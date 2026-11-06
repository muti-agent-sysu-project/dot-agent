"""计量装饰客户端：包装任意 ModelClient，调用即记录（内存 + SQLite 落库）。

增强点（吸收自团队 cloud-edge-agent）：
- **唯一出口**：所有调用过此层，计量不可绕过（含失败尝试）；
- **requested/actual 双档位**：云端不可用且配置降级时改走本地兜底客户端，
  实际档位如实记录，降级率可算，不虚报云端消耗；
- **瞬态重试**：错误分类（瞬态/永久）+ 指数退避，每次尝试**单独**计量
  （申请书口径：失败尝试的 token 也算钱）；
- **usage_missing**：后端未回报 usage 时 token 记 None 并告警，绝不 len 估算。
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import time

from ..types import Caller, GenerationResult, ModelTier
from .base import ModelClient
from .meter import CallMeter
from .openai_compat import GenerationError
from . import store

class LedgerWriteError(RuntimeError):
    """A completed attempt could not be durably recorded; do not call another model."""


# 实测瞬态错误特征（来自团队 RTX 5060/sm_120 与公网环境实测）
_TRANSIENT_MARKERS = (
    "cuda error", "shared object initialization", "process has terminated",
    "internalservererror", "apiconnectionerror", "timeout", "timed out",
    "connection reset", "temporarily unavailable", "502", "503", "504",
)


def _is_transient(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _TRANSIENT_MARKERS)


class MeteredClient(ModelClient):
    def __init__(
        self,
        inner: ModelClient,
        meter: CallMeter,
        *,
        run_id: str = "default",
        fallback: "MeteredClient | None" = None,  # 云端不可用时的本地兜底
        retries: int = 1,
    ) -> None:
        self.inner = inner
        self.meter = meter
        self.run_id = run_id
        self.fallback = fallback
        self.retries = retries
        self.tier = inner.tier
        self.name = inner.name

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        caller: Caller = Caller.EXECUTE,
    ) -> GenerationResult:
        requested_tier = self.tier
        try:
            return await self._with_retries(
                prompt, max_tokens, temperature, caller, requested_tier
            )
        except LedgerWriteError:
            raise
        except RuntimeError as e:
            # 云端不可用（无 key/连接失败）→ 降级本地执行，如实记双档位
            if self.fallback is None or requested_tier is not ModelTier.MC:
                raise
            print("[gateway] ⚠ 云端调用失败，降级本地执行并如实计量", file=sys.stderr)
            return await self.fallback._with_retries(
                prompt, max_tokens, temperature, caller, requested_tier
            )

    async def _with_retries(
        self, prompt: str, max_tokens: int, temperature: float,
        caller: Caller, requested_tier: ModelTier,
    ) -> GenerationResult:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                result = await self.inner.generate(
                    prompt, max_tokens=max_tokens, temperature=temperature, caller=caller
                )
            except Exception as e:  # noqa: BLE001 —— 失败尝试也要计量后决定去留
                last_err = e
                failed = e.result if isinstance(e, GenerationError) else None
                self._log(caller, self.tier, requested_tier,
                          failed.prompt_tokens if failed else 0,
                          failed.completion_tokens if failed else 0,
                          time.perf_counter() - started,
                          failed.usage_missing if failed else True,
                          str(e) if isinstance(e, GenerationError) else type(e).__name__)
                if attempt < self.retries and _is_transient(str(e)):
                    await asyncio.sleep(min(2.0**attempt, 8.0))
                    continue
                raise
            if result.usage_missing:
                print(
                    f"[gateway] ⚠ {self.name} 未返回 usage，token 记为空，"
                    "指标可信度下降（unknown_usage_calls 将计入）", file=sys.stderr,
                )
            self._log(
                caller, self.tier, requested_tier,
                result.prompt_tokens, result.completion_tokens,
                result.latency_s, result.usage_missing, None,
            )
            return result
        raise last_err or RuntimeError("unreachable")

    def _log(
        self, caller: Caller, tier: ModelTier, requested_tier: ModelTier,
        pt: int, ct: int, latency: float, usage_missing: bool, error: str | None,
    ) -> None:
        self.meter.log_call(
            caller=caller, tier=tier, requested_tier=requested_tier,
            prompt_tokens=pt, completion_tokens=ct, latency_s=latency,
            usage_missing=usage_missing, note=error or "",
        )
        try:
            store.record_call(
                caller=caller.value, tier=tier.value, requested_tier=requested_tier.value,
                model=self.name, prompt_tokens=pt, completion_tokens=ct, latency_s=latency,
                run_id=self.run_id, usage_missing=usage_missing, error=error,
            )
        except (OSError, sqlite3.Error):
            raise LedgerWriteError("call_ledger_write_failed") from None
