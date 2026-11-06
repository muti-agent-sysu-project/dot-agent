"""三指标计量器（内存汇总；SQLite 落库见 gateway/store.py）。

口径严格对齐申请书：
  C_API  : 仅云端（MC）调用的 in+out token 总和，包含规划/校验/重试全部调用；
           usage 缺失的调用不并入（known_subtotal 口径），unknown_usage_calls 分列。
  C_time : 请求到结果的墙钟时间（pipeline 记总时长），本地推理计入；
           请求级量纲——分位数必须按请求（tasks）算，不能按调用（calls）算。
  规划开销（团队 cloud-edge-agent 提出）：caller ∈ {classify, decompose,
           dep_judge, summarize} 的云端 token 单独统计——DoT 论文未计量此开销。
"""
from __future__ import annotations

import time
from collections import defaultdict

from ..types import CallLog, Caller, ModelTier

PLANNING_CALLERS = {Caller.CLASSIFY, Caller.DECOMPOSE, Caller.DEP_JUDGE, Caller.SUMMARIZE}


class CallMeter:
    def __init__(self) -> None:
        self.logs: list[CallLog] = []
        self._stage_marks: dict[str, float] = {}
        self.stage_latency: dict[str, float] = defaultdict(float)

    # ---- 计量 ----
    def log(self, log: CallLog) -> None:
        self.logs.append(log)

    def log_call(
        self,
        caller: Caller,
        tier: ModelTier,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        requested_tier: ModelTier = ModelTier.MD,
        usage_missing: bool = False,
        note: str = "",
    ) -> None:
        self.logs.append(
            CallLog(
                caller, tier, prompt_tokens, completion_tokens, latency_s,
                requested_tier=requested_tier, usage_missing=usage_missing, note=note,
            )
        )

    # ---- 阶段计时（调度层时延拆解）----
    def stage(self, name: str) -> None:
        self._stage_marks[name] = time.perf_counter()

    def stage_end(self, name: str) -> float:
        dt = time.perf_counter() - self._stage_marks[name]
        self.stage_latency[name] += dt
        return dt

    # ---- 汇总 ----
    @property
    def cloud_in_tokens(self) -> int:
        return sum(
            l.prompt_tokens for l in self.logs
            if l.tier is ModelTier.MC and not l.usage_missing
        )

    @property
    def cloud_out_tokens(self) -> int:
        return sum(
            l.completion_tokens for l in self.logs
            if l.tier is ModelTier.MC and not l.usage_missing
        )

    def summarize(self, total_time_s: float | None = None) -> dict:
        by_caller: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens": 0})
        for l in self.logs:
            entry = by_caller[l.caller.value]
            entry["calls"] += 1
            if not l.usage_missing:
                entry["tokens"] += l.prompt_tokens + l.completion_tokens
        unknown = sum(1 for l in self.logs if l.usage_missing)
        fallbacks = sum(
            1 for l in self.logs
            if l.tier is not l.requested_tier
        )
        planning = sum(
            l.prompt_tokens + l.completion_tokens
            for l in self.logs
            if l.tier is ModelTier.MC and l.caller in PLANNING_CALLERS
            and not l.usage_missing
        )
        return {
            "C_time_s": round(total_time_s, 4) if total_time_s else None,
            "cloud_calls": sum(1 for l in self.logs if l.tier is ModelTier.MC),
            "C_API_in": self.cloud_in_tokens,
            "C_API_out": self.cloud_out_tokens,
            "C_API_total": self.cloud_in_tokens + self.cloud_out_tokens,
            "C_API_planning": planning,          # 规划开销（DoT 未计量的差异点）
            "unknown_usage_calls": unknown,      # usage 缺失：C_API 为已知部分小计
            "cloud_fallbacks": fallbacks,        # 云端降级本地次数（降级率分子）
            "local_calls": sum(1 for l in self.logs if l.tier is ModelTier.MD),
            "local_latency_s": round(
                sum(l.latency_s for l in self.logs if l.tier is ModelTier.MD), 4
            ),
            "stage_latency_s": {k: round(v, 4) for k, v in self.stage_latency.items()},
            "by_caller": dict(by_caller),
            # 逐次调用的完整记录：面板据此还原中间过程（只含元数据，不含提示词正文）
            "calls": [
                {
                    "seq": i,
                    "caller": l.caller.value,
                    "tier": l.tier.value,
                    "requested_tier": l.requested_tier.value,
                    "in": None if l.usage_missing else l.prompt_tokens,
                    "out": None if l.usage_missing else l.completion_tokens,
                    "latency_s": round(l.latency_s, 3),
                    "usage_missing": l.usage_missing,
                    "note": l.note,
                }
                for i, l in enumerate(self.logs, 1)
            ],
        }
