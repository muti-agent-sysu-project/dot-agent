"""装配演示或真实客户端：LOCAL_BASE_URL 优先于 Ollama。
DOT_REQUIRE_REAL=1 拒绝桩客户端，启动时要求两端短生成及有效 usage。
"""
from __future__ import annotations

import os
import sys

from ..types import ModelTier
from .base import ModelClient
from .demo_behavior import cloud_demo_behavior, local_demo_behavior
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient
from .stub import StubModelClient


def _ollama_alive(host: str) -> bool:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as resp:
            return bool(json.loads(resp.read().decode()))
    except Exception:
        return False


def make_clients(
    local: ModelClient | None = None, cloud: ModelClient | None = None
) -> tuple[ModelClient, ModelClient, dict]:
    """返回 (local_client, cloud_client, runtime_info)。"""
    info: dict = {"local": "stub", "cloud": "stub", "notes": [], "warmup_needed": False}

    strict = os.environ.get("DOT_REQUIRE_REAL") == "1"
    if local is None:
        if os.environ.get("LOCAL_BASE_URL"):
            if not os.environ.get("LOCAL_MODEL"):
                raise RuntimeError("LOCAL_BASE_URL requires LOCAL_MODEL")
            local = OpenAICompatClient(
                base_url=os.environ["LOCAL_BASE_URL"], model=os.environ["LOCAL_MODEL"],
                api_key=os.environ.get("LOCAL_API_KEY", ""), tier=ModelTier.MD,
            )
            info["local"] = f"openai-compat-local:{local.model}"
            info["warmup_needed"] = True
        elif os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_MODEL"):
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if _ollama_alive(host):
                local = OllamaClient()
                info["local"] = f"ollama:{local.model}"
                info["warmup_needed"] = True
            else:
                info["notes"].append(f"OLLAMA 环境变量已设但 {host} 不可达，本地回落 stub")
        else:
            info["notes"].append("未设置 OLLAMA_HOST，本地使用 stub")
    else:
        info["local"] = local.name

    if cloud is None:
        if os.environ.get("CLOUD_API_KEY"):
            cloud = OpenAICompatClient(thinking=os.environ.get("CLOUD_THINKING") or None)
            info["cloud"] = f"openai-compat:{cloud.model}"
        else:
            info["notes"].append("未设置 CLOUD_API_KEY，云端使用 stub")
    else:
        info["cloud"] = cloud.name

    if strict and (local is None or cloud is None or
                   isinstance(local, StubModelClient) or isinstance(cloud, StubModelClient)):
        raise RuntimeError("DOT_REQUIRE_REAL=1 requires real local and cloud clients; stub refused")

    return (
        local or StubModelClient(ModelTier.MD, "slm-stub", behavior=local_demo_behavior),
        cloud or StubModelClient(ModelTier.MC, "cloud-stub", behavior=cloud_demo_behavior),
        info,
    )


async def warmup_real(info: dict, local: ModelClient, cloud: ModelClient) -> None:
    """真实客户端预热（stub 跳过）。server/eval 启动时调用，防冷启动污染 P95。"""
    from .warmup import warmup

    if os.environ.get("DOT_REQUIRE_REAL") == "1":
        from ..types import Caller
        # 思考型云端模型（如 GLM-5.3-flash）的 reasoning 也耗 token，32 不够预热
        for client in (local, cloud):
            # 预热失败只降级告警，绝不让服务崩溃重启循环：
            # 云端限流/余额不足/隧道未就绪都不该阻断正常请求路径
            try:
                result = await client.generate("Reply only OK", max_tokens=256, caller=Caller.WARMUP)
            except Exception as error:  # noqa: BLE001 预热属尽力而为
                print(f"[gateway] preflight {client.name} 失败（跳过）: {error}",
                      file=sys.stderr)
                continue
            if result.usage_missing:
                print(f"[gateway] preflight {client.name}: usage 缺失（计量记 NULL）",
                      file=sys.stderr)
                continue
            print(f"[gateway] preflight {client.name}: usage={result.prompt_tokens}/{result.completion_tokens}")
        return

    if info.get("warmup_needed"):
        dt = await warmup(local)
        print(f"[gateway] 本地模型预热完成 {dt:.1f}s")
    if info.get("cloud", "").startswith("openai-compat"):
        dt = await warmup(cloud)
        print(f"[gateway] 云端预热完成 {dt:.1f}s")


def report(info: dict) -> None:
    for note in info.get("notes", []):
        print(f"[gateway] {note}", file=sys.stderr)
