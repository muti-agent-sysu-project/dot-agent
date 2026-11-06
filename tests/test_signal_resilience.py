"""回归：本地模型截断/空答/连接失败时，请求应降级而不是 500。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.gateway.demo_behavior import cloud_demo_behavior, local_demo_behavior
from dot_agent.gateway.openai_compat import GenerationError
from dot_agent.gateway.stub import StubModelClient
from dot_agent.orchestrator.signals import SignalCollector
from dot_agent.pipeline import AgentPipeline
from dot_agent.types import Caller, ModelTier


class FailingClient(StubModelClient):
    """按 caller 抛截断错误，复现 Qwen 采样撞到 max_tokens 的真实故障。"""

    def __init__(self, *, fail_callers=(Caller.SIGNAL,), name="failing-local"):
        super().__init__(ModelTier.MD, name, behavior=local_demo_behavior)
        self.fail_callers = set(fail_callers)

    async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                       caller=Caller.EXECUTE):
        if caller in self.fail_callers:
            raise GenerationError("model_incomplete_answer", None)
        return await super().generate(
            prompt, max_tokens=max_tokens, temperature=temperature, caller=caller
        )


def _cloud():
    return StubModelClient(ModelTier.MC, "cloud", behavior=cloud_demo_behavior)


def test_signal_collect_returns_none_instead_of_raising():
    collector = SignalCollector(FailingClient())
    scores, greedy = asyncio.run(collector.collect("一个班30人，每5人一组共多少组？"))
    assert greedy is None
    assert scores.alpha_quantile is None
    assert scores.consistency == 0.0  # 采样缺失按不一致处理，倾向升级云端


def test_signal_partial_failure_marks_inconsistent():
    class PartialFail(FailingClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                           caller=Caller.EXECUTE):
            if caller is Caller.SIGNAL and temperature > 0:  # 只让采样（非 greedy）失败
                raise GenerationError("model_incomplete_answer", None)
            return await StubModelClient.generate(
                self, prompt, max_tokens=max_tokens, temperature=temperature, caller=caller
            )

    collector = SignalCollector(PartialFail())
    scores, greedy = asyncio.run(collector.collect("一个班30人，每5人一组共多少组？"))
    assert greedy is not None          # greedy 成功，仍可作为答案
    assert scores.consistency == 0.0   # 但缺样本 → 不能声称一致


def test_pipeline_survives_signal_failure():
    agent = AgentPipeline(FailingClient(), _cloud(), run_id="test-signal-fail")
    result = asyncio.run(agent.run("一个班30人，每5人一组共多少组？"))
    assert result.success is True
    assert result.path == "pipeline"          # 快速通道失败后升级完整流水线
    assert any("本地采样不可用" in line for line in result.trace)


def test_pipeline_returns_failure_not_exception_when_local_dead():
    dead = FailingClient(fail_callers=(Caller.SIGNAL, Caller.EXECUTE, Caller.SUMMARIZE,
                                       Caller.DECOMPOSE, Caller.DEP_JUDGE),
                         name="dead-local")
    agent = AgentPipeline(dead, _cloud(), run_id="test-local-dead")
    result = asyncio.run(agent.run("一个班30人，每5人一组共多少组？"))
    assert result.success is False
    assert result.error == "model_incomplete_answer"
    assert result.path == "failed"
