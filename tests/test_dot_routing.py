"""DoT 式端云分工回归：原题上下文、依赖判断、关键步骤与最终答案。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.gateway.stub import StubModelClient
from dot_agent.orchestrator.scheduler import TaskScheduler
from dot_agent.pipeline import AgentPipeline
from dot_agent.types import Caller, ModelTier, SubTask

QUERY = "鸡兔同笼共10个头28只脚, 兔有几只?"


def _subs(*descs: str) -> list[SubTask]:
    return [SubTask(id=f"s{i}", description=d) for i, d in enumerate(descs, 1)]


def test_key_steps_select_sinks_only():
    subs = _subs("a", "b", "c", "d")
    assert TaskScheduler.key_steps(subs, [("s1", "s3"), ("s2", "s3"),
                                           ("s3", "s4")]) == {"s4"}
    assert TaskScheduler.key_steps(subs, [("s1", "s3"), ("s2", "s3")]) == {"s3", "s4"}


def test_key_steps_ignore_cycle_and_invalid_edges():
    subs = _subs("a", "b", "c")
    assert TaskScheduler.key_steps(subs, [("s1", "s2"), ("s2", "s3"),
                                           ("s3", "s1")]) == set()  # 环不标关键
    assert TaskScheduler.key_steps(subs, [("s9", "s1")]) == set()    # 无效边不标关键
    assert TaskScheduler.key_steps(subs, None) == set()


def _recording_client(tier: ModelTier, calls: list[dict], behavior):
    def wrapped(prompt: str, caller: Caller):
        out = behavior(prompt, caller)
        calls.append({"tier": tier, "caller": caller, "prompt": prompt, "answer": out})
        return out

    return StubModelClient(tier, name=f"{tier.value}-rec", behavior=wrapped)


def _local(prompt: str, caller: Caller):
    if caller is Caller.CLASSIFY:
        return "REASONING"
    if caller is Caller.DECOMPOSE:
        return "1. 求头数\n2. 求兔数"
    if caller is Caller.DEP_JUDGE:  # 不应被调用：依赖判断默认走云端
        return "s1 -> s2"
    return "10"


def _cloud(prompt: str, caller: Caller):
    if caller is Caller.DEP_JUDGE:
        return "s1 -> s2"
    if caller is Caller.EXECUTE:
        return "4只"
    return "4"


def _run():
    local_calls: list[dict] = []
    cloud_calls: list[dict] = []
    agent = AgentPipeline(
        _recording_client(ModelTier.MD, local_calls, _local),
        _recording_client(ModelTier.MC, cloud_calls, _cloud),
        run_id="test-dot-routing", mode="signal",
    )
    result = asyncio.run(agent.run(QUERY))
    return result, local_calls, cloud_calls


def test_subtask_prompt_carries_original_problem_and_answers():
    _, local_calls, cloud_calls = _run()
    sink_prompts = [c["prompt"] for c in cloud_calls
                    if c["caller"] is Caller.EXECUTE]
    assert len(sink_prompts) == 1
    prompt = sink_prompts[0]
    assert f"原问题: {QUERY}" in prompt
    assert "s1:" in prompt          # 已完成的直接前驱答案进入上下文
    assert "求兔数" in prompt


def test_dependency_judge_goes_to_cloud_and_sink_is_key_step():
    result, local_calls, cloud_calls = _run()
    assert not [c for c in local_calls if c["caller"] is Caller.DEP_JUDGE]
    assert [c for c in cloud_calls if c["caller"] is Caller.DEP_JUDGE]

    by_id = {st["id"]: st for st in result.subtasks}
    assert by_id["s2"]["assigned"] == ModelTier.MC.value   # 汇点关键步骤交给云端
    assert by_id["s1"]["assigned"] == ModelTier.MD.value   # 普通子任务留在本地


def test_single_key_step_becomes_final_answer_without_extra_summary():
    result, _, cloud_calls = _run()
    assert result.success and result.path == "pipeline"
    assert result.answer == "4只"
    assert any("唯一关键步骤" in line for line in result.trace)
    assert not [c for c in cloud_calls if c["caller"] is Caller.SUMMARIZE]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all passed")
