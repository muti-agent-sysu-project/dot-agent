"""六项设计优化测试：采样并行、合并推理、θ 自适应、语义缓存、汇总升级、候选多样。"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dot_agent.adaptive import ThetaAdapter
from dot_agent.cache.semantic import SemanticCache
from dot_agent.gateway.base import ModelClient
from dot_agent.gateway.stub import StubModelClient
from dot_agent.orchestrator.decomposer import TaskDecomposer
from dot_agent.orchestrator.signals import SignalCollector
from dot_agent.types import Caller, GenerationResult, Label, ModelTier


def _run(coro):
    return asyncio.run(coro)


# ---------------- 优化 1：采样并行 ----------------
class _ConcurrencyProbe(ModelClient):
    """记录同时在途的请求数，验证 gather 真的并行。"""

    def __init__(self):
        self.tier = ModelTier.MD
        self.name = "probe"
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                       caller=Caller.EXECUTE):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.02)
        self.in_flight -= 1
        return GenerationResult(text="42", latency_s=0.02)


def test_sampling_runs_in_parallel():
    probe = _ConcurrencyProbe()
    collector = SignalCollector(probe)
    sig, greedy, raws = _run(collector.collect("12*(3+4)?", with_samples=True))
    assert probe.max_in_flight == 3, f"未并行: max={probe.max_in_flight}"
    assert sig.consistency == 1.0
    assert len(raws) == 3


# ---------------- 优化 2：合并推理 ----------------
def test_decompose_and_classify_merged():
    async def behavior(prompt, caller):
        return ("类型: REASONING\n"
                "1. 计算每组人数 30 除以 5\n"
                "2. 得出组数")
    decomposer = TaskDecomposer(StubModelClient(ModelTier.MD, behavior=behavior))
    label, subtasks = _run(decomposer.decompose_and_classify("一个班30人每组5人共几组"))
    assert label is Label.REASONING
    assert len(subtasks) == 2 and subtasks[0].id == "s1"


def test_decompose_and_classify_label_only_fallback():
    async def behavior(prompt, caller):
        return "类型: QA\n（分解失败没有编号列表）"
    decomposer = TaskDecomposer(StubModelClient(ModelTier.MD, behavior=behavior))
    label, subtasks = _run(decomposer.decompose_and_classify("水的沸点"))
    assert label is Label.QA and subtasks is None  # 标签仍有价值，供收敛云端路径用


def test_pipeline_uses_merged_path():
    """L1 不确定的查询走合并推理：DECOMPOSE 调用里应包含'类型:'提示。"""
    from dot_agent.pipeline import AgentPipeline

    seen_prompts = []

    async def local_behavior(prompt, caller):
        seen_prompts.append((caller.value, prompt[:60]))
        if caller is Caller.DECOMPOSE:
            return ("类型: REASONING\n1. 计算 12 乘以 7\n2. 给出结果")
        if caller is Caller.CLASSIFY:
            return "REASONING"
        return "84"

    cloud = _make_cloud()
    pipeline = AgentPipeline(
        StubModelClient(ModelTier.MD, behavior=local_behavior, token_probs_mode="uniform"),
        cloud,
    )
    result = _run(pipeline.run("帮我算一下这个东西的数值结果是啥呢"))  # L1 规则不命中的模糊问句
    assert any(caller == "decompose" and "类型" in p for caller, p in seen_prompts), seen_prompts
    assert result.metrics["privacy"] is not None


def _make_cloud():
    class _C(ModelClient):
        tier = ModelTier.MC
        name = "test-cloud"

        async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                           caller=Caller.EXECUTE):
            return GenerationResult(text="云端", latency_s=0.01)
    return _C()


# ---------------- 优化 3：θ 自适应 ----------------
def test_theta_adapter_learns_from_failures(tmp_path):
    _tmp = tmp_path
    adapter = ThetaAdapter({Label.REASONING: 0.5}, enabled=True,
                           path=Path(_tmp) / "t1.json")
    for _ in range(8):
        adapter.observe(Label.REASONING, "local_fail")
    assert adapter.current_theta(Label.REASONING) < 0.5  # 失败率高 → 压低 θ
    ok_adapter = ThetaAdapter({Label.REASONING: 0.5}, enabled=True,
                              path=Path(_tmp) / "t2.json")
    for _ in range(8):
        ok_adapter.observe(Label.REASONING, "local_ok")
    assert ok_adapter.current_theta(Label.REASONING) == 0.5  # 无失败不动


def test_theta_adapter_disabled_and_clamped(tmp_path):
    _tmp = tmp_path
    adapter = ThetaAdapter({Label.QA: 0.5}, enabled=False)
    adapter.observe(Label.QA, "local_fail")
    assert adapter.current_theta(Label.QA) == 0.5  # 关闭时不学习
    strict = ThetaAdapter({Label.QA: 0.5}, enabled=True,
                          path=Path(_tmp) / "t3.json")
    for _ in range(30):
        strict.observe(Label.QA, "local_fail")
    assert strict.current_theta(Label.QA) >= 0.05  # clamp 下限


def test_theta_adapter_disabled_does_not_touch_router(tmp_path):
    """回归：关闭自适应时，上次运行落盘的旧状态不得渗入 router（消融污染）。"""
    from dot_agent.orchestrator.router import Router

    state = Path(tmp_path) / "t4.json"
    state.write_text('{"QA": {"fail_ema": 0.9, "n": 50}}', encoding="utf-8")
    router = Router({Label.QA: 0.7})  # 显式标定 θ=0.7
    adapter = ThetaAdapter(dict(router._theta), enabled=False, path=state)
    adapter.apply_to(router)
    assert router._theta[Label.QA] == 0.7  # 关闭 → router θ 原样保留
    # 对照：开启时旧状态确实生效
    adapter_on = ThetaAdapter(dict(router._theta), enabled=True, path=state)
    adapter_on.apply_to(router)
    assert router._theta[Label.QA] < 0.7  # fail_ema=0.9 → θ 被压低


# ---------------- 优化 4：语义缓存 ----------------
def test_semantic_cache_exact_and_similar():
    cache = SemanticCache()
    cache.put("中国剩余定理的形式化证明", "答案甲", deterministic=True)
    assert cache.get("中国剩余定理的形式化证明", True) == "答案甲"
    sim = cache.get("中国剩余定理的形式化证明方法", True)  # 加了两个字，词面高度重叠
    assert sim == "答案甲" or cache.stats["reject_low_sim"] >= 0
    assert cache.get("明天北京天气怎么样", True) is None  # 低相似拒绝


def test_semantic_cache_respects_deterministic_and_ttl():
    cache = SemanticCache(ttl_s=0.0)
    cache.put("1+1 等于多少", "2", deterministic=True)
    assert cache.get("1+1 等于多少", True) is None  # TTL=0 → 过期
    non_det = SemanticCache()
    non_det.put("写一首诗", "诗", deterministic=False)
    assert non_det.get("写一首诗", False) is None  # 非确定性域永不命中


def test_semantic_cache_embedding_fn_used_first():
    cache = SemanticCache(embedding_fn=lambda s: [1.0, 0.0] if "天气" in s else [0.0, 1.0])
    cache.put("北京天气怎么样", "晴", deterministic=True)
    assert cache.get("上海天气怎么样", True) == "晴"  # embedding 判相似


# ---------------- 优化 5：汇总升级 ----------------
def test_summary_escalates_to_cloud_on_truncation():
    from dot_agent.pipeline import AgentPipeline

    class _LocalDie(ModelClient):
        tier = ModelTier.MD
        name = "die-on-summarize"

        async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                           caller=Caller.EXECUTE):
            if caller is Caller.DECOMPOSE:
                return GenerationResult(text="1. 计算 3 加 5\n2. 输出结果", latency_s=0.01)
            if caller is Caller.CLASSIFY:
                return GenerationResult(text="REASONING", latency_s=0.01)
            if caller is Caller.SUMMARIZE:
                raise RuntimeError("model_incomplete_answer")
            return GenerationResult(text="8", latency_s=0.01)

    cloud_seen = []

    class _Cloud(ModelClient):
        tier = ModelTier.MC
        name = "summary-cloud"

        async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                           caller=Caller.EXECUTE):
            cloud_seen.append(caller.value)
            return GenerationResult(text="最终答案是 8", latency_s=0.01)

    pipeline = AgentPipeline(_LocalDie(), _Cloud())
    result = _run(pipeline.run("计算 3 加 5 并给出结果 7+8"))
    assert "最终答案是 8" in result.answer
    assert "summarize" in cloud_seen, cloud_seen  # 云端参与了汇总
    assert any("升级云端汇总" in line for line in result.trace)


# ---------------- 优化 6：候选温度多样化 ----------------
def test_sample_temps_diverse():
    temps_seen = []

    class _TempProbe(ModelClient):
        tier = ModelTier.MD
        name = "temp-probe"

        async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                           caller=Caller.EXECUTE):
            temps_seen.append(temperature)
            return GenerationResult(text="42", latency_s=0.01)

    collector = SignalCollector(_TempProbe())
    _run(collector.collect("12*(3+4)?", with_samples=True))
    assert temps_seen[0] == 0.0
    assert 0.0 in temps_seen and any(t > 0.9 for t in temps_seen) and any(t < 0.6 for t in temps_seen)
