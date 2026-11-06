"""隐私模块测试：敏感度规则、假名化回填、执行模式组合、判别解析、端到端。"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dot_agent.gateway.stub import StubModelClient
from dot_agent.gateway.base import ModelClient
from dot_agent.privacy import (
    ExecMode, PrivacyGuard, Sensitivity, analyze,
)
from dot_agent.privacy.judge import parse_judge_reply
from dot_agent.privacy.pseudonymizer import EntityMap
from dot_agent.types import Caller, GenerationResult, Label, ModelTier, SignalScores


# ---------------------------------------------------------------- 规则层
def test_phone_id_high():
    r = analyze("联系张三 13812345678 确认")
    assert r.level is Sensitivity.HIGH
    assert ("13812345678", "手机号") in r.entities


def test_domain_keyword_high_without_entity():
    r = analyze("该患者的诊断结果如何")
    assert r.level is Sensitivity.HIGH
    assert r.entities == []


def test_date_low():
    r = analyze("2024年3月5日的安排")
    assert r.level is Sensitivity.LOW


def test_plain_none():
    assert analyze("中国剩余定理的形式化证明").level is Sensitivity.NONE


def test_merge_takes_stricter():
    a, b = analyze("2024年3月5日"), analyze("无")
    assert a.merge(b).level is Sensitivity.LOW
    assert b.merge(analyze("工资明细")).level is Sensitivity.HIGH


# ---------------------------------------------------------------- 假名化
def test_pseudonymize_restore_roundtrip():
    g = PrivacyGuard(enabled=True)
    report = analyze("13812345678 的机主")
    masked = g.pseudonymize("请回拨 13812345678 确认", report)
    assert "13812345678" not in masked
    import re as _re
    assert _re.search(r"\d{11}", masked), masked  # 同长度合成数字
    assert "13812345678" in g.restore(masked)


def test_entity_map_consistent_across_calls():
    m = EntityMap()
    r = analyze("13812345678")
    first = m.pseudonymize("给 13812345678 发消息", r)
    second = m.pseudonymize("再给 13812345678 打电话")  # 盲替换沿用旧映射
    assert first.split()[-2] == second.split()[-2]


def test_leak_check():
    m = EntityMap()
    r = analyze("13812345678")
    m.pseudonymize("x", r)
    assert m.leak_check("包含 13812345678 原文") == 1
    assert m.leak_check("手机号1 已替换") == 0


# ---------------------------------------------------------------- 模式组合
def test_mode_matrix():
    g = PrivacyGuard(enabled=True)
    assert g.classify(False, ["普通问题"])[0] is ExecMode.LOCAL_ONLY
    assert g.classify(True, ["普通问题"])[0] is ExecMode.CLOUD_PLAIN
    assert g.classify(True, ["2024年3月5日开会"])[0] is ExecMode.CLOUD_MASKED
    assert g.classify(True, ["他的病历显示"])[0] is ExecMode.CLOUD_JUDGE


def test_disabled_passthrough():
    g = PrivacyGuard(enabled=False)
    assert g.classify(True, ["病历 病历"])[0] is ExecMode.CLOUD_PLAIN
    assert g.pseudonymize("13812345678") == "13812345678"


# ---------------------------------------------------------------- 判别解析
def test_parse_judge_number():
    cands = ["答案甲", "答案乙", "答案丙"]
    assert parse_judge_reply("2", cands) == ("答案乙", True)
    assert parse_judge_reply("选 3", cands) == ("答案丙", True)
    assert parse_judge_reply("修正: 都不对，应为 42", cands) == ("都不对，应为 42", True)
    assert parse_judge_reply("我不知道", cands) == ("", False)


def test_parse_judge_multi_digit_not_misparsed():
    """回归："42" 不得被解析成候选 4，"12" 不得被解析成候选 1。"""
    cands = ["答案甲", "答案乙", "答案丙"]
    assert parse_judge_reply("42", cands) == ("", False)      # 两位数不是合法编号
    assert parse_judge_reply("12", cands) == ("", False)
    assert parse_judge_reply("候选2", cands) == ("答案乙", True)
    assert parse_judge_reply("修正：应为 42", cands) == ("应为 42", True)


# ---------------------------------------------------------------- 端到端
class _ScriptCloud(ModelClient):
    """记录收到的 prompt，按脚本回话。"""

    def __init__(self):
        self.tier = ModelTier.MC
        self.name = "script-cloud"
        self.seen: list[str] = []

    async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                       caller=Caller.EXECUTE):
        self.seen.append(prompt)
        if caller is Caller.JUDGE:
            return GenerationResult(text="2", latency_s=0.01)
        return GenerationResult(text="云端基于人名1的答案", latency_s=0.01)


def _run(coro):
    return asyncio.run(coro)


def test_e2e_high_sensitivity_uses_judge(monkeypatch=None):
    """敏感子任务转交云端：云端只见假名化候选，答案本地回填。"""
    from dot_agent.pipeline import AgentPipeline

    calls = {"n": 0}

    async def local_behavior(prompt, caller):
        if caller is Caller.DECOMPOSE:
            return "1. 查阅病历里13812345678患者的基础信息 3+5\n2. 计算该患者的用药剂量 3+5"
        if caller is Caller.CLASSIFY:
            return "REASONING"
        calls["n"] += 1
        # 答案长且每次不同：α 分位数低（hard 模式衰减）+ 一致性低 → 难度分过阈值转云端
        return f"候选{calls['n']}: " + "推理过程很长因此token概率衰减 " * 12 + f"最终剂量是{calls['n']}毫克"

    cloud = _ScriptCloud()
    pipeline = AgentPipeline(
        StubModelClient(ModelTier.MD, behavior=local_behavior, token_probs_mode="hard"),
        cloud,
    )
    # 一致性低 → 路由云端；文本含手机号+病历 → CLOUD_JUDGE
    result = _run(pipeline.run("计算病历：13812345678 患者的剂量 3+5 是多少"))
    judge_prompts = [p for p in cloud.seen if "候选" in p]
    assert judge_prompts, f"未发生判别调用, seen={cloud.seen}"
    sent = judge_prompts[0]
    assert "13812345678" not in sent, "手机号泄露到云端"
    import re as _re
    assert _re.search(r"\d{11}", sent), sent  # 合成数字替换
    pm = result.metrics["privacy"]
    assert pm["judge_calls"] >= 1
    assert pm["leak_count"] == 0
    assert "毫克" in result.answer  # 判别选中的候选已回填


def test_e2e_low_sensitivity_masked(monkeypatch=None):
    from dot_agent.pipeline import AgentPipeline

    calls = {"n": 0}

    async def local_behavior(prompt, caller):
        if caller is Caller.DECOMPOSE:
            return "1. 确认2024年3月5日的会面安排\n2. 给出当日日程建议"
        if caller is Caller.CLASSIFY:
            return "REASONING"
        calls["n"] += 1
        return f"候选{calls['n']}: " + "推理过程很长因此token概率衰减 " * 12 + "安排待定"

    cloud = _ScriptCloud()

    async def gen(prompt, *, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE):
        cloud.seen.append(prompt)
        if caller is Caller.JUDGE:
            return GenerationResult(text="无法判断", latency_s=0.01)  # 判别失败 → 假名化生成
        return GenerationResult(text="安排在日期1", latency_s=0.01)

    cloud.generate = gen  # 覆盖实例方法
    pipeline = AgentPipeline(
        StubModelClient(ModelTier.MD, behavior=local_behavior, token_probs_mode="hard"),
        cloud,
    )
    result = _run(pipeline.run("2024年3月5日的安排是什么"))
    # 假名化转交：云端只见"日期1"，不见原文；答案回填后含真实日期
    assert any("日期1" in p and "2024年3月5日" not in p for p in cloud.seen), cloud.seen
    # 云端答案"安排在日期1"回填后应含真实日期（本地汇总是桩文本，看子任务答案）
    assert any("2024年3月5日" in st["answer"] for st in result.subtasks), result.subtasks
    pm = result.metrics["privacy"]
    assert pm["masked_cloud_calls"] >= 1
    assert pm["leak_count"] == 0


# ---------------------------------------------------------------- P2 扩展
def test_numeric_same_length_pseudonym():
    """手机号替换为同长度合成数字，格式效用保持且可逆。"""
    from dot_agent.privacy.pseudonymizer import EntityMap
    m = EntityMap()
    r = analyze("联系 13812345678")
    masked = m.pseudonymize("联系 13812345678", r)
    import re as _re
    digits = _re.findall(r"\b\d{11}\b", masked)
    assert digits and len(digits[0]) == 11 and digits[0] != "13812345678"
    assert "13812345678" in m.restore(masked)
    # 同一原文映射稳定
    m2 = EntityMap()
    assert m2.pseudonymize("致电 13812345678", analyze("13812345678")).split()[-1] == digits[0]


def test_roundtrip_check():
    from dot_agent.privacy.pseudonymizer import EntityMap
    m = EntityMap()
    r = analyze("联系 13812345678")
    original = "联系 13812345678"
    masked = m.pseudonymize(original, r)
    ok = m.verify_roundtrip([original], ["机主是 " + m.restore(masked)])
    assert ok["consistent"]
    # 假名残留 → 不一致
    bad = m.verify_roundtrip([original], [masked])
    assert not bad["consistent"]


def test_plain_disclosure_counted():
    from dot_agent.privacy import PrivacyGuard, ExecMode
    # 隐私关闭（消融对照）时一切明文发出，披露数如实累计
    g = PrivacyGuard(enabled=False)
    mode, _ = g.classify(True, ["13812345678 与 13900000011 都是明文"])
    assert mode is ExecMode.CLOUD_PLAIN
    assert g.ledger.entities_disclosed_plain == 2


def test_views_recorded():
    from dot_agent.privacy import PrivacyGuard, ExecMode
    g = PrivacyGuard(enabled=True)
    g.classify(True, ["2024年3月5日的安排"])
    g.pseudonymize("2024年3月5日的安排")
    g.ledger.views.append({"id": "s1", "mode": "cloud_masked", "text": "日期1"})
    assert g.ledger.summary()["views"][0]["id"] == "s1"


def test_ner_fusion_rules_default():
    from dot_agent.privacy.ner import get_backend, RuleBackend, analyze_with_ner
    backend = get_backend("rules")
    assert isinstance(backend, RuleBackend)
    # transformers 未安装时 load 失败自动退化，不抛异常
    from dot_agent.privacy.ner import TransformersBackend
    tb = TransformersBackend(model_name="nonexistent/model-for-test")
    report = analyze_with_ner("联系 13812345678", tb)
    assert report.level is Sensitivity.HIGH


def test_router_theta_calibration_file():
    from dot_agent.orchestrator.router import Router, load_theta_overrides
    import pathlib as _pl
    p = _pl.Path(tempfile.mkdtemp()) / "router_theta.json"
    p.write_text(json.dumps({"REASONING": 0.7}), encoding="utf-8")
    overrides = load_theta_overrides(p)
    assert overrides[Label.REASONING] == 0.7
    # 仓库可能带有标定产物 config/router_theta.json，此时 Router 用标定值，
    # θ 来源标注为"标定"；未写 config 时回退 POLICY 默认 0.50 且标"默认"
    repo_theta = load_theta_overrides().get(Label.REASONING)
    decision = Router().assign(Label.REASONING, SignalScores(consistency=0.8))
    if repo_theta is not None and repo_theta != 0.5:
        assert f"θ={repo_theta:.2f}(标定)" in decision.reason
    else:
        assert "θ=0.50(默认)" in decision.reason
