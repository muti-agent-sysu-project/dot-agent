"""路由决策：类型 → 策略表（POLICY）+ 信号 → 模型分配。

分类层的最终产出不是标签而是查表策略；分配依据：
  score = 信号组合难度分（orchestrator/signals.py）
  - 无信号      → 策略默认倾向（bias）
  - score > θ   → 云端；否则本地
  - PLANNING 用更低阈值（偏云端），TOOLUSE 默认本地先试、失败升级
阈值 θ 在验证集上扫描；消融时 v0 规则路由即"无信号 + 仅 bias"。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..types import Label, ModelTier, SignalScores


@dataclass(frozen=True)
class Policy:
    decompose: object            # bool | "dynamic"
    graph: object                # bool | "dynamic"
    verify: str                  # consistency / answer_equiv / constraint / exec
    bias: ModelTier              # 无信号时的默认倾向
    theta: float                 # 该类型的分配阈值


POLICY: dict[Label, Policy] = {
    Label.QA: Policy(decompose=False, graph=False, verify="consistency",
                     bias=ModelTier.MD, theta=0.5),
    Label.REASONING: Policy(decompose=True, graph=True, verify="answer_equiv",
                            bias=ModelTier.MD, theta=0.5),
    Label.PLANNING: Policy(decompose=True, graph=True, verify="constraint",
                           bias=ModelTier.MC, theta=0.35),   # 偏云端
    Label.TOOLUSE: Policy(decompose="dynamic", graph="dynamic", verify="exec",
                          bias=ModelTier.MD, theta=0.5),
}


@dataclass
class Decision:
    tier: ModelTier
    policy: Policy
    score: float | None
    reason: str


class Router:
    def __init__(self, theta_default: float | None = None) -> None:
        if theta_default is not None and not 0 <= theta_default <= 1:
            raise ValueError("theta_default must be in [0, 1]")
        self.theta_default = theta_default

    def assign(self, label: Label, signals: SignalScores | None) -> Decision:
        policy = POLICY[label]
        if self.theta_default is not None:
            policy = replace(policy, theta=self.theta_default)
        if signals is None:
            return Decision(policy.bias, policy, None, "no_signal→bias")
        score = signals.difficulty()
        if score is None:
            return Decision(policy.bias, policy, None, "no_valid_signal→bias")
        tier = ModelTier.MC if score > policy.theta else ModelTier.MD
        reason = (
            f"score={score:.2f} {'>' if score > policy.theta else '<='}"
            f" θ={policy.theta} → {tier.value}"
        )
        return Decision(tier, policy, score, reason)
