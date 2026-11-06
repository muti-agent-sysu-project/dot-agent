"""路由决策：类型 → 策略表（POLICY）+ 信号 → 模型分配。

分类层的最终产出不是标签而是查表策略；分配依据：
  score = 信号组合难度分（orchestrator/signals.py）
  - 无信号      → 策略默认倾向（bias）
  - score > θ   → 云端；否则本地
  - PLANNING 用更低阈值（偏云端），TOOLUSE 默认本地先试、失败升级

θ 来源（优先级从高到低）：
  1. 显式传入的 theta_overrides（评测消融用）
  2. config/router_theta.json（scripts/calibrate_theta.py 的标定产物）
  3. POLICY 表里的默认值

消融时 v0 规则路由即"无信号 + 仅 bias"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..types import Label, ModelTier, SignalScores

THETA_CONFIG = Path(__file__).resolve().parents[2] / "config" / "router_theta.json"


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


def load_theta_overrides(path: Path | None = None) -> dict[Label, float]:
    """读标定产物 config/router_theta.json；不存在或格式不对返回空（用默认值）。"""
    p = path or THETA_CONFIG
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[Label, float] = {}
    for key, value in raw.items():
        try:
            label = Label(key)
            theta = float(value)
        except (ValueError, TypeError):
            continue
        if 0.0 < theta < 1.0:
            out[label] = theta
    return out


@dataclass
class Decision:
    tier: ModelTier
    policy: Policy
    score: float | None
    reason: str


class Router:
    def __init__(self, theta_overrides: dict[Label, float] | None = None) -> None:
        # 标定值覆盖 POLICY 默认；显式传入优先于配置文件
        calibrated = load_theta_overrides()
        self._theta: dict[Label, float] = {}
        for label, policy in POLICY.items():
            if theta_overrides and label in theta_overrides:
                self._theta[label] = theta_overrides[label]
            elif label in calibrated:
                self._theta[label] = calibrated[label]
            else:
                self._theta[label] = policy.theta

    def assign(self, label: Label, signals: SignalScores | None) -> Decision:
        policy = POLICY[label]
        theta = self._theta[label]
        if signals is None:
            return Decision(policy.bias, policy, None, "no_signal→bias")
        score = signals.difficulty()
        if score is None:
            return Decision(policy.bias, policy, None, "no_valid_signal→bias")
        tier = ModelTier.MC if score > theta else ModelTier.MD
        origin = "标定" if theta != policy.theta else "默认"
        reason = (
            f"score={score:.2f} {'>' if score > theta else '<='}"
            f" θ={theta:.2f}({origin}) → {tier.value}"
        )
        return Decision(tier, policy, score, reason)
