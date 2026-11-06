"""θ 自适应：用"本地失败→云端成功"的样本流在线微调各任务类型的分配阈值。

机制（无训练、无新模型，纯统计反馈）：
  - executor 观测两类事件：本地答案校验通过（local_ok）、校验不过升级
    云端后成功（local_fail）；
  - 按 Label 维护失败率 EMA；失败率高说明该类任务对本模型偏难，压低 θ
    让更多请求更早转云端（θ 越低越多请求判云端）；
  - theta = clamp(base_theta - step × fail_ema, 0.05, 0.95)，base 来自
    Router 初始 θ（标定文件或默认值）；
  - 每 UPDATE_EVERY 次观测写回 config/router_theta_adaptive.json，重启后
    自动生效（Router 启动时读取标定文件，此处单独落盘避免覆盖人工标定值）。

诚实边界：fail_ema 上升也可能源于任务分布漂移而非模型能力；自适应只
在 DOT_ADAPTIVE=on 时生效（默认 off，消融对照可控）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .types import Label

ADAPTIVE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "router_theta_adaptive.json"
UPDATE_EVERY = 20       # 每多少次观测落盘一次
EMA_ALPHA = 0.2         # EMA 平滑系数
STEP = 0.3              # 失败率→θ 的传导系数
THETA_MIN, THETA_MAX = 0.05, 0.95


class ThetaAdapter:
    def __init__(self, base_theta: dict[Label, float], *,
                 enabled: bool = False, path: Path | None = None) -> None:
        self.enabled = enabled
        self.path = path or ADAPTIVE_CONFIG
        self.base_theta = dict(base_theta)
        # 从上次运行恢复 EMA 状态（跨重启累积）
        self._state: dict[str, dict] = {}
        try:
            self._state = {
                k: v for k, v in json.loads(self.path.read_text(encoding="utf-8")).items()
                if isinstance(v, dict) and "fail_ema" in v
            }
        except (OSError, ValueError):
            pass
        self._since_flush = 0
        self.updates = 0  # 触发过 θ 变更的次数（可观测）

    def observe(self, label: Label, outcome: str) -> None:
        """记录一次本地执行结果：outcome ∈ {"local_ok", "local_fail"}。"""
        if not self.enabled or label not in self.base_theta:
            return
        state = self._state.setdefault(
            label.value, {"fail_ema": 0.0, "n": 0})
        failed = 1.0 if outcome == "local_fail" else 0.0
        state["n"] += 1
        # 前几次样本直接用均值，避免初始 EMA 偏 0 掩盖早期失败
        if state["n"] <= 5:
            state["fail_ema"] = (state["fail_ema"] * (state["n"] - 1) + failed) / state["n"]
        else:
            state["fail_ema"] = (1 - EMA_ALPHA) * state["fail_ema"] + EMA_ALPHA * failed
        self._since_flush += 1
        if self._since_flush >= UPDATE_EVERY:
            self.flush()

    def current_theta(self, label: Label) -> float:
        base = self.base_theta.get(label, 0.5)
        state = self._state.get(label.value)
        if not state or state["n"] < 3:  # 样本太少不动 θ
            return base
        theta = base - STEP * state["fail_ema"]
        return max(THETA_MIN, min(THETA_MAX, theta))

    def apply_to(self, router) -> None:
        """把当前自适应 θ 写入 router（executor 每次分配前调用 apply 或读 current）。

        关闭时必须直接返回：上次自适应运行落盘的旧状态不得渗入本次
        消融对照（DOT_ADAPTIVE=off 必须等价于"无自适应"）。
        """
        if not self.enabled:
            return
        for label in self.base_theta:
            router._theta[label] = self.current_theta(label)

    def flush(self) -> None:
        self._since_flush = 0
        if not self._state:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except OSError:
            pass  # 落盘失败不阻断主流程，状态仍在内存
