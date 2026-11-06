"""核心数据结构：标签、特征、分类结果、子任务、信号、指标记录。

与《任务分类技术实现路线.md》中的标签体系（3 主类+特征位）及
申请书三指标口径（Acc / 云端 token / 完整响应时间）一一对应。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Label(str, Enum):
    """任务主类型（互斥）。"""

    QA = "QA"                # 单跳事实/常识问答
    REASONING = "REASONING"  # 多步推理/数学
    PLANNING = "PLANNING"    # 约束满足/规划
    TOOLUSE = "TOOLUSE"      # 工具调用


class ModelTier(str, Enum):
    MD = "MD"  # 本地小模型（不计云端 token，但耗时计入 C_time）
    MC = "MC"  # 云端大模型（in/out token 全额计量）


class Caller(str, Enum):
    """每次模型调用的用途标记——三指标计量与消融分析按此分组。
    规划开销口径（吸收自团队 cloud-edge-agent）：caller ∈
    {CLASSIFY, DECOMPOSE, DEP_JUDGE, SUMMARIZE} 的云端 token 单独统计——
    DoT 论文未计量这笔开销，正是本项目要报告的差异点。
    """

    CLASSIFY = "classify"
    DECOMPOSE = "decompose"
    DEP_JUDGE = "dep_judge"
    SIGNAL = "signal"        # 难度信号试答（α-quantile/一致性采样）
    EXECUTE = "execute"
    VERIFY = "verify"
    RETRY = "retry"
    JUDGE = "judge"            # 隐私判别模式：云端在本地候选中挑选/修正
    SUMMARIZE = "summarize"
    EVIDENCE_SELECT = "evidence_select"
    EVIDENCE_ANSWER = "evidence_answer"
    EVIDENCE_FALLBACK = "evidence_fallback"
    WARMUP = "warmup"


@dataclass
class TaskFeatures:
    """L1 规则特征层输出的特征包（提取方法见 classifier/l1_rules.py）。"""

    has_tool_schema: bool = False
    action_verbs: int = 0
    constraint_markers: int = 0
    has_math_expr: bool = False
    num_count: int = 0        # 数字出现次数（数值密度，中文应用题常无显式运算符）
    arith_words: int = 0      # 算术关键词: 每/共/倍/平均/相差...
    has_options: bool = False
    question_words: int = 0
    n_tokens: int = 0
    deterministic: bool = False
    ctx_len_seg: str = "0-1k"


@dataclass
class Classification:
    label: Label
    conf: float
    source: str  # "L1" | "L2" | "fallback"
    features: TaskFeatures
    fast_path: bool = False  # True → 跳过分解/建图，本地直答+校验


@dataclass
class GenerationResult:
    """网关统一返回：文本 + token 概率（供 α-quantile）+ 计量字段。

    token_probs 为生成部分每个 token 的采样概率；真实后端不可用时为 None，
    此时 α-quantile 信号退化为仅多次采样一致性（见 orchestrator/signals.py）。
    """

    text: str
    token_probs: list[float] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    # 后端未回报 usage 时为 True：token 数不可信，计量侧记 NULL 不并入
    # 口径（known_subtotal + unknown_usage_calls 分列，不虚报不假报）
    usage_missing: bool = False


@dataclass
class SubTask:
    id: str  # s1, s2, ...
    description: str
    deps: list[str] = field(default_factory=list)
    assigned: ModelTier | None = None
    answer: str = ""
    status: str = "pending"  # pending / running / done / failed
    level: int = 0           # 拓扑层级：同层可并行，跨层串行


@dataclass
class SignalScores:
    """v1 统计信号路由的难度信号。

    alpha_quantile: 试答 token 概率的 α 分位数，越低越难（客观统计量，
    非模型自报置信度，符合申请书口径）。
    consistency: k=3 采样归一化答案的一致率，越低越难。
    exec_ok: 工具执行校验结果（TOOLUSE 域接入执行器后填充）。
    """

    alpha_quantile: float | None = None
    consistency: float | None = None
    exec_ok: bool | None = None

    def difficulty(self, w_alpha: float = 0.5, w_cons: float = 0.5) -> float | None:
        """组合为 0~1 难度分；无任何信号时返回 None（走策略默认倾向）。"""
        parts, weights = [], []
        if self.alpha_quantile is not None:
            parts.append(1.0 - self.alpha_quantile)
            weights.append(w_alpha)
        if self.consistency is not None:
            parts.append(1.0 - self.consistency)
            weights.append(w_cons)
        if self.exec_ok is not None:
            parts.append(0.0 if self.exec_ok else 1.0)
            weights.append(0.3)
        if not parts:
            return None
        return sum(p * w for p, w in zip(parts, weights)) / sum(weights)


@dataclass
class CallLog:
    """单次模型调用计量记录。"""

    caller: Caller
    tier: ModelTier
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    requested_tier: ModelTier = ModelTier.MD  # 与 tier 不同 = 云端降级到本地
    usage_missing: bool = False
    note: str = ""


@dataclass
class RunResult:
    query: str
    answer: str
    label: Label
    path: str  # cache_hit / fast_path / pipeline / direct_cloud
    trace: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # 结构化子任务（前端依赖图可视化）：id/description/deps/assigned/level/status/answer
    subtasks: list[dict] = field(default_factory=list)

    success: bool = True
    error: str | None = None
    provided_sources: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
