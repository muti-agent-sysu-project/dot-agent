"""隐私执行模式：难度判定（本地/云端）× 敏感度 → 云端以什么身份参与。

四种模式：
  LOCAL_ONLY    难度判本地 → 原有行为，云端不参与；
  CLOUD_PLAIN   难度判云端且无敏感内容 → 原有行为，云端见原文；
  CLOUD_MASKED  难度判云端、敏感度为 LOW → 假名化后转交，答案本地回填；
  CLOUD_JUDGE   难度判云端、敏感度为 HIGH → 云端只见假名化问题与本地
                候选答案，从中挑选或修正，不接触完整上下文（黑盒接口
                下任务级投机执行：本地生成、云端判别）。

差异点（与既有工作对照）：
  RouteLLM/DoT/Hera/EdgeWise 只有难度一维，无披露概念；
  PRISM 敏感度在整句级别、单轮问答，无子任务粒度、无依赖图传播、无回填。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .pseudonymizer import EntityMap
from .sensitivity import Sensitivity, SensitivityReport, analyze
from .ner import analyze_with_ner


class ExecMode(str, Enum):
    LOCAL_ONLY = "local_only"
    CLOUD_PLAIN = "cloud_plain"
    CLOUD_MASKED = "cloud_masked"
    CLOUD_JUDGE = "cloud_judge"


@dataclass
class PrivacyLedger:
    """请求级隐私计量：与三指标并列的第四指标"披露"的数据来源。"""

    enabled: bool = False
    mode_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        (m.value for m in ExecMode), 0))
    entities_masked: int = 0
    entities_disclosed_plain: int = 0   # CLOUD_PLAIN 模式下明文发出的实体数
    leak_count: int = 0          # 假名化后仍出现在云端bound文本里的原文数（期望 0）
    roundtrip_failures: int = 0  # 回填一致性校验失败次数（假名残留/实体数不守恒）
    masked_cloud_calls: int = 0
    plain_cloud_calls: int = 0
    judge_calls: int = 0
    # 每次云端转交实际发出的文本（截断保存），面板"云端所见"列的数据源
    views: list[dict] = field(default_factory=list)

    def record(self, mode: ExecMode, disclosed_entities: int = 0) -> None:
        self.mode_counts[mode.value] += 1
        if mode is ExecMode.CLOUD_PLAIN:
            self.entities_disclosed_plain += disclosed_entities

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode_counts": dict(self.mode_counts),
            "entities_masked": self.entities_masked,
            "entities_disclosed_plain": self.entities_disclosed_plain,
            "roundtrip_failures": self.roundtrip_failures,
            "masked_cloud_calls": self.masked_cloud_calls,
            "plain_cloud_calls": self.plain_cloud_calls,
            "judge_calls": self.judge_calls,
            "leak_count": self.leak_count,
            "views": list(self.views),
        }


class PrivacyGuard:
    """每次请求新建一个（含独立的实体映射表）。DOT_PRIVACY=off 时直通。

    ner_backend: 可选 NER 后端（privacy/ner.py），None 或 off 时纯规则。
    """

    def __init__(self, enabled: bool = True, ner_backend=None) -> None:
        self.enabled = enabled
        self.ner_backend = ner_backend
        self.ledger = PrivacyLedger(enabled=enabled)
        self.entity_map = EntityMap() if enabled else None

    def classify(self, wants_cloud: bool, texts: list[str]) -> tuple[ExecMode, SensitivityReport]:
        """难度判定结果 + 参与判定的全部文本（子任务描述与前驱结论）→ 执行模式。

        texts 传多条是因为敏感度沿依赖图传播：前驱答案里的敏感实体会
        被拼进本次转交云端的上文，按最严一档处理。
        """
        report = SensitivityReport()
        for t in texts:
            report = report.merge(analyze_with_ner(t, self.ner_backend))
        if not self.enabled:
            mode = ExecMode.CLOUD_PLAIN if wants_cloud else ExecMode.LOCAL_ONLY
        elif not wants_cloud:
            mode = ExecMode.LOCAL_ONLY
        elif report.level is Sensitivity.HIGH:
            mode = ExecMode.CLOUD_JUDGE
        elif report.level is Sensitivity.LOW:
            mode = ExecMode.CLOUD_MASKED
        else:
            mode = ExecMode.CLOUD_PLAIN
        self.ledger.record(mode, disclosed_entities=len(report.entities))
        return mode, report

    def pseudonymize(self, text: str, report: SensitivityReport | None = None) -> str:
        if not self.enabled or self.entity_map is None:
            return text
        before = self.entity_map.size
        out = self.entity_map.pseudonymize(text, report)
        self.ledger.entities_masked += self.entity_map.size - before
        self.ledger.leak_count += self.entity_map.leak_check(out)
        return out

    def restore(self, text: str, *, original: str | None = None) -> str:
        """云端答案里的假名替换回原文；给了 original 就做回填一致性校验。"""
        if not self.enabled or self.entity_map is None:
            return text
        out = self.entity_map.restore(text)
        if original is not None:
            check = self.entity_map.verify_roundtrip([original], [out])
            if not check["consistent"]:
                self.ledger.roundtrip_failures += 1
        return out
