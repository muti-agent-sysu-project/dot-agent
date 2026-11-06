"""敏感度判定：子任务文本 → 敏感等级 + 命中的敏感实体。

规则层（本模块）做两件事：
  1. 直接标识符：手机号、身份证、银行卡、邮箱、住址、凭据——命中即 HIGH；
  2. 敏感域关键词：病历、诊断、工资、涉案等——命中即 HIGH（语境敏感）。
日期等准标识符记 LOW（单独不足以定位自然人，组合起来可以）。

难度信号决定"要不要云端帮忙"，敏感度决定"云端以什么身份帮忙"——
两信号在 privacy/policy.py 组合成执行模式。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Sensitivity(str, Enum):
    NONE = "none"    # 无敏感内容：云端可见原文
    LOW = "low"      # 准标识符（日期等）：转交前假名化
    HIGH = "high"    # 直接标识符或敏感域：判别模式或仅本地


_LEVEL_RANK = {Sensitivity.NONE: 0, Sensitivity.LOW: 1, Sensitivity.HIGH: 2}

# (模式, 实体类型, 等级)
_PATTERNS: list[tuple[re.Pattern[str], str, Sensitivity]] = [
    # 直接标识符 —— HIGH
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "手机号", Sensitivity.HIGH),
    (re.compile(
        r"(?<!\d)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
    ), "身份证号", Sensitivity.HIGH),
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "银行卡号", Sensitivity.HIGH),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "邮箱", Sensitivity.HIGH),
    (re.compile(r"(?i)(api[_-]?key|secret|token|密码|口令)\s*[:：=]\s*\S{6,}"), "凭据", Sensitivity.HIGH),
    (re.compile(r"[\u4e00-\u9fa5]{1,4}(?:省|市|区|县)(?:[\u4e00-\u9fa5]{1,6}(?:路|街|道)){1,2}\d+号"), "住址", Sensitivity.HIGH),
    # 准标识符 —— LOW
    (re.compile(r"(?<!\d)(?:19|20)\d{2}年\d{1,2}月\d{1,2}日(?!\d)"), "日期", Sensitivity.LOW),
    (re.compile(r"(?<!\d)(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])(?!\d)"), "日期", Sensitivity.LOW),
]

# 敏感域关键词：命中任一即 HIGH（语境敏感，即使无实体）
DOMAIN_KEYWORDS: list[str] = [
    "病历", "诊断", "处方", "患者", "病情",
    "工资", "薪金", "存款", "贷款", "负债", "征信",
    "涉案", "嫌疑人", "被告", "案卷",
    "社保号", "护照号",
]


@dataclass
class SensitivityReport:
    level: Sensitivity = Sensitivity.NONE
    entities: list[tuple[str, str]] = field(default_factory=list)  # (文本, 实体类型)

    def merge(self, other: "SensitivityReport") -> "SensitivityReport":
        """敏感度取更严的一档，实体并集。

        依赖图传播用：子任务携带的前驱结论有敏感内容时，整条链按最严处理。
        """
        level = (self.level if _LEVEL_RANK[self.level] >= _LEVEL_RANK[other.level]
                 else other.level)
        seen = {e[0] for e in self.entities}
        entities = self.entities + [e for e in other.entities if e[0] not in seen]
        return SensitivityReport(level, entities)


def analyze(text: str) -> SensitivityReport:
    """规则层敏感度判定。LLM 辅助判定后续版本再加，规则层先行保证可审计。"""
    entities: list[tuple[str, str]] = []
    level = Sensitivity.NONE
    for pattern, etype, elvl in _PATTERNS:
        found = pattern.findall(text)
        if found:
            entities.extend((m if isinstance(m, str) else m[0], etype) for m in found)
            if elvl is Sensitivity.HIGH:
                level = Sensitivity.HIGH
            elif level is Sensitivity.NONE:
                level = Sensitivity.LOW
    if level is not Sensitivity.HIGH and any(kw in text for kw in DOMAIN_KEYWORDS):
        level = Sensitivity.HIGH
    # 去重（同一实体被多个模式命中时保留首次）
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for text_, etype in entities:
        if text_ not in seen:
            seen.add(text_)
            uniq.append((text_, etype))
    return SensitivityReport(level, uniq)
