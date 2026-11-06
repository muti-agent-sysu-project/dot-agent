"""隐私模块：敏感度判定、假名化回填、执行模式组合、云端判别。

对编排层的唯一入口是 PrivacyGuard；DOT_PRIVACY=off 时全部直通，
行为与旧版完全一致（消融实验的对照组）。
"""
from .policy import ExecMode, PrivacyGuard, PrivacyLedger
from .sensitivity import Sensitivity, SensitivityReport, analyze
from .pseudonymizer import EntityMap, mask_prompt
from .judge import cloud_judge, build_judge_prompt, parse_judge_reply
from .ner import (
    NerBackend, RuleBackend, TransformersBackend, PresidioBackend,
    get_backend, analyze_with_ner,
)

__all__ = [
    "ExecMode", "PrivacyGuard", "PrivacyLedger",
    "Sensitivity", "SensitivityReport", "analyze",
    "EntityMap", "mask_prompt", "cloud_judge",
    "build_judge_prompt", "parse_judge_reply",
    "NerBackend", "RuleBackend", "TransformersBackend", "PresidioBackend",
    "get_backend", "analyze_with_ner",
]
