"""可插拔 NER 敏感度层：规则层为底，本地模型后端可选叠加。

后端协议：detect(text) -> list[(实体文本, 实体类型, HIGH/LOW)]
  - RuleBackend       sensitivity.py 的正则（默认，永远可用，可审计）
  - TransformersBackend  HuggingFace token-classification 模型本地推理
    （默认指向 OpenMed-PII-Chinese 系，278M 参数，Qwen3-4B 同卡可共存；
     transformers 未安装或模型加载失败时自动跳过，不影响主流程）
  - PresidioBackend   微软 Presidio（基线对比用，未安装则跳过）

融合原则：并集去重；任一来源判 HIGH 即 HIGH。
模型推理放在子进程外层由调用方保证不阻塞事件循环（detect 是同步接口，
TransformersBackend 内部用线程池封装）。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .sensitivity import Sensitivity, SensitivityReport, analyze

# 模型标签 → 本项目实体类型/等级 的映射（OpenMed 中文 PII 模型的常见标签）
_LABEL_MAP: dict[str, tuple[str, Sensitivity]] = {
    "PER": ("人名", Sensitivity.HIGH),
    "PERSON": ("人名", Sensitivity.HIGH),
    "NAME": ("人名", Sensitivity.HIGH),
    "PHONE": ("手机号", Sensitivity.HIGH),
    "PHONE_NUMBER": ("手机号", Sensitivity.HIGH),
    "ID": ("身份证号", Sensitivity.HIGH),
    "ID_CARD": ("身份证号", Sensitivity.HIGH),
    "NATIONAL_ID": ("身份证号", Sensitivity.HIGH),
    "EMAIL": ("邮箱", Sensitivity.HIGH),
    "ADDRESS": ("住址", Sensitivity.HIGH),
    "LOCATION": ("住址", Sensitivity.HIGH),
    "BANK": ("银行卡号", Sensitivity.HIGH),
    "BANK_ACCOUNT": ("银行卡号", Sensitivity.HIGH),
    "DATE": ("日期", Sensitivity.LOW),
    "BIRTHDAY": ("日期", Sensitivity.LOW),
}


@dataclass
class NerFinding:
    text: str
    etype: str
    level: Sensitivity


class NerBackend:
    def detect(self, text: str) -> list[NerFinding]:  # pragma: no cover - 接口
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError  # pragma: no cover


class RuleBackend(NerBackend):
    """正则后端：直接复用 sensitivity.analyze 的实体部分。"""

    @property
    def name(self) -> str:
        return "rules"

    def detect(self, text: str) -> list[NerFinding]:
        report = analyze(text)
        return [NerFinding(t, etype, Sensitivity.HIGH if etype != "日期" else Sensitivity.LOW)
                for t, etype in report.entities]


class TransformersBackend(NerBackend):
    """HuggingFace token-classification 后端（OpenMed-PII-Chinese 等）。

    模型通过 DOT_NER_MODEL 指定（HF repo id 或本地路径）；
    聚合策略用 simple（每个实体 span 单独报），置信阈值 DOT_NER_THRESHOLD。
    """

    def __init__(self, model_name: str | None = None, threshold: float | None = None):
        self.model_name = model_name or os.environ.get(
            "DOT_NER_MODEL", "OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1")
        try:
            self.threshold = float(threshold if threshold is not None
                                   else os.environ.get("DOT_NER_THRESHOLD", "0.5"))
        except ValueError:
            self.threshold = 0.5
        self._pipeline = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._failed = False

    @property
    def name(self) -> str:
        return f"transformers:{self.model_name}"

    def _load(self):
        if self._pipeline is None and not self._failed:
            try:
                from transformers import pipeline as hf_pipeline
                self._pipeline = hf_pipeline(
                    "token-classification", model=self.model_name,
                    aggregation_strategy="simple", device=-1)
            except Exception:
                # 模型缺失/依赖未装：标记失败，后续调用不再尝试（规则层兜底）
                self._failed = True
        return self._pipeline

    def detect(self, text: str) -> list[NerFinding]:
        pipe = self._load()
        if pipe is None:
            return []
        try:
            spans = self._pool.submit(pipe, text[:4000]).result(timeout=60)
        except Exception:
            return []
        findings: list[NerFinding] = []
        for span in spans or []:
            if float(span.get("score", 0.0)) < self.threshold:
                continue
            etype, level = _LABEL_MAP.get(
                str(span.get("entity_group", "")).upper(),
                ("实体", Sensitivity.LOW))
            entity = str(span.get("word", "")).strip()
            if entity:
                findings.append(NerFinding(entity, etype, level))
        return findings


class PresidioBackend(NerBackend):
    """微软 Presidio 后端（基线对比用）。中文支持有限，评测时如实记录。"""

    def __init__(self, language: str = "zh"):
        self.language = language
        self._analyzer = None
        self._failed = False

    @property
    def name(self) -> str:
        return f"presidio:{self.language}"

    def _load(self):
        if self._analyzer is None and not self._failed:
            try:
                from presidio_analyzer import AnalyzerEngine
                self._analyzer = AnalyzerEngine()
            except Exception:
                self._failed = True
        return self._analyzer

    def detect(self, text: str) -> list[NerFinding]:
        analyzer = self._load()
        if analyzer is None:
            return []
        try:
            results = analyzer.analyze(text=text[:4000], language="en", return_decision_process=False)
        except Exception:
            return []
        etype_map = {"PHONE_NUMBER": "手机号", "CREDIT_CARD": "银行卡号",
                     "IBAN_CODE": "银行卡号", "EMAIL_ADDRESS": "邮箱",
                     "PERSON": "人名", "URL": "实体", "IP_ADDRESS": "实体"}
        findings: list[NerFinding] = []
        for r in results or []:
            if r.score < 0.5:
                continue
            etype, level = etype_map.get(r.entity_type, ("实体", Sensitivity.HIGH))
            findings.append(NerFinding(text[r.start:r.end], etype, level))
        return findings


def get_backend(name: str | None = None) -> NerBackend:
    """按 DOT_NER_BACKEND 选择后端：rules（默认）/ transformers / presidio / off。"""
    chosen = (name or os.environ.get("DOT_NER_BACKEND", "rules")).lower()
    if chosen == "transformers":
        return TransformersBackend()
    if chosen == "presidio":
        return PresidioBackend()
    return RuleBackend()


def analyze_with_ner(text: str, backend: NerBackend | None) -> SensitivityReport:
    """规则层 + NER 后端融合：实体并集，等级从严。

    后端为 None（off）或 detect 抛异常时退化为纯规则，主流程不受影响。
    """
    report = analyze(text)
    if backend is None:
        return report
    try:
        findings = backend.detect(text)
    except Exception:
        return report
    for f in findings:
        extra = SensitivityReport(
            level=f.level, entities=[(f.text, f.etype)])
        report = report.merge(extra)
    return report
