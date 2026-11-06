"""分类编排：L1 规则层短路 → L2 语义层兜底。

L1 高置信（≥0.90）直接采纳：QA 标记快速通道，其余类型直达策略表，
省掉一次 L2 推理；L1 不确定才调 L2。L3 运行时重分类由主流水线
（pipeline.py）在执行反馈处触发，不在此层。
"""
from __future__ import annotations

from .l1_rules import classify_l1, extract_features, load_rules
from .l2_semantic import SemanticClassifier
from ..types import Classification, Label


class ClassificationPipeline:
    """两个阈值分离：
    - accept_threshold (0.85)：L1 标签被采纳的下限，达标即省一次 L2 推理；
    - fast_conf_threshold (0.95)：QA 直答快速通道要求更高（如选项结构题），
      未达标的 QA 仍直答（POLICY.QA.decompose=False），但 trace 中不标 fast_path。
    """

    def __init__(
        self,
        l2: SemanticClassifier,
        rules: dict | None = None,
        accept_threshold: float = 0.85,
        fast_conf_threshold: float = 0.95,
    ) -> None:
        self.l2 = l2
        self.rules = rules or load_rules()
        self.accept_threshold = accept_threshold
        self.fast_conf_threshold = fast_conf_threshold

    def classify_l1_stage(self, query: str) -> Classification | None:
        """只跑 L1 规则层（零推理成本）；L1 不确定时返回 None。

        None 表示"需要语义层"，由调用方决定走 L2 单独分类还是与分解
        合并推理（TaskDecomposer.decompose_and_classify）——合并可省
        一整段本地推理时延。
        """
        features = extract_features(query, self.rules)
        label, conf = classify_l1(features, self.rules)
        if label is not None and conf >= self.accept_threshold:
            return Classification(
                label=label, conf=conf, source="L1", features=features,
                fast_path=(label is Label.QA and conf >= self.fast_conf_threshold),
            )
        return None

    async def classify(self, query: str) -> Classification:
        features = extract_features(query, self.rules)
        l1 = self.classify_l1_stage(query)
        if l1 is not None:
            return l1
        label2 = await self.l2.classify(query)
        return Classification(label=label2, conf=0.60, source="L2", features=features)
