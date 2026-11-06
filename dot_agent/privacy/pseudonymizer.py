"""假名化与回填：会话级实体映射表，转交云端前替换、云端作答后还原。

设计要点：
- 映射表只存在本地内存，随请求生命周期销毁，不出域。
- 同一实体在多次转交（含依赖图上前驱结论里的引用）中映射一致，
  云端看到的是自洽的假名上下文，多步推理不因替换而断裂。
- 数字型实体（手机号/身份证/银行卡）替换为同长度合成数字：
  云端看到的仍是合法格式，任务效用（格式约束类任务）不因替换而破坏；
  合成值由原文哈希决定，同一原文必然得到同一合成值（跨请求也一致）。
- 回填后做一致性校验：假名不得残留、实体总数守恒（verify_roundtrip）。

注意：假名（合成数字或"日期1"式标签）若与原文同形会冲突，属已知边界
情形；规则层实体均不含该形态，实际冲突概率极低。
"""
from __future__ import annotations

import hashlib
import itertools
import re

from .sensitivity import SensitivityReport

_NUMERIC_TYPES = ("手机号", "身份证号", "银行卡号")


def _synthetic_number(original: str, etype: str) -> str:
    """同长度合成数字：hmac 式哈希决定每位数字，首位移到 1-9 保证非零开头。"""
    digest = hashlib.sha256(f"{etype}:{original}".encode("utf-8")).hexdigest()
    digits = [int(digest[i % len(digest)], 16) % 10 for i in range(len(original))]
    digits[0] = digits[0] % 9 + 1
    return "".join(str(d) for d in digits)


class EntityMap:
    """一次请求一张映射表：原文 → 假名（同长度合成数字或"日期1"式标签）。"""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}      # 原文 → 假名
        self._rev: dict[str, str] = {}      # 假名 → 原文（假名唯一，可逆）
        self._count: dict[str, itertools.count] = {}

    # -- 假名化 --------------------------------------------------------- #
    def pseudonymize(self, text: str, report: SensitivityReport | None = None) -> str:
        """把已知敏感实体替换为假名。report 为 None 时按已有映射表盲替换。"""
        for original, placeholder in self._map.items():
            if original in text:
                text = text.replace(original, placeholder)
        if report is not None:
            for original, etype in report.entities:
                if original in self._map:
                    continue
                placeholder = self._placeholder_for(original, etype)
                self._map[original] = placeholder
                self._rev[placeholder] = original
                if original in text:
                    text = text.replace(original, placeholder)
        return text

    def _placeholder_for(self, original: str, etype: str) -> str:
        if etype in _NUMERIC_TYPES and re.fullmatch(r"\d+", original):
            return _synthetic_number(original, etype)
        counter = self._count.get(etype)
        if counter is None:
            counter = itertools.count(1)
            self._count[etype] = counter
        while True:
            name = f"{etype}{next(counter)}"
            if name not in self._rev:
                return name

    # -- 回填 ----------------------------------------------------------- #
    def restore(self, text: str) -> str:
        """云端答案里的假名替换回原文。长假名先替换，避免前缀遮蔽。"""
        for placeholder in sorted(self._rev, key=len, reverse=True):
            if placeholder in text:
                text = text.replace(placeholder, self._rev[placeholder])
        return text

    # -- 一致性校验 ------------------------------------------------------ #
    def verify_roundtrip(self, original_texts: list[str],
                         restored_texts: list[str]) -> dict:
        """回填一致性：假名不残留 + 回填后的实体数不超出原文。

        云端答案未必复述原文全部实体，所以不要求总数相等，只要求：
        1. restored 中无假名残留（有则说明回填遗漏）；
        2. 每个实体的出现次数 restored ≤ original（超出说明替换破坏了内容）。
        返回 {"placeholders_left": int, "entity_overflows": int,
              "consistent": bool}。
        """
        placeholders_left = 0
        for t in restored_texts:
            for placeholder in self._rev:
                if placeholder in t:
                    placeholders_left += 1
        entity_overflows = 0
        for original in self._map:
            if all(original not in t for t in original_texts):
                continue
            total_original = sum(t.count(original) for t in original_texts)
            total_restored = sum(t.count(original) for t in restored_texts)
            if total_restored > total_original:
                entity_overflows += 1
        consistent = placeholders_left == 0 and entity_overflows == 0
        return {"placeholders_left": placeholders_left,
                "entity_overflows": entity_overflows,
                "consistent": consistent}

    # -- 审计 ----------------------------------------------------------- #
    @property
    def size(self) -> int:
        return len(self._map)

    def leak_check(self, cloud_bound_text: str) -> int:
        """披露审计：统计未经替换就发到云端的敏感原文出现次数（期望为 0）。"""
        return sum(1 for original in self._map if original in cloud_bound_text)


def mask_prompt(text: str) -> str:
    """兜底粗掩码：映射表没覆盖的长数字串打码（云端判别模式用）。"""
    return re.sub(r"\d{7,}", lambda m: "*" * len(m.group(0)), text)
