"""Task Decomposer：云端/本地 SLM + meta-prompt 将请求分解为编号子任务。

- 默认本地分解（省云端 token）；设置 DOT_DECOMPOSE_TIER=cloud 可切到云端。
  2026-11-22 对照：云端分解在 10 题 math 上准确率不变，控制组 18 题不变，
  但云端 token 约翻倍，因此默认仍保留本地。
- 输出强约束编号列表；解析失败重试 1 次，仍失败返回 None，
  由主流水线触发"整题升级云端"（保正确性下限）。
- 子任务数合法区间 [2, 8]：少于 2 无分解必要，多于 8 小模型分解质量
  先行退化（论文 Table 6 口径）。
"""
from __future__ import annotations

import re

from ..gateway.base import ModelClient
from ..types import Caller, SubTask

DECOMPOSE_PROMPT = """请把下面的任务分解为若干可独立求解的子问题。
要求：每个子问题简洁、清晰、相互独立；只输出编号列表，不要解释。

示例:
任务: "小明有12个苹果, 分给3个朋友后每人又还给他2个, 现在小明有几个苹果?"
1. 分给3个朋友后小明剩几个苹果?
2. 朋友共还回多少个苹果?
3. 最终小明有多少个苹果?

任务: "{query}"
输出:"""

_ITEM_RE = re.compile(r"(?m)^\s*(\d+)[\.、\)]\s*(\S.*)$")
MIN_SUBTASKS, MAX_SUBTASKS = 2, 8


class TaskDecomposer:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    async def decompose(self, query: str) -> list[SubTask] | None:
        """返回子任务列表；分解/解析失败返回 None（触发整题升级）。"""
        prompt = DECOMPOSE_PROMPT.format(query=query)
        for attempt in range(2):  # 失败重试一次
            try:
                result = await self.client.generate(
                    prompt, max_tokens=512, temperature=0.0, caller=Caller.DECOMPOSE
                )
            except RuntimeError:
                # 截断/空答/连接失败：重试一次，仍失败交给整题云端回退。
                continue
            items = [(int(m.group(1)), m.group(2).strip()) for m in _ITEM_RE.finditer(result.text)]
            if MIN_SUBTASKS <= len(items) <= MAX_SUBTASKS:
                # 去重（同编号取首个）并按编号排序
                seen: dict[int, str] = {}
                for no, desc in items:
                    seen.setdefault(no, desc)
                return [
                    SubTask(id=f"s{i}", description=desc)
                    for i, (_, desc) in enumerate(sorted(seen.items()), start=1)
                ]
        return None
