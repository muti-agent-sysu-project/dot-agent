"""Task Decomposer：本地 SLM + meta-prompt 将请求分解为编号子任务。

- 分解由本地模型执行（零云端 token），few-shot 样例按任务域维护在
  prompts/ 下，此处为通用占位样例。
- 输出强约束编号列表；解析失败重试 1 次，仍失败返回 None，
  由主流水线触发"整题升级云端"（保正确性下限）。
- 子任务数合法区间 [2, 8]：少于 2 无分解必要，多于 8 小模型分解质量
  先行退化（论文 Table 6 口径）。
"""
from __future__ import annotations

import re

from ..classifier.l2_semantic import DEFAULT_FEWSHOT
from ..gateway.base import ModelClient
from ..types import Caller, Label, SubTask

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
_DECOMPOSE_AND_CLASSIFY_PROMPT = """判断下面任务的类型，并把它分解为若干可独立求解的子问题。
类型只输出一行 "类型: 标签"（标签取 QA / REASONING / PLANNING / TOOLUSE 之一）；
分解只输出编号列表，每行 "数字. 子问题"，不要解释。

{fewshot}

示例:
任务: "小明有12个苹果, 分给3个朋友后每人又还给他2个, 现在小明有几个苹果?"
类型: REASONING
1. 分给3个朋友后小明剩几个苹果?
2. 朋友共还回多少个苹果?
3. 最终小明有多少个苹果?

任务: "{query}"
输出:"""

_LABEL_LINE_RE = re.compile(r"(?m)^\s*类型\s*[:：]\s*(QA|REASONING|PLANNING|TOOLUSE)\s*$",
                            re.IGNORECASE)
MIN_SUBTASKS, MAX_SUBTASKS = 2, 8


class TaskDecomposer:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    async def decompose(self, query: str) -> list[SubTask] | None:
        """返回子任务列表；分解/解析失败返回 None（触发整题升级）。"""
        prompt = DECOMPOSE_PROMPT.format(query=query)
        for attempt in range(2):  # 失败重试一次
            result = await self.client.generate(
                prompt, max_tokens=512, temperature=0.0, caller=Caller.DECOMPOSE
            )
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

    async def decompose_and_classify(
        self, query: str, fewshot: str = DEFAULT_FEWSHOT,
    ) -> tuple[Label | None, list[SubTask] | None]:
        """合并推理：一次本地调用同时产出任务类型与子任务列表。

        适用 L1 规则层不确定的请求——旧路径要 classify（一次推理）+
        decompose（又一次）两段串行，合并后省一整段本地时延。
        返回 (label, subtasks)；label 解析失败为 None，subtasks 失败为 None。
        """
        prompt = _DECOMPOSE_AND_CLASSIFY_PROMPT.format(fewshot=fewshot, query=query)
        for _ in range(2):
            result = await self.client.generate(
                prompt, max_tokens=640, temperature=0.0, caller=Caller.DECOMPOSE
            )
            label_match = _LABEL_LINE_RE.search(result.text)
            label = Label(label_match.group(1).upper()) if label_match else None
            items = [(int(m.group(1)), m.group(2).strip())
                     for m in _ITEM_RE.finditer(result.text)]
            if MIN_SUBTASKS <= len(items) <= MAX_SUBTASKS:
                seen: dict[int, str] = {}
                for no, desc in items:
                    seen.setdefault(no, desc)
                subtasks = [
                    SubTask(id=f"s{i}", description=desc)
                    for i, (_, desc) in enumerate(sorted(seen.items()), start=1)
                ]
                return label, subtasks
            if label is not None:  # 拿到了标签但分解失败：标签仍有价值
                return label, None
        return None, None
