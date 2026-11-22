"""Task Scheduler：依赖判断 → DAG → 拓扑分层 → 逐层并行执行。

- 依赖判断由本地 SLM 完成（零云端 token），输出 "s1 -> s3" 行；
- 分层用标准拓扑层级（level(n) = 0 若无前驱，否则 max(level(前驱))+1），
  与论文"按深度分批、同批并行"语义一致；
- 环 / 解析失败 → 保守退化为线性链（等价于 w/o Graph 消融组）；
- 每个子任务的执行上下文只注入其直接前驱的答案（上下文裁剪落点）。
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from ..gateway.base import ModelClient
from ..types import Caller, SubTask

DEP_JUDGE_PROMPT = """任务: "{query}" 已分解为以下子问题:
{subtask_list}

请判断子问题之间的依赖关系：若子问题 A 的结果是子问题 B 的必要输入，
则 A 必须先完成。只输出依赖行，格式严格为 "s1 -> s3"，每行一条，无依赖则输出空。

依赖关系:"""

_EDGE_RE = re.compile(r"\b(s\d+)\s*->\s*(s\d+)\b")

# 执行器签名：接收子任务与其前驱答案 {subtask_id: answer}，返回最终答案文本
Executor = Callable[[SubTask, dict[str, str]], Awaitable[str]]


class TaskScheduler:
    def __init__(self, client: ModelClient, judge_client: ModelClient | None = None) -> None:
        self.client = client
        # DoT 把依赖分析视为规划关键步骤：默认交给更强的一侧（云端），
        # 图错了会让关键步骤识别和后续执行整体偏航。
        self.judge_client = judge_client or client

    async def judge_deps(self, query: str, subtasks: list[SubTask]) -> list[tuple[str, str]] | None:
        listing = "\n".join(f"{st.id}: {st.description}" for st in subtasks)
        prompt = DEP_JUDGE_PROMPT.format(query=query, subtask_list=listing)
        result = await self.judge_client.generate(
            prompt, max_tokens=256, temperature=0.0, caller=Caller.DEP_JUDGE
        )
        edges = list(dict.fromkeys(_EDGE_RE.findall(result.text.replace("→", "->"))))
        # [(a, b), ...] —— findall 对两个分组返回 tuple
        return edges if edges else None

    @staticmethod
    def key_steps(subtasks: list[SubTask], edges: list[tuple[str, str]] | None) -> set[str]:
        """返回依赖图汇点（无后继）的 id；图无效/成环时返回空集。

        关键步骤由后续流水线交给云端执行。只在依赖图可信时启用，
        避免分解退化或解析噪声把整题误判成关键步骤。
        """
        by_id = {st.id for st in subtasks}
        valid = [(a, b) for a, b in (edges or [])
                 if a in by_id and b in by_id and a != b]
        if not valid:
            return set()
        indeg = {i: 0 for i in by_id}
        succs: dict[str, set[str]] = {i: set() for i in by_id}
        for a, b in valid:
            succs[a].add(b)
            indeg[b] += 1
        queue = [i for i, d in indeg.items() if d == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for nxt in succs[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if visited != len(by_id):  # 有环，退化线性链时不标关键步骤
            return set()
        has_succ = {a for a, _ in valid}
        return {i for i in by_id if i not in has_succ}

    @staticmethod
    def build_levels(
        subtasks: list[SubTask], edges: list[tuple[str, str]] | None
    ) -> list[list[SubTask]]:
        """拓扑分层；edges 为 None 或含环时退化为线性链。"""
        by_id = {st.id: st for st in subtasks}
        if not edges:
            return [[st] for st in subtasks]

        preds: dict[str, set[str]] = {st.id: set() for st in subtasks}
        succs: dict[str, set[str]] = {st.id: set() for st in subtasks}
        for a, b in edges:
            if a in by_id and b in by_id and a != b:
                preds[b].add(a)
                succs[a].add(b)

        level: dict[str, int] = {}
        def compute(n: str, visiting: set[str]) -> int | None:
            if n in visiting:           # 环检测
                return None
            if n in level:
                return level[n]
            visiting.add(n)
            best = 0
            for p in preds[n]:
                v = compute(p, visiting)
                if v is None:
                    return None
                best = max(best, v + 1)
            visiting.discard(n)
            level[n] = best
            return best

        for st in subtasks:
            if compute(st.id, set()) is None:
                return [[s] for s in subtasks]  # 有环 → 线性链

        for st in subtasks:
            st.deps = sorted(preds[st.id])
            st.level = level[st.id]
        buckets: dict[int, list[SubTask]] = {}
        for st in subtasks:
            buckets.setdefault(st.level, []).append(st)
        return [buckets[k] for k in sorted(buckets)]

    async def run(
        self, query: str, subtasks: list[SubTask], executor: Executor
    ) -> dict[str, str]:
        edges = await self.judge_deps(query, subtasks)
        levels = self.build_levels(subtasks, edges)
        # DoT 的关键步骤识别：依赖图汇点没有后继，其答案直接进入最终汇总。
        key_ids = self.key_steps(subtasks, edges)
        for st in subtasks:
            st.key = st.id in key_ids
        answers: dict[str, str] = {}
        for batch in levels:  # 跨层串行，同层并行
            results = await asyncio.gather(
                *(executor(st, answers) for st in batch)
            )
            for st, ans in zip(batch, results):
                answers[st.id] = ans
                st.answer = ans
                st.status = "done"
        return answers
