"""桩客户端的演示行为：无模型环境下演示完整路由流程。

按 prompt 内容模拟本地小模型各角色（分类/分解/依赖判断/直答/汇总），
工具任务由 tools.py 的 mock 沙箱真实执行（答案真实可校验）。
"""
from __future__ import annotations

import re
import json

from ..orchestrator.evidence import REQUEST_MARKER, select_packet

from ..types import Caller

_MATH_DECOMPOSE = """1. 求第一问的数值
2. 求第二问的数值
3. 汇总两个结果"""
_PLAN_DECOMPOSE = """1. 确定总预算
2. 安排每日行程
3. 核算总花费"""
_TOOL_DEP = "s1 -> s2"
_MATH_DEP = "s1 -> s3\ns2 -> s3"


def _task_line(prompt: str) -> str:
    # 取最后一条任务行：分解 prompt 中第一条是 few-shot 示例的任务，真实 query 在末尾
    lines = [l for l in prompt.splitlines() if l.startswith("任务:") or l.startswith("原任务:")]
    return lines[-1] if lines else prompt


def local_demo_behavior(prompt: str, caller: Caller) -> str | None:
    if caller is Caller.EVIDENCE_SELECT:
        payload = json.loads(prompt.split(REQUEST_MARKER, 1)[1])
        packet = select_packet(payload["query"], payload["sources"])
        indices = [i for i, source in enumerate(payload["sources"])
                   if source["source_id"] in packet.source_ids]
        return json.dumps({"evidence_indices": indices})
    task = _task_line(prompt)

    if caller is Caller.CLASSIFY:
        if re.search(r"天气|提醒|计算|发送", task):
            return "TOOLUSE"
        if re.search(r"预算|以内|不超过|至少", task):
            return "PLANNING"
        return "REASONING"

    if "依赖关系" in prompt:
        return _TOOL_DEP if "天气" in task or "提醒" in task else _MATH_DEP

    if "分解为若干" in prompt:
        if re.search(r"天气|提醒|计算", task):
            m = re.search(r"(?:查询|看看|获取)?([一-龥]{2,4}?)(?:明天的)?天气", task)
            city = m.group(1) if m else "北京"
            return f"1. 查询{city}明天的天气\n2. 发送提醒给同事"
        if re.search(r"预算|以内|不超过|至少", task):
            return _PLAN_DECOMPOSE
        return _MATH_DECOMPOSE

    if "原任务" in prompt:
        return "【演示汇总】两问均已求解，答案见各子任务。"

    # 直答 / 信号试答 / 一致性采样：返回固定答案（stub 模式 Acc 无意义，流程真实）
    return "42"


def cloud_demo_behavior(prompt: str, caller: Caller) -> str | None:
    if caller in (Caller.EVIDENCE_ANSWER, Caller.EVIDENCE_FALLBACK):
        payload = json.loads(prompt.split(REQUEST_MARKER, 1)[1])
        source = select_packet(payload["query"], payload["sources"], max_sources=1).sources[0]
        return json.dumps({"answer": "【模拟回答】" + source["text"][:240],
                           "citations": [source["source_id"]]}, ensure_ascii=False)
    head = prompt.splitlines()[0][:36]
    return f"【云端解答】{head}… 答案: 42（stub 演示输出）"
