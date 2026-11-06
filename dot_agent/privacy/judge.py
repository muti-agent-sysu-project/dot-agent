"""云端判别模式：本地多候选，云端挑选或修正（不生成完整上下文）。

难度信号采样（k=3 一致性）本来就会产出多个候选答案——一份数据两用：
采样既测难度，又当判别候选。云端收到的只有假名化问题与编号候选，
相对完整转交的泄露面最小；这是黑盒接口下任务级投机执行的落地形态
（token 级投机解码需要白盒概率分布，商用接口给不了）。
"""
from __future__ import annotations

import re

from ..types import Caller
from ..gateway.base import ModelClient

JUDGE_PROMPT = """你只能看到假名化后的问题和若干候选答案，看不到原始上下文。
请从候选中选出最可靠的一条；若全部有误，给出简短修正（不要展开推理）。
只输出所选候选的编号（如"2"），或"修正: <一句话>"。

问题: {question}

候选:
{candidates}
"""


def build_judge_prompt(question_masked: str, candidates: list[str]) -> str:
    body = "\n".join(f"{i}. {c.strip()[:500]}" for i, c in enumerate(candidates, 1))
    return JUDGE_PROMPT.format(question=question_masked, candidates=body)


def parse_judge_reply(reply: str, candidates: list[str]) -> tuple[str, bool]:
    """解析云端判别结果。

    返回 (答案, 是否成功选中)。识别"2" / "候选2" / "选2" / "修正: xxx"；
    解析失败返回 ("", False)，由调用方退回假名化生成。
    """
    text = reply.strip()
    # 数字边界（(?<!\d)(?!\d)）：防止 "42" 被读成候选 4、"12" 被读成候选 1
    m = re.search(r"(?:候选|选|答案)?\s*(?<!\d)([1-9])(?!\d)\s*号?", text)
    if m and not text.startswith("修正"):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx], True
    m = re.search(r"修正\s*[:：]\s*(.+)", text, re.S)
    if m:
        return m.group(1).strip(), True
    return "", False


async def cloud_judge(
    cloud: ModelClient, question_masked: str, candidates: list[str], *,
    max_tokens: int = 256,
) -> tuple[str, bool]:
    """执行一次判别调用。失败/解析不出 → (候选中位数答案, False) 由调用方降级。"""
    prompt = build_judge_prompt(question_masked, candidates)
    reply = (await cloud.generate(
        prompt, max_tokens=max_tokens, temperature=0.0, caller=Caller.JUDGE
    )).text
    return parse_judge_reply(reply, candidates)
