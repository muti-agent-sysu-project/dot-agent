"""阈值标定：在数据集上扫描各任务类型的分配阈值 θ，产出 config/router_theta.json。

用法（本地无真实模型也能跑，用 --local/--cloud 指向可用端点或 stub）：
  python3 scripts/calibrate_theta.py --dataset eval/data/theta_dev.jsonl \
      --local http://127.0.0.1:18000/v1 --cloud-env

数据集格式（jsonl，每行）：
  {"query": "...", "label": "QA|REASONING|PLANNING|TOOLUSE",
   "answer": "标准答案或判分关键词（可选，缺省判本地答案非空即对）"}

标定逻辑：
  对每条样本收集难度分 score（SignalCollector）与本地答案正确性 correct。
  对候选 θ 扫描：score > θ 时假设转云端（按 correct=1 计，成本记 1），
  否则本地（correct 如实，成本 0）。目标是单位成本的正确率最大化：
      theta* = argmax_θ  P(correct | policy(θ)) - λ · P(cloud | policy(θ))
  λ 默认 0.3（可调）；质量优先时可调小。

诚实边界：本脚本只做离线扫描，不连线上服务；结果写入
config/router_theta.json 后由 Router 自动加载。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.classifier.pipeline import ClassificationPipeline  # noqa: E402
from dot_agent.classifier.l2_semantic import SemanticClassifier  # noqa: E402
from dot_agent.gateway.base import ModelClient  # noqa: E402
from dot_agent.gateway.openai_compat import OpenAICompatClient  # noqa: E402
from dot_agent.orchestrator.router import POLICY, THETA_CONFIG, Label  # noqa: E402
from dot_agent.orchestrator.signals import SignalCollector  # noqa: E402
from dot_agent.types import ModelTier  # noqa: E402

THETAS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ~ 0.95


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row.get("query"), str):
            continue
        try:
            row["label"] = Label(row.get("label", ""))
        except ValueError:
            row["label"] = None  # 未标 label 的样本交给分类器兜底
        rows.append(row)
    return rows


def judge_correct(query: str, answer: str, reference: str | None) -> bool:
    """本地判分：有参考答案做包含匹配，没有则答案非空即算对（保守低估成本）。"""
    if not answer or not answer.strip():
        return False
    if reference:
        return reference.strip() in answer
    return True


async def collect_scores(local: ModelClient, rows: list[dict]) -> list[dict]:
    classifier = ClassificationPipeline(SemanticClassifier(local))
    signals = SignalCollector(local)
    out = []
    for row in rows:
        label = row.get("label")
        if label is None:
            cls = await classifier.classify(row["query"])
            label = cls.label
        sig, greedy = await signals.collect(row["query"])
        score = sig.difficulty() if sig else None
        correct = judge_correct(row["query"], greedy.text if greedy else "",
                                row.get("answer"))
        out.append({"query": row["query"], "label": label,
                    "score": score, "correct_local": correct})
    return out


def scan(samples: list[dict], lam: float) -> dict[str, float]:
    """按类型扫描 θ：目标 = 正确率 − λ × 云端率。score=None 样本不计入。"""
    best: dict[str, float] = {}
    for label in POLICY:
        group = [s for s in samples if s["label"] is label and s["score"] is not None]
        if len(group) < 10:  # 样本太少不可靠，保留默认 θ
            continue
        best_score, best_theta = None, POLICY[label].theta
        for theta in THETAS:
            cloud = [s for s in group if s["score"] > theta]
            local = [s for s in group if s["score"] <= theta]
            # 云端假设全对（上界），本地按实测正确率
            accuracy = (len(cloud) + sum(1 for s in local if s["correct_local"])) / len(group)
            cloud_rate = len(cloud) / len(group)
            objective = accuracy - lam * cloud_rate
            if best_score is None or objective > best_score:
                best_score, best_theta = objective, theta
        best[label.value] = best_theta
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="标定各任务类型的分配阈值 θ")
    parser.add_argument("--dataset", required=True, help="jsonl 数据集路径")
    parser.add_argument("--local", default="http://127.0.0.1:18000/v1",
                        help="本地模型 OpenAI 兼容端点（llama.cpp）")
    parser.add_argument("--lam", type=float, default=0.3, help="云端成本权重 λ")
    parser.add_argument("--output", default=str(THETA_CONFIG))
    args = parser.parse_args()

    rows = load_dataset(Path(args.dataset))
    if not rows:
        print("数据集为空或格式不对", file=sys.stderr)
        return 1
    local = OpenAICompatClient(base_url=args.local, api_key="none",
                               model="qwen-local", tier=ModelTier.MD)
    samples = asyncio.run(collect_scores(local, rows))
    result = scan(samples, args.lam)
    if not result:
        print("每类有效样本不足 10 条，未产出标定值", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"标定完成（{len(samples)} 条样本, λ={args.lam}）：")
    for label, theta in result.items():
        print(f"  {label}: θ={theta}")
    print(f"已写入 {out}（Router 自动加载）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
