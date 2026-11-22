"""评测 runner：四模式对照（对齐申请书口径）。

模式：
  local_only  纯本地基线：整题一次本地调用
  cloud_only  纯云端基线：整题一次云端调用（CoT 式，性能与成本上界参照）
  rule        规则协同 v0：L1/L2 分类 + 类型默认倾向路由（信号关闭）
  signal      信号协同 v1：完整统计信号路由（α-quantile + 一致性 + 执行反馈）

协议（吸收自团队 edge-cloud-prototype 的实验纪律）：
  - 每轮实验一个 run_id，逐题**轮换**各模式执行顺序（防顺序偏差、
    防本地前缀 KV 缓存偏爱先跑的模式）；
  - 报告文件带 run_id，已存在则拒绝覆盖（不静默重复实验）；
  - 所有调用（含基线）过 MeteredClient，SQLite 同口径落库。

用法：
  python eval/runner.py --all            # 四模式全跑
  python eval/runner.py --all --run-id exp2
接真实模型：设 OLLAMA_HOST / CLOUD_API_KEY 后同命令。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from dot_agent import AgentPipeline  # noqa: E402
from dot_agent.gateway.factory import make_clients, warmup_real  # noqa: E402
from dot_agent.gateway.metered import MeteredClient  # noqa: E402
from dot_agent.gateway.meter import CallMeter  # noqa: E402
from dot_agent.orchestrator.verify import answer_equiv, check_constraints  # noqa: E402
from dot_agent.orchestrator.signals import normalize_answer  # noqa: E402
from dot_agent.types import Caller, ModelTier  # noqa: E402

DATASETS = ["gsm8k_style", "mmlu_style", "planning_cn", "tooluse_cn"]


def load_dataset(name: str) -> list[dict]:
    rows = []
    for line in (EVAL_DIR / "datasets" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_letter(answer: str, expected: str) -> bool:
    """选项题判分：先找“答案/正确”附近的独立 A-D，再退回最后一个独立字母。"""
    m = re.search(r"(?:答案|正确|answer|correct)[\s\S]{0,20}?\b([A-D])\b",
                  answer, re.IGNORECASE)
    if m:
        return m.group(1).upper() == expected.upper()
    letters = re.findall(r"(?i)\b([A-D])\b", answer)
    return bool(letters) and letters[-1].upper() == expected.upper()


def score_one(item: dict, answer: str) -> bool:
    kind = item["score"]
    if kind == "num":
        return answer_equiv(answer, item["expected"])
    if kind == "letter":
        return score_letter(answer, item["expected"])
    if kind == "constraint":
        return check_constraints(answer, item.get("constraints", []))
    if kind == "contains":
        return str(item["expected"]) in answer
    return False


class ModeRunner:
    """统一执行入口：pipeline 模式与基线模式都走 MeteredClient 计量。"""

    def __init__(self, kind: str, local, cloud, run_id: str):
        self.kind = kind
        if kind in ("rule", "signal"):
            self.pipeline = AgentPipeline(
                local, cloud, use_signals=(kind == "signal"),
                run_id=run_id, mode=kind,
            )
        else:
            meter = CallMeter()
            tier = ModelTier.MD if kind == "local_only" else ModelTier.MC
            inner = local if kind == "local_only" else cloud
            self.client = MeteredClient(inner, meter, run_id=run_id)

    async def run(self, item: dict) -> dict:
        if self.kind in ("rule", "signal"):
            t0 = time.perf_counter()
            try:
                result = await self.pipeline.run(item["query"])
            except Exception as e:
                # 失败保留在分母里；已发生的调用仍计入成本
                m = self.pipeline.meter.summarize(time.perf_counter() - t0)
                self.pipeline.reset_meter()
                return {
                    "correct": False, "answer": "",
                    "cloud_in": m["C_API_in"], "cloud_out": m["C_API_out"],
                    "unknown": m["unknown_usage_calls"],
                    "time_s": time.perf_counter() - t0,
                    "error": type(e).__name__,
                }
            m = result.metrics
            self.pipeline.reset_meter()
            return {
                "correct": score_one(item, result.answer),
                "answer": result.answer[:500],
                "cloud_in": m["C_API_in"], "cloud_out": m["C_API_out"],
                "unknown": m["unknown_usage_calls"],
                "time_s": m["C_time_s"] or 0.0,
                "error": None if result.success else (result.error or "pipeline_failed"),
            }
        t0 = time.perf_counter()
        try:
            r = await self.client.generate(
                item["query"], max_tokens=512, temperature=0.0, caller=Caller.EXECUTE
            )
        except Exception as e:
            return {
                "correct": False, "answer": "",
                "cloud_in": 0, "cloud_out": 0, "unknown": 0,
                "time_s": time.perf_counter() - t0,
                "error": type(e).__name__,
            }
        dt = time.perf_counter() - t0
        cloud_in = r.prompt_tokens if self.client.tier is ModelTier.MC else 0
        cloud_out = r.completion_tokens if self.client.tier is ModelTier.MC else 0
        return {
            "correct": score_one(item, r.text),
            "answer": r.text[:500],
            "cloud_in": 0 if r.usage_missing else cloud_in,
            "cloud_out": 0 if r.usage_missing else cloud_out,
            "unknown": int(r.usage_missing),
            "time_s": dt,
            "error": None,
        }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    times = sorted(r["time_s"] for r in records)
    p95_idx = min(int(0.95 * (n - 1)), n - 1)
    return {
        "n": n,
        "acc": sum(r["correct"] for r in records) / n if n else 0.0,
        "cloud_in": sum(r["cloud_in"] for r in records),
        "cloud_out": sum(r["cloud_out"] for r in records),
        "unknown": sum(r["unknown"] for r in records),
        "errors": sum(1 for r in records if r.get("error")),
        "t_med": times[n // 2] if n else 0.0,
        "t_p95": times[p95_idx] if n else 0.0,
    }


async def run_round_robin(items: list[dict], runners: dict[str, ModeRunner]) -> dict[str, list[dict]]:
    """逐题轮换模式执行顺序：第 i 题从 modes[i % len] 起轮转。"""
    modes = list(runners)
    results: dict[str, list[dict]] = {m: [] for m in modes}
    for i, item in enumerate(items):
        order = modes[i % len(modes):] + modes[: i % len(modes)]
        for mode in order:
            rec = await runners[mode].run(item)
            results[mode].append(rec)
            print(f"[{mode}] {item['id']} correct={rec['correct']} "
                  f"error={rec.get('error') or '-'}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="四模式全跑（默认仅烟测 signal 模式）")
    parser.add_argument("--run-id", default=None, help="实验分组名（进 SQLite 与报告文件名）")
    parser.add_argument("--output", default=None, help="报告路径（默认 eval/report-{run_id}.md，拒绝覆盖）")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help="只跑指定数据集子集（默认全部）")
    parser.add_argument("--records", default=None, help="逐题记录 JSONL 路径（可选）")
    args = parser.parse_args()

    run_id = args.run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_path = Path(args.output) if args.output else EVAL_DIR / f"report-{run_id}.md"
    if out_path.exists():
        print(f"✗ 报告 {out_path} 已存在，拒绝覆盖。换 --run-id 或 --output。")
        sys.exit(2)

    local, cloud, info = make_clients()
    print(f"运行时: local={info['local']} cloud={info['cloud']}  run_id={run_id}")
    for note in info["notes"]:
        print(f"  注: {note}")
    if info["local"] == "stub":
        print("  ⚠ stub 模式——指标无实际意义，仅验证评测流程可跑通")
    asyncio.run(warmup_real(info, local, cloud))  # 真实模型预热，防冷启动污染 P95

    all_items = [item for ds in args.datasets for item in load_dataset(ds)]
    mode_names = ["signal", "rule", "local_only", "cloud_only"] if args.all else ["signal"]
    runners = {m: ModeRunner(m, local, cloud, run_id) for m in mode_names}
    results = asyncio.run(run_round_robin(all_items, runners))

    if args.records:
        with Path(args.records).open("w", encoding="utf-8") as writer:
            for mode, recs in results.items():
                for item, rec in zip(all_items, recs):
                    writer.write(json.dumps({"id": item["id"], "mode": mode, **rec},
                                            ensure_ascii=False) + "\n")
        print(f"逐题记录: {args.records}")

    lines = [
        "# 评测报告",
        "",
        f"- 运行时: local={info['local']}, cloud={info['cloud']}",
        f"- run_id: {run_id}（逐题轮换模式执行顺序；原始记录见 data/metrics.db）",
        "",
        "| 模式 | n | Acc | C_API(in/out) | unknown_usage | errors | C_time 中位数(s) | C_time P95(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in mode_names:
        s = summarize(results[name])
        lines.append(
            f"| {name} | {s['n']} | {s['acc']:.1%} | {s['cloud_in']}/{s['cloud_out']} "
            f"| {s['unknown']} | {s['errors']} | {s['t_med']:.3f} | {s['t_p95']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
