"""GSM8K 公开基准子集评测：走完整 pipeline，自动判分。

用法（在 ai.chii.cloud 上）：
    cd ~/dot-agent && set -a && source run.env && set +a
    python3 scripts/eval_gsm8k.py --n 30 --theta-demos /tmp/gsm_theta.json

判分规则：GSM8K 标准答案在 #### 之后；从模型答案中抽取最后一个数字
（含千分位逗号/百分号清洗）与标准答案数值比对。
同时统计路由路径与云端调用，观察公开基准上的难度信号表现。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_gsm8k(path: Path, n: int, seed: int = 42) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    rows = df.sample(n=min(n, len(df)), random_state=seed).to_dict("records")
    cases = []
    for i, r in enumerate(rows):
        m = re.search(r"####\s*(\-?[\d,\.]+)", r["answer"])
        if not m:
            continue
        gold = float(m.group(1).replace(",", ""))
        cases.append({"id": f"g{i:03d}", "query": r["question"].strip(), "gold": gold})
    return cases


def post_run(base: str, query: str, timeout: float = 600.0) -> dict:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(f"{base}/api/run", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode())
        except Exception:
            return {"success": False, "error": f"http_{e.code}"}
    data["wall_ms"] = round((time.perf_counter() - t0) * 1000)
    return data


def extract_number(text: str) -> float | None:
    """取答案里最后一个独立数字（清洗千分位逗号与 % ）。"""
    text = text.replace(",", " ").replace("%", " ")
    nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8766")
    ap.add_argument("--parquet", default="datasets/gsm8k_test.parquet")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="eval_gsm8k_report.md")
    ap.add_argument("--pause", type=float, default=0.5)
    args = ap.parse_args()

    cases = load_gsm8k(Path(args.parquet), args.n)
    print(f"载入 {len(cases)} 题", file=sys.stderr)
    results, lats = [], []
    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {c['id']} ...", file=sys.stderr, flush=True)
        d = post_run(args.base, c["query"])
        answer = d.get("answer") or ""
        pred = extract_number(answer)
        ok = pred is not None and abs(pred - c["gold"]) < 1e-6
        metrics = d.get("metrics") or {}
        priv = metrics.get("privacy") or {}
        lats.append(d.get("wall_ms", 0))
        results.append({"id": c["id"], "ok": ok, "gold": c["gold"], "pred": pred,
                        "success": d.get("success"), "error": d.get("error"),
                        "path": d.get("path"), "wall_ms": d.get("wall_ms"),
                        "cloud": bool(priv.get("mode_counts"))})
        if args.pause:
            time.sleep(args.pause)

    oks = [r for r in results if r["ok"]]
    cloud_n = sum(1 for r in results if r["cloud"])
    lines = ["# GSM8K 子集端到端评测", "",
             f"- 题数: {len(results)}，判分正确: {len(oks)}"
             f"（准确率 {len(oks)/len(results):.0%}）",
             f"- 请求成功率: {sum(1 for r in results if r['success'])}/{len(results)}",
             f"- 云端参与: {cloud_n}/{len(results)}",
             f"- 时延 P50/P95: {statistics.median(lats):.0f}ms / "
             f"{sorted(lats)[int(len(lats)*0.95)]:.0f}ms", "",
             "| 题 | 判分 | 抽取值 | 标准值 | 路径 | 时延ms | 云端 |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {'✓' if r['ok'] else '✗'} | {r['pred']} "
                     f"| {r['gold']} | {r['path']} | {r['wall_ms']} "
                     f"| {'是' if r['cloud'] else '否'} |")
    bad = [r for r in results if not r["ok"]]
    if bad:
        lines += ["", "## 判分失败明细（前 5）", ""]
        for r in bad[:5]:
            lines.append(f"- {r['id']}: pred={r['pred']} gold={r['gold']} "
                         f"error={r['error']}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"accuracy": len(oks) / len(results),
                      "cloud_rate": cloud_n / len(results),
                      "p50_ms": statistics.median(lats)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
