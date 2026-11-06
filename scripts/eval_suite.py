"""端到端评测套件：性能 + 路由 + 隐私保护三合一。

用法（在 ai.chii.cloud 上）：
    cd ~/dot-agent && source run.env
    python3 scripts/eval_suite.py --base http://127.0.0.1:8766 \
        --cases datasets/eval_cases.json --out eval_report.md

对每条用例 POST /api/run，收集：
  - 性能：success/error、总时延、云端命中次数（体现云端价值）
  - 路由：path/label 是否符合预期（expect_cloud）
  - 隐私：sensitive 列表中的真实实体是否出现在"云端所见"（ledger.views）
          或最终答案的假名残留里；mask/judge/plain 模式计数
输出 Markdown 报告 + 汇总统计（云端触发率、零泄露率、P50/P95 时延）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_run(base: str, query: str, timeout: float = 600.0) -> dict:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(f"{base}/api/run", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # 业务失败（success=False）返回 502，但 body 仍是完整结果结构
        try:
            data = json.loads(e.read().decode())
        except Exception:
            return {"success": False, "error": f"http_{e.code}", "wall_ms": 0}
    data["wall_ms"] = (time.perf_counter() - t0) * 1000
    return data


def run_suite(base: str, cases: list[dict], pause_s: float) -> list[dict]:
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ...", file=sys.stderr, flush=True)
        data = post_run(base, case["query"])
        metrics = data.get("metrics") or {}
        privacy = metrics.get("privacy") or {}
        views = privacy.get("views") or []
        view_text = " ".join(v.get("text", "") for v in views)
        answer = data.get("answer") or ""

        # 隐私检查：真实实体不得出现在云端所见文本中
        leaks = [s for s in case.get("sensitive", []) if s and s in view_text]
        # 假名残留检查：仅对含敏感实体且确实走过假名化的用例生效。
        # 系统设计类答案天然含 "P1/P2"（优先级）字样，无假名化语境时不算残留。
        residue = []
        if case.get("sensitive") and privacy.get("masked_cloud_calls", 0) + \
                privacy.get("judge_calls", 0) > 0:
            import re
            residue = re.findall(r"(?<![A-Za-z])P\d(?![0-9])", answer)

        mode_counts = privacy.get("mode_counts") or {}
        c_tokens = int(metrics.get("C_API_total", 0) or 0)
        n_cloud = int(metrics.get("cloud_calls", 0) or data.get("n_cloud", 0) or 0)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "success": bool(data.get("success")),
            "error": data.get("error"),
            "wall_ms": round(data.get("wall_ms", 0)),
            "path": data.get("path"),
            "label": data.get("label"),
            "n_cloud": n_cloud,
            "cloud_tokens": c_tokens,
            "cloud_hit": c_tokens > 0 or bool(mode_counts),
            "expect_cloud": case.get("expect_cloud"),
            "leaks": leaks,
            "pseudonym_residue": residue,
            "mode_counts": mode_counts,
            "masked_calls": privacy.get("masked_cloud_calls", 0),
            "plain_calls": privacy.get("plain_cloud_calls", 0),
            "judge_calls": privacy.get("judge_calls", 0),
            "entities_masked": privacy.get("entities_masked", 0),
            "answer_head": answer[:80].replace("\n", " "),
        })
        if pause_s:
            time.sleep(pause_s)
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["success"]]
    lat = sorted(r["wall_ms"] for r in rows)
    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0
    cloud_rows = [r for r in rows if r["expect_cloud"]]
    cloud_hits = [r for r in cloud_rows if r["cloud_hit"]]
    return {
        "total": len(rows),
        "success": len(ok),
        "success_rate": len(ok) / len(rows) if rows else 0,
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "cloud_cases": len(cloud_rows),
        "cloud_hits": len(cloud_hits),
        "cloud_trigger_rate": len(cloud_hits) / len(cloud_rows) if cloud_rows else 0,
        "leak_total": sum(len(r["leaks"]) for r in rows),
        "residue_total": sum(len(r["pseudonym_residue"]) for r in rows),
    }


def write_report(rows: list[dict], summary: dict, out: Path) -> None:
    lines = ["# dot-agent 端到端评测报告", "",
             f"- 用例数: {summary['total']}，成功: {summary['success']}"
             f"（成功率 {summary['success_rate']:.0%}）",
             f"- 时延 P50/P95: {summary['latency_p50_ms']}ms / {summary['latency_p95_ms']}ms",
             f"- 云端触发: 应触发 {summary['cloud_cases']} 条，"
             f"实际命中 {summary['cloud_hits']} 条"
             f"（触发率 {summary['cloud_trigger_rate']:.0%}）",
             f"- 隐私泄露: {summary['leak_total']} 处；假名残留: "
             f"{summary['residue_total']} 处", "",
             "| 用例 | 类别 | 成功 | 路径 | 标签 | 云端 | 时延ms | 泄露 | 残留 | 模式计数 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {'✓' if r['success'] else '✗ ' + str(r['error'])} "
            f"| {r['path']} | {r['label']} | {r['n_cloud']} | {r['wall_ms']} "
            f"| {len(r['leaks'])} | {len(r['pseudonym_residue'])} | {r['mode_counts']} |")
    lines += ["", "## 失败/泄露明细", ""]
    for r in rows:
        if not r["success"] or r["leaks"] or r["pseudonym_residue"]:
            lines.append(f"- **{r['id']}**: error={r['error']}, leaks={r['leaks']}, "
                         f"residue={r['pseudonym_residue']}, answer={r['answer_head']!r}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="dot-agent 端到端评测")
    ap.add_argument("--base", default="http://127.0.0.1:8766")
    ap.add_argument("--cases", default="datasets/eval_cases.json")
    ap.add_argument("--out", default="eval_report.md")
    ap.add_argument("--pause", type=float, default=1.0, help="用例间隔秒数")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    rows = run_suite(args.base, cases, args.pause)
    summary = summarize(rows)
    write_report(rows, summary, Path(args.out))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
