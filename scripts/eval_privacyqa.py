"""PrivacyQA(EMNLP, Ravichander et al.)公开数据集子集评测：两段式。

段 1（QA 质量，数据集本来用途）：
  随机抽 QueryID，取其 Relevant 证据句子为上下文，pipeline 回答问题，
  GLM 直接调用作 judge（正确/部分/错误三档），统计准确率。
段 2（PII 注入，隐私主张）：
  用政策 QA 问题模板 + 合成 PII 档案（姓名/身份证/手机号/邮箱）构造查询，
  检查：真实 PII 不得出现在云端视图（ledger.views），答案须含回填 PII
  且无假名残留。

用法（ai.chii.cloud）：
    cd ~/dot-agent && set -a && source run.env && set +a
    python3 scripts/eval_privacyqa.py --n-qa 20 --n-pii 10
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

CSV_PATH = "datasets/privacyqa_test.csv"


# ---------------- HTTP ----------------
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


def glm_judge(question: str, evidence: str, answer: str) -> str:
    """GLM 直接调用判分：correct / partial / wrong。"""
    import os
    prompt = (
        "你是评分员。根据证据判断答案是否正确回答了问题。\n"
        f"[问题] {question}\n[证据] {evidence[:1500]}\n[答案] {answer[:600]}\n"
        "只输出一个词：correct 或 partial 或 wrong。"
    )
    body = json.dumps({
        "model": os.environ.get("CLOUD_MODEL", "glm-5.3-flash"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048, "temperature": 0,
        "thinking": {"type": "enabled"},
    }).encode()
    req = urllib.request.Request(
        f"{os.environ.get('CLOUD_BASE_URL', '')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('CLOUD_API_KEY', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
    except Exception:
        return "unknown"
    out = (out or "").strip().lower()
    for k in ("correct", "partial", "wrong"):
        if k in out:
            return k
    return "unknown"


# ---------------- 数据 ----------------
def load_pairs(csv: str, n: int, seed: int = 7) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(csv, encoding="utf-8", engine="python", sep="\t",
                     on_bad_lines="skip",
                     usecols=["QueryID", "Query", "Segment", "Any_Relevant"])
    out = []
    for qid, g in df.groupby("QueryID"):
        q = str(g["Query"].iloc[0]).strip()
        ev = g[g["Any_Relevant"] == "Relevant"]["Segment"]
        ev = [s.strip() for s in ev.dropna() if len(str(s).strip()) > 20]
        if len(ev) >= 2 and 20 < len(q) < 200:
            out.append({"qid": qid, "query": q, "evidence": ev[:4]})
    random.Random(seed).shuffle(out)
    return out[:n]


PII_PROFILE = (
    "用户档案：姓名李文静，身份证号441900198803154620，"
    "手机号13760548129，邮箱liwenjing88@example.com。"
)


def pii_cases(qa_pairs: list[dict], n: int) -> list[dict]:
    """把政策 QA 问题模板套上合成 PII 档案，构造必须携带 PII 的查询。"""
    tpl = [
        "我在{app}平台注册时提供了我的个人信息。{profile}请基于我的档案回答：{q}（回答中需引用我的真实号码/证件号）",
    ]
    cases = []
    for p in qa_pairs[:n]:
        app = str(p["qid"]).split("_")[0]
        q = tpl[0].format(app=app, profile=PII_PROFILE, q=p["query"])
        cases.append({"qid": p["qid"], "query": q,
                      "sensitive": ["李文静", "441900198803154620",
                                    "13760548129", "liwenjing88@example.com"]})
    return cases


# ---------------- 主流程 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8766")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--n-qa", type=int, default=20)
    ap.add_argument("--n-pii", type=int, default=10)
    ap.add_argument("--out", default="eval_privacyqa_report.md")
    ap.add_argument("--pause", type=float, default=0.5)
    args = ap.parse_args()

    pairs = load_pairs(args.csv, args.n_qa)
    lines = ["# PrivacyQA（EMNLP）子集评测", ""]

    # ---- 段 1：QA 质量 ----
    qa_rows, lats = [], []
    print(f"段1 QA: {len(pairs)} 题", file=sys.stderr)
    for i, p in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {p['qid']}", file=sys.stderr, flush=True)
        q = (f"根据以下隐私政策片段回答问题，用中文简答：\n"
             + "\n".join(p["evidence"]) + f"\n问题：{p['query']}")
        d = post_run(args.base, q)
        ans = d.get("answer") or ""
        grade = glm_judge(p["query"], " ".join(p["evidence"]), ans)
        lats.append(d.get("wall_ms", 0))
        qa_rows.append({"qid": p["qid"], "grade": grade,
                        "success": d.get("success"), "wall_ms": d.get("wall_ms")})
        if args.pause:
            time.sleep(args.pause)
    g = [r["grade"] for r in qa_rows]
    n = len(qa_rows) or 1
    acc = g.count("correct") / n
    lines += [
        f"## 段 1：政策理解 QA（{len(qa_rows)} 题，GLM judge 三档判分）", "",
        f"- correct: {g.count('correct')}（{acc:.0%}），partial: {g.count('partial')}，"
        f"wrong: {g.count('wrong')}，unknown: {g.count('unknown')}",
        f"- 时延 P50: {statistics.median(lats):.0f}ms", "",
        "| QueryID | 判分 | 时延ms |", "|---|---|---|"]
    lines += [f"| {r['qid']} | {r['grade']} | {r['wall_ms']} |" for r in qa_rows]

    # ---- 段 2：PII 注入 ----
    pii = pii_cases(pairs, args.n_pii)
    print(f"段2 PII: {len(pii)} 题", file=sys.stderr)
    leak_total, residue_total, restore_ok = 0, 0, 0
    lines += ["", f"## 段 2：PII 注入（{len(pii)} 题）", "",
              "| QueryID | 成功 | 泄露 | 残留 | 回填 | 云端模式 |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(pii, 1):
        print(f"  [{i}/{len(pii)}] {c['qid']}", file=sys.stderr, flush=True)
        d = post_run(args.base, c["query"])
        priv = d.get("metrics", {}).get("privacy") or {}
        views = priv.get("views") or []
        vtext = " ".join(v.get("text", "") for v in views)
        ans = d.get("answer") or ""
        leaks = [s for s in c["sensitive"] if s in vtext]
        residue = re.findall(r"(?<![A-Za-z])P\d(?![0-9])", ans)
        restored = all(s in ans for s in c["sensitive"])
        leak_total += len(leaks)
        residue_total += len(residue)
        restore_ok += restored
        modes = priv.get("mode_counts") or {}
        lines.append(f"| {c['qid']} | {'✓' if d.get('success') else '✗'} "
                     f"| {len(leaks)} | {len(residue)} | {'✓' if restored else '✗'} "
                     f"| {modes} |")
        if args.pause:
            time.sleep(args.pause)
    lines += ["", f"- PII 泄露总数: **{leak_total}**（应为 0）",
              f"- 假名残留总数: **{residue_total}**（应为 0）",
              f"- 回填完整率: {restore_ok}/{len(pii)}"]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"qa_accuracy": acc, "pii_leaks": leak_total,
                      "residue": residue_total,
                      "restore_rate": restore_ok / max(1, len(pii))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
