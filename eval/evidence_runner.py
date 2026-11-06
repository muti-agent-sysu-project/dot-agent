"""Separate evidence comparison; never modifies the fixed Stage 5 protocol.

--demo uses explicit simulated clients even if real credentials are configured.
--real uses explicitly configured clients and performs no hidden warmup calls.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dot_agent.pipeline import AgentPipeline
from dot_agent.orchestrator.evidence import validate_sources

MODES = ("full", "rule", "bm25", "local")


def _normalize(text):
    return "".join(text.split()).casefold()


async def run_comparison(items, local, cloud, output_dir, modes=MODES):
    if not items or not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("invalid_comparison")
    seen = set()
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not item["id"] or item["id"] in seen
                or not isinstance(item.get("query"), str) or not item["query"].strip()
                or ("expected" in item and not isinstance(item["expected"], str))):
            raise ValueError("invalid_comparison_item")
        validate_sources(item.get("sources"))
        seen.add(item["id"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = "evidence-" + uuid.uuid4().hex[:12]
    rows = []
    with (output_dir / "results.jsonl").open("x", encoding="utf-8") as writer:
        for index, item in enumerate(items):
            order = list(modes[index % len(modes):]) + list(modes[:index % len(modes)])
            for mode in order:
                agent = AgentPipeline(local, cloud, run_id=run_id, mode="evidence_" + mode)
                agent.cache.enabled = False
                result = await agent.run(item["query"], sources=item["sources"], selector=mode)
                simulated = result.metrics["simulated"]
                quality = None
                if "expected" in item and not simulated:
                    quality = result.success and _normalize(result.answer) == _normalize(item["expected"])
                row = {"item_id": item["id"], "mode": mode, "simulated": simulated,
                       "exact_match": quality, "result": asdict(result)}
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                rows.append(row)
    groups = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        groups[mode] = {
            "units": len(selected),
            "successes": sum(row["result"]["success"] for row in selected),
            "scored_units": sum(row["exact_match"] is not None for row in selected),
            "exact_matches": sum(row["exact_match"] is True for row in selected),
            "cloud_tokens_known_subtotal": sum(row["result"]["metrics"]["C_API_total"] for row in selected),
            "unknown_usage_calls": sum(row["result"]["metrics"]["unknown_usage_calls"] for row in selected),
            "total_time_s": round(sum(row["result"]["metrics"]["C_time_s"] or 0 for row in selected), 4),
        }
    simulated_count = sum(row["simulated"] for row in rows)
    report = {
        "run_id": run_id, "scope": "engineering-comparison; not formal Stage 5",
        "units": len(rows), "modes": list(modes),
        "cohort_sha256": hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "contains_simulation": bool(simulated_count),
        "usage_kind": ("simulated" if simulated_count == len(rows) else
                       "mixed_simulated_and_provider" if simulated_count else "provider_known_subtotal"),
        "quality_metric": "normalized exact match when reference exists; simulation is unscored",
        "by_mode": groups,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--demo", action="store_true")
    identity.add_argument("--real", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets/evidence_smoke.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    items = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.demo:
        from dot_agent.gateway.stub import StubModelClient
        from dot_agent.gateway.demo_behavior import local_demo_behavior, cloud_demo_behavior
        from dot_agent.types import ModelTier
        local = StubModelClient(ModelTier.MD, "local-demo", local_demo_behavior)
        cloud = StubModelClient(ModelTier.MC, "cloud-demo", cloud_demo_behavior)
    else:
        from dot_agent.gateway.factory import make_clients
        from dot_agent.gateway.stub import StubModelClient
        local, cloud, _ = make_clients()
        if isinstance(local, StubModelClient) or isinstance(cloud, StubModelClient):
            parser.error("--real requires both configured real backends")
    try:
        report = asyncio.run(run_comparison(items, local, cloud, args.output, args.modes))
    except FileExistsError:
        parser.error("output directory already exists; choose a new output path")
    print(json.dumps({"units": report["units"], "usage_kind": report["usage_kind"],
                      "summary": str(args.output / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
