"""Four-route comparison keeps failures, labels simulation and refuses overwrite."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eval.evidence_runner import run_comparison
from test_evidence_pipeline import SOURCES, QUERY, answer, pipeline, isolated_ledger


def test_four_routes_share_inputs_and_keep_reference_out_of_prompts(tmp_path):
    _, local, cloud = pipeline(['{"evidence_indices":[1]}'], [answer()] * 4)
    items = [{"id":"one", "query":QUERY, "sources":SOURCES, "expected":"reference-only-never-send"}]
    output = tmp_path / "comparison"
    report = asyncio.run(run_comparison(items, local, cloud, output))
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert report["units"] == len(rows) == 4
    assert {row["mode"] for row in rows} == {"full","rule","bm25","local"}
    assert all(row["exact_match"] is False for row in rows)
    assert all("reference-only-never-send" not in call["prompt"] for call in local.calls + cloud.calls)
    before = len(local.calls) + len(cloud.calls)
    with pytest.raises(FileExistsError):
        asyncio.run(run_comparison(items, local, cloud, output))
    assert len(local.calls) + len(cloud.calls) == before


def test_full_failure_is_counted_and_not_repeated_as_its_own_fallback(tmp_path):
    _, local, cloud = pipeline(cloud_replies=["invalid"])
    items = [{"id":"one", "query":QUERY, "sources":SOURCES, "expected":"answer"}]
    report = asyncio.run(run_comparison(items, local, cloud, tmp_path / "failed", modes=["full"]))
    assert report["units"] == 1 and report["by_mode"]["full"]["successes"] == 0
    assert report["by_mode"]["full"]["scored_units"] == 1
    assert report["by_mode"]["full"]["exact_matches"] == 0
    assert len(cloud.calls) == 1


def test_invalid_cohort_is_rejected_before_creating_output_or_calling_models(tmp_path):
    _, local, cloud = pipeline()
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        asyncio.run(run_comparison([{"id":"one","query":QUERY,"sources":[]}], local, cloud, output))
    assert not output.exists() and not local.calls and not cloud.calls


def test_demo_cli_does_not_read_configured_real_backends(tmp_path):
    output = tmp_path / "demo"
    environment = dict(os.environ, CLOUD_API_KEY="fake-never-use", DOT_REQUIRE_REAL="1",
                       LOCAL_BASE_URL="http://127.0.0.1:1", LOCAL_MODEL="not-a-model",
                       PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "eval/evidence_runner.py", "--demo", "--output", str(output)],
                               cwd=Path(__file__).resolve().parents[1], env=environment,
                               capture_output=True, encoding="utf-8", timeout=20)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["contains_simulation"]
    assert report["usage_kind"] == "simulated"
    assert all(group["scored_units"] == 0 for group in report["by_mode"].values())


def test_zero_call_budget_failure_keeps_demo_identity_and_is_not_scored(tmp_path):
    from dot_agent.gateway.stub import StubModelClient
    from dot_agent.types import ModelTier
    local, cloud = StubModelClient(ModelTier.MD), StubModelClient(ModelTier.MC)
    item = {"id":"oversized-demo", "query":"find data", "expected":"reference",
            "sources":[{"source_id":"large","text":"x" * 50000}]}
    output = tmp_path / "oversized-demo"
    report = asyncio.run(run_comparison([item], local, cloud, output))
    assert local.call_count == cloud.call_count == 0
    assert report["contains_simulation"] and report["usage_kind"] == "simulated"
    assert all(group["scored_units"] == 0 for group in report["by_mode"].values())
    assert all(group["successes"] == 0 for group in report["by_mode"].values())
