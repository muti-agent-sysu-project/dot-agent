import asyncio
import hashlib
import io
import json
import random
from pathlib import Path
from unittest.mock import Mock

import pytest

from dot_agent.collab.privacy_ledger import TrajectoryPrivacyLedger
from dot_agent.collab.types import Action, PolicyDeniedError
from eval import collab_experiment_runner as runner
from eval.finqa_priv import build, packet, stream, TYPES


def test_frozen_policy_zero_weights_and_escaped_literals():
    ledger = TrajectoryPrivacyLedger({"visible": 0, "secret": 2}, raw_forbidden=[b"private"])
    assert ledger.denominator == 2 and ledger.exposure(["visible"]) == 0
    with pytest.raises(PolicyDeniedError):
        ledger.check([], br'{"data":"pri\u0076ate"}')
    assert TrajectoryPrivacyLedger({}, public_scope=True).exposure([]) == 0
    with pytest.raises(ValueError):
        TrajectoryPrivacyLedger({})


def test_incomplete_pilot_fails_closed(tmp_path):
    src, pilot = tmp_path/"src", tmp_path/"pilot"
    src.write_text("[]")
    pilot.write_text("")
    with pytest.raises(ValueError, match="Stage-1"):
        build(src,pilot,tmp_path)


def test_real_gateway_body_receipt_and_unknown(tmp_path, monkeypatch):
    # Mock the HTTP client itself; no socket, environment key, or model service.
    monkeypatch.setattr(runner.os, "getenv", lambda k: None)
    network = Mock()
    monkeypatch.setattr(runner, "build_opener", lambda *args: network)
    cloud = runner.Endpoint("fixture", "https://fixture.invalid/v1/chat/completions", "cloud")
    task, _ = next(stream(random.Random(0)))
    slot = dict(arm="B1_cloud_only",task_id="stream-0")
    def execute(db):
        return asyncio.run(runner.episode(slot, task, runner.Endpoint("unused"), cloud, tmp_path/db,
                                          None, (.05,.05,.2), "real", {}, 41))
    def reply(usage):
        return io.BytesIO(runner.encode(dict(choices=[{"message":{"content":"7"}}], usage=usage)))
    network.open.side_effect = [
        reply(dict(prompt_tokens=30, prompt_tokens_details=dict(cached_tokens=10),
                   completion_tokens=7, completion_tokens_details=dict(reasoning_tokens=5))),
        reply(dict(prompt_tokens=30, completion_tokens=7)), TimeoutError()]
    row, pred, wires, _ = execute("known")
    request = network.open.call_args.args[0]
    assert request.data == wires[0] and hashlib.sha256(request.data).hexdigest() == row["usage"][0]["payload_digest"]
    assert row["C_tokens"] == 37 and row["attempts"] == 1 and pred == "7" and row["P_ledger"] > 0
    partial, pred, _, _ = execute("partial")
    assert pred == "7" and partial["unknown_usage"] == 1 and partial["C_tokens"] is None
    assert partial["usage"][0]["output"] == 7 and partial["usage"][0]["state"] == "unknown"
    lost, _, _, _ = execute("timeout")
    assert lost["status"] == "UnknownUsageError" and lost["P_ledger"] == row["P_ledger"]
    assert lost["unknown_usage"] == 1 and network.open.call_count == 3


def test_local_endpoint_refuses_external_host(monkeypatch):
    monkeypatch.setenv("COLLAB_LOCAL_BASE_URL", "https://example.org/v1")
    with pytest.raises(ValueError, match="origin"):
        runner.Endpoint.configure("local", False)


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    docs = dict(table=[["year","amount"],["2020","20"]], pre_text=["Revenue was 20."], post_text=[],
                qa={"question":"What was revenue?"})
    rows, gt = [], {}
    for i,kind in enumerate(TYPES):
        data, units = packet(docs,kind,random.Random(i))
        tid = f"dev-{i:03}-private"
        rows.append(dict(id=tid, split="dev", input=data))
        gt[tid] = dict(file=data["directory"][runner.selection(data)]["account"],
                       report_group=str(i), sensitive_units=units, answer="20", exe_ans=20)
    for i,(data,gold) in enumerate(stream(random.Random(7))):
        tid = f"stream-{i}"
        rows.append(dict(id=tid, split="stream", input=data))
        gt[tid] = dict(**gold, family_id="growth")
    public, truth = tmp_path/"public_inputs.jsonl", tmp_path/"gt_manifest.json"
    runner.lines(public, rows)
    runner.dump(truth, gt)
    for name,path in [("PUBLIC_SHA",public),("GRADER_SHA",truth)]:
        monkeypatch.setattr(runner, name, hashlib.sha256(path.read_bytes()).hexdigest())
    return tmp_path


def run_fixture(folder, *opts):
    out = folder/"out"
    args = runner.parser().parse_args(["--dataset",str(folder),"--output-dir",str(out),*opts])
    manifest = runner.run(args)
    rows = list(map(json.loads,(out/"results.jsonl").read_bytes().splitlines()))
    return manifest, rows, out


def test_five_arms_and_posthoc_grading(benchmark, monkeypatch):
    order = []
    original_episode, original_read = runner.episode, Path.read_bytes
    async def execute(*args):
        order.append("episode")
        return await original_episode(*args)
    def read(path):
        if path.name == "gt_manifest.json":
            order.append("grader")
        return original_read(path)
    monkeypatch.setattr(runner,"episode",execute)
    monkeypatch.setattr(Path,"read_bytes",read)
    manifest, rows, out = run_fixture(benchmark)
    assert order == ["episode"]*15+["grader"]
    assert manifest["simulated"] and len(rows)==manifest["slots"]==15
    assert all(r["C_tokens"]==sum(u["raw_input"]+u["cache_read"]+u["output"] for u in r["usage"]) for r in rows)
    assert all(r["status"]=="ok" and r["L_s"]>0 and r["E_official"] is None for r in rows)
    assert all(r["P_ledger"]==0 and r["C_tokens"]==0 and r["E_FinQA_exact"]==0 for r in rows if r["arm"]==runner.ARMS[0])
    audit = json.loads((out/"audit.json").read_bytes())
    assert audit["paired_utility"]["clusters"]==3 and len(audit["linkage"][runner.ARMS[1]])==3
    for name in ["cold.tex","cold.md","pareto-0.png"]:
        assert (out/"tables"/name).stat().st_size>100


def test_factorial_stream_and_no_hot_egress(benchmark):
    m, rows, _ = run_fixture(benchmark,"--split","stream","--ablation","all")
    assert len(m["arms"])==16 and len(rows)==80
    for row in rows:
        if row["task_id"] in ("stream-1","stream-2") and row["K"]:
            assert row["steps"]==["skill"] and row["U"]==1
            assert row["usage"]==[] and row["delta_p"]==row["C_tokens"]==0
        else:
            assert "skill" not in row["steps"]


def test_stream_full_reference_retains_recipient_history(benchmark):
    _, rows, _ = run_fixture(benchmark,"--split","stream","--arms","B1_cloud_only")
    assert rows[0]["P_before"]==0
    assert all(a["P_ledger"]==b["P_before"] for a,b in zip(rows,rows[1:]))
    assert all(row["delta_p"]>0 for row in rows)
