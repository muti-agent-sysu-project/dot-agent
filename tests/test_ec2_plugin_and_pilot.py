import asyncio
import hashlib
import io
import json
import random
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from dot_agent.collab.ec2_strategy_plugin import (
    ConstraintPreservingSanitizer, EC2StrategyPlugin, export_skill)
from dot_agent.collab.privacy_ledger import TrajectoryPrivacyLedger, TransactionalBoundary
from dot_agent.collab.sanitizer import Inv, Registry, ViewGenerator
from dot_agent.collab.types import Action, ContractFailedError, Forecast, Scope, Usage
from eval import collab_experiment_runner as runner
from eval.finqa_priv import packet, TYPES
from scripts import run_real_pilot as pilot


@pytest.mark.parametrize("arm,structured", [("B6_paac", False), ("B8_modular", True),
                                            ("B9_joint", True)])
def test_plugin_on_official_advisor_endpoint(tmp_path, arm, structured):
    wires, states = [], []
    scope = Scope("tenant", "recipient", "policy")
    generator = ViewGenerator(Registry(), (Inv("greater","gt",("a","b")),),
                              allowed_predicates=("greater",))
    sanitizer = ConstraintPreservingSanitizer(generator, lambda c: runner.encode(
        dict(view=json.loads(c.payload), max_tokens=c.output_cap)))
    async def transport(scope, wire, cap, attempt):
        wires.append(wire)
        return b'{"choices":[]}', Usage(attempt, 1, 0, 1, "ok", 0)
    boundary = TransactionalBoundary(tmp_path/"history", scope,
        TrajectoryPrivacyLedger({"public":1}), inspect=lambda wire,a,c: (),
        input_bound=len, transport=transport, p_max=0, c_max=100000)
    def forecast(state,c,history):
        states.append(history)
        return Forecast(.2 if c.action == Action.LOCAL else .95, .05, 0, 0, 0, 0)
    plugin = EC2StrategyPlugin(arm, scope, boundary, sanitizer, forecast)
    plugin.record_tool_result("local-handle")
    app = plugin.openclaw_endpoint(lambda wire: ("handle", dict(a=98765,b=12345)), None)
    events = []
    async def receive():
        return {"type":"http.request","body":b'{"model":"fixture","messages":[],"max_tokens":32768}'}
    async def send(event):
        events.append(event)
    asyncio.run(app(dict(type="http", method="POST",path="/v1/chat/completions"), receive, send))
    assert events[0]["status"] == 200 and events[1]["body"] == b'{"choices":[]}'
    assert len(wires) == 1 and b"98765" not in wires[0] and b"12345" not in wires[0]
    view = json.loads(wires[0])["view"]
    if structured:
        assert True in view.values() and ("local-handle",) in states
    else:
        assert view["a"].startswith("[[EC2:")
    assert plugin.finalize_episode().tokens == 2
    assert plugin.finalize_episode().attempts == 1
    with pytest.raises(ContractFailedError):
        plugin.record_tool_result("after-close")
    assert "advisor" in export_skill(tmp_path, arm).read_text()


@pytest.mark.parametrize("bits", [format(i,"04b") for i in range(16)])
def test_factorial_callbacks_and_local_replay(bits):
    boundary, sanitizer, runtime = Mock(), Mock(), Mock()
    boundary.totals.return_value = (0, 0)
    sanitizer.candidates.return_value = ()
    runtime.execute = AsyncMock(return_value="verified-output")
    forecast = lambda *args: Forecast(1, 0, 0, 0, 0, 0)
    plugin = EC2StrategyPlugin("B9_joint", Scope("p","r","v"), boundary,
                               sanitizer, forecast, shdk=bits, runtime=runtime)
    s,h,d,k = (int(b) for b in bits)
    before = plugin.controller.prices
    action, body = asyncio.run(plugin.step_decide("state", {}, output_cap=4,
                                                  skill="artifact", bindings_ref="binding"))
    assert action == Action.LOCAL
    assert plugin.controller.quote == (boundary.quote if h else boundary.standalone_quote)
    if k:
        assert body == "verified-output"
        runtime.execute.assert_awaited_once_with("artifact","binding")
        sanitizer.candidates.assert_not_called()
    else:
        sanitizer.candidates.assert_called_once_with(Action.DELEGATE, {}, (), bool(s), 4)
        assert body is None
    assert plugin.finalize_episode().attempts == 0
    assert (before != plugin.controller.prices) == bool(d)
    boundary.reserve.assert_not_called()


def test_meter_preserves_spend_across_calls(tmp_path):
    meter = pilot.Meter(tmp_path, Decimal("0.000600"), (Decimal(1),)*3)
    wire = runner.encode({"max_tokens":4})
    endpoint = Mock(role="cloud")
    endpoint.call.return_value = ("x", Usage("first", 128, 2, 1, "ok", 0))
    meter.invoke(endpoint, wire, "first")
    with pytest.raises(pilot.PilotStopped, match="cap"):
        meter.invoke(endpoint, wire, "second")
    assert endpoint.call.call_count == 1 and meter.total == Decimal("0.000131")


@pytest.fixture
def setup_pilot(tmp_path, monkeypatch):
    data = dict(table=[["year","sales"],["2020","20"]],pre_text=[],post_text=[],
                qa={"question":"What were sales?"})
    rows, truth = [], {}
    for i in range(16):
        task,units = packet(data,TYPES[i%len(TYPES)],random.Random(i))
        tid = f"dev-{i:03}-private"
        rows.append(dict(id=tid,split="dev",input=task))
        truth[tid] = dict(file=task["directory"][runner.selection(task)]["account"],
                          report_group=str(i), sensitive_units=units, answer="20", exe_ans=20)
    for file, attr, obj in [("public_inputs.jsonl","PUBLIC_SHA",rows),
                            ("gt_manifest.json","GRADER_SHA",truth)]:
        path = tmp_path/file
        (runner.lines if isinstance(obj,list) else runner.dump)(path,obj)
        monkeypatch.setattr(runner,attr,hashlib.sha256(path.read_bytes()).hexdigest())
    card = tmp_path/"card.json"
    runner.dump(card, dict(model="cloud",date="2026-11-01",raw_input=2,cache_read=1,output=6))
    for role in ("local","cloud"):
        prefix = "COLLAB_"+role.upper()
        monkeypatch.setenv(prefix+"_BASE_URL","http://127.0.0.1/v1")
        monkeypatch.setenv(prefix+"_MODEL",role)
    monkeypatch.setenv("COLLAB_CLOUD_RECIPIENT","test-only")
    monkeypatch.setenv("COLLAB_CLOUD_API_KEY","private-test-sentinel")
    return pilot.parser().parse_args(["--dataset",str(tmp_path),"--arm","B6_paac","--price-card",
        str(card),"--output-dir",str(tmp_path/"pilot"),"--journal-dir",str(tmp_path/"ledger")])


@pytest.mark.parametrize("flag", ["preflight_only","dry_run"])
def test_offline_modes_never_call_or_write(setup_pilot, monkeypatch, flag):
    monkeypatch.setattr(runner.Endpoint,"call",lambda *a: pytest.fail("network used"))
    args = setup_pilot
    setattr(args,flag,True)
    estimate = pilot.launch(args)
    assert estimate["tasks"] == 16 and estimate["cloud_calls_at_most"] == 33
    assert not args.output_dir.exists() and not args.journal_dir.exists()


@pytest.mark.parametrize("failure", [None,"unknown","cap"])
def test_pilot_through_http_meter_and_reports(setup_pilot, monkeypatch, failure):
    args = setup_pilot
    if failure == "cap":
        args.max_usd = Decimal("0.000001")
    roles = []
    class Client:
        def open(self, request, timeout):
            role = json.loads(request.data)["model"]
            roles.append(role)
            usage = dict(prompt_tokens=42,prompt_tokens_details={"cached_tokens":2},completion_tokens=1)
            if failure == "unknown" and len(roles)==5:
                del usage["prompt_tokens_details"]
            return io.BytesIO(runner.encode(dict(choices=[{"message":{"content":"0"}}],usage=usage)))
    monkeypatch.setattr(runner,"build_opener",lambda *args: Client())
    if failure:
        with pytest.raises(pilot.PilotStopped):
            pilot.launch(args)
        assert roles == ["local"] if failure == "cap" else len(roles)==5
        if failure == "unknown":
            assert (args.output_dir/"episodes.jsonl").stat().st_size
    else:
        summary = pilot.launch(args)
        assert summary["slots"] == 16
        assert roles[:2] == ["local","cloud"] and roles.count("cloud") <= 33
        for name in ("tables/cold.tex","tables/cold.md","tables/pareto-0.png","audit.json"):
            assert (args.output_dir/"run"/name).stat().st_size
    manifest = json.loads((args.output_dir/"run/run_manifest.json").read_bytes())
    assert manifest["status"] == ("aborted" if failure else "complete")
    for log in args.output_dir.rglob("*.json*"):
        assert b"private-test-sentinel" not in log.read_bytes()
