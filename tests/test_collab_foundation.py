"""Offline contracts for the research foundation."""
import asyncio
import json
import sqlite3
from io import BytesIO
from unittest.mock import Mock

import pytest

from dot_agent.collab.ec2_adapter import annotation_ledger
from dot_agent.collab.privacy_ledger import PrivacyLedger
from dot_agent.collab.risk_router import choose_action
from dot_agent.collab.sanitizer import Registry
from dot_agent.collab.skill_compiler import compile_skill
from dot_agent.collab.types import Action, Span
from dot_agent.gateway import store
from dot_agent.gateway.meter import CallMeter
from dot_agent.gateway.metered import MeteredClient
from dot_agent.gateway.openai_compat import GenerationError, OpenAICompatClient
from dot_agent.gateway.stub import StubModelClient
from dot_agent.orchestrator.router import Router
from dot_agent.pipeline import AgentPipeline
from dot_agent.types import Caller, Label, ModelTier, SignalScores


def test_round_trip_and_local_registry():
    text = "Ada paid 42 to Ada"
    spans = [Span(15, 18, "NAME"), Span(0, 3, "NAME")]
    registry = Registry()
    view = registry.mask(text, spans)
    symbol = registry.mask("Ada", [Span(0, 3, "NAME")])
    assert view == f"{symbol} paid 42 to {symbol}"
    assert registry.restore(view) == text
    assert Registry().mask(text, spans) != view
    with pytest.raises(ValueError):
        Registry().restore(view)
    with pytest.raises(ValueError):
        registry.mask(view, [])


@pytest.mark.parametrize("spans", [
    [Span(0, 1, "X"), Span(0, 2, "Y")], [Span(0, 5, "X")], [Span(0, 1, "bad")],
])
def test_invalid_annotation_is_atomic(spans):
    registry = Registry()
    with pytest.raises(ValueError):
        registry.mask("abcd", spans)
    assert registry.mask("z", [Span(0, 1, "Z")]).endswith(":1]]")


def test_minimal_direct_identifier_detection():
    registry = Registry()
    text = "Mail qa@example.test about item 47."
    assert registry.mask(text) == f"Mail {registry.mask('qa@example.test')} about item 47."
    assert registry.restore(registry.mask(text)) == text
    assert registry.mask(text, []) == registry.mask(text)


def test_annotations_cumulative_unique_exposure():
    ledger = annotation_ledger({"items": [{"entity_id": "row", "sensitive_fields": [
        {"field": "name", "value": "Alice"}, {"field": "team", "value": "Engineering"},
    ]}]})
    assert not ledger.observe("Al")
    assert len(ledger.observe("ice")) == 1
    assert not ledger.observe(" Alice")
    assert ledger.exposure == 0.5
    assert ledger.steps == 3
    assert PrivacyLedger({}).exposure is None
    with pytest.raises(ValueError):
        PrivacyLedger({"bad": ""})


@pytest.mark.parametrize(("flags", "action"), [
    ((True, True, True, True), Action.LOCAL),
    ((False, True, True, True), Action.PROXY),
    ((False, False, True, True), Action.ADVISE),
    ((False, False, False, True), Action.FULL),
    ((False, False, False, False), Action.LOCAL),
])
def test_actions(flags, action):
    local, proxy, advice, full = flags
    assert choose_action(local_ok=local, proxy_sufficient=proxy,
                         advice_sufficient=advice, full_allowed=full) is action


def test_skill_contracts_are_snapshots():
    steps = ["summarize"]
    skill = compile_skill(steps, {"format": "csv"}, {"saved": "yes"})
    steps.clear()
    assert skill.steps == ("summarize",)
    assert skill.can_run({"format": "csv"})
    assert not skill.can_run({})
    assert skill.verified({"saved": "yes"})
    assert not skill.verified({"saved": "no"})
    with pytest.raises(ValueError):
        compile_skill(["summarize"], {}, {})


def test_global_threshold_scan_preserves_label_defaults():
    signals = SignalScores(consistency=0.5)
    assert Router().assign(Label.PLANNING, signals).tier is ModelTier.MC
    for threshold, expected in [(0.0, ModelTier.MC), (0.5, ModelTier.MD), (1, ModelTier.MD)]:
        router = Router(threshold)
        decision = router.assign(Label.PLANNING, signals)
        assert decision.tier is expected and decision.policy.theta == threshold
        assert router.assign(Label.PLANNING, None).tier is ModelTier.MC
    with pytest.raises(ValueError):
        Router(float("nan"))


@pytest.mark.parametrize(("enabled", "calls"), [(False, 1), (True, 3)])
def test_fast_path_sampling(monkeypatch, enabled, calls):
    monkeypatch.setattr(store, "record_call", Mock())
    local = StubModelClient(ModelTier.MD, behavior=lambda *_: "value", base_latency_s=0)
    pipe = AgentPipeline(local, StubModelClient(ModelTier.MC), use_signals=enabled)
    assert asyncio.run(pipe._fast_path("query", "consistency", [])) == ("value", True)
    assert local.call_count == calls
    assert pipe.meter.logs[0].caller is (Caller.SIGNAL if enabled else Caller.EXECUTE)


def response(finish, logs=None):
    message = {"content": None if finish == "length" else "answer", "reasoning_content": "abc"}
    return BytesIO(json.dumps({
        "choices": [{"message": message, "finish_reason": finish, "logprobs": logs}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4,
                  "completion_tokens_details": {"reasoning_tokens": 2}},
    }).encode())


def test_length_retry_bills_both_attempts(tmp_path, monkeypatch):
    client = OpenAICompatClient(tier=ModelTier.MD, api_key="", length_retry_cap=9,
                                request_logprobs=True)
    logs = {"content": [None, {"logprob": 1000}, {"logprob": -1}, {"logprob": True}]}
    client.opener = Mock()
    client.opener.open.side_effect = [response("length"), response("stop", logs)]
    db = tmp_path / "meter.db"
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(store, "_initialized", False)
    result = asyncio.run(MeteredClient(client, CallMeter()).generate("q", max_tokens=6))
    assert result.text == "answer" and result.token_probs == pytest.approx([0.36787944])
    sent = [json.loads(call.args[0].data) for call in client.opener.open.call_args_list]
    assert [body["max_tokens"] for body in sent] == [6, 9]
    assert all(body["logprobs"] for body in sent)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("select finish_reason,reasoning_chars,reasoning_tokens,"
                            "completion_tokens from calls join generation_details using(call_id)"
                            " order by calls.rowid").fetchall()
    assert rows == [("length", 3, 2, 4), ("stop", 3, 2, 4)]


def test_default_truncation_is_marked_without_retry(monkeypatch):
    client = OpenAICompatClient(tier=ModelTier.MD, api_key="")
    client.opener = Mock()
    client.opener.open.return_value = response("length")
    monkeypatch.setattr(store, "record_call", Mock())
    with pytest.raises(GenerationError) as error:
        asyncio.run(MeteredClient(client, CallMeter()).generate("q"))
    assert error.value.result.finish_reason == "length"
    assert "logprobs" not in json.loads(client.opener.open.call_args.args[0].data)
    assert client.opener.open.call_count == 1
