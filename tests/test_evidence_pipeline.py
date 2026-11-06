"""Evidence behavior tests: scripted model replies, no real model calls."""
import asyncio
import json
import sqlite3

import pytest

from dot_agent.gateway import store
from dot_agent.gateway.base import ModelClient
from dot_agent.pipeline import AgentPipeline
from dot_agent.types import Caller, GenerationResult, ModelTier
from dot_agent.orchestrator.evidence import (
    make_packet, parse_answer, parse_indices, select_packet, validate_sources,
)

SOURCES = [
    {"source_id": "intro", "text": "活动地点在教学楼。"},
    {"source_id": "deadline", "text": "报名截止时间是周五17点。"},
    {"source_id": "contact", "text": "咨询请联系活动负责人。"},
]
QUERY = "报名截止时间是什么？"


def answer(text="报名截止时间是周五17点。", citations=None):
    return json.dumps({"answer": text, "citations": citations or ["deadline"]}, ensure_ascii=False)


class ScriptedClient(ModelClient):
    def __init__(self, tier, replies=(), unknown=False):
        self.tier, self.name = tier, "scripted-test"
        self.replies, self.calls, self.unknown = list(replies), [], unknown

    async def generate(self, prompt, *, max_tokens=512, temperature=0.0, caller=Caller.EXECUTE):
        self.calls.append({"prompt": prompt, "caller": caller, "max_tokens": max_tokens})
        await asyncio.sleep(0.002)
        if not self.replies:
            raise AssertionError("unexpected model call")
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return GenerationResult(value, prompt_tokens=7, completion_tokens=2,
                                latency_s=0.002, usage_missing=self.unknown)


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "metrics.db")
    monkeypatch.setattr(store, "_initialized", False)


def pipeline(local_replies=(), cloud_replies=(), **kwargs):
    local = ScriptedClient(ModelTier.MD, local_replies, **kwargs)
    cloud = ScriptedClient(ModelTier.MC, cloud_replies)
    return AgentPipeline(local, cloud, run_id="evidence-test"), local, cloud


def run(agent, **kwargs):
    return asyncio.run(agent.run(QUERY, sources=SOURCES, **kwargs))


@pytest.mark.parametrize("sources", [[], {}, [SOURCES[0], SOURCES[0]],
    [{"source_id": "x", "text": " "}], [{"source_id": "x\ny", "text": "data"}],
    [{"source_id": "x", "text": "data", "requires": ["missing"]}]])
def test_bad_sources_are_rejected(sources):
    with pytest.raises(ValueError):
        validate_sources(sources)


@pytest.mark.parametrize("raw", [
    '{"evidence_indices":[true]}', '{"evidence_indices":[99]}',
    '{"evidence_indices":[1,1]}', '{"evidence_indices":[]}',
    '{"evidence_indices":[1],"evidence_indices":[0]}',
    '{"evidence_indices":[1],"answer":"invented"}', 'not json',
])
def test_bad_index_products_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_indices(raw, SOURCES)


def test_original_text_and_required_table_header_are_preserved():
    sources = [{"source_id":"head", "text":" 名称 | 数量\n"},
               {"source_id":"row", "text":"苹果 | 3", "requires":["head"]}]
    packet = parse_indices('{"evidence_indices":[1]}', sources)
    assert packet.source_ids == ["head", "row"]
    assert packet.sources[0]["text"] == " 名称 | 数量\n"
    with pytest.raises(ValueError):
        make_packet(sources, [1], max_sources=1)


def test_bm25_picks_question_relevant_original_source():
    packet = select_packet(QUERY, SOURCES, "bm25", max_sources=1)
    assert packet.source_ids == ["deadline"]
    assert packet.sources[0]["text"] == SOURCES[1]["text"]


@pytest.mark.parametrize("raw", [answer(citations=["foreign"]),
    '{"answer":"ok","citations":[]}', '{"answer":"","citations":["deadline"]}',
    '{"answer":"ok","citations":["deadline","deadline"]}'])
def test_answer_must_cite_actual_packet(raw):
    with pytest.raises(ValueError):
        parse_answer(raw, ["deadline"])


def test_bm25_uses_main_entry_without_local_calls_and_records_task():
    agent, local, cloud = pipeline(cloud_replies=[answer()])
    result = run(agent)
    assert result.success and result.path == "evidence_bm25"
    assert result.citations == ["deadline"]
    assert set(result.citations) <= set(result.provided_sources)
    assert not local.calls and len(cloud.calls) == 1
    assert result.metrics["C_API_total"] == 9
    assert result.metrics["C_time_s"] >= 0.002
    with sqlite3.connect(store.DB_PATH) as conn:
        assert conn.execute("select count(*) from calls").fetchone()[0] == 1
        assert conn.execute("select path from tasks").fetchone()[0] == "evidence_bm25"


def test_local_selector_only_returns_indices_then_cloud_answers():
    agent, local, cloud = pipeline(['{"evidence_indices":[1]}'], [answer()])
    result = run(agent, selector="local")
    assert result.success and result.provided_sources == ["deadline"]
    assert len(local.calls) == len(cloud.calls) == 1
    assert "evidence_indices" in local.calls[0]["prompt"]
    assert "活动地点" not in cloud.calls[0]["prompt"]


def test_local_contract_failure_goes_directly_to_one_full_fallback():
    agent, local, cloud = pipeline(['{"evidence_indices":[99]}'], [answer()], unknown=True)
    result = run(agent, selector="local")
    assert result.success and result.path == "evidence_full_fallback"
    assert len(local.calls) == len(cloud.calls) == 1
    assert result.provided_sources == [s["source_id"] for s in SOURCES]
    assert result.metrics["unknown_usage_calls"] == 1
    assert result.metrics["C_API_total"] == 9
    assert "index_out_of_range" in " ".join(result.trace)


def test_invalid_cloud_citation_is_retained_and_one_full_fallback_runs():
    agent, _, cloud = pipeline(cloud_replies=[answer(citations=["foreign"]), answer()])
    result = run(agent)
    assert result.success and result.path == "evidence_full_fallback"
    assert len(cloud.calls) == 2 and result.metrics["C_API_total"] == 18
    assert "answer_sources_out_of_range" in " ".join(result.trace)


def test_full_fallback_contract_failure_is_final_and_keeps_all_usage():
    agent, local, cloud = pipeline(cloud_replies=["invalid", "invalid"])
    result = run(agent)
    assert not result.success and result.path == "evidence_failed"
    assert result.error == "invalid_answer_json"
    assert len(cloud.calls) == 2 and not local.calls
    assert result.metrics["C_API_total"] == 18
    with sqlite3.connect(store.DB_PATH) as conn:
        assert conn.execute("select error from tasks").fetchone()[0] == "invalid_answer_json"


def test_evidence_cloud_failure_does_not_silently_answer_with_local_backend():
    secret = "fake-sensitive-error-for-redaction-test"
    agent, local, cloud = pipeline(cloud_replies=[RuntimeError(secret), RuntimeError(secret)])
    result = run(agent)
    assert not result.success and not local.calls and len(cloud.calls) == 2
    assert secret not in repr(result)
    assert result.metrics["unknown_usage_calls"] == 2


def test_full_context_over_budget_is_not_silently_truncated(monkeypatch):
    from dot_agent.orchestrator import evidence
    monkeypatch.setattr(evidence, "MAX_FULL_PROMPT_BYTES", 1000)
    agent, _, cloud = pipeline(cloud_replies=["invalid"])
    sources = SOURCES + [{"source_id":"long", "text":"unrelated " * 1000}]
    result = asyncio.run(agent.run(QUERY, sources=sources))
    assert not result.success and result.error == "full_context_budget_exceeded"
    assert len(cloud.calls) == 1


def test_sources_bypass_question_only_cache_and_change_model_input():
    agent, _, cloud = pipeline(cloud_replies=[answer(), answer("报名截止时间改为周六。")])
    agent.cache.put(QUERY, "stale answer", True)
    first = run(agent)
    changed = [dict(s) for s in SOURCES]
    changed[1]["text"] = "报名截止时间改为周六。"
    agent.reset_meter()
    second = asyncio.run(agent.run(QUERY, sources=changed))
    assert first.answer != second.answer and len(cloud.calls) == 2
    assert "改为周六" in cloud.calls[1]["prompt"]
    assert agent.cache._store[QUERY][0] == "stale answer"


@pytest.mark.parametrize("kwargs", [{"sources": []}, {"sources": SOURCES, "selector":"bogus"}])
def test_invalid_main_entry_input_makes_zero_model_calls(kwargs):
    agent, local, cloud = pipeline()
    with pytest.raises(ValueError):
        asyncio.run(agent.run(QUERY, **kwargs))
    assert not local.calls and not cloud.calls


def test_table_header_dependency_is_kept_by_bm25_even_if_ranked_later():
    sources = [{"source_id":"head", "text":"数量表头"},
               {"source_id":"row", "text":"苹果 数量 3", "requires":["head"]}]
    assert select_packet("苹果数量", sources).source_ids == ["head", "row"]


@pytest.mark.parametrize("failure", [sqlite3.OperationalError("synthetic ledger failure"),
    PermissionError("synthetic directory failure"), OSError("synthetic filesystem failure")],
    ids=["sqlite", "permission", "filesystem"])
def test_ledger_failure_stops_before_any_billable_fallback(monkeypatch, failure):
    def fail(**kwargs):
        raise failure
    monkeypatch.setattr(store, "record_call", fail)
    agent, _, cloud = pipeline(cloud_replies=[answer()])
    result = run(agent)
    assert not result.success and len(cloud.calls) == 1
    assert result.error == "call_ledger_write_failed"
    assert not result.metrics["ledger_complete"]
    assert result.metrics["C_API_total"] == 9


def test_unknown_cloud_usage_stays_unknown_in_sqlite():
    agent, _, cloud = pipeline(cloud_replies=[answer()])
    cloud.unknown = True
    result = run(agent)
    assert result.metrics["C_API_total"] == 0
    assert result.metrics["unknown_usage_calls"] == 1
    assert result.metrics["by_caller"]["evidence_answer"]["tokens"] == 0
    with sqlite3.connect(store.DB_PATH) as conn:
        assert conn.execute("select prompt_tokens,completion_tokens from calls").fetchone() == (None,None)


def test_insufficient_evidence_can_abstain_without_inventing_a_citation():
    text, citations = parse_answer('{"answer":"证据不足","citations":[]}', ["deadline"])
    assert text == "证据不足" and not citations


@pytest.mark.parametrize("selector", ["bm25", "local"])
def test_explicit_demo_runs_structured_evidence_without_real_backends(selector):
    from dot_agent.gateway.stub import StubModelClient
    from dot_agent.gateway.demo_behavior import local_demo_behavior, cloud_demo_behavior
    agent = AgentPipeline(StubModelClient(ModelTier.MD, behavior=local_demo_behavior),
                          StubModelClient(ModelTier.MC, behavior=cloud_demo_behavior))
    result = run(agent, selector=selector)
    assert result.success and result.metrics["simulated"]
    assert result.answer.startswith("【模拟回答】")
    assert result.citations == ["deadline"]


def test_cached_demo_answer_remains_marked_as_simulated():
    from dot_agent.gateway.stub import StubModelClient
    agent = AgentPipeline(StubModelClient(ModelTier.MD), StubModelClient(ModelTier.MC))
    query = "1+1等于多少？"
    agent.cache.put(query, "模拟旧答案", True)
    result = asyncio.run(agent.run(query))
    assert result.path == "cache_hit"
    assert result.metrics["simulated"]


def test_ledger_failure_does_not_activate_legacy_local_fallback(monkeypatch):
    from dot_agent.gateway.metered import LedgerWriteError
    def fail(**kwargs):
        raise PermissionError("synthetic permission failure")
    monkeypatch.setattr(store, "record_call", fail)
    agent, local, cloud = pipeline(cloud_replies=[answer()])
    with pytest.raises(LedgerWriteError):
        asyncio.run(agent.cloud.generate("test"))
    assert len(cloud.calls) == 1 and not local.calls
