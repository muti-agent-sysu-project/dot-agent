"""Local HTTP checks for the actual shared entry point."""
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.error
import urllib.request

import pytest

from dot_agent.server import Handler
from test_evidence_pipeline import SOURCES, QUERY, answer, pipeline, isolated_ledger


@pytest.fixture
def http_api():
    agent, local, cloud = pipeline(local_replies=["42"] * 10, cloud_replies=[answer()] * 4)
    handler = type("EvidenceHandler", (Handler,), {"pipeline": agent, "runtime": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:" + str(server.server_port) + "/api/run", agent, local, cloud
    server.shutdown()
    server.server_close()
    thread.join()


def post(url, body):
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=raw, headers={"Content-Type":"application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read())


def test_sources_reach_actual_http_pipeline_and_citations_are_serialized(http_api):
    url, _, local, cloud = http_api
    status, result = post(url, {"query":QUERY, "sources":SOURCES})
    assert status == 200 and result["success"]
    assert result["path"] == "evidence_bm25" and result["citations"] == ["deadline"]
    assert not local.calls and len(cloud.calls) == 1


@pytest.mark.parametrize("body", [[], {"query":5}, {"query":" "},
    {"query":QUERY,"sources":None}, {"query":QUERY,"sources":[]},
    {"query":QUERY,"sources":SOURCES,"selector":"full"},
    {"query":QUERY,"sources":SOURCES,"selector":"bad"}])
def test_invalid_http_input_is_rejected_before_model_calls(http_api, body):
    url, _, local, cloud = http_api
    status, result = post(url, body)
    assert status == 400 and result["error"]
    assert not local.calls and not cloud.calls


def test_query_only_http_request_remains_supported(http_api):
    url, _, _, _ = http_api
    status, result = post(url, {"query":"你好"})
    assert status == 200 and result["success"]
    assert result["provided_sources"] == result["citations"] == []


def test_http_failure_keeps_metrics_and_redacts_provider_error(http_api):
    url, _, local, cloud = http_api
    cloud.replies = [RuntimeError("fake-sensitive-detail")] * 2
    status, result = post(url, {"query":QUERY,"sources":SOURCES})
    assert status == 502 and not result["success"]
    assert result["metrics"]["unknown_usage_calls"] == 2
    assert "fake-sensitive-detail" not in json.dumps(result)
    assert not local.calls


def test_explicit_server_demo_does_not_load_real_config(monkeypatch):
    from unittest.mock import Mock
    import dot_agent.server as server_module
    def forbidden():
        raise AssertionError("real backend factory must not be called in demo")
    monkeypatch.setattr(server_module, "make_clients", forbidden)
    fake_server = Mock()
    monkeypatch.setattr(server_module, "ThreadingHTTPServer", lambda *args: fake_server)
    monkeypatch.setattr(server_module.Handler, "pipeline", None)
    monkeypatch.setattr(server_module.Handler, "runtime", {})
    server_module.serve("127.0.0.1", 0, demo=True)
    assert server_module.Handler.runtime["cloud"] == "stub"
    fake_server.serve_forever.assert_called_once()
