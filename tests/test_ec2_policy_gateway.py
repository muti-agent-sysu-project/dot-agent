import json
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from eval.active_privacy_attacker import evaluate, local_judge


def test_active_privacy():
    units = [{"entity_id": "a", "name": "Fixture", "sensitive_fields": [{"field": "dept", "value": "violet"}]}]
    def judge(messages):
        assert "violet" not in str(messages)
        return [{"entity": "Fixture", "field": "dept", "value": "Violet", "steps": [0, 1]}]
    result = evaluate([{"messages": []}]*2, units, judge, judge_model="auditor", router_model="router")
    assert result["recovery_rate"] == 1 and result["recovered_with_multiple_cited_steps"] == 1
    with pytest.raises(ValueError):
        evaluate([], units, judge, judge_model="same", router_model="same")
    with pytest.raises(ValueError):
        local_judge("https://example.org/v1", "auditor")


def test_grounded_linkage():
    requests = [{"messages":[{"content":"Fixture A is assigned handle H1"}]},
                {"messages":[{"content":"H1 works in violet"}]}]
    units = [{"entity_id":"synthetic-id-1","name":"Fixture A",
              "sensitive_fields":[{"field":"dept","value":"violet"}]}]
    def judge(messages):
        return [{"entity":"Fixture A","field":"dept","value":"violet","steps":[0, 1]}]
    result = evaluate(requests, units, judge, judge_model="independent", router_model="edge")
    assert result["literal_split_linkage_rate"] == 1 and result["literal_split_units"] == 1


@pytest.mark.parametrize("arm", ["B6_paac", "B8_modular", "B9_joint"])
def test_policy_gateway_loopback(arm, tmp_path):
    from dot_agent.collab.policy_gateway import serve
    received = []
    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":1}}')
    with ThreadingHTTPServer(("127.0.0.1", 0), Model) as model:
        thread = Thread(target=model.serve_forever)
        thread.start()
        try:
            with serve(arm=arm, task_id="fixture", output=tmp_path, edge_port=model.server_port,
                       cloud_url=f"http://127.0.0.1:{model.server_port}/v1", edge_model="edge", cloud_model="cloud") as port:
                conn = HTTPConnection("127.0.0.1", port, timeout=8)
                conn.request("POST", "/v1/chat/completions", json.dumps({"model":"cloud","messages":[{"role":"user","content":"compare 98765 and 12345"}],"max_tokens":32}))
                response = conn.getresponse()
                assert response.status == 200
                response.read()
                conn.close()
        finally:
            model.shutdown()
            thread.join()
    assert len(received) == 1
    if arm == "B6_paac":
        assert received[0]["model"] == "cloud" and "98765" not in str(received)
    else:
        assert received[0]["model"] == "edge"


def test_surrogates_scoped_and_typed():
    from dot_agent.collab.trajectory_view import SessionView
    raw={"name":"Robin","salary":1234.5,"path":"/home/alex/a.csv","token":"fake-demo",
         "org":"Fixture Inc","contact":"fixture@example.invalid","note":"2026-11-27 +1-212-555-0109"}
    a=SessionView()
    masked=a.mask(raw)
    assert a.restore(masked)==raw and masked==a.mask(raw)
    assert SessionView().mask(raw)!=masked
    assert "fixture@example.invalid" not in str(masked)
    assert "fixture" not in a.mask("contact fixture@example.invalid")
    with pytest.raises(ValueError):
        a.restore("__EC2_000000000000_PATH_7__")
    with pytest.raises(ValueError):
        a.restore({"command":a.alias("fake;unsafe","PATH")})


def test_calibrated_forecast(tmp_path):
    from dot_agent.collab.policy_gateway import predictor
    priors=predictor(None)
    path=tmp_path/"fit.json"
    path.write_text(json.dumps({"version":1,"advantage":{**priors,"planning":.15}}))
    assert predictor(path)["planning"]==.15
    path.write_text(json.dumps({"version":1,"advantage":{**priors,"planning":float("nan")}}))
    with pytest.raises(ValueError):
        predictor(path)


def test_sheet_cells_and_session_coreference():
    from dot_agent.collab.trajectory_view import SessionView
    registry=SessionView()
    msg={"role":"tool","content":"employee_name,company_name,amount\n小朱,甲方,12345\n"}
    view=registry.messages({"messages":[msg]})[0]["content"]
    for private in ("小朱","甲方","12345"):
        assert private not in view
        assert registry.mask(private) in view
    assert registry.restore(view)==msg["content"]


def test_projection_requires_adequate_rows():
    from dot_agent.collab.policy_gateway import weakest
    def make(prompt,content):
        return [{"role":"user","content":prompt},{"role":"tool","content":json.dumps(content)}]
    view,kind=weakest(make("Return only the name and salary fields.",[{"name":"A","salary":1,"id":"x"}]))
    assert kind=="column" and json.loads(view[-1]["content"])==[{"name":"A","salary":1}]
    view,kind=weakest(make("Is total greater than budget?",{"total":18,"budget":14,"cost":3}))
    assert kind=="predicate" and json.loads(view[-1]["content"])=={"comparison_result":True}
    _,kind=weakest(make("Is total greater than budget?",{"total":"USD 18","budget":"EUR 14"}))
    assert kind=="typed"


def test_frozen_transcript_packing_and_targets(tmp_path):
    from eval.audit_frozen_transcripts import packed, targets

    gt = tmp_path / "workspace" / "ACE_Bench" / "fixture" / "gt"
    gt.mkdir(parents=True)
    gt.joinpath("sensitive_items.json").write_text(json.dumps({"items": [{
        "sensitive_fields": [{"field": "dept", "value": "Engineering"},
                             {"field": "credential", "value": "excluded"}]
    }]}))
    assert targets(tmp_path, "fixture") == {"engineering"}
    message = {"role": "user", "content": "shared"}
    result = packed([{"messages": [message]}, {"messages": [message]}])
    assert result["entries"] == [message]
    assert [row["messages"]["entry_indices"] for row in result["requests"]] == [[0], [0]]


@contextmanager
def fixture_model(answer):
    class Model(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def do_POST(self):
            req=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            reply=answer(req)
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())
    seen=[]
    with ThreadingHTTPServer(("127.0.0.1",0),Model) as server:
        thread=Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server.server_port,seen
        finally:
            server.shutdown()
            thread.join()


def chat(port, messages, **options):
    conn=HTTPConnection("127.0.0.1",port,timeout=10)
    try:
        conn.request("POST","/v1/chat/completions",json.dumps({"messages":messages,**options}))
        res=conn.getresponse()
        body=res.read()
        assert res.status==200, (res.status,body)
        if options.get("stream"):
            return [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        return json.loads(body)
    finally:
        conn.close()


def reply(message):
    return {"id":"mock","created":1,"model":"fixture","choices":[{"index":0,"message":message,
            "finish_reason":"tool_calls" if message.get("tool_calls") else "stop"}],
            "usage":{"prompt_tokens":12,"completion_tokens":6,"prompt_tokens_details":{"cached_tokens":0}}}


@pytest.mark.parametrize("arm",["B6_paac","B8_modular","B9_joint"])
def test_full_tool_trajectory(arm,tmp_path):
    from dot_agent.collab.policy_gateway import serve
    from dot_agent.collab.trajectory_view import ALIAS
    email="fixture@example.invalid"
    tools=[{"type":"function","function":{"name":"lookup","description":"Find "+email,
            "parameters":{"type":"object","properties":{"email":{"type":"string"}},"required":["email"]}}}]
    def planner(req):
        if req["messages"][-1]["role"]=="tool":
            if req["model"]=="cloud":
                call_id=req["messages"][-2]["tool_calls"][0]["id"]
                assert call_id==req["messages"][-1]["tool_call_id"]
            return reply({"role":"assistant","content":"done"})
        alias=ALIAS.search(req["messages"][-1]["content"])[0]
        return reply({"role":"assistant","content":None,"tool_calls":[{"type":"function","id":"new_call",
                     "function":{"name":"lookup","arguments":json.dumps({"email":alias})}}]})
    with fixture_model(planner) as (up,seen):
        with serve(arm=arm,task_id="test",output=tmp_path,cloud_url=f"http://127.0.0.1:{up}/v1",
                   edge_model="edge",edge_port=up,cloud_model="cloud") as port:
            msgs=[{"role":"user","content":f"Look up {email}"}]
            msg=chat(port,msgs,tools=tools)["choices"][0]["message"]
            args=json.loads(msg["tool_calls"][0]["function"]["arguments"])
            assert args=={"email":email}
            msgs.extend([msg,{"role":"tool","tool_call_id":"new_call","content":{email:"ok"}[args["email"]]}])
            chunks=chat(port,msgs,tools=tools,stream=True)
            assert chunks[0]["choices"][0]["delta"]["content"]=="done"
    cloud=[x for x in seen if x["model"]=="cloud"]
    assert len(cloud)==(2 if arm=="B6_paac" else 1)
    assert email not in json.dumps(cloud)
