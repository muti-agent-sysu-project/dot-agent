import pytest

from eval import collab_experiment_runner as r


@pytest.mark.parametrize(("expression", "expected"), [
    ("(1234.5 - 1100) / 1100", 134.5 / 1100), ("-2 * (+3)", -6)])
def test_expression(expression, expected):
    assert r.arithmetic(expression) == expected


@pytest.mark.parametrize("expr", ["foo()", "2 ** 8", "1/0", "1e500", "1//2", "a.b", "1+1"*200])
def test_bad_expression_is_never_replaced_by_correct_answer(expr):
    assert r.resolve_output(r.encode(dict(expression=expr, unit="number", answer=42)).decode()) == ""


def test_unit_and_nonnumeric_contract():
    assert r.answer(r.resolve_output('{"expression":"100*(9-8)/8","unit":"percent","answer":0}'),
                    .125, percent_scale=.01)
    assert r.resolve_output('{"expression":null,"unit":"number","answer":"undefined"}') == "undefined"


def test_disjoint_flows_and_validated_schema_patch():
    from eval import finqa_workload as w
    tasks, gold = w.generate(False)
    dev,_ = w.generate(True)
    assert len(tasks)==120 and len({v["family_id"] for v in gold.values()})==24
    assert {t["metric"] for t in tasks.values()}.isdisjoint(t["metric"] for t in dev.values())
    family=list(tasks)[:5]
    lib=w.Library()
    assert not lib.learn(tasks[family[0]],'{"expression":"A+B","unit":"number"}')
    assert lib.learn(tasks[family[0]],'{"expression":"(A-B)/A","unit":"number"}')
    for i,event in [(1,"skill"),(2,"skill"),(3,"schema_patch"),(4,"precondition_block")]:
        value,step=lib.replay(tasks[family[i]],False)
        assert step==event and r.answer(value,gold[family[i]]["answer"])
    assert lib.replay(tasks[family[3]],True) is None


def test_second_turn_facts_and_arm_isolation(tmp_path):
    import asyncio
    import random
    from eval.finqa_priv import packet,TYPES
    task,_=packet(dict(qa={"question":"Revenue?"},table=[["year","value"]],pre_text=[],post_text=[]),
                  TYPES[0],random.Random(7))
    db=tmp_path/"shared.db"
    def run(arm,forced):
        return asyncio.run(r.episode(dict(arm=arm,task_id="dev-diagnostic"),task,r.Endpoint("local"),
            r.Endpoint("cloud"),db,None,(.05,.05,.2),"stub",{},41,forced_view=forced))[0]
    b6=run(r.ARMS[2],None)
    b9=run(r.ARMS[4],"relation")
    assert b6["P_before"]==b9["P_before"]==0
    assert len(b9["marginal_exposure"])==2 and all(x>0 for x in b9["marginal_exposure"])
    assert b9["P_ledger"]<=.25

def test_unreceived_charge_survives_new_process_and_cap(tmp_path):
    from decimal import Decimal
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scripts.finqa_pair1 import Accounting,read
    from dot_agent.collab.types import CollaborationError
    meter=Accounting(tmp_path)
    ep=SimpleNamespace(role="cloud",call=Mock(side_effect=TimeoutError))
    wire=r.encode({"max_tokens":1024})
    with pytest.raises(CollaborationError):
        meter.invoke(ep,wire,"lost")
    reopened=Accounting(tmp_path)
    assert reopened.total==meter.total>0
    assert read(reopened.log)[-1]["state"]=="unknown"
    reopened.limit=reopened.total + Decimal(".000001")
    with pytest.raises(CollaborationError):
        reopened.invoke(ep,wire,"another")
    assert ep.call.call_count==1 and reopened.total==meter.total


def test_itt_includes_block_and_unknown_charges():
    from scripts.finqa_pair_reporting import flow_metrics
    rows=[dict(case=case,U=1,L_s=i+1,steps=steps,attempts=0 if steps else 1,
               C_tokens=None if i==0 else 0,C_usd=None if i==0 else 0)
          for i,(case,steps) in enumerate([("cold",[]),("binding",["skill"]),("binding",["skill"]),
                                          ("drift",["schema_patch"]),("negative",["precondition_block"])])]
    out=flow_metrics(rows)
    assert out["itt_tokens"] is None and out["p50"]==3 and out["itt_seconds"]==3
    assert out["hot_hits"]==2 and out["eligible_bindings"]==2 and out["active_blocks"]==1
