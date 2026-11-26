import json

import pytest

from dot_agent.collab.privacy_ledger import ReviewedEnvelopes, TransactionalBoundary
from dot_agent.collab.types import Action, BudgetExhaustedError, Candidate, Scope
from eval import collab_experiment_runner as R
from scripts.finqa_calibrate import amortization


def test_amortization_keeps_failures_and_subtracts_first_premium_once():
    rows = []
    for j in range(5):
        for enabled in (False,True):
            cost = 100 + (50 if j == 0 else -20)*enabled
            rows.append(dict(task_id=f"dev-flow-0-{j}",family_id="dev-flow-0",library_enabled=enabled,
                             reserved_or_billed_tokens=cost,L_s=0,delta_p=0))
    assert amortization(rows,gamma=1)["discounted_net"] == [30,0,0]
    assert amortization(rows,gamma=.5)["discounted_net"][0] < 0
    with pytest.raises(ValueError):
        amortization(rows[:-1])
    with pytest.raises(ValueError):
        amortization([dict(rows[0],task_id="confirm-flow-0-0")])


def test_preview_shared_closure_and_no_double_charge(tmp_path):
    # fcntl is specific to the Linux batch runner; use the portable ledger directly.
    from dot_agent.collab.privacy_ledger import TrajectoryPrivacyLedger
    policy = TrajectoryPrivacyLedger({"dept":1,"quarter":1,"identity":8},
                                    rules=[({"dept","quarter"},"identity")])
    d,q = (Candidate(Action.DELEGATE,k,k.encode(),4) for k in ("dept","quarter"))
    scope=Scope("test","cloud","v1")
    gate=TransactionalBoundary(tmp_path/"db",scope,policy,p_max=.25,c_max=100,
         inspect=ReviewedEnvelopes({(c.payload,c.action,c.output_cap):{c.view_id} for c in (d,q)}),
         transport=None,input_bound=len)
    assert gate.quote_sequence((d,d))==gate.quote(d)
    assert gate.standalone_quote(q)==gate.quote(d)
    with pytest.raises(BudgetExhaustedError):
        gate.quote_sequence((d,q))
    assert gate.totals()==(0,0)
    gate.reserve(scope,d,"sent-or-pending")
    with pytest.raises(BudgetExhaustedError):
        gate.reserve(scope,q,"blocked")


def test_v2_is_separate_arm_and_pro_disables_thinking():
    assert R.AMORTIZED not in R.ARMS and R.factors(R.AMORTIZED)==R.factors("B9_joint")
    wire = R.Endpoint("deepseek-v4-pro").candidate(Action.ADVISE,"probe",{}).payload
    assert json.loads(wire)["thinking"] == {"type":"disabled"}


def test_two_turn_choke_allows_same_local_recovery(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from dot_agent.collab.types import Usage
    from scripts import finqa_closure as bench

    meter = SimpleNamespace(charges={})

    class Calculator(R.Endpoint):
        def call(self, wire, aid):
            data=json.loads(json.loads(wire)["messages"][-1]["content"])
            if "gross" in data:
                answer=(data["gross"]-data["credit"])*(100-data["discount_pct"])/100
            else:
                lines=data.get("lines") or [dict(zip(data["fields"],r)) for r in data["records"]]
                answer=max(lines,key=lambda r:r["gross"]-r["credit"])["line"]
            meter.charges[aid]=0
            return str(answer), Usage(aid,10,0,1,"test",.1)

    profile={v:dict(U=1,tokens=t,seconds=.1) for v,t in (("department",10),("typed",20))}
    for arm in bench.ARMS:
        row=asyncio.run(bench.episode("test",bench.workspace(0,True),arm,Calculator("local"),
            Calculator("cloud"),meter,tmp_path,profile,(.1,.1,.1)))
        assert row["U"]==1 and not row["violation"]
        assert row["blocked"]==int(arm!=R.AMORTIZED)
        assert len(row["usage"])==(2 if arm==R.AMORTIZED else 1)


def test_positive_net_only_changes_first_build(tmp_path):
    import asyncio
    from eval import finqa_workload as W

    tasks,_=W.generate(True)
    first,second=list(tasks)[:2]
    def run(t,library,net):
        profile={"_amortization":{v:{"discounted_net":[net if v=="relation" else 0,0,0]}
                                  for v in ("local","relation","advice","fallback")}}
        return asyncio.run(R.episode(dict(arm=R.AMORTIZED,task_id=t),tasks[t],R.Endpoint("local"),
            R.Endpoint("cloud",role="cloud"),tmp_path/(str(net)+".sqlite"),library,(2.,0.,0.),
            "stub",profile,41))[0]
    library=W.Library()
    assert run(first,library,6000)["steps"][0]=="relation"
    assert run(second,library,6000)["steps"][0]=="local"
    assert run(first,W.Library(),-6000)["steps"][0]=="local"
