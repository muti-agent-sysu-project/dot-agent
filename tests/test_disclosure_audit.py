import pytest

from eval.disclosure_audit import annotation_scores, bootstrap, holm, linkage, mcnemar, permutation
from eval.paper_reporting import aggregate, export, frontier
from eval.disclosure_audit import primary_gate, interaction, pass_three


def observations():
    return [dict(report_group=f,task_id=t,repeat=i,A=a,B=0,stratum=s)
            for f,t,a,s in [("x","a",1,"s"),("x","b",0,"s"),("y","c",0,"s"),("z","d",1,"t")]
            for i in range(3)]


def test_paired_statistics():
    rows = observations()
    b = bootstrap(rows,"A","B",draws=500)
    assert b["clusters"]==3 and b["tasks"]==4 and b["mean_diff"]==.5
    assert b["ci95"][0]<=b["lower95"]<=b["upper95"]<=b["ci95"][1]
    assert b["non_inferior"] and b["leave_one_out"]["x"]==.5
    assert permutation(rows,"A","B")["p"]==.5
    with pytest.raises(ValueError):
        bootstrap(rows+rows,"A","B")
    assert mcnemar([(1,0)]*5)["p"]==.0625
    assert mcnemar([(1,1)])["p"]==1
    assert holm({"c":.04,"b":.03,"a":.01})=={"a":.03,"b":.06,"c":.06}


def test_linkage_controls_and_zero_candidates():
    roster = {"a":dict(shift=0,zone=1),"b":dict(shift=1,zone=0),"c":dict(shift=1,zone=1)}
    case = dict(target="c",family_id="f",background={},messages=[{"shift":1},{"zone":1}])
    cases = [case,dict(case,target="b",counterexample=True,messages=[{"shift":1},{"zone":0}])]
    res = linkage(roster,cases)
    assert [r["top1"] for r in res.values()]==[0,.5,1]
    r = res["full_history"]
    assert (r["n"],r["families"],r["counterexamples"],r["lift"],r["relative_lift"])==(2,1,1,1,None)
    assert r["details"][0]["candidates"]==["c"] and r["mean_confidence"]==1
    assert linkage(roster,[dict(case,messages=[{"zone":3}])])["full_history"]["mean_confidence"]==0
    assert annotation_scores([(1,1,1),(1,0,.5)])["precision"]==pytest.approx(1/3)


def test_joint_gate_and_factorial_contrast():
    rows = [dict(r,B9=1,B8=1,x00=0,x01=1,x10=1,x11=3) for r in observations()]
    exposure = [dict(r,B9=0) for r in rows]
    assert primary_gate(rows,exposure,draws=50)["passed"]
    assert not primary_gate(rows,rows,draws=50)["passed"]
    assert interaction(rows,("x00","x01","x10","x11"),draws=50)["mean_diff"]==1
    assert pass_three([1,0,1])==0
    with pytest.raises(ValueError):
        pass_three([1,1])


def test_report_and_plots(tmp_path):
    r = dict(dataset="EC2",model_pair="local_cloud",arm="B9",task_id="t0",repeat=0,
             family_id="f",U=.6,C_tokens=8,C_usd=.01,L_s=1,E_official=None,P_ledger=.3,
             attempts=100,unknown_usage=1,S=1,H=1,D=1,K=1)
    rows = [dict(r,phase=p) for p in ("cold","ablation","stream")]
    rows.append(dict(rows[-1],task_id="t1",U=0))
    summary = export(rows,rows,tmp_path)
    stream, = [v for v in summary if v["phase"]=="stream"]
    assert (stream["U"],stream["C_tokens"],stream["E_official"],stream["status"])==(.3,16,None,"OK")
    assert "local\\_cloud" in (tmp_path/"cold.tex").read_text()
    assert "1111" in (tmp_path/"ablation.md").read_text()
    assert len(list(tmp_path.glob("*.png")))==3
    assert all(p.stat().st_size>1000 for p in tmp_path.glob("*.png"))
    assert frontier([(1,2),(1,3),(2,2),(3,4)])==[(1,3),(3,4)]
    with pytest.raises(ValueError):
        aggregate(rows[:-1],rows)
    with pytest.raises(ValueError):
        aggregate(rows+rows,rows)
    bad = dict(rows[0],unknown_usage=2,L_s=None)
    flags = aggregate([bad],[bad])[0]["status"]
    assert "BLOCK usage" in flags and "BLOCK latency" in flags
