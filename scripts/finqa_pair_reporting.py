"""Regenerate pair-1 tables and paired statistics offline from the full schedules."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, quantiles

from eval import collab_experiment_runner as R, disclosure_audit as A, paper_reporting as P
from scripts.finqa_pair1 import read


def wide(rows, metric, stream=False):
    by={}
    for r in rows:
        t=r["task_id"] if stream else r["family_id"]
        pair=by.setdefault(t,dict(task_id=t,family_id=r["family_id"],
                                  stratum="all" if stream else r["family_id"].split("/")[-1]))
        if r["arm"] in pair:
            raise ValueError("duplicate within pair")
        pair[r["arm"]]=r[metric]
    return list(by.values())


def flow_metrics(rows):
    cold=[r for r in rows if r["case"]=="cold"]
    binds=[r for r in rows if r["case"]=="binding"]
    negative=[r for r in rows if r["case"]=="negative"]
    q=quantiles([r["L_s"] for r in rows],n=100,method="inclusive")
    def measured(group,key):
        return None if any(r[key] is None for r in group) else fmean(r[key] for r in group)
    return dict(U=fmean(r["U"] for r in rows),tasks=len(rows),families=len(cold),
        hot_hits=sum("skill" in r["steps"] for r in binds),eligible_bindings=len(binds),
        zero_egress=sum(r["attempts"]==0 for r in rows),active_blocks=sum("precondition_block" in r["steps"] for r in negative),
        negative_correct=sum(r["U"] for r in negative),negative_tasks=len(negative),
        installed=sum("compiled" in r["steps"] for r in cold),cold_tokens=measured(cold,"C_tokens"),
        cold_seconds=measured(cold,"L_s"),itt_tokens=measured(rows,"C_tokens"),itt_usd=measured(rows,"C_usd"),
        itt_seconds=measured(rows,"L_s"),p50=q[49],p95=q[94],schema_patches=sum("schema_patch" in r["steps"] for r in rows))


def flow_linkage(root):
    """Held-out roster maps synthetic owners to public metric labels. Bindings change across tasks."""
    packets=json.loads((root/"flow-confirm"/"inputs.json").read_bytes())
    roster={v["owner"]:{"metric":v["metric"]} for v in packets.values()}
    prior_metric=set()
    arms=defaultdict(list)
    for event in read(root/"flow-confirm"/"raw.jsonl"):
        row=event["row"]
        packet=packets[row["task_id"]]
        arm=row["arm"]
        family=packet["metric"]
        key=(arm,family)
        facts=[]
        for wire in event["wires"]:
            obj=json.loads(json.loads(wire)["messages"][-1]["content"])
            f={}
            if family in obj.get("question","") or obj.get("metric")==family:
                f["metric"]=family
            if obj.get("owner") in roster:
                f["identity"]=obj["owner"]
            facts.append(f)
        case=dict(target=packet["owner"],family_id=family,background={},messages=facts)
        # A reusable metric label transfers, but prior owners are distinct bindings.
        history=([{"metric":family}] if key in prior_metric else [])+facts
        arms[arm].append((case,{**case,"messages":history}))
        if any("metric" in f for f in facts):
            prior_metric.add(key)
    output={}
    for arm,cases in arms.items():
        current=A.linkage(roster,[x for x,_ in cases])
        cumulative=A.linkage(roster,[x for _,x in cases])
        current["full_history"]=cumulative["full_history"]
        output[arm]={mode:{k:v for k,v in result.items() if k!="details"} for mode,result in current.items()}
    return output


def run(root):
    folder=root/"tables"
    scheduled=read(root/"confirm"/"schedule.jsonl")+read(root/"flow-confirm"/"schedule.jsonl")
    all_rows=read(root/"confirm"/"results.jsonl")+read(root/"flow-confirm"/"results.jsonl")
    def signature(r):
        return (r["task_id"],r["arm"],r.get("forced_view") or "")
    keys=list(map(signature,scheduled))
    if len(set(keys))!=2320 or sorted(keys)!=sorted(map(signature,all_rows)):
        raise ValueError("expected 800 cold, 320 paired replay, and 1200 flow slots")
    main=[r for r in all_rows if not r["forced_view"]]
    cold=[r for r in main if r["workload"]=="cold"]
    stream=[r for r in main if r["workload"]=="stream"]
    summaries=P.export(main,[r for r in scheduled if not r["forced_view"]],folder)
    R.dump(folder/"main.json",summaries)
    pairs=wide(cold,"U")
    b9=R.ARMS[-1]
    comparisons={}
    for arm in R.ARMS[2:4]:
        comparisons[arm]=dict(bootstrap=A.bootstrap(pairs,b9,arm),mcnemar=A.mcnemar([(r[b9],r[arm]) for r in pairs]))
    adj=A.holm({a:c["mcnemar"]["p"] for a,c in comparisons.items()})
    for a in comparisons:
        comparisons[a]["mcnemar"]["holm_p"]=adj[a]
    R.dump(folder/"paired-cold.json",comparisons)
    diagnostics=[dict(r,arm=r["forced_view"]) for r in all_rows if r["forced_view"]]
    tax={}
    for kind in sorted({r["structure"] for r in diagnostics}):
        paired=wide([r for r in diagnostics if r["structure"]==kind],"U")
        tax[kind]=dict(typed=fmean(r["typed"] for r in paired),minimal=fmean(r["relation"] for r in paired),
            minimal_minus_typed=A.bootstrap(paired,"relation","typed"))
    R.dump(folder/"handoff.json",tax)
    linkage={}
    for arm in R.ARMS:
        traces=[r["linkage"].get(arm) for r in cold if r["arm"]==arm]
        traces=[x for x in traces if x]
        linkage[arm]={k:dict(n=len(traces),top1=fmean(x[k]["top1"] for x in traces),
                             confidence=fmean(x[k]["confidence"] for x in traces)) for k in traces[0]} if traces else None
    R.dump(folder/"linkage-cold.json",linkage)
    R.dump(folder/"linkage-flow.json",flow_linkage(root))
    arms=sorted({r["arm"] for r in stream})
    stats={a:flow_metrics([r for r in stream if r["arm"]==a]) for a in arms}
    R.dump(folder/"flow-itt.json",stats)
    columns=("U","itt_tokens","itt_usd","itt_seconds","p50","p95","hot_hits","eligible_bindings",
             "installed","cold_tokens","cold_seconds","schema_patches","active_blocks","negative_correct","zero_egress")
    def fmt(v):
        return "NA" if v is None else format(v,".5g")
    cells=[["arm",*columns],*[[arm,*[fmt(stats[arm][c]) for c in columns]] for arm in arms]]
    md=["| "+" | ".join(line)+" |" for line in [cells[0],["---"]*len(cells[0]),*cells[1:]]]
    (folder/"flow-itt.md").write_text("\n".join(md)+"\n",encoding="utf8")
    tex=[r"\begin{tabular}{l"+"r"*len(columns)+"}",r"\toprule"]
    for i,line in enumerate(cells):
        tex.append(" & ".join(v.replace("_",r"\_") for v in line)+r" \\")
        if i==0:
            tex.append(r"\midrule")
    (folder/"flow-itt.tex").write_text("\n".join([*tex,r"\bottomrule",r"\end{tabular}"])+"\n",encoding="utf8")
    groups=defaultdict(dict)
    for metric in ("U","C_tokens","L_s"):
        p=wide(stream,metric,stream=True)
        for arm in arms:
            if arm==b9:
                continue
            if any(r[b9] is None or r[arm] is None for r in p):
                continue
            result=A.bootstrap(p,b9,arm,higher_better=metric=="U")
            result["permutation"]=A.permutation(p,b9,arm)
            groups[metric][arm]=result
    for metric,group in groups.items():
        adjusted=A.holm({a:r["permutation"]["p"] for a,r in group.items()})
        for a,r in group.items():
            r["holm_p"]=adjusted[a]
    R.dump(folder/"paired-flow.json",groups)
    bills={}
    for row in read(root/"billing.jsonl"):
        bills[row["id"]]=row
    cloud=[r for r in bills.values() if r["role"]=="cloud"]
    R.dump(folder/"billing-summary.json",dict(attempts=len(cloud), settled_usd=sum(float(x["usd"]) for x in cloud if x["state"]=="settled"),
           upper_usd=sum(float(x["usd"]) for x in cloud),unknown=sum(x["state"]!="settled" for x in cloud)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
