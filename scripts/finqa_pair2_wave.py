"""Run the metered Pair 2 wave, once per stage, from the Linux repository root."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean

from eval import collab_experiment_runner as R
from scripts.finqa_pair1 import digest, read


def plan(root):
    def rows(stage,n):
        items=read(root/stage/"results.jsonl")
        if len(items)!=n or any(not r["task_id"].startswith("dev-") for r in items):
            raise ValueError("incomplete development stage: "+stage)
        return items
    cold=rows("dev-priced",160)
    fit=rows("fit",112)
    stream=rows("flow-dev",90)
    forced=rows("flow-fit",300)
    closure=rows("closure_choke_24/dev",16)
    def cost(items):
        return fmean(r["C_usd"] for r in items)
    policy=sum(r["C_usd"] for r in cold+[
        r for r in fit if r["arm"]==R.ARMS[1]])*5
    handoff=160*sum(cost([r for r in fit if r["forced_view"]==v]) for v in ("typed","relation"))
    # Five unsampled stream arms bounded here by the highest observed dev-path mean.
    # This is a planning scenario; durable per-attempt upper reservations enforce the hard cap.
    worst_path=max(cost([r for r in forced if r["forced_view"]==v and not r["library_enabled"]])
                   for v in ("relation","typed","advice","fallback"))
    flow=4*sum(r["C_usd"] for r in stream)+5*120*worst_path
    bills={r["id"]:r for r in read(root/"billing.jsonl")}
    if any(r["state"]!="settled" for r in bills.values()):
        raise ValueError("unresolved development receipt")
    paid=sum(float(r["usd"]) for r in bills.values())
    estimate=paid+policy+handoff+flow+72*cost(closure)
    R.dump(root/"budget-plan.json",dict(already_metered=paid,cold_main=policy,handoff=handoff,
           stream=flow,closure=72*cost(closure),estimated_total=estimate,hard_cap=5.99))
    if estimate>=5.99:
        raise ValueError("estimated spend exceeds cap")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("--from-frozen",action="store_true",help="reuse completed and validated development freeze")
    args=p.parse_args()
    root=args.root
    root.mkdir(exist_ok=True,parents=True)
    def run(module,*arguments):
        subprocess.run([sys.executable,"-m","scripts."+module,*map(str,arguments)],check=True,timeout=5400)
    if not args.from_frozen:
        run("finqa_pair2_probe",root)
        run("finqa_flow","flow-fit",root,"--pair2")
        run("finqa_pair1","fit",args.dataset,root,"--pair2")
        run("finqa_calibrate","fit",root,"--pair2")
        run("finqa_pair1","dev",args.dataset,root,"--pair2")
        run("finqa_calibrate","freeze",root,"--pair2")
    freeze=json.loads((root/"freeze.json").read_bytes())
    if not freeze["dev_pass"] or digest(root/"calibrated_predictor_pair2.json")!=freeze["profile_sha256"]:
        raise ValueError("invalid development freeze")
    run("finqa_flow","flow-dev",root,"--pair2")
    run("finqa_closure","dev",root)
    plan(root)
    run("finqa_closure","confirm",root)
    run("finqa_pair1","confirm",args.dataset,root,"--pair2")
    run("finqa_flow","flow-confirm",root,"--pair2")
    run("finqa_pair_reporting",root,"--pair2")


if __name__=="__main__":
    main()
