"""Run independent development or frozen formal financial contract streams."""
import argparse
import asyncio
import fcntl
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from eval import collab_experiment_runner as R, finqa_workload as W
from scripts.finqa_pair1 import Accounting, endpoints, digest, read
from scripts.run_real_pilot import append

ABLATIONS = ("B9_no_S","B9_no_H","B9_no_D","B9_no_K","B9_replan")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("flow-fit","flow-confirm"))
    p.add_argument("root",type=Path)
    args=p.parse_args()
    root=args.root
    lock=(root/".wave.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dev=args.stage=="flow-fit"
    config=root/"calibrated_predictor_pair1_v2.json"
    profile={} if dev else json.loads(config.read_bytes())
    if not dev:
        gate=json.loads((root/"freeze.json").read_bytes())
        if not gate["dev_pass"] or gate["profile_sha256"]!=digest(config):
            raise ValueError("unfrozen flow")
    folder=root/args.stage
    folder.mkdir(exist_ok=False)
    packets,labels=W.generate(dev)
    R.dump(folder/"inputs.json",packets)
    # Each job is a complete ordered flow; only independent policies are parallel.
    variants=([(R.ARMS[4],v,learn) for v in ("local","typed","relation","advice","fallback") for learn in (True,False)]
              if dev else [(a,None,True) for a in (*R.ARMS,*ABLATIONS)])
    schedule=[]
    for arm,view,learn in variants:
        for t in packets:
            schedule.append(dict(arm=arm,forced_view=view,library_enabled=learn,task_id=t,repeat=0,
                dataset="FinQA-Priv-contract-dev" if dev else "FinQA-Priv-contract-confirm",
                model_pair="Qwen3-4B+deepseek-flash",phase="ablation" if arm in ABLATIONS else "stream",
                workload="stream", **dict(zip("SHDK",R.factors(arm)))))
    R.lines(folder/"schedule.jsonl",schedule)
    start=time.monotonic()
    manifest=dict(status="running",start=datetime.now(timezone.utc).isoformat(),input_sha256=digest(folder/"inputs.json"),
                  schedule_sha256=digest(folder/"schedule.jsonl"), profile_sha256=digest(config) if not dev else None,
                  code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),workers=6)
    R.dump(folder/"manifest.json",manifest)
    meter=Accounting(root)
    write=threading.Lock()
    with endpoints(meter) as (local,cloud):
        def run(variant):
            arm,view,learn=variant
            library=W.Library() if learn else None
            initial=profile.get("_prices",(.05,.05,.2))
            prices=initial
            journal=folder/f"{arm}-{view}-{learn}.sqlite"
            for slot in schedule:
                if (slot["arm"],slot["forced_view"],slot["library_enabled"])!=variant:
                    continue
                t=slot["task_id"]
                packet=packets[t]
                row,pred,wires,prices=asyncio.run(R.episode(slot,packet,local,cloud,journal,library,
                      prices if not dev else initial,"real",profile,41,forced_view=view))
                if not row["unknown_usage"]:
                    with meter.lock:
                        row["C_usd"]=float(sum(meter.charges[u["attempt_id"]] for u in row["usage"]))
                with write:
                    append(folder/"raw.jsonl",dict(row=row,prediction=pred,wires=[w.decode() for w in wires]))
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(run,variants))
    R.dump(folder/"oracle.json",labels)
    for item in read(folder/"raw.jsonl"):
        row=item["row"]
        gt=labels[row["task_id"]]
        texts=[json.loads(w)["messages"][-1]["content"] for w in item["wires"]]
        exposed=sum(any(v in txt for txt in texts) for v in gt["sensitive_units"])
        row.update(family_id=gt["family_id"],case=gt["case"],prediction=item["prediction"],
                   E_FinQA_exact=exposed/len(gt["sensitive_units"]),
                   U=int(row["status"]=="ok" and R.answer(item["prediction"],gt["answer"])))
        append(folder/"results.jsonl",row)
    manifest.update(status="complete",usd_upper=str(meter.total),wall_s=time.monotonic()-start)
    R.dump(folder/"manifest.json",manifest)


if __name__=="__main__":
    main()
