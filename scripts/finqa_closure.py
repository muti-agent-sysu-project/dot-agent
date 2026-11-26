"""Two-turn synthetic settlement workspaces with real inference and a fixed public roster."""
import argparse
import asyncio
import json
import random
import subprocess
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from dot_agent.collab.privacy_ledger import ReviewedEnvelopes, TransactionalBoundary, TrajectoryPrivacyLedger
from dot_agent.collab.risk_router import JointPrimalDualController, ModularBaselineController
from dot_agent.collab.types import Action, BudgetExhaustedError, CollaborationError, Forecast, Scope
from eval import collab_experiment_runner as R
from scripts.finqa_pair1 import Accounting, digest, endpoints, read
from scripts.run_real_pilot import append


ARMS = (R.ARMS[2], R.ARMS[3], R.AMORTIZED)
DEPTS = ("Treasury", "Sales", "Operations", "Purchasing", "Risk", "Compliance")
PERIODS = ("Q1", "Q2", "Q3", "Q4")
ROSTER = {f"S{i}{j}": (d,q) for i,d in enumerate(DEPTS) for j,q in enumerate(PERIODS)}
CAP = .25
SELECT = "Return only the integer line number with the largest invoice net of credit."
SETTLE = "Return only the amount due: (gross minus credit) times (100 minus discount_pct) divided by 100."


def workspace(index, dev):
    rng = random.Random((41000 if dev else 78000) + index)
    dept, period = list(ROSTER.values())[index]
    pairs = [(rng.randrange(50,950), rng.randrange(1,40)) for _ in range(3)]
    # Resolve ties before freezing the public workspace.
    while len({g-c for g,c in pairs}) != 3:
        pairs = [(rng.randrange(50,950), rng.randrange(1,40)) for _ in range(3)]
    lines = [dict(line=i, gross=g, credit=c) for i,(g,c) in enumerate(pairs)]
    discount = rng.randrange(1,20)
    return dict(department=dept, period=period, lines=lines, discount_pct=discount)


def policy():
    background = {"roster/"+who+"/"+d+"/"+q for who,(d,q) in ROSTER.items()}
    rules = [({f"dept/{d}",f"period/{q}","roster/"+who+"/"+d+"/"+q}, "identified")
             for who,(d,q) in ROSTER.items()]
    return TrajectoryPrivacyLedger({"identified":1}, background=background, rules=rules)


async def episode(tid, task, arm, local, cloud, meter, folder, profile, prices, forced=None):
    start = time.monotonic()
    reviewed, wires, usage, steps, outputs = {}, [], [], [], []
    d, q, lines = task["department"], task["period"], task["lines"]
    def candidate(view, data, prompt, facts):
        c = cloud.candidate(Action.DELEGATE, view, data, prompt)
        reviewed[c.payload,c.action,c.output_cap] = set(facts)
        return c
    # Rich view is a compact per-department export. Typed export represents the same
    # three ledger lines with explicit units, column meanings, and opaque row references.
    rich = candidate("department",{"department":d,"lines":lines},SELECT,{f"dept/{d}"})
    safe = candidate("typed",{"fields":["line","gross","credit"],"currency":"USD",
                               "records":[[row[k] for k in ("line","gross","credit")] for row in lines]},
                     SELECT,set())
    amounts = [candidate("settlement",dict(period=q,gross=r["gross"],credit=r["credit"],
                                           discount_pct=task["discount_pct"]),SETTLE,{f"period/{q}"}) for r in lines]
    async def transmit(scope,wire,cap,aid):
        wires.append(wire.decode())
        out, receipt = cloud.call(wire,aid)
        usage.append(asdict(receipt))
        return out.encode(),receipt
    scope = Scope(arm+"/"+tid,arm+"/cloud","closure-settlement-v1")
    boundary = TransactionalBoundary(folder/(arm+"-"+tid+".sqlite"), scope, policy(),
          inspect=ReviewedEnvelopes(reviewed),input_bound=lambda buf:len(buf)+512,
          transport=transmit,p_max=CAP,c_max=20000)
    async def send(c):
        permit = boundary.reserve(scope,c,uuid.uuid4().hex)
        steps.append(c.view_id)
        out,_ = await boundary.send(permit)
        return out.decode()
    local_usage = []
    def fallback(c):
        wire = json.loads(c.payload)
        data = json.loads(wire["messages"][-1]["content"])
        local_candidate = local.candidate(Action.LOCAL,"local",data,wire["messages"][0]["content"])
        out,receipt = local.call(local_candidate.payload,uuid.uuid4().hex)
        local_usage.append(asdict(receipt))
        steps.append("local_settlement")
        return out
    blocked = 0
    candidates = (rich,safe)
    def forecast(_,c):
        exposure = (max(float(boundary.quote_sequence((c,n))) for n in amounts)
                    if arm == R.AMORTIZED else float(boundary.standalone_quote(c)))
        f = profile[c.view_id]
        return Forecast(f["U"],0.,f["tokens"],f["seconds"],exposure,0.)
    options = dict(scales=(1000,10,1),budgets=(500,5,.1),prices=prices,skills=False)
    controller = (JointPrimalDualController if arm == R.AMORTIZED else ModularBaselineController)(
        lambda *_: candidates, forecast, boundary.quote if arm == R.AMORTIZED else boundary.standalone_quote, **options)
    status, prediction = "ok", ""
    try:
        first = forced or (rich.view_id if arm == R.ARMS[2] else controller.decide(None,candidates).view_id)
        chosen = next(c for c in candidates if c.view_id==first)
        selected = await send(chosen)
        outputs.append(selected)
        if arm == R.ARMS[2]:
            # Same local judge/correction opportunity as the FinQA PAAC adapter.
            judge = local.candidate(Action.LOCAL,"judge",dict(lines=lines,proposal=selected),SELECT)
            value,receipt = local.call(judge.payload,uuid.uuid4().hex)
            local_usage.append(asdict(receipt))
            outputs.append(value)
            steps.append("local_judge")
            if value.strip() in ("0","1","2"):
                selected = value
        if selected.strip() not in ("0","1","2"):
            raise ValueError("invalid line number")
        nxt = amounts[int(selected)]
        try:
            prediction = await send(nxt)
        except BudgetExhaustedError:
            blocked = 1
            prediction = fallback(nxt)
        outputs.append(prediction)
    except (ValueError,CollaborationError) as exc:
        status = type(exc).__name__
    p,_ = boundary.totals()
    wall = time.monotonic()-start
    target = max(r["gross"]-r["credit"] for r in lines)*(100-task["discount_pct"])/100
    return dict(task_id=tid,family_id=tid,arm=arm,forced_view=forced,steps=steps,blocked=blocked,
                U=int(status=="ok" and R.answer(prediction,target)),prediction=prediction,target=target,
                P_ledger=float(p),violation=p>CAP,L_s=wall,C_tokens=sum(u[k] for u in usage for k in
                ("raw_input","cache_read","output")),C_usd=sum(float(meter.charges[u["attempt_id"]]) for u in usage),
                status=status,usage=usage,local_usage=local_usage,wires=wires,outputs=outputs)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("dev","confirm"))
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    root = args.root
    dev = args.stage=="dev"
    lock = (root/".wave.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out = root/"closure_choke_24"
    folder = out/args.stage
    folder.mkdir(parents=True,exist_ok=False)
    config = out/"dev-profile.json"
    params = root/"calibrated_predictor_pair2.json"
    freeze = json.loads((root/"freeze.json").read_bytes())
    if not freeze["dev_pass"] or digest(params)!=freeze["profile_sha256"]:
        raise ValueError("FinQA dev freeze required")
    if not dev and digest(config)!=(out/"dev-profile.sha256").read_text().strip():
        raise ValueError("closure profile hash mismatch")
    prices = json.loads(params.read_bytes())["_prices"]
    forecasts = {} if dev else json.loads(config.read_bytes())
    ids = [("dev-" if dev else "confirm-")+f"settlement-{i:02}" for i in range(8 if dev else 24)]
    tasks = {t:workspace(i,dev) for i,t in enumerate(ids)}
    jobs = ([(t, "fit-"+v, v) for t in ids for v in ("department","typed")] if dev else
            [(t,arm,None) for t in ids for arm in ARMS])
    R.dump(folder/"inputs.json",tasks)
    R.dump(folder/"roster.json",ROSTER)
    R.dump(folder/"schedule.json",jobs)
    manifest = dict(code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),split=args.stage,
           input_hash=digest(folder/"inputs.json"),schedule_hash=digest(folder/"schedule.json"),
           hash_profile=None if dev else digest(config),prices=prices,p_max=CAP,workers=6,start=time.time(),
           max_steps=3,max_transport_retries=0,status="running")
    R.dump(folder/"manifest.json",manifest)
    meter = Accounting(root,pair2=True)
    with endpoints(meter) as (local,cloud):
        def execute(job):
            if meter.stop:
                raise RuntimeError("billing circuit open")
            t,arm,forced = job
            return asyncio.run(episode(t,tasks[t],arm,local,cloud,meter,folder,forecasts,prices,forced))
        with ThreadPoolExecutor(max_workers=6) as pool:
            for row in pool.map(execute,jobs):
                append(folder/"results.jsonl",row)
    if meter.stop:
        raise RuntimeError("incomplete receipts")
    if dev:
        profiles = defaultdict(list)
        for r in read(folder/"results.jsonl"):
            profiles[r["forced_view"]].append(r)
        # Quote current step only for B8. H adds the ordered closure for B9.
        R.dump(config,{v:dict(U=fmean(int(r["outputs"][0].strip()==str(max(tasks[r["task_id"]]["lines"],
                key=lambda a:a["gross"]-a["credit"])["line"])) for r in rows),
                seconds=fmean(r["usage"][0]["elapsed_s"] for r in rows),
                tokens=fmean(sum(r["usage"][0][k] for k in ("raw_input","cache_read","output"))
                             for r in rows)) for v,rows in profiles.items()})
        (out/"dev-profile.sha256").write_text(digest(config))
    manifest.update(status="complete",end=time.time(),rows=len(jobs))
    R.dump(folder/"manifest.json",manifest)


if __name__ == "__main__":
    main()
