"""Metered FinQA-Priv pair-1 stages. Run from the repository with python -m scripts.finqa_pair1."""
import argparse
import asyncio
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dot_agent.collab.types import CollaborationError
from eval import collab_experiment_runner as R
from scripts.run_real_pilot import BUCKETS, INPUT_CAP, MeteredEndpoint, append


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return list(map(json.loads, path.read_bytes().splitlines())) if path.exists() else []


class Accounting:
    """Single stage process, durable per-physical-attempt upper bounds across stages."""
    def __init__(self, root):
        self.lock = threading.Lock()
        self.log = root / "billing.jsonl"
        self.charges = {x["id"]: Decimal(x["usd"]) for x in read(self.log)}
        self.total = sum(self.charges.values(), Decimal(0))
        self.limit = Decimal("3.50")
        self.stop = False

    def record(self, aid, amount, **fields):
        append(self.log, dict(id=aid, usd=str(amount), **fields))
        self.total += amount - self.charges.get(aid, 0)
        self.charges[aid] = amount

    def invoke(self, ep, wire, aid):
        n, cap = len(wire) + 512, json.loads(wire)["max_tokens"]
        if n > INPUT_CAP or not 0 < cap <= R.CAP:
            raise CollaborationError("input or output reservation exceeded")
        cloud = ep.role == "cloud"
        peak = tuple(map(Decimal, ("0.30", "0.006", "1.20"))) if cloud else (Decimal(0),)*3
        reserve = (n * peak[0] + cap * peak[2]) / 10**6
        start = datetime.now(timezone.utc)
        with self.lock:
            if self.stop or self.total + reserve > self.limit:
                self.stop = True
                raise CollaborationError("budget exhausted")
            self.record(aid, reserve, role=ep.role, state="reserved", time=start.isoformat())
        try:
            response, usage = ep.call(wire, aid)
            if usage.tokens < 0 or usage.raw_input + usage.cache_read > n or usage.output > cap:
                raise ValueError("receipt out of bounds")
        except Exception as exc:
            with self.lock:
                self.record(aid, reserve, role=ep.role, state="unknown", error=type(exc).__name__)
            raise CollaborationError(type(exc).__name__) from None
        # Freeze this experiment to the published non-holiday November 26-27 price windows.
        end = datetime.now(timezone.utc)
        def offpeak(t):
            return t.date().isoformat() in ("2026-11-26", "2026-11-27") and t.hour not in (1,2,3,6,7,8,9)
        divisor = 2 if offpeak(start) and offpeak(end) and start.hour == end.hour else 1
        cost = sum(getattr(usage,k)*v for k,v in zip(BUCKETS,peak))/10**6/divisor
        with self.lock:
            self.record(aid, cost, role=ep.role, state="settled", usage=asdict(usage),
                        rates=[str(v/divisor) for v in peak])
        return response, usage


class PooledEndpoint(R.Endpoint):
    def response(self, wire, headers):
        res = self.client.post(self.url, content=wire, headers=headers)
        res.raise_for_status()
        return res.json()


from contextlib import contextmanager


@contextmanager
def endpoints(meter):
    import httpx
    transport = httpx.HTTPTransport(retries=3, trust_env=False)
    os.environ["COLLAB_CLOUD_API_KEY"] = Path(os.environ["FINQA_CLOUD_KEY_FILE"]).read_text().strip()
    with httpx.Client(transport=transport, trust_env=False, timeout=httpx.Timeout(120,connect=8), follow_redirects=False) as client:
        def endpoint(model,url,role):
            ep = PooledEndpoint(model,url+"/chat/completions",role,"COLLAB_"+role.upper(),"deepseek-account")
            object.__setattr__(ep,"client",client)
            return MeteredEndpoint(ep,meter)
        local = endpoint("Qwen/Qwen3-4B","http://127.0.0.1:8000/v1","local")
        cloud = endpoint("deepseek-flash","https://api.deepseek.com/v1","cloud")
        yield local,cloud


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "dev", "confirm"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".wave.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = args.out / "calibrated_predictor_pair1_v2.json"
    profile = {} if args.stage == "fit" else json.loads(config.read_bytes())
    if args.stage == "confirm":
        gate = json.loads((args.out / "freeze.json").read_bytes())
        if gate["profile_sha256"] != digest(config) or not gate["dev_pass"]:
            raise ValueError("dev freeze required")
    folder = args.out / args.stage
    folder.mkdir(exist_ok=False)
    split = "confirm" if args.stage == "confirm" else "dev"
    packets = [r for r in map(json.loads,R.checked(args.dataset/"public_inputs.jsonl",R.PUBLIC_SHA).splitlines())
               if r["split"] == split]
    tasks = {r["id"]:r["input"] for r in packets}
    ids = sorted(t for t in tasks if t.endswith("-private"))
    assert len(ids) == (160 if split == "confirm" else 32)
    schedule = []
    for i,t in enumerate(ids):
        if args.stage == "fit":
            configs = [(R.ARMS[4],v) for v in ("local","advice","fallback","typed","relation")] if i < 16 else []
            configs += [(R.ARMS[1],None)]
        else:
            configs = [(a,None) for a in R.ARMS if not (a == R.ARMS[1] and split == "dev")]
            if split == "confirm":
                configs += [(R.ARMS[4],v) for v in ("typed","relation")]
        for arm,view in configs:
            schedule.append(dict(task_id=t.replace("-private","-public") if arm==R.ARMS[1] else t, arm=arm,
                 forced_view=view, paired_handoff=split=="confirm" and view is not None, phase="cold", repeat=0, workload="cold", dataset="FinQA-Priv-"+split,
                 model_pair="Qwen3-4B+deepseek-flash", **dict(zip("SHDK",R.factors(arm)))))
    R.lines(folder/"schedule.jsonl",schedule)
    start = time.monotonic()
    import subprocess
    manifest = dict(code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(), split=split,
                    tasks=len(schedule), source=R.PUBLIC_SHA, grader=R.GRADER_SHA, profile=digest(config) if profile else None,
                    start=datetime.now(timezone.utc).isoformat(), seed=41, workers=6, status="running",
                    isolation="separate logical arm and diagnostic DBs; shared physical provider account and cache",
                    temperature=0, max_tokens=R.CAP, thinking=False, cloud_limit_usd="3.50")
    R.dump(folder/"manifest.json",manifest)
    meter = Accounting(args.out)
    with endpoints(meter) as (local,cloud):
        out_lock = threading.Lock()
        def run(slot):
            arm,view,t = slot["arm"],slot["forced_view"],slot["task_id"]
            db = folder/(arm+"-"+(view or "policy")+".sqlite")
            row,pred,wires,_ = asyncio.run(R.episode(slot,tasks[t],local,cloud,db,None,(.05,.05,.2),"real",
                                                       profile,41,forced_view=view))
            if not row["unknown_usage"]:
                with meter.lock:
                    row["C_usd"] = float(sum(meter.charges[u["attempt_id"]] for u in row["usage"]))
            with out_lock:
                append(folder/"raw.jsonl",dict(row=row,prediction=pred,wires=[w.decode() for w in wires]))
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(run,schedule))
    truth = json.loads(R.checked(args.dataset/"gt_manifest.json",R.GRADER_SHA))
    for item in read(folder/"raw.jsonl"):
        row = item["row"]
        t = row["task_id"]
        (row,),_,audit = R.grade([(row,item["prediction"],[w.encode() for w in item["wires"]])],tasks,{t:truth[t]})
        row.update(prediction=item["prediction"], structure=truth[t]["structure"], linkage=audit)
        append(folder/"results.jsonl",row)
    manifest.update(wall_s=time.monotonic()-start, usd_upper=str(meter.total), status="complete")
    R.dump(folder/"manifest.json",manifest)


if __name__ == "__main__":
    main()
