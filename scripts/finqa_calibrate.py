"""Fit forecast summaries on development data, verify independent holdout and bind the freeze."""
import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from eval import collab_experiment_runner as R
from scripts.finqa_pair1 import read,digest


def fit(rows, parent):
    # Empirical-Bayes parent mean with four pseudo-families, 80% one-sided Wilson working bound.
    # Descriptive shrinkage estimates, not a simultaneous coverage certificate.
    if not rows or not parent:
        raise ValueError("empty calibration bucket")
    if len({r["family_id"] for r in rows}) != len(rows):
        raise ValueError("one first encounter per family required")
    z=.8416212335729143
    n=len(rows)+4
    q=(sum(r["U"] for r in rows)+4*fmean(r["U"] for r in parent))/n
    qlo=(q+z*z/(2*n)-z*math.sqrt(q*(1-q)/n+z*z/(4*n*n)))/(1+z*z/n)
    return dict(q_lower=qlo,failure_upper=1-qlo,remaining_tokens=fmean(r["reserved_or_billed_tokens"] for r in rows),
                remaining_seconds=fmean(r["L_s"] for r in rows),
                future_exposure=fmean(sum(r["marginal_exposure"][1:]) for r in rows),reuse_value=0,
                independent_families=len(rows))


def fit_prices(rows,profile):
    """Fit a common static operating point to forced dev actions under the published mean budgets."""
    from itertools import product
    groups={}
    for r in rows:
        if r["forced_view"] in ("local","relation","advice","fallback"):
            groups.setdefault(r["family_id"],{})[r["forced_view"]]=r
    cells=[]
    for cp,lp,pp,floor in product((.05,.1,.15,.2,.25,.3,.4,.5),(.05,.10),(.2,.5,1.),(.6,.7,.8)):
        policies=[]
        for h in (False,True):
            picks=[]
            for group in groups.values():
                offers=[]
                for name,r in group.items():
                    f=profile["_by_structure"][r["structure"]][name]
                    delta=next(iter(r["marginal_exposure"]),0.)
                    exposure=delta+(f["future_exposure"] if h else 0.)
                    if name!="local" and (not r["attempts"] or exposure>.25 or f["failure_upper"]>profile["_risk"]):
                        continue
                    score=cp*f["remaining_tokens"]/1000+lp*f["remaining_seconds"]/10+pp*exposure-f["q_lower"]
                    offers.append((score,name,r,f["q_lower"]))
                if any(n!="fallback" and q>=floor for _,n,_,q in offers):
                    offers=[o for o in offers if o[1]!="fallback"]
                picks.append(min(offers,key=lambda o:(o[0],o[1]))[2])
            policies.append(dict(U=fmean(x["U"] for x in picks),T=fmean(x["reserved_or_billed_tokens"] for x in picks),
                                 L=fmean(x["L_s"] for x in picks),P=fmean(x["delta_p"] for x in picks)))
        over=max(max(0,x["T"]/500-1,x["L"]/5-1,x["P"]/.1-1) for x in policies)
        key=(over,-min(x["U"] for x in policies),-fmean(x["U"] for x in policies),
             fmean(x["T"] for x in policies),cp,lp,pp,floor)
        cells.append((key,(cp,lp,pp),floor,policies))
    _,prices,floor,performance=min(cells,key=lambda c:c[0])
    return {"_prices":prices,"_quality_floor":floor,"_price_fit":performance,"_grid_cells":len(cells)}


def calibrate(root):
    cold=read(root/"fit"/"results.jsonl")
    stream=read(root/"flow-fit"/"results.jsonl")
    if len(cold)!=112 or len(stream)!=300 or any(not r["task_id"].startswith("dev-") for r in cold+stream):
        raise ValueError("incomplete or nondevelopment training set")
    views=("typed","relation","advice","local","fallback")
    profile={"_risk":.70,"_stream":{},"_by_structure":{},"_method":"four parent pseudo-families, one-sided 80% Wilson; rho=.70"}
    for v in views:
        parent=[r for r in cold if r["forced_view"]==v]
        profile[v]=fit(parent,parent)
        for kind in {r["structure"] for r in parent}:
            bucket=[r for r in parent if r["structure"]==kind]
            profile["_by_structure"].setdefault(kind,{})[v]=fit(bucket,[r for r in parent if r["structure"]!=kind])
        coldstream=[r for r in stream if r["forced_view"]==v and r["library_enabled"] and r["case"]=="cold"]
        f=fit(coldstream,coldstream)
        pairs={(r["task_id"],r["library_enabled"]):r for r in stream if r["forced_view"]==v and r["case"]=="binding"}
        gains=[]
        for (t,on),row in pairs.items():
            if not on:
                continue
            ref=pairs[t,False]
            # ITT per warm instance; failed installations also remain.
            gains.append(row["U"]-ref["U"] + .05*(ref["reserved_or_billed_tokens"]-row["reserved_or_billed_tokens"])/1000
                         + .05*(ref["L_s"]-row["L_s"])/10)
        f["reuse_value"]=sum(.9**j for j in (1,2))*fmean(gains)
        profile["_stream"][v]=f
    profile.update(fit_prices(cold,profile))
    profile["_hashes"]={stage:digest(root/stage/"results.jsonl") for stage in ("fit","flow-fit")}
    path=root/"calibrated_predictor_pair1_v2.json"
    if path.exists():
        raise FileExistsError(path)
    R.dump(path,profile)


def freeze(root):
    dev_stage="dev-priced" if (root/"dev-priced").is_dir() else "dev"
    rows=read(root/dev_stage/"results.jsonl")+[r for r in read(root/"fit"/"results.jsonl") if r["arm"]==R.ARMS[1]]
    profile=root/"calibrated_predictor_pair1_v2.json"
    if len(rows)!=160 or any(not r["task_id"].startswith("dev-") for r in rows):
        raise ValueError("incomplete dev evaluation")
    arms={}
    for arm in R.ARMS:
        group=[r for r in rows if r["arm"]==arm]
        assert len({r["task_id"] for r in group})==32
        arms[arm]=dict(U=fmean(r["U"] for r in group),calls=sum(r["attempts"] for r in group),
            unknown=sum(r["unknown_usage"] for r in group),scopes_ok=all(r["P_before"]==0 and
            r["principal"].startswith(arm+"/") and r["recipient_group"].startswith(arm+"/") for r in group))
    forecasts=json.loads(profile.read_bytes())
    ok=(all(x["unknown"]<=.01*x["calls"] and x["scopes_ok"] for x in arms.values()) and
        all(arms[a]["calls"]>0 for a in R.ARMS[3:]) and
        any(f["future_exposure"]>0 for f in forecasts["_stream"].values()) and
        any(f["reuse_value"]>0 for f in forecasts["_stream"].values()))
    R.dump(root/"freeze.json",dict(dev_pass=ok,arms=arms,profile_sha256=digest(profile),
        code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        dev_sha256=digest(root/dev_stage/"results.jsonl"),time=datetime.now(timezone.utc).isoformat()))
    if not ok:
        raise ValueError("development verification failed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("fit","freeze"))
    p.add_argument("root",type=Path)
    a=p.parse_args()
    (calibrate if a.mode=="fit" else freeze)(a.root)
