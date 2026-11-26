"""Publish descriptive tables and Pareto plots from normalized, scheduled episodes."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("U","C_tokens","C_usd","L_s","E_official","P_ledger")
KEY = ("dataset","phase","model_pair","arm","task_id","repeat")
PHASES = ("cold","stream","ablation")


def index(rows):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in KEY)
        if key in result or row["phase"] not in PHASES:
            raise ValueError(f"duplicate or unknown phase: {key}")
        result[key] = row
    return result


def aggregate(rows, schedule):
    data, expected = index(rows), index(schedule)
    if not expected or data.keys()!=expected.keys():
        raise ValueError(f"schedule mismatch: missing={len(expected.keys()-data.keys())}, extra={len(data.keys()-expected.keys())}")
    groups = defaultdict(list)
    metrics = (*METRICS, "E_FinQA_exact") if any("E_FinQA_exact" in r for r in rows) else METRICS
    for key,r in data.items():
        for k in metrics:
            x = r[k]
            if x is None and k!="U":
                continue
            if not isinstance(x,(float,int)) or not math.isfinite(x) or x<0 or (k in ("U","E_official","P_ledger","E_FinQA_exact") and x>1):
                raise ValueError(f"invalid {k}: {key}")
        if any(type(r[k])!=int for k in ("attempts","unknown_usage")) or not 0<=r["unknown_usage"]<=r["attempts"]:
            raise ValueError("invalid attempt counts")
        factors = ""
        if r["phase"]=="ablation":
            if any(r[k] not in (0,1) for k in "SHDK"):
                raise ValueError("invalid factors")
            factors = "".join(str(int(r[k])) for k in "SHDK")
        groups[(*key[:4],factors)].append(r)
    result = []
    for (ds,phase,model,arm,factors), episodes in sorted(groups.items()):
        attempts = sum(r["attempts"] for r in episodes)
        unknown = sum(r["unknown_usage"] for r in episodes)
        missing = sum(r["L_s"] is None for r in episodes)
        flags = []
        if unknown>.01*attempts:
            flags.append("BLOCK usage")
        if missing>.01*len(episodes):
            flags.append("BLOCK latency")
        if any(r["C_tokens"] is None or r["C_usd"] is None for r in episodes):
            flags.append("incomplete cost")
        entry = dict(dataset=ds,phase=phase,model_pair=model,arm=arm,SHDK=factors, N=len(episodes),
            usage=f"{unknown}/{attempts}", wall=f"{missing}/{len(episodes)}", status=", ".join(flags) or "OK")
        annotated = [r for r in episodes if r["E_official"] is not None]
        counts = [(r.get("exposed_units"), r.get("total_units")) for r in annotated]
        micro = None
        if counts and all(x is not None and n is not None for x,n in counts):
            if any(not 0<=x<=n or n<=0 for x,n in counts):
                raise ValueError("invalid annotated unit counts")
            micro = sum(x for x,n in counts)/sum(n for x,n in counts)
        entry["E_micro"] = micro
        for k in metrics:
            buckets = defaultdict(list)
            for r in episodes:
                by_flow = r.get("workload", phase)!="cold" and k in ("C_tokens","C_usd","L_s")
                unit = (r["family_id"],r["repeat"]) if by_flow else r["task_id"]
                buckets[unit].append(r[k])
            vals = [(sum(v) if by_flow else mean(v)) for v in buckets.values() if None not in v]
            entry[k] = mean(vals) if vals and (len(vals)==len(buckets) or k=="E_official") else None
            entry[k+"_n"] = len(vals)
        result.append(entry)
    return result


def frontier(points):
    """Minimize x, maximize y. Collapse duplicate operating points."""
    out = []
    for x,y in sorted(set(points),key=lambda v:(v[0],-v[1])):
        if not out or y>out[-1][1]:
            out.append((x,y))
    return out


def export(rows, schedule, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    output = Path(output)
    summary = aggregate(rows,schedule)
    modes = {r.get("simulated") for r in rows}
    if len(modes) != 1:
        raise ValueError("mixed evidence modes")
    simulated = modes == {True}
    output.mkdir(parents=True,exist_ok=True)
    def fmt(v):
        return "NA" if v is None else format(v,".4g") if isinstance(v,float) else str(v)
    def tex(s):
        replacements = {"\\":r"\textbackslash{}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}"}
        return "".join(replacements.get(c, "\\"+c if c in "_%&$#{}" else c) for c in s)
    metrics = (*METRICS, "E_FinQA_exact") if "E_FinQA_exact" in summary[0] else METRICS
    for phase in PHASES:
        cols = ["dataset","model_pair","arm",*(["SHDK"] if phase=="ablation" else []),"N",*metrics,"E_official_n","E_micro","usage","wall","status"]
        records = [[fmt(r[k]) for k in cols] for r in summary if r["phase"]==phase]
        md = ["| "+" | ".join(x.replace("|",r"\|") for x in line)+" |" for line in [cols,["---"]*len(cols),*records]]
        latex = ["\\begin{tabular}{"+len(cols)*"l"+"}",r"\toprule"," & ".join(map(tex,cols))+r" \\",r"\midrule"]
        if simulated:
            md[:0] = ["**Simulated offline workload**", ""]
            latex.insert(0, "% Simulated offline workload")
        latex += [" & ".join(map(tex,line))+r" \\" for line in records]
        (output/(phase+".md")).write_text("\n".join(md)+"\n",encoding="utf8")
        (output/(phase+".tex")).write_text("\n".join([*latex,r"\bottomrule",r"\end{tabular}"])+"\n",encoding="utf8")
    panels = defaultdict(list)
    for r in summary:
        panels[(r["dataset"],r["phase"],r["model_pair"])].append(r)
    for i,((dataset,phase,model),points) in enumerate(panels.items()):
        fig,axes = plt.subplots(1,4,figsize=(13,3.6),layout="constrained")
        exposure = "E_FinQA_exact" if "E_FinQA_exact" in points[0] else "E_official"
        for ax,k,label in zip(axes,(exposure,"P_ledger","C_tokens","L_s"),
                              (exposure,"Ledger exposure","Cloud tokens","Wall time (s)")):
            xy = []
            for r in points:
                if r[k] is None or (k=="C_tokens" and "BLOCK usage" in r["status"]) or (k=="L_s" and "BLOCK latency" in r["status"]):
                    continue
                xy.append((r[k],r["U"]))
                ax.scatter(*xy[-1],s=20)
                ax.annotate(r["arm"],xy[-1],xytext=(3,3),textcoords="offset points",fontsize=7)
            if xy:
                ax.plot(*zip(*frontier(xy)),color=".3",lw=1,zorder=0)
            ax.set(xlabel=label,ylabel="Utility")
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle(("SIMULATED · " if simulated else "") + f"{dataset} · {phase} · {model}")
        fig.savefig(output/f"pareto-{i}.png",dpi=240)
        fig.savefig(output/f"pareto-{i}.pdf")
        plt.close(fig)
    return summary


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results","schedule","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_bytes().splitlines()]
    export(read(args.results),read(args.schedule),args.output_dir)
