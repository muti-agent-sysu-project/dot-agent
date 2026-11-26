"""Paired offline inference and a finite-roster attacker independent of the ledger."""
import itertools
import math
import random
from collections import defaultdict
from statistics import mean


def clusters(rows, a, b):
    """Wide rows contain paired arm values for a task/repeat in one stratum."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    owners = {}
    for r in rows:
        family = r.get("family_id") or r["report_group"]
        st = r.get("stratum", "all")
        if owners.setdefault(family, st) != st:
            raise ValueError("cluster spans strata")
        repeats = data[st][family][r["task_id"]]
        rep = r.get("repeat", 0)
        delta = r[a]-r[b]
        if rep in repeats or not math.isfinite(delta):
            raise ValueError("duplicate or invalid observation")
        repeats[rep] = delta
    if not data:
        raise ValueError("empty sample")
    return {st:{f:[mean(reps.values()) for reps in tasks.values()] for f,tasks in fam.items()}
            for st,fam in data.items()}


def bootstrap(rows, a, b, draws=10000, seed=41, margin=.03, higher_better=True):
    if draws < 1 or margin < 0:
        raise ValueError("positive draws and nonnegative margin required")
    grouped = clusters(rows, a, b)
    strata = [[(sum(ds),len(ds)) for ds in fs.values()] for fs in grouped.values()]
    total = sum(s for group in strata for s,n in group)
    count = sum(n for group in strata for s,n in group)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [v for group in strata for v in rng.choices(group,k=len(group))]
        samples.append(sum(s for s,n in selected)/sum(n for s,n in selected))
    samples.sort()
    def q(p):
        pos = p*(len(samples)-1)
        lo = math.floor(pos)
        return samples[lo]+(pos-lo)*(samples[math.ceil(pos)]-samples[lo])
    family = {f:mean(ds) for fs in grouped.values() for f,ds in fs.items()}
    loo = {f:(total-sum(ds))/(count-len(ds)) for fs in grouped.values()
           for f,ds in fs.items() if len(ds)<count}
    return dict(mean_diff=total/count, lower95=q(.05), upper95=q(.95), ci95=(q(.025),q(.975)),
                non_inferior=q(.05)>-margin if higher_better else q(.95)<margin,
                tasks=count, clusters=len(family), family_differences=family, leave_one_out=loo)


def primary_gate(utility_pairs, annotated_exposure_pairs, a="B9", b="B8", **sampling):
    utility = bootstrap(utility_pairs, a, b, **sampling)
    exposure = bootstrap(annotated_exposure_pairs, a, b, margin=0, higher_better=False, **sampling)
    return dict(utility=utility, exposure=exposure,
                passed=utility["lower95"] > -.03 and exposure["upper95"] < 0)


def interaction(rows, cells, **sampling):
    """Paired D x K interaction at fixed S,H; cells ordered 00,01,10,11."""
    x00,x01,x10,x11 = cells
    pairs = [dict(r, active=r[x11]-r[x01], inactive=r[x10]-r[x00]) for r in rows]
    return bootstrap(pairs,"active","inactive",**sampling)


def pass_three(repeats):
    repeats = tuple(repeats)
    if len(repeats)!=3 or any(v not in (0,1) for v in repeats):
        raise ValueError("Pass^3 requires three completed success indicators")
    return int(all(repeats))


def mcnemar(pairs):
    pairs = list(pairs)
    if not pairs or any(x not in (0,1) or y not in (0,1) for x,y in pairs):
        raise ValueError("expected independent paired binary outcomes")
    b = sum(x>y for x,y in pairs)
    c = sum(x<y for x,y in pairs)
    p = min(1, 2*sum(math.comb(b+c,i) for i in range(min(b,c)+1))/2**(b+c))
    return dict(n=len(pairs), wins=b, losses=c, p=p)


def permutation(rows, a, b, draws=10000, seed=41):
    groups = clusters(rows,a,b)
    ds = [sum(vals) for g in groups.values() for vals in g.values()]
    obs = abs(sum(ds))
    exact = len(ds)<=16
    rng = random.Random(seed)
    signs = (itertools.product((-1,1),repeat=len(ds)) if exact else
             (rng.choices((-1,1),k=len(ds)) for _ in range(draws)))
    hits = sum(abs(sum(x*y for x,y in zip(ds,s)))>=obs-1e-12 for s in signs)
    return dict(p=hits/2**len(ds) if exact else (hits+1)/(draws+1), exact=exact, clusters=len(ds))


def holm(pvalues):
    ans = {}
    floor = 0
    for i,(k,p) in enumerate(sorted(pvalues.items(), key=lambda x:x[1])):
        if not 0<=p<=1:
            raise ValueError("invalid p")
        floor = max(floor,min(1,p*(len(pvalues)-i)))
        ans[k] = floor
    return ans


def linkage(roster, cases):
    """Messages are pre-extracted attribute dictionaries for ONE target alias per case."""
    modes = ("background_only","last_turn_only","full_history")
    details = {k:[] for k in modes}
    families = set()
    for c in cases:
        target = c["target"]
        if target not in roster:
            raise ValueError("truth absent from roster")
        families.add(c["family_id"])
        messages = c["messages"]
        for mode,extra in zip(modes,([],messages[-1:],messages)):
            facts = [c["background"],*extra]
            candidates = sorted(who for who,attrs in roster.items()
                if all(all((who if k=="identity" else attrs.get(k))==v and v is not None
                           for k,v in msg.items()) for msg in facts))
            guess = candidates[0] if candidates else None
            details[mode].append(dict(candidates=candidates, correct=guess==target,
                reduction=len(roster)-len(candidates), confidence=1/len(candidates) if candidates else 0,
                direct=any(m.get("identity")==target for m in extra), counterexample=c.get("counterexample",False)))
    if not families:
        raise ValueError("no cases")
    baseline = mean(x["correct"] for x in details[modes[0]])
    output = {}
    for k,entries in details.items():
        rate = mean(x["correct"] for x in entries)
        output[k] = dict(n=len(entries), families=len(families), top1=rate, lift=rate-baseline,
            relative_lift=rate/baseline-1 if baseline else None, details=entries,
            counterexamples=sum(x["counterexample"] for x in entries),
            direct_hits=sum(x["direct"] for x in entries), mean_confidence=mean(x["confidence"] for x in entries))
    return output


def annotation_scores(pairs):
    """Adjudicated (predicted, actual, inclusion_probability) audit observations."""
    raw = dict.fromkeys(("tp","fp","tn","fn"),0)
    weighted = raw.copy()
    for pred,true,prob in pairs:
        if not 0<prob<=1 or pred not in (0,1) or true not in (0,1):
            raise ValueError("invalid annotation")
        key = ("t" if pred==true else "f")+("p" if pred else "n")
        raw[key] += 1
        weighted[key] += 1/prob
    tp,fp,fn = (weighted[k] for k in ("tp","fp","fn"))
    ratio = lambda a,b: a/b if b else None
    return dict(raw=raw, weighted=weighted, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
                f1=ratio(2*tp,2*tp+fp+fn))
