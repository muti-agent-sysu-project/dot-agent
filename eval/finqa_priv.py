"""Offline paired FinQA benchmark. Only public_inputs.jsonl goes to the runner."""
import argparse
import hashlib
import json
import math
import operator
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

TYPES = ("numeric_order_required", "schema_id_linkage", "irrelevant_sensitive_control")
OPS = dict(add=operator.add, subtract=operator.sub, divide=operator.truediv,
           multiply=operator.mul, exp=operator.pow, greater=lambda a,b: "yes" if a>b else "no")
TOL = 1e-5


def numeric(s):
    s = s.strip().replace(",", "").replace("$", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    return float(s.removesuffix("%"))


def execute(program, table):
    results = []
    def arg(s):
        if s.startswith("#"):
            if not s[1:].isdigit():
                raise ValueError("invalid reference")
            return results[int(s[1:])]
        return numeric(s.removeprefix("const_").replace("m1", "-1"))
    for call in program.split("), "):
        m = re.fullmatch(r"(\w+)\((.*), ([^,]+)", call.removesuffix(")"))
        if not m:
            raise ValueError("invalid instruction")
        op, a, b = m.groups()
        if op.startswith("table_"):
            rows = [r for r in table if r[0] == a]
            if len(rows) != 1 or b != "none":
                raise ValueError("ambiguous row")
            values = list(map(numeric, rows[0][1:]))
            result = dict(table_sum=sum, table_average=mean, table_max=max, table_min=min)[op](values)
        else:
            result = OPS[op](arg(a), arg(b))
        results.append(result)
    return results


def document(row):
    return {k: row[k] for k in ("table", "pre_text", "post_text")}


def packet(row, kind, rng):
    customer = f"Fictional Client {rng.getrandbits(48):012X}"
    keys = rng.sample(range(100000, 999999), 2)
    refs = [f"AC-{k}-{k%97:02}" for k in keys]
    ranks = rng.sample(range(1000, 9000), 2)
    docs = {}
    records = []
    sensitive = [customer]
    for i, ref in enumerate(refs):
        if i and kind == TYPES[2]:
            break
        doc = document(row)
        if i:
            # Synthetic alternative snapshot with different financial amounts.
            doc = json.loads(re.sub(r"\d", lambda m: str((int(m[0])+1)%10), json.dumps(doc, ensure_ascii=False)))
        docs[ref] = doc
        records.append(dict(account=ref, priority=(sorted(ranks, reverse=True) if kind==TYPES[0] else ranks)[i],
                            contract=f"CT-{rng.getrandbits(40):010X}",
                            customer=customer if not i else f"Fictional Client {rng.getrandbits(48):012X}"))
        sensitive.extend([ref, records[-1]["customer"], records[-1]["contract"], str(records[-1]["priority"])])
    rng.shuffle(records)
    docs = dict(sorted(docs.items()))
    if kind == TYPES[0]:
        instruction = "Use the account with the highest priority."
    elif kind == TYPES[1]:
        instruction = f"Look up {customer}'s account. Verify AC-<number>-<number modulo 97, two digits> and open its matching file."
    else:
        instruction = "Use the attached report."
    return dict(instruction=instruction, directory=records, files=docs, question=row["qa"]["question"]), sensitive


def validate(row, payload, kind):
    records = payload["directory"]
    if kind == TYPES[0]:
        rec = max(records, key=lambda r:r["priority"])
        if len({r["priority"] for r in records}) != 2:
            raise ValueError("ambiguous order")
    elif kind == TYPES[1]:
        rec, = [r for r in records if f"{r['customer']}'s account" in payload["instruction"]]
        code = re.fullmatch(r"AC-(\d{6})-(\d{2})", rec["account"])
        if not code or int(code[1]) % 97 != int(code[2]):
            raise ValueError("account checksum")
    else:
        rec, = records
    doc = payload["files"][rec["account"]]
    qa = row["qa"]
    if doc != document(row) or payload["question"] != qa["question"]:
        raise ValueError("source units, evidence or question changed")
    for key in qa["gold_inds"]:
        typ, idx = key.split("_")
        seq = {"table":doc["table"], "text":doc["pre_text"]+doc["post_text"]}[typ]
        if not 0 <= int(idx) < len(seq):
            raise ValueError("unresolved evidence")
        if typ == "text" and seq[int(idx)].strip() != qa["gold_inds"][key].strip():
            raise ValueError("gold text does not match location")
    if not qa["gold_inds"]:
        raise ValueError("missing evidence")
    steps = execute(qa["program"], doc["table"])
    result, gold = steps[-1], qa["exe_ans"]
    if isinstance(gold, str):
        ok = result == gold
    else:
        ok = math.isclose(result, gold, rel_tol=0, abs_tol=TOL)
    if not ok:
        raise ValueError("execution answer mismatch")
    return rec["account"], steps


def stream(rng):
    """Five ordered independent bindings of percentage growth."""
    for i in range(5):
        entity = f"Client-{rng.getrandbits(48):012x}"
        path = f"files/{rng.getrandbits(48):012x}.json"
        old = 0 if i == 3 else rng.randint(10, 1000)
        new = rng.randint(1, 2000)
        fields = ["prior", "current"] if i != 4 else ["previous_amount", "latest_amount"]
        yield (dict(question="Calculate percentage growth from prior to current, or report undefined.",
                    files={path:dict(zip(["customer", *fields], [entity, old, new]))}),
               dict(answer=100*(new-old)/old if old else "undefined", sensitive_units=[entity,path,old,new],
                    case=("precondition_failure" if i==3 else "schema_drift" if i==4 else "regular")))


def build(finqa_dev, stage1_inputs, output_dir, seed=41):
    digest = lambda data: hashlib.sha256(data).hexdigest()
    src = Path(finqa_dev).read_bytes()
    pilot = Path(stage1_inputs).read_bytes()
    rows = json.loads(src)
    groups = defaultdict(list)
    for r in rows:
        groups["/".join(r["id"].split("/")[:2])].append(r)
    seen = set()
    for s in map(json.loads, pilot.splitlines()):
        if s["dataset"].lower() == "finqa":
            group = s["doc_id"].removeprefix("finqa:")
            if group not in groups or not s["id"].startswith("finqa:"+group+"/"):
                raise ValueError("unresolved pilot report")
            seen.add(group)
    required = {"ABMD/2006", "ADI/2011", "AOS/2004", "CNP/2010", "JPM/2008", "MRO/2017"}
    if not required <= seen:
        raise ValueError("incomplete Stage-1 exclusion manifest")
    rng = random.Random(seed)
    # Interleave randomized year strata at systematic midpoints of their
    # empirical mass. Prefixes (including each split) track the year mixture.
    years = defaultdict(list)
    for group in sorted(groups.keys()-seen):
        years[group.split("/")[1]].append(group)
    ranked = []
    for members in years.values():
        rng.shuffle(members)
        ranked.extend(((i+.5)/len(members), rng.random(), g) for i,g in enumerate(members))
    pool = [g for _,_,g in sorted(ranked)]
    public, truth, rejected = [], {}, []
    tried = 0
    for group in pool:
        n = len(truth)//2
        if n == 192:
            break
        split = "confirm" if n < 160 else "dev"
        kind = TYPES[(n if n<160 else n-160)%3]
        candidates = sorted(groups[group], key=lambda r:r["id"])
        rng.shuffle(candidates)
        for row in candidates:
            tried += 1
            try:
                payload, sensitive = packet(row, kind, rng)
                account, steps = validate(row, payload, kind)
            except (ValueError, ArithmeticError, IndexError, KeyError, TypeError) as exc:
                rejected.append(dict(source=row["id"], reason=str(exc)))
                continue
            qa = row["qa"]
            for version in ("public", "private"):
                ident = f"{split}-{n:03}-{version}"
                task = dict(question=qa["question"], **document(row)) if version=="public" else payload
                public.append(dict(id=ident, split=split, input=task))
                truth[ident] = dict(report_group=group, source=row["id"], structure=kind,
                    program=qa["program"], answer=qa["answer"], exe_ans=qa["exe_ans"], steps=steps,
                    abs_tol=TOL, evidence=qa["gold_inds"], file=account if version=="private" else None,
                    sensitive_units=sorted(set(sensitive)) if version=="private" else [])
            break
    if len(truth)!=384:
        raise ValueError("insufficient validated independent reports")
    for i,(task,gt) in enumerate(stream(rng)):
        ident = f"stream-{i}"
        public.append(dict(id=ident, split="stream", input=task))
        truth[ident] = dict(family_id="synthetic-growth", **gt)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)
    files = {"public_inputs.jsonl":"".join(dump(x)+"\n" for x in public),
             "gt_manifest.json":dump(truth)+"\n"}
    hashes = {k:digest(v.encode()) for k,v in files.items()}
    manifest = dict(seed=seed, groups=len(groups), questions=len(rows), seen=sorted(seen),
                    confirm=160, dev=32, stream_instances=5, attempted=tried, validated=192,
                    excluded=len(rejected), exclusion_rate=len(rejected)/tried, rejections=rejected,
                    hashes=hashes, source_sha256=digest(src), pilot_sha256=digest(pilot))
    files["manifest.json"] = dump(manifest)+"\n"
    for name,data in files.items():
        (out/name).write_bytes(data.encode())
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("finqa-dev","stage1-inputs","output-dir"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--seed", type=int, default=41)
    result = build(**vars(p.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k!="rejections"}))
