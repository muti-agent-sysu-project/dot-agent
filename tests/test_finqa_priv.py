import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

from eval.finqa_priv import TYPES, build, execute, packet, stream, validate


BASE = Path(__file__).resolve().parents[2]/"科研方案"/"analysis"
DEV = BASE/"finqa-stage1"/"dev.json"
PILOT = BASE/"stage1-pilot-v2"/"inputs.jsonl"


def test_real_split_and_reproducibility(tmp_path):
    if not DEV.exists() or not PILOT.exists():
        pytest.skip("workspace FinQA snapshot required")
    m = build(DEV,PILOT,tmp_path)
    assert (m["groups"], m["questions"],len(m["seen"]))==(224,883,6)
    assert m["excluded"]+m["validated"]==m["attempted"]
    gt = json.loads((tmp_path/"gt_manifest.json").read_bytes())
    rows = list(map(json.loads,(tmp_path/"public_inputs.jsonl").read_bytes().splitlines()))
    assert len(rows)==len(gt)==389
    src = {r["id"]:r for r in json.loads(DEV.read_bytes())}
    splits = {}
    for split,n in [("confirm",160),("dev",32)]:
        selected = [r for r in rows if r["split"]==split and r["id"].endswith("private")]
        reports = {gt[r["id"]]["report_group"] for r in selected}
        assert len(reports)==len(selected)==n and not reports.intersection(m["seen"])
        assert {gt[r["id"]]["structure"] for r in selected}==set(TYPES)
        splits[split] = reports
        for r in selected:
            g = gt[r["id"]]
            key,steps = validate(src[g["source"]],r["input"],g["structure"])
            assert (key,steps)==(g["file"],g["steps"])
    assert splits["confirm"].isdisjoint(splits["dev"])
    for r in rows:
        for field in ("answer","exe_ans","gold_inds","sensitive_units","steps","program","structure"):
            assert '"'+field+'":' not in json.dumps(r)
    for name,expected in m["hashes"].items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==expected
    assert m==build(DEV,PILOT,tmp_path)


@pytest.mark.parametrize("kind", TYPES)
def test_corrupt_payload(kind):
    row = dict(table=[["amount"],["3"]],pre_text=["amount is 3"],post_text=[],
               qa=dict(question="Double the amount",program="multiply(3, const_2)",
                       gold_inds={"text_0":"amount is 3"},exe_ans=6))
    payload,_ = packet(row,kind,random.Random(1))
    key,_ = validate(row,payload,kind)
    broken = copy.deepcopy(payload)
    broken["files"][key]["table"][0]=["cents"]
    with pytest.raises(ValueError):
        validate(row,broken,kind)
    row["qa"]["exe_ans"]=7
    with pytest.raises(ValueError):
        validate(row,payload,kind)


def test_stream_and_step_order():
    examples = list(stream(random.Random(3)))
    assert len({next(iter(p["files"])) for p,_ in examples})==5
    for p,g in examples:
        rec, = p["files"].values()
        if g["case"]=="schema_drift":
            assert "prior" not in rec
        if g["case"]=="precondition_failure":
            assert rec["prior"]==0 and g["answer"]=="undefined"
        else:
            old,new=list(rec.values())[1:]
            assert g["answer"]==pytest.approx((new-old)/old*100)
    assert execute("subtract(4, 2), divide(#0, const_2)",[])==[2,1]
