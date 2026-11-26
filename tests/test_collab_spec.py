"""Offline adversarial acceptance cases for the frozen collaboration contracts."""
import asyncio
import hashlib
import itertools
import json
import math
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction

import pytest

from dot_agent.collab.ec2_bridge import EC2Bridge, offline_privacy_exposure
from dot_agent.collab.privacy_ledger import TrajectoryPrivacyLedger, TransactionalBoundary
from dot_agent.collab.risk_router import (FamilyEstimator, JointPrimalDualController,
                                         ModularBaselineController, certify_policies, discounted_reuse)
from dot_agent.collab.sanitizer import Inv, Registry, ViewGenerator, minimum_view
from dot_agent.collab.skill_compiler import (Example, LocalTrace, Primitive, TraceEvent,
                                             VerifiedSkillRuntime, export_openclaw_skill_md, validate_dag)
from dot_agent.collab.types import (Action, BudgetExhaustedError, Candidate, ContractFailedError,
                                   Forecast, Node, PolicyDeniedError, Scope, Span, StaleVersionError,
                                   UnknownUsageError, Usage)


SCOPE = Scope("tenant", "colluding-tools", "p1")
PUBLIC = Candidate(Action.ADVISE, "summary", b"public", 2)
LOCAL = Candidate(Action.LOCAL, "local", b"")


def policy():
    return TrajectoryPrivacyLedger({"night": 1, "region": 1, "identified": 5, "public": 1},
                                  background={"public"}, rules=[({"night", "region"}, "identified")],
                                  forbidden={"identified"}, raw_forbidden=[b"forbidden"])


def inspect(wire, action, cap):
    # Complete allowlisted test protocol. Every byte, including metadata, must match.
    if wire == b"forbidden":
        return ()  # raw-byte denial is independent of the schema inspector
    if wire not in (b"night", b"region", b"public", b"identified"):
        raise PolicyDeniedError("unknown_schema")
    return (wire.decode(),)


async def echo(scope, wire, cap, attempt):
    return b"ok", Usage.from_inclusive_input(attempt, 3, 1, min(1, cap), "completed", .5)


def boundary(tmp_path, *, p_max=1., c_max=100, transport=echo, network_allowed=True, scope=SCOPE):
    return TransactionalBoundary(tmp_path / "journal.sqlite", scope, policy(), inspect=inspect,
                                 input_bound=len, transport=transport, p_max=p_max, c_max=c_max,
                                 network_allowed=network_allowed)


def test_aliases_spans_credentials():
    a, b = Registry(), Registry()
    text = 'Ada password="top secret" and Bearer abc.def'
    masked = a.mask(text, [Span(0, 3, "PERSON")])
    assert all(s not in masked for s in ("Ada", "top secret", "abc.def"))
    assert masked.startswith(a.mask("Ada", [Span(0, 3, "PERSON")]))
    assert a.restore(masked) == text
    assert b.mask(text, [Span(0, 3, "PERSON")]) != masked
    with pytest.raises(ValueError):
        b.restore(masked)


def test_invariants_views_and_minimal_order():
    predicates = (Inv("match", "join", ("a", "b")),)
    g = ViewGenerator(Registry(), predicates, public_raw=("a", "b"),
                      allowed_predicates=("match",))
    cases = [{"a": "X", "b": "X"}, {"a": "Q", "b": "R"}]
    choices = g.candidates(Action.DELEGATE, cases[0], cases[1:])
    assert {c.view_id for c in choices} == {"predicate_only", "local_compute_summary",
                                           "typed_proxy", "allowed_raw"}
    refs = json.loads(next(c.payload for c in choices if c.view_id == "typed_proxy"))
    assert refs["a"] == refs["b"] and refs["a"] != "X"
    assert minimum_view(choices, .1, lambda c: .2 if c.view_id == "predicate_only" else 0,
                        lambda c: 0 if c.view_id == "typed_proxy" else 1).view_id == "typed_proxy"
    assert minimum_view(choices, 0, lambda c: 1, lambda c: 0) is None
    assert len(ViewGenerator(Registry(), ()).candidates(Action.ADVISE, {}, [{}])) == 5

    obligations = (Inv("less", "lt", ("x", "y")), Inv("large", "ge", ("x",), 10),
                   Inv("valid", "format", ("s",), pattern=r"\d{2}"),
                   Inv("two", "length", ("s",), 2), Inv("path", "reachable", ("g", "u", "v")))
    vals = dict(x=11, y=12, s="42", g={"a": ["b"]}, u="a", v="b")
    altered = dict(vals, s="x", x=4, g={})
    g = ViewGenerator(Registry(), obligations, public_raw=vals, forbidden_fields=("x",),
                      allowed_predicates=[p.name for p in obligations])
    choices = g.candidates(Action.ADVISE, vals, [altered])
    assert [c.view_id for c in choices] == ["predicate_only"]  # summary fails alternative
    assert all(json.loads(choices[0].payload).values())
    assert [p.evaluate(altered) for p in obligations] == [True, False, False, False, False]
    assert not ViewGenerator(Registry(), obligations).candidates(Action.ADVISE, vals, [])


def test_concurrent_union_and_fixed_denominator(tmp_path):
    b1, b2 = boundary(tmp_path), boundary(tmp_path)
    barrier = threading.Barrier(2)
    def try_reserve(pair):
        b, wire = pair
        c = replace(PUBLIC, payload=wire)
        assert b.quote(c) == Fraction(1, 8)
        barrier.wait(timeout=2)
        try:
            return b.reserve(SCOPE, c, wire.decode())
        except PolicyDeniedError:
            return None
    with ThreadPoolExecutor(2) as pool:
        permits = list(pool.map(try_reserve, [(b1, b"night"), (b2, b"region")]))
    assert sum(p is not None for p in permits) == 1
    assert policy().denominator == 8
    assert policy().exposure({"night", "region"}) == Fraction(7, 8)
    assert policy().delta({"night"}, {"night"}) == 0
    for wire in (b"forbidden", b"identified", b"unknown"):
        with pytest.raises(PolicyDeniedError):
            b1.reserve(SCOPE, replace(PUBLIC, payload=wire), "blocked")
    with pytest.raises(StaleVersionError):
        boundary(tmp_path, scope=replace(SCOPE, policy_version="p2"))
    with pytest.raises(StaleVersionError):
        boundary(tmp_path, c_max=1000)


def test_unknown_receipt_reopen_and_no_double_send(tmp_path):
    calls = []
    async def timeout(scope, wire, cap, aid):
        calls.append(wire)
        reopened = boundary(tmp_path, c_max=7)
        assert reopened.reconcile(aid).status == "in_flight"
        raise TimeoutError
    b = boundary(tmp_path, c_max=7, transport=timeout)
    c = replace(PUBLIC, payload=b"night")
    p = b.reserve(SCOPE, c, "attempt")
    for bad in (replace(p, payload_digest="bad"), replace(p, ledger_version=0)):
        with pytest.raises(ContractFailedError):
            asyncio.run(b.send(bad))
    with pytest.raises(UnknownUsageError):
        asyncio.run(b.send(p))
    reopened = boundary(tmp_path, c_max=7)
    assert reopened.totals() == (Fraction(1, 8), 7)
    assert reopened.reserve(SCOPE, c, "attempt") == p
    assert reopened.reconcile("attempt").status == "unknown"
    with pytest.raises(UnknownUsageError):
        _ = reopened.reconcile("attempt").tokens
    with pytest.raises(ContractFailedError):
        asyncio.run(reopened.send(p))
    with pytest.raises(BudgetExhaustedError):
        reopened.reserve(SCOPE, c, "retry")
    assert calls == [b"night"]


@pytest.mark.parametrize("pmax,cmax,allowed,expected", [(0, 8, True, Action.ADVISE),
    (0, 7, True, Action.LOCAL), (1, 0, True, Action.LOCAL), (1, 100, False, Action.LOCAL)])
def test_zero_budgets_and_safe_fallback(tmp_path, pmax, cmax, allowed, expected):
    b = boundary(tmp_path, p_max=pmax, c_max=cmax, network_allowed=allowed)
    fallback = replace(PUBLIC, action=Action.FALLBACK, payload=b"identified")
    opts = (LOCAL, PUBLIC, fallback)
    ctrl = JointPrimalDualController(lambda s,k: opts, lambda s,c: Forecast(
        .2 if c == LOCAL else .9, 0, 8, 1, 0, 0), b.quote, scales=(1, 1, 1), budgets=(0, 0, 0))
    assert ctrl.decide("s", opts).action == expected
    assert ctrl.decide("s", (fallback,)) is None


def test_prefix_property_random_orders(tmp_path):
    rng = random.Random(41)
    b = boundary(tmp_path, p_max=.125, c_max=95)
    permits, seen = [], set()
    async def trace():
        charged = 0
        for i in range(100):
            wire = rng.choice([b"region", b"night", b"public", b"identified"])
            try:
                permit = b.reserve(SCOPE, replace(PUBLIC, payload=wire), str(i))
                permits.append((permit, wire))
            except (BudgetExhaustedError, PolicyDeniedError):
                pass
            if permits and (i % 3 == 0 or i == 99):
                rng.shuffle(permits)
                for p, w in permits:
                    _, receipt = await b.send(p)
                    charged += receipt.tokens
                    seen.add(w.decode())
                    assert policy().exposure(seen) <= Fraction(1, 8)
                    assert "identified" not in policy().closure(seen)
                    assert b.totals() == (policy().exposure(seen), charged)
                    assert charged <= 95
                permits.clear()
        assert charged > 0
    asyncio.run(trace())


def test_family_calibration_tax_and_certification():
    e = FamilyEstimator({"rare": "parent", "other": "parent"}, min_families=8, delta=.1)
    for i in range(100):
        e.observe("rare", "same", 0, (1, 2, 0))
    assert e.forecast("rare").q_lower == 0 and e.forecast("rare").failure_upper == 1
    for i in range(9):
        e.observe("other", i, 1, (2, 3, .1))
        e.observe_pair("other", i, .4, .7)
    f = e.forecast("rare")
    radius = math.sqrt(math.log(10)/20)
    assert f.q_lower == pytest.approx(.9-radius)
    assert f.failure_upper == pytest.approx(.1+radius)
    assert e.tax_upper("rare") == pytest.approx(-.3 + 2*math.sqrt(math.log(10)/18))
    bounds, certified = certify_policies({"a": [0]*100, "b": [1]*100}, .05, .2)
    assert certified == ("a",) and bounds["a"] == pytest.approx(math.sqrt(math.log(40)/200))


def test_dual_regret_all_box_vertices_and_idempotence():
    rng = random.Random(41)
    eta, scales, targets = .07, (10, 5, 1), (6, 2, .2)
    c = JointPrimalDualController(None, None, None, scales=scales, budgets=targets, eta=eta)
    history = []
    for i in range(180):
        actual = rng.randrange(16), rng.random()*4, rng.random()
        z = tuple((v-b)/s for v,b,s in zip(actual, targets, scales))
        history.append((c.prices, z))
        c.close_episode(str(i), *actual)
        updated = c.prices
        c.close_episode(str(i), *actual)
        assert c.prices == updated and all(0 <= p <= 2 for p in updated)
    for vertex in itertools.product((0., 2.), repeat=3):
        gap = sum(sum((v-p)*d for v,p,d in zip(vertex, prices, z)) for prices,z in history)
        bound = sum(v*v for v in vertex)/(2*eta) + eta/2*sum(sum(x*x for x in z) for _,z in history)
        assert gap <= bound + 1e-10
    with pytest.raises(ValueError):
        JointPrimalDualController(None, None, None, scales=(0,1,1), budgets=targets)
    with pytest.raises(UnknownUsageError):
        c.close_episode("unknown", None, 1, 0)


def test_reuse_risk_and_modular_baseline():
    opts = LOCAL, PUBLIC
    def supply(state, skills):
        return opts
    def estimate(state, c):
        return Forecast(.8 if c == LOCAL else .75, .2 if state == "known" else 1,
                        0, 1, .1, .2 if c == PUBLIC else 0)
    options = dict(scales=(1, 1, 1), budgets=(1, 1, .1))
    for mode, flags, choice in ((JointPrimalDualController, {}, PUBLIC),
                                (JointPrimalDualController, {"skills": False}, LOCAL),
                                (JointPrimalDualController, {"dynamic": False}, LOCAL),
                                (ModularBaselineController, {}, LOCAL)):
        c = mode(supply, estimate, lambda _: .1, **options, **flags)
        assert c.decide("known", opts) == choice
        assert c.decide("unknown", opts) == LOCAL
    assert discounted_reuse([(.5, .2, (4,0,0))], (1,0,0), (10,1,1), 1) == pytest.approx(-.1)

def runtime_fixture(tmp_path, *, idempotent=True):
    writes = []
    def save(state, args, key):
        writes.append(key)
        value = args["value"]
        if state.get("uncertain"):
            state["saved"] = value
            raise TimeoutError
        state["saved"] = -value if state.get("drift") else value
    def fixed(state, args, key):
        state["saved"] = args["value"]
    op = Primitive((("value", "int"),), "int", frozenset(), frozenset({"saved"}),
                   ("allowed",), ("saved",), idempotent, save, lambda state,args: state["saved"])
    ops = {"save.v1": op, "save.v2": replace(op, run=fixed)}
    preds = {"allowed": lambda s,a: a["value"] > 0,
             "saved": lambda s,a: s.get("saved") == a["value"]}
    # Even a raw path, alias, low-domain digest and stack text used as a source
    # identifier must disappear at the parameterization boundary.
    source = "/private/[[EC2:alias]]/Traceback/" + hashlib.sha256(b"1").hexdigest()
    def case(value, **state):
        return Example({source: value}, state, lambda s: s["saved"] == value)
    trace = LocalTrace((TraceEvent("save.v1", (), (("value", source),)),), case(1),
                       (case(5),), (case(-1),))
    traces, runs, edits = {"trace": trace}, {}, {}
    runtime = VerifiedSkillRuntime(ops, preds, "p1", tmp_path/"effects.db",
                                   resolve_trace=traces.__getitem__, resolve_run=runs.__getitem__,
                                   resolve_edit=edits.__getitem__)
    return runtime, traces, runs, edits, writes, case


def test_compile_replay_patch_and_version_drift(tmp_path):
    rt, traces, runs, edits, _, case = runtime_fixture(tmp_path)
    skill = rt.compile("trace", "p1")
    assert skill and skill.parameter_schema == (("p0", "int"),)
    assert all(x not in repr(skill) for x in ("/private", "Traceback", "[[EC2:", hashlib.sha256(b"1").hexdigest()))
    md = export_openclaw_skill_md(skill)
    assert skill.version in md and "/private" not in md
    trace = traces["trace"]
    for alt in ((), (trace.original,), (case(3, drift=True),)):
        traces["bad"] = replace(trace, alternatives=alt)
        assert rt.compile("bad", "p1") is None
    traces["bad"] = replace(trace, negatives=(case(2),))
    assert rt.compile("bad", "p1") is None
    run = str(uuid.uuid4())
    live = {"drift": True}
    runs[run] = {"p0": 17}, live
    with pytest.raises(StaleVersionError):
        asyncio.run(rt.execute(replace(skill, policy_version="p2"), run))
    with pytest.raises(ContractFailedError):
        asyncio.run(rt.execute(skill, run))
    assert live["saved"] == -17
    edits["bad"] = ("other", "save.v2", case(17, drift=True))
    assert rt.patch(skill, run, "bad") is None
    edits["fix"] = ("n0", "save.v2", case(17, drift=True))
    repaired = rt.patch(skill, run, "fix")
    assert repaired and repaired.version != skill.version
    assert skill.nodes[0].op_version == "save.v1"
    assert asyncio.run(rt.execute(repaired, run)) == "verified"
    assert live["saved"] == 17


def test_unknown_non_idempotent_effect_survives_restart(tmp_path):
    rt, traces, runs, _, writes, _ = runtime_fixture(tmp_path, idempotent=False)
    skill = rt.compile("trace", "p1")
    run = str(uuid.uuid4())
    inputs, state = {"p0": 3}, {"uncertain": True}
    runs[run] = inputs, state
    with pytest.raises(ContractFailedError):
        asyncio.run(rt.execute(skill, run))
    assert state["saved"] == 3
    after = len(writes)
    with pytest.raises(ContractFailedError):
        asyncio.run(rt.execute(skill, run))
    assert len(writes) == after
    reopened, _, runs2, _, writes2, _ = runtime_fixture(tmp_path, idempotent=False)
    assert reopened.compile("trace", "p1") == skill
    runs2[run] = inputs, {}
    after = len(writes2)
    with pytest.raises(ContractFailedError):
        asyncio.run(reopened.execute(skill, run))
    assert len(writes2) == after and not runs2[run][1]


@pytest.mark.parametrize("defect", ["cycle", "duplicate", "missing", "writes"])
def test_graph_independent_rejection(tmp_path, defect):
    rt, *_ = runtime_fixture(tmp_path)
    n = Node("a", "save.v1", (), (), ("allowed",), ("saved",), "write", None)
    b = replace(n, key="b")
    graphs = {"cycle": (replace(n, dependencies=("b",)), replace(b, dependencies=("a",))),
              "duplicate": (n,n), "missing": (replace(n, dependencies=("x",)),), "writes": (n,b)}
    with pytest.raises(ContractFailedError):
        validate_dag(graphs[defect], rt.ops)


def test_ec2_parallel_span_compile_cost_and_local_hit(tmp_path):
    clock = [4.]
    pending = set()
    async def send(scope, payload, cap, attempt):
        pending.add(attempt)
        await asyncio.sleep(0)
        assert pending == {"compile", "advise"}
        clock[0] = 7.
        return b"ok", Usage.from_inclusive_input(attempt, 5, 2, 2, "completed", 3.)
    b = boundary(tmp_path, transport=send)
    bridge = EC2Bridge(SCOPE, b, clock=lambda: clock[0])
    async def parallel():
        return await asyncio.gather(*(bridge.intercept(Action.ADVISE, b"public", attempt,
                       output_cap=2, view_id="summary") for attempt in ("compile","advise")))
    results = asyncio.run(parallel())
    assert all((u.raw_input, u.cache_read, u.output) == (3,2,2) for _,u in results)
    metrics = bridge.finish()
    assert (metrics.attempts, metrics.tokens, metrics.wall_s) == (2, 14, 3.)
    rt, _, runs, _, _, _ = runtime_fixture(tmp_path)
    skill = rt.compile("trace", "p1")
    hot = EC2Bridge(SCOPE, b)
    run = str(uuid.uuid4())
    runs[run] = {"p0": 42}, {}
    before = b.totals()
    asyncio.run(rt.execute(skill, run))
    hit = hot.finish()
    assert runs[run][1] == {"saved": 42}
    assert (hit.attempts, hit.tokens) == (0, 0) and b.totals() == before


def test_offline_gt_unit_mapping():
    doc = {"items": [{"entity_id":"u", "sensitive_fields":[{"field":"a"}, {"field":"b"}]}]}
    rows = [{"entity_id":"u", "field":"a", "score":0}, {"entity_id":"u", "field":"b", "score":1}]
    assert offline_privacy_exposure(doc, {"fields":rows}) == .5
    assert offline_privacy_exposure(None, {"fields":rows}) is None
    with pytest.raises(ContractFailedError):
        offline_privacy_exposure(doc, {"fields":[rows[0],rows[0]]})

def test_reviewed_ec2_http_envelope_and_metadata(tmp_path):
    from dot_agent.collab.privacy_ledger import ReviewedEnvelopes
    from dot_agent.collab.ec2_bridge import EC2ChatEndpoint

    wire = b'{"model":"fixed","messages":[{"role":"system","content":"public"}],"max_tokens":2}'
    caps = ReviewedEnvelopes({(wire, Action.ADVISE, 2): {"public"}})
    seen = []
    async def transport(scope, payload, cap, attempt):
        seen.append(payload)
        return b'{"choices":[]}', Usage(attempt, 2, 0, 1, "ok", 1)
    b = TransactionalBoundary(tmp_path/"http.db", SCOPE, policy(), inspect=caps,
                              input_bound=len, transport=transport, p_max=0, c_max=200)
    endpoint = EC2ChatEndpoint(EC2Bridge(SCOPE, b), Action.ADVISE)
    async def post(payload):
        replies = []
        async def receive():
            return {"type":"http.request", "body":payload}
        async def send(evt):
            replies.append(evt)
        await endpoint({"type":"http","method":"POST","path":"/v1/chat/completions"}, receive, send)
        return replies[0]["status"]
    assert asyncio.run(post(wire)) == 200
    assert asyncio.run(post(wire.replace(b"public", b"unreviewed"))) == 403
    assert asyncio.run(post(wire.replace(b'"model"', b'"tools":[],"model"'))) == 403
    assert seen == [wire]


def test_persisted_hot_artifact_and_receipt_breach(tmp_path):
    rt, *_ = runtime_fixture(tmp_path)
    skill = rt.compile("trace", "p1")
    reopened, _, runs, _, _, _ = runtime_fixture(tmp_path)
    invocation = str(uuid.uuid4())
    runs[invocation] = {"p0": 34}, {}
    assert asyncio.run(reopened.execute(skill, invocation)) == "verified"
    async def over(scope, payload, cap, attempt):
        return b"ok", Usage(attempt, 123, 0, 2, "ok", 0)
    b = boundary(tmp_path, transport=over)
    with pytest.raises(ContractFailedError):
        asyncio.run(b.send(b.reserve(SCOPE, PUBLIC, "over")))
    assert b.totals()[1] == 125
    with pytest.raises(PolicyDeniedError):
        boundary(tmp_path).reserve(SCOPE, PUBLIC, "next")


def test_b8_execution_gate_sees_prior_disclosure(tmp_path):
    b = boundary(tmp_path)
    night, region = (replace(PUBLIC, payload=w) for w in (b"night", b"region"))
    asyncio.run(b.send(b.reserve(SCOPE, night, "first")))
    opts = LOCAL, region
    ctl = ModularBaselineController(lambda s,k: opts, lambda s,c: Forecast(
        .9 if c == region else .1, 0, 1, 1, 0, 0), b.standalone_quote,
        scales=(10,10,1), budgets=(10,10,.5))
    selected = ctl.decide("next", opts)
    assert selected == region
    with pytest.raises(PolicyDeniedError):
        b.reserve(SCOPE, selected, "second")
