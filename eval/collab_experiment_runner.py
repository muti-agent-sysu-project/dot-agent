"""Run the pinned synthetic FinQA-Priv v1 benchmark. Grading follows all episodes."""
import ast
import operator
import argparse
import asyncio
import hashlib
import ipaddress
import itertools
import json
import math
import os
import platform
import random
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from dot_agent.collab.privacy_ledger import ReviewedEnvelopes, TrajectoryPrivacyLedger, TransactionalBoundary
from dot_agent.collab.risk_router import JointPrimalDualController, ModularBaselineController
from dot_agent.collab.skill_compiler import Example, LocalTrace, Primitive, TraceEvent, VerifiedSkillRuntime
from dot_agent.collab.types import Action, Candidate, CollaborationError, Forecast, Scope, Usage
from eval import disclosure_audit, paper_reporting

AMORTIZED = "B9_v2_amortized"
ARMS = ("B0_local_only", "B1_cloud_only", "B6_paac", "B8_modular", "B9_joint")
PUBLIC_SHA = "09d685f2e39ac85cb46f54a81302dba2c2a4aa23dac9819abc6463cb62ced294"
GRADER_SHA = "78205ff3787bc3d98813b9a009d93aab6de6ce408ec793ebc3011ad731f22669"
CAP = 1024
PROMPT = ("Solve the financial question using the supplied report and requested period. Return one JSON object "
          'with "expression" (arithmetic using numeric literals, parentheses and + - * /), "unit" '
          '(ratio, percent, number, thousand, million or billion), and "answer". '
          "The expression is evaluated locally. For percentages express points with unit percent, e.g. "
          "100*(12-10)/10. Preserve source scale for unlabelled amounts with unit number. "
          "For yes/no or undefined, use null expression and the literal answer.")
SELECT = "Select the requested report. Return only its integer slot, or -1 if uncertain."


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode()


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("unexpected FinQA-Priv v1 snapshot")
    return data


def dump(path, obj):
    path.write_bytes(encode(obj) + b"\n")


def lines(path, rows):
    path.write_bytes(b"".join(encode(r) + b"\n" for r in rows))


def arithmetic(expression):
    """Finite bounded arithmetic, independent of task data or grading labels."""
    if not isinstance(expression, str) or len(expression) > 512 or not re.fullmatch(r"[0-9.eE+*/() \-]+", expression):
        raise ValueError("invalid arithmetic expression")
    tree = ast.parse(expression.strip(), mode="eval")
    if len(list(ast.walk(tree))) > 128:
        raise ValueError("too many operations")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](calculate(node.left), calculate(node.right))
        else:
            raise ValueError("unsupported operation")
        if not math.isfinite(value):
            raise ValueError("nonfinite result")
        return value

    return calculate(tree.body)


def resolve_output(text):
    """Resolve the same optional tool contract for every arm before offline grading."""
    raw = text.strip()
    fence = re.fullmatch(r"`{3}(?:json)?\s*(.*?)\s*`{3}", raw, re.I | re.S)
    raw = fence[1] if fence else raw
    if not raw.startswith("{"):
        return text
    try:
        obj = json.loads(raw)
        if "expression" not in obj:
            return raw
        if not set(obj) <= {"expression", "unit", "answer"}:
            return ""
        value = arithmetic(obj["expression"]) if obj["expression"] is not None else obj["answer"]
        unit = (obj.get("unit") or "number").lower().strip()
        if unit in ("number", "ratio", "usd", "dollars", "none", "unitless"):
            return str(value)
        if unit in ("percent", "%", "thousand", "million", "billion"):
            return f"{value} {unit}"
    except (ValueError, ArithmeticError, SyntaxError, AttributeError, TypeError, KeyError):
        pass
    return ""


def answer(text, expected, tol=1e-5, *, percent_scale=1, display=""):
    text = text.strip()
    fence = re.fullmatch(r"\x60{3}(?:json|text)?\s*(.*?)\s*\x60{3}", text, re.I | re.S)
    if fence:
        text = fence[1].strip()
    if text.startswith(("{", '"')):
        try:
            value = json.loads(text)
        except ValueError:
            return False
        if isinstance(value, dict):
            value = value.get("answer") if set(value) == {"answer"} else None
        if type(value) not in (str, int, float):
            return False
        text = str(value).strip()
    if isinstance(expected, str):
        return text.strip().lower() == expected.strip().lower()
    numeric = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    text = re.sub(r"^([+-])\s*\$\s*", r"$\1", text)
    match = re.fullmatch(r"(\$?)\s*("+numeric+r")\s*(%|percent|millions?|billions?|thousands?)?", text, re.I)
    if not match:
        return False
    value = float(match[2].replace(",", ""))
    unit = (match[3] or "").lower().removesuffix("s")
    if unit in ("%", "percent"):
        value *= percent_scale
    elif unit:
        canonical = {word.lower() for word in re.findall(r"\b(thousand|million|billion)s?\b", str(display), re.I)}
        if len(canonical) != 1:
            return False
        scales = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
        value *= scales[unit] / scales[canonical.pop()]
    return math.isclose(value, expected, rel_tol=0, abs_tol=tol)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


@dataclass(frozen=True)
class Endpoint:
    model: str
    url: str = ""
    role: str = "local"
    config_prefix: str = ""
    recipient: str = "simulated"

    @classmethod
    def configure(cls, role, simulated, arm=None):
        if simulated:
            return cls("stub-" + role, role=role)
        prefix = "COLLAB_" + role.upper()
        if arm and prefix+"_"+arm.upper()+"_BASE_URL" in os.environ:
            prefix += "_" + arm.upper()
        recipient = os.environ[prefix+"_RECIPIENT"] if role == "cloud" else "loopback"
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", recipient):
            raise ValueError("public recipient group identifier required")
        url = os.environ[prefix + "_BASE_URL"].rstrip("/")
        parts = urlsplit(url)
        loopback = parts.hostname == "localhost"
        try:
            loopback |= ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            pass
        if (not parts.hostname or parts.username or parts.password or parts.query or parts.fragment
                or parts.scheme not in ("http", "https") or (role == "local" and not loopback)
                or (not loopback and parts.scheme != "https")):
            raise ValueError("invalid endpoint origin")
        return cls(os.environ[prefix + "_MODEL"], url + "/chat/completions", role, prefix, recipient)

    def candidate(self, action, view, data, prompt=PROMPT):
        options = {"thinking": {"type": "disabled"}} if self.model in ("deepseek-flash","deepseek-v4-pro") else {}
        wire = encode(dict(**options, model=self.model, temperature=0, max_tokens=CAP, stream=False,
                           messages=[dict(role="system", content=prompt),
                                     dict(role="user", content=encode(data).decode())]))
        return Candidate(action, view, wire, CAP)

    def call(self, wire, aid):
        started = time.monotonic()
        if not self.url:
            msg = json.loads(wire)["messages"]
            task = json.loads(msg[-1]["content"])
            if self.role == "local" and "task" in task and "advice" in task:
                task = task["task"]
            if msg[0]["content"] == SELECT:
                result = str(task.get("selected", 0))
            elif self.role == "local" and "files" in task and "directory" not in task:
                values = growth_inputs(task)
                result = (str(growth(**values)) if len(values)==2 and values["prior"] else "undefined")
            elif set(task) == {"prior", "current"}:
                result = str(growth(**task)) if task["prior"] else "undefined"
            else:
                result = str(int.from_bytes(hashlib.sha256(wire).digest()[:2], "big") % 101)
            return result, Usage(aid, (len(wire)+3)//4, 0, len(result), "simulated",
                                 time.monotonic()-started)
        headers = {"Content-Type": "application/json"}
        key = os.getenv(self.config_prefix + "_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        result = self.response(wire, headers)
        usage = result.get("usage") or {}
        cache = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        native_cache = usage.get("prompt_cache_hit_tokens")
        if cache is None:
            cache = native_cache
        elif native_cache is not None and native_cache != cache:
            cache = None
        # Without cache telemetry the split is unknown, even when input total is known.
        receipt = Usage.from_inclusive_input(aid, usage.get("prompt_tokens"), cache,
                                             usage.get("completion_tokens"), "received",
                                             time.monotonic()-started)
        try:
            body = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            body = ""
        return body if isinstance(body, str) else "", receipt

    def response(self, wire, headers):
        request = Request(self.url, data=wire, headers=headers)
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=90) as response:
            return json.load(response)


def growth(prior, current):
    return 100 * (current - prior) / prior


def growth_inputs(task):
    record, = task["files"].values()
    return {k: record[k] for k in ("prior", "current") if k in record}


class GrowthLibrary:
    """Install only after a first observed solution passes a local arithmetic check."""
    def __init__(self, journal):
        self.skill = None
        self.live = {}
        def run(state, args, key):
            state["result"] = growth(**args)
        op = Primitive((("prior", "int"), ("current", "int")), "float", frozenset(),
                       frozenset({"result"}), ("nonzero",), ("consistent",), True, run,
                       lambda s, a: s["result"])
        self.runtime = VerifiedSkillRuntime({"growth.v1": op}, {
            "nonzero": lambda s, a: a["prior"] != 0,
            "consistent": lambda s, a: math.isclose(s["result"] * a["prior"],
                                                     100*(a["current"]-a["prior"]), abs_tol=1e-8)
        }, "finqa.v1", journal, resolve_trace=lambda _: self.trace, resolve_run=self.live.__getitem__,
           resolve_edit=lambda _: None)

    def learn(self, values, prediction):
        if self.skill or set(values) != {"prior", "current"} or not values["prior"]:
            return
        if not answer(prediction, growth(**values)):
            return
        def case(args):
            return Example(args, {}, lambda s: answer(str(s["result"]), growth(**args)))
        self.trace = LocalTrace((TraceEvent("growth.v1", (), (("prior", "prior"), ("current", "current"))),),
                                case(values), (case(dict(prior=-10, current=7)),),
                                (case(dict(prior=0, current=10)),))
        self.skill = self.runtime.compile("local", "finqa.v1")

    async def hit(self, values):
        if not self.skill or len(values) != 2 or not values["prior"]:
            return None
        key, state = str(uuid.uuid4()), {}
        self.live[key] = (dict(p0=values["prior"], p1=values["current"]), state)
        try:
            await self.runtime.execute(self.skill, key)
            return str(state["result"])
        finally:
            del self.live[key]


def selection(task):
    rows = task["directory"]
    if len(rows) == 1:
        return 0
    if "highest priority" in task["instruction"]:
        return max(range(len(rows)), key=lambda i: rows[i]["priority"])
    slot, = [i for i,r in enumerate(rows) if r["customer"] + "'s account" in task["instruction"]]
    code = re.fullmatch(r"AC-(\d+)-(\d+)", rows[slot]["account"])
    if not code or int(code[1]) % 97 != int(code[2]):
        raise ValueError("invalid account")
    return slot


class Views:
    """Closed vocabulary for the hash-pinned synthetic corpus. Every wire is reviewed locally."""
    def __init__(self, task, cloud, seed):
        self.task, self.cloud = task, cloud
        self.stream = "directory" not in task and "files" in task
        self.public = "table" in task and not self.stream and "directory" not in task
        self.reviewed = {}
        self.rows = task.get("directory", [])
        self.target = selection(task) if self.rows else None
        self.aliases = list(range(len(self.rows)))
        random.Random(seed).shuffle(self.aliases)
        self.literals = ({str(v) for row in self.rows for v in row.values()} if self.rows else
                         {next(iter(task["files"])), *(str(v) for v in next(iter(task["files"].values())).values())}
                         if self.stream else set())
        self.weights = {**{f"field{i}": 1 for i in range(len(self.literals))}, "selection": 1, "report_fingerprint": 1}

    def offer(self, action, name, obj, prompt=PROMPT):
        c = self.cloud.candidate(action, name, obj, prompt)
        body = c.payload.decode()
        facts = {f"field{i}" for i,v in enumerate(sorted(self.literals))
                 if re.search(r"(?<!\d)" + re.escape(v) + r"(?!\d)", body)}
        # Only the full-reference benchmark arm may transmit directory values.
        if name != "full" and facts:
            return None
        if len(self.rows) > 1 and ("selected" in obj or "selection_history" in obj or name == "full"):
            facts.add("selection")
        if self.rows and {"question", "table", "pre_text", "post_text"} <= obj.keys():
            facts.add("report_fingerprint")
        if self.stream:
            facts.update(k for k in ("growth_schema","growth_guard") if k in obj)
        self.reviewed[c.payload, action, CAP] = facts
        return c

    def selector(self, structured):
        if not self.rows:
            return None
        task = {"slots": list(range(len(self.rows))), "goal": "choose report"}
        if structured:
            task["selected"] = self.aliases[self.target]
        else:
            task["directory"] = [
                dict(slot=self.aliases[i], customer=f"C{self.aliases[i]}", priority=f"P{self.aliases[i]}")
                for i in range(len(self.rows))]
            task["goal"] = ("highest priority" if "highest priority" in self.task["instruction"]
                            else f"customer C{self.aliases[self.target]}")
        return self.offer(Action.DELEGATE, "relation" if structured else "typed", task, SELECT)

    def report(self, slot, *, fallback=False, history=()):
        row = self.rows[self.aliases.index(slot)]
        data = dict(question=self.task["question"], **self.task["files"][row["account"]])
        if history:
            data["selection_history"] = history
        return self.offer(Action.FALLBACK if fallback else Action.DELEGATE,
                          "fallback" if fallback else "report", data)

    def advice(self, structured=False):
        data = {"question": self.task["question"]}
        if self.stream:
            values = growth_inputs(self.task)
            if structured:
                data["growth_guard"] = ("schema_drift" if len(values) != 2 else
                                        "defined" if values["prior"] else "undefined")
            else:
                data["growth_schema"] = "two opaque quantities"
        return self.offer(Action.ADVISE, "advice", data,
                          "Suggest a calculation strategy. Do not supply an answer.")


def factors(arm):
    named = {AMORTIZED:(1,1,1,1),"B9_no_S":(0,1,1,1),"B9_no_H":(1,0,1,1),"B9_no_D":(1,1,0,1),
             "B9_no_K":(1,1,1,0),"B9_replan":(1,1,1,1)}
    if arm in named:
        return named[arm]
    if arm in ARMS:
        return {"B8_modular": (1,0,0,1), "B9_joint": (1,1,1,1)}.get(arm, (0,0,0,0))
    return tuple(map(int, arm.removeprefix("SHDK_")))


def configured_arms(names, ablation):
    arms = names.split(",") if names else ([] if ablation else list(ARMS))
    if any(a not in ARMS for a in arms):
        raise ValueError("unknown arm")
    if ablation:
        cells = itertools.product("01", repeat=4) if ablation == "all" else [ablation]
        for cell in cells:
            bits = "".join(cell)
            if not re.fullmatch("[01]{4}", bits):
                raise ValueError("expected four SHDK bits")
            arms.append("SHDK_" + bits)
    if len(set(arms)) != len(arms) or not arms:
        raise ValueError("duplicate or empty arms")
    return arms


def profile_for(mode, profile, candidate, delta, stream):
    name = candidate.view_id
    if mode == "stub":
        # Fabricated forecasts only for exercising the decision paths.
        q = {"local": .86 if stream else .45, "typed": .72, "relation": .91, "advice": .68, "fallback": .94}
        if stream and name == "advice":
            q[name] = .96
        return Forecast(q[name], .1, 0 if name == "local" else len(candidate.payload)/4 + CAP,
                        .1, float(delta), 0.)
    estimate = profile.get(name)
    if estimate is None:
        return Forecast(0, 1, 0, 0, float(delta), 0)
    return Forecast(estimate["q_lower"], estimate["failure_upper"], estimate["remaining_tokens"],
                    estimate["remaining_seconds"], float(delta) + estimate["future_exposure"],
                    estimate["reuse_value"] if stream else 0)


async def episode(slot, task, local, cloud, db, library, prices, mode, profile, seed, *, forced_view=None):
    started = time.monotonic()
    arm = slot["arm"]
    s, h, d, k = factors(arm)
    from eval import finqa_workload as W
    contract = "metric" in task
    view_type = W.ContractViews if contract else Views
    views = view_type(task, cloud, f"{seed}/{slot['task_id']}")
    full = arm == ARMS[1]
    if contract:
        profile = {**profile, **profile.get("_stream", {})}
    elif views.rows:
        kind = ("irrelevant_sensitive_control" if len(views.rows)==1 else "numeric_order_required"
                if "highest priority" in task["instruction"] else "schema_id_linkage")
        profile = {**profile, **profile.get("_by_structure", {}).get(kind, {})}
    selector, advice = views.selector(forced_view == "relation" if forced_view else s), views.advice(s)
    reference = views.offer(Action.FALLBACK, "full", task) if full else None

    fallback = views.report(views.aliases[views.target], fallback=True) if views.rows else None
    # Enumerate only normalized integer model proposals, never reflect free-text cloud output.
    continuations = {}
    if views.rows:
        selected = views.aliases[views.target]
        continuations = {i: views.report(selected, history={"proposed": i, "verified": selected})
                         for i in [-1, *views.aliases]}
    principal = slot["task_id"]
    review = views.reviewed
    weights = views.weights
    if views.stream and not contract:
        # Five frozen independent bindings with four protected field slots and the relation per instance.
        weights = {f"stream-{i}/{name}": 1 for i in range(5)
                   for name in ("field0","field1","field2","field3","selection")}
        shared = {"growth_schema", "growth_guard"}
        weights.update(dict.fromkeys(shared, 1))
        review = {key: {fact if fact in shared else principal+"/"+fact for fact in facts}
                  for key, facts in review.items()}
        principal = "synthetic-growth"
    if contract:
        principal = task["metric"]
    scope = Scope(arm + "/" + principal, arm + "/" + cloud.recipient, "finqa.v1")
    ledger = TrajectoryPrivacyLedger(weights)
    messages, attempts, local_usage = [], [], []
    async def transmit(scope, wire, cap, aid):
        messages.append(wire)
        result, usage = cloud.call(wire, aid)
        return result.encode(), usage
    boundary = TransactionalBoundary(db, scope, ledger, inspect=ReviewedEnvelopes(review),
                                     input_bound=lambda wire: len(wire)+512, transport=transmit,
                                     p_max=1 if full else .25, c_max=250000, network_allowed=arm != ARMS[0])
    p0, _ = boundary.totals()
    steps = []
    marginal_exposure = []
    async def send(candidate):
        if candidate is None:
            raise ValueError("no_legal_view")
        aid = uuid.uuid4().hex
        marginal_exposure.append(float(boundary.quote(candidate)))
        permit = boundary.reserve(scope, candidate, aid)
        attempts.append(aid)
        steps.append(candidate.view_id)
        body, _ = await boundary.send(permit)
        return body.decode()
    def local_call(data, prompt=PROMPT):
        steps.append("local")
        c = local.candidate(Action.LOCAL, "local", data, W.PROMPT if contract and prompt == PROMPT else prompt)
        out, usage = local.call(c.payload, uuid.uuid4().hex)
        local_usage.append(asdict(usage))
        return out

    local_choice = Candidate(Action.LOCAL, "local", b"")
    choices = tuple(c for c in (local_choice, selector, advice, fallback) if c is not None)
    quote = boundary.quote if h else boundary.standalone_quote
    first_build = False
    if arm == AMORTIZED and contract and library is not None:
        first_build = task["metric"] not in library.encountered
        library.encountered.add(task["metric"])
    def forecast(state, candidate):
        delta = 0 if candidate.action == Action.LOCAL else quote(candidate)
        f = profile_for(mode, profile, candidate, delta, views.stream)
        if arm == AMORTIZED:
            # Quote the joint closure once; the quote already contains today's delta.
            exposure = float(delta)
            if candidate.action == Action.DELEGATE:
                if not continuations or any(nxt is None for nxt in continuations.values()):
                    from dot_agent.collab.types import PolicyDeniedError
                    raise PolicyDeniedError("no_legal_continuation")
                exposure = max(float(boundary.quote_sequence((candidate, nxt)))
                               for nxt in continuations.values())
            bonus = 0.
            if first_build:
                net = profile["_amortization"][candidate.view_id]["discounted_net"]
                bonus = max(0., sum(price * resource / scale for price, resource, scale
                                    in zip(prices, net, (1000, 10, 1), strict=True)))
            return Forecast(f.q_lower, f.failure_upper, f.remaining_tokens, f.remaining_seconds,
                            exposure, bonus)
        if h and candidate.action != Action.LOCAL and float(p0) + f.remaining_exposure > .25:
            from dot_agent.collab.types import BudgetExhaustedError
            raise BudgetExhaustedError("predicted continuation exceeds available disclosure")
        if not h:
            return Forecast(f.q_lower, f.failure_upper, f.remaining_tokens, f.remaining_seconds, float(delta), f.reuse_value)
        return f
    options = dict(quality_floor=profile.get("_quality_floor", .6), risk=profile.get("_risk", .35), scales=(1000, 10, 1), budgets=(500, 5, .10), prices=prices, skills=bool(k))
    if arm == ARMS[3]:
        controller = ModularBaselineController(lambda *_: choices, forecast, quote, **options)
    else:
        controller = JointPrimalDualController(lambda *_: choices, forecast, quote, dynamic=bool(d), **options)
    prediction, reason = "", "ok"
    values = growth_inputs(task) if views.stream and not contract else {}
    try:
        event = "skill"
        if contract:
            replay = library.replay(task, arm == "B9_replan") if library and k else None
            hot,event = replay or (None,event)
        else:
            hot = await library.hit(values) if library and k else None
        if hot is not None:
            prediction = hot
            steps.append(event)
        elif full:
            prediction = await send(reference)
        else:
            if forced_view is not None:
                if arm != ARMS[4] or not (slot["task_id"].startswith("dev-") or
                        (slot.get("paired_handoff") and forced_view in ("typed", "relation"))):
                    raise ValueError("forced views restricted to development joint diagnostics")
                choice = next((c for c in choices if c.view_id == forced_view), None)
            elif arm == ARMS[0]:
                choice = local_choice
            elif arm == ARMS[2]:
                choice = selector or advice
            else:
                choice = controller.decide("", choices)
            if choice is None:
                raise ValueError("no_choice")
            if choice.action == Action.LOCAL:
                prediction = local_call(task)
            elif choice.action == Action.FALLBACK:
                prediction = await send(choice)
            elif choice.action == Action.ADVISE:
                hint = await send(choice)
                prediction = local_call({"task": task, "advice": hint})
            else:
                proposed = await send(choice)
                # Local judge validates the cloud's proposed binding against the live directory.
                selected = views.aliases[views.target]
                if arm == ARMS[2]:
                    # Judge has private bindings. Only its bounded, locally verified decision may leave.
                    judged = local_call({"task": task, "alias_slots": views.aliases,
                                         "proposal": proposed}, SELECT)
                    if judged.strip() != str(selected):
                        steps.append("judge_binding_correction")
                if proposed.strip() != str(selected):
                    steps.append("local_binding_correction")
                normalized = next((i for i in views.aliases if str(i) == proposed.strip()), -1)
                prediction = await send(continuations[normalized])
        raw_prediction = prediction
        prediction = resolve_output(prediction)
        if contract:
            prediction = W.resolve(raw_prediction,task)
            if hot is None and library and k and library.learn(task,raw_prediction):
                steps.append("compiled")
        elif library and k:
            library.learn(values, prediction)
    except (CollaborationError, ValueError, OSError, KeyError) as exc:
        reason = type(exc).__name__
    p1, _ = boundary.totals()
    wall = time.monotonic() - started
    receipts = [dict(**asdict(boundary.reconcile(aid)), **boundary.audit_attempt(aid)) for aid in attempts]
    charge = sum(r["reserved_or_billed_tokens"] for r in receipts)
    controller.close_episode(slot["task_id"], charge, wall, float(p1-p0))
    unknown = sum(any(r[b] is None for b in ("raw_input","cache_read","output")) for r in receipts)
    tokens = None if unknown else sum(r[b] for r in receipts for b in ("raw_input","cache_read","output"))
    row = dict(**slot, simulated=mode=="stub", steps=steps, status=reason, L_s=wall, P_ledger=float(p1),
               raw_prediction=locals().get("raw_prediction", prediction), marginal_exposure=marginal_exposure,
               principal=scope.principal, recipient_group=scope.recipient_group, P_before=float(p0), delta_p=float(p1-p0), reserved_or_billed_tokens=charge,
               usage=receipts, local_usage=local_usage, attempts=len(receipts), unknown_usage=unknown,
               C_tokens=tokens, C_usd=0 if not receipts or mode=="stub" else None, E_official=None)
    return row, prediction, messages, controller.prices


def linkage_cases(task, truth, wires):
    """Offline attacker with a public directory of report fingerprints for synthetic accounts."""
    if not task.get("directory"):
        return None
    def fingerprint(document):
        return hashlib.sha256(encode(document)).hexdigest()
    roster = {account: {"report": fingerprint(doc)} for account,doc in task["files"].items()}
    observed = []
    for wire in wires:
        content = json.loads(json.loads(wire)["messages"][-1]["content"])
        facts = {}
        if set(("files", "directory", "instruction")) <= content.keys():
            facts["identity"] = content["directory"][selection(content)]["account"]
        elif all(k in content for k in ("table","pre_text","post_text")):
            facts["report"] = fingerprint({k:content[k] for k in ("table","pre_text","post_text")})
        observed.append(facts)
    case = dict(target=truth["file"], family_id=truth["report_group"], background={}, messages=observed)
    return disclosure_audit.linkage(roster, [case])


def linkage_controls():
    # Balanced two-turn positive controls with matched single-attribute counterexamples.
    roster = {str(i): {"shift": i//4, "zone": i%4} for i in range(16)}
    cases = [dict(target=key, family_id="pair-"+str(int(key)//2), background={},
                  counterexample=bool(int(key)%2),
                  messages=[{"shift": attrs["shift"]}, {"zone": attrs["zone"]}])
             for key, attrs in roster.items()]
    return disclosure_audit.linkage(roster, cases)


def execution_answer(prediction, task, gt):
    expected = gt.get("exe_ans", gt["answer"])
    label = str(gt["answer"])
    scale = 1 if "family_id" in gt else .01
    percent = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*", label)
    if percent and isinstance(expected, (float, int)):
        # Infer whether the annotation reports points or a fraction, never
        # choose the interpretation that favors the model's prediction.
        points = float(percent[1])
        if abs(expected - points) < abs(100 * expected - points):
            scale = 1
    elif re.search(r"multiply\((?:[^,]+,\s*(?:const_)?100|(?:const_)?100,\s*[^)]+)\)$",
                   gt.get("program", "")):
        scale = 1
    units = r"\b(?:in\s+)?(thousands?|millions?|billions?)\b"
    question = task.get("question", "")
    requested = re.findall(r"\bin\s+(thousands?|millions?|billions?)\b", question, re.I)
    if requested:
        label = " ".join(requested)
    elif not re.search(r"percent|ratio|proportion|\bper\b", question, re.I) and not re.search(units, label, re.I):
        document = task.get("files", {}).get(gt.get("file"), task)
        header = (document.get("table") or [[]])[0]
        label = " ".join(str(cell) for cell in header)
    return answer(prediction, expected, gt.get("abs_tol", 1e-5), percent_scale=scale, display=label)


def grade(completed, inputs, truth):
    rows, pairs, attacks = [], {}, {}
    for row, prediction, messages in completed:
        tid, arm = row["task_id"], row["arm"]
        gt = truth[tid]
        row["U"] = int(row["status"]=="ok" and execution_answer(prediction, inputs[tid], gt))
        family = gt.get("report_group") or gt["family_id"]
        row["family_id"] = family
        pairs.setdefault(tid, dict(task_id=tid, family_id=family, stratum=family.split("/")[-1]))[arm] = row["U"]
        # Direct-literal diagnostic on decoded outbound content, separate from official EC² E.
        texts = [json.loads(w)["messages"][-1]["content"] for w in messages]
        units = {str(x) for x in gt["sensitive_units"]}
        hits = sum(any(re.search(r"(?<!\d)"+re.escape(unit)+r"(?!\d)", text) for text in texts)
                   for unit in units)
        row["E_FinQA_exact"] = hits/len(units) if units else None
        audit = linkage_cases(inputs[tid], gt, messages)
        if "family_id" in gt and not audit:
            roster = {}
            for packet in inputs.values():
                rec, = packet["files"].values()
                roster[rec["customer"]] = {}
            target, = inputs[tid]["files"].values()
            observations = []
            for wire in messages:
                content = json.loads(json.loads(wire)["messages"][-1]["content"])
                if "files" in content:
                    record, = content["files"].values()
                    observations.append({"identity": record["customer"]})
                else:
                    observations.append({})
            audit = disclosure_audit.linkage(roster, [dict(target=target["customer"], family_id=family,
                                                         messages=observations, background={})])
        if audit:
            for condition, item in audit.items():
                attacks.setdefault(arm, {}).setdefault(condition, []).append(
                    (family, item["top1"], item["mean_confidence"], item["direct_hits"]))
        rows.append(row)
    summary = {}
    for arm, conditions in attacks.items():
        summary[arm] = {}
        for condition, items in conditions.items():
            n = len(items)
            summary[arm][condition] = dict(tasks=n, families=len({x[0] for x in items}),
                top1=sum(x[1] for x in items)/n, confidence=sum(x[2] for x in items)/n,
                direct_hits=sum(x[3] for x in items))
        baseline = summary[arm]["background_only"]["top1"]
        for item in summary[arm].values():
            item["lift"] = item["top1"] - baseline
            item["relative_lift"] = item["top1"]/baseline - 1 if baseline else None
    return rows, list(pairs.values()), summary


def run(args, *, prepare_endpoints=None, after_episode=None):
    arms = configured_arms(args.arms, args.ablation)
    if args.limit is not None and args.limit < 1:
        raise ValueError("positive limit required")
    public = checked(args.dataset/"public_inputs.jsonl", PUBLIC_SHA)
    public_twin = getattr(args, "public_twin", False)
    if public_twin and (arms != [ARMS[1]] or args.split == "stream"):
        raise ValueError("public twin requires a cold B1 reference")
    suffix = "-public" if public_twin else "-private"
    tasks = [r for r in map(json.loads, public.splitlines()) if r["split"] == args.split
             and (args.split == "stream" or r["id"].endswith(suffix))][:args.limit]
    simulated = args.mode == "stub"
    local = Endpoint.configure("local", simulated) if any(a != ARMS[1] for a in arms) else Endpoint("unused")
    clouds = {a: Endpoint.configure("cloud", simulated, a) for a in arms if a != ARMS[0]}
    if not simulated and clouds:
        if args.journal_dir is None or len({c.recipient for c in clouds.values()}) != len(clouds):
            raise ValueError("real comparison requires durable journal directory and isolated recipient per arm")
        args.journal_dir.mkdir(parents=True, exist_ok=True)
    models = {c.model for c in clouds.values()}
    if len(models) > 1:
        raise ValueError("all compared cloud arms must use the same model")
    cloud_model = next(iter(models), "unused")
    profile_bytes = args.profile.read_bytes() if args.profile else b"{}"
    profile = json.loads(profile_bytes)
    rates = None
    card = json.loads(args.price_card.read_bytes()) if args.price_card else None
    if card:
        if card["model"] != cloud_model or not card["date"]:
            raise ValueError("price card must identify the cloud model and date")
        rates = {k: card[k] for k in ("raw_input","cache_read","output")}
        if any(not math.isfinite(rate) or rate < 0 for rate in rates.values()):
            raise ValueError("invalid USD rate per million tokens")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("fresh output directory required to preserve prior attempts")
    schedule = [dict(task_id=r["id"], repeat=0, arm=a,
                     dataset="FinQA-Priv-"+args.split+("-public" if public_twin else ""), model_pair=local.model+"+"+cloud_model,
                     workload="stream" if args.split=="stream" else "cold",
                     phase="ablation" if a.startswith("SHDK_") else "stream" if args.split=="stream" else "cold",
                     **dict(zip("SHDK", factors(a)))) for a in arms for r in tasks]
    lines(out/"schedule.jsonl", schedule)
    root = Path(__file__).resolve().parents[1]
    def git(*cmd):
        return subprocess.check_output(["git","-C",str(root),*cmd], text=True).strip()
    manifest = dict(simulated=simulated, git_commit=git("rev-parse","HEAD"),
                    working_tree_dirty=bool(git("status","--porcelain")), dataset_sha256=PUBLIC_SHA,
                    grader_sha256=GRADER_SHA, profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
                    mode=args.mode, split=args.split, seed=args.seed, arms=arms, slots=len(schedule),
                    started_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                    python=platform.python_version(), data_revision="finqa-priv-v1/seed-41",
                    grader="finqa-execution-answer-v1", temperature=0, budget_scales=[1000,10,1],
                    average_budgets=[500,5,.1], initial_prices=[.05,.05,.2],
                    local_model=local.model, cloud_model=cloud_model, attempts_per_send=1, concurrency=1,
                    scope="synthetic report-specific principal", token_cap=250000, output_cap=CAP,
                    policy_version="finqa.v1", full_reference_pmax=1, protected_pmax=.25,
                    price_card=({"model": cloud_model, "date": card["date"], **rates} if rates else None),
                    forecast="simulated" if simulated else "external" if profile else "uncalibrated-local-default",
                    status="running")
    dump(out/"run_manifest.json", manifest)
    if prepare_endpoints:
        local, clouds = prepare_endpoints(local, clouds)
    inputs = {r["id"]: r["input"] for r in tasks}
    completed = []
    for arm in arms:
        cloud = clouds.get(arm, Endpoint("unused"))
        journal = (args.journal_dir/"recipient-ledger.sqlite" if cloud.url
                   else out/(arm+"-ledger.sqlite"))
        configs = [slot for slot in schedule if slot["arm"]==arm]
        library = GrowthLibrary(out/(arm+"-skills.sqlite")) if args.split=="stream" and factors(arm)[3] else None
        initial = (.05,.05,.2)
        prices = initial
        for slot in configs:
            row, prediction, messages, prices = asyncio.run(episode(slot, inputs[slot["task_id"]],
                local, cloud, journal, library,
                prices if args.split=="stream" else initial, args.mode, profile, args.seed))
            if rates and not simulated and not row["unknown_usage"]:
                row["C_usd"] = sum(u[k]*rate for u in row["usage"] for k,rate in rates.items())/1e6
            completed.append((row,prediction,messages))
            if after_episode:
                after_episode(row)
    truth = json.loads(checked(args.dataset/"gt_manifest.json", GRADER_SHA))
    rows, pairs, attacks = grade(completed, inputs, truth)
    cluster = (disclosure_audit.bootstrap(pairs, ARMS[4], ARMS[3], seed=args.seed)
               if set(ARMS[3:]) <= set(arms) else None)
    lines(out/"results.jsonl", rows)
    interactions = {}
    for sh in ("00","01","10","11"):
        cells = ["SHDK_"+sh+dk for dk in ("00","01","10","11")]
        if set(cells) <= set(arms):
            interactions[sh] = disclosure_audit.interaction(pairs, cells, seed=args.seed)
    dump(out/"audit.json", dict(simulated=simulated, paired_utility=cluster, linkage=attacks,
                                diagnostic_synthetic_controls=linkage_controls(), D_by_K=interactions))
    paper_reporting.export(rows, schedule, out/"tables")
    manifest.update(status="complete", attempts=sum(r["attempts"] for r in rows),
                    unknown_usage=sum(r["unknown_usage"] for r in rows))
    dump(out/"run_manifest.json", manifest)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("../科研执行/datasets/finqa-priv-v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arms", help="comma-separated core arms")
    p.add_argument("--ablation", nargs="?", const="all", help="all or a four-bit SHDK cell")
    p.add_argument("--split", choices=("dev","confirm","stream"), default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--public-twin", action="store_true", help="use the pinned public B1 reference")
    p.add_argument("--mode", choices=("stub","real"), default="stub")
    p.add_argument("--profile", type=Path, help="frozen development forecasts by view")
    p.add_argument("--price-card", type=Path, help="dated model price card in USD per million")
    p.add_argument("--journal-dir", type=Path, help="shared durable recipient history across real runs")
    p.add_argument("--seed", type=int, default=41)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False))
