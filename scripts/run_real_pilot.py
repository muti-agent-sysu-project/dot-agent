"""Run the first 16 frozen private development tasks with two metered health probes."""
import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from dot_agent.collab.types import Action, CollaborationError, Forecast
from eval import collab_experiment_runner as runner

BUCKETS = ("raw_input", "cache_read", "output")
INPUT_CAP = 32768  # physical request is rejected if its conservative bound exceeds this
ARMS = ("B6_paac", "B8_modular", "B9_joint")


def append(path, obj):
    with path.open("ab") as stream:
        stream.write(runner.encode(obj) + b"\n")


class PilotStopped(CollaborationError):
    pass


class Meter:
    """Single process, serial transport. Persist reservation before any possible send."""
    def __init__(self, directory, limit, rates):
        self.directory, self.limit, self.rates = directory, limit, rates
        self.total = Decimal(0)
        self.stop = None

    def halt(self, reason):
        self.stop = reason
        raise PilotStopped(reason)

    def invoke(self, endpoint, wire, attempt):
        if self.stop:
            raise PilotStopped(self.stop)
        n, cap = len(wire) + 512, json.loads(wire)["max_tokens"]
        if n > INPUT_CAP or type(cap) is not int or not 0 < cap <= runner.CAP:
            self.halt("request exceeds audited token bound")
        rates = self.rates if endpoint.role == "cloud" else (Decimal(0),)*3
        reserve = (n * max(rates[:2]) + cap*rates[2]) / 10**6
        if self.total + reserve > self.limit:
            self.halt("USD cap would be exceeded")
        self.total += reserve
        log = self.directory/"billing.jsonl"
        append(log, dict(id=attempt, role=endpoint.role, state="reserved", usd=str(reserve)))
        try:
            result, usage = endpoint.call(wire, attempt)
            usage.tokens  # requires all exclusive buckets
            if usage.raw_input + usage.cache_read > n or usage.output > cap:
                raise ValueError("invalid meter bound")
        except Exception:
            append(log, dict(id=attempt, role=endpoint.role, state="unreconciled"))
            self.halt("unknown usage or endpoint failure")
        cost = sum(getattr(usage,k)*rate for k,rate in zip(BUCKETS,rates)) / 10**6
        self.total += cost - reserve
        append(log, dict(id=attempt, role=endpoint.role, state="settled", usd=str(cost),
                         usage=asdict(usage)))
        return result, usage


class MeteredEndpoint:
    def __init__(self, endpoint, meter):
        self.endpoint, self.meter = endpoint, meter

    def __getattr__(self, name):
        return getattr(self.endpoint, name)

    def call(self, wire, attempt):
        return self.meter.invoke(self.endpoint, wire, attempt)


def plan(args):
    local = runner.Endpoint.configure("local", False)
    cloud = runner.Endpoint.configure("cloud", False, args.arm)
    if not local.model.strip() or not cloud.model.strip():
        raise ValueError("empty model identifier")
    public = runner.checked(args.dataset/"public_inputs.jsonl", runner.PUBLIC_SHA)
    runner.checked(args.dataset/"gt_manifest.json", runner.GRADER_SHA)
    ids = [r["id"] for r in map(json.loads,public.splitlines())
           if r["split"] == "dev" and r["id"].endswith("-private")][:16]
    if len(set(ids)) != 16:
        raise ValueError("16 unique development inputs required")
    card = json.loads(args.price_card.read_bytes())
    date.fromisoformat(card["date"])
    if card["model"] != cloud.model:
        raise ValueError("price card model mismatch")
    if any(type(card[k]) not in (int,float) or not math.isfinite(card[k]) or card[k]<0 for k in BUCKETS):
        raise ValueError("invalid price card")
    rates = tuple(Decimal(str(card[k])) for k in BUCKETS)
    if not args.max_usd.is_finite() or args.max_usd <= 0:
        raise ValueError("invalid USD limit")
    if args.arm != "B6_paac" and not args.profile:
        raise ValueError("calibrated profile required")
    if args.profile:
        profile = json.loads(args.profile.read_bytes())
        for name in ("local","relation","typed","advice","fallback"):
            Forecast(**{k:profile[name][k] for k in ("q_lower","failure_upper","remaining_tokens",
                     "remaining_seconds","reuse_value")}, remaining_exposure=profile[name]["future_exposure"])
    upper = 33 * (INPUT_CAP * max(rates[:2]) + runner.CAP*rates[2]) / 10**6
    return rates, dict(tasks=len(ids), cloud_calls_at_most=33, local_calls_at_most=17,
                       uncapped_usd_upper=str(upper), max_usd=str(args.max_usd),
                       probes_included=True)


def launch(args):
    rates, estimate = plan(args)
    if args.preflight_only or args.dry_run:
        return dict(mode="offline", **estimate)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir/"run"
    meter = Meter(args.output_dir, args.max_usd, rates)

    def prepare(local, clouds):
        endpoints = [MeteredEndpoint(ep, meter) for ep in (local, clouds[args.arm])]
        for ep in endpoints:
            probe = ep.candidate(Action.DELEGATE, "health", {"probe":"ok"}, "Reply OK.")
            body, _ = ep.call(probe.payload, uuid4().hex)
            if not body.strip():
                meter.halt("empty health response")
        return endpoints[0], {args.arm:endpoints[1]}

    def checkpoint(row):
        append(args.output_dir/"episodes.jsonl", row)
        if meter.stop:
            raise PilotStopped(meter.stop)

    opts = argparse.Namespace(**vars(args), arms=args.arm, mode="real", split="dev",
                              limit=16, seed=41, ablation=None)
    opts.output_dir = out
    try:
        result = runner.run(opts, prepare_endpoints=prepare, after_episode=checkpoint)
    except Exception as error:
        manifest_file = out/"run_manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_bytes())
            manifest.update(status="aborted", reason=meter.stop or type(error).__name__)
            runner.dump(manifest_file,manifest)
        raise PilotStopped(meter.stop or type(error).__name__) from None
    result.update(pilot=estimate, usd_including_probes=str(meter.total))
    runner.dump(out/"run_manifest.json", result)
    return dict(slots=result["slots"], usd_including_probes=str(meter.total), tables=str(out/"tables"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, default="B9_joint")
    p.add_argument("--dataset", type=Path, default=ROOT.parent/"科研执行/datasets/finqa-priv-v1")
    p.add_argument("--price-card", required=True, type=Path)
    p.add_argument("--profile", type=Path)
    p.add_argument("--max-usd", type=Decimal, default=Decimal("2.0"))
    p.add_argument("--output-dir", type=Path, default=ROOT.parent/"科研执行/reports/finqa-real-pilot")
    p.add_argument("--journal-dir", type=Path, default=ROOT.parent/"科研执行/recipient-journal")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    try:
        print(json.dumps(launch(parser().parse_args()), ensure_ascii=False))
    except Exception as exc:
        sys.exit("Pilot stopped: " + (str(exc) if isinstance(exc, PilotStopped) else type(exc).__name__))
