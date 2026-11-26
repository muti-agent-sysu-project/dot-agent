"""Deterministic foundation policy; eligibility is assessed locally."""
from .types import Action


def choose_action(*, local_ok: bool, proxy_sufficient: bool,
                  advice_sufficient: bool, full_allowed: bool) -> Action:
    if local_ok:
        return Action.LOCAL
    if proxy_sufficient:
        return Action.PROXY
    if advice_sufficient:
        return Action.ADVISE
    if full_allowed:
        return Action.FULL
    return Action.LOCAL

from collections import defaultdict
from math import isfinite, log, sqrt
from statistics import fmean
from threading import Lock

from .types import (BudgetExhaustedError, ContractFailedError, Forecast,
                    PolicyDeniedError, UnknownUsageError)


def _radius(n, delta):
    if n < 1 or not 0 < delta < 1:
        raise ValueError("invalid calibration size or error allocation")
    return sqrt(log(1 / delta) / (2 * n))


class FamilyEstimator:
    """Calibration data only. Pre-register parent mapping and per-bound delta.

    Each observed episode belongs to one independent family. Repeated observations
    of the same family are averaged before computing any radius.
    """
    def __init__(self, parents=None, min_families=5, delta=.01):
        _radius(min_families, delta)
        self.parents = dict(parents or {})
        self.minimum, self.delta = min_families, delta
        self.samples = defaultdict(lambda: defaultdict(list))
        self.pairs = defaultdict(lambda: defaultdict(list))
        for bucket in self.parents:
            tuple(self._chain(bucket))

    def _chain(self, key):
        seen = set()
        while key is not None:
            if key in seen:
                raise ValueError("cyclic bucket hierarchy")
            yield key
            seen.add(key)
            key = self.parents.get(key)

    def observe(self, key, family, utility, remaining):
        remaining = tuple(remaining)
        if (not 0 <= utility <= 1 or len(remaining) != 3
                or any(not isfinite(x) or x < 0 for x in remaining)):
            raise ValueError("invalid observation")
        for bucket in self._chain(key):
            self.samples[bucket][family].append((utility, utility <= .7, *remaining))

    def observe_pair(self, key, family, reference_q, candidate_q):
        if not all(0 <= x <= 1 for x in (reference_q, candidate_q)):
            raise ValueError("invalid paired score")
        for bucket in self._chain(key):
            self.pairs[bucket][family].append(reference_q - candidate_q)

    def forecast(self, key):
        for bucket in self._chain(key):
            groups = self.samples.get(bucket, {})
            if len(groups) >= self.minimum:
                means = [tuple(map(fmean, zip(*rows))) for rows in groups.values()]
                q, failure, c, l, p = map(fmean, zip(*means))
                radius = _radius(len(groups), self.delta)
                return Forecast(max(0, q-radius), min(1, failure+radius), c, l, p, 0)
        return Forecast(0, 1, 0, 0, 0, 0)

    def tax_upper(self, key):
        for bucket in self._chain(key):
            groups = self.pairs.get(bucket, {})
            if len(groups) >= self.minimum:
                return min(1, fmean(map(fmean, groups.values())) + 2 * _radius(len(groups), self.delta))
        return 1


def certify_policies(losses, delta, target):
    """Returns simultaneous upper bounds and passing IDs for fixed policies."""
    if not losses or not 0 <= target <= 1:
        raise ValueError("invalid certification family")
    bounds = {}
    for policy, values in losses.items():
        if not values or any(not 0 <= x <= 1 for x in values):
            raise ValueError("invalid independent unit losses")
        bounds[policy] = min(1, fmean(values) + _radius(len(values), delta / len(losses)))
    return bounds, tuple(k for k, bound in bounds.items() if bound <= target)


def discounted_reuse(future, prices, scales, gamma):
    """Each row gives probability of usable installation, delta U, delta resources."""
    if not 0 <= gamma <= 1:
        raise ValueError("invalid discount")
    total = 0.
    for j, (probability, utility, resources) in enumerate(future, 1):
        if not 0 <= probability <= 1:
            raise ValueError("invalid installation probability")
        cost = sum(p * g / s for p, g, s in zip(prices, resources, scales, strict=True))
        total += gamma ** j * probability * (utility - cost)
    return total


class JointPrimalDualController:
    """Candidates and forecasts are functions of a local, grader-free state handle.

    supply(state, skills_enabled) owns capability/deadline filtering. forecast
    supplies a calibrated view-specific Q and total continuation resources.
    quote rechecks the shared boundary; dispatch must still reserve atomically.
    """
    def __init__(self, supply, forecast, quote, *, scales, budgets, eta=.1, upper=2.,
                 prices=(0., 0., 0.), risk=.35, quality_floor=.6, dynamic=True,
                 skills=True, reuse=True):
        self.scales, self.budgets, self.prices = map(tuple, (scales, budgets, prices))
        if (not all(len(v) == 3 for v in (self.scales, self.budgets, self.prices))
                or any(not isfinite(s) or s <= 0 for s in self.scales)
                or any(not isfinite(b) or b < 0 for b in self.budgets)
                or not isfinite(eta) or eta <= 0 or not isfinite(upper) or upper < 0
                or any(not 0 <= x <= upper for x in self.prices)
                or not 0 <= risk <= 1 or not 0 <= quality_floor <= 1):
            raise ValueError("invalid controller parameters")
        self.supply, self.forecast, self.quote = supply, forecast, quote
        self.eta, self.upper = eta, upper
        self.risk, self.floor = risk, quality_floor
        self.dynamic, self.skills, self.reuse = dynamic, skills, reuse
        self._closed = {}
        self._lock = Lock()

    def candidates(self, state_ref):
        return tuple(self.supply(state_ref, self.skills))

    def estimate(self, state_ref, action):
        return self.forecast(state_ref, action)

    def _price(self, c, f, delta):
        resources = (f.remaining_tokens, f.remaining_seconds, f.remaining_exposure)
        reuse = f.reuse_value if self.dynamic and self.skills and self.reuse else 0.
        return resources, reuse

    def decide(self, state_ref, choices):
        eligible = []
        offered = set(self.candidates(state_ref))
        for c in choices:
            if c not in offered:
                continue
            try:
                delta = 0. if c.action == Action.LOCAL else self.quote(c)
                f = self.estimate(state_ref, c)
                if c.action != Action.LOCAL and f.failure_upper > self.risk:
                    continue
            except (BudgetExhaustedError, PolicyDeniedError):
                continue
            resources, reuse = self._price(c, f, delta)
            cost = sum(p * g / s for p, g, s in zip(self.prices, resources, self.scales))
            rank = (cost - f.q_lower - reuse, delta, f.remaining_tokens,
                    c.action != Action.LOCAL, c.action.value, c.view_id, c.payload, c.output_cap)
            eligible.append((rank, c, f.q_lower))
        if any(c.action != Action.FALLBACK and q >= self.floor for _, c, q in eligible):
            eligible = [row for row in eligible if row[1].action != Action.FALLBACK]
        return min(eligible, key=lambda row: row[0])[1] if eligible else None

    def close_episode(self, episode_id, tokens, wall_s, delta_p):
        receipt = tokens, wall_s, delta_p
        if tokens is None:
            raise UnknownUsageError("token_bound_required")
        if (type(tokens) is not int or tokens < 0 or not isfinite(wall_s) or wall_s < 0
                or not 0 <= delta_p <= 1):
            raise ContractFailedError("invalid_episode")
        with self._lock:
            if episode_id in self._closed:
                if self._closed[episode_id] != receipt:
                    raise ContractFailedError("conflicting_episode")
                return
            if self.dynamic:
                self.prices = tuple(min(self.upper, max(0., p + self.eta * (g-b)/s))
                                    for p, g, b, s in zip(self.prices, receipt, self.budgets, self.scales))
            self._closed[episode_id] = receipt


class ModularBaselineController(JointPrimalDualController):
    """B8. Independent single-message privacy quote, fixed tuned resource prices.

    Supply standalone_quote to this controller; the execution bridge still uses
    the same cumulative TransactionalBoundary.reserve as B9.
    """
    def __init__(self, supply, forecast, standalone_quote, **options):
        options.update(dynamic=False, reuse=False)
        super().__init__(supply, forecast, standalone_quote, **options)

    def _price(self, c, f, delta):
        return (f.remaining_tokens, f.remaining_seconds, delta), 0.
