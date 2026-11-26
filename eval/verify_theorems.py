"""Reproduce the finite models in the theory section. Python 3.10+ (standard library).

Run: python -B verify_theorems.py
All examples are synthetic. No network, user records, or real model calls.
The exact oracles choose actions using raw costs and tick-level transition kernels.
The Monte Carlo samples installations, tick drift, assertions and repairs separately.
"""

import itertools
import json
import math
import platform
from dataclasses import dataclass, replace
from fractions import Fraction as F
from pathlib import Path

import random
from statistics import fmean, stdev


SEED = 20261127
COORDS = ("cloud_tokens", "wall_seconds", "incremental_policy_disclosure")


def subsets(items):
    items = tuple(items)
    return (frozenset(group) for n in range(len(items) + 1)
            for group in itertools.combinations(items, n))


def closure(facts, rules):
    result = set(facts)
    while True:
        additions = {head for body, head in rules if body <= result} - result
        if not additions:
            return result
        result.update(additions)


def exposure(facts, rules):
    inferred = closure(facts, rules)
    weights = {"x": 1, "y": 2, "secret": 4}
    return F(sum(weight for atom, weight in weights.items() if atom in inferred), 7)


def verify_propositions(check):
    rules = [(frozenset(("x", "z")), "secret")]
    sent, reserved = set(), {"x"}
    check(exposure(sent | {"z"}, rules) <= F(2, 7), "old snapshot accepts z")
    check(exposure(sent | reserved | {"z"}, rules) > F(2, 7),
          "including outstanding permits rejects z")
    sent.update(reserved)  # Unknown transport status is retained.
    reserved.clear()
    check(exposure(sent | {"x"}, rules) == exposure(sent, rules), "timeout retry idempotence")
    check(exposure(sent | {"z"}, rules) > F(2, 7), "unknown send still blocks z")
    gradients = [(2, -1, 1), (-1, 3, 0), (2, -3, 1), (-1, -1, -2)]
    eta = F(1, 3)
    initial = (F(1, 2), F(1, 2), F(1, 2))
    for comparator in itertools.product((F(0), F(1, 2), F(1)), repeat=3):
        price = initial
        regret = F(0)
        bound = sum((x - u) ** 2 for x, u in zip(initial, comparator)) / (2 * eta)
        for gradient in gradients:
            regret += sum((u - x) * z for u, x, z in zip(comparator, price, gradient))
            bound += eta * sum(z * z for z in gradient) / 2
            price = tuple(min(1, max(0, x + eta * z)) for x, z in zip(price, gradient))
        check(regret <= bound, f"projected comparator {comparator}")
    # The empty observed ledger hides a pending h. Both first credentials pass,
    # but y+z becomes illegal with the retained reservation in the actual gate.
    pending_rules = rules + [(frozenset(("h", "z")), "secret")]
    cap = F(2, 7)
    check(all(exposure({"h", v}, pending_rules) <= cap for v in ("x", "y")), "both first views admissible")
    check(exposure({"y", "z"}, pending_rules) <= cap < exposure({"h", "y", "z"}, pending_rules),
          "observed history alone gives a false open continuation")
    for initial_price, gradient, box in (((0, 0, 0), (2, -1, 4), 0), ((1, 0, 1), (0, 0, 0), 2)):
        for u in itertools.product({F(0), F(box)}, repeat=3):
            updated = [min(box, max(0, v + eta*z)) for v, z in zip(initial_price, gradient)]
            loss = sum((c - v)*z for c, v, z in zip(u, initial_price, gradient))
            check(loss == 0 and updated == list(initial_price), "degenerate dual box or gradient")
    return {"comparators": 27, "shared_snapshot_violation": "5/7 > 2/7",
            "with_pending_h_final_yz": "6/7 > 2/7"}


def verify_closure_separation(check):
    atoms = ("x", "y", "z0", "z1")
    paths = (frozenset(("z0",)), frozenset(("z1",)))
    possible_rules = [(frozenset((view, z)), "secret")
                      for view in ("x", "y") for z in ("z0", "z1")]
    checked_graphs = 0
    for rule_set in subsets(possible_rules):
        for cap in (F(v, 7) for v in range(2, 8)):
            bad = [group for group in subsets(atoms) if exposure(group, rule_set) > cap]
            edges = [group for group in bad if not any(other < group for other in bad)]
            # Execute the two stages through the gate; separately evaluate the cut.
            def admitted(first, suffix):
                history = {first}
                for atom in sorted(suffix):
                    history.add(atom)
                    if exposure(history, rule_set) > cap:
                        return False
                return True

            blocked_x = all(not admitted("x", path) for path in paths)
            open_y = any(admitted("y", path) for path in paths)
            cut_x = all(any(edge <= (path | {"x"}) for edge in edges) for path in paths)
            clear_y = any(not any(edge <= (path | {"y"}) for edge in edges) for path in paths)
            check((blocked_x and open_y) == (cut_x and clear_y), "minimal-edge iff")
            checked_graphs += 1
    family = possible_rules[:2]
    low, high = F(1, 5), F(9, 10)
    for cap, expected_mod in ((F(2, 7), low), (F(5, 7) - F(1, 700), low), (F(5, 7), high)):
        def terminal(view):
            if view == "stop":
                return low
            return high if any(exposure({view} | path, family) <= cap for path in paths) else low

        first_quotes = {v: (1, -exposure({v}, family), -tokens) for v, tokens in (("x", 1), ("y", 2))}
        first_quotes["stop"] = (0, F(0), 0)
        local = max(first_quotes, key=first_quotes.get)
        optimum = max(terminal(view) for view in first_quotes)
        check(terminal(local) == expected_mod and optimum == high, "budget threshold endpoint")
        check(optimum - terminal(local) == high - expected_mod, "attained utility gap")
    check(all(exposure({"x"} | path, ()) <= F(2, 7) for path in paths), "empty K on witness")
    check(exposure({"x", "y"}, ()) > F(2, 7), "an additive choke can survive empty K")
    multihop = [(frozenset(("x", "z0")), "middle"), (frozenset(("middle",)), "secret")]
    check(exposure({"x", "z0"}, multihop) == F(5, 7), "least fixed point uses two rules")
    worlds = (possible_rules[:2], possible_rules[2:])
    first_scores = [tuple(exposure({view}, world) for view in ("x", "y")) for world in worlds]
    check(first_scores[0] == first_scores[1], "indistinguishable first quotes")
    loss_pairs = []
    for t in (F(j, 20) for j in range(21)):
        losses = []
        for world in worlds:
            reward = sum(prob * (high if exposure({v, "z0"}, world) <= F(2, 7) else low)
                         for v, prob in (("x", t), ("y", 1 - t)))
            losses.append(high - reward)
        check(max(losses) >= (high - low) / 2, "arbitrary local rule minimax bound")
        loss_pairs.append(max(losses))
    check(min(loss_pairs) == (high - low) / 2, "fair coin attains minimax bound")
    return {"cut_iff_graph_budget_pairs": checked_graphs, "tight_utility_gap": str(high - low),
            "tight_randomized_minimax": str(min(loss_pairs)), "choke_cap_interval": "[2/7,5/7)"}


def stream_costs(n, alpha, d, premium):
    """Expected raw costs (myopic, optimal) with a known first match, no future hints."""
    # Direct costs 1+d, an installed hit costs 1; unrelated tasks cost both zero.
    direct = 1 + d
    myopic = cold = warm = F(0)
    for slot in range(n, 0, -1):
        arrival = F(1) if slot == 1 else alpha
        cost_if_wait = direct + cold
        cost_if_build = direct + premium + warm
        myopic += arrival * direct
        cold = (1 - arrival) * cold + arrival * min(cost_if_wait, cost_if_build)
        warm += arrival
    return myopic, cold


def verify_stream_separation(check):
    for n, alpha, d, premium in itertools.product(range(1, 8), (F(0), F(1, 3), F(1)),
                                                  (F(0), F(2)), (F(1), F(2))):
        net = (n - 1) * alpha * d - premium
        full_paths = F(0)
        for bits in itertools.product((0, 1), repeat=n-1):
            hits = sum(bits)
            prob = alpha ** hits * (1 - alpha) ** (n - 1 - hits)
            full_paths += prob * (hits * d - premium)
        check(full_paths == net, "expected gain from compiling now")
        myopic, optimum = stream_costs(n, alpha, d, premium)
        check(myopic - optimum == max(0, net), "cost-only Bellman policy vs positive part")
    alpha, d, premium = F(1, 3), F(3), F(4)
    for n in (10, 100, 1000):
        myopic, optimum = stream_costs(n, alpha, d, premium)
        check((myopic - optimum) / n >= alpha * d / 2, "linear lower bound per slot")
    # Signal reveals the next arrival. Enumerate the two paths and both first actions.
    direct, hot, build_cost = 3, 1, 4
    paths = [(direct + hit * direct, build_cost + hit * hot) for hit in (0, 1)]
    blind = min(sum(costs, F(0)) / 2 for costs in zip(*paths))
    informed = sum(min(costs) for costs in paths) / F(2)
    check(blind - informed == F(1, 2), "IID arrivals with a predictive first binding")
    check(stream_costs(2, F(1, 2), F(2), F(1)) == (blind, blind), "blind boundary is exactly neutral")
    myopic, optimum = stream_costs(100, alpha, d, premium)
    return {"first_known_family_arrival": True, "predictive_signal_saving": str(blind - informed),
            "example": {"N": 100, "alpha": str(alpha), "d": int(d), "premium": int(premium),
                        "exact_cumulative_gap": int(myopic - optimum), "mean_gap": float((myopic - optimum)/100)}}


@dataclass(frozen=True)
class Model:
    g0: tuple = (F(8), F(2), F(1, 8))
    g1: tuple = (F(0), F(1, 2), F(0))
    cold: tuple = (F(20), F(5), F(3, 8))
    repair: tuple = (F(3), F(1), F(1, 16))
    replan: tuple = (F(10), F(3), F(1, 8))
    install: F = F(4, 5)
    patch: F = F(1, 2)
    q: F = F(1, 8)
    drift: F = F(1, 16)
    ticks: int = 2


def geom(x, n):
    return n if x == 1 else (1 - x ** n) / (1 - x)


def terms(model, j):
    a = model.cold[j] - model.g0[j]
    d = model.g0[j] - model.g1[j]
    e = model.repair[j] + (1 - model.patch) * model.replan[j]
    return a, d, e


def failure(model):
    return 1 - (1 - model.q) * (1 - model.drift) ** model.ticks


def net_at_failure(model, j, f, gamma, n):
    a, d, e = terms(model, j)
    x = gamma * (1 - (1 - model.patch) * f)
    if n is None and not 0 < gamma < 1:
        raise ValueError("infinite discounted costs require 0 < gamma < 1")
    discounted_count = 1 / (1 - x) if n is None else geom(x, n)
    return -a + model.install * gamma * (d - f * e) * discounted_count


def step_costs(model, j, gamma, direct, active):
    """Cost backups from raw charges. The next opportunity starts with one of two states."""
    clean = model.g1[j] + gamma * active
    patch_ok = model.g1[j] + model.repair[j] + gamma * active
    patch_failed = model.g1[j] + model.repair[j] + model.replan[j] + gamma * direct
    stale = model.patch * patch_ok + (1 - model.patch) * patch_failed
    healthy = (1 - model.q) * clean + model.q * stale
    for _tick in range(model.ticks):
        healthy = (1 - model.drift) * healthy + model.drift * stale
    return model.g0[j] + gamma * direct, healthy


def exact_costs(model, j, n, gamma):
    """Opportunity costs from raw branches, without the saving or failure formula."""
    if n is None:
        if not 0 < gamma < 1:
            raise ValueError("an infinite cost comparison requires discounting")
        direct = model.g0[j] / (1 - gamma)
        # Solve the affine healthy-state Bellman equation after fixing the retired cost.
        intercept = step_costs(model, j, gamma, direct, F(0))[1]
        slope = step_costs(model, j, gamma, direct, F(1))[1] - intercept
        return direct, intercept / (1 - slope)
    direct = active = F(0)
    for _ in range(n):
        direct, active = step_costs(model, j, gamma, direct, active)
    return direct, active


def exact_net(model, j, n, gamma):
    direct, active = exact_costs(model, j, n, gamma)
    no_build = model.g0[j] + gamma * direct
    build = model.cold[j] + gamma * (model.install * active + (1 - model.install) * direct)
    return no_build - build


def patch_gain_from_costs(model, j, k, gamma, *, replan_now):
    """At a failed probe, compare the two complete paths including continuation."""
    direct, active = exact_costs(model, j, k, gamma)
    full_cost = replan_now + gamma * direct
    patch_cost = model.repair[j] + (model.patch * gamma * active
                 + (1 - model.patch) * (model.replan[j] + gamma * direct))
    return full_cost - patch_cost


def min_reuses(a, w, x):
    """Strict threshold for rational inputs, including exact ties near x=1."""
    if a < 0:
        raise ValueError("threshold table assumes a nonnegative cold premium")
    if w <= 0:
        return math.inf
    if x == 1:
        return int(a // w) + 1
    if x == 0:
        return 1 if w > a else math.inf
    if a * (1 - x) >= w:
        return math.inf
    # Bracket then bisect with exact comparisons. Floating logs round near-one
    # inputs to one, and can move an exact equality to the profitable side.
    low, high = 0, 1
    while w * geom(x, high) <= a:
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if w * geom(x, middle) > a:
            high = middle
        else:
            low = middle
    return high


def verify_phase(check):
    base = Model()
    for p, pi, (q, delta), gamma, n in itertools.product(
            (F(0), base.install, F(1)), (F(0), base.patch, F(1)),
            ((F(0), F(0)), (base.q, base.drift), (F(0), F(1)), (F(1), F(0)), (F(1), F(1))),
            (F(1), F(9, 10)), (0, 1, 2, 6)):
        model = replace(base, install=p, patch=pi, q=q, drift=delta)
        for j in range(3):
            check(net_at_failure(model, j, failure(model), gamma, n) == exact_net(model, j, n, gamma),
                  "closed form differs from tick-level backward induction")
    # Validate strict, equal, just-above and asymptotic thresholds, including x=0,1.
    for x, w in itertools.product((F(0), F(1, 2), F(9, 10), F(1)), (F(0), F(1), F(2))):
        for tie_n in range(1, 6):
            tie_value = sum(w * x**i for i in range(tie_n))
            near_tie = (F(0), tie_value - F(1, 500), tie_value, tie_value + F(1, 500))
            for a in {value for value in near_tie if value >= 0}:
                threshold = min_reuses(a, w, x)
                # With permanent health, discounting by x realizes the same test
                # using two complete resource bills. For x=0 force retirement
                # on the first use, without discounting.
                discount = x if x > 0 else F(1)
                direct = 1 + w/discount
                test_model = replace(base, install=F(1), patch=F(0), drift=F(0),
                                     q=F(int(x == 0)), g0=(8, direct, F(1, 8)),
                                     g1=(0, F(1), 0), cold=(20, direct+a, F(3, 8)),
                                     repair=(0, 0, 0), replan=(0, 0, 0))
                if math.isinf(threshold):
                    if 0 < x < 1:
                        upper_gain = exact_net(test_model, 1, None, discount)
                    else:
                        # The only infinite cases on this grid have one attempt
                        # (x=0) or zero per-step saving (x=1).
                        upper_gain = exact_net(test_model, 1, 1, discount)
                        check(x == 0 or test_model.g1[1] == test_model.g0[1], "flat or retired")
                    check(upper_gain <= 0, "strict gain unattainable by complete resource bills")
                else:
                    before = exact_net(test_model, 1, threshold-1, discount)
                    after = exact_net(test_model, 1, threshold, discount)
                    check(before <= 0 < after, "first strict crossing by complete resource bills")
    gamma = F(9, 10)
    thresholds, curves, finite_brackets = {}, {}, {}
    for j, coord in enumerate(COORDS):
        a, d, e = terms(base, j)
        f = failure(base)
        x = 1 - (1 - base.patch) * f
        thresholds[coord] = min_reuses(a, base.install * (d - f * e), x)
        n_star = thresholds[coord]
        check(exact_net(base, j, n_star - 1, F(1)) <= 0 < exact_net(base, j, n_star, F(1)),
              "threshold against independently calculated path costs")
        f_star = ((base.install * gamma * d - a * (1 - gamma))
                  / (gamma * (base.install * e + a * (1 - base.patch))))
        curves[coord] = str(f_star)
        for offset, expected_sign in ((-F(1, 1000), 1), (F(0), 0), (F(1, 1000), -1)):
            val = exact_net(replace(base, q=f_star + offset, drift=F(0)), j, None, gamma)
            check((val > 0) - (val < 0) == expected_sign, "infinite boundary vs raw-cost Bellman solution")
        # Hold primitive charges fixed. Locate a finite crossing using only path costs.
        finite_n = 12
        low, high = F(0), F(1)
        def gain_at(q):
            return exact_net(replace(base, q=q, drift=F(0)), j, finite_n, gamma)
        check(gain_at(low) > 0 > gain_at(high), "cost oracle brackets a physical crossing")
        for _ in range(18):
            mid = (low + high) / 2
            if gain_at(mid) > 0:
                low = mid
            else:
                high = mid
        finite_brackets[coord] = [str(low), str(high)]
        check(net_at_failure(base, j, low, gamma, finite_n) > 0 >=
              net_at_failure(base, j, high, gamma, finite_n), "finite polynomial matches cost bracket")
        # Physical inverse: perturb calendar drift and execute the tick-level process.
        q = f_star / 3
        critical_drift = 1 - ((1 - float(f_star)) / (1 - float(q))) ** (1 / base.ticks)
        for shift, is_positive in ((-1e-5, True), (1e-5, False)):
            scenario = replace(base, q=q, drift=F(str(critical_drift + shift)))
            check((exact_net(scenario, j, None, gamma) > 0) == is_positive, "physical drift crossing by costs")
    # Compare exact branch costs after an observed failure.
    cases = []
    for model in (base, replace(base, patch=F(0)), replace(base, patch=F(1)),
                  replace(base, repair=(F(8), F(1), F(1, 16)))):
        for k, j in itertools.product((0, 1, 8), range(3)):
            replan_now = model.replan[j]
            observed = patch_gain_from_costs(model, j, k, gamma, replan_now=replan_now)
            _, d, e = terms(model, j)
            f = failure(model)
            predicted = (replan_now - model.repair[j] - (1 - model.patch) * model.replan[j]
                         + model.patch * gamma * (d - f * e)
                         * geom(gamma * (1 - (1 - model.patch) * f), k))
            check(observed == predicted, "conditional patch/full continuation advantage")
            if model.repair[0] == 8 and j == 0:
                cases.append(predicted)
    check(cases[0] < 0 < cases[-1], "future retention reverses an unfavorable immediate repair")
    check(exact_net(replace(base, install=F(0)), 0, 4, gamma) < 0, "install failure loss")
    healthy = replace(base, q=F(0), drift=F(0))
    check(exact_net(healthy, 0, 1, F(1)) < 0 < base.install * (base.g0[0] - base.g1[0]),
          "future credit positive while current-inclusive score is negative")
    mixed = [exact_net(base, j, 4, F(1)) for j in range(3)]
    check(sum(mixed) > 0 and min(mixed) < 0, "positive priced value does not imply Pareto gain")
    for n, theta in itertools.product((0, 1, 4, 16), ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3))):
        # Price continuation from the independent DP, then book the cold cost once.
        future_value = F(0)
        cold_premium = F(0)
        for j, price in enumerate(theta):
            direct_cost, active_cost = exact_costs(base, j, n, gamma)
            future_value += price * base.install * gamma * (direct_cost - active_cost)
            cold_premium += price * (base.cold[j] - base.g0[j])
        priced_gain = sum(price * net_at_failure(base, j, failure(base), gamma, n)
                          for j, price in enumerate(theta))
        current_cost_score = future_value - sum(price * cost for price, cost in zip(theta, base.cold))
        direct_score = -sum(price * cost for price, cost in zip(theta, base.g0))
        reuse_net = future_value - cold_premium
        check(current_cost_score - direct_score == reuse_net == priced_gain, "score difference in both conventions")
    # Geometric arrival gaps: sum the probability generating function independently.
    alpha, delta = F(1, 3), F(1, 5)
    expected_survival = alpha * (1 - delta) / (alpha + (1 - alpha) * delta)
    partial = sum(alpha * (1 - alpha)**(m - 1) * (1 - delta)**m for m in range(1, 65))
    check(0 <= expected_survival - partial < F(1, 10**15), "calendar drift accumulated over random gaps")
    # The same latent interarrival length in both gaps violates independent renewal.
    intervals = (1, 2)
    shared = sum(F(1, 2) ** (2*m) for m in intervals) / 2
    fresh = sum(F(1, 2) ** (u+v) for u, v in itertools.product(intervals, repeat=2)) / 4
    check((shared, fresh) == (F(5, 32), F(9, 64)), "dependent versus fresh calendar gaps")

    # Debit each path's actual ledger. Same full payload, different incremental charge.
    def mass(facts):
        return F(len(facts), 2)
    full, patch = {"a", "b"}, {"a"}
    now = mass(full)
    after = mass(patch | full) - mass(patch)
    hit = replace(base, q=F(1), repair=(0, 0, mass(patch)), replan=(0, 0, after))
    real_gain = now - (hit.patch * mass(patch) + (1 - hit.patch) * mass(patch | full))
    check(patch_gain_from_costs(hit, 2, 0, gamma, replan_now=now) == real_gain == F(1, 4),
          "overlapping facts change the immediate replan debit")
    check(hit.patch * after - mass(patch) == -real_gain, "single-cost shortcut reverses the sign")

    # Negative cold excess: discounted latency improves only ABOVE the failure cutoff.
    reverse = replace(base, install=F(1), patch=F(0), drift=F(0), g1=(0, F(7, 2), 0),
                      cold=(20, F(1), F(3, 8)), repair=(0, 0, 0), replan=(0, 0, 0))
    reverse_values = [exact_net(replace(reverse, q=q), 1, None, F(1, 2)) for q in (F(0), F(1, 2), F(1))]
    check(reverse_values == [-F(1, 2), F(0), F(1, 4)], "increasing physical crossing when J<0")
    check(exact_net(reverse, 1, 0, F(1, 2)) > 0, "negative premium is profitable at zero reuses")
    for cold_delta in (-1, 0, 1):
        constant = replace(base, install=F(0), patch=F(1), cold=(base.g0[0]+cold_delta, 5, F(3, 8)))
        for q in (F(0), F(1)):
            check(exact_net(replace(constant, q=q), 0, None, gamma) == -cold_delta, "J=0 endpoint")
    check(min_reuses(F(2), F(1), 1-F(1, 10**30)) == 3, "near-one rational threshold without floating logs")
    return {"base_failure": str(failure(base)), "unweighted_minimum_future_reuses": thresholds,
            "discounted_failure_cutoffs_gamma_0.9": curves, "finite_root_brackets_n12_gamma_0.9": finite_brackets,
            "ledger_patch_saving": str(real_gain), "legacy_single_R_saving": str(-real_gain),
            "negative_premium_D_at_f_0_half_1": list(map(str, reverse_values)),
            "four_reuse_savings_example": [float(v) for v in mixed]}


def binomial(rng, n, p):
    """Split on a uniform order statistic to sample large binomials exactly in law."""
    hits = 0
    while n > 32 and 0 < p < 1:
        k = n // 2
        pivot = rng.betavariate(k, n - k + 1)
        if pivot > p:
            n, p = k - 1, p / pivot
        else:
            hits += k
            n, p = n - k, (p - pivot) / (1 - pivot)
    if p >= 1:
        return hits + n
    if p > 0:
        hits += sum(rng.random() < p for _ in range(n))
    return hits


def monte_carlo(model, n, gamma, rng, count):
    active = installed = binomial(rng, count, float(model.install))
    gains = [float(base - cold) for base, cold in zip(model.g0, model.cold)]
    savings = [float(x - y) for x, y in zip(model.g0, model.g1)]
    repairs, replans = list(map(float, model.repair)), list(map(float, model.replan))
    discount = 1.0
    for _ in range(n):
        discount *= gamma
        healthy = active
        for _tick in range(model.ticks):
            healthy -= binomial(rng, healthy, float(model.drift))
        failures = active - healthy + binomial(rng, healthy, float(model.q))
        retired = failures - binomial(rng, failures, float(model.patch))
        for j in range(len(gains)):
            gains[j] += discount * (active * savings[j] - failures * repairs[j] - retired * replans[j]) / count
        active -= retired
    return gains, installed, active


def verify_simulation(check):
    rng = random.Random(SEED)
    base = Model()
    scenarios = (
        ("positive", base, 16, F(1)),
        ("negative", replace(base, q=F(4, 5), drift=F(1, 4)), 16, F(1)),
        ("discounted", base, 32, F(9, 10)),
    )
    # Set q to the exact discounted token boundary. A long finite run also bounds the tail.
    gamma = F(9, 10)
    a, d, e = terms(base, 0)
    token_root = (base.install * gamma * d - a * (1 - gamma)) / (
        gamma * (base.install * e + a * (1 - base.patch)))
    scenarios += (("discounted_token_boundary", replace(base, q=token_root, drift=F(0)), 180, gamma),)
    count, batches = 10_000_000, 32
    results = {}
    for name, model, n, gamma in scenarios:
        target = [float(net_at_failure(model, j, failure(model), gamma, n)) for j in range(3)]
        sample = []
        installed = survivors = 0
        for _ in range(batches):
            gain, built, active = monte_carlo(model, n, float(gamma), rng, count)
            sample.append(gain)
            installed += built
            survivors += active
        estimate = list(map(fmean, zip(*sample)))
        abs_err = [abs(t - e) for t, e in zip(target, estimate)]
        for coord, err in zip(COORDS, abs_err):
            check(err < 1e-2, f"{name}/{coord} absolute error {err}")
        if name == "positive":
            check(all(x > 0 for x in estimate), "observed positive phase in all original units")
        if name == "negative":
            check(all(x < 0 for x in estimate), "observed negative phase in all original units")
        if name == "discounted_token_boundary":
            infinite = [float(net_at_failure(model, j, failure(model), gamma, None))
                                 for j in range(3)]
            check(abs(infinite[0]) < 1e-14 and max(abs(i-t) for i,t in zip(infinite,target)) < 1e-12,
                  "finite Monte Carlo horizon has negligible tail at the infinite boundary")
        results[name] = {"n_future": n, "gamma": float(gamma), "q": float(model.q),
                         "drift": float(model.drift), "installed": installed, "survivors": survivors,
                         "analytical_total_gain": target, "mc_total_gain": estimate,
                         "absolute_errors": abs_err,
                         "mc_standard_errors": [stdev(col) / math.sqrt(batches) for col in zip(*sample)]}
    worst = max(max(case["absolute_errors"]) for case in results.values())
    return {"seed": SEED, "iid_streams_per_case": count * batches, "independent_batches": batches,
            "coordinate_order": COORDS, "tolerance_absolute_total": 0.01,
            "max_absolute_error": worst, "cases": results}


def binomial_masses(n, p):
    return [math.comb(n, k) * p ** k * (1-p) ** (n-k) for k in range(n+1)]


def verify_audit(check):
    # Includes ledger-only and audit-only atoms and independently frozen direct-miss tags.
    official = (F(0), F(1, 4), F(1, 4), F(1, 2))
    ledger = (F(1, 2), F(1, 4), F(1, 4), F(0))
    direct = (False, True, False, False)
    tv = sum(abs(u - w) for u, w in zip(official, ledger)) / 2
    for y, x in itertools.product(itertools.product((0, 1), repeat=4), repeat=2):
        e = sum(u * flag for u, flag in zip(official, y))
        p = sum(w * flag for w, flag in zip(ledger, x))
        fn = sum(u for u, a, b, t in zip(official, y, x, direct) if a and not b and t)
        sem = sum(u for u, a, b, t in zip(official, y, x, direct) if a and not b and not t)
        fp = sum(u for u, a, b in zip(official, y, x) if b and not a)
        weight = sum((u-w)*b for u, w, b in zip(official, ledger, x))
        check(e - p == fn + sem - fp + weight and abs(weight) <= tv, "exact audit decomposition")
    # Two-point signed gap: E=1,P=0 with probability theta, else E=0,P=1.
    max_failure = 0
    for n, theta, delta in itertools.product((8, 32, 64), (F(1, 5), F(1, 2), F(4, 5)), (0.05, 0.1)):
        radius = math.sqrt(2 * math.log(1/delta) / n)
        misses = sum(prob for k, prob in enumerate(binomial_masses(n, theta))
                     if float(theta) > float(1-theta) + (2*k/n-1) + radius)
        check(float(misses) <= delta, "exact binomial tail exceeds claimed failure probability")
        max_failure = max(max_failure, float(misses))
    n, delta = 40, 0.05
    short_radius = math.sqrt(math.log(1/delta) / (2*n))
    wrong = sum(prob for k, prob in enumerate(binomial_masses(n, F(1, 2))) if 1-2*k/n > short_radius)
    check(float(wrong) > delta, "using width one for signed gap must fail")
    rate_tails = {}
    for n in (16, 64, 256):
        # Exact threshold because sqrt(n) is integral in these witnesses.
        tail = sum(prob for k, prob in enumerate(binomial_masses(n, F(1, 2)))
                   if 2*(n-2*k) >= math.isqrt(n))
        check(tail >= F(3, 32), "one-sided 1/sqrt(n) scale lower witness")
    radius100 = math.sqrt(2 * math.log(1/0.05) / 100)
    # Equal-weight duplicates and IID clusters with all weight on the first
    # both produce a single fair audit score. Enumerate its two actual outcomes.
    duplicate_miss = sum(F(1, 2) for score in (0, 1) if score + radius100 < 0.5)
    weighted_radius = math.sqrt(2 * math.log(1/0.05))  # sum(omega^2)=1
    check(duplicate_miss == F(1, 2) and float(duplicate_miss) > 0.05, "effective sample is one")
    check(all(score + weighted_radius >= 0.5 for score in (0, 1)), "fixed unequal-weight correction")
    clusters = [(1,), (0,)*9]
    check(sum(F(sum(c), len(c)) for c in clusters)/2 == F(1, 2)
          and F(sum(map(sum, clusters)), sum(map(len, clusters))) == F(1, 10), "cluster vs task estimand")
    direct_lexicon = {"red"}
    audit_lexicon = {"red", "the first rainbow colour"}
    transcripts = ({"blue"}, {"the first rainbow colour"})
    proxy_and_audit = [(int(bool(words & direct_lexicon)), int(bool(words & audit_lexicon)))
                       for words in transcripts]
    check(proxy_and_audit == [(0, 0), (0, 1)],
          "a proxy-only rule cannot distinguish hidden semantic publication")
    hidden_text = {"other paraphrase"}
    audited = int(bool(hidden_text & audit_lexicon))
    truth = int(bool(hidden_text & {"other paraphrase"}))
    radius = math.sqrt(2 * math.log(1/0.05) / 32)
    check(audited == 0 and audited + radius < truth == 1,
          "an unobserved channel can defeat certification of a different latent truth")
    return {"decomposition_patterns": 256, "max_exact_noncoverage_on_grid": max_failure,
            "wrong_width_one_noncoverage_at_95pct": float(wrong), "rate_lower_witness_tails": rate_tails,
            "same_proxy_distinct_official_means": [0, 1], "pseudoreplication_noncoverage": float(duplicate_miss)}


def bayes(weights, channel):
    """Finite synthetic experiments: normalized posterior maxima and global Bayes accuracy."""
    total = sum(weights)
    observations = set().union(*(row.keys() for row in channel))
    conditional = {}
    global_accuracy = F(0)
    for obs in observations:
        joint = [F(w, total) * row.get(obs, 0) for w, row in zip(weights, channel)]
        mass = sum(joint)
        if mass:
            conditional[obs] = max(joint) / mass, sum(x > 0 for x in joint)
            global_accuracy += max(joint)
    return conditional, global_accuracy


def verify_linkage(check):
    for k, kappa in itertools.product(range(1, 8), (1, 2, 5)):
        weights = ([kappa] + [1] * (k-1)) * 3
        channel = [{j//k: F(1)} for j in range(3*k)]
        cond, acc = bayes(weights, channel)
        bound = F(kappa, k-1+kappa)
        check(all(m == k and maximum == bound for maximum, m in cond.values()) and acc == bound,
              "skewed partition attains posterior bound everywhere")
    k = 4
    grid = list(itertools.product(range(k), repeat=2))
    channels = {"background_only": [0]*len(grid), "last_turn_only": [c for _, c in grid],
                "full_history": grid}
    accuracies = {}
    for label, values in channels.items():
        cond, acc = bayes([1]*len(grid), [{v: F(1)} for v in values])
        expected = {"background_only": F(1, k*k), "last_turn_only": F(1, k), "full_history": F(1)}[label]
        check(acc == expected and all(v == F(1, size) for v, size in cond.values()), label)
        accuracies[label] = float(acc)
    check(all(sum(i == row for i, _ in grid) == k for row in range(k)), "first turn alone is k-anonymous")
    check(all(len({identity for identity in grid if identity[0] == r and identity[1] == c}) < k
              for r, c in grid), "prefix-count gate rejects every second unmodified view")
    eps = F(1, 100)
    noisy = [{j: 1-eps if i == j else eps/(k-1) for j in range(k)} for i in range(k)]
    cond, acc = bayes([1]*k, noisy)
    rho = (1-eps)/(eps/(k-1))
    check(all(size == k and maximum == 1-eps for maximum, size in cond.values()), "full support, biased channel")
    check(acc == 1-eps and acc == rho/(k-1+rho), "likelihood ratio bound is attained")
    # Same released aliases in both experiments. B' supplies the independent shift;
    # B marginalizes it, so its transcript likelihood is 1/k for every identity.
    shifts = list(range(k))
    public_channel = []
    for identity in range(k):
        aliases = [(identity + shift) % k for shift in shifts]
        public_channel.append({alias: F(aliases.count(alias), k) for alias in set(aliases)})
    check(bayes([1]*k, public_channel)[1] == F(1, k), "background without the shift")
    for shift in shifts:
        conditioned = [{(identity + shift) % k: F(1)} for identity in range(k)]
        check(bayes([1]*k, conditioned)[1] == 1, "same transcript conditional on the shift")
    # Test the exact posterior-mass criterion with unequal priors AND likelihoods.
    for weights, rate in itertools.product(((1, 1, 1, 1), (1, 2, 2, 3)), (F(1, 4), F(2, 5), F(3, 4))):
        skew_channel = [{j: rate if i == j else (1-rate)/(k-1) for j in range(k)} for i in range(k)]
        # Materialize a finite sample space using equiprobable tickets; tally
        # successful guesses directly, with no call to the posterior solver.
        scale = math.lcm(*(p.denominator for row in skew_channel for p in row.values()))
        tickets = [(i, obs) for i, row in enumerate(skew_channel) for obs, p in row.items()
                   for _ in range(int(scale * weights[i] * p))]
        kappa = F(max(weights), min(weights))
        for obs in range(k):
            compatible = [i for i, value in tickets if value == obs]
            best = max(F(compatible.count(guess), len(compatible)) for guess in range(k))
            mass = [w * row[obs] for w, row in zip(weights, skew_channel)]
            rest = sum(mass) - max(mass)
            check((best <= kappa/(k-1+kappa)) == (rest >= (k-1)*max(mass)/kappa),
                  "posterior mass criterion vs direct ticket decisions")
    # Finite calibration channels only: for each joint table enumerate all
    # decisions on the two possible outputs as a separate accuracy oracle.
    channel_count = 0
    for rates in itertools.product((F(0), F(1, 4), F(1, 2), F(3, 4), F(1)), repeat=3):
        channel = [{0: 1-r, 1: r} for r in rates]
        for prior in ((1, 1, 1), (1, 2, 3)):
            post, optimum = bayes(prior, channel)
            guesses = itertools.product(range(3), repeat=2)
            best = max(sum(F(prior[i], sum(prior)) * channel[i][o] for o, i in enumerate(guess))
                       for guess in guesses)
            check(best == optimum, "all two-output decision maps vs Bayes accuracy")
            for o, (top1, size) in post.items():
                support = [i for i in range(3) if channel[i][o] > 0]
                priors = [prior[i] for i in support]
                likelihoods = [channel[i][o] for i in support]
                ratio = F(max(priors), min(priors)) * max(likelihoods) / min(likelihoods)
                check(top1 <= ratio/(size-1+ratio), "positive-support prior and likelihood bound")
            channel_count += 1
    # R is marginally independent of the identity, but is shared with the release.
    pairs = list(itertools.product(range(2), repeat=2))
    hidden = [{o: F(1, 2) for o in range(2)} for _ in range(2)]
    revealed = [{(i ^ r, r): F(1, 2) for r in range(2)} for i in range(2)]
    check(bayes([1, 1], hidden)[1] == F(1, 2) and bayes([1, 1], revealed)[1] == 1
          and all(((i ^ r) ^ r) == i for i, r in pairs), "shared coin must belong to full observation")
    limit_values = [bayes([t, 1, 1], [{"class": F(1)}]*3)[1] for t in (1, 10, 10**6)]
    check(limit_values[0] == F(1, 3) and limit_values[0] < limit_values[1] < limit_values[2] < 1
          and 1-limit_values[-1] < F(1, 10**5), "unbounded skew: supremum, not finite equality")
    gated = [{(r, "denied"): F(1)} for r, _ in grid]
    check(bayes([1]*len(grid), gated)[1] == F(1, k), "identity-independent denial preserves row posterior")
    return {"grid_side": k, "three_condition_accuracy": accuracies, "equal_support_skew_accuracy": float(acc),
            "uniform_cardinality_only_claim": 1/k, "binary_channels_and_priors": channel_count,
            "shared_coin_accuracy_hidden_visible": [0.5, 1], "largest_tested_prior_ratio": 10**6}


def main():
    suites = (("P1_P2", verify_propositions), ("T1a", verify_closure_separation),
              ("T1b", verify_stream_separation), ("T2_exact", verify_phase),
              ("T2_monte_carlo", verify_simulation), ("T3a", verify_audit), ("T3b", verify_linkage))
    report = {"status": "PASS", "scope": "synthetic mathematical verification", "python": platform.python_version(),
              "suites": {}, "passed_assertions": 0}
    for name, suite in suites:
        passed = 0

        def check(condition, label):
            nonlocal passed
            if not condition:
                raise AssertionError(f"{name}: {label}")
            passed += 1

        metrics = suite(check)
        report["suites"][name] = {"assertions": passed, "status": "PASS", "metrics": metrics}
        report["passed_assertions"] += passed
        print(f"{name}: {passed} assertions PASS")
    output = Path(__file__).resolve().with_name("verification_summary.json")
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)
    error = report["suites"]["T2_monte_carlo"]["metrics"]["max_absolute_error"]
    print(f"PASS {report['passed_assertions']} assertions; maximum absolute total-cost MC error {error:.8f}")


if __name__ == "__main__":
    main()
