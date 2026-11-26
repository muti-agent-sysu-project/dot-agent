"""Local exact-match diagnostic over ordered cloud-bound payloads."""
from collections.abc import Mapping


class PrivacyLedger:
    def __init__(self, units: Mapping[str, str]):
        if any(not isinstance(v, str) or not v for v in units.values()):
            raise ValueError("each unit needs a nonempty literal")
        self._units = dict(units)
        self._tail = ""
        self._keep = max((len(v) - 1 for v in units.values()), default=0)
        self.exposed: set[str] = set()
        self.steps = 0

    def observe(self, cloud_text: str) -> frozenset[str]:
        """Return newly exposed unit IDs, including splits across adjacent calls."""
        window = self._tail + cloud_text
        hits = {key for key, value in self._units.items() if value in window}
        new = hits - self.exposed
        self.exposed.update(hits)
        self._tail = window[-self._keep:] if self._keep else ""
        self.steps += 1
        return frozenset(new)

    @property
    def exposure(self) -> float | None:
        return len(self.exposed) / len(self._units) if self._units else None

# The literal diagnostic above belongs to the offline grader. The online boundary
# below uses a separate trusted schema inspector and never reads grader labels.
import hashlib
import json
import math
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from .types import (Action, BudgetExhaustedError, ContractFailedError, Permit,
                    PolicyDeniedError, Scope, StaleVersionError, UnknownUsageError, Usage)


class TrajectoryPrivacyLedger:
    """Fixed Horn policy; O lives in the durable per-recipient attempt journal."""
    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError("frozen ledger policy")
        object.__setattr__(self, key, value)

    def __init__(self, weights, *, background=(), rules=(), forbidden=(), raw_forbidden=(),
                 public_scope=False):
        self._weights = MappingProxyType(dict(weights))
        if ((not any(self._weights.values()) and not (public_scope and not self._weights))
                or any(not math.isfinite(w) or w < 0 for w in self._weights.values())):
            raise ValueError("nonnegative protected weights with positive sum required")
        self._exact = MappingProxyType({k: Fraction(str(w)) for k, w in self._weights.items()})
        self.denominator = sum(self._exact.values())
        self.background = frozenset(background)
        self.rules = tuple((frozenset(lhs), rhs) for lhs, rhs in rules)
        self.forbidden = frozenset(forbidden)
        self.raw_forbidden = tuple(raw_forbidden)
        if any(type(v) is not bytes or not v for v in self.raw_forbidden):
            raise ValueError("expected nonempty forbidden bytes")
        self._public = self.closure(())
        if self._public & self.forbidden:
            raise PolicyDeniedError("infeasible_background")

    def closure(self, observations):
        known = self.background.union(observations)
        while True:
            expanded = known.union(rhs for lhs, rhs in self.rules if lhs <= known)
            if expanded == known:
                return known
            known = expanded

    def exposure(self, observations):
        if not self.denominator:
            return Fraction(0)
        sensitive = self.closure(observations) - self._public
        return sum(w for f, w in self._exact.items() if f in sensitive) / self.denominator

    def delta(self, observations, facts):
        old = frozenset(observations)
        return self.exposure(old.union(facts)) - self.exposure(old)

    def check(self, observations, payload):
        def contains_literal(value):
            if isinstance(value, str):
                return any(x in value.encode("utf-8", "surrogatepass") for x in self.raw_forbidden)
            if isinstance(value, list):
                return any(map(contains_literal, value))
            if isinstance(value, dict):
                return any(map(contains_literal, value)) or any(map(contains_literal, value.values()))
            return False
        forbidden = any(x in payload for x in self.raw_forbidden)
        if self.raw_forbidden and not forbidden:
            try:
                forbidden = contains_literal(json.loads(payload))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # non-JSON protocols are still subject to the trusted inspector
        if forbidden or self.closure(observations) & self.forbidden:
            raise PolicyDeniedError("forbidden_disclosure")

    def fingerprint(self):
        return [sorted(self._weights.items()), sorted(self.background), sorted(self.forbidden),
                sorted((sorted(lhs), rhs) for lhs, rhs in self.rules),
                sorted(v.hex() for v in self.raw_forbidden)]


class ReviewedEnvelopes:
    """Closed-world inspector for a finite schema/fixture: key the trusted fact map
    on the *entire* envelope, protocol and output cap. Unrecognized bytes fail closed.
    Populate from local policy evaluation, never from proposals or EC² annotations.
    """
    def __init__(self, reviewed):
        self._reviewed = MappingProxyType({key: frozenset(facts) for key, facts in reviewed.items()})

    def __call__(self, wire, action, cap):
        try:
            return self._reviewed[wire, action, cap]
        except KeyError:
            raise PolicyDeniedError("unreviewed_envelope") from None


class TransactionalBoundary:
    """A provisioned scope. All instances for a recipient must share its journal.

    inspect(payload, action, output_cap) must reject unrecognized bytes/metadata,
    and verify any wire output limit. input_bound covers the final serialized
    request. transport(scope, bytes, cap, attempt_id) makes exactly one attempt.
    These three trusted endpoint adapters are pinned by policy_version.
    """
    def __init__(self, path, scope, ledger, *, inspect, input_bound, transport,
                 p_max, c_max, network_allowed=True):
        if not 0 <= p_max <= 1 or type(c_max) is not int or c_max < 0:
            raise ValueError("invalid hard budgets")
        if str(path) == ":memory:":
            raise ValueError("durable local journal required")
        self.scope, self.ledger = scope, ledger
        self._inspect, self._bound, self._transport = inspect, input_bound, transport
        self._pmax, self._cmax, self._network = Fraction(str(p_max)), c_max, network_allowed
        self._key = (scope.principal, scope.recipient_group)
        self._path = Path(path)
        self._payloads = {}
        self._lock = RLock()
        signature = hashlib.sha256(json.dumps(
            [scope.policy_version, ledger.fingerprint(), p_max, c_max, network_allowed]
        ).encode()).hexdigest()
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS policies (p TEXT,r TEXT,sig TEXT,PRIMARY KEY(p,r))")
            db.execute("CREATE TABLE IF NOT EXISTS suspended (p TEXT,r TEXT,PRIMARY KEY(p,r))")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, p TEXT, r TEXT, version INTEGER, digest TEXT,
                action TEXT, facts TEXT, cap INTEGER, charge INTEGER, state TEXT, usage TEXT)""")
            db.execute("INSERT OR IGNORE INTO policies VALUES(?,?,?)", (*self._key, signature))
            if db.execute("SELECT sig FROM policies WHERE p=? AND r=?", self._key).fetchone()[0] != signature:
                raise StaleVersionError("policy_requires_migration")

    @contextmanager
    def _transaction(self):
        with self._lock:
            db = sqlite3.connect(self._path, timeout=30)
            try:
                db.row_factory = sqlite3.Row
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            finally:
                db.close()

    def _history(self, db, sent_only=False):
        query = "SELECT facts, charge FROM attempts WHERE p=? AND r=?"
        if sent_only:
            query += " AND state!='reserved'"
        facts, charge = set(), 0
        for row in db.execute(query, self._key):
            facts.update(json.loads(row["facts"]))
            charge += row["charge"]
        return facts, charge

    def _review(self, db, candidate, *, standalone=False):
        self._check_active(db)
        if not self._network or candidate.action == Action.LOCAL:
            raise PolicyDeniedError("network_disabled")
        wire = candidate.payload
        facts = frozenset(self._inspect(wire, candidate.action, candidate.output_cap))
        if not all(type(f) is str and f for f in facts):
            raise PolicyDeniedError("unrecognized_fact")
        bound = self._bound(wire)
        if type(bound) is not int or bound < 0:
            raise ContractFailedError("invalid_input_bound")
        total = bound + candidate.output_cap
        old, charge = (set(), 0) if standalone else self._history(db)
        self.ledger.check(old | facts, wire)
        if self.ledger.exposure(old | facts) > self._pmax or charge + total > self._cmax:
            raise BudgetExhaustedError("reservation_limit")
        return facts, total, self.ledger.delta(old, facts)

    def _check_active(self, db):
        if db.execute("SELECT 1 FROM suspended WHERE p=? AND r=?", self._key).fetchone():
            raise PolicyDeniedError("untrusted_meter_bound")

    def quote(self, candidate):
        with self._transaction() as db:
            return self._review(db, candidate)[2]

    def quote_sequence(self, candidates):
        """Preview ordered disclosures against one snapshot, without reserving anything."""
        with self._transaction() as db:
            facts, charge = self._history(db)
            initial = self.ledger.exposure(facts)
            for candidate in candidates:
                new, total, _ = self._review(db, candidate)
                facts.update(new)
                charge += total
                self.ledger.check(facts, candidate.payload)
                if self.ledger.exposure(facts) > self._pmax or charge > self._cmax:
                    raise BudgetExhaustedError("continuation_limit")
            return self.ledger.exposure(facts) - initial

    def standalone_quote(self, candidate):
        """B8's single-message quote. reserve always checks all outstanding history."""
        with self._transaction() as db:
            return self._review(db, candidate, standalone=True)[2]

    def totals(self):
        """Return possibly sent exposure and conservative sent token charge."""
        with self._transaction() as db:
            facts, charge = self._history(db, sent_only=True)
            return self.ledger.exposure(facts), charge

    def reserve(self, scope, candidate, attempt_id):
        if scope != self.scope:
            raise StaleVersionError("scope_mismatch")
        if not attempt_id:
            raise ContractFailedError("missing_attempt_id")
        digest = hashlib.sha256(candidate.payload).hexdigest()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row:
                if ((row["p"], row["r"]) != self._key or row["digest"] != digest
                        or row["cap"] != candidate.output_cap or row["action"] != candidate.action.value):
                    raise ContractFailedError("attempt_collision")
                version = row["version"]
            else:
                facts, total, _ = self._review(db, candidate)
                version = 1 + db.execute("SELECT COUNT(*) FROM attempts WHERE p=? AND r=?",
                                         self._key).fetchone()[0]
                db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,NULL)",
                           (attempt_id, *self._key, version, digest, candidate.action.value,
                            json.dumps(sorted(facts)), candidate.output_cap, total, "reserved"))
            if not row or row["state"] == "reserved":
                self._payloads[attempt_id] = candidate.payload
        return Permit(attempt_id, scope, version, digest)

    async def send(self, permit):
        aid = permit.attempt_id
        with self._transaction() as db:
            self._check_active(db)
            row = db.execute("SELECT * FROM attempts WHERE id=? AND p=? AND r=?",
                             (aid, *self._key)).fetchone()
            wire = self._payloads.get(aid)
            if (not row or permit != Permit(aid, self.scope, row["version"], row["digest"])
                    or row["state"] != "reserved" or wire is None
                    or hashlib.sha256(wire).hexdigest() != row["digest"]):
                raise ContractFailedError("invalid_or_spent_permit")
            db.execute("UPDATE attempts SET state='in_flight' WHERE id=?", (aid,))
            del self._payloads[aid]
        started = time.monotonic()
        try:
            body, usage = await self._transport(self.scope, wire, row["cap"], aid)
            if not isinstance(usage, Usage) or usage.attempt_id != aid or type(body) is not bytes:
                raise ContractFailedError("invalid_receipt")
        except Exception:
            usage = Usage(aid, None, None, None, "unknown", time.monotonic() - started)
            with self._transaction() as db:
                db.execute("UPDATE attempts SET state='unknown',usage=? WHERE id=?",
                           (json.dumps(asdict(usage)), aid))
            raise UnknownUsageError("attempt_may_have_left") from None
        known_input = (usage.raw_input or 0) + (usage.cache_read or 0)
        breached = ((usage.output or 0) > row["cap"] or known_input > row["charge"] - row["cap"])
        try:
            charge, state = usage.tokens, "settled"
        except UnknownUsageError:
            charge, state = max(row["charge"], known_input + (usage.output or 0)), "unknown"
        with self._transaction() as db:
            if breached:
                db.execute("INSERT OR IGNORE INTO suspended VALUES(?,?)", self._key)
            db.execute("UPDATE attempts SET state=?,charge=?,usage=? WHERE id=?",
                       (state, charge, json.dumps(asdict(usage)), aid))
        if breached:
            raise ContractFailedError("receipt_outside_reservation")
        return body, usage

    def audit_attempt(self, attempt_id):
        """Return a redacted per-attempt receipt, including the conservative unknown charge."""
        with self._transaction() as db:
            row = db.execute("""SELECT digest AS payload_digest, version AS ledger_version,
                action, state, charge AS reserved_or_billed_tokens FROM attempts
                WHERE id=? AND p=? AND r=?""", (attempt_id, *self._key)).fetchone()
        if not row:
            raise ContractFailedError("unknown_attempt")
        return dict(row)

    def reconcile(self, attempt_id):
        with self._transaction() as db:
            row = db.execute("SELECT state,usage FROM attempts WHERE id=? AND p=? AND r=?",
                             (attempt_id, *self._key)).fetchone()
        if not row:
            raise ContractFailedError("unknown_attempt")
        if row["usage"]:
            return Usage(**json.loads(row["usage"]))
        return Usage(attempt_id, None, None, None, row["state"], 0.0)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError("frozen ledger policy")
        object.__setattr__(self, key, value)
