"""Task-local bindings and finite, schema-directed view synthesis."""
import json
import operator
import re
import secrets
from dataclasses import dataclass
from functools import wraps
from threading import RLock

from .types import BudgetExhaustedError, Candidate, ContractFailedError, PolicyDeniedError, Span

_TOKEN = re.compile(r"\[\[EC2:([A-Z][A-Z0-9_]*):([0-9a-f]{24}):([1-9]\d*)\]\]")
_DIRECT = re.compile(
    r"(?P<EMAIL>[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})|(?P<PHONE>(?<!\d)1[3-9]\d{9}(?!\d))"
    r"|(?P<CREDENTIAL>(?i:\b(?:bearer\s+[\w.+/=-]+|(?:api[_-]?key|password|secret|token)"
    r"""["']?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;}]+))|\bsk-[A-Za-z0-9_-]+))"""
)


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class Registry:
    """One instance per task. All remote identifiers are opaque local references."""
    def __init__(self):
        self._space = secrets.token_hex(12)
        self._tokens = {}
        self._originals = {}
        self._lock = RLock()

    @_locked
    def symbol(self, kind, value):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", kind):
            raise ValueError("invalid entity type")
        key = kind, value
        if key not in self._tokens:
            token = f"[[EC2:{kind}:{self._space}:{len(self._tokens) + 1}]]"
            self._tokens[key] = token
            self._originals[token] = value
        return self._tokens[key]

    @_locked
    def mask(self, text: str, spans: list[Span] | None = None) -> str:
        if "[[EC2:" in text:
            raise ValueError("reserved placeholder prefix")
        ordered = sorted(spans or (), key=lambda s: s.start)
        end = 0
        for span in ordered:
            if not end <= span.start < span.end <= len(text):
                raise ValueError("invalid span")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", span.kind):
                raise ValueError("invalid type")
            end = span.end
        detected = [Span(m.start(), m.end(), m.lastgroup) for m in _DIRECT.finditer(text)]
        merged = []
        for span in sorted(ordered + detected, key=lambda s: s.start):
            if merged and span.start < merged[-1].end:
                prior = merged.pop()
                span = Span(prior.start, max(prior.end, span.end), prior.kind)
            merged.append(span)
        pieces, end = [], 0
        for span in merged:
            pieces.extend((text[end:span.start], self.symbol(span.kind, text[span.start:span.end])))
            end = span.end
        return "".join(pieces) + text[end:]

    def restore(self, text):
        def original(m):
            if m[0] not in self._originals:
                raise ValueError("unknown placeholder")
            return self._originals[m[0]]
        return _TOKEN.sub(original, text)


@dataclass(frozen=True)
class Inv:
    """Public schema predicate. Fields name authorized bindings, never raw literals."""
    name: str
    kind: str
    fields: tuple[str, ...]
    threshold: int = 0
    pattern: str = ""

    def evaluate(self, bindings):
        values = [bindings[f] for f in self.fields]
        if self.kind == "join":
            return values[0] == values[1]
        if self.kind in ("lt", "gt", "le", "ge"):
            rhs = self.threshold if len(values) == 1 else values[1]
            return getattr(operator, self.kind)(values[0], rhs)
        if self.kind == "format":
            return re.fullmatch(self.pattern, values[0]) is not None
        if self.kind == "length":
            return len(values[0]) == self.threshold
        if self.kind == "reachable":
            graph, start, goal = values
            pending, visited = [start], set()
            while pending:
                node = pending.pop()
                if node == goal:
                    return True
                if node not in visited:
                    visited.add(node)
                    pending.extend(graph.get(node, ()))
            return False
        raise ContractFailedError("unknown_predicate")


VIEWS = ("omit", "typed_proxy", "predicate_only", "local_compute_summary", "allowed_raw")


class ViewGenerator:
    """Public field/predicate IDs, alternatives and policy are supplied by trusted local schema."""
    def __init__(self, registry, predicates, *, public_raw=(), forbidden_fields=(),
                 forbidden_literals=(), allowed_predicates=()):
        self.registry = registry
        self.predicates = tuple(predicates)
        self.forbidden_fields = frozenset(forbidden_fields)
        self.raw = frozenset(public_raw) - self.forbidden_fields
        self.denied = tuple(forbidden_literals)
        self.allowed = frozenset(allowed_predicates)
        if len({p.name for p in self.predicates}) != len(self.predicates):
            raise ContractFailedError("duplicate_predicate")

    def _render(self, view, values, registry):
        truth = {p.name: p.evaluate(values) for p in self.predicates}
        if not truth.keys() <= self.allowed:
            return None
        if view == "omit":
            obj, replay = {}, {}
        elif view == "typed_proxy":
            if any(p.kind != "join" for p in self.predicates):
                return None
            obj = {f: registry.symbol("REF", values[f]) for p in self.predicates for f in p.fields}
            replay = {p.name: p.evaluate(obj) for p in self.predicates}
        elif view == "predicate_only":
            obj = truth.copy()
            replay = obj
        elif view == "local_compute_summary":
            # A single validation bit suffices only when it recovers every required truth.
            obj = {"all_pass": all(truth.values())}
            replay = dict.fromkeys(truth, obj["all_pass"])
        else:
            obj = {k: values[k] for k in sorted(self.raw)}
            denied = [values[k] for k in self.forbidden_fields if k in values]
            def leaks(value):
                if isinstance(value, dict):
                    return any(leaks(k) or leaks(v) for k, v in value.items())
                if isinstance(value, (list, tuple)):
                    return any(leaks(v) for v in value)
                return any(value == d or (isinstance(value, str) and str(d) in value)
                           for d in denied)
            if leaks(obj):
                return None
            if any(f not in obj for p in self.predicates for f in p.fields):
                return None
            replay = {p.name: p.evaluate(obj) for p in self.predicates}
        if replay != truth:
            return None
        wire = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":")).encode()
        if _DIRECT.search(wire.decode()) or any(v.encode() in wire for v in self.denied):
            return None
        return wire

    def candidates(self, action, bindings, alternatives, output_cap=0):
        alternatives = tuple(alternatives)
        choices = []
        for view in VIEWS:
            wire = self._render(view, bindings, self.registry)
            if wire is not None and all(self._render(view, alt, Registry()) is not None
                                        for alt in alternatives):
                choices.append(Candidate(action, view, wire, output_cap))
        return tuple(choices)


def minimum_view(choices, tolerance, tax_upper, quote):
    """quote re-inspects the final envelope and raises on a failed hard gate."""
    legal = []
    for c in choices:
        if tax_upper(c) <= tolerance:
            try:
                delta = quote(c)
            except (PolicyDeniedError, BudgetExhaustedError):
                continue
            legal.append((delta, len(c.payload), VIEWS.index(c.view_id), c))
    return min(legal, default=(None,), key=lambda row: row[:3])[-1]
