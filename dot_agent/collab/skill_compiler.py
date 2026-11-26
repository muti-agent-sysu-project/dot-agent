"""A declarative, non-executing skill with local state assertions."""
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    steps: tuple[str, ...]
    requires: tuple[tuple[str, str], ...]
    ensures: tuple[tuple[str, str], ...]

    def can_run(self, state: Mapping[str, str]) -> bool:
        return all(state.get(key) == value for key, value in self.requires)

    def verified(self, state: Mapping[str, str]) -> bool:
        return all(state.get(key) == value for key, value in self.ensures)


def compile_skill(steps: list[str], requires: Mapping[str, str],
                  ensures: Mapping[str, str]) -> Skill:
    if not steps or any(not step.strip() for step in steps) or not ensures:
        raise ValueError("steps and postconditions required")
    return Skill(tuple(steps), tuple(sorted(requires.items())), tuple(sorted(ensures.items())))

# The legacy declarative contract above remains available via compile_skill.
import copy
import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
from threading import RLock
from types import MappingProxyType
from typing import Callable

from .types import ContractFailedError, Node, Skill as VerifiedSkill, StaleVersionError


@dataclass(frozen=True)
class Primitive:
    """Trusted deterministic, offline adapter. Predicates and observe read real state.

    run(state, args, commit_key) may mutate only the declared write set.
    observe(state, args) returns the direct successor handle after post passes.
    """
    parameters: tuple[tuple[str, str], ...]
    result_type: str
    reads: frozenset[str]
    writes: frozenset[str]
    pre: tuple[str, ...]
    post: tuple[str, ...]
    idempotent: bool
    run: Callable
    observe: Callable


@dataclass(frozen=True)
class TraceEvent:
    op_version: str
    dependencies: tuple[int, ...]
    # Public argument name -> either a private binding name or a predecessor index.
    sources: tuple[tuple[str, str | int], ...]


@dataclass(repr=False)
class Example:
    inputs: dict
    initial: dict
    accepts: Callable  # independent terminal oracle on the observed final state


@dataclass(repr=False)
class LocalTrace:
    events: tuple[TraceEvent, ...]
    original: Example
    alternatives: tuple[Example, ...]
    negatives: tuple[Example, ...]


_TYPES = {"str": str, "int": int, "bool": bool, "float": float}
_PUBLIC_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")


def validate_dag(nodes, operations):
    nodes = tuple(nodes)
    by_key = {n.key: n for n in nodes}
    if not nodes or len(by_key) != len(nodes):
        raise ContractFailedError("empty_or_duplicate_graph")
    done, levels = set(), []
    while len(done) < len(nodes):
        layer = [n for n in nodes if n.key not in done and set(n.dependencies) <= done]
        if not layer:
            raise ContractFailedError("cycle_or_dangling_edge")
        reads, writes = set(), set()
        for n in layer:
            op = operations.get(n.op_version)
            if not op or n.effect != ("write" if op.writes else "read"):
                raise ContractFailedError("unregistered_effect")
            if op.writes & (reads | writes) or op.reads & writes:
                raise ContractFailedError("layer_conflict")
            reads.update(op.reads)
            writes.update(op.writes)
        levels.append(tuple(layer))
        done.update(n.key for n in layer)
    return tuple(levels)


class VerifiedSkillRuntime:
    """All trace/binding references are local. Only generated IR is exportable.

    resolve_run(uuid) returns (parameter-slot values, live state). Use a fresh
    opaque UUID per invocation, retain it on retry. SQLite stores effect status,
    never argument values, exception text, or small-domain hashes.
    """
    def __init__(self, operations, predicates, policy_version, journal, *, resolve_trace,
                 resolve_run, resolve_edit, retain=True):
        self.ops = MappingProxyType(dict(operations))
        self.predicates = MappingProxyType(dict(predicates))
        self.policy = policy_version
        if not _PUBLIC_ID.fullmatch(policy_version):
            raise ValueError("public policy version required")
        for name, op in self.ops.items():
            names = (name, *(k for k, t in op.parameters), *op.pre, *op.post)
            if (not all(_PUBLIC_ID.fullmatch(k) for k in names)
                    or not op.post or not set(op.pre + op.post) <= self.predicates.keys()
                    or any(t not in _TYPES for _, t in op.parameters)
                    or op.result_type not in _TYPES or len(dict(op.parameters)) != len(op.parameters)):
                raise ValueError("invalid public primitive schema")
        self.journal = journal
        if str(journal) == ":memory:":
            raise ValueError("durable journal required")
        self.trace, self.resolve, self.edits = resolve_trace, resolve_run, resolve_edit
        self.retain = retain
        self._versions, self._tests, self._failures = {}, {}, {}
        self._roots, self._repairs, self._completed = {}, {}, {}
        self._lock = RLock()
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS effects (
                run TEXT,node TEXT,status TEXT,op TEXT,PRIMARY KEY(run,node))""")
            db.execute("CREATE TABLE IF NOT EXISTS invocations (run TEXT PRIMARY KEY,root TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                version TEXT PRIMARY KEY,policy TEXT,root TEXT,ir TEXT)""")
            if retain:
                for version, root, ir in db.execute(
                    "SELECT version,root,ir FROM artifacts WHERE policy=?", (policy_version,)
                ):
                    records = json.loads(ir)
                    nodes = tuple(Node(n["key"], n["op_version"], tuple(n["dependencies"]),
                                       tuple(map(tuple, n["arguments"])), tuple(n["pre"]),
                                       tuple(n["post"]), n["effect"], n["idempotency_slot"])
                                  for n in records["nodes"])
                    if any(n.op_version not in self.ops for n in nodes):
                        continue  # tool was removed; old artifact cannot execute here
                    skill = self._artifact(nodes, tuple(map(tuple, records["parameter_schema"])))
                    if skill.version != version:
                        raise StaleVersionError("artifact_registry_drift")
                    self._versions[version], self._roots[version] = skill, root

    @contextmanager
    def _db(self):
        with closing(sqlite3.connect(self.journal)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def _lower(self, events):
        slots, nodes = {}, []
        for index, event in enumerate(events):
            op = self.ops.get(event.op_version)
            if op is None:
                raise ContractFailedError("unregistered_op")
            sources = dict(event.sources)
            if set(sources) != dict(op.parameters).keys() or len(sources) != len(event.sources):
                raise ContractFailedError("invalid_arguments")
            if any(type(i) is not int or i < 0 for i in event.dependencies):
                raise ContractFailedError("invalid_dependency")
            args = []
            for arg, typ in op.parameters:
                source = sources[arg]
                if type(source) is int:
                    if (source not in event.dependencies or not 0 <= source < len(events)
                            or self.ops[events[source].op_version].result_type != typ):
                        raise ContractFailedError("invalid_data_edge")
                    ref = f"n{source}"
                elif type(source) is str:
                    # Only provenance identity is compared. Even equal raw values
                    # from different sources remain independently bindable.
                    ref, old_type = slots.setdefault(source, (f"p{len(slots)}", typ))
                    if old_type != typ:
                        raise ContractFailedError("inconsistent_slot_type")
                else:
                    raise ContractFailedError("literal_in_ir")
                args.append((arg, ref))
            nodes.append(Node(f"n{index}", event.op_version, tuple(f"n{i}" for i in event.dependencies),
                              tuple(args), op.pre, op.post, "write" if op.writes else "read",
                              "invocation" if op.idempotent and op.writes else None))
        return tuple(nodes), slots

    def _artifact(self, nodes, schema):
        validate_dag(nodes, self.ops)
        contracts = []
        for n in nodes:
            op = self.ops[n.op_version]
            contracts.append((op.parameters, op.result_type, sorted(op.reads), sorted(op.writes),
                              op.pre, op.post, op.idempotent))
        public = json.dumps([self.policy, schema, [asdict(n) for n in nodes], contracts], sort_keys=True)
        version = hashlib.sha256(public.encode()).hexdigest()
        return VerifiedSkill(version, self.policy, schema, nodes)

    def _publish(self, skill, root):
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                       (skill.version, self.policy, root, json.dumps(asdict(skill))))
        self._versions[skill.version], self._roots[skill.version] = skill, root

    def _claim(self, run, node, op):
        with self._db() as db:
            row = db.execute("SELECT status,op FROM effects WHERE run=? AND node=?",
                             (run, node.key)).fetchone()
            if row:
                if row[0] == "done":
                    if row[1] == node.op_version or not op.idempotent:
                        return False
                elif not op.idempotent or row[0] == "in_flight":
                    raise ContractFailedError("unknown_commit")
            db.execute("INSERT OR REPLACE INTO effects VALUES(?,?,'in_flight',?)",
                       (run, node.key, node.op_version))
        return True

    def _mark(self, run, node, status):
        with self._db() as db:
            db.execute("UPDATE effects SET status=? WHERE run=? AND node=?", (status, run, node))

    def _interpret(self, skill, inputs, state, *, run=None):
        if set(inputs) != dict(skill.parameter_schema).keys() or any(
            type(inputs[k]) is not _TYPES[t] for k, t in skill.parameter_schema
        ):
            raise ContractFailedError("binding_schema")
        handles = {}
        repair = self._repairs.get(skill.version)
        if run and repair:
            parent, affected = repair
            handles.update({key: value for key, value in self._completed.get((run, parent), {}).items()
                            if key not in affected})
        for layer in validate_dag(skill.nodes, self.ops):
            for n in layer:
                if n.key in handles:
                    continue
                op = self.ops[n.op_version]
                args = {name: inputs[ref] if ref in inputs else handles[ref] for name, ref in n.arguments}
                try:
                    if not all(self.predicates[p](state, args) for p in n.pre):
                        raise ContractFailedError("precondition")
                    durable = bool(run and op.writes)
                    if not durable or self._claim(run, n, op):
                        try:
                            op.run(state, args, (run, n.key) if n.idempotency_slot else None)
                        except Exception:
                            if durable:
                                self._mark(run, n.key, "unknown")
                            raise ContractFailedError("unknown_commit") from None
                        if durable:
                            self._mark(run, n.key, "done")
                    if not all(self.predicates[p](state, args) for p in n.post):
                        raise ContractFailedError("postcondition")
                    handles[n.key] = op.observe(state, args)
                    if type(handles[n.key]) is not _TYPES[op.result_type]:
                        raise ContractFailedError("result_schema")
                    if run:
                        self._completed.setdefault((run, skill.version), {})[n.key] = handles[n.key]
                except Exception as err:
                    if run:
                        codes = {"unknown_commit", "precondition", "postcondition", "result_schema"}
                        code = err.args[0] if (type(err) is ContractFailedError and len(err.args) == 1
                                               and type(err.args[0]) is str and err.args[0] in codes) else "tool_contract"
                        self._failures[run] = (skill.version, n.key, code)
                    raise ContractFailedError("node_contract") from None

    def _verify(self, skill, slots, trace):
        if not trace.alternatives or not trace.negatives:
            return False
        def bind(case):
            return {ref: case.inputs[src] for src, (ref, _) in slots.items()}
        original = bind(trace.original)
        for index, case in enumerate((trace.original, *trace.alternatives)):
            values = bind(case)
            if index and values == original and case.initial == trace.original.initial:
                return False
            state = copy.deepcopy(case.initial)
            self._interpret(skill, values, state)
            if not case.accepts(state):
                return False
        for case in trace.negatives:
            try:
                self._interpret(skill, bind(case), copy.deepcopy(case.initial))
            except ContractFailedError:
                continue
            return False
        return True

    def compile(self, trace_ref, policy_version):
        if self.policy != policy_version:
            raise StaleVersionError("policy_mismatch")
        if not self.retain:
            return None
        trace = self.trace(trace_ref)
        try:
            nodes, slots = self._lower(trace.events)
            skill = self._artifact(nodes, tuple(slots.values()))
            if not self._verify(skill, slots, trace):
                return None
        except (ContractFailedError, KeyError, TypeError, ValueError):
            return None
        with self._lock:
            self._publish(skill, self._roots.get(skill.version, skill.version))
            self._tests[skill.version] = slots, trace
        return skill

    async def execute(self, skill, bindings_ref):
        if self.policy != skill.policy_version or self._versions.get(skill.version) != skill:
            raise StaleVersionError("unvalidated_skill")
        if str(uuid.UUID(bindings_ref)) != bindings_ref:
            raise ContractFailedError("invalid_invocation")
        inputs, state = self.resolve(bindings_ref)
        with self._lock:
            root = self._roots[skill.version]
            with self._db() as db:
                db.execute("INSERT OR IGNORE INTO invocations VALUES(?,?)", (bindings_ref, root))
                if db.execute("SELECT root FROM invocations WHERE run=?", (bindings_ref,)).fetchone() != (root,):
                    raise ContractFailedError("invocation_collision")
            self._interpret(skill, inputs, state, run=bindings_ref)
            self._failures.pop(bindings_ref, None)
        return "verified"

    def patch(self, skill, failure_ref, edit_ref):
        """Failure handle is the invocation ID. Edits select one registered replacement."""
        with self._lock:
            failure = self._failures.get(failure_ref)
            if (not failure or failure[0] != skill.version or failure[2] == "unknown_commit"
                    or skill.version not in self._tests or self._versions.get(skill.version) != skill
                    or self.policy != skill.policy_version):
                return None
            node_key, replacement, regression = self.edits(edit_ref)
            if failure[1] != node_key or replacement not in self.ops:
                return None
            old = next(n for n in skill.nodes if n.key == node_key)
            a, b = self.ops[old.op_version], self.ops[replacement]
            if (replacement == old.op_version or a.parameters != b.parameters
                    or a.result_type != b.result_type or a.reads != b.reads or a.writes != b.writes
                    or a.pre != b.pre or a.post != b.post or a.idempotent != b.idempotent):
                return None
            nodes = tuple(replace(n, op_version=replacement) if n == old else n for n in skill.nodes)
            candidate = self._artifact(nodes, skill.parameter_schema)
            slots, trace = self._tests[skill.version]
            suite = replace(trace, alternatives=(*trace.alternatives, regression))
            try:
                if not self._verify(candidate, slots, suite):
                    return None
            except (ContractFailedError, KeyError, TypeError, ValueError):
                return None
            self._publish(candidate, self._roots[skill.version])
            self._tests[candidate.version] = slots, suite
            affected = {node_key}
            for layer in validate_dag(nodes, self.ops):
                affected.update(n.key for n in layer if affected.intersection(n.dependencies))
            self._repairs[candidate.version] = skill.version, affected
            return candidate


def export_openclaw_skill_md(skill):
    if not re.fullmatch(r"[0-9a-f]{64}", skill.version):
        raise ContractFailedError("invalid_artifact_handle")
    return ("---\nname: verified-local-skill\ndescription: Run a verified local artifact.\n---\n"
            f"Invoke the registered local skill runtime with artifact ID `{skill.version}` "
            "and a local binding handle.\n")
