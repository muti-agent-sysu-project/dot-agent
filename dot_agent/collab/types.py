"""Frozen contracts shared by the local controller, interpreter and egress boundary."""
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Protocol


class CollaborationError(Exception):
    """Only stable error classes, never exception text, are fit for routing."""


class PolicyDeniedError(CollaborationError): pass
class BudgetExhaustedError(CollaborationError): pass
class StaleVersionError(CollaborationError): pass
class ContractFailedError(CollaborationError): pass
class UnknownUsageError(CollaborationError): pass


class Action(str, Enum):
    LOCAL = "Local-Exec"
    DELEGATE = "Minimal-Proxy-Cloud-Delegate"
    ADVISE = "Local-Compute-Cloud-Advise"
    FALLBACK = "Constrained-Cloud-Fallback"
    PROXY = DELEGATE
    FULL = FALLBACK


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class Scope:
    principal: str
    recipient_group: str
    policy_version: str


@dataclass(frozen=True)
class Candidate:
    action: Action
    view_id: str
    payload: bytes = field(repr=False)
    output_cap: int = 0

    def __post_init__(self):
        if (type(self.payload) is not bytes or type(self.output_cap) is not int
                or self.output_cap < 0 or not isinstance(self.action, Action)):
            raise ContractFailedError("invalid_candidate")


@dataclass(frozen=True)
class Forecast:
    q_lower: float
    failure_upper: float
    remaining_tokens: float
    remaining_seconds: float
    remaining_exposure: float
    reuse_value: float

    def __post_init__(self):
        if (not 0 <= self.q_lower <= 1 or not 0 <= self.failure_upper <= 1
                or any(not isfinite(x) or x < 0 for x in
                       (self.remaining_tokens, self.remaining_seconds, self.remaining_exposure))
                or not isfinite(self.reuse_value)):
            raise ContractFailedError("invalid_forecast")


@dataclass(frozen=True)
class Usage:
    attempt_id: str
    raw_input: int | None
    cache_read: int | None
    output: int | None
    status: str
    elapsed_s: float

    def __post_init__(self):
        if (not isfinite(self.elapsed_s) or self.elapsed_s < 0 or
                any(x is not None and (type(x) is not int or x < 0)
                    for x in (self.raw_input, self.cache_read, self.output))):
            raise ContractFailedError("invalid_usage")

    @classmethod
    def from_inclusive_input(cls, attempt_id, input_tokens, cached, output, status, elapsed_s):
        raw = None if cached is None or input_tokens is None else input_tokens - cached
        return cls(attempt_id, raw, cached, output, status, elapsed_s)

    @property
    def tokens(self) -> int:
        if None in (self.raw_input, self.cache_read, self.output):
            raise UnknownUsageError("incomplete_usage")
        return self.raw_input + self.cache_read + self.output


@dataclass(frozen=True)
class Node:
    key: str
    op_version: str
    dependencies: tuple[str, ...]
    arguments: tuple[tuple[str, str], ...]
    pre: tuple[str, ...]
    post: tuple[str, ...]
    effect: str
    idempotency_slot: str | None


@dataclass(frozen=True)
class Skill:
    version: str
    policy_version: str
    parameter_schema: tuple[tuple[str, str], ...]
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class Permit:
    attempt_id: str
    scope: Scope
    ledger_version: int
    payload_digest: str


class Boundary(Protocol):
    def reserve(self, scope: Scope, candidate: Candidate, attempt_id: str) -> Permit: ...
    async def send(self, permit: Permit) -> tuple[bytes, Usage]: ...
    def reconcile(self, attempt_id: str) -> Usage: ...


class Controller(Protocol):
    def candidates(self, state_ref: str) -> tuple[Candidate, ...]: ...
    def estimate(self, state_ref: str, action: Candidate) -> Forecast: ...
    def decide(self, state_ref: str, choices: tuple[Candidate, ...]) -> Candidate | None: ...
    def close_episode(self, episode_id: str, tokens: int | None, wall_s: float,
                      delta_p: float) -> None: ...


class SkillRuntime(Protocol):
    def compile(self, trace_ref: str, policy_version: str) -> Skill | None: ...
    async def execute(self, skill: Skill, bindings_ref: str) -> str: ...
    def patch(self, skill: Skill, failure_ref: str, edit_ref: str) -> Skill | None: ...
