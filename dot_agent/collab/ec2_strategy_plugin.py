"""Per-episode callbacks and a non-streaming OpenClaw cloud-endpoint adapter."""
import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from .ec2_bridge import EC2Bridge, EC2ChatEndpoint
from .risk_router import JointPrimalDualController, ModularBaselineController
from .sanitizer import ViewGenerator
from .skill_compiler import VerifiedSkillRuntime
from .types import Action, Candidate, ContractFailedError, PolicyDeniedError

DEFAULTS = {"B6_paac": "0000", "B8_modular": "1001", "B9_joint": "1111"}


@dataclass(frozen=True)
class ConstraintPreservingSanitizer:
    """Envelope builder attaches the actual model, tool schema and public instructions."""
    generator: ViewGenerator
    envelope: Callable[[Candidate], bytes]

    def candidates(self, action, bindings, alternatives, structured, cap):
        if structured:
            views = self.generator.candidates(action, bindings, alternatives, cap)
        else:
            symbols = {k: self.generator.registry.symbol("REF", v) for k, v in bindings.items()}
            views = (Candidate(action, "typed_proxy", json.dumps(symbols).encode(), cap),)
        return tuple(replace(c, payload=self.envelope(c)) for c in views)


class EC2StrategyPlugin:
    """One serialized trajectory per instance. Keep the recipient boundary across episodes.

    forecast(state, candidate, tool_results) supplies remaining costs EXCLUDING this
    message's disclosure. Local state and tool_results never enter a wire implicitly.
    """
    def __init__(self, arm, scope, boundary, sanitizer, forecast, *, shdk=None,
                 runtime: VerifiedSkillRuntime | None = None, scales=(1000, 10, 1), budgets=(500, 5, .1),
                 prices=(.05, .05, .2)):
        bits = DEFAULTS.get(arm, "") if shdk is None else shdk
        if arm not in DEFAULTS or len(bits) != 4 or set(bits) - set("01"):
            raise ValueError("invalid arm or SHDK")
        if arm != "B9_joint" and bits != DEFAULTS[arm]:
            raise ValueError("factorial overrides use the B9_joint arm")
        self.s, self.h, self.d, self.k = (v == "1" for v in bits)
        self.arm, self.sanitizer, self.runtime = arm, sanitizer, runtime
        self.bridge = EC2Bridge(scope, boundary)
        self.episode_id = uuid4().hex
        self.start_p, self.start_charge = boundary.totals()
        self.results = []
        self.closed = False
        self._step_lock = asyncio.Lock()
        quote = boundary.quote if self.h else boundary.standalone_quote

        def estimate(state, c):
            f = forecast(state, c, tuple(self.results))
            delta = 0 if c.action == Action.LOCAL else float(quote(c))
            return replace(f, remaining_exposure=f.remaining_exposure + delta)

        cls = ModularBaselineController if arm == "B8_modular" and not self.d else JointPrimalDualController
        self.controller = cls(lambda state, skills: self.choices, estimate, quote,
                              scales=scales, budgets=budgets, prices=prices,
                              dynamic=self.d, skills=self.k)
        self.choices = ()

    def record_tool_result(self, local_handle):
        if self.closed:
            raise ContractFailedError("episode_closed")
        self.results.append(local_handle)

    async def step_decide(self, state, bindings, *, alternatives=(), output_cap,
                          action=Action.DELEGATE, skill=None, bindings_ref=None):
        """Return (route, result). A local route without a skill delegates to the local host."""
        async with self._step_lock:
            return await self._step(state, bindings, alternatives, output_cap,
                                    action, skill, bindings_ref)

    async def _step(self, state, bindings, alternatives, output_cap, action, skill, bindings_ref):
        if self.closed:
            raise ContractFailedError("episode_closed")
        if self.k and skill is not None:
            if self.runtime is None:
                raise ContractFailedError("missing_skill_runtime")
            return Action.LOCAL, await self.runtime.execute(skill, bindings_ref)
        choices = self.sanitizer.candidates(action, bindings, tuple(alternatives), self.s, output_cap)
        local = Candidate(Action.LOCAL, "local", b"")
        self.choices = (local, *choices)
        if self.arm == "B6_paac":
            choice = choices[0] if choices else local
        else:
            choice = self.controller.decide(state, self.choices)
        if choice is None or choice.action == Action.LOCAL:
            return Action.LOCAL, None
        body, _ = await self.bridge.intercept(choice.action, choice.payload, uuid4().hex,
                                              view_id=choice.view_id, output_cap=output_cap)
        return choice.action, body

    def finalize_episode(self):
        if self._step_lock.locked():
            raise ContractFailedError("pending_step")
        if not self.closed:
            self.metrics = self.bridge.finish()
            p, charge = self.bridge.boundary.totals()
            self.controller.close_episode(self.episode_id, charge-self.start_charge,
                                          self.metrics.wall_s, float(p-self.start_p))
            self.closed = True
        return self.metrics

    def openclaw_endpoint(self, project, local_chat, *, action=Action.ADVISE):
        """Mount at CLOUD_BASE_URL for the stock advisory runner.

        Trusted project(wire) returns (state, bindings) using the local task schema.
        Both transports must return OpenAI-compatible completion response bytes.
        """
        plugin = self

        class Adapter:
            async def intercept(self, route, wire, attempt_id, *, view_id, output_cap):
                if type(output_cap) is not int or output_cap <= 0:
                    raise PolicyDeniedError("invalid_cap")
                state, bindings = project(wire)
                chosen, body = await plugin.step_decide(state, bindings, output_cap=output_cap,
                                                        action=route)
                if chosen == Action.LOCAL:
                    body = await local_chat(wire)
                return body, None

        return EC2ChatEndpoint(Adapter(), action)


def export_skill(directory: Path, arm="B9_joint"):
    """Write OpenClaw discovery instructions. The endpoint enforces the policy."""
    if arm not in DEFAULTS:
        raise ValueError("unknown strategy")
    path = Path(directory) / "ec2-collab" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nname: ec2-collab\ndescription: Request policy-reviewed assistance "
                    "for the current tool step.\n---\n\n"
                    f"Use the installed advisor tool for {arm}. Keep bindings and tool results "
                    "local. Continue locally when a request is denied.\n", encoding="utf8")
    return path
