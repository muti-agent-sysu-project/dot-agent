"""In-process interception point for the EC² cloud adapter and offline evaluation.

Install intercept in place of the runner's final cloud send, once per physical attempt.
The adapter's immutable request includes model, instructions, history, tool schemas and
other recipient-visible metadata. The transport configured on Boundary owns the endpoint.
"""
import json
import time
from dataclasses import dataclass
from threading import Lock

from .types import Candidate, ContractFailedError, UnknownUsageError


@dataclass(frozen=True)
class ServiceMetrics:
    attempts: int
    known_tokens: int
    unknown_attempts: int
    wall_s: float

    @property
    def tokens(self):
        return None if self.unknown_attempts else self.known_tokens


class EC2Bridge:
    """One task clock including local work, compile, repair, retry and service judge.

    Offline grading is a separate function with no reference from the controller.
    """
    def __init__(self, scope, boundary, *, clock=time.monotonic):
        self.scope, self.boundary = scope, boundary
        self._clock, self._start = clock, clock()
        self._attempts = set()
        self._end = None
        self._lock = Lock()
        self._active = 0

    async def intercept(self, action, request_bytes, attempt_id, *, view_id, output_cap):
        c = Candidate(action, view_id, request_bytes, output_cap)
        with self._lock:
            if self._end is not None:
                raise ContractFailedError("episode_closed")
            permit = self.boundary.reserve(self.scope, c, attempt_id)
            self._attempts.add(attempt_id)
            self._active += 1
        try:
            return await self.boundary.send(permit)
        finally:
            with self._lock:
                self._active -= 1

    def finish(self):
        with self._lock:
            if self._active:
                raise ContractFailedError("pending_service_attempts")
            if self._end is None:
                self._end = self._clock()
            known = unknown = 0
            for attempt in self._attempts:
                receipt = self.boundary.reconcile(attempt)
                try:
                    known += receipt.tokens
                except UnknownUsageError:
                    unknown += 1
            return ServiceMetrics(len(self._attempts), known, unknown, self._end - self._start)


def offline_privacy_exposure(annotation, verdicts):
    """Apply EC² unit mapping to completed official per-field verdicts, after the run.

    None denotes missing annotation or unscored evidence. Exact-match diagnostics
    in ec2_adapter are deliberately separate from the official semantic verdicts.
    """
    if annotation is None or verdicts is None:
        return None
    expected = [(item["entity_id"], field["field"]) for item in annotation["items"]
                for field in item["sensitive_fields"]]
    if not expected:
        return None
    scores = {}
    for row in verdicts["fields"]:
        key = row["entity_id"], row["field"]
        if key in scores or type(row["score"]) is not int or row["score"] not in (0, 1):
            raise ContractFailedError("invalid_grader_verdict")
        scores[key] = row["score"]
    if len(set(expected)) != len(expected) or set(scores) != set(expected):
        raise ContractFailedError("grader_unit_mismatch")
    return sum(scores.values()) / len(expected)

class EC2ChatEndpoint:
    """Local ASGI target for EC² CLOUD_BASE_URL (/v1, nonstreaming upstream).

    Each instance serves one configured task, recipient and protocol. Only the
    reviewed body is forwarded; the transport supplies fixed endpoint metadata.
    No grader files, inbound credentials or request headers enter the budget state.
    """
    def __init__(self, bridge, action, *, max_bytes=2**20):
        self.bridge, self.action, self.max_bytes = bridge, action, max_bytes

    async def __call__(self, scope, receive, send):
        from uuid import uuid4
        from .types import CollaborationError

        if scope["type"] != "http":
            raise ValueError("HTTP only")
        code, body = 404, b'{"error":"unknown_route"}'
        if (scope.get("method") == "POST" and scope.get("path") == "/v1/chat/completions"
                and not scope.get("query_string")):
            try:
                chunks = bytearray()
                while True:
                    event = await receive()
                    if event["type"] != "http.request":
                        return
                    if len(chunks) + len(event["body"]) > self.max_bytes:
                        raise ValueError("oversized_request")
                    chunks.extend(event["body"])
                    if not event.get("more_body"):
                        break
                wire = bytes(chunks)
                request = json.loads(wire)
                if (type(request) is not dict or request.get("stream", False) is not False
                        or sum(k in request for k in ("max_tokens", "max_completion_tokens")) != 1):
                    raise ValueError("invalid_chat_request")
                cap = request.get("max_tokens", request.get("max_completion_tokens"))
                body, _ = await self.bridge.intercept(self.action, wire, uuid4().hex,
                                                      view_id="ec2_chat", output_cap=cap)
                code = 200
            except (ValueError, UnicodeError, RecursionError):
                code, body = 400, b'{"error":"invalid_request"}'
            except CollaborationError as error:
                code = 502 if isinstance(error, UnknownUsageError) else 403
                body = json.dumps({"error": type(error).__name__}).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
