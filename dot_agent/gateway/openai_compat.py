"""OpenAI-compatible local/cloud gateway; no secret-bearing response errors."""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from ..types import Caller, GenerationResult, ModelTier
from .base import ModelClient


class GenerationError(RuntimeError):
    """A failed answer can still have billable, provider-reported usage."""
    def __init__(self, code: str, result: GenerationResult):
        super().__init__(code)
        self.result = result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _make_opener(base_url: str):
    handlers = [_NoRedirect()]
    if urlsplit(base_url).hostname in ("127.0.0.1", "localhost", "::1"):
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


class OpenAICompatClient(ModelClient):
    tier = ModelTier.MC

    def __init__(self, base_url=None, api_key=None, model=None, timeout_s=120.0,
                 *, tier=ModelTier.MC, thinking=None, request_logprobs=False,
                 length_retry_cap=0):
        self.tier = tier
        self.base_url = (base_url or os.environ.get("CLOUD_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
        self.api_key = (os.environ.get("CLOUD_API_KEY", "") if api_key is None else api_key)
        self.model = model or os.environ.get("CLOUD_MODEL", "deepseek-chat")
        self.timeout_s = timeout_s
        if thinking not in (None, "enabled", "disabled"):
            raise ValueError("thinking must be enabled or disabled")
        self.thinking = thinking
        self.request_logprobs = request_logprobs
        if type(length_retry_cap) is not int or length_retry_cap < 0:
            raise ValueError("length_retry_cap must be a nonnegative integer")
        self.length_retry_cap = length_retry_cap
        self.name = f"{'local' if tier is ModelTier.MD else 'cloud'}:{self.model}"
        self.opener = _make_opener(self.base_url)

    async def generate(self, prompt, *, max_tokens=512, temperature=0.0,
                       caller=Caller.EXECUTE):
        # Blocking urllib must not serialize the orchestrator's independent tasks.
        return await asyncio.to_thread(self._generate, prompt, max_tokens, temperature)

    def _generate(self, prompt, max_tokens, temperature):
        if self.tier is ModelTier.MC and not self.api_key:
            raise RuntimeError("CLOUD_API_KEY is required")
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": temperature}
        if self.request_logprobs:
            payload["logprobs"] = True
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(), headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with self.opener.open(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            code = e.code
            e.close()
            raise RuntimeError(f"model_http_{code}") from None
        except (urllib.error.URLError, OSError):
            raise RuntimeError("model_connection_error") from None
        except (ValueError, UnicodeError):
            raise RuntimeError("model_invalid_json") from None
        latency = time.perf_counter() - t0
        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        known = all(type(n) is int and n >= 0 for n in (pt, ct))
        result = GenerationResult(text="", prompt_tokens=pt if known else 0,
            completion_tokens=ct if known else 0, latency_s=latency, usage_missing=not known)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content")
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise GenerationError("model_invalid_response", result) from None
        result.finish_reason = finish if isinstance(finish, str) else None
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            result.reasoning_chars = len(reasoning)
        details = usage.get("completion_tokens_details")
        count = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if type(count) is int and count >= 0:
            result.reasoning_tokens = count  # subset of completion_tokens
        if finish not in ("stop", None):
            raise GenerationError("model_incomplete_answer", result)
        if not isinstance(text, str) or not text.strip():
            raise GenerationError("model_empty_answer", result)
        logs = choice.get("logprobs")
        items = logs.get("content") if isinstance(logs, dict) else None
        if isinstance(items, list):
            probs = []
            for item in items:
                value = item.get("logprob") if isinstance(item, dict) else None
                if type(value) in (int, float) and -1e308 <= value <= 0:
                    probs.append(math.exp(value))
            result.token_probs = probs or None
        result.text = text
        return result
