import json
import os
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from .config import Endpoint


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass
class Call:
    model: str
    answer: str = ''
    elapsed_ms: float = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    success: bool = False
    attempted: bool = False
    error: str | None = None
    finish_reason: str | None = None


class ChatClient:
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        # 回环请求不走系统代理。云端保留环境代理设置。
        from urllib.parse import urlsplit
        handlers = [NoRedirect()]
        if urlsplit(endpoint.base_url).hostname in ('127.0.0.1', 'localhost', '::1'):
            handlers.append(ProxyHandler({}))
        self.opener = build_opener(*handlers)

    def complete(self, prompt, system_prompt, max_tokens, temperature, response_format=None):
        start = time.perf_counter()
        result = Call(self.endpoint.model)
        try:
            key = os.environ.get(self.endpoint.api_key_env) if self.endpoint.api_key_env else None
            if self.endpoint.api_key_env and not key:
                result.error = 'missing_api_key'
                return result
            payload = {'model': self.endpoint.model,
                       'messages': [{'role': 'system', 'content': system_prompt},
                                    {'role': 'user', 'content': prompt}],
                       'temperature': temperature, 'max_tokens': max_tokens,
                       'stream': False}
            if self.endpoint.thinking is not None:
                payload['thinking'] = {'type': self.endpoint.thinking}
            if response_format is not None:
                payload['response_format'] = response_format
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            if key:
                headers['Authorization'] = f'Bearer {key}'
            request = Request(self.endpoint.base_url.rstrip('/') + '/chat/completions',
                              data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                              headers=headers, method='POST')
            result.attempted = True
            with self.opener.open(request, timeout=self.endpoint.timeout_seconds) as response:
                data = json.loads(response.read(8 * 1024 * 1024).decode('utf-8'))
            if not isinstance(data, dict):
                raise ValueError('invalid response')
            if isinstance(data.get('model'), str):
                result.model = data['model']
            usage = data.get('usage')
            if isinstance(usage, dict):
                for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    value = usage.get(name)
                    if type(value) is int and value >= 0:
                        setattr(result, name, value)
            choice = data['choices'][0]
            answer = choice['message']['content']
            result.finish_reason = choice.get('finish_reason')
            if not isinstance(answer, str) or not answer.strip():
                result.error = 'empty_answer'
            else:
                result.answer = answer.strip()
                result.success = result.finish_reason in ('stop', None)
                if not result.success:
                    result.error = 'incomplete_answer'
        except HTTPError as error:
            result.error = f'http_{error.code}'
            error.close()
        except (TimeoutError, URLError, OSError):
            result.error = 'transport_error'
        except (ValueError, KeyError, TypeError, IndexError, UnicodeError):
            result.error = 'invalid_response'
        finally:
            result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result
