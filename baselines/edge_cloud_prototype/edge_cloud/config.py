import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: float = 120
    thinking: str | None = None

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (url.scheme not in ('http', 'https') or not url.hostname or
                url.username or url.password or url.query or url.fragment):
            raise ValueError('base_url 必须为无凭据、无 query 的 HTTP(S) 地址')
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('model 不可为空')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('timeout_seconds 必须为有限正数')
        if self.thinking not in (None, 'enabled', 'disabled'):
            raise ValueError('thinking 必须为 enabled 或 disabled')


@dataclass(frozen=True)
class Config:
    local: Endpoint
    cloud: Endpoint | None = None
    max_tokens: int = 256
    temperature: float = 0
    system_prompt: str = '准确回答用户问题，遵守用户要求的输出格式。'
    long_prompt_chars: int = 300
    complex_keywords: tuple = field(default_factory=lambda: (
        '证明', '多步推理', '约束规划', '编写代码', '代码实现', 'prove', 'write code'))
    fallback_on_local_error: bool = True
    dot_max_subtasks: int = 4
    dot_context_max_bytes: int = 9000
    dot_planner_max_tokens: int = 512

    def __post_init__(self):
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError('max_tokens 必须为正整数')
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError('temperature 必须为 0 到 2 的有限数值')
        if type(self.long_prompt_chars) is not int or self.long_prompt_chars < 1:
            raise ValueError('long_prompt_chars 必须为正整数')
        if not all(isinstance(k, str) and k.strip() for k in self.complex_keywords):
            raise ValueError('复杂任务关键词不可为空')
        if type(self.dot_max_subtasks) is not int or not 1 <= self.dot_max_subtasks <= 4:
            raise ValueError('dot_max_subtasks 必须为 1 到 4 的整数')
        for name in ('dot_context_max_bytes', 'dot_planner_max_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须为正整数')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('配置必须为 JSON 对象')
        data['local'] = Endpoint(**data['local'])
        data['cloud'] = Endpoint(**data['cloud']) if data.get('cloud') else None
        if 'complex_keywords' in data:
            data['complex_keywords'] = tuple(data['complex_keywords'])
        return cls(**data)

    def public_metadata(self):
        # 不复制 endpoint、环境变量值或原始配置到实验报告。
        return {'local_model': self.local.model,
                'cloud_model': self.cloud.model if self.cloud else None,
                'max_tokens': self.max_tokens, 'temperature': self.temperature,
                'system_prompt': self.system_prompt,
                'long_prompt_chars': self.long_prompt_chars,
                'complex_keywords': list(self.complex_keywords),
                'fallback_on_local_error': self.fallback_on_local_error,
                'dot_max_subtasks': self.dot_max_subtasks,
                'dot_context_max_bytes': self.dot_context_max_bytes,
                'dot_planner_max_tokens': self.dot_planner_max_tokens,
                'cloud_thinking': self.cloud.thinking if self.cloud else None}
