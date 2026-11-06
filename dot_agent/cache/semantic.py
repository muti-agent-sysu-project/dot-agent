"""语义缓存：确定性域内做"精确匹配 + 相似度匹配"两级命中。

相似度用字符 3-gram Jaccard（纯标准库，零依赖）；embedding 相似度留
set embedding_fn 接口，接 bge-small-zh 等编码器后自动走语义匹配，
未接入时词面相似度兜底——中文字面相近的查询词面相似度仍然可用。

防误命中三道闸（对应 vCache/GPTCache 文献结论：语义缓存收益高度依赖
查询分布与阈值）：
  1. 只对 deterministic 域开放（非确定性任务永远 miss）；
  2. 高阈值 SIM_THRESHOLD（0.92），低于精确命中的信任度；
  3. 容量上限 500 条 + TTL，防止陈旧答案长尾污染。
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Callable

MAX_ENTRIES = 500
SIM_THRESHOLD = 0.92


def _ngrams(text: str, n: int = 3) -> Counter:
    text = "".join(text.split())  # 去空白后按字符切 n-gram
    if len(text) < n:
        return Counter([text])
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _jaccard(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


class SemanticCache:
    def __init__(self, enabled: bool = True, ttl_s: float = 3600.0,
                 sim_threshold: float = SIM_THRESHOLD,
                 embedding_fn: Callable[[str], list[float]] | None = None) -> None:
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.sim_threshold = sim_threshold
        self.embedding_fn = embedding_fn  # 可选语义向量函数
        self._store: dict[str, tuple[str, float, list[float] | None]] = {}
        self.stats = {"hit": 0, "miss": 0, "sim_hit": 0, "reject_low_sim": 0}

    # -- 向量相似度（接入 embedding_fn 后启用） ------------------------- #
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def _best_match(self, query: str) -> tuple[str, float] | None:
        """在存量里找最相似条目：优先 embedding 余弦，未接入或失败用词面 Jaccard。"""
        if not self._store:
            return None
        if self.embedding_fn is not None:
            try:
                qvec = self.embedding_fn(query)
                scored = [(k, self._cosine(qvec, vec))
                          for k, (_, _, vec) in self._store.items() if vec]
                if scored:
                    return max(scored, key=lambda x: x[1])
            except Exception:
                pass  # embedding 失败退回词面
        qgrams = _ngrams(query)
        scored = [(k, _jaccard(qgrams, _ngrams(k))) for k in self._store]
        return max(scored, key=lambda x: x[1])

    def get(self, query: str, deterministic: bool) -> str | None:
        if not self.enabled or not deterministic:
            self.stats["miss"] += 1
            return None
        now = time.time()
        entry = self._store.get(query)
        if entry and now - entry[1] < self.ttl_s:
            self.stats["hit"] += 1
            return entry[0]
        match = self._best_match(query)
        if match:
            key, score = match
            hit_entry = self._store.get(key)
            if (hit_entry and now - hit_entry[1] < self.ttl_s
                    and score >= self.sim_threshold):
                self.stats["sim_hit"] += 1
                return hit_entry[0]
            self.stats["reject_low_sim"] += 1
        self.stats["miss"] += 1
        return None

    def put(self, query: str, answer: str, deterministic: bool) -> None:
        if not self.enabled or not deterministic:
            return
        if len(self._store) >= MAX_ENTRIES:
            # 淘汰最旧一条（FIFO 足够，LRU 收益不值一把额外结构）
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        vec = None
        if self.embedding_fn is not None:
            try:
                vec = self.embedding_fn(query)
            except Exception:
                vec = None
        self._store[query] = (answer, time.time(), vec)
