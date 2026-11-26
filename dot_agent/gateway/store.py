"""SQLite 指标落库（吸收自团队 cloud-edge-agent 的计量设计）。

双表结构：
- calls：每次模型调用一行（含失败尝试）。请求级指标不能落在这里算分位数
  ——一次请求拆成 5 次调用会被算成 5 倍样本，P50/P95 直接失真。
- tasks：一次用户请求一行，C_time（请求级量纲）与该请求的云端 token 汇总。

口径要点：
- usage 缺失时 token 记 NULL 而非 0（诚实边界：不虚报也不假报），
  汇总侧用 known_subtotal + unknown_usage_calls 分列（吸收 edge-cloud-prototype）。
- 每次重试尝试单独落一行：申请书口径"含重试的每次尝试"。
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  call_id           TEXT PRIMARY KEY,
  run_id            TEXT,
  task_id           TEXT,
  ts                TEXT,
  caller            TEXT,      -- classify/decompose/dep_judge/signal/execute/verify/retry/summarize
  tier              TEXT,      -- MD/MC（实际执行档位，降级后为准）
  requested_tier    TEXT,      -- 调用方要求档位；与 tier 不同 = 云端降级到本地
  model             TEXT,
  prompt_tokens     INTEGER,   -- NULL = 后端未回报 usage（不可信，勿并入口径）
  completion_tokens INTEGER,
  latency_ms        REAL,
  cloud_tokens      INTEGER DEFAULT 0,  -- 仅 tier=MC 时非零
  usage_missing     INTEGER DEFAULT 0,
  is_cache_hit      INTEGER DEFAULT 0,
  error             TEXT
);
CREATE TABLE IF NOT EXISTS generation_details (
  call_id TEXT PRIMARY KEY REFERENCES calls(call_id),
  finish_reason TEXT,
  reasoning_chars INTEGER,
  reasoning_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);

CREATE TABLE IF NOT EXISTS tasks (
  task_id        TEXT PRIMARY KEY,
  run_id         TEXT,
  ts             TEXT,
  mode           TEXT,        -- auto/local_only/cloud_only/rule/signal
  question       TEXT,
  answer         TEXT,
  label          TEXT,
  path           TEXT,
  n_subtasks     INTEGER DEFAULT 0,
  n_local        INTEGER DEFAULT 0,
  n_cloud        INTEGER DEFAULT 0,
  total_ms       REAL,        -- 请求级 C_time
  plan_ms        REAL,        -- 分类+分解+依赖判断（规划开销，DoT 未计量的部分）
  exec_ms        REAL,
  cloud_tokens   INTEGER,
  unknown_usage_calls INTEGER DEFAULT 0,
  error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
"""

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "metrics.db"

_initialized = False


def _ensure(db_path: Path | None) -> Path:
    """惰性建表：首次写入前保证 schema 存在（幂等）。"""
    global _initialized
    path = db_path or DB_PATH
    if not _initialized or db_path is not None:
        with _LOCK, _connect(path) as conn:
            conn.executescript(SCHEMA)
        _initialized = True
    return path


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init(db_path: Path | None = None) -> None:
    with _LOCK, _connect(db_path or DB_PATH) as conn:
        conn.executescript(SCHEMA)


def record_call(
    *,
    caller: str,
    tier: str,
    requested_tier: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_s: float,
    run_id: str = "default",
    task_id: str = "",
    usage_missing: bool = False,
    is_cache_hit: bool = False,
    error: str | None = None,
    db_path: Path | None = None,
    finish_reason: str | None = None,
    reasoning_chars: int | None = None,
    reasoning_tokens: int | None = None,
) -> str:
    call_id = uuid.uuid4().hex
    known = not usage_missing and prompt_tokens is not None
    cloud_tokens = (
        (prompt_tokens or 0) + (completion_tokens or 0)
        if tier == "MC" and known
        else 0
    )
    row = (
        call_id, run_id, task_id,
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        caller, tier, requested_tier, model,
        None if usage_missing else prompt_tokens,
        None if usage_missing else completion_tokens,
        latency_s * 1000, cloud_tokens, int(usage_missing), int(is_cache_hit), error,
    )
    path = _ensure(db_path)
    with _LOCK, _connect(path) as conn:
        conn.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row
        )
        if any(v is not None for v in (finish_reason, reasoning_chars, reasoning_tokens)):
            conn.execute("INSERT INTO generation_details VALUES (?,?,?,?)",
                         (call_id, finish_reason, reasoning_chars, reasoning_tokens))
    return call_id


def record_task(
    *,
    run_id: str = "default",
    mode: str = "auto",
    question: str,
    answer: str,
    label: str,
    path: str,
    n_subtasks: int,
    n_local: int,
    n_cloud: int,
    total_ms: float,
    plan_ms: float,
    exec_ms: float,
    cloud_tokens: int,
    unknown_usage_calls: int = 0,
    error: str | None = None,
    db_path: Path | None = None,
) -> str:
    task_id = uuid.uuid4().hex[:12]
    row = (
        task_id, run_id,
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        mode, question, answer, label, path, n_subtasks, n_local, n_cloud,
        total_ms, plan_ms, exec_ms, cloud_tokens, unknown_usage_calls, error,
    )
    path = _ensure(db_path)
    with _LOCK, _connect(path) as conn:
        conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    return task_id
