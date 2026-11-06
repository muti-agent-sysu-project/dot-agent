"""实验记录库：每次请求的完整结果落盘为 JSON，支持检索、统计、导出。

设计：
- 存储 data/runs/<日期>/<id>.json，一天一目录，id 为时间戳+随机后缀，
  文件名即 id，人工可直接翻目录。
- 与 SQLite 账本（gateway/store.py，数字口径）互补：这里是全量结果，
  回看实验细节用。
- 全部纯函数 + 显式 store 对象，HTTP 层只做参数校验（可脱离服务器测试）。
- 并发用一把锁（实验台单用户场景，够用且不会错）。
"""
from __future__ import annotations

import json
import random
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "runs"

_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _new_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%d-%H%M%S") + "-" + f"{random.randint(0, 0xFFFF):04x}"


def make_record(result_dict: dict, record_id: str | None = None,
                ts: str | None = None) -> dict:
    """从 /api/run 的响应构造落盘记录：全量保留，另加索引字段。"""
    metrics = result_dict.get("metrics") or {}
    return {
        "id": record_id or _new_id(_now()),
        "ts": ts or _now().isoformat(timespec="seconds"),
        "query": result_dict.get("query", ""),
        "answer": result_dict.get("answer", ""),
        "label": result_dict.get("label", ""),
        "path": result_dict.get("path", ""),
        "success": result_dict.get("success", False),
        "error": result_dict.get("error"),
        "trace": result_dict.get("trace", []),
        "subtasks": result_dict.get("subtasks", []),
        "provided_sources": result_dict.get("provided_sources", []),
        "citations": result_dict.get("citations", []),
        "metrics": metrics,
        # 索引字段（筛选用，从 metrics 提出来避免读取时全量解析）
        "c_time_s": metrics.get("C_time_s"),
        "cloud_tokens": metrics.get("C_API_total", 0),
        "cloud_calls": metrics.get("cloud_calls", 0),
        "local_calls": metrics.get("local_calls", 0),
        "simulated": metrics.get("simulated", False),
    }


class RecordStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # -- 写 ------------------------------------------------------------- #
    def save(self, record: dict) -> str:
        day = record["id"][:8]
        target = self._day_dir(day) / f"{record['id']}.json"
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return record["id"]

    # -- 读 ------------------------------------------------------------- #
    def _day_dir(self, day: str) -> Path:
        return self.root / day

    def list(self, *, limit: int = 50, label: str | None = None,
             path: str | None = None, contains: str | None = None,
             success: bool | None = None) -> list[dict]:
        """按时间倒序列出摘要（不含 trace/subtasks，列表页够用）。"""
        rows: list[dict] = []
        days = sorted((d for d in self.root.iterdir() if d.is_dir()),
                      reverse=True)
        with _LOCK:
            for day in days:
                for f in sorted(day.glob("*.json"), reverse=True):
                    try:
                        rec = json.loads(f.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        continue
                    if label and rec.get("label") != label:
                        continue
                    if path and rec.get("path") != path:
                        continue
                    if success is not None and bool(rec.get("success")) is not success:
                        continue
                    if contains and contains not in (rec.get("query") or ""):
                        continue
                    # .get 容错：单条旧格式/损坏记录缺字段不应拖垮整个列表查询
                    rows.append({k: rec.get(k) for k in (
                        "id", "ts", "query", "label", "path", "success", "error",
                        "c_time_s", "cloud_tokens", "cloud_calls", "local_calls",
                        "simulated")})
                    if len(rows) >= limit:
                        return rows
        return rows

    def get(self, record_id: str) -> dict | None:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", record_id):
            return None
        target = self._day_dir(record_id[:8]) / f"{record_id}.json"
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    # -- 统计 ------------------------------------------------------------ #
    def stats(self, *, label: str | None = None) -> dict:
        rows = self.list(limit=10000, label=label)
        times = sorted(r["c_time_s"] for r in rows if r["c_time_s"] is not None)
        n = len(rows)
        ok = sum(1 for r in rows if r["success"])
        cloud_tokens = [r["cloud_tokens"] for r in rows if r["cloud_tokens"]]
        label_counts: dict[str, int] = {}
        path_counts: dict[str, int] = {}
        for r in rows:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
            path_counts[r["path"]] = path_counts.get(r["path"], 0) + 1

        def pct(values: list[float], q: float) -> float | None:
            if not values:
                return None
            idx = min(int(q * len(values)), len(values) - 1)
            return round(values[idx], 4)

        return {
            "total": n,
            "success": ok,
            "success_rate": round(ok / n, 4) if n else None,
            "c_time_avg": round(sum(times) / len(times), 4) if times else None,
            "c_time_p50": pct(times, 0.5),
            "c_time_p95": pct(times, 0.95),
            "cloud_tokens_avg": round(sum(cloud_tokens) / len(cloud_tokens), 1) if cloud_tokens else None,
            "cloud_tokens_total": sum(cloud_tokens),
            "simulated_runs": sum(1 for r in rows if r.get("simulated")),
            "label_counts": label_counts,
            "path_counts": path_counts,
        }

    # -- 导出 ------------------------------------------------------------ #
    def _iter_all(self, exclude_simulated: bool = False):
        for day in sorted(d for d in self.root.iterdir() if d.is_dir()):
            for f in sorted(day.glob("*.json")):
                try:
                    record = json.loads(f.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                if exclude_simulated and record.get("simulated"):
                    continue
                yield record

    def export_jsonl(self, *, exclude_simulated: bool = False) -> str:
        """全量导出为 JSONL（一行一条，论文评测的标准格式）。

        落盘文件带缩进（人工可读），这里解析后重新紧凑序列化，保证一行一条。
        exclude_simulated: 排除 stub 模拟记录，防止假数据混进论文评测。
        """
        lines = [json.dumps(record, ensure_ascii=False)
                 for record in self._iter_all(exclude_simulated)]
        return "\n".join(lines) + ("\n" if lines else "")

    def export_csv(self, *, exclude_simulated: bool = False) -> str:
        """摘要字段导出 CSV（表格用）。

        以 = + - @ 开头的单元格值会被表格软件当公式执行（CSV 注入），
        统一加 `'` 前缀转义。
        """
        import csv
        import io

        def safe(value) -> str:
            text = str(value)
            return "'" + text if text[:1] in ("=", "+", "-", "@") else text

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "ts", "label", "path", "success", "error",
                         "c_time_s", "cloud_tokens", "cloud_calls",
                         "local_calls", "simulated", "query"])
        for record in self._iter_all(exclude_simulated):
            r = record  # 摘要字段直接从全量记录取；.get 容错坏记录
            writer.writerow([safe(r.get("id") or ""), safe(r.get("ts") or ""),
                             safe(r.get("label") or ""), safe(r.get("path") or ""),
                             r.get("success"), safe(r.get("error") or ""),
                             r.get("c_time_s") or "", r.get("cloud_tokens"),
                             r.get("cloud_calls"), r.get("local_calls"),
                             r.get("simulated"),
                             safe((r.get("query") or "").replace("\n", " ")[:200])])
        return buf.getvalue()
