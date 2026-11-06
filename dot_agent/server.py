"""演示服务端：纯标准库（http.server），零额外依赖。

端点：
  GET  /               前端页面
  GET  /api/runtime    网关运行时信息（stub/真实模型）
  GET  /api/history    本次进程内的历史请求（内存，重启即清）
  GET  /api/runs       实验记录列表（落盘，支持筛选）：?limit=&label=&path=&q=&success=
  GET  /api/runs/<id>  单条实验记录全文
  GET  /api/stats      记录汇总统计：?label=
  GET  /api/export     导出：?format=jsonl|csv
  POST /api/run        执行一次完整流水线 {query: string}
  POST /api/batch      批量执行 {queries: [string...]}（串行，上限 50）

模型选择由 gateway.factory 按环境变量决定：
未设置 → 桩客户端（前端零模型可完整演示路由流程）
OLLAMA_HOST + CLOUD_API_KEY → 真实端云协同

运行: python -m dot_agent.server  (默认 0.0.0.0:8765)
"""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .gateway.factory import make_clients, warmup_real
from .pipeline import AgentPipeline
from .orchestrator.evidence import validate_sources
from .records import RecordStore, make_record

WEB_DIR = Path(__file__).resolve().parent / "web"
HISTORY: list[dict] = []
_HISTORY_LOCK = threading.Lock()
_MAX_HISTORY = 30
_RECORDS = RecordStore()
_BATCH_MAX = 50


def _one(params: dict, key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def _int_param(params: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(params.get(key, [default])[0])
    except (ValueError, TypeError):
        return default
    return max(lo, min(hi, value))


def _bool_param(params: dict, key: str) -> bool | None:
    raw = _one(params, key)
    if raw is None:
        return None
    return raw.lower() in ("1", "true", "yes")


def _serialize(result) -> dict:
    return {
        "query": result.query,
        "answer": result.answer,
        "label": result.label.value,
        "path": result.path,
        "trace": result.trace,
        "metrics": result.metrics,
        "subtasks": result.subtasks,
        "success": result.success,
        "error": result.error,
        "provided_sources": result.provided_sources,
        "citations": result.citations,
    }


class Handler(BaseHTTPRequestHandler):
    pipeline: AgentPipeline = None  # type: ignore[assignment]
    runtime: dict = {}
    run_lock = threading.Lock()  # pipeline 内部缓存/计量非线程安全，串行执行

    def log_message(self, fmt, *args):  # 精简日志
        print(f"[server] {fmt % args}")

    # ---------------- GET ----------------
    _STATIC_TYPES = {
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
        ".map": "application/json; charset=utf-8",
    }

    def _serve_asset(self, rel: str):
        """只放行 web 目录内的静态资源，挡住目录穿越。"""
        target = (WEB_DIR / rel).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        ctype = self._STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send_file(target, ctype)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path.startswith("/vendor/"):
            # 内置 KaTeX / marked，离线也能渲染公式与 Markdown
            self._serve_asset(self.path.split("?", 1)[0].lstrip("/"))
        elif path == "/app.js":
            self._send_file(WEB_DIR / "app.js", "text/javascript; charset=utf-8")
        elif path == "/style.css":
            self._send_file(WEB_DIR / "style.css", "text/css; charset=utf-8")
        elif path == "/api/runtime":
            self._send_json(self.runtime)
        elif path == "/api/history":
            with _HISTORY_LOCK:
                self._send_json({"history": HISTORY})
        elif path == "/api/runs":
            params = parse_qs(urlsplit(self.path).query)
            rows = _RECORDS.list(
                limit=_int_param(params, "limit", 50, 1, 500),
                label=_one(params, "label"),
                path=_one(params, "path"),
                contains=_one(params, "q"),
                success=_bool_param(params, "success"),
            )
            self._send_json({"runs": rows})
        elif path.startswith("/api/runs/"):
            record_id = path.removeprefix("/api/runs/")
            record = _RECORDS.get(record_id)
            if record is None:
                self._send_json({"error": "record_not_found"}, status=404)
            else:
                self._send_json(record)
        elif path == "/api/stats":
            params = parse_qs(urlsplit(self.path).query)
            self._send_json(_RECORDS.stats(label=_one(params, "label")))
        elif path == "/api/export":
            params = parse_qs(urlsplit(self.path).query)
            fmt = _one(params, "format") or "jsonl"
            exclude_sim = _bool_param(params, "exclude_simulated") or False
            if fmt == "csv":
                payload = _RECORDS.export_csv(exclude_simulated=exclude_sim)
                suffix = "runs.csv" if not exclude_sim else "runs-real.csv"
                self._send_text(payload, "text/csv; charset=utf-8", suffix)
            elif fmt == "jsonl":
                payload = _RECORDS.export_jsonl(exclude_simulated=exclude_sim)
                suffix = "runs.jsonl" if not exclude_sim else "runs-real.jsonl"
                self._send_text(payload, "application/x-ndjson; charset=utf-8",
                                suffix)
            else:
                self._send_json({"error": "invalid_format"}, status=400)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    # ---------------- POST ----------------
    def do_POST(self) -> None:
        if self.path == "/api/batch":
            self._handle_batch()
            return
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "invalid_content_length"}, status=400)
            return
        if not 0 < length <= 1024 * 1024:
            self._send_json({"error": "request_body_size_invalid"}, status=413)
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            self._send_json({"error": "invalid_json"}, status=400)
            return
        try:
            if not isinstance(body, dict) or not isinstance(body.get("query"), str):
                raise ValueError("invalid_query")
            query = body["query"].strip()
            if not query:
                raise ValueError("invalid_query")
            selector = body.get("selector", "bm25")
            if selector not in ("bm25", "local"):
                raise ValueError("invalid_selector")
            options = {}
            if "sources" in body:
                options = {"sources": validate_sources(body["sources"]), "selector": selector}
            elif selector != "bm25":
                raise ValueError("selector_requires_sources")
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return

        with self.run_lock:  # asyncio.run 每请求独立事件循环
            try:
                self.pipeline.reset_meter()  # 逐请求计量，避免跨请求累积
                result = asyncio.run(self.pipeline.run(query, **options))
            except Exception as e:  # 演示服务不崩，错误回传前端并留堆栈
                traceback.print_exc()
                self._send_json(
                    {"error": "pipeline_failed", "detail": type(e).__name__,
                     "success": False}, status=500,
                )
                return
        data = _serialize(result)
        with _HISTORY_LOCK:
            HISTORY.insert(0, data)
            del HISTORY[_MAX_HISTORY:]
        record = make_record(data)
        try:
            _RECORDS.save(record)
        except OSError as e:
            print(f"[server] 实验记录落盘失败: {type(e).__name__}")
        data["record_id"] = record["id"]
        self._send_json(data, status=200 if result.success else 502)

    def _handle_batch(self) -> None:
        """批量执行：串行跑完返回摘要列表。上限 50 条，防误操作拖垮服务。"""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "invalid_content_length"}, status=400)
            return
        if not 0 < length <= 1024 * 1024:
            self._send_json({"error": "request_body_size_invalid"}, status=413)
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            self._send_json({"error": "invalid_json"}, status=400)
            return
        queries = body.get("queries") if isinstance(body, dict) else None
        if (not isinstance(queries, list)
                or not queries
                or any(not isinstance(q, str) or not q.strip() for q in queries)):
            self._send_json({"error": "invalid_queries"}, status=400)
            return
        if len(queries) > _BATCH_MAX:
            self._send_json({"error": "batch_too_large", "limit": _BATCH_MAX},
                            status=400)
            return
        results = []
        with self.run_lock:
            for query in queries:
                try:
                    self.pipeline.reset_meter()
                    result = asyncio.run(self.pipeline.run(query.strip()))
                    data = _serialize(result)
                    record = make_record(data)
                    try:
                        _RECORDS.save(record)
                    except OSError:
                        pass
                    results.append({
                        "record_id": record["id"], "query": query.strip()[:200],
                        "label": data["label"], "path": data["path"],
                        "success": data["success"], "error": data["error"],
                        "c_time_s": data["metrics"].get("C_time_s"),
                        "cloud_tokens": data["metrics"].get("C_API_total", 0),
                        "cloud_calls": data["metrics"].get("cloud_calls", 0),
                        "local_calls": data["metrics"].get("local_calls", 0),
                    })
                except Exception as e:
                    traceback.print_exc()
                    results.append({"query": query.strip()[:200],
                                    "success": False,
                                    "error": "pipeline_failed:" + type(e).__name__})
        self._send_json({"results": results, "total": len(results)})

    # ---------------- helpers ----------------
    def _send_json(self, obj: dict, status: int = 200) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, text: str, ctype: str, filename: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path, ctype: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(host: str = "0.0.0.0", port: int = 8765, *, demo: bool = False) -> None:
    if demo:
        from .gateway.stub import StubModelClient
        from .gateway.demo_behavior import local_demo_behavior, cloud_demo_behavior
        from .types import ModelTier
        local = StubModelClient(ModelTier.MD, name="local-demo", behavior=local_demo_behavior)
        cloud = StubModelClient(ModelTier.MC, name="cloud-demo", behavior=cloud_demo_behavior)
        info = {"local": "stub", "cloud": "stub", "notes": ["显式离线演示，不读取真实模型配置"]}
    else:
        local, cloud, info = make_clients()
        asyncio.run(warmup_real(info, local, cloud))  # 真实模型预热，防冷启动污染首请求
    Handler.pipeline = AgentPipeline(local, cloud, run_id="server")
    Handler.runtime = info
    for note in info.get("notes", []):
        print(f"[gateway] {note}")
    print(f"运行时: local={info['local']}  cloud={info['cloud']}")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"演示面板: http://localhost:{port}  (Ctrl+C 退出)")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="显式使用演示模型，不读取真实后端配置")
    args = parser.parse_args()
    serve(args.host, args.port, demo=args.demo)
