"""实验记录库测试：落盘、检索、统计、导出（不依赖 HTTP 与网络）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dot_agent.records import RecordStore, make_record


def _sample(i: int, **kw) -> dict:
    base = {
        "query": f"测试问题 {i}",
        "answer": f"答案 {i}",
        "label": "REASONING" if i % 2 else "QA",
        "path": "pipeline" if i % 3 else "fast_path",
        "success": kw.get("success", i % 5 != 0),
        "error": None if i % 5 else "model_connection_error",
        "trace": ["step"],
        "subtasks": [{"id": "s1", "assigned": "MD"}],
        "provided_sources": [], "citations": [],
        "metrics": {"C_time_s": 0.1 * i + 0.05, "C_API_total": 100 + i,
                    "cloud_calls": i % 2, "local_calls": 3,
                    "simulated": True,
                    "privacy": {"enabled": True, "leak_count": 0,
                                "entities_masked": 2, "mode_counts": {"cloud_judge": 1}}},
    }
    base.update({k: v for k, v in kw.items() if k != "success"})
    if "simulated" in kw:  # simulated 在 metrics 里，顶层覆盖不影响 make_record
        base["metrics"]["simulated"] = kw["simulated"]
    return make_record(base)


def test_make_record_indexes_metrics():
    rec = _sample(1)
    assert abs(rec["c_time_s"] - 0.15) < 1e-9
    assert rec["cloud_tokens"] == 101
    assert rec["id"].count("-") == 2  # 日期-时间-后缀


def test_save_get_list_roundtrip(tmp_path):
    store = RecordStore(root=tmp_path / "runs")
    ids = [store.save(_sample(i)) for i in range(6)]
    assert len(ids) == 6
    got = store.get(ids[0])
    assert got["id"] == ids[0]
    assert got["subtasks"] == [{"id": "s1", "assigned": "MD"}]
    rows = store.list()
    assert len(rows) == 6
    assert "trace" not in rows[0]  # 列表页是摘要
    assert all(r["c_time_s"] is not None for r in rows)


def test_list_filters(tmp_path):
    store = RecordStore(root=tmp_path / "runs")
    for i in range(6):
        store.save(_sample(i))
    assert len(store.list(label="QA")) == 3
    assert len(store.list(path="pipeline")) == 4
    assert len(store.list(success=False)) == 2
    assert len(store.list(contains="问题 3")) == 1
    assert store.list(limit=2).__len__() == 2


def test_get_rejects_bad_id(tmp_path):
    store = RecordStore(root=tmp_path / "runs")
    assert store.get("../escape") is None
    assert store.get("not-an-id") is None


def test_stats(tmp_path):
    store = RecordStore(root=tmp_path / "runs")
    for i in range(10):
        store.save(_sample(i))
    s = store.stats()
    assert s["total"] == 10
    assert s["success"] == 8
    assert s["success_rate"] == 0.8
    assert s["c_time_avg"] is not None and s["c_time_p50"] <= s["c_time_p95"]
    assert s["cloud_tokens_total"] == sum(100 + i for i in range(10))
    assert s["label_counts"] == {"QA": 5, "REASONING": 5}
    # 按 label 过滤
    s_qa = store.stats(label="QA")
    assert s_qa["total"] == 5


def test_export_jsonl_csv(tmp_path):
    store = RecordStore(root=tmp_path / "runs")
    for i in range(3):
        store.save(_sample(i))
    jsonl = store.export_jsonl()
    lines = [json.loads(l) for l in jsonl.strip().splitlines()]
    assert len(lines) == 3 and lines[0]["query"].startswith("测试问题")
    csv_text = store.export_csv()
    header = csv_text.splitlines()[0]
    assert "c_time_s" in header and "cloud_tokens" in header
    assert len(csv_text.splitlines()) == 4  # 表头 + 3 行


def test_csv_injection_escaped(tmp_path):
    """以 = 开头的查询导出后不得被表格软件当公式执行。"""
    store = RecordStore(root=tmp_path / "runs")
    rec = make_record({"query": "=cmd|' /C calc'!A0", "answer": "x", "label": "QA",
                       "path": "fast_path", "success": True, "metrics": {"C_API_total": 5}})
    store.save(rec)
    line = [l for l in store.export_csv().splitlines() if "cmd" in l][0]
    assert "'=cmd" in line  # 公式前缀已被转义，表格软件不会再执行


def test_export_exclude_simulated(tmp_path):
    """导出必须能排除 stub 模拟记录，防止假数据混进论文评测。"""
    store = RecordStore(root=tmp_path / "runs")
    store.save(_sample(1))                      # simulated=True（_sample 默认）
    store.save(_sample(2, simulated=False))     # 真实
    assert len(store.export_jsonl().strip().splitlines()) == 2
    real = store.export_jsonl(exclude_simulated=True).strip().splitlines()
    assert len(real) == 1 and json.loads(real[0])["simulated"] is False
    assert len(store.export_csv(exclude_simulated=True).splitlines()) == 2  # 表头+1


def test_malformed_record_does_not_break_queries(tmp_path):
    """回归：单条缺字段/坏记录不得让 list/stats/export 整体 KeyError 崩溃。"""
    store = RecordStore(root=tmp_path / "runs")
    store.save(_sample(1))
    # 手工写一条旧格式坏记录：缺 c_time_s / label 等索引字段
    day = store.root / sorted(d.name for d in store.root.iterdir() if d.is_dir())[0]
    (day / "19990101-000000-ffff.json").write_text('{"id": "x"}', encoding="utf-8")
    rows = store.list()
    assert any(r["id"] == "x" for r in rows)          # 坏记录可见但不出错
    assert any(r["id"] != "x" and r["label"] for r in rows)  # 好记录不受影响
    stats = store.stats()
    assert stats["total"] == 2
    csv_out = store.export_csv()                  # 导出也不崩
    assert "x," in csv_out and "测试问题 1" in csv_out
