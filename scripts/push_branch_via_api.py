#!/usr/bin/env python3
"""把当前工作区整棵树推成远端分支（github.com 直连被墙，走 api.github.com）。

与 scripts/push_via_api.py 的区别：那个脚本推 main 并强制覆盖历史；
本脚本以远端 main 为父提交、只新建/更新一条分支，main 不受影响，便于开 PR 审阅。

用法：python3 scripts/push_branch_via_api.py [--dry-run]
环境变量：PUSH_BRANCH 分支名，默认 feat/ui-trace-and-budgets-20261101
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "muti-agent-sysu-project/dot-agent"
BRANCH = os.environ.get("PUSH_BRANCH", "feat/ui-trace-and-budgets-20261101")
ROOT = Path(__file__).resolve().parents[1]
GH = "/tmp/gh_2.100.0_linux_amd64/bin/gh"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "data"}
SKIP_SUFFIX = {".pyc", ".log", ".tmp", ".db"}

MESSAGE = """修复请求 500 与输出预算过小，面板补齐中间过程与公式渲染

- signals：采样截断/空答返回 None 并按“缺失”处理，不再抛穿成 500
- pipeline：本地试答不可用时升级完整流水线；汇总失败退化为拼接子任务答案
- 预算：本地答案 768、采样 384、汇总 1536、云端答案 1024，均可用 DOT_BUDGET_* 覆盖
- 子任务答案保留 2000 字；计量新增逐次调用记录（仅元数据，不含提示词正文）
- 面板：新增阶段耗时、子任务明细、逐次调用表；答案支持 Markdown 与 KaTeX
- 内置 katex/marked 到 web/vendor，不依赖外网 CDN
- 回归：新增 tests/test_signal_resilience.py 四项
"""


def gh(method: str, path: str, payload: dict | None = None) -> dict:
    cmd = [GH, "api", "--method", method, f"repos/{REPO}/{path}"]
    if payload is not None:
        cmd += ["--input", "-"]
    proc = subprocess.run(cmd, input=json.dumps(payload).encode() if payload else None,
                          capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"{method} {path} 失败: {proc.stderr.decode()[:300]}")
    return json.loads(proc.stdout.decode() or "{}")


def collect() -> list[str]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                             capture_output=True).stdout.decode().split("\0")
    files = {p for p in tracked if p}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            rel = str(Path(base, name).relative_to(ROOT))
            if Path(rel).suffix in SKIP_SUFFIX:
                continue
            files.add(rel)
    return sorted(files)


def main() -> int:
    dry = "--dry-run" in sys.argv
    files = collect()
    blobs_only = [f for f in files if Path(f).suffix not in {".woff2"}]
    print(f"待推送 {len(files)} 个文件（其中字体 {len(files) - len(blobs_only)} 个）")
    if dry:
        for f in files[:12]:
            print("  ", f)
        print("  …")
        return 0

    main_sha = gh("GET", "git/ref/heads/main")["object"]["sha"]
    print(f"远端 main: {main_sha[:7]}（本分支以它为父提交，不改动 main）")

    items, failed = [], []
    for i, rel in enumerate(files, 1):
        try:
            data = (ROOT / rel).read_bytes()
        except OSError as exc:
            failed.append((rel, str(exc)))
            continue
        blob = gh("POST", "git/blobs",
                  {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        items.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if i % 20 == 0:
            print(f"  已上传 {i}/{len(files)}")

    print(f"blob 完成 {len(items)}，失败 {len(failed)}")
    tree = gh("POST", "git/trees", {"tree": items})
    commit = gh("POST", "git/commits",
                {"message": MESSAGE, "tree": tree["sha"], "parents": [main_sha]})
    print(f"提交 {commit['sha'][:7]}")

    try:
        gh("POST", "git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})
        print(f"已新建分支 {BRANCH}")
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise
        gh("PATCH", f"git/refs/heads/{BRANCH}", {"sha": commit["sha"], "force": True})
        print(f"已更新分支 {BRANCH}")

    print(f"https://github.com/{REPO}/tree/{BRANCH}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
