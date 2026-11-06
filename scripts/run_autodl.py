"""Load the existing remote secret and start the team mainline (Linux only)."""
import argparse
import asyncio
import os
from pathlib import Path
import runpy
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
SECRET = Path("/root/autodl-tmp/edge-cloud-prototype/prototype/runtime/secrets/deepseek_api_key")


def load_key(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 8192:
        raise ValueError("invalid secret file")
    if os.name == "posix" and info.st_mode & 0o077:
        raise ValueError("secret permissions must be 600")
    key = path.read_text(encoding="utf-8").strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("invalid secret")
    os.environ["CLOUD_API_KEY"] = key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "server", "eval"])
    args, rest = parser.parse_known_args()
    if sys.platform != "linux":
        parser.exit(2, "仅在 AutoDL Linux 执行；请勿复制密钥到本机。\n")
    try:
        load_key(SECRET)
    except (OSError, ValueError):
        parser.exit(2, "无法安全加载远程密钥；检查原文件和600权限。\n")
    defaults = {"LOCAL_BASE_URL": "http://127.0.0.1:8000/v1", "LOCAL_MODEL": "qwen-local",
                "CLOUD_BASE_URL": "https://api.deepseek.com", "CLOUD_MODEL": "deepseek-flash",
                "CLOUD_THINKING": "disabled"}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ["DOT_REQUIRE_REAL"] = "1"
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [sys.argv[0], *rest]
    if args.command == "check":
        if rest:
            parser.error("check does not accept extra arguments")
        from dot_agent.gateway.factory import make_clients, warmup_real
        local, cloud, info = make_clients()
        asyncio.run(warmup_real(info, local, cloud))
    elif args.command == "server":
        runpy.run_module("dot_agent.server", run_name="__main__")
    else:
        runpy.run_path(str(ROOT / "eval/runner.py"), run_name="__main__")


if __name__ == "__main__":
    main()
