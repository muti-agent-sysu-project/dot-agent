"""前台启动 GPU 服务；Ctrl+C 停止，避免遗留后台进程。"""
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    if platform.system() != 'Linux':
        raise SystemExit('仅在 AutoDL Linux 运行')
    state = json.loads((ROOT / 'runtime/deployment.json').read_text(encoding='utf-8'))
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join(str(ROOT / p) for p in state['library_dirs']) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['LLAMA_CACHE'] = str(ROOT / 'runtime/cache')
    command = [str(ROOT / state['server']), '-m', str(ROOT / state['model']),
               '--host', '127.0.0.1', '--port', '8000', '--alias', 'qwen-local',
               '-ngl', '99', '-c', '4096', '-np', '1', '--jinja', '--reasoning-budget', '0',
               '--chat-template-kwargs', json.dumps({'enable_thinking': False}), '-lv', '4']
    print('服务只绑定 127.0.0.1:8000，当前终端保持打开。', flush=True)
    try:
        return subprocess.call(command, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
