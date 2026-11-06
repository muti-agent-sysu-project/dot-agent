"""在 AutoDL 内加载私密文件并运行 CLI；密钥不进入参数或输出。"""
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_key(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('密钥必须为普通文件')
    if os.name == 'posix' and info.st_mode & 0o077:
        raise PermissionError('密钥文件权限必须仅允许当前账户访问，请设为600')
    if info.st_size > 8192:
        raise ValueError('密钥文件过大')
    key = path.read_text(encoding='utf-8').strip()
    if not key:
        raise ValueError('密钥文件为空')
    if any(char.isspace() for char in key):
        raise ValueError('密钥包含空白字符')
    os.environ['EDGE_CLOUD_API_KEY'] = key


def main():
    if sys.platform != 'linux':
        print('该入口仅在AutoDL Linux使用，密钥不得复制到本机。', file=sys.stderr)
        return 2
    try:
        load_key(ROOT / 'runtime/secrets/deepseek_api_key')
    except (OSError, ValueError):
        print('无法安全加载密钥：检查指定文件是否存在、非空且权限为600。', file=sys.stderr)
        return 2
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from edge_cloud.__main__ import main as cli_main
    sys.argv = [sys.argv[0], '--config', str(ROOT / 'config.deepseek.example.json'), *sys.argv[1:]]
    return cli_main()


if __name__ == '__main__':
    raise SystemExit(main())
