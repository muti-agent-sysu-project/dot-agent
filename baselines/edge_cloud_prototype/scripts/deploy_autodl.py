"""仅在已确认租用的 AutoDL Linux 实例执行；标准库，无 pip 安装。"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verified(path, artifact):
    return path.is_file() and path.stat().st_size == artifact['size'] and digest(path) == artifact['sha256']


def download(artifact, directory):
    path = directory / artifact['name']
    if verified(path, artifact):
        print('已验证，复用：', artifact['name'], flush=True)
        return path
    if path.exists():
        raise RuntimeError('已有文件校验不通过，请先人工检查：' + artifact['name'])
    partial = path.with_suffix(path.suffix + '.part')
    for url in artifact['urls']:
        try:
            print('下载：', artifact['name'], flush=True)
            with urlopen(url, timeout=60) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            if not verified(partial, artifact):
                raise RuntimeError('下载文件尺寸或 SHA256 不一致')
            partial.replace(path)
            return path
        except (OSError, URLError, RuntimeError):
            # 不打印可能包含签名 URL 的异常。
            print('该来源失败，尝试下一个已配置来源。', flush=True)
    raise RuntimeError('所有下载来源均失败；保留 part 文件供检查。')


def extract(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        # Python 3.10 兼容：显式检查路径及链接，拒绝设备文件。
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError('归档路径越界')
            if member.issym() or member.islnk():
                link = ((target.parent if member.issym() else destination) / member.linkname).resolve()
                if not link.is_relative_to(destination):
                    raise RuntimeError('归档链接越界')
            elif not (member.isfile() or member.isdir()):
                raise RuntimeError('归档包含特殊文件')
        if hasattr(tarfile, 'data_filter'):
            bundle.extractall(destination, members=members, filter='data')
        else:
            bundle.extractall(destination, members=members)


def preflight():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('只支持 AutoDL Linux x86_64，不在本机部署')
    executable = shutil.which('nvidia-smi')
    if not executable:
        raise RuntimeError('未发现 nvidia-smi；请确认实例以 GPU 模式开机')
    info = subprocess.check_output([executable, '--query-gpu=name,memory.total,driver_version',
                                    '--format=csv,noheader'], text=True)
    print('GPU：', info.strip())
    # CUDA 12.8 完整运行时选用原生兼容驱动，避免只靠低版本兼容模式。
    drivers = [line.rsplit(',', 1)[-1].strip() for line in info.strip().splitlines()]
    if any(int(version.split('.')[0]) < 570 for version in drivers):
        raise RuntimeError('此运行时要求驱动主版本 >=570，请换已核对配置，不改系统驱动')
    missing = [name for name in ('cmake', 'nvcc', 'g++', 'make') if not shutil.which(name)]
    if missing:
        raise RuntimeError('CUDA源码编译缺少工具：' + ', '.join(missing))
    cmake_version = subprocess.check_output(['cmake', '--version'], text=True)
    match = re.search(r'cmake version (\d+)\.(\d+)', cmake_version)
    if not match or tuple(map(int, match.groups())) < (3, 18):
        raise RuntimeError('CUDA源码编译需要 CMake >=3.18')
    cuda_version = subprocess.check_output(['nvcc', '--version'], text=True)
    match = re.search(r'release (\d+)\.(\d+)', cuda_version)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        raise RuntimeError('此部署路径需要 CUDA toolkit >=12.8，请检查实际镜像')
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('项目磁盘可用空间不足 10GiB')
    with socket.socket() as check:
        check.bind(('127.0.0.1', 8000))
    return info


def build_runtime(archive, binaries, revision):
    sources = ROOT / 'runtime/source'
    extract(archive, sources)
    source = sources / ('llama.cpp-' + revision)
    if not (source / 'CMakeLists.txt').is_file():
        raise RuntimeError('源码目录与固定commit不一致')
    capabilities = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True)
    values = [value.strip() for value in capabilities.splitlines() if value.strip()]
    if not values or not all(re.fullmatch(r'\d+\.\d+', value) for value in values):
        raise RuntimeError('无法读取GPU计算能力，不启用CPU替代')
    architectures = ';'.join(sorted({value.replace('.', '') for value in values}))
    binaries.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / 'runtime/tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['TMPDIR'] = str(temporary)
    configure = ['cmake', '-S', str(source), '-B', str(binaries),
                 '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON',
                 '-DCMAKE_CUDA_ARCHITECTURES=' + architectures,
                 '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                 '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON']
    build = ['cmake', '--build', str(binaries), '--target', 'llama-server',
             '--parallel', str(min(os.cpu_count() or 1, 6))]
    print('在实例内编译CUDA服务，进度见 runtime/build.log。', flush=True)
    with (ROOT / 'runtime/build.log').open('a', encoding='utf-8') as log:
        for command in (configure, build):
            subprocess.run(command, check=True, cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='只检查环境，不下载')
    args = parser.parse_args()
    gpu = preflight()
    if args.preflight:
        print('环境预检查通过。')
        return
    manifest = json.loads((ROOT / 'scripts/deployment-manifest.json').read_text())
    downloads = ROOT / 'runtime/downloads'
    binaries = ROOT / 'runtime/llama'
    downloads.mkdir(parents=True, exist_ok=True)
    for artifact in manifest['artifacts']:
        path = download(artifact, downloads)
        if artifact['kind'] == 'source':
            build_runtime(path, binaries, manifest['runtime_revision'])
    servers = list(binaries.rglob('llama-server'))
    if len(servers) != 1:
        raise RuntimeError('未找到唯一 llama-server')
    executable = servers[0]
    executable.chmod(executable.stat().st_mode | 0o100)
    env = os.environ.copy()
    dirs = sorted({str(path.parent) for path in binaries.rglob('*.so*')})
    env['LD_LIBRARY_PATH'] = ':'.join(dirs + [env.get('LD_LIBRARY_PATH', '')])
    # 先验证系统动态库兼容，不启动或静默退回 CPU。
    version = subprocess.check_output([str(executable), '--version'], env=env,
                                      stderr=subprocess.STDOUT, text=True)
    model = downloads / 'Qwen3-4B-Q4_K_M.gguf'
    state = {'server': str(executable.relative_to(ROOT)), 'model': str(model.relative_to(ROOT)),
             'library_dirs': [str(Path(p).relative_to(ROOT)) for p in dirs],
             'gpu': gpu.strip(), 'runtime_version': version.strip(),
             'model_revision': manifest['model_revision'],
             'runtime_revision': manifest['runtime_revision'], 'runtime_build': 'source_cuda'}
    (ROOT / 'runtime/deployment.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print('下载和运行时检查完成。启动：python scripts/start_autodl.py')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        # 本脚本只处理公开下载/环境检查；仍不输出远程异常正文。
        detail = str(error) if isinstance(error, RuntimeError) else '检查 GPU、磁盘或运行时动态库兼容性。'
        print('部署未完成：', type(error).__name__, detail)
        raise SystemExit(2)
