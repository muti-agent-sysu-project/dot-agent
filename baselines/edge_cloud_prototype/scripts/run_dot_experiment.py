"""AutoDL 上运行已冻结的小规模协议，预检与对照分开记录。"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_autodl import load_key
from edge_cloud.config import Config
from edge_cloud.system import System
from edge_cloud.evaluation import evaluate
from edge_cloud.__main__ import doctor


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('仅在AutoDL运行')
    os.chdir(ROOT)
    load_key(ROOT / 'runtime/secrets/deepseek_api_key')
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'reports'):
        raise ValueError('输出必须位于项目reports目录')
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config.load(ROOT / 'config.deepseek.example.json')
    system = System(cfg)
    files = [ROOT / 'config.deepseek.example.json', ROOT / 'data/dot_smoke.jsonl',
             ROOT / 'EXPERIMENT_PROTOCOL.md', *sorted((ROOT / 'edge_cloud').glob('*.py')),
             ROOT / 'scripts/run_dot_experiment.py', ROOT / 'scripts/run_autodl.py',
             ROOT / 'scripts/check_dot_dataset.py']
    frozen = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    save(output / 'protocol.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'simulated': False,
        'files_sha256': frozen, 'config': cfg.public_metadata(),
        'modes': ['local', 'cloud', 'hybrid', 'dot'], 'tasks': 6, 'repeats': 1,
        'client_location': 'AutoDL instance', 'instance_id': '6fcb40b7f5-2694e599',
        'preflight_excluded_from_comparison': True,
        'application_answer_cache': False, 'service_caches_controlled': False})
    computed = subprocess.run([sys.executable, 'scripts/check_dot_dataset.py'],
                              capture_output=True, text=True, check=True)
    save(output / 'dataset-check.json', json.loads(computed.stdout))
    health = doctor(system)
    save(output / 'doctor.json', health)
    if any(row['status'] != 'ready' for row in health.values()):
        save(output / 'status.json', {'status': 'failed_preflight', 'reason': 'doctor'})
        return 2
    print('Both endpoints ready; starting one short cloud generation.', flush=True)
    probe = System(replace(cfg, max_tokens=32)).ask('6加7等于多少？只输出整数。', 'cloud')
    save(output / 'cloud-probe.json', probe)
    if (not probe['success'] or not probe['calls'] or
            any(call[field] is None for call in probe['calls']
                for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'))):
        save(output / 'status.json', {'status': 'failed_preflight', 'reason': 'cloud_generation_or_usage'})
        print('Cloud probe failed or usage missing; no comparison started.', flush=True)
        return 2
    warmup = system.ask('3加4等于多少？只输出整数。', 'local')
    save(output / 'local-warmup.json', warmup)
    if not warmup['success']:
        save(output / 'status.json', {'status': 'failed_preflight', 'reason': 'local_generation'})
        return 2
    print('Generation and usage verified; starting 6 tasks x 4 modes, one repeat.', flush=True)
    report = evaluate(system, ROOT / 'data/dot_smoke.jsonl',
                      ['local', 'cloud', 'hybrid', 'dot'], 1, output / 'comparison')
    assert frozen == {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in files}, '实验期间文件发生变化'
    errors = sum(row['errors'] for row in report['results'].values())
    save(output / 'status.json', {'status': 'complete' if not errors else 'complete_with_request_errors',
                                 'errors': errors, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                                 'preflight_cloud_tokens': probe['cloud_tokens']})
    print(json.dumps(report['results'], ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('Experiment stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(2)
