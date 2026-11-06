import hashlib
import json
import math
import statistics
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .system import MODES, STAGES


def load_tasks(path):
    tasks = []
    seen = set()
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if (not isinstance(task, dict) or not isinstance(task.get('id'), str)
                or not task['id'].strip() or task['id'] in seen
                or not isinstance(task.get('prompt'), str) or not task['prompt'].strip()
                or not isinstance(task.get('answers'), list) or not task['answers']
                or not all(isinstance(a, str) and a.strip() for a in task['answers'])):
            raise ValueError('题集行必须包含唯一 id、非空 prompt 和非空字符串 answers 列表')
        seen.add(task['id'])
        tasks.append(task)
    if not tasks:
        raise ValueError('题集不可为空')
    return tasks


def normalize(text):
    return unicodedata.normalize('NFKC', text).strip().rstrip('。.!！').strip().casefold()


def score(answer, task):
    # 固定 exact match，只宽容 Unicode、首尾空白、句末标点。
    return normalize(answer) in {normalize(value) for value in task['answers']}


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(rows):
    summaries = {}
    for mode in MODES:
        group = [row for row in rows if row['mode'] == mode]
        if not group:
            continue
        timings = [row['elapsed_ms'] for row in group]
        unknown = sum(row['unknown_usage_calls'] for row in group)
        known_tokens = sum(call['total_tokens'] for row in group for call in row['calls']
                           if call['role'] == 'cloud' and call['attempted']
                           and call['total_tokens'] is not None)
        stage_metrics = {}
        for stage in STAGES:
            all_stage_calls = [call for row in group for call in row['calls']
                               if call['stage'] == stage]
            stage_calls = [call for call in all_stage_calls
                           if call['role'] == 'cloud' and call['attempted']]
            local_calls = [call for call in all_stage_calls
                           if call['role'] == 'local' and call['attempted']]
            subtotal = sum(call['total_tokens'] for call in stage_calls
                           if call['total_tokens'] is not None)
            unknown_stage = sum(call['total_tokens'] is None for call in stage_calls)
            local_subtotal = sum(call['total_tokens'] for call in local_calls
                                 if call['total_tokens'] is not None)
            local_unknown = sum(call['total_tokens'] is None for call in local_calls)
            stage_metrics[stage] = {
                'cloud_tokens': None if unknown_stage or any(row.get('simulated') for row in group) else subtotal,
                'known_cloud_tokens_subtotal': subtotal,
                'unknown_usage_calls': unknown_stage,
                'cloud_attempts': len(stage_calls),
                'elapsed_ms': sum(call['elapsed_ms'] for call in stage_calls),
                'local_tokens': None if local_unknown else local_subtotal,
                'known_local_tokens_subtotal': local_subtotal,
                'local_unknown_usage_calls': local_unknown,
                'local_attempts': len(local_calls),
                'all_call_elapsed_ms': sum(call['elapsed_ms'] for call in all_stage_calls)}
        summaries[mode] = {
            'requests': len(group), 'successes': sum(row['success'] for row in group),
            'errors': sum(not row['success'] for row in group),
            'success_rate': statistics.mean(row['success'] for row in group),
            'accuracy': statistics.mean(bool(row.get('correct')) for row in group),
            'cloud_tokens': None if unknown or any(row['cloud_tokens'] is None for row in group) else known_tokens,
            'known_cloud_tokens_subtotal': known_tokens, 'unknown_usage_calls': unknown,
            'cloud_attempts': sum(call['role'] == 'cloud' and call['attempted']
                                  for row in group for call in row['calls']),
            'mean_ms': statistics.mean(timings), 'p50_ms': percentile(timings, 0.5),
            'p95_ms': percentile(timings, 0.95),
            'fallback_rate': statistics.mean(
                bool(row.get('upgrade_count', 0) or row.get('degradation_count', 0))
                for row in group),
            'upgrade_count': sum(row.get('upgrade_count', 0) for row in group),
            'degradation_count': sum(row.get('degradation_count', 0) for row in group),
            'stage_metrics': stage_metrics,
            'simulated': any(row.get('simulated', False) for row in group)}
    return summaries


def evaluate(system, tasks_path, modes, repeats, output):
    tasks_path = Path(tasks_path)
    tasks = load_tasks(tasks_path)
    if repeats < 1 or not modes or any(mode not in MODES for mode in modes):
        raise ValueError('需要有效模式和正数 repeats')
    if len(set(modes)) != len(modes):
        raise ValueError('评测模式不可重复')
    output = Path(output)
    # 完整保留每次实验，拒绝覆盖历史。
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    experiment_id = str(uuid.uuid4())
    with (output / 'requests.jsonl').open('x', encoding='utf-8') as stream:
        for repeat in range(repeats):
            for index, task in enumerate(tasks):
                shift = (index + repeat) % len(modes)
                order = modes[shift:] + modes[:shift]
                for mode in order:
                    row = system.ask(task['prompt'], mode)
                    row.update(task_id=task['id'], category=task.get('category', 'unspecified'),
                               repeat=repeat, experiment_id=experiment_id,
                               correct=row['success'] and score(row['answer'], task))
                    for call in row['calls']:
                        call.update(experiment_id=experiment_id, task_id=task['id'], repeat=repeat)
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'experiment_id': experiment_id,
        'simulated': any(row.get('simulated', False) for row in rows),
        'dataset_sha256': hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        'tasks': len(tasks), 'repeats': repeats,
        'scoring': 'normalized_exact_match', 'ordering': 'rotated_modes_per_task',
        'includes_warmup': False, 'cache_enabled': False,
        'config': system.config.public_metadata(),
        'results': summarize(rows),
        'limitations': ['评测器不执行预热，外部预热不计入本报告。',
                        '冒烟题集不代表标准基准。', '少量样本 P95 不能用于稳定性结论。',
                        '失败请求计入准确率分母和完整响应时间。',
                        '未知用量为 null，已知部分单独列出。',
                        '模型服务内部缓存/供应商缓存未控制，需正式实验前检查。']}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
