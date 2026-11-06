"""用普通算术和穷举核对固定演示题，不调用模型。"""
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = 'dd9f8cb480e8ae8d7a3ba0ae90a8b0a16518a9fd3af1eca8f7f5902354479638'


def verify_frozen_dataset(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        raise ValueError('题集SHA256与运行前固定协议不符')
    return digest


def main():
    path = ROOT / 'data/dot_smoke.jsonl'
    digest = verify_frozen_dataset(path)
    tasks = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    total = 12 * 8 + 5 * 14
    water_boxes = (48 * 2 + 6 - 1) // 6
    plans = {'A': 80 + max(18 - 10, 0) * 8,
             'B': 110 + max(18 - 16, 0) * 6}
    books = {'A': (35, 46), 'B': (42, 49), 'C': (28, 45)}
    pairs = [pair for pair in itertools.combinations(books, 2)
             if sum(books[book][0] for book in pair) <= 80]
    selected = max(pairs, key=lambda pair: sum(books[book][1] for book in pair))
    expected = {'d01': str(17 + 25),
                'd02': str(total - (20 if total >= 150 else 0)),
                'd03': str(300 - water_boxes * 15),
                'd04': min((p for p in plans if plans[p] <= 130), key=plans.get),
                'd05': str(60 - sum((5, 8, 4))),
                'd06': ''.join(selected)}
    assert len(tasks) == len(expected)
    assert {task['id'] for task in tasks} == set(expected)
    for task in tasks:
        assert task['answers'] == [expected[task['id']]], task['id']
    print(json.dumps({'verified_answers': expected,
                      'method': 'integer arithmetic and exhaustive pair selection; no model',
                      'dataset_sha256': digest},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
