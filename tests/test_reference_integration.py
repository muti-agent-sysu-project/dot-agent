"""Offline integration checks: runnable reference and failure propagation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'eval/run_reference.py'


class ReferenceIntegrationTests(unittest.TestCase):
    def test_offline_four_modes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='.reference-test-', dir=ROOT) as tmp:
            output = Path(tmp) / 'report'
            command = [sys.executable, str(ENTRY), 'demo', '--dataset',
                       'data/dot_smoke.jsonl', '--modes', 'local', 'cloud',
                       'hybrid', 'dot', '--output', str(output)]
            first = subprocess.run(command, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            rows = [json.loads(line) for line in
                    (output / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 24)
            self.assertEqual({r['mode'] for r in rows}, {'local', 'cloud', 'hybrid', 'dot'})
            self.assertTrue(all(r['simulated'] and r['cloud_tokens'] is None for r in rows))
            before = (output / 'requests.jsonl').read_bytes()
            second = subprocess.run(command, capture_output=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual((output / 'requests.jsonl').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
