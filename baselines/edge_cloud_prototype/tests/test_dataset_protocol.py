import tempfile
import unittest
from pathlib import Path
from scripts.check_dot_dataset import verify_frozen_dataset


class DatasetProtocolTests(unittest.TestCase):
    def test_changed_prompt_with_original_answers_is_rejected(self):
        source = Path(__file__).resolve().parents[1] / 'data/dot_smoke.jsonl'
        verify_frozen_dataset(source)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / 'changed.jsonl'
            changed.write_bytes(source.read_bytes().replace(b'17', b'18', 1))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_frozen_dataset(changed)
