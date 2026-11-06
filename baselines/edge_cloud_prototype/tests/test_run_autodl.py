import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RunAutoDLTests(unittest.TestCase):
    def loader(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_autodl.py'
        spec = importlib.util.spec_from_file_location('run_autodl_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_empty_secret_rejected_without_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'secret'
            path.write_text('  \n', encoding='utf-8')
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, '为空'):
                self.loader().load_key(path)

    def test_key_is_loaded_into_environment_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'secret'
            path.write_text('unit-test-secret\n', encoding='utf-8')
            path.chmod(0o600)
            with patch.dict(os.environ, {}, clear=True):
                self.loader().load_key(path)
                self.assertEqual(os.environ['EDGE_CLOUD_API_KEY'], 'unit-test-secret')

    @unittest.skipUnless(os.name == 'posix', 'POSIX file permissions')
    def test_public_readable_secret_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'secret'
            path.write_text('unit-test-secret', encoding='utf-8')
            path.chmod(0o644)
            with self.assertRaisesRegex(PermissionError, '权限'):
                self.loader().load_key(path)
