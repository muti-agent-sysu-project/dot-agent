import hashlib
import io
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.deploy_autodl import extract, verified, build_runtime, preflight


class DeploymentTests(unittest.TestCase):
    def test_artifact_digest_and_size_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            path.write_bytes(b'example')
            artifact = {'size': 7, 'sha256': hashlib.sha256(b'example').hexdigest()}
            self.assertTrue(verified(path, artifact))
            path.write_bytes(b'forged!')
            self.assertFalse(verified(path, artifact))

    def test_archive_rejects_escape_and_external_link(self):
        for name, link in (('../escape', None), ('link', '../../escape')):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'bundle.tar.gz'
                with tarfile.open(archive, 'w:gz') as bundle:
                    item = tarfile.TarInfo(name)
                    if link:
                        item.type = tarfile.SYMTYPE
                        item.linkname = link
                        bundle.addfile(item)
                    else:
                        item.size = 1
                        bundle.addfile(item, io.BytesIO(b'x'))
                with self.assertRaises(RuntimeError):
                    extract(archive, root / 'out')
                self.assertFalse((root / 'escape').exists())

    def test_archive_extracts_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bundle.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                item = tarfile.TarInfo('bin/test')
                item.size = 2
                bundle.addfile(item, io.BytesIO(b'ok'))
            extract(archive, root / 'out')
            self.assertEqual((root / 'out/bin/test').read_bytes(), b'ok')

    def test_source_build_requires_cuda_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'runtime/source/llama.cpp-commit'
            source.mkdir(parents=True)
            (source / 'CMakeLists.txt').write_text('project(example)', encoding='utf-8')
            import subprocess
            failed = subprocess.CalledProcessError(1, ['cmake'])
            with patch('scripts.deploy_autodl.ROOT', root), patch('scripts.deploy_autodl.extract'), \
                 patch('scripts.deploy_autodl.subprocess.check_output', return_value='8.6\n'), \
                 patch('scripts.deploy_autodl.subprocess.run', side_effect=failed) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    build_runtime(root / 'source.tar.gz', root / 'runtime/llama', 'commit')
                self.assertIn('-DGGML_CUDA=ON', run.call_args.args[0])
                self.assertFalse((root / 'runtime/deployment.json').exists())

    def test_preflight_rejects_windows_before_tools_or_downloads(self):
        with patch('scripts.deploy_autodl.platform.system', return_value='Windows'), \
             patch('scripts.deploy_autodl.shutil.which') as which:
            with self.assertRaisesRegex(RuntimeError, '不在本机部署'):
                preflight()
            which.assert_not_called()

    @unittest.skipUnless(sys.platform == 'linux' and shutil.which('cmake'),
                         'requires Linux CMake')
    def test_server_target_registered_with_upstream_tools_dependency(self):
        # Fixed upstream commit puts tools/server behind both build switches.
        # Run real CMake with the same dependency contract, rather than mocking
        # successful subprocesses and missing a nonexistent build target.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'runtime/source/llama.cpp-commit'
            source.mkdir(parents=True)
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.18)\n'
                'project(server_target_contract NONE)\n'
                'if(LLAMA_BUILD_TOOLS AND LLAMA_BUILD_SERVER)\n'
                '  add_custom_target(llama-server)\n'
                'endif()\n', encoding='utf-8')
            with patch('scripts.deploy_autodl.ROOT', root), \
                 patch('scripts.deploy_autodl.extract'), \
                 patch('scripts.deploy_autodl.subprocess.check_output', return_value='8.6\n'):
                build_runtime(root / 'source.tar.gz', root / 'runtime/llama', 'commit')


if __name__ == '__main__':
    unittest.main()
