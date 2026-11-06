"""Run the imported, unchanged edge-cloud reference CLI in its own directory.

Config, dataset and output paths are relative to the reference directory unless
absolute. No credentials or model services are discovered or started here.
"""
from pathlib import Path
import subprocess
import sys

REFERENCE = Path(__file__).resolve().parents[1] / 'baselines/edge_cloud_prototype'


def main():
    return subprocess.call(
        [sys.executable, '-X', 'utf8', '-m', 'edge_cloud', *sys.argv[1:]],
        cwd=REFERENCE,
    )


if __name__ == '__main__':
    raise SystemExit(main())
