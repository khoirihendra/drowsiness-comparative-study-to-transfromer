"""Small, dependency-free helpers for reproducible, non-overwriting runs."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def provenance():
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ('numpy', 'opencv-python', 'opencv-python-headless', 'mediapipe',
                 'tensorflow', 'tensorflow-cpu', 'scikit-learn', 'xgboost', 'keras'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    sources = sorted(root.glob('*.py')) + sorted((root / 'src').rglob('*.py'))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                python=platform.python_version(), packages=packages,
                source_sha256={str(p.relative_to(root)): sha256(p) for p in sources})


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
