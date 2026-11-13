"""Trusted local Version B package locations and integrity checks; no downloads."""
from __future__ import annotations

from hashlib import sha256
from functools import lru_cache
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / 'resources' / 'version_b'
EVALUATIONS = ROOT / 'evaluation' / 'version_b'
ARCHITECTURES = {
    'frozen_hybrid': 'Frozen DistilBERT + XGBoost',
    'tfidf_xgb': 'TF-IDF + XGBoost',
    'finetuned_classifier': 'Standalone fine-tuned DistilBERT',
    'finetuned_hybrid': 'Fine-tuned DistilBERT + XGBoost',
}


class VersionBError(ValueError):
    pass


@lru_cache(maxsize=128)
def _file_digest(filename: str, identity: tuple) -> str:
    digest = sha256()
    with Path(filename).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    try:
        stat = path.stat()
        identity = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        return _file_digest(str(path.resolve()), identity)
    except OSError as exc:
        raise VersionBError(f'Required local artifact is unavailable: {path.name}') from exc


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError) as exc:
        raise VersionBError(f'Cannot read package metadata: {path.name}') from exc
    if not isinstance(data, dict):
        raise VersionBError(f'Package metadata must contain an object: {path.name}')
    return data


def model_directory() -> Path:
    return Path(os.environ.get('SENTILOKAL_MODEL_DIR', ROOT / 'models' / 'version_b')).expanduser().resolve()


def local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise VersionBError('A package artifact filename is missing.')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise VersionBError('Package paths must stay inside the configured artifact directory.')
    return target


def deployment_config(resources: Path = RESOURCES) -> dict:
    data = read_json(resources / 'deployment_config.json')
    if data.get('negative_label') != 0 or data.get('positive_label') != 1 or data.get('score_calibrated') is not False:
        raise VersionBError('The deployment label/score semantics do not match the approved Version B export.')
    if set(data.get('models', {})) != set(ARCHITECTURES):
        raise VersionBError('The deployment configuration must describe the four approved architectures.')
    for entry in data['models'].values():
        threshold = entry.get('threshold')
        if type(threshold) not in (int, float) or not 0 < threshold < 1:
            raise VersionBError('The recorded negative-class threshold is invalid.')
    if data.get('dashboard_default_architecture') not in ARCHITECTURES or type(data.get('max_length')) is not int:
        raise VersionBError('The recorded default architecture or sequence length is invalid.')
    return data


def verified_artifact(root: Path, filename: str, config: dict) -> Path:
    path = local_file(root, filename)
    expected = config.get('model_artifact_sha256', {}).get(filename)
    if not expected or file_sha256(path) != expected:
        raise VersionBError(f'Artifact checksum does not match the approved deployment: {filename}')
    return path
