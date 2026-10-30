"""Future adapter for the exact locally trained DistilBERT–XGBoost artifacts.

Stage 1 never calls this loader. No downloads, fitting, balancing, alternate
normalizers, tokenizer defaults, or invented confidence values are provided.
Optional model dependencies are imported only by explicit future integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class ModelIntegrationError(ValueError):
    """The exact training-time inference configuration is not available."""


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def preprocessing_fingerprint(files: Mapping[str, Path], settings: Mapping) -> str:
    """Fingerprint the approved code, normalization resources and training settings.

    Logical file names must stay identical between training export and serving.
    The registered callable is trusted application code; files are never executed.
    """
    if not files or any(not isinstance(name, str) or not name for name in files):
        raise ModelIntegrationError('Register the exact preprocessing code and normalization resource files.')
    try:
        hashes = {name: _file_hash(Path(path)) for name, path in files.items()}
        payload = json.dumps({'files': hashes, 'settings': dict(settings)}, sort_keys=True, allow_nan=False, separators=(',', ':'))
    except (OSError, TypeError, ValueError) as exc:
        raise ModelIntegrationError('Preprocessing files or settings cannot be verified.') from exc
    return sha256(payload.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class RegisteredPreprocessor:
    transform: Callable[[str], str]
    files: Mapping[str, Path]
    settings: Mapping


@dataclass(frozen=True)
class PredictionBatch:
    labels: Sequence[int]
    confidence: Sequence[float] | None = None

    def validate(self, expected_rows: int) -> tuple[np.ndarray, np.ndarray | None]:
        labels = np.asarray(self.labels, dtype=object)
        if labels.shape != (expected_rows,) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v not in (0, 1) for v in labels):
            raise ModelIntegrationError('Predictions must contain exactly one binary integer label per original review.')
        scores = None
        if self.confidence is not None:
            raw = np.asarray(self.confidence, dtype=object)
            if raw.shape != (expected_rows,) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in raw):
                raise ModelIntegrationError('Confidence must contain one numeric probability per review, or be absent.')
            scores = raw.astype(float)
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ModelIntegrationError('Confidence probabilities must be finite values in [0, 1].')
        return labels.astype(int), scores


def _local_path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ModelIntegrationError(f'Manifest requires {name}.')
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()) or not path.exists():
        raise ModelIntegrationError(f'{name} must exist within the manifest directory.')
    return path


def read_model_manifest(path: str | Path) -> dict:
    """Validate the recorded contract and exact saved model files, with no guesses."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelIntegrationError('The owner-provided model manifest is unavailable or invalid.') from exc
    if not isinstance(manifest, dict):
        raise ModelIntegrationError('Model manifest must be a JSON object.')
    for name in ('model_id', 'checkpoint', 'preprocessing_id'):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ModelIntegrationError(f'Manifest requires {name} from the training run.')
    if not isinstance(manifest.get('pooling'), str) or manifest['pooling'] not in {'cls', 'mean'}:
        raise ModelIntegrationError("pooling must explicitly be 'cls' or attention-mask 'mean', matching training.")
    if type(manifest.get('max_length')) is not int or not 1 <= manifest['max_length'] <= 512:
        raise ModelIntegrationError('max_length must be the training-time integer in [1, 512].')
    if manifest.get('schema_version') != 1 or type(manifest['schema_version']) is not int:
        raise ModelIntegrationError('Manifest requires schema_version=1.')
    if manifest.get('label_mapping') != {'0': 'Negative', '1': 'Positive'}:
        raise ModelIntegrationError('label_mapping must explicitly map 0 to Negative and 1 to Positive.')
    settings = manifest.get('tokenizer_settings')
    if not isinstance(settings, dict) or set(settings) != {'padding', 'truncation', 'add_special_tokens', 'padding_side', 'truncation_side'}:
        raise ModelIntegrationError('Record all tokenizer_settings: padding, truncation, add_special_tokens, padding_side, truncation_side.')
    if not (settings['padding'] is False or settings['padding'] in ('longest', 'max_length')) or type(settings['truncation']) is not bool or type(settings['add_special_tokens']) is not bool:
        raise ModelIntegrationError('Unsupported tokenizer padding, truncation, or special-token settings.')
    if settings['padding_side'] not in ('left', 'right') or settings['truncation_side'] not in ('left', 'right'):
        raise ModelIntegrationError('Record the saved tokenizer padding_side and truncation_side.')
    if manifest.get('encoder_inputs') not in (['input_ids'], ['input_ids', 'attention_mask']):
        raise ModelIntegrationError('encoder_inputs must explicitly record the tensors used during training.')
    if manifest.get('encoder_dtype') not in ('float32', 'float16', 'bfloat16') or manifest.get('feature_dtype') not in ('float32', 'float64'):
        raise ModelIntegrationError('Record encoder_dtype and feature_dtype from training.')
    if not _is_digest(manifest.get('preprocessing_fingerprint')):
        raise ModelIntegrationError('Record the exact preprocessing_fingerprint from the training export.')
    classification = manifest.get('classification')
    if not isinstance(classification, dict) or classification.get('method') not in ('predict', 'probability_threshold') or classification.get('confidence') not in ('none', 'predicted_class_probability'):
        raise ModelIntegrationError('Record the classification method and supported confidence policy.')
    objective = classification.get('objective')
    if objective not in ('binary:logistic', 'binary:hinge'):
        raise ModelIntegrationError('Record the saved binary XGBoost objective; unsupported objectives need an explicit adapter.')
    needs_probabilities = classification['method'] == 'probability_threshold' or classification['confidence'] == 'predicted_class_probability'
    if needs_probabilities and objective != 'binary:logistic':
        raise ModelIntegrationError('This objective does not support the declared probability output.')
    if classification['method'] == 'probability_threshold':
        threshold = classification.get('decision_threshold')
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ModelIntegrationError('Record the actual probability decision_threshold in (0, 1).')
    elif 'decision_threshold' in classification:
        raise ModelIntegrationError('A decision_threshold must not be silently applied to classifier.predict.')
    libraries = manifest.get('library_versions')
    if not isinstance(libraries, dict) or any(not isinstance(libraries.get(name), str) or not libraries[name].strip() for name in ('torch', 'transformers', 'tokenizers', 'xgboost')):
        raise ModelIntegrationError('Record the exact training library_versions for torch, transformers, tokenizers, and xgboost.')
    tokenizer = _local_path(path.parent, manifest.get('tokenizer_path'), 'tokenizer_path')
    encoder = _local_path(path.parent, manifest.get('encoder_path'), 'encoder_path')
    classifier = _local_path(path.parent, manifest.get('classifier_path'), 'classifier_path')
    if not tokenizer.is_dir() or not encoder.is_dir() or not any(tokenizer.iterdir()) or not any(encoder.iterdir()):
        raise ModelIntegrationError('Tokenizer and encoder must be nonempty local saved-model directories.')
    if not classifier.is_file() or classifier.suffix.lower() not in ('.json', '.ubj'):
        raise ModelIntegrationError('Classifier must be a local XGBoost JSON/UBJ file; pickle/joblib is unsupported.')
    files = [classifier] + [file for directory in (tokenizer, encoder) for file in directory.rglob('*') if file.is_file()]
    expected = manifest.get('artifact_sha256')
    inventory = {}
    for file in files:
        if not file.resolve().is_relative_to(path.parent.resolve()):
            raise ModelIntegrationError('Model artifacts must remain inside the manifest directory.')
        inventory[file.relative_to(path.parent.resolve()).as_posix()] = _file_hash(file)
    if not isinstance(expected, dict) or expected != inventory:
        raise ModelIntegrationError('artifact_sha256 must match every saved tokenizer, encoder and classifier file exactly.')
    return dict(manifest, tokenizer_path=tokenizer, encoder_path=encoder, classifier_path=classifier)


def _probabilities(classifier, embeddings: np.ndarray, rows: int) -> np.ndarray:
    if not callable(getattr(classifier, 'predict_proba', None)):
        raise ModelIntegrationError('The trained classifier does not support the declared probability output.')
    values = np.asarray(classifier.predict_proba(embeddings))
    if values.shape != (rows, 2) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1):
        raise ModelIntegrationError('Classifier did not return valid binary class probabilities.')
    return values


@dataclass
class HybridModel:
    manifest: dict
    tokenizer: Any
    encoder: Any
    classifier: Any
    preprocess: Callable[[str], str]

    def predict(self, reviews: Sequence[str], *, batch_size: int = 16) -> PredictionBatch:
        """Run one-to-one CPU inference only when explicitly connected after training."""
        if type(batch_size) is not int or batch_size < 1:
            raise ModelIntegrationError('batch_size must be a positive integer.')
        if isinstance(reviews, (str, bytes)) or any(not isinstance(review, str) for review in reviews):
            raise ModelIntegrationError('Inference reviews must be strings.')
        policy = self.manifest['classification']
        labels, confidence = [], [] if policy['confidence'] != 'none' else None
        if len(reviews) == 0:
            return PredictionBatch(labels, confidence)
        import torch
        self.encoder.eval()
        settings = self.manifest['tokenizer_settings']
        for offset in range(0, len(reviews), batch_size):
            texts = [self.preprocess(text) for text in reviews[offset:offset + batch_size]]
            if any(not isinstance(text, str) for text in texts):
                raise ModelIntegrationError('Preprocessing must return one string per original review; no rows may be silently dropped.')
            tokens = self.tokenizer(texts, padding=settings['padding'], truncation=settings['truncation'], add_special_tokens=settings['add_special_tokens'],
                                    max_length=self.manifest['max_length'], return_attention_mask=True, return_token_type_ids=False, return_tensors='pt')
            with torch.inference_mode():
                hidden = self.encoder(**{name: tokens[name] for name in self.manifest['encoder_inputs']}).last_hidden_state
                if self.manifest['pooling'] == 'cls':
                    embeddings = hidden[:, 0, :]
                else:
                    mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            features = embeddings.to(dtype=getattr(torch, self.manifest['feature_dtype'])).cpu().numpy()
            probabilities = None
            if policy['method'] == 'probability_threshold' or confidence is not None:
                probabilities = _probabilities(self.classifier, features, len(texts))
            predictions = (probabilities[:, 1] >= policy['decision_threshold']).astype(int) if policy['method'] == 'probability_threshold' else np.asarray(self.classifier.predict(features))
            batch = PredictionBatch(predictions)
            predictions, _ = batch.validate(len(texts))
            labels.extend(predictions.tolist())
            if confidence is not None:
                confidence.extend(probabilities[np.arange(len(texts)), predictions].tolist())
        result = PredictionBatch(labels, confidence)
        result.validate(len(reviews))
        return result


def load_trained_model(manifest_path: str | Path, preprocessing_registry: Mapping[str, RegisteredPreprocessor]) -> HybridModel:
    """Explicit future factory. Never call with an uploaded/untrusted manifest."""
    manifest = read_model_manifest(manifest_path)
    profile = preprocessing_registry.get(manifest['preprocessing_id'])
    if not isinstance(profile, RegisteredPreprocessor) or not callable(profile.transform):
        raise ModelIntegrationError('Register the original training preprocessing callable, files and settings; no substitute is permitted.')
    if preprocessing_fingerprint(profile.files, profile.settings) != manifest['preprocessing_fingerprint']:
        raise ModelIntegrationError('The registered preprocessing code, normalization resources or settings differ from training.')
    try:
        from transformers import AutoModel, AutoTokenizer
        from xgboost import XGBClassifier
        import torch  # noqa: F401
    except ImportError as exc:
        raise ModelIntegrationError('Install the exact training versions of torch, transformers, tokenizers, and xgboost before integration.') from exc
    if any(version(name) != manifest['library_versions'][name] for name in ('torch', 'transformers', 'tokenizers', 'xgboost')):
        raise ModelIntegrationError('Installed inference libraries differ from the recorded training versions.')
    tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer_path'], local_files_only=True, trust_remote_code=False)
    if any(getattr(tokenizer, name) != manifest['tokenizer_settings'][name] for name in ('padding_side', 'truncation_side')):
        raise ModelIntegrationError('Saved tokenizer settings do not match the training contract.')
    encoder = AutoModel.from_pretrained(manifest['encoder_path'], local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, manifest['encoder_dtype']))
    if encoder.config.model_type != 'distilbert' or manifest['max_length'] > encoder.config.max_position_embeddings:
        raise ModelIntegrationError('The saved encoder must be the trained DistilBERT with the recorded sequence length.')
    encoder.to('cpu')
    classifier = XGBClassifier()
    classifier.load_model(manifest['classifier_path'])
    if not np.array_equal(classifier.classes_, [0, 1]) or classifier.n_features_in_ != encoder.config.dim:
        raise ModelIntegrationError('Classifier label order or feature dimension does not match the trained encoder.')
    if classifier.get_xgb_params().get('objective') != manifest['classification']['objective']:
        raise ModelIntegrationError('The saved classifier objective differs from the training contract.')
    return HybridModel(manifest, tokenizer, encoder, classifier, profile.transform)
