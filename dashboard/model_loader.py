"""Optional future inference adapter for locally trained DistilBERT + XGBoost.

No model is downloaded, trained, balanced, or loaded during Stage 1. Importing
this module does not require Transformers, PyTorch, or XGBoost. The training
pipeline must explicitly register its original preprocessing callable.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class ModelIntegrationError(ValueError):
    """The exact training-time inference configuration is not available."""


def _local_path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ModelIntegrationError(f"Manifest requires {name}.")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()) or not path.exists():
        raise ModelIntegrationError(f"{name} must exist within the manifest's directory.")
    return path


def read_model_manifest(path: str | Path) -> dict:
    """Require an explicit inference contract; do not guess pooling or text rules."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ModelIntegrationError("Model manifest must be a JSON object.")
    for name in ("model_id", "checkpoint", "preprocessing_id"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ModelIntegrationError(f"Manifest requires {name} from the training run.")
    if not isinstance(manifest.get("pooling"), str) or manifest["pooling"] not in {"cls", "mean"}:
        raise ModelIntegrationError("pooling must explicitly be 'cls' or attention-mask 'mean', matching training.")
    if type(manifest.get("max_length")) is not int or not 1 <= manifest["max_length"] <= 512:
        raise ModelIntegrationError("max_length must be the training-time integer in [1, 512].")
    if manifest.get("label_mapping") != {"0": "Negative", "1": "Positive"}:
        raise ModelIntegrationError("label_mapping must explicitly map 0 to Negative and 1 to Positive.")
    if type(manifest.get("decision_threshold")) not in {int, float} or not 0 < manifest["decision_threshold"] < 1:
        raise ModelIntegrationError("decision_threshold must be explicitly recorded in (0, 1).")
    tokenizer = _local_path(path.parent, manifest.get("tokenizer_path"), "tokenizer_path")
    encoder = _local_path(path.parent, manifest.get("encoder_path"), "encoder_path")
    classifier = _local_path(path.parent, manifest.get("classifier_path"), "classifier_path")
    if not tokenizer.is_dir() or not encoder.is_dir():
        raise ModelIntegrationError("tokenizer_path and encoder_path must be local saved model directories.")
    if classifier.suffix.lower() not in {".json", ".ubj"} or not classifier.is_file():
        raise ModelIntegrationError("classifier_path must be an XGBoost JSON or UBJ model; pickle/joblib is unsupported.")
    return dict(manifest, tokenizer_path=tokenizer, encoder_path=encoder, classifier_path=classifier)


@dataclass
class HybridModel:
    manifest: dict
    tokenizer: Any
    encoder: Any
    classifier: Any
    preprocess: Callable[[str], str]

    def predict(self, reviews: Sequence[str], *, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Use the original callable and declared pooling; return labels and confidence.

        This explicit API is not called by the exploratory dashboard. Confidence
        is the classifier's probability for the selected class, not accuracy.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ModelIntegrationError("batch_size must be a positive integer.")
        if isinstance(reviews, (str, bytes)) or any(not isinstance(review, str) for review in reviews):
            raise ModelIntegrationError("Inference reviews must be strings.")
        if len(reviews) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        import torch
        labels, confidence = [], []
        self.encoder.eval()
        for offset in range(0, len(reviews), batch_size):
            texts = [self.preprocess(text) for text in reviews[offset:offset + batch_size]]
            if any(not isinstance(text, str) for text in texts):
                raise ModelIntegrationError("Registered preprocessing must return a string for each review.")
            tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.manifest["max_length"], return_tensors="pt")
            with torch.inference_mode():
                hidden = self.encoder(**tokens).last_hidden_state
                if self.manifest["pooling"] == "cls":
                    embeddings = hidden[:, 0, :]
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            probabilities = np.asarray(self.classifier.predict_proba(embeddings.cpu().numpy()))
            if probabilities.shape != (len(texts), 2) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
                raise ModelIntegrationError("Classifier did not return valid binary class probabilities.")
            predictions = (probabilities[:, 1] >= self.manifest["decision_threshold"]).astype(int)
            labels.extend(predictions.tolist())
            confidence.extend(probabilities[np.arange(len(texts)), predictions].tolist())
        return np.array(labels), np.array(confidence)


def load_trained_model(manifest_path: str | Path, preprocessing_registry: Mapping[str, Callable[[str], str]]) -> HybridModel:
    """Load exact local artifacts only after explicit application integration.

    Wrap this function with st.cache_resource in a future inference UI. Preserve
    the preprocessing_id in that cache key when training configurations change.
    Never load a manifest supplied by an untrusted party.
    """
    manifest = read_model_manifest(manifest_path)
    preprocess = preprocessing_registry.get(manifest["preprocessing_id"])
    if not callable(preprocess):
        raise ModelIntegrationError("Register the original training preprocessing callable under preprocessing_id; no substitute is permitted.")
    try:
        from transformers import AutoModel, AutoTokenizer
        from xgboost import XGBClassifier
        import torch  # noqa: F401 -- fails early with a helpful optional-dependency error
    except ImportError as exc:
        raise ModelIntegrationError("Optional inference dependencies are not installed. Install matching training versions of torch, transformers, and xgboost first.") from exc
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(manifest["encoder_path"], local_files_only=True, trust_remote_code=False, use_safetensors=True)
    if encoder.config.model_type != "distilbert":
        raise ModelIntegrationError("The encoder must be the trained DistilBERT model.")
    classifier = XGBClassifier()
    classifier.load_model(manifest["classifier_path"])
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise ModelIntegrationError("Classifier class order must be [0, 1].")
    if classifier.n_features_in_ != encoder.config.dim:
        raise ModelIntegrationError("XGBoost feature count does not match the encoder dimension.")
    return HybridModel(manifest, tokenizer, encoder, classifier, preprocess)
