"""Four Version B architecture adapters using owner-approved local artifacts only.

No training, balancing, threshold selection, held-out inference, remote downloads,
or user-supplied executable artifacts. Model outputs are uncalibrated scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.presentation import public_reviews
from dashboard.version_b import (ARCHITECTURES, RESOURCES, VersionBError, deployment_config,
                                 file_sha256, local_file, model_directory, read_json, verified_artifact)
from dashboard.version_b_preprocessing import VersionBPreprocessor


@dataclass(frozen=True)
class PredictionOutput:
    labels: np.ndarray
    negative_scores: np.ndarray
    architecture: str
    threshold: float


def negative_column(probabilities, classes) -> np.ndarray:
    values = np.asarray(probabilities)
    order = np.asarray(classes)
    if order.shape != (2,) or set(order.tolist()) != {0, 1} or any(isinstance(x, (bool, np.bool_)) for x in order):
        raise VersionBError('Model class ordering must identify exactly 0=Negative and 1=Positive.')
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1):
        raise VersionBError('The model did not return valid two-class scores.')
    # Preserve native score precision: NumPy threshold casting at a float32
    # boundary must match the supplied reference implementation exactly.
    return values[:, int(np.flatnonzero(order == 0)[0])]


def labels_from_negative_scores(scores, threshold: float) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 1 or values.dtype.kind not in 'fiu' or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise VersionBError('Negative-class scores must be finite values in [0, 1].')
    if type(threshold) not in (int, float) or not 0 < threshold < 1:
        raise VersionBError('Use the recorded full-precision negative threshold.')
    # Score is for class 0; do not apply the threshold to P(positive).
    return np.where(values >= threshold, 0, 1).astype(int)


def _transformer_settings(root: Path, architecture: str, config: dict, resources: Path = RESOURCES) -> dict:
    data = read_json(resources / 'inference_profile.json')
    if file_sha256(local_file(resources, data['reference_file'])) != data.get('reference_sha256'):
        raise VersionBError('The recorded reference inference source does not match its checksum.')
    settings = data.get(architecture)
    required = {'tokenizer_path','padding','truncation','add_special_tokens','padding_side','truncation_side','encoder_inputs','encoder_dtype','feature_dtype','negative_threshold_comparison'}
    if not isinstance(settings, dict) or not required.issubset(settings):
        raise VersionBError('The reference transformer inference settings are incomplete.')
    if settings['negative_threshold_comparison'] != '>=':
        raise VersionBError('The reference threshold tie rule differs; review the adapter explicitly before enabling inference.')
    if settings['padding'] not in (True, False, 'longest', 'max_length') or type(settings['truncation']) is not bool or type(settings['add_special_tokens']) is not bool:
        raise VersionBError('The recorded tokenizer settings are unsupported.')
    if settings['padding_side'] not in ('left','right') or settings['truncation_side'] not in ('left','right'):
        raise VersionBError('Record the reference tokenizer padding and truncation sides.')
    if settings['encoder_inputs'] not in (['input_ids'], ['input_ids','attention_mask']):
        raise VersionBError('Record the exact encoder input tensors from the reference implementation.')
    if settings['encoder_dtype'] not in ('float32','float16','bfloat16') or settings['feature_dtype'] not in ('float32','float64'):
        raise VersionBError('Record the exact encoder and embedding dtypes.')
    if architecture != 'finetuned_classifier' and settings.get('pooling') not in ('cls','mean'):
        raise VersionBError('Hybrid embedding pooling is missing; it must not be guessed.')
    if architecture == 'frozen_hybrid' and not settings.get('encoder_path'):
        raise VersionBError('The saved frozen encoder path is missing.')
    if settings.get('cpu_batch_size') != 4 or settings.get('use_fast') is not True:
        raise VersionBError('The CPU batch size and fast tokenizer must match the supplied reference.')
    return settings


def _verify_tokenizer_assets(root: Path, settings: dict, resources: Path):
    profile = read_json(resources/'inference_profile.json')
    prefix = settings['tokenizer_path'] + '/'
    inventory = {name:digest for name,digest in profile.get('artifact_sha256',{}).items() if name.startswith(prefix)}
    if not inventory:
        raise VersionBError('The reference tokenizer/config checksum inventory is missing.')
    for name, digest in inventory.items():
        if file_sha256(local_file(root,name)) != digest:
            raise VersionBError(f'The saved tokenizer or checkpoint config differs from the supplied integration kit: {name}')


def _verify_frozen_encoder(root: Path, encoder_path: Path, config: dict):
    manifest = read_json(root/'frozen_encoder_manifest.json')
    if manifest.get('pretrained_model_id') != config['pretrained_model_id'] or not isinstance(manifest.get('revision'),str) or not manifest['revision'].strip():
        raise VersionBError('The local frozen encoder must record its original pretrained model ID and revision.')
    hashes = manifest.get('artifact_sha256',{})
    for filename in ('config.json','model.safetensors'):
        if file_sha256(encoder_path/filename) != hashes.get(filename):
            raise VersionBError('The saved frozen encoder differs from its owner-approved revision/checksum inventory.')


def _check_tokenizer(path: Path):
    if not path.is_dir() or not (path/'tokenizer_config.json').is_file() or not any((path/name).is_file() for name in ('tokenizer.json','vocab.txt')):
        raise VersionBError(f'Saved tokenizer files are missing from {path.name}.')


def readiness(architecture: str, directory: Path | None = None, resources: Path = RESOURCES) -> tuple[bool, str]:
    try:
        config = deployment_config(resources)
        if architecture not in ARCHITECTURES:
            raise VersionBError('Choose one of the four approved deployment architectures.')
        root = model_directory() if directory is None else directory.resolve()
        entry = config['models'][architecture]
        if architecture != 'finetuned_classifier':
            verified_artifact(root, entry['model_file'], config)
        if architecture == 'tfidf_xgb':
            verified_artifact(root, entry['vectorizer_file'], config)
        else:
            checkpoint = local_file(root, config['fine_tuned_checkpoint_folder'])
            if architecture != 'frozen_hybrid':
                if not (checkpoint/'config.json').is_file():
                    raise VersionBError(f'Missing {checkpoint.name}/config.json and saved checkpoint/tokenizer files.')
                verified_artifact(checkpoint, 'model.safetensors', config)
            settings = _transformer_settings(root, architecture, config, resources)
            tokenizer_path = local_file(root, settings['tokenizer_path'])
            _check_tokenizer(tokenizer_path)
            _verify_tokenizer_assets(root,settings,resources)
            encoder_path = local_file(root, settings['encoder_path']) if architecture == 'frozen_hybrid' else checkpoint
            if not (encoder_path/'config.json').is_file() or not (encoder_path/'model.safetensors').is_file():
                raise VersionBError(f'Saved DistilBERT files are missing from {encoder_path.name}.')
            if architecture == 'frozen_hybrid':
                _verify_frozen_encoder(root,encoder_path,config)
        dependencies = ['regex','xgboost','joblib','sklearn'] if architecture == 'tfidf_xgb' else ['regex','torch','transformers','safetensors'] + ([] if architecture == 'finetuned_classifier' else ['xgboost'])
        missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
        if missing:
            raise VersionBError('Install inference dependencies: ' + ', '.join(missing))
        if architecture == 'tfidf_xgb' and version('scikit-learn') != '1.6.1':
            raise VersionBError('Install scikit-learn==1.6.1 to match the saved TF-IDF vectorizer.')
        return True, 'Local artifacts verified; ready to load. Scores are uncalibrated.'
    except (VersionBError, OSError, ValueError, KeyError, TypeError) as exc:
        return False, str(exc)


class VersionBPredictor:
    def __init__(self, architecture, config, preprocessor, *, classifier=None, vectorizer=None, tokenizer=None, encoder=None, settings=None):
        self.architecture, self.config, self.preprocessor = architecture, config, preprocessor
        self.classifier, self.vectorizer = classifier, vectorizer
        self.tokenizer, self.encoder, self.settings = tokenizer, encoder, settings
        self.threshold = config['models'][architecture]['threshold']

    def predict(self, reviews: Sequence[str], *, batch_size: int | None = None) -> PredictionOutput:
        if batch_size is None:
            batch_size = 16 if self.architecture == 'tfidf_xgb' else self.settings['cpu_batch_size']
        if isinstance(reviews, (str, bytes)) or any(not isinstance(text, str) for text in reviews) or type(batch_size) is not int or batch_size < 1:
            raise VersionBError('Provide a sequence of original review strings and a positive batch size.')
        labels, scores = [], []
        for start in range(0, len(reviews), batch_size):
            texts = [self.preprocessor.transform(text) for text in reviews[start:start+batch_size]]
            if any(not text.strip() for text in texts):
                raise VersionBError('A review is empty after the exact Version B preprocessing. No rows were dropped; review the input before analyzing this batch.')
            if self.architecture == 'tfidf_xgb':
                features = self.vectorizer.transform(texts)
                probabilities = self.classifier.predict_proba(features)
                negative = negative_column(probabilities, self.classifier.classes_)
            else:
                import torch
                settings = self.settings
                tokens = self.tokenizer(texts, padding=settings['padding'], truncation=settings['truncation'], add_special_tokens=settings['add_special_tokens'],
                                        max_length=self.config['max_length'], return_attention_mask=True, return_token_type_ids=False, return_tensors='pt')
                inputs = {key: tokens[key] for key in settings['encoder_inputs']}
                with torch.inference_mode():
                    output = self.encoder(**inputs) if self.architecture == 'frozen_hybrid' else self.encoder(**inputs,output_hidden_states=True,return_dict=True)
                    if self.architecture == 'finetuned_classifier':
                        probabilities = torch.softmax(output.logits.float(), dim=-1).cpu().numpy()
                        negative = negative_column(probabilities, [0, 1])
                    else:
                        hidden = output.last_hidden_state if self.architecture == 'frozen_hybrid' else output.hidden_states[-1]
                        if settings['pooling'] == 'cls':
                            features = hidden[:,0,:]
                        else:
                            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                            features = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                        features = features.to(dtype=getattr(torch, settings['feature_dtype'])).cpu().numpy()
                        negative = negative_column(self.classifier.predict_proba(features), self.classifier.classes_)
            if len(negative) != len(texts):
                raise VersionBError('The model changed the number of reviews. No partial output was accepted.')
            scores.append(negative.copy())
            labels.extend(labels_from_negative_scores(negative, self.threshold).tolist())
        negative_scores = np.concatenate(scores) if scores else np.array([],dtype=float)
        return PredictionOutput(np.asarray(labels, dtype=int), negative_scores, self.architecture, self.threshold)


def load_predictor(architecture: str, directory: Path | None = None, resources: Path = RESOURCES) -> VersionBPredictor:
    root = model_directory() if directory is None else directory.resolve()
    available, message = readiness(architecture, root, resources)
    if not available:
        raise VersionBError(message)
    config = deployment_config(resources)
    preprocessor = VersionBPreprocessor(resources, root/'cldr')
    classifier, vectorizer, tokenizer, encoder, settings = None, None, None, None, None
    if architecture != 'finetuned_classifier':
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.load_model(verified_artifact(root, config['models'][architecture]['model_file'], config))
        if set(classifier.classes_.tolist()) != {0,1}:
            raise VersionBError('The saved XGBoost classifier does not use the approved binary label encoding.')
    if architecture == 'tfidf_xgb':
        import joblib
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        # This exact owner-supplied serialization records sklearn 1.6.1.
        if sklearn.__version__ != '1.6.1':
            raise VersionBError('The trusted TF-IDF vectorizer requires scikit-learn==1.6.1, matching its saved metadata.')
        vectorizer = joblib.load(verified_artifact(root, config['models'][architecture]['vectorizer_file'], config))
        if type(vectorizer) is not TfidfVectorizer or vectorizer.preprocessor is not None or vectorizer.tokenizer is not None:
            raise VersionBError('The verified vectorizer is not the expected saved TF-IDF transformer.')
        if len(vectorizer.get_feature_names_out()) != classifier.n_features_in_:
            raise VersionBError('Saved vectorizer and classifier feature dimensions differ.')
    else:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        settings = _transformer_settings(root, architecture, config, resources)
        tokenizer = AutoTokenizer.from_pretrained(local_file(root, settings['tokenizer_path']), use_fast=settings['use_fast'], local_files_only=True, trust_remote_code=False)
        if tokenizer.padding_side != settings['padding_side'] or tokenizer.truncation_side != settings['truncation_side']:
            raise VersionBError('The saved tokenizer sides differ from the recorded inference settings.')
        checkpoint = local_file(root, settings['encoder_path']) if architecture == 'frozen_hybrid' else local_file(root, config['fine_tuned_checkpoint_folder'])
        cls = AutoModel if architecture == 'frozen_hybrid' else AutoModelForSequenceClassification
        encoder = cls.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch,settings['encoder_dtype']))
        if encoder.config.model_type != 'distilbert' or config['max_length'] > encoder.config.max_position_embeddings:
            raise VersionBError('Saved transformer architecture or sequence length is incompatible.')
        if architecture != 'frozen_hybrid' and encoder.config.num_labels != 2:
            raise VersionBError('The saved standalone classifier must have two output logits.')
        labels = {int(key):str(value).casefold() for key,value in getattr(encoder.config,'id2label',{}).items()}
        if set(labels.values()) & {'negative','positive'} and labels != {0:'negative',1:'positive'}:
            raise VersionBError('The transformer label names conflict with 0=Negative, 1=Positive.')
        if classifier is not None and classifier.n_features_in_ != encoder.config.dim:
            raise VersionBError('The saved encoder and hybrid XGBoost dimensions differ.')
        encoder.to('cpu'); encoder.eval()
    return VersionBPredictor(architecture, config, preprocessor, classifier=classifier, vectorizer=vectorizer, tokenizer=tokenizer, encoder=encoder, settings=settings)


def _cache_identity(root: Path, resources: Path, architecture: str) -> str:
    config = deployment_config(resources)
    files = [resources/'deployment_config.json', resources/'preprocessv3_cldr_ab.py',resources/'malay_normalization_map_v3.json',resources/'cldr_ms_observed_tts.tsv',resources/'inference_profile.json',resources/'reference_dashboard_app.py.txt']
    entry = config['models'][architecture]
    files += [local_file(root,entry[key]) for key in ('model_file','vectorizer_file') if key in entry]
    for directory in [root/config['fine_tuned_checkpoint_folder'], root/'frozen_distilbert', root/'saved_tokenizer', root/'cldr']:
        if directory.is_dir():
            files += [file for file in directory.rglob('*') if file.is_file()]
    files.append(root/'frozen_encoder_manifest.json')
    identities = [(str(path),path.stat().st_size,path.stat().st_mtime_ns) for path in files if path.is_file()]
    return sha256(json.dumps(identities).encode()).hexdigest()


@st.cache_resource(show_spinner='Loading trained sentiment model…')
def _cached_predictor(architecture: str, directory: str, resources: str, identity: str):
    return load_predictor(architecture, Path(directory), Path(resources))


def get_predictor(architecture: str, directory: Path | None = None, resources: Path = RESOURCES):
    root = model_directory() if directory is None else directory.resolve()
    if architecture not in ARCHITECTURES:
        raise VersionBError('Unknown deployed architecture.')
    try:
        return _cached_predictor(architecture, str(root), str(resources), _cache_identity(root, resources, architecture))
    except VersionBError:
        raise
    except Exception as exc:
        raise VersionBError('The saved model could not load with the installed dependencies. Verify the local artifacts and documented package versions.') from exc


def predict_review_frame(frame: pd.DataFrame, architecture: str, *, predictor=None, directory=None) -> pd.DataFrame:
    if 'review_text' not in frame or frame.review_text.isna().any() or any(not isinstance(text,str) for text in frame.review_text):
        raise VersionBError('Original review text is required for every row.')
    model = predictor if predictor is not None else get_predictor(architecture, directory)
    output = model.predict(frame.review_text.tolist())
    if not isinstance(output, PredictionOutput) or np.asarray(output.labels).dtype.kind not in 'iu':
        raise VersionBError('The backend must return binary integer labels and raw negative-class scores.')
    scores = np.asarray(output.negative_scores)
    if output.threshold != deployment_config()['models'][architecture]['threshold']:
        raise VersionBError('The prediction threshold differs from the approved deployment configuration.')
    expected = labels_from_negative_scores(scores, output.threshold)
    if len(scores) != len(frame) or not np.array_equal(expected, output.labels) or output.architecture != architecture:
        raise VersionBError('Prediction results do not align one-to-one with the requested input and architecture.')
    result = public_reviews(frame.drop(columns=['predicted_sentiment','confidence'], errors='ignore'))
    result['predicted_label'] = output.labels
    result['predicted_sentiment'] = np.where(output.labels==0,'Negative','Positive')
    result['negative_score'] = scores
    result['prediction_model'] = ARCHITECTURES[architecture]
    result['negative_threshold'] = output.threshold
    return result
