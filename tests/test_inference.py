"""Inference contracts tested with temporary files and deterministic test doubles.

No checkpoint is trained, loaded or substituted. Fixture outputs are test data,
never production sentiment results. Optional ML packages are not required.
"""
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from dashboard.inference import MODEL_PENDING_MESSAGE, PredictionService
from dashboard.model_loader import (HybridModel, ModelIntegrationError, PredictionBatch,
                                    RegisteredPreprocessor, load_trained_model,
                                    preprocessing_fingerprint, read_model_manifest)


class PredictionServiceTests(unittest.TestCase):
    def test_no_model_means_no_predictions_or_fallback(self):
        service = PredictionService()
        self.assertFalse(service.available)
        with self.assertRaisesRegex(ModelIntegrationError, MODEL_PENDING_MESSAGE):
            service.predict_reviews(pd.DataFrame({'review_text': ['bagus']}))

    def test_all_original_rows_survive_without_balancing_or_rating_dependency(self):
        frame = pd.DataFrame({'review_text': [' SAME! ', ' SAME! ', 'baru', 'lagi'],
                              'rating': [3, np.nan, 1, 5], 'product_id': ['private'] * 4,
                              'shop_id': ['do not expose'] * 4, 'predicted_sentiment': ['untrusted'] * 4,
                              'confidence': [9] * 4}, index=[9, 2, 3, 1])
        before = frame.copy(deep=True)
        model = Mock()
        model.predict.return_value = PredictionBatch([0, 1, 1, 0])
        result = PredictionService(model).predict_reviews(frame, batch_size=2)
        model.predict.assert_called_once_with(frame.review_text.tolist(), batch_size=2)
        self.assertEqual(result.review_text.tolist(), frame.review_text.tolist())
        self.assertEqual(result.predicted_label.tolist(), [0, 1, 1, 0])
        self.assertEqual(result.predicted_sentiment.tolist(), ['Negative', 'Positive', 'Positive', 'Negative'])
        self.assertNotIn('confidence', result)
        self.assertNotIn('product_id', result)
        self.assertNotIn('shop_id', result)
        pd.testing.assert_frame_equal(frame, before)

    def test_supported_confidence_and_empty_result_export_contract(self):
        model = Mock()
        model.predict.return_value = PredictionBatch([1], [.8])
        result = PredictionService(model).predict_reviews(pd.DataFrame({'review_text': ['original']}))
        self.assertEqual(result.confidence.tolist(), [.8])
        model.predict.return_value = PredictionBatch([], None)
        empty = PredictionService(model).predict_reviews(pd.DataFrame({'review_text': []}))
        self.assertTrue(empty.empty)
        self.assertIn('predicted_label', empty)

    def test_invalid_outputs_are_rejected_instead_of_filled_with_fake_values(self):
        invalid = [PredictionBatch([1, 0]), PredictionBatch([True]), PredictionBatch(['Positive']),
                   PredictionBatch([.5]), PredictionBatch([2]), PredictionBatch([0], [float('nan')]),
                   PredictionBatch([0], [1.1]), PredictionBatch([0], []), PredictionBatch([0], ['.8'])]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ModelIntegrationError):
                model = Mock()
                model.predict.return_value = result
                PredictionService(model).predict_reviews(pd.DataFrame({'review_text': ['original']}))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for directory in ['tokenizer', 'encoder']:
            (self.root / directory).mkdir()
        for filename in ['tokenizer/config.json', 'encoder/model.safetensors', 'classifier.ubj']:
            (self.root / filename).write_bytes(b'test fixture only; not trained model data')
        self.preprocess = self.root / 'approved_fixture.py'
        self.preprocess.write_text('# test-only preprocessing source identity\n')
        self.profile = RegisteredPreprocessor(lambda text: text, {'pipeline': self.preprocess}, {'normalization_version': 'test-only'})
        self.manifest = {
            'schema_version': 1, 'model_id': 'test-only', 'checkpoint': 'test-only',
            'preprocessing_id': 'approved-fixture',
            'preprocessing_fingerprint': preprocessing_fingerprint(self.profile.files, self.profile.settings),
            'pooling': 'mean', 'max_length': 7,
            'tokenizer_settings': {'padding': 'max_length', 'truncation': True, 'add_special_tokens': True, 'padding_side': 'right', 'truncation_side': 'right'},
            'encoder_inputs': ['input_ids', 'attention_mask'], 'encoder_dtype': 'float32', 'feature_dtype': 'float32',
            'label_mapping': {'0': 'Negative', '1': 'Positive'},
            'classification': {'method': 'predict', 'confidence': 'none', 'objective': 'binary:hinge'},
            'library_versions': {name: 'test-only' for name in ['torch', 'transformers', 'tokenizers', 'xgboost']},
            'tokenizer_path': 'tokenizer', 'encoder_path': 'encoder', 'classifier_path': 'classifier.ubj',
            'artifact_sha256': {name: sha256((self.root / name).read_bytes()).hexdigest() for name in ['tokenizer/config.json', 'encoder/model.safetensors', 'classifier.ubj']},
        }
        self.path = self.root / 'manifest.json'

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.manifest))
        return self.path

    def test_complete_saved_contract_validates_without_importing_model_packages(self):
        result = read_model_manifest(self.write())
        self.assertEqual(result['pooling'], 'mean')
        self.assertEqual(result['tokenizer_settings'], self.manifest['tokenizer_settings'])
        self.assertEqual(result['classification']['confidence'], 'none')

    def test_changed_or_extra_artifact_is_rejected(self):
        self.write()
        (self.root / 'tokenizer/config.json').write_text('changed after training')
        with self.assertRaisesRegex(ModelIntegrationError, 'artifact_sha256'):
            read_model_manifest(self.path)

    def test_unknown_tokenization_and_missing_settings_are_not_guessed(self):
        del self.manifest['tokenizer_settings']['padding_side']
        with self.assertRaisesRegex(ModelIntegrationError, 'tokenizer_settings'):
            read_model_manifest(self.write())

    def test_confidence_and_classification_policies_must_match_model_support(self):
        self.manifest['classification']['confidence'] = 'predicted_class_probability'
        with self.assertRaisesRegex(ModelIntegrationError, 'does not support'):
            read_model_manifest(self.write())
        self.manifest['classification'] = {'method': 'probability_threshold', 'confidence': 'none', 'objective': 'binary:logistic'}
        with self.assertRaisesRegex(ModelIntegrationError, 'decision_threshold'):
            read_model_manifest(self.write())

    def test_different_normalization_rules_are_rejected_before_model_loading(self):
        self.write()
        self.preprocess.write_text('# test-only changed normalization rules\n')
        with self.assertRaisesRegex(ModelIntegrationError, 'differ from training'):
            load_trained_model(self.path, {'approved-fixture': self.profile})

    def test_missing_preprocessor_never_uses_any_repository_pipeline_as_fallback(self):
        with self.assertRaisesRegex(ModelIntegrationError, 'no substitute'):
            load_trained_model(self.write(), {})

    def test_preprocessing_fingerprint_covers_config_and_resource_files(self):
        original = preprocessing_fingerprint(self.profile.files, self.profile.settings)
        self.assertNotEqual(original, preprocessing_fingerprint(self.profile.files, {'normalization_version': 'changed'}))
        self.preprocess.write_text('changed')
        self.assertNotEqual(original, preprocessing_fingerprint(self.profile.files, self.profile.settings))


class Tensor:
    """NumPy-backed tensor double for checking pooling without loading PyTorch."""
    def __init__(self, data): self.data = np.asarray(data)
    @property
    def dtype(self): return self.data.dtype
    def __getitem__(self, key): return Tensor(self.data[key])
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.data, axis))
    def to(self, dtype): return Tensor(self.data.astype(dtype))
    def sum(self, axis): return Tensor(self.data.sum(axis))
    def clamp(self, min): return Tensor(self.data.clip(min=min))
    def __mul__(self, other): return Tensor(self.data * other.data)
    def __truediv__(self, other): return Tensor(self.data / other.data)
    def cpu(self): return self
    def numpy(self): return self.data


class HybridAdapterTests(unittest.TestCase):
    def run_adapter(self, pooling, policy, classifier):
        tokenizer = Mock(side_effect=lambda texts, **kwargs: {'input_ids': Tensor(np.ones((len(texts), 3))), 'attention_mask': Tensor([[1, 1, 0]] * len(texts))})
        encoder = Mock(side_effect=lambda **kwargs: SimpleNamespace(last_hidden_state=Tensor([[[1., 3.], [3., 5.], [100., 100.]]] * len(kwargs['input_ids'].data))))
        preprocess = Mock(side_effect=lambda text: 'approved fixture: ' + text)
        manifest = {'pooling': pooling, 'max_length': 7, 'feature_dtype': 'float32', 'classification': policy,
                    'encoder_inputs': ['input_ids', 'attention_mask'],
                    'tokenizer_settings': {'padding': 'max_length', 'truncation': True, 'add_special_tokens': True}}
        model = HybridModel(manifest, tokenizer, encoder, classifier, preprocess)
        torch = SimpleNamespace(inference_mode=nullcontext, float32=np.float32)
        with patch.dict(sys.modules, {'torch': torch}):
            output = model.predict([' ORIGINAL ', 'duplicate', 'duplicate'], batch_size=2)
        return output, tokenizer, encoder, preprocess

    def test_original_normalizer_tokenizer_settings_and_masked_mean_are_used(self):
        classifier = Mock()
        classifier.predict.side_effect = lambda values: np.zeros(len(values), dtype=int)
        result, tokenizer, encoder, preprocess = self.run_adapter('mean', {'method': 'predict', 'confidence': 'none'}, classifier)
        np.testing.assert_array_equal(result.labels, [0, 0, 0])
        self.assertIsNone(result.confidence)
        self.assertEqual([call.args[0] for call in preprocess.call_args_list], [' ORIGINAL ', 'duplicate', 'duplicate'])
        self.assertEqual(tokenizer.call_args_list[0].args[0], ['approved fixture:  ORIGINAL ', 'approved fixture: duplicate'])
        self.assertEqual(tokenizer.call_args_list[0].kwargs['max_length'], 7)
        self.assertEqual(tokenizer.call_args_list[0].kwargs['padding'], 'max_length')
        self.assertTrue(tokenizer.call_args_list[0].kwargs['add_special_tokens'])
        self.assertEqual(set(encoder.call_args.kwargs), {'input_ids', 'attention_mask'})
        for call in classifier.predict.call_args_list:
            np.testing.assert_array_equal(call.args[0], [[2., 4.]] * len(call.args[0]))
            self.assertEqual(call.args[0].dtype, np.float32)
        classifier.predict_proba.assert_not_called()
        encoder.eval.assert_called_once()

    def test_cls_threshold_and_selected_class_probability_follow_contract(self):
        classifier = Mock()
        classifier.predict_proba.side_effect = lambda values: np.array([[.2, .8]] * len(values))
        policy = {'method': 'probability_threshold', 'decision_threshold': .9, 'confidence': 'predicted_class_probability'}
        result, *_ = self.run_adapter('cls', policy, classifier)
        self.assertEqual(result.labels, [0, 0, 0])
        self.assertEqual(result.confidence, [.2, .2, .2])
        classifier.predict.assert_not_called()
        for call in classifier.predict_proba.call_args_list:
            np.testing.assert_array_equal(call.args[0], [[1., 3.]] * len(call.args[0]))

    def test_invalid_probability_values_do_not_become_confidence(self):
        classifier = Mock()
        classifier.predict_proba.side_effect = lambda values: np.array([[.8, .8]] * len(values))
        with self.assertRaisesRegex(ModelIntegrationError, 'valid binary class probabilities'):
            self.run_adapter('cls', {'method': 'probability_threshold', 'decision_threshold': .5, 'confidence': 'none'}, classifier)


if __name__ == '__main__':
    unittest.main()
