"""Adapter tests use explicit doubles; only ActualTfidfTests uses trained weights."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from dashboard.prediction_backend import (PredictionOutput, VersionBPredictor, _cache_identity,
                                          _transformer_settings, get_predictor, labels_from_negative_scores,
                                          load_predictor, negative_column, predict_review_frame, readiness)
from dashboard.version_b import ARCHITECTURES, RESOURCES, ROOT, VersionBError, deployment_config, verified_artifact


class TensorDouble:
    """Only the tensor operations needed to verify adapter routing and pooling."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype

    def __getitem__(self, key): return TensorDouble(self.values[key])
    def __mul__(self, other): return TensorDouble(self.values * other.values)
    def __truediv__(self, other): return TensorDouble(self.values / other.values)
    def unsqueeze(self, axis): return TensorDouble(np.expand_dims(self.values, axis))
    def sum(self, axis): return TensorDouble(self.values.sum(axis))
    def clamp(self, min): return TensorDouble(np.maximum(self.values, min))
    def to(self, dtype): return self if getattr(dtype,'type',None)=='cpu' else TensorDouble(self.values.astype(dtype))
    def float(self): return self.to(np.float32)
    def cpu(self): return self
    def numpy(self): return self.values


def softmax_double(value, dim):
    scores = np.exp(value.values - value.values.max(axis=dim, keepdims=True))
    return TensorDouble(scores / scores.sum(axis=dim, keepdims=True))


TORCH_DOUBLE = SimpleNamespace(inference_mode=nullcontext, softmax=softmax_double, float32=np.float32, float64=np.float64)


class NegativeDecisionTests(unittest.TestCase):
    def test_native_float32_boundary_matches_reference_without_widening_scores(self):
        for architecture, entry in deployment_config()['models'].items():
            threshold=entry['threshold']
            scores=np.array([threshold],dtype=np.float32)
            probabilities=np.column_stack([scores,1-scores])
            extracted=negative_column(probabilities,[0,1])
            with self.subTest(architecture=architecture):
                self.assertEqual(extracted.dtype,np.float32)
                np.testing.assert_array_equal(labels_from_negative_scores(extracted,threshold),np.where(scores>=float(threshold),0,1))

    def test_each_exact_config_threshold_applies_to_class_zero_with_negative_tie(self):
        for key, entry in deployment_config()['models'].items():
            threshold = entry['threshold']
            with self.subTest(architecture=key):
                self.assertEqual(labels_from_negative_scores([0, np.nextafter(threshold, 0), threshold, 1], threshold).tolist(), [1,1,0,0])
        self.assertEqual(negative_column([[.1,.9],[.8,.2]], [1,0]).tolist(), [.9,.2])
        self.assertEqual(negative_column([[.1,.9]], [0,1]).tolist(), [.1])

    def test_invalid_class_order_scores_or_thresholds_fail(self):
        for scores, classes in [([[.1,.9]], [0,0]), ([[.1,.9]], [False,True]), ([[.1,.9]], ['Negative','Positive']),
                                ([[.1,.8]], [0,1]), ([[float('nan'),1]], [0,1]), ([[1.1,-.1]], [0,1])]:
            with self.subTest(scores=scores, classes=classes), self.assertRaises(VersionBError):
                negative_column(scores, classes)
        for scores, threshold in [([float('inf')], .2), ([-.1], .2), ([.3], 0), ([.3], float('nan'))]:
            with self.assertRaises(VersionBError):
                labels_from_negative_scores(scores, threshold)


class AdapterDoubleTests(unittest.TestCase):
    def test_adapters_match_reviewed_reference_functions_with_explicit_doubles(self):
        """Execute only two reviewed pure functions, with every external model doubled."""
        import ast
        source = ast.parse((RESOURCES/'reference_dashboard_app.py.txt').read_text())
        functions = [node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in {'bert_forward','run_predictions'}]
        self.assertEqual(len(functions),2)
        for node in functions:
            node.decorator_list=[]
        reference_code=compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'supplied-reference-functions','exec')
        for architecture in ARCHITECTURES:
            config=deployment_config()
            preprocessor=Mock()
            preprocessor.transform.side_effect=lambda text:text+' normalized'
            classifier=Mock(classes_=np.array([0,1]))
            classifier.predict_proba.side_effect=lambda features:np.tile([.4,.6],(len(features),1))
            vectorizer=Mock()
            vectorizer.transform.side_effect=lambda texts:np.ones((len(texts),2))
            def tokenizer(texts,**kwargs):
                self.assertEqual(kwargs['max_length'],256)
                self.assertTrue(kwargs['padding'])
                return {'input_ids':TensorDouble(np.zeros((len(texts),2))),
                        'attention_mask':TensorDouble(np.ones((len(texts),2)))}
            def encoder(**kwargs):
                count=len(kwargs['input_ids'].values)
                hidden=TensorDouble(np.tile([[2.,4.],[8.,10.]],(count,1,1)))
                return SimpleNamespace(last_hidden_state=hidden,hidden_states=(hidden,),logits=TensorDouble(np.tile([2.,0.],(count,1))))
            reference_prep=SimpleNamespace(preprocess_text=lambda text,*args,**kwargs:preprocessor.transform(text))
            classifiers={key:classifier for key in ARCHITECTURES}
            classifiers['vectorizer']=vectorizer
            encoder_tuple=(tokenizer,encoder,SimpleNamespace(type='cpu'))
            scope={'np':np,'pd':pd,'torch':TORCH_DOUBLE,'assets':lambda:(config,reference_prep,{}, {}, None),
                   'classifiers':lambda:classifiers,'frozen_encoder':lambda:encoder_tuple,'finetuned_encoder':lambda:encoder_tuple}
            exec(reference_code,scope)
            originals=['same','other','same','four','five','six']
            settings=None if architecture=='tfidf_xgb' else _transformer_settings(ROOT/'models/version_b',architecture,config)
            adapter=VersionBPredictor(architecture,config,preprocessor,classifier=classifier,vectorizer=vectorizer,tokenizer=tokenizer,encoder=encoder,settings=settings)
            with self.subTest(architecture=architecture),patch.dict(sys.modules,{'torch':TORCH_DOUBLE}):
                recorded=scope['run_predictions'](originals,architecture)
                result=adapter.predict(originals)
                self.assertEqual(recorded.prediction.tolist(),np.where(result.labels==0,'Negative','Positive').tolist())
                np.testing.assert_allclose(recorded.negative_score_uncalibrated,result.negative_scores,rtol=0,atol=0)

    def test_four_interfaces_preserve_raw_order_duplicate_rows_and_batch_boundaries(self):
        for architecture in ARCHITECTURES:
            config = deployment_config()
            preprocessor = Mock()
            preprocessor.transform.side_effect = lambda text: text + ' normalized'
            classifier = Mock(classes_=np.array([0,1]))
            classifier.predict_proba.side_effect = lambda features: np.tile([.4,.6], (len(features),1))
            vectorizer = Mock()
            vectorizer.transform.side_effect = lambda texts: np.array([[len(text),1] for text in texts])
            observed = []
            def tokenizer(texts, **kwargs):
                observed.extend(texts)
                self.assertEqual(kwargs['max_length'], 256)
                self.assertEqual(kwargs['padding'], 'max_length')
                self.assertTrue(kwargs['truncation'])
                self.assertFalse(kwargs['return_token_type_ids'])
                return {'input_ids':TensorDouble(np.zeros((len(texts),3))),
                        'attention_mask':TensorDouble(np.tile([1,1,0],(len(texts),1)))}
            def encoder(**kwargs):
                if architecture != 'frozen_hybrid':
                    self.assertTrue(kwargs.pop('output_hidden_states'))
                    self.assertTrue(kwargs.pop('return_dict'))
                self.assertEqual(set(kwargs), {'input_ids','attention_mask'})
                count = len(kwargs['input_ids'].values)
                hidden = TensorDouble(np.tile([[2.,4.],[6.,8.],[100.,100.]],(count,1,1)))
                return SimpleNamespace(logits=TensorDouble(np.tile([2.,0.],(count,1))),
                                       last_hidden_state=hidden, hidden_states=(hidden,))
            settings = {'padding':'max_length','truncation':True,'add_special_tokens':True,
                        'encoder_inputs':['input_ids','attention_mask'],'pooling':'mean','feature_dtype':'float32'}
            model = VersionBPredictor(architecture, config, preprocessor, classifier=classifier, vectorizer=vectorizer,
                                      tokenizer=tokenizer, encoder=encoder, settings=settings)
            with self.subTest(architecture=architecture), patch.dict(sys.modules, {'torch':TORCH_DOUBLE}):
                result = model.predict(['same','other','same'], batch_size=2)
                self.assertEqual([call.args[0] for call in preprocessor.transform.call_args_list], ['same','other','same'])
                self.assertEqual(len(result.labels), 3)
                self.assertEqual(result.labels.tolist(), [0,0,0])
                self.assertEqual(result.negative_scores[0], result.negative_scores[2])
                if architecture == 'tfidf_xgb':
                    self.assertEqual(vectorizer.transform.call_args_list[0].args[0], ['same normalized','other normalized'])
                    vectorizer.fit.assert_not_called()
                else:
                    self.assertEqual(observed, ['same normalized','other normalized','same normalized'])
                if architecture in ('frozen_hybrid','finetuned_hybrid'):
                    np.testing.assert_array_equal(classifier.predict_proba.call_args_list[0].args[0], [[4,6],[4,6]])
                classifier.fit.assert_not_called()

    def test_no_row_drops_for_empty_preprocessed_review(self):
        model = VersionBPredictor('tfidf_xgb', deployment_config(), Mock(transform=lambda value:''), classifier=Mock(), vectorizer=Mock())
        with self.assertRaisesRegex(VersionBError, 'No rows were dropped'):
            model.predict(['emoji-only'])
        model.classifier.predict_proba.assert_not_called()

    def test_public_outputs_separate_ratings_and_discard_uploaded_prediction_claims(self):
        frame = pd.DataFrame({'review_text':['same','bad','same'], 'rating':[1,5,1], 'sentiment':['Negative','Positive','Negative'],
                              'item_id':['secret']*3, 'cleaned_text':['private']*3, 'predicted_sentiment':['invented']*3, 'confidence':[1.]*3})
        model = Mock()
        model.predict.return_value = PredictionOutput(np.array([1,0,1]), np.array([.1,.8,.1]), 'tfidf_xgb', .28)
        result = predict_review_frame(frame,'tfidf_xgb',predictor=model)
        model.predict.assert_called_once_with(['same','bad','same'])
        self.assertEqual(result.review_text.tolist(), frame.review_text.tolist())
        self.assertEqual(result.sentiment.tolist(), frame.sentiment.tolist())
        self.assertEqual(result.predicted_sentiment.tolist(), ['Positive','Negative','Positive'])
        self.assertTrue({'item_id','cleaned_text','confidence'}.isdisjoint(result.columns))
        self.assertEqual(frame.predicted_sentiment.tolist(), ['invented']*3)
        model.predict.return_value = PredictionOutput(np.array([1]), np.array([.1]), 'tfidf_xgb', .28)
        with self.assertRaisesRegex(VersionBError, 'align'):
            predict_review_frame(frame,'tfidf_xgb',predictor=model)

    def test_text_only_inputs_are_supported_and_threshold_mismatch_is_rejected(self):
        model = Mock()
        model.predict.return_value = PredictionOutput(np.array([0]), np.array([.5]), 'tfidf_xgb', .28)
        result = predict_review_frame(pd.DataFrame({'review_text':['original']}),'tfidf_xgb',predictor=model)
        self.assertTrue(result.sentiment.isna().all())
        model.predict.return_value = PredictionOutput(np.array([0]), np.array([.5]), 'tfidf_xgb', .49)
        with self.assertRaisesRegex(VersionBError, 'threshold differs'):
            predict_review_frame(pd.DataFrame({'review_text':['original']}),'tfidf_xgb',predictor=model)


class LocalLoadingTests(unittest.TestCase):
    def test_missing_models_never_download_or_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            for architecture in ARCHITECTURES:
                ready, reason = readiness(architecture,Path(folder))
                self.assertFalse(ready)
                self.assertTrue('unavailable' in reason or 'Missing' in reason)
                with self.assertRaises(VersionBError):
                    load_predictor(architecture,Path(folder))

    def test_corrupted_vectorizer_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tfidf_vectorizer.joblib'
            path.write_bytes(b'untrusted serialized contents')
            with patch('joblib.load') as deserialize, self.assertRaisesRegex(VersionBError, 'checksum'):
                verified_artifact(Path(folder), path.name, deployment_config())
            deserialize.assert_not_called()

    def test_reference_profile_records_cls_and_rejects_unknown_pooling(self):
        with tempfile.TemporaryDirectory() as folder:
            import json
            import shutil
            resources = Path(folder)/'resources'
            shutil.copytree(RESOURCES,resources)
            settings = _transformer_settings(Path(folder),'frozen_hybrid',deployment_config())
            self.assertEqual(settings['pooling'],'cls')
            self.assertTrue(settings['padding'])
            self.assertEqual(settings['cpu_batch_size'],4)
            path = resources/'inference_profile.json'
            profile = json.loads(path.read_text())
            profile['frozen_hybrid']['pooling'] = 'unrecorded'
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(VersionBError, 'pooling'):
                _transformer_settings(Path(folder),'frozen_hybrid',deployment_config(),resources)

    def test_resource_cache_reuses_only_matching_artifact_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            from dashboard.prediction_backend import _cached_predictor
            _cached_predictor.clear()
            model = object()
            with patch('dashboard.prediction_backend.load_predictor', return_value=model) as load:
                self.assertIs(get_predictor('tfidf_xgb', Path(folder)), model)
                self.assertIs(get_predictor('tfidf_xgb', Path(folder)), model)
                self.assertEqual(load.call_count,1)
                (Path(folder)/'frozen_encoder_manifest.json').write_text('{}')
                self.assertIs(get_predictor('tfidf_xgb', Path(folder)), model)
                self.assertEqual(load.call_count,2)
            _cached_predictor.clear()


@unittest.skipUnless((ROOT/'models/version_b/tfidf_vectorizer.joblib').is_file() and importlib.util.find_spec('xgboost') is not None, 'Owner-supplied real TF-IDF artifacts/dependencies not provisioned')
class ActualTfidfTests(unittest.TestCase):
    def test_actual_saved_tfidf_runs_new_text_and_duplicates_without_ratings(self):
        model = load_predictor('tfidf_xgb', ROOT/'models/version_b')
        originals = ['Barang bagus!', 'Sound rosak, x boleh guna.', 'Barang bagus!']
        before = originals.copy()
        result = model.predict(originals, batch_size=2)
        self.assertEqual(result.labels.tolist(), [1,0,1])
        self.assertEqual(result.threshold, deployment_config()['models']['tfidf_xgb']['threshold'])
        self.assertEqual(result.negative_scores[0], result.negative_scores[2])
        self.assertEqual(originals,before)


@unittest.skipUnless((ROOT/'models/version_b/saved_tokenizer/tokenizer.json').exists() and importlib.util.find_spec('transformers') is not None,'Supplied saved tokenizers/dependencies not provisioned')
class ActualTokenizerTests(unittest.TestCase):
    def test_saved_tokenizers_reproduce_reference_ids_and_masks_without_transformer_weights(self):
        from transformers import AutoTokenizer
        from dashboard.prediction_backend import _verify_tokenizer_assets
        originals=['Sound BAGUS 👍','x boleh guna','Sound BAGUS 👍','bagus '*300]
        for architecture in ['frozen_hybrid','finetuned_hybrid']:
            settings=_transformer_settings(ROOT/'models/version_b',architecture,deployment_config())
            _verify_tokenizer_assets(ROOT/'models/version_b',settings,RESOURCES)
            tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/version_b'/settings['tokenizer_path'],use_fast=True,local_files_only=True,trust_remote_code=False)
            reference=tokenizer(originals,padding=True,truncation=True,max_length=256,return_tensors='np')
            adapter=tokenizer(originals,padding=settings['padding'],truncation=settings['truncation'],max_length=256,
                              add_special_tokens=True,return_attention_mask=True,return_token_type_ids=False,return_tensors='np')
            with self.subTest(architecture=architecture):
                self.assertEqual(tokenizer.padding_side,settings['padding_side'])
                self.assertEqual(tokenizer.truncation_side,settings['truncation_side'])
                for name in ['input_ids','attention_mask']:
                    np.testing.assert_array_equal(reference[name],adapter[name])
                if architecture=='frozen_hybrid':
                    self.assertIn('token_type_ids',reference)
                    self.assertNotIn('token_type_ids',adapter)


if __name__ == '__main__':
    unittest.main()
