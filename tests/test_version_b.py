"""Recorded exports and source identity tests; temporary fixtures are not findings."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from dashboard.final_evaluations import load_final_evaluations
from dashboard.version_b import ARCHITECTURES, EVALUATIONS, RESOURCES, ROOT, VersionBError, deployment_config, file_sha256
from dashboard.version_b_preprocessing import VersionBPreprocessor
from dashboard.version_b_research import load_version_b_predictions, load_version_b_research


class RecordedEvaluationTests(unittest.TestCase):
    def test_actual_eight_exports_remain_in_separate_test_groups(self):
        report = load_final_evaluations()
        original = pd.read_csv(EVALUATIONS / 'day3_all_models_test_metrics.csv')
        self.assertEqual(len(report.rows), 8)
        for split, count, negative in [('80_20', 3762, 61), ('70_30', 5625, 71)]:
            group = report.group(split)
            self.assertEqual(set(group.architecture), set(ARCHITECTURES))
            self.assertEqual(group.test_count.tolist(), [count] * 4)
            self.assertEqual(group.actual_negative.tolist(), [negative] * 4)
        pd.testing.assert_series_equal(report.rows.macro_f1, original.macro_f1)
        pd.testing.assert_series_equal(report.rows.negative_PR_AUC, original.negative_PR_AUC)
        self.assertEqual(report.metrics_bytes, (EVALUATIONS / 'day3_all_models_test_metrics.csv').read_bytes())

    def test_missing_partial_duplicate_or_inconsistent_exports_are_not_filled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(load_final_evaluations(root))
            metrics = root / 'day3_all_models_test_metrics.csv'
            matrix = root / 'day3_all_models_confusion.csv'
            shutil.copy2(EVALUATIONS / metrics.name, metrics)
            with self.assertRaises(VersionBError):
                load_final_evaluations(root)
            shutil.copy2(EVALUATIONS / matrix.name, matrix)
            frame = pd.read_csv(metrics)
            for field, value in [('macro_f1', .99), ('negative_PR_AUC', float('nan')), ('test_count', 0)]:
                corrupt = frame.copy()
                corrupt.loc[0, field] = value
                corrupt.to_csv(metrics, index=False)
                with self.subTest(field=field), self.assertRaises(VersionBError):
                    load_final_evaluations(root)
            duplicate = pd.concat([frame.iloc[:-1], frame.iloc[:1]], ignore_index=True)
            duplicate.to_csv(metrics, index=False)
            with self.assertRaises(VersionBError):
                load_final_evaluations(root)


class ResearchIdentityTests(unittest.TestCase):
    """Small, temporary source fixtures exercise joins without changing real CSVs."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.resources = self.root / 'resources'
        shutil.copytree(RESOURCES, self.resources)
        self.frame = pd.DataFrame({'source_row_id':['r3','r1','r2'], 'stars':[5,1,5], 'sentiment':[1,0,1],
                                  'comment':['same original','bad original','same original'],
                                  'cleaned_comment':['same original','bad original','same original'],
                                  'item_id':['14750761025'] * 3})
        self.dates = pd.DataFrame({'source_row_id':['r1','r2','r3'], 'datetime':['2024-01-01','2024-02-01','2024-03-01']})
        self.install_source()

    def tearDown(self):
        self.temp.cleanup()

    def install_source(self):
        source, metadata = self.root/'version_B_model_ready.csv', self.root/'version_B_dashboard_metadata.csv'
        self.frame.to_csv(source, index=False)
        self.dates.to_csv(metadata, index=False)
        manifest = json.loads((self.resources/'research_manifest.json').read_text())
        manifest.update(expected_rows=3, dataset_sha256=file_sha256(source), metadata_sha256=file_sha256(metadata))
        (self.resources/'research_manifest.json').write_text(json.dumps(manifest))
        config = json.loads((self.resources/'deployment_config.json').read_text())
        config['training_source_sha256'] = manifest['dataset_sha256']
        (self.resources/'deployment_config.json').write_text(json.dumps(config))

    def save_predictions(self, frame, **extra):
        path = self.root/'research_predictions.csv'
        frame.to_csv(path, index=False)
        config = deployment_config(self.resources)
        manifest = {'training_source_sha256':config['training_source_sha256'], 'predictions_sha256':file_sha256(path),
                    'architecture':'tfidf_xgb', 'negative_threshold':config['models']['tfidf_xgb']['threshold'], **extra}
        (self.root/'research_predictions_manifest.json').write_text(json.dumps(manifest))

    def test_dates_join_by_id_and_original_order_duplicates_are_preserved(self):
        data = load_version_b_research(self.root, self.resources)
        self.assertEqual(data.reviews.review_id.tolist(), ['version_b:r3','version_b:r1','version_b:r2'])
        self.assertEqual(data.reviews.review_date.dt.month.tolist(), [3,1,2])
        self.assertEqual(data.reviews.review_text.tolist(), self.frame.comment.tolist())
        self.assertEqual(data.reviews.sentiment.tolist(), ['Positive','Negative','Positive'])

    def test_rejects_duplicate_missing_foreign_metadata_ids_and_tampered_source(self):
        for ids in [['r1','r1','r3'], ['r1','r2','foreign'], ['r1','r2','']]:
            self.dates.source_row_id = ids
            self.install_source()
            with self.subTest(ids=ids), self.assertRaises(VersionBError):
                load_version_b_research(self.root, self.resources)
        with (self.root/'version_B_model_ready.csv').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(VersionBError, 'provenance'):
            load_version_b_research(self.root, self.resources)

    def test_prediction_overlay_requires_approved_ids_source_and_model_provenance(self):
        self.assertIsNone(load_version_b_predictions(self.root, self.resources))
        self.save_predictions(pd.DataFrame({'source_row_id':['r2'], 'predicted_sentiment':[0]}))
        result = load_version_b_predictions(self.root, self.resources)
        self.assertEqual(result.review_id.tolist(), ['version_b:r2'])
        self.assertEqual(result.predicted_sentiment.tolist(), ['Negative'])
        for ids in [['r2','r2'], ['legacy:r2']]:
            self.save_predictions(pd.DataFrame({'source_row_id':ids, 'predicted_sentiment':[0]*len(ids)}))
            with self.subTest(ids=ids), self.assertRaises(VersionBError):
                load_version_b_predictions(self.root, self.resources)
        self.save_predictions(pd.DataFrame({'source_row_id':['r2'], 'predicted_sentiment':[0]}), negative_threshold=.99)
        with self.assertRaisesRegex(VersionBError, 'threshold'):
            load_version_b_predictions(self.root, self.resources)


class ExactPreprocessingTests(unittest.TestCase):
    def test_normalization_preserves_version_b_case_and_reserved_words(self):
        preprocessor = VersionBPreprocessor()
        self.assertEqual(preprocessor.transform('padu mantap hauk baguss'), 'padu mantap hauk baguss')
        self.assertEqual(preprocessor.transform('Sound BAGUS'), 'Sound BAGUS')

    def test_corrupt_resource_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'resources'
            shutil.copytree(RESOURCES, root)
            (root/'preprocessv3_cldr_ab.py').write_text('raise RuntimeError("must never execute")')
            with self.assertRaisesRegex(VersionBError, 'checksum mismatch'):
                VersionBPreprocessor(root)

    @unittest.skipUnless((ROOT/'models/version_b/version_B_model_ready.csv').exists(), 'Private Version B source not provisioned')
    def test_actual_18705_saved_cleaned_texts_match_exact_preprocessor(self):
        path = ROOT/'models/version_b/version_B_model_ready.csv'
        self.assertEqual(file_sha256(path), deployment_config()['training_source_sha256'])
        source = pd.read_csv(path, keep_default_na=False)
        preprocessor = VersionBPreprocessor(cldr_dir=ROOT/'models/version_b/cldr')
        cleaned = source.comment.map(preprocessor.transform)
        self.assertEqual(len(cleaned), 18705)
        self.assertEqual(int(cleaned.ne(source.cleaned_comment).sum()), 0)


if __name__ == '__main__':
    unittest.main()
