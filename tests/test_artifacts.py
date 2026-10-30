"""Synthetic test-only result fixtures; these are never dashboard findings."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dashboard.artifacts import (
    ArtifactError, attach_predictions, comparable_evaluations,
    load_evaluation, load_predictions, load_results,
)
from dashboard.model_loader import ModelIntegrationError, read_model_manifest


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = {
            "model_id": "synthetic-test-model", "model_name": "Test fixture only",
            "test_set_id": "synthetic-held-out", "split": "test", "held_out": True,
            "evaluation_config": {"averaging": "binary", "positive_label": 1, "zero_division": 0, "decision_threshold": 0.5},
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self, metadata=None, name="evaluation_test.json"):
        path = self.root / name
        path.write_text(json.dumps(metadata or self.metadata), encoding="utf-8")
        return path

    def recorded(self):
        return dict(self.metadata, confusion_matrix=[[2, 1], [1, 2]], n_samples=6,
                    metrics={"accuracy": 2/3, "precision": 2/3, "recall": 2/3, "f1": 2/3})

    def test_missing_results_are_empty_not_mocked(self):
        result = load_results(self.root / "absent")
        self.assertEqual(result.evaluations, [])
        self.assertIsNone(result.predictions)
        self.assertEqual(result.errors, [])

    def test_metrics_from_actual_test_rows(self):
        pd.DataFrame({"review_id": ["a", "b", "c", "d"], "y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1]}).to_csv(self.root / "test.csv", index=False)
        evaluation = load_evaluation(self.write(dict(self.metadata, predictions_file="test.csv")))
        self.assertEqual(evaluation.confusion_matrix, [[1, 1], [0, 2]])
        self.assertEqual(evaluation.metrics["accuracy"], .75)
        self.assertAlmostEqual(evaluation.metrics["f1"], .8)
        self.assertEqual(evaluation.n_samples, 4)

    def test_recorded_metrics_must_agree_with_matrix(self):
        metadata = self.recorded()
        self.assertAlmostEqual(load_evaluation(self.write(metadata)).metrics["f1"], 2/3)
        metadata["metrics"]["f1"] = .99
        with self.assertRaisesRegex(ArtifactError, "inconsistent"):
            load_evaluation(self.write(metadata))

    def test_no_training_or_missing_provenance_allowed(self):
        for change in ({"split": "train"}, {"held_out": False}, {"test_set_id": ""}, {"evaluation_config": {}}):
            with self.subTest(change=change), self.assertRaises(ArtifactError):
                load_evaluation(self.write(dict(self.recorded(), **change)))

    def test_no_empty_or_malformed_matrix(self):
        for matrix in ([[0, 0], [0, 0]], [[1, 1.1], [1, 2]], [[-1, 2], [1, 2]], [[1, 2, 3], [1, 2, 3]]):
            with self.subTest(matrix=matrix), self.assertRaises(ArtifactError):
                load_evaluation(self.write(dict(self.recorded(), confusion_matrix=matrix)))

    def test_no_path_escape_or_duplicate_test_ids(self):
        with self.assertRaises(ArtifactError):
            load_evaluation(self.write(dict(self.metadata, predictions_file="../test.csv")))
        pd.DataFrame({"review_id": ["a", "a"], "y_true": [0, 1], "y_pred": [0, 1]}).to_csv(self.root / "test.csv", index=False)
        with self.assertRaises(ArtifactError):
            load_evaluation(self.write(dict(self.metadata, predictions_file="test.csv")))

    def test_prediction_join_preserves_missing_coverage_and_ids(self):
        reviews = pd.DataFrame({"review_id": ["001", "002"], "sentiment": ["Positive", "Negative"]})
        path = self.root / "predictions.csv"
        pd.DataFrame({"review_id": ["002"], "predicted_sentiment": [1], "confidence": [.7]}).to_csv(path, index=False)
        predictions = load_predictions(path)
        self.assertEqual(predictions.review_id.tolist(), ["002"])
        merged = attach_predictions(reviews, predictions)
        self.assertEqual(len(merged), 2)
        self.assertTrue(pd.isna(merged.loc[0, "predicted_sentiment"]))
        self.assertEqual(merged.loc[1, "predicted_sentiment"], "Positive")
        self.assertEqual(merged.sentiment.tolist(), ["Positive", "Negative"])

    def test_invalid_predictions_and_collision_rejected(self):
        reviews = pd.DataFrame({"review_id": ["a"], "sentiment": ["Negative"]})
        for predictions in (
            pd.DataFrame({"review_id": ["a", "a"], "predicted_sentiment": [0, 1]}),
            pd.DataFrame({"review_id": ["a"], "predicted_sentiment": ["neutral"]}),
            pd.DataFrame({"review_id": ["a"], "predicted_sentiment": [1], "confidence": [1.1]}),
            pd.DataFrame({"review_id": ["b"], "predicted_sentiment": [1]}),
        ):
            with self.subTest(), self.assertRaises(ArtifactError):
                attach_predictions(reviews, predictions)
        with self.assertRaisesRegex(ArtifactError, "overwrite"):
            attach_predictions(reviews.assign(predicted_sentiment="Positive"), pd.DataFrame({"review_id": ["a"], "predicted_sentiment": [1]}))

    def test_incompatible_baselines_do_not_compare(self):
        reference = load_evaluation(self.write(self.recorded()))
        same = load_evaluation(self.write(dict(self.recorded(), model_id="same"), "evaluation_same.json"))
        different_test = load_evaluation(self.write(dict(self.recorded(), model_id="other", test_set_id="another-test"), "evaluation_other.json"))
        different_config = load_evaluation(self.write(dict(self.recorded(), model_id="threshold", evaluation_config=dict(self.metadata["evaluation_config"], decision_threshold=.7)), "evaluation_threshold.json"))
        self.assertEqual([x.model_id for x in comparable_evaluations([reference, same, different_test, different_config], reference)], [reference.model_id, "same"])

    def test_bundle_surfaces_invalid_result_errors(self):
        self.write(dict(self.recorded(), split="training"))
        bundle = load_results(self.root)
        self.assertEqual(bundle.evaluations, [])
        self.assertEqual(len(bundle.errors), 1)
        self.assertIn("split='test'", bundle.errors[0])

    def test_all_negative_predictions_have_zero_binary_f1(self):
        metadata = dict(self.metadata, confusion_matrix=[[4, 0], [0, 0]], n_samples=4,
                        metrics={"accuracy": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        result = load_evaluation(self.write(metadata))
        self.assertEqual(result.metrics["f1"], 0.0)
        self.assertEqual(result.confusion_matrix, [[4, 0], [0, 0]])

    def test_malformed_model_manifest_types_are_reported(self):
        base = {"model_id": "test-only", "checkpoint": "recorded", "preprocessing_id": "original", "pooling": ["cls"]}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(base))
        with self.assertRaisesRegex(ModelIntegrationError, "pooling"):
            read_model_manifest(path)
        base.update(pooling="cls", max_length=True)
        path.write_text(json.dumps(base))
        with self.assertRaisesRegex(ModelIntegrationError, "max_length"):
            read_model_manifest(path)

    def test_model_manifest_requires_exact_configuration(self):
        path = self.root / "manifest.json"
        path.write_text('{}')
        with self.assertRaisesRegex(ModelIntegrationError, "model_id"):
            read_model_manifest(path)


if __name__ == "__main__":
    unittest.main()
