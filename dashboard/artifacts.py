"""Validated, opt-in result-file integration; never generates model predictions.

Evaluation files must describe an explicitly held-out test set. Rating-derived
labels in the exploration dataset are never interpreted as predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


class ArtifactError(ValueError):
    """An artifact does not satisfy the documented result contract."""


@dataclass(frozen=True)
class EvaluationResult:
    model_id: str
    model_name: str
    test_set_id: str
    split: str
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    n_samples: int
    averaging: str
    positive_label: int
    metadata: dict[str, Any]
    source: str = ""


@dataclass
class ResultsBundle:
    evaluations: list[EvaluationResult] = field(default_factory=list)
    predictions: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)


def _label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or pd.isna(value):
        raise ArtifactError("Sentiment labels must be 0/1 or Negative/Positive, with no missing values.")
    key = str(value).strip().casefold()
    choices = {"negative": 0, "positive": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1}
    if key not in choices:
        raise ArtifactError(f"Invalid sentiment label: {value!r}.")
    return choices[key]


def _labels(values: pd.Series) -> np.ndarray:
    return np.array([_label(value) for value in values], dtype=int)


def _identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "review_id" not in frame:
        raise ArtifactError("Prediction CSV requires canonical review_id values exported by the dashboard.")
    ids = frame["review_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ArtifactError("review_id must be nonempty and unique in every prediction file.")
    frame["review_id"] = ids
    return frame


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load real saved predictions without attempting inference or deriving labels."""
    frame = _identifiers(pd.read_csv(path, dtype={"review_id": "string"}))
    if frame.empty:
        raise ArtifactError("Prediction file is empty.")
    if "predicted_sentiment" not in frame:
        raise ArtifactError("Prediction CSV requires predicted_sentiment.")
    codes = _labels(frame["predicted_sentiment"])
    output = frame[["review_id"]].copy()
    output["predicted_sentiment"] = np.where(codes == 1, "Positive", "Negative")
    if "confidence" in frame:
        confidence = pd.to_numeric(frame["confidence"], errors="coerce")
        if confidence.isna().any() or not confidence.between(0, 1).all():
            raise ArtifactError("confidence must contain genuine numeric probabilities in [0, 1].")
        output["confidence"] = confidence
    return output


def attach_predictions(reviews: pd.DataFrame, predictions: pd.DataFrame | None) -> pd.DataFrame:
    """Join by canonical review_id, preserving all rows and showing missing coverage.

    The review ID, not a row number or comment text, controls the match. Extra
    prediction IDs may belong to source rows excluded by dashboard filters.
    """
    if predictions is None:
        return reviews.copy()
    predictions = _identifiers(predictions)
    if "predicted_sentiment" not in predictions:
        raise ArtifactError("Predictions require predicted_sentiment.")
    codes = _labels(predictions["predicted_sentiment"])
    predictions["predicted_sentiment"] = np.where(codes == 1, "Positive", "Negative")
    allowed = ["review_id", "predicted_sentiment"]
    if "confidence" in predictions:
        values = pd.to_numeric(predictions["confidence"], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ArtifactError("confidence must contain genuine numeric probabilities in [0, 1].")
        predictions["confidence"] = values
        allowed.append("confidence")
    collisions = (set(allowed) - {"review_id"}) & set(reviews.columns)
    if collisions:
        raise ArtifactError("Prediction join would overwrite existing columns: " + ", ".join(sorted(collisions)))
    if "review_id" not in reviews:
        raise ArtifactError("Review data requires canonical review_id.")
    if not reviews.empty and not reviews["review_id"].astype("string").isin(predictions["review_id"]).any():
        raise ArtifactError("No prediction IDs match this dataset. Export review_id from the selected dataset first.")
    return reviews.merge(predictions[allowed], on="review_id", how="left", validate="many_to_one", sort=False)


def _required_text(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"Evaluation requires a nonempty {key}.")
    return value.strip()


def _metrics_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = matrix.ravel()
    total = int(matrix.sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {"accuracy": float((tn + tp) / total), "precision": precision, "recall": recall, "f1": f1}


def _test_file(base: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename.strip():
        raise ArtifactError("predictions_file must name a local CSV within the results directory.")
    target = (base / filename).resolve()
    if not target.is_relative_to(base.resolve()) or target.suffix.lower() != ".csv":
        raise ArtifactError("predictions_file must be a CSV within the results directory.")
    return target


def load_evaluation(path: str | Path) -> EvaluationResult:
    """Validate test predictions or cross-check recorded metrics against their CM.

    The contract fixes binary averaging, positive label 1, and zero_division 0.
    Additional evaluation_config fields are preserved and must match for baseline
    comparison (for example thresholds or repeated-evaluation settings).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ArtifactError("Evaluation JSON must contain an object.")
    model_id = _required_text(data, "model_id")
    model_name = _required_text(data, "model_name")
    test_set_id = _required_text(data, "test_set_id")
    if data.get("split") != "test" or data.get("held_out") is not True:
        raise ArtifactError("Evaluation must explicitly declare split='test' and held_out=true.")
    config = data.get("evaluation_config")
    if not isinstance(config, dict) or config.get("averaging") != "binary" or type(config.get("positive_label")) is not int or config.get("positive_label") != 1 or type(config.get("zero_division")) is not int or config.get("zero_division") != 0:
        raise ArtifactError("evaluation_config must explicitly set averaging='binary', positive_label=1, zero_division=0.")
    if "predictions_file" in data:
        frame = _identifiers(pd.read_csv(_test_file(path.parent, data["predictions_file"]), dtype={"review_id": "string"}))
        if frame.empty or not {"y_true", "y_pred"}.issubset(frame.columns):
            raise ArtifactError("Test CSV requires nonempty review_id, y_true, and y_pred columns.")
        true, pred = _labels(frame["y_true"]), _labels(frame["y_pred"])
        matrix = confusion_matrix(true, pred, labels=[0, 1])
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="binary", pos_label=1, zero_division=0)
        metrics = {"accuracy": float(accuracy_score(true, pred)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
        n_samples = len(frame)
        if "n_samples" in data and (type(data["n_samples"]) is not int or data["n_samples"] != n_samples):
            raise ArtifactError("Recorded n_samples does not match the test prediction rows.")
        if "confusion_matrix" in data and data["confusion_matrix"] != matrix.tolist():
            raise ArtifactError("Recorded confusion_matrix does not match the test predictions.")
        if "metrics" in data:
            supplied_metrics = data["metrics"]
            if not isinstance(supplied_metrics, dict):
                raise ArtifactError("Recorded metrics must contain an object.")
            for name, value in metrics.items():
                supplied = supplied_metrics.get(name)
                if isinstance(supplied, bool) or not isinstance(supplied, Real) or not math.isfinite(supplied) or not math.isclose(supplied, value, abs_tol=0.00005):
                    raise ArtifactError(f"Recorded {name} does not match the test predictions.")
        # Preserve the test membership for stronger comparisons when available.
        data = dict(data, test_review_ids=sorted(frame["review_id"].tolist()))
    else:
        raw_matrix = data.get("confusion_matrix")
        if not isinstance(raw_matrix, list) or len(raw_matrix) != 2 or any(not isinstance(row, list) or len(row) != 2 for row in raw_matrix):
            raise ArtifactError("Recorded evaluation requires a 2x2 confusion_matrix in [[TN, FP], [FN, TP]] order.")
        if any(type(value) is not int or value < 0 for row in raw_matrix for value in row):
            raise ArtifactError("Confusion matrix counts must be nonnegative integers.")
        matrix = np.array(raw_matrix, dtype=np.int64)
        n_samples = data.get("n_samples")
        if type(n_samples) is not int or n_samples <= 0 or n_samples != int(matrix.sum()):
            raise ArtifactError("n_samples must be positive and equal the confusion matrix total.")
        metrics = data.get("metrics")
        if not isinstance(metrics, dict) or not {"accuracy", "precision", "recall", "f1"}.issubset(metrics):
            raise ArtifactError("Recorded evaluation requires accuracy, precision, recall, and f1 metrics.")
        expected = _metrics_from_matrix(matrix)
        for key, value in expected.items():
            supplied = metrics[key]
            if isinstance(supplied, bool) or not isinstance(supplied, Real) or not math.isfinite(supplied) or not 0 <= supplied <= 1 or not math.isclose(float(supplied), value, abs_tol=0.00005):
                raise ArtifactError(f"Recorded {key} is invalid or inconsistent with the confusion matrix. Use at least five decimal places.")
        metrics = {key: float(metrics[key]) for key in expected}
    return EvaluationResult(model_id, model_name, test_set_id, "test", metrics, matrix.tolist(), n_samples, "binary", 1, data, str(path))


def comparable_evaluations(evaluations: list[EvaluationResult], reference: EvaluationResult | None = None) -> list[EvaluationResult]:
    """Return results sharing declared test membership and all evaluation settings.

    Compare relative to the selected model, never pick the best score as reference.
    When both files expose test IDs, their exact membership must also agree.
    """
    if not evaluations:
        return []
    reference = reference or evaluations[0]
    def compatible(candidate: EvaluationResult) -> bool:
        settings = candidate.metadata.get("evaluation_config") == reference.metadata.get("evaluation_config")
        same = candidate.test_set_id == reference.test_set_id and candidate.n_samples == reference.n_samples and settings
        a, b = candidate.metadata.get("test_review_ids"), reference.metadata.get("test_review_ids")
        return same and (a is None or b is None or a == b)
    return [item for item in evaluations if compatible(item)]


def load_results(results_dir: str | Path, *, include_evaluations: bool = True) -> ResultsBundle:
    """Discover evaluation*.json and predictions.csv; invalid files remain visible.

    Missing artifacts are an ordinary empty state. A malformed existing artifact
    is reported rather than replaced with sample values.
    """
    directory = Path(results_dir)
    bundle = ResultsBundle()
    if not directory.exists():
        return bundle
    if not directory.is_dir():
        bundle.errors.append("Results path must be a directory.")
        return bundle
    for path in sorted(directory.glob("evaluation*.json")) if include_evaluations else []:
        try:
            result = load_evaluation(path)
            if any(old.model_id == result.model_id for old in bundle.evaluations):
                raise ArtifactError(f"Duplicate model_id {result.model_id!r}; keep one evaluation file per model.")
            bundle.evaluations.append(result)
        except (ArtifactError, OSError, ValueError, TypeError, OverflowError) as exc:
            bundle.errors.append(f"{path.name}: {exc}")
    predictions = directory / "predictions.csv"
    if predictions.exists():
        try:
            bundle.predictions = load_predictions(predictions)
        except (ArtifactError, OSError, ValueError, TypeError) as exc:
            bundle.errors.append(f"predictions.csv: {exc}")
    return bundle
