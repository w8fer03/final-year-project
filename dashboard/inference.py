"""Module-independent future prediction boundary, disabled in the current UI.

A real trained predictor must be explicitly connected to this legacy service.
No environment switch, star label, keyword heuristic, or uploaded artifact enables
predictions. Original rows/text are retained one-to-one, including duplicates.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from dashboard.model_loader import ModelIntegrationError, PredictionBatch, load_trained_model
from dashboard.presentation import public_reviews

MODEL_PENDING_MESSAGE = 'Trained-model artifacts are not configured for this inference service.'


class ReviewPredictor(Protocol):
    def predict(self, reviews: Sequence[str], *, batch_size: int = 16) -> PredictionBatch: ...


class PredictionService:
    def __init__(self, predictor: ReviewPredictor | None = None):
        self._predictor = predictor

    @property
    def available(self) -> bool:
        return self._predictor is not None

    def predict_reviews(self, reviews: pd.DataFrame, *, batch_size: int = 16) -> pd.DataFrame:
        if not self.available:
            raise ModelIntegrationError(MODEL_PENDING_MESSAGE)
        if 'review_text' not in reviews or reviews.review_text.isna().any() or any(not isinstance(text, str) for text in reviews.review_text):
            raise ModelIntegrationError('Original review text must be supplied as strings for every row.')
        if type(batch_size) is not int or batch_size < 1:
            raise ModelIntegrationError('batch_size must be a positive integer.')
        batch = self._predictor.predict(reviews.review_text.tolist(), batch_size=batch_size)
        if not isinstance(batch, PredictionBatch):
            raise ModelIntegrationError('The trained predictor must return a PredictionBatch.')
        labels, confidence = batch.validate(len(reviews))
        # Discard untrusted CSV prediction columns; only this predictor supplies them.
        result = public_reviews(reviews.drop(columns=['predicted_sentiment', 'confidence'], errors='ignore'))
        result['predicted_label'] = labels
        result['predicted_sentiment'] = ['Positive' if label == 1 else 'Negative' for label in labels]
        if confidence is not None:
            result['confidence'] = confidence
        return result


def load_prediction_service(manifest_path, preprocessing_registry) -> PredictionService:
    """Legacy owner connection point; the active Version B UI uses its own backend."""
    return PredictionService(load_trained_model(manifest_path, preprocessing_registry))
