"""Minimal research display frames, separate from the original preprocessing data."""
from dashboard.data import Dataset

DASHBOARD_COLUMNS = ['review_id', 'product_id', 'product_name', 'review_text', 'rating', 'review_date', 'sentiment', 'eligible']


def dashboard_research_dataset(dataset: Dataset) -> Dataset:
    """Keep the cached display dataset free of unrelated source metadata.

    Two internal keys remain for grouping and future validated prediction joins;
    customer views/exports still apply the stricter public projection. The source
    CSV, original text, dates, eligibility and quality counts are unchanged.
    """
    return Dataset(source=dataset.source, records=dataset.records[DASHBOARD_COLUMNS].copy(),
                   reviews=dataset.reviews[DASHBOARD_COLUMNS].copy(), quality=dict(dataset.quality))


