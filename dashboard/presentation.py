"""Customer-facing projections that keep source identifiers internal.

Internal review frames retain listing/review IDs for provenance and prediction
joins. Views and customer CSVs use this explicit allowlist instead. Review text
and preprocessing outputs are never rewritten by the presentation layer.
"""

from __future__ import annotations

import pandas as pd

from dashboard.products import PRODUCT_GROUPS


PUBLIC_REVIEW_COLUMNS = [
    "product_name", "review_text", "rating", "review_date", "sentiment",
]
OPTIONAL_PREDICTION_COLUMNS = ["predicted_sentiment", "confidence"]


def public_product_name(name, product_id=None) -> str:
    """Keep supplied names; replace missing or identifier-derived fallback labels.

    Pass the internal listing ID when available so a literal ID cannot become a
    customer label. Confirmed product names always retain their exact spelling.
    """
    if name is None or pd.isna(name):
        return "Other product"
    label = str(name).strip()
    if label in PRODUCT_GROUPS:
        return label
    if not label or label.casefold().startswith(("product ", "unknown product")):
        return "Other product"
    if product_id is not None and not pd.isna(product_id):
        identifier = str(product_id).strip()
        if identifier and identifier in label:
            return "Other product"
    # A bare numeric listing ID is also unsafe when no ID argument is supplied.
    if label.isdecimal():
        return "Other product"
    return label


def public_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the review fields suitable for customer tables and CSV downloads.

    Optional prediction columns must originate from the validated result loader;
    this projection does not establish model provenance. Missing fields become
    empty values so an empty selection still has a stable public CSV schema.
    The index is reset because source identifiers may also be stored in an index.
    """
    names = frame["product_name"] if "product_name" in frame else pd.Series(None, index=frame.index, dtype=object)
    ids = frame["product_id"] if "product_id" in frame else pd.Series(None, index=frame.index, dtype=object)
    output = frame.reindex(columns=PUBLIC_REVIEW_COLUMNS).copy()
    output["product_name"] = [public_product_name(name, listing_id) for name, listing_id in zip(names, ids)]
    for column in OPTIONAL_PREDICTION_COLUMNS:
        if column in frame:
            output[column] = frame[column].copy()
    return output.reset_index(drop=True)
