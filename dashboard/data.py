"""Read-only normalization and aggregations for existing review datasets.

These transformations prepare display data; they do not alter review text,
replace the project's preprocessing, rebalance classes, or modify source files.
Every source record is retained, including duplicates. Neutral/invalid ratings
and records explicitly excluded by preprocessing stay in ``records`` but are
absent from the binary-analysis ``reviews`` frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import pandas as pd

from dashboard.products import PRODUCT_NAMES


SENTIMENTS = ["Positive", "Negative"]
ALIASES = {
    "source_review_id": ["cmtid", "review_id", "comment_id"],
    "product_id": ["product_id", "item_id", "itemid"],
    "product_name": ["product_name", "item_name", "product_title"],
    "review_text": ["review_text", "comment", "customer_comment", "review"],
    "cleaned_text": ["cleaned_comment_v3", "cleaned_text", "cleaned_comment"],
    "rating": ["rating", "stars", "star_rating"],
    "review_date": ["review_date", "datetime", "date", "created_at"],
}


@dataclass
class Dataset:
    source: str
    records: pd.DataFrame
    reviews: pd.DataFrame
    quality: dict[str, int]


def _column(frame: pd.DataFrame, key: str) -> str | None:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    return next((lookup[a] for a in ALIASES[key] if a in lookup), None)


def load_dataset(path_or_buffer: str | Path | BinaryIO, source_name: str | None = None) -> Dataset:
    """Load a CSV with explicit schema mapping and rating-derived labels.

    Product ID, original review text, rating and review date columns are required.
    Values may be missing and are reported, rather than invented. Dates follow
    month/day/year for ambiguous slash dates, matching the supplied Shopee files.
    A unique source review ID is used directly; repeated IDs receive a stable
    source-row suffix. This supports unambiguous future prediction-file joins.
    """
    try:
        raw = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read review CSV: {exc}") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    mapping = {key: _column(raw, key) for key in ALIASES}
    missing = [key for key in ["product_id", "review_text", "rating", "review_date"] if mapping[key] is None]
    if missing:
        raise ValueError("Missing required review columns: " + ", ".join(missing) + ". Supported names include item_id, comment, stars and datetime.")

    frame = raw.copy()
    for key, source_col in mapping.items():
        frame[key] = raw[source_col].copy() if source_col else ""
    frame["product_id"] = frame["product_id"].str.strip()
    frame["source_product_name"] = frame["product_name"].copy()
    frame["product_name"] = frame["product_name"].str.strip()
    no_product = frame["product_id"].eq("")
    frame.loc[no_product, "product_id"] = "Unknown"
    no_name = frame["product_name"].eq("")
    frame.loc[no_name, "product_name"] = "Product " + frame.loc[no_name, "product_id"]
    frame.loc[no_product & no_name, "product_name"] = "Unknown product"
    confirmed_names = frame["product_id"].map(PRODUCT_NAMES)
    frame.loc[confirmed_names.notna(), "product_name"] = confirmed_names.loc[confirmed_names.notna()]

    source_ids = frame["source_review_id"].str.strip()
    duplicates = source_ids.ne("") & source_ids.duplicated(keep=False)
    frame["review_id"] = source_ids
    for index in frame.index[source_ids.eq("") | duplicates]:
        base = source_ids.at[index] or "row"
        frame.at[index, "review_id"] = f"{base}__row_{index + 1}"
    # Avoid collision with an existing literal ID matching a generated suffix.
    if frame["review_id"].duplicated().any():
        frame["review_id"] = [f"row_{i + 1}" for i in frame.index]

    rating = pd.to_numeric(frame["rating"], errors="coerce")
    frame["rating"] = rating
    valid_rating = rating.isin([1, 2, 3, 4, 5])
    neutral = rating.eq(3)
    frame["sentiment"] = pd.Series(None, index=frame.index, dtype=object)
    frame.loc[rating.isin([1, 2]), "sentiment"] = "Negative"
    frame.loc[rating.isin([4, 5]), "sentiment"] = "Positive"
    frame["sentiment_code"] = frame["sentiment"].map({"Negative": 0, "Positive": 1}).astype("Int64")

    date_strings = frame["review_date"].str.strip()
    frame["review_date"] = pd.to_datetime(date_strings, format="mixed", errors="coerce", dayfirst=False, utc=True).dt.tz_convert(None)
    missing_dates = date_strings.eq("")
    bad_dates = ~missing_dates & frame["review_date"].isna()
    if "preprocess_status" in raw:
        status = raw["preprocess_status"].str.strip().str.lower()
        excluded = status.ne("") & status.ne("keep")
    else:
        excluded = pd.Series(False, index=frame.index)
    frame["eligible"] = valid_rating & ~neutral & ~excluded
    existing_reason = raw["exclusion_reason"] if "exclusion_reason" in raw else pd.Series("", index=frame.index)
    frame["exclusion_reason"] = existing_reason
    frame.loc[~valid_rating, "exclusion_reason"] = "Invalid or missing star rating"
    frame.loc[neutral, "exclusion_reason"] = "Three-star review excluded from binary analysis"
    frame.loc[excluded & frame["exclusion_reason"].eq(""), "exclusion_reason"] = "Excluded by existing preprocessing status"

    supplied_labels = next((raw[c] for c in ["actual_sentiment", "sentiment", "sentiment_original"] if c in raw), None)
    mismatches = 0
    if supplied_labels is not None:
        supplied = supplied_labels.str.strip().str.lower().map({"positive": "Positive", "negative": "Negative", "1": "Positive", "0": "Negative"})
        frame["source_sentiment"] = supplied_labels
        mismatches = int((supplied.notna() & frame["sentiment"].notna() & supplied.ne(frame["sentiment"])).sum())

    quality = {
        "total_records": len(frame),
        "binary_records": int(frame["eligible"].sum()),
        "neutral_records": int(neutral.sum()),
        "invalid_rating_records": int((~valid_rating).sum()),
        "excluded_preprocessing_records": int(excluded.sum()),
        "missing_dates": int(missing_dates.sum()),
        "invalid_dates": int(bad_dates.sum()),
        "empty_reviews": int(frame["review_text"].str.strip().eq("").sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "repeated_review_ids": int((source_ids.ne("") & source_ids.duplicated()).sum()),
        "label_mismatches": mismatches,
        "missing_product_ids": int(no_product.sum()),
    }
    source = source_name or (str(path_or_buffer) if isinstance(path_or_buffer, (str, Path)) else getattr(path_or_buffer, "name", "Uploaded CSV"))
    return Dataset(source=str(source), records=frame, reviews=frame.loc[frame["eligible"]].copy(), quality=quality)


def filter_reviews(
    frame: pd.DataFrame,
    products: Sequence[str] | None = None,
    sentiments: Sequence[str] | None = None,
    start=None,
    end=None,
    include_undated: bool = True,
    product_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply inclusive calendar-date and categorical filters without mutation."""
    keep = pd.Series(True, index=frame.index)
    if products is not None:
        keep &= frame["product_id"].isin(products)
    if product_names is not None:
        keep &= frame["product_name"].isin(product_names)
    if sentiments is not None:
        keep &= frame["sentiment"].isin(sentiments)
    dates = frame["review_date"]
    if start is not None or end is not None:
        date_keep = dates.notna()
        if start is not None:
            date_keep &= dates.ge(pd.Timestamp(start).normalize())
        if end is not None:
            date_keep &= dates.lt(pd.Timestamp(end).normalize() + pd.Timedelta(days=1))
        if include_undated:
            date_keep |= dates.isna()
        keep &= date_keep
    elif not include_undated:
        keep &= dates.notna()
    return frame.loc[keep].copy()


def sentiment_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    positive = int(frame["sentiment"].eq("Positive").sum())
    negative = int(frame["sentiment"].eq("Negative").sum())
    total = positive + negative
    return {
        "total_reviews": total,
        "total_products": int(frame["product_name"].nunique()),
        "total_product_ids": int(frame["product_id"].nunique()),
        "positive_reviews": positive,
        "negative_reviews": negative,
        "positive_percentage": 100 * positive / total if total else 0.0,
        "negative_percentage": 100 * negative / total if total else 0.0,
    }


def product_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["product_name", "product_ids", "total_listing_ids", "total_reviews", "positive_reviews", "negative_reviews", "positive_percentage", "negative_percentage"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    counts = frame.groupby(["product_name", "sentiment"], dropna=False).size().unstack(fill_value=0)
    result = counts.reindex(columns=SENTIMENTS, fill_value=0).rename(columns={"Positive": "positive_reviews", "Negative": "negative_reviews"}).reset_index()
    listings = frame.groupby("product_name")["product_id"].agg(
        product_ids=lambda values: ", ".join(sorted(set(values))),
        total_listing_ids="nunique",
    ).reset_index()
    result = result.merge(listings, on="product_name", validate="one_to_one")
    result["total_reviews"] = result["positive_reviews"] + result["negative_reviews"]
    result["positive_percentage"] = result["positive_reviews"].div(result["total_reviews"]).mul(100)
    result["negative_percentage"] = result["negative_reviews"].div(result["total_reviews"]).mul(100)
    result.columns.name = None
    return result[columns].sort_values(["total_reviews", "product_name"], ascending=[False, True]).reset_index(drop=True)


def monthly_sentiment(frame: pd.DataFrame, frequency: str = "MS") -> pd.DataFrame:
    """Return observed-date counts, filling missing monthly periods with zero."""
    dated = frame.dropna(subset=["review_date"])
    if dated.empty:
        return pd.DataFrame(columns=["month", "sentiment", "count"])
    if frequency != "MS":
        raise ValueError("Only monthly frequency 'MS' is supported.")
    grouped = dated.groupby([pd.Grouper(key="review_date", freq=frequency), "sentiment"]).size()
    periods = pd.date_range(dated["review_date"].min().to_period("M").start_time, dated["review_date"].max().to_period("M").start_time, freq=frequency)
    full = pd.MultiIndex.from_product([periods, SENTIMENTS], names=["month", "sentiment"])
    return grouped.rename_axis(["month", "sentiment"]).reindex(full, fill_value=0).rename("count").reset_index()
