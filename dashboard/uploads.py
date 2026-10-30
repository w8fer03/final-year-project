"""In-memory descriptive analysis for user uploads, separate from research inputs.

Only original review text is required. Optional ratings provide proxy labels;
missing ratings never cause text sentiment to be inferred. All source rows stay
in records, including duplicates, unrated rows and excluded three-star rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import io

import pandas as pd

from dashboard.data import ALIASES, Dataset
from dashboard.presentation import public_product_name, public_reviews
from dashboard.products import PRODUCT_NAMES


@dataclass
class UploadedDataset(Dataset):
    has_ratings: bool
    has_dates: bool
    has_products: bool


def load_uploaded_dataset(content: bytes, filename: str) -> UploadedDataset:
    if not filename.lower().endswith(".csv"):
        raise ValueError("Choose a CSV file ending in .csv.")
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("The CSV is too large. Please use a file of 50 MB or less.")
    try:
        rows = csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        header = next(rows, [])
        columns = [name.strip().casefold() for name in header]
        if len(set(columns)) != len(columns):
            raise ValueError("Column names must be unique, ignoring spaces and letter case.")
        if any(len(row) > len(header) for row in rows):
            raise ValueError("Some CSV rows have more fields than the header. Put comments containing commas inside double quotes.")
        raw = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8-sig", skip_blank_lines=False)
    except (pd.errors.EmptyDataError, UnicodeError, pd.errors.ParserError, csv.Error) as exc:
        raise ValueError("This file could not be read as a UTF-8 CSV. Check its encoding, commas, and quoted fields.") from exc
    raw.columns = [str(c).strip() for c in raw.columns]
    normalized = [str(c).casefold() for c in raw.columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Column names must be unique, ignoring spaces and letter case.")
    lookup = dict(zip(normalized, raw.columns))
    mapping = {key: next((lookup[a] for a in aliases if a in lookup), None) for key, aliases in ALIASES.items()}
    if mapping["review_text"] is None:
        raise ValueError("A review text column is required. Use review_text, comment, customer_comment, or review. Ratings, dates, and products are optional.")
    if raw.empty:
        raise ValueError("The CSV contains column headers but no review records.")
    # Select the few display fields; arbitrary uploaded predictions and cleaned
    # fields are neither trusted nor retained in this session's analysis frame.
    frame = pd.DataFrame(index=raw.index)
    frame["review_text"] = raw[mapping["review_text"]].copy()
    frame["product_id"] = raw[mapping["product_id"]].str.strip() if mapping["product_id"] else ""
    names = raw[mapping["product_name"]].str.strip() if mapping["product_name"] else pd.Series("", index=raw.index)
    safe_names = []
    anonymous = {}
    for name, product_id in zip(names, frame.product_id):
        if product_id in PRODUCT_NAMES:
            safe_names.append(PRODUCT_NAMES[product_id])
        else:
            safe = public_product_name(name, product_id)
            if safe == "Other product":
                if product_id:
                    safe = anonymous.setdefault(product_id, f"Uploaded product {len(anonymous) + 1}")
                else:
                    safe = "Unspecified product"
            safe_names.append(safe)
    frame["product_name"] = safe_names
    frame["rating"] = pd.to_numeric(raw[mapping["rating"]], errors="coerce") if mapping["rating"] else float("nan")
    date_strings = raw[mapping["review_date"]].str.strip() if mapping["review_date"] else pd.Series("", index=raw.index)
    frame["review_date"] = pd.to_datetime(date_strings, format="mixed", errors="coerce", dayfirst=False, utc=True).dt.tz_convert(None)
    frame["sentiment"] = pd.Series(None, index=raw.index, dtype=object)
    frame.loc[frame.rating.isin([1, 2]), "sentiment"] = "Negative"
    frame.loc[frame.rating.isin([4, 5]), "sentiment"] = "Positive"
    excluded = raw.preprocess_status.str.strip().str.casefold().isin(["", "keep"]).eq(False) if "preprocess_status" in raw else pd.Series(False, index=raw.index)
    frame["eligible"] = frame.rating.isin([1, 2, 4, 5]) & ~excluded
    # Existing exclusions also exclude labels from displayed binary statistics.
    frame.loc[excluded, "sentiment"] = None
    quality = {
        "total_records": len(frame), "binary_records": int(frame.eligible.sum()),
        "neutral_records": int(frame.rating.eq(3).sum()),
        "invalid_rating_records": int((~frame.rating.isin([1, 2, 3, 4, 5])).sum()),
        "excluded_preprocessing_records": int(excluded.sum()),
        "missing_dates": int(date_strings.eq("").sum()),
        "invalid_dates": int((date_strings.ne("") & frame.review_date.isna()).sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
    }
    return UploadedDataset(source=filename, records=frame, reviews=frame[frame.eligible].copy(), quality=quality,
                           has_ratings=mapping["rating"] is not None,
                           has_dates=mapping["review_date"] is not None,
                           has_products=bool(frame.product_id.ne("").any() or names.ne("").any()))


def uploaded_public_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    # Always omit arbitrary prediction columns, even if a future caller passes them.
    return public_reviews(frame.drop(columns=["predicted_sentiment", "confidence"], errors="ignore"))
