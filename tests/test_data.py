import io
import hashlib
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from dashboard.data import filter_reviews, load_dataset, monthly_sentiment, product_summary, sentiment_summary
from dashboard.products import PRODUCT_GROUPS, PRODUCT_NAMES


def csv_data(body: str):
    return io.StringIO("item_id,comment,stars,datetime,cmtid\n" + body)


class DataTests(unittest.TestCase):
    def test_labels_neutral_invalid_and_missing_date(self):
        dataset = load_dataset(csv_data(
            "12,bagus,5,1/31/2024 23:59,a\n"
            "12,rosak,1,2/1/2024,b\n"
            "13,okay,3,2/1/2024,c\n"
            "13,invalid,6,bad,d\n"
            "13,missing,4,,e\n"
        ))
        self.assertEqual(dataset.reviews.sentiment.tolist(), ["Positive", "Negative", "Positive"])
        self.assertEqual(dataset.quality["neutral_records"], 1)
        self.assertEqual(dataset.quality["invalid_rating_records"], 1)
        self.assertEqual(dataset.quality["invalid_dates"], 1)
        self.assertEqual(dataset.quality["missing_dates"], 1)
        self.assertEqual(len(dataset.records), 5)

    def test_filter_end_date_inclusive_and_undated_explicit(self):
        frame = load_dataset(csv_data("12,bagus,5,1/31/2024 23:59,a\n12,rosak,1,2/1/2024,b\n13,missing,4,,c\n")).reviews
        result = filter_reviews(frame, start=date(2024, 1, 1), end=date(2024, 1, 31), include_undated=False)
        self.assertEqual(result.review_id.tolist(), ["a"])
        self.assertEqual(len(filter_reviews(frame, end=date(2024, 1, 31), include_undated=True)), 2)
        self.assertTrue(filter_reviews(frame, products=[]).empty)
        self.assertTrue(filter_reviews(frame, sentiments=[]).empty)

    def test_duplicates_are_retained_and_identifiable(self):
        dataset = load_dataset(csv_data("12,bagus,5,1/1/2024,a\n12,bagus,5,1/1/2024,a\n12,rosak,2,1/2/2024,b\n"))
        self.assertEqual(len(dataset.reviews), 3)
        self.assertTrue(dataset.reviews.review_id.is_unique)
        self.assertEqual(dataset.quality["duplicate_rows"], 1)
        self.assertEqual(dataset.quality["repeated_review_ids"], 1)
        self.assertEqual(dataset.reviews.source_review_id.tolist(), ["a", "a", "b"])

    def test_aggregations_counts_percentages_and_missing_months(self):
        dataset = load_dataset(csv_data("12,bagus,5,1/1/2024,a\n12,rosak,2,3/2/2024,b\n13,great,4,3/4/2024,c\n"))
        summary = sentiment_summary(dataset.reviews)
        self.assertEqual(summary["total_reviews"], 3)
        self.assertAlmostEqual(summary["positive_percentage"], 200 / 3)
        products = product_summary(dataset.reviews)
        self.assertEqual(products.total_reviews.sum(), 3)
        self.assertTrue((products.positive_percentage + products.negative_percentage).eq(100).all())
        monthly = monthly_sentiment(dataset.reviews)
        self.assertEqual(monthly["count"].sum(), 3)
        self.assertEqual(len(monthly), 6)
        self.assertEqual(monthly.loc[monthly.month.dt.month.eq(2), "count"].sum(), 0)

    def test_preprocessing_status_and_label_mismatches(self):
        data = io.StringIO("item_id,comment,stars,datetime,cmtid,preprocess_status,sentiment,cleaned_comment_v3\n12,bagus,5,1/1/2024,a,keep,Negative,bagus\n13,rosak,1,1/1/2024,b,exclude,Negative,rosak\n")
        dataset = load_dataset(data)
        self.assertEqual(len(dataset.reviews), 1)
        self.assertEqual(dataset.reviews.sentiment.tolist(), ["Positive"])
        self.assertEqual(dataset.quality["label_mismatches"], 1)
        self.assertEqual(dataset.quality["excluded_preprocessing_records"], 1)
        self.assertEqual(dataset.reviews.cleaned_text.tolist(), ["bagus"])

    def test_empty_and_missing_columns(self):
        empty = load_dataset(csv_data(""))
        self.assertTrue(empty.reviews.empty)
        self.assertEqual(sentiment_summary(empty.reviews)["total_reviews"], 0)
        self.assertTrue(product_summary(empty.reviews).empty)
        self.assertTrue(monthly_sentiment(empty.reviews).empty)
        with self.assertRaisesRegex(ValueError, "Missing required review columns"):
            load_dataset(io.StringIO("review,rating\nbagus,5\n"))

    def test_bundled_cleaned_dataset_actual_totals(self):
        path = Path(__file__).resolve().parents[1] / "data" / "dataset_preprocessed_v3_clean.csv"
        if not path.exists():
            self.skipTest("Bundled uploaded dataset not present")
        dataset = load_dataset(path)
        summary = sentiment_summary(dataset.reviews)
        self.assertEqual(summary["total_reviews"], 19295)
        self.assertEqual(summary["total_products"], 5)
        self.assertEqual(summary["total_product_ids"], 17)
        self.assertEqual(summary["positive_reviews"], 19013)
        self.assertEqual(summary["negative_reviews"], 282)
        self.assertTrue(dataset.reviews.review_id.is_unique)
        self.assertEqual(dataset.quality["invalid_dates"], 0)

    def test_user_confirmed_groups_map_all_seventeen_listing_ids(self):
        self.assertEqual(len(PRODUCT_GROUPS), 5)
        self.assertEqual(len(PRODUCT_NAMES), 17)
        self.assertEqual(sum(map(len, PRODUCT_GROUPS.values())), 17)
        rows = "".join(f"{listing_id},bagus,5,1/1/2024,review-{i}\n" for i, listing_id in enumerate(PRODUCT_NAMES))
        dataset = load_dataset(csv_data(rows))
        self.assertEqual(set(dataset.reviews.product_id), set(PRODUCT_NAMES))
        self.assertEqual(set(dataset.reviews.product_name), set(PRODUCT_GROUPS))
        for row in dataset.reviews.itertuples():
            self.assertEqual(row.product_name, PRODUCT_NAMES[row.product_id])
        summary = sentiment_summary(dataset.reviews)
        self.assertEqual(summary["total_products"], 5)
        self.assertEqual(summary["total_product_ids"], 17)
        for name, ids in PRODUCT_GROUPS.items():
            filtered = filter_reviews(dataset.reviews, product_names=[name])
            self.assertEqual(set(filtered.product_id), set(ids))
            intersected = filter_reviews(dataset.reviews, product_names=[name], products=[ids[0]])
            self.assertEqual(intersected.product_id.tolist(), [ids[0]])
        self.assertTrue(filter_reviews(dataset.reviews, product_names=[]).empty)

    def test_unknown_ids_keep_names_and_known_ids_use_confirmed_names(self):
        data = io.StringIO("item_id,product_name,comment,stars,datetime,cmtid\n14750761025,Old source name,bagus,5,1/1/2024,a\nnew-1,Custom source name,good,4,1/1/2024,b\nnew-2,,rosak,1,1/1/2024,c\n")
        dataset = load_dataset(data)
        rows = dataset.reviews.set_index("product_id")
        self.assertEqual(rows.loc["14750761025", "product_name"], "Baseus H17 Encok 3.5mm Wired Earphone")
        self.assertEqual(rows.loc["14750761025", "source_product_name"], "Old source name")
        self.assertEqual(rows.loc["new-1", "product_name"], "Custom source name")
        self.assertEqual(rows.loc["new-2", "product_name"], "Product new-2")

    def test_bundled_group_counts_ids_and_source_files_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        paths = {
            "dataset_preprocessed_v3_clean.csv": "6164d277eece96037fb4d47972d08aed2e7b67c1eef88d8383aeaf0cc3c80ae0",
            "dataset v3 labelled 2.csv": "efd8342adf02d7170668176f32c11cbd06b2853ef329d5d35cf21023006e88cd",
        }
        if not all((root / "data" / name).exists() for name in paths):
            self.skipTest("Provide both local review CSVs for dataset provenance checks")
        for name, digest in paths.items():
            path = root / "data" / name
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            dataset = load_dataset(path)
            self.assertEqual(before, digest)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
            self.assertEqual(dataset.records.product_id.tolist(), raw.item_id.tolist())
            self.assertEqual(dataset.records.source_review_id.tolist(), raw.cmtid.tolist())
            # Stable row IDs predate product grouping and do not depend on names.
            repeated = raw.cmtid.duplicated(keep=False)
            expected_ids = [f"{value}__row_{i + 1}" if repeated.iloc[i] else value for i, value in enumerate(raw.cmtid)]
            self.assertEqual(dataset.records.review_id.tolist(), expected_ids)
            self.assertEqual(set(raw.item_id), set(PRODUCT_NAMES))
        dataset = load_dataset(root / "data" / "dataset_preprocessed_v3_clean.csv")
        groups = product_summary(dataset.reviews).set_index("product_name")
        expected = {
            "Baseus H17 Encok 3.5mm Wired Earphone": (2251, 2213, 38),
            "Baseus C17 Type-C Earphones": (4338, 4232, 106),
            "Baseus Encok CZ11 / HZ11": (3068, 3002, 66),
            "Baseus WM01 TWS Wireless Earphones": (5415, 5395, 20),
            "Baseus H19": (4223, 4171, 52),
        }
        self.assertEqual(len(groups), 5)
        self.assertEqual(groups.total_reviews.sum(), 19295)
        for name, counts in expected.items():
            row = groups.loc[name]
            self.assertEqual(tuple(row[["total_reviews", "positive_reviews", "negative_reviews"]]), counts)
            self.assertEqual(set(row.product_ids.split(", ")), set(PRODUCT_GROUPS[name]))
            self.assertEqual(row.total_listing_ids, len(PRODUCT_GROUPS[name]))
            self.assertAlmostEqual(row.positive_percentage + row.negative_percentage, 100)


if __name__ == "__main__":
    unittest.main()
