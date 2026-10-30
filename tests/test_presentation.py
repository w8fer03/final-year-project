"""Customer output excludes source identifiers and internal cleaned text."""

import io
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from dashboard.presentation import PUBLIC_REVIEW_COLUMNS, public_product_name, public_reviews
from dashboard.products import PRODUCT_GROUPS


class PresentationTests(unittest.TestCase):
    def test_customer_csv_and_table_exclude_identifier_columns_and_index(self):
        source = pd.DataFrame({
            "review_id": ["internal-review-a", "internal-review-b"],
            "source_review_id": ["source-review-a", "source-review-b"],
            "product_id": ["14750761025", "98765432101"],
            "shop_id": ["internal-shop", "internal-shop"],
            "cmtid": ["source-review-a", "source-review-b"],
            "product_name": ["Baseus H17 Encok 3.5mm Wired Earphone", "Product 98765432101"],
            "review_text": ["Sound bagus", "Rosak"],
            "cleaned_text": ["INTERNAL_CLEANED_SENTINEL_A", "INTERNAL_CLEANED_SENTINEL_B"],
            "cleaned_comment_v3": ["INTERNAL_V3_SENTINEL_A", "INTERNAL_V3_SENTINEL_B"],
            "rating": [5, 1],
            "review_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "sentiment": ["Positive", "Negative"],
            "unrelated_internal_diagnostic": ["private-a", "private-b"],
        }).set_index("review_id", drop=False)
        before = source.copy(deep=True)
        customer = public_reviews(source)
        self.assertEqual(PUBLIC_REVIEW_COLUMNS, ["product_name", "review_text", "rating", "review_date", "sentiment"])
        self.assertEqual(customer.columns.tolist(), PUBLIC_REVIEW_COLUMNS)
        self.assertEqual(customer.index.tolist(), [0, 1])
        self.assertIsNone(customer.index.name)
        self.assertEqual(customer.product_name.tolist(), ["Baseus H17 Encok 3.5mm Wired Earphone", "Other product"])
        csv = customer.to_csv(index=False)
        for identifier in ["14750761025", "98765432101", "internal-review-a", "source-review-a", "internal-shop", "private-a", "cleaned_text", "cleaned_comment_v3", "INTERNAL_CLEANED_SENTINEL", "INTERNAL_V3_SENTINEL"]:
            self.assertNotIn(identifier, csv)
        exported = pd.read_csv(io.StringIO(csv))
        self.assertEqual(exported.columns.tolist(), PUBLIC_REVIEW_COLUMNS)
        self.assertEqual(exported.review_text.tolist(), source.review_text.tolist())
        assert_frame_equal(source, before)

    def test_known_names_preserved_unknown_identifier_fallbacks_hidden(self):
        for name, listing_ids in PRODUCT_GROUPS.items():
            for listing_id in listing_ids:
                self.assertEqual(public_product_name(name, listing_id), name)
        for label, listing_id in [
            ("Product 123456789", "123456789"),
            ("Product new-listing", "new-listing"),
            ("Unknown product", None),
            ("Unknown product 123456789", "123456789"),
            ("123456789", None),
            ("Raw listing 123456789", "123456789"),
            (None, None), (pd.NA, None), ("", None),
        ]:
            with self.subTest(label=label):
                self.assertEqual(public_product_name(label, listing_id), "Other product")
        self.assertEqual(public_product_name("Owner-supplied earphones", "new-listing"), "Owner-supplied earphones")

    def test_validated_predictions_remain_optional_and_unchanged(self):
        source = pd.DataFrame({"product_name": ["Baseus H19"], "review_text": ["bagus"],
                               "sentiment": ["Positive"], "product_id": ["25502231352"]})
        self.assertEqual(public_reviews(source).columns.tolist(), PUBLIC_REVIEW_COLUMNS)
        self.assertNotIn("cleaned_text", public_reviews(source).columns)
        predictions = source.assign(predicted_sentiment="Negative", confidence=.71)
        projected = public_reviews(predictions)
        self.assertEqual(projected.columns.tolist(), PUBLIC_REVIEW_COLUMNS + ["predicted_sentiment", "confidence"])
        self.assertEqual(projected.predicted_sentiment.tolist(), ["Negative"])
        self.assertEqual(projected.confidence.tolist(), [.71])

    def test_empty_frames_have_stable_public_schema(self):
        empty = public_reviews(pd.DataFrame(columns=["review_id", "source_review_id", "product_id"]))
        self.assertEqual(empty.columns.tolist(), PUBLIC_REVIEW_COLUMNS)
        self.assertTrue(empty.empty)


if __name__ == "__main__":
    unittest.main()
