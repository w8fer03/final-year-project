"""Temporary CSV fixtures validate uploads; they are never dashboard findings."""
import io
import unittest

from dashboard.uploads import load_uploaded_dataset, uploaded_public_reviews


class UploadDataTests(unittest.TestCase):
    def test_text_only_retains_every_row_without_inventing_sentiment(self):
        data = load_uploaded_dataset(b'review\nbagus\nrosak\n""\n', "text.csv")
        self.assertEqual(len(data.records), 3)
        self.assertEqual(data.records.review_text.tolist(), ["bagus", "rosak", ""])
        self.assertTrue(data.reviews.empty)
        self.assertTrue(data.records.sentiment.isna().all())
        self.assertFalse(data.has_ratings)
        self.assertFalse(data.has_dates)
        self.assertFalse(data.has_products)

    def test_optional_columns_and_binary_denominator(self):
        csv = b'product_name,review,stars,date\nEarphone A,bagus,5,2024-01-01\nEarphone A,rosak,1,2024-02-01\nEarphone B,biasa,3,\nEarphone B,unknown,6,bad\nEarphone B,missing,,\n'
        data = load_uploaded_dataset(csv, "ratings.csv")
        self.assertEqual(len(data.records), 5)
        self.assertEqual(len(data.reviews), 2)
        self.assertEqual(data.reviews.sentiment.tolist(), ["Positive", "Negative"])
        self.assertTrue(data.records.sentiment.iloc[2:].isna().all())
        self.assertEqual(data.quality["neutral_records"], 1)
        self.assertEqual(data.quality["invalid_rating_records"], 2)
        self.assertEqual(data.quality["invalid_dates"], 1)
        self.assertTrue(data.has_products and data.has_ratings and data.has_dates)

    def test_distinct_anonymous_products_use_safe_labels(self):
        data = load_uploaded_dataset(b'item_id,comment,stars\n987654321,good,5\n123456789,bad,1\n987654321,okay,4\n', "ids.csv")
        self.assertEqual(data.records.product_name.tolist(), ["Uploaded product 1", "Uploaded product 2", "Uploaded product 1"])
        public = uploaded_public_reviews(data.records).to_csv(index=False)
        self.assertNotIn("987654321", public)
        self.assertNotIn("123456789", public)
        self.assertNotIn("product_id", public)

    def test_known_mapping_duplicates_and_dates_preserved(self):
        data = load_uploaded_dataset(b'item_id,comment,stars,datetime\n14750761025,good,5,2020-07-14\n14750761025,good,5,2020-07-14\n', "baseus.csv")
        self.assertEqual(len(data.records), 2)
        self.assertEqual(data.quality["duplicate_rows"], 1)
        self.assertEqual(data.records.product_name.iloc[0], "Baseus H17 Encok 3.5mm Wired Earphone")
        self.assertEqual(str(data.records.review_date.min().date()), "2020-07-14")

    def test_untrusted_predictions_and_cleaned_text_are_not_used(self):
        csv = b'review,stars,predicted_sentiment,confidence,cleaned_text\nbagus,1,Positive,.99,PRIVATE\n'
        data = load_uploaded_dataset(csv, "columns.csv")
        self.assertEqual(data.records.sentiment.tolist(), ["Negative"])
        public = uploaded_public_reviews(data.records)
        self.assertEqual(public.columns.tolist(), ["product_name", "review_text", "rating", "review_date", "sentiment"])
        self.assertNotIn("PRIVATE", public.to_csv(index=False))
        self.assertNotIn("predicted_sentiment", data.records)

    def test_existing_preprocessing_exclusions_are_respected(self):
        data = load_uploaded_dataset(b'review,stars,preprocess_status\ngood,5,exclude\nbad,1,keep\n', "excluded.csv")
        self.assertEqual(len(data.records), 2)
        self.assertEqual(data.reviews.sentiment.tolist(), ["Negative"])
        self.assertTrue(data.records.sentiment.iloc[:1].isna().all())

    def test_case_insensitive_column_aliases(self):
        data = load_uploaded_dataset(b' COMMENT ,STAR_RATING\ngood,4\n', "case.csv")
        self.assertEqual(data.reviews.sentiment.tolist(), ["Positive"])

    def test_duplicate_header_names_are_rejected_before_pandas_can_mangle_them(self):
        for header in [b'review,review', b'review, REVIEW ']:
            with self.assertRaisesRegex(ValueError, 'Column names must be unique'):
                load_uploaded_dataset(header + b'\ngood,bad\n', 'duplicate.csv')

    def test_invalid_csv_errors_are_actionable(self):
        cases = [(b'', 'empty.csv', 'UTF-8 CSV'), (b'product_name\nA\n', 'missing.csv', 'review text column'),
                 (b'review\n', 'headers.csv', 'no review records'), (b'\xff', 'encoding.csv', 'UTF-8 CSV'),
                 (b'review,stars\n"unclosed,5\n', 'malformed.csv', 'UTF-8 CSV'),
                 (b'review,stars\nhello,world,5\n', 'extra-fields.csv', 'more fields than the header'),
                 (b'review\ngood\n', 'wrong.txt', '.csv')]
        for content, filename, expected in cases:
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, expected):
                load_uploaded_dataset(content, filename)


if __name__ == '__main__':
    unittest.main()
