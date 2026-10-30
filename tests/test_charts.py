"""Chart contract tests; all review counts below are synthetic test fixtures."""
import unittest

import pandas as pd

from dashboard.charts import product_comparison, rating_imbalance, review_length_chart


class OverviewAnalysisTests(unittest.TestCase):
    def test_lengths_count_original_words_and_retain_empty_comments(self):
        reviews = pd.DataFrame({"review_text": ["bagus sangat", "good  sound", "", None, "satu\nlagi\tperkataan"],
                                "cleaned_text": ["PRIVATE_PREPROCESSING_VALUE"] * 5})
        fig = review_length_chart(reviews)
        self.assertEqual(list(fig.data[0].x), [0, 2, 3])
        self.assertEqual(list(fig.data[0].y), [2, 2, 1])
        for private in ["bagus sangat", "PRIVATE_PREPROCESSING_VALUE", "cleaned_text"]:
            self.assertNotIn(private, fig.to_json())
        self.assertEqual(reviews.review_text.iloc[0], "bagus sangat")

    def test_rating_treemap_counts_observed_binary_ratings_only(self):
        reviews = pd.DataFrame({"rating": [1, 2, 2, 4, 5, 5, 5]})
        fig = rating_imbalance(reviews)
        self.assertEqual(list(fig.data[0].labels), ["1 stars", "2 stars", "4 stars", "5 stars"])
        self.assertEqual(list(fig.data[0].values), [1, 2, 1, 3])
        self.assertEqual(sum(fig.data[0].values), len(reviews))
        filtered = rating_imbalance(reviews[reviews.rating.eq(5)])
        self.assertEqual(list(filtered.data[0].labels), ["5 stars"])
        self.assertEqual(list(filtered.data[0].values), [3])
        self.assertEqual(list(filtered.data[0].marker.colors), ["#FFE77A"])

    def test_empty_length_and_rating_charts_are_serializable(self):
        for fig in [review_length_chart(pd.DataFrame({"review_text": []})),
                    rating_imbalance(pd.DataFrame({"rating": []}))]:
            fig.to_json()


class ProductComparisonTests(unittest.TestCase):
    def test_named_groups_show_counts_without_listing_details(self):
        names = [
            "Baseus H17 Encok 3.5mm Wired Earphone",
            "Baseus C17 Type-C Earphones",
            "Baseus Encok CZ11 / HZ11",
            "Baseus WM01 TWS Wireless Earphones",
            "Baseus H19",
        ]
        summary = pd.DataFrame({
            "product_name": names,
            "product_ids": ["101, 102", "201", "301, 302", "401, 402, 403", "501"],
            "total_listing_ids": [2, 1, 2, 3, 1],
            "total_reviews": [10, 20, 30, 40, 50],
            "positive_reviews": [8, 16, 24, 32, 40],
            "negative_reviews": [2, 4, 6, 8, 10],
            "positive_percentage": [80.] * 5,
            "negative_percentage": [20.] * 5,
        })
        fig = product_comparison(summary)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(list(fig.data[0].y), names)
        self.assertEqual(list(fig.data[1].y), names)
        self.assertEqual(len(set(fig.data[0].y)), 5)
        self.assertEqual(sum(fig.data[0].x) + sum(fig.data[1].x), 150)
        for i, row in summary.iterrows():
            self.assertEqual(len(fig.data[0].customdata[i]), 3)
            self.assertEqual(fig.data[0].customdata[i][2], row.total_reviews)
        self.assertNotIn("Listing", fig.data[0].hovertemplate)
        self.assertNotIn("product_ids", fig.to_json())
        self.assertEqual(fig.layout.yaxis.title.text, "Product")
        self.assertTrue(all(trace.orientation == "h" for trace in fig.data))
        fig.to_json()

    def test_empty_summary_is_safe(self):
        summary = pd.DataFrame(columns=[
            "product_name", "product_ids", "total_listing_ids", "total_reviews",
            "positive_reviews", "negative_reviews", "positive_percentage", "negative_percentage",
        ])
        fig = product_comparison(summary)
        self.assertEqual(len(fig.data[0].x), 0)
        fig.to_json()


if __name__ == "__main__":
    unittest.main()
