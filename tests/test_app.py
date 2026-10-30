"""Streamlit interaction tests using real review data and temporary result fixtures.

Synthetic evaluation rows exist only in a temporary directory owned by each test;
they are never copied into the application results directory or presented as FYP
experimental findings.
"""
from datetime import date
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard import charts
from dashboard.data import filter_reviews, load_dataset, sentiment_summary
from dashboard.products import PRODUCT_GROUPS


ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "dataset_preprocessed_v3_clean.csv"
LABELLED = ROOT / "data" / "dataset v3 labelled 2.csv"
C17 = "Baseus C17 Type-C Earphones"
H17 = "Baseus H17 Encok 3.5mm Wired Earphone"


def widget(elements, label):
    return next(item for item in elements if item.label == label)


@unittest.skipUnless(CLEANED.exists() and LABELLED.exists(), "Provide both local review CSVs to run dataset-dependent app checks")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(CLEANED)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.results = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {"SHOPEE_RESULTS_DIR": self.temp.name})
        self.environment.start()
        self.original_data_path = os.environ.pop("SHOPEE_DATA_PATH", None)

    def tearDown(self):
        if self.original_data_path is not None:
            os.environ["SHOPEE_DATA_PATH"] = self.original_data_path
        self.environment.stop()
        self.temp.cleanup()

    def app(self, page="Dashboard Overview"):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertNoException(app)
        if page != "Home":
            self.navigate(app, page)
        return app

    def assertNoException(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def navigate(self, app, name):
        widget(app.radio, "Navigation").set_value(name).run()
        self.assertNoException(app)

    def metrics(self, app):
        return {item.label: item.value for item in app.metric}

    def assertNoPublicIdentifiers(self, app):
        """Inspect serialized/displayed content, not hidden widget state values."""
        forbidden_labels = {"Product ID", "Listing IDs", "Listings", "Source review ID", "Review ID", "Source columns", "Existing cleaned review", "Compare existing cleaned text"}
        forbidden_columns = {"product_id", "product_ids", "total_listing_ids", "source_review_id", "review_id", "shop_id", "item_id", "cmtid", "cleaned_text"}
        text = []
        for kind in ["markdown", "caption", "text", "info", "warning", "error", "success", "subheader", "title", "header", "code", "json"]:
            text.extend(str(item.value) for item in app.get(kind))
        text.extend(item.label for item in app.expander)
        for table in app.dataframe:
            self.assertTrue((forbidden_labels | forbidden_columns).isdisjoint(table.value.columns))
            text.append(table.value.to_csv(index=False))
        for kind in ["selectbox", "multiselect", "radio"]:
            for item in app.get(kind):
                text.extend(str(option) for option in item.options)
        text.extend(chart.proto.spec for chart in app.get("plotly_chart"))
        rendered = "\n".join(text)
        for label in forbidden_labels | forbidden_columns:
            self.assertNotIn(label, rendered)
        for listing_ids in PRODUCT_GROUPS.values():
            for listing_id in listing_ids:
                self.assertNotIn(listing_id, rendered)

    def assertPublicExport(self, exported):
        allowed = {"product_name", "review_text", "rating", "review_date", "sentiment", "predicted_sentiment", "confidence"}
        self.assertTrue(set(exported.columns).issubset(allowed))
        self.assertTrue({"product_name", "review_text", "rating", "review_date", "sentiment"}.issubset(exported.columns))
        for listing_ids in PRODUCT_GROUPS.values():
            for listing_id in listing_ids:
                self.assertNotIn(listing_id, exported.to_csv(index=False))

    def test_home_landing_and_primary_navigation_actions(self):
        app = self.app(page="Home")
        self.assertEqual(widget(app.radio, "Navigation").value, "Home")
        self.assertNoPublicIdentifiers(app)
        widget(app.button, "Open dashboard").click().run()
        self.assertNoException(app)
        self.assertEqual(widget(app.radio, "Navigation").value, "Dashboard Overview")
        self.assertEqual(self.metrics(app)["Total reviews"], "19,295")
        self.navigate(app, "Home")
        widget(app.button, "View products").click().run()
        self.assertNoException(app)
        self.assertEqual(widget(app.radio, "Navigation").value, "Product Sentiment Analysis")
        self.assertEqual(len(app.dataframe[0].value), 5)
        self.assertNoPublicIdentifiers(app)
        self.navigate(app, "Home")
        next(item for item in app.button if item.key == "explore_baseus-c17").click().run()
        self.assertNoException(app)
        self.assertEqual(widget(app.radio, "Navigation").value, "Product Sentiment Analysis")
        self.assertEqual(widget(app.multiselect, "Products").value, [C17])
        self.assertEqual(widget(app.selectbox, "Focus product").value, C17)
        self.assertEqual(self.metrics(app)["Total reviews"], "4,338")
        self.assertNoPublicIdentifiers(app)

    def test_all_pages_and_stage_one_empty_evaluation(self):
        app = self.app()
        self.assertEqual(self.metrics(app), {
            "Total reviews": "19,295", "Products": "5", "Positive reviews": "19,013",
            "Negative reviews": "282", "Positive share": "98.54%",
        })
        self.assertEqual(len(app.get("plotly_chart")), 3)
        self.assertEqual(set(widget(app.multiselect, "Products").options), set(PRODUCT_GROUPS))
        self.assertNoPublicIdentifiers(app)
        for page in ["Home", "Product Sentiment Analysis", "Customer Review Explorer", "Model Performance"]:
            self.navigate(app, page)
            self.assertNoPublicIdentifiers(app)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("Evaluation results are not available yet" in item.value for item in app.markdown))
        self.assertTrue(any("Baseline model evaluation results are not available" in item.value for item in app.info))

    def test_product_focus_and_percentage_yearly_trend(self):
        app = self.app()
        self.navigate(app, "Product Sentiment Analysis")
        comparison = app.dataframe[0].value
        self.assertEqual(len(comparison), 5)
        self.assertEqual(comparison.columns.tolist(), ["Product", "Total reviews", "Positive", "Negative", "Positive %", "Negative %"])
        # AppTest.value reads the complete Arrow payload, including columns that
        # could be restored with Streamlit's column toggle. Projection, rather
        # than column_config visibility alone, keeps listing fields out of it.
        for table in app.dataframe:
            self.assertTrue({"Listing IDs", "Listings", "product_ids", "total_listing_ids"}.isdisjoint(table.value.columns))
        comparison_spec = app.get("plotly_chart")[0].proto.spec
        self.assertNotIn("Listing IDs", comparison_spec)
        self.assertNotIn("Listings", comparison_spec)
        for listing_ids in PRODUCT_GROUPS.values():
            for listing_id in listing_ids:
                self.assertNotIn(listing_id, comparison_spec)
        self.assertEqual(comparison["Total reviews"].sum(), 19295)
        self.assertEqual(set(widget(app.selectbox, "Focus product").options), {"All Products", *PRODUCT_GROUPS})
        for name, listings, total in [(C17, 4, 4338), (H17, 2, 2251)]:
            with self.subTest(product=name):
                widget(app.selectbox, "Focus product").set_value(name).run()
                self.assertNoException(app)
                expected = self.dataset.reviews[self.dataset.reviews.product_name.eq(name)]
                self.assertEqual(expected.product_id.nunique(), listings)
                self.assertEqual(len(expected), total)
                self.assertEqual(self.metrics(app)["Total reviews"], f"{total:,}")
                row = comparison[comparison.Product.eq(name)].iloc[0]
                self.assertEqual(set(expected.product_id), set(PRODUCT_GROUPS[name]))
                self.assertEqual(row["Total reviews"], total)
        widget(app.radio, "Trend measure").set_value("Sentiment percentage").run()
        widget(app.selectbox, "Time grouping").set_value("Yearly").run()
        self.assertNoException(app)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        spec = json.loads(app.get("plotly_chart")[1].proto.spec)
        self.assertIn("%", spec["layout"]["yaxis"]["title"]["text"])

    def test_global_product_sentiment_and_inclusive_date_filters(self):
        app = self.app()
        widget(app.multiselect, "Products").set_value([C17]).run()
        widget(app.multiselect, "Rating-derived sentiment").set_value(["Negative"]).run()
        dates = (date(2023, 1, 1), date(2025, 12, 31))
        widget(app.date_input, "Review date range").set_value(dates).run()
        self.assertNoException(app)
        expected = filter_reviews(self.dataset.reviews, product_names=[C17], sentiments=["Negative"], start=dates[0], end=dates[1])
        summary = sentiment_summary(expected)
        self.assertGreater(summary["total_reviews"], 0)
        self.assertEqual(self.metrics(app)["Total reviews"], f"{summary['total_reviews']:,}")
        self.assertEqual(self.metrics(app)["Positive reviews"], "0")
        self.assertEqual(self.metrics(app)["Positive share"], "0.00%")
        self.assertEqual(self.metrics(app)["Products"], "1")
        self.assertGreater(expected.product_id.nunique(), 1)
        self.navigate(app, "Product Sentiment Analysis")
        self.assertEqual(app.dataframe[0].value["Total reviews"].sum(), len(expected))
        self.assertEqual(app.dataframe[0].value["Product"].tolist(), [C17])

    def test_reset_filters_restores_full_dataset_and_all_products(self):
        app = self.app()
        widget(app.multiselect, "Products").set_value([C17]).run()
        widget(app.multiselect, "Rating-derived sentiment").set_value(["Negative"]).run()
        widget(app.date_input, "Review date range").set_value((date(2023, 1, 1), date(2025, 12, 31))).run()
        widget(app.checkbox, "Include undated reviews").uncheck().run()
        self.assertNoException(app)
        self.assertEqual(self.metrics(app)["Products"], "1")
        self.assertLess(int(self.metrics(app)["Total reviews"].replace(",", "")), len(self.dataset.reviews))
        widget(app.button, "Reset filters").click().run()
        self.assertNoException(app)
        self.assertEqual(set(widget(app.multiselect, "Products").value), set(PRODUCT_GROUPS))
        self.assertEqual(set(widget(app.multiselect, "Rating-derived sentiment").value), {"Positive", "Negative"})
        self.assertTrue(widget(app.checkbox, "Include undated reviews").value)
        dates = self.dataset.reviews.review_date.dropna()
        self.assertEqual(widget(app.date_input, "Review date range").value, (dates.min().date(), dates.max().date()))
        self.assertEqual(self.metrics(app)["Total reviews"], "19,295")
        self.assertEqual(self.metrics(app)["Products"], "5")
        self.assertEqual(self.metrics(app)["Positive share"], "98.54%")

    def test_empty_global_selection_and_explorer_search(self):
        app = self.app()
        widget(app.multiselect, "Products").set_value([]).run()
        self.assertNoException(app)
        self.assertEqual(self.metrics(app)["Total reviews"], "0")
        self.assertEqual(self.metrics(app)["Positive share"], "—")
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.navigate(app, "Customer Review Explorer")
        self.assertTrue(any("No matching reviews" in item.value for item in app.info))

    def test_explorer_malay_case_insensitive_search_rating_details_and_export(self):
        app = self.app()
        self.navigate(app, "Customer Review Explorer")
        widget(app.text_input, "Search original reviews").set_value("BaGuS").run()
        widget(app.multiselect, "Star rating").set_value([4, 5]).run()
        self.assertNoException(app)
        expected = self.dataset.reviews
        expected = expected[expected.review_text.str.contains("bagus", case=False, regex=False) & expected.rating.isin([4, 5])]
        table = app.dataframe[0].value
        self.assertTrue(table["Original review"].str.contains("bagus", case=False, regex=False).all())
        self.assertTrue(table["Stars"].isin([4, 5]).all())
        self.assertNotIn("Model prediction", table.columns)
        self.assertTrue(any(f"**{len(expected):,} reviews**" in item.value for item in app.markdown))
        buttons = app.get("download_button")
        self.assertEqual(len(buttons), 1)
        self.assertTrue(buttons[0].proto.url.endswith(".csv"))
        details = widget(app.selectbox, "Open review details")
        details.set_value(expected.review_id.iloc[len(table) - 1]).run()
        self.assertNoException(app)
        self.assertIn(table["Original review"].iloc[-1], [item.value for item in app.text])
        self.assertTrue(any("Model prediction: not available" in item.value for item in app.caption))
        widget(app.text_input, "Search original reviews").set_value("no_review_should_contain_this_2718281828").run()
        self.assertNoException(app)
        self.assertEqual(len(app.dataframe), 0)
        self.assertTrue(any("No matching reviews" in item.value for item in app.info))

    def test_group_filter_export_and_details_show_product_names_without_identifiers(self):
        app = self.app()
        widget(app.multiselect, "Products").set_value([C17]).run()
        with patch("streamlit.download_button", wraps=st.download_button) as export:
            self.navigate(app, "Customer Review Explorer")
            exported = pd.read_csv(io.BytesIO(export.call_args.args[1]))
        expected = self.dataset.reviews[self.dataset.reviews.product_name.eq(C17)]
        self.assertEqual(len(exported), 4338)
        self.assertPublicExport(exported)
        self.assertEqual(set(exported.product_name), {C17})
        self.assertEqual(exported.review_text.tolist(), expected.review_text.tolist())
        table = app.dataframe[0].value
        self.assertEqual(set(table["Product"]), {C17})
        self.assertNoPublicIdentifiers(app)
        review = table.iloc[-1]
        widget(app.selectbox, "Open review details").set_value(expected.review_id.iloc[len(table) - 1]).run()
        self.assertNoException(app)
        details_text = "\n".join(item.value for elements in [app.markdown, app.caption, app.text] for item in elements)
        self.assertIn(f"Product: {C17}", details_text)
        self.assertNoPublicIdentifiers(app)
        self.assertIn(review["Original review"], [item.value for item in app.text])

    def test_switch_to_labelled_source_and_empty_sentiment_filter(self):
        app = self.app()
        widget(app.selectbox, "Review dataset").set_value(LABELLED).run()
        self.assertNoException(app)
        self.assertEqual(self.metrics(app)["Total reviews"], "23,527")
        self.assertEqual(self.metrics(app)["Positive reviews"], "23,159")
        self.assertEqual(self.metrics(app)["Negative reviews"], "368")
        widget(app.multiselect, "Rating-derived sentiment").set_value([]).run()
        self.assertNoException(app)
        self.assertEqual(self.metrics(app)["Total reviews"], "0")

    def test_real_labelled_source_has_clean_public_tables_exports_and_hover_data(self):
        app = self.app()
        widget(app.selectbox, "Review dataset").set_value(LABELLED).run()
        self.assertNoException(app)
        self.assertEqual(self.metrics(app)["Total reviews"], "23,527")
        with patch("dashboard.charts.product_comparison", wraps=charts.product_comparison) as render_chart:
            self.navigate(app, "Product Sentiment Analysis")
            chart_input = render_chart.call_args.args[0]
        self.assertEqual(chart_input.columns.tolist(), ["product_name", "total_reviews", "positive_reviews", "negative_reviews", "positive_percentage", "negative_percentage"])
        self.assertEqual(app.dataframe[0].value["Total reviews"].sum(), 23527)
        self.assertNoPublicIdentifiers(app)
        with patch("streamlit.download_button", wraps=st.download_button) as export:
            self.navigate(app, "Customer Review Explorer")
            exported = pd.read_csv(io.BytesIO(export.call_args.args[1]))
        self.assertEqual(len(exported), 23527)
        self.assertEqual(exported.columns.tolist(), ["product_name", "review_text", "rating", "review_date", "sentiment"])
        self.assertPublicExport(exported)
        self.assertNoPublicIdentifiers(app)
        self.assertEqual(app.dataframe[0].value.columns.tolist(), ["Product", "Original review", "Stars", "Review date", "Rating-derived sentiment"])
        labelled = load_dataset(LABELLED).reviews
        self.assertEqual(exported.review_text.fillna("").tolist(), labelled.review_text.tolist())
        widget(app.selectbox, "Open review details").set_value(labelled.review_id.iloc[0]).run()
        self.assertNoException(app)
        self.assertIn(labelled.review_text.iloc[0], [item.value for item in app.text])
        self.assertNoPublicIdentifiers(app)

    def test_internal_cleaned_text_sentinel_never_reaches_public_review_outputs(self):
        fixture = self.dataset.records.head(3).copy()
        sentinel = "INTERNAL_CLEANED_SENTINEL_NEVER_PUBLISH_7319"
        for alias in ["cleaned_text", "cleaned_comment_v3", "cleaned_comment"]:
            fixture[alias] = sentinel
        source = self.results / "internal_cleaned_review_source.csv"
        fixture.to_csv(source, index=False)
        os.environ["SHOPEE_DATA_PATH"] = str(source)
        app = self.app()
        with patch("streamlit.download_button", wraps=st.download_button) as export:
            self.navigate(app, "Customer Review Explorer")
            payload = export.call_args.args[1]
            exported = pd.read_csv(io.BytesIO(payload))
        self.assertPublicExport(exported)
        self.assertEqual(len(exported), 3)
        self.assertEqual(exported.review_text.tolist(), fixture.review_text.tolist())
        self.assertNotIn(sentinel, payload.decode("utf-8-sig"))
        self.assertNoPublicIdentifiers(app)
        for kind in ["text", "markdown", "caption"]:
            self.assertFalse(any(sentinel in str(item.value) for item in app.get(kind)))
        for table in app.dataframe:
            self.assertNotIn(sentinel, table.value.to_csv(index=False))
        self.assertIn(fixture.review_text.iloc[0], [item.value for item in app.text])

    def test_future_valid_held_out_results_appear_and_malformed_are_reported(self):
        # Test-only labels exercise a known asymmetric confusion matrix.
        pd.DataFrame({"review_id": ["fixture-a", "fixture-b", "fixture-c", "fixture-d"],
                      "y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1]}).to_csv(self.results / "test.csv", index=False)
        config = {
            "model_id": "temporary-test-fixture", "model_name": "Temporary test fixture",
            "test_set_id": "temporary-held-out", "split": "test", "held_out": True,
            "evaluation_config": {"averaging": "binary", "positive_label": 1, "zero_division": 0},
            "predictions_file": "test.csv",
        }
        evaluation = self.results / "evaluation_fixture.json"
        evaluation.write_text(json.dumps(config))
        app = self.app()
        self.navigate(app, "Model Performance")
        self.assertEqual(self.metrics(app), {"F1-score · primary": "0.800", "Accuracy": "0.750", "Precision": "0.667", "Recall": "1.000"})
        self.assertEqual(len(app.get("plotly_chart")), 1)
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(spec["data"][0]["z"], [[1, 1], [0, 2]])
        evaluation.write_text(json.dumps(dict(config, split="train")))
        app.run()
        self.assertNoException(app)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("Result file not loaded" in item.value for item in app.warning))

    def test_validated_predictions_use_internal_ids_and_public_export_omits_identifiers(self):
        first, second = self.dataset.reviews.iloc[:2].itertuples(index=False)
        pd.DataFrame({"review_id": [first.review_id, second.review_id],
                      "predicted_sentiment": [first.sentiment, "Positive" if second.sentiment == "Negative" else "Negative"],
                      "confidence": [.7, .6]}).to_csv(self.results / "predictions.csv", index=False)
        app = self.app()
        with patch("streamlit.download_button", wraps=st.download_button) as export:
            self.navigate(app, "Customer Review Explorer")
            exported = pd.read_csv(io.BytesIO(export.call_args.args[1]))
        self.assertPublicExport(exported)
        self.assertIn("predicted_sentiment", exported.columns)
        self.assertEqual(len(exported), len(self.dataset.reviews))
        self.assertEqual(exported.predicted_sentiment.notna().sum(), 2)
        self.assertEqual(exported.review_text.iloc[0], first.review_text)
        self.assertTrue(any("Prediction coverage: 2 of 19,295" in item.value for item in app.caption))
        self.assertIn("Model confidence", app.dataframe[0].value.columns)

        widget(app.selectbox, "Prediction agreement with star label").set_value("Matches rating label").run()
        self.assertNoException(app)
        table = app.dataframe[0].value
        self.assertEqual(table["Original review"].tolist(), [first.review_text])
        self.assertNoPublicIdentifiers(app)
        self.assertEqual(table["Model prediction"].tolist(), [first.sentiment])

        widget(app.selectbox, "Prediction agreement with star label").set_value("Differs from rating label").run()
        self.assertNoException(app)
        self.assertEqual(app.dataframe[0].value["Original review"].tolist(), [second.review_text])
        self.assertNoPublicIdentifiers(app)

        widget(app.selectbox, "Prediction agreement with star label").set_value("No prediction").run()
        self.assertNoException(app)
        table = app.dataframe[0].value
        self.assertTrue(table["Model prediction"].isna().all())
        self.assertTrue(any("**19,293 reviews**" in item.value for item in app.markdown))
        self.navigate(app, "Model Performance")
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)

    def test_review_csv_predictions_are_ignored_until_separate_results_validate_them(self):
        fixture = self.dataset.records.head(3).copy()
        fixture["predicted_sentiment"] = "Unvalidated raw CSV value"
        fixture["confidence"] = 9.99
        source = self.results / "review_source.csv"
        fixture.to_csv(source, index=False)
        os.environ["SHOPEE_DATA_PATH"] = str(source)
        app = self.app()
        self.navigate(app, "Customer Review Explorer")
        table = app.dataframe[0].value
        self.assertNotIn("Model prediction", table.columns)
        self.assertNotIn("Model confidence", table.columns)
        self.assertTrue(any("Model predictions are not available yet" in item.value for item in app.info))
        self.assertFalse(any(item.label == "Prediction agreement with star label" for item in app.selectbox))

        first = load_dataset(source).reviews.iloc[0]
        pd.DataFrame({"review_id": [first.review_id], "predicted_sentiment": [first.sentiment],
                      "confidence": [.8]}).to_csv(self.results / "predictions.csv", index=False)
        app.run()
        self.assertNoException(app)
        table = app.dataframe[0].value
        self.assertEqual(table["Model prediction"].dropna().tolist(), [first.sentiment])
        self.assertEqual(table["Model confidence"].dropna().tolist(), [.8])
        self.assertTrue(any("Prediction coverage: 1 of 3" in item.value for item in app.caption))
        self.assertFalse(any("Predictions could not be joined" in item.value for item in app.warning))


class HomeWithoutDatasetTests(unittest.TestCase):
    def test_home_renders_without_a_configured_review_file(self):
        # An unavailable configured source must not block the public landing
        # page. No repository files are removed or replaced by this check.
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "unavailable-reviews.csv"
            with patch.dict(os.environ, {"SHOPEE_DATA_PATH": str(missing), "SHOPEE_RESULTS_DIR": folder}):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                self.assertEqual([item.value for item in app.exception], [])
                self.assertEqual(widget(app.radio, "Navigation").value, "Home")
                self.assertEqual(len(app.error), 0)
                self.assertFalse(any(item.label == "Review dataset" for item in app.selectbox))
                self.assertEqual(sum(item.label == "Explore product" for item in app.button), 5)
                widget(app.button, "Open dashboard").click().run()
                self.assertEqual([item.value for item in app.exception], [])
                self.assertTrue(any("configured review dataset is unavailable" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
