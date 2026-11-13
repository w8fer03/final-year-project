"""SentiLokal review analytics. Run from this directory: streamlit run app.py."""
from __future__ import annotations

import io
from base64 import b64encode
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.runtime import refresh_charts
from dashboard.navigation import section_anchor, scroll_to_section
from dashboard.artifacts import attach_predictions, comparable_evaluations, load_results
from dashboard.data import filter_reviews, load_dataset, product_summary, sentiment_summary
from dashboard.media import product_media
from dashboard.presentation import public_product_name, public_reviews
from dashboard.products import PRODUCT_GROUPS, PRODUCT_NAMES
from dashboard.research import DEFAULT_FILENAME, ResearchDataError, read_research_source
from dashboard.research_view import dashboard_research_dataset
from dashboard.final_evaluations import load_final_evaluations
from dashboard.performance_page import render_final_performance
from dashboard.version_b import EVALUATIONS, VersionBError
from dashboard.version_b_research import load_version_b_research, load_version_b_predictions
from dashboard.upload_page import remember_upload_controls, render_upload_page

ROOT = Path(__file__).resolve().parent
APP_NAME = "SentiLokal"
APP_DESCRIPTOR = "Malaysian E-Commerce Review Sentiment Analytics"
APP_TITLE = f"{APP_NAME} – {APP_DESCRIPTOR}"
RESEARCH_SECTIONS = ["Home", "Dashboard Overview", "Product Sentiment Analysis", "Customer Review Explorer", "Model Performance"]
RESEARCH_MODULE = "Baseus Research Dashboard"
UPLOAD_MODULE = "Upload & Analyze"
MODULES = [RESEARCH_MODULE, UPLOAD_MODULE]
PAGES = RESEARCH_SECTIONS
charts = refresh_charts()

st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide")
font_faces = "\n".join(
    '@font-face{font-family:"Poppins";font-style:normal;font-weight:' + str(weight)
    + ';font-display:swap;src:url(data:font/ttf;base64,'
    + b64encode((ROOT / "assets/fonts" / f"Poppins-{name}.ttf").read_bytes()).decode("ascii")
    + ') format("truetype");}'
    for weight, name in [(400, "Regular"), (500, "Medium"), (600, "SemiBold"), (700, "Bold")]
)
st.markdown(f"<style>{font_faces}\n{(ROOT / 'assets/style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading review data…")
def cached_dataset(content: bytes, name: str, product_catalog: tuple[tuple[str, str], ...]):
    # The catalog is part of the cache key so correcting a name refreshes the data.
    return dashboard_research_dataset(load_dataset(io.BytesIO(content), source_name=name))


def plot(fig, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def empty(title: str, message: str):
    # Only call with application-owned static strings.
    st.markdown(f'<div class="empty-state"><h3>{title}</h3><p>{message}</p></div>', unsafe_allow_html=True)


def kpis(df: pd.DataFrame, include_products: bool = True, prefix: str = ""):
    s = sentiment_summary(df)
    entries = [("Total reviews", f"{s['total_reviews']:,}")]
    if include_products:
        entries.append(("Products", f"{s['total_products']:,}"))
    entries += [("Positive reviews", f"{s['positive_reviews']:,}"),
                ("Negative reviews", f"{s['negative_reviews']:,}"),
                ("Positive share", f"{s['positive_percentage']:.2f}%" if len(df) else "—")]
    for index, (col, (label, value)) in enumerate(zip(st.columns(len(entries)), entries)):
        with col:
            metric_key = "total" if index == 0 else label.lower().replace(" ", "_")
            with st.container(key=f"kpi_{prefix}{metric_key}"):
                st.metric(label, value)
    st.caption("Based on your current filters · Star-rating labels · Original records retained")


def panel_title(title: str, subtitle: str):
    st.markdown(f'<div class="panel-heading"><h3>{escape(title)}</h3><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def reset_filters(source_name, products, date_bounds):
    """Reset sidebar controls before Streamlit creates the widgets."""
    st.session_state[f"product_names_{source_name}"] = list(products)
    st.session_state[f"sentiments_{source_name}"] = ["Positive", "Negative"]
    st.session_state[f"undated_{source_name}"] = True
    if date_bounds:
        st.session_state[f"dates_{source_name}"] = date_bounds


def navigate(page: str, product: str | None = None):
    st.session_state.application_module = RESEARCH_MODULE
    st.session_state.navigation = page
    queue_section_scroll()
    if product is not None:
        st.session_state.requested_product = product


def queue_section_scroll():
    """Selection moves within the document; it never chooses what to render."""
    if st.session_state.get("active_workspace") == "research":
        remember_research_controls()
    elif st.session_state.get("active_workspace") == "upload":
        remember_upload_controls()
    # The isolated upload document shares the existing top-of-page anchor.
    # This also works when a running server has cached the older navigation module.
    st.session_state.pending_section = "Home" if st.session_state.navigation == "Upload & Analyze" else st.session_state.navigation
    st.session_state.scroll_revision = st.session_state.get("scroll_revision", 0) + 1


def remember_research_controls():
    keys = {"navigation", "focus_product_name", "review_page", "rows_per_page"}
    saved = st.session_state.get("research_controls", {}).copy()
    saved.update({key: st.session_state[key] for key in st.session_state
                  if key in keys or key.startswith(("research_", "product_names_", "sentiments_", "dates_", "undated_"))
                  if key != "research_controls"})
    st.session_state.research_controls = saved


def restore_research_controls():
    for key, value in st.session_state.get("research_controls", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def product_photo(name: str, *, compact: bool = False):
    media = product_media(name)
    uri = media.image_data_uri
    if uri:
        st.markdown(f'<div class="product-photo"><img src="{uri}" alt="{escape(media.alt, quote=True)}" /></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="product-photo product-photo-empty" role="img" aria-label="Product photo not available"><svg viewBox="0 0 64 64" aria-hidden="true"><path d="M12 36V29a20 20 0 0 1 40 0v7M12 33h7v20h-7a5 5 0 0 1-5-5V38a5 5 0 0 1 5-5Zm40 0h-7v20h7a5 5 0 0 0 5-5V38a5 5 0 0 0-5-5Z"/></svg><span>Product photo not available yet</span></div>', unsafe_allow_html=True)
    if not compact and media.source_url:
        st.link_button("Photo source", media.source_url)
    return media


def render_home():
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<div class="landing-eyebrow">BASEUS PRODUCTS · SHOPEE MALAYSIA</div>', unsafe_allow_html=True)
        st.title(APP_NAME)
        st.markdown(f'<p class="hero-sub">{APP_DESCRIPTOR}</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="landing-headline">Your reviews.<br>A clearer perspective.</h2>', unsafe_allow_html=True)
        st.write("Explore customer feedback across Baseus earphones. Compare product ratings, follow historical sentiment patterns, and read the original Malay, English, and mixed-language comments.")
        a, b = st.columns(2)
        a.button("Open dashboard", on_click=navigate, args=("Dashboard Overview",), type="primary", use_container_width=True)
        b.button("View products", on_click=navigate, args=("Product Sentiment Analysis",), use_container_width=True)
        st.caption("Current insights use customer star ratings. Trained-model predictions will be integrated when genuine results are available.")
    with right:
        st.markdown('<div class="landing-visual"><span class="landing-visual-label">CUSTOMER FEEDBACK, MADE CLEARER</span><strong>Baseus</strong><div class="landing-visual-rule"></div><p>Five products.<br>One place to explore.</p><span class="landing-visual-foot">SentiLokal · Review analytics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-section"><span>THE PRODUCT COLLECTION</span><h2>Get to know the products</h2><p>Choose a product to explore its customer reviews and rating-derived sentiment.</p></div>', unsafe_allow_html=True)
    names = list(PRODUCT_GROUPS)
    for start in range(0, len(names), 3):
        columns = st.columns(3, gap="medium")
        for column, name in zip(columns, names[start:start + 3]):
            with column:
                media = product_media(name)
                with st.container(border=True, key=f"panel_catalog_{media.slug}"):
                    product_photo(name)
                    st.markdown(f'<div class="product-card-copy"><span>{escape(media.category)}</span><h3>{escape(name)}</h3></div>', unsafe_allow_html=True)
                    st.button("Explore product", key=f"explore_{media.slug}", on_click=navigate, args=("Product Sentiment Analysis", name), use_container_width=True)
    st.markdown('<div class="landing-section"><span>HOW TO EXPLORE</span><h2>From the big picture to the original review</h2></div>', unsafe_allow_html=True)
    for column, (number, title, text) in zip(st.columns(3), [
        ("01", "See the overview", "Understand review volume, star ratings, and sentiment across the collection."),
        ("02", "Compare the products", "Explore differences in customer ratings and historical trends."),
        ("03", "Read the feedback", "Search and read the original customer comments in their own words."),
    ]):
        with column:
            st.markdown(f'<div class="journey-card"><span>{number}</span><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)
    st.caption("SentiLokal is an academic research prototype. Product names and any sourced photographs belong to their respective owners; no affiliation or endorsement is implied.")


def date_caption(df):
    dates = df.review_date.dropna()
    if len(dates):
        st.caption(f"Review dates: {dates.min():%d %b %Y} – {dates.max():%d %b %Y} · {df.review_date.isna().sum():,} undated records in selection")
    else:
        st.caption("No valid review dates in this selection.")


def trend(df, key, percentage=False, frequency="MS"):
    if df.review_date.notna().any():
        plot(charts.trend_chart(df, frequency=frequency, percentage=percentage), key)
        missing = int(df.review_date.isna().sum())
        st.caption(f"Historical review dates only. {missing:,} undated records omitted from this chart. No live monitoring.")
    else:
        st.info("A sentiment trend needs valid review dates. Undated reviews remain available in the other analyses.")


def render_overview(dataset, df):
    st.subheader("Dashboard Overview")
    st.caption("The complete picture in one place · All charts follow your sidebar filters.")
    with st.container(key="analytics_board"):
        kpis(df)
        date_caption(df)
        if df.empty:
            st.info("No reviews match the current filters. Expand the selection in the sidebar.")
        else:
            # A compact two-column board keeps every analysis on the same page.
            left, right = st.columns(2, gap="medium")
            with left:
                with st.container(border=True, key="panel_imbalance"):
                    panel_title("Rating class imbalance", "Review counts by original star rating")
                    plot(charts.rating_imbalance(df), "overview_imbalance")
            with right:
                with st.container(border=True, key="panel_lengths"):
                    panel_title("Review length analysis", "Whitespace-separated words in original comments")
                    plot(charts.review_length_chart(df), "overview_lengths")
            left, right = st.columns(2, gap="medium")
            with left:
                with st.container(border=True, key="panel_ratings"):
                    panel_title("Star rating distribution", "1–2 stars: negative · 4–5 stars: positive")
                    plot(charts.rating_bars(df).update_layout(height=310), "overview_ratings")
            with right:
                with st.container(border=True, key="panel_comparison"):
                    panel_title("Product sentiment comparison", "Positive and negative review counts by product")
                    summary = public_product_summary(df)
                    plot(charts.product_comparison(summary).update_layout(height=310), "overview_products")
            left, right = st.columns([1, 1.4], gap="medium")
            with left:
                with st.container(border=True, key="panel_distribution"):
                    panel_title("Overall sentiment", "Customer star ratings, before class balancing")
                    plot(charts.sentiment_donut(df).update_layout(height=265), "overview_donut")
            with right:
                with st.container(border=True, key="panel_history"):
                    panel_title("Sentiment over time", "Historical monthly review counts")
                    if df.review_date.notna().any():
                        plot(charts.trend_chart(df).update_layout(height=265), "overview_trend")
                        st.caption(f"{int(df.review_date.isna().sum()):,} undated records omitted from this chart.")
                    else:
                        st.info("A sentiment trend needs valid review dates. Undated reviews remain in totals.")
            st.caption("The original class imbalance is preserved. Three-star and invalid ratings are excluded. Star labels may differ from comment sentiment; no balancing or training is performed.")
    with st.expander("About this dataset", expanded=False):
        st.caption("These quality counts describe the entire selected source file, before sidebar filters.")
        labels = {
            "total_records": "Source records", "binary_records": "Eligible binary records",
            "neutral_records": "3-star records excluded", "invalid_rating_records": "Invalid / missing ratings excluded",
            "excluded_preprocessing_records": "Excluded by existing preprocessing status",
            "empty_reviews": "Empty original comments (retained)", "missing_dates": "Missing dates",
            "invalid_dates": "Unparseable dates", "duplicate_rows": "Exact duplicate rows (retained)",
            "label_mismatches": "Stored labels differing from ratings",
            "missing_product_ids": "Records without product information",
        }
        st.dataframe(pd.DataFrame([{"Check": label, "Records": dataset.quality.get(key, 0)} for key, label in labels.items()]), hide_index=True, use_container_width=True)
        st.caption("The official cleaned research dataset is loaded automatically. Ratings 1–2 → Negative (0), 4–5 → Positive (1). Three-star and invalid-rating records never enter binary statistics. Existing preprocessing exclusions are respected. Original CSVs are unchanged.")

def public_product_summary(df):
    """Only customer-facing aggregates may cross the chart and table boundary."""
    return product_summary(df)[["product_name", "total_reviews", "positive_reviews", "negative_reviews", "positive_percentage", "negative_percentage"]]


def product_comparison_table(summary):
    # Project the display data so listing metadata cannot be re-enabled in the column menu.
    st.dataframe(summary.rename(columns={
        "product_name": "Product", "total_reviews": "Total reviews",
        "positive_reviews": "Positive", "negative_reviews": "Negative",
        "positive_percentage": "Positive %", "negative_percentage": "Negative %",
    }), hide_index=True, use_container_width=True,
        column_config={
            "Product": st.column_config.TextColumn(width="large"),
            "Total reviews": st.column_config.NumberColumn(width="small"),
            "Positive": st.column_config.NumberColumn(width="small"),
            "Negative": st.column_config.NumberColumn(width="small"),
            "Positive %": st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100, width="small"),
            "Negative %": st.column_config.NumberColumn(format="%.2f%%", width="small"),
        })


def render_products(df):
    st.subheader("Product Sentiment Analysis")
    st.caption("Compare customer feedback across products, then explore one product in detail.")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    summary = public_product_summary(df)
    options = ["All Products"] + sorted(df.product_name.unique().tolist())
    with st.container(border=True, key="panel_focus"):
        label, control = st.columns([1, 1.3])
        with label:
            panel_title("Your product view", "Choose a product for the summary cards and trend")
        with control:
            selection = st.selectbox("Focus product", options, key="focus_product_name")
    focus = df if selection == "All Products" else df[df.product_name == selection]
    if selection in PRODUCT_GROUPS:
        photo, description = st.columns([1, 2.5])
        with photo:
            product_photo(selection, compact=True)
        with description:
            st.markdown(f"#### {selection}")
            st.caption(product_media(selection).category)
            st.write("The cards and trend below show customer feedback for this product within your current filters.")
    kpis(focus, include_products=False, prefix="focus_")
    with st.container(border=True, key="panel_product_table"):
        panel_title("Product comparison", "A clear view of review counts and sentiment share")
        product_comparison_table(summary)
        st.caption("Select a column heading to sort. Counts reflect your sidebar filters.")
    with st.container(border=True, key="panel_product_trend"):
        panel_title("Product sentiment over time", selection)
        c1, c2 = st.columns([2, 1])
        percentage = c1.radio("Trend measure", ["Review count", "Sentiment percentage"], horizontal=True, key="research_trend_measure") == "Sentiment percentage"
        freq = c2.selectbox("Time grouping", ["Monthly", "Yearly"], key="research_trend_group")
        trend(focus, "product_trend", percentage=percentage, frequency="MS" if freq == "Monthly" else "YS")


def review_table(df):
    return public_reviews(df).rename(columns={
        "product_name": "Product",
        "review_text": "Original review", "rating": "Stars", "review_date": "Review date",
        "sentiment": "Rating-derived sentiment", "predicted_sentiment": "Model prediction", "confidence": "Model confidence",
    })


def render_explorer(df, results):
    st.subheader("Customer Review Explorer")
    st.caption("Read the original Malay, English, and mixed-language comments. Star labels do not establish the sentiment expressed in the text.")
    # Predictions in an arbitrary review CSV are not a validated model result.
    df = df.drop(columns=["predicted_sentiment", "confidence"], errors="ignore")
    if results.predictions is not None:
        try:
            df = attach_predictions(df, results.predictions)
        except ValueError:
            st.warning("Saved model predictions are unavailable for this dataset.")
    has_predictions = "predicted_sentiment" in df and df.predicted_sentiment.notna().any()
    if not has_predictions:
        st.info("Model predictions are not available yet. The labels below are derived from customer star ratings.")
    else:
        st.caption(f"Prediction coverage: {df.predicted_sentiment.notna().sum():,} of {len(df):,} selected records. These explorer comparisons are not held-out model evaluation.")
    with st.container(border=True, key="panel_review_search"):
        panel_title("Find a review", "Search the original customer comments")
        left, right = st.columns([2, 1])
        search = left.text_input("Search original reviews", placeholder="Search for bagus, rosak, sound, battery…", key="research_search")
        stars = right.multiselect("Star rating", [1, 2, 4, 5], default=[1, 2, 4, 5], key="research_stars")
    if search:
        df = df[df.review_text.str.contains(search, case=False, regex=False, na=False)]
    df = df[df.rating.isin(stars)]
    if has_predictions:
        status = st.selectbox("Prediction agreement with star label", ["All reviews", "Matches rating label", "Differs from rating label", "No prediction"], key="research_agreement")
        known = df.predicted_sentiment.notna()
        if status == "Matches rating label":
            df = df[known & df.predicted_sentiment.eq(df.sentiment)]
        elif status == "Differs from rating label":
            df = df[known & df.predicted_sentiment.ne(df.sentiment)]
        elif status == "No prediction":
            df = df[~known]
    st.write(f"**{len(df):,} reviews** match your selection")
    if df.empty:
        st.info("No matching reviews. Try a different keyword, rating, or sidebar filter.")
        return
    st.download_button("Download filtered reviews · CSV", public_reviews(df).to_csv(index=False).encode("utf-8-sig"), "filtered_shopee_reviews.csv", "text/csv", use_container_width=False)
    c1, c2 = st.columns(2)
    page_size = c1.selectbox("Reviews per page", [25, 50, 100], key="rows_per_page")
    pages = max(1, (len(df) + page_size - 1) // page_size)
    if st.session_state.get("review_page", 1) > pages:
        st.session_state.review_page = 1
    page = c2.number_input("Page", min_value=1, max_value=pages, step=1, key="review_page")
    visible = df.iloc[(page - 1) * page_size:page * page_size]
    st.dataframe(review_table(visible), hide_index=True, use_container_width=True,
                 column_config={"Original review": st.column_config.TextColumn(width="large"),
                                "Review date": st.column_config.DatetimeColumn(format="DD MMM YYYY HH:mm")})
    st.caption(f"Page {page} of {pages}. Export includes all {len(df):,} matching records.")
    review_labels = {row.review_id: f"Review {(page - 1) * page_size + index + 1} · {row.review_text[:70]}" for index, row in enumerate(visible.itertuples(index=False))}
    selected_id = st.selectbox("Open review details", visible.review_id.tolist(), format_func=review_labels.get)
    row = visible[visible.review_id == selected_id].iloc[0]
    with st.container(border=True, key="panel_review_detail"):
        st.markdown("#### Original customer review")
        st.text(row.review_text if row.review_text.strip() else "[No original comment supplied]")
        a, b, c = st.columns(3)
        a.write(f"Product: {row.product_name}")
        b.write(f"Stars: {int(row.rating)} / 5")
        c.write(f"Date: {row.review_date:%d %b %Y %H:%M}" if pd.notna(row.review_date) else "Date unavailable")
        if row.sentiment == "Positive":
            st.success("Rating-derived sentiment: Positive (1)")
        else:
            st.error("Rating-derived sentiment: Negative (0)")
        if "predicted_sentiment" in row and pd.notna(row.predicted_sentiment):
            st.write(f"Model prediction: **{row.predicted_sentiment}**")
            if "confidence" in row and pd.notna(row.confidence):
                st.write(f"Recorded model confidence: {row.confidence:.3f}")
        else:
            st.caption("Model prediction: not available")
    with st.expander("Linguistic characteristics"):
        st.info("Validated linguistic annotations for code-mixing, slang, abbreviations, emojis, and informal spelling are not available. No language classifications or frequencies are inferred.")
        st.caption("Preprocessing diagnostics, when present in the source, are not treated as validated linguistic annotations. Monolingual reviews remain included.")


def render_performance(results):
    st.subheader("Model Performance")
    try:
        report = load_final_evaluations(Path(os.environ.get("SENTILOKAL_EVALUATION_DIR", EVALUATIONS)))
    except VersionBError as exc:
        st.warning(str(exc))
        report = None
    if report is not None:
        render_final_performance(report)
        return
    st.caption("Held-out evaluation · Negative = 0 · Positive = 1 · F1-score is the primary project metric")
    if not results.evaluations:
        empty("Evaluation results are not available yet", "The recorded evaluation files are not installed in this deployment. Valid held-out results will appear here once they are supplied.")
        st.markdown("#### Evaluation metrics")
        st.info("Accuracy, precision, recall, and F1-score are pending held-out test evaluation.")
        st.markdown("#### Confusion matrix")
        st.info("A confusion matrix requires genuine test labels and predictions. No matrix is available yet.")
        st.markdown("#### Baseline comparison")
        st.info("Baseline model evaluation results are not available.")
    else:
        index = st.selectbox("Evaluation run", range(len(results.evaluations)), format_func=lambda i: results.evaluations[i].model_name)
        result = results.evaluations[index]
        st.success(f"Recorded held-out result: {result.model_name} · {result.n_samples:,} test examples")
        for col, name in zip(st.columns(4), ["f1", "accuracy", "precision", "recall"]):
            col.metric("F1-score · primary" if name == "f1" else name.title(), f"{result.metrics[name]:.3f}")
        a, b = st.columns([1.2, 1])
        with a:
            plot(charts.confusion_heatmap(result.confusion_matrix), "performance_confusion")
        with b:
            st.markdown("#### Evaluation configuration")
            st.json({"Model": result.model_name, "Dataset split": "Held-out test", "Examples": result.n_samples,
                     "Metric averaging": result.averaging, "Positive class": result.positive_label})
        st.markdown("#### Baseline comparison")
        # Compare only runs sharing the selected evaluation's full declared configuration.
        compatible = comparable_evaluations(results.evaluations, reference=result)
        if len(compatible) > 1:
            metric = st.selectbox("Comparison metric", ["f1", "accuracy", "precision", "recall"])
            plot(charts.metric_comparison(compatible, metric=metric), "performance_comparison")
            st.caption("This is a descriptive comparison of recorded test results; it does not select or tune a deployment model.")
        else:
            st.info("Baseline model evaluation results are not available for the same test set and evaluation configuration.")
    with st.expander("How to read these metrics"):
        st.write("F1-score balances precision and recall for the positive class. Precision measures how many positive predictions are correct; recall measures how many actual positives were found. Accuracy measures the share of all correct predictions and can hide errors on a minority class.")
        st.caption("Star-rating labels are proxies, not independently verified text sentiment. Results reflect only the supplied held-out set; class balancing and training are outside this dashboard.")
    with st.expander("About the proposed model"):
        st.write("The completed study evaluated frozen and fine-tuned DistilBERT hybrids, TF-IDF + XGBoost, and standalone fine-tuned DistilBERT. Install the recorded final CSV exports to view their results.")
        st.caption("Evaluation and prediction features become available when genuine trained-model results have been supplied.")


def switch_module():
    if st.session_state.get("active_workspace") == "research":
        remember_research_controls()
    elif st.session_state.get("active_workspace") == "upload":
        remember_upload_controls()
    st.session_state.pending_section = "Home" if st.session_state.application_module == UPLOAD_MODULE else st.session_state.get("research_controls", {}).get("navigation", "Dashboard Overview")
    st.session_state.scroll_revision = st.session_state.get("scroll_revision", 0) + 1


def main():
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand"><span class="brand-mark">◈</span><div>{APP_NAME}<small>MALAYSIAN REVIEW ANALYTICS</small></div></div><div class="sidebar-label">MODULES</div>', unsafe_allow_html=True)
        st.selectbox("Application module", MODULES, key="application_module", on_change=switch_module)
    if st.session_state.application_module == UPLOAD_MODULE:
        st.session_state.active_workspace = "upload"
        section_anchor("Home")
        with st.container(key="upload_workspace"):
            render_upload_page()
        finish_document()
        return
    st.session_state.active_workspace = "research"
    restore_research_controls()
    if st.session_state.get("navigation") not in RESEARCH_SECTIONS:
        st.session_state.navigation = "Dashboard Overview"
        st.session_state.pending_section = "Dashboard Overview"
    with st.sidebar:
        st.markdown('<div class="sidebar-label">RESEARCH SECTIONS</div>', unsafe_allow_html=True)
        st.radio("Navigation", RESEARCH_SECTIONS, key="navigation", label_visibility="collapsed", on_change=queue_section_scroll)
        st.caption("All five research sections stay on one page. Your uploads have their own module.")
    with st.container(key="section_home"):
        section_anchor("Home")
        render_home()
    st.divider()
    profile = os.environ.get("SENTILOKAL_RESEARCH_PROFILE", "legacy")
    final_directory = Path(os.environ.get("SENTILOKAL_EVALUATION_DIR", EVALUATIONS))
    final_exports_installed = any((final_directory / filename).exists() for filename in ("day3_all_models_test_metrics.csv", "day3_all_models_confusion.csv"))
    # The Version B report is read from aggregate exports. Legacy evaluation
    # loading may calculate metrics from test CSVs, so skip it on this path.
    results = load_results(Path(os.environ.get("SHOPEE_RESULTS_DIR", ROOT / "results")), include_evaluations=not final_exports_installed and profile == "legacy")
    content = None
    name = DEFAULT_FILENAME
    source_error = None
    with st.sidebar:
        st.divider()
        st.markdown('<div class="sidebar-label">RESEARCH DATA</div>', unsafe_allow_html=True)
    dataset = None
    try:
        if profile == "version_b":
            dataset = load_version_b_research()
            name = "version_b_cldr"
            # Legacy CSV predictions never cross into the Version B namespace.
            results.predictions = None
            try:
                results.predictions = load_version_b_predictions()
            except VersionBError as exc:
                results.errors.append(str(exc))
        elif profile == "legacy":
            research_source = read_research_source(ROOT)
            content, name = research_source.content, research_source.path.name
            dataset = cached_dataset(content, name, tuple(sorted(PRODUCT_NAMES.items())))
        else:
            raise VersionBError("Research profile must be legacy or version_b; no automatic source substitution is allowed.")
    except (ResearchDataError, VersionBError) as exc:
        source_error = str(exc)
    except (ValueError, OSError, UnicodeError, pd.errors.ParserError):
        source_error = "The configured research file could not be read. Restore the approved source and metadata before continuing."
    with st.sidebar:
        st.caption(("Version B CLDR · Explicitly configured research source" if profile == "version_b" else "Original cleaned dataset · Legacy research source") if dataset is not None else "Official research dataset · Setup required")
    if dataset is None:
        for destination, key in [("Dashboard Overview", "overview"), ("Product Sentiment Analysis", "products"), ("Customer Review Explorer", "explorer")]:
            with st.container(key=f"section_{key}"):
                section_anchor(destination)
                st.subheader(destination)
                if source_error:
                    st.warning(source_error)
                else:
                    st.info("The official research dataset needs to be configured by the project owner. Your own CSV can be explored on Upload & Analyze.")
        with st.expander("Research dataset setup · Project owner"):
            st.write("The approved research source is missing or could not be verified. Restore the approved data file or configure its read-only server path before the research charts can load.")
            st.code("data/dataset_preprocessed_v3_clean.csv\n# Or set SENTILOKAL_RESEARCH_DATA_PATH to a read-only server file.", language="text")
            st.caption("See README.md and docs/deployment.md for exact setup and checksum verification. User uploads stay separate and cannot replace the research dataset.")
        with st.container(key="section_performance"):
            section_anchor("Model Performance")
            render_performance(results)
        finish_document()
        return
    df = dataset.reviews.copy()
    df["product_name"] = [public_product_name(name, product_id) for name, product_id in zip(df.product_name, df.product_id)]
    requested = st.session_state.pop("requested_product", None)
    if requested:
        dates = df.review_date.dropna()
        bounds = (dates.min().date(), dates.max().date()) if len(dates) else None
        reset_filters(name, tuple(sorted(df.product_name.unique())), bounds)
        if requested in set(df.product_name):
            st.session_state[f"product_names_{name}"] = [requested]
            st.session_state.focus_product_name = requested
    with st.sidebar:
        st.caption(f"{len(df):,} review records · {df.product_name.nunique()} products")
        st.divider()
        st.markdown('<div class="sidebar-label">FILTER YOUR VIEW</div>', unsafe_allow_html=True)
        valid_dates = df.review_date.dropna()
        date_bounds = (valid_dates.min().date(), valid_dates.max().date()) if len(valid_dates) else None
        st.button("Reset filters", on_click=reset_filters, args=(name, tuple(sorted(df.product_name.unique())), date_bounds), use_container_width=True)
        products = st.multiselect("Products", sorted(df.product_name.unique()), default=sorted(df.product_name.unique()), key=f"product_names_{name}")
        sentiments = st.multiselect("Rating-derived sentiment", ["Positive", "Negative"], default=["Positive", "Negative"], key=f"sentiments_{name}")
        dates = df.review_date.dropna()
        start = end = None
        if len(dates):
            selection = st.date_input("Review date range", value=(dates.min().date(), dates.max().date()), min_value=dates.min().date(), max_value=dates.max().date(), key=f"dates_{name}")
            if len(selection) == 2:
                start, end = selection
            else:
                st.caption("Choose an end date to apply the date filter.")
        include_undated = st.checkbox("Include undated reviews", value=True, key=f"undated_{name}")
        st.caption("Undated records can appear in totals but are omitted from trends.")
        st.divider()
        st.markdown('<div class="sidebar-footnote"><strong>FYP research workspace</strong><br>Historical review analysis<br>Four trained architectures · Version B results</div>', unsafe_allow_html=True)
    filtered = filter_reviews(df, product_names=products, sentiments=sentiments, start=start, end=end, include_undated=include_undated)
    with st.container(key="section_overview"):
        section_anchor("Dashboard Overview")
        st.caption("Research source: Version B CLDR (approved final research dataset)." if profile == "version_b" else "Research source: original cleaned dataset. The final experiment results below refer to the separate Version B CLDR dataset.")
        st.markdown('<div class="status-note"><span class="status-badge">DATA EXPLORATION</span><span>Sentiment uses star ratings: <b>1–2★ Negative</b> · <b>4–5★ Positive</b>. Model predictions are separate.</span></div>', unsafe_allow_html=True)
        selected_dates = filtered.review_date.dropna()
        period = f"{selected_dates.min():%d %b %Y} – {selected_dates.max():%d %b %Y}" if len(selected_dates) else "No dated reviews selected"
        st.markdown(f'<div class="selection-summary"><span><i></i>{len(filtered):,} reviews selected</span><span>{filtered.product_name.nunique()} products</span><span>{period}</span></div>', unsafe_allow_html=True)
        if results.errors:
            st.warning("Result file not loaded: some model results are unavailable. Valid results and review exploration remain accessible.")
        render_overview(dataset, filtered)
    st.divider()
    with st.container(key="section_products"):
        section_anchor("Product Sentiment Analysis")
        render_products(filtered)
    st.divider()
    with st.container(key="section_explorer"):
        section_anchor("Customer Review Explorer")
        render_explorer(filtered, results)
    st.divider()
    with st.container(key="section_performance"):
        section_anchor("Model Performance")
        st.caption("Review filters do not change held-out evaluation results.")
        render_performance(results)
    finish_document()


def finish_document():
    if st.session_state.get("active_workspace") == "research":
        remember_research_controls()
    st.divider()
    description = "Uploaded review exploration · Star-rating-derived labels" if st.session_state.get("active_workspace") == "upload" else "Historical Shopee review analysis · Original customer comments preserved"
    st.caption(f"{APP_NAME} · {description}")
    destination = st.session_state.pop("pending_section", None)
    if destination in PAGES:
        scroll_to_section(destination, st.session_state.get("scroll_revision", 0))


if __name__ == "__main__":
    main()
