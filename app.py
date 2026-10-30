"""SentiLokal review analytics. Run from this directory: streamlit run app.py."""
from __future__ import annotations

import io
import os
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.artifacts import attach_predictions, comparable_evaluations, load_results
from dashboard.data import filter_reviews, load_dataset, product_summary, sentiment_summary
from dashboard.media import product_media
from dashboard.presentation import public_product_name, public_reviews
from dashboard.products import PRODUCT_GROUPS, PRODUCT_NAMES

ROOT = Path(__file__).resolve().parent
APP_NAME = "SentiLokal"
APP_DESCRIPTOR = "Malaysian E-Commerce Review Sentiment Analytics"
APP_TITLE = f"{APP_NAME} – {APP_DESCRIPTOR}"
PAGES = ["Home", "Dashboard Overview", "Product Sentiment Analysis", "Customer Review Explorer", "Model Performance"]

st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading review data…")
def cached_dataset(content: bytes, name: str, product_catalog: tuple[tuple[str, str], ...]):
    # The catalog is part of the cache key so correcting a name refreshes the data.
    return load_dataset(io.BytesIO(content), source_name=name)


def plot(fig, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def empty(title: str, message: str):
    # Only call with application-owned static strings.
    st.markdown(f'<div class="empty-state"><h3>{title}</h3><p>{message}</p></div>', unsafe_allow_html=True)


def kpis(df: pd.DataFrame, include_products: bool = True):
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
            with st.container(key=f"kpi_{metric_key}"):
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
    st.session_state.navigation = page
    if product is not None:
        st.session_state.requested_product = product


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
        ("03", "Read the feedback", "Search actual comments and compare the original and existing cleaned text."),
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
    st.caption("Understand the collection before modelling: volume, rating-derived sentiment, and historical patterns.")
    kpis(df)
    date_caption(df)
    if df.empty:
        st.info("No reviews match the current filters. Expand the selection in the sidebar.")
    else:
        a, b = st.columns([1, 1.7], gap="medium")
        with a:
            with st.container(border=True, key="panel_distribution"):
                panel_title("Sentiment distribution", "How customers rated their experience")
                plot(charts.sentiment_donut(df), "overview_donut")
        with b:
            with st.container(border=True, key="panel_history"):
                panel_title("Sentiment over time", "Monthly review volume across the selected period")
                trend(df, "overview_trend")
        a, b = st.columns([1.7, 1], gap="medium")
        with a:
            with st.container(border=True, key="panel_ratings"):
                panel_title("Original star ratings", "The ratings behind your sentiment labels")
                plot(charts.rating_bars(df), "overview_ratings")
        with b:
            with st.container(border=True, key="panel_method"):
                panel_title("Understanding the results", "Explore the data before modelling")
                st.markdown('<div class="method-row"><span class="sentiment-dot positive"></span><div><strong>4–5 stars</strong><p>Positive rating-derived sentiment</p></div></div><div class="method-row"><span class="sentiment-dot negative"></span><div><strong>1–2 stars</strong><p>Negative rating-derived sentiment</p></div></div>', unsafe_allow_html=True)
                summary = sentiment_summary(df)
                st.info(f"{summary['positive_percentage']:.2f}% positive · {summary['negative_percentage']:.2f}% negative in this selection.")
                st.caption("The original class imbalance is preserved. Star labels can differ from the sentiment expressed in a comment. No balancing or training is performed here.")
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
        st.caption("The two provided CSVs are alternative versions, not concatenated. Ratings 1–2 → Negative (0), 4–5 → Positive (1). Three-star and invalid-rating records never enter binary statistics. Existing preprocessing exclusions are respected. Original CSVs and cleaned text are unchanged.")


def render_products(df):
    st.subheader("Product Sentiment Analysis")
    st.caption("Compare customer feedback across products, then explore one product in detail.")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    summary = product_summary(df)
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
    kpis(focus, include_products=False)
    with st.container(border=True, key="panel_comparison"):
        panel_title("Compare products", "Positive and negative review counts for your sidebar selection")
        plot(charts.product_comparison(summary), "product_comparison")
    with st.container(border=True, key="panel_product_table"):
        panel_title("Product comparison", "A clear view of review counts and sentiment share")
        # Project the display data so listing metadata cannot be re-enabled in the column menu.
        display_columns = ["product_name", "total_reviews", "positive_reviews", "negative_reviews", "positive_percentage", "negative_percentage"]
        st.dataframe(summary[display_columns].rename(columns={
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
        st.caption("Select a column heading to sort. Counts reflect your sidebar filters.")
    with st.container(border=True, key="panel_product_trend"):
        panel_title("Product sentiment over time", selection)
        c1, c2 = st.columns([2, 1])
        percentage = c1.radio("Trend measure", ["Review count", "Sentiment percentage"], horizontal=True) == "Sentiment percentage"
        freq = c2.selectbox("Time grouping", ["Monthly", "Yearly"])
        trend(focus, "product_trend", percentage=percentage, frequency="MS" if freq == "Monthly" else "YS")


def review_table(df):
    return public_reviews(df).rename(columns={
        "product_name": "Product",
        "review_text": "Original review", "cleaned_text": "Existing cleaned review", "rating": "Stars", "review_date": "Review date",
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
        search = left.text_input("Search original reviews", placeholder="Search for bagus, rosak, sound, battery…")
        stars = right.multiselect("Star rating", [1, 2, 4, 5], default=[1, 2, 4, 5])
    if search:
        df = df[df.review_text.str.contains(search, case=False, regex=False, na=False)]
    df = df[df.rating.isin(stars)]
    if has_predictions:
        status = st.selectbox("Prediction agreement with star label", ["All reviews", "Matches rating label", "Differs from rating label", "No prediction"])
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
        with st.expander("Compare existing cleaned text"):
            if row.cleaned_text.strip():
                st.text(row.cleaned_text)
                st.caption("Read directly from the existing dataset. The dashboard does not run or change preprocessing.")
            else:
                st.info("Cleaned text is not available for this review.")
    with st.expander("Linguistic characteristics"):
        st.info("Validated linguistic annotations for code-mixing, slang, abbreviations, emojis, and informal spelling are not available. No language classifications or frequencies are inferred.")
        st.caption("Preprocessing diagnostics, when present in the source, are not treated as validated linguistic annotations. Monolingual reviews remain included.")


def render_performance(results):
    st.subheader("Model Performance")
    st.caption("Held-out evaluation · Negative = 0 · Positive = 1 · F1-score is the primary project metric")
    if not results.evaluations:
        empty("Evaluation results are not available yet", "The project is currently in data preprocessing. No trained model, test predictions, or evaluation metrics have been supplied. Valid held-out results will appear here once they are added.")
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
            best_score = max(e.metrics['f1'] for e in compatible)
            best_names = ", ".join(e.model_name for e in compatible if e.metrics['f1'] == best_score)
            st.write(f"Highest recorded F1-score in this comparable group: **{best_names}** ({best_score:.3f}).")
        else:
            st.info("Baseline model evaluation results are not available for the same test set and evaluation configuration.")
    with st.expander("How to read these metrics"):
        st.write("F1-score balances precision and recall for the positive class. Precision measures how many positive predictions are correct; recall measures how many actual positives were found. Accuracy measures the share of all correct predictions and can hide errors on a minority class.")
        st.caption("Star-rating labels are proxies, not independently verified text sentiment. Results reflect only the supplied held-out set; class balancing and training are outside this dashboard.")
    with st.expander("About the proposed model"):
        st.write("Proposed architecture: DistilBERT contextual features → XGBoost binary classifier. No trained checkpoint, pooling method, split ratio, or cross-validation configuration is assumed.")
        st.caption("Evaluation and prediction features become available when genuine trained-model results have been supplied.")


def main():
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand"><span class="brand-mark">◈</span><div>{APP_NAME}<small>MALAYSIAN REVIEW ANALYTICS</small></div></div><div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed")
    if page == "Home":
        render_home()
        return
    with st.sidebar:
        st.divider()
        st.markdown('<div class="sidebar-label">YOUR DATA</div>', unsafe_allow_html=True)
        files = sorted((ROOT / "data").glob("*.csv"))
        preferred = next((p for p in files if "preprocessed" in p.name), None)
        default = files.index(preferred) if preferred else 0
        configured = os.environ.get("SHOPEE_DATA_PATH")
        with st.expander("Upload a CSV", expanded=False):
            upload = st.file_uploader("Use another review CSV", type=["csv"])
        if upload is not None:
            content, name = upload.getvalue(), upload.name
        elif configured:
            try:
                path = Path(configured).expanduser()
                content, name = path.read_bytes(), path.name
            except OSError:
                st.error("The configured review dataset is unavailable. Please choose another file.")
                st.stop()
        elif files:
            names = {"dataset_preprocessed_v3_clean.csv": "Cleaned dataset", "dataset v3 labelled 2.csv": "Labelled dataset"}
            source = st.selectbox("Review dataset", files, index=default, format_func=lambda p: names.get(p.name, p.name))
            content, name = source.read_bytes(), source.name
        else:
            st.info("Add a review CSV to data/ or upload one above.")
            st.stop()
    try:
        dataset = cached_dataset(content, name, tuple(sorted(PRODUCT_NAMES.items())))
    except (ValueError, OSError, UnicodeError, pd.errors.ParserError):
        st.error("The review file could not be loaded. Please use a supported dataset containing products, original comments, star ratings, and review dates.")
        st.stop()
    results = load_results(Path(os.environ.get("SHOPEE_RESULTS_DIR", ROOT / "results")))
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
        st.markdown('<div class="sidebar-footnote"><strong>FYP research workspace</strong><br>Historical review analysis<br>DistilBERT + XGBoost · Integration pending</div>', unsafe_allow_html=True)
    filtered = filter_reviews(df, product_names=products, sentiments=sentiments, start=start, end=end, include_undated=include_undated)
    st.markdown(f'<div class="workspace-bar"><div class="workspace-name"><span class="mini-mark">◈</span> {APP_NAME.upper()} WORKSPACE</div><span class="workspace-tag">BASEUS · SHOPEE MALAYSIA</span></div>', unsafe_allow_html=True)
    st.title(APP_NAME)
    st.markdown(f'<p class="hero-sub">{APP_DESCRIPTOR}</p>', unsafe_allow_html=True)
    st.markdown('<div class="status-note"><span class="status-badge">DATA EXPLORATION</span><span>Sentiment uses star ratings: <b>1–2★ Negative</b> · <b>4–5★ Positive</b>. Model predictions are separate.</span></div>', unsafe_allow_html=True)
    if page != "Model Performance":
        selected_dates = filtered.review_date.dropna()
        period = f"{selected_dates.min():%d %b %Y} – {selected_dates.max():%d %b %Y}" if len(selected_dates) else "No dated reviews selected"
        st.markdown(f'<div class="selection-summary"><span><i></i>{len(filtered):,} reviews selected</span><span>{filtered.product_name.nunique()} products</span><span>{period}</span></div>', unsafe_allow_html=True)
    if results.errors:
        st.warning("Result file not loaded: some model results are unavailable. Valid results and review exploration remain accessible.")
    if page == "Dashboard Overview":
        render_overview(dataset, filtered)
    elif page == "Product Sentiment Analysis":
        render_products(filtered)
    elif page == "Customer Review Explorer":
        render_explorer(filtered, results)
    else:
        st.caption("Review filters do not change held-out evaluation results.")
        render_performance(results)
    st.divider()
    st.caption(f"{APP_NAME} · Historical Shopee review analysis · Original comments and existing cleaned text preserved")


if __name__ == "__main__":
    main()
