"""Session-only Upload & Analyze interface; never reads research result files."""
from __future__ import annotations

from hashlib import sha256

import pandas as pd
import streamlit as st

from dashboard.data import product_summary, sentiment_summary
from dashboard.runtime import refresh_charts
from dashboard.prediction_controls import clear_batch_predictions, prediction_summary, render_batch_action, render_model_controls
from dashboard.uploads import load_uploaded_dataset, uploaded_public_reviews

CONTROL_KEYS = (
    "upload_products", "upload_sentiments", "upload_dates", "upload_undated",
    "upload_search", "upload_stars", "upload_page_size", "upload_page",
    "upload_model", "upload_single_text", "upload_predicted_filter",
)


def clear_uploaded_dataset():
    generation = st.session_state.get("user_upload_generation", 0) + 1
    for key in list(st.session_state):
        if key.startswith(("user_upload_", "upload_")):
            del st.session_state[key]
    st.session_state.user_upload_generation = generation


def remember_upload_controls():
    st.session_state.user_upload_controls = {key: st.session_state[key] for key in CONTROL_KEYS if key in st.session_state}


def restore_upload_controls():
    for key, value in st.session_state.get("user_upload_controls", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value


def _panel(title, subtitle):
    st.subheader(title)
    st.caption(subtitle)


def render_upload_page():
    charts = refresh_charts()
    st.title("Upload & Analyze")
    st.markdown('<p class="hero-sub">Explore your own customer reviews in a separate workspace.</p>', unsafe_allow_html=True)
    st.markdown('<div class="status-note"><span class="status-badge">YOUR UPLOAD</span><span>Your CSV affects only this page and stays in your session. It is not saved to project files. Use <b>Clear Uploaded Dataset</b> to remove it from this session.</span></div>', unsafe_allow_html=True)
    restore_upload_controls()
    architecture, available = render_model_controls()
    with st.container(border=True, key="panel_upload_file"):
        _panel("Choose your review data", "A review text column is required. Ratings, dates, and product information are optional.")
        with st.expander("Supported CSV columns"):
            st.write("Review text: review_text, comment, customer_comment, or review. Optional stars: rating, stars, or star_rating. Optional dates: review_date, datetime, date, or created_at. Optional product names: product_name, item_name, or product_title.")
            st.caption("UTF-8 CSV · Maximum 50 MB · Product identifiers, if supplied, are used internally for grouping and never displayed or exported.")
        generation = st.session_state.get("user_upload_generation", 0)
        upload = st.file_uploader("Upload your review CSV", type=["csv"], key=f"upload_file_{generation}")
        st.button("Clear Uploaded Dataset", key="clear_user_upload", on_click=clear_uploaded_dataset)
    if upload is not None:
        content = upload.getvalue()
        digest = sha256(content).hexdigest()
        if digest != st.session_state.get("user_upload_digest") or upload.name != st.session_state.get("user_upload_filename"):
            for key in CONTROL_KEYS:
                if key not in ('upload_model','upload_single_text'):
                    st.session_state.pop(key, None)
            st.session_state.user_upload_controls = {key:st.session_state[key] for key in ('upload_model','upload_single_text') if key in st.session_state}
            clear_batch_predictions()
            st.session_state.user_upload_digest = digest
            st.session_state.user_upload_filename = upload.name
            st.session_state.user_upload_dataset = None
            try:
                st.session_state.user_upload_dataset = load_uploaded_dataset(content, upload.name)
                st.session_state.pop("user_upload_error", None)
            except ValueError as exc:
                st.session_state.user_upload_error = str(exc)
    if st.session_state.get("user_upload_filename"):
        st.write("Selected file:", st.session_state.user_upload_filename)
    if st.session_state.get("user_upload_error"):
        st.error(st.session_state.user_upload_error)
        return
    dataset = st.session_state.get("user_upload_dataset")
    if dataset is None:
        st.markdown('<div class="empty-state"><h3>Your analysis starts here</h3><p>Upload a CSV to see its overview, available charts, and searchable reviews. The official Baseus research dashboard stays separate.</p></div>', unsafe_allow_html=True)
        remember_upload_controls()
        return
    render_batch_action(dataset.records,architecture,available)
    frame = dataset.records
    with st.container(border=True, key="panel_upload_filters"):
        _panel("Filter your uploaded data", "These controls apply only to your uploaded reviews.")
        left, right = st.columns(2)
        products = left.multiselect("Uploaded products", sorted(frame.product_name.unique()), default=sorted(frame.product_name.unique()), key="upload_products")
        sentiments = right.multiselect("Uploaded rating-derived sentiment", ["Positive", "Negative", "Not classified"], default=["Positive", "Negative", "Not classified"], key="upload_sentiments")
        keep = frame.product_name.isin(products) & frame.sentiment.fillna("Not classified").isin(sentiments)
        dates = frame.review_date.dropna()
        if len(dates):
            left, right = st.columns([2, 1])
            selection = left.date_input("Uploaded review date range", value=(dates.min().date(), dates.max().date()), min_value=dates.min().date(), max_value=dates.max().date(), key="upload_dates")
            undated = right.checkbox("Include undated uploaded reviews", value=True, key="upload_undated")
            if len(selection) == 2:
                date_keep = frame.review_date.ge(pd.Timestamp(selection[0])) & frame.review_date.lt(pd.Timestamp(selection[1]) + pd.Timedelta(days=1))
                keep &= date_keep | (frame.review_date.isna() & undated)
            elif not undated:
                keep &= frame.review_date.notna()
    selected = frame.loc[keep].copy()
    binary = selected.loc[selected.eligible]
    summary = sentiment_summary(binary)
    with st.container(key="upload_overview"):
        _panel("Uploaded Dataset Overview", "Descriptive statistics from your uploaded records.")
        entries = [("Total uploaded reviews", f"{len(selected):,}"),
                   ("Uploaded products", str(selected.product_name.nunique()) if dataset.has_products else "—"),
                   ("Positive uploaded reviews", f"{summary['positive_reviews']:,}" if dataset.has_ratings else "—"),
                   ("Negative uploaded reviews", f"{summary['negative_reviews']:,}" if dataset.has_ratings else "—")]
        for col, (label, value) in zip(st.columns(4), entries):
            col.metric(label, value)
        left, right = st.columns(2)
        left.metric("Positive share · binary ratings", f"{summary['positive_percentage']:.2f}%" if len(binary) else "—")
        right.metric("Negative share · binary ratings", f"{summary['negative_percentage']:.2f}%" if len(binary) else "—")
        st.caption(f"{len(selected):,} of {len(frame):,} uploaded records selected · {len(binary):,} eligible for binary sentiment. Shares use eligible 1–2 / 4–5-star reviews only. Three-star, invalid or missing ratings and existing preprocessing exclusions are not classified. All original rows are retained.")
        dated = selected.review_date.dropna()
        if len(dated):
            st.caption(f"Observed review dates: {dated.min():%d %b %Y} – {dated.max():%d %b %Y} · {selected.review_date.isna().sum():,} undated records")
        else:
            st.caption("No valid review dates are available in this selection.")
    if not dataset.has_ratings:
        left, right = st.columns(2)
        lengths = selected.review_text.str.split().str.len()
        left.metric("Reviews with original comments", f"{int(selected.review_text.str.strip().ne('').sum()):,}")
        right.metric("Median review length · words", f"{lengths.median():.1f}" if len(lengths) else "—")
        st.info("This CSV has no star-rating column. Saved-model analysis does not require ratings. No rating-derived labels have been generated. Use Analyze Reviews with an available trained model to obtain separate model predictions.")
    elif binary.empty:
        st.info("No eligible binary star ratings are available in this selection. Three-star reviews are excluded from binary sentiment analysis.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="panel_upload_sentiment"):
            _panel("Uploaded sentiment distribution", "1–2 stars = Negative · 4–5 stars = Positive")
            if len(binary):
                st.plotly_chart(charts.sentiment_donut(binary), use_container_width=True, key="upload_donut")
            else:
                st.info("This chart needs eligible star ratings.")
    with right:
        with st.container(border=True, key="panel_upload_ratings"):
            _panel("Uploaded star ratings", "Original 1–5-star counts; 3 stars remain excluded from binary labels")
            if selected.rating.isin([1, 2, 3, 4, 5]).any():
                st.plotly_chart(charts.rating_bars(selected, include_neutral=True), use_container_width=True, key="upload_rating_chart")
            else:
                st.info("No valid star ratings are available for this chart.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="panel_upload_trend"):
            _panel("Uploaded sentiment over time", "Monthly counts from eligible dated reviews")
            if binary.review_date.notna().any():
                st.plotly_chart(charts.trend_chart(binary), use_container_width=True, key="upload_trend")
                st.caption(f"{int(binary.review_date.isna().sum()):,} undated binary reviews omitted from this chart.")
            else:
                st.info("This chart needs both eligible star ratings and valid dates.")
    with right:
        with st.container(border=True, key="panel_upload_products"):
            _panel("Uploaded product comparison", "Sentiment counts grouped by product")
            if dataset.has_products and len(binary):
                columns = ["product_name", "total_reviews", "positive_reviews", "negative_reviews", "positive_percentage", "negative_percentage"]
                st.plotly_chart(charts.product_comparison(product_summary(binary)[columns]), use_container_width=True, key="upload_comparison")
            else:
                st.info("This chart needs product information and eligible star ratings.")
    prediction_summary(selected,charts)
    with st.container(border=True, key="panel_upload_explorer"):
        _panel("Uploaded Review Explorer", "Search original comments and download matching records.")
        left, right = st.columns([2, 1])
        search = left.text_input("Search uploaded reviews", key="upload_search", placeholder="Search original comments…")
        stars = right.multiselect("Uploaded star rating", ["1", "2", "3", "4", "5", "Missing / invalid"], default=["1", "2", "3", "4", "5", "Missing / invalid"], key="upload_stars")
        rating_labels = selected.rating.map(lambda value: str(int(value)) if pd.notna(value) and value in [1, 2, 3, 4, 5] else "Missing / invalid")
        visible = selected.loc[rating_labels.isin(stars)]
        if search:
            visible = visible[visible.review_text.str.contains(search, case=False, regex=False, na=False)]
        predictions = st.session_state.get("user_upload_predictions")
        if predictions is not None:
            predicted_filter = st.selectbox("Model prediction filter",["All reviews","Predicted positive","Predicted negative"],key="upload_predicted_filter")
            if predicted_filter != "All reviews":
                wanted = 1 if predicted_filter == "Predicted positive" else 0
                visible = visible.loc[predictions.loc[visible.index].predicted_label.eq(wanted)]
        st.write(f"**{len(visible):,} uploaded reviews** match your selection")
        export = uploaded_public_reviews(visible) if predictions is None else predictions.loc[visible.index].reset_index(drop=True).copy()
        st.download_button("Download uploaded reviews · CSV", export.to_csv(index=False).encode("utf-8-sig"), "uploaded_reviews_filtered.csv", "text/csv", key="upload_export")
        left, right = st.columns(2)
        size = left.selectbox("Uploaded reviews per page", [25, 50, 100], key="upload_page_size")
        pages = max(1, (len(visible) + size - 1) // size)
        if st.session_state.get("upload_page", 1) > pages:
            st.session_state.upload_page = 1
        page = right.number_input("Uploaded review page", min_value=1, max_value=pages, step=1, key="upload_page")
        table = export.iloc[(page - 1) * size:page * size].rename(columns={"product_name": "Product", "review_text": "Original review", "rating": "Stars", "review_date": "Review date", "sentiment": "Rating-derived sentiment"})
        table = table.drop(columns=["predicted_label","prediction_model","negative_threshold"],errors="ignore").rename(columns={"predicted_sentiment":"Model-predicted sentiment","negative_score":"Negative score · uncalibrated"})
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={"Original review": st.column_config.TextColumn(width="large"), "Review date": st.column_config.DatetimeColumn(format="DD MMM YYYY HH:mm")})
        st.caption(f"Page {page} of {pages}. Export includes all {len(visible):,} matching records. Blank rating-derived sentiment means no eligible star label is available. Model predictions, when analyzed, are separate columns; scores are uncalibrated.")
    remember_upload_controls()
