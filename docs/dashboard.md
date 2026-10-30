# Shopee Sentiment Analysis Dashboard

**Mixed-Language Electronic Product Reviews | DistilBERT + XGBoost**

A Streamlit exploration dashboard for the FYP *Sentiment Analysis of Mixed-Language Electronic Product Reviews on Shopee Malaysia Using DistilBERT and XGBoost*.

The project is at **data preprocessing**, before class balancing, training, and evaluation. Current statistics use **star-rating-derived labels**, never model predictions: 1–2 stars → Negative (0), 4–5 stars → Positive (1). Three-star and invalid-rating records are excluded from binary analyses. Customer star ratings are proxy labels and may disagree with the actual comment.

## Run locally

Requires Python 3.10 or newer. From the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead. In the prepared cloud workspace:

```bash
cd /workspace/final-year-project
source .venv/bin/activate
streamlit run app.py
```

No API keys, databases, or GPU are needed. Streamlit and Plotly are the only UI technologies. The included `.streamlit/config.toml` configures the light theme and disables usage telemetry.

The interface uses a light mint and teal theme, white rounded panels, a high-contrast summary card, and consistent green/red sentiment indicators. Charts and filters remain native Streamlit/Plotly controls with keyboard focus states and a narrow-screen layout.

## Four pages

- **Dashboard Overview:** filtered record/product counts, positive/negative shares, sentiment donut, original star ratings, historical monthly trends, class imbalance, source quality report.
- **Product Sentiment Analysis:** compare the five named products using a six-column table (Product, Total reviews, Positive, Negative, Positive %, Negative %), focus-product KPIs, and monthly/yearly trends. Listing IDs and listing counts are excluded from the comparison table and chart tooltips; original IDs remain available in review details, exports, and the dataset-quality information.
- **Customer Review Explorer:** literal case-insensitive Malay/English keyword search, star filters, paginated sortable tables, original and existing cleaned text, individual review details, filtered CSV download. Genuine stored predictions can add coverage and agreement filters later.
- **Model Performance:** informative empty states now. Valid held-out evaluation files automatically enable F1, accuracy, precision, recall, a confusion matrix, recorded configuration, and compatible baseline comparisons. Sidebar exploration filters never modify evaluation metrics.

Global product, sentiment, and inclusive date filters update all exploration charts and totals. Clearing a product/sentiment/rating selection means no matches. Undated records can be included in totals but never contribute to time charts. Empty selections and unavailable optional data have explanatory states.

The sidebar **Reset filters** button restores all products, both sentiments, the full observed date range, and inclusion of undated records. The selection summary shows the current review count, product count, and observed period. The focus-product dropdown changes its summary cards and trend; the comparison table/chart continue to reflect the sidebar selection.

## Local datasets and actual scope

The prepared workspace has byte-identical copies of both uploaded CSVs in `data/`, selected as alternative versions in the sidebar. They are **not merged**. These files stay local and are excluded from the Git branch. After cloning, place your own original files in `data/`, use `SHOPEE_DATA_PATH`, or upload a CSV in the sidebar. No review data is downloaded from GitHub by the application.

| Source | Source records | Positive | Negative | Products | Listing IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dataset_preprocessed_v3_clean.csv` (default) | 19,295 | 19,013 | 282 | 5 | 17 |
| `dataset v3 labelled 2.csv` | 23,527 | 23,159 | 368 | 5 | 17 |

Both supplied files span **14 July 2020–15 August 2026**, rather than the proposal's planned January 2022–June 2026. The dashboard presents the observed dates without silently restricting the research period. Use filters to examine the desired subset.

### Product names and listing IDs

The 17 Shopee listing IDs represent the **five products** below, following the mapping supplied by the project owner. The product KPI, sidebar product filter, comparison chart/table, and focus-product trends group records by these names. Review details and CSV exports retain each original listing ID and canonical review ID. Grouping affects display aggregates only: no source rows are merged, removed, or rewritten.

| Product | Listing IDs |
| --- | --- |
| Baseus H17 Encok 3.5mm Wired Earphone | 14750761025, 27956652952 |
| Baseus C17 Type-C Earphones | 29224862918, 19495504570, 15004032602, 27104868593 |
| Baseus Encok CZ11 / HZ11 | 28305736055, 28007706814, 29005735910 |
| Baseus WM01 TWS Wireless Earphones | 27020823629, 11315577801, 4039048325 |
| Baseus H19 | 25502231352, 28656652968, 22050094510, 23987350200, 7678007496 |

Maintain this mapping in `dashboard/products.py`. Known IDs use the supplied product names. Unknown IDs keep a provided source name, or an ID-based fallback when no name is present. The dataset overview identifies unmapped IDs; additional uploads are not silently assigned to one of the five known products. Changes to the mapping invalidate the data cache.

Repeated source review IDs and duplicate records are retained and reported; totals therefore count **source records**, not a claim of deduplicated customers/reviews. The cleaned file has 2,091 repeated-ID occurrences after the first, with no exact duplicate rows. The labelled file has 5,610 repeated-ID occurrences and 4,077 exact duplicate rows. Deduplication or revised research scope belongs in a separate data decision. No balancing, training, inference, or rewriting of the existing preprocessing pipeline occurs.

### Bring another CSV

Use the sidebar CSV uploader, place a file in `data/`, or set `SHOPEE_DATA_PATH` to a local CSV. The environment override takes precedence over local directory files; uploads take precedence over both.

The schema layer maps these source columns without changing the source file:

| Source | Canonical field |
| --- | --- |
| `item_id` / `product_id` | `product_id` |
| `comment` / `review_text` | `review_text` |
| `stars` / `rating` | `rating` |
| `datetime` / `review_date` | `review_date` |
| `cmtid` / `review_id` | `source_review_id` |
| `cleaned_comment_v3` / `cleaned_text` / `cleaned_comment` | `cleaned_text` |

Product ID, original comment, star rating, and review date **columns** are required. Missing values are counted; invalid dates remain undated. Ambiguous slash dates use month/day/year as in the provided Shopee files. Existing nonempty `preprocess_status` values other than `keep` are respected as exclusions from binary statistics. Existing labels are compared with ratings but do not override the rating-based definition.

`review_id` is a unique canonical source-row identity: unique `cmtid` values are retained, repeated/missing ones receive a stable source-row suffix. Export the selected dataset's canonical `review_id` values before preparing prediction files. CSV export uses canonical column names. No arbitrary prediction columns in uploaded review CSVs are trusted; prediction results must use the separate validated result contract.

## Future integration

See [model_integration.md](model_integration.md) for complete schemas. Place real `evaluation_<model>.json` files and optionally `predictions.csv` in `results/` (or set `SHOPEE_RESULTS_DIR`). The app discovers them on every rerun, so adding results requires no UI rebuild. Invalid files show errors while data exploration remains usable. No sample metrics or predictions are shipped in `results/`.

Evaluation requires an explicit held-out test declaration, test-set identity, and evaluation configuration. Supply test `y_true`/`y_pred` CSVs or consistent recorded metrics plus a confusion matrix and sample count. Baselines must share the test set and evaluation settings. File validation checks structure and numerical consistency; it cannot independently prove that test data was held out during training.

`dashboard/model_loader.py` provides a lazy, optional local DistilBERT–XGBoost adapter requiring the actual tokenizer, safetensors encoder, XGBoost JSON/UBJ, pooling, threshold, and registered original preprocessing function. No model dependencies load during Stage 1. Install matching training-time PyTorch/Transformers/XGBoost versions only when integrating actual trained artifacts. The adapter has not been exercised with trained artifacts because none exist yet.

## Project structure

```text
app.py                       Streamlit navigation and pages
dashboard/data.py            Read-only schema normalization, filters, aggregates
dashboard/products.py        Owner-supplied product-name / listing-ID mapping
dashboard/charts.py          Reusable Plotly figures
dashboard/artifacts.py       Validated prediction and held-out evaluation loading
dashboard/model_loader.py    Optional future model adapter
assets/style.css             Dashboard presentation
.streamlit/config.toml       Streamlit theme/settings
data/                        Local CSV inputs (ignored by Git)
results/                     Real future predictions/evaluation files (empty)
models/                      Future local trained artifacts (empty)
docs/dashboard.md            Dashboard setup and usage guide
docs/model_integration.md     Result and model contracts
tests/                       Transformation, integration, and app interaction tests
```

## Validate

```bash
python -m unittest discover -s tests -v
```

Tests use small explicitly synthetic, temporary fixtures where needed to check future file validation; those fixtures are never loaded as dashboard findings. Application tests also exercise the real local datasets using Streamlit AppTest. Tests requiring those two private CSV files are explicitly skipped in a fresh clone until the files are provided; skipped tests do not validate dataset-dependent behavior.

The model-performance page intentionally has no current scores or confusion matrix. Validated linguistic annotations are also unavailable; no code-mixing, slang, emoji, or informal-spelling frequencies are inferred from the text or preprocessing diagnostics.
