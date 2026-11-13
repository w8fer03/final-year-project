# SentiLokal – Malaysian E-Commerce Review Sentiment Analytics

**Mixed-Language Electronic Product Reviews | DistilBERT + XGBoost**

SentiLokal is a Streamlit exploration dashboard for Malaysian e-commerce review sentiment analytics, developed for the FYP *Sentiment Analysis of Mixed-Language Electronic Product Reviews on Shopee Malaysia Using DistilBERT and XGBoost*.

The completed experiments evaluated four architectures on each of two test splits. Research charts continue to use **star-rating-derived labels**: 1–2 stars → Negative (0), 4–5 stars → Positive (1). Three-star and invalid-rating records are excluded from binary analyses. These proxy labels can disagree with the comment or a model prediction. The Model Performance section imports the genuine eight-record Version B exports; optional upload inference is separate.

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

Poppins is bundled locally in `assets/fonts/` with its SIL Open Font License; browsers do not need Google Fonts access. The base text size is 17px, with slightly larger labels and chart text. Streamlit 1.45 or newer is required for the custom-font theme settings.

The interface uses a Baseus-inspired yellow and charcoal theme, warm neutral backgrounds, white rounded panels, a yellow summary card with dark text, and consistent green/red sentiment indicators. Charts and filters remain native Streamlit/Plotly controls with keyboard focus states and a narrow-screen layout.

## Updating a running dashboard

From the repository directory in another terminal, update the dashboard branch using a fast-forward pull:

```bash
git switch codex/shopee-sentiment-dashboard
git pull --ff-only origin codex/shopee-sentiment-dashboard
```

Then stop the running Streamlit process with **Ctrl+C** in its terminal and restart it from the repository directory:

```bash
.venv/bin/python -m streamlit run app.py
```

A full server restart remains recommended after pulling application changes. The app also checks its local chart source on each rerun and refreshes a stale imported chart module when its source changes or a required chart function is missing. This prevents the `rating_imbalance` AttributeError seen when Streamlit had rerun a newer `app.py` while retaining older chart code. This check reads only the repository's chart source; no uploaded content is executed.

## Research page and Upload & Analyze

The sidebar **Application module** selector contains exactly **Baseus Research Dashboard** and **Upload & Analyze**. A new session starts in the research module at **Dashboard Overview**, showing the real results automatically. The original radio list retains the five research section shortcuts. These selections render the same complete research document in this order: Home, Dashboard Overview, Product Sentiment Analysis, Customer Review Explorer, and Model Performance. Selecting a section or a landing-page action scrolls within that document. **Upload & Analyze** is the separate second module, not a research section. Its page never reads the research result files or replaces the research input. Research controls and upload controls are stored separately and restored when returning to their page.

- **Home:** an introduction to SentiLokal and five product cards. **Open dashboard**, **View products**, and **Explore product** scroll to the analytical sections; a product card selects that product for exploration. Home works before the approved research dataset is provisioned. Without it, the analytical sections retain their headings and show deployment setup instructions; the model-status section remains available.
- **Dashboard Overview:** filtered record/product counts, positive/negative shares, six charts (rating-imbalance treemap, original-review length distribution, star ratings, product sentiment comparison, sentiment donut, monthly trends), and the source quality report.
- **Product Sentiment Analysis:** compare the five named products using a six-column table (Product, Total reviews, Positive, Negative, Positive %, Negative %), focus-product KPIs, and monthly/yearly trends. Product names identify the collection throughout the customer interface.
- **Customer Review Explorer:** literal case-insensitive Malay/English keyword search, star filters, paginated sortable tables, original customer comments, individual review details, filtered CSV download. Genuine stored predictions can add coverage and agreement filters later.
- **Model Performance:** informative empty states now. Valid held-out evaluation files automatically enable F1, accuracy, precision, recall, a confusion matrix, recorded configuration, and compatible baseline comparisons. Sidebar exploration filters never modify evaluation metrics.

Customer screens and CSV downloads exclude listing IDs, review IDs, cleaned text, source-schema details, and internal artifact metadata. `dashboard/presentation.py` projects the public review fields: product name, original review, stars, date, and rating-derived sentiment, with genuine model prediction/confidence fields only when available. Source identifiers and existing cleaned text remain in private backend data for provenance, preprocessing continuity, and future model integration; their source values are unchanged. Generic labels derived from an unknown listing ID appear as **Other product**.

Global product, sentiment, and inclusive date filters update all exploration charts and totals. Clearing a product/sentiment/rating selection means no matches. Undated records can be included in totals but never contribute to time charts. Empty selections and unavailable optional data have explanatory states.

Review length counts whitespace-separated words in each original customer comment, including zero for empty comments. Only aggregated length counts enter the chart. This does not use cleaned text, model tokens, or inferred linguistic annotations. The rating treemap counts eligible 1-, 2-, 4-, and 5-star reviews; three-star and invalid ratings remain excluded. Aspect-level charts from the layout reference are not generated because validated aspect annotations are not supplied. Search and star filters in the review explorer affect its table and download; the six summary charts continue to reflect the sidebar selection.

The sidebar **Reset filters** button restores all products, both sentiments, the full observed date range, and inclusion of undated records. The selection summary shows the current review count, product count, and observed period. The focus-product dropdown changes its summary cards and trend; the comparison table/chart continue to reflect the sidebar selection.

## Local product photographs

The five product cards and focused-product panel resolve photographs through `dashboard/media.py` using `assets/products/catalog.toml`. The catalog contains one named entry per product. No fabricated, generated, unrelated, or remotely hotlinked product photographs are supplied. The current catalog has empty image paths, so the interface states **Product photo not available yet**.

Official Baseus/Shopee photo access returned HTTP 403 in this environment, preventing verification and download of the requested photographs. Required access additions were saved in the cloud environment configuration draft. Review and save those changes in environment settings, then publish the environment before retrying photo retrieval; a saved draft does not itself apply network changes or establish that the remote service permits access. Genuine image files can also be supplied locally without fetching them from the application.

To add a verified photograph, place its JPEG, PNG, or WebP file under `assets/products/` and set that product's `image_path` to the relative filename. Record the actual original public HTTPS page or image URL in `source_url` for provenance; do not invent URLs or include credentials. The catalog name must match the product exactly. The loader checks that the path stays within `assets/products/`, that its format is supported, and that the image decodes successfully. These file checks do not independently establish product identity: the owner must confirm that the photograph depicts the correct product and may be used. Valid source links appear beside available photographs.

Photo acquisition is a separate owner task. The running dashboard reads local assets only, makes no external image requests, and does not depend on Baseus or Shopee network access. Adding verified files and updating the catalog enables the images on rerun.

## Local datasets and actual scope

The prepared workspace has byte-identical copies of both originally supplied CSVs in `data/`. Only the approved **`dataset_preprocessed_v3_clean.csv`** automatically feeds the research dashboard. The labelled alternative is not selected, merged, or used as a fallback. There is no research CSV uploader or dataset selector. The owner has since committed the approved cleaned file under `data/` and a duplicate at the repository root. The app reads only the approved `data/` file (or an explicit read-only server path). A normal clone now has this source; a code-only/private deployment may still require provisioning, as described in [deployment.md](deployment.md). Source repository permissions must protect internal fields; public dashboard projections alone do not hide Git contents. The application does not download research data from GitHub.

| Source | Source records | Positive | Negative | Products | Listing IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dataset_preprocessed_v3_clean.csv` (default) | 19,295 | 19,013 | 282 | 5 | 17 |
| `dataset v3 labelled 2.csv` | 23,527 | 23,159 | 368 | 5 | 17 |

Both supplied files span **14 July 2020–15 August 2026**, rather than the proposal's planned January 2022–June 2026. The dashboard presents the observed dates without silently restricting the research period. Use filters to examine the desired subset.

### Product names and listing IDs

The 17 Shopee listing IDs represent the **five products** below, following the mapping supplied by the project owner. This table documents private backend configuration for the project owner; the customer interface and downloads show product names without source identifiers. The product KPI, sidebar product filter, comparison chart/table, and focus-product trends group records by these names. Grouping affects display aggregates only: no source rows are merged, removed, or rewritten.

| Product | Listing IDs |
| --- | --- |
| Baseus H17 Encok 3.5mm Wired Earphone | 14750761025, 27956652952 |
| Baseus C17 Type-C Earphones | 29224862918, 19495504570, 15004032602, 27104868593 |
| Baseus Encok CZ11 / HZ11 | 28305736055, 28007706814, 29005735910 |
| Baseus WM01 TWS Wireless Earphones | 27020823629, 11315577801, 4039048325 |
| Baseus H19 | 25502231352, 28656652968, 22050094510, 23987350200, 7678007496 |

Maintain this mapping in `dashboard/products.py`. Known IDs use the supplied product names. Unknown IDs retain a provided source name internally, or an ID-based fallback when no name is present. The presentation layer replaces identifier-derived fallback labels with **Other product** and does not display unmapped IDs. Additional uploads are not silently assigned to one of the five known products. Changes to the mapping invalidate the data cache.

Repeated source review IDs and duplicate records are retained and reported; totals count **source records**. The legacy cleaned file has 2,091 repeated-ID occurrences after the first, with no exact duplicate rows. The labelled file has 5,610 repeated-ID occurrences and 4,077 exact duplicate rows. The approved Version B source has 18,705 unique source-row keys. No balancing, training, deduplication, or rewriting of the original preprocessing scripts occurs. Upload inference invokes a separately vendored, checksum-verified copy of the supplied Version B preprocessing code.

### Upload & Analyze

Choose **Upload & Analyze** from the sidebar **Application module** selector to open its dedicated file uploader. Review text is required; ratings, dates, and products are optional. Supported names are documented on the page. Uploads are parsed without writing files and without Streamlit's global data cache. Each session retains its own parsed records until cleared or the session expires. **Clear Uploaded Dataset** clears the session analysis, filters, filename, and file-selector generation so the old CSV is not immediately reloaded.

The page retains its rating-based statistics, charts, and paginated explorer. Three-star and unrated rows remain visible with blank rating-derived sentiment. Select one of four trained architectures and click **Analyze Reviews** to run an available saved model on every original uploaded row. Text-only CSVs need no ratings for model prediction. A manual single-review input is also available. Genuine outputs add predicted totals, a separate distribution chart, a prediction filter, and export columns. Scores are explicitly uncalibrated; incoming prediction/confidence columns are discarded. Unavailable models show specific setup errors while exploration remains usable.

Upload product/sentiment/date filters apply only to uploaded analysis. Its search and star filters affect its review table and export. No uploaded file, option, or prediction column changes the official research dataset, its filters, or its genuine held-out results. Research source configuration is server-side: `SENTILOKAL_RESEARCH_DATA_PATH` (or legacy `SHOPEE_DATA_PATH` if the new variable is absent), with approved checksum verification. See [deployment.md](deployment.md).

### Research schema for owner integration

For owner integration, the private schema layer maps these source columns without changing the source file. This schema is not displayed in the dashboard:

| Source | Canonical field |
| --- | --- |
| `item_id` / `product_id` | `product_id` |
| `comment` / `review_text` | `review_text` |
| `stars` / `rating` | `rating` |
| `datetime` / `review_date` | `review_date` |
| `cmtid` / `review_id` | `source_review_id` |
| `cleaned_comment_v3` / `cleaned_text` / `cleaned_comment` | `cleaned_text` |

For the **research loader**, product ID, original comment, star rating, and review date columns are required. The separate upload loader requires only original comments. Missing values are counted; invalid dates remain undated. Ambiguous slash dates use month/day/year as in the provided Shopee files. Existing nonempty `preprocess_status` values other than `keep` are respected as exclusions from binary statistics. Existing labels are compared with ratings but do not override the rating-based definition.

For the legacy source, `review_id` is a unique canonical identity: unique `cmtid` values are retained, repeated/missing ones receive a stable source-row suffix. Version B uses its separate `version_b:` source-key namespace. These keys remain private. See the [research prediction contract](model_integration.md#research-source-and-saved-predictions). Uploaded CSV prediction claims are discarded; only explicit saved-model analysis can add predictions.

## Final model integration

See [model_integration.md](model_integration.md) for the active Version B backend, exact negative-score thresholds, artifact requirements, source migration, and verification limits. The supplied two aggregate CSV exports under `evaluation/version_b/` populate Model Performance automatically. Comparisons stay within each test split, emphasize macro and negative F1, and do not select a deployment model using held-out performance.

The legacy `results/` contract is retained for compatibility when no final aggregate report is installed. Current Version B evaluation reporting never runs held-out inference or recomputes metrics from test predictions. It validates aggregate consistency and preserves the recorded values, including negative PR-AUC. Per-review held-out membership was not supplied, so comparison groups rely on the owner's split declarations and consistent support counts.

`dashboard/prediction_backend.py` is the active four-architecture interface. The older `dashboard/inference.py` and `dashboard/model_loader.py` remain independent legacy APIs. Real TF-IDF inference, saved-tokenizer parity, and full-corpus preprocessing parity have been verified locally. The code-only reference confirms CLS features and dynamic padding. Transformer encoder tests use doubles until the original fine-tuned weights and frozen encoder are supplied.

## Project structure

```text
app.py                       Single-document dashboard and section rendering
dashboard/research.py        Approved local/read-only research source and checksum
dashboard/research_view.py    Minimal cached display data; original fields preserved
dashboard/uploads.py         Session-only optional-schema upload normalization
dashboard/upload_page.py     Separate Upload & Analyze interface
dashboard/navigation.py      Sidebar section anchors and local scrolling component
dashboard/runtime.py         Refresh stale local chart imports after code updates
dashboard/data.py            Read-only schema normalization, filters, aggregates
dashboard/products.py        Owner-supplied product-name / listing-ID mapping
dashboard/presentation.py    Public review fields and safe product labels
dashboard/media.py           Local product-photo validation and resolution
dashboard/charts.py          Reusable Plotly figures
dashboard/artifacts.py       Validated prediction and held-out evaluation loading
dashboard/model_loader.py    Verified future training-contract adapter
dashboard/inference.py        Unavailable-by-default prediction service boundary
assets/style.css             Dashboard presentation
assets/products/catalog.toml Product-photo filenames and source provenance
.streamlit/config.toml       Streamlit theme/settings
data/                        Owner-added approved research input; other CSVs ignored
results/                     Real future predictions/evaluation files (empty)
models/version_b/            Ignored owner-provisioned weights, vectorizer, private CSVs and CLDR XML
resources/version_b/         Verified preprocessing resources and deployment/source manifests
evaluation/version_b/        The two genuine aggregate held-out exports
dashboard/prediction_backend.py Four trained-model adapters and resource cache
dashboard/prediction_controls.py Session-only batch and single-review inference controls
dashboard/version_b_research.py Explicit source-ID-based Version B research loader
dashboard/final_evaluations.py Eight-record aggregate validation without held-out inference
docs/dashboard.md            Dashboard setup and usage guide
docs/deployment.md           Private research data provisioning instructions
scripts/check_research_data.py Read-only research deployment readiness check
docs/model_integration.md     Result and model contracts
tests/                       Transformation, integration, and app interaction tests
```

## Validate

```bash
python -m unittest discover -s tests -v
```

Tests use small explicitly synthetic, temporary fixtures where needed to check future file validation; those fixtures are never loaded as dashboard findings. Application tests also exercise the real local datasets using Streamlit AppTest. Tests requiring those two private CSV files are explicitly skipped in a fresh clone until the files are provided; skipped tests do not validate dataset-dependent behavior.

The model-performance page intentionally has no current scores or confusion matrix. Validated linguistic annotations are also unavailable; no code-mixing, slang, emoji, or informal-spelling frequencies are inferred from the text or preprocessing diagnostics.
