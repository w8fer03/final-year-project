# Shopee Reviews Scraper: A Technical Evolution Case Study

A collection of python-based crawlers designed to harvest product reviews (usernames, star ratings, comment texts, product variants, and timestamps) from **Shopee**. 

Rather than just presenting a final product, this repository is organized as a **case study** to showcase the technical evolution from a standard UI-based web scraper to a highly optimized, signature-bypassing hybrid API injector.

---

## 📈 The Technical Evolution

Crawling Shopee reviews is notoriously challenging due to their advanced Anti-bot mechanisms (which block standard automated browsers with Captchas) and secure Internal APIs (which return `403 Forbidden` if signatures, cookies, or timestamps are invalid). 

Below is the journey of how those hurdles were systematically resolved:

### 🔴 Phase 1: V1 - Selenium DOM Scraping
- **Directory**: [`v1_selenium_dom/`](v1_selenium_dom)
- **Concept**: Bypasses the initial anti-bot system by attaching Selenium to a **real, active Google Chrome browser** in debugging mode (`--remote-debugging-port=9222`). Once attached, it scrolls the page to trigger lazy loading and scrapes reviews directly from the HTML DOM tree using CSS selectors.
- **Limitation**: Brittle (breaks when Shopee modifies class names) and slow (must paginate manually in the browser UI, loading only 6 comments per page click).
- **Read More**: Detailed guide in English at [README.md](v1_selenium_dom/README.md).

### 🟢 Phase 2: V2 - Hybrid API Injection (Recommended)
- **Directory**: [`v2_api_injection/`](v2_api_injection)
- **Concept**: A breakthrough optimization. It still connects to the debugged Chrome instance, but instead of clicking elements and parsing HTML, it **injects asynchronous JavaScript fetches** directly into the browser context. Because the requests are made *inside* the authenticated browser under the `shopee.vn` origin, they seamlessly inherit all valid cookies, security tokens, and API signatures.
- **Advantage**: Bypasses the 403 blocks effortlessly. It is 100x faster, retrieves up to 59 structured reviews in a single call, and is completely immune to CSS UI updates.
- **Read More**: Detailed guide in English at [README.md](v2_api_injection/README.md).

---

## 📊 Direct Performance & Stability Comparison

| Metric | 🔴 V1: Selenium DOM Scraping | 🟢 V2: Hybrid API Injection |
| :--- | :--- | :--- |
| **Data Fetch Mechanism** | Parsing UI HTML/DOM elements | Direct JSON fetch from Internal API |
| **Speed & Throughput** | ~2 reviews / sec (6 reviews per page click) | **~40+ reviews / sec** (Up to 59 reviews per API call) |
| **Data Depth & Insights** | **Basic** (Only visible front-end texts: user, overall star rating, comment text, variant name, and date). | **Extremely Rich** (Extracts deeper backend metadata invisible to the DOM, including granular sub-ratings for product/seller/delivery, quick feedback tags, review helpfulness like counts, attachment media counters, repeat purchase indicators, and unique IDs). |
| **Maintenance Level** | **High** (Breaks whenever CSS class names change) | **Very Low** (Backend API structures are highly stable) |
| **Anti-bot Resilience** | Medium (Frequent UI interactions might trigger bot verification) | **Extremely High** (Simulates pure backend network traffic from a real browser) |
| **Data Integrity** | Manual string cleaning, prone to selector errors | 100% clean, raw JSON output straight from Shopee's database |

---

## 📁 Repository Structure

```text
scrape_shopee/
├── README.md                      <-- Root documentation (this file)
│
├── 📁 v1_selenium_dom/            <-- Phase 1: DOM Scraping (Legacy)
│   ├── shopee_crawler_v1.py       <-- Original Selenium DOM script
│   └── README.md                  <-- Detailed V1 setup guide (English)
│
├── 📁 v2_api_injection/           <-- Phase 2: Hybrid API Injection (Recommended)
│   ├── shopee_crawler_v2_api.py   <-- High-performance API-injection script
│   ├── fetch_raw_sample.py        <-- Utility script to fetch raw JSON reviews
│   └── README.md                  <-- Detailed V2 setup & schema guide (English)
│
├── 📁 preprocess/                 <-- Phase 3: NLP Review Cleaning & Standardizing
│   ├── preprocess.py              <-- Python preprocessor class (Teencode + Spellcheck)
│   ├── preprocess.ipynb           <-- Jupyter Notebook demo of the pipeline
│   ├── teencode.json              <-- Static dictionary for slang translation
│   ├── vietnamese_dictionary.txt  <-- Standard Vietnamese dictionary (74K entries)
│   └── README.md                  <-- Deep architectural tradeoffs & benchmarks (English)
│
└── 📁 outputs/                    <-- Local outputs (Git-ignored directory)
```

---

## ⚡ Quick Start (Version 2)

To run the highly optimized Version 2 immediately:

1. **Launch Google Chrome in Debug Mode** from your command line:
   ```cmd
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\shopee_data"
   ```
2. **Install requirements**:
   ```bash
   pip install selenium pandas
   ```
3. **Configure the target product** in [`shopee_crawler_v2_api.py`](v2_api_injection/shopee_crawler_v2_api.py):
   ```python
   PRODUCT_URL = "https://shopee.vn/your-product-link"
   ```
4. **Run the script**:
   ```bash
   python v2_api_injection/shopee_crawler_v2_api.py
   ```
5. Check your directory for `shopee_reviews_api.csv` once complete!

---

## 🧠 What I Learned & Key Takeaways

1. **Working with anti-bot shields**: Classic headless browsers are easily spotted. Attaching to an active Chrome browser via CDP (Chrome DevTools Protocol) is an incredibly powerful way to bypass front-end telemetry checks.
2. **Shifting from UI to API**: Scraping the DOM is slow and highly fragile. Whenever possible, extracting data from internal APIs is superior.
3. **The Power of Origin Injection**: When APIs are protected by custom cryptographic request signatures, executing fetches directly inside the browser's context is an elegant, robust solution that eliminates the need for reverse-engineering complex obfuscated JavaScript signature generators.


## SentiLokal – Malaysian E-Commerce Review Sentiment Analytics

SentiLokal is a Streamlit dashboard for Malaysian e-commerce review sentiment analytics. One scrollable research page contains the Home introduction and five product cards, overview charts, product comparisons, customer review exploration, and final model performance. The **Application module** selector has exactly two choices: **Baseus Research Dashboard** (default) and **Upload & Analyze**. The research module retains its yellow/Poppins design and five radio section shortcuts. Research charts use rating-derived labels; the Model Performance section imports the actual eight Version B evaluation records in separate 80:20 and 70:30 groups. Source identifiers and cleaned text remain internal.

See [the dashboard guide](docs/dashboard.md) for setup, local product photographs, dataset provenance, navigation, and future model integration. Product photographs use a local validated catalog; verified photos are currently unavailable because official Baseus/Shopee access returned HTTP 403. The cards show an honest empty-photo state. From the repository root:

The research dashboard defaults to the approved `dataset_preprocessed_v3_clean.csv`. There is no customer uploader or dataset selector on that dashboard. The final 18,705-review Version B source is available through the explicit, checksum-verified migration in [deployment.md](docs/deployment.md#version-b-final-research-source). The original source and checksum are preserved.

**The current branch includes the cleaned CSV added by the project owner at `data/dataset_preprocessed_v3_clean.csv`.** A normal fresh clone therefore loads the approved data automatically. A deployment that omits data must provision that file or point `SENTILOKAL_RESEARCH_DATA_PATH` at an approved read-only server mount. The app opens it only for reading and verifies its SHA-256. Repository access must still protect any raw source metadata; the app exposes only its public field allowlist. See [exact Codespace and deployment setup](docs/deployment.md), including the checksum and read-only storage configuration. Missing data produces an explicit setup state, never fabricated results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_research_data.py
streamlit run app.py
```

With the branch dataset present (or after private owner provisioning), the check command should succeed and the research dashboard loads without a CSV upload. For a server mount, set the path before those commands:

```bash
export SENTILOKAL_RESEARCH_DATA_PATH=/srv/sentilokal/research/dataset_preprocessed_v3_clean.csv
```

To explore a different CSV, choose **Upload & Analyze**. Only original review text is required. Select a saved model and click **Analyze Reviews** to generate actual predictions, including for text-only CSVs; a single-review input is also available. Prediction labels and uncalibrated negative scores are separate from optional star-rating labels. Search, prediction filters, and CSV export retain original comments and duplicate rows. Uploads and their predictions stay in the individual session's memory and never replace research data. **Clear Uploaded Dataset** removes this analysis.

The actual supplied **TF-IDF + XGBoost** artifacts have been loaded and smoke-tested locally. The code-only kit's saved tokenizers and reference inference settings are verified and integrated. The other three architectures remain unavailable until the original fine-tuned weights and frozen encoder are provisioned. Their encoder tests use doubles, not real weights. See [model integration and the complete changed-file inventory](docs/model_integration.md). Model files, tokenizers, and new private Version B CSVs are ignored by Git and require separate provisioning on a fresh host.

```bash
python scripts/check_version_b.py --research --check-preprocessing --smoke-tfidf
# After the remaining transformer files are installed:
python scripts/check_version_b.py --require-all
```

Run dashboard checks with `python -m unittest discover -s tests -v`. This dashboard does not train models, balance classes, or change the existing preprocessing scripts.
