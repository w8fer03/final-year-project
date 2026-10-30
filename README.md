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

SentiLokal is a Streamlit dashboard for Malaysian e-commerce review sentiment analytics. One scrollable research page contains the Home introduction and five product cards, overview charts, product comparisons, customer review exploration, and future model performance. The sidebar radio list jumps to these sections; **Upload & Analyze** opens a separate page for user data. Current insights use star-rating-derived sentiment; model-evaluation states remain empty until valid held-out results are supplied. Source listing and review IDs and existing cleaned text stay internal and are excluded from customer screens and CSV downloads.

See [the dashboard guide](docs/dashboard.md) for setup, local product photographs, dataset provenance, navigation, and future model integration. Product photographs use a local validated catalog; verified photos are currently unavailable because official Baseus/Shopee access returned HTTP 403. The cards show an honest empty-photo state. From the repository root:

The research dashboard automatically loads **only the approved `dataset_preprocessed_v3_clean.csv`**. There is no uploader or dataset selector on that dashboard. All statistics are calculated from this file, with the existing five-product grouping and observed review dates.

**A new clone or deployment has no private CSV.** Provision your original approved file as `data/dataset_preprocessed_v3_clean.csv`, or point `SENTILOKAL_RESEARCH_DATA_PATH` at an approved read-only server file. The loader verifies its SHA-256 before using it. See [exact Codespace and deployment setup](docs/deployment.md), including the checksum and read-only storage configuration. Missing data produces an explicit setup state, never fabricated results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_research_data.py
streamlit run app.py
```

After owner provisioning, the check command should succeed and the research dashboard loads without a CSV upload. For a server mount, set the path before those commands:

```bash
export SENTILOKAL_RESEARCH_DATA_PATH=/srv/sentilokal/research/dataset_preprocessed_v3_clean.csv
```

To explore a different CSV, choose **Upload & Analyze** in the sidebar. Only a review-text column is required; ratings, dates, and products are optional. The page provides validation, available charts, product/sentiment filters, review search, and filtered CSV download. Uploads stay in the individual Streamlit session's server memory, are not written to disk or cached across users, and never replace the research dataset. **Clear Uploaded Dataset** removes the uploaded analysis and resets its file selector. Text-only uploads receive no sentiment classifications; automated prediction awaits the genuine trained model.

Run dashboard checks with `python -m unittest discover -s tests -v`. This dashboard does not train models, balance classes, or change the existing preprocessing scripts.
