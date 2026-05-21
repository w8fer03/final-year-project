# Shopee Reviews Scraper: A Technical Evolution Case Study

A collection of python-based crawlers designed to harvest product reviews (usernames, star ratings, comment texts, product variants, and timestamps) from **Shopee**. 

Rather than just presenting a final product, this repository is organized as a **case study** to showcase the technical evolution from a standard UI-based web scraper to a highly optimized, signature-bypassing hybrid API injector.

---

## 📈 The Technical Evolution

Crawling Shopee reviews is notoriously challenging due to their advanced Anti-bot mechanisms (which block standard automated browsers with Captchas) and secure Internal APIs (which return `403 Forbidden` if signatures, cookies, or timestamps are invalid). 

Below is the journey of how those hurdles were systematically resolved:

### 🔴 Phase 1: V1 - Selenium DOM Scraping
- **Directory**: [`v1_selenium_dom/`](file:///c:/Users/Ku%20Bach/Desktop/Code/scrape_shopee/v1_selenium_dom)
- **Concept**: Bypasses the initial anti-bot system by attaching Selenium to a **real, active Google Chrome browser** in debugging mode (`--remote-debugging-port=9222`). Once attached, it scrolls the page to trigger lazy loading and scrapes reviews directly from the HTML DOM tree using CSS selectors.
- **Limitation**: Brittle (breaks when Shopee modifies class names) and slow (must paginate manually in the browser UI, loading only 6 comments per page click).
- **Read More**: Detailed guide in Vietnamese at [README_v1.md](file:///c:/Users/Ku%20Bach/Desktop/Code/scrape_shopee/v1_selenium_dom/README_v1.md).

### 🟢 Phase 2: V2 - Hybrid API Injection (Recommended)
- **Directory**: [`v2_api_injection/`](file:///c:/Users/Ku%20Bach/Desktop/Code/scrape_shopee/v2_api_injection)
- **Concept**: A breakthrough optimization. It still connects to the debugged Chrome instance, but instead of clicking elements and parsing HTML, it **injects asynchronous JavaScript fetches** directly into the browser context. Because the requests are made *inside* the authenticated browser under the `shopee.vn` origin, they seamlessly inherit all valid cookies, security tokens, and API signatures.
- **Advantage**: Bypasses the 403 blocks effortlessly. It is 100x faster, retrieves up to 59 structured reviews in a single call, and is completely immune to CSS UI updates.
- **Read More**: Detailed guide in English at [README_v2.md](file:///c:/Users/Ku%20Bach/Desktop/Code/scrape_shopee/v2_api_injection/README_v2.md).

---

## 📊 Direct Performance & Stability Comparison

| Metric | 🔴 V1: Selenium DOM Scraping | 🟢 V2: Hybrid API Injection |
| :--- | :--- | :--- |
| **Data Fetch Mechanism** | Parsing UI HTML/DOM elements | Direct JSON fetch from Internal API |
| **Speed & Throughput** | ~2 reviews / sec (6 reviews per page click) | **~40+ reviews / sec** (Up to 59 reviews per API call) |
| **Maintenance Level** | **High** (Breaks whenever CSS class names change) | **Very Low** (Backend API structures are highly stable) |
| **Anti-bot Resilience** | Medium (Frequent UI interactions might trigger bot verification) | **Extremely High** (Simulates pure backend network traffic from a real browser) |
| **Data Integrity** | Manual string cleaning, prone to selector errors | 100% clean, raw JSON output straight from Shopee's database |

---

## 📁 Repository Structure

```text
scrape_shopee/
├── README.md                      <-- Root documentation (this file)
│
├── 📁 v1_selenium_dom/            <-- Version 1 (DOM Scraping)
│   ├── shopee_crawler_v1.py       <-- Original Selenium DOM script
│   └── README_v1.md               <-- Detailed V1 setup guide (Vietnamese)
│
└── 📁 v2_api_injection/           <-- Version 2 (Optimized API Injection)
    ├── shopee_crawler_v2_api.py   <-- High-performance JS-injection script
    └── README_v2.md               <-- Technical walkthrough of V2 (English)
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
3. **Configure the target product** in [`shopee_crawler_v2_api.py`](file:///c:/Users/Ku%20Bach/Desktop/Code/scrape_shopee/v2_api_injection/shopee_crawler_v2_api.py):
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
