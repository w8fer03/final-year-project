# Shopee Reviews Crawling Mechanism: Selenium + Remote Debugging (V1)

This directory contains **Version 1** of the Shopee Product Reviews Scraper. It uses Selenium attached to a real, active Google Chrome browser in debugging mode to scrape review comments from the browser's rendered DOM tree.

---

## 1. Goal Overview

The objective of this module is to build an automated tool to harvest product reviews (ratings) from Shopee product pages. The gathered data includes:
- Buyer usernames
- Rating star count (1 to 5 stars)
- Review comment text
- Review timestamp
- Product variant details (purchased model/size)

---

## 2. Why Avoid Other Common Methods?

Before settling on the Remote Debugging solution, we evaluated and rejected three standard scraping approaches due to Shopee's strict anti-bot security parameters:

- **Method 1: Utilizing the official Shopee API (`/api/v2/product/get_comment`)**
  - *Problem:* Shopee requires a valid `access_token` for this endpoint. This token is only issued to registered sellers or official partners via the Shopee Open Platform.
  - *Conclusion:* Infeasible for gathering public data from an ordinary user's perspective.
- **Method 2: Directly calling Shopee's internal rating API**
  - *Problem:* Shopee validates request headers, cookies, User-Agent integrity, user sessions, timestamps, and custom cryptographic request signatures. These valid headers are only generated inside a real, active browser session.
  - *Outcome:* Directly calling this endpoint from Python's standard `requests` library without authenticating signatures results in immediate **403 Forbidden** blocks.
- **Method 3: Standard automated Selenium (new browser instance)**
  - *Problem:* Shopee features an advanced anti-bot telemetry engine that quickly identifies properties unique to automated test environments (e.g. `navigator.webdriver` set to true) and triggers insurmountable CAPTCHA challenges.
  - *Conclusion:* Overcoming continuous CAPTCHAs programmatically is slow, costly, and unreliable.

---

## 3. The Solution: Remote Debugging (Controlling a Real Browser)

Instead of attempting to spoof human behaviors on an automated browser, we run the automation script directly inside a **real, active Google Chrome instance** already running on the developer's computer.

### How it Works:
1. **Debugging Mode:** We launch a real Chrome session via the command line with the `--remote-debugging-port=9222` flag. This exposes a communication port (Port 9222).
2. **Chrome DevTools Protocol (CDP):** This protocol allows external programs (such as our Python Selenium script) to connect to and control the active Chrome session.
3. **Bypassing Protections:** Because this is a real user browser session containing actual cookies, local history, and browser configurations, Shopee detects the traffic as standard human browsing and does not activate its anti-bot challenges.

---

## 4. Technical Implementation Details

### 1. Connecting to the Active Chrome Browser via CDP

```python
def get_driver_remote():
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)
    return driver
```
- **Mechanism:** Chrome running in debug mode acts as a CDP server. Python's Selenium client connects directly to it via WebSockets.
- **Benefit:** It attaches to an existing browser rather than launching a new test instance, hiding automation indicators.

### 2. Navigating to the Product and Handling Lazy Load

```python
driver.get(PRODUCT_URL)
driver.execute_script("window.scrollTo(0, 2500);")
```
- **Explanation:** Shopee uses **Lazy Loading** (content only loads when scrolled into view). The script scrolls down to the reviews section to trigger API calls and render the reviews into the DOM.

### 3. Harvesting Reviews from the Rendered DOM

```python
review_items = driver.find_elements(By.CSS_SELECTOR, "div.q2b7Oq")
```
We select elements based on Shopee's front-end DOM structure:

| Data Field | Selector | Target Element |
| :--- | :--- | :--- |
| Username | `.InK5kS` | Class containing buyer username |
| Comment Content | `.YNedDV` | Class containing review text |
| Time & Variant | `.XYk98l` | Contains both timestamp and purchased option |
| Rating Stars | `.icon-rating-solid` | Counts solid star icons to determine rating (1-5) |

### 4. Handling Pagination

```python
next_btn = driver.find_element(By.CSS_SELECTOR, "button.shopee-icon-button--right")
driver.execute_script("arguments[0].click();", next_btn)
```
- **Explanation:** The page URL does not change during review pagination. The pagination buttons exist dynamically inside the DOM. Selenium must locate and click the "Next" button using JS execution to load the subsequent page of comments.

### 5. Exporting Data to CSV

```python
df = pd.DataFrame(data)
df.to_csv("shopee_reviews.csv", index=False, encoding="utf-8-sig")
```
- **Explanation:** Pandas structure makes export clean. Writing with the `utf-8-sig` encoding ensures Vietnamese diacritics are correctly displayed inside spreadsheet programs like Microsoft Excel.

---

## 5. Conclusion & Workflow Summary

This DOM-based Selenium method provides a robust way to establish automation while maintaining an authentic identity. It successfully bypasses anti-bot barriers and CAPTCHAs, rendering data collection stable.

### Workflow Summary
1. Launch Chrome in debugging mode via CMD:
   ```cmd
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\shopee_data"
   ```
2. Log in manually if required.
3. Execute the Python script:
   ```bash
   python v1_selenium_dom/shopee_crawler_v1.py
   ```
4. Selenium hooks into the active Chrome instance at `127.0.0.1:9222`.
5. Opens the target Shopee product page.
6. Scrolls down to lazy-load reviews.
7. Parses each review's data from the rendered HTML DOM.
8. Clicks **Next** to paginate through review screens.
9. Exports the gathered datasets into a structured **CSV** file.
