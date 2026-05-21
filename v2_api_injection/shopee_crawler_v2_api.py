# -*- coding: utf-8 -*-
import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_driver_remote():
    chrome_options = Options()
    # Connect to port 9222 opened in CMD
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)
    # Set timeout for async script execution
    driver.set_script_timeout(30)
    return driver

def extract_ids_from_url(url):
    """
    Extracts shop_id and item_id from Shopee product URL.
    Supports both popular URL formats:
    1. https://shopee.vn/product/shopid/itemid
    2. https://shopee.vn/Product-Name-i.shopid.itemid
    """
    # Format 1: /product/shop_id/item_id
    match1 = re.search(r'product/(\d+)/(\d+)', url)
    if match1:
        return match1.group(1), match1.group(2)
        
    # Format 2: -i.shop_id.item_id
    match2 = re.search(r'-i\.(\d+)\.(\d+)', url)
    if match2:
        return match2.group(1), match2.group(2)
        
    return None, None

def collect_reviews_api(driver, shop_id, item_id, total_to_crawl=100):
    all_reviews = []
    limit = 59  # Max reviews per Shopee API call
    offset = 0
    
    # JavaScript code to fetch data from the internal API
    # Using arguments to pass parameters from Python dynamically
    js_fetch_script = """
    const callback = arguments[arguments.length - 1];
    const limit = arguments[0];
    const offset = arguments[1];
    const item_id = arguments[2];
    const shop_id = arguments[3];
    
    const url = `https://shopee.vn/api/v2/item/get_ratings?filter=0&limit=${limit}&offset=${offset}&itemid=${item_id}&shopid=${shop_id}&type=0`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => callback({ success: true, data: data }))
        .catch(err => callback({ success: false, error: err.message }));
    """
    
    print(f"[*] Starting API crawl for up to {total_to_crawl} reviews...")
    
    while len(all_reviews) < total_to_crawl:
        print(f"[*] Fetching data from offset {offset}...")
        
        # Execute asynchronous JS script inside the browser context
        result = driver.execute_async_script(js_fetch_script, limit, offset, item_id, shop_id)
        
        if not result or not result.get("success"):
            error_msg = result.get("error") if result else "No response from browser"
            print(f"[!] Error calling API via browser: {error_msg}")
            break
            
        data = result.get("data", {})
        ratings = data.get("data", {}).get("ratings")
        
        if not ratings:
            print("[*] No more reviews found.")
            break
            
        for rate in ratings:
            author = rate.get("author_username", "An danh")
            stars = rate.get("rating_star", 0)
            comment = (rate.get("comment") or "").replace("\n", " ")
            
            # Extract product variant choices if any
            variants = []
            product_items = rate.get("product_items", [])
            for item in product_items:
                val = item.get("value")
                if val:
                    variants.append(val)
            variant_str = ", ".join(variants)
            
            # Convert Unix timestamp to formatted string
            ctime = rate.get("ctime", 0)
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctime))
            
            info_str = f"{formatted_time}"
            if variant_str:
                info_str += f" | Phan loai hang: {variant_str}"
                
            all_reviews.append({
                "user": author,
                "stars": stars,
                "comment": comment,
                "info": info_str
            })
            
        print(f"[+] Successfully fetched {len(all_reviews)} reviews.")
        
        # If returned reviews are less than the limit, it means we reached the end
        if len(ratings) < limit:
            print("[*] Received fewer reviews than limit. Stopping.")
            break
            
        offset += limit
        time.sleep(1.5) # Soft delay between requests to avoid rate limits
        
    return all_reviews[:total_to_crawl]

def main():
    PRODUCT_URL = "https://shopee.vn/product/737801055/21742804545"
    
    # 1. Extract IDs
    shop_id, item_id = extract_ids_from_url(PRODUCT_URL)
    if not shop_id or not item_id:
        print("[!] Could not extract Shop ID or Item ID from product URL!")
        return
        
    print(f"[+] Successfully extracted - Shop ID: {shop_id} | Item ID: {item_id}")
    
    # 2. Connect to Remote Debugging Chrome
    try:
        driver = get_driver_remote()
    except Exception as e:
        print("[!] Could not connect to Chrome Remote Debugging on port 9222!")
        print("[!] Please make sure Chrome is opened in debug mode on port 9222 and logged in.")
        print(f"Error details: {e}")
        return
        
    try:
        # Check if browser is currently on Shopee to avoid Same-Origin Policy errors
        current_url = driver.current_url
        if "shopee.vn" not in current_url:
            print("[*] Browser is not on Shopee. Navigating to Shopee to set origin...")
            driver.get("https://shopee.vn")
            time.sleep(2)
            
        # 3. Collect reviews
        # Crawling 150 reviews (takes ~3 seconds, equivalent to 25 page loads in standard Selenium)
        data = collect_reviews_api(driver, shop_id, item_id, total_to_crawl=500)
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv("shopee_reviews_api.csv", index=False, encoding="utf-8-sig")
            print(f"[+] Saved {len(data)} clean reviews to shopee_reviews_api.csv")
        else:
            print("[!] No data found or crawl failed.")
            
    finally:
        # We do NOT quit the driver since it controls the user's main active browser!
        print("[*] Session complete. Keeping browser open.")

if __name__ == "__main__":
    main()
