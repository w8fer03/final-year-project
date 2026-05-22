# -*- coding: utf-8 -*-
import os
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
                val = item.get("model_name") or item.get("value") or item.get("options")
                if val:
                    if isinstance(val, list):
                        variants.extend([str(v) for v in val if v])
                    else:
                        variants.append(str(val))
            variant_str = ", ".join(variants)
            
            # Convert Unix timestamp to formatted date and time
            ctime = rate.get("ctime", 0)
            local_time = time.localtime(ctime) if ctime else None
            datetime_str = time.strftime('%Y-%m-%d %H:%M:%S', local_time) if local_time else ""
            
            # Extract detailed rating categories if available
            detailed = rate.get("detailed_rating") or {}
            
            # Extract template tags as a comma-separated string
            tags = rate.get("template_tags") or []
            tags_str = ", ".join(tags) if tags else ""
            
            all_reviews.append({
                "user": author,
                "userid": rate.get("userid"),
                "stars": stars,
                "product_quality": detailed.get("product_quality"),
                "seller_service": detailed.get("seller_service"),
                "delivery_service": detailed.get("delivery_service"),
                "comment": comment,
                "template_tags": tags_str,
                "datetime": datetime_str,
                "variant": variant_str if variant_str else "",
                "like_count": rate.get("like_count", 0),
                "image_count": len(rate.get("images") or []),
                "video_count": len(rate.get("videos") or []),
                "is_repeated": rate.get("is_repeated_purchase", False),
                "cmtid": rate.get("cmtid")
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
    PRODUCT_URL = "https://shopee.vn/%C4%90%C3%A8n-Treo-M%C3%A0n-H%C3%ACnh-C%E1%BA%A3m-%E1%BB%A8ng-B%E1%BA%A3o-V%E1%BB%87-M%E1%BA%AFt-%C4%90%C3%A8n-M%C3%A0n-H%C3%ACnh-PC-3-Ch%E1%BA%BF-%C4%90%E1%BB%99-S%C3%A1ng-C%E1%BA%A3m-%E1%BB%A8ng-%C4%90i%E1%BB%81u-Khi%E1%BB%83n-T%E1%BB%AB-Xa-i.1644821661.57200786209?extraParams=%7B%22display_model_id%22%3A350070001933%2C%22model_selection_logic%22%3A3%7D&sp_atk=55d84077-d99a-4456-a494-864c8ffaecb3&xptdk=55d84077-d99a-4456-a494-864c8ffaecb3"
    
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
        data = collect_reviews_api(driver, shop_id, item_id, total_to_crawl=50)
        
        if data:
            df = pd.DataFrame(data)
            
            # Get path of outputs/ directory relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            outputs_dir = os.path.join(os.path.dirname(current_dir), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            output_file = os.path.join(outputs_dir, "shopee_reviews_api.csv")
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"[+] Saved {len(data)} clean reviews to {output_file}")
        else:
            print("[!] No data found or crawl failed.")
            
    finally:
        # We do NOT quit the driver since it controls the user's main active browser!
        print("[*] Session complete. Keeping browser open.")

if __name__ == "__main__":
    main()
