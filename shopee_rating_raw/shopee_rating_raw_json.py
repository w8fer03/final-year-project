import os
import time
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_driver_remote():
    chrome_options = Options()
    # Kết nối vào cổng 9222 của trình duyệt đang mở
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_script_timeout(30)
    return driver

def extract_ids_from_url(url):
    match1 = re.search(r'product/(\d+)/(\d+)', url)
    if match1:
        return match1.group(1), match1.group(2)
        
    match2 = re.search(r'-i\.(\d+)\.(\d+)', url)
    if match2:
        return match2.group(1), match2.group(2)
        
    return None, None

def main():
    PRODUCT_URL = "https://shopee.vn/Set-10-20-Chu%C3%B4ng-Tr%C3%B2n-Nh%E1%BB%8F-M%C3%A0u-K%E1%BA%B9o-14-*-17mm-L%C3%A0m-M%C3%B3c-Kh%C3%B3a-%C4%90i%E1%BB%87n-Tho%E1%BA%A1i-diy-i.379903869.21181281963?extraParams=%7B%22display_model_id%22%3A214393124876%7D"
    
    shop_id, item_id = extract_ids_from_url(PRODUCT_URL)
    if not shop_id or not item_id:
        print("[!] Could not extract Shop ID or Item ID from product URL!")
        return
        
    print(f"[+] Extracted - Shop ID: {shop_id} | Item ID: {item_id}")
    
    try:
        driver = get_driver_remote()
    except Exception as e:
        print(f"[!] Could not connect to Chrome Remote Debugging: {e}")
        return
        
    try:
        # Đảm bảo trình duyệt đang ở Shopee
        current_url = driver.current_url
        if "shopee.vn" not in current_url:
            print("[*] Browser is not on Shopee. Navigating...")
            driver.get("https://shopee.vn")
            time.sleep(2)
            
        # Chúng ta chỉ fetch một vài bản ghi (ví dụ limit=5) để cấu trúc JSON không bị quá dài
        limit = 2
        offset = 0
        
        js_fetch_script = """
        const callback = arguments[arguments.length - 1];
        const limit = arguments[0];
        const offset = arguments[1];
        const item_id = arguments[2];
        const shop_id = arguments[3];
        
        const url = `https://shopee.vn/api/v2/item/get_ratings?filter=0&limit=${limit}&offset=${offset}&itemid=${item_id}&shopid=${shop_id}&type=0`;
        
        fetch(url)
            .then(response => response.json())
            .then(data => callback({ success: true, data: data }))
            .catch(err => callback({ success: false, error: err.message }));
        """
        
        print("[*] Fetching raw JSON from Shopee API...")
        result = driver.execute_async_script(js_fetch_script, limit, offset, item_id, shop_id)
        
        if result and result.get("success"):
            raw_data = result.get("data")
            
            # Get path of outputs/ directory relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            outputs_dir = os.path.join(os.path.dirname(current_dir), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            output_file = os.path.join(outputs_dir, "shopee_reviews_raw.json")
            # Lưu dữ liệu gốc chưa chỉnh sửa dưới dạng JSON đẹp
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=4, ensure_ascii=False)
                
            print(f"[+] Successfully saved raw, unmodified JSON to: {output_file}")
        else:
            error_msg = result.get("error") if result else "No response"
            print(f"[!] Failed to fetch raw API data: {error_msg}")
            
    finally:
        print("[*] Test session complete.")

if __name__ == "__main__":
    main()
