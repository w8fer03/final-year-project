# -*- coding: utf-8 -*-
import time, json, pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_driver_remote():
    chrome_options = Options()
    # Kết nối tới cổng 9222 mà bạn đã mở ở CMD
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def collect_reviews(driver, max_pages=3):
    all_reviews = []
    
    for p in range(max_pages):
        print(f"[*] Đang quét trang review {p + 1}...")
        time.sleep(3) # Đợi trang render đánh giá
        
        # Tìm các khối đánh giá
        review_items = driver.find_elements(By.CSS_SELECTOR, "div.q2b7Oq")
        
        for item in review_items:
            try:
                # Lấy tên, số sao, bình luận và thời gian
                author = item.find_element(By.CSS_SELECTOR, ".InK5kS").text
                content = item.find_element(By.CSS_SELECTOR, ".YNedDV").text
                time_and_variant = item.find_element(By.CSS_SELECTOR, ".XYk98l").text
                
                # Đếm số sao (dựa trên class active)
                stars = len(item.find_elements(By.CSS_SELECTOR, ".icon-rating-solid"))
                
                all_reviews.append({
                    "user": author,
                    "stars": stars,
                    "comment": content.replace("\n", " "),
                    "info": time_and_variant
                })
            except Exception as e:
                continue
        
        # Bấm nút sang trang tiếp theo (Pagination)
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "button.shopee-icon-button--right")
            driver.execute_script("arguments[0].click();", next_btn)
        except:
            print("[!] Đã hết trang hoặc không tìm thấy nút Next.")
            break
            
    return all_reviews

def main():
    # URL sản phẩm shopee cần crawl
    PRODUCT_URL = "https://shopee.vn/product/737801055/21742804545"
    
    driver = get_driver_remote()
    
    print(f"[*] Đang điều khiển trình duyệt tới: {PRODUCT_URL}")
    driver.get(PRODUCT_URL)
    
    # Cuộn xuống để Shopee load phần đánh giá
    print("[*] Đang cuộn xuống phần đánh giá...")
    driver.execute_script("window.scrollTo(0, 2500);")
    time.sleep(2)
    
    # Nhập max_pages = số lượng trang muốn crawl
    # 1 trang shopee sẽ có 6 comments.
    data = collect_reviews(driver, max_pages=3)
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv("shopee_reviews.csv", index=False, encoding="utf-8-sig")
        print(f"[✓] Đã lưu {len(data)} đánh giá vào file shopee_reviews.csv")
    else:
        print("[!] Không tìm thấy dữ liệu.")

if __name__ == "__main__":
    main()