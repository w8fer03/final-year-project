# GIẢI THÍCH CƠ CHẾ CRAWL RATING SHOPEE BẰNG SELENIUM + REMOTE DEBUGGING

## 1. Tổng quan mục tiêu
Mục tiêu của dự án là xây dựng một công cụ tự động thu thập dữ liệu đánh giá (ratings) từ các trang sản phẩm trên Shopee. Dữ liệu bao gồm:
-   Tên người dùng.
-   Số sao đánh giá.
-   Nội dung bình luận.
-   Thời gian bình luận.
-   Phân loại hàng hóa.
## 2. Tại sao không sử dụng các phương pháp phổ biến khác?
Trước khi đi đến giải pháp hiện tại, chúng ta xem xét và loại bỏ 3 phương pháp thông thường vì các rào cản kỹ thuật từ hệ thống bảo mật của Shopee:
-   **Cách 1: Sử dụng API chính thức của Shopee (`/api/v2/product/get_comment`)**
    -   _Vấn đề:_ Shopee yêu cầu tham số `access_token` hợp lệ. Tham số này chỉ cấp cho người bán (sellers) hoặc đối tác đã đăng ký chính thức qua Shopee Open Platform.
    -   _Kết luận:_ Không khả thi đối với việc crawl dữ liệu công khai từ góc độ người dùng thông thường.
-   **Cách 2: Khai thác API nội bộ của Shopee (Internal API)**
    -   _Vấn đề:_ Shopee kiểm tra IP, Cookie, User-Agent, User Session, Timestamp, Signature. Các header hợp lệ chỉ xuất phát từ trình duyệt thực tế.
    -   _Hệ quả:_ Nếu gọi trực tiếp từ script mà không có đầy đủ các thông số bảo mật này, hệ thống sẽ trả về lỗi **403 Forbidden** (Truy cập bị chặn).
-   **Cách 3: Sử dụng Selenium (mở browser tự động)**
    -   _Vấn đề:_ Shopee có hệ thống Anti-bot rất mạnh, dễ dàng phát hiện ra các thuộc tính đặc trưng của trình duyệt tự động và ngay lập tức chặn bằng Captcha.
    -   _Kết luận:_ Việc vượt qua Captcha liên tục của Shopee bằng bot là rất khó khăn và tốn kém.
## 3. Giải pháp tối ưu: Remote Debugging (Điều khiển trình duyệt thực)
Thay vì cố gắng "giả lập" một con người, chúng ta sử dụng trình duyệt thật đang chạy trên máy tính.
### Nguyên lý hoạt động:
1.  **Chế độ Debug:** Chúng ta mở trình duyệt Chrome thật bằng dòng lệnh với tham số `--remote-debugging-port=9222`. Lúc này, Chrome sẽ mở ra một "cổng giao tiếp" (Port 9222).
2.  **Chrome DevTools Protocol (CDP):** Đây là giao thức cho phép các phần mềm bên ngoài (như script Python) gửi lệnh điều khiển đến Chrome.
3.  **Vượt rào cản:** Vì đây là trình duyệt người dùng đang sử dụng, đã đăng nhập và có đầy đủ Cookie/History thật, Shopee sẽ nhận diện đây là hành vi của người dùng bình thường, từ đó không kích hoạt cơ chế Anti-bot.
## 4. Giải thích chi tiết cách hoạt động
### 1. Kết nối tới Chrome thật bằng DevTools Protocol

```
def get_driver_remote():
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

```
-   **Cơ chế:** Chrome ở chế độ debug đóng vai trò server CDP. Python Selenium là client kết nối qua WebSocket.
-   **Lợi ích:** Không tạo browser mới $\Rightarrow$ tránh bị Shopee phát hiện.
### 2. Điều khiển Chrome mở sản phẩm
```
driver.get(PRODUCT_URL)
driver.execute_script("window.scrollTo(0, 2500);")
```
-   **Giải thích:** Shopee sử dụng kỹ thuật **"Lazy Load"** (chỉ tải dữ liệu khi người dùng cuộn tới). Do đó, script phải thực hiện lệnh cuộn xuống phần đánh giá trước khi bắt đầu quét.
### 3. Thu thập review từ DOM đã render

```
review_items = driver.find_elements(By.CSS_SELECTOR, "div.q2b7Oq")

```

Chọn các selector dựa trên cấu trúc DOM của Shopee:
|Thông tin|Selector|Ý nghĩa|
|--|--|--|
|Tên người dùng| `.InK5kS` | Class chứa tên user |
|Nội dung bình luận|`.YNedDV`|Class chứa nội dung|
|Thời gian + phân loại|`.XYk98l`|Gộp thời gian + biến thể sản phẩm|
|Số sao|`.icon-rating-solid`|Mỗi icon solid = 1 sao|


### 4. Lặp từng trang đánh giá (Pagination)
```
next_btn = driver.find_element(By.CSS_SELECTOR, "button.shopee-icon-button--right")
driver.execute_script("arguments[0].click();", next_btn)
```
-   **Giải thích:** URL của trang không thay đổi khi chuyển trang. Nút “Next” chỉ xuất hiện trong DOM. Selenium phải bấm nút next để Shopee load thêm review.

### 5. Lưu dữ liệu ra CSV

```
df = pd.DataFrame(data)
df.to_csv("shopee_reviews.csv", index=False, encoding="utf-8-sig")
```
-   **Giải thích:** Pandas giúp xử lý và xuất dữ liệu sạch. `utf-8-sig` đảm bảo tiếng Việt hiển thị đúng trong Excel.

## 5. Kết luận
Phương pháp này là sự kết hợp hoàn hảo giữa tự động hóa và duy trì danh tính thực. Nó giải quyết triệt để vấn đề bị chặn bởi API hay Captcha, giúp việc lấy dữ liệu đánh giá sản phẩm trở nên ổn định và hiệu quả nhất trong bối cảnh các trang thương mại điện tử ngày càng thắt chặt an ninh.

### Tóm tắt toàn bộ quy trình hoạt động
1.  Người dùng mở Chrome ở chế độ debugging bằng cách nhập `"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\shopee_data"` trên cmd.
2.  Đăng nhập Shopee thủ công.
3.  Chạy Python script (Từ sau bước này Python sẽ tự động crawl, người dùng không cần làm gì cả).
4.  Selenium kết nối vào Chrome thật qua `127.0.0.1:9222`.
5.  Mở trang sản phẩm.
6.  Cuộn xuống và load review.
7.  Parse từng review từ DOM.
8.  Bấm nút **Next** để phân trang.
9.  Lưu dữ liệu ra file **CSV**.
