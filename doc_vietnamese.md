# Tìm hiểu Hệ thống Định Lượng Bằng Trực Giác 🇻🇳

Tài liệu này sẽ giải thích **Tư duy Kiến trúc bằng những ví dụ dân dã nhất**.

---

## Phần 1: Khám Phá Feature Engineering (14 Chỉ số Bắt Mạch)
Thay vì cho AI đọc tin tức, chúng ta ép AI nhìn công ty qua 14 chỉ số kinh điển (Các chỉ số được định nghĩa trong mảng `features`). Chúng chia làm 4 Nhóm để AI đoán xem 3 tháng nữa công ty sẽ Vút bay hay Đổ sập:

### Nhóm 1: Sức Khỏe Sinh Lời (Làm Ra Tiền Không?)
- **EPS (Lợi Nhuận Trên Mỗi Cổ Phiếu)**: Năm nay ông chủ Tỏi kiếm được 10 tỷ, chia cho 10 triệu anh em cô bác đang hùn vốn. 1 cổ phiếu sinh ra bao nhiêu tiền thật?
- **ROE (Tỷ Suất Sinh Lời Ròng Trên Vốn Chủ Sở Hữu)**: Góp cho công ty 100 đồng tiền thịt, 1 năm sau công ty trả lại cho cổ đông bao nhiêu đồng lãi? Khách sạn Vinpearl (VPL) hồi 2008 ROE bị nhiễu do mới xây xong chôn vốn.
- **net_income_ratio (Biên Lợi Nhuận Ròng)**: Bán được 100 đồng doanh thu, trừ sạch tiền thuê bò, điện nước, thuế... bỏ túi được mấy đồng lãi?

### Nhóm 2: Định Giá Đắt/Rẻ (Ông Đang Bán Hớ Hay Bán Đắt?)
- **PE (Price to Earning)**: Công ty kiếm được 1 ngàn đồng/năm. Thị trường đang quát giá Cổ phiếu mua đứt nó là 20 ngàn. Tức là PE = 20 (Phải ôm 20 năm mới vãn hồi vốn). Càng cao chứng tỏ càng bị đu đỉnh (hoặc công ty có kỳ vọng cực lớn).
- **PB (Price to Book)**: Công ty có cái nhà, cái xưởng đem bán ve chai được 1 tỷ (Book Value). Nhưng thị trường đang rao giá 2 tỷ. PB = 2. 

### Nhóm 3: Thanh Khoản (Có Sắp Phá Sản Vì Thiếu Tiền Mặt?)
- **cur_ratio (Tỷ lệ thanh toán Hiện hành)** & **quick_ratio (Thanh toán Nhanh)** & **cash_ratio (Tiền mặt)**: Cả 3 chỉ số này xoay quanh 1 việc: Nếu Sáng Mai 100 ông giang hồ kéo đến siết nợ Ngắn Hạn, công ty có đủ bao nhiêu Tài Sản / Tiền Mặt ngay trong két sắt để vứt ra trả ngay không? Nếu `cash_ratio` quá thấp, công ty cực dễ bị chết sốc.
- **acc_rec_turnover (Vòng Quay Khoản Phải Thu)**: Dân gian gọi là **Trình Đi Đòi Nợ**. Chỉ số cao tức là công ty bán chịu hàng xong, mấy ngày sau đã léo nhéo đòi được tiền về két sắt ngay. Thấp tức là tiền bị khách giam, dòng máu bị tắc!

### Nhóm 4: Gánh Nặng Nợ Nần (Có Bị Xiết Vốn?)
- **debt_ratio (Tỷ Lệ Nợ/Tài Sản)** & **debt_to_equity (Nợ/Vốn Chủ)**: Đi kinh doanh nhưng mở xưởng mất 10 đồng, vay giang hồ tận 8 đồng. Dính mùa Covid không trả nổi lãi là đi đứt!

👉 **Cách AI Học:** Nó sẽ ngấu nghiến sự phối hợp cả 14 con số này từ 30 Công ty trong VN30 suốt 3 năm ròng rã, để tìm đúc ra 1 quy luật: *"Công ty nào PE < 15, Vòng Đòi Nợ Nhanh, Tiền Mặt Cao -> Dự báo y_return tăng trưởng giá tốt"*.

---

## Phần 2: Quyết định Xử lý Hệ thống (Domain Decisions)

### Quyết định 1: Tại sao phải tính `Log Return` thay vì Phần Trăm (%) Bình thường?
**Câu chuyện:** Nếu bạn có 100 triệu USD. Hôm nay thị trường Sập **-50%**, bạn còn 50 Triệu. Để gỡ lại mức bờ 100 Triệu, ngày mai thị trường phải Bay lên tận **+100%**. (Số -50 và +100 hoàn toàn không đối xứng nhau bằng tỷ lệ cấp số cộng, nên nếu đút vào máy học AI, thuật toán Deep Learning sẽ bị hoang tưởng về phương sai).
**Quyết định:** Dùng `% Logarit`. Trong miền Log, giảm 50% là `-0.693`, và tăng gấp đôi là `+0.693`. Nó xoá bỏ mọi sự dối trá của Lãi kép, giúp Lợi nhuận mang tính "Cộng dồn" (Additive) hoàn hảo!

### Quyết định 2: Tội Ác Nhìn Lén Tương Lai (Look-ahead Bias / Data Leakage)
**Tội ác:** Trong Data Science thông thường (như bài học vạch đường để xe tự lái), ta thường đổ hết dữ liệu trộn lộn xộn ngẫu nhiên (Shuffle = True), sau đó bốc 80% học, 20% Train.
Nếu làm vậy với Chứng khoán: Tức là bạn cho con Robot AI học trước "Báo cáo Tài chính Của Năm 2024", rồi ném bài kiểm tra "Chứng khoán năm 2022" ra bắt nó thi! Nó là Cỗ Máy Thời Gian! Mọi điểm số nó thi đậu đều là đồ giả khi cầm ra đánh tiền thật ngoài đời.
**Cách Trị Tội (Tính năng `StandardScaler`):** Hàm tỷ lệ hóa (Scaler) trong khung Code của chúng ta chỉ được gọi lệnh `.fit()` trên dữ liệu của Tập Huấn Luyện Quá khứ. Nó bị ép buộc không bao giờ được phép dùng kính hiển vi soi Đáy/Đỉnh của Tập Test (Bài thi tương lai).

### Quyết định 3: Walk-Forward Validation (Cỗ Máy Chạy Thi Giám Thị)
Thay vì chẻ một phát Train 80% / Test 20%. Bạn quyết định thiết kế phương pháp **Walk-Forward Cuốn Chiếu**.
**Ẩn dụ:** Tưởng tượng ông Thầy giáo dạy Cổ phiếu. Ông có trong tay Đề thi Đại học từ 2011 đến 2024.
- Bước 1: Ông đưa Tài liệu Lịch sử từ 2011-2015 cho 4 học sinh (4 Model AI). Bắt chúng nó cày bừa. Sau đó Tịch thu tài liệu. Đưa đề năm 2016 ra bắt làm Tươi. Ghi điểm vào sổ (File Predictions). Xóa Trí Nhớ Bọn Học sinh.
- Bước 2: Ông lại gom Học sinh mới, đưa lại tài liệu từ 2012-2016. Tịch thu. Bắt thi làm bài năm 2017. Ghi điểm vào số. Xóa trí nhớ.
👉 Quy trình cực kì tàn khốc và khép kín này khiến con AI vĩnh viễn không bao giờ được học lật trước Đề thi của tương lai, một chiến trường sát phạt 100% minh bạch!

### Quyết định 4: Bức tường Than Ngăn Cắt Train / Trade
Thay vì trộn code mua bán trực tiếp vào file AI. Tôi đã xây dựng Lớp vỏ `backtest_engine.py` nhận vào Ma Trận Trọng Số.
- File học Máy (ML) tuyệt đối không biết "Lãi/Lỗ Tiền Thật". Nó chỉ ra Cáo thị: Khay A chứa FPT đang ngon, cho 20% Lúa.
- Sàn đấu `bt` là kẻ thù độc lập. Nó nhận Cáo thị đó, xách tiền ra Sàn giao dịch thực chiến. Trừ luôn cả Phí Nộp Sàn (Hoa hồng cho Broker), chịu những cú Trượt Gía (Slippage) nếu Mua Đuổi không khớp được lệnh.
Kết quả `CAGR` và `Max Drawdown` đẻ ra từ file `backtest_engine` là Lợi Nhuận Thức Tỉnh (Realized Equity), là tiền có thể mang ra ngân hàng tiêu xài chứ không phải là Giấc mơ sai số MSE nữa!
