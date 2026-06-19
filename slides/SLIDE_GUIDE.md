# Hướng dẫn thuyết trình Slide - Pizza Delivery DSS

Tài liệu này cung cấp kịch bản nói và hướng dẫn trình bày chi tiết cho từng slide trong tệp tin Beamer [pizza_dss_slides.tex](file:///C:/Users/doand/Desktop/Workspace/1-Projects/ml-dss/pizza_delivery_dss/slides/pizza_dss_slides.tex).

---

## 1. Phân bổ thời gian (Tổng cộng 10-12 phút)

- **Mở đầu & Dữ liệu (Slide 1-5):** 2 phút.
- **EDA & Phân tích Nghiệp vụ (Slide 6-11):** 3 phút.
- **Mô hình học máy (Slide 12-16):** 3 phút.
- **Hệ hỗ trợ quyết định DSS & BI (Slide 17-21):** 2 phút.
- **Tái lập, Phản biện \& Giới hạn (Slide 22-27):** 2 phút.

---

## 2. Kịch bản nói chi tiết từng Slide

| Slide | Tiêu đề | Hình ảnh minh họa | Kịch bản nói chi tiết |
|---|---|---|---|
| **1** | Title Slide | Hình nền đỏ HUST | "Xin chào các thầy và các bạn. Hôm nay nhóm 29 xin phép trình bày đồ án học phần Hệ hỗ trợ quyết định với đề tài: Hệ hỗ trợ quyết định điều phối giao pizza dựa trên dự báo đơn giao trễ." |
| **2** | Mục lục / Agenda | Cột nội dung 4 phần | "Bài thuyết trình của nhóm gồm 4 phần chính: (I) Giới thiệu & Dữ liệu; (II) Phân tích EDA; (III) Mô hình hóa học máy; và (IV) Thiết kế hệ DSS cùng Dashboard." |
| **3** | 1. Bài toán \& Quyết định cần hỗ trợ | `pipeline_overview.png` | "Bài toán của chúng ta là điều phối giao hàng hiệu quả. Điều phối viên cần xác định sớm đơn nào có nguy cơ trễ trước khi giao để can thiệp. Hệ DSS của nhóm nhận thông tin đơn hàng lúc vừa đặt, dùng AI dự báo xác suất trễ, rồi dịch thành điểm rủi ro và gợi ý hành động cụ thể." |
| **4** | 2. Dữ liệu, Rò rỉ (Leakage) \& Ngưỡng mục tiêu | `delay_threshold_inference.png` | "Dữ liệu gồm 1.004 đơn hàng. Để chống rò rỉ dữ liệu (leakage), nhóm cấm tuyệt đối các biến sau giao hàng như duration hay delay. Tiếp theo, nhóm dùng thuật toán quét ngưỡng (Sweep Threshold) trên duration và tìm ra ranh giới nhãn trễ ngầm định là 30 phút." |
| **5** | 3. Khám định Dữ liệu (Synthetic Forensics) | `generator_deterministic_formula_errors.png` | "Đặc biệt, nhóm đã khám định dữ liệu (data forensics) và chứng minh bằng toán học rằng đây là dữ liệu sinh tự động. Nhóm tìm ra 7 công thức tất định sai số bằng 0 (như ước lượng duration bằng 2.4 lần khoảng cách). Vì vậy nhóm diễn giải kết quả rất thận trọng, tránh overclaim." |
| **6** | 4. Phân tích Khám phá (EDA) - Phân phối Nhãn | `delay_distribution.png` | "Dữ liệu bị mất cân bằng nhãn nhẹ với khoảng 21\% đơn bị trễ. Do đó, chúng ta không thể dùng metric Accuracy một mình vì nếu đoán 'Luôn đúng giờ' đã đạt 79\% nhưng không bắt được đơn trễ nào. Nhóm bắt buộc dùng F2-Score và MCC làm thước đo chính." |
| **7** | 5. EDA - Các yếu tố rủi ro vận hành chính | `delay_rate_by_traffic.png` | "Qua EDA, hai yếu tố dẫn dắt rủi ro trễ lớn nhất là Giao thông (kẹt xe làm trễ tăng vọt) và Khoảng cách di chuyển xa. Khi kết hợp kẹt xe với quãng đường dài, rủi ro giao trễ gần như là chắc chắn." |
| **8** | 6. EDA - Hành vi Khách hàng \& Sở thích | `size_preference.png` | "Khách hàng ưa chuộng nhất là pizza size Medium và loại pizza Non-Veg. Tuy nhiên, loại bánh Cheese Burst và các size XL/Large lại có tỷ lệ trễ bếp cao hơn bình thường, là thông tin quan trọng để điều phối chú ý cảnh báo." |
| **9** | 7. Dự báo Nhu cầu \& Lập kế hoạch nhân sự | `monthly_demand_forecast.png` | "Để minh họa quy trình hoạch định, nhóm dùng mô hình seasonal-naive dự báo nhu cầu đơn hàng 6 tháng tới. Do dữ liệu synthetic có MAPE cao (45.7\%), nhóm chỉ dùng làm demo. Từ phân phối giờ cao điểm lúc 19h, nhóm đề xuất kế hoạch phân bổ nhân sự tối ưu." |
| **10** | 8. Dự báo Xu hướng Sở thích | `preference_size_share_forecast.png` | "Nhóm cũng thực hiện dự báo xu hướng thị phần của các size bánh và loại pizza trong tương lai. Kết quả cho thấy xu hướng thị hiếu gần như đi ngang và đây là artifact của hàm sinh dữ liệu tự động." |
| **11** | 9. Kiến thức Môn học Áp dụng trong Dự án | Bảng đối chiếu | "Đồ án này bao phủ toàn diện các kiến thức môn học từ DSS, phân loại supervised, so sánh mô hình, đánh giá stability, phân cụm unsupervised K-Means, luật kết hợp Apriori cho đến tối ưu hóa vận tải." |
| **12** | 10. Feature Engineering: Compact vs Full Feature Set | `correlation_heatmap.png` | "Nhóm thiết kế hai tập đặc trưng: Full Set chứa 17 đặc trưng và Compact Set loại bỏ các biến trùng lặp thông tin (như estimated duration hay complexity). Thử nghiệm trên mô hình Logistic Regression cho thấy Compact Set đạt F2 dev tốt hơn (0.943 vs 0.928) và giảm thiểu overfit." |
| **13** | 11. So sánh Mô hình \& Tại sao loại Bayes? | `model_comparison.png` | "Nhóm so sánh 6 classifier trên tập Dev. Logistic Regression chiến thắng với F2 tốt nhất (0.9434). Trong khi đó, Gaussian Naive Bayes đạt điểm thấp kỷ lục (F2 chỉ 0.66) do vi phạm nặng nề giả định độc lập đặc trưng (các biến khoảng cách, traffic phụ thuộc chặt chẽ)." |
| **14** | 12. Siêu tham số \& Tối ưu Ngưỡng F-beta | `fbeta_threshold_curve.png` | "Nhóm tune siêu tham số C của mô hình LR trên Train bằng 5-Fold CV. Bản tuned C=0.3 cho kết quả dev kém hơn nên nhóm giữ default C=1.0. Quét ngưỡng F2 tối ưu trên Dev cho kết quả tốt nhưng khi chuyển sang Test làm tăng FP, nên nhóm giữ ngưỡng kỹ thuật 0.5." |
| **15** | 13. Kiểm thử Độ ổn định (Stability Audit) | `model_stability_f2_distribution.png` | "Để chứng minh kết quả không phải do ăn may một lần chia tập, nhóm chạy thử nghiệm stability chia lại ngẫu nhiên 100 lần trên pool train+dev. F2-score đạt trung bình 0.9419 với độ lệch chuẩn rất nhỏ, chứng tỏ mô hình có tính ổn định cực cao." |
| **16** | 14. Kết quả trên Test set \& So sánh Baseline | Bảng kết quả | "Khi đánh giá độc lập một lần trên tập Test khóa, mô hình đạt Accuracy 96.02\%, F2 đạt 0.9491, và Recall đạt 97.62\% (chỉ bỏ lọt duy nhất 1 đơn trễ). Kết quả này vượt trội hoàn toàn so với hai baseline đoán ngây thơ." |
| **17** | 15. Tầng Quyết định DSS: Xây dựng Risk Score | `risk_component_breakdown.png` | "Ở tầng hỗ trợ quyết định, nhóm xây dựng công thức điểm rủi ro Risk Score từ 0 đến 100. Điểm này kết hợp 55\% từ xác suất AI dự báo với 45\% áp lực vận hành (traffic, distance, complexity,...). Giúp người dùng hiểu rõ tại sao một đơn hàng bị rủi ro." |
| **18** | 16. DSS: Calibration \& Độ Nhạy Ngưỡng | `risk_calibration.png` | "Hiệu chuẩn cho thấy tỷ lệ trễ thực tế tăng vọt từ nhóm Low (0\%) lên Medium (7.3\%) và High (88.6\%). Nhóm đặt ngưỡng phân luồng ưu tiên 35/65, bắt được 92.9\% tổng số đơn trễ rơi vào nhóm High Priority." |
| **19** | 17. DSS: Hàng đợi Ưu tiên \& Khuyến nghị Hành động | Bảng khuyến nghị hành động | "Tương ứng với mỗi phân luồng ưu tiên, hệ thống đưa ra khuyến nghị hành động nghiệp vụ cụ thể cho điều phối viên: Nhóm High cần gán tài xế ưu tiên và chuẩn bị tài xế dự phòng; nhóm Medium cần theo dõi sát; nhóm Low xử lý thường." |
| **20** | 18. Tối ưu Vận tải (Hungarian Assignment Scenario) | `transport_assignment_cost.png` | "Để minh họa tầng prescriptive DSS, nhóm lấy các đơn hàng High Priority và gán tối ưu cho đội tài xế giả lập. Sử dụng thuật toán Hungarian để tối thiểu hóa ma trận chi phí (phạt trễ, thưởng cùng tuyến), đạt chi phí trung bình tối ưu là 32.84." |
| **21** | 19. Ứng dụng Thực tế: Streamlit Dashboard - Tổng quan | `dashboard_overview.jpg` | "Nhóm xây dựng ứng dụng web Streamlit Dashboard hoàn chỉnh gồm 8 tab. Tab Overview hiển thị nhanh KPI vận hành thời gian thực và biểu đồ, tab EDA cho phép lọc nhanh tỷ lệ trễ động theo nhu cầu người dùng." |
| **22** | 20. Dashboard: Ra Quyết định \& Forensics | `dashboard_single_order.jpg` | "Tab Single Order Demo cho phép điều phối viên nhập thông tin 1 đơn hàng và nhận ngay điểm rủi ro bóc tách chi tiết. Tab Delay Queue hiển thị hàng đợi ưu tiên và gợi ý gán tài xế tối ưu trực tiếp." |
| **23** | 21. Minh chứng: Power BI Data Pack \& Khả năng tái lập | Khối mã lệnh | "Dự án cung cấp gói dữ liệu sạch Power BI (Fact/Dim CSV) kèm hướng dẫn dựng chi tiết. Đặc biệt, toàn bộ dự án có khả năng tái lập 100\% chỉ bằng 1 lệnh chạy duy nhất `python -m scripts.run_all` vượt qua 30 unit tests." |
| **24** | 22. Các câu hỏi phản biện \& Giải pháp đối phó | Danh sách hỏi - đáp | "Nhóm đã chuẩn bị sẵn câu trả lời tự vệ cho các câu hỏi hội đồng hay bắt bẻ như: Vì sao không dùng duration làm feature (leakage); Vì sao biết ngưỡng trễ là 30 phút (sweep threshold); Tại sao không dùng Bayes (giả định độc lập bị vi phạm)." |
| **25** | 23. Giới hạn \& Caveats cần lưu ý | Danh sách caveats | "Nhóm thẳng thắn nêu rõ giới hạn: dữ liệu Kaggle là synthetic nên điểm mô hình cao là do cấu trúc dữ liệu sinh, các phần dự báo nhu cầu hay chất lượng thương hiệu chỉ mang tính chất minh họa phương pháp luận." |
| **26** | 24. Kết luận \& Hướng phát triển | Cột kết luận | "Đồ án đã xây dựng thành công một chu trình DSS khép kín nghiêm ngặt và minh bạch. Hướng phát triển tiếp theo là thu thập dữ liệu GPS tài xế thật, tích hợp chi phí thực tế và kiểm thử hệ thống với người vận hành thực tế." |
| **27** | Cám ơn / Q\&A | Hình end-empty HUST | "Nhóm xin chân thành cảm ơn các thầy cô giáo đã lắng nghe. Nhóm sẵn sàng nhận câu hỏi phản biện từ phía hội đồng." |

---

## 3. Quy tắc khi chỉnh sửa Slide Beamer

- Biên dịch lại slide PDF sau khi thay đổi mã nguồn bằng lệnh:
  ```powershell
  python -m scripts.build_slides_pdf
  ```
- File PDF kết quả thuyết trình chính thức sẽ được ghi đè tại: `slides/PIZZA_DSS_SLIDE_DECK.pdf`.
