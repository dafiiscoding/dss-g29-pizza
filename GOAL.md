# GOAL — Pizza Delivery DSS (Đồ án môn Hệ hỗ trợ quyết định)

> File mục tiêu **chuẩn, duy nhất** của dự án. Cả nhóm đọc file này đầu tiên.
> Đây là **đồ án môn học** (môn Hệ hỗ trợ quyết định — TS. Lê Hải Hà; TS. Trịnh
> Đình Thăng), **không phải sản phẩm thật**. Mục tiêu là làm đúng và đủ quy trình
> môn học, lấy điểm, chứ không phải triển khai ngoài đời.
> Cập nhật: 19/06/2026.

---

## 0. Một câu để cả nhóm nhớ

Xây một **Hệ hỗ trợ quyết định (DSS)** dự báo **đơn pizza có nguy cơ giao trễ**,
rồi biến dự báo đó thành **Risk Score → Priority → Hành động → Hàng đợi ưu tiên**,
và trình bày đầy đủ theo chuẩn môn học (report + slide + dashboard + notebook).

---

## 1. Trọng tâm: cái gì ăn điểm chính, cái gì là phụ

Dataset là dữ liệu Kaggle **gần như synthetic/“rác”** (forensics chứng minh nhiều
cột là công thức tất định). Vì vậy nhóm **không dồn sức** vào các phân tích dễ bị
data rác làm sai lệch. Phân chia rõ:

**🎯 TRỌNG TÂM (chấm điểm chính — làm kỹ, làm sâu):**
1. **Dự báo đơn trễ** (`is_delayed`): chống leakage, so 6 classifier, chọn theo
   F2, báo test + baseline. → đây là phần "ăn điểm" chính của đồ án.
2. **Tầng quyết định DSS**: Risk Score 0–100 (có bóc tách thành phần), Priority
   (Low/Medium/High), Recommended Action, Delay Queue.
3. **Trung thực dữ liệu (forensics)**: chứng minh data là synthetic → giải thích
   vì sao điểm mô hình cao mà không overclaim. Đây là điểm cộng học thuật, đi
   kèm trọng tâm.

**➕ PHỤ (chỉ cần ĐỦ để phủ yêu cầu môn — trình bày gọn, ghi rõ là demo):**
- Clustering (K-Means), Association rules, Hypothesis testing.
- Customer behavior, Forecast demand/staffing, Preference trend, Recommendation.
- Brand/location/state analysis.
- Tối ưu vận tải (assignment tài xế giả lập).

> ⚠️ Quy ước vàng: **mọi phần PHỤ phải kèm caveat “dữ liệu synthetic, chỉ minh hoạ
> quy trình, không kết luận thị trường/sản xuất thật.”** Không phần phụ nào được
> trình bày như thành tựu.

---

## 2. Kiến thức môn học thể hiện trong dự án (để đối chiếu yêu cầu môn)

| Kiến thức môn | Thể hiện ở đâu | Mức đầu tư |
|---|---|---|
| DSS & ra quyết định | Risk Score → Priority → Action → Queue | 🎯 Trọng tâm |
| Supervised learning | Dự báo `is_delayed` | 🎯 Trọng tâm |
| Decision Tree / Naive Bayes / k-NN / SVM / Ensemble | So sánh 6 classifier | 🎯 Trọng tâm |
| Đánh giá mô hình | Train/dev/test, baseline, Accuracy/F1/F2/MCC, CI | 🎯 Trọng tâm |
| Feature engineering & chống leakage | Compact feature, cấm cột hậu nghiệm | 🎯 Trọng tâm |
| Weighted scoring model | Risk Score = cộng có trọng số các component | 🎯 Trọng tâm |
| Data quality / forensics | Reverse-engineering generator, deterministic formula | 🎯 Trọng tâm |
| Clustering (K-Means) | Phân nhóm đơn theo đặc trưng trước giao | ➕ Phụ |
| Association rules | Luật traffic/distance/size/peak → delayed | ➕ Phụ |
| Hypothesis testing | Chi-square payment/size/type/location/traffic | ➕ Phụ |
| Time series (forecast/staffing) | Demand theo tháng, staffing theo giờ | ➕ Phụ |
| Recommendation | Rule theo popularity/context | ➕ Phụ |
| Customer behavior | Size/type/location/restaurant preference | ➕ Phụ |
| Transportation/Optimization | Gán tài xế giả lập, chi phí tối thiểu | ➕ Phụ |
| BI/Dashboard | Streamlit + Power BI pack | ➕ Phụ (hỗ trợ trình bày) |

---

## 3. Yêu cầu môn học / deliverable phải nộp

- [ ] **Báo cáo PDF** (LaTeX) — có flow: bài toán → dữ liệu → forensics → EDA →
      modeling → DSS → (phụ) → giới hạn.
- [ ] **Slide PDF** (Beamer) — mỗi slide một thông điệp, có caveat data rác.
- [ ] **6 notebook** chạy 0 lỗi, là minh chứng quy trình.
- [ ] **Source code** (`src/`, `scripts/`) tái lập được.
- [ ] **Dashboard Streamlit** + **Power BI data pack**.
- [ ] **Dataset gốc** + mô tả nguồn/license.
- [ ] **Phụ lục bắt buộc**: phân công nhóm, tự chấm, **khai báo dùng AI**.

---

## 4. Tiêu chí ĐẠT (đo được, đơn giản)

**Phần trọng tâm:**
- [ ] Pipeline `P(trễ) → Risk Score → Priority → Action → Queue` chạy đầy đủ.
- [ ] So sánh đủ 6 classifier; chọn model trên **dev theo F2**; báo **test 1 lần**.
- [ ] Luôn có **2 baseline** (always-on-time, always-delayed) cạnh kết quả.
- [ ] Risk Score bóc tách được 6 thành phần, trọng số minh bạch.
- [ ] Forensics chứng minh **≥7 công thức tất định** (`max_abs_error ≈ 0`).
- [ ] Có giải thích thẳng: vì sao cấm cột duration/delay; vì sao điểm cao (data
      gần tất định) chứ không phải "đoán giỏi".

**Phần phụ:** mỗi chủ đề chỉ cần **có artifact + 1 đoạn trình bày + caveat**. Không
cần tối ưu.

**Trình bày:** người mới trong nhóm đọc 1 vòng là hiểu mỗi bước làm gì, đọc được
bảng/hình, hiểu kết luận và giới hạn (thuật ngữ tra `docs/GLOSSARY.md`).

---

## 5. Protocol không thương lượng (luật cứng)

1. Đây là **đồ án môn học** trên data synthetic → **không overclaim** bất cứ kết
   quả nào là dùng được ngoài đời.
2. Chỉ dùng feature **biết trước lúc nhận đơn**. Cấm tuyệt đối:
   `delivery_duration_min`, `delivery_time`, `delay_min`,
   `delivery_efficiency_min_per_km`, `restaurant_avg_time`.
3. Fit trên **train**; chọn/tune trên **dev** (theo F2); báo **test một lần** sau
   khi khóa, kèm khoảng tin cậy.
4. **F2 là metric chính** (bỏ sót đơn trễ tốn hơn báo nhầm) nhưng báo kèm
   Accuracy, Balanced Accuracy, F1, MCC, confusion matrix.
5. Risk/Priority là **chính sách minh bạch**, không phải nhãn tối ưu thống kê.
6. Vận tải: đơn **thật**, tài xế/fleet **giả lập**.
7. **Một nơi sinh số** (`scripts/`), notebook/report/slide chỉ **đọc lại** số đó.

---

## 6. Số khóa hiện tại (locked — đối chiếu báo cáo)

- Corpus 1.004 đơn; 210 trễ / 794 đúng giờ; **tỷ lệ trễ 20,92%**.
- `is_delayed` ≡ `delivery_duration_min > 30` (mismatch 0).
- Split stratified **602 / 201 / 201**; feature `compact_nonredundant` (12).
- **Model khóa: Logistic Regression** (C=1.0, chọn theo F2 dev).
- Test: Acc 0,9602; Bal.Acc 0,9661; Precision 0,8542; Recall 0,9762; F1 0,9111;
  **F2 0,9491**; MCC 0,8889.
- Baseline always-on-time F2=0; always-delayed F2=0,5691.
- Vận tải: 12 đơn High, 6 tài xế giả lập, mean cost 32,84.

---

## 7. Definition of Done (coi như xong khi)

- [ ] Phần trọng tâm ở §4 đều đạt.
- [ ] Phần phụ đều có artifact + caveat (không thiếu chủ đề trong §2).
- [ ] `python -m scripts.run_all` pass **18/18**; tests pass; notebook 0 lỗi.
- [ ] Report PDF + slide PDF build được; số khớp `reports/metrics/*`.
- [ ] Trả lời được các câu hỏi thầy hay hỏi:
  - Vì sao cấm duration/delay làm feature?
  - Vì sao biết ngưỡng trễ ~30–35 phút?
  - Điểm cao có phải may mắn / có ý nghĩa gì khi data là rác?
  - Risk Score 0–100 tính thế nào?
- [ ] Phụ lục phân công + tự chấm + khai báo AI đã điền thật.

---

## 8. Phân công nhóm

> (Để cuối — điền sau khi chốt nội dung. Gợi ý: gắn mỗi người với phần TRỌNG TÂM
> hoặc PHỤ ở §1 + một deliverable ở §3.)

| Thành viên | MSSV | Phụ trách | Mục liên quan |
|---|---|---|---|
| Đoàn Danh Long | 20237354 | … | … |
| Vũ Quang Vinh | 20237408 | … | … |
| Nguyễn Đăng Đức | 20237312 | … | … |
| Nguyễn Tuấn Anh | 20237293 | … | … |
