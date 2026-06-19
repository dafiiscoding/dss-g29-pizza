# Hướng dẫn thuyết trình Slide — Pizza Delivery DSS

> Khớp với `pizza_dss_slides.tex` / `PIZZA_DSS_SLIDE_DECK.pdf` bản hiện tại
> (**33 frame nội dung** + title + mục lục + cảm ơn). Mỗi mục dưới đây là một
> frame; số thứ tự trùng số trên tiêu đề slide.
> Thời lượng gợi ý: **20–25 phút** + hỏi đáp.

## Bố cục & phân vai gợi ý

| Phần | Frame | Người trình bày (gợi ý) |
|---|---|---|
| Mở đầu + Phần I — Bài toán & dữ liệu | Title → 5 | Đoàn Danh Long |
| Phần II — Dự đoán đơn trễ | 6 → 16 | Vũ Quang Vinh + Đoàn Danh Long |
| Phần III — Từ dự đoán đến quyết định | 17 → 22 | Nguyễn Tuấn Anh |
| Phần IV — Mở rộng & trình bày | 23 → 33 | Nguyễn Đăng Đức |

Thông điệp xuyên suốt cần lặp lại: **xương sống là "dự đoán đơn trễ → ra quyết
định"; các phân tích nghiệp vụ chỉ là phần mở rộng; dữ liệu là giả lập nên không
thổi phồng kết quả.**

---

## Title + Mục lục
- Giới thiệu nhóm, đề tài. Một câu chốt: "Hệ hỗ trợ quyết định đoán trước đơn nào
  dễ trễ và biến dự đoán đó thành hành động điều phối."
- Mục lục: chỉ ra 4 phần, nhấn Phần II–III là lõi.

## Phần I — Bài toán & dữ liệu
1. **Bài toán & quyết định** — bối cảnh (giờ cao điểm, đơn trễ), 3 lớp DSS: đầu vào → dự đoán → đề xuất. Chỉ vào sơ đồ pipeline.
2. **Dữ liệu & chống rò rỉ** — 1.004 đơn; nhấn: cấm cột sau-giao-hàng vì chúng sinh ra nhãn (rò rỉ = xem trước đáp án).
3. **Suy luận ngưỡng trễ** — không tự đặt SLA; dò ngưỡng → 30 phút khớp 0 sai lệch. Điểm chặt chẽ, nói chậm.
4. **Truy vết — vì sao** — thừa nhận dữ liệu giả lập; biến điểm yếu thành phân tích; không thổi phồng.
5. **Truy vết — bằng chứng** — 7 công thức tất định (sai số ~0). Kết: biến trùng thông tin → loại ở bước chọn đặc trưng.

## Phần II — Dự đoán đơn trễ (LÕI)
6. **Phân phối nhãn & cách đánh giá** — chỉ ~21% đơn trễ → không nhìn mỗi độ chính xác; chọn F2 (phạt bỏ sót).
7. **Giao thông** — yếu tố gây trễ số 1; sẽ đưa vào điểm rủi ro.
8. **Quãng đường & độ phức tạp** — xa + đông = gần chắc trễ; complexity đóng góp nhỏ hơn.
9. **Chọn đặc trưng** — bộ đầy đủ (17) vs rút gọn (12); vì sao ưu tiên rút gọn.
10. **Rút gọn thắng** — bảng F2/MCC: rút gọn cao hơn dù ít biến → khóa bộ 12 biến.
11. **So sánh 6 mô hình** — Hồi quy Logistic tốt nhất theo F2; nêu nhanh các mô hình khác.
12. **Vì sao loại Naive Bayes** — điểm thấp + giả định độc lập bị vi phạm; đã thử chứ không bỏ qua.
13. **Tinh chỉnh siêu tham số** — chọn-xong-mới-tinh-chỉnh; bản gốc tốt hơn nên giữ (trung thực).
14. **Quét ngưỡng quyết định** — ngưỡng tối ưu trên dev không "transfer" tốt sang test → giữ 0,5.
15. **Độ ổn định (100 lần)** — F2 ~0,94 ổn định, không phải may mắn một lần chia tập.
16. **Kết quả test & baseline** — bảng test; vượt 2 baseline; bỏ sót 1/42 đơn. Nhắc: điểm cao một phần do dữ liệu gần tất định.

## Phần III — Từ dự đoán đến quyết định (LÕI)
17. **Điểm rủi ro: công thức** — kết hợp xác suất mô hình (55%) + áp lực vận hành (45%); là chính sách minh bạch, không phải tối ưu thống kê.
18. **Bóc tách thành phần** — cho điều phối viên thấy *vì sao* một đơn rủi ro cao.
19. **Hiệu chuẩn** — tỷ lệ trễ thực tế tăng dần theo nhóm (0% → 7,3% → 88,6%) ⇒ điểm rủi ro "đúng".
20. **Độ nhạy ngưỡng** — nhóm cao bắt 92,9% đơn trễ; ngưỡng là nút điều chỉnh tải.
21. **Hàng đợi & hành động** — bảng Cao/Vừa/Thấp → hành động cụ thể.
22. **Xếp tài xế** — bài toán phân công (Hungary); đơn thật, đội xe giả lập; 12 đơn/6 tài xế.

## Phần IV — Mở rộng & trình bày
23. **[Mở rộng] Hành vi khách hàng** — size/type; nhắc caveat giả lập.
24. **[Mở rộng] Dự báo nhu cầu** — sai số cao, chỉ minh hoạ cách lập kế hoạch.
25. **[Mở rộng] Xếp ca & xu hướng** — cao điểm 19h; xu hướng là đặc tính bộ sinh dữ liệu.
26. **Đối chiếu rubric** — 4 nhóm yêu cầu môn đều phủ; chi tiết 20 mục ở Bảng 7.1 báo cáo.
27. **Bảng điều khiển — tổng quan & khám phá** — chỉ vào ảnh chụp, nói vai trò người điều phối.
28. **Bảng điều khiển — quyết định & chất lượng dữ liệu** — thử một đơn, hàng đợi, cảnh báo dữ liệu giả lập.
29. **Tái lập** — một lệnh dựng lại, bộ kiểm thử pass; số liệu sinh một nơi.
30–31. **Câu hỏi phản biện** — đọc lướt; đây là "đạn" thủ sẵn cho phần hỏi đáp.
32. **Giới hạn** — nói thẳng: không quảng cáo điểm 96% là năng lực thật; chưa dùng mô hình deep mới.
33. **Kết luận & hướng phát triển** — chốt giá trị: quy trình + truy vết + ra quyết định minh bạch.

## Cảm ơn
- Mời câu hỏi. Đội cử 1 người điều phối hỏi đáp, dựa vào frame 30–31 và phần Giới hạn.

---

## Mẹo trình bày
- Mỗi frame **một thông điệp**; đừng đọc nguyên chữ trên slide.
- Khi bị hỏi "điểm cao thế dùng được không?" → trả lời theo frame 31: dữ liệu giả lập, giá trị là quy trình.
- Giữ nhịp: Phần I ~4', Phần II ~9', Phần III ~6', Phần IV ~5'.
