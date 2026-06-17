import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_dss_optimization_powerbi.ipynb"


def code_cell(source):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def md_cell(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def img_cell(name):
    return code_cell(f'Image(filename=str(PROJECT_ROOT / "reports" / "figures" / "{name}"))')


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        md_cell(
            """
            # Module 04 - DSS, tối ưu vận tải và Power BI pack

            **Câu hỏi.** Sau khi có xác suất trễ, làm sao biến nó thành *quyết định
            hành động* (theo dõi/điều phối) một cách minh bạch và có thể giải thích?

            **Phương pháp.** Tầng prescriptive 3 bước: (1) `delay_risk_score` =
            trộn có trọng số xác suất mô hình với các "áp lực" vận hành; (2) phân
            `priority` Low/Medium/High theo ngưỡng; (3) gán tài xế giả lập theo chi
            phí tối thiểu. Mỗi bước đều được *kiểm chứng* và *bóc tách*.

            **Quyết định đầu ra.** Hàng đợi ưu tiên + hành động khuyến nghị + gán
            tài xế kịch bản.
            """
        ),
        md_cell(
            """
            > **Đối tượng & cách đọc.** Notebook cho **người mới** trong nhóm. Lời
            > giải thích bằng tiếng Việt; thuật ngữ tiếng Anh (F2, MCC, bootstrap…)
            > được định nghĩa ở mục **"0. Định nghĩa"** và đầy đủ trong
            > `docs/GLOSSARY.md`. Mỗi mục theo mạch *mục tiêu → phân tích → insight*;
            > mỗi bảng/biểu đồ có **một dòng đọc-hiểu ngay bên dưới**.
            """
        ),
        code_cell(
            """
            import sys
            from pathlib import Path

            PROJECT_ROOT = Path.cwd()
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            import numpy as np
            import pandas as pd
            from IPython.display import Image
            from pizza_dss.config import PRIORITY_LOW_MAX, PRIORITY_MEDIUM_MAX
            from pizza_dss.dashboard_data import build_dashboard_data
            from pizza_dss.data_loader import load_processed_splits
            from pizza_dss.decision_rules import calculate_delay_risk_components, calculate_delay_risk_score, get_priority_level
            from pizza_dss.modeling import load_best_model, predict_delay_probability
            from pizza_dss.powerbi import build_powerbi_pack
            from pizza_dss.transport_optimization import build_transport_artifacts, solve_transport_assignment
            """
        ),
        md_cell(
            """
            ## 0. Định nghĩa tầng DSS

            - **Delay Risk Score (0-100)** = `0.55*P(trễ)` + `0.15*traffic` +
              `0.12*distance` + `0.08*peak` + `0.06*complexity` + `0.04*weekend`.
              Trọng số là **chính sách minh bạch** do nhóm chọn (ưu tiên tín hiệu mô
              hình), không phải tối ưu thống kê.
            - **Priority**: Low nếu score ≤ 35, Medium nếu ≤ 65, còn lại High
              (ngưỡng `PRIORITY_LOW_MAX` / `PRIORITY_MEDIUM_MAX` trong `config`).
            - **Calibration**: kiểm tra đơn risk cao có *thực sự* trễ nhiều hơn
              không (so risk band với delay rate thật).
            - **Bài toán phân công (assignment)**: gán đơn ↔ tài xế sao cho tổng chi
              phí nhỏ nhất (Hungarian; greedy nếu thiếu scipy).
            """
        ),
        md_cell("## 1. Priority queue"),
        code_cell(
            """
            queue = build_dashboard_data()
            queue.head(10)
            """
        ),
        md_cell("**Insight.** Hàng đợi sắp theo `delay_risk_score` giảm dần — đơn rủi ro nhất nổi lên đầu để quản lý xử lý trước."),
        code_cell('queue["priority"].value_counts().reindex(["Low", "Medium", "High"]).rename_axis("priority").reset_index(name="orders")'),
        img_cell("priority_distribution.png"),
        md_cell("**Insight (hình).** Phần lớn đơn ở Low/Medium; nhóm High là thiểu số cần can thiệp — đúng tinh thần phân bổ nguồn lực có trọng tâm."),
        md_cell(
            """
            ### 1a. Bóc tách công thức Risk Score (ví dụ tính tay)

            Lấy một đơn test thật, tính từng thành phần × trọng số = đóng góp, để
            thấy điểm số đến từ đâu.
            """
        ),
        code_cell(
            """
            _, _, test_df = load_processed_splits()
            model = load_best_model()
            probs = predict_delay_probability(model, test_df)
            order = test_df.iloc[0]
            p = float(probs[0])
            components = calculate_delay_risk_components(order, p)
            weights = {"model_probability": 0.55, "traffic_pressure": 0.15, "distance_pressure": 0.12,
                       "peak_pressure": 0.08, "complexity_pressure": 0.06, "weekend_pressure": 0.04}
            breakdown = pd.DataFrame([
                {"component": k, "value": round(components[k], 2), "weight": w,
                 "contribution": round(components[k] * w, 2)}
                for k, w in weights.items()
            ])
            breakdown
            """
        ),
        code_cell(
            """
            {
                "order_id": order["order_id"],
                "model_probability": round(p, 4),
                "delay_risk_score": calculate_delay_risk_score(order, p),
                "priority": get_priority_level(calculate_delay_risk_score(order, p)),
            }
            """
        ),
        md_cell("**Insight.** Tổng `contribution` đúng bằng `delay_risk_score`; thành phần xác suất mô hình (0.55) chi phối, các áp lực vận hành tinh chỉnh thêm. Nhờ vậy mỗi điểm số đều *giải thích được* cho người vận hành."),
        md_cell(
            """
            ### 1b. Kiểm chứng calibration: risk cao có trễ thật nhiều hơn?
            """
        ),
        code_cell(
            """
            queue["risk_band"] = pd.cut(queue["delay_risk_score"], [-1, PRIORITY_LOW_MAX, PRIORITY_MEDIUM_MAX, 1000],
                                        labels=["Low", "Medium", "High"])
            queue.groupby("risk_band", observed=True)["true_is_delayed"].agg(
                orders="count", actual_delayed="sum", actual_delay_rate="mean"
            ).reset_index()
            """
        ),
        md_cell("**Insight (calibration).** `actual_delay_rate` tăng dần Low → Medium → High ⇒ Risk Score xếp hạng rủi ro đúng hướng với thực tế trên test. Đây là bằng chứng tầng DSS không chỉ đẹp về hình thức mà còn bám outcome."),
        md_cell("### 1c. Độ nhạy ngưỡng priority"),
        code_cell(
            """
            rows = []
            for low, high in [(30, 60), (PRIORITY_LOW_MAX, PRIORITY_MEDIUM_MAX), (40, 70)]:
                band = pd.cut(queue["delay_risk_score"], [-1, low, high, 1000], labels=["Low", "Medium", "High"])
                counts = band.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
                rows.append({"low_cut": low, "high_cut": high, **counts.to_dict()})
            pd.DataFrame(rows)
            """
        ),
        img_cell("risk_score_histogram.png"),
        md_cell("**Insight (độ nhạy).** Số đơn High thay đổi theo ngưỡng — đây là *lựa chọn năng lực xử lý*, không phải sự thật khách quan. Báo cáo nêu rõ ngưỡng 35/65 là chính sách có thể điều chỉnh theo nguồn lực thực."),
        md_cell("## 2. Bài toán vận tải / phân công"),
        code_cell(
            """
            summary = build_transport_artifacts()
            summary
            """
        ),
        code_cell(
            """
            assignments = solve_transport_assignment(top_n=12)
            assignments
            """
        ),
        img_cell("transport_assignment_cost.png"),
        md_cell(
            """
            **Insight.** Chi phí gán = thời gian di chuyển (theo distance/speed) +
            phạt ưu tiên + phạt traffic cao − bonus cùng khu vực. Hungarian chọn
            tổ hợp gán **tổng chi phí nhỏ nhất**; greedy là phương án dự phòng nếu
            thiếu scipy. **Lưu ý:** đơn hàng là thật nhưng driver/capacity là *giả
            lập* vì dataset không có bảng tài xế — đây là minh hoạ prescriptive.
            """
        ),
        md_cell("## 3. Power BI-ready pack"),
        code_cell("build_powerbi_pack()"),
        code_cell('sorted([p.name for p in (PROJECT_ROOT / "powerbi").iterdir()])'),
        md_cell(
            """
            **Insight.** Pack gồm bảng fact (orders, delay_queue, transport,
            forecast, staffing, rules, audit, mix) + dim (restaurant, location,
            date) + `measures.dax` + spec. Đây là **data pack** để dựng `.pbix`
            trong Power BI Desktop, chưa phải file `.pbix` sẵn.
            """
        ),
        md_cell("## 4. Insight → quyết định tầng DSS"),
        code_cell(
            """
            pd.DataFrame([
                {"cau_hoi": "Risk Score lấy ở đâu?", "bang_chung": "Tổng contribution = score; model prob chiếm 0.55.",
                 "quyet_dinh": "Trình bày như chính sách minh bạch, bóc tách được, không phải nhãn học."},
                {"cau_hoi": "Risk có đúng thực tế?", "bang_chung": "actual_delay_rate tăng theo risk band.",
                 "quyet_dinh": "Dùng Risk Score để xếp hàng đợi; vẫn nêu là hỗ trợ, không tự động điều phối."},
                {"cau_hoi": "Ngưỡng 35/65 cố định?", "bang_chung": "Số đơn High đổi theo ngưỡng.",
                 "quyet_dinh": "Coi ngưỡng là tham số theo năng lực xử lý, ghi rõ trong báo cáo."},
                {"cau_hoi": "Tối ưu vận tải thật chưa?", "bang_chung": "Không có bảng tài xế thật.",
                 "quyet_dinh": "Giữ là kịch bản giả lập; không tuyên bố tối ưu dispatch thực tế."},
            ])
            """
        ),
    ]
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    ).execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
