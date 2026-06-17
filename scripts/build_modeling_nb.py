import sys
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_modeling.ipynb"


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
            # Module 03 - Mô hình hóa và tầng quyết định

            **Câu hỏi.** Trước khi giao, có thể dự báo đơn nào nguy cơ trễ để ưu
            tiên theo dõi/điều phối không, và mô hình nào đáng tin nhất?

            **Phương pháp.** Supervised learning theo đúng kỷ luật chống rò rỉ:

            1. *Fit* preprocessing + model **chỉ trên train**.
            2. *Chọn* mô hình trên **dev** theo F2 (ưu tiên không bỏ sót đơn trễ).
            3. *Kiểm tra độ ổn định* bằng cross-validation trên train.
            4. *Khóa* mô hình rồi **báo test một lần**, kèm khoảng tin cậy bootstrap.

            **Quyết định đầu ra.** Một mô hình khóa + xác suất trễ để tầng DSS biến
            thành Risk Score/Priority.

            > Lưu ý đọc kết quả: dữ liệu là synthetic gần tất định (xem Module 06),
            > nên điểm số rất cao phản ánh độ dễ của dữ liệu, **không** phải năng
            > lực dự báo thực tế. Giá trị học thuật nằm ở *quy trình* và *cách đọc*.
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

            import pandas as pd
            from IPython.display import Image
            from pizza_dss.config import ACTIVE_FEATURE_COLUMNS, COMPACT_FEATURE_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
            from pizza_dss.data_loader import load_processed_splits, validate_feature_contract
            from pizza_dss.decision_rules import get_dss_decision
            from pizza_dss.modeling import (
                bootstrap_test_metrics,
                compare_feature_sets,
                compare_models,
                cross_validate_models,
                evaluate_baselines,
                load_best_model,
                metric_row,
                predict_delay_probability,
                train_and_evaluate,
            )
            """
        ),
        md_cell(
            """
            **Khoá artifact chuẩn.** Cell dưới chạy `train_and_evaluate()` để sinh
            model khóa + các hình đánh giá (so sánh model, confusion matrix, ROC,
            hệ số, PR-curve) dùng chung cho notebook, slide và báo cáo.
            """
        ),
        code_cell("_ = train_and_evaluate()"),
        md_cell(
            """
            ## 0. Định nghĩa các metric (đọc trước khi xem bảng)

            Lớp "trễ" là thiểu số (~21%), nên **không dùng Accuracy một mình**.

            - **Accuracy**: tỷ lệ dự đoán đúng trên tổng. Dễ bị lớp đa số (on-time)
              đánh lừa.
            - **Balanced Accuracy**: trung bình của Recall hai lớp; công bằng với
              lớp thiểu số.
            - **Precision** = TP/(TP+FP): trong các đơn bị *cảnh báo trễ*, bao nhiêu
              % thực sự trễ. Cao = ít báo động giả.
            - **Recall** (Sensitivity) = TP/(TP+FN): trong các đơn *thực sự trễ*,
              bắt được bao nhiêu %. Cao = ít bỏ sót.
            - **F1** = trung bình điều hòa của Precision và Recall (cân bằng).
            - **F2** = F-beta với beta=2, **ưu tiên Recall gấp đôi Precision**. Chọn
              vì bỏ sót đơn trễ (FN) tốn chi phí vận hành hơn cảnh báo dư (FP).
            - **MCC** (Matthews): tương quan giữa dự đoán và thực tế trên cả 4 ô của
              confusion matrix; nằm trong [-1, 1], 0 = đoán mò. Bền với mất cân bằng.
            - **ROC-AUC**: khả năng xếp hạng (đơn trễ có xác suất cao hơn đơn không
              trễ); 0.5 = ngẫu nhiên, 1.0 = hoàn hảo.
            - **TN/FP/FN/TP**: bốn ô confusion matrix (đúng-âm/dương-giả/âm-giả/đúng-dương).
            """
        ),
        md_cell("## 1. Split và feature contract"),
        code_cell(
            """
            validate_feature_contract()
            train_df, dev_df, test_df = load_processed_splits()
            pd.DataFrame([
                {"split": "train", "rows": len(train_df), "delayed": int(train_df[TARGET_COLUMN].sum()),
                 "delayed_rate": round(train_df[TARGET_COLUMN].mean(), 4)},
                {"split": "dev", "rows": len(dev_df), "delayed": int(dev_df[TARGET_COLUMN].sum()),
                 "delayed_rate": round(dev_df[TARGET_COLUMN].mean(), 4)},
                {"split": "test", "rows": len(test_df), "delayed": int(test_df[TARGET_COLUMN].sum()),
                 "delayed_rate": round(test_df[TARGET_COLUMN].mean(), 4)},
            ])
            """
        ),
        md_cell(
            """
            **Insight.** `validate_feature_contract()` chạy không lỗi nghĩa là không
            feature nào nằm trong danh sách rò rỉ. Tỷ lệ trễ ~21% được giữ đều ở cả
            ba split (stratified). Test chỉ có ~42 đơn trễ — đây là lý do ở mục 5 ta
            phải kèm khoảng tin cậy bootstrap thay vì tin tuyệt đối vào điểm số.
            """
        ),
        md_cell("## 2. Baseline bắt buộc trên dev"),
        code_cell("evaluate_baselines(dev_df[TARGET_COLUMN]).round(4)"),
        md_cell(
            """
            **Định nghĩa.** *Always on-time* = đoán mọi đơn không trễ; *Always
            delayed* = đoán mọi đơn trễ. Đây là mốc tối thiểu mọi mô hình phải vượt.

            **Insight.** Always-on-time đạt Accuracy ~0.79 nhưng **F2 = 0, MCC = 0**
            (không bắt được đơn trễ nào) — minh chứng vì sao Accuracy gây hiểu lầm.
            Always-delayed có Recall = 1 nhưng Precision thấp. Mô hình tốt phải vượt
            cả hai về F2 và MCC.
            """
        ),
        md_cell("## 3. So sánh feature set: full vs compact"),
        code_cell(
            """
            {
                "full_pre_dispatch": FEATURE_COLUMNS,
                "compact_nonredundant": COMPACT_FEATURE_COLUMNS,
                "active_for_locked_model": ACTIVE_FEATURE_COLUMNS,
            }
            """
        ),
        code_cell("compare_feature_sets(train_df, dev_df).round(4)"),
        md_cell(
            """
            **Insight.** Compact bỏ các cột tất định (`estimated_duration_min`,
            `topping_density`, `pizza_complexity`, `traffic_impact`) và mã hóa size
            bằng `pizza_size_score` (1-4). Dev F2 của compact ngang hoặc nhỉnh hơn
            full dù **ít feature hơn** → bằng chứng các cột công thức là *redundant*,
            không thêm thông tin. Vì vậy mô hình khóa dùng compact (đơn giản, ít
            nguy cơ overfit).
            """
        ),
        md_cell(
            """
            ## 3b. Vì sao chọn đúng sáu classifier này?

            Sáu mô hình phủ các họ thuật toán trong học phần, từ tuyến tính tới
            ensemble, để so sánh công bằng.
            """
        ),
        code_cell(
            """
            pd.DataFrame([
                {"model": "Logistic Regression", "ho": "Tuyến tính/xác suất",
                 "gia_dinh": "Log-odds tuyến tính theo feature", "vi_sao": "Baseline mạnh, hệ số diễn giải được"},
                {"model": "Decision Tree", "ho": "Cây",
                 "gia_dinh": "Chia ngưỡng đệ quy, phi tuyến", "vi_sao": "Dễ giải thích luật if-then"},
                {"model": "Naive Bayes", "ho": "Xác suất sinh",
                 "gia_dinh": "Feature độc lập có điều kiện, Gaussian", "vi_sao": "Nhanh, mốc tham chiếu"},
                {"model": "k-NN", "ho": "Dựa lân cận",
                 "gia_dinh": "Đơn giống nhau thì cùng nhãn", "vi_sao": "Phi tham số, nhạy với scaling"},
                {"model": "SVM (RBF)", "ho": "Biên cực đại",
                 "gia_dinh": "Tách lớp bằng kernel phi tuyến", "vi_sao": "Mạnh khi biên phức tạp"},
                {"model": "Random Forest", "ho": "Ensemble bagging",
                 "gia_dinh": "Trung bình nhiều cây ngẫu nhiên", "vi_sao": "Giảm phương sai, robust"},
            ])
            """
        ),
        md_cell(
            """
            **Insight.** Mỗi họ có điểm mạnh khác nhau; so sánh trên cùng feature
            set và cùng metric giúp lựa chọn dựa trên bằng chứng, không cảm tính.
            """
        ),
        md_cell("## 4. So sánh sáu classifier trên dev (một lần split)"),
        code_cell(
            """
            dev_metrics, fitted = compare_models(train_df, dev_df)
            dev_metrics.round(4)
            """
        ),
        md_cell(
            """
            **Cách đọc.** Bảng đã sắp theo F2 giảm dần (rồi Balanced Accuracy, MCC).
            Cột tn/fp/fn/tp cho thấy cơ cấu lỗi của từng mô hình.

            **Insight.** Logistic Regression đứng đầu F2/MCC; Naive Bayes có Recall
            cao nhưng nhiều FP nên Precision và F2 thấp. Đây mới là **một lần** chia
            dev nên cần kiểm tra độ ổn định ở mục 4b.
            """
        ),
        img_cell("model_comparison.png"),
        md_cell("**Insight (hình).** Trực quan hóa khoảng cách F2/Balanced-Accuracy/MCC giữa các mô hình: nhóm tuyến tính/SVM/cây vượt trội, Naive Bayes tụt lại."),
        img_cell("roc_curves.png"),
        md_cell("**Insight (hình).** ROC nằm sát góc trên-trái = tách lớp tốt; AUC trong chú thích xác nhận Logistic Regression và SVM xếp hạng rủi ro tốt nhất."),
        md_cell(
            """
            ## 4b. Cross-validation trên train (kiểm tra độ ổn định)

            **Định nghĩa.** Stratified k-fold: chia train thành k phần giữ nguyên tỷ
            lệ trễ, lần lượt train trên k-1 phần và đánh giá trên phần còn lại. Báo
            mean ± std để thấy mô hình *ổn định* hay *may rủi theo split*.
            """
        ),
        code_cell("cross_validate_models(train_df, k=5).round(4)"),
        md_cell(
            """
            **Insight.** F2 trung bình 5-fold cao và **độ lệch chuẩn nhỏ** ở các mô
            hình dẫn đầu → kết quả không phải nhờ một split may mắn. Logistic
            Regression vừa cao vừa ổn định, củng cố lựa chọn làm mô hình khóa.
            """
        ),
        md_cell("## 5. Test sau khi khóa mô hình (báo một lần)"),
        code_cell(
            """
            best_name = dev_metrics.iloc[0]["model"]
            best_model = fitted[best_name]
            prob = predict_delay_probability(best_model, test_df)
            pred = prob >= 0.5
            pd.DataFrame([metric_row(best_name, test_df[TARGET_COLUMN], pred, prob)]).round(4)
            """
        ),
        md_cell("**Insight.** Mô hình khóa vượt xa cả hai baseline trên test. Recall cao = ít bỏ sót đơn trễ, đúng mục tiêu vận hành."),
        img_cell("confusion_matrix_test.png"),
        md_cell(
            """
            **Insight (hình).** Đọc theo chi phí vận hành: ô **FN** (đơn trễ bị bỏ
            sót) là tốn nhất vì khách không được cảnh báo; ô **FP** chỉ tốn công theo
            dõi dư. Mô hình giữ FN rất thấp — phù hợp việc tối ưu F2.
            """
        ),
        img_cell("model_coefficients.png"),
        md_cell("**Insight (hình).** Hệ số dương lớn ở distance/traffic cao = các yếu tố này tăng odds trễ, khớp EDA và công thức Risk Score (Module 04)."),
        md_cell(
            """
            ## 5b. Khoảng tin cậy bootstrap cho metric test

            **Định nghĩa.** Lấy mẫu lại có hoàn lại từ 201 đơn test nhiều lần, tính
            metric mỗi lần, lấy phân vị 2.5%-97.5% làm khoảng tin cậy 95%. Cần thiết
            vì test chỉ ~42 đơn trễ, một vài FN có thể làm điểm dao động mạnh.
            """
        ),
        code_cell("bootstrap_test_metrics(best_model, test_df, n_boot=2000)"),
        md_cell(
            """
            **Insight.** Khoảng tin cậy của F2/MCC khá **rộng** (do n nhỏ), nên phải
            trình bày điểm test kèm CI, **không** tuyên bố một con số tuyệt đối. Đây
            là cách đọc trung thực với cỡ mẫu nhỏ.
            """
        ),
        img_cell("pr_curve.png"),
        md_cell(
            """
            **Insight (hình).** Đường Precision-Recall và điểm F2-optimal cho thấy vì
            sao ngưỡng vận hành có thể lệch khỏi 0.5: muốn tối đa F2 (ưu tiên Recall),
            ngưỡng tối ưu thường thấp hơn, đánh đổi thêm vài FP để bắt thêm đơn trễ.
            """
        ),
        md_cell("## 6. Baseline test (đối chứng)"),
        code_cell("evaluate_baselines(test_df[TARGET_COLUMN]).round(4)"),
        md_cell("**Insight.** Đặt cạnh mục 5 để thấy mức cải thiện thực của mô hình so với đoán-một-lớp; luôn báo baseline cạnh kết quả cuối."),
        md_cell("## 7. Ví dụ tầng DSS trên hàng đợi test"),
        code_cell(
            """
            model = load_best_model()
            probabilities = predict_delay_probability(model, test_df)
            rows = []
            for i, (_, order) in enumerate(test_df.head(10).iterrows()):
                decision = get_dss_decision(order, probabilities[i])
                rows.append({
                    "order_id": order["order_id"],
                    "traffic_level": order["traffic_level"],
                    "distance_km": order["distance_km"],
                    "true_is_delayed": order["is_delayed"],
                    **decision,
                })
            pd.DataFrame(rows).sort_values("delay_risk_score", ascending=False)
            """
        ),
        md_cell(
            """
            **Insight.** Xác suất trễ được biến thành `delay_risk_score`, `priority`
            và `recommended_action`. Risk Score là **chính sách minh bạch**, không
            phải nhãn học từ dữ liệu — chi tiết công thức ở Module 04.
            """
        ),
        md_cell("## 8. Insight → quyết định mô hình hóa"),
        code_cell(
            """
            pd.DataFrame([
                {"cau_hoi": "Metric chọn mô hình?", "bang_chung": "Trễ là thiểu số ~21%; FN đắt hơn FP.",
                 "quyet_dinh": "Chọn theo F2 trên dev, báo kèm Balanced Acc/MCC, không dùng Accuracy đơn lẻ."},
                {"cau_hoi": "Feature set nào?", "bang_chung": "Compact F2 >= full dù ít cột; cột công thức redundant.",
                 "quyet_dinh": "Khóa mô hình trên compact (12 feature)."},
                {"cau_hoi": "Mô hình nào?", "bang_chung": "LogReg dẫn đầu F2/MCC, CV ổn định, hệ số diễn giải được.",
                 "quyet_dinh": "Khóa Logistic Regression."},
                {"cau_hoi": "Điểm test tin được không?", "bang_chung": "n=201 (~42 trễ); CI bootstrap rộng.",
                 "quyet_dinh": "Báo test kèm CI; không overclaim; nêu rõ dữ liệu synthetic."},
            ])
            """
        ),
    ]
    return nb


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Saved executed notebook -> {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
