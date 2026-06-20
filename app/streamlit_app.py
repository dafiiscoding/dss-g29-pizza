import sys
from html import escape
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.config import ACTIVE_FEATURE_COLUMNS, FIGURES_DIR, METRICS_DIR
from pizza_dss.business_analysis import (
    customer_preference_tables,
    forecast_metrics,
    forecast_monthly_demand,
    hourly_staffing_plan,
    hypothesis_tests,
    recommendation_rules,
    synthetic_data_audit,
)
from pizza_dss.dashboard_data import load_dashboard_data, make_single_order_frame
from pizza_dss.data_loader import load_dataset
from pizza_dss.decision_rules import explain_delay_risk_score, get_dss_decision
from pizza_dss.modeling import load_best_model, predict_delay_probability
from pizza_dss.transport_optimization import solve_transport_assignment

# --- Việt hoá tên cột & giá trị hiển thị (chỉ ảnh hưởng phần hiển thị) ---
VN_COLS = {
    "order_id": "Mã đơn",
    "restaurant_name": "Nhà hàng",
    "location": "Khu vực",
    "distance_km": "Cự ly (km)",
    "traffic_level": "Giao thông",
    "delayed_probability": "Xác suất trễ",
    "delay_risk_score": "Điểm rủi ro",
    "priority": "Ưu tiên",
    "recommended_action": "Hành động gợi ý",
    "component": "Thành phần",
    "component_score": "Điểm thành phần",
    "weight": "Trọng số",
    "weighted_contribution": "Đóng góp",
    "rationale": "Diễn giải",
    "driver_id": "Tài xế",
    "driver_slot": "Lượt xe",
    "estimated_assignment_cost": "Chi phí gán",
    "model": "Mô hình",
    "accuracy": "Độ chính xác",
    "balanced_accuracy": "Độ chính xác cân bằng",
    "precision": "Precision (đúng khi cảnh báo)",
    "recall": "Recall (bắt đơn trễ)",
    "f1": "F1",
    "f2": "F2",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "tn": "Đúng giờ đoán đúng",
    "fp": "Báo trễ nhầm",
    "fn": "Bỏ sót trễ",
    "tp": "Trễ bắt đúng",
    "param_C": "Tham số C",
    "cv_f2_mean": "F2 trung bình CV",
    "cv_f2_std": "Độ lệch CV",
    "delta_dev_f2_vs_default": "Chênh F2 so với mặc định",
    "decision": "Quyết định",
    "decision_reason": "Lý do quyết định",
    "note": "Ghi chú",
    "pizza_size_label": "Cỡ bánh",
    "pizza_size": "Cỡ bánh",
    "pizza_type": "Loại bánh",
    "payment_method": "Phương thức thanh toán",
    "payment_category": "Nhóm thanh toán",
    "order_share": "Tỷ trọng",
    "order_period": "Tháng",
    "actual_orders": "Thực tế",
    "forecast_orders": "Dự báo",
    "order_hour": "Giờ",
    "severity": "Mức độ",
    "column": "Cột dữ liệu",
    "recovered_formula": "Công thức phục dựng",
    "max_abs_error": "Sai số lớn nhất",
    "matches_exactly": "Khớp tuyệt đối",
    "check": "Phát hiện",
    "evidence": "Bằng chứng",
    "interpretation": "Diễn giải",
    "rule_type": "Loại luật",
    "context": "Ngữ cảnh",
    "recommendation": "Gợi ý",
    "context_orders": "Số đơn trong ngữ cảnh",
    "matched_orders": "Số đơn khớp",
    "confidence": "Độ tin cậy",
    "rule": "Luật kiểm tra",
    "threshold_minutes": "Ngưỡng phút",
    "mismatches": "Số dòng lệch",
    "mismatch_rate": "Tỷ lệ lệch",
    "exact_match": "Khớp nhãn gốc",
    "false_positives": "Báo trễ nhầm",
    "false_negatives": "Bỏ sót trễ",
    "predicted_delayed": "Số đơn dự báo trễ",
    "actual_delayed": "Số đơn trễ thật",
    "metric": "Chỉ số",
    "mean": "Trung bình",
    "std": "Độ lệch chuẩn",
    "min": "Nhỏ nhất",
    "p05": "P5",
    "p50": "Trung vị",
    "p95": "P95",
    "max": "Lớn nhất",
    "feature": "Feature",
    "role": "Vai trò nghiệp vụ",
    "preprocessing": "Cách xử lý",
    "why_used": "Vì sao dùng",
    "strength": "Điểm mạnh",
    "limitation": "Hạn chế",
    "model_decision": "Kết luận",
    "score_rule": "Quy tắc điểm",
    "number_origin": "Nguồn gốc con số",
    "source": "Nguồn",
}
VN_PRIORITY = {"High": "Cao", "Medium": "Vừa", "Low": "Thấp"}
VN_PRIORITY_DETAIL = {
    "High": "Cao - xử lý ngay",
    "Medium": "Vừa - theo dõi sát",
    "Low": "Thấp - xử lý thường",
}
VN_TRAFFIC = {"High": "Đông", "Medium": "Vừa", "Low": "Thoáng"}
VN_SIZE = {"Small": "Nhỏ", "Medium": "Vừa", "Large": "Lớn", "XL": "Rất lớn"}
VN_PIZZA_TYPE = {
    "BBQ Chicken": "Gà BBQ",
    "Cheese Burst": "Nhiều phô mai",
    "Deep Dish": "Đế dày",
    "Gluten-Free": "Không gluten",
    "Margarita": "Margarita",
    "Non-Veg": "Mặn / thịt",
    "Sicilian": "Sicilia",
    "Stuffed Crust": "Viền nhồi",
    "Thai Chicken": "Gà kiểu Thái",
    "Thin Crust": "Đế mỏng",
    "Veg": "Chay",
    "Vegan": "Thuần chay",
}
VN_PAYMENT = {
    "Card": "Thẻ",
    "Cash": "Tiền mặt",
    "Domino's Cash": "Tiền mặt Domino's",
    "Hut Points": "Điểm Hut",
    "UPI": "UPI",
    "Wallet": "Ví điện tử",
}
VN_PAYMENT_CATEGORY = {"Online": "Trực tuyến", "Offline": "Tại quầy / tiền mặt"}
VN_MODEL = {
    "Logistic Regression": "Hồi quy Logistic",
    "Decision Tree": "Cây quyết định",
    "Naive Bayes": "Naive Bayes",
    "k-NN": "k-NN",
    "SVM": "SVM",
    "Random Forest": "Rừng ngẫu nhiên",
    "Always on-time": "Luôn đoán đúng giờ",
    "Always delayed": "Luôn đoán trễ",
    "default_lr": "Logistic mặc định",
    "tuned_lr": "Logistic sau fine-tune",
}
VN_DECISION = {"locked": "Chọn dùng", "not_locked": "Không dùng"}
VN_METRIC = {
    "accuracy": "Độ chính xác",
    "balanced_accuracy": "Độ chính xác cân bằng",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "f2": "F2",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
}
VN_COMPONENT = {
    "model_probability": "Xác suất từ mô hình",
    "traffic_pressure": "Áp lực giao thông",
    "distance_pressure": "Áp lực cự ly",
    "peak_pressure": "Áp lực giờ cao điểm",
    "complexity_pressure": "Độ phức tạp món",
    "weekend_pressure": "Áp lực cuối tuần",
}
VN_SEVERITY = {"critical": "Nghiêm trọng", "warning": "Cảnh báo", "info": "Thông tin"}
ACTION_TRANSLATIONS = {
    "Escalate to dispatch lead, prepare backup driver, and notify customer support before the promised delivery window is missed.": (
        "Ưu tiên xử lý ngay: báo điều phối trưởng, chuẩn bị tài xế dự phòng "
        "và nhắn bộ phận chăm sóc khách hàng trước khi đơn trễ hẹn giao."
    )
}


def translate_action(action):
    return ACTION_TRANSLATIONS.get(str(action), str(action))


def _map_series(series, mapping):
    return series.map(lambda value: mapping.get(str(value), value))


def localize_values(df):
    out = df.copy()
    mappings = {
        "priority": VN_PRIORITY,
        "traffic_level": VN_TRAFFIC,
        "pizza_size": VN_SIZE,
        "pizza_type": VN_PIZZA_TYPE,
        "payment_method": VN_PAYMENT,
        "payment_category": VN_PAYMENT_CATEGORY,
        "model": VN_MODEL,
        "decision": VN_DECISION,
        "metric": VN_METRIC,
        "component": VN_COMPONENT,
        "severity": VN_SEVERITY,
    }
    for column, mapping in mappings.items():
        if column in out.columns:
            out[column] = _map_series(out[column], mapping)
    if "recommended_action" in out.columns:
        out["recommended_action"] = out["recommended_action"].map(translate_action)
    if "true_is_delayed" in out.columns:
        out["true_is_delayed"] = out["true_is_delayed"].map(
            lambda value: "Trễ" if str(value).lower() == "true" else "Đúng giờ"
        )
    if "exact_match" in out.columns:
        out["exact_match"] = out["exact_match"].map(
            lambda value: "Có" if str(value).lower() == "true" else "Không"
        )
    if "matches_exactly" in out.columns:
        out["matches_exactly"] = out["matches_exactly"].map(
            lambda value: "Có" if str(value).lower() == "true" else "Không"
        )
    return out


def vn(df, drop=None):
    """Bỏ cột thừa rồi đổi tên cột sang tiếng Việt cho dễ đọc."""
    out = df.copy()
    if drop:
        out = out.drop(columns=[c for c in drop if c in out.columns])
    out = localize_values(out)
    out = out.rename(columns={k: v for k, v in VN_COLS.items() if k in out.columns})
    seen = {}
    unique_columns = []
    for column in out.columns:
        seen[column] = seen.get(column, 0) + 1
        unique_columns.append(column if seen[column] == 1 else f"{column} ({seen[column]})")
    out.columns = unique_columns
    return out


def only_columns(df, columns):
    return df[[column for column in columns if column in df.columns]].copy()


def format_probability_columns(df, columns):
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "")
    return out


def format_score_columns(df, columns, digits=4):
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda value: round(float(value), digits) if pd.notna(value) else value)
    return out


def metric_definition_table():
    return pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "role": "Tỷ lệ dự đoán đúng trên toàn bộ đơn.",
                "why_used": "Dễ hiểu nhưng dễ gây ảo giác khi dữ liệu lệch lớp.",
            },
            {
                "metric": "balanced_accuracy",
                "role": "Trung bình Recall của hai lớp: trễ và đúng giờ.",
                "why_used": "Không để lớp đúng giờ chiếm đa số che mất lỗi bỏ sót đơn trễ.",
            },
            {
                "metric": "precision",
                "role": "Trong các đơn bị cảnh báo trễ, tỷ lệ trễ thật là bao nhiêu.",
                "why_used": "Giúp kiểm soát báo động giả để điều phối viên không bị quá tải.",
            },
            {
                "metric": "recall",
                "role": "Trong các đơn trễ thật, mô hình bắt được bao nhiêu.",
                "why_used": "Quan trọng vì bỏ sót đơn trễ gây ảnh hưởng trực tiếp tới khách hàng.",
            },
            {
                "metric": "f2",
                "role": "F-score đặt Recall nặng hơn Precision.",
                "why_used": "Phù hợp mục tiêu DSS: ưu tiên phát hiện sớm đơn có nguy cơ trễ.",
            },
            {
                "metric": "mcc",
                "role": "Độ tương quan giữa nhãn thật và nhãn dự đoán: 1 hoàn hảo, 0 gần như đoán mò, -1 dự đoán ngược.",
                "why_used": "Dùng đủ TP/TN/FP/FN nên đánh giá cân bằng hơn Accuracy khi dữ liệu lệch lớp.",
            },
        ]
    )


def feature_role_table():
    return pd.DataFrame(
        [
            {
                "feature": "distance_km",
                "role": "Cự ly giao hàng.",
                "preprocessing": "Chuẩn hoá số.",
                "why_used": "Cự ly càng xa càng ít dư địa thời gian; EDA cho thấy rủi ro tăng mạnh sau khoảng 6 km.",
            },
            {
                "feature": "toppings_count",
                "role": "Số topping của đơn.",
                "preprocessing": "Chuẩn hoá số.",
                "why_used": "Đại diện tải khâu chuẩn bị bếp; nhiều topping thường làm đơn phức tạp hơn.",
            },
            {
                "feature": "order_hour",
                "role": "Giờ đặt đơn.",
                "preprocessing": "Chuẩn hoá số.",
                "why_used": "Bắt hiệu ứng giờ cao điểm, đặc biệt khung tối 18h-20h.",
            },
            {
                "feature": "pizza_size_score",
                "role": "Cỡ bánh mã hoá thứ bậc Small=1 đến XL=4.",
                "preprocessing": "Tạo từ pizza_size rồi chuẩn hoá số.",
                "why_used": "Cỡ bánh có thứ tự tự nhiên; mã hoá thứ bậc giữ quan hệ lớn/nhỏ và giảm số cột.",
            },
            {
                "feature": "restaurant_name",
                "role": "Chuỗi cửa hàng.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Giữ tín hiệu khác biệt vận hành/brand nếu có, đồng thời được kiểm tra ablation để tránh diễn giải quá mức.",
            },
            {
                "feature": "location",
                "role": "Khu vực giao.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Đại diện bối cảnh địa lý và mật độ giao hàng theo thành phố/khu vực.",
            },
            {
                "feature": "pizza_type",
                "role": "Loại bánh.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Một số loại bánh có thể liên quan đến độ phức tạp chuẩn bị.",
            },
            {
                "feature": "traffic_level",
                "role": "Mức giao thông.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Driver rủi ro rõ nhất trong EDA; đường đông làm tỷ lệ trễ tăng mạnh.",
            },
            {
                "feature": "payment_category",
                "role": "Nhóm thanh toán online/offline.",
                "preprocessing": "Gộp từ payment_method, sau đó one-hot.",
                "why_used": "Giữ tín hiệu thanh toán ở mức gọn hơn, tránh nhiều nhãn payment_method nhỏ lẻ.",
            },
            {
                "feature": "is_peak_hour",
                "role": "Có phải giờ cao điểm.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Biến vận hành dễ hiểu, hỗ trợ DSS giải thích áp lực bếp/tài xế.",
            },
            {
                "feature": "is_weekend",
                "role": "Có phải cuối tuần.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Bắt thay đổi nhu cầu cuối tuần, nhưng chỉ giữ vai trò tín hiệu phụ.",
            },
            {
                "feature": "order_month",
                "role": "Tháng đặt hàng.",
                "preprocessing": "One-hot encoding.",
                "why_used": "Bắt yếu tố mùa vụ ở mức đơn giản, phù hợp dữ liệu nhỏ.",
            },
        ]
    )


def model_analysis_table():
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "role": "Dự báo xác suất bằng ranh giới tuyến tính, có cân bằng lớp khi huấn luyện.",
                "strength": "F2 và MCC cao nhất; có xác suất rõ ràng; dễ đưa vào dashboard.",
                "limitation": "Nếu dữ liệu thật có quan hệ quá phức tạp thì mô hình tuyến tính có thể kém hơn.",
                "model_decision": "Chọn làm mô hình khóa.",
            },
            {
                "model": "SVM",
                "role": "Mô hình biên tối đa, dùng kernel RBF và hiệu chuẩn xác suất.",
                "strength": "Accuracy cao và MCC gần Logistic Regression; xử lý được biên phi tuyến.",
                "limitation": "Recall thấp hơn Logistic Regression, khó giải thích hơn và phải hiệu chuẩn thêm để lấy xác suất.",
                "model_decision": "Không chọn vì mục tiêu chính là bắt trễ theo F2/Recall.",
            },
            {
                "model": "Decision Tree",
                "role": "Cây luật if-else dễ đọc.",
                "strength": "Dễ trình bày thành các luật if-else; Recall khá.",
                "limitation": "Dễ overfit trên dữ liệu nhỏ; MCC thấp hơn hai mô hình đầu.",
                "model_decision": "Dùng để so sánh, không khóa.",
            },
            {
                "model": "Random Forest",
                "role": "Tập hợp nhiều cây để giảm overfit từng cây đơn lẻ.",
                "strength": "Precision rất cao, ít báo trễ nhầm.",
                "limitation": "Recall thấp hơn, bỏ sót nhiều đơn trễ hơn nên F2 giảm.",
                "model_decision": "Không phù hợp mục tiêu ưu tiên bắt trễ.",
            },
            {
                "model": "k-NN",
                "role": "Dự đoán theo các đơn gần nhất trong không gian feature.",
                "strength": "Đơn giản, không cần giả định dạng hàm.",
                "limitation": "Nhạy với thang đo/khoảng cách sau one-hot; Recall và F2 thấp.",
                "model_decision": "Không chọn.",
            },
            {
                "model": "Naive Bayes",
                "role": "Baseline xác suất dựa trên giả định feature độc lập.",
                "strength": "Recall khá cao, chạy nhanh, làm mốc so sánh tốt.",
                "limitation": "Precision và MCC thấp vì giả định độc lập không hợp dữ liệu có nhiều feature liên quan chặt.",
                "model_decision": "Không chọn, nhưng giữ để chứng minh đã thử đủ nhóm mô hình.",
            },
        ]
    )


def risk_score_source_table():
    return pd.DataFrame(
        [
            {
                "component": "model_probability",
                "weight": 0.55,
                "score_rule": "delayed_probability x 100",
                "number_origin": "Xác suất lấy từ mô hình Logistic Regression đã khóa.",
                "source": "Mô hình học máy",
            },
            {
                "component": "traffic_pressure",
                "weight": 0.15,
                "score_rule": "Low=20, Medium=60, High=100",
                "number_origin": "Nhóm đặt thang 20/60/100 theo thứ tự rủi ro trong EDA; High có tỷ lệ trễ cao nhất.",
                "source": "EDA + quy tắc nhóm đặt",
            },
            {
                "component": "distance_pressure",
                "weight": 0.12,
                "score_rule": "min(distance_km / 10 x 100, 100)",
                "number_origin": "10 km là cự ly lớn nhất trong dataset, dùng để chuẩn hóa về thang 0-100.",
                "source": "Min/max dữ liệu",
            },
            {
                "component": "peak_pressure",
                "weight": 0.08,
                "score_rule": "Giờ cao điểm=100, không cao điểm=20",
                "number_origin": "Cờ nhị phân; giờ cao điểm có ảnh hưởng nhưng thấp hơn traffic và cự ly.",
                "source": "Feature vận hành + quy tắc nhóm đặt",
            },
            {
                "component": "complexity_pressure",
                "weight": 0.06,
                "score_rule": "min(pizza_complexity / 20 x 100, 100)",
                "number_origin": "20 là độ phức tạp tối đa quan sát được: 5 topping x size_score XL=4.",
                "source": "Công thức FE + min/max dữ liệu",
            },
            {
                "component": "weekend_pressure",
                "weight": 0.04,
                "score_rule": "Cuối tuần=70, ngày thường=40",
                "number_origin": "Tín hiệu cuối tuần yếu hơn nên chỉ cho điểm vừa và trọng số nhỏ.",
                "source": "EDA + quy tắc nhóm đặt",
            },
        ]
    )


# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="Hệ hỗ trợ quyết định giao Pizza - Nhóm 29",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Look (HUST Red theme accents)
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .main-title {
        color: #d2143a;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #d2143a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        min-height: 88px;
    }
    .insight-box {
        background-color: #f0f7f9;
        border-left: 5px solid #0e7490;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    .section-header {
        color: #1e293b;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .tiny-note {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề hệ thống
st.markdown("<h1 class='main-title'>🍕 Hệ hỗ trợ quyết định điều phối giao Pizza</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Chấm điểm rủi ro trễ đơn, xếp hàng đợi ưu tiên, gợi ý hành động và phân công tài xế — <b>Nhóm 29</b></p>", unsafe_allow_html=True)

# Định nghĩa các Tab lớn
tabs = st.tabs(
    [
        "🖥️ Vận hành điều phối",
        "📈 Kinh doanh & Dự báo",
        "🕵️ Kiểm định dữ liệu",
        "🧠 Mô hình & Đánh giá",
    ]
)

# =====================================================================
# TAB 1: DISPATCHER CONSOLE (OPERATIONAL DSS)
# =====================================================================
with tabs[0]:
    st.markdown("<h3 class='section-header'>Bảng Điều Phối Đơn Hàng Giao Pizza Trực Tuyến</h3>", unsafe_allow_html=True)
    
    df_raw = load_dataset()
    queue_data = load_dashboard_data()
    if "recommended_action" in queue_data.columns:
        queue_data["recommended_action"] = queue_data["recommended_action"].map(translate_action)
    
    # Bộ lọc trên Sidebar để lọc danh sách hàng đợi giao hàng
    with st.sidebar:
        st.header("⚙️ Bộ lọc vận hành")
        prio_filter = st.selectbox(
            "Mức ưu tiên",
            ["All", "High", "Medium", "Low"],
            format_func=lambda x: "Tất cả" if x == "All" else VN_PRIORITY.get(x, x),
        )
        traffic_filter = st.selectbox(
            "Mật độ giao thông",
            ["All"] + sorted(queue_data["traffic_level"].unique()),
            format_func=lambda x: "Tất cả" if x == "All" else VN_TRAFFIC.get(x, x),
        )
        
    # Áp dụng bộ lọc
    filtered_queue = queue_data.copy()
    if prio_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["priority"] == prio_filter]
    if traffic_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["traffic_level"] == traffic_filter]
        
    # Hiển thị Metric vận hành
    m_cols = st.columns(4)
    with m_cols[0]:
        st.markdown(f"<div class='metric-card'><b>Tổng đơn trong hàng đợi</b><br><span style='font-size:1.8rem; font-weight:700; color:#d2143a;'>{len(filtered_queue)} đơn</span></div>", unsafe_allow_html=True)
    with m_cols[1]:
        high_count = (filtered_queue["priority"] == "High").sum()
        st.markdown(f"<div class='metric-card'><b>Đơn hàng nguy cơ CAO</b><br><span style='font-size:1.8rem; font-weight:700; color:#d97706;'>{high_count} đơn</span></div>", unsafe_allow_html=True)
    with m_cols[2]:
        mean_risk = filtered_queue["delay_risk_score"].mean()
        st.markdown(f"<div class='metric-card'><b>Điểm rủi ro trung bình</b><br><span style='font-size:1.8rem; font-weight:700; color:#0e7490;'>{mean_risk:.1f} / 100</span></div>", unsafe_allow_html=True)
    with m_cols[3]:
        delayed_rate = df_raw["is_delayed"].mean() * 100
        st.markdown(f"<div class='metric-card'><b>Tỉ lệ trễ lịch sử (Baseline)</b><br><span style='font-size:1.8rem; font-weight:700; color:#64748b;'>{delayed_rate:.1f}%</span></div>", unsafe_allow_html=True)
        
    st.write("")
    
    # Layout 2 cột: Trái là hàng đợi đơn hàng, Phải là Chi tiết đơn được click
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.subheader("📋 Hàng đợi ưu tiên giao hàng")
        st.caption("Các đơn đang chờ giao được sắp xếp tự động theo Điểm rủi ro trễ. Bấm chọn mã đơn ở cột phải để xem chi tiết & hành động gợi ý.")

        # Chỉ giữ các cột cần cho người điều phối (bỏ cột hành động dài — đã có ở panel chi tiết)
        display_queue = filtered_queue[[
            "order_id", "restaurant_name", "location", "distance_km",
            "traffic_level", "delayed_probability", "delay_risk_score", "priority"
        ]].copy()

        # Định dạng dễ đọc; hàm vn() sẽ Việt hoá giá trị High/Medium/Low.
        display_queue["delayed_probability"] = display_queue["delayed_probability"].map(lambda x: f"{x:.1%}")
        display_queue["delay_risk_score"] = display_queue["delay_risk_score"].round(1)

        st.dataframe(vn(display_queue), hide_index=True, width="stretch", height=400)

        export_queue = only_columns(filtered_queue, [
            "order_id", "restaurant_name", "location", "distance_km", "traffic_level",
            "delayed_probability", "delay_risk_score", "priority", "recommended_action"
        ])
        st.download_button(
            "📥 Tải xuống danh sách hàng đợi đã Việt hoá",
            vn(export_queue).to_csv(index=False).encode("utf-8-sig"),
            "hang_doi_rui_ro_giao_pizza.csv",
            "text/csv",
        )
        
        # Nhận xét nghiệp vụ dưới bảng
        st.markdown("""
        <div class='insight-box'>
            <b>💡 Nhận xét vận hành hàng đợi:</b>
            <ul>
                <li>Hàng đợi sắp xếp theo mức ưu tiên <b>Cao &rarr; Vừa &rarr; Thấp</b> để điều phối viên xử lý tuần tự.</li>
                <li>Đơn <b>ưu tiên Cao (Điểm rủi ro &gt; 65)</b> cần hành động ngay: chuẩn bị tài xế dự phòng hoặc kích hoạt gói giao hỏa tốc.</li>
                <li>Đơn <b>ưu tiên Vừa (Điểm rủi ro 35–65)</b> cần theo dõi sát thời gian làm bánh tại bếp.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with right_col:
        st.subheader("🔍 Chi tiết & Giải thích rủi ro đơn hàng")
        
        # Cho phép chọn đơn hàng từ danh sách để kiểm tra chi tiết
        order_list = filtered_queue["order_id"].tolist()
        if order_list:
            selected_order_id = st.selectbox("Chọn mã đơn để kiểm tra:", order_list)
            order_info = filtered_queue[filtered_queue["order_id"] == selected_order_id].iloc[0]
            
            # Khung thông tin quyết định
            prob = order_info["delayed_probability"]
            score = order_info["delay_risk_score"]
            priority = order_info["priority"]
            priority_label = VN_PRIORITY_DETAIL.get(priority, priority)
            action = translate_action(order_info["recommended_action"])
            
            if priority == "High":
                st.error(f"🔴 **MỨC ƯU TIÊN: {priority_label} (Rủi ro trễ: {prob:.1%})**")
                st.markdown(f"<div class='warning-box'><b>Hành động gợi ý:</b> {escape(action)}</div>", unsafe_allow_html=True)
            elif priority == "Medium":
                st.warning(f"🟡 **MỨC ƯU TIÊN: {priority_label} (Rủi ro trễ: {prob:.1%})**")
                st.markdown(f"<div class='warning-box'><b>Hành động gợi ý:</b> {escape(action)}</div>", unsafe_allow_html=True)
            else:
                st.success(f"🟢 **MỨC ƯU TIÊN: {priority_label} (Rủi ro trễ: {prob:.1%})**")
                st.markdown(f"<div class='success-box'><b>Hành động gợi ý:</b> {escape(action)}</div>", unsafe_allow_html=True)
                
            # Tạo dữ liệu giả lập cho explain_delay_risk_score
            # Để giải thích ta lấy dòng dữ liệu tương ứng
            raw_order = df_raw[df_raw["order_id"] == selected_order_id].iloc[0]
            # Tính toán breakdown
            explanation = pd.DataFrame(explain_delay_risk_score(raw_order, prob))
            explanation_chart = explanation.copy()
            explanation_chart["component_label"] = explanation_chart["component"].map(VN_COMPONENT).fillna(explanation_chart["component"])
            
            st.plotly_chart(
                px.bar(
                    explanation_chart.sort_values("weighted_contribution"),
                    x="weighted_contribution",
                    y="component_label",
                    orientation="h",
                    title=f"Đóng góp vào Điểm rủi ro ({score:.1f} điểm)",
                    labels={"weighted_contribution": "Điểm đóng góp sau trọng số", "component_label": "Thành phần rủi ro"},
                    color="weighted_contribution",
                    color_continuous_scale="Reds"
                ),
                use_container_width=True
            )
            
            st.dataframe(
                vn(explanation[["component", "weighted_contribution", "rationale"]]),
                hide_index=True,
                width="stretch"
            )
            with st.expander("Cách đọc bảng giải thích rủi ro"):
                st.markdown(
                    "- **Xác suất từ mô hình** là bằng chứng thống kê chính, chiếm 55% điểm rủi ro.\n"
                    "- **Giao thông, cự ly, giờ cao điểm, độ phức tạp và cuối tuần** là áp lực vận hành bổ sung.\n"
                    "- Điểm rủi ro càng cao thì đơn càng cần được đẩy lên đầu hàng đợi điều phối."
                )
            with st.expander("Nguồn gốc các số trong Risk Score"):
                st.dataframe(vn(risk_score_source_table()), hide_index=True, width="stretch")
                st.markdown(
                    "<div class='tiny-note'>"
                    "Các trọng số cộng bằng 1.0. Đây là bộ quy tắc nhóm tự đặt để dễ giải thích: mô hình chiếm hơn nửa điểm, "
                    "giao thông và cự ly đứng sau vì EDA cho thấy liên quan mạnh đến trễ; giờ cao điểm, độ phức tạp và cuối tuần có trọng số nhỏ hơn."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Không có đơn hàng nào khớp bộ lọc.")
            
    st.markdown("---")
    
    # Phân công tài xế tối ưu
    st.subheader("🚚 Gợi ý phân công tài xế")
    st.caption("Dùng thuật toán Hungary để gán tài xế giả lập cho các đơn rủi ro cao sao cho tổng chi phí gán nhỏ nhất trong kịch bản demo.")
    
    assignments = solve_transport_assignment(queue=queue_data, top_n=12)
    
    t_cols = st.columns(3)
    t_cols[0].metric("Số đơn hàng được gán", len(assignments))
    t_cols[1].metric("Số tài xế/lượt giao được dùng", assignments["driver_slot"].nunique())
    t_cols[2].metric("Chi phí gán trung bình", f"{assignments['estimated_assignment_cost'].mean():.2f}")
    
    left_t, right_t = st.columns([3, 2])
    with left_t:
        st.write("**Bảng gán chi tiết tài xế cho đơn hàng nguy cơ cao:**")
        assign_view = assignments[[
            "order_id", "priority", "traffic_level", "distance_km",
            "driver_id", "driver_slot", "estimated_assignment_cost"
        ]].copy()
        assign_view["priority"] = assign_view["priority"].map(VN_PRIORITY).fillna(assign_view["priority"])
        assign_view["traffic_level"] = assign_view["traffic_level"].map(VN_TRAFFIC).fillna(assign_view["traffic_level"])
        st.dataframe(vn(assign_view), hide_index=True, width="stretch")
    with right_t:
        st.plotly_chart(
            px.bar(
                assignments.sort_values("estimated_assignment_cost"),
                x="estimated_assignment_cost",
                y="order_id",
                color="driver_id",
                orientation="h",
                title="Chi phí gán ước tính theo từng đơn hàng",
                labels={"estimated_assignment_cost": "Chi phí gán", "order_id": "Mã đơn hàng"}
            ),
            use_container_width=True
        )
        
    st.markdown("""
    <div class='insight-box'>
        <b>🚚 Nhận xét và logic phân công tài xế:</b>
        <ul>
            <li>Chi phí gán được tính từ: khoảng cách, mức rủi ro đơn, mức kẹt xe và điểm thưởng nếu cùng khu vực giao.</li>
            <li>Mục tiêu là ưu tiên đơn rủi ro cao trước, tránh để tài xế xử lý đơn ít rủi ro trong khi đơn dễ trễ bị chờ lâu.</li>
            <li><i>Lưu ý:</i> Đội tài xế là giả lập vì dữ liệu Kaggle không có bảng tài xế thật.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Mô phỏng đơn hàng mới ở dưới cùng
    st.markdown("---")
    st.subheader("🆕 Mô phỏng rủi ro đơn hàng mới")
    st.caption("Nhập các thông số của một đơn hàng chuẩn bị tiếp nhận để kiểm tra mức độ rủi ro trễ và hành động khuyến nghị.")
    
    sim_model = load_best_model()
    
    sim_cols = st.columns(3)
    values = {}
    with sim_cols[0]:
        values["restaurant_name"] = st.selectbox("Nhà hàng", sorted(df_raw["restaurant_name"].unique()), key="sim_res")
        values["location"] = st.selectbox("Khu vực giao", sorted(df_raw["location"].unique()), key="sim_loc")
        values["pizza_size"] = st.selectbox("Cỡ bánh", sorted(df_raw["pizza_size"].unique()), format_func=lambda x: VN_SIZE.get(x, x), key="sim_size")
        values["pizza_type"] = st.selectbox("Loại bánh", sorted(df_raw["pizza_type"].unique()), format_func=lambda x: VN_PIZZA_TYPE.get(x, x), key="sim_type")
    with sim_cols[1]:
        values["traffic_level"] = st.selectbox("Mật độ giao thông", ["Low", "Medium", "High"], index=1, format_func=lambda x: VN_TRAFFIC.get(x, x), key="sim_traffic")
        values["payment_method"] = st.selectbox("Phương thức thanh toán", sorted(df_raw["payment_method"].unique()), format_func=lambda x: VN_PAYMENT.get(x, x), key="sim_pay")
        values["payment_category"] = st.selectbox("Nhóm thanh toán", sorted(df_raw["payment_category"].unique()), format_func=lambda x: VN_PAYMENT_CATEGORY.get(x, x), key="sim_pay_cat")
        values["order_month"] = st.selectbox("Tháng đặt hàng", sorted(df_raw["order_month"].unique()), key="sim_month")
    with sim_cols[2]:
        values["distance_km"] = st.slider("Cự ly giao hàng (km)", 0.5, 10.0, 4.0, 0.1, key="sim_dist")
        values["toppings_count"] = st.slider("Số lượng topping", 1, 5, 3, key="sim_toppings")
        values["order_hour"] = st.slider("Giờ đặt hàng", 0, 23, 19, key="sim_hour")
        values["is_peak_hour"] = st.checkbox("Giờ cao điểm", value=True, key="sim_peak")
        values["is_weekend"] = st.checkbox("Ngày cuối tuần", value=False, key="sim_weekend")
        
    # Tính các biến phái sinh tất định cho phần mô phỏng
    values["estimated_duration_min"] = values["distance_km"] * 2.4
    values["traffic_impact"] = {"Low": 1, "Medium": 2, "High": 3}[values["traffic_level"]]
    values["pizza_size_score"] = {"Small": 1, "Medium": 2, "Large": 3, "XL": 4}.get(values["pizza_size"], 2)
    values["pizza_complexity"] = values["toppings_count"] * values["pizza_size_score"]
    values["topping_density"] = values["toppings_count"] / max(values["distance_km"], 0.1)

    sim_df = make_single_order_frame(values)
    sim_prob = predict_delay_probability(sim_model, sim_df)[0]
    sim_decision = get_dss_decision(sim_df.iloc[0], sim_prob)
    
    sim_res_cols = st.columns(3)
    sim_res_cols[0].metric("Xác suất trễ dự báo", f"{sim_decision['delayed_probability']:.1%}")
    sim_res_cols[1].metric("Điểm rủi ro tích hợp", f"{sim_decision['delay_risk_score']:.1f}")
    sim_res_cols[2].metric("Mức ưu tiên đề xuất", VN_PRIORITY_DETAIL.get(sim_decision["priority"], sim_decision["priority"]))
    
    st.info(f"👉 **Hành động gợi ý:** {translate_action(sim_decision['recommended_action'])}")

# =====================================================================
# TAB 2: BUSINESS & FORECAST INSIGHTS (STRATEGIC DSS)
# =====================================================================
with tabs[1]:
    st.markdown("<h3 class='section-header'>Dự báo Nhu cầu & Hành vi Khách hàng (Chiến lược)</h3>", unsafe_allow_html=True)
    st.caption("Hỗ trợ Ban quản trị lập kế hoạch nhân sự và chuẩn bị nguyên vật liệu.")
    
    left_f, right_f = st.columns(2)
    
    with left_f:
        st.subheader("📅 Dự báo nhu cầu đơn hàng hàng tháng")
        forecast = forecast_monthly_demand(df_raw)
        f_metrics = forecast_metrics(forecast)
        
        # Biểu đồ demand forecast
        fig_demand = px.line(
            forecast,
            x="order_period",
            y=["actual_orders", "forecast_orders"],
            markers=True,
            title="Sản lượng đơn hàng thực tế vs. Dự báo (Seasonal-Naive)",
            labels={"order_period": "Tháng/Thời kỳ", "value": "Số lượng đơn", "variable": "Loại"}
        )
        st.plotly_chart(fig_demand, use_container_width=True)
        
        # Chỉ số chất lượng forecast
        f_cols = st.columns(3)
        f_cols[0].metric("Số tháng đánh giá", f_metrics["backtest_rows"])
        f_cols[1].metric("Sai số MAE", f"{f_metrics['mae']:.1f}")
        f_cols[2].metric("Chỉ số MAPE", f"{f_metrics['mape']:.1f}%")
        
        st.markdown("""
        <div class='insight-box'>
            <b>📊 Phân tích Dự báo nhu cầu:</b>
            <ul>
                <li>Sai số dự báo <b>MAPE ở mức 45.7%</b> là khá cao, do chuỗi thời gian ngắn và dữ liệu sinh (synthetic) có nhiễu cao.</li>
                <li><i>Khuyến nghị chiến lược:</i> Chỉ sử dụng mô hình dự báo này cho việc ước lượng khung nguyên vật liệu cơ bản, không nên dùng để đặt mục tiêu KPI doanh thu chặt chẽ.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with right_f:
        st.subheader("⏰ Kịch bản phân bổ nhân sự theo giờ (Staffing Scenario)")
        staffing = hourly_staffing_plan(df_raw)
        
        fig_staffing = px.bar(
            staffing, 
            x="order_hour", 
            y="scenario_orders_per_day",
            title="Phân bổ số đơn hàng kỳ vọng theo giờ trong ngày (Quy mô 100 đơn/ngày)",
            labels={"order_hour": "Giờ trong ngày", "scenario_orders_per_day": "Số đơn kỳ vọng"}
        )
        st.plotly_chart(fig_staffing, use_container_width=True)
        
        st.markdown("""
        <div class='insight-box'>
            <b>⏰ Phân tích nhân sự giờ cao điểm:</b>
            <ul>
                <li>Khung giờ cao điểm lớn nhất là <b>18h - 20h</b> (Đỉnh điểm lúc 19h đạt ~10% tổng đơn trong ngày).</li>
                <li><i>Khuyến nghị vận hành:</i> Bố trí gấp đôi số lượng nhân sự làm bánh và tài xế trực ca từ 17h30 đến 20h30 để tránh ùn ứ đơn tại bếp.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Khám phá khách hàng
    st.subheader("🍕 Khám phá hành vi khách hàng & Quy tắc gợi ý sản phẩm")
    prefs = customer_preference_tables(df_raw)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cơ cấu cỡ bánh Pizza:**")
        st.dataframe(vn(prefs["size_mix"]), hide_index=True, width="stretch")
        st.plotly_chart(
            px.pie(prefs["size_mix"], values="order_share", names="pizza_size_label", title="Tỷ lệ đặt hàng theo cỡ bánh"),
            use_container_width=True
        )
    with col2:
        st.write("**Cơ cấu loại nhân Pizza:**")
        st.dataframe(vn(prefs["type_mix"]), hide_index=True, width="stretch")
        st.plotly_chart(
            px.pie(prefs["type_mix"], values="order_share", names="pizza_type", title="Tỷ lệ đặt hàng theo loại nhân bánh"),
            use_container_width=True
        )
        
    st.subheader("💡 Luật kết hợp đề xuất sản phẩm dựa trên Context")
    rules_view = only_columns(
        recommendation_rules(df_raw).head(15),
        ["rule_type", "context", "recommendation", "context_orders", "matched_orders", "confidence"],
    )
    st.dataframe(vn(rules_view), hide_index=True, width="stretch")
    
    st.markdown("""
    <div class='insight-box'>
        <b>💡 Phân tích hành vi & quy tắc đề xuất:</b>
        <ul>
            <li>Cỡ bánh được yêu thích nhất là <b>Medium (chiếm ~39.4%)</b>, tiếp theo là Large (~30.1%).</li>
            <li>Pizza nhân <b>Non-Veg (Thịt) dẫn đầu tuyệt đối với ~69.6%</b> lượng đơn hàng.</li>
            <li><i>Quy tắc đề xuất sản phẩm:</i> Dựa trên tần suất bán chạy theo từng khung giờ và địa điểm. Ví dụ: cuối tuần hoặc giờ cao điểm thì gợi ý combo cỡ Medium kèm topping bán chạy ở khu vực đó.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# TAB 3: DATA FORENSICS & QUALITY (TRUTH ENGINE)
# =====================================================================
with tabs[2]:
    st.markdown("<h3 class='section-header'>Kiểm tra chất lượng và nguồn gốc dữ liệu</h3>", unsafe_allow_html=True)
    st.caption("Các kiểm tra cho thấy bộ dữ liệu có nhiều dấu hiệu được sinh tự động, nên cần thận trọng khi diễn giải.")
    
    # 1. Realism Audit
    st.subheader("🔍 Kiểm tra tính tự nhiên của dữ liệu")
    left_q, right_q = st.columns([3, 2])
    with left_q:
        synthetic = synthetic_data_audit(df_raw)
        synthetic_view = only_columns(synthetic, ["check", "severity", "evidence", "interpretation"])
        st.dataframe(vn(synthetic_view), hide_index=True, width="stretch")
    with right_q:
        synthetic_counts = synthetic.copy()
        synthetic_counts["severity"] = synthetic_counts["severity"].map(VN_SEVERITY).fillna(synthetic_counts["severity"])
        fig_synthetic = px.bar(
            synthetic_counts["severity"].value_counts().reset_index(name="checks"),
            x="severity",
            y="checks",
            title="Số lỗi phát hiện theo mức độ nghiêm trọng",
            labels={"severity": "Mức độ nghiêm trọng", "checks": "Số lỗi"}
        )
        st.plotly_chart(fig_synthetic, use_container_width=True)
        
    st.markdown("""
    <div class='warning-box'>
        <b>🕵️ Nhận xét về dữ liệu:</b>
        <ul>
            <li>Kiểm định cho thấy <b>100% cột dữ liệu có dấu hiệu sinh lập toán học</b> (Không phải dữ liệu thực tế thu thập từ nhà hàng).</li>
            <li>Ví dụ điển hình: Các giá trị khoảng cách hình học <code>distance_km</code> phân bố đều một cách phi thực tế, và thời gian giao hàng <code>delivery_duration_min</code> chỉ nhận các giá trị là bội số của 5.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. 7 deterministic formulas & Threshold Inference
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("📐 7 Công thức tất định tìm thấy trong dữ liệu")
        st.caption("Nhóm đã truy vết và phục dựng thành công các công thức toán học ẩn sinh ra dữ liệu.")
        
        formula_path = METRICS_DIR / "generator_deterministic_formulas.csv"
        if formula_path.exists():
            formula_view = only_columns(
                pd.read_csv(formula_path),
                ["column", "recovered_formula", "max_abs_error", "matches_exactly"],
            )
            st.dataframe(vn(formula_view), hide_index=True, width="stretch")
        else:
            st.warning("Không tìm thấy file generator_deterministic_formulas.csv")
            
        fig_err_path = FIGURES_DIR / "generator_deterministic_formula_errors.png"
        if fig_err_path.exists():
            st.image(str(fig_err_path), caption="Sai số của công thức phục dựng chạm mức 10^-15 (tức khớp tuyệt đối)")
            
    with col_d2:
        st.subheader("⏱️ Suy luận ngưỡng trễ")
        st.caption("Làm thế nào để biết bao nhiêu phút giao hàng thì bị coi là 'trễ'?")
        
        threshold_path = METRICS_DIR / "delay_threshold_inference.csv"
        if threshold_path.exists():
            # Show top vài dòng
            threshold_view = only_columns(
                pd.read_csv(threshold_path).head(10),
                ["rule", "threshold_minutes", "mismatches", "mismatch_rate", "exact_match", "predicted_delayed", "actual_delayed"],
            )
            threshold_view = format_probability_columns(threshold_view, ["mismatch_rate"])
            st.dataframe(vn(threshold_view), hide_index=True, width="stretch")
        else:
            st.warning("Không tìm thấy file delay_threshold_inference.csv")
            
        fig_thresh_path = FIGURES_DIR / "delay_threshold_inference.png"
        if fig_thresh_path.exists():
            st.image(str(fig_thresh_path), caption="Sai lệch nhãn chạm 0 tại ranh giới 30-35 phút")
            
    st.markdown("""
    <div class='insight-box'>
        <b>🕵️ Bài học rút ra:</b>
        <ul>
            <li><b>Sai số phục dựng công thức cực nhỏ (10^-15):</b> Chứng minh tuyệt đối dữ liệu sinh bằng thuật toán. Ví dụ: Thời gian giao hàng dự tính <code>estimated_duration_min</code> luôn bằng đúng <code>distance_km * 2.4</code>.</li>
            <li><b>Ngưỡng nhãn trễ:</b> Nhãn trễ <code>is_delayed</code> không được định nghĩa trước, nhưng dò ngưỡng cho thấy quy tắc gán nhãn của bộ sinh là: nếu thời gian thực tế <code>delivery_duration_min > 30</code> phút thì gán nhãn 1 (Trễ), ngược lại gán 0. Điều này giải thích ranh giới nhãn.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Brand Uniformity & Ablation
    st.subheader("🏢 Phân tích tên nhà hàng: tín hiệu thật hay do bộ sinh tạo?")
    st.caption("Kiểm tra xem các chuỗi cửa hàng Pizza khác nhau có thực sự có tỷ lệ trễ khác nhau không, hay chỉ là do nhiễu ngẫu nhiên trong bộ sinh.")
    
    brand_cols = st.columns(2)
    with brand_cols[0]:
        brand_test_path = METRICS_DIR / "brand_homogeneity_tests.csv"
        if brand_test_path.exists():
            st.dataframe(pd.read_csv(brand_test_path), hide_index=True, width="stretch")
        else:
            st.write("Không tìm thấy brand_homogeneity_tests.csv")
            
        fig_brand_path = FIGURES_DIR / "brand_delay_rate_homogeneity.png"
        if fig_brand_path.exists():
            st.image(str(fig_brand_path), caption="Biểu đồ tỷ lệ trễ chồng lấp giữa các Brand")
            
    with brand_cols[1]:
        brand_ablation_path = METRICS_DIR / "brand_ablation.csv"
        if brand_ablation_path.exists():
            st.dataframe(pd.read_csv(brand_ablation_path), hide_index=True, width="stretch")
        else:
            st.write("Không tìm thấy brand_ablation.csv")
            
    st.markdown("""
    <div class='insight-box'>
        <b>🏢 Thử bỏ biến tên nhà hàng:</b>
        <ul>
            <li>Kiểm định Chi-square cho thấy <b>không có sự khác biệt có ý nghĩa thống kê</b> về tỷ lệ trễ giữa các Brand (P-value > 0.05).</li>
            <li><b>Thử bỏ biến tên nhà hàng:</b> Huấn luyện khi có và không có biến này cho thấy F2 gần như không đổi.</li>
            <li><i>Kết luận:</i> Tên nhà hàng (Brand) chỉ là một thuộc tính gán nhãn ngẫu nhiên trong dữ liệu giả lập, không đại diện cho chất lượng vận hành thật của từng thương hiệu.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# TAB 4: ML MODEL REGISTRY & BENCHMARKS (TECHNICAL CORE)
# =====================================================================
with tabs[3]:
    st.markdown("<h3 class='section-header'>Đánh giá mô hình học máy</h3>", unsafe_allow_html=True)
    st.caption("Tập trung vào câu hỏi: mô hình nào bắt đơn trễ tốt, vì sao chọn mô hình đó, và feature được xử lý như thế nào.")
    
    dev_path = METRICS_DIR / "model_dev_comparison.csv"
    test_path = METRICS_DIR / "model_test_metrics.csv"
    baseline_path = METRICS_DIR / "baseline_test_metrics.csv"
    tuning_path = METRICS_DIR / "default_vs_tuned_lr.csv"
    stability_summary_path = METRICS_DIR / "model_stability_summary.csv"
    stability_figure_path = FIGURES_DIR / "model_stability_f2_distribution.png"
    
    if not dev_path.exists() or not test_path.exists():
        st.warning("Vui lòng chạy `python -m scripts.run_all` trước để sinh các chỉ số kiểm thử.")
    else:
        st.subheader("🧾 Dữ liệu dùng cho mô hình")
        data_cols = st.columns(4)
        delayed_orders = int(df_raw["is_delayed"].sum())
        on_time_orders = int((~df_raw["is_delayed"]).sum())
        data_cols[0].metric("Tổng số đơn", f"{len(df_raw):,}".replace(",", "."))
        data_cols[1].metric("Đơn trễ", delayed_orders)
        data_cols[2].metric("Đơn đúng giờ", on_time_orders)
        data_cols[3].metric("Feature đang dùng", len(ACTIVE_FEATURE_COLUMNS))

        st.markdown("""
        <div class='warning-box'>
            <b>Lưu ý chống rò rỉ dữ liệu:</b>
            <code>delivery_time</code>, <code>delivery_duration_min</code>, <code>delay_min</code>
            và các biến sau giao hàng không được đưa vào mô hình. Dù có thể phục dựng
            <code>delivery_duration_min - delay_min = estimated_duration_min</code>, phép tính này vẫn dùng thông tin chỉ biết sau khi giao xong.
            Vì vậy nó chỉ dùng để kiểm định dữ liệu, không dùng làm feature dự báo lúc nhận đơn.
        </div>
        """, unsafe_allow_html=True)

        st.write("**Các chỉ số đánh giá dùng trong báo cáo:**")
        st.dataframe(vn(metric_definition_table()), hide_index=True, width="stretch")

        with st.expander("Xem 12 feature đang dùng và vai trò của từng feature", expanded=True):
            feature_view = feature_role_table()
            feature_view = feature_view[feature_view["feature"].isin(ACTIVE_FEATURE_COLUMNS)]
            st.dataframe(vn(feature_view), hide_index=True, width="stretch")
            st.markdown(
                "<div class='tiny-note'>"
                "Feature engineering chính: mã hoá cỡ bánh thành điểm thứ bậc, chuẩn hoá biến số, one-hot biến phân loại, "
                "gộp nhóm thanh toán và loại bỏ biến rò rỉ/trùng công thức để mô hình gọn hơn."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 1. So sánh mô hình trên tập Dev
        st.subheader("📊 Bảng 1: Tổng hợp so sánh 6 mô hình trên tập phát triển")
        left_m, right_m = st.columns([3, 2])
        
        with left_m:
            dev_df = pd.read_csv(dev_path)
            dev_view = only_columns(dev_df, ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f2", "mcc"])
            dev_view = format_score_columns(dev_view, ["accuracy", "balanced_accuracy", "precision", "recall", "f2", "mcc"])
            st.dataframe(vn(dev_view), hide_index=True, width="stretch")
        with right_m:
            # Vẽ biểu đồ F2 của các model
            dev_chart = dev_df.copy()
            dev_chart["model_label"] = dev_chart["model"].map(VN_MODEL).fillna(dev_chart["model"])
            fig_compare = px.bar(
                dev_chart,
                x="model_label",
                y="f2",
                title="F2 của từng mô hình (càng cao càng bắt trễ tốt)",
                labels={"model_label": "Mô hình", "f2": "F2 - ưu tiên Recall"},
                color="f2",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
        st.markdown("""
        <div class='insight-box'>
            <b>📊 Phân tích so sánh mô hình:</b>
            <ul>
                <li><b>Hồi quy Logistic</b> đứng đầu trên tập phát triển với F2 = 0.9434 và MCC = 0.9117, nên được chọn làm mô hình khóa.</li>
                <li><b>F2</b> được ưu tiên vì bài toán cần bắt sớm đơn trễ; bỏ sót đơn trễ tốn kém hơn báo động nhầm.</li>
                <li><b>Naive Bayes</b> có MCC thấp vì giả định các feature độc lập không hợp dữ liệu: cự ly, traffic, độ phức tạp và các biến dẫn xuất có quan hệ chặt.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.write("**Phân tích đồng đều từng mô hình:**")
        st.dataframe(vn(model_analysis_table()), hide_index=True, width="stretch")
        
        st.markdown("---")
        
        # 2. Kết quả kiểm thử khóa & Baseline
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("🎯 Hiệu suất mô hình khóa trên tập kiểm thử")
            test_view = only_columns(pd.read_csv(test_path), ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "mcc"])
            test_view = format_score_columns(test_view, ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "mcc"])
            st.dataframe(vn(test_view), hide_index=True, width="stretch")
            
            fig_cm_path = FIGURES_DIR / "confusion_matrix_test.png"
            if fig_cm_path.exists():
                st.image(str(fig_cm_path), caption="Confusion Matrix trên tập Test (Chỉ bỏ sót 1 đơn trễ thực tế)")
        with col_t2:
            st.subheader("🏁 So sánh với các mô hình đoán ngây thơ (Baseline)")
            baseline_view = only_columns(pd.read_csv(baseline_path), ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f2", "mcc"])
            baseline_view = format_score_columns(baseline_view, ["accuracy", "balanced_accuracy", "precision", "recall", "f2", "mcc"])
            st.dataframe(vn(baseline_view), hide_index=True, width="stretch")
            
        st.markdown("""
        <div class='insight-box'>
            <b>🏁 Ý nghĩa so sánh Baseline:</b>
            <ul>
                <li><b>Baseline Always-on-Time:</b> Đoán tất cả đơn đều đúng giờ. Đạt Accuracy 79.1% nhưng <b>F2 = 0%</b> và <b>Recall = 0%</b> (Không phát hiện được bất kỳ đơn trễ nào, hoàn toàn vô dụng cho DSS).</li>
                <li><b>Baseline Always-Delayed:</b> Đoán tất cả đơn đều trễ. Đạt Recall 100% nhưng báo động giả tràn lan (Precision 20.9%), gây quá tải nghiêm trọng cho vận hành.</li>
                <li>Mô hình <b>Logistic Regression của nhóm đạt F2 ~94.9%</b>, cân bằng cực tốt giữa việc phát hiện sớm đơn trễ và hạn chế báo động giả.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 3. Tuning & Stability
        st.subheader("🛡️ Bảng 2: So sánh fine-tune Logistic Regression và kiểm tra độ ổn định")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            if stability_summary_path.exists():
                st.write("**Thống kê hiệu suất qua 100 lần chia tập ngẫu nhiên:**")
                stability_view = only_columns(pd.read_csv(stability_summary_path), ["metric", "mean", "std", "p05", "p50", "p95"])
                stability_view = format_score_columns(stability_view, ["mean", "std", "p05", "p50", "p95"])
                st.dataframe(vn(stability_view), hide_index=True, width="stretch")
            if stability_figure_path.exists():
                st.image(str(stability_figure_path), caption="Phân phối điểm F2 qua 100 lần chạy ngẫu nhiên")
                
        with col_s2:
            if tuning_path.exists():
                st.write("**Kết quả so sánh mô hình mặc định và mô hình sau fine-tune:**")
                tuning_view = only_columns(pd.read_csv(tuning_path), [
                    "model", "param_C", "cv_f2_mean", "f2", "mcc", "delta_dev_f2_vs_default", "decision"
                ])
                tuning_view = format_score_columns(tuning_view, ["cv_f2_mean", "f2", "mcc", "delta_dev_f2_vs_default"])
                st.dataframe(vn(tuning_view), hide_index=True, width="stretch")
                
        st.markdown("""
        <div class='insight-box'>
            <b>🛡️ Đánh giá độ ổn định và tinh chỉnh:</b>
            <ul>
                <li><b>Chạy lại 100 lần chia tập:</b> F2 trung bình đạt 0.941, độ lệch chuẩn 0.015. Kết quả không phụ thuộc vào một lần chia dữ liệu duy nhất.</li>
                <li><b>Quyết định fine-tune:</b> CV chọn C=0.3 nhỉnh hơn nhẹ, nhưng khi kiểm tra lại trên dev thì F2 giảm 0.054 so với C=1.0. Vì vậy nhóm giữ cấu hình mặc định.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
