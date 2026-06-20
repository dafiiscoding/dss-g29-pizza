import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pizza_dss.config import FEATURE_COLUMNS, FIGURES_DIR, METRICS_DIR
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
from pizza_dss.transport_optimization import solve_transport_assignment, transport_cost_policy_spec

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="Pizza Delivery DSS - Group 29",
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
</style>
""", unsafe_allow_html=True)

# Tiêu đề hệ thống
st.markdown("<h1 class='main-title'>🍕 Hệ Hỗ Trợ Quyết Định Giao Pizza (Pizza Delivery DSS)</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống chấm điểm rủi ro trễ đơn hàng, ưu tiên điều phối và tối ưu hóa phân công tài xế — <b>Nhóm 29</b></p>", unsafe_allow_html=True)

# Định nghĩa các Tab lớn
tabs = st.tabs(
    [
        "🖥️ Dispatcher Console (Vận Hành)",
        "📈 Business & Forecast (Chiến Lược)",
        "🕵️ Data Forensics & Quality (Kiểm Định)",
        "🧠 ML Registry & Benchmarks (Kỹ Thuật)",
    ]
)

# =====================================================================
# TAB 1: DISPATCHER CONSOLE (OPERATIONAL DSS)
# =====================================================================
with tabs[0]:
    st.markdown("<h3 class='section-header'>Bảng Điều Phối Đơn Hàng Giao Pizza Trực Tuyến</h3>", unsafe_allow_html=True)
    
    df_raw = load_dataset()
    queue_data = load_dashboard_data()
    
    # Bộ lọc trên Sidebar để lọc danh sách hàng đợi giao hàng
    with st.sidebar:
        st.header("⚙️ Bộ lọc Vận Hành")
        prio_filter = st.selectbox("Mức ưu tiên (Priority)", ["All", "High", "Medium", "Low"])
        traffic_filter = st.selectbox("Mật độ giao thông (Traffic)", ["All"] + sorted(queue_data["traffic_level"].unique()))
        
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
        st.caption("Các đơn hàng đang chờ giao được sắp xếp tự động dựa trên Điểm Rủi ro Trễ (Delay Risk Score).")
        
        # Chọn các cột hiển thị đẹp mắt cho người điều phối
        display_queue = filtered_queue[[
            "order_id", "restaurant_name", "location", "distance_km", 
            "traffic_level", "delayed_probability", "delay_risk_score", 
            "priority", "recommended_action"
        ]].copy()
        
        # Định dạng phần trăm và làm tròn điểm rủi ro
        display_queue["delayed_probability"] = display_queue["delayed_probability"].map(lambda x: f"{x:.1%}")
        display_queue["delay_risk_score"] = display_queue["delay_risk_score"].round(1)
        
        st.dataframe(display_queue, hide_index=True, width="stretch", height=400)
        
        st.download_button(
            "📥 Tải xuống danh sách hàng đợi",
            filtered_queue.to_csv(index=False).encode("utf-8"),
            "pizza_delay_priority_queue.csv",
            "text/csv",
        )
        
        # Nhận xét nghiệp vụ dưới bảng
        st.markdown("""
        <div class='insight-box'>
            <b>💡 Nhận xét Vận Hành Hàng Đợi:</b>
            <ul>
                <li>Hàng đợi được sắp xếp ưu tiên theo <b>Priority (High -> Medium -> Low)</b> để nhân viên điều phối xử lý tuần tự.</li>
                <li>Đơn hàng có <b>Priority = High (Risk Score > 65)</b> yêu cầu hành động ngay: chuẩn bị tài xế dự phòng hoặc kích hoạt gói giao hỏa tốc.</li>
                <li>Đơn hàng có <b>Priority = Medium (Risk Score từ 35 đến 65)</b> cần theo dõi sát thời gian làm bánh tại bếp.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with right_col:
        st.subheader("🔍 Chi tiết & Giải thích rủi ro đơn hàng")
        
        # Cho phép chọn đơn hàng từ danh sách để kiểm tra chi tiết
        order_list = filtered_queue["order_id"].tolist()
        if order_list:
            selected_order_id = st.selectbox("Chọn Mã đơn hàng (Order ID) để kiểm tra:", order_list)
            order_info = filtered_queue[filtered_queue["order_id"] == selected_order_id].iloc[0]
            
            # Khung thông tin quyết định
            prob = order_info["delayed_probability"]
            score = order_info["delay_risk_score"]
            priority = order_info["priority"]
            action = order_info["recommended_action"]
            
            if priority == "High":
                st.error(f"🔴 **MỨC ƯU TIÊN: {priority} (Rủi ro trễ: {prob:.1%})**")
                st.markdown(f"<div class='warning-box'><b>Hành động gợi ý:</b> {action}</div>", unsafe_allow_html=True)
            elif priority == "Medium":
                st.warning(f"🟡 **MỨC ƯU TIÊN: {priority} (Rủi ro trễ: {prob:.1%})**")
                st.markdown(f"<div class='warning-box'><b>Hành động gợi ý:</b> {action}</div>", unsafe_allow_html=True)
            else:
                st.success(f"🟢 **MỨC ƯU TIÊN: {priority} (Rủi ro trễ: {prob:.1%})**")
                st.markdown(f"<div class='success-box'><b>Hành động gợi ý:</b> {action}</div>", unsafe_allow_html=True)
                
            # Tạo dữ liệu giả lập cho explain_delay_risk_score
            # Để giải thích ta lấy dòng dữ liệu tương ứng
            raw_order = df_raw[df_raw["order_id"] == selected_order_id].iloc[0]
            # Tính toán breakdown
            explanation = pd.DataFrame(explain_delay_risk_score(raw_order, prob))
            
            st.plotly_chart(
                px.bar(
                    explanation.sort_values("weighted_contribution"),
                    x="weighted_contribution",
                    y="component",
                    orientation="h",
                    title=f"Đóng góp vào Điểm rủi ro ({score:.1f} điểm)",
                    labels={"weighted_contribution": "Điểm đóng góp (trọng số)", "component": "Thành phần rủi ro"},
                    color="weighted_contribution",
                    color_continuous_scale="Reds"
                ),
                use_container_width=True
            )
            
            st.dataframe(
                explanation[["component", "component_score", "weight", "weighted_contribution", "rationale"]],
                hide_index=True,
                width="stretch"
            )
        else:
            st.info("Không có đơn hàng nào khớp bộ lọc.")
            
    st.markdown("---")
    
    # Phân công tài xế tối ưu
    st.subheader("🚚 Phân công tài xế tối ưu (Prescriptive Optimization)")
    st.caption("Ứng dụng bài toán phân công (Assignment Problem - thuật toán Hungary) để gán tài xế giả lập cho các đơn hàng High-Priority nhằm tối ưu tổng chi phí giao nhận.")
    
    assignments = solve_transport_assignment(queue=queue_data, top_n=12)
    
    t_cols = st.columns(3)
    t_cols[0].metric("Số đơn hàng được gán", len(assignments))
    t_cols[1].metric("Số tài xế/lượt giao được dùng", assignments["driver_slot"].nunique())
    t_cols[2].metric("Chi phí gán trung bình", f"{assignments['estimated_assignment_cost'].mean():.2f}")
    
    left_t, right_t = st.columns([3, 2])
    with left_t:
        st.write("**Bảng gán chi tiết tài xế cho đơn hàng nguy cơ cao:**")
        st.dataframe(assignments[[
            "order_id", "priority", "traffic_level", "distance_km", 
            "driver_id", "driver_slot", "estimated_assignment_cost"
        ]], hide_index=True, width="stretch")
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
            <li><b>Ma trận chi phí gán (Assignment Cost)</b> được tính tự động từ: Khoảng cách địa lý + Phạt rủi ro đơn hàng (Priority Penalty) + Phạt kẹt xe (Traffic Penalty) - Thưởng cùng địa điểm giao hàng (Same-location bonus).</li>
            <li>Thuật toán tối ưu hóa giúp giảm tổng quãng đường giao hàng của đội xe, tránh trường hợp tài xế giao đơn Low-priority trước trong khi đơn High-priority bị bỏ qua.</li>
            <li><i>Lưu ý:</i> Kịch bản này sử dụng thông tin đơn hàng thật từ Kaggle nhưng đội xe (tài xế) là giả lập do dữ liệu gốc không có bảng tài xế thực tế.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Simulator đơn hàng mới ở dưới cùng
    st.markdown("---")
    st.subheader("🆕 Mô phỏng rủi ro đơn hàng mới (Simulator)")
    st.caption("Nhập các thông số của một đơn hàng chuẩn bị tiếp nhận để kiểm tra mức độ rủi ro trễ và hành động khuyến nghị.")
    
    sim_model = load_best_model()
    
    sim_cols = st.columns(3)
    values = {}
    with sim_cols[0]:
        values["restaurant_name"] = st.selectbox("Nhà hàng (Restaurant)", sorted(df_raw["restaurant_name"].unique()), key="sim_res")
        values["location"] = st.selectbox("Khu vực (Location)", sorted(df_raw["location"].unique()), key="sim_loc")
        values["pizza_size"] = st.selectbox("Cỡ pizza (Pizza size)", sorted(df_raw["pizza_size"].unique()), key="sim_size")
        values["pizza_type"] = st.selectbox("Loại pizza (Pizza type)", sorted(df_raw["pizza_type"].unique()), key="sim_type")
    with sim_cols[1]:
        values["traffic_level"] = st.selectbox("Tình trạng giao thông (Traffic)", ["Low", "Medium", "High"], index=1, key="sim_traffic")
        values["payment_method"] = st.selectbox("Phương thức thanh toán", sorted(df_raw["payment_method"].unique()), key="sim_pay")
        values["payment_category"] = st.selectbox("Nhóm thanh toán", sorted(df_raw["payment_category"].unique()), key="sim_pay_cat")
        values["order_month"] = st.selectbox("Tháng đặt hàng", sorted(df_raw["order_month"].unique()), key="sim_month")
    with sim_cols[2]:
        values["distance_km"] = st.slider("Khoảng cách giao hàng (Distance km)", 0.5, 10.0, 4.0, 0.1, key="sim_dist")
        values["toppings_count"] = st.slider("Số lượng toppings", 1, 5, 3, key="sim_toppings")
        values["order_hour"] = st.slider("Giờ đặt hàng (Hour)", 0, 23, 19, key="sim_hour")
        values["is_peak_hour"] = st.checkbox("Giờ cao điểm", value=True, key="sim_peak")
        values["is_weekend"] = st.checkbox("Ngày cuối tuần (Weekend)", value=False, key="sim_weekend")
        
    # Tính các biến phái sinh tất định cho Simulator
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
    sim_res_cols[2].metric("Mức ưu tiên đề xuất", sim_decision["priority"])
    
    st.info(f"👉 **Hành động gợi ý:** {sim_decision['recommended_action']}")

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
        st.write("**Cơ cấu cỡ bánh Pizza (Size preference):**")
        st.dataframe(prefs["size_mix"], hide_index=True, width="stretch")
        st.plotly_chart(
            px.pie(prefs["size_mix"], values="order_share", names="pizza_size_label", title="Tỷ lệ đặt hàng theo cỡ bánh"),
            use_container_width=True
        )
    with col2:
        st.write("**Cơ cấu loại nhân Pizza (Type preference):**")
        st.dataframe(prefs["type_mix"], hide_index=True, width="stretch")
        st.plotly_chart(
            px.pie(prefs["type_mix"], values="order_share", names="pizza_type", title="Tỷ lệ đặt hàng theo loại nhân bánh"),
            use_container_width=True
        )
        
    st.subheader("💡 Luật kết hợp đề xuất sản phẩm dựa trên Context")
    st.dataframe(recommendation_rules(df_raw).head(15), hide_index=True, width="stretch")
    
    st.markdown("""
    <div class='insight-box'>
        <b>💡 Phân tích hành vi & quy tắc đề xuất:</b>
        <ul>
            <li>Cỡ bánh được yêu thích nhất là <b>Medium (chiếm ~39.4%)</b>, tiếp theo là Large (~30.1%).</li>
            <li>Pizza nhân <b>Non-Veg (Thịt) dẫn đầu tuyệt đối với ~69.6%</b> lượng đơn hàng.</li>
            <li><i>Quy tắc đề xuất sản phẩm (Recommendation Heuristics):</i> Dựa trên tần suất bán chạy theo từng khung giờ và địa điểm. Ví dụ: Vào cuối tuần hoặc giờ cao điểm, hệ thống tự động gợi ý các combo Pizza cỡ Medium kèm topping bán chạy nhất tại khu vực tương ứng để tăng giá trị đơn hàng (AOV).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# TAB 3: DATA FORENSICS & QUALITY (TRUTH ENGINE)
# =====================================================================
with tabs[2]:
    st.markdown("<h3 class='section-header'>Data Forensics: Giải phẫu dữ liệu sinh lập (Synthetic Proofs)</h3>", unsafe_allow_html=True)
    st.caption("Bằng chứng toán học chứng minh dữ liệu Kaggle là giả lập và các tác động của nó.")
    
    # 1. Realism Audit
    st.subheader("🔍 Kiểm định tính chân thực của dữ liệu (Realism Audit)")
    left_q, right_q = st.columns([3, 2])
    with left_q:
        synthetic = synthetic_data_audit(df_raw)
        st.dataframe(synthetic, hide_index=True, width="stretch")
    with right_q:
        fig_synthetic = px.bar(
            synthetic["severity"].value_counts().reset_index(name="checks"),
            x="severity",
            y="checks",
            title="Số lỗi phát hiện theo mức độ nghiêm trọng",
            labels={"severity": "Mức độ nghiêm trọng", "checks": "Số lỗi"}
        )
        st.plotly_chart(fig_synthetic, use_container_width=True)
        
    st.markdown("""
    <div class='warning-box'>
        <b>🕵️ Nhận xét về Tính chân thực của dữ liệu:</b>
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
            st.dataframe(pd.read_csv(formula_path), hide_index=True, width="stretch")
        else:
            st.warning("Không tìm thấy file generator_deterministic_formulas.csv")
            
        fig_err_path = FIGURES_DIR / "generator_deterministic_formula_errors.png"
        if fig_err_path.exists():
            st.image(str(fig_err_path), caption="Sai số của công thức phục dựng chạm mức 10^-15 (tức khớp tuyệt đối)")
            
    with col_d2:
        st.subheader("⏱️ Suy luận Ngưỡng Trễ (Label Threshold Inference)")
        st.caption("Làm thế nào để biết bao nhiêu phút giao hàng thì bị coi là 'trễ'?")
        
        threshold_path = METRICS_DIR / "delay_threshold_inference.csv"
        if threshold_path.exists():
            # Show top vài dòng
            st.dataframe(pd.read_csv(threshold_path).head(10), hide_index=True, width="stretch")
        else:
            st.warning("Không tìm thấy file delay_threshold_inference.csv")
            
        fig_thresh_path = FIGURES_DIR / "delay_threshold_inference.png"
        if fig_thresh_path.exists():
            st.image(str(fig_thresh_path), caption="Sai lệch nhãn chạm 0 tại ranh giới 30-35 phút")
            
    st.markdown("""
    <div class='insight-box'>
        <b>🕵️ Bài học rút ra từ Data Forensics:</b>
        <ul>
            <li><b>Sai số phục dựng công thức cực nhỏ (10^-15):</b> Chứng minh tuyệt đối dữ liệu sinh bằng thuật toán. Ví dụ: Thời gian giao hàng dự tính <code>estimated_duration_min</code> luôn bằng đúng <code>distance_km * 2.4</code>.</li>
            <li><b>Ngưỡng nhãn trễ:</b> Nhãn trễ <code>is_delayed</code> không được định nghĩa trước, nhưng dò ngưỡng cho thấy quy tắc gán nhãn của bộ sinh là: nếu thời gian thực tế <code>delivery_duration_min > 30</code> phút thì gán nhãn 1 (Trễ), ngược lại gán 0. Điều này giải thích ranh giới nhãn.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Brand Uniformity & Ablation
    st.subheader("🏢 Phân tích Brand: Sự thật hay Artifact?")
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
        <b>🏢 Phân tích Brand Ablation (Loại bỏ Brand):</b>
        <ul>
            <li>Kiểm định Chi-square cho thấy <b>không có sự khác biệt có ý nghĩa thống kê</b> về tỷ lệ trễ giữa các Brand (P-value > 0.05).</li>
            <li><b>Brand Ablation:</b> Thử nghiệm huấn luyện mô hình khi có biến Brand và không có biến Brand cho thấy độ chính xác (F2 score) gần như không đổi.</li>
            <li><i>Kết luận:</i> Tên nhà hàng (Brand) chỉ là một thuộc tính gán nhãn ngẫu nhiên trong dữ liệu giả lập, không đại diện cho chất lượng vận hành thật của từng thương hiệu.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# TAB 4: ML MODEL REGISTRY & BENCHMARKS (TECHNICAL CORE)
# =====================================================================
with tabs[3]:
    st.markdown("<h3 class='section-header'>Đánh giá Mô hình Học Máy (Machine Learning Registry)</h3>", unsafe_allow_html=True)
    st.caption("Quy trình huấn luyện, so sánh mô hình khóa và kiểm thử nghiêm ngặt.")
    
    dev_path = METRICS_DIR / "model_dev_comparison.csv"
    test_path = METRICS_DIR / "model_test_metrics.csv"
    baseline_path = METRICS_DIR / "baseline_test_metrics.csv"
    tuning_path = METRICS_DIR / "default_vs_tuned_lr.csv"
    stability_summary_path = METRICS_DIR / "model_stability_summary.csv"
    stability_figure_path = FIGURES_DIR / "model_stability_f2_distribution.png"
    
    if not dev_path.exists() or not test_path.exists():
        st.warning("Vui lòng chạy `python -m scripts.run_all` trước để sinh các chỉ số kiểm thử.")
    else:
        # 1. So sánh mô hình trên tập Dev
        st.subheader("📊 So sánh 6 Classifier trên tập Phát triển (Dev Set)")
        left_m, right_m = st.columns([3, 2])
        
        with left_m:
            dev_df = pd.read_csv(dev_path)
            st.dataframe(dev_df, hide_index=True, width="stretch")
        with right_m:
            # Vẽ biểu đồ F2 của các model
            fig_compare = px.bar(
                dev_df, 
                x="model", 
                y="f2",
                title="So sánh F2-Score giữa các thuật toán",
                labels={"model": "Mô hình", "f2": "F2 Score (Ưu tiên bắt trễ)"},
                color="f2",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
        st.markdown("""
        <div class='insight-box'>
            <b>📊 Phân tích so sánh mô hình:</b>
            <ul>
                <li>Mô hình được chọn là <b>Logistic Regression (F2 = 0.9491)</b> vì tính ổn định cao, đơn giản, dễ giải thích trọng số và đạt hiệu suất cao trên dữ liệu dạng công thức phẳng này.</li>
                <li>Mô hình Naive Bayes bị loại vì đạt điểm rất thấp (F2 chỉ ~0.5), nguyên nhân do giả định các biến độc lập của Naive Bayes bị vi phạm nghiêm trọng (các biến như distance_km và estimated_duration_min phụ thuộc tuyến tính 100% vào nhau).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. Kết quả kiểm thử khóa & Baseline
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("🎯 Hiệu suất mô hình khóa trên Tập Kiểm Thử (Locked Test Set)")
            st.dataframe(pd.read_csv(test_path), hide_index=True, width="stretch")
            
            fig_cm_path = FIGURES_DIR / "confusion_matrix_test.png"
            if fig_cm_path.exists():
                st.image(str(fig_cm_path), caption="Confusion Matrix trên tập Test (Chỉ bỏ sót 1 đơn trễ thực tế)")
        with col_t2:
            st.subheader("🏁 So sánh với các mô hình đoán ngây thơ (Baselines)")
            st.dataframe(pd.read_csv(baseline_path), hide_index=True, width="stretch")
            
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
        st.subheader("🛡️ Độ ổn định mô hình (Stability Audit) & Tinh chỉnh")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            if stability_summary_path.exists():
                st.write("**Thống kê hiệu suất qua 100 lần phân rã ngẫu nhiên (100-run Bootstrap):**")
                st.dataframe(pd.read_csv(stability_summary_path), hide_index=True, width="stretch")
            if stability_figure_path.exists():
                st.image(str(stability_figure_path), caption="Phân phối điểm F2 qua 100 lần chạy ngẫu nhiên")
                
        with col_s2:
            if tuning_path.exists():
                st.write("**Kết quả tinh chỉnh siêu tham số (Hyperparameter Tuning):**")
                st.dataframe(pd.read_csv(tuning_path)[[
                    "model", "param_C", "cv_f2_mean", "f2", "mcc", "decision"
                ]], hide_index=True, width="stretch")
                
        st.markdown("""
        <div class='insight-box'>
            <b>🛡️ Đánh giá Độ ổn định & Tinh chỉnh:</b>
            <ul>
                <li><b>100-run Audit:</b> Điểm F2 trung bình đạt 0.941 với độ lệch chuẩn cực nhỏ (0.015). Điều này chứng minh mô hình hoạt động ổn định và kết quả tốt không phải do ăn may một lần chia tập Train/Dev/Test.</li>
                <li><b>Quyết định Tuning:</b> Thử nghiệm tinh chỉnh tham số C của Logistic Regression cho thấy mặc định C=1.0 đạt F2 tốt hơn trên tập validation so với C=0.3, vì vậy nhóm quyết định giữ mô hình mặc định.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
