
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px


from translations import (
    HOTEL_LABELS,
    MONTH_LABELS,
    MARKET_SEGMENT_LABELS,
    DEPOSIT_LABELS
)


# ==================================
# 1. ĐƯỜNG DẪN
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "raw" / "hotel_bookings.csv"
)

SUMMARY_PATH = (
    BASE_DIR / "models" / "evaluation_summary.json"
)

MONTHS = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December"
]

COLOR_MAP = {
    "Đã hủy": "#EF4444",
    "Không hủy": "#10B981"
}


# ==================================
# 2. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ==================================

@st.cache_data
def load_dashboard_data():

    df = pd.read_csv(DATA_PATH)

    # Loại bỏ đơn không có khách
    total_guests = df[
        ["adults", "children", "babies"]
    ].sum(axis=1, min_count=3)

    df = df[total_guests != 0].copy()

    # Loại bỏ giá phòng âm
    df = df[df["adr"] >= 0].copy()

    # Tạo cột ngày nhận phòng
    month_map = {
        month: i + 1
        for i, month in enumerate(MONTHS)
    }

    df["arrival_month_num"] = (
        df["arrival_date_month"].map(month_map)
    )

    df["arrival_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": df["arrival_month_num"],
        "day": df["arrival_date_day_of_month"]
    }, errors="coerce")

    # Tạo cột tổng số đêm
    df["total_nights"] = (
        df["stays_in_weekend_nights"]
        + df["stays_in_week_nights"]
    )

    # Chuyển nhãn sang tiếng Việt
    df["booking_status"] = df["is_canceled"].map({
        0: "Không hủy",
        1: "Đã hủy"
    })

    # Tạo nhóm thời gian đặt trước
    df["lead_time_group"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, float("inf")],
        labels=[
            "0 - 7 ngày",
            "8 - 30 ngày",
            "31 - 90 ngày",
            "91 - 180 ngày",
            "Trên 180 ngày"
        ]
    )

    return df


# ==================================
# 3. ĐỌC KẾT QUẢ MÔ HÌNH
# ==================================

@st.cache_data
def load_model_summary():

    with open(
        SUMMARY_PATH, "r", encoding="utf-8"
    ) as f:

        return json.load(f)


# ==================================
# 4. HÀM VẼ TỶ LỆ HỦY
# ==================================



def cancellation_chart(df, column, title):

    stats = (
        df.groupby(column, observed=True)
        .agg(
            total_bookings=("is_canceled", "size"),
            canceled_bookings=("is_canceled", "sum"),
            cancel_rate=("is_canceled", "mean")
        )
        .reset_index()
    )

    stats["cancel_rate"] *= 100

    # Từ điển Việt hóa
    label_maps = {
        "hotel": HOTEL_LABELS,
        "deposit_type": DEPOSIT_LABELS,
        "market_segment": MARKET_SEGMENT_LABELS
    }

    # Tạo cột riêng để hiển thị tiếng Việt
    if column in label_maps:

        stats["display_label"] = (
            stats[column]
            .astype(str)
            .map(lambda x: label_maps[column].get(x, x))
        )

    else:

        stats["display_label"] = stats[column].astype(str)

    #st.write(stats[[column, "display_label"]])

    # Vẽ biểu đồ bằng cột tiếng Việt
    fig = px.bar(
        stats,
        x="display_label",
        y="cancel_rate",
        color="cancel_rate",
        color_continuous_scale=[
            "#10B981",
            "#F59E0B",
            "#EF4444"
        ],
        range_color=[0, 100],
        text_auto=".1f",
        hover_data={
            "display_label": True,
            "total_bookings": True,
            "canceled_bookings": True,
            "cancel_rate": ":.2f"
        },
        labels={
            "display_label": "Nhóm",
            "total_bookings": "Tổng đơn đặt phòng",
            "canceled_bookings": "Số đơn đã hủy",
            "cancel_rate": "Tỷ lệ hủy (%)"
        },
        title=title
    )

    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Tỷ lệ hủy (%)",
        yaxis_range=[0, 110],
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.update_xaxes(
        tickangle=-25 if column == "market_segment" else 0,
        automargin=True
    )

    

    return fig



# ==================================
# 5. GIAO DIỆN DASHBOARD
# ==================================

def show_dashboard():

    st.title("Dashboard tổng quan")

    st.write(
        "Phân tích tình hình đặt phòng và hủy phòng "
        "dựa trên dữ liệu lịch sử của khách sạn."
    )

    df = load_dashboard_data()

    st.caption(
        "Nguồn dữ liệu lịch sử: 2015 - 2017. "
        "Không phải dữ liệu đặt phòng thời gian thực."
    )

    # ----------------------------------
    # 5.1. BỘ LỌC
    # ----------------------------------

    st.subheader("Bộ lọc dữ liệu")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        hotel_filter = st.selectbox(
            "Loại khách sạn",
            ["Tất cả"] + sorted(df["hotel"].unique().tolist()),
            format_func=lambda x: HOTEL_LABELS.get(x, x)
        )

    with col2:

        year_filter = st.selectbox(
            "Năm nhận phòng",
            ["Tất cả"] + sorted(
                df["arrival_date_year"].unique().tolist()
            )
        )

    with col3:

        month_filter = st.selectbox(
            "Tháng nhận phòng",
            ["Tất cả"] + MONTHS,
            format_func=lambda x: MONTH_LABELS.get(x, x)
        )

    with col4:

        segment_filter = st.selectbox(
            "Phân khúc thị trường",
            ["Tất cả"] + sorted(
                df["market_segment"].dropna().unique().tolist()
            ),
            format_func=lambda x: MARKET_SEGMENT_LABELS.get(x, x)
        )

    filtered_df = df.copy()

    if hotel_filter != "Tất cả":

        filtered_df = filtered_df[
            filtered_df["hotel"] == hotel_filter
        ]

    if year_filter != "Tất cả":

        filtered_df = filtered_df[
            filtered_df["arrival_date_year"] == year_filter
        ]

    if month_filter != "Tất cả":

        filtered_df = filtered_df[
            filtered_df["arrival_date_month"] == month_filter
        ]

    if segment_filter != "Tất cả":

        filtered_df = filtered_df[
            filtered_df["market_segment"] == segment_filter
        ]

    if filtered_df.empty:

        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc."
        )

        return

    st.divider()

    # ----------------------------------
    # 5.2. CÁC CHỈ SỐ KPI
    # ----------------------------------

    total = len(filtered_df)

    canceled = int(
        filtered_df["is_canceled"].sum()
    )

    not_canceled = total - canceled

    cancel_rate = canceled / total

    st.subheader("Tổng quan đặt phòng")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tổng đơn đặt phòng",
        f"{total:,}"
    )

    col2.metric(
        "Số đơn đã hủy",
        f"{canceled:,}"
    )

    col3.metric(
        "Số đơn không hủy",
        f"{not_canceled:,}"
    )

    col4.metric(
        "Tỷ lệ hủy",
        f"{cancel_rate:.2%}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Giá phòng TB (EUR)",
        f"{filtered_df['adr'].mean():,.2f}"
    )

    col2.metric(
        "Số đêm đặt TB",
        f"{filtered_df['total_nights'].mean():.2f}"
    )

    col3.metric(
        "Số ngày đặt trước TB",
        f"{filtered_df['lead_time'].mean():.1f}"
    )

    col4.metric(
        "Tỷ lệ không hủy",
        f"{1 - cancel_rate:.2%}"
    )

    st.divider()

    # ----------------------------------
    # 5.3. BIỂU ĐỒ TỔNG QUAN
    # ----------------------------------

    st.subheader("Tình hình đặt phòng")

    col1, col2 = st.columns([1, 2])

    with col1:

        status_data = pd.DataFrame({
            "Trạng thái": [
                "Đã hủy",
                "Không hủy"
            ],
            "Số lượng": [
                canceled,
                not_canceled
            ]
        })

        fig = px.pie(
            status_data,
            names="Trạng thái",
            values="Số lượng",
            hole=0.6,
            color="Trạng thái",
            color_discrete_map=COLOR_MAP,
            title="Tỷ lệ hủy phòng"
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        fig.update_layout(
            height=380,
            legend=dict(orientation="h"),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        monthly = (
            filtered_df
            .groupby(
                pd.Grouper(
                    key="arrival_date",
                    freq="MS"
                )
            )
            .agg(
                total_bookings=("is_canceled", "size"),
                canceled_bookings=("is_canceled", "sum")
            )
            .reset_index()
        )

        monthly["not_canceled_bookings"] = (
            monthly["total_bookings"]
            - monthly["canceled_bookings"]
        )

        monthly = monthly.rename(columns={
            "total_bookings": "Tổng đơn",
            "canceled_bookings": "Đã hủy",
            "not_canceled_bookings": "Không hủy"
        })

        fig = px.line(
            monthly,
            x="arrival_date",
            y=[
                "Tổng đơn",
                "Đã hủy",
                "Không hủy"
            ],
            markers=True,
            color_discrete_map={
                "Tổng đơn": "#3B82F6",
                **COLOR_MAP
            },
            title="Đặt phòng theo tháng nhận phòng"
        )

        fig.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Số đơn đặt phòng",
            legend_title=None,
            height=380,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ----------------------------------
    # 5.4. PHÂN TÍCH TỶ LỆ HỦY
    # ----------------------------------

    st.subheader("Các đặc điểm liên quan đến hủy phòng")

    col1, col2 = st.columns(2)

    with col1:

        fig = cancellation_chart(
            filtered_df,
            "hotel",
            "Tỷ lệ hủy theo loại khách sạn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = cancellation_chart(
            filtered_df,
            "deposit_type",
            "Tỷ lệ hủy theo hình thức đặt cọc"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        fig = cancellation_chart(
            filtered_df,
            "market_segment",
            "Tỷ lệ hủy theo phân khúc thị trường"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = cancellation_chart(
            filtered_df,
            "lead_time_group",
            "Tỷ lệ hủy theo thời gian đặt trước"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        "Các biểu đồ thể hiện mối liên hệ thống kê, "
        "không khẳng định quan hệ nguyên nhân - kết quả."
    )

    st.divider()

    # ----------------------------------
    # 5.5. PHÂN BỐ THỜI GIAN ĐẶT TRƯỚC
    # ----------------------------------

    st.subheader("Phân bố thời gian đặt trước")

    fig = px.histogram(
        filtered_df,
        x="lead_time",
        color="booking_status",
        color_discrete_map=COLOR_MAP,
        nbins=50,
        barmode="overlay",
        opacity=0.65
    )

    fig.update_layout(
        xaxis_title="Số ngày đặt trước",
        yaxis_title="Số đơn đặt phòng",
        legend_title="Trạng thái",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ----------------------------------
    # 5.6. THÔNG TIN MÔ HÌNH
    # ----------------------------------

    st.subheader("Mô hình Machine Learning đang sử dụng")

    summary = load_model_summary()

    model_info = summary["stratified_final"]

    metrics = model_info["test_metrics_oof_opt"]

    st.write(
        f"**Mô hình:** {model_info['model']}"
    )

    st.write(
        f"**Threshold:** "
        f"{model_info['oof_optimal_threshold']:.2f}"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    col4.metric(
        "F1-Score",
        f"{metrics['f1']:.2%}"
    )

    col5.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.2%}"
    )

    st.caption(
        "Kết quả đánh giá trên tập test Stratified "
        "với threshold tối ưu từ OOF. "
        "Các chỉ số này không thay đổi theo bộ lọc Dashboard."
    )

    st.divider()

    # ----------------------------------
    # 5.7. DỮ LIỆU CHI TIẾT
    # ----------------------------------

    with st.expander("Xem dữ liệu đặt phòng chi tiết"):

        display_cols = [
            "hotel",
            "arrival_date",
            "lead_time",
            "adults",
            "children",
            "total_nights",
            "adr",
            "market_segment",
            "deposit_type",
            "booking_status"
        ]

        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Tổng cộng {len(filtered_df):,} bản ghi."
        )

        st.download_button(
            label="Tải xuống dữ liệu đã lọc",
            data=filtered_df[display_cols].to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="hotel_booking_statistics.csv",
            mime="text/csv"
        )
