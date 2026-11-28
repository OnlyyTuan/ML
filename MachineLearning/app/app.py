import pandas as pd
from dashboard import show_dashboard

from datetime import date

import streamlit as st

from prediction import (
    predict_booking,
    MODEL_NAME,
    THRESHOLD
)

from translations import (
    HOTEL_LABELS,
    DEPOSIT_LABELS,
    CUSTOMER_LABELS,
    MEAL_LABELS,
    MARKET_SEGMENT_LABELS,
    DISTRIBUTION_LABELS
)


import dashboard
import translations

# ==============================
# 1. CẤU HÌNH TRANG
# ==============================

st.set_page_config(
    page_title="Hotel Booking Analytics",
    page_icon="🏨",
    layout="wide"
)




# ==============================
# 2. SIDEBAR
# ==============================

st.sidebar.title("🏨 Hotel Booking Analytics")

page = st.sidebar.radio(
    "Chức năng",
    [
        "Tổng quan",
        "Dự đoán đơn lẻ",
        "Dự đoán hàng loạt"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Thông tin mô hình")
st.sidebar.write(f"**Model:** {MODEL_NAME}")
st.sidebar.write(f"**Threshold:** {THRESHOLD:.2f}")


# ==============================
# 3. TRANG TỔNG QUAN
# ==============================

if page == "Tổng quan":

    show_dashboard()


# ==============================
# 4. DỰ ĐOÁN ĐƠN LẺ
# ==============================

elif page == "Dự đoán đơn lẻ":

    st.title("Dự đoán hủy đặt phòng")

    st.write(
        "Nhập thông tin đơn đặt phòng để đánh giá "
        "khả năng khách hàng hủy phòng."
    )

    with st.form("booking_form"):

        # --------------------------
        # Nhóm 1: Thông tin đặt phòng
        # --------------------------

        st.subheader("1. Thông tin đặt phòng")

        col1, col2, col3 = st.columns(3)

        with col1:

            hotel = st.selectbox(
                "Loại khách sạn",
                options=list(HOTEL_LABELS.keys()),
                format_func=lambda x: HOTEL_LABELS[x]
            )

            lead_time = st.number_input(
                "Số ngày đặt trước",
                min_value=0,
                value=90,
                step=1
            )

        with col2:

            arrival_date = st.date_input(
                "Ngày nhận phòng dự kiến",
                value=date(2017, 8, 25)
            )

            reserved_room_type = st.selectbox(
                "Loại phòng đã đặt",
                ["A", "B", "C", "D", "E", "F", "G", "H", "L"]
            )

        with col3:

            adr = st.number_input(
                "Giá phòng trung bình mỗi đêm (EUR)",
                min_value=0.0,
                value=120.0,
                step=10.0
            )

            deposit_type = st.selectbox(
                "Hình thức đặt cọc",
                options=list(DEPOSIT_LABELS.keys()),
                format_func=lambda x: DEPOSIT_LABELS[x]
            )

        st.divider()

        # --------------------------
        # Nhóm 2: Thông tin khách hàng
        # --------------------------

        st.subheader("2. Thông tin khách hàng")

        col1, col2, col3 = st.columns(3)

        with col1:

            adults = st.number_input(
                "Số người lớn",
                min_value=0,
                value=2,
                step=1
            )

            children = st.number_input(
                "Số trẻ em",
                min_value=0,
                value=1,
                step=1
            )

        with col2:

            babies = st.number_input(
                "Số em bé",
                min_value=0,
                value=0,
                step=1
            )

            country = st.text_input(
                "Mã quốc gia",
                value="PRT",
                help="Ví dụ: PRT, GBR, FRA, DEU, VNM"
            )

        with col3:

            customer_type = st.selectbox(
                "Loại khách hàng",
                options=list(CUSTOMER_LABELS.keys()),
                format_func=lambda x: CUSTOMER_LABELS[x]
            )

            repeated_guest = st.checkbox(
                "Khách hàng từng lưu trú"
            )

        st.divider()

        # --------------------------
        # Nhóm 3: Thông tin lưu trú
        # --------------------------

        st.subheader("3. Thông tin lưu trú")

        col1, col2, col3 = st.columns(3)

        with col1:

            weekend_nights = st.number_input(
                "Số đêm cuối tuần",
                min_value=0,
                value=2,
                step=1
            )

            week_nights = st.number_input(
                "Số đêm trong tuần",
                min_value=0,
                value=3,
                step=1
            )

        with col2:

            meal = st.selectbox(
                "Gói ăn uống",
                options=list(MEAL_LABELS.keys()),
                format_func=lambda x: MEAL_LABELS[x]
            )

            parking_spaces = st.number_input(
                "Số chỗ đỗ xe yêu cầu",
                min_value=0,
                value=0,
                step=1
            )

        with col3:

            special_requests = st.number_input(
                "Số yêu cầu đặc biệt",
                min_value=0,
                value=1,
                step=1
            )

        st.divider()

        # --------------------------
        # Nhóm 4: Lịch sử đặt phòng
        # --------------------------

        st.subheader("4. Lịch sử đặt phòng")

        col1, col2 = st.columns(2)

        with col1:

            previous_cancellations = st.number_input(
                "Số lần từng hủy phòng",
                min_value=0,
                value=2,
                step=1
            )

        with col2:

            previous_not_canceled = st.number_input(
                "Số lần đặt phòng không hủy",
                min_value=0,
                value=8,
                step=1
            )

        st.divider()

        # --------------------------
        # Nhóm 5: Kênh đặt phòng
        # --------------------------

        st.subheader("5. Kênh đặt phòng")

        col1, col2 = st.columns(2)

        with col1:

            market_segment = st.selectbox(
                "Phân khúc thị trường",
                options=list(MARKET_SEGMENT_LABELS.keys()),
                format_func=lambda x: MARKET_SEGMENT_LABELS[x]
            )

            agent = st.number_input(
                "Mã đại lý (0 nếu không có)",
                min_value=0,
                value=9,
                step=1
            )

        with col2:

            distribution_channel = st.selectbox(
                "Kênh phân phối",
                options=list(DISTRIBUTION_LABELS.keys()),
                format_func=lambda x: DISTRIBUTION_LABELS[x]
            )

            company = st.number_input(
                "Mã công ty (0 nếu không có)",
                min_value=0,
                value=0,
                step=1
            )

        submitted = st.form_submit_button(
            "Dự đoán hủy phòng",
            type="primary",
            use_container_width=True
        )

    # ==============================
    # 5. XỬ LÝ DỰ ĐOÁN
    # ==============================

    if submitted:

        if adults + children + babies == 0:

            st.error("Tổng số khách phải lớn hơn 0.")

        else:

            input_data = {
                "hotel": hotel,
                "lead_time": lead_time,
                "arrival_date": arrival_date,
                "reserved_room_type": reserved_room_type,
                "adr": adr,
                "deposit_type": deposit_type,

                "adults": adults,
                "children": children,
                "babies": babies,
                "country": country.strip().upper() or "Unknown",
                "customer_type": customer_type,
                "is_repeated_guest": int(repeated_guest),

                "stays_in_weekend_nights": weekend_nights,
                "stays_in_week_nights": week_nights,
                "meal": meal,
                "required_car_parking_spaces": parking_spaces,
                "total_of_special_requests": special_requests,

                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_not_canceled,

                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "agent": agent if agent > 0 else None,
                "company": company if company > 0 else None
            }

            try:

                with st.spinner("Đang thực hiện dự đoán..."):

                    result = predict_booking(input_data).iloc[0]

                cancel_prob = float(
                    result["cancel_probability"]
                )

                not_cancel_prob = float(
                    result["not_cancel_probability"]
                )

                prediction = int(result["prediction"])

                # ==============================
                # 6. HIỂN THỊ KẾT QUẢ
                # ==============================

                st.divider()
                st.header("Kết quả dự đoán")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Xác suất hủy phòng",
                    f"{cancel_prob:.2%}"
                )

                col2.metric(
                    "Xác suất không hủy",
                    f"{not_cancel_prob:.2%}"
                )

                col3.metric(
                    "Ngưỡng phân loại",
                    f"{THRESHOLD:.0%}"
                )

                st.write("**Xác suất hủy phòng**")
                st.progress(cancel_prob)

                st.write("**Xác suất không hủy phòng**")
                st.progress(not_cancel_prob)

                if prediction == 1:

                    st.error("CẢNH BÁO: Đơn đặt phòng có nguy cơ hủy.")

                else:

                    st.success("KẾT QUẢ: Dự đoán không hủy phòng.")

                st.info(
                    f"Khuyến nghị: {result['recommendation']}"
                )

                with st.expander("Xem dữ liệu đầu vào"):

                    st.json(input_data, expanded=False)

            except Exception as e:

                st.error(f"Lỗi khi thực hiện dự đoán: {e}")


# ==============================
# 7. DỰ ĐOÁN HÀNG LOẠT
# ==============================


elif page == "Dự đoán hàng loạt":

    st.title("Dự đoán hàng loạt")

    st.write(
        "Tải lên danh sách đơn đặt phòng để đánh giá "
        "nguy cơ hủy phòng của nhiều khách hàng cùng lúc."
    )

    # ==================================
    # 1. CÁC CỘT DỮ LIỆU BẮT BUỘC
    # ==================================

    required_cols = [
        "hotel",
        "lead_time",
        "reserved_room_type",
        "adr",
        "deposit_type",
        "adults",
        "children",
        "babies",
        "country",
        "customer_type",
        "is_repeated_guest",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "meal",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "market_segment",
        "distribution_channel",
        "agent",
        "company"
    ]

    date_cols = [
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month"
    ]

    # ==================================
    # 2. TẢI FILE CSV
    # ==================================

    uploaded_file = st.file_uploader(
        "Tải lên file CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.success(
                f"Đã tải lên {len(df):,} đơn đặt phòng."
            )

            # Hiển thị dữ liệu ban đầu
            st.subheader("Dữ liệu đầu vào")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            # ==================================
            # 3. KIỂM TRA DỮ LIỆU
            # ==================================

            missing_cols = [
                col for col in required_cols
                if col not in df.columns
            ]

            # Hỗ trợ hai định dạng ngày
            if "arrival_date" not in df.columns:

                missing_cols += [
                    col for col in date_cols
                    if col not in df.columns
                ]

            if missing_cols:

                st.error(
                    "File CSV thiếu các cột bắt buộc:"
                )

                st.write(missing_cols)

            elif df.empty:

                st.warning("File CSV không có dữ liệu.")

            else:

                # ==================================
                # 4. THỰC HIỆN DỰ ĐOÁN
                # ==================================

                if st.button(
                    "Thực hiện dự đoán",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Đang phân tích danh sách đặt phòng..."
                    ):

                        input_df = df.copy()

                        numeric_cols = [
                            "lead_time",
                            "adr",
                            "adults",
                            "children",
                            "babies",
                            "is_repeated_guest",
                            "stays_in_weekend_nights",
                            "stays_in_week_nights",
                            "required_car_parking_spaces",
                            "total_of_special_requests",
                            "previous_cancellations",
                            "previous_bookings_not_canceled"
                        ]

                        if "arrival_date" not in input_df.columns:
                            numeric_cols += [
                                "arrival_date_year",
                                "arrival_date_week_number",
                                "arrival_date_day_of_month"
                            ]

                        for col in numeric_cols:

                            input_df[col] = pd.to_numeric(
                                input_df[col],
                                errors="raise"
                            )

                        # Kiểm tra giá trị thiếu
                        non_nullable = [
                            col for col in numeric_cols
                            if col != "children"
                        ]

                        if input_df[non_nullable].isna().any().any():
                            raise ValueError(
                                "Dữ liệu số bắt buộc bị thiếu."
                            )

                        # Kiểm tra giá trị âm
                        if (input_df[numeric_cols] < 0).any().any():
                            raise ValueError(
                                "Dữ liệu chứa giá trị âm không hợp lệ."
                            )

                        # Kiểm tra tổng số khách
                        total_guests = input_df[
                            ["adults", "children", "babies"]
                        ].sum(axis=1, min_count=3)

                        if (total_guests == 0).any():
                            raise ValueError(
                                "Có đơn đặt phòng không có khách."
                            )

                        # Chuẩn hóa mã quốc gia
                        input_df["country"] = (
                            input_df["country"]
                            .str.strip()
                            .str.upper()
                            .replace("", "Unknown")
                            .replace("UNKNOWN", "Unknown")
                        )

                        # Gọi mô hình
                        predictions = predict_booking(input_df)

                        # Ghép dữ liệu và kết quả
                        result_df = df.copy()

                        for col in predictions.columns:
                            result_df[col] = predictions[col]

                        # Lưu kết quả vào session state
                        st.session_state["batch_results"] = result_df

                        st.session_state["batch_file"] = (
                            uploaded_file.getvalue()
                        )

                # ==================================
                # 5. HIỂN THỊ KẾT QUẢ
                # ==================================

                if (
                    "batch_results" in st.session_state
                    and st.session_state.get("batch_file")
                    == uploaded_file.getvalue()
                ):

                    result_df = st.session_state["batch_results"]

                    st.divider()
                    st.header("Tổng quan kết quả")

                    total = len(result_df)

                    canceled = int(
                        result_df["prediction"].sum()
                    )

                    not_canceled = total - canceled

                    cancel_rate = canceled / total

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Tổng đơn đặt phòng",
                        f"{total:,}"
                    )

                    col2.metric(
                        "Có nguy cơ hủy",
                        f"{canceled:,}"
                    )

                    col3.metric(
                        "Dự đoán không hủy",
                        f"{not_canceled:,}"
                    )

                    col4.metric(
                        "Tỷ lệ dự đoán hủy",
                        f"{cancel_rate:.2%}"
                    )

                    # Biểu đồ thống kê
                    chart_data = pd.DataFrame({
                        "Kết quả": [
                            "Có nguy cơ hủy",
                            "Dự đoán không hủy"
                        ],
                        "Số lượng": [
                            canceled,
                            not_canceled
                        ]
                    })

                    st.bar_chart(
                        chart_data.set_index("Kết quả")
                    )

                    # ==================================
                    # 6. LỌC VÀ SẮP XẾP KẾT QUẢ
                    # ==================================

                    st.subheader("Danh sách kết quả dự đoán")

                    filter_option = st.selectbox(
                        "Lọc kết quả",
                        [
                            "Tất cả",
                            "Có nguy cơ hủy",
                            "Dự đoán không hủy"
                        ]
                    )

                    filtered_df = result_df.copy()

                    if filter_option == "Có nguy cơ hủy":

                        filtered_df = filtered_df[
                            filtered_df["prediction"] == 1
                        ]

                    elif filter_option == "Dự đoán không hủy":

                        filtered_df = filtered_df[
                            filtered_df["prediction"] == 0
                        ]

                    # Sắp xếp theo xác suất hủy giảm dần
                    filtered_df = filtered_df.sort_values(
                        "cancel_probability",
                        ascending=False
                    )

                    display_cols = [
                        "hotel",
                        "lead_time",
                        "adr",
                        "deposit_type",
                        "market_segment",
                        "cancel_probability",
                        "prediction_label",
                        "recommendation"
                    ]

                    if "booking_id" in filtered_df.columns:
                        display_cols.insert(0, "booking_id")

                    st.dataframe(
                        filtered_df[display_cols],
                        use_container_width=True,
                        column_config={
                            "hotel": "Khách sạn",
                            "lead_time": "Số ngày đặt trước",
                            "adr": "Giá phòng/đêm",
                            "deposit_type": "Đặt cọc",
                            "market_segment": "Phân khúc",
                            "cancel_probability":
                                st.column_config.ProgressColumn(
                                    "Xác suất hủy",
                                    min_value=0.0,
                                    max_value=1.0,
                                    format="percent"
                                ),
                            "prediction_label": "Dự đoán",
                            "recommendation": "Khuyến nghị"
                        }
                    )

                    st.caption(
                        f"Đang hiển thị {len(filtered_df):,} "
                        f"trên {total:,} đơn đặt phòng."
                    )

                    # ==================================
                    # 7. XUẤT FILE KẾT QUẢ
                    # ==================================

                    st.subheader("Xuất kết quả")

                    csv_data = filtered_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="Tải xuống kết quả CSV",
                        data=csv_data,
                        file_name="booking_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(f"Lỗi xử lý dữ liệu: {e}")
