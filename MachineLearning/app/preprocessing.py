
from pathlib import Path
import json

import numpy as np
import pandas as pd


# 1. Đường dẫn và metadata
BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    BASE_DIR / "data" / "processed" / "feature_metadata.json"
)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

NUM_COLS = metadata["num_cols"]
LOW_CAT_COLS = metadata["low_card_cat_cols"]
HIGH_CAT_COLS = metadata["high_card_cat_cols"]

FEATURE_COLS = NUM_COLS + LOW_CAT_COLS + HIGH_CAT_COLS


# 2. Hàm tiền xử lý dữ liệu đầu vào
def preprocess_input(input_data):

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # 3. Xử lý ngày nhận phòng
    if "arrival_date" in df.columns:

        arrival_date = pd.to_datetime(
            df["arrival_date"], errors="raise"
        )

        df["arrival_date_year"] = arrival_date.dt.year

        df["arrival_date_week_number"] = (
            arrival_date.dt.isocalendar().week.astype(int)
        )

        df["arrival_date_day_of_month"] = arrival_date.dt.day
        df["arrival_month_num"] = arrival_date.dt.month

    else:
        month_map = {
            "January": 1, "February": 2,
            "March": 3, "April": 4,
            "May": 5, "June": 6,
            "July": 7, "August": 8,
            "September": 9, "October": 10,
            "November": 11, "December": 12
        }

        df["arrival_month_num"] = (
            df["arrival_date_month"].map(month_map).astype(int)
        )

    # 4. Xử lý dữ liệu phân loại bị thiếu
    df["country"] = df["country"].fillna("Unknown").astype(str)

    for col in LOW_CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)

    # 5. Xử lý mã đại lý và công ty
    for col in ["agent", "company"]:

        values = df[col].replace(
            ["", "Unknown"], np.nan
        )

        df[f"has_{col}"] = values.notna().astype(int)

        df[col] = values.apply(
            lambda x: str(int(float(x)))
            if pd.notna(x) else "Unknown"
        )

    # 6. Feature Engineering
    df["total_nights"] = (
        df["stays_in_weekend_nights"]
        + df["stays_in_week_nights"]
    )

    df["total_guests"] = df[
        ["adults", "children", "babies"]
    ].sum(axis=1, min_count=3)

    df["weekend_ratio"] = np.where(
        df["total_nights"] > 0,
        df["stays_in_weekend_nights"] / df["total_nights"],
        0.0
    ).round(4)

    df["total_previous_bookings"] = (
        df["previous_cancellations"]
        + df["previous_bookings_not_canceled"]
    )

    df["prev_cancel_ratio"] = np.where(
        df["total_previous_bookings"] > 0,
        df["previous_cancellations"]
        / df["total_previous_bookings"],
        0.0
    ).round(4)

    # 7. Mã hóa chu kỳ tháng
    df["arrival_month_sin"] = np.sin(
        2 * np.pi * df["arrival_month_num"] / 12
    ).round(6)

    df["arrival_month_cos"] = np.cos(
        2 * np.pi * df["arrival_month_num"] / 12
    ).round(6)

    # 8. Kiểm tra và sắp xếp đặc trưng
    missing_cols = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Thiếu các đặc trưng: {missing_cols}"
        )

    return df[FEATURE_COLS].copy()







