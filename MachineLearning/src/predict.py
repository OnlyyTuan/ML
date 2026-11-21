"""
Tải mô hình và dự đoán thời gian giao hàng cho dữ liệu mới.

Input là CSV có các cột đặc trưng (giống data/processed/*.csv). Các đặc trưng
suy ra được (thời gian, khoảng cách, freight_ratio) sẽ tự tính nếu thiếu.

    python -m src.predict --input new_orders.csv --output predictions.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from src.data_prep import add_geo_features, add_time_features
from src.utils import load_config, resolve


def load_model(config_path=None) -> dict:
    cfg = load_config(config_path)
    return joblib.load(resolve(cfg["paths"]["model_path"]))


def prepare_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = df.copy()
    if "order_purchase_timestamp" in df and "purchase_month" not in df:
        df = add_time_features(df)
    if "distance_km" not in df and {"customer_lat", "customer_lng", "seller_lat", "seller_lng",
                                    "customer_state", "seller_state"} <= set(df):
        df = add_geo_features(df)
    if "freight_ratio" not in df and {"total_freight", "total_price"} <= set(df):
        df["freight_ratio"] = (df["total_freight"] / df["total_price"].replace(0, np.nan)).fillna(0)
    for c in features:          # cột thiếu -> NaN (HistGradientBoosting xử lý được)
        if c not in df:
            df[c] = np.nan
    return df[features]


def predict(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    X = prepare_features(df, bundle["features"])
    return np.clip(bundle["pipeline"].predict(X), 0, None)


def predict_one(order: dict, bundle: dict) -> float:
    return float(predict(pd.DataFrame([order]), bundle)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    bundle = load_model(args.config)
    df = pd.read_csv(args.input)
    df["predicted_delivery_days"] = predict(df, bundle).round(2)
    df.to_csv(args.output, index=False)
    print(f"Predicted {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
