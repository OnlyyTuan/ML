"""
Tiền xử lý dữ liệu và trích xuất đặc trưng.

Đọc 6 file CSV gốc của Olist trong data/raw, tạo 1 dòng / đơn hàng với target
delivery_days, rồi chia train/val/test theo thời gian và lưu vào data/processed.

    python -m src.data_prep [--config configs/config.yaml]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, resolve

BR_LAT = (-35.0, 5.5)   # khung toạ độ Brazil
BR_LNG = (-75.0, -34.0)

DATE_COLS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]


# ---------------------------------------------------------------- orders
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng thời gian đều biết được tại lúc đặt hàng (dùng chung cho predict)."""
    df = df.copy()
    purchase = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_year"] = purchase.dt.year
    df["purchase_month"] = purchase.dt.month
    df["purchase_dayofweek"] = purchase.dt.day_of_week
    df["purchase_hour"] = purchase.dt.hour
    df["purchase_is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    approved = pd.to_datetime(df.get("order_approved_at", pd.NaT), errors="coerce")
    df["approval_hours"] = (approved - purchase).dt.total_seconds() / 3600
    estimated = pd.to_datetime(df.get("order_estimated_delivery_date", pd.NaT), errors="coerce")
    df["estimated_days"] = (estimated - purchase).dt.total_seconds() / 86400
    return df


def preprocess_orders(orders: pd.DataFrame, outlier_quantile: float = 0.995) -> pd.DataFrame:
    """Tạo target + đặc trưng thời gian. Chỉ giữ đơn đã giao."""
    df = orders.copy()
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df[df["order_status"] == "delivered"]
    df = df.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])

    purchase = df["order_purchase_timestamp"]
    df["delivery_days"] = (df["order_delivered_customer_date"] - purchase).dt.total_seconds() / 86400
    df = df[df["delivery_days"] > 0]
    if len(df) and outlier_quantile < 1:
        df = df[df["delivery_days"] <= df["delivery_days"].quantile(outlier_quantile)]

    df = add_time_features(df)
    df["approval_hours"] = df["approval_hours"].fillna(df["approval_hours"].median())
    # order_delivered_carrier_date chỉ biết SAU khi đặt hàng -> leakage, không dùng
    keep = ["order_id", "customer_id", "order_purchase_timestamp", "delivery_days",
            "purchase_year", "purchase_month", "purchase_dayofweek", "purchase_hour",
            "purchase_is_weekend", "approval_hours", "estimated_days"]
    return df[keep]


# ---------------------------------------------------------------- products / items
def preprocess_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.rename(columns={
        "product_name_lenght": "product_name_length",            # dataset gốc viết sai
        "product_description_lenght": "product_description_length",
    })
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    num_cols = ["product_photos_qty", "product_weight_g",
                "product_length_cm", "product_height_cm", "product_width_cm"]
    for c in num_cols:
        df[c] = df[c].fillna(df.groupby("product_category_name")[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())
    df["product_volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    return df[["product_id", "product_category_name", "product_photos_qty",
               "product_weight_g", "product_volume_cm3"]]


def preprocess_items(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Gộp các món trong đơn: giá, phí ship, số lượng, cân nặng, thể tích."""
    it = items.merge(products, on="product_id", how="left")
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        max_price=("price", "max"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        max_weight_g=("product_weight_g", "max"),
        total_volume_cm3=("product_volume_cm3", "sum"),
        mean_photos_qty=("product_photos_qty", "mean"),
    ).reset_index()
    agg["freight_ratio"] = (agg["total_freight"] / agg["total_price"].replace(0, np.nan)).fillna(0)
    # người bán + danh mục "chính" = món đắt nhất trong đơn
    main = (it.sort_values("price", ascending=False)
              .drop_duplicates("order_id")[["order_id", "seller_id", "product_category_name"]])
    return agg.merge(main, on="order_id", how="left")


# ---------------------------------------------------------------- geolocation
def preprocess_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """zip code prefix -> median (lat, lng), bỏ điểm ngoài Brazil."""
    g = geo[geo["geolocation_lat"].between(*BR_LAT) & geo["geolocation_lng"].between(*BR_LNG)]
    return (g.groupby("geolocation_zip_code_prefix")
             .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
             .reset_index()
             .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"}))


def attach_geo(df: pd.DataFrame, geo: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Gắn lat/lng theo zip; thiếu thì lấy trung bình theo bang."""
    zip_col, state_col = f"{prefix}_zip_code_prefix", f"{prefix}_state"
    out = df.merge(geo, left_on=zip_col, right_on="zip_code_prefix", how="left").drop(columns="zip_code_prefix")
    for c in ("lat", "lng"):
        out[c] = out[c].fillna(out.groupby(state_col)[c].transform("mean"))
    return out.rename(columns={"lat": f"{prefix}_lat", "lng": f"{prefix}_lng"})


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_km"] = haversine_km(df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    if "customer_city" in df and "seller_city" in df:
        df["same_city"] = (df["customer_city"] == df["seller_city"]).astype(int)
    return df


# ---------------------------------------------------------------- pipeline
RAW_FILES = ["orders", "order_items", "customers", "sellers", "products", "geolocation"]


def load_raw(raw_dir: Path) -> dict:
    return {n: pd.read_csv(Path(raw_dir) / f"olist_{n}_dataset.csv") for n in RAW_FILES}


def build_dataset(raw: dict, outlier_quantile: float = 0.995) -> pd.DataFrame:
    orders = preprocess_orders(raw["orders"], outlier_quantile)
    items = preprocess_items(raw["order_items"], preprocess_products(raw["products"]))
    geo = preprocess_geolocation(raw["geolocation"])
    customers = attach_geo(raw["customers"], geo, "customer")[
        ["customer_id", "customer_state", "customer_city", "customer_lat", "customer_lng"]]
    sellers = attach_geo(raw["sellers"], geo, "seller")[
        ["seller_id", "seller_state", "seller_city", "seller_lat", "seller_lng"]]

    df = (orders.merge(items, on="order_id", how="inner")
                .merge(customers, on="customer_id", how="left")
                .merge(sellers, on="seller_id", how="left"))
    df = add_geo_features(df).dropna(subset=["distance_km"])
    df = df.drop(columns=["customer_id", "customer_city", "seller_city"])
    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


def time_split(df: pd.DataFrame, val_size: float, test_size: float):
    """Chia theo thứ tự thời gian: train (cũ nhất) / val / test (mới nhất)."""
    n = len(df)
    n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
    n_train = n - n_val - n_test
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def main(config_path=None):
    cfg = load_config(config_path)
    raw_dir, out_dir = resolve(cfg["paths"]["raw_dir"]), resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(load_raw(raw_dir), cfg["preprocess"]["outlier_quantile"])
    train, val, test = time_split(df, cfg["preprocess"]["val_size"], cfg["preprocess"]["test_size"])
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):>7,} rows -> {out_dir / f'{name}.csv'}")
    print(df["delivery_days"].describe().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
