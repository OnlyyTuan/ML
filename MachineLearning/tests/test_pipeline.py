"""Unit test cho tiền xử lý dữ liệu và suy luận, dùng dữ liệu giả lập nhỏ."""
import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from src import data_prep, evaluate, predict, train
from src.utils import load_config, regression_metrics

STATES = {"SP": (-23.5, -46.6), "RJ": (-22.9, -43.2), "BA": (-12.9, -38.5), "PR": (-25.4, -49.3)}


@pytest.fixture
def raw():
    """Sinh dữ liệu thô giống format Olist (300 đơn)."""
    rng = np.random.default_rng(0)
    n = 300
    states = list(STATES)
    zips = {s: 10000 + i * 1000 for i, s in enumerate(states)}

    geo = pd.DataFrame([
        {"geolocation_zip_code_prefix": zips[s], "geolocation_lat": lat + rng.normal(0, .05),
         "geolocation_lng": lng + rng.normal(0, .05), "geolocation_city": s.lower(), "geolocation_state": s}
        for s, (lat, lng) in STATES.items() for _ in range(3)
    ] + [{"geolocation_zip_code_prefix": zips["SP"], "geolocation_lat": 40.0, "geolocation_lng": 10.0,
          "geolocation_city": "x", "geolocation_state": "SP"}])   # điểm lỗi ngoài Brazil

    cust_state = rng.choice(states, n)
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)], "customer_unique_id": [f"u{i}" for i in range(n)],
        "customer_zip_code_prefix": [zips[s] for s in cust_state],
        "customer_city": [s.lower() for s in cust_state], "customer_state": cust_state})
    sellers = pd.DataFrame({
        "seller_id": [f"s{i}" for i in range(len(states))],
        "seller_zip_code_prefix": [zips[s] for s in states],
        "seller_city": [s.lower() for s in states], "seller_state": states})
    products = pd.DataFrame({
        "product_id": ["p0", "p1", "p2"], "product_category_name": ["moveis", "beleza", None],
        "product_name_lenght": [40, 30, np.nan], "product_description_lenght": [200, 100, np.nan],
        "product_photos_qty": [1, 2, np.nan], "product_weight_g": [5000, 300, np.nan],
        "product_length_cm": [50, 10, 20], "product_height_cm": [40, 5, 10], "product_width_cm": [30, 10, 10]})

    purchase = pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.uniform(0, 500, n), unit="D")
    seller_idx = rng.integers(0, len(states), n)
    dist = np.array([abs(STATES[cust_state[i]][0] - STATES[states[seller_idx[i]]][0]) for i in range(n)])
    days = 3 + dist * 1.5 + rng.exponential(2, n)
    status = np.where(rng.random(n) < 0.05, "canceled", "delivered")
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)], "customer_id": customers["customer_id"], "order_status": status,
        "order_purchase_timestamp": purchase,
        "order_approved_at": purchase + pd.Timedelta(hours=1),
        "order_delivered_carrier_date": purchase + pd.Timedelta(days=1),
        "order_delivered_customer_date": np.where(status == "delivered",
                                                  purchase + pd.to_timedelta(days, unit="D"), pd.NaT),
        "order_estimated_delivery_date": purchase + pd.Timedelta(days=20)})
    items = pd.DataFrame({
        "order_id": orders["order_id"], "order_item_id": 1,
        "product_id": rng.choice(["p0", "p1", "p2"], n), "seller_id": [f"s{i}" for i in seller_idx],
        "shipping_limit_date": purchase, "price": rng.uniform(10, 500, n), "freight_value": rng.uniform(5, 50, n)})
    return {"orders": orders, "order_items": items, "customers": customers,
            "sellers": sellers, "products": products, "geolocation": geo}


@pytest.fixture
def project(tmp_path, raw):
    """Dự án tạm: ghi raw CSV + config trỏ tới tmp_path."""
    import yaml
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name, df in raw.items():
        df.to_csv(raw_dir / f"olist_{name}_dataset.csv", index=False)
    cfg = copy.deepcopy(load_config())
    cfg["paths"] = {"raw_dir": str(raw_dir), "processed_dir": str(tmp_path / "processed"),
                    "model_path": str(tmp_path / "model.joblib"), "metrics_path": str(tmp_path / "metrics.json")}
    cfg["model"]["params"].update(max_iter=50, min_samples_leaf=5, early_stopping=False)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp_path, cfg_path, cfg


# ------------------------------------------------------------------ data
def test_orders_target_and_filter(raw):
    out = data_prep.preprocess_orders(raw["orders"], outlier_quantile=1.0)
    assert (out["delivery_days"] > 0).all()
    assert len(out) == (raw["orders"]["order_status"] == "delivered").sum()
    assert "order_delivered_carrier_date" not in out          # không leakage


def test_geolocation_drops_points_outside_brazil(raw):
    geo = data_prep.preprocess_geolocation(raw["geolocation"])
    assert geo["lat"].between(*data_prep.BR_LAT).all()
    assert geo["zip_code_prefix"].is_unique


def test_products_fill_missing(raw):
    p = data_prep.preprocess_products(raw["products"])
    assert p.isna().sum().sum() == 0
    assert (p["product_category_name"] == "unknown").sum() == 1


def test_haversine_known_distance():
    # São Paulo -> Rio de Janeiro ~ 360 km
    d = data_prep.haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 340 < d < 380


def test_build_dataset_and_time_split(raw):
    df = data_prep.build_dataset(raw, outlier_quantile=1.0)
    assert df["order_id"].is_unique
    assert df["order_purchase_timestamp"].is_monotonic_increasing
    tr, va, te = data_prep.time_split(df, 0.15, 0.15)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["order_purchase_timestamp"].max() <= te["order_purchase_timestamp"].min()


def test_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["MAE"] == 0 and m["R2"] == 1


# ------------------------------------------------------------------ end-to-end + inference
def test_full_pipeline(project):
    tmp, cfg_path, cfg = project
    data_prep.main(cfg_path)
    metrics = train.main(cfg_path)
    assert metrics["val"]["MAE"] < metrics["val_baseline_olist_estimate"]["MAE"]
    res = evaluate.main(cfg_path)
    assert (tmp / "metrics.json").exists() and res["n_test"] > 0

    bundle = joblib.load(tmp / "model.joblib")
    test = pd.read_csv(tmp / "processed" / "test.csv")
    preds = predict.predict(test, bundle)
    assert preds.shape == (len(test),) and (preds >= 0).all()

    # dự đoán 1 đơn chỉ với thông tin thô + danh mục chưa từng thấy
    one = predict.predict_one({
        "order_purchase_timestamp": "2018-06-01 10:00:00",
        "order_estimated_delivery_date": "2018-06-20",
        "customer_lat": -12.9, "customer_lng": -38.5, "customer_state": "BA",
        "seller_lat": -23.5, "seller_lng": -46.6, "seller_state": "SP",
        "total_price": 100, "total_freight": 20, "product_category_name": "never_seen",
    }, bundle)
    assert np.isfinite(one) and one > 0

