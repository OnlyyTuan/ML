# Dự đoán thời gian giao hàng – Olist Brazilian E-Commerce

- **Dataset:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Loại bài toán:** Học có giám sát – Hồi quy
- **Input:** thông tin đơn hàng, sản phẩm, người bán, khách hàng, địa lý
- **Output:** `delivery_days` – số ngày từ lúc đặt hàng đến khi khách nhận hàng
- **Mô hình:** `HistGradientBoostingRegressor` (scikit-learn), loss MAE
- **Baseline:** thời gian giao ước tính Olist hiển thị cho khách (`estimated_days`) và median của tập train

## Cấu trúc

```
MachineLearning/
├── data/
│   ├── raw/            # 6 file CSV gốc của Olist (không chỉnh sửa)
│   ├── processed/      # train.csv / val.csv / test.csv
│   └── external/       # dữ liệu bổ sung
├── notebooks/
│   ├── 01_eda.ipynb        # Khám phá dữ liệu (EDA)
│   └── 02_baseline.ipynb   # Giai đoạn 1: baseline & kiểm tra dự đoán
├── src/
│   ├── data_prep.py    # tiền xử lý, trích xuất đặc trưng, chia train/val/test
│   ├── train.py        # huấn luyện, lưu models/delivery_model.joblib
│   ├── evaluate.py     # đánh giá trên test, lưu models/metrics.json
│   ├── predict.py      # tải mô hình, dự đoán dữ liệu mới
│   └── utils.py        # đọc config, metric
├── models/
├── configs/config.yaml # đường dẫn, danh sách feature, siêu tham số
├── tests/test_pipeline.py
├── requirements.txt
└── README.md
```

## Cách chạy

Yêu cầu **Python 3.11**. Các lệnh chạy từ thư mục `MachineLearning/`.

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Tải dataset từ Kaggle và giải nén 6 file sau vào `data/raw/`: `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_customers_dataset.csv`, `olist_sellers_dataset.csv`, `olist_products_dataset.csv`, `olist_geolocation_dataset.csv`.

```bash
python -m src.data_prep
```
Giai đoạn 1 – Baseline: mở và chạy `notebooks/02_baseline.ipynb`.

```bash
python -m src.train
```
```bash
python -m src.evaluate
```
```bash
python -m src.predict --input data/processed/test.csv --output predictions.csv
```
```bash
pytest -q
```

## Tiền xử lý

| File | Xử lý |
|---|---|
| `orders` | Chỉ giữ đơn `delivered`; target `delivery_days`; bỏ ≤ 0 và top 0.5%; đặc trưng tháng, thứ, giờ, cuối tuần, `approval_hours`, `estimated_days`. Bỏ `order_delivered_carrier_date` vì leakage. |
| `order_items` | Gộp theo đơn: số món, số sản phẩm, số người bán, tổng/max giá, phí ship, `freight_ratio`. Người bán chính = món đắt nhất. |
| `products` | Điền thiếu bằng median theo danh mục; thể tích = dài×cao×rộng; tổng cân nặng/thể tích của đơn. |
| `geolocation` | Bỏ toạ độ ngoài Brazil; median lat/lng theo zip prefix. |
| `customers` / `sellers` | Gắn toạ độ theo zip, thiếu thì lấy trung bình bang. |
| Ghép | `distance_km` (Haversine), `same_state`, `same_city`. |

Dữ liệu được chia **theo thời gian đặt hàng** (70% train / 15% val / 15% test) để mô phỏng việc dự đoán đơn hàng tương lai.
