# Olist E-Commerce Delivery Time Prediction

## 1. Bài toán
Dự đoán thời gian giao hàng thực tế (`delivery_time_days`) **tại thời điểm khách hàng đặt hàng (Purchase-time Prediction)** trên bộ dữ liệu công khai *Brazilian E-Commerce Public Dataset by Olist* (96,470 đơn hàng `delivered` hợp lệ).

## 2. Kết quả chính (Holdout Test Set)
| Mô hình | Test MAE (ngày) | Test RMSE (ngày) | Test R² | Trong ±3 ngày | Trong ±5 ngày |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`LightGBM_L1_Tuned` (Best)** | **3.880** | **5.268** | **0.070** | **47.6%** | **71.2%** |
| `XGBoost_L1_Tuned` | 3.913 | 5.300 | 0.058 | 46.8% | 71.8% |
| `Ridge_Regression` | 4.337 | 5.900 | -0.167 | 45.8% | 66.3% |
| `Baseline_Median` | 5.087 | 6.284 | -0.324 | 27.7% | 52.1% |
| `Baseline_Olist_Estimated` | 10.722 | 13.112 | -4.764 | 14.1% | 23.0% |

## 3. Cấu trúc thư mục
```text
MachineLearning/
├── eda.ipynb               # Khám phá dữ liệu, đánh giá Baseline Olist & xuất biểu đồ EDA
├── preprocess.ipynb        # Làm sạch, Purchase-time Feature Engineering, Chronological TE & Split
├── train.ipynb             # Thí nghiệm Encoding, Ablation, TimeSeriesSplit Tuning, Quantile & Evaluation
├── REPORT.md               # Báo cáo đồ án đầy đủ 7 chương
├── requirements.txt        # Phiên bản thư viện đã pin (Python 3.11)
├── raw/                    # 9 file CSV gốc từ Kaggle (không commit lên Git)
├── processed/              # train.csv, val.csv, test.csv, cutoff_*.csv, feature_metadata.json
├── model/                  # best_model.joblib, te_maps.json, các bảng CSV kết quả
├── eda_figures/            # Biểu đồ chương EDA
└── model_figures/          # Biểu đồ đánh giá & phân tích lỗi mô hình
```

## 4. Cài đặt môi trường
```bash
conda create -n ml python=3.11 -y
conda activate ml
pip install -r MachineLearning/requirements.txt
```

## 5. Tải dữ liệu
Tải bộ dữ liệu [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) từ Kaggle, giải nén toàn bộ 9 file `.csv` vào thư mục `MachineLearning/raw/`.

## 6. Thứ tự chạy lại toàn bộ pipeline
Mở các notebook trong thư mục `MachineLearning/` và chạy lần lượt từ đầu đến cuối:
1. `eda.ipynb`
2. `preprocess.ipynb`
3. `train.ipynb`

## 7. Báo cáo chi tiết
Xem toàn bộ nội dung báo cáo 7 chương tại [REPORT.md](REPORT.md).
