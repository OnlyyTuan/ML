
from pathlib import Path
from functools import lru_cache
import json

import joblib
import numpy as np
import pandas as pd

from preprocessing import preprocess_input, FEATURE_COLS


# 1. Đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model_pipeline.joblib"

SUMMARY_PATH = BASE_DIR / "models" / "evaluation_summary.json"


# 2. Đọc thông tin mô hình
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    evaluation_summary = json.load(f)

MODEL_INFO = evaluation_summary["stratified_final"]

MODEL_NAME = MODEL_INFO["model"]
THRESHOLD = MODEL_INFO["oof_optimal_threshold"]


# 3. Nạp mô hình
@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# 4. Dự đoán
def predict_booking(input_data):

    model = load_model()

    # Tiền xử lý dữ liệu đầu vào
    X = preprocess_input(input_data)

    # Kiểm tra đặc trưng của pipeline
    if hasattr(model, "feature_names_in_"):

        expected_cols = list(model.feature_names_in_)

        if set(expected_cols) != set(X.columns):
            raise ValueError(
                "Đặc trưng đầu vào không khớp với mô hình."
            )

        X = X[expected_cols]

    # Xác định vị trí của lớp hủy phòng (1)
    classes = list(model.classes_)
    cancel_index = classes.index(1)

    # Dự đoán xác suất
    probabilities = model.predict_proba(X)[:, cancel_index]

    # Phân loại theo threshold tối ưu
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Tạo bảng kết quả
    results = pd.DataFrame({
        "cancel_probability": probabilities,

        "not_cancel_probability": 1 - probabilities,

        "prediction": predictions,

        "prediction_label": np.where(
            predictions == 1,
            "Có nguy cơ hủy",
            "Dự đoán không hủy"
        ),

        "recommendation": np.where(
            predictions == 1,
            "Nên liên hệ khách hàng để xác nhận đặt phòng.",
            "Tiếp tục theo dõi đơn đặt phòng."
        )
    }, index=X.index)

    return results




