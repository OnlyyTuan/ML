"""Tiện ích dùng chung: đọc config, đường dẫn, metric."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p) -> Path:
    """Đường dẫn tương đối trong config được tính từ thư mục gốc dự án."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "within_3_days": float(np.mean(np.abs(err) <= 3)),
    }
