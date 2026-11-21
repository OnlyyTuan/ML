"""
Huấn luyện mô hình và lưu file model.

    python -m src.train [--config configs/config.yaml]
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils import load_config, regression_metrics, resolve


def build_pipeline(cfg: dict) -> Pipeline:
    num, cat = cfg["features"]["numeric"], cfg["features"]["categorical"]
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), cat),
         ("num", "passthrough", num)],
        verbose_feature_names_out=False,
    )
    # sau ColumnTransformer, các cột categorical đứng đầu
    model = HistGradientBoostingRegressor(
        categorical_features=list(range(len(cat))), **cfg["model"]["params"])
    return Pipeline([("pre", pre), ("model", model)])


def main(config_path=None):
    cfg = load_config(config_path)
    proc = resolve(cfg["paths"]["processed_dir"])
    feats = cfg["features"]["categorical"] + cfg["features"]["numeric"]
    target = cfg["features"]["target"]

    train = pd.read_csv(proc / "train.csv")
    val = pd.read_csv(proc / "val.csv")

    pipe = build_pipeline(cfg)
    pipe.fit(train[feats], train[target])

    metrics = {
        "train": regression_metrics(train[target], pipe.predict(train[feats])),
        "val": regression_metrics(val[target], pipe.predict(val[feats])),
        # baseline: thời gian ước tính Olist đưa ra cho khách
        "val_baseline_olist_estimate": regression_metrics(val[target], val["estimated_days"]),
    }

    model_path = resolve(cfg["paths"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": feats, "target": target}, model_path)
    print(f"Saved model -> {model_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
