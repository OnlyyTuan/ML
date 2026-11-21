"""
Đánh giá mô hình trên tập kiểm thử (test.csv) và so sánh với baseline.

    python -m src.evaluate [--config configs/config.yaml]
"""
import argparse
import json

import joblib
import pandas as pd

from src.utils import load_config, regression_metrics, resolve


def main(config_path=None):
    cfg = load_config(config_path)
    bundle = joblib.load(resolve(cfg["paths"]["model_path"]))
    pipe, feats, target = bundle["pipeline"], bundle["features"], bundle["target"]

    proc = resolve(cfg["paths"]["processed_dir"])
    test = pd.read_csv(proc / "test.csv")
    pred = pipe.predict(test[feats])
    train_median = pd.read_csv(proc / "train.csv")[target].median()

    results = {
        "n_test": int(len(test)),
        "model": regression_metrics(test[target], pred),
        "baseline_olist_estimate": regression_metrics(test[target], test["estimated_days"]),
        "baseline_train_median": regression_metrics(test[target], [train_median] * len(test)),
    }

    # MAE theo bang của khách hàng (10 bang nhiều đơn nhất)
    by_state = (test.assign(abs_err=(pred - test[target]).abs())
                    .groupby("customer_state")["abs_err"].agg(["count", "mean"])
                    .sort_values("count", ascending=False).head(10))
    results["mae_by_customer_state"] = by_state["mean"].round(2).to_dict()

    out = resolve(cfg["paths"]["metrics_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved metrics -> {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
