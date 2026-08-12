"""
Prediction logic. Loads the current global model + encoders, builds a
single-row feature vector for the requested store/product/date, and returns
predicted demand plus the recommended restock quantity.
"""
import numpy as np
import pandas as pd

from app.ml.features import build_inference_row, get_feature_columns
from app.ml.train_model import load_latest_model


def predict_demand(store_id: str, payload: dict, history_df: pd.DataFrame) -> dict:
    booster, encoders, version = load_latest_model()

    row = dict(payload)
    row["store_id"] = store_id  # ALWAYS from the authenticated session, never client input

    feature_row = build_inference_row(row, encoders, history_df)
    feature_cols = get_feature_columns()

    # Guard against a feature column missing on the very first prediction
    # after a schema change - fill with 0 rather than raising.
    for col in feature_cols:
        if col not in feature_row.columns:
            feature_row[col] = 0

    X = feature_row[feature_cols]
    pred = float(booster.predict(X)[0])
    pred = max(pred, 0.0)  # demand cannot be negative

    inventory_level = float(payload.get("inventory_level") or 0.0)
    restock_qty = max(pred - inventory_level, 0.0)

    return {
        "predicted_demand": round(pred, 2),
        "recommended_restock_qty": round(restock_qty, 2),
        "model_version": version,
    }
