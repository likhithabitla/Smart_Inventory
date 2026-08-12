"""
Model training.

MODEL CHOICE: LightGBM over XGBoost
  - Native categorical feature support (categorical_feature=[...]) avoids
    one-hot blow-up for store_id/product_id, which can have high cardinality
    as the number of stores/SKUs grows. XGBoost's categorical support is
    newer/more restrictive across versions.
  - Histogram-based split finding trains noticeably faster on the kind of
    wide, growing, mixed-type panel data this app accumulates, which matters
    since /train is triggered manually by shop owners and should feel
    responsive rather than kicking off a long batch job.
  - Leaf-wise growth typically gets better accuracy than XGBoost's level-wise
    default at equivalent training time on tabular data of this shape
    (moderate rows, mixed categorical/numeric features).
  - First-class `sample_weight` support, which is required here for the
    exponential recency-decay weighting.
  Both libraries would work; LightGBM was chosen for the categorical-feature
  ergonomics and training speed given this app's "retrain on demand" UX.
"""
import os
import json
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from app.config import settings
from app.ml.features import build_training_frame, get_feature_columns, TARGET_COLUMN, CATEGORICAL_COLUMNS
from app.ml.weighting import compute_sample_weights, compute_lambda

MODEL_FILE_TEMPLATE = "global_model_{version}.joblib"
ENCODERS_FILE_TEMPLATE = "encoders_{version}.joblib"
LATEST_POINTER_FILE = "latest.json"


def _model_dir() -> str:
    os.makedirs(settings.model_dir, exist_ok=True)
    return settings.model_dir


def train_global_model(all_rows_df: pd.DataFrame, triggering_store_id: str) -> Dict:
    """
    Trains ONE global LightGBM model on ALL rows currently in inventory_data
    (across every store), using store_id/product_id as categorical features
    so store-specific patterns are still captured (see features.py docstring
    for the full rationale). Called whenever any store owner hits /train, so
    the shared model continuously improves as any store uploads new data.

    Returns metadata dict including metrics and the model_version string.
    """
    if len(all_rows_df) < 2:
        raise ValueError("Not enough data across all stores to train a model.")

    df = all_rows_df.dropna(subset=[TARGET_COLUMN]).copy()
    if len(df) < 2:
        raise ValueError("No rows with a non-null 'demand' target to train on.")

    feature_df, encoders = build_training_frame(df, encoders=None)
    feature_cols = get_feature_columns()
    cat_feature_names = [c + "_enc" for c in CATEGORICAL_COLUMNS]

    X = feature_df[feature_cols]
    y = feature_df[TARGET_COLUMN].astype(float)

    # --- exponential recency-decay sample weights ---------------------
    reference_date = pd.to_datetime(feature_df["date"]).max()
    weights = compute_sample_weights(
        feature_df["date"], reference_date, settings.decay_half_life_days
    )

    # Train/validation split (time-agnostic random split is fine here since
    # recency weighting - not chronological holdout - is how we handle
    # recency; a random split gives a more stable MAE estimate on small
    # per-store datasets than a strict time split would).
    if len(X) >= 10:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train, w_train = X, y, weights
        X_val, y_val = X, y

    train_set = lgb.Dataset(X_train, label=y_train, weight=w_train,
                             categorical_feature=cat_feature_names, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": max(5, len(X_train) // 50),
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    preds_val = booster.predict(X_val)
    preds_val = np.clip(preds_val, 0, None)  # demand can't be negative
    mae = float(mean_absolute_error(y_val, preds_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))

    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    model_dir = _model_dir()

    joblib.dump(booster, os.path.join(model_dir, MODEL_FILE_TEMPLATE.format(version=version)))
    joblib.dump(encoders, os.path.join(model_dir, ENCODERS_FILE_TEMPLATE.format(version=version)))

    with open(os.path.join(model_dir, LATEST_POINTER_FILE), "w") as f:
        json.dump({"version": version}, f)

    return {
        "model_version": version,
        "rows_used_total": int(len(df)),
        "rows_used_this_store": int((df["store_id"] == triggering_store_id).sum()),
        "train_mae": mae,
        "train_rmse": rmse,
        "trained_at": datetime.utcnow(),
        "lambda_decay": float(compute_lambda(settings.decay_half_life_days)),
        "half_life_days": settings.decay_half_life_days,
    }


def load_latest_model() -> Tuple[lgb.Booster, Dict, str]:
    model_dir = _model_dir()
    pointer_path = os.path.join(model_dir, LATEST_POINTER_FILE)
    if not os.path.exists(pointer_path):
        raise FileNotFoundError("No trained model yet. Call POST /train first.")
    with open(pointer_path) as f:
        version = json.load(f)["version"]

    booster = joblib.load(os.path.join(model_dir, MODEL_FILE_TEMPLATE.format(version=version)))
    encoders = joblib.load(os.path.join(model_dir, ENCODERS_FILE_TEMPLATE.format(version=version)))
    return booster, encoders, version
