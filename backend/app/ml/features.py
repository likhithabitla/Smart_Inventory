"""
Feature engineering pipeline shared by training and prediction.

WHY ONE GLOBAL MODEL (not one model per store):
  1. Cold start: a brand-new store has zero or few rows. A per-store model
     would have nothing to learn from and would produce garbage predictions
     until months of data accumulate. A global model transfers patterns
     learned from OTHER stores (seasonality, promotion effects, weather
     sensitivity) to new stores immediately.
  2. Shared structure: units_sold responses to discounts/promotions/weather
     follow broadly similar retail dynamics across stores. Pooling data
     gives LightGBM far more rows to learn these general relationships from,
     which reduces variance versus fitting a small tree ensemble per store.
  3. Operational simplicity: one model file, one training job, one set of
     metrics to monitor - instead of N models drifting out of sync, each
     needing separate retraining schedules and separate quality monitoring.
  4. Store-specific behavior is NOT lost - see point 2 below, store_id is
     itself an input feature, so the model can and does learn store-specific
     offsets/interactions (e.g. "store S7 systematically sells 20% more of
     category X during promotions") directly from that categorical feature
     combined with LightGBM's native categorical splits.

WHY STORE_ID / PRODUCT_ID ENCODING IS NECESSARY:
  Without identity features, the model can only learn generic relationships
  ("more discount -> more demand") but cannot distinguish store A's downtown
  location from store B's suburban location, or a fast-moving SKU from a
  slow one. Including store_id and product_id as categorical features lets
  gradient boosting split on them directly, effectively learning a per-
  store/per-product baseline and interaction effects (e.g. store_id x
  weather_condition), while still sharing the rest of the tree structure
  across all stores. This is the standard "entity embedding via categorical
  splits" pattern for global boosted-tree models on panel/multi-entity data.

LEAKAGE AVOIDANCE:
  Lag and rolling features are computed using values STRICTLY BEFORE the
  current row's date (shift(1) before any rolling window), for each
  store_id + product_id group independently, sorted by date. This guarantees
  we never use same-day or future demand to predict the current day.
"""
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = [
    "store_id", "product_id", "category", "region",
    "weather_condition", "seasonality",
]

NUMERIC_COLUMNS = [
    "inventory_level", "units_ordered", "price", "discount",
    "promotion", "competitor_pricing", "epidemic",
]

LAG_FEATURE_COLUMNS = [
    "demand_lag_1", "demand_lag_7", "rolling_mean_7", "rolling_mean_30",
]

TARGET_COLUMN = "demand"


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds demand_lag_1, demand_lag_7, rolling_mean_7, rolling_mean_30,
    computed per (store_id, product_id) group, sorted by date, using only
    PAST values (shift(1) applied before rolling) to avoid leakage.
    """
    df = df.sort_values(["store_id", "product_id", "date"]).copy()
    grp = df.groupby(["store_id", "product_id"])["demand"]

    df["demand_lag_1"] = grp.shift(1)
    df["demand_lag_7"] = grp.shift(7)

    # rolling means computed on the already-shifted (past-only) series
    shifted = grp.shift(1)
    df["rolling_mean_7"] = (
        shifted.groupby([df["store_id"], df["product_id"]])
        .rolling(window=7, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["rolling_mean_30"] = (
        shifted.groupby([df["store_id"], df["product_id"]])
        .rolling(window=30, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # New product/store combos with no history yet: fall back to 0 (the
    # model also sees store_id/product_id/category so it can still learn
    # a sane baseline for cold-start items).
    for col in LAG_FEATURE_COLUMNS:
        df[col] = df[col].fillna(0.0)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if df[col].notna().any() else 0)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: Dict[str, Dict[str, int]] = None
                         ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Label-encodes categorical columns into integer codes that LightGBM
    treats as native categorical features (via categorical_feature=...).
    If `encoders` is provided (inference time), reuse the existing mapping
    and map unseen categories to a reserved "unknown" code so prediction
    never crashes on a brand-new store/product/category value.
    """
    df = df.copy()
    fitted = encoders is None
    if fitted:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if fitted:
            uniques = sorted(df[col].astype(str).unique().tolist())
            mapping = {val: i + 1 for i, val in enumerate(uniques)}  # 0 reserved for unknown
            mapping["__unknown__"] = 0
            encoders[col] = mapping
        mapping = encoders[col]
        df[col + "_enc"] = df[col].astype(str).map(mapping).fillna(mapping.get("__unknown__", 0)).astype(int)

    return df, encoders


def get_feature_columns() -> List[str]:
    return (
        [c + "_enc" for c in CATEGORICAL_COLUMNS]
        + NUMERIC_COLUMNS
        + LAG_FEATURE_COLUMNS
        + ["day_of_week", "day_of_month", "month", "week_of_year", "is_weekend"]
    )


def build_training_frame(raw_df: pd.DataFrame, encoders: Dict = None
                          ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Full pipeline: missing-value handling -> lag features -> time features
    -> categorical encoding. Returns (feature_df, encoders).
    """
    df = handle_missing_values(raw_df)
    df = build_lag_features(df)
    df = add_time_features(df)
    df, encoders = encode_categoricals(df, encoders)
    return df, encoders


def build_inference_row(single_row: dict, encoders: Dict[str, Dict[str, int]],
                         history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a single-row feature frame for /predict.

    `history_df` should be that store+product's most recent uploaded rows
    (already sorted by date) so lag/rolling features reflect real recent
    history instead of defaulting to 0 for existing products.
    """
    row = dict(single_row)
    df_row = pd.DataFrame([row])
    df_row = handle_missing_values(df_row)
    df_row = add_time_features(df_row)

    # Compute lag/rolling features from real history if we have it
    lag_1 = lag_7 = roll_7 = roll_30 = 0.0
    if history_df is not None and len(history_df) > 0:
        hist = history_df.sort_values("date")
        demand_series = hist["demand"].dropna()
        if len(demand_series) >= 1:
            lag_1 = float(demand_series.iloc[-1])
        if len(demand_series) >= 7:
            lag_7 = float(demand_series.iloc[-7])
        roll_7 = float(demand_series.tail(7).mean()) if len(demand_series) > 0 else 0.0
        roll_30 = float(demand_series.tail(30).mean()) if len(demand_series) > 0 else 0.0

    df_row["demand_lag_1"] = lag_1
    df_row["demand_lag_7"] = lag_7
    df_row["rolling_mean_7"] = roll_7
    df_row["rolling_mean_30"] = roll_30

    df_row, _ = encode_categoricals(df_row, encoders)
    return df_row
