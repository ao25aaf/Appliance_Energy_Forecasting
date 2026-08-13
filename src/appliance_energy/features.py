"""
Feature engineering for the XGBoost feature-based model.

Matches the logic in notebooks/05_feature_based_models.ipynb exactly. Three feature
groups, all built with strict no-leakage rules: lag and rolling features always use
.shift(1) or later, so no feature ever sees the current or future target value.
"""

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 168]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/day-of-week/weekend features plus their cyclical sin-cos encodings."""
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_rolling_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Add lag and rolling mean/std features on `target`, always shifted so no feature
    ever sees the current or future target value.
    """
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out[target].shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = out[target].shift(1)
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
    return out


def make_feature_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build the full supervised-learning feature table (time + lag + rolling features)
    and drop the warm-up rows left with missing values.
    """
    out = add_time_features(df)
    out = add_lag_rolling_features(out, target=target)
    return out.dropna()


def feature_groups(feature_cols):
    """
    Group feature column names into interpretable categories (lag, rolling, time,
    indoor sensor, outdoor weather, other) for aggregated importance analysis.
    """
    groups = {
        "lag": [], "rolling": [], "time": [],
        "sensor_indoor": [], "weather_outdoor": [], "other": [],
    }

    for col in feature_cols:
        if col.startswith("lag_"):
            groups["lag"].append(col)
        elif col.startswith("roll_"):
            groups["rolling"].append(col)
        elif col in (
            "hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ):
            groups["time"].append(col)
        elif col in ("T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"):
            # Checked before the broader indoor-sensor rule below, since T_out would
            # otherwise incorrectly match startswith("T") meant for T1..T9.
            groups["weather_outdoor"].append(col)
        elif col.startswith("T") or (col.startswith("RH_") and col != "RH_out"):
            groups["sensor_indoor"].append(col)
        else:
            groups["other"].append(col)

    return groups
