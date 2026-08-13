import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.features import (  # noqa: E402
    make_feature_table, add_lag_rolling_features, feature_groups,
)


def _make_dummy_df(n=300):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Appliances": np.arange(n, dtype=float),
        "T_out": np.random.RandomState(0).normal(10, 2, n),
        "T1": np.random.RandomState(1).normal(20, 1, n),
        "RH_1": np.random.RandomState(2).normal(40, 5, n),
    }, index=idx)
    return df


def test_lag_features_do_not_use_future_target_values():
    df = _make_dummy_df()
    out = add_lag_rolling_features(df, target="Appliances")

    valid = out["lag_1"].dropna()
    shifted_true = df["Appliances"].shift(1).loc[valid.index]
    assert np.allclose(valid.values, shifted_true.values)


def test_rolling_features_use_only_past_observations():
    df = _make_dummy_df()
    out = add_lag_rolling_features(df, target="Appliances")

    row = out.iloc[50]
    window_vals = df["Appliances"].iloc[47:50]  # i-3 .. i-1, excludes i
    assert abs(row["roll_mean_3"] - window_vals.mean()) < 1e-9


def test_feature_table_has_no_missing_target_values():
    df = _make_dummy_df()
    ml_data = make_feature_table(df, target="Appliances")
    assert ml_data["Appliances"].isna().sum() == 0


def test_feature_groups_classifies_known_columns_correctly():
    cols = ["lag_1", "roll_mean_3", "hour", "is_weekend", "T1", "RH_1", "T_out", "Windspeed"]
    groups = feature_groups(cols)

    assert "lag_1" in groups["lag"]
    assert "roll_mean_3" in groups["rolling"]
    assert "hour" in groups["time"]
    assert "is_weekend" in groups["time"]
    assert "T1" in groups["sensor_indoor"]
    assert "RH_1" in groups["sensor_indoor"]
    assert "T_out" in groups["weather_outdoor"]
    assert "Windspeed" in groups["weather_outdoor"]
