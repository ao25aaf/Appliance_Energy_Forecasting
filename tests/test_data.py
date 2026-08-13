import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.data import clean, TARGET  # noqa: E402


def _make_raw_10min_df(n=200):
    idx = pd.date_range("2016-01-01", periods=n, freq="10min")
    df = pd.DataFrame({
        TARGET: np.random.RandomState(0).uniform(30, 300, n),
        "T1": np.random.RandomState(1).normal(20, 1, n),
    }, index=idx)
    return df


def test_clean_resamples_to_hourly():
    df = _make_raw_10min_df(n=600)  # 100 hours of 10-min data
    out = clean(df, freq="h")
    # 10-minute data resampled to hourly should give roughly n/6 rows
    assert 90 <= len(out) <= 100


def test_clean_drops_rows_with_missing_target():
    df = _make_raw_10min_df(n=200)
    df.loc[df.index[:6], TARGET] = np.nan  # blank out the first hour entirely
    out = clean(df, freq="h")
    assert out[TARGET].isna().sum() == 0


def test_clean_has_no_missing_values_after_interpolation():
    df = _make_raw_10min_df(n=300)
    df.loc[df.index[50], "T1"] = np.nan  # a single scattered gap
    out = clean(df, freq="h")
    assert out.isna().sum().sum() == 0
