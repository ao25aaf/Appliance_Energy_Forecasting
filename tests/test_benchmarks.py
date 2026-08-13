import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.models.benchmarks import (  # noqa: E402
    mean_forecast, naive_forecast, seasonal_naive_forecast, drift_forecast,
    all_benchmarks, rolling_forecast, rolling_benchmarks,
)


def _train_and_index():
    train = pd.Series(np.arange(1, 101, dtype=float))
    index = pd.RangeIndex(100, 105)
    return train, index


def test_forecast_lengths_match_test_period():
    train, index = _train_and_index()
    for fc in [
        mean_forecast(train, index),
        naive_forecast(train, index),
        seasonal_naive_forecast(train, index, seasonality=24, name="x"),
        drift_forecast(train, index),
    ]:
        assert len(fc) == len(index)


def test_naive_forecast_repeats_last_value():
    train, index = _train_and_index()
    fc = naive_forecast(train, index)
    assert (fc == train.iloc[-1]).all()


def test_all_benchmarks_returns_five_models():
    train = pd.Series(np.arange(1, 501, dtype=float))
    index = pd.RangeIndex(500, 505)
    result = all_benchmarks(train, index)
    assert set(result.keys()) == {
        "mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift"
    }


def test_rolling_forecast_uses_only_true_history_never_own_output():
    """The core no-leakage guarantee behind the rolling 24-hour design: each day's
    forecast must be built from real, already-observed values, never a value the
    function itself produced for an earlier day."""
    # A series where every value is unique makes it easy to tell where a forecast
    # value actually came from.
    y = pd.Series(np.arange(0, 500, dtype=float), index=pd.RangeIndex(500))
    test_start_idx = 400
    n_days = 4

    result = rolling_forecast(naive_forecast, y, test_start_idx, n_days, horizon=24)

    assert len(result) == n_days * 24
    # Day 2's naive forecast should equal the true value at the end of day 1
    # (index 423, i.e. one before day 2 starts at 424), not anything invented.
    day_2_forecast_value = result.iloc[24]
    assert day_2_forecast_value == y.iloc[423]


def test_rolling_benchmarks_returns_five_models_with_correct_length():
    y = pd.Series(np.sin(np.linspace(0, 50, 1000)) + 100, index=pd.RangeIndex(1000))
    test_start_idx = 800
    n_days = 5  # 120 hours, well within the 800-observation history

    result = rolling_benchmarks(y, test_start_idx, n_days, daily_period=24, weekly_period=168)

    assert set(result.keys()) == {
        "mean", "naive", "seasonal_naive_daily", "seasonal_naive_weekly", "drift"
    }
    for name, forecast in result.items():
        assert len(forecast) == n_days * 24, f"{name} has wrong length"
