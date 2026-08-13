import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appliance_energy.evaluation import mase, evaluate_forecast, mae, rmse, bias  # noqa: E402


def test_mase_zero_for_perfect_forecast():
    y_train = pd.Series(np.sin(np.linspace(0, 20, 300)) + 10)
    y_true = pd.Series([5.0, 6.0, 7.0], index=pd.RangeIndex(3))
    y_pred = y_true.copy()

    result = mase(y_true, y_pred, y_train, seasonality=24)
    assert abs(result) < 1e-9


def test_mase_below_one_means_better_than_seasonal_naive():
    # A noisy-but-seasonal training series gives a non-zero seasonal error scale
    # (a perfectly repeating pattern would make consecutive seasonal differences
    # zero, which is an edge case MASE explicitly guards against, not what this
    # test is checking).
    rng = np.random.RandomState(0)
    base = np.tile([1.0, 2.0, 3.0, 4.0], 30)
    y_train = pd.Series(base + rng.normal(0, 0.1, len(base)))
    y_true = pd.Series([10.0, 20.0, 30.0], index=pd.RangeIndex(3))
    y_pred_good = pd.Series([10.5, 20.5, 30.5], index=pd.RangeIndex(3))  # small error
    y_pred_bad = pd.Series([100.0, 200.0, 300.0], index=pd.RangeIndex(3))  # huge error

    good = mase(y_true, y_pred_good, y_train, seasonality=24)
    bad = mase(y_true, y_pred_bad, y_train, seasonality=24)
    assert good < bad


def test_evaluate_forecast_returns_expected_keys():
    y_train = pd.Series(np.arange(100, dtype=float))
    idx = pd.RangeIndex(5)
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    y_pred = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], index=idx)

    result = evaluate_forecast("test_model", y_true, y_pred, y_train, seasonality=24)

    for key in ["model", "MAE", "RMSE", "MASE", "Bias", "n_points"]:
        assert key in result

    assert result["model"] == "test_model"
    assert result["n_points"] == 5
    assert abs(result["Bias"] - 0.5) < 1e-9


def test_mae_rmse_bias_basic_correctness():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]

    assert abs(mae(y_true, y_pred) - (1 + 0 + 1) / 3) < 1e-9
    assert abs(bias(y_true, y_pred) - (1 + 0 - 1) / 3) < 1e-9
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)  # RMSE >= MAE always holds


def test_evaluate_forecast_ignores_missing_values():
    y_train = pd.Series(np.arange(100, dtype=float))
    idx = pd.RangeIndex(4)
    y_true = pd.Series([1.0, 2.0, np.nan, 4.0], index=idx)
    y_pred = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    result = evaluate_forecast("test_model", y_true, y_pred, y_train, seasonality=24)
    assert result["n_points"] == 3
    assert abs(result["MAE"]) < 1e-9  # remaining 3 points match perfectly
