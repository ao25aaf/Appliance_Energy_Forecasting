"""
Evaluation metrics: MAE, RMSE, MASE, Bias.

Matches the evaluate_forecast/mase/mae/rmse/bias functions used identically across
notebooks 03, 04, 05, 06, and 07, so every model in this project is scored the same way.
"""

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true, y_pred) -> float:
    """Mean signed error; positive means the model over-forecasts on average."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def mase(y_true, y_pred, y_train, seasonality: int = 24) -> float:
    """
    Mean absolute scaled error, scaled by the in-sample seasonal naive error.
    Below 1.0 means the model beats the seasonal naive benchmark.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonality:
        return np.nan

    seasonal_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(seasonal_errors)

    if scale == 0:
        return np.nan

    model_mae = np.mean(np.abs(y_true - y_pred))
    return float(model_mae / scale)


def evaluate_forecast(model_name, y_true, y_pred, y_train, seasonality: int = 24) -> dict:
    """Evaluate a forecast using MAE, RMSE, MASE, and bias."""
    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, index=y_true.index, dtype=float)

    valid_mask = y_true.notna() & y_pred.notna()
    y_true_valid = y_true.loc[valid_mask]
    y_pred_valid = y_pred.loc[valid_mask]

    return {
        "model": model_name,
        "MAE": mae(y_true_valid, y_pred_valid),
        "RMSE": rmse(y_true_valid, y_pred_valid),
        "MASE": mase(y_true_valid, y_pred_valid, y_train, seasonality=seasonality),
        "Bias": bias(y_true_valid, y_pred_valid),
        "n_points": int(valid_mask.sum()),
    }
