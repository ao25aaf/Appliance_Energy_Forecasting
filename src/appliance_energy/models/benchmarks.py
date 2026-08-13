"""
Part 3 benchmark models, plus the rolling 24-hour forecast wrapper used by every
model in this project (Part 3's brief explicitly asks for a 24-hour horizon).

Matches notebooks/03_benchmark_models.ipynb exactly.
"""

import pandas as pd


def mean_forecast(y_train, index):
    """Forecast every horizon step as the training-period mean."""
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train, index):
    """Forecast every horizon step as the last observed training value."""
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train, index, seasonality, name):
    """
    Forecast by repeating the value from `seasonality` steps ago, rolling forward
    recursively once the training history runs out.
    """
    if len(y_train) < seasonality:
        raise ValueError("There exist no season in training data")

    horizon = len(index)
    history = list(y_train.values)
    values = []

    for _ in range(horizon):
        seasonal_value = history[-seasonality]
        values.append(seasonal_value)
        history.append(seasonal_value)

    return pd.Series(values, index=index, name=name)


def drift_forecast(y_train, index):
    """Forecast a straight line extrapolated from the first to the last training value."""
    if len(y_train) < 2:
        raise ValueError("Drift forecast requires at least two training observations.")

    horizon = len(index)
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)
    values = [y_train.iloc[-1] + slope * step for step in range(1, horizon + 1)]

    return pd.Series(values, index=index, name="drift")


def all_benchmarks(y_train, index, daily_period=24, weekly_period=168):
    """
    Compute all five benchmark forecasts (mean, naive, daily/weekly seasonal naive,
    drift) and return them as a dict keyed by model name.
    """
    return {
        "mean": mean_forecast(y_train, index),
        "naive": naive_forecast(y_train, index),
        "seasonal_naive_daily": seasonal_naive_forecast(
            y_train, index, daily_period, "seasonal_naive_daily",
        ),
        "seasonal_naive_weekly": seasonal_naive_forecast(
            y_train, index, weekly_period, "seasonal_naive_weekly",
        ),
        "drift": drift_forecast(y_train, index),
    }


def rolling_forecast(forecast_fn, y, test_start_idx, n_days, horizon=24, **kwargs):
    """
    Apply a single-shot forecast function repeatedly, once per test day, using only
    genuinely known history at each day's origin (never the model's own prior output).

    This is what makes every benchmark a genuine rolling 24-hour forecast rather than
    one long multi-step forecast: forecast_fn must have signature
    (y_train, index, **kwargs) -> pd.Series, matching the functions above.
    """
    parts = []
    for day in range(n_days):
        origin = test_start_idx + day * horizon
        history = y.iloc[:origin]
        index = y.index[origin:origin + horizon]
        parts.append(forecast_fn(history, index, **kwargs))
    return pd.concat(parts)


def rolling_benchmarks(y, test_start_idx, n_days, daily_period=24, weekly_period=168, horizon=24):
    """Convenience wrapper: all five benchmarks, each as a genuine rolling 24-hour forecast."""
    return {
        "mean": rolling_forecast(mean_forecast, y, test_start_idx, n_days, horizon=horizon),
        "naive": rolling_forecast(naive_forecast, y, test_start_idx, n_days, horizon=horizon),
        "seasonal_naive_daily": rolling_forecast(
            seasonal_naive_forecast, y, test_start_idx, n_days, horizon=horizon,
            seasonality=daily_period, name="seasonal_naive_daily",
        ),
        "seasonal_naive_weekly": rolling_forecast(
            seasonal_naive_forecast, y, test_start_idx, n_days, horizon=horizon,
            seasonality=weekly_period, name="seasonal_naive_weekly",
        ),
        "drift": rolling_forecast(drift_forecast, y, test_start_idx, n_days, horizon=horizon),
    }
