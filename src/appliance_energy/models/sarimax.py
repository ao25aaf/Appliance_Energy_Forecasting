"""
Part 4 SARIMAX model: order selection, fitting, and the rolling 24-hour forecast.

Matches notebooks/04_sarimax_models.ipynb. The AIC grid search itself lives in
grid_search() below and is NOT run by default from the pipeline script (it takes
20-100+ seconds per fit on a single core); the best order already found,
SARIMAX(1,0,3)x(1,1,1,24), is used as the default unless a fresh search is
requested.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BEST_ORDER = (1, 0, 3)
SEASONAL_ORDER = (1, 1, 1, 24)
EXOG_COLS = ["T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]


def grid_search(train_y, p_range=range(0, 4), d_range=range(0, 1), q_range=range(0, 4),
                 seasonal_order=SEASONAL_ORDER):
    """
    AIC grid search over (p, d, q), with seasonal_order held fixed. Reduced range
    by default (see notebooks/04 for why the full brief-specified p=[0,6], d=[0,2],
    q=[0,6] grid is not run by default, compute time on a single core).
    """
    from itertools import product
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rows = []
    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = SARIMAX(
                train_y, order=(p, d, q), seasonal_order=seasonal_order,
                trend="c" if d == 0 else "n",
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=75)
            aic, bic = fit.aic, fit.bic
        except Exception:
            aic, bic = np.nan, np.nan
        rows.append({"p": p, "d": d, "q": q, "AIC": aic, "BIC": bic})

    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def fit_sarimax(train_y, train_X=None, order=BEST_ORDER, seasonal_order=SEASONAL_ORDER):
    """Fit SARIMAX on the training data with optional exogenous variables."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train_y, exog=train_X, order=order, seasonal_order=seasonal_order,
        trend="c", enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=100)


def rolling_sarimax_forecast(fit, y, X, test_start_idx, n_days, horizon=24):
    """
    Genuine rolling 24-hour SARIMAX forecast: forecasts 24 hours, then reveals the
    real outcome and updates the filter state (refit=False keeps the AIC-selected
    parameters fixed rather than re-optimizing 14 times, standard practice for
    rolling-origin evaluation and far cheaper than a full refit each day).
    """
    all_means, all_cis = [], []
    current_fit = fit

    for day in range(n_days):
        origin = test_start_idx + day * horizon
        day_exog = X.iloc[origin:origin + horizon] if X is not None else None

        fc = current_fit.get_forecast(steps=horizon, exog=day_exog)
        all_means.append(fc.predicted_mean)
        all_cis.append(fc.conf_int(alpha=0.05))

        if day < n_days - 1:
            new_obs = y.iloc[origin:origin + horizon]
            new_exog = X.iloc[origin:origin + horizon] if X is not None else None
            current_fit = current_fit.append(new_obs, exog=new_exog, refit=False)

    mean_fc = pd.concat(all_means)
    ci = pd.concat(all_cis)
    return mean_fc, ci
