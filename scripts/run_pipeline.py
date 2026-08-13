"""
scripts/run_pipeline.py

The full analysis pipeline, runnable end to end from a fresh clone, matching the
README's 10-step description:
    1. Load or download the dataset.
    2. Clean and prepare the time series.
    3. Create time, lag, rolling, sensor, and weather features.
    4. Split the data into training and test sets.
    5. Fit benchmark models.
    6. Fit the SARIMAX model.
    7. Fit the feature-based model.
    8. Fit or call the foundation model.
    9. Evaluate all forecasts.
    10. Save forecasts, metrics, and plots.

By default this uses the already-selected SARIMAX order and already-tuned XGBoost
hyperparameters (both found once via the slower searches in notebooks 04 and 05)
rather than re-running those searches, which take significantly longer. Pass
--full-search to redo both from scratch.

The foundation model (Chronos-2) requires `torch` and `chronos-forecasting` to be
installed, and internet access to Hugging Face to download the pretrained weights.
If it can't be loaded, this script reports that clearly and continues without it,
it does not substitute a placeholder result under the foundation_model name.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --full-search
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from appliance_energy.data import load_and_clean, RAW_URL, TARGET
from appliance_energy.features import make_feature_table
from appliance_energy.models.benchmarks import rolling_benchmarks
from appliance_energy.models.sarimax import (
    BEST_ORDER, SEASONAL_ORDER, EXOG_COLS, grid_search, fit_sarimax, rolling_sarimax_forecast,
)
from appliance_energy.models.feature_models import TUNED_PARAMS, tune, fit_feature_model, predict
from appliance_energy.evaluation import evaluate_forecast

TEST_STEPS = 14 * 24
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-search", action="store_true",
                         help="Re-run the SARIMAX grid search and XGBoost hyperparameter "
                              "search from scratch (slow) instead of using the already-found "
                              "best order/params.")
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    forecast_dir = PROJECT_ROOT / "outputs" / "forecasts"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    for d in [fig_dir, forecast_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 1 & 2: load / download and clean ---
    processed_path = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"
    if processed_path.exists():
        print(f"Using existing processed data at {processed_path}")
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading and cleaning from {RAW_URL} ...")
        df = load_and_clean(RAW_URL, processed_path=processed_path)
    df.index.freq = "h"
    y = df[TARGET]

    # --- 3: feature table (time, lag, rolling, sensor, weather) ---
    print("Building feature table...")
    ml_data = make_feature_table(df, target=TARGET)
    feature_cols = [c for c in ml_data.columns if c != TARGET]

    # --- 4: train/test split ---
    train_y, test_y = y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]
    ml_train, ml_test = ml_data.iloc[:-TEST_STEPS], ml_data.iloc[-TEST_STEPS:]
    n_days = TEST_STEPS // 24
    test_start_idx = len(y) - TEST_STEPS

    print(f"\nData: {df.shape[0]} hourly rows")
    print(f"Train: {train_y.index.min()} to {train_y.index.max()} ({len(train_y)} hours)")
    print(f"Test:  {test_y.index.min()} to {test_y.index.max()} ({len(test_y)} hours)")

    forecasts = {}

    # --- 5: benchmarks ---
    print("\n[5/8] Fitting benchmark models (rolling 24h)...")
    forecasts.update(rolling_benchmarks(y, test_start_idx, n_days, DAILY_PERIOD, WEEKLY_PERIOD))

    # --- 6: SARIMAX ---
    print("[6/8] Fitting SARIMAX...")
    exog_cols = [c for c in EXOG_COLS if c in df.columns]
    X = df[exog_cols]
    train_X, test_X = X.iloc[:-TEST_STEPS], X.iloc[-TEST_STEPS:]

    order = BEST_ORDER
    if args.full_search:
        print("  Running full grid search (this takes a while)...")
        grid = grid_search(train_y)
        order = tuple(int(v) for v in grid.iloc[0][["p", "d", "q"]])
        print(f"  Selected order: {order}")

    sarimax_fit = fit_sarimax(train_y, train_X, order=order, seasonal_order=SEASONAL_ORDER)
    sarimax_mean, sarimax_ci = rolling_sarimax_forecast(sarimax_fit, y, X, test_start_idx, n_days)
    sarimax_mean.index = test_y.index
    forecasts["sarimax"] = sarimax_mean

    # --- 7: feature-based model ---
    print("[7/8] Fitting XGBoost feature model...")
    X_train_ml, y_train_ml = ml_train[feature_cols], ml_train[TARGET]
    X_test_ml = ml_test[feature_cols]

    params = TUNED_PARAMS
    if args.full_search:
        print("  Running full hyperparameter search (this takes a while)...")
        model, best_params = tune(X_train_ml, y_train_ml)
        print(f"  Best params: {best_params}")
    else:
        model = fit_feature_model(X_train_ml, y_train_ml, params=params)

    feature_pred = predict(model, X_test_ml)
    forecasts["feature_model"] = feature_pred.reindex(test_y.index)

    # --- 8: foundation model (Chronos-2), best-effort, no placeholder ---
    print("[8/8] Attempting Chronos-2 foundation model...")
    try:
        from appliance_energy.models.foundation import load_chronos_pipeline, rolling_chronos_forecast
        pipeline = load_chronos_pipeline()
        chronos_pred = rolling_chronos_forecast(pipeline, y, test_start_idx, n_days)
        chronos_pred.index = test_y.index
        forecasts["foundation_model"] = chronos_pred
        print("  Chronos-2 forecast complete.")
    except Exception as e:
        print(f"  Chronos-2 could not be run here: {type(e).__name__}: {e}")
        print("  Skipping the foundation model rather than substituting a placeholder result.")

    # --- 9: evaluate ---
    print("\nEvaluating all forecasts...")
    results = [
        evaluate_forecast(name, test_y, pred, train_y, seasonality=DAILY_PERIOD)
        for name, pred in forecasts.items()
    ]
    results_df = pd.DataFrame(results).sort_values("MASE").reset_index(drop=True)
    print("\nModel comparison (sorted by MASE, lower is better):")
    print(results_df.round(3).to_string(index=False))

    # --- 10: save forecasts, metrics, plots ---
    all_forecasts = pd.DataFrame({"actual": test_y})
    for name, pred in forecasts.items():
        all_forecasts[name] = pred
    all_forecasts.to_csv(forecast_dir / "all_forecasts.csv")
    results_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 6))
        test_y.plot(ax=ax, label="Actual", color="black", linewidth=2)
        for name, pred in forecasts.items():
            pred.plot(ax=ax, label=name, alpha=0.75, linewidth=1)
        ax.set_title("All model forecasts vs actual, test period")
        ax.set_ylabel("Appliances (Wh)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "forecast_comparison.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved plot to {fig_dir / 'forecast_comparison.png'}")
    except Exception as e:
        print(f"Could not save plot: {e}")

    print(f"\nSaved forecasts to {forecast_dir / 'all_forecasts.csv'}")
    print(f"Saved metrics to {metrics_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
