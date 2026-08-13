"""
scripts/run_pipeline.py

A runnable entry point exercising the reusable src/appliance_energy package end to
end: download/clean the data, build the rolling 24-hour benchmark forecasts, and
evaluate them, printing a results table.

This intentionally does NOT reproduce the full analysis (SARIMAX's grid search,
XGBoost's hyperparameter search, and Chronos-2 all take real time and, for
Chronos-2, real internet access to Hugging Face). Those live in the notebooks,
which are the authoritative, fully-executed record of this project, this script
exists to prove src/ is genuinely runnable and correct on its own, and as a quick
sanity check after any change to the package.

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from appliance_energy.data import load_and_clean, RAW_URL, TARGET  # noqa: E402
from appliance_energy.models.benchmarks import rolling_benchmarks  # noqa: E402
from appliance_energy.evaluation import evaluate_forecast  # noqa: E402

import pandas as pd  # noqa: E402

TEST_STEPS = 14 * 24
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168


def main():
    processed_path = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"

    if processed_path.exists():
        print(f"Using existing processed data at {processed_path}")
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading and cleaning from {RAW_URL} ...")
        df = load_and_clean(RAW_URL, processed_path=processed_path)

    df.index.freq = "h"
    y = df[TARGET]

    train_y = y.iloc[:-TEST_STEPS]
    test_y = y.iloc[-TEST_STEPS:]
    n_days = TEST_STEPS // 24
    test_start_idx = len(y) - TEST_STEPS

    print(f"\nData: {df.shape[0]} hourly rows")
    print(f"Train: {train_y.index.min()} to {train_y.index.max()} ({len(train_y)} hours)")
    print(f"Test:  {test_y.index.min()} to {test_y.index.max()} ({len(test_y)} hours)")

    print("\nComputing rolling 24-hour benchmark forecasts...")
    forecasts = rolling_benchmarks(y, test_start_idx, n_days, DAILY_PERIOD, WEEKLY_PERIOD)

    results = [
        evaluate_forecast(name, test_y, pred, train_y, seasonality=DAILY_PERIOD)
        for name, pred in forecasts.items()
    ]
    results_df = pd.DataFrame(results).sort_values("MASE").reset_index(drop=True)

    print("\nBenchmark results (sorted by MASE, lower is better):")
    print(results_df.round(3).to_string(index=False))

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_dir / "benchmark_comparison.csv", index=False)
    print(f"\nSaved to {metrics_dir / 'benchmark_comparison.csv'}")

    print(
        "\nFor the full analysis, SARIMAX order selection, the tuned XGBoost feature "
        "model, the genuine Chronos-2 result, and the final model comparison, see "
        "notebooks/04 through notebooks/07."
    )


if __name__ == "__main__":
    main()
