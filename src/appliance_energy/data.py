"""
Data loading and cleaning for the Appliances Energy Prediction dataset.

Matches the logic in notebooks/01_data_download_and_cleaning.ipynb exactly, extracted
here so it can be unit tested and reused from scripts without depending on the notebook.
"""

from pathlib import Path

import pandas as pd

TARGET = "Appliances"
RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"


def load_data(source) -> pd.DataFrame:
    """
    Download (or load from a local path) the raw 10-minute Appliances Energy
    Prediction dataset and parse its timestamp. Drops rv1/rv2, random noise columns
    included in the original dataset for feature-selection testing, not real signal.
    """
    df = pd.read_csv(source)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df = df.drop(columns=["rv1", "rv2"], errors="ignore")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Drop rows with a missing target, resample to `freq` (mean), and interpolate any
    small gaps left by resampling.
    """
    out = df.dropna(subset=[TARGET]).copy()

    resampled = out.resample(freq).mean()
    resampled = resampled.interpolate("time")
    resampled = resampled.dropna()

    return resampled


def load_and_clean(source, processed_path: Path = None, freq: str = "h") -> pd.DataFrame:
    """Full load -> clean -> (optionally save) pipeline."""
    raw = load_data(source)
    hourly = clean(raw, freq=freq)

    if processed_path is not None:
        processed_path = Path(processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(processed_path)

    return hourly
