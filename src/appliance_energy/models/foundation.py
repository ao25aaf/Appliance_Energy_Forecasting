"""
Part 7 foundation model: Chronos-2, zero-shot, target-only.

Matches notebooks/06_foundation_model.ipynb exactly. Requires `torch` and
`chronos-forecasting` to be installed, and internet access to Hugging Face to
download the pretrained weights on first use. There is deliberately no fallback
or placeholder here: if Chronos-2 can't be loaded, this raises rather than
silently substituting a different model under the same name. The assignment
brief asks for a foundation model, not a foundation model or a substitute.
"""


def load_chronos_pipeline():
    """Load the Chronos-2 pipeline once, reused across every rolling forecast call."""
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.float32,
    )


def chronos_forecast(pipeline, context_series, horizon):
    """
    A single zero-shot forecast call given an already-loaded pipeline. Returns a
    numpy array of `horizon` point forecasts (the predicted mean).
    """
    import torch

    context = torch.tensor(context_series.values, dtype=torch.float32)
    _, mean = pipeline.predict_quantiles(inputs=[context], prediction_length=horizon)
    return mean[0].squeeze().numpy()


def rolling_chronos_forecast(pipeline, y, test_start_idx, n_days, horizon=24):
    """
    Genuine rolling 24-hour Chronos-2 forecast: n_days separate zero-shot calls,
    each using only real history available at that day's origin (never the model's
    own prior output). Returns a pandas Series indexed to match the test period.
    """
    import pandas as pd

    all_preds = []
    for day in range(n_days):
        origin = test_start_idx + day * horizon
        history = y.iloc[:origin]
        index = y.index[origin:origin + horizon]
        day_forecast = chronos_forecast(pipeline, history, horizon=horizon)
        all_preds.append(pd.Series(day_forecast, index=index))

    return pd.concat(all_preds)
