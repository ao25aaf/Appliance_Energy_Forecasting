"""
Parts 5 and 6 feature-based model: XGBoost fit on the engineered feature table.

Matches notebooks/05_feature_based_models.ipynb. The tuned hyperparameters already
found by RandomizedSearchCV are used as the default; tune() re-runs that search
(takes several minutes) if a fresh one is requested.
"""

TUNED_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.01,
    subsample=0.8, colsample_bytree=1.0,
)


def tune(X_train, y_train, random_state=0):
    """
    Re-run the RandomizedSearchCV + TimeSeriesSplit hyperparameter search (not run
    by default from the pipeline script, takes several minutes). TimeSeriesSplit is
    used specifically so validation folds never precede their training data.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    param_distributions = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    base_model = XGBRegressor(random_state=random_state, n_jobs=1)

    search = RandomizedSearchCV(
        base_model, param_distributions=param_distributions, n_iter=15,
        scoring="neg_mean_absolute_error", cv=tscv, random_state=random_state, n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def fit_feature_model(X_train, y_train, params=None, random_state=0):
    """Fit XGBoost with the given (or tuned-default) hyperparameters."""
    from xgboost import XGBRegressor

    params = params or TUNED_PARAMS
    model = XGBRegressor(**params, random_state=random_state, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Point forecast from a fitted feature model. No rolling loop needed, the
    feature table's lag/rolling columns are already built from real history at
    every hour, so a single batch prediction is already a genuine rolling
    24-hour forecast by construction (see notebooks/05 for the full argument)."""
    import pandas as pd
    return pd.Series(model.predict(X_test), index=X_test.index, name="feature_model")
