"""Short-term (day-ahead) demand forecasting — answers RQ2.

Model ladder, exactly as prescribed by the case study methodology:

1. ``SeasonalNaive24``  — value 24 h earlier (the honest baseline).
2. ``MovingAverage168`` — mean of the same hour over the previous 7 days.
3. ``Ridge``            — regularised linear model on the feature matrix.
4. ``RandomForest``     — non-linear ensemble.
5. ``LightGBM``         — gradient boosting (skipped gracefully if absent).

All models are trained per building on a chronological train/test split
(last ``test_days`` held out). Baselines use only lagged actuals, ML models
use the leakage-safe feature matrix from ``src.features.engineering``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import SETTINGS
from src.models.evaluation import add_skill_vs_baseline, evaluate

logger = logging.getLogger(__name__)

try:  # optional dependency
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False


@dataclass
class ForecastResult:
    metrics: pd.DataFrame      # building_id, model, MAE, RMSE, MAPE_%, sMAPE_%, skill
    predictions: pd.DataFrame  # timestamp, building_id, actual, model, prediction
    feature_importance: pd.DataFrame  # building_id, feature, importance (tree models)


def time_split(df: pd.DataFrame, test_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: last ``test_days`` are the hold-out set."""
    test_days = test_days or SETTINGS.forecast.test_days
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]


def _make_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=4,
            random_state=random_state, n_jobs=-1,
        ),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            random_state=random_state, verbosity=-1,
        )
    return models


def _baseline_predictions(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Baselines computed from lag features already present in the matrix."""
    preds = {"SeasonalNaive24": test["lag_24h"].to_numpy()}
    week_lags = [c for c in ("lag_24h", "lag_48h", "lag_72h", "lag_168h") if c in test.columns]
    preds["MovingAverage168"] = test[week_lags].mean(axis=1).to_numpy()
    return preds


def run_forecasting(
    features: pd.DataFrame,
    feature_cols: list[str],
    test_days: int | None = None,
) -> ForecastResult:
    """Train and evaluate all models for every building."""
    fc = SETTINGS.forecast
    metric_rows, pred_frames, importance_rows = [], [], []

    for building_id, group in features.groupby("building_id", sort=True):
        group = group.sort_values("timestamp")
        train, test = time_split(group, test_days)
        # exclude imputed hours from evaluation so metrics reflect real data
        eval_mask = ~test["imputed"].to_numpy() if "imputed" in test.columns else np.ones(len(test), bool)
        if len(train) < 24 * 21 or len(test) < 24:
            logger.warning("Skipping %s: not enough data", building_id)
            continue

        y_train = train["meter_reading"].to_numpy()
        y_test = test["meter_reading"].to_numpy()
        X_train, X_test = train[feature_cols], test[feature_cols]

        all_preds = _baseline_predictions(test)
        for name, model in _make_models(fc.random_state).items():
            model.fit(X_train, y_train)
            all_preds[name] = model.predict(X_test)
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                for f, v in zip(feature_cols, imp / (imp.sum() or 1)):
                    importance_rows.append(
                        {"building_id": building_id, "model": name,
                         "feature": f, "importance": round(float(v), 4)}
                    )

        for name, preds in all_preds.items():
            preds = np.clip(preds, 0, None)
            metric_rows.append(
                {"building_id": building_id, "model": name,
                 **evaluate(y_test[eval_mask], preds[eval_mask])}
            )
            pred_frames.append(pd.DataFrame({
                "timestamp": test["timestamp"].to_numpy(),
                "building_id": building_id,
                "actual": y_test,
                "model": name,
                "prediction": np.round(preds, 3),
            }))

    metrics = add_skill_vs_baseline(pd.DataFrame(metric_rows))
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    importance = pd.DataFrame(importance_rows)
    return ForecastResult(metrics, predictions, importance)


def best_model_per_building(metrics: pd.DataFrame) -> pd.DataFrame:
    """Lowest-MAE model per building — used by the dashboard summary."""
    idx = metrics.groupby("building_id")["MAE"].idxmin()
    return metrics.loc[idx].reset_index(drop=True)
