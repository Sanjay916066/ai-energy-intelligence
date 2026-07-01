"""Forecasting tests (RQ2)."""

import pandas as pd

from src.config import SETTINGS
from src.features.engineering import build_feature_matrix
from src.models.forecasting import best_model_per_building, run_forecasting, time_split


def test_time_split_is_chronological(clean_data):
    clean, _ = clean_data
    one = clean[clean["building_id"] == clean["building_id"].iloc[0]]
    train, test = time_split(one, test_days=14)
    assert train["timestamp"].max() < test["timestamp"].min()
    span_hours = (test["timestamp"].max() - test["timestamp"].min()) / pd.Timedelta(hours=1)
    assert span_hours <= 14 * 24


def test_forecasting_end_to_end(clean_data, dataset):
    clean, _ = clean_data
    feats, cols = build_feature_matrix(clean, dataset.weather)
    result = run_forecasting(feats, cols, test_days=7)

    required_metrics = {"MAE", "RMSE", "MAPE_%", "sMAPE_%", "skill_vs_baseline_%"}
    assert required_metrics <= set(result.metrics.columns)
    assert {"SeasonalNaive24", "MovingAverage168", "Ridge", "RandomForest"} <= set(
        result.metrics["model"]
    )
    # predictions are non-negative and aligned to the test window
    assert (result.predictions["prediction"] >= 0).all()
    assert result.predictions["timestamp"].nunique() <= 7 * 24

    # at least one ML model beats the naive baseline on every building
    ml = result.metrics[~result.metrics["model"].isin(["SeasonalNaive24", "MovingAverage168"])]
    assert (ml.groupby("building_id")["skill_vs_baseline_%"].max() > 0).all()


def test_best_model_selection():
    metrics = pd.DataFrame([
        {"building_id": "B1", "model": "A", "MAE": 5.0},
        {"building_id": "B1", "model": "B", "MAE": 3.0},
        {"building_id": "B2", "model": "A", "MAE": 1.0},
        {"building_id": "B2", "model": "B", "MAE": 2.0},
    ])
    best = best_model_per_building(metrics)
    chosen = best.set_index("building_id")["model"]
    assert chosen["B1"] == "B" and chosen["B2"] == "A"


def test_baseline_is_true_seasonal_naive(clean_data, dataset):
    clean, _ = clean_data
    feats, cols = build_feature_matrix(clean, dataset.weather)
    result = run_forecasting(feats, cols, test_days=7)
    naive = result.predictions[result.predictions["model"] == "SeasonalNaive24"]
    b = naive["building_id"].iloc[0]
    one_clean = clean[clean["building_id"] == b].set_index("timestamp")["meter_reading"]
    row = naive[naive["building_id"] == b].iloc[10]
    assert row["prediction"] == max(0.0, round(one_clean[row["timestamp"] - pd.Timedelta(hours=24)], 3))
