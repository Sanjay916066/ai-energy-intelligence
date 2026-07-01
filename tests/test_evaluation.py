"""Metric-function tests."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import add_skill_vs_baseline, evaluate, mae, mape, rmse, smape


def test_perfect_prediction():
    y = [10.0, 20.0, 30.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_known_values():
    y_true = [100.0, 200.0]
    y_pred = [110.0, 180.0]
    assert mae(y_true, y_pred) == pytest.approx(15.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(y_true, y_pred) == pytest.approx((10 + 10) / 2)


def test_mape_ignores_zero_actuals():
    assert np.isfinite(mape([0.0, 100.0], [5.0, 110.0]))


def test_nan_handling():
    assert mae([1.0, np.nan, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(0.5)


def test_evaluate_returns_all_required_metrics():
    result = evaluate([1, 2, 3], [1.1, 2.2, 2.7])
    assert set(result) == {"MAE", "RMSE", "MAPE_%", "sMAPE_%"}


def test_skill_vs_baseline():
    metrics = pd.DataFrame([
        {"building_id": "B1", "model": "SeasonalNaive24", "MAE": 10.0},
        {"building_id": "B1", "model": "RandomForest", "MAE": 5.0},
        {"building_id": "B1", "model": "BadModel", "MAE": 20.0},
    ])
    out = add_skill_vs_baseline(metrics)
    skills = out.set_index("model")["skill_vs_baseline_%"]
    assert skills["RandomForest"] == pytest.approx(50.0)
    assert skills["BadModel"] == pytest.approx(-100.0)  # honest negative reporting
