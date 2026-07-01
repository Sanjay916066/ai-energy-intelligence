"""Feature-engineering tests, incl. the leakage guard."""

import pandas as pd

from src.config import SETTINGS
from src.features.engineering import add_calendar_features, build_feature_matrix


def test_calendar_features():
    df = pd.DataFrame({"timestamp": pd.date_range("2023-06-05", periods=48, freq="h")})
    out = add_calendar_features(df)
    assert out["hour"].iloc[0] == 0
    assert out["weekday"].iloc[0] == 0          # Monday
    assert out["is_weekend"].iloc[0] == 0
    ops, ope = SETTINGS.business.operating_hours
    assert out.loc[out["hour"] == ops, "is_operating_hour"].iloc[0] == 1
    assert out.loc[out["hour"] == 23, "is_operating_hour"].iloc[0] == 0


def test_no_leakage_all_lags_at_least_horizon(clean_data):
    clean, _ = clean_data
    horizon = SETTINGS.forecast.horizon_hours
    _, feature_cols = build_feature_matrix(clean, horizon=horizon)
    lag_hours = [int(c.split("_")[1].rstrip("h")) for c in feature_cols if c.startswith("lag_")]
    assert lag_hours, "expected lag features"
    assert min(lag_hours) >= horizon


def test_feature_matrix_is_finite(clean_data, dataset):
    clean, _ = clean_data
    feats, cols = build_feature_matrix(clean, dataset.weather)
    assert not feats[cols].isna().any().any()
    assert not feats["meter_reading"].isna().any()
    assert "air_temperature" in cols


def test_lag_values_match_shifted_target(clean_data):
    clean, _ = clean_data
    feats, _ = build_feature_matrix(clean)
    one = feats[feats["building_id"] == feats["building_id"].iloc[0]].set_index("timestamp")
    ts = one.index[500]
    expected = one.loc[ts - pd.Timedelta(hours=24), "meter_reading"]
    assert one.loc[ts, "lag_24h"] == expected
