"""Data-quality profiling tests (RQ1)."""

import numpy as np
import pandas as pd

from src.data.quality import profile_building, profile_dataset


def _perfect_series(hours: int = 24 * 30) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=hours, freq="h")
    return pd.DataFrame({"timestamp": idx, "meter_reading": 100.0 + np.arange(hours) % 10})


def test_perfect_data_scores_high():
    meta = pd.Series({"site_id": "A", "primary_use": "Office", "sqm": 1000, "year_built": 2000})
    result = profile_building(_perfect_series(), meta)
    assert result["missing_rate"] == 0.0
    assert result["duplicate_timestamps"] == 0
    assert result["negative_count"] == 0
    assert result["quality_score"] >= 99.0
    assert result["rating"] == "good"


def test_missing_values_detected():
    df = _perfect_series()
    df.loc[10:59, "meter_reading"] = np.nan
    result = profile_building(df)
    assert result["missing_rate"] > 0.05


def test_duplicates_and_negatives_detected():
    df = _perfect_series()
    df = pd.concat([df, df.head(5)], ignore_index=True)  # 5 duplicate timestamps
    df.loc[3, "meter_reading"] = -50.0
    result = profile_building(df)
    assert result["duplicate_timestamps"] == 5
    assert result["negative_count"] == 1


def test_gap_detection():
    df = _perfect_series()
    df = df.drop(index=range(100, 130))  # 30-hour hole
    result = profile_building(df)
    assert result["gap_count"] >= 1
    assert result["longest_gap_hours"] >= 29


def test_extreme_peaks_detected():
    df = _perfect_series()
    df.loc[200, "meter_reading"] = 100_000.0
    result = profile_building(df)
    assert result["extreme_peak_count"] >= 1


def test_scorecard_covers_all_buildings(dataset):
    scorecard = profile_dataset(dataset.meters, dataset.metadata)
    assert set(scorecard["building_id"]) == set(dataset.metadata["building_id"])
    assert scorecard["quality_score"].between(0, 100).all()
    # injected issues must be visible
    assert (scorecard["duplicate_timestamps"] > 0).all()
    assert (scorecard["negative_count"] > 0).all()
