"""Cleaning-rule tests."""

import numpy as np
import pandas as pd

from src.data.cleaning import clean_building, clean_dataset


def test_duplicates_merged_and_grid_complete(clean_data):
    clean, report = clean_data
    for _, group in clean.groupby("building_id"):
        assert not group["timestamp"].duplicated().any()
        deltas = group["timestamp"].sort_values().diff().dropna()
        assert (deltas == pd.Timedelta(hours=1)).all()
    assert report.totals()["duplicate_timestamps_merged"] > 0


def test_negatives_removed(clean_data):
    clean, report = clean_data
    assert (clean["meter_reading"].dropna() >= 0).all()
    assert report.totals()["negatives_nullified"] > 0


def test_short_gaps_interpolated_long_gaps_kept():
    idx = pd.date_range("2023-01-01", periods=24 * 10, freq="h")
    values = np.full(len(idx), 50.0)
    values[30:33] = np.nan     # 3-hour gap -> interpolated
    values[100:130] = np.nan   # 30-hour gap -> left missing
    df = pd.DataFrame({"timestamp": idx, "meter_reading": values})
    cleaned, actions = clean_building(df)
    assert cleaned["meter_reading"].iloc[30:33].notna().all()
    assert cleaned["meter_reading"].iloc[110:120].isna().all()
    assert actions["values_interpolated"] >= 3
    assert actions["values_left_missing"] >= 20


def test_imputed_flag_marks_only_filled_hours():
    idx = pd.date_range("2023-01-01", periods=48, freq="h")
    values = np.full(48, 10.0)
    values[5:7] = np.nan
    df = pd.DataFrame({"timestamp": idx, "meter_reading": values})
    cleaned, _ = clean_building(df)
    assert cleaned["imputed"].sum() == 2
    assert cleaned["imputed"].iloc[5] and cleaned["imputed"].iloc[6]


def test_spikes_preserved_for_anomaly_detection():
    idx = pd.date_range("2023-01-01", periods=100, freq="h")
    values = np.full(100, 20.0)
    values[50] = 900.0  # extreme spike must survive cleaning
    df = pd.DataFrame({"timestamp": idx, "meter_reading": values})
    cleaned, _ = clean_building(df)
    assert cleaned["meter_reading"].max() == 900.0
