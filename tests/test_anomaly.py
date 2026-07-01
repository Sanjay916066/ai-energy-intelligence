"""Anomaly-detection tests (RQ3) — validated against injected ground truth."""

import numpy as np
import pandas as pd

from src.models.anomaly import (
    _run_lengths,
    detect_anomalies,
    detect_building,
    summarize_episodes,
)


def test_run_lengths():
    mask = np.array([False, True, True, True, False, True, False])
    lengths = _run_lengths(mask)
    assert list(lengths) == [0, 3, 3, 3, 0, 1, 0]


def _flat_series(hours: int = 24 * 60, level: float = 50.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    values = level + rng.normal(0, 2.0, hours)
    return pd.DataFrame({"timestamp": idx, "meter_reading": values})


def test_spike_detected():
    df = _flat_series()
    df.loc[500, "meter_reading"] = 5000.0
    found = detect_building(df)
    spikes = found[found["detector"] == "spike"]
    assert df.loc[500, "timestamp"] in set(spikes["timestamp"])


def test_zero_run_detected_during_operating_hours():
    df = _flat_series()
    # Tuesday 09:00 onwards, 6 zero hours
    target = df.index[(df["timestamp"].dt.weekday == 1) & (df["timestamp"].dt.hour == 9)][1]
    df.loc[target : target + 5, "meter_reading"] = 0.0
    found = detect_building(df)
    assert (found["detector"] == "zero_run").any()


def test_flatline_detected():
    df = _flat_series()
    df.loc[300:340, "meter_reading"] = 42.0  # 41 identical readings
    found = detect_building(df)
    assert (found["detector"] == "flatline").any()


def test_detects_injected_ground_truth(dataset, clean_data):
    """Detector recall on the synthetic ground-truth night episodes."""
    clean, _ = clean_data
    anomalies = detect_anomalies(clean)
    assert not anomalies.empty
    assert anomalies["severity"].between(0, 100).all()

    truth = dataset.ground_truth
    night_events = truth[truth["type"] == "night_episode"]
    for _, ev in night_events.iterrows():
        window = anomalies[
            (anomalies["building_id"] == ev["building_id"])
            & (anomalies["timestamp"] >= ev["start"])
            & (anomalies["timestamp"] <= ev["start"] + pd.Timedelta(hours=ev["hours"]))
        ]
        assert len(window) > 0, f"missed night episode in {ev['building_id']}"


def test_episode_summary(clean_data):
    clean, _ = clean_data
    anomalies = detect_anomalies(clean)
    episodes = summarize_episodes(anomalies)
    assert len(episodes) <= len(anomalies)
    assert (episodes["hours"] >= 1).all()
    assert (episodes["end"] >= episodes["start"]).all()
    # ranked by severity, worst first
    assert episodes["peak_severity"].is_monotonic_decreasing
