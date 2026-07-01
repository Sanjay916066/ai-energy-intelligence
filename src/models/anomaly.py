"""Anomaly detection — answers RQ3 ("Can abnormal consumption be detected?").

Hybrid strategy required by the case study: transparent *rule-based* flags
combined with an *ML detector* (Isolation Forest), merged and ranked by a
severity score with operational context so a facility manager can triage.

Detectors
---------
- ``spike``      robust z-score (MAD) vs the same hour-of-week profile
- ``night_load`` night consumption far above the building's typical night load
- ``zero_run``   >= N consecutive zero readings during operating hours
- ``flatline``   constant non-zero value for many hours (stuck sensor)
- ``ml_outlier`` Isolation Forest on consumption + context features

Severity = weighted blend of deviation magnitude and detector confidence,
scaled 0-100. Anomalies with multiple detector hits rank higher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import SETTINGS

DETECTOR_WEIGHTS = {
    "spike": 1.0,
    "night_load": 0.9,
    "zero_run": 0.8,
    "flatline": 0.6,
    "ml_outlier": 0.5,
}


def _hour_of_week_profile(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Median and MAD of consumption per hour-of-week (168 slots)."""
    how = df["timestamp"].dt.weekday * 24 + df["timestamp"].dt.hour
    grouped = df["meter_reading"].groupby(how)
    med = grouped.median()
    mad = grouped.apply(lambda s: float((s - s.median()).abs().median()))
    return med.reindex(range(168)).ffill(), mad.reindex(range(168)).ffill()


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """For each True position, length of the maximal True run containing it."""
    lengths = np.zeros(len(mask), dtype=int)
    start = None
    for i, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            lengths[start:i] = i - start
            start = None
    return lengths


def detect_building(df: pd.DataFrame) -> pd.DataFrame:
    """All detectors for one building's clean hourly slice."""
    ap = SETTINGS.anomaly
    op_start, op_end = SETTINGS.business.operating_hours
    df = df.sort_values("timestamp").reset_index(drop=True)
    values = df["meter_reading"].to_numpy()
    hours = df["timestamp"].dt.hour.to_numpy()
    weekdays = df["timestamp"].dt.weekday.to_numpy()
    finite = np.isfinite(values)

    med_prof, mad_prof = _hour_of_week_profile(df.loc[finite])
    how = weekdays * 24 + hours
    expected = med_prof.to_numpy()[how]
    mad = np.maximum(mad_prof.to_numpy()[how], 1e-6)
    rz = 0.6745 * (values - expected) / mad

    hits: list[pd.DataFrame] = []

    def collect(mask: np.ndarray, kind: str, strength: np.ndarray) -> None:
        mask = mask & finite
        if not mask.any():
            return
        hits.append(pd.DataFrame({
            "timestamp": df["timestamp"].to_numpy()[mask],
            "value": values[mask],
            "expected": np.round(expected[mask], 3),
            "detector": kind,
            "strength": np.round(np.clip(strength[mask], 0, 10), 3),
        }))

    # 1) point spikes
    collect(rz > ap.spike_zscore, "spike", rz / ap.spike_zscore)

    # 2) abnormal night load
    night = (hours >= SETTINGS.business.night_hours[0]) | (hours < SETTINGS.business.night_hours[1])
    night_typical = np.nanmedian(values[night & finite]) if (night & finite).any() else 0.0
    if night_typical > 0:
        ratio = values / night_typical
        collect(night & (ratio > ap.night_load_factor), "night_load", ratio / ap.night_load_factor)

    # 3) zero runs during operating hours
    operating = (hours >= op_start) & (hours < op_end) & (weekdays < 5)
    zero_mask = operating & finite & (values <= 1e-9)
    runs = _run_lengths(zero_mask)
    collect(runs >= ap.zero_run_hours, "zero_run", runs / ap.zero_run_hours)

    # 4) flatline (stuck sensor): identical non-zero value repeated
    same_as_prev = np.zeros(len(values), dtype=bool)
    same_as_prev[1:] = finite[1:] & finite[:-1] & (np.abs(np.diff(values)) < 1e-9) & (values[1:] > 0)
    flat_runs = _run_lengths(same_as_prev)
    collect(flat_runs >= ap.flatline_hours, "flatline", flat_runs / ap.flatline_hours)

    # 5) Isolation Forest on context features
    feats = pd.DataFrame({
        "value": values, "hour": hours, "weekday": weekdays,
        "residual": values - expected, "rz": rz,
    }).loc[finite]
    if len(feats) > 100:
        iso = IsolationForest(
            contamination=ap.isolation_forest_contamination,
            random_state=ap.random_state, n_estimators=200,
        )
        labels = iso.fit_predict(feats)
        scores = -iso.score_samples(feats)  # higher = more anomalous
        ml_mask = np.zeros(len(values), dtype=bool)
        ml_mask[feats.index[labels == -1]] = True
        strength = np.zeros(len(values))
        strength[feats.index] = scores / max(scores.max(), 1e-9) * 2
        collect(ml_mask, "ml_outlier", strength)

    if not hits:
        return pd.DataFrame(columns=["timestamp", "value", "expected", "detector", "strength"])
    return pd.concat(hits, ignore_index=True)


def detect_anomalies(clean: pd.DataFrame) -> pd.DataFrame:
    """Run all detectors per building; merge multi-detector hits and rank.

    Returns one row per (building, timestamp) with the detectors that fired,
    a 0-100 severity, and an operational context label.
    """
    frames = []
    for building_id, group in clean.groupby("building_id", sort=True):
        found = detect_building(group)
        if not found.empty:
            found.insert(0, "building_id", building_id)
            frames.append(found)
    if not frames:
        return pd.DataFrame(columns=[
            "building_id", "timestamp", "value", "expected",
            "detectors", "n_detectors", "severity", "context",
        ])

    raw = pd.concat(frames, ignore_index=True)
    raw["weighted"] = raw.apply(
        lambda r: DETECTOR_WEIGHTS[r["detector"]] * min(r["strength"], 3.0), axis=1
    )
    merged = (
        raw.groupby(["building_id", "timestamp"])
        .agg(
            value=("value", "first"),
            expected=("expected", "first"),
            detectors=("detector", lambda s: "+".join(sorted(set(s)))),
            n_detectors=("detector", "nunique"),
            weighted=("weighted", "sum"),
        )
        .reset_index()
    )
    max_w = merged["weighted"].max()
    merged["severity"] = (merged["weighted"] / max_w * 100).round(1) if max_w > 0 else 0.0
    merged["context"] = merged["timestamp"].apply(_context_label)
    return (
        merged.drop(columns="weighted")
        .sort_values("severity", ascending=False)
        .reset_index(drop=True)
    )


def _context_label(ts: pd.Timestamp) -> str:
    op_start, op_end = SETTINGS.business.operating_hours
    if ts.weekday() >= 5:
        return "weekend"
    if ts.hour >= 22 or ts.hour < 6:
        return "night"
    if op_start <= ts.hour < op_end:
        return "operating hours"
    return "off-shift"


def summarize_episodes(anomalies: pd.DataFrame, gap_hours: int = 3) -> pd.DataFrame:
    """Group consecutive anomalous hours into operational *episodes* so the
    dashboard shows '3-night equipment-left-on episode', not 20 raw rows."""
    if anomalies.empty:
        return pd.DataFrame(columns=[
            "building_id", "start", "end", "hours", "detectors", "peak_severity", "context",
        ])
    rows = []
    for building_id, group in anomalies.groupby("building_id"):
        group = group.sort_values("timestamp")
        gap = group["timestamp"].diff() > pd.Timedelta(hours=gap_hours)
        for _, ep in group.groupby(gap.cumsum()):
            rows.append({
                "building_id": building_id,
                "start": ep["timestamp"].min(),
                "end": ep["timestamp"].max(),
                "hours": len(ep),
                "detectors": "+".join(sorted(set("+".join(ep["detectors"]).split("+")))),
                "peak_severity": float(ep["severity"].max()),
                "context": ep["context"].mode().iloc[0],
            })
    return (
        pd.DataFrame(rows)
        .sort_values("peak_severity", ascending=False)
        .reset_index(drop=True)
    )
