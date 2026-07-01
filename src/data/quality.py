"""Data-quality profiling (answers RQ1: "How reliable is the available data?").

Produces a per-building scorecard covering every dimension named in the case
study: missingness, duplicate timestamps, gaps, zero readings, unrealistic
peaks, negative values and metadata completeness, aggregated into a 0-100
quality score with a traffic-light rating.

Profiling runs on the *raw* data — before cleaning — so the scorecard is
evidence of what the cleaning step had to repair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SETTINGS

METADATA_FIELDS = ["site_id", "primary_use", "sqm", "year_built"]

# weights of each penalty dimension in the 0-100 score
_SCORE_WEIGHTS = {
    "missing_rate": 40.0,
    "duplicate_rate": 15.0,
    "gap_rate": 15.0,
    "zero_rate": 10.0,
    "extreme_rate": 10.0,
    "negative_rate": 5.0,
    "metadata_incompleteness": 5.0,
}


def _gap_stats(ts: pd.Series) -> tuple[int, int]:
    """(number of gaps > 1h, longest gap in hours) on a sorted unique series."""
    if len(ts) < 2:
        return 0, 0
    deltas = ts.diff().dropna().dt.total_seconds() / 3600.0
    gaps = deltas[deltas > 1.0]
    if gaps.empty:
        return 0, 0
    return int(len(gaps)), int(gaps.max() - 1)


def profile_building(df: pd.DataFrame, metadata_row: pd.Series | None = None) -> dict:
    """Quality metrics for one building's raw slice (timestamp, meter_reading)."""
    q = SETTINGS.quality
    df = df.sort_values("timestamp")
    values = df["meter_reading"]

    n_rows = len(df)
    dup_mask = df["timestamp"].duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())
    unique_ts = df.loc[~dup_mask, "timestamp"]

    expected = pd.date_range(unique_ts.min(), unique_ts.max(), freq=q.expected_freq)
    n_expected = len(expected)
    n_present_valid = int(values[~dup_mask].notna().sum())
    missing_rate = 1.0 - n_present_valid / n_expected if n_expected else 1.0

    gap_count, longest_gap = _gap_stats(unique_ts.reset_index(drop=True))

    finite = values[np.isfinite(values)]
    zero_rate = float((finite == 0).mean()) if len(finite) else 0.0
    n_negative = int((finite < 0).sum())

    # robust extreme-peak detection: modified z-score via MAD, plus a hard
    # multiple-of-median guard for meters whose MAD is ~0
    positive = finite[finite > 0]
    if len(positive) >= 24:
        med = float(positive.median())
        mad = float((positive - med).abs().median())
        rz = 0.6745 * (finite - med) / mad if mad > 0 else pd.Series(0.0, index=finite.index)
        extreme_mask = (rz > q.peak_zscore_threshold) | (finite > q.max_plausible_multiplier * med)
        n_extreme = int(extreme_mask.sum())
    else:
        n_extreme = 0

    if metadata_row is not None:
        present = sum(pd.notna(metadata_row.get(f)) for f in METADATA_FIELDS)
        metadata_completeness = present / len(METADATA_FIELDS)
    else:
        metadata_completeness = 0.0

    rates = {
        "missing_rate": missing_rate,
        "duplicate_rate": n_duplicates / n_rows if n_rows else 0.0,
        "gap_rate": min(1.0, longest_gap / (24 * 7)),  # week-long gap => full penalty
        "zero_rate": zero_rate,
        "extreme_rate": min(1.0, n_extreme / max(n_rows, 1) * 100),  # extremes are rare; amplify
        "negative_rate": min(1.0, n_negative / max(n_rows, 1) * 100),
        "metadata_incompleteness": 1.0 - metadata_completeness,
    }
    score = 100.0 - sum(_SCORE_WEIGHTS[k] * min(1.0, max(0.0, v)) for k, v in rates.items())
    score = round(max(0.0, score), 1)

    return {
        "n_rows": n_rows,
        "n_expected_hours": n_expected,
        "missing_rate": round(missing_rate, 4),
        "duplicate_timestamps": n_duplicates,
        "gap_count": gap_count,
        "longest_gap_hours": longest_gap,
        "zero_rate": round(zero_rate, 4),
        "negative_count": n_negative,
        "extreme_peak_count": n_extreme,
        "metadata_completeness": round(metadata_completeness, 2),
        "quality_score": score,
        "rating": "good" if score >= 90 else ("acceptable" if score >= 75 else "poor"),
    }


def profile_dataset(meters: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scorecard with one row per building, sorted worst-first."""
    meta_idx = (
        metadata.set_index("building_id") if metadata is not None else pd.DataFrame()
    )
    rows = []
    for building_id, group in meters.groupby("building_id", sort=True):
        meta_row = meta_idx.loc[building_id] if building_id in getattr(meta_idx, "index", []) else None
        rows.append({"building_id": building_id, **profile_building(group, meta_row)})
    return (
        pd.DataFrame(rows)
        .sort_values("quality_score")
        .reset_index(drop=True)
    )
