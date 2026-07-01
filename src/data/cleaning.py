"""Cleaning / repair of raw meter data.

Documented, conservative rules (every action is counted in the returned
``CleaningReport`` so the technical report can state exactly what was done):

1. Coerce timestamps, sort, and average duplicate timestamps.
2. Re-index every building to a complete hourly grid.
3. Replace negative readings with NaN (physically impossible for consumption).
4. Linearly interpolate gaps up to ``max_interpolation_gap_hours``; longer
   gaps stay NaN and remain visible to downstream steps.
5. Add an ``imputed`` flag so imputed hours can be excluded from model
   evaluation and highlighted in the dashboard.

Deliberately NOT removed here: extreme spikes and zero-runs — those are
candidate *anomalies* (RQ3), not data errors, so they are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import SETTINGS


@dataclass
class CleaningReport:
    per_building: dict[str, dict] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(self.per_building, orient="index")
            .rename_axis("building_id")
            .reset_index()
        )

    def totals(self) -> dict:
        frame = self.to_frame()
        numeric = frame.select_dtypes("number")
        return {k: int(v) for k, v in numeric.sum().items()}


def clean_building(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean one building's slice; returns (hourly frame, action counts)."""
    max_gap = SETTINGS.cleaning.max_interpolation_gap_hours
    actions: dict[str, int] = {}

    out = df[["timestamp", "meter_reading"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    actions["bad_timestamps_dropped"] = int(out["timestamp"].isna().sum())
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    n_before = len(out)
    out = out.groupby("timestamp", as_index=False)["meter_reading"].mean()
    actions["duplicate_timestamps_merged"] = n_before - len(out)

    full_index = pd.date_range(
        out["timestamp"].min(), out["timestamp"].max(), freq="h"
    )
    series = out.set_index("timestamp")["meter_reading"].reindex(full_index)
    actions["grid_hours_added"] = len(full_index) - len(out)

    negatives = series < 0
    actions["negatives_nullified"] = int(negatives.sum())
    series[negatives] = pd.NA

    was_nan = series.isna()
    series = series.astype("float64").interpolate(
        method="linear", limit=max_gap, limit_area="inside"
    )
    imputed = was_nan & series.notna()
    actions["values_interpolated"] = int(imputed.sum())
    actions["values_left_missing"] = int(series.isna().sum())

    result = pd.DataFrame(
        {"timestamp": full_index, "meter_reading": series.to_numpy(), "imputed": imputed.to_numpy()}
    )
    return result, actions


def clean_dataset(meters: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean all buildings; returns long frame + report."""
    report = CleaningReport()
    frames = []
    for building_id, group in meters.groupby("building_id", sort=True):
        cleaned, actions = clean_building(group)
        cleaned.insert(1, "building_id", building_id)
        report.per_building[str(building_id)] = actions
        frames.append(cleaned)
    return pd.concat(frames, ignore_index=True), report
