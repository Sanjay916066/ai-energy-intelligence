"""Synthetic industrial energy dataset generator.

Produces BDG2-shaped data (long-format hourly meter readings + building
metadata + hourly weather) with realistic industrial load patterns and
*deliberately injected* data-quality issues and anomalies, so the whole
pipeline (quality profiling, cleaning, forecasting, anomaly detection,
recommendations) can be demonstrated offline and tested deterministically.

Injected issues per building (recorded in the returned ``ground_truth``):
- missing blocks (sensor outage)
- duplicate timestamps with conflicting values
- negative glitch readings
- extreme point spikes
- abnormal night-consumption episodes
- stuck-at-zero runs during operating hours
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import SETTINGS

PRIMARY_USES = ["Manufacturing", "Warehouse", "Office", "Laboratory", "Food processing"]


@dataclass
class SyntheticDataset:
    meters: pd.DataFrame          # columns: timestamp, building_id, meter_reading
    metadata: pd.DataFrame        # building_id, site_id, primary_use, sqm, year_built
    weather: pd.DataFrame         # timestamp, air_temperature
    ground_truth: pd.DataFrame = field(default_factory=pd.DataFrame)  # injected events


def _weather(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Smooth seasonal + diurnal temperature signal (deg C)."""
    hours = np.arange(len(index))
    day_of_year = index.dayofyear.to_numpy()
    seasonal = 10.0 - 12.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    diurnal = 4.0 * np.sin(2 * np.pi * (index.hour.to_numpy() - 9) / 24)
    noise = rng.normal(0, 1.2, len(hours))
    return pd.DataFrame(
        {"timestamp": index, "air_temperature": (seasonal + diurnal + noise).round(2)}
    )


def _base_profile(
    index: pd.DatetimeIndex,
    temperature: np.ndarray,
    rng: np.random.Generator,
    sqm: float,
    primary_use: str,
) -> np.ndarray:
    """Hourly kWh profile: base load + shift load + temperature response + noise."""
    hour = index.hour.to_numpy()
    weekday = index.weekday.to_numpy()
    op_start, op_end = SETTINGS.business.operating_hours

    scale = sqm / 1000.0
    base = 12.0 * scale                                   # 24/7 standby consumption
    in_shift = ((hour >= op_start) & (hour < op_end) & (weekday < 5)).astype(float)
    # ramp up/down at shift edges
    ramp = 0.5 * (((hour == op_start) | (hour == op_end - 1)) & (weekday < 5))
    shift_intensity = {"Manufacturing": 55.0, "Food processing": 48.0,
                       "Laboratory": 40.0, "Office": 30.0, "Warehouse": 20.0}[primary_use]
    production = shift_intensity * scale * (in_shift - ramp)
    saturday = 0.25 * shift_intensity * scale * ((weekday == 5) & (hour >= 8) & (hour < 14))

    # electric heating/cooling response (V-shaped around 16 degC)
    thermal = 0.9 * scale * np.abs(temperature - 16.0)

    noise = rng.normal(0, 1.5 * scale, len(index))
    load = base + production + saturday + thermal + noise
    return np.clip(load, 0.5 * scale, None)


def _inject_issues(
    df: pd.DataFrame, rng: np.random.Generator, building_id: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Corrupt one building's series in-place-ish; return events for ground truth."""
    events: list[dict] = []
    n = len(df)
    values = df["meter_reading"].to_numpy().copy()
    op_start, op_end = SETTINGS.business.operating_hours

    def log(kind: str, start: int, hours: int) -> None:
        events.append(
            {"building_id": building_id, "type": kind,
             "start": df["timestamp"].iloc[start], "hours": hours}
        )

    # 1) missing blocks (2 outages of 4-36 h)
    for _ in range(2):
        start = int(rng.integers(24, n - 48))
        length = int(rng.integers(4, 36))
        values[start : start + length] = np.nan
        log("missing_block", start, length)

    # 2) extreme spikes (3 single-hour surges)
    for _ in range(3):
        i = int(rng.integers(0, n))
        values[i] = values[i] * rng.uniform(6, 10) if np.isfinite(values[i]) else values[i]
        log("spike", i, 1)

    # 3) night-consumption episode: equipment left running for 3 nights
    night_start = int(rng.integers(n // 4, n // 2))
    hours = df["timestamp"].dt.hour.to_numpy()
    mask = np.zeros(n, dtype=bool)
    mask[night_start : night_start + 72] = True
    night_mask = mask & ((hours >= 22) | (hours < 6))
    values[night_mask] = values[night_mask] * 3.5
    log("night_episode", night_start, 72)

    # 4) stuck-at-zero during operating hours (meter/equipment fault, 6 h)
    weekday = df["timestamp"].dt.weekday.to_numpy()
    op_mask = (hours >= op_start) & (hours < op_end) & (weekday < 5)
    op_idx = np.flatnonzero(op_mask[: n - 12])
    z = int(rng.choice(op_idx))
    values[z : z + 6] = 0.0
    log("zero_run", z, 6)

    # 5) negative glitches
    for _ in range(2):
        i = int(rng.integers(0, n))
        values[i] = -abs(values[i]) if np.isfinite(values[i]) else -1.0
        log("negative", i, 1)

    out = df.copy()
    out["meter_reading"] = values

    # 6) duplicate timestamps with conflicting values (raw-data artefact)
    dup_rows = out.iloc[rng.integers(0, n, size=5)].copy()
    dup_rows["meter_reading"] = dup_rows["meter_reading"] * rng.uniform(0.9, 1.1, 5)
    for ts in dup_rows["timestamp"]:
        events.append({"building_id": building_id, "type": "duplicate_timestamp",
                       "start": ts, "hours": 0})
    out = pd.concat([out, dup_rows], ignore_index=True)
    return out, events


def generate(
    n_buildings: int = 8,
    start: str = "2023-01-01",
    end: str = "2023-12-31 23:00",
    seed: int = 42,
    inject_issues: bool = True,
) -> SyntheticDataset:
    """Generate the full synthetic dataset. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h")
    weather = _weather(index, rng)
    temperature = weather["air_temperature"].to_numpy()

    meta_rows, frames, all_events = [], [], []
    for i in range(n_buildings):
        building_id = f"Site_A_Bldg_{i + 1:02d}"
        primary_use = PRIMARY_USES[i % len(PRIMARY_USES)]
        sqm = float(rng.integers(2_000, 20_000))
        meta_rows.append(
            {"building_id": building_id, "site_id": "Site_A",
             "primary_use": primary_use, "sqm": sqm,
             "year_built": int(rng.integers(1975, 2018))}
        )
        load = _base_profile(index, temperature, rng, sqm, primary_use)
        df = pd.DataFrame(
            {"timestamp": index, "building_id": building_id,
             "meter_reading": np.round(load, 3)}
        )
        if inject_issues:
            df, events = _inject_issues(df, rng, building_id)
            all_events.extend(events)
        frames.append(df)

    meters = pd.concat(frames, ignore_index=True)
    # shuffle rows slightly so the raw file is not perfectly ordered (realistic)
    meters = meters.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    metadata = pd.DataFrame(meta_rows)
    ground_truth = pd.DataFrame(all_events)
    return SyntheticDataset(meters, metadata, weather, ground_truth)
