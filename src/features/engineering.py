"""Feature engineering for forecasting and anomaly detection.

All features named in the case study: hour, day, weekday/weekend, season,
rolling statistics, lag variables, and weather where available.

Leakage guard: for a forecast horizon of ``h`` hours, only lags >= h and
rolling windows shifted by ``h`` are used, so every feature is known at
prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SETTINGS

CALENDAR_FEATURES = [
    "hour", "weekday", "is_weekend", "month", "day_of_year",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_operating_hour",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + cyclical encodings from the ``timestamp`` column."""
    out = df.copy()
    ts = out["timestamp"]
    op_start, op_end = SETTINGS.business.operating_hours
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["is_operating_hour"] = (
        (out["hour"] >= op_start) & (out["hour"] < op_end) & (out["weekday"] < 5)
    ).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame, lags: tuple[int, ...], target: str = "meter_reading"
) -> pd.DataFrame:
    """Per-building lag features (df must be sorted by building, timestamp)."""
    out = df.copy()
    grouped = out.groupby("building_id", sort=False)[target]
    for lag in lags:
        out[f"lag_{lag}h"] = grouped.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...],
    shift: int,
    target: str = "meter_reading",
) -> pd.DataFrame:
    """Rolling mean/std computed on data at least ``shift`` hours old."""
    out = df.copy()
    shifted = out.groupby("building_id", sort=False)[target].shift(shift)
    for window in windows:
        roll = shifted.groupby(out["building_id"], sort=False).rolling(
            window, min_periods=max(3, window // 4)
        )
        out[f"rollmean_{window}h"] = roll.mean().reset_index(level=0, drop=True)
        out[f"rollstd_{window}h"] = roll.std().reset_index(level=0, drop=True)
    return out


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join hourly weather; forward-fill small weather gaps."""
    if weather is None or weather.empty:
        return df
    w = weather.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    w = w.drop_duplicates("timestamp").sort_values("timestamp")
    w["air_temperature"] = w["air_temperature"].ffill(limit=6)
    return df.merge(w[["timestamp", "air_temperature"]], on="timestamp", how="left")


def build_feature_matrix(
    clean: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    horizon: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Full feature table for supervised forecasting.

    Returns (frame including timestamp/building_id/target, feature column names).
    Rows whose lag features are undefined (series start) are dropped.
    """
    fc = SETTINGS.forecast
    horizon = horizon or fc.horizon_hours
    lags = tuple(l for l in fc.lags if l >= horizon) or (horizon,)

    df = clean.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_features(df, fc.rolling_windows, shift=horizon)
    df = merge_weather(df, weather)

    feature_cols = CALENDAR_FEATURES + [f"lag_{l}h" for l in lags]
    feature_cols += [f"rollmean_{w}h" for w in fc.rolling_windows]
    feature_cols += [f"rollstd_{w}h" for w in fc.rolling_windows]
    if "air_temperature" in df.columns:
        df["air_temperature"] = df["air_temperature"].fillna(df["air_temperature"].median())
        feature_cols.append("air_temperature")

    lag_cols = [f"lag_{l}h" for l in lags]
    df = df.dropna(subset=lag_cols + ["meter_reading"]).reset_index(drop=True)

    # rolling stats can be NaN near series starts and long gaps: fall back to
    # the shortest lag (for means) and 0 (for stds) so models get finite input
    fallback = df[f"lag_{min(lags)}h"]
    for w in fc.rolling_windows:
        df[f"rollmean_{w}h"] = df[f"rollmean_{w}h"].fillna(fallback)
        df[f"rollstd_{w}h"] = df[f"rollstd_{w}h"].fillna(0.0)
    return df, feature_cols
