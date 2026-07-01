"""Central configuration for the energy-intelligence pipeline.

All tunable business and modelling parameters live here so that the pipeline,
dashboard and tests share a single source of truth.  Values can be overridden
per-run through the CLI (see ``src/pipeline.py``) or by editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# BDG2 open dataset (https://github.com/buds-lab/building-data-genome-project-2)
# ---------------------------------------------------------------------------
BDG2_BASE_URL = (
    "https://raw.githubusercontent.com/buds-lab/"
    "building-data-genome-project-2/master/data"
)
BDG2_FILES = {
    "metadata": f"{BDG2_BASE_URL}/metadata/metadata.csv",
    "weather": f"{BDG2_BASE_URL}/weather/weather.csv",
    "electricity": f"{BDG2_BASE_URL}/meters/cleaned/electricity_cleaned.csv",
}

# ---------------------------------------------------------------------------
# Business parameters (industrial context — override to match a real partner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BusinessParams:
    """Economic and environmental conversion factors."""

    electricity_price_eur_per_kwh: float = 0.25
    co2_kg_per_kwh: float = 0.35          # grid emission factor
    demand_charge_eur_per_kw_year: float = 120.0  # peak-demand tariff component
    operating_hours: tuple[int, int] = (6, 20)    # typical industrial shift window
    night_hours: tuple[int, int] = (22, 6)        # wrap-around: 22:00 -> 06:00


@dataclass(frozen=True)
class QualityParams:
    """Thresholds used by the data-quality scorecard."""

    expected_freq: str = "h"
    peak_zscore_threshold: float = 6.0    # robust z-score above which a reading is "unrealistic"
    max_plausible_multiplier: float = 12.0  # reading > multiplier * median flagged as extreme


@dataclass(frozen=True)
class CleaningParams:
    """Rules applied when repairing the raw meter series."""

    max_interpolation_gap_hours: int = 6  # linear-fill only short gaps; longer stay NaN


@dataclass(frozen=True)
class ForecastParams:
    """Forecast experiment set-up."""

    horizon_hours: int = 24               # day-ahead forecasting
    test_days: int = 14                   # hold-out window at the end of the series
    lags: tuple[int, ...] = (24, 25, 26, 48, 72, 168)  # all >= horizon to avoid leakage
    rolling_windows: tuple[int, ...] = (24, 168)
    random_state: int = 42


@dataclass(frozen=True)
class AnomalyParams:
    """Anomaly-detection thresholds."""

    spike_zscore: float = 5.0             # robust z-score for point spikes
    night_load_factor: float = 1.6        # night load above factor * typical night load
    zero_run_hours: int = 3               # consecutive operating-hour zeros => outage/meter fault
    flatline_hours: int = 12              # constant non-zero value for this long => stuck sensor
    isolation_forest_contamination: float = 0.01
    random_state: int = 42


@dataclass(frozen=True)
class Settings:
    business: BusinessParams = field(default_factory=BusinessParams)
    quality: QualityParams = field(default_factory=QualityParams)
    cleaning: CleaningParams = field(default_factory=CleaningParams)
    forecast: ForecastParams = field(default_factory=ForecastParams)
    anomaly: AnomalyParams = field(default_factory=AnomalyParams)


SETTINGS = Settings()
