"""Data ingestion layer.

Two interchangeable sources feed the same downstream pipeline (the "input
layer swap" recommended by the case study):

* ``synthetic`` — offline, deterministic, issues injected (default).
* ``bdg2``      — the Building Data Genome Project 2 open dataset; files are
                  downloaded once into ``data/raw`` and a configurable subset
                  of buildings is melted into the long format used everywhere:
                  ``timestamp | building_id | meter_reading`` (kWh).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import BDG2_FILES, RAW_DIR
from src.data.synthetic import SyntheticDataset, generate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BDG2 download / load
# ---------------------------------------------------------------------------
def download_bdg2(files: list[str] | None = None, force: bool = False) -> dict[str, Path]:
    """Stream-download BDG2 CSVs into ``data/raw``. Returns local paths.

    Note: ``electricity_cleaned.csv`` is ~1 GB — only fetched when requested.
    """
    files = files or ["metadata", "weather", "electricity"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in files:
        url = BDG2_FILES[name]
        dest = RAW_DIR / f"bdg2_{name}.csv"
        paths[name] = dest
        if dest.exists() and not force:
            logger.info("%s already present, skipping download", dest.name)
            continue
        logger.info("Downloading %s -> %s", url, dest)
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    return paths


def load_bdg2(
    site: str = "Fox",
    max_buildings: int = 12,
    buildings: list[str] | None = None,
) -> SyntheticDataset:
    """Load a subset of BDG2 as the standard long-format dataset container.

    Buildings are chosen either explicitly or as the ``max_buildings`` meters
    with the fewest missing values on the chosen site (keeps the prototype
    small per the case-study risk plan: "start with 5-20 buildings").
    """
    paths = download_bdg2()
    metadata = pd.read_csv(paths["metadata"])

    if buildings is None:
        site_buildings = metadata.loc[
            metadata["site_id"] == site, "building_id"
        ].tolist()
        usecols_probe = ["timestamp"] + site_buildings
        wide = pd.read_csv(paths["electricity"], usecols=lambda c: c in usecols_probe)
        completeness = wide.drop(columns="timestamp").notna().mean()
        buildings = completeness.sort_values(ascending=False).head(max_buildings).index.tolist()
    else:
        wide = pd.read_csv(
            paths["electricity"], usecols=lambda c: c == "timestamp" or c in set(buildings)
        )

    long_df = wide.melt(
        id_vars="timestamp", value_vars=buildings,
        var_name="building_id", value_name="meter_reading",
    )
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])

    weather = pd.read_csv(paths["weather"])
    weather = weather.loc[weather["site_id"] == site, ["timestamp", "airTemperature"]]
    weather = weather.rename(columns={"airTemperature": "air_temperature"})
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    meta = metadata.loc[
        metadata["building_id"].isin(buildings),
        ["building_id", "site_id", "primaryspaceusage", "sqm", "yearbuilt"],
    ].rename(columns={"primaryspaceusage": "primary_use", "yearbuilt": "year_built"})

    return SyntheticDataset(
        meters=long_df, metadata=meta.reset_index(drop=True), weather=weather
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def load_dataset(
    source: str = "synthetic",
    n_buildings: int = 8,
    seed: int = 42,
    site: str = "Fox",
) -> SyntheticDataset:
    """Return the raw dataset from the requested source."""
    if source == "synthetic":
        return generate(n_buildings=n_buildings, seed=seed)
    if source == "bdg2":
        return load_bdg2(site=site, max_buildings=n_buildings)
    raise ValueError(f"Unknown source {source!r}; expected 'synthetic' or 'bdg2'.")
