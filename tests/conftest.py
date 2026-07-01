"""Shared fixtures: a small deterministic synthetic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import clean_dataset  # noqa: E402
from src.data.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    """Two buildings, one year, deterministic, issues injected."""
    return generate(n_buildings=2, seed=7)


@pytest.fixture(scope="session")
def clean_data(dataset):
    clean, report = clean_dataset(dataset.meters)
    return clean, report
