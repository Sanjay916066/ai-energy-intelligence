"""KPI, recommendation and what-if tests (RQ4)."""

import pandas as pd

from src.config import SETTINGS
from src.recommendations import (
    compute_kpis,
    generate_recommendations,
    what_if_night_reduction,
    what_if_peak_reduction,
)


def test_kpis_complete_and_consistent(clean_data, dataset):
    clean, _ = clean_data
    kpis = compute_kpis(clean, dataset.metadata)
    assert set(kpis["building_id"]) == set(dataset.metadata["building_id"])
    assert (kpis["peak_kw"] >= kpis["base_load_kw"]).all()
    assert kpis["load_factor"].between(0, 1).all()
    assert kpis["night_share_%"].between(0, 100).all()
    bp = SETTINGS.business
    expected_cost = (kpis["total_kwh"] * bp.electricity_price_eur_per_kwh).round(0)
    pd.testing.assert_series_equal(kpis["cost_eur"], expected_cost, check_names=False)


def test_recommendations_are_ranked_and_quantified(clean_data, dataset):
    clean, _ = clean_data
    kpis = compute_kpis(clean, dataset.metadata)
    recs = generate_recommendations(kpis)
    assert not recs.empty
    assert recs["priority"].isin(["high", "medium", "low"]).all()
    prio_rank = recs["priority"].map({"high": 0, "medium": 1, "low": 2})
    assert prio_rank.is_monotonic_increasing
    assert (recs["est_saving_eur_yr"] >= 0).all()
    assert recs["evidence"].str.len().gt(10).all()
    assert recs["action"].str.len().gt(10).all()


def test_what_if_peak_reduction(clean_data):
    clean, _ = clean_data
    one = clean[clean["building_id"] == clean["building_id"].iloc[0]]
    res = what_if_peak_reduction(one, 10)
    assert res["new_peak_kw"] < res["current_peak_kw"]
    assert res["demand_charge_saving_eur_yr"] > 0
    assert res["kwh_shifted"] >= 0


def test_what_if_night_reduction_scales_linearly(clean_data):
    clean, _ = clean_data
    one = clean[clean["building_id"] == clean["building_id"].iloc[0]]
    r10 = what_if_night_reduction(one, 10)
    r20 = what_if_night_reduction(one, 20)
    assert abs(r20["saved_kwh_yr"] - 2 * r10["saved_kwh_yr"]) <= max(2.0, 0.01 * r20["saved_kwh_yr"])
