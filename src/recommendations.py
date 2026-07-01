"""KPIs, recommendation cards, and what-if scenarios — answers RQ4.

Turns the analytical outputs into stakeholder-facing artefacts:

* ``compute_kpis``            — per-building energy KPI table
* ``generate_recommendations`` — ranked, quantified action cards
* ``what_if_peak_reduction``  — scenario: shave X % off the peak
* ``what_if_night_reduction`` — scenario: cut night/standby load by X %

Savings are estimates based on the configurable tariff, demand charge and
grid CO2 factor in ``src.config`` — stated as such in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SETTINGS


def _night_mask(ts: pd.Series) -> pd.Series:
    start, end = SETTINGS.business.night_hours
    return (ts.dt.hour >= start) | (ts.dt.hour < end)


def compute_kpis(clean: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-building KPI table over the full cleaned period."""
    bp = SETTINGS.business
    rows = []
    for building_id, g in clean.groupby("building_id", sort=True):
        v = g["meter_reading"]
        finite = v.notna()
        total_kwh = float(v.sum())
        hours = int(finite.sum())
        if hours == 0:
            continue
        peak_idx = v.idxmax()
        night = _night_mask(g["timestamp"]) & finite
        weekend = (g["timestamp"].dt.weekday >= 5) & finite
        avg_kw = total_kwh / hours
        peak_kw = float(v.max())
        base_load = float(v[finite].quantile(0.05))
        rows.append({
            "building_id": building_id,
            "total_kwh": round(total_kwh, 1),
            "avg_daily_kwh": round(total_kwh / (hours / 24), 1),
            "peak_kw": round(peak_kw, 1),
            "peak_timestamp": g.loc[peak_idx, "timestamp"],
            "load_factor": round(avg_kw / peak_kw, 3) if peak_kw > 0 else np.nan,
            "base_load_kw": round(base_load, 1),
            "night_share_%": round(float(v[night].sum()) / total_kwh * 100, 1),
            "weekend_share_%": round(float(v[weekend].sum()) / total_kwh * 100, 1),
            "cost_eur": round(total_kwh * bp.electricity_price_eur_per_kwh, 0),
            "co2_tonnes": round(total_kwh * bp.co2_kg_per_kwh / 1000, 2),
        })
    kpis = pd.DataFrame(rows)
    if metadata is not None and not kpis.empty:
        kpis = kpis.merge(
            metadata[["building_id", "primary_use", "sqm"]], on="building_id", how="left"
        )
        kpis["kwh_per_sqm"] = (kpis["total_kwh"] / kpis["sqm"]).round(1)
    return kpis


# ---------------------------------------------------------------------------
# Recommendation rules
# ---------------------------------------------------------------------------
# Rule thresholds are industry heuristics; each card explains its evidence.
NIGHT_SHARE_THRESHOLD = 25.0     # % of energy at night considered high
LOAD_FACTOR_THRESHOLD = 0.45     # below this, peaks are expensive vs average use
BASE_LOAD_RATIO_THRESHOLD = 0.35  # base load > 35% of average load => standby waste


def generate_recommendations(
    kpis: pd.DataFrame,
    anomaly_episodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ranked action cards with estimated annual kWh / EUR / CO2 impact."""
    bp = SETTINGS.business
    cards: list[dict] = []

    def card(building_id, priority, title, evidence, action, saving_kwh) -> None:
        saving_kwh = max(0.0, float(saving_kwh))
        cards.append({
            "building_id": building_id,
            "priority": priority,
            "title": title,
            "evidence": evidence,
            "action": action,
            "est_saving_kwh_yr": round(saving_kwh, 0),
            "est_saving_eur_yr": round(saving_kwh * bp.electricity_price_eur_per_kwh, 0),
            "est_saving_co2_t_yr": round(saving_kwh * bp.co2_kg_per_kwh / 1000, 2),
        })

    for _, k in kpis.iterrows():
        b = k["building_id"]
        annualizer = 365 * 24 / max(k["total_kwh"] / max(k["avg_daily_kwh"] / 24, 1e-9), 1)

        if k["night_share_%"] > NIGHT_SHARE_THRESHOLD:
            excess_share = (k["night_share_%"] - 15.0) / 100  # target: 15 % night share
            card(
                b, "high",
                "Reduce night / standby consumption",
                f"{k['night_share_%']}% of energy is consumed at night "
                f"(22:00-06:00); efficient industrial sites are near 15%.",
                "Audit equipment left running overnight (compressors, HVAC, "
                "lighting); add shutdown schedules or interlocks.",
                k["total_kwh"] * excess_share * 0.6 * annualizer,  # 60% of excess is addressable
            )

        if pd.notna(k["load_factor"]) and k["load_factor"] < LOAD_FACTOR_THRESHOLD:
            shave = k["peak_kw"] * 0.10
            demand_saving_eur = shave * bp.demand_charge_eur_per_kw_year
            card(
                b, "high",
                "Peak-load reduction (10% shaving target)",
                f"Load factor {k['load_factor']} — peak {k['peak_kw']} kW vs "
                f"base {k['base_load_kw']} kW; peaks drive demand charges.",
                f"Stagger start-ups around {pd.Timestamp(k['peak_timestamp']):%H:%M}; "
                "consider load-shifting or storage. "
                f"~{demand_saving_eur:.0f} EUR/yr demand-charge saving at "
                f"{bp.demand_charge_eur_per_kw_year} EUR/kW-yr.",
                0.0,  # peak shaving saves on the demand charge, not kWh
            )
            cards[-1]["est_saving_eur_yr"] = round(demand_saving_eur, 0)

        avg_kw = k["avg_daily_kwh"] / 24
        if avg_kw > 0 and k["base_load_kw"] / avg_kw > BASE_LOAD_RATIO_THRESHOLD:
            reducible = (k["base_load_kw"] - BASE_LOAD_RATIO_THRESHOLD * avg_kw) * 8760 * 0.3
            card(
                b, "medium",
                "Lower the permanent base load",
                f"Base load {k['base_load_kw']} kW is "
                f"{k['base_load_kw'] / avg_kw:.0%} of average load.",
                "Sub-meter always-on circuits; eliminate idle losses "
                "(transformers, ventilation running 24/7, leaking compressed air).",
                reducible,
            )

        if k["weekend_share_%"] > 20.0:
            card(
                b, "medium",
                "Review weekend consumption",
                f"{k['weekend_share_%']}% of energy is used on weekends.",
                "Verify weekend production schedule; disable non-essential "
                "systems on non-production days.",
                k["total_kwh"] * (k["weekend_share_%"] - 12) / 100 * 0.4 * annualizer,
            )

    if anomaly_episodes is not None and not anomaly_episodes.empty:
        top = anomaly_episodes[anomaly_episodes["peak_severity"] >= 50]
        for b, group in top.groupby("building_id"):
            worst = group.iloc[0]
            card(
                b, "high",
                "Investigate detected consumption anomalies",
                f"{len(group)} high-severity episode(s); worst: "
                f"{worst['detectors']} starting {worst['start']:%Y-%m-%d %H:%M} "
                f"({worst['context']}, {worst['hours']} h).",
                "Cross-check with maintenance logs and shift plans; assign an "
                "owner to each episode in the dashboard.",
                0.0,
            )

    if not cards:
        return pd.DataFrame(columns=[
            "building_id", "priority", "title", "evidence", "action",
            "est_saving_kwh_yr", "est_saving_eur_yr", "est_saving_co2_t_yr",
        ])
    frame = pd.DataFrame(cards)
    prio_rank = frame["priority"].map({"high": 0, "medium": 1, "low": 2})
    return (
        frame.assign(_rank=prio_rank)
        .sort_values(["_rank", "est_saving_eur_yr"], ascending=[True, False])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# What-if scenarios (dashboard interactive)
# ---------------------------------------------------------------------------
def what_if_peak_reduction(clean_building: pd.DataFrame, reduction_pct: float) -> dict:
    """Effect of capping the load at (1 - pct) * current peak.

    Shifted energy is assumed to be *rescheduled*, not saved, so the benefit
    is the demand-charge reduction; the kWh moved is reported for context.
    """
    bp = SETTINGS.business
    v = clean_building["meter_reading"].dropna()
    peak = float(v.max())
    cap = peak * (1 - reduction_pct / 100)
    shifted_kwh = float((v - cap).clip(lower=0).sum())
    return {
        "current_peak_kw": round(peak, 1),
        "new_peak_kw": round(cap, 1),
        "kwh_shifted": round(shifted_kwh, 1),
        "hours_above_cap": int((v > cap).sum()),
        "demand_charge_saving_eur_yr": round((peak - cap) * bp.demand_charge_eur_per_kw_year, 0),
    }


def what_if_night_reduction(clean_building: pd.DataFrame, reduction_pct: float) -> dict:
    """Effect of cutting night-hour consumption by ``reduction_pct`` %."""
    bp = SETTINGS.business
    night = _night_mask(clean_building["timestamp"])
    night_kwh = float(clean_building.loc[night, "meter_reading"].sum())
    period_days = max(
        (clean_building["timestamp"].max() - clean_building["timestamp"].min()).days, 1
    )
    saved = night_kwh * reduction_pct / 100 * (365 / period_days)
    return {
        "night_kwh_period": round(night_kwh, 1),
        "saved_kwh_yr": round(saved, 0),
        "saved_eur_yr": round(saved * bp.electricity_price_eur_per_kwh, 0),
        "saved_co2_t_yr": round(saved * bp.co2_kg_per_kwh / 1000, 2),
    }
