"""AI-Driven Energy Intelligence dashboard (Streamlit).

Run from the project root:
    streamlit run dashboard/app.py

Pages (tabs):
    1. Overview          — KPIs, consumption trends, peak-load heatmap
    2. Data Quality      — RQ1 scorecard, cleaning report
    3. Forecasting       — RQ2 model comparison, forecast vs actual
    4. Anomalies         — RQ3 ranked episodes, drill-down plots
    5. Actions & What-if — RQ4 recommendation cards, interactive scenarios

If the pipeline has not been run yet the app offers to run it (synthetic
source) directly from the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, SETTINGS  # noqa: E402
from src.pipeline import artifact_path, run as run_pipeline  # noqa: E402
from src.recommendations import what_if_night_reduction, what_if_peak_reduction  # noqa: E402

st.set_page_config(
    page_title="AI Energy Intelligence", page_icon="⚡", layout="wide"
)

PRIORITY_COLOR = {"high": "#d62728", "medium": "#ff7f0e", "low": "#2ca02c"}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading processed data ...")
def load_artifacts() -> dict:
    return {
        "clean": pd.read_parquet(artifact_path("clean_data")),
        "weather": pd.read_parquet(artifact_path("weather")),
        "metadata": pd.read_csv(artifact_path("metadata")),
        "scorecard": pd.read_csv(artifact_path("quality_scorecard")),
        "cleaning": pd.read_csv(artifact_path("cleaning_report")),
        "metrics": pd.read_csv(artifact_path("forecast_metrics")),
        "predictions": pd.read_parquet(artifact_path("forecast_predictions")),
        "importance": pd.read_csv(artifact_path("feature_importance")),
        "anomalies": pd.read_csv(artifact_path("anomalies"), parse_dates=["timestamp"]),
        "episodes": pd.read_csv(artifact_path("anomaly_episodes"), parse_dates=["start", "end"]),
        "kpis": pd.read_csv(artifact_path("kpis"), parse_dates=["peak_timestamp"]),
        "recommendations": pd.read_csv(artifact_path("recommendations")),
    }


def ensure_pipeline() -> bool:
    if artifact_path("run_summary").exists():
        return True
    st.warning("No processed data found — run the pipeline first.")
    if st.button("Run pipeline now (synthetic data, ~1 min)", type="primary"):
        with st.spinner("Running full pipeline ..."):
            run_pipeline(source="synthetic")
        st.cache_data.clear()
        st.rerun()
    return False


def eur(x: float) -> str:
    return f"{x:,.0f} €"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_overview(data: dict, buildings: list[str], date_range) -> None:
    clean = filter_df(data["clean"], buildings, date_range)
    kpis = data["kpis"][data["kpis"]["building_id"].isin(buildings)]

    c1, c2, c3, c4, c5 = st.columns(5)
    total_kwh = clean["meter_reading"].sum()
    bp = SETTINGS.business
    c1.metric("Total consumption", f"{total_kwh / 1000:,.1f} MWh")
    c2.metric("Peak load", f"{clean['meter_reading'].max():,.0f} kW")
    c3.metric("Energy cost", eur(total_kwh * bp.electricity_price_eur_per_kwh))
    c4.metric("CO₂ emissions", f"{total_kwh * bp.co2_kg_per_kwh / 1000:,.1f} t")
    c5.metric("Buildings", f"{clean['building_id'].nunique()}")

    st.subheader("Consumption trend")
    daily = (
        clean.set_index("timestamp")
        .groupby("building_id")["meter_reading"]
        .resample("D").sum().reset_index()
    )
    fig = px.line(
        daily, x="timestamp", y="meter_reading", color="building_id",
        labels={"meter_reading": "kWh/day", "timestamp": ""},
    )
    fig.update_layout(height=380, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Average load profile (hour × weekday)")
        one = clean[clean["building_id"] == buildings[0]] if len(buildings) > 1 else clean
        prof = one.assign(
            hour=one["timestamp"].dt.hour, weekday=one["timestamp"].dt.day_name()
        )
        pivot = prof.pivot_table(
            index="weekday", columns="hour", values="meter_reading", aggfunc="mean"
        ).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        st.caption(f"Building: {buildings[0]}" if len(buildings) > 1 else "")
        st.plotly_chart(
            px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                      labels={"color": "kW"}).update_layout(height=340),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Building KPI comparison")
        show_cols = ["building_id", "total_kwh", "peak_kw", "load_factor",
                     "night_share_%", "weekend_share_%", "cost_eur", "co2_tonnes"]
        if "kwh_per_sqm" in kpis.columns:
            show_cols.append("kwh_per_sqm")
        st.dataframe(kpis[show_cols], use_container_width=True, height=340,
                     hide_index=True)


def tab_quality(data: dict, buildings: list[str]) -> None:
    st.subheader("Data-quality scorecard (raw data, before cleaning)")
    st.caption(
        "Answers RQ1 — missingness, duplicate timestamps, gaps, zero readings, "
        "unrealistic peaks and metadata completeness, blended into a 0-100 score."
    )
    sc = data["scorecard"][data["scorecard"]["building_id"].isin(buildings)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean quality score", f"{sc['quality_score'].mean():.1f} / 100")
    c2.metric("Worst building", f"{sc.iloc[0]['building_id']} ({sc.iloc[0]['quality_score']})")
    c3.metric("Duplicate timestamps", int(sc["duplicate_timestamps"].sum()))

    styled = sc.style.background_gradient(
        subset=["quality_score"], cmap="RdYlGn", vmin=50, vmax=100
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Cleaning actions applied")
    st.caption(
        "Documented repair rules: duplicate merge, hourly re-gridding, negative "
        f"removal, interpolation of gaps ≤ {SETTINGS.cleaning.max_interpolation_gap_hours} h. "
        "Longer gaps stay missing; spikes/zero-runs are kept for anomaly detection."
    )
    cl = data["cleaning"][data["cleaning"]["building_id"].isin(buildings)]
    st.dataframe(cl, use_container_width=True, hide_index=True)

    melted = sc.melt(
        id_vars="building_id",
        value_vars=["missing_rate", "zero_rate"],
        var_name="issue", value_name="rate",
    )
    st.plotly_chart(
        px.bar(melted, x="building_id", y="rate", color="issue", barmode="group",
               labels={"rate": "share of readings"}).update_layout(height=320),
        use_container_width=True,
    )


def tab_forecasting(data: dict, buildings: list[str]) -> None:
    st.subheader("Model comparison (day-ahead forecast, hold-out test window)")
    st.caption(
        "Answers RQ2 — every model is compared against the seasonal-naive "
        "baseline; positive skill = better than baseline. Imputed hours are "
        "excluded from the metrics."
    )
    metrics = data["metrics"][data["metrics"]["building_id"].isin(buildings)]
    st.dataframe(
        metrics.style.background_gradient(subset=["skill_vs_baseline_%"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

    b = st.selectbox("Building", buildings, key="fc_building")
    preds = data["predictions"]
    preds = preds[preds["building_id"] == b]
    models = sorted(preds["model"].unique())
    chosen = st.multiselect(
        "Models to plot", models,
        default=[m for m in models if m in ("SeasonalNaive24", "RandomForest", "LightGBM")] or models[:2],
    )

    if chosen:
        actual = preds[preds["model"] == chosen[0]][["timestamp", "actual"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual["timestamp"], y=actual["actual"],
            name="Actual", line=dict(color="#444", width=2),
        ))
        for m in chosen:
            pm = preds[preds["model"] == m]
            fig.add_trace(go.Scatter(
                x=pm["timestamp"], y=pm["prediction"], name=m, line=dict(width=1.4),
            ))
        fig.update_layout(height=420, yaxis_title="kW", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    imp = data["importance"]
    imp = imp[(imp["building_id"] == b) & (imp["model"] == "RandomForest")]
    if not imp.empty:
        st.subheader("Feature importance (Random Forest)")
        top = imp.nlargest(12, "importance")
        st.plotly_chart(
            px.bar(top, x="importance", y="feature", orientation="h")
            .update_layout(height=350, yaxis=dict(autorange="reversed")),
            use_container_width=True,
        )


def tab_anomalies(data: dict, buildings: list[str]) -> None:
    st.subheader("Anomaly episodes, ranked by severity")
    st.caption(
        "Answers RQ3 — hybrid detection: rule-based (spikes, night load, "
        "zero-runs, flatlines) + Isolation Forest. Consecutive hours are "
        "grouped into operational episodes."
    )
    eps = data["episodes"][data["episodes"]["building_id"].isin(buildings)]
    anoms = data["anomalies"][data["anomalies"]["building_id"].isin(buildings)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes", len(eps))
    c2.metric("High severity (≥ 70)", int((eps["peak_severity"] >= 70).sum()))
    c3.metric("Night/weekend episodes", int(eps["context"].isin(["night", "weekend"]).sum()))

    min_sev = st.slider("Minimum severity", 0, 100, 40, 5)
    show = eps[eps["peak_severity"] >= min_sev]
    st.dataframe(
        show.style.background_gradient(subset=["peak_severity"], cmap="Reds"),
        use_container_width=True, hide_index=True,
    )

    if not show.empty:
        st.subheader("Episode drill-down")
        idx = st.selectbox(
            "Episode", show.index,
            format_func=lambda i: (
                f"{show.loc[i, 'building_id']} — {show.loc[i, 'start']:%Y-%m-%d %H:%M} "
                f"({show.loc[i, 'detectors']}, severity {show.loc[i, 'peak_severity']:.0f})"
            ),
        )
        ep = show.loc[idx]
        clean = data["clean"]
        window = clean[
            (clean["building_id"] == ep["building_id"])
            & (clean["timestamp"] >= ep["start"] - pd.Timedelta(days=2))
            & (clean["timestamp"] <= ep["end"] + pd.Timedelta(days=2))
        ]
        marked = anoms[
            (anoms["building_id"] == ep["building_id"])
            & (anoms["timestamp"] >= ep["start"])
            & (anoms["timestamp"] <= ep["end"])
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=window["timestamp"], y=window["meter_reading"],
            name="Consumption", line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=marked["timestamp"], y=marked["value"], mode="markers",
            name="Anomalous hours", marker=dict(color="#d62728", size=9, symbol="x"),
        ))
        fig.update_layout(height=380, yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)


def tab_actions(data: dict, buildings: list[str]) -> None:
    st.subheader("Recommended actions")
    st.caption(
        "Answers RQ4 — quantified with the configurable tariff "
        f"({SETTINGS.business.electricity_price_eur_per_kwh} €/kWh), demand charge "
        f"({SETTINGS.business.demand_charge_eur_per_kw_year} €/kW·yr) and CO₂ factor "
        f"({SETTINGS.business.co2_kg_per_kwh} kg/kWh). Savings are estimates."
    )
    recs = data["recommendations"][data["recommendations"]["building_id"].isin(buildings)]

    total = recs["est_saving_eur_yr"].sum()
    co2 = recs["est_saving_co2_t_yr"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Actions identified", len(recs))
    c2.metric("Estimated saving potential", f"{eur(total)}/yr")
    c3.metric("CO₂ reduction potential", f"{co2:,.1f} t/yr")

    for _, r in recs.iterrows():
        color = PRIORITY_COLOR.get(r["priority"], "#888")
        with st.container(border=True):
            st.markdown(
                f"**{r['title']}** — {r['building_id']} "
                f"<span style='color:{color};font-weight:600'>[{r['priority'].upper()}]</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"*Evidence:* {r['evidence']}")
            st.markdown(f"*Action:* {r['action']}")
            if r["est_saving_eur_yr"] > 0:
                st.markdown(
                    f"💰 **{eur(r['est_saving_eur_yr'])}/yr** · "
                    f"⚡ {r['est_saving_kwh_yr']:,.0f} kWh/yr · "
                    f"🌍 {r['est_saving_co2_t_yr']} t CO₂/yr"
                )

    st.divider()
    st.subheader("What-if scenarios")
    b = st.selectbox("Building", buildings, key="wi_building")
    one = data["clean"][data["clean"]["building_id"] == b]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Peak-load shaving**")
        pct = st.slider("Peak reduction (%)", 5, 30, 10, 5, key="wi_peak")
        res = what_if_peak_reduction(one, pct)
        st.metric("New peak", f"{res['new_peak_kw']:,.0f} kW",
                  delta=f"-{res['current_peak_kw'] - res['new_peak_kw']:,.0f} kW",
                  delta_color="inverse")
        st.metric("Demand-charge saving", f"{eur(res['demand_charge_saving_eur_yr'])}/yr")
        st.caption(
            f"{res['kwh_shifted']:,.0f} kWh over {res['hours_above_cap']} h "
            "would need shifting/storage."
        )
    with col2:
        st.markdown("**Night-load reduction**")
        pct_n = st.slider("Night consumption cut (%)", 10, 60, 30, 10, key="wi_night")
        res_n = what_if_night_reduction(one, pct_n)
        st.metric("Energy saved", f"{res_n['saved_kwh_yr']:,.0f} kWh/yr")
        st.metric("Cost saved", f"{eur(res_n['saved_eur_yr'])}/yr")
        st.metric("CO₂ avoided", f"{res_n['saved_co2_t_yr']} t/yr")


# ---------------------------------------------------------------------------
# Helpers + main
# ---------------------------------------------------------------------------
def filter_df(df: pd.DataFrame, buildings: list[str], date_range) -> pd.DataFrame:
    out = df[df["building_id"].isin(buildings)]
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        out = out[(out["timestamp"] >= start) & (out["timestamp"] < end)]
    return out


def main() -> None:
    st.title("⚡ AI-Driven Energy Intelligence for Industrial Buildings")
    st.caption(
        "Data quality · forecasting · anomaly detection · decision support — "
        "Collaborative Industrial Project, Case Study 1"
    )
    if not ensure_pipeline():
        st.stop()
    data = load_artifacts()

    all_buildings = sorted(data["clean"]["building_id"].unique())
    with st.sidebar:
        st.header("Filters")
        buildings = st.multiselect("Buildings", all_buildings, default=all_buildings)
        tmin = data["clean"]["timestamp"].min().date()
        tmax = data["clean"]["timestamp"].max().date()
        date_range = st.date_input(
            "Date range", (tmin, tmax), min_value=tmin, max_value=tmax
        )
        st.divider()
        if st.button("Re-run pipeline (synthetic)"):
            with st.spinner("Running ..."):
                run_pipeline(source="synthetic")
            st.cache_data.clear()
            st.rerun()
        st.caption(
            "To use real BDG2 data run\n`python -m src.pipeline --source bdg2`\n"
            "then reload this app."
        )

    if not buildings:
        st.info("Select at least one building.")
        st.stop()

    tabs = st.tabs(
        ["📊 Overview", "🧹 Data Quality", "📈 Forecasting", "🚨 Anomalies", "✅ Actions & What-if"]
    )
    with tabs[0]:
        tab_overview(data, buildings, date_range)
    with tabs[1]:
        tab_quality(data, buildings)
    with tabs[2]:
        tab_forecasting(data, buildings)
    with tabs[3]:
        tab_anomalies(data, buildings)
    with tabs[4]:
        tab_actions(data, buildings)


if __name__ == "__main__":
    main()
