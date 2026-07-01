# ⚡ AI-Driven Energy Intelligence for Industrial Buildings

**Case Study 1 — Collaborative Industrial Project (2 master students)**

A reproducible prototype that converts raw energy-meter data into reliable
insights, forecasts, anomaly alerts and management recommendations:

| Research question | Answered by |
|---|---|
| **RQ1** — How reliable is the available energy data? | Data-quality scorecard ([src/data/quality.py](src/data/quality.py)) |
| **RQ2** — Can short-term energy demand be predicted? | Baselines + ML forecasting with MAE/RMSE/MAPE/sMAPE ([src/models/forecasting.py](src/models/forecasting.py)) |
| **RQ3** — Can abnormal consumption be detected? | Hybrid rule-based + Isolation Forest detection, severity-ranked ([src/models/anomaly.py](src/models/anomaly.py)) |
| **RQ4** — Can the system support practical decisions? | Streamlit dashboard, recommendation cards, what-if scenarios ([dashboard/app.py](dashboard/app.py)) |

## Quick start

```bash
# 1. Environment (Python >= 3.10)
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Run the full pipeline (synthetic industrial data, ~30 s, fully offline)
python -m src.pipeline --source synthetic --buildings 8

# 3. Launch the dashboard
streamlit run dashboard/app.py

# 4. Run the tests
python -m pytest
```

### Using the real BDG2 open dataset

The pipeline was designed against the
[Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2)
(1,636 non-residential buildings, hourly meters, 2016–2017 — Miller et al.,
*Scientific Data* 7, 368, 2020):

```bash
python -m src.pipeline --source bdg2 --site Fox --buildings 12
```

This downloads `metadata.csv`, `weather.csv` and `electricity_cleaned.csv`
(~1 GB, one-time) into `data/raw/` and selects the most complete meters of the
chosen site — following the case-study risk plan ("start with 5–20 buildings").

**Default = synthetic source**: a deterministic generator
([src/data/synthetic.py](src/data/synthetic.py)) producing BDG2-shaped
industrial load data with *injected, ground-truthed* quality issues and
anomalies (missing blocks, duplicate timestamps, negative glitches, spikes,
night-consumption episodes, zero-runs). This keeps the whole workflow
demonstrable offline and lets tests verify detector recall — and mirrors the
intended production path: **swap the input layer, keep the pipeline**.

## Project structure

```
project-energy-intelligence/
├── data/                  # raw / interim / processed (git-ignored; no sensitive data in repo)
├── notebooks/             # 01 EDA · 02 data quality · 03 forecasting · 04 anomalies+recommendations
├── src/
│   ├── config.py          # all business & model parameters (tariff, CO2 factor, thresholds)
│   ├── data/              # ingestion (BDG2 + synthetic), quality profiling, cleaning
│   ├── features/          # calendar/lag/rolling/weather features (leakage-safe)
│   ├── models/            # forecasting ladder, metrics, hybrid anomaly detection
│   ├── recommendations.py # KPIs, action cards, what-if scenarios
│   └── pipeline.py        # end-to-end orchestrator (CLI)
├── dashboard/app.py       # Streamlit decision-support dashboard
├── tests/                 # 36 unit + integration tests (pytest)
├── reports/               # technical report + one-page industrial summary
└── requirements.txt
```

## Methodology (per the case-study stages)

1. **Data understanding** — long format `timestamp | building_id | meter_reading`
   (kWh), building metadata, hourly weather.
2. **Data-quality profiling** — runs on the *raw* data: missingness, duplicate
   timestamps, gaps, zero readings, robust-z extreme peaks, negatives, metadata
   completeness → 0–100 score + traffic-light rating.
3. **Cleaning** — documented conservative rules: merge duplicates, hourly
   re-gridding, nullify negatives, interpolate gaps ≤ 6 h (flagged `imputed`);
   spikes/zero-runs are *kept* — they are anomaly candidates, not data errors.
4. **Feature engineering** — hour/weekday/season (cyclically encoded), lags,
   rolling statistics, temperature. Leakage guard: for horizon *h*, only lags ≥ *h*
   and rolling windows shifted by *h*.
5. **Forecasting (day-ahead)** — SeasonalNaive24 & MovingAverage168 baselines vs
   Ridge, Random Forest, LightGBM; chronological hold-out (last 14 days); imputed
   hours excluded from metrics; **skill vs baseline reported honestly, even when
   negative**.
6. **Anomaly detection** — rules (spike vs hour-of-week profile, night load,
   operating-hour zero-runs, flatlines) + Isolation Forest; merged, severity 0–100,
   grouped into operational *episodes* with context (night/weekend/shift).
7. **Decision support** — KPI table (peak, load factor, night/weekend share, base
   load, cost, CO₂, kWh/m²), quantified recommendation cards, interactive peak-shaving
   and night-reduction what-if scenarios.

## Dashboard

Five tabs mapped to the research questions: **Overview** (KPIs, trends, load-profile
heatmap) · **Data Quality** (scorecard + cleaning audit) · **Forecasting** (model
comparison, forecast vs actual, feature importance) · **Anomalies** (ranked episodes
with drill-down) · **Actions & What-if** (recommendation cards, scenario sliders).
Sidebar filters by building and date range; the pipeline can be (re)run from the app.

## Configuration

Business assumptions live in [src/config.py](src/config.py) and should be adapted
to a real partner: electricity price (0.25 €/kWh), demand charge (120 €/kW·yr),
grid CO₂ factor (0.35 kg/kWh), operating hours (06–20), forecast horizon (24 h),
detector thresholds. All savings shown are estimates derived from these factors.

## Team roles (per the case study)

- **Student 1 — Data & Dashboard:** `src/data/`, `src/pipeline.py`, `dashboard/`,
  data-quality report, user guide.
- **Student 2 — AI & Evaluation:** `src/features/`, `src/models/`,
  `src/recommendations.py`, notebooks 03–04, model evaluation.
- **Joint:** integration, report ([reports/technical_report.md](reports/technical_report.md)),
  industrial summary ([reports/industrial_summary.md](reports/industrial_summary.md)),
  final presentation.

## Limitations & next steps

- Savings estimates use flat-tariff heuristics; a real engagement needs the
  partner's tariff structure and sub-metering.
- Anomaly validation on real data requires stakeholder plausibility review
  (no labels) — the synthetic ground truth substitutes for this in CI.
- Extensions: LSTM/Temporal CNN models, probabilistic forecasts, weather
  forecasts as future covariates, multi-site comparison, alerting integration.
