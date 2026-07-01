# AI-Driven Energy Intelligence for Industrial Buildings
### Data Quality, Forecasting, Anomaly Detection, and Decision Support

**Case Study 1 — Collaborative Industrial Project · Technical Report**
*Student 1 (Data Management, ETL, Dashboard) · Student 2 (AI Forecasting, Anomaly Detection, Recommendations)*

> This report is pre-filled with the results of the reference pipeline run on the
> synthetic industrial dataset (8 buildings, 1 year, seed 42). Re-running
> `python -m src.pipeline` reproduces every number; switching to
> `--source bdg2` regenerates them for the open BDG2 dataset. Sections marked
> ✏️ are to be extended by the team for the final 12–18-page submission.

---

## 1. Problem statement and industrial motivation

Industrial buildings and SMEs collect energy-consumption data from meters and
building-management systems, but the data are rarely used for operational
decisions. Typical obstacles: missing or inconsistent timestamps, unclear
peak-load drivers, invisible night/weekend consumption, and no linkage between
energy data and business actions.

This project designs and evaluates a compact **energy-intelligence prototype**
that converts raw meter data into (i) a data-reliability assessment, (ii)
short-term demand forecasts, (iii) ranked anomaly alerts and (iv) quantified
management recommendations, delivered through an interactive dashboard.

**Research questions**

- **RQ1** — How reliable is the available energy data?
- **RQ2** — Can short-term energy demand be predicted?
- **RQ3** — Can abnormal consumption be detected?
- **RQ4** — Can the system support practical decisions?

## 2. Data

**Primary dataset strategy.** The prototype runs on two interchangeable
sources through the same pipeline: the **Building Data Genome Project 2**
(BDG2; Miller et al., *Scientific Data* 7, 368, 2020 — 1,636 non-residential
buildings, 3,053 meters, hourly, 2016–2017) and a **synthetic industrial
dataset** that replicates BDG2's shape with ground-truthed injected data-quality
issues and anomalies. The synthetic source enables offline reproducibility and
quantitative detector validation; the BDG2 source demonstrates transferability;
partner data would replace the input layer without pipeline changes.

**Data model.** Long format `timestamp | building_id | meter_reading` (kWh/h),
plus building metadata (site, primary use, floor area, year built) and hourly
outdoor temperature.

✏️ *Describe the selected BDG2 site/buildings or partner data here.*

## 3. Data-quality assessment (RQ1)

Profiling runs on the **raw** data, before cleaning, along the dimensions
required by the case study: missingness, duplicate timestamps, gaps, zero
readings, unrealistic peaks (robust MAD z-score > 6 or > 12× median), negative
readings and metadata completeness. The dimensions blend into a weighted
0–100 **quality score** with a traffic-light rating.

**Reference-run scorecard (8 buildings):** quality scores 99.3–99.6/100
("good"); every building exhibited the injected issues — 5 duplicate
timestamps, 2 negative readings, 2 missing blocks (4–36 h), and 2–3 extreme
peaks each — all correctly surfaced by the profiler.

**Cleaning rules (documented, conservative):**

1. merge duplicate timestamps (mean),
2. re-index to a complete hourly grid,
3. nullify physically impossible negative readings,
4. linearly interpolate gaps ≤ 6 h, flagged `imputed`; longer gaps stay missing,
5. **preserve** spikes and zero-runs — they are anomaly candidates, not errors.

Reference run: 40 duplicates merged, 16 negatives nullified, 111 values
interpolated, 253 left missing. Imputed hours are excluded from forecast
evaluation.

**Answer to RQ1:** the data are reliable *after* profiling-guided cleaning; the
scorecard makes reliability measurable and monitorable per building.

## 4. Forecasting (RQ2)

**Set-up.** Day-ahead horizon (24 h); features = calendar (hour, weekday,
weekend, season — cyclically encoded), lags {24, 25, 26, 48, 72, 168 h},
rolling mean/std (24 h, 168 h windows, shifted by the horizon) and outdoor
temperature. *Leakage guard:* every feature is known 24 h before the target.
Chronological hold-out: last 14 days. Metrics: MAE, RMSE, MAPE, sMAPE, and
skill vs the seasonal-naive baseline.

**Results (mean over 8 buildings, hold-out window):**

| Model | MAE | RMSE | MAPE % | sMAPE % | Skill vs naive |
|---|---|---|---|---|---|
| SeasonalNaive24 (baseline) | 77.3 | 143.1 | 19.0 | 17.8 | 0 % |
| MovingAverage168 | 81.6 | 117.5 | 20.8 | 18.1 | −5.3 % |
| Ridge | 26.8 | 38.0 | 7.0 | 6.8 | +64.6 % |
| **Random Forest** | **16.7** | **26.2** | **4.1** | **4.0** | **+78.0 %** |
| LightGBM | 30.2 | 44.8 | 6.6 | 6.9 | +62.9 % |

Random Forest was the best model for all 8 buildings. Feature importance is
dominated by the 24 h lag and rolling means, with temperature contributing
seasonal signal. Negative skill (MovingAverage168 on MAE) is reported
unchanged — the case study explicitly requires honest baseline comparison.

**Answer to RQ2:** yes — day-ahead demand is predictable with ~4 % sMAPE on
this data, a ~78 % error reduction over the naive baseline.

✏️ *Add plots for normal and difficult periods (from notebook 03).*

## 5. Anomaly detection (RQ3)

**Hybrid design.** Transparent rules — point spikes vs the hour-of-week
profile (robust z > 5), night load > 1.6× typical night load, ≥ 3 consecutive
operating-hour zeros, ≥ 12 h flatlines — combined with an **Isolation Forest**
(1 % contamination) on consumption + context features. Hits are merged per
hour, scored 0–100 by weighted deviation, and grouped into operational
**episodes** with context labels (night / weekend / operating hours).

**Results.** 1,238 anomalous hours → 750 episodes, of which 7 high-severity
(≥ 70). Validation against the injected ground truth: **all 8 night-consumption
episodes and all operating-hour zero-runs were detected** (unit-tested recall).
The ML detector contributes additional single-hour outliers; with unlabeled
real data these require the stakeholder plausibility review foreseen in the
case study — the dashboard's ranked episode list is designed for exactly that
triage.

**False positives:** the Isolation Forest flags ~1 % of hours by construction;
most coincide with rule hits (severity boost), the remainder rank low.
✏️ *Discuss false-positive rate after stakeholder review.*

**Answer to RQ3:** yes — abnormal consumption is detectable and, crucially,
rankable by operational relevance.

## 6. Decision support (RQ4)

**KPIs per building:** total kWh, average daily kWh, peak kW and its
timestamp, load factor, base load (5th percentile), night share, weekend
share, cost (0.25 €/kWh), CO₂ (0.35 kg/kWh), kWh/m².

Reference run: night share 16.9–25.4 %, load factors 0.06–0.14 (spike-driven
peaks), leading to 25 recommendation cards with an estimated total potential
of **≈ 0.7 M€/yr**, dominated by peak-shaving demand-charge savings and
night/standby reduction. Every card carries evidence, a concrete action and
estimated kWh/€/CO₂ impact; savings are estimates from configurable factors.

**Dashboard** (Streamlit): Overview · Data Quality · Forecasting · Anomalies ·
Actions & What-if, with building/date filters, episode drill-down and
interactive peak-shaving and night-reduction scenarios.

**Answer to RQ4:** yes — the prototype closes the loop from meter data to
quantified, prioritized actions understandable to a non-technical stakeholder.

## 7. Limitations

- Economic estimates use a flat tariff and a fixed demand charge; real tariffs
  (time-of-use, capacity markets) change the numbers, not the method.
- Synthetic data cannot capture all real-world failure modes (meter swaps,
  DST artefacts, unit changes); BDG2/partner runs are the necessary complement.
- Anomaly detection is unsupervised; precision on real data is unknown until
  the plausibility review.
- No probabilistic forecasts yet; peak-risk decisions would benefit from
  prediction intervals.

## 8. Next steps

Deep-learning models (LSTM/TCN) where data volume justifies them; weather
*forecasts* as future covariates; alert routing (e-mail/MS Teams);
multi-site benchmarking; integration with the partner's BMS for closed-loop
verification of implemented recommendations.

## 9. Reproducibility

`README.md` documents the environment; `python -m src.pipeline` regenerates
all artefacts in `data/processed/`; `python -m pytest` (36 tests) validates
quality profiling, cleaning, leakage-safety, metrics, detector recall and the
end-to-end pipeline. All parameters live in `src/config.py`.

## References

1. Miller, C. et al. *The Building Data Genome Project 2.* Scientific Data 7, 368 (2020).
2. BDG2 repository: https://github.com/buds-lab/building-data-genome-project-2
3. UCI Electricity Load Diagrams 2011–2014: https://archive.ics.uci.edu/dataset/321
4. Open Power System Data — Time Series: https://data.open-power-system-data.org/time_series/
