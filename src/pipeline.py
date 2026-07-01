"""End-to-end pipeline: raw data -> dashboard-ready tables.

Stages (matching the case-study methodology table):
    1. ingest          raw meter data + metadata + weather
    2. quality profile scorecard on the RAW data
    3. clean           documented repair rules
    4. features        leakage-safe supervised matrix
    5. forecast        baselines + ML, per-building metrics
    6. anomalies       hybrid detection, episode summary
    7. KPIs + recommendations + persisted artefacts

Run:
    python -m src.pipeline --source synthetic --buildings 8
    python -m src.pipeline --source bdg2 --site Fox --buildings 12
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src.config import INTERIM_DIR, PROCESSED_DIR, SETTINGS
from src.data.cleaning import clean_dataset
from src.data.ingestion import load_dataset
from src.data.quality import profile_dataset
from src.features.engineering import build_feature_matrix
from src.models.anomaly import detect_anomalies, summarize_episodes
from src.models.forecasting import best_model_per_building, run_forecasting
from src.recommendations import compute_kpis, generate_recommendations

logger = logging.getLogger(__name__)

ARTIFACTS = {
    "clean_data": "clean_data.parquet",
    "raw_sample": "raw_sample.parquet",
    "weather": "weather.parquet",
    "metadata": "metadata.csv",
    "quality_scorecard": "quality_scorecard.csv",
    "cleaning_report": "cleaning_report.csv",
    "forecast_metrics": "forecast_metrics.csv",
    "forecast_predictions": "forecast_predictions.parquet",
    "feature_importance": "feature_importance.csv",
    "anomalies": "anomalies.csv",
    "anomaly_episodes": "anomaly_episodes.csv",
    "kpis": "kpis.csv",
    "recommendations": "recommendations.csv",
    "run_summary": "run_summary.json",
}


def artifact_path(name: str, out_dir: Path | None = None) -> Path:
    return (out_dir or PROCESSED_DIR) / ARTIFACTS[name]


def run(
    source: str = "synthetic",
    n_buildings: int = 8,
    site: str = "Fox",
    test_days: int | None = None,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict:
    """Execute the full pipeline and persist every artefact. Returns summary."""
    t0 = time.time()
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[1/7] Ingesting data (source=%s)", source)
    ds = load_dataset(source=source, n_buildings=n_buildings, seed=seed, site=site)

    logger.info("[2/7] Profiling raw data quality")
    scorecard = profile_dataset(ds.meters, ds.metadata)

    logger.info("[3/7] Cleaning")
    clean, cleaning_report = clean_dataset(ds.meters)

    logger.info("[4/7] Feature engineering")
    features, feature_cols = build_feature_matrix(clean, ds.weather)

    logger.info("[5/7] Forecasting (%d features)", len(feature_cols))
    forecast = run_forecasting(features, feature_cols, test_days=test_days)

    logger.info("[6/7] Anomaly detection")
    anomalies = detect_anomalies(clean)
    episodes = summarize_episodes(anomalies)

    logger.info("[7/7] KPIs and recommendations")
    kpis = compute_kpis(clean, ds.metadata)
    recommendations = generate_recommendations(kpis, episodes)

    # ------------------------------------------------------------------ save
    clean.to_parquet(artifact_path("clean_data", out_dir), index=False)
    ds.meters.head(50_000).to_parquet(artifact_path("raw_sample", out_dir), index=False)
    ds.weather.to_parquet(artifact_path("weather", out_dir), index=False)
    ds.metadata.to_csv(artifact_path("metadata", out_dir), index=False)
    scorecard.to_csv(artifact_path("quality_scorecard", out_dir), index=False)
    cleaning_report.to_frame().to_csv(artifact_path("cleaning_report", out_dir), index=False)
    forecast.metrics.to_csv(artifact_path("forecast_metrics", out_dir), index=False)
    forecast.predictions.to_parquet(artifact_path("forecast_predictions", out_dir), index=False)
    forecast.feature_importance.to_csv(artifact_path("feature_importance", out_dir), index=False)
    anomalies.to_csv(artifact_path("anomalies", out_dir), index=False)
    episodes.to_csv(artifact_path("anomaly_episodes", out_dir), index=False)
    kpis.to_csv(artifact_path("kpis", out_dir), index=False)
    recommendations.to_csv(artifact_path("recommendations", out_dir), index=False)

    best = best_model_per_building(forecast.metrics)
    summary = {
        "source": source,
        "n_buildings": int(clean["building_id"].nunique()),
        "period": [str(clean["timestamp"].min()), str(clean["timestamp"].max())],
        "rows_clean": int(len(clean)),
        "mean_quality_score": float(scorecard["quality_score"].mean().round(1)),
        "cleaning_totals": cleaning_report.totals(),
        "best_models": best.set_index("building_id")["model"].to_dict(),
        "mean_best_mae": float(best["MAE"].mean().round(3)),
        "mean_skill_vs_baseline_%": float(best["skill_vs_baseline_%"].mean().round(1)),
        "n_anomalies": int(len(anomalies)),
        "n_episodes": int(len(episodes)),
        "n_recommendations": int(len(recommendations)),
        "est_total_saving_eur_yr": float(recommendations["est_saving_eur_yr"].sum()),
        "forecast_horizon_hours": SETTINGS.forecast.horizon_hours,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(artifact_path("run_summary", out_dir), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Pipeline finished in %.1fs -> %s", summary["runtime_seconds"], out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy-intelligence pipeline")
    parser.add_argument("--source", choices=["synthetic", "bdg2"], default="synthetic")
    parser.add_argument("--buildings", type=int, default=8,
                        help="number of buildings (5-20 recommended)")
    parser.add_argument("--site", default="Fox", help="BDG2 site id (bdg2 source only)")
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(
        source=args.source, n_buildings=args.buildings,
        site=args.site, test_days=args.test_days, seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
