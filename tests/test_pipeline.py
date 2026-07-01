"""End-to-end pipeline smoke test: every required artefact is produced."""

import json

from src.pipeline import ARTIFACTS, artifact_path, run


def test_pipeline_produces_all_artifacts(tmp_path):
    summary = run(source="synthetic", n_buildings=2, test_days=7, seed=3, out_dir=tmp_path)

    for name in ARTIFACTS:
        assert artifact_path(name, tmp_path).exists(), f"missing artefact: {name}"

    assert summary["n_buildings"] == 2
    assert summary["rows_clean"] > 0
    assert 0 <= summary["mean_quality_score"] <= 100
    assert summary["n_anomalies"] > 0
    assert summary["n_recommendations"] > 0
    assert summary["best_models"]  # a best model chosen for every building

    with open(artifact_path("run_summary", tmp_path), encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["n_buildings"] == summary["n_buildings"]
