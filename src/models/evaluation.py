"""Forecast evaluation metrics required by the case study:
MAE, RMSE, MAPE and sMAPE, plus skill relative to the naive baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def mae(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    """MAPE in %, computed only where the actual value is non-zero."""
    yt, yp = _as_arrays(y_true, y_pred)
    nz = np.abs(yt) > eps
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100)


def smape(y_true, y_pred, eps: float = 1e-9) -> float:
    """Symmetric MAPE in % (bounded 0-200, robust to near-zero actuals)."""
    yt, yp = _as_arrays(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2 + eps
    return float(np.mean(np.abs(yt - yp) / denom) * 100)


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAPE_%": round(mape(y_true, y_pred), 2),
        "sMAPE_%": round(smape(y_true, y_pred), 2),
    }


def add_skill_vs_baseline(
    metrics: pd.DataFrame, baseline_name: str = "SeasonalNaive24"
) -> pd.DataFrame:
    """Add ``skill_vs_baseline_%`` = MAE improvement over the naive baseline.

    Positive = better than baseline. The case study requires an honest
    baseline comparison, so this column is reported even when negative.
    """
    out = metrics.copy()
    base = (
        out.loc[out["model"] == baseline_name]
        .set_index("building_id")["MAE"]
    )
    out["skill_vs_baseline_%"] = out.apply(
        lambda r: round((1 - r["MAE"] / base[r["building_id"]]) * 100, 1)
        if r["building_id"] in base.index and base[r["building_id"]] > 0
        else float("nan"),
        axis=1,
    )
    return out
