"""Validation-only per-label decision threshold optimizer and sensitivity analyzer (Suite A2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.schema import HALLUCINATION_LABELS
from src.evaluation.metrics import compute_multilabel_metrics
from src.models.calibration import assert_validation_only
from src.utils.io import read_json, utc_now_iso, write_json


def optimize_validation_thresholds(
    val_probs: np.ndarray,
    val_y_true: np.ndarray,
    *,
    split_names: Sequence[str] | None = None,
    min_tau: float = 0.05,
    max_tau: float = 0.95,
    step: float = 0.01,
    baseline_tau: float = 0.50,
) -> dict[str, Any]:
    """
    Sweep tau_k in [0.05, 0.95] step 0.01 exclusively on validation predictions to maximize per-label F1.

    Ties are broken cleanly toward baseline_tau (0.50) to reduce arbitrary threshold drift.
    """
    assert_validation_only(split_names, caller="optimize_validation_thresholds")
    yp = np.asarray(val_probs, dtype=np.float64)
    yt = np.asarray(val_y_true, dtype=np.int64)
    if yp.ndim != 2 or yp.shape != yt.shape or yp.shape[1] != len(HALLUCINATION_LABELS):
        raise ValueError(f"Expected identical (N, 5) arrays, got probs={yp.shape}, y={yt.shape}")
    if len(yp) == 0:
        raise ValueError("Cannot optimize thresholds on empty validation predictions.")

    grid = np.round(np.arange(float(min_tau), float(max_tau) + float(step) * 0.5, float(step)), 4)
    best_thresholds: list[float] = []
    sensitivity_curves: dict[str, list[dict[str, float]]] = {}

    for k, lbl in enumerate(HALLUCINATION_LABELS):
        col_t = yt[:, k]
        col_p = yp[:, k]
        curve: list[dict[str, float]] = []
        best_tau = float(baseline_tau)
        best_f1 = -1.0
        best_dist = float("inf")

        for tau in grid:
            pred = (col_p >= tau).astype(np.int64)
            prec = float(precision_score(col_t, pred, zero_division=0))
            rec = float(recall_score(col_t, pred, zero_division=0))
            f1 = float(f1_score(col_t, pred, zero_division=0))
            curve.append(
                {
                    "tau": float(tau),
                    "precision": round(prec, 6),
                    "recall": round(rec, 6),
                    "f1": round(f1, 6),
                }
            )
            dist = abs(float(tau) - float(baseline_tau))
            if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and dist < best_dist):
                best_f1 = f1
                best_tau = float(tau)
                best_dist = dist

        best_thresholds.append(round(best_tau, 4))
        sensitivity_curves[lbl] = curve

    val_fixed = compute_multilabel_metrics(yt, yp, baseline_tau)
    val_tuned = compute_multilabel_metrics(yt, yp, best_thresholds)

    return {
        "tuned_at_utc": utc_now_iso(),
        "n_val_samples": int(len(yp)),
        "grid_spec": {"min_tau": min_tau, "max_tau": max_tau, "step": step, "n_points": len(grid)},
        "baseline_tau": float(baseline_tau),
        "per_label_thresholds": {
            lbl: best_thresholds[k] for k, lbl in enumerate(HALLUCINATION_LABELS)
        },
        "threshold_vector": best_thresholds,
        "val_metrics_fixed_050": val_fixed,
        "val_metrics_tuned": val_tuned,
        "sensitivity_curves": sensitivity_curves,
    }


def save_thresholds(result: dict[str, Any], path: str | Path) -> Path:
    """Persist tuned threshold payload JSON to disk."""
    p = Path(path)
    write_json(p, result)
    return p


def load_thresholds(path: str | Path) -> list[float]:
    """Load ordered 5-label threshold vector from thresholds JSON."""
    payload = read_json(path)
    vec = payload.get("threshold_vector")
    if not isinstance(vec, list) or len(vec) != len(HALLUCINATION_LABELS):
        raise ValueError(f"Malformed thresholds file {path}: expected 5 floats.")
    return [float(x) for x in vec]

