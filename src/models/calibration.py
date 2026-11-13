"""Validation-only post-hoc Temperature Scaling and calibration diagnostics (Suite A3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import optimize, special

from src.data.schema import HALLUCINATION_LABELS
from src.evaluation.metrics import compute_brier_score, compute_ece
from src.utils.io import read_json, utc_now_iso, write_json


def _stable_bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Numerically stable mean binary cross-entropy loss from raw logits."""
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def assert_validation_only(split_names: Sequence[str] | None, *, caller: str) -> None:
    """Enforce zero test/train leakage during calibration or threshold tuning."""
    if split_names is None:
        return
    bad = sorted({str(s) for s in split_names if str(s) != "val"})
    if bad:
        raise AssertionError(
            f"{caller} accepts ONLY validation split records ('val'); received disallowed splits: {bad}"
        )


class TemperatureScaler:
    """
    Post-hoc multi-label Temperature Scaling fitted strictly on validation logits.

    Supports both scalar global temperature (T_1=...=T_5=T) and per-label temperatures
    T = [T_FI, T_FF, T_II, T_LI, T_CI] > 0.
    """

    def __init__(self, *, per_label: bool = True, init_temp: float = 1.0) -> None:
        self.per_label = bool(per_label)
        n = len(HALLUCINATION_LABELS)
        self.temperatures = np.full(n, float(init_temp), dtype=np.float64)
        self.fitted: bool = False
        self.fit_report: dict[str, Any] = {}

    def fit(
        self,
        val_logits: np.ndarray,
        val_y_true: np.ndarray,
        *,
        split_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Fit positive temperatures via L-BFGS-B minimizing validation BCEWithLogitsLoss."""
        assert_validation_only(split_names, caller="TemperatureScaler.fit")
        z = np.asarray(val_logits, dtype=np.float64)
        y = np.asarray(val_y_true, dtype=np.float64)
        if z.ndim != 2 or z.shape != y.shape or z.shape[1] != len(HALLUCINATION_LABELS):
            raise ValueError(f"Expected identical (N, 5) arrays, got logits={z.shape}, y={y.shape}")
        if len(z) == 0:
            raise ValueError("Cannot fit TemperatureScaler on empty validation array.")

        raw_probs = special.expit(z)
        before_loss = _stable_bce_with_logits(z, y)

        if self.per_label:
            temps = np.ones(z.shape[1], dtype=np.float64)
            for k in range(z.shape[1]):
                zk = z[:, k]
                yk = y[:, k]
                res = optimize.minimize(
                    lambda log_t: _stable_bce_with_logits(zk / np.exp(log_t[0]), yk),
                    x0=np.array([0.0], dtype=np.float64),
                    bounds=[(-3.5, 3.5)],
                    method="L-BFGS-B",
                )
                temps[k] = float(np.exp(res.x[0]))
            self.temperatures = temps
        else:
            res = optimize.minimize(
                lambda log_t: _stable_bce_with_logits(z / np.exp(log_t[0]), y),
                x0=np.array([0.0], dtype=np.float64),
                bounds=[(-3.5, 3.5)],
                method="L-BFGS-B",
            )
            t_scalar = float(np.exp(res.x[0]))
            self.temperatures = np.full(z.shape[1], t_scalar, dtype=np.float64)

        self.fitted = True
        cal_probs = self.predict_proba(z)
        after_loss = _stable_bce_with_logits(self.transform_logits(z), y)

        raw_ece = float(np.mean([compute_ece(y[:, k], raw_probs[:, k])["ece"] for k in range(z.shape[1])]))
        cal_ece = float(np.mean([compute_ece(y[:, k], cal_probs[:, k])["ece"] for k in range(z.shape[1])]))
        raw_brier = float(np.mean([compute_brier_score(y[:, k], raw_probs[:, k]) for k in range(z.shape[1])]))
        cal_brier = float(np.mean([compute_brier_score(y[:, k], cal_probs[:, k]) for k in range(z.shape[1])]))

        self.fit_report = {
            "fitted_at_utc": utc_now_iso(),
            "per_label": self.per_label,
            "n_val_samples": int(len(z)),
            "temperatures": {
                lbl: round(float(self.temperatures[k]), 6)
                for k, lbl in enumerate(HALLUCINATION_LABELS)
            },
            "val_bce_before": round(before_loss, 6),
            "val_bce_after": round(after_loss, 6),
            "val_mean_ece_before": round(raw_ece, 6),
            "val_mean_ece_after": round(cal_ece, 6),
            "val_mean_brier_before": round(raw_brier, 6),
            "val_mean_brier_after": round(cal_brier, 6),
        }
        return dict(self.fit_report)

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        return z / np.maximum(self.temperatures[None, :], 1e-6)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw logits and return calibrated sigmoid scores."""
        return special.expit(self.transform_logits(logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_label": self.per_label,
            "fitted": self.fitted,
            "temperatures_vector": [float(t) for t in self.temperatures.tolist()],
            "fit_report": self.fit_report,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        write_json(p, self.to_dict())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        payload = read_json(path)
        scaler = cls(per_label=bool(payload.get("per_label", True)))
        scaler.temperatures = np.asarray(payload["temperatures_vector"], dtype=np.float64)
        scaler.fitted = bool(payload.get("fitted", True))
        scaler.fit_report = dict(payload.get("fit_report") or {})
        return scaler


def probs_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safely invert bounded probabilities into finite logits for post-hoc scaling."""
    clipped = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return special.logit(clipped)

