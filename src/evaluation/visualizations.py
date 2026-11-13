"""Publication-ready matplotlib visualizations for co-occurrence, PR curves, calibration, training, and E4 slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.data.schema import HALLUCINATION_LABELS
from src.evaluation.metrics import compute_ece

SHORT_LABEL_MAP = {
    "factual_inconsistency": "Cat I (FI)",
    "factual_fabrication": "Cat II (FF)",
    "instruction_inconsistency": "Cat III (II)",
    "logical_inconsistency": "Cat IV (LI)",
    "context_inconsistency": "Cat V (CI)",
}


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_label_cooccurrence_heatmap(
    y_true: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Hallucination Label Co-Occurrence Matrix",
    normalize_by_row: bool = False,
) -> Path:
    """Save 5x5 co-occurrence heatmap across hallucination categories."""
    yt = np.asarray(y_true, dtype=np.float64)
    mat = yt.T @ yt
    if normalize_by_row:
        diag = np.maximum(np.diag(mat)[:, None], 1.0)
        mat = mat / diag

    labels = [SHORT_LABEL_MAP[lbl] for lbl in HALLUCINATION_LABELS]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)

    thresh = mat.max() * 0.55 if mat.size else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text_val = f"{mat[i, j]:.2f}" if normalize_by_row else f"{int(mat[i, j])}"
            ax.text(
                j,
                i,
                text_val,
                ha="center",
                va="center",
                color="white" if mat[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    target = _ensure_parent(out_path)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_per_label_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Per-Category Precision-Recall Curves",
) -> Path:
    """Plot precision-recall curves and Average Precision (PR-AUC) for all five labels."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)

    for idx, lbl in enumerate(HALLUCINATION_LABELS):
        col_t = yt[:, idx]
        col_p = yp[:, idx]
        if len(np.unique(col_t)) < 2:
            ax.plot([], [], label=f"{SHORT_LABEL_MAP[lbl]} (N/A single class)")
            continue
        prec, rec, _ = precision_recall_curve(col_t, col_p)
        ap = average_precision_score(col_t, col_p)
        ax.plot(rec, prec, lw=1.8, label=f"{SHORT_LABEL_MAP[lbl]} (AP={ap:.3f})")

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    target = _ensure_parent(out_path)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_reliability_diagrams(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    out_path: str | Path,
    *,
    calibrated_probs: np.ndarray | None = None,
    n_bins: int = 10,
    title: str = "10-Bin Reliability Diagrams (Raw vs Temperature-Scaled)",
) -> Path:
    """Plot per-label calibration curves showing empirical positive rate vs mean predicted score."""
    yt = np.asarray(y_true, dtype=np.int64)
    rp = np.asarray(raw_probs, dtype=np.float64)
    cp = np.asarray(calibrated_probs, dtype=np.float64) if calibrated_probs is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), dpi=150)
    flat_axes = axes.ravel()

    for idx, lbl in enumerate(HALLUCINATION_LABELS):
        ax = flat_axes[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Perfect")
        raw_info = compute_ece(yt[:, idx], rp[:, idx], n_bins=n_bins)
        rx = [b["mean_predicted_prob"] for b in raw_info["bins"] if b["count"] > 0]
        ry = [b["empirical_pos_rate"] for b in raw_info["bins"] if b["count"] > 0]
        ax.plot(rx, ry, marker="o", ms=4, lw=1.6, label=f"Raw (ECE={raw_info['ece']:.3f})")
        if cp is not None:
            cal_info = compute_ece(yt[:, idx], cp[:, idx], n_bins=n_bins)
            cx = [b["mean_predicted_prob"] for b in cal_info["bins"] if b["count"] > 0]
            cy = [b["empirical_pos_rate"] for b in cal_info["bins"] if b["count"] > 0]
            ax.plot(cx, cy, marker="s", ms=4, lw=1.6, label=f"TempScaled (ECE={cal_info['ece']:.3f})")
        ax.set_title(SHORT_LABEL_MAP[lbl], fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Empirical frequency")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=7.5, loc="upper left")

    flat_axes[5].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    target = _ensure_parent(out_path)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_training_curves(
    history_rows: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    title: str = "Training Loss and Validation Macro-F1 Progression",
) -> Path:
    """Plot epoch-by-epoch train loss alongside validation Macro-F1 and loss."""
    epochs = [float(r.get("epoch", idx + 1)) for idx, r in enumerate(history_rows)]
    tr_loss = [float(r.get("train_loss", 0.0)) for r in history_rows]
    val_loss = [r.get("val_loss") for r in history_rows]
    val_f1 = [r.get("val_macro_f1") for r in history_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=150)
    ax1.plot(epochs, tr_loss, marker="o", label="Train Loss")
    if all(v is not None for v in val_loss):
        ax1.plot(epochs, [float(v) for v in val_loss], marker="s", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss Curves")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    if any(v is not None for v in val_f1):
        ax2.plot(
            epochs,
            [0.0 if v is None else float(v) for v in val_f1],
            marker="^",
            color="tab:green",
            label="Val Macro-F1 (tau=0.50)",
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_title("Validation Macro-F1")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend()

    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    target = _ensure_parent(out_path)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_combination_breakdown(
    sliced_metrics: dict[str, Any],
    out_path: str | Path,
    *,
    slice_key: str = "by_cardinality",
    metric_name: str = "macro_f1",
    title: str = "Performance Breakdown Across Label Cardinality Tiers",
) -> Path:
    """Save horizontal bar chart comparing Macro-F1 / Subset Accuracy across Suite E4 slices."""
    group_map = sliced_metrics.get(slice_key, {})
    labels = sorted(group_map.keys())
    values = [float(group_map[k].get(metric_name, 0.0)) for k in labels]
    counts = [int(group_map[k].get("n_samples", 0)) for k in labels]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.5 * len(labels) + 1.5)), dpi=150)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color="steelblue", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lbl} (n={cnt})" for lbl, cnt in zip(labels, counts)])
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, values):
        ax.text(min(1.01, val + 0.015), bar.get_y() + bar.get_height() / 2.0, f"{val:.3f}", va="center", fontsize=8.5)

    fig.tight_layout()
    target = _ensure_parent(out_path)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target

