"""Comprehensive multi-label, binary Task B1, calibration, E4 slicing, and statistical significance metrics."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.schema import HALLUCINATION_LABELS, MultiLabelRecord


def resolve_ground_truth_matrix(
    records: Sequence[MultiLabelRecord | dict[str, Any]],
) -> np.ndarray:
    """
    Extract canonical ground-truth Nx5 binary matrix from records.

    Strict scientific hierarchy:
      - If adjudicated_gold_labels is present, ALWAYS prefer human gold labels.
      - Else for synthetic/pilot splits, fall back to label_vector or intended_labels.to_vector().
      - NEVER use auto_detected_labels (LLM judge) as ground truth!
      - Natural test records (`data_origin == "natural"` or `split == "test_natural"`)
        MUST carry adjudicated_gold_labels or explicit gold label_vector.
    """
    rows: list[list[int]] = []
    for rec in records:
        if isinstance(rec, MultiLabelRecord):
            origin = rec.data_origin
            split = rec.split
            gold = rec.adjudicated_gold_labels
            lvec = rec.label_vector
            intended = rec.intended_labels.to_vector() if rec.intended_labels is not None else None
            sid = rec.sample_id
        else:
            origin = rec.get("data_origin", "synthetic")
            split = rec.get("split", "train")
            gold = rec.get("adjudicated_gold_labels")
            lvec = rec.get("label_vector")
            ind = rec.get("intended_labels")
            if isinstance(ind, dict):
                intended = [int(ind[lbl]) for lbl in HALLUCINATION_LABELS]
            else:
                intended = None
            sid = rec.get("sample_id", "<unknown>")

        if gold is not None:
            vec = [int(x) for x in gold]
        elif origin == "natural" or split == "test_natural":
            raise ValueError(
                f"Natural sample {sid!r} lacks adjudicated_gold_labels; LLM judge predictions "
                "must never substitute for human gold ground truth."
            )
        elif lvec is not None:
            vec = [int(x) for x in lvec]
        elif intended is not None:
            vec = [int(x) for x in intended]
        else:
            raise ValueError(f"Record {sid!r} has neither adjudicated_gold_labels nor intended ground truth.")

        if len(vec) != len(HALLUCINATION_LABELS):
            raise ValueError(f"Record {sid!r} ground truth length {len(vec)} != {len(HALLUCINATION_LABELS)}")
        rows.append(vec)
    return np.asarray(rows, dtype=np.int64)


def _normalize_threshold_array(
    thresholds: float | Sequence[float] | np.ndarray,
    n_labels: int = 5,
) -> np.ndarray:
    if isinstance(thresholds, (int, float)):
        return np.full(n_labels, float(thresholds), dtype=np.float64)
    arr = np.asarray(list(thresholds), dtype=np.float64)
    if arr.shape != (n_labels,):
        raise ValueError(f"Expected {n_labels} thresholds, got shape {arr.shape}")
    return arr


def binarize_probabilities(
    y_prob: np.ndarray,
    thresholds: float | Sequence[float] | np.ndarray = 0.50,
) -> np.ndarray:
    """Convert probability array (N, K) into binary predictions using scalar or per-label thresholds."""
    probs = np.asarray(y_prob, dtype=np.float64)
    thr = _normalize_threshold_array(thresholds, probs.shape[1])
    return (probs >= thr[None, :]).astype(np.int64)


def compute_ece(
    y_true_1d: np.ndarray,
    y_prob_1d: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute 10-bin Expected Calibration Error plus per-bin statistics for reliability diagrams."""
    y_t = np.asarray(y_true_1d, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob_1d, dtype=np.float64), 0.0, 1.0)
    n = len(y_t)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out: list[dict[str, Any]] = []
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = float(edges[idx]), float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (y_p >= lo) & (y_p <= hi)
        else:
            mask = (y_p >= lo) & (y_p < hi)
        count = int(mask.sum())
        if count > 0:
            acc = float(y_t[mask].mean())
            conf = float(y_p[mask].mean())
            ece += (count / max(1, n)) * abs(acc - conf)
        else:
            acc = None
            conf = None
        bins_out.append(
            {
                "bin_lower": round(lo, 4),
                "bin_upper": round(hi, 4),
                "count": count,
                "empirical_pos_rate": None if acc is None else round(acc, 6),
                "mean_predicted_prob": None if conf is None else round(conf, 6),
            }
        )
    return {
        "ece": round(float(ece), 6),
        "n_bins": n_bins,
        "bins": bins_out,
    }


def compute_brier_score(y_true_1d: np.ndarray, y_prob_1d: np.ndarray) -> float:
    """Compute Brier score MSE between probabilities and binary ground truth."""
    y_t = np.asarray(y_true_1d, dtype=np.float64)
    y_p = np.asarray(y_prob_1d, dtype=np.float64)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean((y_p - y_t) ** 2))


def _safe_pr_auc(y_true_1d: np.ndarray, y_prob_1d: np.ndarray) -> float | None:
    if len(y_true_1d) == 0 or len(np.unique(y_true_1d)) < 2:
        return None
    return float(average_precision_score(y_true_1d, y_prob_1d))


def _safe_roc_auc(y_true_1d: np.ndarray, y_prob_1d: np.ndarray) -> float | None:
    if len(y_true_1d) == 0 or len(np.unique(y_true_1d)) < 2:
        return None
    return float(roc_auc_score(y_true_1d, y_prob_1d))


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: float | Sequence[float] | np.ndarray = 0.50,
    *,
    latency_ms_per_sample: float | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    n_ece_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute complete Task M5, Task B1 binary check, per-label PR/ROC-AUC, calibration, and truncation telemetry.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)
    if yt.ndim != 2 or yp.shape != yt.shape or yt.shape[1] != len(HALLUCINATION_LABELS):
        raise ValueError(f"Invalid shapes y_true={yt.shape}, y_prob={yp.shape}")

    n = yt.shape[0]
    thr_arr = _normalize_threshold_array(thresholds, yt.shape[1])
    y_pred = binarize_probabilities(yp, thr_arr)

    overall = {
        "n_samples": int(n),
        "micro_precision": round(float(precision_score(yt, y_pred, average="micro", zero_division=0)), 6),
        "micro_recall": round(float(recall_score(yt, y_pred, average="micro", zero_division=0)), 6),
        "micro_f1": round(float(f1_score(yt, y_pred, average="micro", zero_division=0)), 6),
        "macro_precision": round(float(precision_score(yt, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(yt, y_pred, average="macro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(yt, y_pred, average="macro", zero_division=0)), 6),
        "weighted_precision": round(float(precision_score(yt, y_pred, average="weighted", zero_division=0)), 6),
        "weighted_recall": round(float(recall_score(yt, y_pred, average="weighted", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(yt, y_pred, average="weighted", zero_division=0)), 6),
        "samples_precision": round(float(precision_score(yt, y_pred, average="samples", zero_division=0)), 6),
        "samples_recall": round(float(recall_score(yt, y_pred, average="samples", zero_division=0)), 6),
        "samples_f1": round(float(f1_score(yt, y_pred, average="samples", zero_division=0)), 6),
        "hamming_loss": round(float(hamming_loss(yt, y_pred)), 6),
        "subset_accuracy": round(float(accuracy_score(yt, y_pred)), 6),
        "thresholds_used": [round(float(t), 4) for t in thr_arr.tolist()],
    }

    per_label: dict[str, Any] = {}
    pr_aucs: list[float] = []
    roc_aucs: list[float] = []
    eces: list[float] = []
    briers: list[float] = []

    for idx, lbl in enumerate(HALLUCINATION_LABELS):
        col_t = yt[:, idx]
        col_p = yp[:, idx]
        col_pred = y_pred[:, idx]
        tp = int(((col_t == 1) & (col_pred == 1)).sum())
        fp = int(((col_t == 0) & (col_pred == 1)).sum())
        fn = int(((col_t == 1) & (col_pred == 0)).sum())
        tn = int(((col_t == 0) & (col_pred == 0)).sum())
        pr_val = _safe_pr_auc(col_t, col_p)
        roc_val = _safe_roc_auc(col_t, col_p)
        ece_info = compute_ece(col_t, col_p, n_bins=n_ece_bins)
        brier_val = compute_brier_score(col_t, col_p)
        if pr_val is not None:
            pr_aucs.append(pr_val)
        if roc_val is not None:
            roc_aucs.append(roc_val)
        eces.append(ece_info["ece"])
        briers.append(brier_val)

        per_label[lbl] = {
            "precision": round(float(precision_score(col_t, col_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(col_t, col_pred, zero_division=0)), 6),
            "f1": round(float(f1_score(col_t, col_pred, zero_division=0)), 6),
            "support": int(col_t.sum()),
            "predicted_positives": int(col_pred.sum()),
            "pr_auc": None if pr_val is None else round(pr_val, 6),
            "roc_auc": None if roc_val is None else round(roc_val, 6),
            "ece": ece_info["ece"],
            "ece_bins": ece_info["bins"],
            "brier_score": round(brier_val, 6),
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "threshold": round(float(thr_arr[idx]), 4),
        }

    overall["mean_pr_auc"] = round(float(np.mean(pr_aucs)), 6) if pr_aucs else None
    overall["mean_roc_auc"] = round(float(np.mean(roc_aucs)), 6) if roc_aucs else None

    calibration = {
        "mean_ece": round(float(np.mean(eces)), 6) if eces else 0.0,
        "mean_brier_score": round(float(np.mean(briers)), 6) if briers else 0.0,
        "per_label_ece": {lbl: per_label[lbl]["ece"] for lbl in HALLUCINATION_LABELS},
        "per_label_brier": {lbl: per_label[lbl]["brier_score"] for lbl in HALLUCINATION_LABELS},
    }

    # Strict Task B1 separation: binary "any hallucination present" check
    any_true = (yt.sum(axis=1) > 0).astype(np.int64)
    any_pred = (y_pred.sum(axis=1) > 0).astype(np.int64)
    any_score = yp.max(axis=1)
    task_b1 = {
        "definition": "binary any_hallucination_present (sum(y)>0); max_k s_k for continuous score",
        "positive_rate_true": round(float(any_true.mean()), 6) if n else 0.0,
        "accuracy": round(float(accuracy_score(any_true, any_pred)), 6) if n else 0.0,
        "precision": round(float(precision_score(any_true, any_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(any_true, any_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(any_true, any_pred, zero_division=0)), 6),
        "pr_auc": (lambda v: None if v is None else round(v, 6))(_safe_pr_auc(any_true, any_score)),
        "roc_auc": (lambda v: None if v is None else round(v, 6))(_safe_roc_auc(any_true, any_score)),
    }

    operational: dict[str, Any] = {}
    if latency_ms_per_sample is not None:
        operational["latency_ms_per_sample"] = round(float(latency_ms_per_sample), 4)
    if metadata_rows is not None and len(metadata_rows) > 0:
        operational["context_truncated_pct"] = round(
            100.0 * sum(1 for m in metadata_rows if m.get("context_truncated")) / len(metadata_rows), 2
        )
        operational["response_truncated_pct"] = round(
            100.0 * sum(1 for m in metadata_rows if m.get("response_truncated")) / len(metadata_rows), 2
        )

    return {
        "overall": overall,
        "per_label": per_label,
        "calibration": calibration,
        "task_b1_binary": task_b1,
        "operational": operational,
    }


def _cardinality_bucket(vec_row: np.ndarray) -> str:
    k = int(vec_row.sum())
    return "G4+" if k >= 4 else f"G{k}"


def _rec_attr(rec: MultiLabelRecord | dict[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(rec, MultiLabelRecord):
        return getattr(rec, name, default)
    return rec.get(name, default)


def compute_sliced_e4_metrics(
    records: Sequence[MultiLabelRecord | dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: float | Sequence[float] | np.ndarray = 0.50,
    *,
    metadata_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute Suite E4 sliced evaluations across domain, dataset, context regime,
    label cardinality, exact label combinations / key pairs, and truncation status.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)
    if len(records) != len(yt):
        raise ValueError(f"records len {len(records)} != y_true len {len(yt)}")

    def _slice_indices(groups: dict[str, list[int]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, idxs in sorted(groups.items()):
            if not idxs:
                continue
            arr = np.asarray(idxs, dtype=np.int64)
            sub_meta = [metadata_rows[i] for i in idxs] if metadata_rows is not None else None
            sub = compute_multilabel_metrics(yt[arr], yp[arr], thresholds, metadata_rows=sub_meta)
            out[key] = {
                "n_samples": len(idxs),
                "macro_f1": sub["overall"]["macro_f1"],
                "micro_f1": sub["overall"]["micro_f1"],
                "hamming_loss": sub["overall"]["hamming_loss"],
                "subset_accuracy": sub["overall"]["subset_accuracy"],
                "mean_ece": sub["calibration"]["mean_ece"],
                "mean_brier_score": sub["calibration"]["mean_brier_score"],
                "task_b1_f1": sub["task_b1_binary"]["f1"],
                "context_inconsistency_fp_rate_on_zero": (
                    round(
                        float(
                            (
                                (yt[arr][:, 4] == 0)
                                & (binarize_probabilities(yp[arr], thresholds)[:, 4] == 1)
                            ).sum()
                            / max(1, int((yt[arr][:, 4] == 0).sum()))
                        ),
                        6,
                    )
                    if int((yt[arr][:, 4] == 0).sum()) > 0
                    else 0.0
                ),
            }
        return out

    by_domain: dict[str, list[int]] = defaultdict(list)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    by_context: dict[str, list[int]] = defaultdict(list)
    by_cardinality: dict[str, list[int]] = defaultdict(list)
    by_combination: dict[str, list[int]] = defaultdict(list)
    by_pair_cooccurrence: dict[str, list[int]] = defaultdict(list)
    by_truncation: dict[str, list[int]] = defaultdict(list)

    short_names = ("FI", "FF", "II", "LI", "CI")
    for idx, rec in enumerate(records):
        by_domain[str(_rec_attr(rec, "domain"))].append(idx)
        by_dataset[str(_rec_attr(rec, "source_dataset"))].append(idx)
        ctx_key = "with_context" if bool(_rec_attr(rec, "has_context")) else "no_context"
        by_context[ctx_key].append(idx)

        row_t = yt[idx]
        by_cardinality[_cardinality_bucket(row_t)].append(idx)
        bits = "".join(str(int(v)) for v in row_t.tolist())
        by_combination[bits].append(idx)

        active_idx = [j for j, v in enumerate(row_t.tolist()) if v == 1]
        for a_i in range(len(active_idx)):
            for b_i in range(a_i + 1, len(active_idx)):
                pair_key = f"{short_names[active_idx[a_i]]}+{short_names[active_idx[b_i]]}"
                by_pair_cooccurrence[pair_key].append(idx)

        if metadata_rows is not None and idx < len(metadata_rows):
            trunc_key = "context_truncated" if metadata_rows[idx].get("context_truncated") else "context_intact"
            by_truncation[trunc_key].append(idx)

    return {
        "by_domain": _slice_indices(by_domain),
        "by_dataset": _slice_indices(by_dataset),
        "by_context_regime": _slice_indices(by_context),
        "by_cardinality": _slice_indices(by_cardinality),
        "by_combination_bits": _slice_indices(by_combination),
        "by_cooccurring_pair": _slice_indices(by_pair_cooccurrence),
        "by_truncation": _slice_indices(by_truncation),
    }


def _build_cluster_index(
    n_samples: int,
    group_ids: Sequence[str] | None = None,
) -> list[np.ndarray]:
    if group_ids is None:
        return [np.asarray([i], dtype=np.int64) for i in range(n_samples)]
    if len(group_ids) != n_samples:
        raise ValueError(f"group_ids len {len(group_ids)} != n_samples {n_samples}")
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        buckets[str(gid)].append(i)
    return [np.asarray(idxs, dtype=np.int64) for _, idxs in sorted(buckets.items())]


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    group_ids: Sequence[str] | None = None,
    thresholds: float | Sequence[float] | np.ndarray = 0.50,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute grouped 95% bootstrap confidence intervals over unique item/question clusters.

    Preserves shared source_group_id clusters (essential for multi-model test_natural)
    so CI width is never artificially narrowed by correlated question repeats.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float64)
    y_pred = binarize_probabilities(yp, thresholds)
    clusters = _build_cluster_index(len(yt), group_ids)
    n_clusters = len(clusters)
    if n_clusters == 0:
        return {}

    rng = np.random.default_rng(seed)
    tracked = ("macro_f1", "micro_f1", "hamming_loss", "subset_accuracy", "mean_ece", "mean_brier_score")
    samples: dict[str, list[float]] = {k: [] for k in tracked}

    for _ in range(max(1, int(n_bootstrap))):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([clusters[c] for c in drawn])
        bt, bp, bpred = yt[idx], yp[idx], y_pred[idx]
        samples["macro_f1"].append(float(f1_score(bt, bpred, average="macro", zero_division=0)))
        samples["micro_f1"].append(float(f1_score(bt, bpred, average="micro", zero_division=0)))
        samples["hamming_loss"].append(float(hamming_loss(bt, bpred)))
        samples["subset_accuracy"].append(float(accuracy_score(bt, bpred)))
        samples["mean_ece"].append(
            float(np.mean([compute_ece(bt[:, k], bp[:, k])["ece"] for k in range(bt.shape[1])]))
        )
        samples["mean_brier_score"].append(
            float(np.mean([compute_brier_score(bt[:, k], bp[:, k]) for k in range(bt.shape[1])]))
        )

    out: dict[str, dict[str, float]] = {}
    for metric_name, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[metric_name] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 6),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 6),
        }
    return out


def mcnemar_exact_or_chi2_test(
    y_true_1d: np.ndarray,
    pred_a_1d: np.ndarray,
    pred_b_1d: np.ndarray,
) -> dict[str, Any]:
    """Compare paired binary predictions using exact McNemar test (<25 discordant) or chi-squared."""
    yt = np.asarray(y_true_1d, dtype=np.int64)
    pa = np.asarray(pred_a_1d, dtype=np.int64)
    pb = np.asarray(pred_b_1d, dtype=np.int64)
    corr_a = pa == yt
    corr_b = pb == yt
    b = int((corr_a & ~corr_b).sum())  # A right, B wrong
    c = int((~corr_a & corr_b).sum())  # A wrong, B right
    discordant = b + c
    if discordant == 0:
        return {"b": b, "c": c, "discordant": 0, "method": "no_discordant", "statistic": 0.0, "p_value": 1.0}
    if discordant < 25:
        res = stats.binomtest(k=min(b, c), n=discordant, p=0.5, alternative="two-sided")
        return {
            "b": b,
            "c": c,
            "discordant": discordant,
            "method": "exact_binomial",
            "statistic": float(min(b, c)),
            "p_value": round(float(res.pvalue), 8),
        }
    stat = ((abs(b - c) - 1.0) ** 2) / discordant
    pval = float(stats.chi2.sf(stat, df=1))
    return {
        "b": b,
        "c": c,
        "discordant": discordant,
        "method": "edwards_chi2",
        "statistic": round(float(stat), 6),
        "p_value": round(pval, 8),
    }


def paired_bootstrap_significance(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    thresholds_a: float | Sequence[float] | np.ndarray = 0.50,
    thresholds_b: float | Sequence[float] | np.ndarray = 0.50,
    group_ids: Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Paired cluster bootstrap test comparing Model B against Model A on Macro-F1, Micro-F1, and Subset Accuracy.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    pa = binarize_probabilities(prob_a, thresholds_a)
    pb = binarize_probabilities(prob_b, thresholds_b)
    clusters = _build_cluster_index(len(yt), group_ids)
    n_clusters = len(clusters)
    if n_clusters == 0:
        return {}

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"macro_f1": [], "micro_f1": [], "subset_accuracy": []}
    for _ in range(max(1, int(n_bootstrap))):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([clusters[c] for c in drawn])
        bt, bpa, bpb = yt[idx], pa[idx], pb[idx]
        deltas["macro_f1"].append(
            float(f1_score(bt, bpb, average="macro", zero_division=0) - f1_score(bt, bpa, average="macro", zero_division=0))
        )
        deltas["micro_f1"].append(
            float(f1_score(bt, bpb, average="micro", zero_division=0) - f1_score(bt, bpa, average="micro", zero_division=0))
        )
        deltas["subset_accuracy"].append(
            float(accuracy_score(bt, bpb) - accuracy_score(bt, bpa))
        )

    observed = {
        "macro_f1": float(f1_score(yt, pb, average="macro", zero_division=0) - f1_score(yt, pa, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, pb, average="micro", zero_division=0) - f1_score(yt, pa, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(yt, pb) - accuracy_score(yt, pa)),
    }

    summary: dict[str, Any] = {}
    for k, vals in deltas.items():
        arr = np.asarray(vals, dtype=np.float64)
        # Centered bootstrap p-value under H0: delta == 0
        p_ge = float((np.abs(arr - arr.mean()) >= abs(observed[k])).mean())
        summary[k] = {
            "observed_diff_b_minus_a": round(observed[k], 6),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 6),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 6),
            "two_sided_p_value": round(min(1.0, max(1.0 / (len(arr) + 1), p_ge)), 6),
        }

    # Add per-label & exact-match McNemar summary
    exact_a = (pa == yt).all(axis=1).astype(np.int64)
    exact_b = (pb == yt).all(axis=1).astype(np.int64)
    summary["mcnemar_exact_match"] = mcnemar_exact_or_chi2_test(np.ones_like(exact_a), exact_a, exact_b)
    summary["mcnemar_per_label"] = {
        lbl: mcnemar_exact_or_chi2_test(yt[:, j], pa[:, j], pb[:, j])
        for j, lbl in enumerate(HALLUCINATION_LABELS)
    }
    return summary


def wilcoxon_paired_test(
    scores_a: Sequence[float] | np.ndarray,
    scores_b: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Run paired Wilcoxon signed-rank test on per-sample continuous or ordinal metric arrays."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError("scores_a and scores_b must share non-empty identical shape.")
    diff = b - a
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff_b_minus_a": 0.0, "zero_all": True}
    res = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    pval = float(res.pvalue) if not math.isnan(float(res.pvalue)) else 1.0
    return {
        "statistic": round(float(res.statistic), 6),
        "p_value": round(pval, 8),
        "mean_diff_b_minus_a": round(float(diff.mean()), 6),
        "zero_all": False,
    }

