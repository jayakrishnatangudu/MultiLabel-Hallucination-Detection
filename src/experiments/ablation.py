"""Phase 5 Suite E3 ablation data slicers (A1, A7, A8), A7 shortcut audit, and matrix builder."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

from src.data.schema import HALLUCINATION_LABELS, MultiLabelRecord
from src.evaluation.metrics import binarize_probabilities, resolve_ground_truth_matrix
from src.utils.config import load_yaml_config
from src.utils.io import read_jsonl, write_json


def _rec_field(rec: MultiLabelRecord | dict[str, Any], key: str) -> Any:
    return getattr(rec, key) if isinstance(rec, MultiLabelRecord) else rec.get(key)


def slice_a1_data_composition(
    records: Sequence[MultiLabelRecord],
    condition: str,
) -> list[MultiLabelRecord]:
    """
    Suite A1 training data slicer:
      - 'clean_plus_single': G0 + G1 only (tests whether single-label models detect multi-label cases)
      - 'clean_plus_multi': G0 + G2..G5 only
      - 'clean_plus_single_and_multi': full G0..G5 mixture
    """
    y = resolve_ground_truth_matrix(records)
    card = y.sum(axis=1)
    if condition == "clean_plus_single":
        return [r for r, c in zip(records, card) if c in (0, 1)]
    if condition == "clean_plus_multi":
        return [r for r, c in zip(records, card) if c == 0 or c >= 2]
    if condition == "clean_plus_single_and_multi":
        return list(records)
    raise ValueError(f"Unknown A1 condition={condition!r}")


def build_a7_context_control_split(
    records: Sequence[MultiLabelRecord],
    condition: str,
) -> list[MultiLabelRecord]:
    """
    Suite A7 shortcut-learning control:
      - 'naive_disjoint': removes CI=0 records that have context (`pubmedqa`, `cuad_qa`), replicating
        the naive flaw where has_context=True correlates trivially with CI=1.
      - 'context_unified': retains clean/perturbed CI=0 records inside context datasets.
    """
    y = resolve_ground_truth_matrix(records)
    ci_col = y[:, HALLUCINATION_LABELS.index("context_inconsistency")]
    if condition == "context_unified":
        return list(records)
    if condition == "naive_disjoint":
        return [
            r
            for r, ci in zip(records, ci_col)
            if not (bool(r.has_context) and int(ci) == 0)
        ]
    raise ValueError(f"Unknown A7 condition={condition!r}")


def summarize_a7_ci_false_positives(
    records: Sequence[MultiLabelRecord | dict[str, Any]],
    y_prob: np.ndarray,
    thresholds: float | Sequence[float] | np.ndarray = 0.50,
) -> dict[str, Any]:
    """
    Measure Context Inconsistency false-positive shortcut rate specifically on
    context-present CI=0 controls (`pubmedqa`, `cuad_qa`).
    """
    y_true = resolve_ground_truth_matrix(records)
    y_pred = binarize_probabilities(y_prob, thresholds)
    ci_idx = HALLUCINATION_LABELS.index("context_inconsistency")

    ctx_ci0_idxs = [
        i
        for i, rec in enumerate(records)
        if bool(_rec_field(rec, "has_context")) and int(y_true[i, ci_idx]) == 0
    ]
    ctx_ci1_idxs = [
        i
        for i, rec in enumerate(records)
        if bool(_rec_field(rec, "has_context")) and int(y_true[i, ci_idx]) == 1
    ]
    n_ci0 = len(ctx_ci0_idxs)
    fp_ci = sum(int(y_pred[i, ci_idx] == 1) for i in ctx_ci0_idxs)
    n_ci1 = len(ctx_ci1_idxs)
    tp_ci = sum(int(y_pred[i, ci_idx] == 1) for i in ctx_ci1_idxs)
    return {
        "context_present_ci0_count": n_ci0,
        "context_present_ci0_false_positives": fp_ci,
        "context_present_ci0_fp_rate": round(fp_ci / max(1, n_ci0), 6) if n_ci0 else 0.0,
        "context_present_ci1_count": n_ci1,
        "context_present_ci1_recall": round(tp_ci / max(1, n_ci1), 6) if n_ci1 else 0.0,
    }


def build_a8_cross_domain_fold(
    records: Sequence[MultiLabelRecord],
    *,
    held_out_domain: str,
) -> tuple[list[MultiLabelRecord], list[MultiLabelRecord], list[MultiLabelRecord]]:
    """
    Suite A8 cross-domain leave-one-domain-out fold:
    Returns `(train_in_domain, val_in_domain, test_held_out_domain)` while enforcing split isolation.
    """
    if held_out_domain not in ("medical", "legal", "finance"):
        raise ValueError(f"Unsupported domain={held_out_domain!r}")
    train_recs = [r for r in records if r.split == "train" and r.domain != held_out_domain]
    val_recs = [r for r in records if r.split == "val" and r.domain != held_out_domain]
    test_recs = [r for r in records if r.split == "test_synthetic" and r.domain == held_out_domain]
    # Also permit explicit val fallback when running small test fixtures without test_synthetic rows
    if not test_recs:
        test_recs = [r for r in records if r.domain == held_out_domain and r.split != "train"]
    return train_recs, val_recs, test_recs


def build_ablation_matrix(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return declarative runnable experiment specifications covering Suites A1 through A8."""
    return [
        {
            "ablation_id": "A1_data_composition",
            "research_question": "Does training on single-label data detect co-occurring hallucinations?",
            "conditions": ["clean_plus_single", "clean_plus_multi", "clean_plus_single_and_multi"],
            "backbone": "deberta_v3",
        },
        {
            "ablation_id": "A2_threshold_sensitivity",
            "research_question": "Gain & stability of val-only per-label threshold tuning vs fixed 0.50.",
            "conditions": ["tau_0.50", "val_tuned_per_label_tau"],
            "grid": train_cfg.get("thresholding", {}),
        },
        {
            "ablation_id": "A3_calibration_temperature_scaling",
            "research_question": "ECE & Brier improvement under global vs per-label Temperature Scaling.",
            "conditions": ["raw_sigmoid", "global_temperature", "per_label_temperature"],
        },
        {
            "ablation_id": "A4_context_processing",
            "research_question": "512-token response-preserving truncation vs sliding window vs ModernBERT 2048/4096.",
            "conditions": [
                {"backbone": "deberta_v3", "context_mode": "budgeted_truncation", "max_length": 512},
                {"backbone": "deberta_v3", "context_mode": "sliding_window", "max_length": 512},
                {"backbone": "modernbert", "context_mode": "long_context", "max_length": int(model_cfg["backbones"]["modernbert"]["max_length"])},
            ],
        },
        {
            "ablation_id": "A5_head_architecture",
            "research_question": "Independent 5-sigmoids vs Classifier Chain vs Label-Interaction Residual head.",
            "conditions": ["independent_sigmoid", "classifier_chain", "label_interaction"],
            "backbone": "deberta_v3",
        },
        {
            "ablation_id": "A6_imbalance_loss",
            "research_question": "Unweighted BCE vs capped pos_weight BCE vs Asymmetric Focal Loss.",
            "conditions": ["bce", "weighted_bce", "asymmetric_focal"],
            "backbone": "deberta_v3",
        },
        {
            "ablation_id": "A7_context_shortcut_control",
            "research_question": "Effect of context-unified non-CI controls on PubMedQA/CUAD CI false positives.",
            "conditions": ["naive_disjoint", "context_unified"],
            "backbone": "deberta_v3",
        },
        {
            "ablation_id": "A8_cross_domain_leave_one_out",
            "research_question": "Zero-shot cross-domain transfer across Medical, Legal, and Finance.",
            "held_out_domains": ["medical", "legal", "finance"],
            "backbone": "deberta_v3",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and export Suite E3 ablation matrix (A1-A8).")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--output-json", default="results/ablation_matrix.json")
    parser.add_argument("--inspect-a1-jsonl", default=None, help="Optional JSONL to inspect A1 slice counts.")
    args = parser.parse_args()

    m_cfg = load_yaml_config(args.model_config)
    t_cfg = load_yaml_config(args.training_config)
    matrix = build_ablation_matrix(m_cfg, t_cfg)
    payload: dict[str, Any] = {"ablations": matrix}
    if args.inspect_a1_jsonl:
        recs = [MultiLabelRecord(**d) for d in read_jsonl(args.inspect_a1_jsonl)]
        payload["a1_counts"] = {
            cond: len(slice_a1_data_composition(recs, cond))
            for cond in ("clean_plus_single", "clean_plus_multi", "clean_plus_single_and_multi")
        }
    write_json(args.output_json, payload)
    print(f"Exported {len(matrix)} ablation specifications -> {args.output_json}")


if __name__ == "__main__":
    main()

