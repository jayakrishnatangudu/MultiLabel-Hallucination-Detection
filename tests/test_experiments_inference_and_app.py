"""Tests for Phase 5 ablations/comparisons, Phase 6 CPU inference, Streamlit UI helpers, and Colab runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.components.helpers import (
    EXAMPLE_PRESETS,
    discover_available_models,
    export_prediction_json_bytes,
    format_prediction_rows,
    format_truncation_summary,
    render_sentence_highlights_html,
)
from app.main import run_app
from scripts.colab_workflow import build_stage_commands, inspect_runtime_readiness
from src.data.schema import HALLUCINATION_LABELS
from src.evaluation.metrics import compute_multilabel_metrics, resolve_ground_truth_matrix
from src.experiments.ablation import (
    build_a7_context_control_split,
    build_a8_cross_domain_fold,
    build_ablation_matrix,
    slice_a1_data_composition,
    summarize_a7_ci_false_positives,
)
from src.experiments.comparison import (
    aggregate_multi_seed_metrics,
    compare_two_models,
    generate_results_tables,
)
from src.models.baseline import run_classical_baseline_experiment
from src.models.inference import HallucinationPredictor, UNVERIFIED_PREDICTION_DISCLAIMER
from src.utils.config import load_yaml_config
from tests.test_baselines_calibration_thresholds import _make_semantic_dataset


def test_ablation_slicers_and_a7_ci_shortcut_audit() -> None:
    tr = _make_semantic_dataset(split="train", n=30, seed=5)
    va = _make_semantic_dataset(split="val", n=16, seed=6)
    te = _make_semantic_dataset(split="test_synthetic", n=16, seed=7)

    a1_single = slice_a1_data_composition(tr, "clean_plus_single")
    y_single = resolve_ground_truth_matrix(a1_single)
    assert all(k in (0, 1) for k in y_single.sum(axis=1))

    a1_multi = slice_a1_data_composition(tr, "clean_plus_multi")
    y_multi = resolve_ground_truth_matrix(a1_multi)
    assert all(k == 0 or k >= 2 for k in y_multi.sum(axis=1))

    # A7 context-unified vs naive_disjoint
    naive = build_a7_context_control_split(tr, "naive_disjoint")
    unified = build_a7_context_control_split(tr, "context_unified")
    assert len(unified) == len(tr)
    y_naive = resolve_ground_truth_matrix(naive)
    for rec, row in zip(naive, y_naive):
        if rec.has_context:
            assert row[HALLUCINATION_LABELS.index("context_inconsistency")] == 1

    # Summarize CI false-positive rate on context-present CI=0 items
    y_tr = resolve_ground_truth_matrix(tr)
    probs = np.zeros_like(y_tr, dtype=np.float64)
    probs[:, 4] = 0.85  # Force CI shortcut false positives across all items
    a7_sum = summarize_a7_ci_false_positives(tr, probs, thresholds=0.50)
    assert a7_sum["context_present_ci0_count"] > 0
    assert a7_sum["context_present_ci0_fp_rate"] == pytest.approx(1.0)

    # A8 cross-domain fold isolation
    tr_f, va_f, te_f = build_a8_cross_domain_fold(tr + va + te, held_out_domain="finance")
    assert all(r.domain != "finance" for r in tr_f + va_f)
    assert all(r.domain == "finance" for r in te_f)

    matrix = build_ablation_matrix(load_yaml_config("configs/model.yaml"), load_yaml_config("configs/training.yaml"))
    assert len(matrix) == 8
    assert [m["ablation_id"] for m in matrix] == [
        "A1_data_composition",
        "A2_threshold_sensitivity",
        "A3_calibration_temperature_scaling",
        "A4_context_processing",
        "A5_head_architecture",
        "A6_imbalance_loss",
        "A7_context_shortcut_control",
        "A8_cross_domain_leave_one_out",
    ]


def test_multi_seed_aggregation_significance_and_auto_tables(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    y_t = (rng.random((25, 5)) < 0.45).astype(np.int64)
    prob_a = np.clip(y_t * 0.30 + rng.uniform(0.30, 0.55, size=(25, 5)), 0.01, 0.99)
    prob_b = np.clip(y_t * 0.85 + rng.uniform(0.02, 0.15, size=(25, 5)), 0.01, 0.99)

    seed_evals = [
        {"metrics_calibrated_val_tuned_tau": compute_multilabel_metrics(y_t, prob_b, 0.50)},
        {"metrics_calibrated_val_tuned_tau": compute_multilabel_metrics(y_t, prob_b * 0.98, 0.50)},
        {"metrics_calibrated_val_tuned_tau": compute_multilabel_metrics(y_t, prob_b * 0.96, 0.50)},
    ]
    agg = aggregate_multi_seed_metrics(seed_evals)
    assert agg["status"] == "COMPLETED"
    assert agg["n_seeds"] == 3
    assert agg["overall"]["macro_f1"]["mean"] > 0.7

    cmp_res = compare_two_models(
        y_true=y_t,
        prob_a=prob_a,
        prob_b=prob_b,
        model_a_name="Baseline",
        model_b_name="Improved",
        n_bootstrap=30,
    )
    assert cmp_res["paired_bootstrap"]["macro_f1"]["observed_diff_b_minus_a"] > 0.0

    paths = generate_results_tables(
        [
            {"model_name": "E1.2_TFIDF", "split_tag": "test_synthetic", "aggregated": agg},
            {"model_name": "E2.3_DeBERTa", "split_tag": "test_synthetic", "aggregated": {"status": "PENDING_EMPIRICAL_RUN", "n_seeds": 0}},
        ],
        tmp_path / "tables",
    )
    md_text = paths["markdown"].read_text(encoding="utf-8")
    tex_text = paths["latex"].read_text(encoding="utf-8")
    assert "E1.2_TFIDF" in md_text
    assert "PENDING_EMPIRICAL_RUN" in md_text
    assert "\\begin{table*}" in tex_text


def test_cpu_inference_and_streamlit_helpers_end_to_end(tmp_path: Path) -> None:
    tr = _make_semantic_dataset(split="train", n=40, seed=101)
    va = _make_semantic_dataset(split="val", n=20, seed=102)
    te = _make_semantic_dataset(split="test_synthetic", n=15, seed=103)

    run_dir = tmp_path / "runs" / "tfidf_demo"
    run_classical_baseline_experiment(
        train_records=tr,
        val_records=va,
        test_records=te,
        output_dir=run_dir,
        variant="ovr_logreg",
        n_bootstrap=10,
    )

    discovered = discover_available_models([tmp_path / "runs"])
    assert len(discovered) == 1 and discovered[0]["kind"] == "TF-IDF Baseline"

    predictor = HallucinationPredictor(run_dir)
    preset = EXAMPLE_PRESETS["Multi-Label Distorted + Fabricated Response"]
    res = predictor.predict_single(
        response_text="<script>alert(1)</script> " + preset["response"],
        question_text=preset["question"],
        context_text=preset["context"],
        use_calibration=True,
        threshold_mode="val_tuned",
    )
    assert res.disclaimer == UNVERIFIED_PREDICTION_DISCLAIMER
    assert res.calibration_applied is True
    assert 0.0 <= res.any_hallucination_binary_score <= 1.0

    rows = format_prediction_rows(res)
    assert len(rows) == 5
    trunc = format_truncation_summary(res.truncation_metadata)
    assert trunc["total_tokens"] > 0

    rendered_html = render_sentence_highlights_html(res.sentence_highlights)
    assert "<script>" not in rendered_html and "&lt;script&gt;" in rendered_html
    assert len(export_prediction_json_bytes(res)) > 100
    assert callable(run_app)

    # Colab script helper validation
    cmds = build_stage_commands(python_exe="python")
    assert "phase4_train_eval" in cmds and len(cmds["phase4_train_eval"]) == 4
    assert isinstance(inspect_runtime_readiness(), dict)

