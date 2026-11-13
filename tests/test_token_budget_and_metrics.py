"""Unit and integration tests for TokenBudgetAllocator, metrics, E4 slicing, statistical tests, and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.schema import HALLUCINATION_LABELS, LabelDictionary, LicenseMetadata, MultiLabelRecord
from src.data.token_budget import (
    TokenBudgetAllocator,
    export_split_token_budget_report,
    summarize_truncation_rates,
)
from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_brier_score,
    compute_ece,
    compute_multilabel_metrics,
    compute_sliced_e4_metrics,
    mcnemar_exact_or_chi2_test,
    paired_bootstrap_significance,
    resolve_ground_truth_matrix,
    wilcoxon_paired_test,
)
from src.evaluation.visualizations import (
    plot_combination_breakdown,
    plot_label_cooccurrence_heatmap,
    plot_per_label_pr_curves,
    plot_reliability_diagrams,
    plot_training_curves,
)


def _dummy_record(
    idx: int,
    bits: str,
    *,
    domain: str = "medical",
    dataset: str = "pubmedqa",
    has_context: bool = True,
    split: str = "test_synthetic",
    data_origin: str = "synthetic",
    gold: list[int] | None = None,
) -> MultiLabelRecord:
    vec = [int(c) for c in bits]
    return MultiLabelRecord(
        sample_id=f"SAMPLE_{idx:03d}",
        source_dataset=dataset,
        source_record_id=f"SRC_{idx:03d}",
        source_group_id=f"GRP_{idx // 2:03d}",
        domain=domain,  # type: ignore[arg-type]
        data_origin=data_origin,  # type: ignore[arg-type]
        has_context=has_context,
        question=f"Question {idx} regarding clinical evidence?",
        context=("Long evidence clause. " * 25) if has_context else None,
        original_answer="Original factual reference answer with detailed clinical guidance.",
        generated_answer="Generated candidate answer " + ("stating distorted details " * 5),
        generation_method="synthetic_perturbation" if data_origin == "synthetic" else "natural_llm_response",
        generation_strategy="strategy_a",
        intended_labels=None if data_origin == "natural" else LabelDictionary(**dict(zip(HALLUCINATION_LABELS, vec))),
        adjudicated_gold_labels=gold,
        label_vector=gold if data_origin == "natural" else vec,
        licensing=LicenseMetadata(
            upstream_license="MIT",
            license_status="verified_open",
            redistribution_permitted=True,
            attribution="Test Fixture",
            notes="Unit test synthetic record.",
        ),
        split=split,  # type: ignore[arg-type]
    )


def test_token_budget_allocator_preserves_response_before_long_context() -> None:
    allocator = TokenBudgetAllocator(
        max_length=128,
        response_budget=64,
        question_budget=24,
        min_question_budget=10,
    )
    q = "What is the primary emergency treatment recommended for anaphylaxis in adults?"
    r = " ".join([f"resptoken{i}" for i in range(45)])
    c = " ".join([f"ctxtoken{i}" for i in range(250)])

    enc = allocator.encode_sample(question=q, response=r, context=c)
    meta = enc.metadata.to_dict()
    assert set(meta.keys()) == {
        "question_tokens_orig",
        "question_tokens_kept",
        "question_truncated",
        "response_tokens_orig",
        "response_tokens_kept",
        "response_truncated",
        "context_tokens_orig",
        "context_tokens_kept",
        "context_truncated",
        "total_sequence_tokens",
        "max_length_limit",
    }
    assert meta["total_sequence_tokens"] <= 128
    assert meta["max_length_limit"] == 128
    assert meta["response_truncated"] is False
    assert meta["response_tokens_kept"] == 45
    assert meta["context_truncated"] is True
    assert meta["context_tokens_kept"] < meta["context_tokens_orig"]

    # Empty context expands response capacity naturally
    r_long = " ".join([f"noctxresp{i}" for i in range(85)])
    enc_no_ctx = allocator.encode_sample(question="Short question?", response=r_long, context=None)
    assert enc_no_ctx.metadata.response_truncated is False
    assert enc_no_ctx.metadata.context_tokens_orig == 0

    # Extreme overflow response preserves question floor before trimming response tail
    r_huge = " ".join([f"hugeresp{i}" for i in range(200)])
    enc_huge = allocator.encode_sample(question=q, response=r_huge, context=c)
    assert enc_huge.metadata.question_tokens_kept >= 10
    assert enc_huge.metadata.response_truncated is True
    assert enc_huge.metadata.total_sequence_tokens <= 128

    # Sliding window retains full response inside every chunk
    chunks = allocator.build_sliding_window_chunks(question=q, response=r, context=c, stride=16, max_chunks=8)
    assert len(chunks) >= 2
    assert all(ch.total_chunks == len(chunks) for ch in chunks)
    assert all(not ch.metadata.response_truncated for ch in chunks)
    assert summarize_truncation_rates([enc.metadata, enc_no_ctx.metadata])["count"] == 2


def test_resolve_ground_truth_rejects_unannotated_natural_samples() -> None:
    unannotated_nat = MultiLabelRecord(
        sample_id="NAT_UNANN",
        source_dataset="medquad",
        source_record_id="NAT_SRC_1",
        source_group_id="GRP_NAT",
        domain="medical",
        data_origin="natural",
        has_context=False,
        question="Question?",
        original_answer="Original factual reference answer with enough length.",
        generated_answer="Natural generated candidate answer text.",
        generation_method="natural_llm_response",
        generation_strategy="natural_qa",
        licensing=LicenseMetadata(
            upstream_license="MIT",
            license_status="verified_open",
            redistribution_permitted=True,
            attribution="Test Fixture",
            notes="Unit test natural record.",
        ),
        split="test_natural",
    )
    with pytest.raises(ValueError, match="lacks adjudicated_gold_labels"):
        resolve_ground_truth_matrix([unannotated_nat])


def test_multilabel_metrics_calibration_e4_slicing_and_statistics(tmp_path: Path) -> None:
    bits_list = [
        "00000",
        "10000",
        "11000",
        "10101",
        "01010",
        "11111",
        "00001",
        "00110",
        "11010",
        "00000",
    ]
    records = [
        _dummy_record(
            i,
            bits,
            domain="legal" if i % 2 == 0 else "medical",
            dataset="cuad_qa" if i % 2 == 0 else "pubmedqa",
            has_context=True,
        )
        for i, bits in enumerate(bits_list)
    ]
    y_true = resolve_ground_truth_matrix(records)
    rng = np.random.default_rng(7)
    y_prob = np.clip(y_true * 0.80 + rng.uniform(0.05, 0.25, size=y_true.shape), 0.01, 0.99)
    meta = [{"context_truncated": i % 3 == 0, "response_truncated": False} for i in range(len(records))]

    m = compute_multilabel_metrics(
        y_true, y_prob, thresholds=0.50, latency_ms_per_sample=4.2, metadata_rows=meta
    )
    assert set(m["per_label"].keys()) == set(HALLUCINATION_LABELS)
    assert m["overall"]["macro_f1"] > 0.70
    assert m["task_b1_binary"]["recall"] >= 0.80
    assert 0.0 <= m["calibration"]["mean_ece"] <= 1.0
    assert m["operational"]["context_truncated_pct"] == pytest.approx(40.0)
    assert compute_brier_score(y_true[:, 0], y_prob[:, 0]) >= 0.0
    assert compute_ece(y_true[:, 0], y_prob[:, 0], n_bins=10)["n_bins"] == 10

    sliced = compute_sliced_e4_metrics(records, y_true, y_prob, thresholds=0.50, metadata_rows=meta)
    assert set(sliced["by_domain"].keys()) == {"legal", "medical"}
    assert "G0" in sliced["by_cardinality"]
    assert "FI+FF" in sliced["by_cooccurring_pair"]
    assert set(sliced["by_truncation"].keys()) == {"context_intact", "context_truncated"}

    gids = [r.source_group_id for r in records]
    ci = bootstrap_confidence_intervals(y_true, y_prob, group_ids=gids, n_bootstrap=30, seed=42)
    assert ci["macro_f1"]["ci95_low"] <= ci["macro_f1"]["mean"] <= ci["macro_f1"]["ci95_high"]

    # Significance helpers: McNemar exact + Edwards chi2 + paired bootstrap + Wilcoxon
    mc_small = mcnemar_exact_or_chi2_test(np.ones(10), np.array([1]*8 + [0]*2), np.array([1]*5 + [0]*5))
    assert mc_small["method"] == "exact_binomial"
    y_large = np.ones(60, dtype=np.int64)
    pa_large = np.array([1] * 50 + [0] * 10)
    pb_large = np.array([1] * 20 + [0] * 40)
    mc_large = mcnemar_exact_or_chi2_test(y_large, pa_large, pb_large)
    assert mc_large["method"] == "edwards_chi2"

    y_worse = np.clip(y_true * 0.45 + rng.uniform(0.1, 0.5, size=y_true.shape), 0.01, 0.99)
    sig = paired_bootstrap_significance(y_true, y_worse, y_prob, group_ids=gids, n_bootstrap=30, seed=11)
    assert "macro_f1" in sig and "mcnemar_exact_match" in sig
    wil = wilcoxon_paired_test(np.linspace(0.2, 0.5, 12), np.linspace(0.5, 0.9, 12))
    assert wil["p_value"] < 0.05

    # Generate all 5 matplotlib visualizations and confirm non-empty PNG outputs
    p1 = plot_label_cooccurrence_heatmap(y_true, tmp_path / "cooc.png")
    p2 = plot_per_label_pr_curves(y_true, y_prob, tmp_path / "pr.png")
    p3 = plot_reliability_diagrams(y_true, y_prob, tmp_path / "rel.png", calibrated_probs=y_prob * 0.95)
    p4 = plot_training_curves(
        [
            {"epoch": 1, "train_loss": 0.62, "val_loss": 0.58, "val_macro_f1": 0.61},
            {"epoch": 2, "train_loss": 0.40, "val_loss": 0.41, "val_macro_f1": 0.78},
        ],
        tmp_path / "curves.png",
    )
    p5 = plot_combination_breakdown(sliced, tmp_path / "card.png")
    for fp in (p1, p2, p3, p4, p5):
        assert fp.exists() and fp.stat().st_size > 1000


def test_export_split_token_budget_report(tmp_path: Path) -> None:
    """Verify export_split_token_budget_report computes per-dataset truncation stats and writes JSON."""
    from src.utils.io import read_json, write_jsonl

    split_path = tmp_path / "mini_val.jsonl"
    long_contract = " ".join(f"clause_{i}" for i in range(650))
    write_jsonl(
        split_path,
        [
            {
                "source_record_id": "rec_cuad",
                "source_dataset": "cuad_qa",
                "has_context": True,
                "question": "What is the governing law?",
                "original_answer": "Delaware law governs this agreement.",
                "context": long_contract,
            },
            {
                "source_record_id": "rec_med",
                "source_dataset": "medquad",
                "has_context": False,
                "question": "What causes anemia?",
                "original_answer": "Low hemoglobin levels.",
                "context": None,
            },
        ],
    )
    out_json = tmp_path / "budget_report.json"
    report = export_split_token_budget_report(
        {"val": split_path},
        max_lengths=[512, 2048],
        output_json=out_json,
    )
    assert out_json.exists()
    saved = read_json(out_json)
    val_entry = saved["splits"]["val"]
    assert val_entry["record_count"] == 2
    assert val_entry["max_lengths"]["512"]["by_source_dataset"]["cuad_qa"]["context_truncated_rate"] == 1.0
    assert val_entry["max_lengths"]["2048"]["by_source_dataset"]["cuad_qa"]["context_truncated_rate"] == 0.0
    assert val_entry["max_lengths"]["512"]["by_source_dataset"]["medquad"]["context_truncated_rate"] == 0.0
    assert val_entry["max_lengths"]["512"]["sliding_window_mean_chunks"] >= 2.0
