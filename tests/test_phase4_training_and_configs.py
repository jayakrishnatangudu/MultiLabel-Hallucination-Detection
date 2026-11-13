"""Tests for Phase 4 model/training YAML contracts, pure classifier helpers, split isolation, and locked evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.classifier import (
    aggregate_chunk_logits_numpy,
    asymmetric_focal_loss_numpy,
    compute_pos_weights_from_matrix,
    resolve_backbone_spec,
)
from src.training.evaluate import evaluate_logits_on_split
from src.training.train import enforce_split_isolation, prepare_encoded_records
from src.data.token_budget import TokenBudgetAllocator
from src.utils.config import load_yaml_config
from tests.test_baselines_calibration_thresholds import _make_semantic_dataset


def test_model_and_training_yaml_match_verified_architecture_specs() -> None:
    m_cfg = load_yaml_config("configs/model.yaml")
    t_cfg = load_yaml_config("configs/training.yaml")

    assert resolve_backbone_spec(m_cfg, "distilbert")["hf_model_id"] == "distilbert/distilbert-base-uncased"
    assert resolve_backbone_spec(m_cfg, "roberta")["hf_model_id"] == "FacebookAI/roberta-base"
    deb = resolve_backbone_spec(m_cfg, "deberta_v3")
    assert deb["hf_model_id"] == "microsoft/deberta-v3-base"
    assert deb["max_length"] == 512
    assert deb["explicit_max_length_required"] is True
    mod = resolve_backbone_spec(m_cfg, "modernbert")
    assert mod["hf_model_id"] == "answerdotai/ModernBERT-base"
    assert mod["max_position_embeddings"] == 8192

    assert t_cfg["random_seeds"] == [42, 101, 202]
    assert t_cfg["thresholding"]["min_tau"] == 0.05
    assert t_cfg["thresholding"]["max_tau"] == 0.95
    assert t_cfg["thresholding"]["step"] == 0.01


def test_classifier_numpy_helpers_and_phase4_eval_pipeline(tmp_path: Path) -> None:
    y = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    pw = compute_pos_weights_from_matrix(y, cap=5.0)
    assert pw[0] == pytest.approx(2.0)
    assert pw[1] == pytest.approx(1.0)

    chunk_logits = np.array([[1.0, -1.0, 0.0, 2.0, -2.0], [3.0, -3.0, 0.0, 0.0, -2.0]])
    assert aggregate_chunk_logits_numpy(chunk_logits, mode="max")[0] == pytest.approx(3.0)
    assert aggregate_chunk_logits_numpy(chunk_logits, mode="mean")[0] == pytest.approx(2.0)
    lse = aggregate_chunk_logits_numpy(chunk_logits, mode="logsumexp")
    assert lse.shape == (5,)
    assert asymmetric_focal_loss_numpy(chunk_logits, y[:2]) > 0.0

    tr = _make_semantic_dataset(split="train", n=12, seed=21)
    va = _make_semantic_dataset(split="val", n=8, seed=22)
    te = _make_semantic_dataset(split="test_synthetic", n=8, seed=23)

    iso = enforce_split_isolation(tr, va, {"test_synthetic": te})
    assert iso["train_count"] == 12 and iso["test_synthetic_count"] == 8
    with pytest.raises(AssertionError, match="Group leakage"):
        enforce_split_isolation(tr, va, {"test_synthetic": [tr[0].model_copy(update={"split": "test_synthetic"})]})

    alloc = TokenBudgetAllocator(max_length=128)
    enc_std, meta_std = prepare_encoded_records(te, allocator=alloc, context_mode="budgeted_truncation")
    assert len(enc_std) == 8 and "input_ids" in enc_std[0]
    enc_sw, _ = prepare_encoded_records(te, allocator=alloc, context_mode="sliding_window", sliding_stride=8)
    assert len(enc_sw) == 8 and "chunks_input_ids" in enc_sw[0]

    # Locked split evaluation from raw logits without needing PyTorch
    rng = np.random.default_rng(77)
    raw_logits = rng.normal(0.0, 1.5, size=(8, 5))
    ev = evaluate_logits_on_split(
        records=te,
        raw_logits=raw_logits,
        split_tag="test_synthetic",
        out_dir=tmp_path / "phase4_eval",
        tuned_thresholds=[0.5] * 5,
        metadata_rows=meta_std,
        n_bootstrap=20,
    )
    assert ev["split_tag"] == "test_synthetic"
    assert (tmp_path / "phase4_eval" / "plots" / "test_synthetic_pr_curves.png").exists()

