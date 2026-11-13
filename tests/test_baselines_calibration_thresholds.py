"""End-to-end tests for TemperatureScaler, val-only thresholds, TF-IDF baselines (E1.1-E1.3), and Zero-Shot LLM (E1.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.schema import HALLUCINATION_LABELS, LabelDictionary, LicenseMetadata, MultiLabelRecord
from src.generation.backend import MockLLMClient
from src.models.baseline import (
    ClassicalBaselineSuite,
    ZeroShotLLMDetectorBaseline,
    run_classical_baseline_experiment,
)
from src.models.calibration import TemperatureScaler, probs_to_logits
from src.training.thresholds import load_thresholds, optimize_validation_thresholds, save_thresholds


def _make_semantic_dataset(
    *,
    split: str,
    n: int,
    seed: int,
    data_origin: str = "synthetic",
) -> list[MultiLabelRecord]:
    rng = np.random.default_rng(seed)
    cues = [
        ("reversed contradiction against established clinical mechanism ", "verified guideline baseline mechanism "),
        ("invented bogus citation archive-2099 journal report ", "standard published evidence review "),
        ("ignored user format instruction refused requested step ", "followed explicit question requirements carefully "),
        ("circular premise proves false conclusion contradicts step one ", "coherent deductive reasoning step by step "),
        ("extrapolates beyond provided passage context snippet ", "faithful reading strictly inside context "),
    ]
    records: list[MultiLabelRecord] = []
    for i in range(n):
        if i % 6 == 0:
            vec = [0, 0, 0, 0, 0]
        else:
            vec = [int(rng.random() < 0.45) for _ in range(4)] + [int(i % 2 == 0 and rng.random() < 0.5)]
            if sum(vec) == 0:
                vec[i % 4] = 1
        has_ctx = i % 2 == 0
        if not has_ctx:
            vec[4] = 0
        parts = [cues[k][0] if vec[k] == 1 else cues[k][1] for k in range(5)]
        ans_text = " ".join(parts) + f" example item {i} filler phrase."
        gid = f"GRP_{split}_{i:04d}"
        records.append(
            MultiLabelRecord(
                sample_id=f"{split.upper()}_{i:04d}",
                source_dataset="pubmedqa" if has_ctx else "medquad",
                source_record_id=f"SRC_{split}_{i:04d}",
                source_group_id=gid,
                domain="medical" if i % 2 == 0 else "finance",
                data_origin=data_origin,  # type: ignore[arg-type]
                has_context=has_ctx,
                question=f"Question {i} about domain facts?",
                context="Reference evidence states standard published baseline facts." if has_ctx else None,
                original_answer="Original baseline domain response with accurate clinical reasoning.",
                generated_answer=ans_text,
                generation_method="natural_llm_response" if data_origin == "natural" else "synthetic_perturbation",
                generation_strategy="natural_qa" if data_origin == "natural" else "strategy_a",
                intended_labels=None if data_origin == "natural" else LabelDictionary(**dict(zip(HALLUCINATION_LABELS, vec))),
                adjudicated_gold_labels=vec if data_origin == "natural" else None,
                label_vector=vec,
                licensing=LicenseMetadata(
                    upstream_license="MIT",
                    license_status="verified_open",
                    redistribution_permitted=True,
                    attribution="Test Fixture",
                    notes="Unit test synthetic record.",
                ),
                split=split,  # type: ignore[arg-type]
            )
        )
    return records


def test_calibration_and_threshold_optimizers_enforce_val_only_guards(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    y_val = (rng.random((40, 5)) < 0.4).astype(np.int64)
    logits_val = (y_val * 2.5 - 1.2) + rng.normal(0.0, 0.5, size=(40, 5))

    scaler = TemperatureScaler(per_label=True)
    with pytest.raises(AssertionError, match="ONLY validation"):
        scaler.fit(logits_val, y_val, split_names=["train"] * 40)

    rep = scaler.fit(logits_val, y_val, split_names=["val"] * 40)
    assert all(t > 0.0 for t in rep["temperatures"].values())
    cal_path = scaler.save(tmp_path / "calibration.json")
    reloaded_scaler = TemperatureScaler.load(cal_path)
    assert np.allclose(reloaded_scaler.temperatures, scaler.temperatures)

    # Scalar T mode
    scalar = TemperatureScaler(per_label=False)
    scalar_rep = scalar.fit(logits_val, y_val, split_names=["val"] * 40)
    assert len(set(scalar_rep["temperatures"].values())) == 1

    probs_val = reloaded_scaler.predict_proba(logits_val)
    with pytest.raises(AssertionError, match="ONLY validation"):
        optimize_validation_thresholds(probs_val, y_val, split_names=["test_synthetic"] * 40)

    thr = optimize_validation_thresholds(probs_val, y_val, split_names=["val"] * 40)
    assert thr["grid_spec"]["n_points"] == 91
    assert len(thr["sensitivity_curves"]["factual_inconsistency"]) == 91
    save_thresholds(thr, tmp_path / "thresholds.json")
    assert load_thresholds(tmp_path / "thresholds.json") == thr["threshold_vector"]
    assert probs_to_logits(probs_val).shape == logits_val.shape


@pytest.mark.parametrize("variant", ["ovr_logreg", "linear_svc", "classifier_chain"])
def test_classical_baseline_variants_end_to_end(variant: str, tmp_path: Path) -> None:
    tr = _make_semantic_dataset(split="train", n=60, seed=1)
    va = _make_semantic_dataset(split="val", n=30, seed=2)
    te = _make_semantic_dataset(split="test_synthetic", n=30, seed=3)
    nat = _make_semantic_dataset(split="test_natural", n=24, seed=4, data_origin="natural")

    run_dir = tmp_path / f"e1_{variant}"
    report = run_classical_baseline_experiment(
        train_records=tr,
        val_records=va,
        test_records=te,
        natural_records=nat,
        output_dir=run_dir,
        variant=variant,
        n_bootstrap=20,
        seed=42,
    )
    assert (run_dir / "baseline_model.joblib").exists()
    assert (run_dir / "calibration.json").exists()
    assert (run_dir / "thresholds.json").exists()
    assert (run_dir / "baseline_report.json").exists()
    assert (run_dir / "plots" / "test_synthetic_pr_curves.png").exists()
    assert (run_dir / "plots" / "test_natural_reliability.png").exists()
    assert report["test_synthetic_eval"]["metrics_calibrated_val_tuned_tau"]["overall"]["macro_f1"] > 0.70
    assert report["test_natural_eval"]["n_samples"] == 24

    loaded = ClassicalBaselineSuite.load(run_dir / "baseline_model.joblib")
    p_loaded, _, _, _ = loaded.predict_scores_and_logits(te[:5])
    assert p_loaded.shape == (5, len(HALLUCINATION_LABELS))


def test_zero_shot_llm_detector_baseline_offline() -> None:
    te = _make_semantic_dataset(split="test_synthetic", n=6, seed=9)

    def _mock_judge(sys_p: str, usr_p: str, call_n: int, temp: float) -> str:
        return json.dumps(
            {
                "factual_inconsistency": 1 if "contradiction" in usr_p else 0,
                "factual_fabrication": 1 if "invented" in usr_p else 0,
                "instruction_inconsistency": 1 if "ignored" in usr_p else 0,
                "logical_inconsistency": 1 if "circular" in usr_p else 0,
                "context_inconsistency": 1 if "extrapolates" in usr_p else 0,
            }
        )

    client = MockLLMClient(
        model_id="openai/gpt-oss-120b",
        provider="mock_groq",
        model_family="gpt",
        responder=_mock_judge,
    )
    detector = ZeroShotLLMDetectorBaseline(client=client)
    res = detector.predict(te)
    assert res["n_samples"] == 6
    assert res["json_parse_rate"] == pytest.approx(1.0)
    assert res["y_prob"].shape == (6, 5)
