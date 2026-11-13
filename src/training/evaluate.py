"""Locked single-pass Phase 4-5 checkpoint & prediction evaluator keeping synthetic and natural splits strictly unpooled."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import special

from src.data.schema import MultiLabelRecord
from src.data.token_budget import TokenBudgetAllocator
from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_multilabel_metrics,
    compute_sliced_e4_metrics,
    resolve_ground_truth_matrix,
)
from src.evaluation.visualizations import (
    plot_combination_breakdown,
    plot_label_cooccurrence_heatmap,
    plot_per_label_pr_curves,
    plot_reliability_diagrams,
)
from src.models.calibration import TemperatureScaler
from src.models.classifier import (
    MultiLabelHallucinationClassifier,
    require_torch_and_transformers,
    torch_available,
)
from src.training.thresholds import load_thresholds
from src.training.train import (
    load_tokenizer_for_backbone,
    prepare_encoded_records,
)
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_jsonl, utc_now_iso, write_json

if torch_available():
    import torch
    from torch.utils.data import DataLoader
    from src.training.train import _EncodedDataset, _collate_batch, predict_logits_dataloader
else:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


def evaluate_logits_on_split(
    *,
    records: Sequence[MultiLabelRecord],
    raw_logits: np.ndarray,
    split_tag: str,
    out_dir: str | Path,
    scaler: TemperatureScaler | None = None,
    tuned_thresholds: list[float] | None = None,
    latency_ms_per_sample: float | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate raw model logits once against a locked split without cross-split pooling.

    Exports PR curves, 10-bin reliability diagrams, co-occurrence heatmap, E4 slicing,
    and cluster-aware 95% bootstrap confidence intervals.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true = resolve_ground_truth_matrix(records)
    raw_probs = special.expit(np.asarray(raw_logits, dtype=np.float64))
    cal_probs = scaler.predict_proba(raw_logits) if scaler is not None else raw_probs
    group_ids = [r.source_group_id for r in records]
    tau = tuned_thresholds if tuned_thresholds is not None else [0.50] * y_true.shape[1]

    m_raw_050 = compute_multilabel_metrics(
        y_true, raw_probs, 0.50, latency_ms_per_sample=latency_ms_per_sample, metadata_rows=metadata_rows
    )
    m_cal_050 = compute_multilabel_metrics(
        y_true, cal_probs, 0.50, latency_ms_per_sample=latency_ms_per_sample, metadata_rows=metadata_rows
    )
    m_cal_tuned = compute_multilabel_metrics(
        y_true, cal_probs, tau, latency_ms_per_sample=latency_ms_per_sample, metadata_rows=metadata_rows
    )
    sliced = compute_sliced_e4_metrics(records, y_true, cal_probs, tau, metadata_rows=metadata_rows)
    ci_95 = bootstrap_confidence_intervals(
        y_true,
        cal_probs,
        group_ids=group_ids,
        thresholds=tau,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    plots_dir = out / "plots"
    plot_per_label_pr_curves(y_true, cal_probs, plots_dir / f"{split_tag}_pr_curves.png", title=f"{split_tag} PR Curves")
    plot_reliability_diagrams(
        y_true,
        raw_probs,
        plots_dir / f"{split_tag}_reliability.png",
        calibrated_probs=cal_probs if scaler is not None else None,
        title=f"{split_tag} Reliability Diagrams",
    )
    plot_label_cooccurrence_heatmap(y_true, plots_dir / f"{split_tag}_cooccurrence.png", title=f"{split_tag} Label Co-Occurrence")
    plot_combination_breakdown(sliced, plots_dir / f"{split_tag}_cardinality.png", title=f"{split_tag} Cardinality F1")

    return {
        "split_tag": split_tag,
        "n_samples": len(records),
        "unique_source_groups": len(set(group_ids)),
        "metrics_raw_tau_050": m_raw_050,
        "metrics_calibrated_tau_050": m_cal_050,
        "metrics_calibrated_val_tuned_tau": m_cal_tuned,
        "e4_sliced_metrics": sliced,
        "bootstrap_95ci": ci_95,
        "raw_logits_preview_shape": list(np.asarray(raw_logits).shape),
    }


def evaluate_saved_checkpoint(
    *,
    run_dir: str | Path,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    test_synthetic_records: Sequence[MultiLabelRecord] | None = None,
    test_synthetic_gold_records: Sequence[MultiLabelRecord] | None = None,
    test_natural_records: Sequence[MultiLabelRecord] | None = None,
    output_dir: str | Path | None = None,
    n_bootstrap: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load best_checkpoint.pt + calibration.json + thresholds.json and evaluate splits separately."""
    require_torch_and_transformers("evaluate_saved_checkpoint")
    rdir = Path(run_dir)
    out = Path(output_dir) if output_dir is not None else rdir
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(rdir / "best_checkpoint.pt", map_location=device, weights_only=False)
    hf_id = str(ckpt["hf_model_id"])
    max_len = int(ckpt["max_length"])
    tokenizer = load_tokenizer_for_backbone(hf_id, max_len)
    tb_cfg = model_cfg.get("token_budget") or {}
    allocator = TokenBudgetAllocator(
        tokenizer=tokenizer,
        max_length=max_len,
        response_budget=int(tb_cfg.get("response_budget", 288)),
        question_budget=int(tb_cfg.get("question_budget", 96)),
    )

    model = MultiLabelHallucinationClassifier(
        hf_model_id=hf_id,
        num_labels=int(ckpt["num_labels"]),
        pooling=str(ckpt["pooling"]),
        head_type=str(ckpt["head_type"]),
        dropout=float(ckpt["dropout"]),
        chunk_aggregation=str(ckpt.get("chunk_aggregation", "logsumexp")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    scaler = TemperatureScaler.load(rdir / "calibration.json") if (rdir / "calibration.json").exists() else None
    tuned_tau = load_thresholds(rdir / "thresholds.json") if (rdir / "thresholds.json").exists() else None

    ctx_mode = str(ckpt.get("context_mode", "budgeted_truncation"))
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    use_amp = bool(train_cfg["schedule"].get("fp16_if_cuda", True)) and device.type == "cuda"
    bs_eval = int(train_cfg["schedule"].get("eval_batch_size", 32))
    boots = int(n_bootstrap or train_cfg["evaluation"].get("n_bootstrap", 1000))

    results_by_split: dict[str, Any] = {}
    for split_tag, recs in (
        ("test_synthetic", test_synthetic_records),
        ("test_synthetic_gold", test_synthetic_gold_records),
        ("test_natural", test_natural_records),
    ):
        if not recs:
            continue
        enc, meta = prepare_encoded_records(
            recs,
            allocator=allocator,
            context_mode=ctx_mode,
            sliding_stride=int(tb_cfg.get("sliding_window_stride", 128)),
            max_chunks=int(tb_cfg.get("sliding_window_max_chunks", 16)),
        )
        y_t = resolve_ground_truth_matrix(recs)
        loader = DataLoader(
            _EncodedDataset(enc, y_t),
            batch_size=bs_eval if ctx_mode != "sliding_window" else 4,
            shuffle=False,
            collate_fn=lambda b: _collate_batch(b, pad_id),
        )
        logits, lat_ms = predict_logits_dataloader(model, loader, device, use_amp=use_amp)
        np.save(out / f"{split_tag}_logits.npy", logits)
        results_by_split[split_tag] = evaluate_logits_on_split(
            records=recs,
            raw_logits=logits,
            split_tag=split_tag,
            out_dir=out,
            scaler=scaler,
            tuned_thresholds=tuned_tau,
            latency_ms_per_sample=lat_ms,
            metadata_rows=meta,
            n_bootstrap=boots,
            seed=seed,
        )

    report = {
        "evaluated_at_utc": utc_now_iso(),
        "run_dir": str(rdir),
        "hf_model_id": hf_id,
        "backbone_key": ckpt.get("backbone_key"),
        "head_type": ckpt.get("head_type"),
        "pooling": ckpt.get("pooling"),
        "context_mode": ctx_mode,
        "max_length": max_len,
        "device": str(device),
        "splits_evaluated": sorted(results_by_split.keys()),
        "evaluations": results_by_split,
    }
    write_json(out / "eval_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked single-pass Phase 4-5 checkpoint evaluator.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory containing best_checkpoint.pt.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--test-synthetic", default=None)
    parser.add_argument("--test-synthetic-gold", default=None)
    parser.add_argument("--test-natural", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m_cfg = load_yaml_config(args.model_config)
    t_cfg = load_yaml_config(args.training_config)

    def _maybe_read(path_str: str | None) -> list[MultiLabelRecord] | None:
        if not path_str:
            return None
        return [MultiLabelRecord(**d) for d in read_jsonl(resolve_project_path(path_str))]

    rep = evaluate_saved_checkpoint(
        run_dir=resolve_project_path(args.run_dir),
        model_cfg=m_cfg,
        train_cfg=t_cfg,
        test_synthetic_records=_maybe_read(args.test_synthetic),
        test_synthetic_gold_records=_maybe_read(args.test_synthetic_gold),
        test_natural_records=_maybe_read(args.test_natural),
        output_dir=resolve_project_path(args.output_dir) if args.output_dir else None,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for tag, val in rep["evaluations"].items():
        print(tag, val["metrics_calibrated_val_tuned_tau"]["overall"])


if __name__ == "__main__":
    main()

