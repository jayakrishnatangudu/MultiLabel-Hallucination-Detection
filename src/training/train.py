"""Phase 4-5 PyTorch transformer fine-tuning, E1.4 5-binary-encoder runner, calibration, and thresholding."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np

from src.data.schema import HALLUCINATION_LABELS, MultiLabelRecord
from src.data.token_budget import TokenBudgetAllocator, summarize_truncation_rates
from src.evaluation.metrics import compute_multilabel_metrics, resolve_ground_truth_matrix
from src.evaluation.visualizations import plot_training_curves
from src.models.calibration import TemperatureScaler
from src.models.classifier import (
    MultiLabelHallucinationClassifier,
    build_loss_fn,
    require_torch_and_transformers,
    resolve_backbone_spec,
    torch_available,
    transformers_available,
)
from src.training.thresholds import optimize_validation_thresholds, save_thresholds
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_jsonl, sha256_json, utc_now_iso, write_json

if torch_available():
    import torch
    from torch.utils.data import DataLoader, Dataset
else:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

if transformers_available():
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
else:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]
    get_cosine_schedule_with_warmup = None  # type: ignore[assignment]


def seed_everything(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    if torch_available():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def enforce_split_isolation(
    train_records: Sequence[MultiLabelRecord],
    val_records: Sequence[MultiLabelRecord],
    test_batches: dict[str, Sequence[MultiLabelRecord]] | None = None,
) -> dict[str, Any]:
    """Assert zero source_group_id overlap and valid split labels across train, val, and tests."""
    tr_bad = [r.sample_id for r in train_records if r.split != "train"]
    va_bad = [r.sample_id for r in val_records if r.split != "val"]
    if tr_bad:
        raise AssertionError(f"Non-train records in train split: {tr_bad[:5]}")
    if va_bad:
        raise AssertionError(f"Non-val records in val split: {va_bad[:5]}")

    tr_groups = {r.source_group_id for r in train_records}
    va_groups = {r.source_group_id for r in val_records}
    tr_va_overlap = tr_groups & va_groups
    if tr_va_overlap:
        raise AssertionError(f"Group leakage between train and val: {sorted(tr_va_overlap)[:5]}")

    summary: dict[str, Any] = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_unique_groups": len(tr_groups),
        "val_unique_groups": len(va_groups),
    }
    for name, recs in (test_batches or {}).items():
        if not recs:
            continue
        t_groups = {r.source_group_id for r in recs}
        if t_groups & tr_groups:
            raise AssertionError(f"Group leakage between train and {name}: {sorted(t_groups & tr_groups)[:5]}")
        if t_groups & va_groups:
            raise AssertionError(f"Group leakage between val and {name}: {sorted(t_groups & va_groups)[:5]}")
        summary[f"{name}_count"] = len(recs)
        summary[f"{name}_unique_groups"] = len(t_groups)
    return summary


def load_tokenizer_for_backbone(hf_model_id: str, max_length: int) -> Any:
    """Load HF AutoTokenizer while explicitly clamping model_max_length (critical for DeBERTa-v3)."""
    require_torch_and_transformers("load_tokenizer_for_backbone")
    tok = AutoTokenizer.from_pretrained(hf_model_id)
    tok.model_max_length = int(max_length)
    return tok


def prepare_encoded_records(
    records: Sequence[MultiLabelRecord],
    *,
    allocator: TokenBudgetAllocator,
    context_mode: str = "budgeted_truncation",
    sliding_stride: int = 128,
    max_chunks: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tokenize records using shared field-aware TokenBudgetAllocator and capture metadata."""
    encoded: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for rec in records:
        if context_mode == "sliding_window":
            chunks = allocator.build_sliding_window_chunks(
                question=rec.question,
                response=rec.generated_answer,
                context=rec.context,
                stride=sliding_stride,
                max_chunks=max_chunks,
            )
            encoded.append(
                {
                    "sample_id": rec.sample_id,
                    "chunks_input_ids": [c.input_ids for c in chunks],
                    "chunks_attention_mask": [c.attention_mask for c in chunks],
                }
            )
            m = chunks[0].metadata.to_dict()
            m["num_chunks"] = len(chunks)
            meta_rows.append(m)
        else:
            item = allocator.encode_sample(
                question=rec.question,
                response=rec.generated_answer,
                context=rec.context,
            )
            encoded.append(
                {
                    "sample_id": rec.sample_id,
                    "input_ids": item.input_ids,
                    "attention_mask": item.attention_mask,
                }
            )
            meta_rows.append(item.metadata.to_dict())
    return encoded, meta_rows


if torch_available():

    class _EncodedDataset(Dataset):  # type: ignore[misc]
        def __init__(self, encoded_items: list[dict[str, Any]], y_matrix: np.ndarray) -> None:
            self.items = encoded_items
            self.y = torch.tensor(y_matrix, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {**self.items[idx], "labels": self.y[idx]}

    def _collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
        if "chunks_input_ids" in batch[0]:
            # Sliding-window evaluation keeps per-item chunks explicit
            return {
                "is_sliding": True,
                "items": batch,
                "labels": torch.stack([b["labels"] for b in batch], dim=0),
            }
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for b in batch:
            pad_n = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad_n)
            attention_masks.append(b["attention_mask"] + [0] * pad_n)
        return {
            "is_sliding": False,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.stack([b["labels"] for b in batch], dim=0),
        }

    def predict_logits_dataloader(
        model: MultiLabelHallucinationClassifier,
        loader: DataLoader,
        device: Any,
        *,
        use_amp: bool = False,
        target_col_idx: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run batched inference returning raw logits `(N, C)` and mean latency ms/sample."""
        model.eval()
        logits_chunks: list[np.ndarray] = []
        t0 = time.perf_counter()
        total_items = 0
        with torch.no_grad():
            for batch in loader:
                if batch["is_sliding"]:
                    b_logits: list[torch.Tensor] = []
                    for item in batch["items"]:
                        c_ids = [torch.tensor([ids], dtype=torch.long, device=device) for ids in item["chunks_input_ids"]]
                        c_msk = [torch.tensor([m], dtype=torch.long, device=device) for m in item["chunks_attention_mask"]]
                        b_logits.append(model.forward_sliding_chunks(c_ids, c_msk))
                    out = torch.stack(b_logits, dim=0)
                    total_items += len(batch["items"])
                else:
                    in_ids = batch["input_ids"].to(device)
                    att = batch["attention_mask"].to(device)
                    total_items += int(in_ids.shape[0])
                    with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
                        out = model(in_ids, att)
                if target_col_idx is not None and out.shape[1] == 1:
                    logits_chunks.append(out.detach().cpu().numpy())
                else:
                    logits_chunks.append(out.detach().cpu().numpy())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stacked = np.vstack(logits_chunks) if logits_chunks else np.zeros((0, model.num_labels), dtype=np.float64)
        return stacked, elapsed_ms / max(1, total_items)


def train_transformer_run(
    *,
    train_records: Sequence[MultiLabelRecord],
    val_records: Sequence[MultiLabelRecord],
    backbone_key: str,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    output_dir: str | Path,
    seed: int = 42,
    head_type: str | None = None,
    pooling: str | None = None,
    context_mode: str | None = None,
    loss_type: str | None = None,
    max_length_override: int | None = None,
    epochs_override: int | None = None,
    target_label_idx: int | None = None,
) -> dict[str, Any]:
    """
    Fine-tune a multi-label (or single-label for E1.4) transformer classifier on Colab GPU/CPU,
    select best checkpoint via val Macro-F1, and fit val-only temperature scaling + thresholds.
    """
    require_torch_and_transformers("train_transformer_run")
    seed_everything(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    enforce_split_isolation(train_records, val_records)
    bb_spec = resolve_backbone_spec(model_cfg, backbone_key)
    hf_id = str(bb_spec["hf_model_id"])
    max_len = int(max_length_override or bb_spec["max_length"])

    arch_cfg = model_cfg.get("architecture") or {}
    tb_cfg = model_cfg.get("token_budget") or {}
    use_head = head_type or str(arch_cfg.get("head_type", "independent_sigmoid"))
    use_pool = pooling or str(arch_cfg.get("pooling", "masked_mean"))
    use_ctx_mode = context_mode or str(arch_cfg.get("context_mode", "budgeted_truncation"))
    use_loss_cfg = dict(train_cfg.get("loss") or {})
    if loss_type:
        use_loss_cfg["type"] = loss_type

    tokenizer = load_tokenizer_for_backbone(hf_id, max_len)
    allocator = TokenBudgetAllocator(
        tokenizer=tokenizer,
        max_length=max_len,
        response_budget=int(tb_cfg.get("response_budget", 288)),
        question_budget=int(tb_cfg.get("question_budget", 96)),
        min_question_budget=int(tb_cfg.get("min_question_budget", 32)),
        special_token_overhead=int(tb_cfg.get("special_token_overhead", 4)),
    )

    tr_enc, tr_meta = prepare_encoded_records(
        train_records,
        allocator=allocator,
        context_mode=use_ctx_mode,
        sliding_stride=int(tb_cfg.get("sliding_window_stride", 128)),
        max_chunks=int(tb_cfg.get("sliding_window_max_chunks", 16)),
    )
    va_enc, va_meta = prepare_encoded_records(
        val_records,
        allocator=allocator,
        context_mode=use_ctx_mode,
        sliding_stride=int(tb_cfg.get("sliding_window_stride", 128)),
        max_chunks=int(tb_cfg.get("sliding_window_max_chunks", 16)),
    )

    y_tr_full = resolve_ground_truth_matrix(train_records)
    y_va_full = resolve_ground_truth_matrix(val_records)
    y_tr = y_tr_full[:, target_label_idx : target_label_idx + 1] if target_label_idx is not None else y_tr_full
    y_va = y_va_full[:, target_label_idx : target_label_idx + 1] if target_label_idx is not None else y_va_full
    num_labels = 1 if target_label_idx is not None else len(HALLUCINATION_LABELS)

    sched_cfg = train_cfg["schedule"]
    opt_cfg = train_cfg["optimizer"]
    epochs = int(epochs_override or sched_cfg.get("epochs", 6))
    bs_tr = int(sched_cfg.get("train_batch_size", 16))
    bs_va = int(sched_cfg.get("eval_batch_size", 32))
    grad_acc = max(1, int(sched_cfg.get("gradient_accumulation_steps", 2)))
    patience = int(sched_cfg.get("early_stopping_patience", 2))
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    tr_loader = DataLoader(
        _EncodedDataset(tr_enc, y_tr),
        batch_size=bs_tr if use_ctx_mode != "sliding_window" else 1,
        shuffle=True,
        collate_fn=lambda b: _collate_batch(b, pad_id),
    )
    va_loader = DataLoader(
        _EncodedDataset(va_enc, y_va),
        batch_size=bs_va if use_ctx_mode != "sliding_window" else 4,
        shuffle=False,
        collate_fn=lambda b: _collate_batch(b, pad_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(sched_cfg.get("fp16_if_cuda", True)) and device.type == "cuda"
    model = MultiLabelHallucinationClassifier(
        hf_model_id=hf_id,
        num_labels=num_labels,
        pooling=use_pool,
        head_type="independent_sigmoid" if target_label_idx is not None else use_head,
        dropout=float(arch_cfg.get("dropout", 0.15)),
        chunk_aggregation=str(arch_cfg.get("chunk_aggregation", "logsumexp")),
    ).to(device)

    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": float(opt_cfg.get("backbone_learning_rate", 2e-5))},
        {"params": list(model.head.parameters()), "lr": float(opt_cfg.get("head_learning_rate", 1e-4))},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(opt_cfg.get("weight_decay", 0.01)))
    total_steps = max(1, (len(tr_loader) // grad_acc) * epochs)
    warmup_steps = int(total_steps * float(opt_cfg.get("warmup_ratio", 0.10)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = build_loss_fn(use_loss_cfg, y_train=y_tr, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    history: list[dict[str, Any]] = []
    ckpt_path = out / "best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(tr_loader, start=1):
            targets = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if batch["is_sliding"]:
                    b_logits = []
                    for item in batch["items"]:
                        c_ids = [torch.tensor([ids], dtype=torch.long, device=device) for ids in item["chunks_input_ids"]]
                        c_msk = [torch.tensor([m], dtype=torch.long, device=device) for m in item["chunks_attention_mask"]]
                        b_logits.append(model.forward_sliding_chunks(c_ids, c_msk))
                    logits = torch.stack(b_logits, dim=0)
                else:
                    logits = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        targets=targets,
                    )
                loss = criterion(logits, targets) / grad_acc

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * grad_acc
            n_steps += 1
            if step_idx % grad_acc == 0 or step_idx == len(tr_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        va_logits, va_lat = predict_logits_dataloader(model, va_loader, device, use_amp=use_amp)
        with torch.no_grad():
            t_logits = torch.tensor(va_logits, dtype=torch.float32)
            t_targets = torch.tensor(y_va, dtype=torch.float32)
            val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(t_logits, t_targets).item())
        va_probs = 1.0 / (1.0 + np.exp(-np.clip(va_logits, -30.0, 30.0)))

        if num_labels == 1:
            from sklearn.metrics import f1_score as sk_f1

            val_macro = float(sk_f1(y_va.ravel(), (va_probs.ravel() >= 0.50).astype(np.int64), zero_division=0))
        else:
            val_m = compute_multilabel_metrics(y_va, va_probs, 0.50, latency_ms_per_sample=va_lat)
            val_macro = float(val_m["overall"]["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running_loss / max(1, n_steps), 6),
                "val_loss": round(val_loss, 6),
                "val_macro_f1": round(val_macro, 6),
                "val_latency_ms": round(va_lat, 3),
            }
        )
        if val_macro > best_score:
            best_score = val_macro
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone_key": backbone_key,
                    "hf_model_id": hf_id,
                    "max_length": max_len,
                    "pooling": use_pool,
                    "head_type": "independent_sigmoid" if target_label_idx is not None else use_head,
                    "dropout": float(arch_cfg.get("dropout", 0.15)),
                    "chunk_aggregation": str(arch_cfg.get("chunk_aggregation", "logsumexp")),
                    "context_mode": use_ctx_mode,
                    "num_labels": num_labels,
                    "target_label_idx": target_label_idx,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1_050": best_score,
                    "seed": seed,
                },
                ckpt_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    plot_training_curves(history, out / "training_curves.png")

    # Reload best checkpoint cleanly for val calibration + threshold fitting
    best_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    best_val_logits, best_val_lat = predict_logits_dataloader(model, va_loader, device, use_amp=use_amp)

    val_cal_report: dict[str, Any] = {}
    val_thr_summary: dict[str, Any] = {}
    if num_labels == len(HALLUCINATION_LABELS):
        val_splits = [r.split for r in val_records]
        cal_cfg = train_cfg.get("calibration") or {}
        if cal_cfg.get("enabled", True):
            temp_scaler = TemperatureScaler(per_label=bool(cal_cfg.get("per_label", True)))
            val_cal_report = temp_scaler.fit(best_val_logits, y_va, split_names=val_splits)
            temp_scaler.save(out / "calibration.json")
            cal_probs = temp_scaler.predict_proba(best_val_logits)
        else:
            cal_probs = 1.0 / (1.0 + np.exp(-np.clip(best_val_logits, -30.0, 30.0)))

        thr_cfg = train_cfg.get("thresholding") or {}
        thr_res = optimize_validation_thresholds(
            cal_probs,
            y_va,
            split_names=val_splits,
            min_tau=float(thr_cfg.get("min_tau", 0.05)),
            max_tau=float(thr_cfg.get("max_tau", 0.95)),
            step=float(thr_cfg.get("step", 0.01)),
            baseline_tau=float(thr_cfg.get("baseline_tau", 0.50)),
        )
        save_thresholds(thr_res, out / "thresholds.json")
        val_thr_summary = {
            "per_label_thresholds": thr_res["per_label_thresholds"],
            "val_macro_f1_fixed_050": thr_res["val_metrics_fixed_050"]["overall"]["macro_f1"],
            "val_macro_f1_tuned": thr_res["val_metrics_tuned"]["overall"]["macro_f1"],
        }

    metadata = {
        "completed_at_utc": utc_now_iso(),
        "backbone_key": backbone_key,
        "hf_model_id": hf_id,
        "max_length": max_len,
        "pooling": use_pool,
        "head_type": "independent_sigmoid" if target_label_idx is not None else use_head,
        "context_mode": use_ctx_mode,
        "loss_type": use_loss_cfg.get("type", "bce"),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1_050": round(best_score, 6),
        "val_latency_ms_per_sample": round(best_val_lat, 3),
        "train_truncation_summary": summarize_truncation_rates(tr_meta),
        "val_truncation_summary": summarize_truncation_rates(va_meta),
        "val_calibration_report": val_cal_report,
        "val_threshold_tuning_summary": val_thr_summary,
        "history": history,
        "config_sha256": sha256_json({"model": model_cfg, "training": train_cfg}),
    }
    write_json(out / "train_summary.json", metadata)
    return metadata


def train_independent_binary_encoders(
    *,
    train_records: Sequence[MultiLabelRecord],
    val_records: Sequence[MultiLabelRecord],
    backbone_key: str,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    output_dir: str | Path,
    seed: int = 42,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """
    Suite E1.4: Train five independent single-label encoders (one per category FI..CI)
    and align their validation logits for direct multi-task sharing comparison vs E2.3.
    """
    require_torch_and_transformers("train_independent_binary_encoders")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_label_runs: dict[str, Any] = {}
    val_logits_cols: list[np.ndarray] = []

    bb_spec = resolve_backbone_spec(model_cfg, backbone_key)
    hf_id = str(bb_spec["hf_model_id"])
    max_len = int(bb_spec["max_length"])
    tokenizer = load_tokenizer_for_backbone(hf_id, max_len)
    tb_cfg = model_cfg.get("token_budget") or {}
    allocator = TokenBudgetAllocator(
        tokenizer=tokenizer,
        max_length=max_len,
        response_budget=int(tb_cfg.get("response_budget", 288)),
        question_budget=int(tb_cfg.get("question_budget", 96)),
    )
    va_enc, _ = prepare_encoded_records(val_records, allocator=allocator)
    y_va = resolve_ground_truth_matrix(val_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    va_loader = DataLoader(
        _EncodedDataset(va_enc, y_va),
        batch_size=int(train_cfg["schedule"].get("eval_batch_size", 32)),
        shuffle=False,
        collate_fn=lambda b: _collate_batch(b, pad_id),
    )

    for k, lbl in enumerate(HALLUCINATION_LABELS):
        sub_dir = out / f"head_{lbl}"
        run_sum = train_transformer_run(
            train_records=train_records,
            val_records=val_records,
            backbone_key=backbone_key,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            output_dir=sub_dir,
            seed=seed,
            epochs_override=epochs_override,
            target_label_idx=k,
        )
        per_label_runs[lbl] = run_sum
        state = torch.load(sub_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
        sub_model = MultiLabelHallucinationClassifier(
            hf_model_id=hf_id,
            num_labels=1,
            pooling=state["pooling"],
            head_type="independent_sigmoid",
            dropout=state["dropout"],
        ).to(device)
        sub_model.load_state_dict(state["model_state_dict"])
        col_logits, _ = predict_logits_dataloader(sub_model, va_loader, device)
        val_logits_cols.append(col_logits[:, 0])

    combined_val_logits = np.column_stack(val_logits_cols)
    val_splits = [r.split for r in val_records]
    scaler = TemperatureScaler(per_label=True)
    cal_report = scaler.fit(combined_val_logits, y_va, split_names=val_splits)
    scaler.save(out / "calibration.json")
    cal_probs = scaler.predict_proba(combined_val_logits)
    thr_res = optimize_validation_thresholds(cal_probs, y_va, split_names=val_splits)
    save_thresholds(thr_res, out / "thresholds.json")

    summary = {
        "suite": "E1.4_five_independent_binary_encoders",
        "backbone_key": backbone_key,
        "seed": seed,
        "per_label_runs": per_label_runs,
        "val_calibration_report": cal_report,
        "val_threshold_tuning_summary": {
            "per_label_thresholds": thr_res["per_label_thresholds"],
            "val_macro_f1_fixed_050": thr_res["val_metrics_fixed_050"]["overall"]["macro_f1"],
            "val_macro_f1_tuned": thr_res["val_metrics_tuned"]["overall"]["macro_f1"],
        },
    }
    write_json(out / "e1_4_train_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 transformer fine-tuning runner (CUDA Colab / CPU).")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--backbone", default="deberta_v3", choices=["distilbert", "roberta", "deberta_v3", "modernbert"])
    parser.add_argument("--mode", default="multilabel", choices=["multilabel", "e1_4_independent_binary"])
    parser.add_argument("--head-type", default=None, choices=["independent_sigmoid", "classifier_chain", "label_interaction"])
    parser.add_argument("--pooling", default=None, choices=["masked_mean", "cls"])
    parser.add_argument("--context-mode", default=None, choices=["budgeted_truncation", "sliding_window", "long_context"])
    parser.add_argument("--loss-type", default=None, choices=["bce", "weighted_bce", "asymmetric_focal"])
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    model_cfg = load_yaml_config(args.model_config)
    train_cfg = load_yaml_config(args.training_config)
    tr_path = resolve_project_path(args.train_file or train_cfg["paths"]["train_file"])
    va_path = resolve_project_path(args.val_file or train_cfg["paths"]["val_file"])
    tr_recs = [MultiLabelRecord(**d) for d in read_jsonl(tr_path)]
    va_recs = [MultiLabelRecord(**d) for d in read_jsonl(va_path)]
    seeds = args.seeds or [int(s) for s in train_cfg.get("random_seeds", [42])]
    out_root = resolve_project_path(args.output_root or f"experiments/{args.backbone}_{args.mode}")

    for s in seeds:
        run_dir = out_root / f"seed_{s}"
        if args.mode == "e1_4_independent_binary":
            train_independent_binary_encoders(
                train_records=tr_recs,
                val_records=va_recs,
                backbone_key=args.backbone,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                output_dir=run_dir,
                seed=s,
                epochs_override=args.epochs,
            )
        else:
            train_transformer_run(
                train_records=tr_recs,
                val_records=va_recs,
                backbone_key=args.backbone,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                output_dir=run_dir,
                seed=s,
                head_type=args.head_type,
                pooling=args.pooling,
                context_mode=args.context_mode,
                loss_type=args.loss_type,
                max_length_override=args.max_length,
                epochs_override=args.epochs,
            )


if __name__ == "__main__":
    main()

