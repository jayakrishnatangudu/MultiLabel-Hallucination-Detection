"""CPU/GPU single-sample inference engine with calibration, thresholds, truncation telemetry, and sentence overlap highlights."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from scipy import special

from src.data.schema import HALLUCINATION_LABELS
from src.data.token_budget import TokenBudgetAllocator
from src.models.baseline import ClassicalBaselineSuite
from src.models.calibration import TemperatureScaler
from src.models.classifier import (
    MultiLabelHallucinationClassifier,
    require_torch_and_transformers,
    torch_available,
)
from src.training.thresholds import load_thresholds
from src.utils.io import utc_now_iso

if torch_available():
    import torch
    from transformers import AutoTokenizer
else:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


UNVERIFIED_PREDICTION_DISCLAIMER = (
    "UNVERIFIED AUTOMATED PREDICTION: Scores and predicted hallucination labels are "
    "model outputs only, not adjudicated ground truth."
)

CATEGORY_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "factual_inconsistency": {
        "code": "Cat I (FI)",
        "title": "Factual Inconsistency",
        "description": "Distorts or contradicts established facts within the supplied answer/domain.",
    },
    "factual_fabrication": {
        "code": "Cat II (FF)",
        "title": "Factual Fabrication",
        "description": "Invents non-existent studies, statutes, dates, numbers, entities, or citations.",
    },
    "instruction_inconsistency": {
        "code": "Cat III (II)",
        "title": "Instruction Inconsistency",
        "description": "Fails to answer the user's actual question scope, constraints, or sub-parts.",
    },
    "logical_inconsistency": {
        "code": "Cat IV (LI)",
        "title": "Logical Inconsistency",
        "description": "Exhibits internal self-contradiction, invalid deduction, or numerical mismatch.",
    },
    "context_inconsistency": {
        "code": "Cat V (CI)",
        "title": "Context Inconsistency",
        "description": "Contradicts or extrapolates beyond the supplied reference context passage.",
    },
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def compute_sentence_alignment_highlights(
    response_text: str,
    context_text: str | None,
    *,
    overlap_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Lightweight lexical token-overlap heuristic marking how strongly each response sentence aligns with context.

    Intended strictly as an inspectable UI visual aid (not causal model attribution).
    """
    cleaned_resp = (response_text or "").strip()
    if not cleaned_resp:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned_resp) if s.strip()]
    ctx_clean = (context_text or "").strip()
    if not ctx_clean:
        return [
            {
                "sentence_index": idx,
                "sentence": sent,
                "context_lexical_overlap": None,
                "status": "no_context_supplied",
            }
            for idx, sent in enumerate(sentences)
        ]

    ctx_tokens = set(_WORD_RE.findall(ctx_clean.lower()))
    out: list[dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        toks = set(_WORD_RE.findall(sent.lower()))
        overlap = len(toks & ctx_tokens) / max(1, len(toks)) if toks else 0.0
        out.append(
            {
                "sentence_index": idx,
                "sentence": sent,
                "context_lexical_overlap": round(float(overlap), 4),
                "status": "supported_overlap" if overlap >= overlap_threshold else "low_context_support",
            }
        )
    return out


@dataclass
class InferenceResult:
    """Structured prediction payload returned by HallucinationPredictor.predict_single."""

    predicted_at_utc: str
    disclaimer: str
    model_info: dict[str, Any]
    raw_scores: dict[str, float]
    calibrated_scores: dict[str, float]
    thresholds_used: dict[str, float]
    triggered_labels: dict[str, int]
    active_categories: list[str]
    any_hallucination_binary_score: float
    truncation_metadata: dict[str, Any]
    sentence_highlights: list[dict[str, Any]]
    latency_ms: float
    calibration_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HallucinationPredictor:
    """
    Unified CPU/GPU predictor supporting both fine-tuned transformer checkpoints (`best_checkpoint.pt`)
    and classical TF-IDF joblib runs (`baseline_model.joblib`).
    """

    def __init__(self, run_dir: str | Path, *, device: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Model run directory does not exist: {self.run_dir}")

        cal_path = self.run_dir / "calibration.json"
        thr_path = self.run_dir / "thresholds.json"
        self.scaler: TemperatureScaler | None = TemperatureScaler.load(cal_path) if cal_path.exists() else None
        self.tuned_thresholds: list[float] = (
            load_thresholds(thr_path) if thr_path.exists() else [0.50] * len(HALLUCINATION_LABELS)
        )

        ckpt_path = self.run_dir / "best_checkpoint.pt"
        joblib_path = self.run_dir / "baseline_model.joblib"
        if ckpt_path.exists():
            self.backend_kind = "transformer"
            self._init_transformer(ckpt_path, device=device)
        elif joblib_path.exists():
            self.backend_kind = "classical_tfidf"
            self.baseline_model = ClassicalBaselineSuite.load(joblib_path)
            self.allocator = TokenBudgetAllocator(max_length=self.baseline_model.max_length)
            self.model_info = {
                "backend_kind": "classical_tfidf",
                "variant": self.baseline_model.variant,
                "run_dir": str(self.run_dir),
                "max_length": self.baseline_model.max_length,
            }
        else:
            raise FileNotFoundError(
                f"Neither 'best_checkpoint.pt' nor 'baseline_model.joblib' found in {self.run_dir}"
            )

    def _init_transformer(self, ckpt_path: Path, *, device: str | None) -> None:
        require_torch_and_transformers("HallucinationPredictor[transformer]")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        hf_id = str(ckpt["hf_model_id"])
        max_len = int(ckpt["max_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.tokenizer.model_max_length = max_len
        self.allocator = TokenBudgetAllocator(tokenizer=self.tokenizer, max_length=max_len)
        self.model = MultiLabelHallucinationClassifier(
            hf_model_id=hf_id,
            num_labels=int(ckpt["num_labels"]),
            pooling=str(ckpt["pooling"]),
            head_type=str(ckpt["head_type"]),
            dropout=float(ckpt["dropout"]),
            chunk_aggregation=str(ckpt.get("chunk_aggregation", "logsumexp")),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.context_mode = str(ckpt.get("context_mode", "budgeted_truncation"))
        self.model_info = {
            "backend_kind": "transformer",
            "hf_model_id": hf_id,
            "backbone_key": ckpt.get("backbone_key"),
            "head_type": ckpt.get("head_type"),
            "pooling": ckpt.get("pooling"),
            "context_mode": self.context_mode,
            "max_length": max_len,
            "device": str(self.device),
            "run_dir": str(self.run_dir),
        }

    def predict_single(
        self,
        *,
        response_text: str,
        question_text: str = "",
        context_text: str | None = None,
        use_calibration: bool = True,
        threshold_mode: str = "val_tuned",
    ) -> InferenceResult:
        """Score a single `(question, response, context)` tuple and return full structured telemetry."""
        resp_clean = (response_text or "").strip()
        if not resp_clean:
            raise ValueError("Candidate response text cannot be empty.")
        q_clean = (question_text or "").strip() or "Evaluate the candidate response for hallucinations."
        c_clean = (context_text or "").strip() or None

        t0 = time.perf_counter()
        if self.backend_kind == "classical_tfidf":
            synth_row = {
                "question": q_clean,
                "generated_answer": resp_clean,
                "context": c_clean,
            }
            _, logits_arr, meta_rows, _ = self.baseline_model.predict_scores_and_logits([synth_row])
            raw_logits = logits_arr[0]
            trunc_meta = meta_rows[0]
        else:
            if getattr(self, "context_mode", "budgeted_truncation") == "sliding_window":
                chunks = self.allocator.build_sliding_window_chunks(
                    question=q_clean, response=resp_clean, context=c_clean
                )
                with torch.no_grad():
                    c_ids = [torch.tensor([c.input_ids], dtype=torch.long, device=self.device) for c in chunks]
                    c_msk = [torch.tensor([c.attention_mask], dtype=torch.long, device=self.device) for c in chunks]
                    raw_logits = self.model.forward_sliding_chunks(c_ids, c_msk).detach().cpu().numpy()
                trunc_meta = chunks[0].metadata.to_dict()
                trunc_meta["num_chunks"] = len(chunks)
            else:
                enc = self.allocator.encode_sample(question=q_clean, response=resp_clean, context=c_clean)
                with torch.no_grad():
                    in_ids = torch.tensor([enc.input_ids], dtype=torch.long, device=self.device)
                    att = torch.tensor([enc.attention_mask], dtype=torch.long, device=self.device)
                    raw_logits = self.model(in_ids, att).squeeze(0).detach().cpu().numpy()
                trunc_meta = enc.metadata.to_dict()

        raw_probs = special.expit(raw_logits)
        cal_applied = bool(use_calibration and self.scaler is not None)
        if cal_applied and self.scaler is not None:
            cal_probs = self.scaler.predict_proba(raw_logits[None, :])[0]
        else:
            cal_probs = raw_probs

        if threshold_mode == "val_tuned":
            thr_vec = self.tuned_thresholds
        elif threshold_mode == "fixed_050":
            thr_vec = [0.50] * len(HALLUCINATION_LABELS)
        else:
            raise ValueError(f"Unsupported threshold_mode={threshold_mode!r}")

        raw_dict = {lbl: round(float(raw_probs[i]), 6) for i, lbl in enumerate(HALLUCINATION_LABELS)}
        cal_dict = {lbl: round(float(cal_probs[i]), 6) for i, lbl in enumerate(HALLUCINATION_LABELS)}
        thr_dict = {lbl: round(float(thr_vec[i]), 4) for i, lbl in enumerate(HALLUCINATION_LABELS)}
        trig_dict = {
            lbl: int(cal_probs[i] >= thr_vec[i])
            for i, lbl in enumerate(HALLUCINATION_LABELS)
        }
        active = [lbl for lbl in HALLUCINATION_LABELS if trig_dict[lbl] == 1]
        highlights = compute_sentence_alignment_highlights(resp_clean, c_clean)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceResult(
            predicted_at_utc=utc_now_iso(),
            disclaimer=UNVERIFIED_PREDICTION_DISCLAIMER,
            model_info=self.model_info,
            raw_scores=raw_dict,
            calibrated_scores=cal_dict,
            thresholds_used=thr_dict,
            triggered_labels=trig_dict,
            active_categories=active,
            any_hallucination_binary_score=round(float(np.max(cal_probs)), 6),
            truncation_metadata=trunc_meta,
            sentence_highlights=highlights,
            latency_ms=round(elapsed_ms, 2),
            calibration_applied=cal_applied,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-sample hallucination inference CLI (CPU or GPU).")
    parser.add_argument("--run-dir", required=True, help="Directory with baseline_model.joblib or best_checkpoint.pt.")
    parser.add_argument("--response", required=True, help="Candidate LLM response to inspect.")
    parser.add_argument("--question", default="", help="Optional question.")
    parser.add_argument("--context", default=None, help="Optional reference context passage.")
    parser.add_argument("--threshold-mode", default="val_tuned", choices=["val_tuned", "fixed_050"])
    args = parser.parse_args()

    predictor = HallucinationPredictor(args.run_dir)
    res = predictor.predict_single(
        response_text=args.response,
        question_text=args.question,
        context_text=args.context,
        threshold_mode=args.threshold_mode,
    )
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

