"""Pure-Python helpers powering the Streamlit multi-label hallucination inspection UI."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from src.data.schema import HALLUCINATION_LABELS
from src.models.inference import CATEGORY_DESCRIPTIONS, InferenceResult
from src.utils.config import PROJECT_ROOT

EXAMPLE_PRESETS: dict[str, dict[str, str]] = {
    "Clean Grounded Clinical Summary": {
        "question": "What is the initial emergency medication for acute anaphylaxis?",
        "response": (
            "Intramuscular epinephrine injected into the mid-outer thigh is the immediate first-line "
            "treatment for acute anaphylaxis. Prompt administration helps reverse airway edema and hypotension."
        ),
        "context": (
            "Emergency clinical guidelines state that intramuscular epinephrine is the first-line "
            "pharmacologic treatment for acute anaphylaxis across all age groups."
        ),
    },
    "Multi-Label Distorted + Fabricated Response": {
        "question": "Can acetaminophen cure chronic arterial hypertension long-term?",
        "response": (
            "Yes, taking daily high-dose acetaminophen cures chronic hypertension permanently according "
            "to the Global Cardio-Trial 2027 published in New England Cardiology Letters. Therefore, "
            "you should double the dose whenever blood pressure rises above 140 mmHg, even though blood pressure needs no medication."
        ),
        "context": (
            "Acetaminophen is an analgesic and antipyretic agent without sustained antihypertensive "
            "efficacy. First-line hypertension control uses lifestyle therapy and prescribed antihypertensives."
        ),
    },
    "No-Context Financial Question": {
        "question": "How does compound interest differ from simple interest?",
        "response": (
            "Simple interest accrues only on the original principal balance, whereas compound interest "
            "is calculated on both the initial principal and previously accumulated interest periods."
        ),
        "context": "",
    },
}


def discover_available_models(search_roots: list[Path] | None = None) -> list[dict[str, str]]:
    """Scan experiments/ and results/ for saved best_checkpoint.pt or baseline_model.joblib runs."""
    roots = search_roots or [PROJECT_ROOT / "experiments", PROJECT_ROOT / "results"]
    found: list[dict[str, str]] = []
    seen_dirs: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for fname, kind in (("best_checkpoint.pt", "Transformer"), ("baseline_model.joblib", "TF-IDF Baseline")):
            for fp in sorted(root.rglob(fname)):
                run_dir = fp.parent.resolve()
                if run_dir in seen_dirs:
                    continue
                seen_dirs.add(run_dir)
                try:
                    rel = str(run_dir.relative_to(PROJECT_ROOT))
                except ValueError:
                    rel = str(run_dir)
                found.append({"label": f"[{kind}] {rel}", "path": str(run_dir), "kind": kind})
    return found


def format_prediction_rows(result: InferenceResult) -> list[dict[str, Any]]:
    """Build UI display table rows pairing raw scores, calibrated scores, thresholds, and status."""
    rows: list[dict[str, Any]] = []
    for lbl in HALLUCINATION_LABELS:
        meta = CATEGORY_DESCRIPTIONS[lbl]
        trig = int(result.triggered_labels[lbl])
        rows.append(
            {
                "Category": f"{meta['code']} — {meta['title']}",
                "Key": lbl,
                "Raw Score": round(float(result.raw_scores[lbl]), 4),
                "Calibrated Score": round(float(result.calibrated_scores[lbl]), 4),
                "Threshold": round(float(result.thresholds_used[lbl]), 4),
                "Status": "TRIGGERED" if trig == 1 else "Clean (0)",
                "Triggered": bool(trig),
            }
        )
    return rows


def format_truncation_summary(trunc_meta: dict[str, Any]) -> dict[str, Any]:
    """Summarize Field-Aware TokenBudgetAllocator telemetry and generate warning flags for UI."""
    warnings: list[str] = []
    if trunc_meta.get("response_truncated"):
        warnings.append("Candidate response tail exceeded token reserve and was truncated.")
    if trunc_meta.get("context_truncated"):
        warnings.append(
            f"Reference context was truncated ({trunc_meta.get('context_tokens_kept')}/"
            f"{trunc_meta.get('context_tokens_orig')} tokens kept; response preserved first)."
        )
    if trunc_meta.get("question_truncated"):
        warnings.append("Question exceeded token allocation and was truncated.")
    return {
        "total_tokens": int(trunc_meta.get("total_sequence_tokens", 0)),
        "max_length": int(trunc_meta.get("max_length_limit", 512)),
        "question_tokens": f"{trunc_meta.get('question_tokens_kept', 0)}/{trunc_meta.get('question_tokens_orig', 0)}",
        "response_tokens": f"{trunc_meta.get('response_tokens_kept', 0)}/{trunc_meta.get('response_tokens_orig', 0)}",
        "context_tokens": f"{trunc_meta.get('context_tokens_kept', 0)}/{trunc_meta.get('context_tokens_orig', 0)}",
        "warnings": warnings,
    }


def render_sentence_highlights_html(highlights: list[dict[str, Any]]) -> str:
    """Render safe escaped HTML pills highlighting per-sentence context lexical overlap."""
    if not highlights:
        return "<p>No sentences to inspect.</p>"
    spans: list[str] = []
    for item in highlights:
        sent_esc = html.escape(str(item["sentence"]))
        status = item.get("status", "no_context_supplied")
        ov = item.get("context_lexical_overlap")
        if status == "supported_overlap":
            bg = "#e6f4ea"
            border = "#34a853"
            badge = f"overlap={ov:.2f}" if isinstance(ov, (int, float)) else ""
        elif status == "low_context_support":
            bg = "#fce8e6"
            border = "#ea4335"
            badge = f"low overlap={ov:.2f}" if isinstance(ov, (int, float)) else ""
        else:
            bg = "#f1f3f4"
            border = "#9aa0a6"
            badge = "no context"
        spans.append(
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'padding:8px 12px;margin:6px 0;border-radius:4px;">'
            f'<span style="font-size:0.85em;color:#444;font-weight:600;">[{html.escape(badge)}]</span> '
            f'{sent_esc}</div>'
        )
    return "\n".join(spans)


def export_prediction_json_bytes(result: InferenceResult) -> bytes:
    """Serialize structured InferenceResult to UTF-8 JSON bytes for Streamlit download."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")

