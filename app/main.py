"""Phase 6 Streamlit Research Interface for Multi-Label Hallucination Inspection."""

from __future__ import annotations

from pathlib import Path
import sys

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]

# Ensure project root is importable when invoked via `streamlit run app/main.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.helpers import (
    EXAMPLE_PRESETS,
    discover_available_models,
    export_prediction_json_bytes,
    format_prediction_rows,
    format_truncation_summary,
    render_sentence_highlights_html,
)
from src.models.inference import (
    CATEGORY_DESCRIPTIONS,
    HallucinationPredictor,
    UNVERIFIED_PREDICTION_DISCLAIMER,
)


def run_app() -> None:
    """Render the interactive multi-label hallucination inspection Streamlit application."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed in the active environment. Install `streamlit` (`pip install streamlit`) "
            "and launch via `streamlit run app/main.py`."
        )

    st.set_page_config(
        page_title="Multi-Label Hallucination Detector",
        layout="wide",
    )
    st.title("Multi-Label LLM Hallucination Detection Workbench")
    st.warning(UNVERIFIED_PREDICTION_DISCLAIMER)

    with st.sidebar:
        st.header("Model & Inference Controls")
        discovered = discover_available_models()
        custom_path = st.text_input(
            "Custom model run directory",
            value="",
            help="Path containing best_checkpoint.pt (DeBERTa/RoBERTa/DistilBERT/ModernBERT) or baseline_model.joblib.",
        )

        selected_dir: str | None = custom_path.strip() or (discovered[0]["path"] if discovered else None)
        if discovered:
            labels = [d["label"] for d in discovered]
            choice = st.selectbox("Discovered trained runs", options=labels, index=0)
            if not custom_path.strip():
                selected_dir = discovered[labels.index(choice)]["path"]
        else:
            st.info(
                "No saved checkpoint (`best_checkpoint.pt`) or classical baseline (`baseline_model.joblib`) "
                "was found under `experiments/` or `results/`. Run `python -m src.models.baseline` or "
                "`scripts/colab_workflow.py` first, then specify or reload a run directory."
            )

        threshold_mode = st.radio(
            "Threshold Policy",
            options=["val_tuned", "fixed_050"],
            format_func=lambda m: "Validation-Tuned tau* (thresholds.json)" if m == "val_tuned" else "Fixed Baseline tau = 0.50",
        )
        use_calibration = st.checkbox(
            "Apply val-fitted Temperature Scaling (if available)",
            value=True,
        )

        st.divider()
        st.subheader("Taxonomy Reference (Cat I-V)")
        for info in CATEGORY_DESCRIPTIONS.values():
            st.markdown(f"**{info['code']} — {info['title']}**: {info['description']}")

    preset_name = st.selectbox("Load example preset (optional)", options=["(Custom Input)", *EXAMPLE_PRESETS.keys()])
    preset = EXAMPLE_PRESETS.get(preset_name, {"question": "", "response": "", "context": ""})

    col_q, col_ctx = st.columns(2)
    with col_q:
        question_text = st.text_area("Question (optional)", value=preset["question"], height=100)
    with col_ctx:
        context_text = st.text_area("Reference Context (optional; used for Cat V CI check)", value=preset["context"], height=100)

    response_text = st.text_area("Candidate LLM Response (required)", value=preset["response"], height=160)

    if st.button("Analyze Response for Hallucination Categories", type="primary"):
        if not selected_dir:
            st.error("No valid trained run directory is selected.")
            return
        if not response_text.strip():
            st.error("Please supply a non-empty Candidate LLM Response.")
            return
        try:
            predictor = HallucinationPredictor(selected_dir)
            result = predictor.predict_single(
                response_text=response_text,
                question_text=question_text,
                context_text=context_text,
                use_calibration=use_calibration,
                threshold_mode=threshold_mode,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Triggered Categories", str(len(result.active_categories)))
        m2.metric("Task B1 Binary Max Score", f"{result.any_hallucination_binary_score:.3f}")
        m3.metric("Temperature Calibrated", "Yes" if result.calibration_applied else "No (raw)")
        m4.metric("Inference Latency", f"{result.latency_ms:.1f} ms")

        if result.active_categories:
            st.error("Triggered Hallucination Types: " + ", ".join(result.active_categories))
        else:
            st.success("No hallucination categories crossed active thresholds (Clean Prediction).")

        st.subheader("Per-Category Confidence Scores & Decision Status")
        st.dataframe(format_prediction_rows(result), width="stretch", hide_index=True)

        trunc = format_truncation_summary(result.truncation_metadata)
        with st.expander("Token Budgeting & Truncation Telemetry", expanded=bool(trunc["warnings"])):
            st.write(
                f"Sequence tokens: **{trunc['total_tokens']} / {trunc['max_length']}** | "
                f"Question: `{trunc['question_tokens']}` | Response: `{trunc['response_tokens']}` | "
                f"Context: `{trunc['context_tokens']}`"
            )
            for w in trunc["warnings"]:
                st.warning(w)

        st.subheader("Response Sentence Inspection (Lexical Context Overlap Aid)")
        st.caption("Heuristic token-overlap visual aid against supplied reference context (not causal model attribution).")
        st.markdown(render_sentence_highlights_html(result.sentence_highlights), unsafe_allow_html=True)

        st.download_button(
            label="Download Prediction Result (JSON)",
            data=export_prediction_json_bytes(result),
            file_name="hallucination_prediction.json",
            mime="application/json",
        )


if __name__ == "__main__":  # pragma: no cover
    if st is not None:
        run_app()

