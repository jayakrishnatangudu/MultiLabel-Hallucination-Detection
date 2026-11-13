"""Stage 1 deterministic rejection rules and Stage 2 blind cross-judge auditing."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
from typing import Any

from src.data.schema import AutoDetectionTier, HALLUCINATION_LABELS
from src.generation.backend import LLMClient, LLMResponse
from src.generation.prompt_builder import LoadedPromptTemplate, render_cross_judge

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)
_BOILERPLATE_PREFIX_RE = re.compile(
    r"^\s*(?:here is (?:the )?(?:requested )?(?:hallucinated|perturbed|updated) (?:answer|response)\s*:\s*)",
    flags=re.IGNORECASE,
)


@dataclass
class CandidateValidationResult:
    """Outcome of deterministic Stage 1 filters on one generator attempt."""

    passed: bool
    extracted_response: str
    parsed_json: dict[str, Any] | None
    rejection_reasons: list[str]
    normalized_edit_distance: float
    length_ratio: float


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object cleanly from raw text, tolerating fenced markdown blocks."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    fenced = _FENCED_JSON_RE.search(cleaned)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(cleaned[brace_start : brace_end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def compute_normalized_word_edit_distance(original: str, candidate: str) -> float:
    """Return 1 - SequenceMatcher ratio over lowercased tokens (0.0 = identical, 1.0 = disjoint)."""
    a_tokens = original.lower().split()
    b_tokens = candidate.lower().split()
    if not a_tokens and not b_tokens:
        return 0.0
    sim = SequenceMatcher(None, a_tokens, b_tokens).ratio()
    return round(1.0 - sim, 4)


def validate_candidate_generation(
    *,
    raw_output: str,
    response_field: str,
    original_answer: str,
    intended_bits: str,
    has_context: bool,
    val_cfg: dict[str, Any],
) -> CandidateValidationResult:
    """Apply Stage 1 deterministic filters (JSON schema, refusal, no-op identity, length, CI rule)."""
    reasons: list[str] = []
    parsed = extract_json_object(raw_output)
    if parsed is None:
        return CandidateValidationResult(
            passed=False,
            extracted_response="",
            parsed_json=None,
            rejection_reasons=["json_parse_failure"],
            normalized_edit_distance=0.0,
            length_ratio=0.0,
        )

    candidate = str(parsed.get(response_field, "")).strip()
    if not candidate and response_field != "generated_response":
        candidate = str(parsed.get("generated_response", "")).strip()
    elif not candidate and response_field != "perturbed_response":
        candidate = str(parsed.get("perturbed_response", "")).strip()

    candidate = _BOILERPLATE_PREFIX_RE.sub("", candidate).strip()
    if not candidate:
        return CandidateValidationResult(
            passed=False,
            extracted_response="",
            parsed_json=parsed,
            rejection_reasons=["empty_response"],
            normalized_edit_distance=0.0,
            length_ratio=0.0,
        )

    lowered = candidate.lower()
    for pat in val_cfg.get("refusal_patterns", []):
        if str(pat).lower() in lowered:
            reasons.append(f"refusal_detected:{pat}")
            break

    if bool(val_cfg.get("enforce_zero_ci_when_no_context", True)) and not has_context and intended_bits[4] == "1":
        reasons.append("invalid_ci_without_context")

    orig_words = max(1, len(original_answer.split()))
    cand_words = len(candidate.split())
    length_ratio = round(cand_words / orig_words, 4)
    min_ratio = float(val_cfg.get("min_length_ratio", 0.60))
    max_ratio = float(val_cfg.get("max_length_ratio", 2.20))
    short_thresh = int(val_cfg.get("short_answer_word_threshold", 25))
    short_cap = int(val_cfg.get("short_answer_max_words_cap", 95))

    if length_ratio < min_ratio and cand_words < 12:
        reasons.append(f"response_too_short:ratio={length_ratio}")
    if orig_words < short_thresh:
        if cand_words > short_cap:
            reasons.append(f"response_too_long:words={cand_words}>cap={short_cap}")
    elif length_ratio > max_ratio:
        reasons.append(f"response_too_long:ratio={length_ratio}")

    edit_dist = compute_normalized_word_edit_distance(original_answer, candidate)
    is_perturbed = any(ch == "1" for ch in intended_bits)
    min_edit = float(val_cfg.get("min_normalized_edit_distance_for_perturbed", 0.05))
    if is_perturbed and edit_dist < min_edit:
        reasons.append(f"identity_or_noop_edit:dist={edit_dist}")

    return CandidateValidationResult(
        passed=len(reasons) == 0,
        extracted_response=candidate,
        parsed_json=parsed,
        rejection_reasons=reasons,
        normalized_edit_distance=edit_dist,
        length_ratio=length_ratio,
    )


def audit_with_cross_judge(
    *,
    judge_client: LLMClient,
    judge_template: LoadedPromptTemplate,
    judge_decoding: dict[str, Any],
    domain: str,
    question: str,
    has_context: bool,
    prompt_context: str | None,
    reference_answer: str,
    candidate_response: str,
    intended_bits: str | None,
    seed: int | None = None,
) -> tuple[AutoDetectionTier, LLMResponse, list[str]]:
    """
    Run blind diagnostic cross-judge auditing without exposing intended labels.

    Outputs populate AutoDetectionTier ONLY (target_missed, spillover_flagged, predicted_labels);
    human_observed_labels and adjudicated_gold_labels are never touched here.
    """
    sys_msg, usr_msg = render_cross_judge(
        judge_template,
        domain=domain,
        question=question,
        has_context=has_context,
        prompt_context=prompt_context,
        reference_answer=reference_answer,
        candidate_response=candidate_response,
    )
    resp = judge_client.generate(
        system_prompt=sys_msg,
        user_prompt=usr_msg,
        temperature=float(judge_decoding.get("temperature", 0.0)),
        top_p=float(judge_decoding.get("top_p", 1.0)),
        max_tokens=int(judge_decoding.get("max_tokens", 512)),
        seed=seed,
    )

    parsed = extract_json_object(resp.raw_text)
    diag_flags: list[str] = []
    if parsed is None or not isinstance(parsed.get("labels"), dict):
        diag_flags.append("cross_judge_parse_failure")
        tier = AutoDetectionTier(
            judge_model=judge_client.model_id,
            judge_provider=getattr(judge_client, "provider", None),
            judge_model_family=getattr(judge_client, "model_family", None),
            predicted_labels=None,
            rule_checks_passed=True,
            target_missed=False,
            spillover_flagged=False,
            coherence_score_1_to_5=None,
            diagnostic_notes="Cross-judge output could not be parsed cleanly.",
        )
        return tier, resp, diag_flags

    label_dict = parsed["labels"]
    pred_vec = [int(bool(label_dict.get(k, 0))) for k in HALLUCINATION_LABELS]
    if not has_context and pred_vec[4] == 1:
        # Diagnostic catch: judge tried to mark CI without context
        pred_vec[4] = 0
        diag_flags.append("judge_attempted_ci_without_context")

    raw_coh = parsed.get("coherence_score_1_to_5")
    coherence: float | None = None
    if raw_coh is not None:
        try:
            coherence = max(1.0, min(5.0, float(raw_coh)))
        except (TypeError, ValueError):
            coherence = None

    target_missed = False
    spillover_flagged = False
    if intended_bits is not None:
        intended_vec = [int(ch) for ch in intended_bits]
        target_missed = any(i == 1 and p == 0 for i, p in zip(intended_vec, pred_vec))
        spillover_flagged = any(i == 0 and p == 1 for i, p in zip(intended_vec, pred_vec))
        if target_missed:
            diag_flags.append("cross_judge_target_missed")
        if spillover_flagged:
            diag_flags.append("cross_judge_spillover_flagged")
    if coherence is not None and coherence < 3.0:
        diag_flags.append(f"low_coherence_score:{coherence}")

    tier = AutoDetectionTier(
        judge_model=judge_client.model_id,
        judge_provider=getattr(judge_client, "provider", None),
        judge_model_family=getattr(judge_client, "model_family", None),
        predicted_labels=pred_vec,  # type: ignore[arg-type]
        rule_checks_passed=True,
        target_missed=target_missed,
        spillover_flagged=spillover_flagged,
        coherence_score_1_to_5=coherence,
        diagnostic_notes=str(parsed.get("diagnostic_notes") or "").strip() or None,
    )
    return tier, resp, diag_flags

