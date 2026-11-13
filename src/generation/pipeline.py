"""Phase 2 generation orchestration, retry/rejection logging, blinded annotation I/O, and CLI."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schema import (
    AutoDetectionTier,
    GenerationAttemptRecord,
    HALLUCINATION_LABELS,
    MultiLabelRecord,
    SourceQARecord,
)
from src.generation.backend import (
    BackendHTTPRequestError,
    LLMClient,
    build_live_client,
    estimate_worst_case_call_cost_usd as _estimate_worst_case_call_cost_usd,
    init_budget_state as _init_budget_state,
    probe_live_backends,
    record_actual_spend as _record_actual_spend,
    reserve_budget_for_call as _reserve_budget_for_call,
    verify_generation_and_judge_backends,
)
from src.generation.pilot_metrics import compute_pilot_e0_report
from src.generation.prompt_builder import (
    LoadedPromptTemplate,
    load_prompt_template,
    render_cross_judge,
    render_natural_qa,
    render_strategy_a_step,
    render_strategy_b_joint,
    render_strategy_c_span_editing,
)
from src.generation.sampler import (
    binary_str_to_label_dict,
    extract_evidence_anchored_window,
    sample_full_generation_spec,
    sample_natural_test_spec,
    sample_pilot_e0_spec,
)
from src.generation.validators import (
    audit_with_cross_judge,
    extract_json_object,
    validate_candidate_generation,
)
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_jsonl, sha256_json, utc_now_iso, write_json, write_jsonl
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

BLINDED_ANNOTATOR_FIELDS: frozenset[str] = frozenset(
    {
        "annotation_item_id",
        "domain",
        "has_context",
        "question",
        "context",
        "original_answer",
        "candidate_response",
    }
)


def _resolve_prompt_context(
    src: SourceQARecord,
    cfg: dict[str, Any],
) -> tuple[str | None, bool]:
    """Return prompt-scoped context (windowed for CUAD prompts only; raw canonical preserved elsewhere)."""
    prompt_context = src.context
    evidence_window_used = False
    cuad_win_cfg = cfg["source_policies"].get(src.source_dataset, {}).get("evidence_window", {})
    if src.has_context and cuad_win_cfg.get("enabled_for_prompting_only", False):
        prompt_context, evidence_window_used = extract_evidence_anchored_window(
            src.context,
            src.original_answer,
            max_words=int(cuad_win_cfg.get("max_prompt_context_words", 1800)),
            preamble_words=int(cuad_win_cfg.get("preamble_words", 250)),
            window_words_around_span=int(cuad_win_cfg.get("window_words_around_span", 750)),
        )
    return prompt_context, evidence_window_used


def _run_single_slot_generation(
    *,
    sample_id: str,
    base_spec: dict[str, Any],
    strategy_name: str,
    step_order_name: str | None,
    step_sequence: list[str] | None,
    generator_client: LLMClient,
    judge_client: LLMClient,
    templates: dict[str, LoadedPromptTemplate],
    cfg: dict[str, Any],
    split_name: str = "train",
    budget_state: dict[str, Any] | None = None,
) -> tuple[MultiLabelRecord | None, list[GenerationAttemptRecord], MultiLabelRecord | None]:
    """
    Execute generation with retry budget, preserving all attempts & rejected candidates.

    Returns (accepted_or_flagged_record, attempt_logs, rejected_record_if_exhausted).
    """
    src: SourceQARecord = base_spec["source_record"]
    bits: str = base_spec["intended_bits"]
    active_labels: list[str] = base_spec["active_labels"]
    forbidden_labels: list[str] = base_spec["forbidden_labels"]
    seed = int(cfg.get("random_seed", 42))
    gen_cfg = cfg["backends"]["synthetic_generator"]
    dec_cfg = gen_cfg["decoding"]
    max_retries = int(gen_cfg.get("max_retries_per_sample", 3))
    temp_step = float(gen_cfg.get("retry_temperature_step", 0.1))
    val_cfg = cfg["automated_validation"]
    judge_dec = cfg["backends"]["cross_judge_auditor"]["decoding"]

    # CUAD long-context Evidence-Anchored Window (for prompt ONLY; canonical context remains full)
    prompt_context, evidence_window_used = _resolve_prompt_context(src, cfg)

    attempt_logs: list[GenerationAttemptRecord] = []
    last_rejected_reasons: list[str] = []
    last_extracted = ""
    base_temp = float(dec_cfg.get("temperature", 0.7))
    top_p = float(dec_cfg.get("top_p", 0.95))
    max_tokens = int(dec_cfg.get("max_tokens", 1024))

    if strategy_name == "strategy_a":
        prompt_tpl = templates["strategy_a_sequential"]
    elif strategy_name == "strategy_b":
        prompt_tpl = templates["strategy_b_joint"]
    elif strategy_name == "strategy_c":
        prompt_tpl = templates["strategy_c_span_editing"]
    else:
        raise ValueError(f"Unsupported generation strategy: {strategy_name}")

    for attempt_idx in range(1, max_retries + 1):
        cur_temp = round(min(1.2, base_temp + (attempt_idx - 1) * temp_step), 2)
        attempt_seed = seed + attempt_idx - 1
        span_plan_meta: list[dict[str, str]] | None = None

        total_in_tok = 0
        total_out_tok = 0
        total_cost = 0.0
        total_latency_ms = 0.0
        raw_transcript_parts: list[str] = []

        if strategy_name == "strategy_a":
            assert step_sequence is not None and len(step_sequence) == len(active_labels)
            cur_text = src.original_answer
            preserved: list[str] = []
            step_failed_reasons: list[str] = []

            for s_pos, target_cat in enumerate(step_sequence, start=1):
                sys_msg, usr_msg = render_strategy_a_step(
                    prompt_tpl,
                    domain=src.domain,
                    question=src.question,
                    has_context=src.has_context,
                    prompt_context=prompt_context,
                    current_response=cur_text,
                    target_category=target_cat,
                    step_index=s_pos,
                    total_steps=len(step_sequence),
                    preserved_active_labels=preserved,
                    forbidden_labels=forbidden_labels,
                )
                if not _reserve_budget_for_call(
                    budget_state,
                    client=generator_client,
                    system_prompt=sys_msg,
                    user_prompt=usr_msg,
                    max_output_tokens=max_tokens,
                    sample_id=f"{sample_id}[attempt={attempt_idx},step={s_pos}]",
                ):
                    return None, attempt_logs, None
                try:
                    resp = generator_client.generate(
                        system_prompt=sys_msg,
                        user_prompt=usr_msg,
                        temperature=cur_temp,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=attempt_seed + s_pos,
                    )
                    _record_actual_spend(budget_state, resp.cost_usd)
                    total_in_tok += resp.input_tokens
                    total_out_tok += resp.output_tokens
                    total_cost += resp.cost_usd
                    total_latency_ms += resp.latency_ms
                    raw_transcript_parts.append(f"[step_{s_pos}:{target_cat}] {resp.raw_text}")

                    step_val = validate_candidate_generation(
                        raw_output=resp.raw_text,
                        response_field="perturbed_response",
                        original_answer=cur_text,
                        intended_bits="10000",  # Force single-step non-identity edit check
                        has_context=src.has_context,
                        val_cfg=val_cfg,
                    )
                    if not step_val.passed:
                        step_failed_reasons = [f"step_{s_pos}_{target_cat}:{r}" for r in step_val.rejection_reasons]
                        break
                    cur_text = step_val.extracted_response
                    preserved.append(target_cat)
                except (BackendHTTPRequestError, Exception) as exc:
                    logger.warning(
                        "Request/network error during sequential generation step for %s attempt %d step %d: %s",
                        sample_id,
                        attempt_idx,
                        s_pos,
                        exc,
                    )
                    err_code = getattr(exc, "status_code", None) or "conn"
                    step_failed_reasons = [f"step_{s_pos}_{target_cat}:http_error_{err_code}"]
                    break

            raw_output_combined = "\n".join(raw_transcript_parts)
            if step_failed_reasons:
                val_res_passed = False
                rejection_reasons = step_failed_reasons
                extracted_text = cur_text
                norm_edit = 0.0
                len_ratio = 1.0
            else:
                # Re-validate full sequential result against original_answer & length bounds
                synthetic_json = json.dumps({"generated_response": cur_text}, ensure_ascii=False)
                final_val = validate_candidate_generation(
                    raw_output=synthetic_json,
                    response_field="generated_response",
                    original_answer=src.original_answer,
                    intended_bits=bits,
                    has_context=src.has_context,
                    val_cfg=val_cfg,
                )
                val_res_passed = final_val.passed
                rejection_reasons = final_val.rejection_reasons
                extracted_text = final_val.extracted_response or cur_text
                norm_edit = final_val.normalized_edit_distance
                len_ratio = final_val.length_ratio

        else:
            if strategy_name == "strategy_b":
                sys_msg, usr_msg = render_strategy_b_joint(
                    prompt_tpl,
                    domain=src.domain,
                    source_dataset=src.source_dataset,
                    question=src.question,
                    has_context=src.has_context,
                    prompt_context=prompt_context,
                    original_answer=src.original_answer,
                    intended_bits=bits,
                    active_labels=active_labels,
                    forbidden_labels=forbidden_labels,
                )
            else:
                sys_msg, usr_msg, span_plan_meta = render_strategy_c_span_editing(
                    prompt_tpl,
                    domain=src.domain,
                    source_dataset=src.source_dataset,
                    question=src.question,
                    has_context=src.has_context,
                    prompt_context=prompt_context,
                    original_answer=src.original_answer,
                    intended_bits=bits,
                    active_labels=active_labels,
                    forbidden_labels=forbidden_labels,
                )

            if not _reserve_budget_for_call(
                budget_state,
                client=generator_client,
                system_prompt=sys_msg,
                user_prompt=usr_msg,
                max_output_tokens=max_tokens,
                sample_id=f"{sample_id}[attempt={attempt_idx}]",
            ):
                return None, attempt_logs, None
            try:
                resp = generator_client.generate(
                    system_prompt=sys_msg,
                    user_prompt=usr_msg,
                    temperature=cur_temp,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=attempt_seed,
                )
                _record_actual_spend(budget_state, resp.cost_usd)
                total_in_tok = resp.input_tokens
                total_out_tok = resp.output_tokens
                total_cost = resp.cost_usd
                total_latency_ms = resp.latency_ms
                raw_output_combined = resp.raw_text

                final_val = validate_candidate_generation(
                    raw_output=resp.raw_text,
                    response_field="generated_response",
                    original_answer=src.original_answer,
                    intended_bits=bits,
                    has_context=src.has_context,
                    val_cfg=val_cfg,
                )
                val_res_passed = final_val.passed
                rejection_reasons = final_val.rejection_reasons
                extracted_text = final_val.extracted_response
                norm_edit = final_val.normalized_edit_distance
                len_ratio = final_val.length_ratio
            except (BackendHTTPRequestError, Exception) as exc:
                logger.warning(
                    "Request/network error during single-pass generation for %s attempt %d: %s",
                    sample_id,
                    attempt_idx,
                    exc,
                )
                err_code = getattr(exc, "status_code", None) or "conn"
                val_res_passed = False
                rejection_reasons = [f"http_error_{err_code}"]
                raw_output_combined = f"[HTTP_ERROR: {exc}]"
                extracted_text = None
                norm_edit = 0.0
                len_ratio = 0.0

        gen_params = {
            "generation_provider": getattr(generator_client, "provider", None),
            "generation_model_family": getattr(generator_client, "model_family", None),
            "temperature": cur_temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": attempt_seed,
            "step_order_name": step_order_name,
            "step_order_labels": step_sequence,
            "evidence_window_used": evidence_window_used,
            "prompt_sha256": prompt_tpl.template_sha256,
            "input_tokens": total_in_tok,
            "output_tokens": total_out_tok,
            "cost_usd": round(total_cost, 8),
            "latency_ms": round(total_latency_ms, 2),
            "normalized_edit_distance": norm_edit,
            "length_ratio": len_ratio,
        }
        if span_plan_meta is not None:
            gen_params["span_assignment_plan"] = span_plan_meta

        attempt_logs.append(
            GenerationAttemptRecord(
                attempt_id=f"{sample_id}__att_{attempt_idx}",
                sample_id=sample_id,
                base_item_id=base_spec["base_item_id"],
                source_dataset=src.source_dataset,
                source_subset=src.source_subset,
                source_record_id=src.source_record_id,
                source_document_id=src.source_document_id,
                source_group_id=src.source_group_id,
                domain=src.domain,
                split=split_name,  # type: ignore[arg-type]
                generation_strategy=strategy_name,
                step_order_name=step_order_name,
                step_order_labels=step_sequence,
                generation_model=generator_client.model_id,
                prompt_version=prompt_tpl.prompt_version,
                generation_parameters=gen_params,
                attempt_number=attempt_idx,
                intended_vector=bits,
                raw_output=raw_output_combined,
                extracted_response=extracted_text,
                accepted=val_res_passed,
                rejection_reasons=rejection_reasons,
                latency_ms=round(total_latency_ms, 2),
                input_tokens=total_in_tok,
                output_tokens=total_out_tok,
                cost_usd=round(total_cost, 8),
                timestamp_utc=utc_now_iso(),
            )
        )

        last_rejected_reasons = rejection_reasons
        last_extracted = extracted_text

        if val_res_passed:
            # Stage 2: Blind diagnostic cross-judge (strictly advisory)
            j_sys, j_usr = render_cross_judge(
                templates["cross_judge_auditor"],
                domain=src.domain,
                question=src.question,
                has_context=src.has_context,
                prompt_context=prompt_context,
                reference_answer=src.original_answer,
                candidate_response=extracted_text,
            )
            if not _reserve_budget_for_call(
                budget_state,
                client=judge_client,
                system_prompt=j_sys,
                user_prompt=j_usr,
                max_output_tokens=int(judge_dec.get("max_tokens", 512)),
                sample_id=f"{sample_id}[judge]",
            ):
                return None, attempt_logs, None
            try:
                auto_tier, judge_resp, diag_flags = audit_with_cross_judge(
                    judge_client=judge_client,
                    judge_template=templates["cross_judge_auditor"],
                    judge_decoding=judge_dec,
                    domain=src.domain,
                    question=src.question,
                    has_context=src.has_context,
                    prompt_context=prompt_context,
                    reference_answer=src.original_answer,
                    candidate_response=extracted_text,
                    intended_bits=bits,
                    seed=seed,
                )
                _record_actual_spend(budget_state, judge_resp.cost_usd)
                gen_params["judge_cost_usd"] = judge_resp.cost_usd
                gen_params["judge_latency_ms"] = judge_resp.latency_ms
                gen_params["judge_input_tokens"] = judge_resp.input_tokens
                gen_params["judge_output_tokens"] = judge_resp.output_tokens
            except BackendHTTPRequestError as j_exc:
                logger.warning("Judge call failed for %s: %s", sample_id, j_exc)
                auto_tier = AutoDetectionTier(
                    judge_model=judge_client.model_id,
                    judge_provider=getattr(judge_client, "provider", None),
                    judge_model_family=getattr(judge_client, "model_family", None),
                    predicted_labels=None,
                    rule_checks_passed=True,
                    target_missed=False,
                    spillover_flagged=False,
                    coherence_score_1_to_5=None,
                    diagnostic_notes=f"Cross-judge HTTP error: {j_exc}",
                )
                diag_flags = [f"judge_http_error_{j_exc.status_code or 'conn'}"]
                gen_params["judge_cost_usd"] = 0.0
                gen_params["judge_latency_ms"] = 0.0
                gen_params["judge_input_tokens"] = 0
                gen_params["judge_output_tokens"] = 0
            gen_params["base_item_id"] = base_spec["base_item_id"]
            gen_params["in_order_sensitivity_subset"] = bool(
                base_spec.get("in_order_sensitivity_subset", False)
            )

            flagged = bool(auto_tier.target_missed or auto_tier.spillover_flagged or diag_flags)
            status = "auto_flagged" if flagged else "auto_passed"
            intended_dict = binary_str_to_label_dict(bits)
            record = MultiLabelRecord(
                sample_id=sample_id,
                source_dataset=src.source_dataset,
                source_subset=src.source_subset,
                source_record_id=src.source_record_id,
                source_document_id=src.source_document_id,
                source_group_id=src.source_group_id,
                domain=src.domain,
                data_origin="synthetic",
                has_context=src.has_context,
                question=src.question,
                context=src.context,  # Canonical full raw context preserved!
                original_answer=src.original_answer,
                generated_answer=extracted_text,
                generation_method=strategy_name,
                generation_strategy=strategy_name,
                generation_model=generator_client.model_id,
                prompt_version=prompt_tpl.prompt_version,
                generation_parameters=gen_params,
                intended_labels=intended_dict,
                auto_detected_labels=auto_tier,
                label_vector=intended_dict.to_vector(),
                validation_status=status,  # type: ignore[arg-type]
                rejection_reasons=diag_flags,
                attempted_count=attempt_idx,
                licensing=src.licensing,
                split=split_name,  # type: ignore[arg-type]
            )
            return record, attempt_logs, None

    # Exhausted retry budget: emit preserved rejected record
    intended_dict = binary_str_to_label_dict(bits)
    rejected_rec = MultiLabelRecord(
        sample_id=sample_id,
        source_dataset=src.source_dataset,
        source_subset=src.source_subset,
        source_record_id=src.source_record_id,
        source_document_id=src.source_document_id,
        source_group_id=src.source_group_id,
        domain=src.domain,
        data_origin="synthetic",
        has_context=src.has_context,
        question=src.question,
        context=src.context,
        original_answer=src.original_answer,
        generated_answer=last_extracted,
        generation_method=strategy_name,
        generation_strategy=strategy_name,
        generation_model=generator_client.model_id,
        prompt_version=prompt_tpl.prompt_version,
        generation_parameters={
            "generation_provider": getattr(generator_client, "provider", None),
            "generation_model_family": getattr(generator_client, "model_family", None),
            "base_item_id": base_spec["base_item_id"],
            "in_order_sensitivity_subset": bool(base_spec.get("in_order_sensitivity_subset", False)),
            "step_order_name": step_order_name,
            "step_order_labels": step_sequence,
            "retries_exhausted": max_retries,
        },
        intended_labels=intended_dict,
        auto_detected_labels=None,
        label_vector=None,
        validation_status="rejected",
        rejection_reasons=last_rejected_reasons or ["max_retries_exhausted"],
        attempted_count=max_retries,
        licensing=src.licensing,
        split=split_name,  # type: ignore[arg-type]
    )
    return None, attempt_logs, rejected_rec


def export_blinded_annotation_packet(
    records: list[MultiLabelRecord],
    out_packet_path: str | Path,
    out_key_path: str | Path,
    *,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Export an annotator-safe blinded JSONL packet.

    Explicitly strips intended_labels, generation_strategy, generation_model, data_origin,
    source_dataset, and auto_detected_labels so human annotations remain independent.
    """
    rng = np.random.default_rng(seed)
    order = list(range(len(records)))
    rng.shuffle(order)

    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    for seq_idx, rec_idx in enumerate(order):
        rec = records[rec_idx]
        ann_id = f"ANN_{seq_idx:05d}"
        packet_obj = {
            "annotation_item_id": ann_id,
            "domain": rec.domain,
            "has_context": rec.has_context,
            "question": rec.question,
            "context": rec.context,
            "original_answer": rec.original_answer,
            "candidate_response": rec.generated_answer,
        }
        assert set(packet_obj.keys()) == BLINDED_ANNOTATOR_FIELDS
        packet_rows.append(packet_obj)
        key_rows.append(
            {
                "annotation_item_id": ann_id,
                "sample_id": rec.sample_id,
                "source_group_id": rec.source_group_id,
                "source_dataset": rec.source_dataset,
                "data_origin": rec.data_origin,
                "generation_strategy": rec.generation_strategy,
                "intended_labels": rec.intended_labels.to_vector() if rec.intended_labels else None,
            }
        )

    p1 = write_jsonl(out_packet_path, packet_rows)
    p2 = write_jsonl(out_key_path, key_rows)
    return p1, p2


def apply_human_annotations(
    records: list[MultiLabelRecord],
    blind_key_rows: list[dict[str, Any]],
    rater_annotations: list[dict[str, Any]],
    *,
    adjudicator_decisions: dict[str, list[int]] | None = None,
) -> list[MultiLabelRecord]:
    """
    Attach blinded human labels and compute independent adjudicated_gold_labels.

    Auto-detected LLM cross-judge predictions are never read or used for ground truth.
    """
    ann_to_sample = {row["annotation_item_id"]: row["sample_id"] for row in blind_key_rows}
    by_sample_raters: dict[str, dict[str, list[int]]] = defaultdict(dict)
    by_sample_notes: dict[str, list[str]] = defaultdict(list)

    for ann in rater_annotations:
        sample_id = ann.get("sample_id") or ann_to_sample.get(str(ann.get("annotation_item_id", "")))
        if not sample_id:
            continue
        rater_id = str(ann["rater_id"])
        vec = [int(x) for x in ann["labels"]]
        if len(vec) != 5:
            raise ValueError(f"Human label vector for {sample_id}/{rater_id} must have length 5.")
        by_sample_raters[sample_id][rater_id] = vec
        if ann.get("explanation"):
            by_sample_notes[sample_id].append(f"{rater_id}: {ann['explanation']}")

    updated: list[MultiLabelRecord] = []
    adj_map = adjudicator_decisions or {}
    for rec in records:
        raters = by_sample_raters.get(rec.sample_id)
        if not raters:
            updated.append(rec)
            continue

        vectors = list(raters.values())
        n_raters = len(vectors)
        resolved = True
        gold_vec: list[int] = []
        for bit_idx in range(5):
            ones = sum(v[bit_idx] for v in vectors)
            zeros = n_raters - ones
            if ones > zeros:
                gold_vec.append(1)
            elif zeros > ones:
                gold_vec.append(0)
            else:
                resolved = False
                break

        if not resolved and rec.sample_id in adj_map:
            gold_vec = [int(x) for x in adj_map[rec.sample_id]]
            resolved = True

        if resolved and not rec.has_context and gold_vec[4] == 1:
            raise ValueError(
                f"Annotated gold label CI=1 invalid on context-free sample {rec.sample_id}"
            )

        if resolved:
            updated.append(
                rec.model_copy(
                    update={
                        "human_observed_labels": raters,
                        "adjudicated_gold_labels": gold_vec,
                        "label_vector": gold_vec,
                        "validation_status": "human_verified",
                        "annotation_explanation": " | ".join(by_sample_notes.get(rec.sample_id, [])) or None,
                    }
                )
            )
        else:
            updated.append(
                rec.model_copy(
                    update={
                        "human_observed_labels": raters,
                        "rejection_reasons": list(rec.rejection_reasons) + ["needs_adjudication_tie"],
                    }
                )
            )
    return updated


def _load_resume_state(
    out_dir: Path,
    accepted_file: str,
    rejected_file: str,
    attempts_file: str,
) -> tuple[list[MultiLabelRecord], list[MultiLabelRecord], list[GenerationAttemptRecord], set[str]]:
    """Load existing JSONL generation checkpoints so interrupted runs resume cleanly."""
    accepted: list[MultiLabelRecord] = []
    rejected: list[MultiLabelRecord] = []
    attempts: list[GenerationAttemptRecord] = []
    acc_path = out_dir / accepted_file
    rej_path = out_dir / rejected_file
    att_path = out_dir / attempts_file
    if acc_path.exists():
        accepted = [MultiLabelRecord(**d) for d in read_jsonl(acc_path)]
    if rej_path.exists():
        rejected = [MultiLabelRecord(**d) for d in read_jsonl(rej_path)]
    if att_path.exists():
        attempts = [GenerationAttemptRecord(**d) for d in read_jsonl(att_path)]
    completed_ids = {r.sample_id for r in accepted} | {r.sample_id for r in rejected}
    return accepted, rejected, attempts, completed_ids


def _run_clean_control_slot(
    *,
    c_spec: dict[str, Any],
    split_name: str,
    judge_client: LLMClient,
    judge_template: LoadedPromptTemplate,
    judge_dec: dict[str, Any],
    cfg: dict[str, Any],
    seed: int,
    budget_state: dict[str, Any],
) -> MultiLabelRecord | None:
    """Audit an unmodified clean control item with the blind cross-judge and build MultiLabelRecord."""
    src: SourceQARecord = c_spec["source_record"]
    prompt_context, _ = _resolve_prompt_context(src, cfg)
    j_sys, j_usr = render_cross_judge(
        judge_template,
        domain=src.domain,
        question=src.question,
        has_context=src.has_context,
        prompt_context=prompt_context,
        reference_answer=src.original_answer,
        candidate_response=src.original_answer,
    )
    if not _reserve_budget_for_call(
        budget_state,
        client=judge_client,
        system_prompt=j_sys,
        user_prompt=j_usr,
        max_output_tokens=int(judge_dec.get("max_tokens", 512)),
        sample_id=f"{c_spec['base_item_id']}[clean_judge]",
    ):
        return None
    try:
        auto_tier, j_resp, diag_flags = audit_with_cross_judge(
            judge_client=judge_client,
            judge_template=judge_template,
            judge_decoding=judge_dec,
            domain=src.domain,
            question=src.question,
            has_context=src.has_context,
            prompt_context=prompt_context,
            reference_answer=src.original_answer,
            candidate_response=src.original_answer,
            intended_bits="00000",
            seed=seed,
        )
        _record_actual_spend(budget_state, j_resp.cost_usd)
        j_cost = j_resp.cost_usd
        j_lat = j_resp.latency_ms
    except BackendHTTPRequestError as exc:
        logger.warning("Clean control judge failed for %s: %s", c_spec["base_item_id"], exc)
        auto_tier = None
        j_cost = 0.0
        j_lat = 0.0
    intended_dict = binary_str_to_label_dict("00000")
    return MultiLabelRecord(
        sample_id=c_spec["base_item_id"],
        source_dataset=src.source_dataset,
        source_subset=src.source_subset,
        source_record_id=src.source_record_id,
        source_document_id=src.source_document_id,
        source_group_id=src.source_group_id,
        domain=src.domain,
        data_origin="synthetic",
        has_context=src.has_context,
        question=src.question,
        context=src.context,
        original_answer=src.original_answer,
        generated_answer=src.original_answer,
        generation_method="clean_control",
        generation_strategy="clean_control",
        generation_model=None,
        prompt_version=cfg["prompts"]["version"],
        generation_parameters={
            "base_item_id": c_spec["base_item_id"],
            "judge_cost_usd": j_cost,
            "judge_latency_ms": j_lat,
        },
        intended_labels=intended_dict,
        auto_detected_labels=auto_tier,
        label_vector=[0, 0, 0, 0, 0],
        validation_status="auto_flagged" if (auto_tier.spillover_flagged or diag_flags) else "auto_passed",
        rejection_reasons=diag_flags,
        attempted_count=1,
        licensing=src.licensing,
        split=split_name,  # type: ignore[arg-type]
    )


def _flush_checkpoint(
    out_dir: Path,
    accepted_file: str,
    rejected_file: str,
    attempts_file: str,
    accepted_records: list[MultiLabelRecord],
    rejected_records: list[MultiLabelRecord],
    all_attempts: list[GenerationAttemptRecord],
) -> None:
    """Persist intermediate generation state to disk so interrupted sessions lose minimal work."""
    write_jsonl(out_dir / accepted_file, accepted_records)
    write_jsonl(out_dir / rejected_file, rejected_records)
    write_jsonl(out_dir / attempts_file, all_attempts)


def run_pilot_e0(
    config_path: str | Path,
    *,
    execute_live: bool = False,
    generator_client: LLMClient | None = None,
    judge_client: LLMClient | None = None,
    output_dir: str | Path | None = None,
    source_records_override: list[SourceQARecord] | None = None,
    resume: bool = False,
    max_new_slots: int | None = None,
    checkpoint_every: int = 1,
) -> dict[str, Any]:
    """
    Run the Pilot E0 paired pipeline across Clean G0, Strategies A/B/C, and E0-Order.

    Supports resumable checkpointing (resume=True) and bounded free-tier batches
    (max_new_slots). Enforces backend preflight verification when running live.
    """
    cfg = load_yaml_config(config_path)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_project_path(cfg["paths"]["pilot_output_dir"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if generator_client is None or judge_client is None:
        if not execute_live:
            raise RuntimeError(
                "Refusing to run Pilot E0 without --execute-live or explicit mock LLM clients."
            )
        verify_generation_and_judge_backends(cfg, check_live=True)
        generator_client = build_live_client(cfg["backends"]["synthetic_generator"])
        judge_client = build_live_client(cfg["backends"]["cross_judge_auditor"])

    if source_records_override is not None:
        source_records = source_records_override
    else:
        raw_split = read_jsonl(resolve_project_path(cfg["paths"]["source_splits"]["train"]))
        source_records = [SourceQARecord(**d) for d in raw_split]

    paired_specs, clean_specs, sample_summary = sample_pilot_e0_spec(source_records, cfg)
    templates = {
        key: load_prompt_template(cfg["prompts"][key])
        for key in (
            "strategy_a_sequential",
            "strategy_b_joint",
            "strategy_c_span_editing",
            "cross_judge_auditor",
        )
    }

    if resume:
        accepted_records, rejected_records, all_attempts, completed_ids = _load_resume_state(
            out_dir,
            "pilot_records.jsonl",
            "rejected_samples.jsonl",
            "generation_attempts.jsonl",
        )
    else:
        accepted_records, rejected_records, all_attempts, completed_ids = [], [], [], set()

    budget_state = _init_budget_state(cfg)
    judge_dec = cfg["backends"]["cross_judge_auditor"]["decoding"]
    seed = int(cfg.get("random_seed", 42))
    new_slots_processed = 0

    def _can_process_more() -> bool:
        if budget_state["halted_safely"]:
            return False
        if max_new_slots is not None and new_slots_processed >= max_new_slots:
            return False
        return True

    # 1. Clean controls (G0) + blind cross-judge diagnostic
    for c_spec in clean_specs:
        if not _can_process_more():
            break
        if c_spec["base_item_id"] in completed_ids:
            continue
        clean_rec = _run_clean_control_slot(
            c_spec=c_spec,
            split_name="train",
            judge_client=judge_client,
            judge_template=templates["cross_judge_auditor"],
            judge_dec=judge_dec,
            cfg=cfg,
            seed=seed,
            budget_state=budget_state,
        )
        if clean_rec is None:
            break
        accepted_records.append(clean_rec)
        completed_ids.add(clean_rec.sample_id)
        new_slots_processed += 1
        if checkpoint_every > 0 and new_slots_processed % checkpoint_every == 0:
            _flush_checkpoint(
                out_dir,
                "pilot_records.jsonl",
                "rejected_samples.jsonl",
                "generation_attempts.jsonl",
                accepted_records,
                rejected_records,
                all_attempts,
            )

    # 2. Paired runs across identical base items: Strategy A (forward), Strategy B, Strategy C
    for spec in paired_specs:
        if not _can_process_more():
            break
        base_id = spec["base_item_id"]
        fwd_seq = spec["step_orders"]["forward"]
        for strat_key, order_name, seq in [
            ("strategy_a", "forward", fwd_seq),
            ("strategy_b", None, None),
            ("strategy_c", None, None),
        ]:
            if not _can_process_more():
                break
            sample_id = f"{base_id}__{strat_key}" + (f"_{order_name}" if order_name else "")
            if sample_id in completed_ids:
                continue
            rec, atts, rej = _run_single_slot_generation(
                sample_id=sample_id,
                base_spec=spec,
                strategy_name=strat_key,
                step_order_name=order_name,
                step_sequence=seq,
                generator_client=generator_client,
                judge_client=judge_client,
                templates=templates,
                cfg=cfg,
                split_name="train",
                budget_state=budget_state,
            )
            all_attempts.extend(atts)
            if rec is not None:
                accepted_records.append(rec)
                completed_ids.add(sample_id)
            if rej is not None:
                rejected_records.append(rej)
                completed_ids.add(sample_id)
            new_slots_processed += 1
            if checkpoint_every > 0 and new_slots_processed % checkpoint_every == 0:
                _flush_checkpoint(
                    out_dir,
                    "pilot_records.jsonl",
                    "rejected_samples.jsonl",
                    "generation_attempts.jsonl",
                    accepted_records,
                    rejected_records,
                    all_attempts,
                )

        # 3. Strategy A Order-Sensitivity extra orders (reverse & randomized) on tagged 18-item subset
        if _can_process_more() and spec.get("in_order_sensitivity_subset"):
            for extra_order in ("reverse", "randomized"):
                if not _can_process_more():
                    break
                seq = spec["step_orders"][extra_order]
                sample_id = f"{base_id}__strategy_a_{extra_order}"
                if sample_id in completed_ids:
                    continue
                rec, atts, rej = _run_single_slot_generation(
                    sample_id=sample_id,
                    base_spec=spec,
                    strategy_name="strategy_a",
                    step_order_name=extra_order,
                    step_sequence=seq,
                    generator_client=generator_client,
                    judge_client=judge_client,
                    templates=templates,
                    cfg=cfg,
                    split_name="train",
                    budget_state=budget_state,
                )
                all_attempts.extend(atts)
                if rec is not None:
                    accepted_records.append(rec)
                    completed_ids.add(sample_id)
                if rej is not None:
                    rejected_records.append(rej)
                    completed_ids.add(sample_id)
                new_slots_processed += 1
                if checkpoint_every > 0 and new_slots_processed % checkpoint_every == 0:
                    _flush_checkpoint(
                        out_dir,
                        "pilot_records.jsonl",
                        "rejected_samples.jsonl",
                        "generation_attempts.jsonl",
                        accepted_records,
                        rejected_records,
                        all_attempts,
                    )

    # Enforce strict split integrity across outputs
    bad_splits = [r.sample_id for r in accepted_records + rejected_records if r.split != "train"]
    if bad_splits:
        raise AssertionError(f"Split leakage detected in pilot outputs: {bad_splits}")

    _flush_checkpoint(
        out_dir,
        "pilot_records.jsonl",
        "rejected_samples.jsonl",
        "generation_attempts.jsonl",
        accepted_records,
        rejected_records,
        all_attempts,
    )
    export_blinded_annotation_packet(
        accepted_records,
        out_dir / "blinded_annotation_packet.jsonl",
        out_dir / "blind_key_mapping.jsonl",
        seed=seed,
    )

    metrics = compute_pilot_e0_report(
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        attempt_records=all_attempts,
    )
    full_summary = {
        "sampling_summary": sample_summary,
        "pilot_metrics": metrics,
        "budget_guard": budget_state,
        "resume_mode": resume,
        "new_slots_processed_this_run": new_slots_processed,
        "config_sha256": sha256_json(cfg),
    }
    write_json(out_dir / "pilot_e0_summary.json", full_summary)
    return full_summary


def run_full_generation(
    config_path: str | Path,
    *,
    split_name: str = "train",
    strategy_name: str = "strategy_a",
    execute_live: bool = False,
    generator_client: LLMClient | None = None,
    judge_client: LLMClient | None = None,
    output_dir: str | Path | None = None,
    source_records_override: list[SourceQARecord] | None = None,
    total_target_override: int | None = None,
    resume: bool = False,
    max_new_slots: int | None = None,
    checkpoint_every: int = 1,
) -> dict[str, Any]:
    """
    Execute split-locked full synthetic corpus generation with resume & budget guard.

    Supports generating train, val, or test_synthetic independently using the
    approved winner strategy (default 'strategy_b').
    """
    pool_key = {"train": "train", "val": "val", "test_synthetic": "test_synthetic_pool"}.get(split_name)
    if pool_key is None:
        raise ValueError(f"Invalid split_name={split_name!r}; expected 'train', 'val', or 'test_synthetic'.")

    cfg = load_yaml_config(config_path)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_project_path(cfg["paths"]["full_output_dir"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if generator_client is None or judge_client is None:
        if not execute_live:
            raise RuntimeError(
                "Refusing to run full generation without --execute-live or explicit mock LLM clients."
            )
        verify_generation_and_judge_backends(cfg, check_live=True)
        generator_client = build_live_client(cfg["backends"]["synthetic_generator"])
        judge_client = build_live_client(cfg["backends"]["cross_judge_auditor"])

    if source_records_override is not None:
        source_records = source_records_override
    else:
        raw_split = read_jsonl(resolve_project_path(cfg["paths"]["source_splits"][pool_key]))
        source_records = [SourceQARecord(**d) for d in raw_split]

    specs, sample_summary = sample_full_generation_spec(
        source_records,
        cfg,
        split_name=split_name,
        total_target_override=total_target_override,
        strategy_name=strategy_name,
    )
    templates = {
        key: load_prompt_template(cfg["prompts"][key])
        for key in (
            "strategy_a_sequential",
            "strategy_b_joint",
            "strategy_c_span_editing",
            "cross_judge_auditor",
        )
    }

    acc_name = f"synthetic_{split_name}.jsonl"
    rej_name = f"rejected_{split_name}.jsonl"
    att_name = f"attempts_{split_name}.jsonl"
    if resume:
        accepted_records, rejected_records, all_attempts, completed_ids = _load_resume_state(
            out_dir, acc_name, rej_name, att_name
        )
    else:
        accepted_records, rejected_records, all_attempts, completed_ids = [], [], [], set()

    budget_state = _init_budget_state(cfg)
    judge_dec = cfg["backends"]["cross_judge_auditor"]["decoding"]
    seed = int(cfg.get("random_seed", 42))
    new_slots_processed = 0

    for spec in specs:
        if budget_state["halted_safely"]:
            break
        if max_new_slots is not None and new_slots_processed >= max_new_slots:
            break
        sample_id = spec["base_item_id"] if spec["is_clean"] else f"{spec['base_item_id']}__{strategy_name}"
        if sample_id in completed_ids:
            continue

        if spec["is_clean"]:
            clean_rec = _run_clean_control_slot(
                c_spec=spec,
                split_name=split_name,
                judge_client=judge_client,
                judge_template=templates["cross_judge_auditor"],
                judge_dec=judge_dec,
                cfg=cfg,
                seed=seed,
                budget_state=budget_state,
            )
            if clean_rec is None:
                break
            accepted_records.append(clean_rec)
            completed_ids.add(clean_rec.sample_id)
        else:
            seq = spec["step_orders"]["forward"] if strategy_name == "strategy_a" else None
            rec, atts, rej = _run_single_slot_generation(
                sample_id=sample_id,
                base_spec=spec,
                strategy_name=strategy_name,
                step_order_name="forward" if strategy_name == "strategy_a" else None,
                step_sequence=seq,
                generator_client=generator_client,
                judge_client=judge_client,
                templates=templates,
                cfg=cfg,
                split_name=split_name,
                budget_state=budget_state,
            )
            all_attempts.extend(atts)
            if rec is not None:
                accepted_records.append(rec)
                completed_ids.add(sample_id)
            if rej is not None:
                rejected_records.append(rej)
                completed_ids.add(sample_id)
        new_slots_processed += 1
        if checkpoint_every > 0 and new_slots_processed % checkpoint_every == 0:
            _flush_checkpoint(
                out_dir, acc_name, rej_name, att_name, accepted_records, rejected_records, all_attempts
            )

    bad_splits = [r.sample_id for r in accepted_records + rejected_records if r.split != split_name]
    if bad_splits:
        raise AssertionError(f"Split leakage detected in full generation outputs: {bad_splits}")

    _flush_checkpoint(out_dir, acc_name, rej_name, att_name, accepted_records, rejected_records, all_attempts)
    summary = {
        "split_name": split_name,
        "strategy_name": strategy_name,
        "sampling_summary": sample_summary,
        "accepted_count": len(accepted_records),
        "rejected_count": len(rejected_records),
        "attempt_count": len(all_attempts),
        "budget_guard": budget_state,
        "resume_mode": resume,
        "new_slots_processed_this_run": new_slots_processed,
        "config_sha256": sha256_json(cfg),
    }
    write_json(out_dir / f"full_generation_{split_name}_summary.json", summary)
    return summary


def run_natural_generation(
    config_path: str | Path,
    *,
    execute_live: bool = False,
    respondent_clients: dict[str, LLMClient] | None = None,
    judge_client: LLMClient | None = None,
    output_dir: str | Path | None = None,
    source_records_override: list[SourceQARecord] | None = None,
    resume: bool = False,
    max_new_slots: int | None = None,
    checkpoint_every: int = 1,
) -> dict[str, Any]:
    """
    Generate natural multi-model QA responses on held-out test_natural_pool.

    Leaves adjudicated_gold_labels=None until blind human annotation completes
    (cross-judge auto_detected_labels are saved strictly for human agreement studies).
    """
    # Natural QA outputs route to natural_output_dir by default
    cfg = load_yaml_config(config_path)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_project_path(cfg["paths"].get("natural_output_dir") or cfg["paths"]["full_output_dir"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if respondent_clients is None or judge_client is None:
        if not execute_live:
            raise RuntimeError(
                "Refusing to run test_natural generation without --execute-live or explicit mock clients."
            )
        verify_generation_and_judge_backends(cfg, check_live=True)
        judge_client = judge_client or build_live_client(cfg["backends"]["cross_judge_auditor"])
        respondent_clients = {}
        for m_spec in cfg["backends"]["natural_qa_models"]:
            respondent_clients[m_spec["model_id"]] = build_live_client(m_spec)

    if source_records_override is not None:
        source_records = source_records_override
    else:
        raw_split = read_jsonl(resolve_project_path(cfg["paths"]["source_splits"]["test_natural_pool"]))
        source_records = [SourceQARecord(**d) for d in raw_split]

    specs, sample_summary = sample_natural_test_spec(source_records, cfg)
    nat_tpl = load_prompt_template(cfg["prompts"]["natural_qa_generation"])
    judge_tpl = load_prompt_template(cfg["prompts"]["cross_judge_auditor"])
    judge_dec = cfg["backends"]["cross_judge_auditor"]["decoding"]
    seed = int(cfg.get("random_seed", 42))

    acc_name = "test_natural_unannotated.jsonl"
    rej_name = "rejected_test_natural.jsonl"
    att_name = "attempts_test_natural.jsonl"
    if resume:
        records, rejected_records, _, completed_ids = _load_resume_state(out_dir, acc_name, rej_name, att_name)
    else:
        records, rejected_records, completed_ids = [], [], set()

    budget_state = _init_budget_state(cfg)
    new_slots = 0
    for idx, spec in enumerate(specs):
        if budget_state["halted_safely"]:
            break
        if max_new_slots is not None and new_slots >= max_new_slots:
            break
        sample_id = spec["natural_item_id"]
        if sample_id in completed_ids:
            continue

        src: SourceQARecord = spec["source_record"]
        m_spec = spec["respondent_model_spec"]
        client = respondent_clients.get(m_spec["model_id"])
        if client is None:
            raise KeyError(f"No LLMClient supplied for natural respondent model {m_spec['model_id']!r}")
        prompt_context, ev_meta = _resolve_prompt_context(src, cfg)
        sys_msg, usr_msg = render_natural_qa(
            nat_tpl,
            domain=src.domain,
            question=src.question,
            has_context=src.has_context,
            prompt_context=prompt_context,
        )
        max_out = int(m_spec.get("max_tokens", 512))
        if not _reserve_budget_for_call(
            budget_state,
            client=client,
            system_prompt=sys_msg,
            user_prompt=usr_msg,
            max_output_tokens=max_out,
            sample_id=sample_id,
        ):
            break
        try:
            resp = client.generate(
                system_prompt=sys_msg,
                user_prompt=usr_msg,
                temperature=float(m_spec.get("temperature", 0.7)),
                top_p=float(m_spec.get("top_p", 0.9)),
                max_tokens=max_out,
                seed=seed + idx,
            )
            _record_actual_spend(budget_state, resp.cost_usd)
        except BackendHTTPRequestError as g_exc:
            logger.warning("Respondent model %s failed for %s: %s", m_spec["model_id"], sample_id, g_exc)
            continue

        # Run blind cross-judge solely for diagnostic comparison against future human gold labels
        auto_tier = None
        j_sys, j_usr = render_cross_judge(
            judge_tpl,
            domain=src.domain,
            question=src.question,
            has_context=src.has_context,
            prompt_context=prompt_context,
            reference_answer=src.original_answer,
            candidate_response=resp.raw_text.strip(),
        )
        if _reserve_budget_for_call(
            budget_state,
            client=judge_client,
            system_prompt=j_sys,
            user_prompt=j_usr,
            max_output_tokens=int(judge_dec.get("max_tokens", 512)),
            sample_id=f"{sample_id}[judge]",
        ):
            try:
                auto_tier, j_resp, _ = audit_with_cross_judge(
                    judge_client=judge_client,
                    judge_template=judge_tpl,
                    judge_decoding=judge_dec,
                    domain=src.domain,
                    question=src.question,
                    has_context=src.has_context,
                    prompt_context=prompt_context,
                    reference_answer=src.original_answer,
                    candidate_response=resp.raw_text.strip(),
                    intended_bits="00000",
                    seed=seed + idx,
                )
                _record_actual_spend(budget_state, j_resp.cost_usd)
            except BackendHTTPRequestError as j_exc:
                logger.warning("Natural judge call failed for %s: %s", sample_id, j_exc)
                auto_tier = None

        rec = MultiLabelRecord(
            sample_id=sample_id,
            source_dataset=src.source_dataset,
            source_subset=src.source_subset,
            source_record_id=src.source_record_id,
            source_document_id=src.source_document_id,
            source_group_id=src.source_group_id,
            domain=src.domain,
            data_origin="natural",
            has_context=src.has_context,
            question=src.question,
            context=src.context,
            original_answer=src.original_answer,
            generated_answer=resp.raw_text.strip(),
            generation_method="natural_llm_response",
            generation_strategy="natural_qa",
            generation_model=m_spec["model_id"],
            prompt_version=cfg["prompts"]["version"],
            generation_parameters={
                "question_bundle_id": spec["question_bundle_id"],
                "evidence_window_used": ev_meta,
                "cost_usd": resp.cost_usd,
                "latency_ms": resp.latency_ms,
            },
            intended_labels=None,
            auto_detected_labels=auto_tier,
            adjudicated_gold_labels=None,
            label_vector=None,
            validation_status="pending",
            attempted_count=1,
            licensing=src.licensing,
            split="test_natural",
        )
        records.append(rec)
        completed_ids.add(sample_id)
        new_slots += 1
        if checkpoint_every > 0 and new_slots % checkpoint_every == 0:
            write_jsonl(out_dir / acc_name, records)

    write_jsonl(out_dir / acc_name, records)
    export_blinded_annotation_packet(
        records,
        out_dir / "test_natural_blinded_annotation_packet.jsonl",
        out_dir / "test_natural_blind_key_mapping.jsonl",
        seed=seed + 77,
    )
    summary = {
        "sampling_summary": sample_summary,
        "generated_count": len(records),
        "budget_guard": budget_state,
        "resume_mode": resume,
        "new_slots_processed_this_run": new_slots,
        "config_sha256": sha256_json(cfg),
    }
    write_json(out_dir / "natural_generation_summary.json", summary)
    return summary


def _serialize_spec_row(spec: dict[str, Any]) -> dict[str, Any]:
    """Serialize a deterministic sampler specification dict into a flat JSONL row."""
    src: SourceQARecord = spec["source_record"]
    row: dict[str, Any] = {
        "source_record_id": src.source_record_id,
        "source_group_id": src.source_group_id,
        "source_dataset": src.source_dataset,
        "source_subset": src.source_subset,
        "domain": src.domain,
        "has_context": src.has_context,
    }
    for key in (
        "base_item_id",
        "natural_item_id",
        "split",
        "strategy_name",
        "is_clean",
        "intended_bits",
        "cardinality_tier",
        "in_order_sensitivity_subset",
        "step_orders",
        "respondent_model_spec",
    ):
        if key in spec:
            row[key] = spec[key]
    return row


def export_full_generation_plans(
    cfg: dict[str, Any],
    *,
    strategy_name: str = "strategy_a",
    total_target_override: int | None = None,
    output_dir: str | Path | None = None,
    source_pools_override: dict[str, list[SourceQARecord]] | None = None,
) -> dict[str, Any]:
    """Export deterministic full synthetic generation plans across train, val, and test_synthetic."""
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_project_path(cfg["paths"].get("natural_output_dir") or cfg["paths"]["full_output_dir"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    total_planned_items = 0
    for split_name, pool_key in (("train", "train"), ("val", "val"), ("test_synthetic", "test_synthetic_pool")):
        if source_pools_override is not None and split_name in source_pools_override:
            source_records = source_pools_override[split_name]
        else:
            raw_split = read_jsonl(resolve_project_path(cfg["paths"]["source_splits"][pool_key]))
            source_records = [SourceQARecord(**d) for d in raw_split]
        specs, split_summary = sample_full_generation_spec(
            source_records,
            cfg,
            split_name=split_name,
            total_target_override=total_target_override,
            strategy_name=strategy_name,
        )
        rows = [_serialize_spec_row(s) for s in specs]
        write_jsonl(out_dir / f"plan_{split_name}.jsonl", rows)
        split_summaries[split_name] = {
            **split_summary,
            "exported_item_count": len(rows),
        }
        total_planned_items += len(rows)

    aggregate = {
        "strategy_name": strategy_name,
        "total_planned_items": total_planned_items,
        "splits": split_summaries,
        "config_sha256": sha256_json(cfg),
    }
    write_json(out_dir / "full_sampling_summary.json", aggregate)
    return aggregate


def export_natural_generation_plan(
    cfg: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    source_records_override: list[SourceQARecord] | None = None,
) -> dict[str, Any]:
    """Export deterministic held-out test_natural generation plan across all configured respondent models."""
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_project_path(cfg["paths"].get("natural_output_dir", cfg["paths"]["full_output_dir"]))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if source_records_override is not None:
        source_records = source_records_override
    else:
        raw_split = read_jsonl(resolve_project_path(cfg["paths"]["source_splits"]["test_natural_pool"]))
        source_records = [SourceQARecord(**d) for d in raw_split]

    specs, summary = sample_natural_test_spec(source_records, cfg)
    rows = [_serialize_spec_row(s) for s in specs]
    write_jsonl(out_dir / "test_natural_plan.jsonl", rows)
    full_summary = {
        **summary,
        "exported_item_count": len(rows),
        "config_sha256": sha256_json(cfg),
    }
    write_json(out_dir / "natural_sampling_summary.json", full_summary)
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 generation backend verifier and generation runner.")
    parser.add_argument("--config", default="configs/generation.yaml", help="Path to generation config YAML.")
    parser.add_argument(
        "--check-backend",
        action="store_true",
        help="Verify generator and cross-judge env vars, pricing, and /models availability without generating.",
    )
    parser.add_argument(
        "--verify-live-smoke",
        action="store_true",
        help="Verify live /models endpoints + one structured-JSON probe per backend (never runs Pilot E0).",
    )
    parser.add_argument(
        "--export-pilot-plan",
        action="store_true",
        help="Sample Pilot E0 paired/clean/order-sensitivity spec from source_train.jsonl without calling LLMs.",
    )
    parser.add_argument(
        "--export-full-plan",
        action="store_true",
        help="Sample deterministic train/val/test_synthetic full generation plans without calling LLMs.",
    )
    parser.add_argument(
        "--export-natural-plan",
        action="store_true",
        help="Sample deterministic test_natural multi-model generation plan without calling LLMs.",
    )
    parser.add_argument(
        "--execute-pilot",
        action="store_true",
        help="Run Pilot E0 live after passing strict backend verification.",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Run split-locked full synthetic generation after Pilot E0 sign-off.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test_synthetic"],
        help="Target split for --run-full.",
    )
    parser.add_argument(
        "--strategy",
        default="strategy_b",
        choices=["strategy_a", "strategy_b", "strategy_c"],
        help="Winner perturbation strategy for --run-full / --export-full-plan (Strategy B selected by Pilot E0).",
    )
    parser.add_argument(
        "--total-target-override",
        type=int,
        default=None,
        help="Optional smaller split sample count for smoke/staged runs.",
    )
    parser.add_argument(
        "--run-natural",
        action="store_true",
        help="Generate natural multi-model responses on held-out test_natural_pool.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted pilot/full/natural generation run from existing JSONL outputs.",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Maximum new generation slots to process in this session (free-tier pacing guard).",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.check_backend:
        status = verify_generation_and_judge_backends(cfg, check_live=True)
        print(status)
    if args.verify_live_smoke:
        report_path = resolve_project_path(cfg["paths"]["reports_dir"]) / "backend_live_verification.json"
        report = probe_live_backends(cfg, write_report_path=report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    exported_summaries: dict[str, Any] = {}
    if args.export_pilot_plan:
        raw_split = read_jsonl(resolve_project_path(cfg["paths"]["source_splits"]["train"]))
        source_records = [SourceQARecord(**d) for d in raw_split]
        paired_specs, clean_specs, summary = sample_pilot_e0_spec(source_records, cfg)
        out_dir = resolve_project_path(cfg["paths"]["pilot_output_dir"])
        write_jsonl(out_dir / "pilot_e0_paired_plan.jsonl", [_serialize_spec_row(s) for s in paired_specs])
        write_jsonl(out_dir / "pilot_e0_clean_plan.jsonl", [_serialize_spec_row(s) for s in clean_specs])
        write_json(out_dir / "pilot_e0_sampling_summary.json", summary)
        exported_summaries["pilot_e0"] = summary
        logger.info("Exported Pilot E0 deterministic plan: %s", summary)
    if args.export_full_plan:
        full_summary = export_full_generation_plans(
            cfg,
            strategy_name=args.strategy,
            total_target_override=args.total_target_override,
        )
        exported_summaries["full_synthetic"] = full_summary
        logger.info("Exported Full synthetic deterministic plan: %s", full_summary["total_planned_items"])
    if args.export_natural_plan:
        nat_summary = export_natural_generation_plan(cfg)
        exported_summaries["test_natural"] = nat_summary
        logger.info("Exported Natural test deterministic plan: %s", nat_summary["exported_item_count"])
    if exported_summaries:
        reports_dir = resolve_project_path(cfg["paths"]["reports_dir"])
        write_json(reports_dir / "phase2_sampling_plans_report.json", exported_summaries)
    if args.execute_pilot:
        run_pilot_e0(
            args.config,
            execute_live=True,
            resume=args.resume,
            max_new_slots=args.max_slots,
        )
    if args.run_full:
        run_full_generation(
            args.config,
            split_name=args.split,
            strategy_name=args.strategy,
            execute_live=True,
            total_target_override=args.total_target_override,
            resume=args.resume,
            max_new_slots=args.max_slots,
        )
    if args.run_natural:
        run_natural_generation(
            args.config,
            execute_live=True,
            resume=args.resume,
            max_new_slots=args.max_slots,
        )


if __name__ == "__main__":
    main()
