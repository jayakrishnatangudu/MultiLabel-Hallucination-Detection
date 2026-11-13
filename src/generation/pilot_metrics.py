"""Comparative metrics, IAA (Cohen's/Fleiss' kappa), spillover, latency, and cost reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.data.schema import GenerationAttemptRecord, HALLUCINATION_LABELS, MultiLabelRecord


def cohens_kappa(y_a: list[int], y_b: list[int]) -> float | None:
    """Compute binary Cohen's kappa between two aligned rater vectors."""
    if len(y_a) != len(y_b) or not y_a:
        return None
    n = len(y_a)
    n11 = sum(1 for a, b in zip(y_a, y_b) if a == 1 and b == 1)
    n00 = sum(1 for a, b in zip(y_a, y_b) if a == 0 and b == 0)
    n10 = sum(1 for a, b in zip(y_a, y_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(y_a, y_b) if a == 0 and b == 1)
    p_obs = (n11 + n00) / n
    p_a1 = (n11 + n10) / n
    p_b1 = (n11 + n01) / n
    p_exp = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
    if abs(1.0 - p_exp) < 1e-12:
        return 1.0 if abs(p_obs - 1.0) < 1e-12 else 0.0
    return round((p_obs - p_exp) / (1.0 - p_exp), 4)


def fleiss_kappa(ratings_matrix: list[list[int]]) -> float | None:
    """
    Compute binary Fleiss' kappa across M raters per item.

    Each element of ratings_matrix is a list of 0/1 ratings from R raters (R >= 2 constant).
    """
    if not ratings_matrix:
        return None
    n_items = len(ratings_matrix)
    n_raters = len(ratings_matrix[0])
    if n_raters < 2 or any(len(row) != n_raters for row in ratings_matrix):
        return None

    p_i_list: list[float] = []
    total_ones = 0
    for row in ratings_matrix:
        c1 = sum(int(x) for x in row)
        c0 = n_raters - c1
        total_ones += c1
        p_i = (c1 * (c1 - 1) + c0 * (c0 - 1)) / (n_raters * (n_raters - 1))
        p_i_list.append(p_i)

    p_bar = sum(p_i_list) / n_items
    p1 = total_ones / (n_items * n_raters)
    p0 = 1.0 - p1
    p_e = p1 * p1 + p0 * p0
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else 0.0
    return round((p_bar - p_e) / (1.0 - p_e), 4)


def compute_inter_annotator_agreement(records: list[MultiLabelRecord]) -> dict[str, Any]:
    """Compute per-label and macro pairwise Cohen's kappa and Fleiss' kappa across blinded raters."""
    multi_rated = [r for r in records if len(r.human_observed_labels) >= 2]
    if not multi_rated:
        return {"annotated_multi_rater_items": 0, "per_label": {}}

    per_label: dict[str, Any] = {}
    cohen_vals: list[float] = []
    fleiss_vals: list[float] = []

    for bit_idx, lbl in enumerate(HALLUCINATION_LABELS):
        y_first: list[int] = []
        y_second: list[int] = []
        fleiss_rows: list[list[int]] = []
        rater_counts: set[int] = set()

        for rec in multi_rated:
            sorted_raters = [rec.human_observed_labels[k][bit_idx] for k in sorted(rec.human_observed_labels)]
            y_first.append(sorted_raters[0])
            y_second.append(sorted_raters[1])
            fleiss_rows.append(sorted_raters)
            rater_counts.add(len(sorted_raters))

        exact_pct = round(sum(1 for a, b in zip(y_first, y_second) if a == b) / len(y_first), 4)
        ck = cohens_kappa(y_first, y_second)
        fk = fleiss_kappa(fleiss_rows) if len(rater_counts) == 1 else None
        if ck is not None:
            cohen_vals.append(ck)
        if fk is not None:
            fleiss_vals.append(fk)

        per_label[lbl] = {
            "n_items": len(multi_rated),
            "first_pair_agreement": exact_pct,
            "first_pair_cohens_kappa": ck,
            "fleiss_kappa": fk,
        }

    return {
        "annotated_multi_rater_items": len(multi_rated),
        "macro_cohens_kappa": round(sum(cohen_vals) / len(cohen_vals), 4) if cohen_vals else None,
        "macro_fleiss_kappa": round(sum(fleiss_vals) / len(fleiss_vals), 4) if fleiss_vals else None,
        "per_label": per_label,
    }


def _evaluate_subset_fidelity(
    records: list[MultiLabelRecord],
    *,
    reference_tier: str,
) -> dict[str, Any]:
    """
    Evaluate target-label recall, unintended spillover, and exact match against EITHER
    adjudicated_gold_labels ('human_gold') OR diagnostic auto_detected_labels ('cross_judge').
    """
    per_label_tp: Counter[str] = Counter()
    per_label_intended_pos: Counter[str] = Counter()
    per_label_fp: Counter[str] = Counter()
    per_label_intended_neg: Counter[str] = Counter()
    exact_matches = 0
    evaluated = 0

    for rec in records:
        if rec.intended_labels is None:
            continue
        if reference_tier == "human_gold":
            obs = rec.adjudicated_gold_labels
        elif reference_tier == "cross_judge":
            obs = rec.auto_detected_labels.predicted_labels if rec.auto_detected_labels else None
        else:
            raise ValueError(f"Unsupported reference tier: {reference_tier}")
        if obs is None:
            continue

        evaluated += 1
        tgt = rec.intended_labels.to_vector()
        if obs == tgt:
            exact_matches += 1
        for lbl, t_bit, o_bit in zip(HALLUCINATION_LABELS, tgt, obs):
            if t_bit == 1:
                per_label_intended_pos[lbl] += 1
                if o_bit == 1:
                    per_label_tp[lbl] += 1
            else:
                per_label_intended_neg[lbl] += 1
                if o_bit == 1:
                    per_label_fp[lbl] += 1

    per_label_stats: dict[str, Any] = {}
    recalls: list[float] = []
    spillovers: list[float] = []
    for lbl in HALLUCINATION_LABELS:
        pos_n = per_label_intended_pos[lbl]
        neg_n = per_label_intended_neg[lbl]
        rec_rate = round(per_label_tp[lbl] / pos_n, 4) if pos_n else None
        sp_rate = round(per_label_fp[lbl] / neg_n, 4) if neg_n else None
        if rec_rate is not None:
            recalls.append(rec_rate)
        if sp_rate is not None:
            spillovers.append(sp_rate)
        per_label_stats[lbl] = {
            "intended_positives": pos_n,
            "target_recall": rec_rate,
            "intended_negatives": neg_n,
            "unintended_spillover_rate": sp_rate,
        }

    total_pos = sum(per_label_intended_pos.values())
    total_neg = sum(per_label_intended_neg.values())
    return {
        "reference_tier": reference_tier,
        "evaluated_records": evaluated,
        "exact_intended_vector_match_rate": round(exact_matches / evaluated, 4) if evaluated else None,
        "micro_target_recall": round(sum(per_label_tp.values()) / total_pos, 4) if total_pos else None,
        "macro_target_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "micro_unintended_spillover": round(sum(per_label_fp.values()) / total_neg, 4) if total_neg else None,
        "macro_unintended_spillover": round(sum(spillovers) / len(spillovers), 4) if spillovers else None,
        "per_label": per_label_stats,
    }


def _summarize_group_metrics(
    group_records: list[MultiLabelRecord],
    group_rejected: list[MultiLabelRecord],
    group_attempts: list[GenerationAttemptRecord],
) -> dict[str, Any]:
    attempted_slots = len(group_records) + len(group_rejected)
    accepted_slots = len(group_records)
    rejected_slots = len(group_rejected)
    total_api_attempts = len(group_attempts)
    failed_attempts = sum(1 for a in group_attempts if not a.accepted)

    # Initial attempt (attempt_number == 1) failure rate across slots
    att1 = [a for a in group_attempts if a.attempt_number == 1]
    att1_failed = sum(1 for a in att1 if not a.accepted)
    reason_counts = Counter(
        reason for att in group_attempts if not att.accepted for reason in att.rejection_reasons
    )

    coherences = [
        rec.auto_detected_labels.coherence_score_1_to_5
        for rec in group_records
        if rec.auto_detected_labels and rec.auto_detected_labels.coherence_score_1_to_5 is not None
    ]
    total_gen_cost = round(sum(a.cost_usd for a in group_attempts), 8)
    total_judge_cost = round(
        sum(float(rec.generation_parameters.get("judge_cost_usd", 0.0)) for rec in group_records), 8
    )
    total_lat_ms = round(sum(a.latency_ms for a in group_attempts), 2)

    return {
        "attempted_slots": attempted_slots,
        "accepted_slots": accepted_slots,
        "rejected_slots": rejected_slots,
        "post_retry_invalid_sample_rate": round(rejected_slots / attempted_slots, 4) if attempted_slots else 0.0,
        "total_llm_retry_attempts": total_api_attempts,
        "failed_retry_attempts": failed_attempts,
        "initial_attempt_invalid_rate": round(att1_failed / len(att1), 4) if att1 else 0.0,
        "rejection_reasons_breakdown": dict(reason_counts),
        "mean_coherence_score": round(sum(coherences) / len(coherences), 4) if coherences else None,
        "total_generation_cost_usd": total_gen_cost,
        "total_judge_cost_usd": total_judge_cost,
        "total_cost_usd": round(total_gen_cost + total_judge_cost, 8),
        "mean_cost_per_accepted_usd": round((total_gen_cost + total_judge_cost) / accepted_slots, 8)
        if accepted_slots
        else None,
        "total_generation_latency_ms": total_lat_ms,
        "mean_latency_per_accepted_ms": round(total_lat_ms / accepted_slots, 2) if accepted_slots else None,
        "total_input_tokens": sum(a.input_tokens for a in group_attempts),
        "total_output_tokens": sum(a.output_tokens for a in group_attempts),
        "diagnostic_cross_judge_fidelity": _evaluate_subset_fidelity(
            group_records, reference_tier="cross_judge"
        ),
        "human_gold_fidelity": _evaluate_subset_fidelity(group_records, reference_tier="human_gold"),
    }


def compute_pilot_e0_report(
    *,
    accepted_records: list[MultiLabelRecord],
    rejected_records: list[MultiLabelRecord],
    attempt_records: list[GenerationAttemptRecord],
) -> dict[str, Any]:
    """
    Compute complete Pilot E0 metrics: attempted/accepted counts, paired A/B/C comparison,
    Strategy A E0-Order (forward/reverse/randomized) comparison, IAA, coherence, latency, cost.
    """
    clean_records = [r for r in accepted_records if r.generation_strategy == "clean_control"]
    perturbed_records = [r for r in accepted_records if r.generation_strategy != "clean_control"]

    # Main paired comparison excludes extra E0-Order reverse/randomized Strategy A variants
    def _is_main_paired(rec_or_att: Any) -> bool:
        if isinstance(rec_or_att, GenerationAttemptRecord):
            return rec_or_att.generation_strategy in {"strategy_b", "strategy_c"} or (
                rec_or_att.generation_strategy == "strategy_a" and rec_or_att.step_order_name == "forward"
            )
        order_n = rec_or_att.generation_parameters.get("step_order_name")
        return rec_or_att.generation_strategy in {"strategy_b", "strategy_c"} or (
            rec_or_att.generation_strategy == "strategy_a" and order_n == "forward"
        )

    by_strategy: dict[str, Any] = {}
    for strat in ("strategy_a", "strategy_b", "strategy_c"):
        s_acc = [r for r in perturbed_records if r.generation_strategy == strat and _is_main_paired(r)]
        s_rej = [r for r in rejected_records if r.generation_strategy == strat and _is_main_paired(r)]
        s_att = [a for a in attempt_records if a.generation_strategy == strat and _is_main_paired(a)]
        by_strategy[strat] = _summarize_group_metrics(s_acc, s_rej, s_att)

    # Paired exact item alignment check across Strategies A, B, C
    paired_base_ids: dict[str, set[str]] = defaultdict(set)
    for r in perturbed_records:
        if _is_main_paired(r):
            bid = str(r.generation_parameters.get("base_item_id", ""))
            paired_base_ids[str(r.generation_strategy)].add(bid)

    # Strategy A Order-Sensitivity comparison across the 18-item subset (forward vs reverse vs randomized)
    os_base_ids = {
        str(r.generation_parameters.get("base_item_id"))
        for r in perturbed_records + rejected_records
        if r.generation_parameters.get("in_order_sensitivity_subset")
    }
    order_sensitivity: dict[str, Any] = {}
    for order_name in ("forward", "reverse", "randomized"):
        o_acc = [
            r
            for r in perturbed_records
            if r.generation_strategy == "strategy_a"
            and r.generation_parameters.get("step_order_name") == order_name
            and str(r.generation_parameters.get("base_item_id")) in os_base_ids
        ]
        o_rej = [
            r
            for r in rejected_records
            if r.generation_strategy == "strategy_a"
            and r.generation_parameters.get("step_order_name") == order_name
            and str(r.generation_parameters.get("base_item_id")) in os_base_ids
        ]
        o_att = [
            a
            for a in attempt_records
            if a.generation_strategy == "strategy_a"
            and a.step_order_name == order_name
            and a.base_item_id in os_base_ids
        ]
        order_sensitivity[order_name] = _summarize_group_metrics(o_acc, o_rej, o_att)

    return {
        "overall_counts": {
            "clean_control_count": len(clean_records),
            "perturbed_attempted_slots": len(perturbed_records) + len(rejected_records),
            "perturbed_accepted_slots": len(perturbed_records),
            "perturbed_rejected_slots": len(rejected_records),
            "total_llm_retry_attempts": len(attempt_records),
        },
        "paired_strategy_comparison": by_strategy,
        "paired_alignment_base_counts": {k: len(v) for k, v in paired_base_ids.items()},
        "strategy_a_order_sensitivity": {
            "subset_size": len(os_base_ids),
            "orders": order_sensitivity,
        },
        "clean_control_diagnostics": {
            "cross_judge_fidelity": _evaluate_subset_fidelity(clean_records, reference_tier="cross_judge"),
            "human_gold_fidelity": _evaluate_subset_fidelity(clean_records, reference_tier="human_gold"),
        },
        "inter_annotator_agreement": compute_inter_annotator_agreement(accepted_records),
    }
