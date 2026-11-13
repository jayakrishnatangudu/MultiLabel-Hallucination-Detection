"""Stratified sampling, CUAD evidence windowing, clause segmentation, and step ordering."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

import numpy as np

from src.data.schema import HALLUCINATION_LABELS, LabelDictionary, SourceQARecord

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SEMI_SPLIT_RE = re.compile(r"\s*;\s*")


def binary_str_to_label_dict(bits: str) -> LabelDictionary:
    """Convert a 5-character binary string ('FI FF II LI CI') into a LabelDictionary."""
    if len(bits) != 5 or any(ch not in {"0", "1"} for ch in bits):
        raise ValueError(f"Expected 5-bit label string, got: {bits!r}")
    return LabelDictionary(
        factual_inconsistency=int(bits[0]),  # type: ignore[arg-type]
        factual_fabrication=int(bits[1]),  # type: ignore[arg-type]
        instruction_inconsistency=int(bits[2]),  # type: ignore[arg-type]
        logical_inconsistency=int(bits[3]),  # type: ignore[arg-type]
        context_inconsistency=int(bits[4]),  # type: ignore[arg-type]
    )


def active_labels_from_bits(bits: str) -> list[str]:
    """Return ordered list of active hallucination label keys from a 5-bit string."""
    if len(bits) != 5:
        raise ValueError(f"Expected 5-bit label string, got: {bits!r}")
    return [label for bit, label in zip(bits, HALLUCINATION_LABELS) if bit == "1"]


def cardinality_tier(bits: str) -> str:
    """Map binary label vector string to cardinality tier name (G0..G5)."""
    weight = sum(int(ch) for ch in bits)
    return f"G{weight}"


def split_response_clauses(text: str) -> list[str]:
    """
    Split an answer into coherent sentence/clause spans for Strategy C editing.

    Prefers sentence boundaries; falls back to semicolon clause boundaries for long
    semicolon-concatenated extracts (common in CUAD-QA).
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []

    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    if len(raw_parts) <= 1 and ";" in cleaned:
        semi_parts = [p.strip() for p in _SEMI_SPLIT_RE.split(cleaned) if p.strip()]
        if len(semi_parts) > 1:
            raw_parts = semi_parts

    # Merge ultra-short (<3-word) fragments into adjacent clauses to prevent orphan spans
    merged: list[str] = []
    for part in raw_parts:
        if merged and len(part.split()) < 3:
            merged[-1] = f"{merged[-1]} {part}".strip()
        elif merged and len(merged[-1].split()) < 3:
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part)
    return merged


def extract_evidence_anchored_window(
    context: str | None,
    original_answer: str,
    max_words: int = 1800,
    preamble_words: int = 250,
    window_words_around_span: int = 750,
) -> tuple[str | None, bool]:
    """
    Create a bounded prompt window around ground-truth evidence spans inside long documents.

    Returns (prompt_context, window_was_applied). Never alters the canonical record's full context.
    """
    if context is None:
        return None, False
    words = context.split()
    if len(words) <= max_words:
        return context, False

    lowered_ctx = context.lower()
    char_to_word = []
    pos = 0
    for w_idx, w in enumerate(words):
        found_at = context.find(w, pos)
        if found_at == -1:
            found_at = pos
        char_to_word.append((found_at, w_idx))
        pos = found_at + len(w)

    def _char_idx_to_word_idx(c_idx: int) -> int:
        best = 0
        for c_pos, w_idx in char_to_word:
            if c_pos <= c_idx:
                best = w_idx
            else:
                break
        return best

    span_candidates = [s.strip() for s in original_answer.split(";") if len(s.strip()) >= 8]
    if not span_candidates:
        span_candidates = [original_answer.strip()]

    intervals: list[tuple[int, int]] = [(0, min(len(words), preamble_words))]
    half_win = max(100, window_words_around_span // 2)

    for span in span_candidates:
        probe = " ".join(span.split()[:12]).lower()
        idx = lowered_ctx.find(probe)
        if idx == -1 and len(probe) > 25:
            probe = probe[:25]
            idx = lowered_ctx.find(probe)
        if idx != -1:
            w_center = _char_idx_to_word_idx(idx)
            w_span_len = max(20, len(span.split()))
            start_w = max(0, w_center - half_win)
            end_w = min(len(words), w_center + w_span_len + half_win)
            intervals.append((start_w, end_w))

    # Merge overlapping word intervals deterministically
    intervals.sort()
    merged_intervals: list[tuple[int, int]] = []
    for s, e in intervals:
        if not merged_intervals or s > merged_intervals[-1][1]:
            merged_intervals.append((s, e))
        else:
            prev_s, prev_e = merged_intervals[-1]
            merged_intervals[-1] = (prev_s, max(prev_e, e))

    chunks: list[str] = []
    budget_left = max_words
    for idx_i, (s, e) in enumerate(merged_intervals):
        if budget_left <= 0:
            break
        take_end = min(e, s + budget_left)
        if idx_i > 0:
            chunks.append("[...]")
        chunks.append(" ".join(words[s:take_end]))
        budget_left -= take_end - s

    return "\n".join(chunks), True


def compute_strategy_a_orders(
    active_labels: list[str],
    taxonomy_cfg: dict[str, Any],
    seed: int,
    item_idx: int,
) -> dict[str, list[str]]:
    """Compute forward, reverse, and deterministic randomized step sequences for active labels."""
    active_set = set(active_labels)
    fwd_base = list(
        taxonomy_cfg.get(
            "sequential_injection_order",
            [
                "instruction_inconsistency",
                "factual_inconsistency",
                "context_inconsistency",
                "factual_fabrication",
                "logical_inconsistency",
            ],
        )
    )
    rev_base = list(taxonomy_cfg.get("reverse_injection_order", list(reversed(fwd_base))))
    forward = [lbl for lbl in fwd_base if lbl in active_set]
    reverse = [lbl for lbl in rev_base if lbl in active_set]

    rng = np.random.default_rng(seed + 1009 + item_idx * 17)
    randomized = list(forward)
    if len(randomized) >= 3:
        for _ in range(12):
            rng.shuffle(randomized)
            if randomized != forward and randomized != reverse:
                break
    elif len(randomized) == 2:
        randomized = list(reverse)

    return {"forward": forward, "reverse": reverse, "randomized": randomized}


def _is_eligible_for_bits(record: SourceQARecord, bits: str, cfg: dict[str, Any]) -> bool:
    """Check whether a source record satisfies context and length/clause constraints for a label vector."""
    if bits[4] == "1" and not record.has_context:
        return False

    tier = cardinality_tier(bits)
    constraints = cfg["eligibility_constraints"].get(tier, {})
    if constraints.get("require_context", False) and not record.has_context:
        return False

    min_words = int(constraints.get("min_answer_words", 3))
    min_clauses = int(constraints.get("min_clauses", 1))
    ans_words = int(record.metadata.get("answer_words", len(record.original_answer.split())))
    n_clauses = len(split_response_clauses(record.original_answer))

    if tier == "G3":
        # Allow G3 when either clause structure OR >=40 answer words holds
        return (n_clauses >= min_clauses or ans_words >= min_words) and ans_words >= 20
    if tier in {"G4", "G5"}:
        return n_clauses >= min_clauses or ans_words >= min_words
    return ans_words >= min_words and (n_clauses >= min_clauses or ans_words >= 25)


def _select_reserved_context_non_ci_slots(
    combo_targets: list[str],
    ctx_non_ci_quotas: dict[str, int],
    ctx_datasets: list[str],
    spare_ctx_total: int,
) -> dict[int, str]:
    """
    Deterministically reserve exact CI=0 context-present slots per tier (e.g., 2 G1 + 2 G2)
    across distinct bit-vectors and balanced across context datasets.
    """
    clean_quotas = {str(k): int(v) for k, v in ctx_non_ci_quotas.items() if int(v) > 0}
    if not clean_quotas:
        return {}
    total_reserved = sum(clean_quotas.values())
    if total_reserved > spare_ctx_total:
        raise ValueError(
            f"context_non_ci_tier_quotas total ({total_reserved}) exceeds available spare context "
            f"supply after CI=1 demand ({spare_ctx_total})."
        )

    ordered_ctx = sorted(ctx_datasets)
    reserved_pref_ds: dict[int, str] = {}
    for tier_idx, (tier_name, need) in enumerate(sorted(clean_quotas.items())):
        by_bits: dict[str, list[int]] = defaultdict(list)
        for slot_idx, bits in enumerate(combo_targets):
            if cardinality_tier(bits) == tier_name and bits[4] == "0":
                by_bits[bits].append(slot_idx)
        avail_count = sum(len(v) for v in by_bits.values())
        if avail_count < need:
            raise ValueError(
                f"Insufficient CI=0 targets in {tier_name} ({avail_count}) for "
                f"context_non_ci_tier_quotas[{tier_name!r}]={need}."
            )

        # Prioritize combinations with largest counts (then descending bit order) round-robin for diversity
        ranked_bits = sorted(
            by_bits.keys(),
            key=lambda b: (-len(by_bits[b]), tuple(-int(c) for c in b)),
        )
        picked_indices: list[int] = []
        round_offset = 0
        while len(picked_indices) < need:
            progressed = False
            for bits_key in ranked_bits:
                if round_offset < len(by_bits[bits_key]):
                    picked_indices.append(by_bits[bits_key][round_offset])
                    progressed = True
                    if len(picked_indices) >= need:
                        break
            if not progressed:
                break
            round_offset += 1

        for j, slot_idx in enumerate(picked_indices):
            pref_ds = ordered_ctx[(tier_idx + j) % len(ordered_ctx)] if ordered_ctx else ""
            reserved_pref_ds[slot_idx] = pref_ds

    return reserved_pref_ds


def sample_pilot_e0_spec(
    source_records: list[SourceQARecord],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Construct the paired 80-item + 20-clean-control Pilot E0 sampling specification.

    Enforces strict source_train split isolation, zero source_group_id overlap between
    perturbed and clean controls, domain/context balance, and paired identity across A/B/C.
    """
    pilot_cfg = cfg["pilot_e0"]
    required_pool = str(pilot_cfg.get("source_split_pool", "train"))
    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    pool: list[SourceQARecord] = []
    for rec in source_records:
        if rec.split != required_pool:
            raise ValueError(
                f"Pilot source pool violation: record {rec.record_id} has split={rec.split!r}, "
                f"expected {required_pool!r}."
            )
        pool.append(rec)

    ds_quotas = pilot_cfg["domain_dataset_quotas_base_items"]
    combo_quotas: dict[str, int] = dict(pilot_cfg["combination_quotas_paired_base"])

    # Expand exact combination targets (total 80) sorted from hardest (G5 -> G1)
    combo_targets: list[str] = []
    for bits, cnt in sorted(
        combo_quotas.items(),
        key=lambda kv: (-sum(int(c) for c in kv[0]), kv[0]),
    ):
        combo_targets.extend([bits] * int(cnt))

    # Build dataset slot assignments while guaranteeing CI=1 targets only land on context datasets
    ds_initial_perturbed = {ds: int(vals["perturbed"]) for ds, vals in ds_quotas.items()}
    ds_remaining_perturbed = dict(ds_initial_perturbed)
    ctx_datasets = [ds for ds, pol in cfg["source_policies"].items() if pol["has_context"]]
    no_ctx_datasets = [ds for ds, pol in cfg["source_policies"].items() if not pol["has_context"]]

    # Track per-source-dataset pools shuffled deterministically (respecting PubMedQA pqa_labeled quota)
    by_dataset: dict[str, list[SourceQARecord]] = defaultdict(list)
    for rec in sorted(pool, key=lambda r: r.record_id):
        by_dataset[rec.source_dataset].append(rec)
    for ds, recs in by_dataset.items():
        idxs = list(range(len(recs)))
        rng.shuffle(idxs)
        if ds == "pubmedqa":
            labeled = [recs[i] for i in idxs if recs[i].source_subset == "pqa_labeled"]
            unlabeled = [recs[i] for i in idxs if recs[i].source_subset != "pqa_labeled"]
            interleaved: list[SourceQARecord] = []
            li, ui = 0, 0
            while li < len(labeled) or ui < len(unlabeled):
                if li < len(labeled):
                    interleaved.append(labeled[li])
                    li += 1
                for _ in range(3):
                    if ui < len(unlabeled):
                        interleaved.append(unlabeled[ui])
                        ui += 1
            by_dataset[ds] = interleaved
        else:
            by_dataset[ds] = [recs[i] for i in idxs]

    used_group_ids: set[str] = set()
    used_record_ids: set[str] = set()

    def _pick_record(
        bits: str,
        candidate_datasets: list[str],
        *,
        preferred_first: str | None = None,
        min_reserve_by_ds: dict[str, int] | None = None,
    ) -> SourceQARecord | None:
        # Rotate dataset order by remaining quota FRACTION so smaller datasets stay proportionally represented across G1..G4
        reserves = min_reserve_by_ds or {}
        eligible_ds = [
            d
            for d in candidate_datasets
            if ds_remaining_perturbed.get(d, 0) > (reserves.get(d, 0) if d != preferred_first else 0)
        ]
        ordered_ds = sorted(
            eligible_ds,
            key=lambda d: (-(ds_remaining_perturbed[d] / max(1, ds_initial_perturbed[d])), d),
        )
        if preferred_first and preferred_first in eligible_ds:
            ordered_ds = [preferred_first] + [d for d in ordered_ds if d != preferred_first]
        for ds in ordered_ds:
            for rec in by_dataset[ds]:
                if rec.record_id in used_record_ids or rec.source_group_id in used_group_ids:
                    continue
                if _is_eligible_for_bits(rec, bits, cfg):
                    return rec
        return None

    total_ci1_targets = sum(1 for b in combo_targets if b[4] == "1")
    spare_ctx_total = sum(ds_initial_perturbed.get(d, 0) for d in ctx_datasets) - total_ci1_targets
    reserved_ctx_pref_ds = _select_reserved_context_non_ci_slots(
        combo_targets,
        pilot_cfg.get("context_non_ci_tier_quotas") or {},
        ctx_datasets,
        spare_ctx_total,
    )
    ci1_assigned = 0

    paired_specs: list[dict[str, Any]] = []
    for idx, bits in enumerate(combo_targets):
        rem_ctx_supply = sum(ds_remaining_perturbed.get(d, 0) for d in ctx_datasets)
        rem_ci1_demand = total_ci1_targets - ci1_assigned
        rem_reserved_ahead_by_ds = Counter(
            ds for r_idx, ds in reserved_ctx_pref_ds.items() if r_idx > idx and ds
        )
        rem_reserved_total = sum(1 for r_idx in reserved_ctx_pref_ds if r_idx >= idx)
        unreserved_spare_ctx = rem_ctx_supply - rem_ci1_demand - rem_reserved_total

        preferred_first: str | None = None
        if bits[4] == "1":
            candidates = ctx_datasets
        elif idx in reserved_ctx_pref_ds:
            candidates = ctx_datasets
            preferred_first = reserved_ctx_pref_ds[idx] or None
        elif unreserved_spare_ctx > 0:
            candidates = ctx_datasets + no_ctx_datasets
        else:
            candidates = no_ctx_datasets

        chosen = _pick_record(
            bits,
            candidates,
            preferred_first=preferred_first,
            min_reserve_by_ds=rem_reserved_ahead_by_ds if idx not in reserved_ctx_pref_ds else None,
        )
        if chosen is None and bits[4] == "1":
            chosen = _pick_record(bits, ctx_datasets)
        elif chosen is None and bits[4] != "1" and idx not in reserved_ctx_pref_ds and unreserved_spare_ctx > 0:
            chosen = _pick_record(bits, no_ctx_datasets + ctx_datasets)
        if chosen is None:
            raise RuntimeError(f"Could not find eligible train record for vector {bits} (slot {idx}).")

        if bits[4] == "1":
            ci1_assigned += 1

        used_record_ids.add(chosen.record_id)
        used_group_ids.add(chosen.source_group_id)
        ds_remaining_perturbed[chosen.source_dataset] -= 1

        active_labels = active_labels_from_bits(bits)
        forbidden_labels = [lbl for lbl in HALLUCINATION_LABELS if lbl not in active_labels]
        step_orders = compute_strategy_a_orders(active_labels, cfg["taxonomy"], seed, idx)

        paired_specs.append(
            {
                "base_item_id": f"PILOT_BASE_{idx:03d}",
                "source_record": chosen,
                "intended_bits": bits,
                "cardinality_tier": cardinality_tier(bits),
                "active_labels": active_labels,
                "forbidden_labels": forbidden_labels,
                "step_orders": step_orders,
                "in_order_sensitivity_subset": False,
            }
        )

    assert all(rem == 0 for rem in ds_remaining_perturbed.values()), (
        f"Unfulfilled pilot dataset quotas: {ds_remaining_perturbed}"
    )

    ctx_non_ci_items = [
        s for s in paired_specs if s["source_record"].has_context and s["intended_bits"][4] == "0"
    ]
    ctx_non_ci_by_tier = dict(Counter(s["cardinality_tier"] for s in ctx_non_ci_items))
    configured_ctx_non_ci = {
        str(k): int(v) for k, v in (pilot_cfg.get("context_non_ci_tier_quotas") or {}).items() if int(v) > 0
    }
    if configured_ctx_non_ci:
        assert ctx_non_ci_by_tier == configured_ctx_non_ci, (
            f"Context-present CI=0 tier breakdown {ctx_non_ci_by_tier} != configured {configured_ctx_non_ci}"
        )

    # Select clean controls (20 unmodified references) disjoint by source_group_id
    clean_specs: list[dict[str, Any]] = []
    clean_idx = 0
    for ds, vals in ds_quotas.items():
        need = int(vals["clean"])
        picked = 0
        for rec in by_dataset[ds]:
            if picked >= need:
                break
            if rec.record_id in used_record_ids or rec.source_group_id in used_group_ids:
                continue
            if _is_eligible_for_bits(rec, "00000", cfg):
                used_record_ids.add(rec.record_id)
                used_group_ids.add(rec.source_group_id)
                clean_specs.append(
                    {
                        "base_item_id": f"PILOT_CLEAN_{clean_idx:03d}",
                        "source_record": rec,
                        "intended_bits": "00000",
                        "cardinality_tier": "G0",
                        "active_labels": [],
                        "forbidden_labels": list(HALLUCINATION_LABELS),
                    }
                )
                clean_idx += 1
                picked += 1
        assert picked == need, f"Insufficient clean records for {ds}: {picked}/{need}"

    # Tag the clearly identified Strategy A order-sensitivity subset (10 G2 + 8 G3 = 18 items)
    os_cfg = pilot_cfg.get("order_sensitivity_subset", {})
    if os_cfg.get("enabled", True):
        target_g2 = 10
        target_g3 = 8
        g2_items = [s for s in paired_specs if s["cardinality_tier"] == "G2"]
        g3_items = [s for s in paired_specs if s["cardinality_tier"] == "G3"]

        def _diverse_pick(items: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
            picked: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for it in items:
                by_ds[it["source_record"].source_dataset].append(it)
            ds_cycle = sorted(by_ds.keys())
            while len(picked) < k:
                progressed = False
                for ds_k in ds_cycle:
                    while by_ds[ds_k] and by_ds[ds_k][0]["base_item_id"] in seen_ids:
                        by_ds[ds_k].pop(0)
                    if by_ds[ds_k]:
                        cand = by_ds[ds_k].pop(0)
                        picked.append(cand)
                        seen_ids.add(cand["base_item_id"])
                        progressed = True
                        if len(picked) >= k:
                            break
                if not progressed:
                    break
            return picked

        chosen_os = _diverse_pick(g2_items, target_g2) + _diverse_pick(g3_items, target_g3)
        for item in chosen_os:
            item["in_order_sensitivity_subset"] = True

    # Build complete cross-tabulated balance across G0..G5 x domain x context x dataset
    all_base_items = clean_specs + paired_specs
    tier_order = ("G0", "G1", "G2", "G3", "G4", "G5")
    tier_breakdown: dict[str, Any] = {}
    for tier in tier_order:
        t_items = [s for s in all_base_items if s["cardinality_tier"] == tier]
        tier_breakdown[tier] = {
            "base_items": len(t_items),
            "by_domain": {
                dom: sum(1 for s in t_items if s["source_record"].domain == dom)
                for dom in ("medical", "legal", "finance")
            },
            "by_context": {
                "with_context": sum(1 for s in t_items if s["source_record"].has_context),
                "no_context": sum(1 for s in t_items if not s["source_record"].has_context),
            },
            "by_dataset": dict(Counter(s["source_record"].source_dataset for s in t_items)),
        }

    os_items = [s for s in paired_specs if s["in_order_sensitivity_subset"]]
    os_steps_per_order = sum(len(s["active_labels"]) for s in os_items)
    strategy_a_forward_calls = sum(len(s["active_labels"]) for s in paired_specs)
    strategy_b_calls = len(paired_specs)
    strategy_c_calls = len(paired_specs)
    core_gen_calls = strategy_a_forward_calls + strategy_b_calls + strategy_c_calls
    extra_orders_count = 2 if os_items else 0  # reverse + randomized
    os_extra_outputs = len(os_items) * extra_orders_count
    os_extra_gen_calls = os_steps_per_order * extra_orders_count
    max_retries = int(cfg["backends"]["synthetic_generator"].get("max_retries_per_sample", 3))

    summary = {
        "source_split_pool": required_pool,
        "unique_source_questions": len(all_base_items),
        "unique_group_ids": len(used_group_ids),
        "paired_perturbed_base_count": len(paired_specs),
        "clean_control_count": len(clean_specs),
        "order_sensitivity_subset_count": len(os_items),
        "paired_by_dataset": dict(Counter(s["source_record"].source_dataset for s in paired_specs)),
        "paired_by_domain": dict(Counter(s["source_record"].domain for s in paired_specs)),
        "paired_by_context": dict(Counter(str(s["source_record"].has_context) for s in paired_specs)),
        "paired_by_context_and_ci": {
            "with_context_ci_1": sum(
                1 for s in paired_specs if s["source_record"].has_context and s["intended_bits"][4] == "1"
            ),
            "with_context_ci_0": len(ctx_non_ci_items),
            "with_context_ci_0_by_tier": ctx_non_ci_by_tier,
            "with_context_ci_0_items": [
                {
                    "base_item_id": s["base_item_id"],
                    "source_dataset": s["source_record"].source_dataset,
                    "domain": s["source_record"].domain,
                    "cardinality_tier": s["cardinality_tier"],
                    "intended_bits": s["intended_bits"],
                }
                for s in ctx_non_ci_items
            ],
            "no_context_ci_0": sum(
                1 for s in paired_specs if not s["source_record"].has_context and s["intended_bits"][4] == "0"
            ),
        },
        "paired_by_tier": dict(Counter(s["cardinality_tier"] for s in paired_specs)),
        "clean_by_dataset": dict(Counter(s["source_record"].source_dataset for s in clean_specs)),
        "tier_by_domain_and_context": tier_breakdown,
        "order_sensitivity_subset_breakdown": {
            "subset_items": len(os_items),
            "by_tier": dict(Counter(s["cardinality_tier"] for s in os_items)),
            "by_domain": dict(Counter(s["source_record"].domain for s in os_items)),
            "by_context": {
                "with_context": sum(1 for s in os_items if s["source_record"].has_context),
                "no_context": sum(1 for s in os_items if not s["source_record"].has_context),
            },
            "by_dataset": dict(Counter(s["source_record"].source_dataset for s in os_items)),
        },
        "budget_and_call_accounting": {
            "core_pilot_budget": {
                "unique_source_questions": len(all_base_items),
                "perturbed_strategy_slots": len(paired_specs) * 3,
                "clean_control_slots": len(clean_specs),
                "total_core_evaluation_items": len(paired_specs) * 3 + len(clean_specs),
                "nominal_generator_calls": {
                    "strategy_a_forward_sequential_steps": strategy_a_forward_calls,
                    "strategy_b_joint": strategy_b_calls,
                    "strategy_c_span_editing": strategy_c_calls,
                    "clean_control": 0,
                    "subtotal": core_gen_calls,
                },
                "nominal_cross_judge_calls": len(paired_specs) * 3 + len(clean_specs),
                "total_core_nominal_llm_calls": core_gen_calls + len(paired_specs) * 3 + len(clean_specs),
            },
            "order_sensitivity_addon": {
                "is_additional_to_260": True,
                "extra_unique_source_questions": 0,
                "reused_subset_base_items": len(os_items),
                "additional_comparative_outputs": os_extra_outputs,
                "nominal_generator_calls_reverse_plus_randomized": os_extra_gen_calls,
                "nominal_cross_judge_calls": os_extra_outputs,
                "total_addon_nominal_llm_calls": os_extra_gen_calls + os_extra_outputs,
            },
            "combined_pilot_totals_when_order_addon_enabled": {
                "unique_source_questions": len(all_base_items),
                "total_perturbed_outputs_attempted": len(paired_specs) * 3 + os_extra_outputs,
                "clean_controls": len(clean_specs),
                "total_evaluation_items": len(paired_specs) * 3 + len(clean_specs) + os_extra_outputs,
                "nominal_generator_calls": core_gen_calls + os_extra_gen_calls,
                "worst_case_max_retry_generator_calls": (core_gen_calls + os_extra_gen_calls) * max_retries,
                "nominal_cross_judge_calls": len(paired_specs) * 3 + len(clean_specs) + os_extra_outputs,
                "total_nominal_llm_calls": (
                    core_gen_calls
                    + os_extra_gen_calls
                    + len(paired_specs) * 3
                    + len(clean_specs)
                    + os_extra_outputs
                ),
            },
        },
    }
    return paired_specs, clean_specs, summary


def _allocate_proportional_quotas(weights: dict[str, float], target_total: int) -> dict[str, int]:
    """Allocate integer counts summing to target_total using largest-remainder rounding."""
    if target_total < 0:
        raise ValueError(f"target_total must be non-negative, got {target_total}")
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    total_w = sum(positive.values())
    if target_total == 0 or total_w <= 0.0:
        return {k: 0 for k in positive}

    raw = {k: (w / total_w) * target_total for k, w in positive.items()}
    floors = {k: int(val) for k, val in raw.items()}
    remainder = target_total - sum(floors.values())
    ranked = sorted(
        positive.keys(),
        key=lambda k: (-(raw[k] - floors[k]), -positive[k], k),
    )
    for k in ranked[:remainder]:
        floors[k] += 1
    return floors


def sample_full_generation_spec(
    source_records: list[SourceQARecord],
    cfg: dict[str, Any],
    *,
    split_name: str = "train",
    total_target_override: int | None = None,
    strategy_name: str = "strategy_a",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build deterministic split-locked generation specifications for full generation.

    Allocates dataset and context/combination quotas from cfg['full_generation']
    while strictly restricting source records to split_name's designated pool so
    no source_group_id ever leaks across train, val, and test splits.
    """
    pool_map = {
        "train": "train",
        "val": "val",
        "test_synthetic": "test_synthetic_pool",
    }
    if split_name not in pool_map:
        raise ValueError(f"Unsupported full_generation split_name={split_name!r}; expected {tuple(pool_map)}.")
    required_pool = pool_map[split_name]

    fg_cfg = cfg["full_generation"]
    seed_offset = {"train": 101, "val": 202, "test_synthetic": 303}[split_name]
    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + seed_offset)

    pool: list[SourceQARecord] = []
    for rec in source_records:
        if rec.split != required_pool:
            raise ValueError(
                f"Full generation pool violation for {split_name!r}: record {rec.record_id} has "
                f"split={rec.split!r}, expected {required_pool!r}."
            )
        pool.append(rec)

    split_ratio = float(fg_cfg["split_ratios"][split_name])
    default_split_total = int(round(int(fg_cfg["total_target_records"]) * split_ratio))
    split_total = int(total_target_override) if total_target_override is not None else default_split_total

    ds_quotas = _allocate_proportional_quotas(
        {ds: float(val) for ds, val in fg_cfg["dataset_quotas_total"].items()},
        split_total,
    )
    ctx_datasets = [ds for ds, pol in cfg["source_policies"].items() if pol["has_context"]]
    no_ctx_datasets = [ds for ds, pol in cfg["source_policies"].items() if not pol["has_context"]]
    with_ctx_total = sum(ds_quotas.get(ds, 0) for ds in ctx_datasets)
    no_ctx_total = split_total - with_ctx_total

    no_ctx_weights = {
        k: float(v)
        for k, v in fg_cfg["combination_quotas"]["no_context"].items()
        if k != "total"
    }
    with_ctx_weights = {
        k: float(v)
        for k, v in fg_cfg["combination_quotas"]["with_context"].items()
        if k != "total"
    }
    no_ctx_combo_quotas = _allocate_proportional_quotas(no_ctx_weights, no_ctx_total)
    with_ctx_combo_quotas = _allocate_proportional_quotas(with_ctx_weights, with_ctx_total)

    # Order slots hardest-first (G5 -> G1 -> G0) inside each context regime
    combo_slots: list[tuple[bool, str]] = []
    for is_ctx, qmap in ((True, with_ctx_combo_quotas), (False, no_ctx_combo_quotas)):
        for bits, cnt in sorted(
            qmap.items(),
            key=lambda kv: (-sum(int(c) for c in kv[0]), kv[0]),
        ):
            combo_slots.extend([(is_ctx, bits)] * int(cnt))

    # Deterministic per-dataset pools ordered to maximize unique source_group_id diversity first,
    # then spreading round-robin across held-out contract/document groups inside this same split.
    by_dataset_ordered: dict[str, list[SourceQARecord]] = {}
    for ds in sorted(ds_quotas.keys()):
        recs = [r for r in pool if r.source_dataset == ds]
        idxs = list(range(len(recs)))
        rng.shuffle(idxs)
        shuffled = [recs[i] for i in idxs]
        group_buckets: dict[str, list[SourceQARecord]] = defaultdict(list)
        for rec in shuffled:
            group_buckets[rec.source_group_id].append(rec)
        # Round-robin across unique groups so Pass 1 exhausts distinct source_group_ids first
        rr_recs: list[SourceQARecord] = []
        bucket_keys = sorted(group_buckets.keys())
        rng.shuffle(bucket_keys)
        while any(group_buckets[k] for k in bucket_keys):
            for gk in bucket_keys:
                if group_buckets[gk]:
                    rr_recs.append(group_buckets[gk].pop(0))
        by_dataset_ordered[ds] = rr_recs

    ds_remaining = dict(ds_quotas)
    used_record_ids: set[str] = set()
    used_group_ids: set[str] = set()
    specs: list[dict[str, Any]] = []

    for idx, (is_ctx, bits) in enumerate(combo_slots):
        candidate_ds = ctx_datasets if is_ctx else no_ctx_datasets
        eligible_ds = sorted(
            [d for d in candidate_ds if ds_remaining.get(d, 0) > 0],
            key=lambda d: (-(ds_remaining[d] / max(1, ds_quotas[d])), d),
        )
        chosen: SourceQARecord | None = None
        for ds in eligible_ds:
            for rec in by_dataset_ordered.get(ds, []):
                if rec.record_id in used_record_ids:
                    continue
                if _is_eligible_for_bits(rec, bits, cfg):
                    chosen = rec
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(
                f"Could not find eligible {split_name} record for vector={bits}, has_context={is_ctx} "
                f"(remaining={ds_remaining})."
            )

        used_record_ids.add(chosen.record_id)
        used_group_ids.add(chosen.source_group_id)
        ds_remaining[chosen.source_dataset] -= 1
        active_labels = active_labels_from_bits(bits)
        forbidden_labels = [lbl for lbl in HALLUCINATION_LABELS if lbl not in active_labels]
        step_orders = (
            compute_strategy_a_orders(active_labels, cfg["taxonomy"], seed + seed_offset, idx)
            if active_labels
            else {"forward": []}
        )
        specs.append(
            {
                "base_item_id": f"FULL_{split_name.upper()}_{idx:05d}",
                "source_record": chosen,
                "split": split_name,
                "generation_strategy": "clean_control" if bits == "00000" else strategy_name,
                "intended_bits": bits,
                "cardinality_tier": cardinality_tier(bits),
                "active_labels": active_labels,
                "forbidden_labels": forbidden_labels,
                "step_orders": step_orders,
                "is_clean": bits == "00000",
            }
        )

    assert all(rem == 0 for rem in ds_remaining.values()), (
        f"Unfulfilled full_generation quotas for {split_name}: {ds_remaining}"
    )

    summary = {
        "split_name": split_name,
        "source_split_pool": required_pool,
        "total_target_slots": len(specs),
        "strategy_name": strategy_name,
        "unique_source_records": len(used_record_ids),
        "unique_group_ids": len(used_group_ids),
        "by_dataset": dict(Counter(s["source_record"].source_dataset for s in specs)),
        "by_domain": dict(Counter(s["source_record"].domain for s in specs)),
        "by_context": dict(Counter(str(s["source_record"].has_context) for s in specs)),
        "by_tier": dict(Counter(s["cardinality_tier"] for s in specs)),
        "combination_counts": dict(Counter(s["intended_bits"] for s in specs)),
    }
    return specs, summary


def sample_natural_test_spec(
    source_records: list[SourceQARecord],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Sample held-out test_natural questions and group ALL multi-model answers per question.

    Preserves shared source_group_id and source_document_id across all LLM respondents to the
    same question so evaluation and bootstrap confidence intervals avoid question-level leakage.
    """
    fg_cfg = cfg["full_generation"]["human_gold_subsets"]
    models = cfg["backends"]["natural_qa_models"]
    n_models = max(1, len(models))
    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 77)

    natural_pool = [r for r in source_records if r.split == "test_natural_pool"]
    if len(natural_pool) != len(source_records):
        raise ValueError("test_natural sampling accepts ONLY split='test_natural_pool' records.")

    # Target total responses divided across balanced question quotas
    target_total_responses = int(fg_cfg.get("test_natural_target", 360))
    n_questions = target_total_responses // n_models

    # Balanced ratios matching full generation (20% medquad, 20% pubmedqa, 15% legal_qa_v1, 20% cuad_qa, 25% fiqa)
    shares = {
        "medquad": 0.20,
        "pubmedqa": 0.20,
        "legal_qa_v1": 0.15,
        "cuad_qa": 0.20,
        "fiqa": 0.25,
    }
    ds_q_quotas = {ds: int(round(n_questions * sh)) for ds, sh in shares.items()}
    diff = n_questions - sum(ds_q_quotas.values())
    if diff != 0:
        ds_q_quotas["fiqa"] += diff

    by_ds: dict[str, list[SourceQARecord]] = defaultdict(list)
    for rec in sorted(natural_pool, key=lambda r: r.record_id):
        by_ds[rec.source_dataset].append(rec)

    selected_specs: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    used_record_ids: set[str] = set()
    q_counter = 0

    for ds, need_q in ds_q_quotas.items():
        cands = list(by_ds[ds])
        idxs = list(range(len(cands)))
        rng.shuffle(idxs)
        picked_recs: list[SourceQARecord] = []
        # Pass 1: maximize unique source_group_ids first
        for i in idxs:
            if len(picked_recs) >= need_q:
                break
            rec = cands[i]
            if rec.record_id in used_record_ids or rec.source_group_id in used_groups:
                continue
            used_groups.add(rec.source_group_id)
            used_record_ids.add(rec.record_id)
            picked_recs.append(rec)

        # Pass 2 (e.g., CUAD contracts grouped by document inside test_natural_pool):
        # spread additional distinct questions round-robin across held-out document groups
        if len(picked_recs) < need_q:
            group_buckets: dict[str, list[SourceQARecord]] = defaultdict(list)
            for i in idxs:
                rec = cands[i]
                if rec.record_id not in used_record_ids:
                    group_buckets[rec.source_group_id].append(rec)
            bucket_keys = sorted(group_buckets.keys())
            while len(picked_recs) < need_q and any(group_buckets[k] for k in bucket_keys):
                for gk in bucket_keys:
                    if len(picked_recs) >= need_q:
                        break
                    if group_buckets[gk]:
                        rec = group_buckets[gk].pop(0)
                        used_groups.add(rec.source_group_id)
                        used_record_ids.add(rec.record_id)
                        picked_recs.append(rec)

        assert len(picked_recs) == need_q, (
            f"Insufficient test_natural_pool records for {ds}: {len(picked_recs)}/{need_q}"
        )
        for rec in picked_recs:
            for m_spec in models:
                selected_specs.append(
                    {
                        "natural_item_id": f"NAT_{q_counter:04d}_{m_spec['model_id'].split('/')[-1]}",
                        "question_bundle_id": f"NAT_Q_{q_counter:04d}",
                        "source_record": rec,
                        "source_group_id": rec.source_group_id,
                        "respondent_model_spec": m_spec,
                    }
                )
            q_counter += 1

    by_ds_groups = {
        ds: len({s["source_group_id"] for s in selected_specs if s["source_record"].source_dataset == ds})
        for ds in ds_q_quotas
    }
    summary = {
        "source_split_pool": "test_natural_pool",
        "unique_questions": q_counter,
        "total_natural_response_slots": len(selected_specs),
        "models_per_question": n_models,
        "unique_group_ids": len(used_groups),
        "by_dataset_questions": ds_q_quotas,
        "by_dataset_unique_groups": by_ds_groups,
    }
    return selected_specs, summary
