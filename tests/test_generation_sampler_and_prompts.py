"""Unit tests for Phase 2 pilot sampler, CUAD windowing, clause segmentation, and prompts."""

from __future__ import annotations

import pytest

from src.data.schema import LicenseMetadata, ProvenanceMetadata, SourceQARecord
from src.generation.prompt_builder import (
    load_prompt_template,
    render_cross_judge,
    render_natural_qa,
    render_strategy_a_step,
    render_strategy_b_joint,
    render_strategy_c_span_editing,
)
from src.generation.sampler import (
    compute_strategy_a_orders,
    extract_evidence_anchored_window,
    sample_natural_test_spec,
    sample_pilot_e0_spec,
    split_response_clauses,
)
from src.utils.config import load_yaml_config


def _make_source_records_pool(split_name: str = "train") -> list[SourceQARecord]:
    records: list[SourceQARecord] = []
    specs = [
        ("medquad", "medical", False, "train", 30),
        ("pubmedqa", "medical", True, "pqa_labeled", 30),
        ("legal_qa_v1", "legal", False, "train", 25),
        ("cuad_qa", "legal", True, "train", 30),
        ("fiqa", "finance", False, "train", 35),
    ]
    idx = 0
    for ds, domain, has_ctx, subset, count in specs:
        lic_status = "unresolved" if ds == "legal_qa_v1" else ("verified_open" if ds == "pubmedqa" else "upstream_caution")
        redist = ds not in {"legal_qa_v1", "fiqa"}
        for k in range(count):
            idx += 1
            ans = (
                f"Sentence one explains core mechanism {k} for {ds}. "
                f"Sentence two specifies secondary clinical or statutory evidence {k} clearly. "
                f"Sentence three outlines scope boundaries {k} accurately. "
                f"Sentence four gives quantitative guidance {k} for evaluation. "
                f"Sentence five confirms final compliance criteria {k}."
            )
            ctx = (
                f"Full reference context document {k} for {ds}. "
                + ans
                + " "
                + (" ".join(["filler"] * 40))
            ) if has_ctx else None
            records.append(
                SourceQARecord(
                    record_id=f"{ds}_{idx:05d}",
                    source_dataset=ds,
                    source_subset=subset,
                    source_record_id=f"orig_{idx}",
                    source_document_id=f"doc_{ds}_{k}" if has_ctx else None,
                    source_group_id=f"grp_{ds}_{k}",
                    domain=domain,  # type: ignore[arg-type]
                    has_context=has_ctx,
                    question=f"What are the domain requirements and criteria number {k} in {ds}?",
                    context=ctx,
                    original_answer=ans,
                    metadata={"answer_words": len(ans.split()), "question_words": 10},
                    provenance=ProvenanceMetadata(
                        source_url="https://example.org",
                        upstream_url="https://example.org",
                        dataset_version="v1",
                        downloaded_at_utc="2026-11-12T00:00:00+00:00",
                        raw_filename="raw.csv",
                        raw_file_sha256="0" * 64,
                        preprocessing_config_sha256="1" * 64,
                    ),
                    licensing=LicenseMetadata(
                        hf_card_license=None,
                        upstream_license="MIT" if ds == "pubmedqa" else "unresolved" if ds == "legal_qa_v1" else "CC-BY-4.0",
                        license_status=lic_status,  # type: ignore[arg-type]
                        redistribution_permitted=redist,
                        attribution="Test attribution",
                        notes="Test license notes.",
                    ),
                    split=split_name,  # type: ignore[arg-type]
                )
            )
    return records


def test_evidence_window_preserves_answer_span_inside_word_limit() -> None:
    target_clause = "This Agreement shall be governed by the laws of the State of New York."
    long_contract = (
        "PREAMBLE TITLE CONTRACT AGREEMENT PARTIES DEFINITIONS SECTION ONE "
        + " ".join(["preamble"] * 220)
        + " "
        + " ".join(["middle"] * 1200)
        + " SECTION FOURTEEN GOVERNING LAW. "
        + target_clause
        + " "
        + " ".join(["trailer"] * 1500)
    )
    windowed, applied = extract_evidence_anchored_window(
        long_contract, target_clause, max_words=600, preamble_words=100, window_words_around_span=200
    )
    assert applied is True
    assert windowed is not None
    assert len(windowed.split()) <= 605
    assert target_clause in windowed


def test_clause_splitting_and_strategy_a_order_sequences() -> None:
    semis = "Clause one is clearly binding on the parties; Clause two sets a thirty day deadline; Clause three defines arbitration."
    clauses = split_response_clauses(semis)
    assert len(clauses) == 3

    cfg = load_yaml_config("configs/generation.yaml")
    orders = compute_strategy_a_orders(
        ["factual_inconsistency", "factual_fabrication", "logical_inconsistency"],
        cfg["taxonomy"],
        seed=42,
        item_idx=3,
    )
    assert orders["forward"] == ["factual_inconsistency", "factual_fabrication", "logical_inconsistency"]
    assert orders["reverse"] == ["logical_inconsistency", "factual_fabrication", "factual_inconsistency"]
    assert set(orders["randomized"]) == set(orders["forward"])
    assert orders["randomized"] not in (orders["forward"], orders["reverse"])


def test_pilot_e0_sampler_balance_pair_lock_and_split_guard() -> None:
    cfg = load_yaml_config("configs/generation.yaml")
    train_pool = _make_source_records_pool("train")
    paired_specs, clean_specs, summary = sample_pilot_e0_spec(train_pool, cfg)

    assert len(paired_specs) == 80
    assert len(clean_specs) == 20
    assert summary["order_sensitivity_subset_count"] == 18
    assert summary["paired_by_dataset"] == {
        "medquad": 16,
        "pubmedqa": 16,
        "legal_qa_v1": 12,
        "cuad_qa": 16,
        "fiqa": 20,
    }
    assert summary["paired_by_context"] == {"False": 48, "True": 32}
    assert summary["paired_by_tier"] == {"G5": 2, "G4": 5, "G3": 18, "G2": 30, "G1": 25}
    ctx_ci = summary["paired_by_context_and_ci"]
    assert ctx_ci["with_context_ci_1"] == 28
    assert ctx_ci["with_context_ci_0"] == 4
    assert ctx_ci["with_context_ci_0_by_tier"] == {"G1": 2, "G2": 2}
    assert ctx_ci["no_context_ci_0"] == 48
    g1_ctx_ds = {it["source_dataset"] for it in ctx_ci["with_context_ci_0_items"] if it["cardinality_tier"] == "G1"}
    g2_ctx_ds = {it["source_dataset"] for it in ctx_ci["with_context_ci_0_items"] if it["cardinality_tier"] == "G2"}
    assert g1_ctx_ds == {"pubmedqa", "cuad_qa"}
    assert g2_ctx_ds == {"pubmedqa", "cuad_qa"}
    assert summary["unique_source_questions"] == 100
    assert summary["budget_and_call_accounting"]["core_pilot_budget"]["total_core_evaluation_items"] == 260
    assert summary["budget_and_call_accounting"]["core_pilot_budget"]["nominal_generator_calls"]["subtotal"] == 329
    assert summary["budget_and_call_accounting"]["order_sensitivity_addon"]["is_additional_to_260"] is True
    assert summary["budget_and_call_accounting"]["order_sensitivity_addon"]["additional_comparative_outputs"] == 36
    assert (
        summary["budget_and_call_accounting"]["combined_pilot_totals_when_order_addon_enabled"][
            "total_evaluation_items"
        ]
        == 296
    )
    assert (
        summary["budget_and_call_accounting"]["combined_pilot_totals_when_order_addon_enabled"][
            "nominal_generator_calls"
        ]
        == 417
    )
    assert set(summary["tier_by_domain_and_context"].keys()) == {"G0", "G1", "G2", "G3", "G4", "G5"}

    # Zero source_group_id overlap between paired perturbed base items and clean controls
    paired_groups = {s["source_record"].source_group_id for s in paired_specs}
    clean_groups = {s["source_record"].source_group_id for s in clean_specs}
    assert paired_groups.isdisjoint(clean_groups)
    assert all(s["source_record"].split == "train" for s in paired_specs + clean_specs)

    # Reject any non-train record entering pilot pool
    bad_pool = list(train_pool)
    bad_pool[0] = bad_pool[0].model_copy(update={"split": "test_synthetic_pool"})
    with pytest.raises(ValueError, match="Pilot source pool violation"):
        sample_pilot_e0_spec(bad_pool, cfg)


def test_natural_sampler_groups_all_model_answers_by_question() -> None:
    cfg = load_yaml_config("configs/generation.yaml")
    nat_pool = _make_source_records_pool("test_natural_pool")
    # Simulate CUAD contract-document grouping (multiple distinct clause questions per contract group ID)
    doc_grouped_pool = [
        rec.model_copy(update={"source_group_id": f"grp_cuad_doc_{idx % 2}"})
        if rec.source_dataset == "cuad_qa"
        else rec
        for idx, rec in enumerate(nat_pool)
    ]
    small_cfg = dict(cfg)
    small_cfg["full_generation"] = {
        **cfg["full_generation"],
        "human_gold_subsets": {**cfg["full_generation"]["human_gold_subsets"], "test_natural_target": 60},
    }
    specs, summary = sample_natural_test_spec(doc_grouped_pool, small_cfg)
    assert summary["unique_questions"] == 20
    assert len(specs) == 60
    assert summary["unique_group_ids"] < summary["unique_questions"]
    # Every question bundle must map 1-to-1 to one source_record_id and one source_group_id across all 3 models
    bundles_group: dict[str, set[str]] = {}
    bundles_record: dict[str, set[str]] = {}
    for sp in specs:
        bundles_group.setdefault(sp["question_bundle_id"], set()).add(sp["source_group_id"])
        bundles_record.setdefault(sp["question_bundle_id"], set()).add(sp["source_record"].record_id)
    assert len(bundles_group) == 20
    assert all(len(gids) == 1 for gids in bundles_group.values())
    assert all(len(rids) == 1 for rids in bundles_record.values())
    assert len({sp["source_record"].record_id for sp in specs}) == 20


def test_versioned_prompt_templates_render_cleanly_and_blind_judge() -> None:
    cfg = load_yaml_config("configs/generation.yaml")
    tpl_a = load_prompt_template(cfg["prompts"]["strategy_a_sequential"])
    tpl_b = load_prompt_template(cfg["prompts"]["strategy_b_joint"])
    tpl_c = load_prompt_template(cfg["prompts"]["strategy_c_span_editing"])
    tpl_j = load_prompt_template(cfg["prompts"]["cross_judge_auditor"])
    tpl_n = load_prompt_template(cfg["prompts"]["natural_qa_generation"])

    sa_sys, sa_usr = render_strategy_a_step(
        tpl_a,
        domain="medical",
        question="What is metformin's primary action?",
        has_context=False,
        prompt_context=None,
        current_response="Metformin lowers hepatic glucose production.",
        target_category="factual_inconsistency",
        step_index=1,
        total_steps=2,
        preserved_active_labels=[],
        forbidden_labels=["context_inconsistency"],
    )
    assert "factual_inconsistency" in sa_usr and len(sa_sys) > 20

    _, sb_usr = render_strategy_b_joint(
        tpl_b,
        domain="medical",
        source_dataset="medquad",
        question="What is metformin's primary action?",
        has_context=False,
        prompt_context=None,
        original_answer="Metformin lowers hepatic glucose production.",
        intended_bits="11000",
        active_labels=["factual_inconsistency", "factual_fabrication"],
        forbidden_labels=["instruction_inconsistency", "logical_inconsistency", "context_inconsistency"],
    )
    assert "11000" in sb_usr

    _, sc_usr, ops = render_strategy_c_span_editing(
        tpl_c,
        domain="finance",
        source_dataset="fiqa",
        question="How is Roth IRA growth taxed?",
        has_context=False,
        prompt_context=None,
        original_answer="Roth IRA contributions use after-tax dollars. Qualified withdrawals remain tax-free.",
        intended_bits="10010",
        active_labels=["factual_inconsistency", "logical_inconsistency"],
        forbidden_labels=["factual_fabrication", "instruction_inconsistency", "context_inconsistency"],
    )
    assert len(ops) == 2 and "S1" in sc_usr and "S2" in sc_usr

    _, sj_usr = render_cross_judge(
        tpl_j,
        domain="finance",
        question="How is Roth IRA growth taxed?",
        has_context=False,
        prompt_context=None,
        reference_answer="Qualified Roth IRA withdrawals are tax-free.",
        candidate_response="Roth withdrawals are always taxed at 40%.",
    )
    assert "intended" not in sj_usr.lower() and "strategy" not in sj_usr.lower()

    _, sn_usr = render_natural_qa(
        tpl_n,
        domain="legal",
        question="What is the governing law?",
        has_context=True,
        prompt_context="Governed by Delaware law.",
    )
    assert "Governed by Delaware law." in sn_usr

