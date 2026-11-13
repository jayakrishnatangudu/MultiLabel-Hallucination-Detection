"""Unit tests for canonical Phase 1 and Phase 2+ schemas."""

import pytest
from pydantic import ValidationError

from src.data.schema import (
    AutoDetectionTier,
    GenerationAttemptRecord,
    HALLUCINATION_LABELS,
    LabelDictionary,
    LicenseMetadata,
    MultiLabelRecord,
    ProvenanceMetadata,
    SourceQARecord,
)


def _sample_provenance() -> ProvenanceMetadata:
    return ProvenanceMetadata(
        source_url="https://example.org/ds",
        upstream_url="https://example.org/upstream",
        dataset_version="abc123",
        downloaded_at_utc="2026-11-12T00:00:00+00:00",
        raw_filename="sample.csv",
        raw_file_sha256="deadbeef",
        preprocessing_config_sha256="cafebabe",
    )


def _sample_license(
    license_status: str = "verified_open",
    redistribution_permitted: bool = True,
) -> LicenseMetadata:
    return LicenseMetadata(
        hf_card_license="mit",
        upstream_license="MIT",
        license_status=license_status,
        redistribution_permitted=redistribution_permitted,
        attribution="Test Author (2025)",
        notes="Verified test license.",
    )


def test_source_record_valid_with_and_without_context() -> None:
    rec_no_ctx = SourceQARecord(
        record_id="medquad_000001",
        source_dataset="medquad",
        source_subset="train",
        source_record_id="0",
        source_group_id="grp_medquad_1",
        domain="medical",
        has_context=False,
        question="What are the symptoms of anemia?",
        context=None,
        original_answer="Fatigue, weakness, and pale skin are common symptoms.",
        provenance=_sample_provenance(),
        licensing=_sample_license("upstream_caution", True),
    )
    assert rec_no_ctx.has_context is False

    rec_ctx = SourceQARecord(
        record_id="pubmedqa_000001",
        source_dataset="pubmedqa",
        source_subset="pqa_labeled",
        source_record_id="21645374",
        source_document_id="pubmedqa_21645374",
        source_group_id="grp_pubmedqa_21645374",
        domain="medical",
        has_context=True,
        question="Does compound X reduce fever?",
        context="Clinical trial abstract showing reduction in fever.",
        original_answer="Yes, compound X reduced fever in the trial.",
        provenance=_sample_provenance(),
        licensing=_sample_license("verified_open", True),
    )
    assert rec_ctx.has_context is True


def test_source_record_rejects_context_mismatch_and_blank_fields() -> None:
    with pytest.raises(ValidationError):
        SourceQARecord(
            record_id="pubmedqa_bad",
            source_dataset="pubmedqa",
            source_subset="pqa_labeled",
            source_record_id="1",
            source_group_id="grp_1",
            domain="medical",
            has_context=True,
            question="Valid question?",
            context="   ",
            original_answer="Valid answer.",
            provenance=_sample_provenance(),
            licensing=_sample_license(),
        )

    with pytest.raises(ValidationError):
        SourceQARecord(
            record_id="medquad_blank",
            source_dataset="medquad",
            source_subset="train",
            source_record_id="1",
            source_group_id="grp_1",
            domain="medical",
            has_context=False,
            question="   ",
            context=None,
            original_answer="Valid answer.",
            provenance=_sample_provenance(),
            licensing=_sample_license(),
        )


def test_legal_qa_v1_requires_unresolved_non_redistributable_license() -> None:
    with pytest.raises(ValidationError):
        SourceQARecord(
            record_id="legal_qa_bad_lic",
            source_dataset="legal_qa_v1",
            source_subset="train",
            source_record_id="0",
            source_group_id="grp_legal_0",
            domain="legal",
            has_context=False,
            question="Is verbal agreement binding?",
            context=None,
            original_answer="Depends on jurisdiction and subject matter.",
            provenance=_sample_provenance(),
            licensing=_sample_license("verified_open", True),
        )

    valid_legal = SourceQARecord(
        record_id="legal_qa_ok",
        source_dataset="legal_qa_v1",
        source_subset="train",
        source_record_id="0",
        source_group_id="grp_legal_0",
        domain="legal",
        has_context=False,
        question="Is verbal agreement binding?",
        context=None,
        original_answer="Depends on jurisdiction and subject matter.",
        provenance=_sample_provenance(),
        licensing=_sample_license("unresolved", False),
    )
    assert valid_legal.licensing.license_status == "unresolved"
    assert valid_legal.licensing.redistribution_permitted is False


def test_multilabel_record_tiers_and_length_five() -> None:
    labels = LabelDictionary(
        factual_inconsistency=1,
        factual_fabrication=1,
        instruction_inconsistency=0,
        logical_inconsistency=0,
        context_inconsistency=1,
    )
    assert len(HALLUCINATION_LABELS) == 5
    assert labels.to_vector() == [1, 1, 0, 0, 1]

    rec = MultiLabelRecord(
        sample_id="MLH_000001",
        source_dataset="pubmedqa",
        source_record_id="21645374",
        source_group_id="grp_pubmedqa_21645374",
        domain="medical",
        data_origin="synthetic",
        has_context=True,
        question="Does compound X work?",
        context="Trial showed efficacy.",
        original_answer="Yes.",
        generated_answer="No, and Trial NCT9999 proved otherwise.",
        generation_method="span_editing",
        intended_labels=labels,
        auto_detected_labels=AutoDetectionTier(
            judge_model="mock-judge",
            predicted_labels=[1, 1, 0, 0, 1],
            rule_checks_passed=True,
        ),
        human_observed_labels={"rater_a": [1, 1, 0, 0, 1]},
        adjudicated_gold_labels=[1, 1, 0, 0, 1],
        label_vector=[1, 1, 0, 0, 1],
        validation_status="human_verified",
        licensing=_sample_license(),
    )
    assert rec.adjudicated_gold_labels == [1, 1, 0, 0, 1]

    with pytest.raises(ValidationError):
        MultiLabelRecord(
            sample_id="MLH_BAD_NAT",
            source_dataset="pubmedqa",
            source_record_id="21645374",
            source_group_id="grp_1",
            domain="medical",
            data_origin="natural",
            has_context=True,
            question="Q",
            context="C",
            generated_answer="A",
            generation_method="natural_llm",
            intended_labels=labels,
            licensing=_sample_license(),
        )


def test_multilabel_phase2_provenance_and_ci_context_guards() -> None:
    # Context-absent sample must reject intended or gold CI=1
    with pytest.raises(ValidationError, match="context_inconsistency cannot be 1"):
        MultiLabelRecord(
            sample_id="MLH_BAD_CI",
            source_dataset="medquad",
            source_subset="train",
            source_record_id="10",
            source_group_id="grp_10",
            domain="medical",
            data_origin="synthetic",
            has_context=False,
            question="What causes asthma?",
            context=None,
            original_answer="Airway inflammation.",
            generated_answer="Airway inflammation.",
            generation_method="strategy_c",
            generation_strategy="strategy_c",
            intended_labels=LabelDictionary(
                factual_inconsistency=0,
                factual_fabrication=0,
                instruction_inconsistency=0,
                logical_inconsistency=0,
                context_inconsistency=1,
            ),
            licensing=_sample_license("upstream_caution", True),
        )

    # Rejected record preserves empty/rejected answer + rejection_reasons cleanly
    rej = MultiLabelRecord(
        sample_id="MLH_REJ_01",
        source_dataset="cuad_qa",
        source_subset="test",
        source_record_id="doc_1_q1",
        source_document_id="doc_1",
        source_group_id="grp_doc_1",
        domain="legal",
        data_origin="synthetic",
        has_context=True,
        question="What is the governing law?",
        context="This Agreement is governed by New York law.",
        original_answer="New York law.",
        generated_answer="",
        generation_method="strategy_b",
        generation_strategy="strategy_b",
        generation_model="mock-llama",
        prompt_version="v1.0",
        generation_parameters={"temperature": 0.7, "seed": 42},
        validation_status="rejected",
        rejection_reasons=["json_parse_failure"],
        attempted_count=3,
        licensing=_sample_license("upstream_caution", True),
        split="test_synthetic",
    )
    assert rej.generation_strategy == "strategy_b"
    assert rej.rejection_reasons == ["json_parse_failure"]

    att = GenerationAttemptRecord(
        attempt_id="att_1",
        sample_id="MLH_REJ_01",
        base_item_id="BASE_001",
        source_dataset="cuad_qa",
        source_subset="test",
        source_record_id="doc_1_q1",
        source_document_id="doc_1",
        source_group_id="grp_doc_1",
        domain="legal",
        split="test_synthetic",
        generation_strategy="strategy_b",
        generation_model="mock-llama",
        prompt_version="v1.0",
        attempt_number=1,
        intended_vector="10001",
        raw_output="broken",
        accepted=False,
        rejection_reasons=["json_parse_failure"],
        timestamp_utc="2026-11-12T00:00:00+00:00",
    )
    assert att.accepted is False


