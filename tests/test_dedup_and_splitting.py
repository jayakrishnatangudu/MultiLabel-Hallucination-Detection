"""Unit tests for near-duplicate clustering guards, group splitting, and leakage verification."""

from __future__ import annotations

import pytest

from src.data.preprocessing import _cluster_near_duplicate_questions, normalize_match_key
from src.data.schema import LicenseMetadata, ProvenanceMetadata, SourceQARecord
from src.data.splitting import SPLIT_ORDER, assign_group_splits, verify_zero_split_leakage


_ND_CFG = {
    "enabled": True,
    "tfidf_cosine_threshold": 0.90,
    "min_rare_token_jaccard": 0.65,
    "rare_token_max_df_ratio": 0.15,
    "enforce_same_qtype_if_present": True,
    "max_audit_borderline_pairs": 15,
    "borderline_cosine_low": 0.78,
}


def _make_record(
    idx: int,
    group_id: str,
    question: str,
    answer: str = "Valid detailed medical answer text.",
    dataset: str = "medquad",
    subset: str = "train",
    domain: str = "medical",
    has_context: bool = False,
    context: str | None = None,
    doc_id: str | None = None,
    split: str | None = None,
) -> SourceQARecord:
    return SourceQARecord(
        record_id=f"rec_{idx}",
        source_dataset=dataset,
        source_subset=subset,
        source_record_id=f"raw_{idx}",
        source_document_id=doc_id,
        source_group_id=group_id,
        domain=domain,
        has_context=has_context,
        question=question,
        context=context,
        original_answer=answer,
        metadata={},
        provenance=ProvenanceMetadata(
            source_url="https://example.org/ds",
            upstream_url="https://example.org/up",
            dataset_version="abc123",
            downloaded_at_utc="2026-11-12T00:00:00+00:00",
            raw_filename="file.csv",
            raw_file_sha256="0" * 64,
            preprocessing_config_sha256="1" * 64,
        ),
        licensing=LicenseMetadata(
            hf_card_license="mit",
            upstream_license="MIT",
            license_status="verified_open",
            redistribution_permitted=True,
            attribution="Test et al. (2025)",
            notes="Test attribution",
        ),
        split=split,
    )


def test_near_duplicate_merges_paraphrases_not_distinct_entities() -> None:
    questions = [
        ("What is (are) Parkinson's Disease ?", "information"),
        ("What is (are) Parkinson disease ?", "information"),
        ("What is (are) Q Fever ?", "information"),
        ("What is (are) Fever ?", "information"),
        ("What is (are) What I need to know about Hepatitis A ?", "information"),
        ("What is (are) What I need to know about Hepatitis B ?", "information"),
        ("What is (are) What I need to know about Hepatitis C ?", "information"),
        ("What are the symptoms of Asthma ?", "symptoms"),
        ("What are the symptoms of Diabetes ?", "symptoms"),
        ("What is the treatment for Migraine ?", "treatment"),
        ("What is the treatment for Migraine ?", "prevention"),
    ]
    items = [
        {
            "q_key": f"{normalize_match_key(q)}||{i}",
            "question_norm": q,
            "qtype": qtype,
        }
        for i, (q, qtype) in enumerate(questions)
    ]

    qkey_to_cluster, _ = _cluster_near_duplicate_questions(items, _ND_CFG)
    clusters = [qkey_to_cluster[item["q_key"]] for item in items]

    # Parkinson's Disease vs Parkinson disease MUST merge
    assert clusters[0] == clusters[1]
    # Q Fever vs Fever MUST NOT merge
    assert clusters[2] != clusters[3]
    # Hepatitis A, B, C MUST remain distinct
    assert len({clusters[4], clusters[5], clusters[6]}) == 3
    # Template-similar Asthma vs Diabetes MUST remain distinct
    assert clusters[7] != clusters[8]
    # Same wording across different qtype MUST remain distinct when qtype guard is enabled
    assert clusters[9] != clusters[10]


def test_assign_group_splits_preserves_groups_and_determinism() -> None:
    records: list[SourceQARecord] = []
    rec_idx = 0
    for g_idx in range(30):
        gid = f"grp_med_{g_idx}"
        q_text = f"Medical question number {g_idx} about condition alpha_{g_idx}?"
        # Give each group 2 distinct answers so group integrity is tested
        for ans_num in range(2):
            records.append(
                _make_record(
                    idx=rec_idx,
                    group_id=gid,
                    question=q_text,
                    answer=f"Detailed medical explanation variant {ans_num} for condition {g_idx}.",
                    dataset="medquad",
                    subset="train",
                )
            )
            rec_idx += 1

    for g_idx in range(20):
        gid = f"grp_fin_{g_idx}"
        records.append(
            _make_record(
                idx=rec_idx,
                group_id=gid,
                question=f"Finance question number {g_idx} about portfolio beta_{g_idx}?",
                answer=f"Detailed finance explanation for question {g_idx}.",
                dataset="fiqa",
                subset="test",
                domain="finance",
            )
        )
        rec_idx += 1

    split_cfg = {
        "train_ratio": 0.80,
        "val_ratio": 0.10,
        "test_synthetic_ratio": 0.08,
        "test_natural_ratio": 0.02,
    }
    split_a, report_a = assign_group_splits(records, split_cfg=split_cfg, seed=42)
    split_b, _ = assign_group_splits(records, split_cfg=split_cfg, seed=42)

    assert [r.split for r in split_a] == [r.split for r in split_b]
    assert set(r.split for r in split_a).issubset(set(SPLIT_ORDER))
    assert report_a["leakage_audit"]["leaked_groups"] == 0
    assert report_a["leakage_audit"]["leaked_documents"] == 0
    assert report_a["leakage_audit"]["leaked_context_free_questions"] == 0

    # Verify every multi-answer group occupies exactly one split
    group_splits: dict[str, set[str]] = {}
    for rec in split_a:
        assert rec.split is not None
        group_splits.setdefault(rec.source_group_id, set()).add(rec.split)
    assert all(len(s) == 1 for s in group_splits.values())


def test_verify_zero_split_leakage_catches_group_doc_and_cf_question_leakage() -> None:
    unset_rec = _make_record(0, "g0", "What is X?", split=None)
    with pytest.raises(ValueError, match="unset split"):
        verify_zero_split_leakage([unset_rec])

    # 1. Shared source_group_id across splits
    r1 = _make_record(1, "g_shared", "Question one?", split="train")
    r2 = _make_record(2, "g_shared", "Question two?", split="val")
    with pytest.raises(AssertionError, match="Split leakage detected"):
        verify_zero_split_leakage([r1, r2])

    # 2. Shared source_document_id across splits despite distinct group IDs
    d1 = _make_record(
        3,
        "g_doc_a",
        "Clause one?",
        dataset="cuad_qa",
        domain="legal",
        has_context=True,
        context="Full contract text context.",
        doc_id="doc_99",
        split="train",
    )
    d2 = _make_record(
        4,
        "g_doc_b",
        "Clause two?",
        dataset="cuad_qa",
        domain="legal",
        has_context=True,
        context="Full contract text context.",
        doc_id="doc_99",
        split="test_synthetic_pool",
    )
    with pytest.raises(AssertionError, match="Split leakage detected"):
        verify_zero_split_leakage([d1, d2])

    # 3. Context-free identical question across splits despite distinct group IDs
    c1 = _make_record(5, "g_q1", "What causes lupus?", split="train")
    c2 = _make_record(6, "g_q2", "What causes lupus?!", split="test_natural_pool")
    with pytest.raises(AssertionError, match="Split leakage detected"):
        verify_zero_split_leakage([c1, c2])

