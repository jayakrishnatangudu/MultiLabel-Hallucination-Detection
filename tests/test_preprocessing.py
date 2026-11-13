"""Unit tests for Phase 1 normalization, filtering, raw immutability, and provenance propagation."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd

from src.data.preprocessing import (
    normalize_display_text,
    normalize_match_key,
    preprocess_dataset,
)
from src.utils.io import sha256_file


_BASE_PRE_CFG = {
    "min_question_chars": 10,
    "min_answer_chars": 15,
    "min_question_words": 3,
    "min_answer_words": 3,
    "max_answer_words": 50,
    "strip_qa_prefixes": True,
    "near_duplicate": {
        "enabled": True,
        "tfidf_cosine_threshold": 0.90,
        "min_rare_token_jaccard": 0.65,
        "rare_token_max_df_ratio": 0.02,
        "enforce_same_qtype_if_present": True,
        "max_audit_borderline_pairs": 10,
        "borderline_cosine_low": 0.78,
    },
}


def test_text_normalization_preserves_meaning() -> None:
    assert normalize_display_text("Q:   What is   asthma? ?  ") == "What is asthma?"
    assert normalize_display_text("A:\u00a0Asthma\t\tis  chronic.") == "Asthma is chronic."
    assert normalize_display_text("Q: Legal question?", strip_qa_prefix=False) == "Q: Legal question?"
    assert normalize_match_key("What is (are) Parkinson's Disease?") == "what is are parkinson s disease"


def test_preprocess_medquad_filtering_exact_dedup_and_raw_immutability(tmp_path: Path) -> None:
    raw_csv = tmp_path / "medDataset_processed.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"qtype": "information", "Question": "What is asthma?", "Answer": "Asthma is a chronic airway condition."},
            {"qtype": "information", "Question": "What is asthma?", "Answer": "Asthma is a chronic airway condition."},
            {"qtype": "information", "Question": "What is asthma?", "Answer": "It causes inflammation and narrowing of airways."},
            {"qtype": "symptoms", "Question": "Short?", "Answer": "Too short question."},
            {"qtype": "information", "Question": "What is (are) ?", "Answer": "Broken scrape missing disease topic completely."},
            {"qtype": "treatment", "Question": "What is pneumonia treatment?", "Answer": " ".join(["word"] * 60)},
        ]
    )
    df.to_csv(raw_csv, index=False, encoding="utf-8")
    raw_sha_before = sha256_file(raw_csv)

    manifest_entry = {
        "hf_repo": "keivalya/MedQuad-MedicalQnADataset",
        "source_url": "https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset",
        "upstream_url": "https://github.com/abachaa/MedQuAD",
        "dataset_version": "test_sha_medquad",
        "domain": "medical",
        "has_context": False,
        "licensing": {
            "hf_card_license": None,
            "upstream_license": "CC-BY-4.0",
            "license_status": "upstream_caution",
            "redistribution_permitted": True,
            "attribution": "Ben Abacha & Demner-Fushman (2019).",
            "notes": "Test MedQuAD license metadata.",
        },
        "files": [
            {
                "subset": "train",
                "remote_path": "medDataset_processed.csv",
                "local_path": str(raw_csv),
                "sha256": raw_sha_before,
                "downloaded_at_utc": "2026-11-12T00:00:00+00:00",
            }
        ],
    }

    records, stats = preprocess_dataset(
        dataset_key="medquad",
        manifest_entry=manifest_entry,
        pre_cfg=_BASE_PRE_CFG,
        pre_cfg_sha="cfgsha_test",
        processed_dir=processed_dir,
    )

    assert sha256_file(raw_csv) == raw_sha_before
    assert len(records) == 2
    assert stats["attrition_breakdown"]["dropped_exact_tuple_duplicates"] == 1
    assert stats["attrition_breakdown"]["dropped_too_short"] == 1
    assert stats["attrition_breakdown"]["dropped_empty_topic_template"] == 1
    assert stats["attrition_breakdown"]["dropped_too_long_answer"] == 1
    assert records[0].source_group_id == records[1].source_group_id
    assert records[0].provenance.raw_file_sha256 == raw_sha_before
    assert records[0].licensing.attribution == "Ben Abacha & Demner-Fushman (2019)."


def test_preprocess_cuad_skips_impossible_and_groups_by_document(tmp_path: Path) -> None:
    zip_path = tmp_path / "data.zip"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    template_q = "Highlight the parts (if any) of this contract related to Governing Law."
    payload = {
        "data": [
            {
                "title": "Contract Alpha",
                "paragraphs": [
                    {
                        "context": "Contract Alpha shall be governed by the laws of New York without conflict rules.",
                        "qas": [
                            {
                                "id": "alpha_q1",
                                "question": template_q,
                                "is_impossible": False,
                                "answers": [{"text": "governed by the laws of New York"}],
                            },
                            {
                                "id": "alpha_q2",
                                "question": "Highlight the parts related to Anti-Assignment.",
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            },
            {
                "title": "Contract Beta",
                "paragraphs": [
                    {
                        "context": "Contract Beta shall be governed exclusively by the laws of California.",
                        "qas": [
                            {
                                "id": "beta_q1",
                                "question": template_q,
                                "is_impossible": False,
                                "answers": [{"text": "governed exclusively by the laws of California"}],
                            }
                        ],
                    }
                ],
            },
        ]
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train_separate_questions.json", json.dumps(payload))

    raw_sha = sha256_file(zip_path)
    manifest_entry = {
        "hf_repo": "TheAtticusProject/cuad",
        "source_url": "https://huggingface.co/datasets/TheAtticusProject/cuad",
        "upstream_url": "https://github.com/TheAtticusProject/cuad",
        "dataset_version": "test_sha_cuad",
        "domain": "legal",
        "has_context": True,
        "licensing": {
            "hf_card_license": "cc-by-4.0",
            "upstream_license": "CC-BY-4.0",
            "license_status": "upstream_caution",
            "redistribution_permitted": True,
            "attribution": "Hendrycks et al. (2021).",
            "notes": "Test CUAD metadata.",
        },
        "files": [
            {
                "subset": "archive",
                "remote_path": "data.zip",
                "local_path": str(zip_path),
                "sha256": raw_sha,
                "downloaded_at_utc": "2026-11-12T00:00:00+00:00",
            }
        ],
        "archive_members": [
            {"subset": "train", "member_path": "train_separate_questions.json"}
        ],
    }

    records, stats = preprocess_dataset(
        dataset_key="cuad_qa",
        manifest_entry=manifest_entry,
        pre_cfg=_BASE_PRE_CFG,
        pre_cfg_sha="cfgsha_test",
        processed_dir=processed_dir,
    )

    assert sha256_file(zip_path) == raw_sha
    assert stats["attrition_breakdown"]["dropped_unanswerable_cuad_impossible"] == 1
    assert len(records) == 2
    assert records[0].source_document_id != records[1].source_document_id
    assert records[0].source_group_id != records[1].source_group_id
    assert records[0].has_context is True


def test_legal_qa_licensing_propagated_as_unresolved(tmp_path: Path) -> None:
    raw_csv = tmp_path / "legal_qa_full.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "question": "Q: Can a landlord enter a rental property without notice?",
                "answer": "A: Most jurisdictions require reasonable advance notice except emergencies.",
            }
        ]
    ).to_csv(raw_csv, index=False, encoding="utf-8")

    raw_sha = sha256_file(raw_csv)
    manifest_entry = {
        "hf_repo": "dzunggg/legal-qa-v1",
        "source_url": "https://huggingface.co/datasets/dzunggg/legal-qa-v1",
        "upstream_url": "https://huggingface.co/datasets/dzunggg/legal-qa-v1",
        "dataset_version": "test_sha_legal",
        "domain": "legal",
        "has_context": False,
        "licensing": {
            "hf_card_license": None,
            "upstream_license": "unresolved",
            "license_status": "unresolved",
            "redistribution_permitted": False,
            "attribution": "dzunggg/legal-qa-v1.",
            "notes": "License unresolved.",
        },
        "files": [
            {
                "subset": "train",
                "remote_path": "legal_qa_full.csv",
                "local_path": str(raw_csv),
                "sha256": raw_sha,
                "downloaded_at_utc": "2026-11-12T00:00:00+00:00",
            }
        ],
    }

    records, _ = preprocess_dataset(
        dataset_key="legal_qa_v1",
        manifest_entry=manifest_entry,
        pre_cfg=_BASE_PRE_CFG,
        pre_cfg_sha="cfgsha_test",
        processed_dir=processed_dir,
    )
    assert len(records) == 1
    assert records[0].question == "Can a landlord enter a rental property without notice?"
    assert records[0].original_answer.startswith("Most jurisdictions")
    assert records[0].licensing.license_status == "unresolved"
    assert records[0].licensing.redistribution_permitted is False
