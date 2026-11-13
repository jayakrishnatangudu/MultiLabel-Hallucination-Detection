"""Read-only preprocessing, cleaning, deduplication, and near-duplicate auditing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.schema import LicenseMetadata, ProvenanceMetadata, SourceQARecord
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_json, sha256_json, sha256_text, summarize_lengths, write_json, write_jsonl
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)

_QA_PREFIX_RE = re.compile(r"^\s*(?:Q|A)\s*:\s*", re.IGNORECASE)
_MULTI_QMARK_RE = re.compile(r"(?:\?\s*){2,}$")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_POSSESSIVE_RE = re.compile(r"(?:'|’|`)s\b", re.IGNORECASE)
_SUBTYPE_PREFIXES: frozenset[str] = frozenset(
    {
        "type", "hepatitis", "vitamin", "factor", "group", "class",
        "stage", "grade", "phase", "trisomy", "chromosome", "blood",
    }
)

_QA_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did",
        "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
        "into", "is", "it", "its", "may", "me", "my", "of", "on", "or", "our",
        "should", "so", "than", "that", "the", "their", "them", "there", "these",
        "they", "this", "to", "us", "was", "we", "were", "what", "when", "where",
        "which", "who", "whom", "why", "will", "with", "would", "you", "your",
        "symptoms", "symptom", "cause", "causes", "treatment", "treatments",
        "prevent", "prevention", "diagnose", "diagnosed", "diagnosis", "outlook",
        "risk", "people", "person", "disease", "condition", "inherited", "genetic",
        "information", "explain", "tell", "about", "please",
    }
)


def normalize_display_text(text: str | None, strip_qa_prefix: bool = True) -> str:
    """Normalize whitespace and formatting artifacts without altering meaning."""
    if text is None:
        return ""
    cleaned = str(text).replace("\u00a0", " ").strip()
    if strip_qa_prefix:
        cleaned = _QA_PREFIX_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = _MULTI_QMARK_RE.sub("?", cleaned)
    return cleaned


def normalize_match_key(text: str) -> str:
    """Canonical lowercase alphanumeric key for exact duplicate comparisons."""
    lowered = text.lower()
    stripped = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def word_count(text: str | None) -> int:
    """Return whitespace-delimited word count."""
    if not text:
        return 0
    return len(text.split())


class _DisjointSet:
    """Path-compressed union-find structure for grouping related records."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb
        return True


def _extract_raw_rows_for_dataset(
    dataset_key: str,
    manifest_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read immutable raw files in read-only mode into raw intermediate dicts."""
    raw_items: list[dict[str, Any]] = []
    files = manifest_entry["files"]

    if dataset_key == "medquad":
        fmeta = files[0]
        df = pd.read_csv(fmeta["local_path"], dtype=str, keep_default_na=False)
        for idx, row in df.iterrows():
            raw_items.append(
                {
                    "source_subset": fmeta["subset"],
                    "source_record_id": str(idx),
                    "source_document_id": None,
                    "question": row.get("Question", ""),
                    "context": None,
                    "answer": row.get("Answer", ""),
                    "extra_meta": {"qtype": row.get("qtype", "").strip()},
                    "raw_filename": fmeta["remote_path"],
                    "raw_file_sha256": fmeta["sha256"],
                    "downloaded_at_utc": fmeta["downloaded_at_utc"],
                }
            )
    elif dataset_key == "legal_qa_v1":
        fmeta = files[0]
        df = pd.read_csv(fmeta["local_path"], dtype=str, keep_default_na=False)
        for idx, row in df.iterrows():
            raw_items.append(
                {
                    "source_subset": fmeta["subset"],
                    "source_record_id": str(idx),
                    "source_document_id": None,
                    "question": row.get("question", ""),
                    "context": None,
                    "answer": row.get("answer", ""),
                    "extra_meta": {},
                    "raw_filename": fmeta["remote_path"],
                    "raw_file_sha256": fmeta["sha256"],
                    "downloaded_at_utc": fmeta["downloaded_at_utc"],
                }
            )
    elif dataset_key == "fiqa":
        for fmeta in files:
            with Path(fmeta["local_path"]).open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            for idx, row in enumerate(rows):
                raw_items.append(
                    {
                        "source_subset": fmeta["subset"],
                        "source_record_id": f"{fmeta['subset']}_{idx}",
                        "source_document_id": None,
                        "question": row.get("input", ""),
                        "context": None,
                        "answer": row.get("output", ""),
                        "extra_meta": {"instruction": str(row.get("instruction", "")).strip()},
                        "raw_filename": fmeta["remote_path"],
                        "raw_file_sha256": fmeta["sha256"],
                        "downloaded_at_utc": fmeta["downloaded_at_utc"],
                    }
                )
    elif dataset_key == "pubmedqa":
        for fmeta in files:
            df = pd.read_parquet(fmeta["local_path"])
            for idx, row in df.iterrows():
                ctx_obj = row.get("context")
                if isinstance(ctx_obj, dict):
                    ctx_parts = ctx_obj.get("contexts", [])
                    if hasattr(ctx_parts, "tolist"):
                        ctx_parts = ctx_parts.tolist()
                    ctx_text = "\n".join(str(p).strip() for p in ctx_parts if str(p).strip())
                else:
                    ctx_text = str(ctx_obj or "")
                pubid = str(row.get("pubid", f"{fmeta['subset']}_{idx}"))
                extra: dict[str, Any] = {"expert_labeled": fmeta["subset"] == "pqa_labeled"}
                if "final_decision" in row and pd.notna(row["final_decision"]):
                    extra["final_decision"] = str(row["final_decision"])
                raw_items.append(
                    {
                        "source_subset": fmeta["subset"],
                        "source_record_id": pubid,
                        "source_document_id": f"pubmedqa_doc_{pubid}",
                        "question": "" if pd.isna(row.get("question")) else str(row.get("question")),
                        "context": ctx_text,
                        "answer": "" if pd.isna(row.get("long_answer")) else str(row.get("long_answer")),
                        "extra_meta": extra,
                        "raw_filename": fmeta["remote_path"],
                        "raw_file_sha256": fmeta["sha256"],
                        "downloaded_at_utc": fmeta["downloaded_at_utc"],
                    }
                )
    elif dataset_key == "cuad_qa":
        fmeta = files[0]
        members = manifest_entry["archive_members"]
        with zipfile.ZipFile(fmeta["local_path"], "r") as zf:
            for m_spec in members:
                member_name = m_spec["member_path"]
                subset = m_spec["subset"]
                payload = json.loads(zf.read(member_name).decode("utf-8"))
                for doc in payload.get("data", []):
                    title = str(doc.get("title", "")).strip()
                    doc_id = f"cuad_doc_{sha256_text(title)[:12]}"
                    for para in doc.get("paragraphs", []):
                        ctx_text = str(para.get("context", ""))
                        for qa in para.get("qas", []):
                            answers_list = qa.get("answers") or []
                            unique_spans: list[str] = []
                            seen_spans: set[str] = set()
                            for ans in answers_list:
                                txt = str(ans.get("text", "")).strip()
                                if txt and txt not in seen_spans:
                                    seen_spans.add(txt)
                                    unique_spans.append(txt)
                            joined_ans = "; ".join(unique_spans)
                            raw_items.append(
                                {
                                    "source_subset": subset,
                                    "source_record_id": str(qa.get("id", "")),
                                    "source_document_id": doc_id,
                                    "question": str(qa.get("question", "")),
                                    "context": ctx_text,
                                    "answer": joined_ans,
                                    "extra_meta": {
                                        "contract_title": title,
                                        "is_impossible": bool(qa.get("is_impossible", False)),
                                        "num_answer_spans": len(answers_list),
                                    },
                                    "raw_filename": f"data.zip::{member_name}",
                                    "raw_file_sha256": fmeta["sha256"],
                                    "downloaded_at_utc": fmeta["downloaded_at_utc"],
                                }
                            )
    else:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")

    return raw_items


def _cluster_match_text(text: str) -> str:
    """Normalize text for similarity clustering while stripping English possessives and preserving 1-char identifiers."""
    without_possessive = _POSSESSIVE_RE.sub("", text)
    return normalize_match_key(without_possessive)


def _extract_rare_guard_tokens(text: str, match_text: str) -> set[str]:
    """Extract content tokens plus 1-char biomedical/alphanumeric qualifiers and subtype bigrams."""
    raw_tokens = match_text.split()
    toks = {tok for tok in raw_tokens if tok not in _QA_STOPWORDS}
    for idx in range(len(raw_tokens) - 1):
        prev_tok, cur_tok = raw_tokens[idx], raw_tokens[idx + 1]
        if prev_tok in _SUBTYPE_PREFIXES and (len(cur_tok) == 1 or cur_tok.isdigit() or cur_tok in {"i", "ii", "iii", "iv", "v", "vi", "a", "b"}):
            toks.add(f"{prev_tok}_{cur_tok}")
    if re.search(r"(?<!^)\bA\b", text):
        toks.add("_cap_a")
    return toks


def _cluster_near_duplicate_questions(
    unique_items: list[dict[str, Any]],
    nd_cfg: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Cluster near-duplicate questions using TF-IDF cosine + rare-content-token overlap.

    Designed conservatively so template-similar questions targeting distinct entities
    (e.g., 'What are symptoms of Asthma?' vs 'What are symptoms of Diabetes?',
    or 'Hepatitis A' vs 'Hepatitis B', or 'Q Fever' vs 'Fever') are never merged.
    """
    n = len(unique_items)
    if n <= 1 or not nd_cfg.get("enabled", True):
        return {item["q_key"]: idx for idx, item in enumerate(unique_items)}, {
            "enabled": False,
            "unique_questions_evaluated": n,
            "merged_pairs_count": 0,
            "clusters_with_size_gt_1": 0,
            "max_cluster_size": 1 if n > 0 else 0,
            "sample_merged_pairs": [],
            "sample_borderline_unmerged_pairs": [],
        }

    texts = [item["question_norm"] for item in unique_items]
    match_texts = [_cluster_match_text(t) for t in texts]
    token_lists: list[set[str]] = []
    doc_freq: Counter[str] = Counter()
    for text, mtext in zip(texts, match_texts):
        toks = _extract_rare_guard_tokens(text, mtext)
        token_lists.append(toks)
        doc_freq.update(toks)

    max_df_ratio = float(nd_cfg.get("rare_token_max_df_ratio", 0.02))
    max_df_count = max(2, int(max_df_ratio * n))
    rare_tokens_per_q = [
        {tok for tok in toks if doc_freq[tok] <= max_df_count}
        for toks in token_lists
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(match_texts)

    cos_thresh = float(nd_cfg.get("tfidf_cosine_threshold", 0.90))
    jaccard_thresh = float(nd_cfg.get("min_rare_token_jaccard", 0.65))
    borderline_low = float(nd_cfg.get("borderline_cosine_low", 0.78))
    enforce_qtype = bool(nd_cfg.get("enforce_same_qtype_if_present", True))
    max_audit = int(nd_cfg.get("max_audit_borderline_pairs", 25))

    dsu = _DisjointSet(n)
    merged_pairs: list[dict[str, Any]] = []
    borderline_unmerged: list[dict[str, Any]] = []
    total_merged_pairs = 0
    rejected_by_rare_guard = 0
    rejected_by_qtype_guard = 0

    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sims = (matrix[start:end] @ matrix.T).toarray()
        for local_i in range(end - start):
            i = start + local_i
            row_sims = sims[local_i]
            cand_indices = np.where(row_sims >= borderline_low)[0]
            for j in cand_indices:
                if j <= i:
                    continue
                sim = float(row_sims[j])
                qtype_i = unique_items[i].get("qtype")
                qtype_j = unique_items[j].get("qtype")
                if enforce_qtype and qtype_i and qtype_j and qtype_i != qtype_j:
                    if sim >= cos_thresh:
                        rejected_by_qtype_guard += 1
                    continue

                rare_i = rare_tokens_per_q[i]
                rare_j = rare_tokens_per_q[j]
                union_rare = rare_i | rare_j
                if union_rare:
                    rare_jaccard = len(rare_i & rare_j) / len(union_rare)
                else:
                    # Fallback to full content tokens when no rare tokens exist
                    full_union = token_lists[i] | token_lists[j]
                    rare_jaccard = (
                        len(token_lists[i] & token_lists[j]) / len(full_union)
                        if full_union
                        else 0.0
                    )

                if sim >= cos_thresh and rare_jaccard >= jaccard_thresh:
                    dsu.union(i, j)
                    total_merged_pairs += 1
                    if len(merged_pairs) < max_audit:
                        merged_pairs.append(
                            {
                                "q1": texts[i],
                                "q2": texts[j],
                                "cosine": round(sim, 4),
                                "rare_token_jaccard": round(rare_jaccard, 4),
                            }
                        )
                else:
                    if sim >= cos_thresh:
                        rejected_by_rare_guard += 1
                    if len(borderline_unmerged) < max_audit:
                        borderline_unmerged.append(
                            {
                                "q1": texts[i],
                                "q2": texts[j],
                                "cosine": round(sim, 4),
                                "rare_token_jaccard": round(rare_jaccard, 4),
                                "reason": "low_rare_overlap" if sim >= cos_thresh else "below_cosine_threshold",
                            }
                        )

    cluster_sizes = Counter(dsu.find(i) for i in range(n))
    multi_clusters = [sz for sz in cluster_sizes.values() if sz > 1]
    qkey_to_cluster = {unique_items[i]["q_key"]: dsu.find(i) for i in range(n)}
    audit_summary = {
        "enabled": True,
        "unique_questions_evaluated": n,
        "merged_pairs_count": total_merged_pairs,
        "rejected_by_rare_token_guard": rejected_by_rare_guard,
        "rejected_by_qtype_guard": rejected_by_qtype_guard,
        "clusters_with_size_gt_1": len(multi_clusters),
        "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        "sample_merged_pairs": merged_pairs,
        "sample_borderline_unmerged_pairs": borderline_unmerged,
    }
    return qkey_to_cluster, audit_summary


def preprocess_dataset(
    dataset_key: str,
    manifest_entry: dict[str, Any],
    pre_cfg: dict[str, Any],
    pre_cfg_sha: str,
    processed_dir: Path,
) -> tuple[list[SourceQARecord], dict[str, Any]]:
    """Normalize, clean, deduplicate, group, and validate a single dataset."""
    raw_rows = _extract_raw_rows_for_dataset(dataset_key, manifest_entry)
    strip_prefixes = bool(pre_cfg.get("strip_qa_prefixes", True))
    min_q_chars = int(pre_cfg.get("min_question_chars", 10))
    min_a_chars = int(pre_cfg.get("min_answer_chars", 15))
    min_q_words = int(pre_cfg.get("min_question_words", 3))
    min_a_words = int(pre_cfg.get("min_answer_words", 3))
    max_a_words = int(pre_cfg.get("max_answer_words", 1500))

    attrition: Counter[str] = Counter()
    attrition["raw_total_rows"] = len(raw_rows)

    missing_raw_stats: Counter[str] = Counter()
    cleaned_candidates: list[dict[str, Any]] = []
    seen_exact_tuples: set[tuple[str, str, str]] = set()
    subset_counts_raw: Counter[str] = Counter()
    subset_counts_final: Counter[str] = Counter()

    for row in raw_rows:
        subset_counts_raw[row["source_subset"]] += 1
        q_raw = row["question"]
        a_raw = row["answer"]
        c_raw = row["context"]

        if not str(q_raw).strip():
            missing_raw_stats["missing_question_raw"] += 1
        if not str(a_raw).strip():
            missing_raw_stats["missing_answer_raw"] += 1
        if manifest_entry["has_context"] and (c_raw is None or not str(c_raw).strip()):
            missing_raw_stats["missing_context_raw"] += 1

        if dataset_key == "cuad_qa" and row["extra_meta"].get("is_impossible"):
            attrition["dropped_unanswerable_cuad_impossible"] += 1
            continue

        q_norm = normalize_display_text(q_raw, strip_qa_prefix=strip_prefixes)
        a_norm = normalize_display_text(a_raw, strip_qa_prefix=strip_prefixes)
        c_norm = normalize_display_text(c_raw, strip_qa_prefix=False) if c_raw is not None else None

        if not q_norm or not a_norm:
            attrition["dropped_empty_question_or_answer"] += 1
            continue
        if manifest_entry["has_context"] and not c_norm:
            attrition["dropped_empty_context"] += 1
            continue
        if (
            len(q_norm) < min_q_chars
            or word_count(q_norm) < min_q_words
            or len(a_norm) < min_a_chars
            or word_count(a_norm) < min_a_words
        ):
            attrition["dropped_too_short"] += 1
            continue
        if word_count(a_norm) > max_a_words:
            attrition["dropped_too_long_answer"] += 1
            continue

        q_key = normalize_match_key(q_norm)
        if not manifest_entry["has_context"] and not any(
            tok not in _QA_STOPWORDS for tok in q_key.split()
        ):
            attrition["dropped_empty_topic_template"] += 1
            continue
        c_key = normalize_match_key(c_norm or "")
        a_key = normalize_match_key(a_norm)

        exact_tuple = (q_key, c_key, a_key)
        if exact_tuple in seen_exact_tuples:
            attrition["dropped_exact_tuple_duplicates"] += 1
            continue
        seen_exact_tuples.add(exact_tuple)

        cleaned_candidates.append(
            {
                **row,
                "question_norm": q_norm,
                "context_norm": c_norm,
                "answer_norm": a_norm,
                "q_key": q_key,
                "c_key": c_key,
                "qc_key": f"{q_key}||{c_key}",
            }
        )

    # Track identical (question, context) rows that have distinct valid answers
    qc_counter = Counter(item["qc_key"] for item in cleaned_candidates)
    same_qc_multi_answer_rows = sum(cnt - 1 for cnt in qc_counter.values() if cnt > 1)
    same_qc_multi_answer_groups = sum(1 for cnt in qc_counter.values() if cnt > 1)

    # Assign source_group_id:
    # - Document-grounded datasets (pubmedqa, cuad_qa) group by source_document_id, plus link identical q_keys in pubmedqa
    # - Context-free QA datasets (medquad, legal_qa_v1, fiqa) group by exact q_key + audited near-duplicate cluster
    if manifest_entry["has_context"]:
        if dataset_key == "pubmedqa":
            dsu = _DisjointSet(len(cleaned_candidates))
            doc_first: dict[str, int] = {}
            q_first: dict[str, int] = {}
            for idx, item in enumerate(cleaned_candidates):
                doc_id = item["source_document_id"]
                if doc_id in doc_first:
                    dsu.union(doc_first[doc_id], idx)
                else:
                    doc_first[doc_id] = idx
                qk = item["q_key"]
                if qk in q_first:
                    dsu.union(q_first[qk], idx)
                else:
                    q_first[qk] = idx
            group_ids = [f"grp_{dataset_key}_{dsu.find(idx):06d}" for idx in range(len(cleaned_candidates))]
        else:
            # cuad_qa shares 41 template questions across distinct contracts: group strictly by contract document
            group_ids = [
                f"grp_{item['source_document_id']}"
                for item in cleaned_candidates
            ]
        nd_audit = {
            "enabled": False,
            "reason": "Document-grounded dataset grouped primarily by source_document_id to avoid merging template questions across distinct contracts/abstracts.",
            "unique_documents": len({item["source_document_id"] for item in cleaned_candidates}),
        }
    else:
        unique_q_items: list[dict[str, Any]] = []
        seen_q_keys: set[str] = set()
        for item in cleaned_candidates:
            if item["q_key"] not in seen_q_keys:
                seen_q_keys.add(item["q_key"])
                unique_q_items.append(
                    {
                        "q_key": item["q_key"],
                        "question_norm": item["question_norm"],
                        "qtype": item["extra_meta"].get("qtype"),
                    }
                )
        qkey_to_cluster, nd_audit = _cluster_near_duplicate_questions(
            unique_q_items, pre_cfg.get("near_duplicate", {})
        )
        group_ids = [
            f"grp_{dataset_key}_{qkey_to_cluster[item['q_key']]:06d}"
            for item in cleaned_candidates
        ]

    licensing_model = LicenseMetadata(**manifest_entry["licensing"])
    records: list[SourceQARecord] = []
    q_words: list[int] = []
    a_words: list[int] = []
    c_words: list[int] = []

    for idx, item in enumerate(cleaned_candidates):
        qw = word_count(item["question_norm"])
        aw = word_count(item["answer_norm"])
        cw = word_count(item["context_norm"])
        q_words.append(qw)
        a_words.append(aw)
        if item["context_norm"]:
            c_words.append(cw)

        rec = SourceQARecord(
            record_id=f"{dataset_key}_{idx:06d}",
            source_dataset=dataset_key,
            source_subset=item["source_subset"],
            source_record_id=item["source_record_id"],
            source_document_id=item["source_document_id"],
            source_group_id=group_ids[idx],
            domain=manifest_entry["domain"],
            has_context=bool(manifest_entry["has_context"]),
            question=item["question_norm"],
            context=item["context_norm"],
            original_answer=item["answer_norm"],
            metadata={
                **item["extra_meta"],
                "question_words": qw,
                "answer_words": aw,
                "context_words": cw,
                "question_chars": len(item["question_norm"]),
                "answer_chars": len(item["answer_norm"]),
                "context_chars": len(item["context_norm"] or ""),
            },
            provenance=ProvenanceMetadata(
                source_url=manifest_entry["source_url"],
                upstream_url=manifest_entry["upstream_url"],
                dataset_version=manifest_entry["dataset_version"],
                downloaded_at_utc=item["downloaded_at_utc"],
                raw_filename=item["raw_filename"],
                raw_file_sha256=item["raw_file_sha256"],
                preprocessing_config_sha256=pre_cfg_sha,
            ),
            licensing=licensing_model,
        )
        records.append(rec)
        subset_counts_final[item["source_subset"]] += 1

    out_path = processed_dir / f"{dataset_key}.jsonl"
    write_jsonl(out_path, records)

    dataset_stats = {
        "dataset_key": dataset_key,
        "domain": manifest_entry["domain"],
        "has_context": bool(manifest_entry["has_context"]),
        "processed_path": str(out_path),
        "raw_total_rows": attrition["raw_total_rows"],
        "final_clean_rows": len(records),
        "total_dropped_rows": attrition["raw_total_rows"] - len(records),
        "retention_rate": round(len(records) / attrition["raw_total_rows"], 4) if attrition["raw_total_rows"] else 0.0,
        "subset_counts_raw": dict(subset_counts_raw),
        "subset_counts_final": dict(subset_counts_final),
        "attrition_breakdown": dict(attrition),
        "missing_raw_fields": dict(missing_raw_stats),
        "exact_qc_multi_answer_extra_rows": same_qc_multi_answer_rows,
        "exact_qc_multi_answer_groups": same_qc_multi_answer_groups,
        "unique_source_groups": len({r.source_group_id for r in records}),
        "near_duplicate_audit": nd_audit,
        "length_stats_words": {
            "question": summarize_lengths(q_words),
            "answer": summarize_lengths(a_words),
            "context": summarize_lengths(c_words),
        },
        "licensing": manifest_entry["licensing"],
    }
    logger.info(
        "Preprocessed %-12s | raw=%6d -> clean=%6d (groups=%6d, exact_dups_dropped=%d)",
        dataset_key,
        dataset_stats["raw_total_rows"],
        dataset_stats["final_clean_rows"],
        dataset_stats["unique_source_groups"],
        attrition["dropped_exact_tuple_duplicates"],
    )
    return records, dataset_stats


def run_preprocessing(config_path: str | Path) -> dict[str, Any]:
    """Run read-only preprocessing across all datasets in the raw manifest."""
    config = load_yaml_config(config_path)
    manifest_path = resolve_project_path(config["paths"]["raw_manifest"])
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    report_path = resolve_project_path(config["paths"]["preprocessing_report"])

    manifest = read_json(manifest_path)
    pre_cfg = config["preprocessing"]
    pre_cfg_sha = sha256_json(pre_cfg)

    all_records: list[SourceQARecord] = []
    per_dataset_stats: dict[str, Any] = {}

    for dataset_key, manifest_entry in manifest["datasets"].items():
        records, stats = preprocess_dataset(
            dataset_key=dataset_key,
            manifest_entry=manifest_entry,
            pre_cfg=pre_cfg,
            pre_cfg_sha=pre_cfg_sha,
            processed_dir=processed_dir,
        )
        all_records.extend(records)
        per_dataset_stats[dataset_key] = stats

    combined_path = processed_dir / "all_sources_clean.jsonl"
    write_jsonl(combined_path, all_records)

    report = {
        "schema_version": "1.0",
        "preprocessing_config": pre_cfg,
        "preprocessing_config_sha256": pre_cfg_sha,
        "total_raw_rows": sum(s["raw_total_rows"] for s in per_dataset_stats.values()),
        "total_final_clean_rows": len(all_records),
        "total_unique_groups": len({r.source_group_id for r in all_records}),
        "combined_processed_path": str(combined_path),
        "datasets": per_dataset_stats,
    }
    write_json(report_path, report)
    logger.info("Saved preprocessing report: %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess immutable raw datasets into canonical JSONL.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    args = parser.parse_args()
    run_preprocessing(args.config)


if __name__ == "__main__":
    main()

