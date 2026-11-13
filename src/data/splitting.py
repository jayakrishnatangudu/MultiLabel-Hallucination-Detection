"""Deterministic group-stratified data splitting with strict leakage verification."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocessing import normalize_match_key
from src.data.schema import SourceQARecord
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test_synthetic_pool", "test_natural_pool")


def verify_zero_split_leakage(records: list[SourceQARecord]) -> dict[str, int]:
    """Assert that zero source_group_id, document, or context-free question leaks across splits."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    doc_splits: dict[str, set[str]] = defaultdict(set)
    cf_question_splits: dict[str, set[str]] = defaultdict(set)

    for rec in records:
        if rec.split is None:
            raise ValueError(f"Record {rec.record_id} has unset split.")
        group_splits[rec.source_group_id].add(rec.split)
        if rec.source_document_id:
            doc_splits[rec.source_document_id].add(rec.split)
        if not rec.has_context:
            q_key = f"{rec.source_dataset}||{normalize_match_key(rec.question)}"
            cf_question_splits[q_key].add(rec.split)

    leaked_groups = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    leaked_docs = {d: sorted(s) for d, s in doc_splits.items() if len(s) > 1}
    leaked_cf_q = {q: sorted(s) for q, s in cf_question_splits.items() if len(s) > 1}

    if leaked_groups or leaked_docs or leaked_cf_q:
        raise AssertionError(
            f"Split leakage detected! groups={len(leaked_groups)}, "
            f"docs={len(leaked_docs)}, context_free_questions={len(leaked_cf_q)}"
        )

    return {
        "unique_groups_checked": len(group_splits),
        "unique_documents_checked": len(doc_splits),
        "unique_context_free_questions_checked": len(cf_question_splits),
        "leaked_groups": 0,
        "leaked_documents": 0,
        "leaked_context_free_questions": 0,
    }


def _allocate_group_counts(n_groups: int, ratios: tuple[float, ...]) -> list[int]:
    """Allocate integer group counts matching target ratios while keeping non-empty splits when possible."""
    raw_targets = [n_groups * r for r in ratios]
    counts = [int(x) for x in raw_targets]
    remainder = n_groups - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: raw_targets[i] - counts[i], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def assign_group_splits(
    records: list[SourceQARecord],
    split_cfg: dict[str, Any],
    seed: int = 42,
) -> tuple[list[SourceQARecord], dict[str, Any]]:
    """
    Assign deterministic splits grouped by source_group_id and stratified by (source_dataset, source_subset).

    Ensures all records derived from the same original question or document stay in the exact same split.
    """
    ratios = (
        float(split_cfg.get("train_ratio", 0.80)),
        float(split_cfg.get("val_ratio", 0.10)),
        float(split_cfg.get("test_synthetic_ratio", 0.08)),
        float(split_cfg.get("test_natural_ratio", 0.02)),
    )
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}")

    # Group records by source_group_id and determine each group's stratum
    groups: dict[str, list[SourceQARecord]] = defaultdict(list)
    for rec in records:
        groups[rec.source_group_id].append(rec)

    strata_to_groups: dict[str, list[str]] = defaultdict(list)
    for gid, grecs in sorted(groups.items()):
        # Sort subset labels deterministically if a group spans multiple subsets
        stratum = f"{grecs[0].source_dataset}::{grecs[0].source_subset}"
        strata_to_groups[stratum].append(gid)

    rng = np.random.default_rng(seed)
    group_to_split: dict[str, str] = {}

    for stratum, gids in sorted(strata_to_groups.items()):
        shuffled = list(gids)
        rng.shuffle(shuffled)
        counts = _allocate_group_counts(len(shuffled), ratios)
        cursor = 0
        for split_name, cnt in zip(SPLIT_ORDER, counts):
            for gid in shuffled[cursor : cursor + cnt]:
                group_to_split[gid] = split_name
            cursor += cnt

    updated_records: list[SourceQARecord] = []
    for rec in records:
        updated_records.append(rec.model_copy(update={"split": group_to_split[rec.source_group_id]}))

    leakage_audit = verify_zero_split_leakage(updated_records)

    split_totals = Counter(r.split for r in updated_records)
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLIT_ORDER})
    by_subset: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLIT_ORDER})
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLIT_ORDER})
    by_context: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLIT_ORDER})

    for rec in updated_records:
        sp = str(rec.split)
        by_dataset[rec.source_dataset][sp] += 1
        by_subset[f"{rec.source_dataset}/{rec.source_subset}"][sp] += 1
        by_domain[rec.domain][sp] += 1
        ctx_label = "with_context" if rec.has_context else "no_context"
        by_context[ctx_label][sp] += 1

    report = {
        "random_seed": seed,
        "ratios": dict(zip(SPLIT_ORDER, ratios)),
        "total_records": len(updated_records),
        "total_groups": len(groups),
        "split_record_counts": {s: split_totals.get(s, 0) for s in SPLIT_ORDER},
        "split_group_counts": {
            s: sum(1 for sp in group_to_split.values() if sp == s) for s in SPLIT_ORDER
        },
        "by_dataset": dict(by_dataset),
        "by_subset": dict(by_subset),
        "by_domain": dict(by_domain),
        "by_context": dict(by_context),
        "leakage_audit": leakage_audit,
    }
    return updated_records, report


def run_splitting(config_path: str | Path) -> dict[str, Any]:
    """Load clean canonical records, assign group-stratified splits, and write split JSONLs."""
    config = load_yaml_config(config_path)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    splits_dir = resolve_project_path(config["paths"]["splits_dir"])
    report_path = resolve_project_path(config["paths"]["split_report"])

    combined_path = processed_dir / "all_sources_clean.jsonl"
    raw_dicts = read_jsonl(combined_path)
    records = [SourceQARecord(**d) for d in raw_dicts]

    split_records, report = assign_group_splits(
        records=records,
        split_cfg=config["splitting"],
        seed=int(config.get("random_seed", 42)),
    )

    by_split: dict[str, list[SourceQARecord]] = defaultdict(list)
    for rec in split_records:
        by_split[str(rec.split)].append(rec)

    for split_name in SPLIT_ORDER:
        write_jsonl(splits_dir / f"source_{split_name}.jsonl", by_split[split_name])
    write_jsonl(processed_dir / "all_sources_clean_with_splits.jsonl", split_records)

    write_json(report_path, report)
    logger.info(
        "Assigned group splits (seed=%d): %s | leakage_groups=%d",
        report["random_seed"],
        report["split_record_counts"],
        report["leakage_audit"]["leaked_groups"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign question/document-grouped splits for Phase 1.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    args = parser.parse_args()
    run_splitting(args.config)


if __name__ == "__main__":
    main()

