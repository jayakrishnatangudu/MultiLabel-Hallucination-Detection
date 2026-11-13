"""Generate human-readable Markdown data quality, licensing, and split report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_json
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def build_phase1_markdown_report(
    manifest: dict[str, Any],
    pre_report: dict[str, Any],
    split_report: dict[str, Any],
) -> str:
    """Render a comprehensive Markdown report from Phase 1 JSON artifacts."""
    lines: list[str] = [
        "# Phase 1 Dataset Acquisition, Preprocessing & Quality Report",
        "",
        f"- **Raw Manifest Timestamp (UTC)**: `{manifest['generated_at_utc']}`",
        f"- **Preprocessing Config SHA-256**: `{pre_report['preprocessing_config_sha256']}`",
        f"- **Split Random Seed**: `{split_report['random_seed']}`",
        f"- **Total Raw Records**: `{pre_report['total_raw_rows']:,}`",
        f"- **Total Cleaned Records**: `{pre_report['total_final_clean_rows']:,}`",
        f"- **Total Unique Question/Document Groups**: `{pre_report['total_unique_groups']:,}`",
        "",
        "## 1. Dataset-Wise Attrition & Length Statistics",
        "",
        "| Dataset | Domain | Context | Raw Rows | Clean Rows | Dropped | Retention | Unique Groups | Median Q/A/C Words |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for key, ds in pre_report["datasets"].items():
        ls = ds["length_stats_words"]
        q_med = ls["question"]["median"]
        a_med = ls["answer"]["median"]
        c_med = ls["context"]["median"] if ds["has_context"] else "—"
        lines.append(
            f"| `{key}` | {ds['domain']} | {ds['has_context']} | {ds['raw_total_rows']:,} | "
            f"{ds['final_clean_rows']:,} | {ds['total_dropped_rows']:,} | "
            f"{ds['retention_rate']*100:.1f}% | {ds['unique_source_groups']:,} | "
            f"{q_med} / {a_med} / {c_med} |"
        )

    lines.extend(
        [
            "",
            "### Attrition & Missingness Details",
            "",
            "| Dataset | Missing Raw Q / A / C | CUAD Unanswerable | Empty Q/A/C | Too Short | Too Long | Exact Tuple Dups | Multi-Answer Same (Q,C) Groups |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, ds in pre_report["datasets"].items():
        m = ds["missing_raw_fields"]
        a = ds["attrition_breakdown"]
        lines.append(
            f"| `{key}` | {m.get('missing_question_raw', 0)} / {m.get('missing_answer_raw', 0)} / {m.get('missing_context_raw', 0)} | "
            f"{a.get('dropped_unanswerable_cuad_impossible', 0):,} | "
            f"{a.get('dropped_empty_question_or_answer', 0) + a.get('dropped_empty_context', 0):,} | "
            f"{a.get('dropped_too_short', 0):,} | {a.get('dropped_too_long_answer', 0):,} | "
            f"{a.get('dropped_exact_tuple_duplicates', 0):,} | "
            f"{ds['exact_qc_multi_answer_groups']:,} ({ds['exact_qc_multi_answer_extra_rows']:,} extra rows) |"
        )

    lines.extend(["", "## 2. Near-Duplicate Clustering Audit", ""])
    for key, ds in pre_report["datasets"].items():
        nd = ds["near_duplicate_audit"]
        if not nd.get("enabled"):
            lines.append(f"- **`{key}`**: {nd.get('reason', 'Disabled')} (unique documents: `{nd.get('unique_documents', 'n/a')}`).")
            continue
        lines.append(
            f"- **`{key}`**: evaluated `{nd['unique_questions_evaluated']:,}` unique questions -> "
            f"merged `{nd['merged_pairs_count']:,}` near-duplicate pairs into `{nd['clusters_with_size_gt_1']:,}` multi-member clusters "
            f"(max cluster size `{nd['max_cluster_size']}`), while rejecting `{nd['rejected_by_rare_token_guard']:,}` "
            f"template-similar pairs via rare-token guard and `{nd['rejected_by_qtype_guard']:,}` via `qtype` mismatch."
        )
        if nd["sample_merged_pairs"]:
            lines.append("  - *Sample merged pairs*:")
            for p in nd["sample_merged_pairs"][:5]:
                lines.append(
                    f"    - `{p['cosine']}` (rare Jaccard `{p['rare_token_jaccard']}`): "
                    f"\"{p['q1']}\" <-> \"{p['q2']}\""
                )
        if nd["sample_borderline_unmerged_pairs"]:
            lines.append("  - *Sample borderline unmerged pairs (verified distinct entities/topics)*:")
            for p in nd["sample_borderline_unmerged_pairs"][:5]:
                lines.append(
                    f"    - `{p['cosine']}` (rare Jaccard `{p['rare_token_jaccard']}`, {p['reason']}): "
                    f"\"{p['q1']}\" != \"{p['q2']}\""
                )

    lines.extend(
        [
            "",
            "## 3. Source Licensing, Attribution & Redistribution Status",
            "",
            "| Dataset | Version (Commit SHA) | HF Card Tag | Verified Upstream License | Status | Redistribute? | Required Attribution |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for key, entry in manifest["datasets"].items():
        lic = entry["licensing"]
        lines.append(
            f"| `{key}` | `{entry['dataset_version'][:12]}` | `{lic['hf_card_license']}` | "
            f"`{lic['upstream_license']}` | `{lic['license_status']}` | `{lic['redistribution_permitted']}` | "
            f"{lic['attribution']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Question/Document-Grouped Data Split Distribution",
            "",
            f"- **Leakage Check**: `{split_report['leakage_audit']}`",
            "",
            "| Dataset / Subset | Train (80%) | Val (10%) | Test Synthetic Pool (8%) | Test Natural Pool (2%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for subset_key, counts in split_report["by_subset"].items():
        lines.append(
            f"| `{subset_key}` | {counts['train']:,} | {counts['val']:,} | "
            f"{counts['test_synthetic_pool']:,} | {counts['test_natural_pool']:,} |"
        )
    tot = split_report["split_record_counts"]
    lines.append(
        f"| **Total Records** | **{tot['train']:,}** | **{tot['val']:,}** | "
        f"**{tot['test_synthetic_pool']:,}** | **{tot['test_natural_pool']:,}** |"
    )
    gtot = split_report["split_group_counts"]
    lines.append(
        f"| **Total Groups** | **{gtot['train']:,}** | **{gtot['val']:,}** | "
        f"**{gtot['test_synthetic_pool']:,}** | **{gtot['test_natural_pool']:,}** |"
    )
    lines.append("")
    return "\n".join(lines)


def generate_phase1_report(config_path: str | Path) -> Path:
    """Read Phase 1 JSON artifacts and write the Markdown report."""
    config = load_yaml_config(config_path)
    manifest = read_json(resolve_project_path(config["paths"]["raw_manifest"]))
    pre_report = read_json(resolve_project_path(config["paths"]["preprocessing_report"]))
    split_report = read_json(resolve_project_path(config["paths"]["split_report"]))

    markdown = build_phase1_markdown_report(manifest, pre_report, split_report)
    out_path = resolve_project_path(config["paths"]["markdown_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote Phase 1 data report: %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 Markdown quality report.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    args = parser.parse_args()
    generate_phase1_report(args.config)


if __name__ == "__main__":
    main()

