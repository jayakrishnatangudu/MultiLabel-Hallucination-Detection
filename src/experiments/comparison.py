"""Multi-seed aggregation, statistical comparison tests, and automated Markdown/CSV/LaTeX result tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.schema import HALLUCINATION_LABELS
from src.evaluation.metrics import (
    binarize_probabilities,
    paired_bootstrap_significance,
    wilcoxon_paired_test,
)
from src.utils.io import read_json, utc_now_iso, write_json

SHORT_CATS = ("FI", "FF", "II", "LI", "CI")


def aggregate_multi_seed_metrics(
    seed_eval_dicts: Sequence[dict[str, Any]],
    *,
    metric_section: str = "metrics_calibrated_val_tuned_tau",
) -> dict[str, Any]:
    """
    Aggregate mean and sample std across multi-seed evaluation payloads (e.g., seeds 42, 101, 202).
    """
    if not seed_eval_dicts:
        return {"n_seeds": 0, "status": "PENDING_EMPIRICAL_RUN"}

    overall_keys = (
        "macro_f1",
        "micro_f1",
        "hamming_loss",
        "subset_accuracy",
        "mean_pr_auc",
        "mean_roc_auc",
    )
    cal_keys = ("mean_ece", "mean_brier_score")
    agg_overall: dict[str, dict[str, float | None]] = {}

    for k in overall_keys:
        vals = [
            float(d[metric_section]["overall"][k])
            for d in seed_eval_dicts
            if d.get(metric_section, {}).get("overall", {}).get(k) is not None
        ]
        if vals:
            agg_overall[k] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
            }
        else:
            agg_overall[k] = {"mean": None, "std": None}

    agg_cal: dict[str, dict[str, float]] = {}
    for k in cal_keys:
        vals = [float(d[metric_section]["calibration"][k]) for d in seed_eval_dicts]
        agg_cal[k] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
        }

    per_label_f1: dict[str, dict[str, float]] = {}
    for lbl in HALLUCINATION_LABELS:
        vals = [float(d[metric_section]["per_label"][lbl]["f1"]) for d in seed_eval_dicts]
        per_label_f1[lbl] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
        }

    b1_vals = [float(d[metric_section]["task_b1_binary"]["f1"]) for d in seed_eval_dicts]
    return {
        "n_seeds": len(seed_eval_dicts),
        "status": "COMPLETED",
        "overall": agg_overall,
        "calibration": agg_cal,
        "per_label_f1": per_label_f1,
        "task_b1_f1": {
            "mean": round(float(np.mean(b1_vals)), 6),
            "std": round(float(np.std(b1_vals, ddof=1)) if len(b1_vals) > 1 else 0.0, 6),
        },
    }


def _per_sample_f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute per-item F1 for paired Wilcoxon testing."""
    tp = (y_true * y_pred).sum(axis=1).astype(np.float64)
    fp = ((1 - y_true) * y_pred).sum(axis=1).astype(np.float64)
    fn = (y_true * (1 - y_pred)).sum(axis=1).astype(np.float64)
    denom = 2.0 * tp + fp + fn
    # Exact clean 0/0 match gets F1=1.0 cleanly aligned with sklearn zero_division behavior
    return np.where(denom > 0.0, (2.0 * tp) / np.maximum(denom, 1.0), 1.0)


def compare_two_models(
    *,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    model_a_name: str,
    model_b_name: str,
    thresholds_a: float | Sequence[float] | np.ndarray = 0.50,
    thresholds_b: float | Sequence[float] | np.ndarray = 0.50,
    group_ids: Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run paired cluster bootstrap, McNemar exact/chi2, and Wilcoxon signed-rank comparisons."""
    boot_sig = paired_bootstrap_significance(
        y_true,
        prob_a,
        prob_b,
        thresholds_a=thresholds_a,
        thresholds_b=thresholds_b,
        group_ids=group_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    pa = binarize_probabilities(prob_a, thresholds_a)
    pb = binarize_probabilities(prob_b, thresholds_b)
    wil = wilcoxon_paired_test(
        _per_sample_f1_scores(y_true, pa),
        _per_sample_f1_scores(y_true, pb),
    )
    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "paired_bootstrap": boot_sig,
        "wilcoxon_per_sample_f1": wil,
    }


def _fmt_mean_std(entry: dict[str, Any] | None) -> str:
    if not entry or entry.get("mean") is None:
        return "PENDING_EMPIRICAL_RUN"
    mean = float(entry["mean"])
    std = float(entry.get("std") or 0.0)
    return f"{mean:.4f} ± {std:.4f}"


def generate_results_tables(
    model_summaries: Sequence[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Path]:
    """
    Auto-generate Markdown, CSV, and LaTeX benchmark tables directly from serialized JSON metrics.

    Eliminates manual number transcription; missing runs render explicitly as PENDING_EMPIRICAL_RUN.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "comparison_summary.csv"
    md_path = out / "comparison_tables.md"
    tex_path = out / "comparison_tables.tex"

    headers = [
        "Model / Run",
        "Split",
        "Seeds",
        "Macro-F1",
        "Micro-F1",
        "Subset Acc",
        "Hamming Loss",
        "Mean PR-AUC",
        "Mean ECE",
        "Mean Brier",
        "FI F1",
        "FF F1",
        "II F1",
        "LI F1",
        "CI F1",
    ]
    rows: list[list[str]] = []
    for item in model_summaries:
        name = str(item["model_name"])
        split_tag = str(item.get("split_tag", "test_synthetic"))
        agg = item.get("aggregated") or {}
        if agg.get("status") != "COMPLETED":
            rows.append([name, split_tag, str(agg.get("n_seeds", 0))] + ["PENDING_EMPIRICAL_RUN"] * (len(headers) - 3))
            continue
        ov = agg["overall"]
        cal = agg["calibration"]
        pl = agg["per_label_f1"]
        rows.append(
            [
                name,
                split_tag,
                str(agg["n_seeds"]),
                _fmt_mean_std(ov.get("macro_f1")),
                _fmt_mean_std(ov.get("micro_f1")),
                _fmt_mean_std(ov.get("subset_accuracy")),
                _fmt_mean_std(ov.get("hamming_loss")),
                _fmt_mean_std(ov.get("mean_pr_auc")),
                _fmt_mean_std(cal.get("mean_ece")),
                _fmt_mean_std(cal.get("mean_brier_score")),
                *[_fmt_mean_std(pl.get(lbl)) for lbl in HALLUCINATION_LABELS],
            ]
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    md_lines = [
        "# Automated Multi-Label Hallucination Detection Comparison Tables",
        f"_Generated automatically at {utc_now_iso()} — zero manual transcription._",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # LaTeX tabular export
    col_spec = "llc" + "r" * (len(headers) - 3)
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(h.replace("_", "\\_") for h in headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        escaped = [c.replace("_", "\\_").replace("±", "$\\pm$") for c in r]
        tex_lines.append(" & ".join(escaped) + " \\\\")
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Multi-label hallucination detection results across baselines and encoder architectures.}",
            "\\label{tab:main_comparison}",
            "\\end{table*}",
        ]
    )
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    write_json(
        out / "comparison_manifest.json",
        {
            "generated_at_utc": utc_now_iso(),
            "n_models": len(model_summaries),
            "files": {
                "csv": str(csv_path),
                "markdown": str(md_path),
                "latex": str(tex_path),
            },
        },
    )
    return {"csv": csv_path, "markdown": md_path, "latex": tex_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed reports and generate Markdown/CSV/LaTeX tables.")
    parser.add_argument("--report-files", nargs="*", default=[], help="List of eval_report.json or baseline_report.json paths.")
    parser.add_argument("--split-tag", default="test_synthetic")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for rf in args.report_files:
        payload = read_json(rf)
        if "evaluations" in payload and args.split_tag in payload["evaluations"]:
            eval_dict = payload["evaluations"][args.split_tag]
            name = str(payload.get("backbone_key") or Path(rf).parent.name)
            summaries.append(
                {
                    "model_name": name,
                    "split_tag": args.split_tag,
                    "aggregated": aggregate_multi_seed_metrics([eval_dict]),
                }
            )
        elif f"{args.split_tag}_eval" in payload:
            eval_dict = payload[f"{args.split_tag}_eval"]
            name = f"TFIDF_{payload.get('variant', 'baseline')}"
            summaries.append(
                {
                    "model_name": name,
                    "split_tag": args.split_tag,
                    "aggregated": aggregate_multi_seed_metrics([eval_dict]),
                }
            )
    if not summaries:
        summaries = [
            {"model_name": "E1.2_TFIDF_OvR_LogReg", "split_tag": args.split_tag, "aggregated": {"n_seeds": 0, "status": "PENDING_EMPIRICAL_RUN"}},
            {"model_name": "E2.1_DistilBERT", "split_tag": args.split_tag, "aggregated": {"n_seeds": 0, "status": "PENDING_EMPIRICAL_RUN"}},
            {"model_name": "E2.2_RoBERTa_base", "split_tag": args.split_tag, "aggregated": {"n_seeds": 0, "status": "PENDING_EMPIRICAL_RUN"}},
            {"model_name": "E2.3_DeBERTa_v3_base", "split_tag": args.split_tag, "aggregated": {"n_seeds": 0, "status": "PENDING_EMPIRICAL_RUN"}},
            {"model_name": "E2.4_ModernBERT_base", "split_tag": args.split_tag, "aggregated": {"n_seeds": 0, "status": "PENDING_EMPIRICAL_RUN"}},
        ]
    paths = generate_results_tables(summaries, args.output_dir)
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()

