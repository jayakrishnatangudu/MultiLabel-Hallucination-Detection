"""Phase 3 Suite E1 classical TF-IDF baselines and zero-shot/few-shot LLM detector evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import joblib
import numpy as np
from scipy import special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.data.schema import HALLUCINATION_LABELS, MultiLabelRecord
from src.data.token_budget import TokenBudgetAllocator, summarize_truncation_rates
from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_multilabel_metrics,
    compute_sliced_e4_metrics,
    resolve_ground_truth_matrix,
)
from src.evaluation.visualizations import (
    plot_combination_breakdown,
    plot_label_cooccurrence_heatmap,
    plot_per_label_pr_curves,
    plot_reliability_diagrams,
)
from src.generation.backend import LLMClient, build_live_client
from src.generation.prompt_builder import format_context_block, load_prompt_template
from src.generation.validators import extract_json_object
from src.models.calibration import TemperatureScaler, probs_to_logits
from src.training.thresholds import optimize_validation_thresholds, save_thresholds
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_jsonl, utc_now_iso, write_json


class _SafeBinaryWrapper(BaseEstimator, ClassifierMixin):
    """Wrap binary linear models to gracefully handle single-class toy folds in tests."""

    __module__ = "src.models.baseline"

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.constant_value_: int | None = None

    def fit(self, X: Any, y: np.ndarray) -> "_SafeBinaryWrapper":
        uniq = np.unique(y)
        if len(uniq) < 2:
            self.constant_value_ = int(uniq[0]) if len(uniq) == 1 else 0
            self.classes_ = np.array([0, 1], dtype=np.int64)
            return self
        self.constant_value_ = None
        self.estimator.fit(X, y)
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = X.shape[0]
        if self.constant_value_ is not None:
            p1 = np.full(n, 0.99 if self.constant_value_ == 1 else 0.01, dtype=np.float64)
            return np.column_stack([1.0 - p1, p1])
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        df = np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        p1 = special.expit(df)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def build_field_aware_text_inputs(
    records: Sequence[MultiLabelRecord | dict[str, Any]],
    *,
    max_length: int = 512,
    response_budget: int = 288,
    question_budget: int = 96,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Format records through shared TokenBudgetAllocator so E1 and E2 inspect identical kept fields."""
    allocator = TokenBudgetAllocator(
        max_length=max_length,
        response_budget=response_budget,
        question_budget=question_budget,
    )
    texts: list[str] = []
    meta_rows: list[dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, MultiLabelRecord):
            q, r, c = rec.question, rec.generated_answer, rec.context
        else:
            q = str(rec["question"])
            r = str(rec["generated_answer"])
            c = rec.get("context")
        enc = allocator.encode_sample(question=q, response=r, context=c)
        texts.append(enc.formatted_text)
        meta_rows.append(enc.metadata.to_dict())
    return texts, meta_rows


class ClassicalBaselineSuite:
    """
    TF-IDF multi-label classifier family for Experiment Suite E1.2 and E1.3:
      - 'ovr_logreg': Word + char_wb TF-IDF + independent LogisticRegression heads
      - 'linear_svc': Word + char_wb TF-IDF + calibrated LinearSVC per category
      - 'classifier_chain': Word + char_wb TF-IDF + ordered ClassifierChain(LogisticRegression)
    """

    __module__ = "src.models.baseline"

    def __init__(
        self,
        *,
        variant: str = "ovr_logreg",
        max_features_word: int = 25000,
        max_features_char: int = 15000,
        c_reg: float = 1.0,
        class_weight: str | None = "balanced",
        random_seed: int = 42,
        max_length: int = 512,
    ) -> None:
        if variant not in ("ovr_logreg", "linear_svc", "classifier_chain"):
            raise ValueError(f"Unsupported classical baseline variant={variant!r}")
        self.variant = variant
        self.max_features_word = int(max_features_word)
        self.max_features_char = int(max_features_char)
        self.c_reg = float(c_reg)
        self.class_weight = class_weight
        self.random_seed = int(random_seed)
        self.max_length = int(max_length)

        self.vectorizer = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=self.max_features_word,
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=self.max_features_char,
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
            ]
        )
        self.classifier_: Any = None
        self.independent_heads_: list[_SafeBinaryWrapper] | None = None

    def fit(
        self,
        train_records: Sequence[MultiLabelRecord | dict[str, Any]],
        *,
        y_train: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit TF-IDF feature union and chosen linear multi-label estimator strictly on train data."""
        texts, meta = build_field_aware_text_inputs(train_records, max_length=self.max_length)
        yt = resolve_ground_truth_matrix(train_records) if y_train is None else np.asarray(y_train, dtype=np.int64)
        t0 = time.perf_counter()
        X_tr = self.vectorizer.fit_transform(texts)

        if self.variant == "classifier_chain":
            # Guard rare constant columns cleanly while keeping ClassifierChain on all dynamic datasets
            if all(len(np.unique(yt[:, k])) >= 2 for k in range(yt.shape[1])):
                base_lr = LogisticRegression(
                    C=self.c_reg,
                    class_weight=self.class_weight,
                    max_iter=1000,
                    random_state=self.random_seed,
                )
                self.classifier_ = ClassifierChain(base_lr, order="random", random_state=self.random_seed)
                self.classifier_.fit(X_tr, yt)
                self.independent_heads_ = None
            else:
                self.independent_heads_ = [
                    _SafeBinaryWrapper(
                        LogisticRegression(
                            C=self.c_reg,
                            class_weight=self.class_weight,
                            max_iter=1000,
                            random_state=self.random_seed + k,
                        )
                    ).fit(X_tr, yt[:, k])
                    for k in range(yt.shape[1])
                ]
        else:
            heads: list[_SafeBinaryWrapper] = []
            for k in range(yt.shape[1]):
                if self.variant == "ovr_logreg":
                    est: Any = LogisticRegression(
                        C=self.c_reg,
                        class_weight=self.class_weight,
                        max_iter=1000,
                        random_state=self.random_seed + k,
                    )
                else:
                    n_pos = int((yt[:, k] == 1).sum())
                    n_neg = int((yt[:, k] == 0).sum())
                    svc = LinearSVC(
                        C=self.c_reg,
                        class_weight=self.class_weight,
                        random_state=self.random_seed + k,
                        max_iter=3000,
                    )
                    if min(n_pos, n_neg) >= 3:
                        est = CalibratedClassifierCV(svc, cv=min(3, min(n_pos, n_neg)), method="sigmoid")
                    else:
                        est = svc
                heads.append(_SafeBinaryWrapper(est).fit(X_tr, yt[:, k]))
            self.independent_heads_ = heads
            self.classifier_ = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "variant": self.variant,
            "n_train": int(len(texts)),
            "n_features": int(X_tr.shape[1]),
            "fit_time_ms": round(elapsed_ms, 2),
            "train_truncation_summary": summarize_truncation_rates(meta),
        }

    def predict_scores_and_logits(
        self,
        records: Sequence[MultiLabelRecord | dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
        """Return `(probs, logits, token_metadata, latency_ms_per_sample)`."""
        texts, meta = build_field_aware_text_inputs(records, max_length=self.max_length)
        t0 = time.perf_counter()
        X = self.vectorizer.transform(texts)
        if self.classifier_ is not None:
            probs = np.asarray(self.classifier_.predict_proba(X), dtype=np.float64)
        elif self.independent_heads_ is not None:
            cols = [head.predict_proba(X)[:, 1] for head in self.independent_heads_]
            probs = np.column_stack(cols)
        else:
            raise RuntimeError("ClassicalBaselineSuite must be fitted before calling predict.")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logits = probs_to_logits(probs)
        return probs, logits, meta, elapsed_ms / max(1, len(texts))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalBaselineSuite":
        import sys

        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            if not hasattr(main_mod, "ClassicalBaselineSuite"):
                setattr(main_mod, "ClassicalBaselineSuite", cls)
            if not hasattr(main_mod, "_SafeBinaryWrapper"):
                setattr(main_mod, "_SafeBinaryWrapper", _SafeBinaryWrapper)
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object from {path} is {type(obj)}, expected ClassicalBaselineSuite")
        return obj


def _parse_zero_shot_predictions(raw_text: str) -> tuple[np.ndarray, bool]:
    """Extract ordered 5-dim binary prediction vector from structured zero-shot JSON response."""
    obj = extract_json_object(raw_text)
    if obj is None:
        return np.zeros(len(HALLUCINATION_LABELS), dtype=np.float64), False
    labels_obj = obj.get("labels", obj)
    if not isinstance(labels_obj, dict):
        return np.zeros(len(HALLUCINATION_LABELS), dtype=np.float64), False
    probs: list[float] = []
    for lbl in HALLUCINATION_LABELS:
        raw_val = labels_obj.get(lbl, 0)
        if isinstance(raw_val, bool):
            probs.append(1.0 if raw_val else 0.0)
        elif isinstance(raw_val, (int, float)):
            v = float(raw_val)
            probs.append(float(np.clip(v, 0.0, 1.0)))
        else:
            probs.append(1.0 if str(raw_val).strip().lower() in ("1", "true", "yes") else 0.0)
    return np.asarray(probs, dtype=np.float64), True


class ZeroShotLLMDetectorBaseline:
    """
    Suite E1.5 structured zero-shot LLM hallucination detector using prompts/v1_0/zero_shot_detector.md.

    Supports both live free-tier/configured clients and offline MockLLMClient evaluation.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        prompt_path: str | Path = "prompts/v1_0/zero_shot_detector.md",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.client = client
        self.template = load_prompt_template(prompt_path)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def predict(
        self,
        records: Sequence[MultiLabelRecord | dict[str, Any]],
        *,
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """Run zero-shot detection across candidate records (with deterministic truncation telemetry)."""
        subset = list(records[:max_samples] if max_samples is not None else records)
        probs_rows: list[np.ndarray] = []
        parse_success = 0
        total_cost = 0.0
        latencies: list[float] = []

        for rec in subset:
            if isinstance(rec, MultiLabelRecord):
                domain = rec.domain
                question = rec.question
                has_context = rec.has_context
                context = rec.context
                cand = rec.generated_answer
            else:
                domain = str(rec["domain"])
                question = str(rec["question"])
                has_context = bool(rec["has_context"])
                context = rec.get("context")
                cand = str(rec["generated_answer"])

            sys_msg = self.template.system_prompt
            usr_msg = self.template.user_template.format(
                domain=domain,
                question=question,
                has_context=str(has_context).lower(),
                context_block=format_context_block(context),
                candidate_response=cand,
            )
            resp = self.client.generate(
                system_prompt=sys_msg,
                user_prompt=usr_msg,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            vec, ok = _parse_zero_shot_predictions(resp.raw_text)
            if ok:
                parse_success += 1
            probs_rows.append(vec)
            total_cost += resp.cost_usd
            latencies.append(resp.latency_ms)

        y_prob = np.vstack(probs_rows) if probs_rows else np.zeros((0, len(HALLUCINATION_LABELS)), dtype=np.float64)
        return {
            "y_prob": y_prob,
            "n_samples": len(subset),
            "json_parse_rate": round(parse_success / max(1, len(subset)), 6),
            "total_cost_usd": round(total_cost, 6),
            "mean_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 2),
            "model_id": getattr(self.client, "model_id", "unknown"),
            "provider": getattr(self.client, "provider", "unknown"),
        }


def run_classical_baseline_experiment(
    *,
    train_records: Sequence[MultiLabelRecord | dict[str, Any]],
    val_records: Sequence[MultiLabelRecord | dict[str, Any]],
    test_records: Sequence[MultiLabelRecord | dict[str, Any]],
    output_dir: str | Path,
    variant: str = "ovr_logreg",
    natural_records: Sequence[MultiLabelRecord | dict[str, Any]] | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train TF-IDF baseline on train, fit calibration & thresholds strictly on val,
    and evaluate once on locked test_synthetic (and optional unpooled test_natural).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Assert strict split identities before fitting anything
    val_splits = [r.split if isinstance(r, MultiLabelRecord) else r["split"] for r in val_records]

    model = ClassicalBaselineSuite(variant=variant, random_seed=seed)
    fit_info = model.fit(train_records)
    model.save(out / "baseline_model.joblib")

    y_val = resolve_ground_truth_matrix(val_records)
    val_probs, val_logits, val_meta, val_lat = model.predict_scores_and_logits(val_records)

    scaler = TemperatureScaler(per_label=True)
    cal_report = scaler.fit(val_logits, y_val, split_names=val_splits)
    scaler.save(out / "calibration.json")
    val_cal_probs = scaler.predict_proba(val_logits)

    thr_result = optimize_validation_thresholds(val_cal_probs, y_val, split_names=val_splits)
    save_thresholds(thr_result, out / "thresholds.json")
    tuned_tau = thr_result["threshold_vector"]

    def _eval_split(
        recs: Sequence[MultiLabelRecord | dict[str, Any]],
        split_tag: str,
    ) -> dict[str, Any]:
        y_t = resolve_ground_truth_matrix(recs)
        raw_p, logits, meta, lat = model.predict_scores_and_logits(recs)
        cal_p = scaler.predict_proba(logits)
        gids = [r.source_group_id if isinstance(r, MultiLabelRecord) else r["source_group_id"] for r in recs]

        m_raw_050 = compute_multilabel_metrics(y_t, raw_p, 0.50, latency_ms_per_sample=lat, metadata_rows=meta)
        m_cal_050 = compute_multilabel_metrics(y_t, cal_p, 0.50, latency_ms_per_sample=lat, metadata_rows=meta)
        m_cal_tuned = compute_multilabel_metrics(y_t, cal_p, tuned_tau, latency_ms_per_sample=lat, metadata_rows=meta)
        sliced = compute_sliced_e4_metrics(recs, y_t, cal_p, tuned_tau, metadata_rows=meta)
        ci = bootstrap_confidence_intervals(
            y_t,
            cal_p,
            group_ids=gids,
            thresholds=tuned_tau,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        plots_dir = out / "plots"
        plot_per_label_pr_curves(y_t, cal_p, plots_dir / f"{split_tag}_pr_curves.png")
        plot_reliability_diagrams(
            y_t,
            raw_p,
            plots_dir / f"{split_tag}_reliability.png",
            calibrated_probs=cal_p,
        )
        plot_label_cooccurrence_heatmap(y_t, plots_dir / f"{split_tag}_cooccurrence.png")
        plot_combination_breakdown(sliced, plots_dir / f"{split_tag}_cardinality.png")
        return {
            "split": split_tag,
            "n_samples": len(recs),
            "metrics_raw_tau_050": m_raw_050,
            "metrics_calibrated_tau_050": m_cal_050,
            "metrics_calibrated_val_tuned_tau": m_cal_tuned,
            "e4_sliced_metrics": sliced,
            "bootstrap_95ci": ci,
        }

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "suite": "E1_classical_baseline",
        "variant": variant,
        "seed": seed,
        "fit_info": fit_info,
        "val_calibration_report": cal_report,
        "val_threshold_tuning_summary": {
            "per_label_thresholds": thr_result["per_label_thresholds"],
            "val_macro_f1_fixed_050": thr_result["val_metrics_fixed_050"]["overall"]["macro_f1"],
            "val_macro_f1_tuned": thr_result["val_metrics_tuned"]["overall"]["macro_f1"],
        },
        "test_synthetic_eval": _eval_split(test_records, "test_synthetic"),
        "val_latency_ms_per_sample": round(val_lat, 4),
        "val_truncation": summarize_truncation_rates(val_meta),
    }
    if natural_records:
        report["test_natural_eval"] = _eval_split(natural_records, "test_natural")

    write_json(out / "baseline_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Suite E1 classical TF-IDF and zero-shot LLM baselines.")
    parser.add_argument("--variant", default="ovr_logreg", choices=["ovr_logreg", "linear_svc", "classifier_chain", "zero_shot_llm"])
    parser.add_argument("--train-file", default="data/generated/full/synthetic_train.jsonl")
    parser.add_argument("--val-file", default="data/generated/full/synthetic_val.jsonl")
    parser.add_argument("--test-file", default="data/generated/full/synthetic_test_synthetic.jsonl")
    parser.add_argument("--natural-file", default=None, help="Optional annotated test_natural.jsonl (must contain adjudicated_gold_labels).")
    parser.add_argument("--output-dir", default="results/baselines/e1_ovr_logreg")
    parser.add_argument("--generation-config", default="configs/generation_free_groq.yaml", help="Used when variant=zero_shot_llm.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = resolve_project_path(args.output_dir)
    if args.variant == "zero_shot_llm":
        cfg = load_yaml_config(args.generation_config)
        client = build_live_client(cfg["backends"]["cross_judge_auditor"])
        recs = [MultiLabelRecord(**d) for d in read_jsonl(resolve_project_path(args.test_file))]
        y_t = resolve_ground_truth_matrix(recs[: args.max_samples] if args.max_samples else recs)
        detector = ZeroShotLLMDetectorBaseline(client=client)
        out = detector.predict(recs, max_samples=args.max_samples)
        metrics = compute_multilabel_metrics(y_t, out["y_prob"], 0.50, latency_ms_per_sample=out["mean_latency_ms"])
        payload = {"zero_shot_summary": {k: v for k, v in out.items() if k != "y_prob"}, "metrics": metrics}
        write_json(out_dir / "zero_shot_report.json", payload)
        print(payload["metrics"]["overall"])
        return

    tr = [MultiLabelRecord(**d) for d in read_jsonl(resolve_project_path(args.train_file))]
    va = [MultiLabelRecord(**d) for d in read_jsonl(resolve_project_path(args.val_file))]
    te = [MultiLabelRecord(**d) for d in read_jsonl(resolve_project_path(args.test_file))]
    nat = [MultiLabelRecord(**d) for d in read_jsonl(resolve_project_path(args.natural_file))] if args.natural_file else None
    rep = run_classical_baseline_experiment(
        train_records=tr,
        val_records=va,
        test_records=te,
        natural_records=nat,
        output_dir=out_dir,
        variant=args.variant,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(rep["test_synthetic_eval"]["metrics_calibrated_val_tuned_tau"]["overall"])


if __name__ == "__main__":
    main()

