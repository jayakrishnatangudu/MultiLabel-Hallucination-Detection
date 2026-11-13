# Proposed Research Methodology (Revised)

## 1. Problem Formulation

Given an input tuple $x = (q, c, r)$ consisting of a user question/directive $q$, an optional reference context passage $c \in \mathcal{C} \cup \{\varnothing\}$, and a candidate LLM response $r$, the task is **response-level multi-label fine-grained hallucination classification**:

$$f_\theta(q, c, r) \rightarrow \mathbf{z} \in \mathbb{R}^5, \quad \mathbf{s} = \sigma(\mathbf{z}) \in (0, 1)^5, \quad \hat{y}_k = \mathbb{I}(s_k \ge \tau_k)$$

preserving the five taxonomy categories from Saxena et al. (2025) and Huang et al. (2025) without merging:

| Index | Key | Category | Definition |
|---:|---|---|---|
| 0 | `factual_inconsistency` (FI) | Factual Inconsistency | Contradicts established real-world facts (e.g., wrong numerical limits, entity attributes, mechanisms). |
| 1 | `factual_fabrication` (FF) | Factual Fabrication | Introduces invented, unverifiable entities, clinical trials, citations, or statutory mechanisms. |
| 2 | `instruction_inconsistency` (II) | Instruction Inconsistency | Diverges from the user's explicit question/directive (answers a related but different query). |
| 3 | `logical_inconsistency` (LI) | Logical Inconsistency | Exhibits internal contradictions or invalid reasoning transitions within the response itself. |
| 4 | `context_inconsistency` (CI) | Context Inconsistency | Contradicts, omits required caveats from, or fabricates claims unsupported by an explicitly supplied context $c$. |

Raw sigmoid outputs $\mathbf{s}$ are treated as **uncalibrated confidence scores** unless explicitly calibrated via validation-fitted temperature scaling and verified via ECE and Brier Score. Thresholds $\boldsymbol{\tau} = (\tau_1, \dots, \tau_5)$ are tuned **exclusively on the validation split** and compared against a fixed `0.50` baseline.

---

## 2. Stage B — Verified Source Dataset Acquisition & Licensing Audit

Because `https://github.com/ashitas96/fine_grained_hallucinated_data` hosts only a 71-byte `README.md` (`2024-06-11`) with no dataset files, we reconstruct clean source pools directly from the five publicly verified datasets (`2026-11-12`):

| Domain | Dataset Source | Verified Availability & Size | Verified License Status | Context $c$ |
|---|---|---|---|---|
| Medical | `keivalya/MedQuad-MedicalQnADataset` (mirror of `abachaa/MedQuAD`) | Accessible on HF (`train`: **16,407** rows) | HF card lists `None`; upstream `abachaa/MedQuAD` repo `LICENSE.txt` specifies **CC BY 4.0** (with 3 MedlinePlus subsets' answers removed for copyright compliance) | Absent ($\varnothing$) |
| Legal | `dzunggg/legal-qa-v1` | Accessible on HF (`train`: **3,742** rows) | **No explicit license metadata** (`card_license: None` on HF) — usage restricted to research reproducibility & flagged in ethics/licensing section | Absent ($\varnothing$) |
| Finance | `llamafactory/fiqa` | Accessible on HF (`train`: **5,500**, `test`: **1,000**) | **Apache-2.0** (`apache-2.0` on HF card; derived from Maia et al., 2018 FiQA) | Absent ($\varnothing$) |
| Medical (Context) | `qiaojin/PubMedQA` | Accessible on HF (`pqa_labeled`: **1,000**; `pqa_unlabeled`: **61,249**; `pqa_artificial`: **211,269**) | **MIT** (`mit` on HF card) | Present (PubMed abstract) |
| Legal (Context) | `TheAtticusProject/cuad` (`data.zip`, 18.3 MB) | Accessible via GitHub release zip (`HTTP 200`, `18,309,308` bytes; HF `cuad-qa.py` points to same URL) | **CC BY 4.0** (`cc-by-4.0` on HF `cuad` & `cuad-qa` cards) | Present (Contract clause) |

### Preprocessing, Deduplication & Context-Shortcut Control
1. Store immutable raw files under `data/raw/`. Normalize records into `data/processed/`.
2. Drop non-questions, empty answers, and extremely short snippets unsuitable for multi-error perturbation (logged with exact attrition counts).
3. Deduplicate exact strings and group near-duplicate questions (TF-IDF cosine $\ge 0.85$ / MinHash) under a single `source_group_id`.
4. **Context-Shortcut Control**: Because `MedQuAD`, `LegalQAv1`, and `FIQA` lack explicit contexts ($c = \varnothing \Rightarrow \text{CI} = 0$), we generate **all** label regimes—clean (`00000`), single non-CI (`FI`, `FF`, `II`, `LI`), and non-CI multi-label combinations (`FI+FF`, `FI+LI`, etc.)—**within `PubMedQA` and `CUAD-QA` as well**, preventing models from learning `(c is present) => (CI = 1)`.

---

## 3. Stage C — Controlled Multi-Label Dataset Engineering

### 3.1 Combination Validity Rules
Rather than naively generating all $2^5 = 32$ binary vectors:
- **G0 — Clean (`[0,0,0,0,0]`)**: Original ground-truth answers across all 5 sources.
- **G1 — Singletons (5 patterns)**: `FI`, `FF`, `II`, `LI`, and `CI` (`CI` only when $c \neq \varnothing$).
- **G2 — Validated Pairs (up to 10 patterns)**:
  - *Content pairs*: `FI+FF`, `FI+LI`, `FF+LI`, `FI+CI`, `FF+CI`, `LI+CI` (`CI` pairs only when $c \neq \varnothing$).
  - *Instruction pairs*: `II+FI`, `II+FF`, `II+LI`, `II+CI` — constrained so the response exhibits *topical-shift instruction deviation* (related entity, wrong question/task) and embeds the second factual/logical/context error inside that off-directive answer.
- **G3 — Validated Triples**: Separable multi-clause combinations (`FI+FF+LI`, `FI+FF+CI`, `FI+LI+CI`, `FF+LI+CI`, `II+FI+FF`, `II+FF+LI`, `II+FI+CI`, `II+FF+CI`) restricted to multi-sentence items ($\ge 3$ sentences).
- **G4 & G5 — Stress Subsets (4–5 Labels)**: Strictly restricted to rich multi-sentence context-grounded records ($\ge 4$–$5$ sentences) with mandatory human verification; never scaled blindly if pilot audit shows semantic collapse or low human agreement.

### 3.2 Generation Strategy Comparison (Pilot Study E0)
Compare three generation protocols on identical stratified items with fixed seeds and saved prompt templates (`prompts/`):
- **Strategy A — Sequential Generation**: Iterative single-category perturbations ($r_0 \to r_1 \to r_2 \dots$) with explicit constraint preservation.
- **Strategy B — Joint Compositional Prompting**: One-pass prompt specifying exact target labels, semantic separation rules (`FI` vs. `FF`, `FI` vs. `CI`), and required per-label span justifications.
- **Strategy C — Controlled Span-Level Editing**: Sentence/clause segmentation assigning distinct perturbation operations to disjoint text spans, followed by minimal connective smoothing.

Generator backend is configurable via `configs/generation.yaml` (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`—which requires accepting Meta's gated `llama3` Community License—or an explicitly logged open/API generator if hardware requires). Every record logs exact model ID, prompt version, decoding parameters (`temperature`, `top_p`, `max_tokens`), and seed where supported.

---

## 4. Multi-Tier Dataset Validation & Ground-Truth Governance

**Core Scientific Rule**: Neither raw synthetic generator intent nor automated LLM-judge outputs may be silently equated with verified ground truth. Every record explicitly separates four label tiers in its JSON schema:

```json
{
  "sample_id": "MLH_000001",
  "source_dataset": "pubmedqa",
  "source_record_id": "21645374",
  "source_group_id": "grp_pubmedqa_00412",
  "domain": "medical",
  "data_origin": "synthetic",
  "has_context": true,
  "question": "...",
  "context": "...",
  "original_answer": "...",
  "generated_answer": "...",
  "generation_method": "span_editing",
  "generation_model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "prompt_version": "v1.0",
  "intended_labels": {
    "factual_inconsistency": 1,
    "factual_fabrication": 1,
    "instruction_inconsistency": 0,
    "logical_inconsistency": 0,
    "context_inconsistency": 1
  },
  "auto_detected_labels": {
    "judge_model": "...",
    "predicted_labels": [1, 1, 0, 0, 1],
    "rule_checks_passed": true,
    "spillover_flagged": false
  },
  "human_observed_labels": {
    "annotator_1": [1, 1, 0, 0, 1],
    "annotator_2": [1, 1, 0, 0, 1],
    "annotator_3": [1, 0, 0, 0, 1]
  },
  "adjudicated_gold_labels": [1, 1, 0, 0, 1],
  "validation_status": "human_verified",
  "annotation_explanation": "..."
}
```

### Definition of the Four Validation Tiers
1. **`intended_labels`**: Target 5-bit vector requested during synthetic generation (null for natural LLM responses).
2. **`auto_detected_labels`**: Outputs from automated rule checks (JSON compliance, non-refusal, non-identity edit distance, Termite structural retention) and a blind diagnostic cross-judge LLM. Used strictly for filtering/flagging anomalies (`auto_passed` / `auto_flagged`), **never promoted to gold ground truth**.
3. **`human_observed_labels`**: Independent 5-bit labels recorded by blinded human annotators who inspect `(question, context, original_answer, generated_answer)` **without seeing `intended_labels` or `auto_detected_labels`**.
4. **`adjudicated_gold_labels`**: Final consensus/adjudicated 5-bit ground truth (majority consensus across annotators + explicit tie resolution), populated exclusively for `human_verified` samples.

We compute **per-label and overall Fleiss' $\kappa$**, **pairwise Cohen's $\kappa$**, and **synthetic-intent vs. human-gold discrepancy (spillover rate)**.

---

## 5. Separate Synthetic & Natural Evaluation Sets + Splitting

1. **Question-Grouped Synthetic Split (`80:10:10`)**:
   - Group all synthetic & clean variants by `source_group_id`.
   - Split via multi-label group stratification (`seed=42`) into `train`, `val`, and `test_synthetic`.
   - Automated pytest assertions verify zero `source_group_id` or near-duplicate question leakage across splits.
2. **Separate Natural LLM Evaluation Set (`test_natural`)**:
   - Select held-out questions from `MedQuAD`, `LegalQAv1`, `FIQA` (`test`), `PubMedQA`, and `CUAD-QA` **strictly disjoint from `train` and `val`**.
   - Generate *natural, unprompted-for-hallucination* responses from 2–3 instruction-tuned LLMs (including genuine domain queries where models naturally fail or succeed).
   - Blindly human-annotate and adjudicate all 5 labels (`adjudicated_gold_labels`).
   - **Never merge `test_natural` into `test_synthetic`**: report every detector's performance separately on `test_synthetic` (intended labels), `test_synthetic_gold` (human-verified synthetic subset), and `test_natural` (natural human-verified LLM responses).
