# Experiment Plan (Revised)

## 1. Scientific Integrity & Evaluation Rules
1. **Fair Task Separation**: Never conflate binary accuracy (Saxena et al., 2025) with multi-label F1. Binary evaluation (`Task B1`: any label $>0$) and 5-label evaluation (`Task M5`: $\hat{\mathbf{y}} \in \{0,1\}^5$) are reported separately.
2. **Strict Split Hygiene**: Models train on `train`; hyperparameters, temperature scaling ($T$), and per-label thresholds ($\boldsymbol{\tau}^*$) are selected **exclusively on `val`**; final evaluations run once on locked **`test_synthetic`**, **`test_synthetic_gold`**, and **`test_natural`** (kept strictly separate).
3. **LLM Judges Never Treated as Ground Truth**: Automated LLM-judge predictions serve only as diagnostic spillover checks and as an explicit *zero-shot/few-shot baseline* (`E1.5`), never as test ground truth.
4. **Statistical Rigor**: Report 95% bootstrap confidence intervals ($B=1000$) over test items, multi-seed mean $\pm$ std across 3 seeds for primary models, and paired significance tests (McNemar / paired bootstrap / Wilcoxon) for key model and ablation comparisons.

---

## 2. Experiment Suites

### Suite E0 — Synthetic Strategy Pilot (Phase 2)
Compare **Sequential (A)**, **Joint (B)**, and **Span-Level Editing (C)** on 150–300 stratified source records spanning all domains and label counts:
- Metrics: format validity rate, refusal rate, edit-compliance rate, blind cross-judge target agreement, **unintended spillover rate**, and blind human Fleiss'/Cohen's $\kappa$ against `intended_labels`.

### Suite E1 — Baselines & Base-Paper Contextualization (Phase 3)
1. **E1.1 — Binary Hallucination Detection Check (`Task B1`)**: Reconstructed single-category + clean slice evaluated with binary prediction (`max_k s_k` or dedicated binary classifier) to contextualize binary recall across Cat I–V.
2. **E1.2 — TF-IDF + One-vs-Rest Logistic Regression**: Word + char n-grams on structured input with independent binary LR heads.
3. **E1.3 — TF-IDF + LinearSVC / Classifier Chain Baseline**.
4. **E1.4 — Five Independent Binary Encoders (`DeBERTa-v3-base` / `DistilBERT`)**: Five separate single-label models (one per category) compared against the single shared-backbone 5-sigmoid multi-label model (`E2.3`) to quantify parameter efficiency and multi-task sharing gains.
5. **E1.5 — Zero-Shot & Few-Shot LLM Judge Baseline**: Structured 5-label LLM prompting evaluated against human-verified `test_synthetic_gold` and `test_natural`.

### Suite E2 — Transformer Multi-Label Models (Phase 4)
All models below use the **primary independent 5-sigmoid head** with `BCEWithLogitsLoss`, field-aware token budgeting preserving the full response, fixed `0.50` threshold baseline, and validation-tuned per-label thresholds $\boldsymbol{\tau}^*$:
1. **E2.1 — Lightweight Baseline**: `distilbert/distilbert-base-uncased` (~67.0M params, 512 tokens).
2. **E2.2 — Comparison Encoder**: `FacebookAI/roberta-base` (~124.7M params, 512 usable tokens).
3. **E2.3 — Primary Recommended Model**: `microsoft/deberta-v3-base` (~184M params, 512 tokens, explicit `max_length=512`).
4. **E2.4 — Separate Long-Context Experiment**: `answerdotai/ModernBERT-base` (~149.7M params, tested at extended context lengths e.g. 2,048 / 4,096 tokens on full dataset and `PubMedQA`/`CUAD-QA` long-context slices).

### Suite E3 — Controlled Experiments & Ablations (Phase 5)
| ID | Research Question | Controlled Conditions |
|---|---|---|
| **A1 (Data Composition)** | Does training on single-label data detect co-occurring hallucinations? | (i) Clean + Single only vs. (ii) Clean + Multi only vs. (iii) Clean + Single + Multi, evaluated across Clean, Single, Multi (`test_synthetic`), and `test_natural`. |
| **A2 (Threshold Selection & Sensitivity)** | What is the gain and stability of validation-only per-label threshold tuning? | Fixed $\tau = 0.50$ vs. Val-tuned $\boldsymbol{\tau}^*$ (per-label F1) + full sweep sensitivity curves ($\tau \in [0.05, 0.95]$) logged exclusively from `val` before applying to test. |
| **A3 (Calibration & Temperature Scaling)** | How well calibrated are raw sigmoid outputs and does temperature scaling improve reliability? | Raw sigmoid scores vs. Val-fitted global $T$ vs. per-label $\mathbf{T}$, measured via **ECE** and **Brier Score**. |
| **A4 (Context Processing)** | How do truncation, sliding window, and long-context encoding compare on long inputs? | (i) Budgeted response-preserving 512-token truncation (`DeBERTa-v3-base`), (ii) Sliding-window context pooling (`DeBERTa-v3-base`), (iii) Native long-context encoding (`ModernBERT-base`)—tracked with exact truncation metadata rates. |
| **A5 (Head Architecture — Controlled)** | Do label-dependency heads outperform independent 5-sigmoids? | Primary independent 5-sigmoid head vs. Classifier-Chain head vs. Label-Interaction residual head on `microsoft/deberta-v3-base`. |
| **A6 (Imbalance Loss)** | Does weighting improve minority/complex categories (`LI`, `CI`, triples)? | Unweighted `BCEWithLogitsLoss` vs. capped `pos_weight` BCE vs. Asymmetric Focal Loss. |
| **A7 (Context-Shortcut Control)** | Do non-CI controls inside `PubMedQA`/`CUAD-QA` stop context-presence shortcuts? | Naive disjoint pool vs. Context-unified control pool (measuring false-positive `CI` rate on clean/non-CI `PubMedQA`/`CUAD-QA`). |
| **A8 (Cross-Domain Leave-One-Out)** | Does multi-label detection transfer across domains? | Train on 2 domains $\to$ zero-shot evaluate on held-out 3rd domain (all 3 folds). |

### Suite E4 — Granular Error, Natural-Set & Combination Analysis (Phase 5)
- **Synthetic vs. Natural Domain Gap**: Compare `test_synthetic`, `test_synthetic_gold`, and `test_natural` side by side without pooling.
- **By Label Cardinality**: Performance across $\|y\|_1 \in \{0, 1, 2, 3, 4+\}$.
- **By Category Pair & Interaction**: Precision/Recall/F1/Exact-Match for every frequent pair (`FI+FF`, `FI+LI`, `FF+LI`, `FI+CI`, `FF+CI`, `LI+CI`, `II+*`), explicitly testing error-confusion hypotheses.
- **Truncation-Stratified Analysis**: Compare accuracy on `context_truncated == False` vs. `context_truncated == True` subsets.

---

## 3. Comprehensive Evaluation Metrics Logged Per Run

- **Multi-Label Classification**: Micro / Macro / Weighted / Samples Precision, Recall, F1; Hamming Loss ($\downarrow$); Subset Accuracy / Exact Match Ratio ($\uparrow$).
- **Per-Category**: Precision, Recall, F1, Support, PR-AUC (Average Precision), ROC-AUC, 2×2 Confusion Matrix.
- **Calibration**: Per-label and mean **ECE** (10-bin reliability diagrams) and **Brier Score** before and after optional temperature scaling.
- **Operational & Token Metadata**: Mean inference latency (ms/sample on CPU/GPU), parameter count, and percentage of test samples with `context_truncated` / `response_truncated`.
