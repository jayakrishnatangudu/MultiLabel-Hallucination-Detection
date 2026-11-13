# Research Gap Statement

## 1. Summary of Prior Work Capabilities vs. Unsolved Gaps

Recent hallucination research (2023–2026) has moved well beyond simple response-level binary classification. For example:
- **FAVA** (Mishra et al., COLM 2024) and **RAGTruth** (Niu et al., ACL 2024) perform **span-level** detection across fine-grained categories.
- **SciHal25** (Li et al., SDP 2025) frames scientific claim verification against scholarly abstracts as multi-label classification.
- **FG-PRM** (Li et al., EMNLP 2025 Findings) applies a six-type taxonomy—overlapping strongly with Huang et al. (2025)—to step-level **mathematical reasoning** (GSM8K, MATH).
- **Mu-SHROOM** (SemEval-2025 Task 3) benchmarks multilingual span-level overgeneration, and **HaluCheck** (Qiu et al., 2026) evaluates detection/localization/correction across factual text and agent trajectories.

Therefore, **we explicitly do NOT claim that fine-grained taxonomies or multi-label hallucination detection in general are entirely unexplored**. Instead, our literature audit identifies **four specific, verifiable research gaps** at the intersection of high-stakes QA (Saxena et al., 2025) and multi-label detection:

---

## 2. Specific Research Gaps

### Gap 1 — Single-Category and Binary Collapse in High-Stakes Domain QA
Saxena et al. (2025) define five distinct factuality and faithfulness hallucination categories across medical (`MedQuAD`, `PubMedQA`), legal (`LegalQAv1`, `CUAD-QA`), and financial (`FIQA`) QA. However:
1. Their synthetic dataset enforces a strict **single-category-per-instance** assumption (no response has $>1$ active hallucination label).
2. Their downstream experiments collapse all five categories into **binary detection** (`Hallucinated` vs. `Non-Hallucinated`), predicting neither individual categories nor co-occurring label subsets.
No validated multi-label benchmark currently captures **simultaneous co-occurrence across all five Huang et al. (2025) categories** in medical, legal, and financial QA.

### Gap 2 — Unresolved Context-Inconsistency Disjointness Across Datasets
In Saxena et al. (2025), Category V (**Context Inconsistency**) is studied exclusively on `PubMedQA` and `CUAD-QA`, while Categories I–IV (**Factual Inconsistency**, **Factual Fabrication**, **Instruction Inconsistency**, **Logical Inconsistency**) are generated exclusively on context-free QA (`MedQuAD`, `LegalQAv1`, `FIQA`).
Because of this disjoint design:
- Existing data cannot test whether a detector can distinguish **Context Inconsistency** from **Factual Inconsistency** or **Factual Fabrication** when *both* occur in the same context-grounded response (e.g., contradicting a supplied clinical abstract while also inventing a fake clinical trial).
- If naively combined, a classifier can exploit a trivial dataset shortcut (`presence of context passage` $\Rightarrow$ `Context Inconsistency`).

### Gap 3 — Lack of Controlled Multi-Category Synthesis & Spillover Analysis
Generating multi-label hallucinated responses is qualitatively harder than single-category prompting:
- Naive joint prompts can fail to inject all requested labels, collapse semantic coherence, or introduce **unintended secondary hallucinations** (e.g., rewriting an answer to violate an instruction inadvertently fabricates new claims or creates logical contradictions).
- No prior controlled comparison evaluates **Sequential Generation**, **Joint Prompt Generation**, and **Controlled Span-Level Editing** on multi-label faithfulness, edit separability, unintended spillover rate, and downstream detector generalization.

### Gap 4 — Uncalibrated LLM-Text Detectors vs. Lightweight Multi-Label Encoders & Combination-Level Error Analysis
Saxena et al. (2025) rely on fine-tuning 7B–13B decoder LLMs to emit `'Hallucinated'` / `'Non-Hallucinated'` tokens, evaluated solely via accuracy. This design:
- Suffers from high inference cost and non-probabilistic thresholding (closed zero-shot LLMs showed severe bias toward predicting `Non-Hallucinated`).
- Lacks per-category calibrated probability outputs, per-label threshold optimization, and empirical analysis of **which hallucination combinations confuse detectors** (e.g., `Factual Inconsistency + Factual Fabrication` vs. `Factual Inconsistency + Logical Inconsistency`), as well as cross-domain and single-only vs. multi-label training ablations.

---

## 3. Concrete Research Contributions of This Project

Addressing Gaps 1–4 yields five clearly demarcated, testable contributions beyond Saxena et al. (2025):

1. **Publicly Reproducible Dataset Reconstruction & Multi-Label Benchmark**: Given that the base paper's GitHub repository (`ashitas96/fine_grained_hallucinated_data`) hosts no dataset files, we construct an open, reproducible pipeline from the 5 verified source datasets (`MedQuAD`, `LegalQAv1`, `FIQA`, `PubMedQA`, `CUAD-QA`), covering clean (`0`), single-label (`1`), and valid multi-label (`2`–`5` simultaneous labels) responses under a unified 5-bit schema with strict question-grouped, leakage-free splits.
2. **Controlled Multi-Category Generation & Strategy Benchmark**: Systematic empirical comparison of **Sequential**, **Joint**, and **Controlled Span-Level Editing** strategies—governed by explicit combination-validity rules and audited for label spillover—plus a **context-unified control** where `PubMedQA`/`CUAD-QA` receive *all* 5 single and multi-label patterns (preventing context-presence shortcuts).
3. **Human-Audited Synthetic & Natural Evaluation Sets**: Multi-annotator verification strictly separating `intended_labels`, `auto_detected_labels`, `human_observed_labels`, and `adjudicated_gold_labels` (reporting Fleiss' $\kappa$, pairwise Cohen's $\kappa$, and unintended spillover rates), paired with a **separate human-annotated natural LLM response test set** (`test_natural`).
4. **Threshold-Tuned & Calibration-Evaluated Multi-Label Encoders**: Independent 5-sigmoid encoder classifiers (`microsoft/deberta-v3-base` primary, `distilbert/distilbert-base-uncased` lightweight baseline, `FacebookAI/roberta-base` comparison, plus `answerdotai/ModernBERT-base` as a separate long-context experiment) using field-aware response-preserving token budgeting, validation-only per-label threshold tuning vs. `0.50` fixed baseline, and calibration assessment via **ECE** and **Brier Score** (with optional validation-fitted temperature scaling), benchmarked against TF-IDF, One-vs-Rest, 5-encoder binary ensembles, and zero/few-shot LLM judges.
5. **Combination-Difficulty, Cross-Domain, and Ablation Analysis**: Controlled experiments testing whether training on single-label data transfers to multi-label inputs, how joint single+multi training improves co-occurring detection, and which category pairs cause maximal confusion.

