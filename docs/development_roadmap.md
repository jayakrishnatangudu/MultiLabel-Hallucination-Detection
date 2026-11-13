# Project Implementation Roadmap

This roadmap executes the project strictly in verifiable, testable stages while honoring local hardware constraints (16 GB CPU locally; Google Colab GPU / external inference API for heavy LLM generation and transformer fine-tuning).

---

## Phase 0 — Workspace & Research Audit *(Current Phase — Awaiting Approval)*
- **Status**: Complete upon user review of Phase 0 deliverables.
- **Deliverables**:
  - `docs/base_paper_analysis.md`
  - `docs/literature_review.md`
  - `docs/research_gap.md`
  - `docs/proposed_methodology.md`
  - `docs/architecture.md`
  - `docs/experiment_plan.md`
  - `docs/development_roadmap.md`
- **Decision Gate**: User approves taxonomy scope, dataset reconstruction plan (due to empty base repo), generation backend/budget, and target dataset scale before Phase 1 coding begins.

---

## Phase 1 — Repository Setup, Source Dataset Acquisition & Analysis
- **Tasks**:
  1. Initialize Git repository, `.gitignore`, `requirements.txt`, and modular `src/` + `configs/` + `tests/` scaffolding.
  2. Implement `src/data/download.py` to fetch `MedQuAD`, `LegalQAv1`, `FIQA`, `PubMedQA`, and `CUAD-QA` into immutable `data/raw/`.
  3. Implement `src/data/preprocessing.py` with unified canonical record format, token-length statistics, exact deduplication, and near-duplicate clustering (`source_group_id`).
  4. Write automated unit tests (`tests/test_preprocessing.py`) and generate a verified **Dataset Quality & Distribution Report**.
- **Exit Criteria**: Clean, deduplicated source pools in `data/processed/` + 100% passing preprocessing/schema tests.

---

## Phase 2 — Controlled Multi-Label Data Generation & Validation
- **Tasks**:
  1. Author versioned prompts in `prompts/` (5 single-label + Sequential, Joint, and Span-Editing multi-label + Cross-Judge audit prompt).
  2. Implement `src/data/generation.py` with provider abstraction (Colab/HF local Llama-3-8B or API endpoint), deterministic config logging, refusal retries, and schema checks.
  3. Run **Experiment E0** (pilot comparing Strategies A, B, C on 150–300 stratified items) + small blinded manual inspection.
  4. Select best synthesis strategy (or hybrid) and scale dataset generation across Clean (`G0`), Single (`G1`), and Validated Multi-Label (`G2`–`G4/G5`), including `PubMedQA`/`CUAD-QA` non-CI controls.
  5. Implement `src/data/validation.py` (automated checks logged under `auto_detected_labels`, never promoted to gold), `src/data/annotation.py` (blinded export/import for `human_observed_labels`, `adjudicated_gold_labels`, Fleiss'/Cohen's $\kappa$), and `src/data/splitting.py` (group-stratified 80:10:10 synthetic splits + separate `test_natural` holdout + zero-leakage assertions).
  6. Conduct blinded human annotation on both a stratified synthetic gold subset (`test_synthetic_gold`) and a natural LLM response test set (`test_natural`).
- **Exit Criteria**: Validated JSONL splits in `data/splits/`, separate `test_natural.jsonl`, zero cross-split question leakage verified by pytest, and label-distribution + human-agreement report.

---

## Phase 3 — Evaluation Suite & Baseline Models
- **Tasks**:
  1. Implement `src/evaluation/metrics.py` and `src/evaluation/visualizations.py` with unit tests verifying Micro/Macro/Weighted F1, Hamming Loss, Subset Accuracy, PR-AUC, ECE, Brier Score, and per-label confusion matrices.
  2. Implement traditional ML baselines (`src/models/baseline.py`): TF-IDF + OvR Logistic Regression, LinearSVC, and Classifier Chain.
  3. Implement single-label binary reproduction check (`E1.1`), 5 independent binary encoder baseline (`E1.4`), and optional zero/few-shot LLM judge baseline (`E1.5`).
- **Exit Criteria**: Baseline benchmark table saved under `results/` with locked evaluation scripts.

---

## Phase 4 — Proposed Multi-Label Transformer Detector
- **Tasks**:
  1. Implement `src/data/token_budget.py` (field-aware token budgeting preserving full responses + logging truncation metadata) and `src/models/classifier.py` (independent 5-sigmoid primary head + optional experimental heads).
  2. Implement `src/training/train.py`, `src/training/thresholds.py` (validation-only per-label sweeps + fixed 0.50 baseline), `src/models/calibration.py` (ECE/Brier + optional val-fitted temperature scaling), and `src/training/evaluate.py`.
  3. Train and evaluate `distilbert/distilbert-base-uncased` (lightweight baseline), `FacebookAI/roberta-base` (comparison), `microsoft/deberta-v3-base` (primary model), and `answerdotai/ModernBERT-base` (separate long-context experiment).
- **Exit Criteria**: Trained checkpoints, validation-tuned threshold configs, calibration metrics, and separate `test_synthetic` / `test_synthetic_gold` / `test_natural` logs.

---

## Phase 5 — Ablations, Error Analysis & Statistical Testing
- **Tasks**:
  1. Implement `src/experiments/ablation.py` and `src/experiments/comparison.py`.
  2. Run Ablations `A1`–`A8` (data composition, threshold tuning & sensitivity, calibration/temperature scaling, context truncation vs. sliding-window vs. ModernBERT long-context, independent 5-sigmoid vs. classifier-chain/interaction heads, imbalance loss, context-shortcut control, leave-one-domain-out transfer).
  3. Run Combination Difficulty analysis (`E4`), bootstrap confidence intervals, and significance tests.
  4. Generate publication-ready charts (co-occurrence heatmaps, PR curves, loss curves, combination breakdown) under `results/`.
- **Exit Criteria**: Complete empirical evidence matrix backing every research claim (including negative/inconclusive findings).

---

## Phase 6 — Research Demonstration Application
- **Tasks**:
  1. Build clean inference module (`src/models/inference.py`) supporting CPU execution of the trained encoder, validation-tuned thresholds, and optional temperature-scaled scores.
  2. Build Streamlit application (`app/main.py` + `app/components/`) accepting Response, optional Question, and optional Reference Context.
  3. Display binary overall status, all triggered categories, raw/calibrated model scores (clearly marked as unverified model predictions unless verified externally), category descriptions, optional heuristic sentence-level context alignment highlights, and downloadable structured JSON.
  4. Write end-to-end app validation and inference tests.
- **Exit Criteria**: Tested, responsive local Streamlit app backed by trained checkpoints.

---

## Phase 7 — Research Paper & Defense Preparation
- **Tasks**:
  1. Auto-generate experiment result tables from saved JSON logs (zero manual transcription errors).
  2. Draft all 18 manuscript sections (Abstract through References) in `docs/manuscript/` aligned with target venue guidelines (checking scope, indexing, fees, and review standards before recommending venues).
  3. Verify all citations, DOIs, limitations, threats to validity, and reproducibility checklist (`README.md`, model/dataset cards).
  4. Prepare final project report and presentation slides.

