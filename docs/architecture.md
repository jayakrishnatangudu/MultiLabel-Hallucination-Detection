# System & Model Architecture Proposal (Revised)

## 1. Hardware & Deployment Constraints Analysis

Verified local environment (`2026-11-12`):
- **CPU**: 13th Gen Intel Core i5-13420H (8 cores, 12 logical processors)
- **RAM**: 16 GB (`16,786,755,584` bytes)
- **GPU**: Intel(R) UHD Graphics only (**no local NVIDIA CUDA GPU**)
- **Storage**: ~225.7 GB free on `D:\`
- **External Compute**: Google Colab GPU (e.g., T4 / L4 / A100 depending on session allocation) for transformer fine-tuning and open-weight LLM generation.

**Architectural Consequence**: Fine-tuning 7B–13B decoder LLMs (as done in Saxena et al., 2025) requires substantial GPU VRAM and slow autoregressive decoding per input. In contrast, compact **encoder transformers (~67M–184M parameters)** train efficiently in Google Colab and execute local CPU inference fast enough for interactive inspection in a Streamlit research UI.

---

## 2. Verified Backbone Model Specifications

All model identifiers, parameter counts, positional limits, and licenses below were directly verified via the Hugging Face Hub API and `config.json` / `tokenizer_config.json` (`2026-11-12`):

| Role in Project | Verified HF Model ID | Checkpoint Parameters | Backbone / Embedding Breakdown | `hidden_size` | `max_position_embeddings` | `tokenizer.model_max_length` | License |
|---|---|---:|---|---:|---:|---:|---|
| **Lightweight Baseline** | `distilbert/distilbert-base-uncased` | `66,985,530` (~67.0M) | 6-layer distilled BERT | 768 | 512 | 512 | `apache-2.0` |
| **Comparison Encoder** | `FacebookAI/roberta-base` | `124,697,947` (~124.7M) | 12-layer encoder (86M backbone per DeBERTa-v3 table) | 768 | 514 (512 usable tokens) | 512 | `mit` |
| **Primary Model** | `microsoft/deberta-v3-base` | ~184M total (per model card) | 12 layers; 86M backbone + 98M embedding (128K vocab) | 768 | 512 | `None` (must pass explicit `max_length`) | `mit` |
| **Separate Long-Context Experiment** | `answerdotai/ModernBERT-base` | `149,655,232` (~149.7M) | Long-context bidirectional encoder | 768 | 8,192 | 8,192 | `apache-2.0` |

*Implementation Note*: Because `microsoft/deberta-v3-base` has `tokenizer.model_max_length = None` in `tokenizer_config.json`, all tokenization calls **must explicitly enforce** `max_length` via our field-aware token budgeter.

---

## 3. Core Multi-Label Classifier Architecture

### 3.1 Primary Baseline Architecture (Independent 5-Sigmoid Head)
```text
Field-Aware Tokenized Input:
  [CLS] Question: {question_tokens}
        Response: {response_tokens}        <-- strictly preserved before context
        Context: {context_tokens_or_NONE} [SEP]
        │
        ▼
Pretrained Transformer Encoder Backbone
  Primary: microsoft/deberta-v3-base
  Lightweight Baseline: distilbert/distilbert-base-uncased
  Comparison: FacebookAI/roberta-base
  Long-Context Experiment: answerdotai/ModernBERT-base
        │
        ▼
Masked Mean Pooling over Non-Padding Tokens (configurable with [CLS] pooling ablation)
        │
        ▼
Primary Independent 5-Sigmoid Head:
  LayerNorm(d) -> Dropout(p) -> Linear(d, d // 2) -> GELU -> Dropout(p) -> Linear(d // 2, 5)
        │
        ▼
Raw Logits z = [z_FI, z_FF, z_II, z_LI, z_CI]
        │
        ├── Training Loss: BCEWithLogitsLoss (unweighted baseline; pos_weight / Focal in ablation)
        ├── Raw Scores: s_k = Sigmoid(z_k) ∈ (0, 1)    [NOT assumed calibrated]
        ├── Optional Post-Hoc Calibration: p_k = Sigmoid(z_k / T_k) via val-fitted Temperature Scaling
        └── Binary Predictions: y_hat_k = 1 if score_k >= tau_k else 0  (tau_k tuned on val ONLY)
```

### 3.2 Controlled Head Variants (Experimental Only — Not Assumed Primary)
The **independent 5-sigmoid head** is our primary architecture across all backbones. More complex multi-label heads are tested **strictly as controlled experiments** in Phase 5:
1. **Classifier Chain Head / Sequential Conditioning**: Predicts labels sequentially conditioned on previously predicted probabilities/logits to test whether modeling directional dependencies (e.g., `II` $\to$ `FI/FF`) improves Subset Accuracy.
2. **Label-Interaction / Co-Occurrence Residual Head**: Lightweight label-embedding bilinear or residual cross-label layer on top of the 5 logits to test whether explicit label-correlation modeling outperforms independent sigmoids.

---

## 4. Field-Aware Token Budgeting & Long-Context Processing

Long `CUAD-QA` contract clauses and `PubMedQA` abstracts frequently exceed 512 tokens. Naively placing `context` before `response` truncates the candidate response, rendering `FI`, `FF`, and `LI` invisible.

### 4.1 Explicit Token Budget Allocator (`src/data/token_budget.py` / `classifier.py`)
Given encoder budget $L_{\max}$ (e.g., `512` for DeBERTa-v3/RoBERTa/DistilBERT; `2048` or `4096`+ for ModernBERT) and special-token overhead $S$:
1. Tokenize `question`, `response`, and `context` **independently** without special tokens.
2. **Priority 1 — Candidate Response Preservation**: Reserve up to $B_{\text{resp}}$ (e.g., 288 tokens; expandable up to $L_{\max} - S - \text{len}(q_{\text{kept}})$ when `context` is empty) so the entire candidate response is preserved whenever possible. If an exceptionally long response exceeds $L_{\max} - S - B_{\text{q\_min}}$, truncate only the tail of the response *after* preserving the full question header and record `response_truncated = True`.
3. **Priority 2 — Question Allocation**: Reserve up to $B_q$ (e.g., 96 tokens) after response reservation (or keep full short questions).
4. **Priority 3 — Context Allocation**: Allocate all remaining capacity $B_c = L_{\max} - S - \text{len}(q_{\text{kept}}) - \text{len}(r_{\text{kept}})$ to `context`.
5. **Mandatory Truncation Metadata Logged Per Sample**:
   - `question_tokens_orig`, `question_tokens_kept`, `question_truncated`
   - `response_tokens_orig`, `response_tokens_kept`, `response_truncated`
   - `context_tokens_orig`, `context_tokens_kept`, `context_truncated`
   - `total_sequence_tokens`, `max_length_limit`

### 4.2 Controlled Comparison of Context Handling Strategies
Rather than silently relying on truncation, we explicitly compare three context-processing strategies on identical splits (especially `PubMedQA` and `CUAD-QA`):
1. **Budgeted Truncation (512 tokens)**: Preserve full `(question, response)` and keep leading `context` tokens up to 512.
2. **Sliding-Window Context Aggregation (512 tokens)**: Split long `context` into overlapping chunks $\{c^{(1)}, \dots, c^{(M)}\}$ (e.g., stride 128), pair **every** chunk with the full `(question, response)`, encode in parallel, and aggregate chunk representations/logits (log-sum-exp / max-pool / attention-pool) so zero context information is dropped within a 512-token backbone.
3. **Native Long-Context Encoding (`answerdotai/ModernBERT-base`)**: Feed full `(question, response, context)` up to `2048`–`4096` (up to native `8192` limit) in a single pass as a dedicated long-context experiment.

---

## 5. Calibration Evaluation & Optional Temperature Scaling

Raw neural sigmoid outputs $\sigma(z_k)$ frequently exhibit overconfidence or miscalibration—especially under synthetic-to-natural domain shift or class imbalance—and **must NOT be described as calibrated probabilities by default**.

- **Diagnostic Calibration Metrics** (logged on both `val` and `test` splits, per-label and macro-averaged):
  1. **Expected Calibration Error (ECE)** (e.g., 10-bin reliability binning + reliability diagrams).
  2. **Brier Score** ($\frac{1}{N} \sum_i (s_{i,k} - y_{i,k})^2$ per label $k$ and mean Brier across labels).
- **Optional Post-Hoc Temperature Scaling** (controlled experiment):
  - Fit scalar temperature $T$ (or per-label vector $\mathbf{T} = [T_1, \dots, T_5]$) by minimizing `BCEWithLogitsLoss` on the **validation split only** (`L-BFGS` / `Adam`, test split untouched).
  - Report before-and-after ECE, Brier Score, and thresholded F1 to evaluate whether temperature scaling improves probability reliability without distorting ranking metrics (PR-AUC / ROC-AUC).

---

## 6. Validation-Only Thresholding Policy

- **Baseline Reference**: Always report metrics at fixed `tau = 0.50` across all 5 categories.
- **Validation-Only Tuning**: Sweep $\tau_k \in [0.05, 0.95]$ (step `0.01`) exclusively on **validation split** predictions to maximize per-label $F_1$ (also tracking sensitivity curves for Precision, Recall, $F_1$, Macro-F1, and Hamming Loss).
- **Strict Leakage Guard**: Threshold optimizers must enforce split assertions verifying zero overlap with `test_synthetic` or `test_natural`. Test splits are evaluated **once** using locked validation-selected $\boldsymbol{\tau}^*$.

---

## 7. Modular Software Structure

```text
MultiLabel-Hallucination-Detection/
├── configs/                  # dataset.yaml, generation.yaml, model.yaml, training.yaml
├── prompts/                  # Versioned single/multi generation + diagnostic judge prompts
├── data/
│   ├── raw/                  # Immutable downloaded source datasets
│   ├── processed/            # Deduplicated canonical records + source_group_id
│   ├── generated/            # Raw synthetic outputs + generation provenance
│   ├── validated/            # Multi-tier records (intended / auto_detected / human_observed / adjudicated_gold)
│   └── splits/               # Question-grouped 80:10:10 synthetic splits + separate test_natural.jsonl
├── src/
│   ├── data/                 # download, preprocessing, token_budget, generation, validation, annotation, splitting
│   ├── models/               # classifier, chain_heads (experimental), baseline, calibration, inference
│   ├── training/             # train, evaluate, thresholds
│   ├── evaluation/           # metrics (F1, Hamming, Exact Match, PR-AUC, ECE, Brier), visualizations
│   ├── experiments/          # ablation, comparison, statistical_tests
│   └── utils/                # config, seeding, logging, schema validation
├── app/                      # Streamlit research demo UI (model predictions clearly distinguished from verified truth)
├── tests/                    # Automated unit & integration tests
├── experiments/              # Logged configs, checkpoints, TensorBoard/CSV logs
└── results/                  # Reproducible tables, reliability plots, error-analysis exports
```
