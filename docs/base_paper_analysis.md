# Base Paper Analysis

## 1. Bibliographic & Document Overview

- **Primary Base Paper (`references/base_research_paper.pdf`)**:
  - **Title**: *A framework to synthetically generate fine-grained hallucinated data*
  - **Authors**: Ashita Saxena, Dishank Aggarwal, Naveen Badathala, and Pushpak Bhattacharyya (CFILT Lab, IIT Bombay, India)
  - **Journal**: *Language Resources and Evaluation* (Springer), Vol. 59, pp. 4223–4252 (2025)
  - **Timeline**: Received 11 June 2024; Accepted 30 June 2025; Published online 27 September 2025
  - **DOI**: `10.1007/s10579-025-09864-x`
  - **Article License**: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
  - **Claimed Data Repository**: `https://github.com/ashitas96/fine_grained_hallucinated_data/`

- **Initial Proposal Presentation (`references/0th_review_ppt.pptx` / `0th_review_ppt.pdf`)**:
  - **Title**: *Improving Fine-Grained Hallucination Detection Through Multi-Label Classification Of Complex LLM Responses* (Prasad V. Potluri Siddhartha Institute of Technology, Dept. of CSE; Guide: Dr. P. Anil Kumar; Team: N. Yavanika, L. Pujitha, K. Akshit).
  - **Role**: High-level initial project proposal framing the extension from Saxena et al.'s single-category detection to multi-label detection (`Accuracy`, `Precision & Recall`, `Macro-F1 & Micro-F1`, `Hamming Loss`, `Exact Match Ratio`, `Per-Category F1`), alongside Python/PyTorch/Hugging Face/Scikit-learn/RAG tooling and 16 GB RAM / GPU recommendations.
  - **Scope Hygiene**: Unrelated template artifacts present in underlying presentation text (e.g., KV-cache eviction / SLM streaming comparison text behind Slide 7) are excluded from this research project as instructed.

---

## 2. Section-by-Section Analysis of Saxena et al. (2025)

### Section 1 & Section 2 — Introduction & Motivation
- **Problem**: LLMs exhibit plausible yet incorrect or unverifiable generations ("hallucinations") even in strong models such as ChatGPT. While prior work examined isolated tasks (summarization, dialogue), systematic fine-grained hallucination resources in **high-stakes domains** (medical, legal, finance) were lacking.
- **Core Contributions Claimed**:
  1. A prompting strategy (using Llama3-8B) to synthetically generate five fine-grained hallucination types.
  2. A synthetic benchmark of **52,646** hallucinated QA instances spanning medical, legal, and finance domains.
  3. Systematic zero-shot evaluation of 5 open-source and 2 closed-source LLMs.
  4. Supervised fine-tuning (FT) of 5 open-source LLMs, showing average binary detection accuracy improvements of **+15.8%** (medical), **+11.4%** (legal), and **+11.6%** (finance) on test splits.

### Section 3 — Hallucination Taxonomy
Saxena et al. adopt the hierarchical taxonomy from Huang et al. (2023/2025), separating **Factuality Hallucinations** from **Faithfulness Hallucinations**:
1. **Factual Inconsistency (Cat I — Factuality)**: Real-world grounded claims containing factual contradictions (e.g., naming Yuri Gagarin instead of Neil Armstrong as the first person on the Moon).
2. **Factual Fabrication (Cat II — Factuality)**: Invented or unverifiable claims with no empirical grounding in established real-world knowledge (e.g., fabricating clinical trials or historical origins of unicorns).
3. **Instruction Inconsistency (Cat III — Faithfulness)**: Output diverges from explicit user directives or answers a related but different question rather than what was asked.
4. **Context Inconsistency (Cat V — Faithfulness)**: Output contradicts, omits, or fabricates facts relative to an explicitly supplied context passage.
5. **Logical Inconsistency (Cat IV — Faithfulness)**: Internal logical contradictions within the model's own generated output or reasoning chain.

### Section 4 — Related Work
Positions the paper relative to task-specific mitigation/verification (Zhao et al., 2020; Dziri et al., 2021; Jiang et al., 2023), attribution/entailment metrics (BEGIN, AIS, PARENT, TRUE), and LLM hallucination benchmarks (`HaluEval` by Li et al., 2023; `FELM` by Zhao et al., 2024 / Chen et al., 2023). The authors differentiate their work by targeting **high-stakes domain QA** (medical, legal, finance) and synthetic fine-grained generation.

### Section 5 — Source Datasets Used
- **Context-Free QA Sources (used for Cat I, II, III, IV)**:
  - **MedQuAD** (Ben Abacha & Demner-Fushman, 2019; `keivalya/MedQuad-MedicalQnADataset`): 16,407 NIH medical QA pairs across 37 question types; filtered by token length to **7,218** QA pairs.
  - **LegalQAv1** (`dzunggg/legal-qa-v1`): 3,742 legal QA pairs; filtered by length to **3,586** pairs (Sect. 5.2), though Sect. 6.3 states **3,742** were used.
  - **FIQA** (`llamafactory/fiqa`, subset of Maia et al., 2018): Filtered for noisy/non-question entries to **3,111** pairs (Sect. 5.3), though Sect. 6.3 states **2,150** pairs.
- **Context-Grounded QA Sources (used exclusively for Cat V — Context Inconsistency)**:
  - **PubMedQA** (Jin et al., 2019): Biomedical QA with PubMed abstract context + `long_answer`; **1,180** pairs reported in Sect. 6.3.
  - **CUAD-QA** (Hendrycks et al., 2021): Legal contract clauses + questions + short answers; **1,010** pairs reported in Sect. 6.3.

### Section 6 — Data Generation Setup & Validation

#### 6.1–6.2 Generation Methodology
- **Generator Model & Decoding**: **Llama3-8B** in inference mode (`temperature = 0.7`, `max_tokens = 1024`, `top_p = 1.0`). Chosen over GPT-3.5-Turbo based on a qualitative study (Appendix, pp. 4246–4247) showing Llama3-8B preserves sentence coherence better while following structural perturbation instructions.
- **Category-Specific Prompting Strategies**:
  - **Cat I — Termite Strategy (Fig. 2)**: Introduces dependency-parsing *modifiers* and instructs the LLM to selectively replace modifiers (leaving sentence structure intact like termites eating inner wood) to contradict the original factual detail, outputting `Hallucinated Answer` + `Replacement Details`.
  - **Cat II — Factual Fabrication (Fig. 3)**: Instructs the LLM to inject contextually relevant but unverifiable/fabricated facts into the original answer, outputting `Hallucinated Answer` + `Fabricated Information/Fact`.
  - **Cat III — Instruction Inconsistency (Fig. 4)**: Instructs the LLM to produce a response relevant to the topic/entities in the question that answers something *not* asked by the exact question.
  - **Cat V — Context Inconsistency (Fig. 5)**: Takes `(context, question, original_answer)` from PubMedQA/CUAD-QA and instructs the LLM to introduce contradictions or unsupported elements relative to the context, outputting `Hallucinated Answer` + `Contextual Inconsistency Introduced`.
  - **Cat IV — Logical Inconsistency (Fig. 6)**: Instructs the LLM to introduce an internal contradiction into the answer, outputting `Hallucinated answer` + `Logical Inconsistency Introduced`. *(Note: Fig. 6's caption in the published PDF contains a copy-paste error calling it "instruction inconsistency", while the prompt text inside Fig. 6 is for logical inconsistency.)*

#### 6.3 Post-Processing
1. Discard instances failing the structured output format.
2. Strip boilerplate prefixes (e.g., `"Here is the requested hallucinated answer:"`).
3. Detect safety/refusal responses (e.g., `"I'm sorry, I can't help with that"`) and re-prompt up to **5 times** before discarding.

#### 6.4 Qualitative Human Evaluation
- **Annotators**: 3 proficient English speakers (2 Master's students, 1 M.A. Linguistics; aged 24–30; 2 male, 1 female).
- **Sample Size**: **2,400** generated instances (1,000 Medical from MedQuAD + PubMedQA; 1,000 Legal from LegalQA + CUAD-QA; 400 Finance from FIQA).
- **Task**: Binary verification (`1` if both hallucinated and aligned with the specified target category, else `0`).
- **Agreement**: **Fleiss' $\kappa = 0.81$**; pairwise Cohen's $\kappa$: A–B = `0.83`, B–C = `0.79`, A–C = `0.84` (average = `0.82`).

### Section 7 — Dataset Statistics (Table 9)
Total generated **hallucinated** instances = **52,646**:

| Domain / Source | Cat I (Factual Incons.) | Cat II (Factual Fab.) | Cat III (Instruction Incons.) | Cat IV (Logical Incons.) | Cat V (Context Incons.) | Total |
|---|---:|---:|---:|---:|---:|---:|
| Medical (`MedQuAD`) | 6,373 | 6,994 | 7,117 | 4,902 | — | 25,386 |
| Legal (`LegalQAv1`) | 2,871 | 1,679 | 2,693 | 471 | — | 7,714 |
| Finance (`FIQA`) | 2,885 | 3,060 | 3,092 | 2,977 | — | 12,014 |
| Medical Context (`PubMedQA`) | — | — | — | — | 4,322 | 4,322 |
| Legal Context (`CUAD-QA`) | — | — | — | — | 3,210 | 3,210 |
| **Total** | **12,129** | **11,733** | **12,902** | **8,350** | **7,532** | **52,646** |

*Observation*: Notice severe generation attrition in Legal Cat IV (**471** instances vs. 2,871 in Cat I), and notice that PubMedQA (4,322) and CUAD-QA (3,210) exceed the stated 1,180 / 1,010 source counts, indicating multi-pass sampling per source item.

### Section 8 — Experiments & Results
- **Evaluated Models**:
  - *Zero-Shot (Tables 10–12, full benchmark)*: Mistral 7B, Zephyr 7B, Llama2-chat-7B, Llama2-chat-13B, Llama3-8B, `gemini-1.5-flash-001`, `GPT-3.5-Turbo`.
  - *Zero-Shot vs. Fine-Tuned (Tables 13–15, 10% test split)*: Same 5 open-source LLMs; **80:10:10** train/val/test split grouped so **no test question appears in train or val**.
  - *Fine-Tuning Hyperparameters*: Adam optimizer, `epochs = 2`, `lr = 2e-4`, `batch_size = 8`.
- **Detection Task Formulation (CRITICAL)**:
  - Both zero-shot and fine-tuned experiments use a **binary prompt**: *"determine whether the answer is hallucinated or not. Output 'Non-Hallucinated' if its non-hallucinated, 'Hallucinated' if it is hallucinated."* (`max_tokens = 10`).
  - Even though results tables break accuracy down by source category (Cat I–V + Non-Hallu.), **the classifier itself does NOT predict which hallucination category occurred** — it only predicts binary `Hallucinated` vs. `Non-Hallucinated`!
- **Key Reported Results**:
  - Zero-shot closed models (`Gemini`, `GPT-3.5-Turbo`) show strong non-hallucinated bias (e.g., Medical Non-Hallu accuracy 94.1% / 87.3%) while missing ~70–82% of subtle Factual Inconsistency (18.5% / 24.3%) and Factual Fabrication (19.7% / 29.4%).
  - Fine-tuning substantially raises detection accuracy across all 5 open models; best fine-tuned model (`Llama3-8B`) reaches Average Accuracy **72.7%** (Med, +10.3 over ZS; mean gain across 5 models reported as +15.8%), **73.7%** (Legal, +10.6 over ZS; mean +11.4%), **72.0%** (Finance, +11.0 over ZS; mean +11.6%).

### Section 9 & Appendix — Conclusion, Future Work, & Natural Error Analysis
- **Explicit Future Direction**: *"Another promising direction is to explore the effect of introducing multiple hallucination types within a single instance to better mimic real-world complexity and stress-test model reliability."* (p. 4246).
- **Appendix Natural LLM Study**: 200 sampled questions/domain from MedQuAD, LegalQAv1, FIQA answered by Mistral-7B, Llama3-7B, GPT-3.5-Turbo and annotated by 3 raters (majority $\ge 2/3$). Across all domains, **Factual Fabrication** and **Factual Inconsistency** dominate natural errors (Med: 33.7% FF, 28.9% FI, 21.7% LI, 15.7% II; Legal: 36.8% FI, 26.5% FF, 20.6% II, 16.2% LI; Finance: 42.4% FI, 21.2% FF, 19.7% LI, 16.7% II).

---

## 3. Repository Audit & Critical Limitations

### Live Audit of `ashitas96/fine_grained_hallucinated_data` (Verified 2026-11-12)
Querying the GitHub REST API (`https://api.github.com/repos/ashitas96/fine_grained_hallucinated_data`) revealed:
- **Repository Contents**: Only a single 71-byte `README.md` (`# A Framework to Synthetically Generate Fine-grained Hallucinated Data`), 2 commits dated `2024-06-11`, `main` branch only, **0 releases**, **0 forks**, **no license file**, and **zero dataset files or generation/training scripts**.
- **Hugging Face Search**: No dataset matching `fine_grained_hallucinated` or authored by `ashitas96` exists on Hugging Face Hub.
- **Implication**: The 52,646 generated instances are **not publicly downloadable**. However, all five underlying source datasets (`MedQuAD`, `LegalQAv1`, `FIQA`, `PubMedQA`, `CUAD-QA`) ARE publicly accessible, and the exact generation prompts (Figs. 2–6) and hyperparameters are published in the paper—allowing us to faithfully reconstruct single-label samples where needed and build our multi-label extension transparently.

### Methodological Gaps & Weaknesses in the Base Paper
1. **Binary Detection Collapse**: The models never perform 5-class or multi-label category prediction; they only predict `Hallucinated` vs. `Non-Hallucinated`.
2. **Single-Category Assumption**: Each synthetic instance injects exactly one hallucination type, ignoring co-occurring hallucinations within complex responses.
3. **Dataset Disjointness for Context Inconsistency**: Cat V (`Context Inconsistency`) exists only on `PubMedQA` and `CUAD-QA`, while Cat I–IV exist only on context-free QA (`MedQuAD`, `LegalQAv1`, `FIQA`). No unified instance tests Context Inconsistency alongside Cat I–IV!
4. **Accuracy-Only Evaluation**: No Precision, Recall, F1, AUROC, PR-AUC, Hamming Loss, or confusion matrices are reported—despite severe imbalances (e.g., Legal Cat IV = 471 vs. Cat I = 2,871) and non-hallucinated prediction bias in baseline LLMs.
5. **Compute-Heavy Detector Choice**: Fine-tuning 7B–13B decoder LLMs via text-generation (`lr=2e-4`, LoRA details unspecified) is expensive at inference time and lacks calibrated per-category probabilities for threshold tuning.
6. **Human Annotation Scope**: Validation checked *binary alignment* with a single intended category (`1`/`0`), not whether unintended secondary categories (e.g., a Termite replacement also breaking internal logic, or an Instruction deviation introducing fabrication) accidentally appeared!
7. **Inconsistent Dataset Filtering Counts**: Reported source counts differ slightly between Section 5 (7,218 / 3,586 / 3,111) and Section 6.3 (7,218 / 3,742 / 2,150).

