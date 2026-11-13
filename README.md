# Multi-Label Fine-Grained Hallucination Detection Across Domain-Specific QA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](http://localhost:8501)
[![Tests Passing](https://img.shields.io/badge/Tests-51%2F51%20Passing-brightgreen.svg)]()
[![Paper: IEEE Format](https://img.shields.io/badge/Paper-IEEEtran-red.svg)](docs/manuscript/paper.tex)

An end-to-end research framework and empirical benchmark for **multi-label fine-grained hallucination detection** across medical, legal, and financial question answering. 

Unlike existing benchmarks that artificially enforce a mutually exclusive single-label assumption, this framework models compound, multi-modal hallucination phenomena across a formal 5-dimensional category space $\mathbf{y} \in \{0, 1\}^5$:
1. `factual_inconsistency` (**FI**): Contradicts established real-world facts or clinical guidelines.
2. `factual_fabrication` (**FF**): Asserts non-existent entities, case citations, statutory sections, or dates.
3. `instruction_inconsistency` (**II**): Violates explicit query constraints, formatting rules, or sub-questions.
4. `logical_inconsistency` (**LI**): Exhibits internal self-contradiction, circular reasoning, or non-sequiturs.
5. `context_inconsistency` (**CI**): Contradicts or extrapolates beyond reference context ($y_{\text{CI}} \equiv 0$ when $c = \emptyset$).

---

## Key Empirical Findings

1. **Synthetic Generation Paradigm (Pilot E0, 296 Slots)**:
   - **Strategy A (Sequential Chaining)** suffered from severe compounding semantic drift ($38.8\%$ invalid rate).
   - **Strategy B (Joint Single-Pass)** won empirically, achieving a **$2.5\%$ invalid rate**, **$93.95\%$ macro target recall**, and **$41.03\%$ exact vector match**.
2. **Detection Benchmark**:
   - `TFIDF_ovr_logreg` achieved **0.7253 Macro-$F_1$** and **0.7775 Mean PR-AUC**, with Context Inconsistency reaching **0.8750 $F_1$**.
   - `DistilBERT` fine-tuning demonstrated superior probability calibration, achieving an Expected Calibration Error (ECE) of **0.0789** after validation temperature scaling.
3. **Sub-10ms Inference Guarantee**:
   - The interactive diagnostic engine operates with a mean latency of **$6.45\text{ ms}$** on CPU.

| Model Architecture | Macro-$F_1$ | Micro-$F_1$ | Subset Acc | Hamming Loss | Mean PR-AUC | Mean ECE | Mean Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TFIDF_ovr_logreg`** | **0.7253** | **0.6744** | **0.2400** | **0.2240** | 0.7775 | 0.1624 | **0.1255** |
| `TFIDF_linear_svc` | 0.6844 | 0.6420 | 0.2000 | 0.2320 | **0.7995** | 0.1808 | 0.1348 |
| `TFIDF_classifier_chain` | 0.6413 | 0.5979 | 0.1200 | 0.3120 | 0.7105 | 0.1745 | 0.1475 |
| `DistilBERT` (Fine-Tuned) | 0.5469 | 0.5333 | 0.0800 | 0.3840 | 0.5252 | **0.0789** | 0.1714 |

---

## Repository Structure

```
├── app/                        # Interactive Streamlit Diagnostic Workbench
│   ├── main.py                 # Streamlit application entrypoint
│   └── components/             # UI helpers, gauges, and attribution highlights
├── configs/                    # Reproducible YAML configurations
│   ├── dataset.yaml            # Dataset splits and source definitions
│   ├── generation_free_groq.yaml # Free-tier LLM generation pipeline
│   ├── model.yaml              # Transformer and baseline architectures
│   └── training.yaml           # Training hyper-parameters and threshold tuning
├── data/
│   ├── synthetic/full/         # Clean, leak-free synthetic train/val/test splits
│   └── natural/                # Multi-model natural QA unannotated test set
├── docs/                       # Research specifications and manuscripts
│   ├── manuscript/             # Camera-ready IEEE paper, figures, and Overleaf zip
│   │   ├── main.tex            # IEEEtran conference LaTeX source
│   │   ├── references.bib      # BibTeX bibliography
│   │   ├── images/             # All 8 publication figures (PNG & PDF)
│   │   └── overleaf_project.zip# 1-click Overleaf upload bundle
│   └── architecture.md         # Comprehensive system architecture
├── notebooks/
│   └── colab_runner.ipynb      # Turnkey Google Colab multi-GPU scaling runner
├── results/                    # Serialized benchmark reports, tables & models
│   ├── baselines/              # Trained OvR LogReg, LinearSVC, ClassifierChain
│   ├── runs/seed_42/           # DistilBERT checkpoints, calibration, eval reports
│   └── tables/                 # Auto-generated LaTeX, Markdown, CSV comparison tables
├── src/                        # Core Python package
│   ├── data/                   # Token budget allocator, preprocessing, split hash
│   ├── generation/             # Joint & sequential synthetic generators, auditor
│   ├── models/                 # Inference engine, calibration, baseline suite
│   └── training/               # Multi-label PyTorch trainer & threshold tuner
└── tests/                      # Pytest suite (51 / 51 passing)
```

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/MultiLabel-Hallucination-Detection.git
cd MultiLabel-Hallucination-Detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify environment and module integrity:
```bash
pytest -q
# Expected: 51 passed
```

### 3. Launch Interactive Streamlit UI
Run the local diagnostic workbench:
```bash
streamlit run app/main.py
```
Access at `http://localhost:8501` to test custom QA pairs, visualize real-time confidence gauges, and inspect sentence-level attribution highlights.

### 4. Overleaf LaTeX Paper
To compile or edit the research paper in Overleaf:
1. Download or upload `docs/manuscript/overleaf_project.zip` to Overleaf (**New Project $\to$ Upload Project**).
2. Click **Recompile** to generate the two-column IEEE formatted PDF.

---

## Citation

If you use this benchmark, code, or methodology in your research, please cite:

```bibtex
@inproceedings{tangudu2026multilabel,
  author    = {Uddagiri Sirisha and Jaya Krishna Tangudu and Siva Sai Swaroop Yanduri and Pavan Deep P and Chandra Shekhar},
  title     = {An Automated Multi-Label Fine-Grained Hallucination Detection and Calibration Framework Across Domain-Specific Question Answering},
  booktitle = {Proceedings of the IEEE Conference on Natural Language Processing and Knowledge Engineering},
  year      = {2026}
}
```
