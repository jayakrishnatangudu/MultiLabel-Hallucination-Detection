# Human Annotation & Ground-Truth Governance Guidelines (Phase 2)

## 1. Core Scientific Rules

1. **Independence of Ground Truth**: Neither synthetic generator `intended_labels` nor automated `auto_detected_labels` (rule checks or LLM cross-judge predictions) may be substituted for human ground truth.
2. **Strict Blinding**: Human annotators view ONLY:
   - `domain`, `question`, `context` (`None` if context-free), `original_answer` (as a domain verification aid when grading synthetic & reference-backed responses), and `candidate_response`.
   - Annotators **NEVER** see `intended_labels`, `auto_detected_labels`, `generation_strategy`, `generation_model`, or whether an item is synthetic (`perturbed`), synthetic-clean (`[0,0,0,0,0]`), or natural LLM output (`test_natural`).
   - Note on `test_natural`: When judging authentic LLM responses, raters inspect `(question, context, original_answer, candidate_response)` with explicit instruction that `original_answer` is *one* valid reference (not necessarily exhaustive—other factually accurate phrasings are `FI=0`!).
3. **Co-Occurrence Without Label Collapse**: Each of the five binary labels is scored independently (`0` or `1`). Zero, one, or multiple labels may be `1` on a single response.

---

## 2. Operational Definitions & Decision Tests

### Label 0 — `factual_inconsistency` (`FI`)
- **Definition**: The response states an assertion about established real-world entities, mechanisms, numbers, timelines, legal rules, or financial accounting principles that **contradicts verifiable reality**.
- **Operational Test**: *"Replacing or reversing a known real-world fact makes the claim demonstrably FALSE—not merely fabricated from thin air, and not merely misquoting a passage-local experiment/contract number."*
- **Mark `FI = 1`**: False physiological mechanism (`"insulin raises blood sugar"`), wrong tax threshold/rule (`"Roth IRA contributions are tax-deductible"`), inverted statutory rule, wrong disease transmission mechanism.
- **Mark `FI = 0`**: An accurate paraphrase using slightly different words from `original_answer`; OR misstating a local study sample size (`n=50` vs `n=500` in a PubMed abstract) -> that is `CI`, NOT `FI`.

### Label 1 — `factual_fabrication` (`FF`)
- **Definition**: The response introduces **invented, non-existent, or unverifiable specific entities, sources, statistics, citations, clinical trials, regulatory forms, statutes, or court precedents**.
- **Operational Test**: *"Did the response conjure up a concrete external authority, fake entity, or nonexistent metric/precedent rather than mutating a real known concept (`FI`) or merely inventing a clause attributed to the provided document (`CI`)?"*
- **Mark `FF = 1`**: `"According to the 2022 Oslo Multicenter Trial (n=12,400)..."`, `"IRS Form 8991-Q allows..."`, `"In *Henderson v. Apex Trust* (2019), the Supreme Court held..."`, invented enzyme/gene/drug brand names.

### Label 2 — `instruction_inconsistency` (`II`)
- **Definition**: The response fails to follow the user's explicit question, task directive, or constraint (e.g., discusses causes/symptoms when asked for treatment/prevention, ignores a conditional qualification, or answers a related but wrong question).
- **Operational Test**: *"Did the response answer what the user actually asked—regardless of whether the statements inside it are true or false?"*
- **Note**: A response can be **factually accurate (`FI=0, FF=0`) yet still `II=1`** (accurate facts answering the wrong aspect), OR **`II=1` + `FI=1` simultaneously** (answers the wrong aspect AND inserts a factual error inside that off-topic discussion).

### Label 3 — `logical_inconsistency` (`LI`)
- **Definition**: The response contradicts itself internally, states mutually incompatible numbers/conditions, or draws an invalid inference within its own text.
- **Operational Test**: *"Can two statements inside `{candidate_response}` both be true at the same time, or does the conclusion negate the stated premise?"*
- **Example**: Sentence 1 says `"Always administer drug X before meals due to rapid absorption"`, and Sentence 3 says `"Never take drug X near meals; only dose 4 hours post-prandially"`, or `"Interest is completely non-deductible; therefore taxpayers may deduct 50% of this interest under Schedule A"`.

### Label 4 — `context_inconsistency` (`CI`)
- **Definition** (`has_context = True` ONLY): The response contradicts factual statements in `{context}`, inverts the context's conclusion/polarity, drops required explicit caveats/exceptions, or asserts document-specific details as stated in `{context}` that are unsupported/absent from `{context}`.
- **Hard Constraint**: Whenever `has_context = False` (`MedQuAD`, `LegalQAv1`, `FIQA`), **`CI` MUST BE `0`**.

---

## 3. Disambiguating `FI` vs. `FF` vs. `CI` (Critical Boundary Rules)

Confusion between `FI`, `FF`, and `CI` is the highest source of annotation noise. Annotators **must** follow one overarching invariant plus the decision matrix below:

> **Core Invariant (`CI` Never Auto-Triggers `FI`)**: A contradiction with the supplied context (`CI = 1`) **must NOT automatically be treated as a real-world factual contradiction (`FI = 1`)**. Only mark `FI = 1` when the response asserts a falsehood against verifiable external real-world/domain knowledge *beyond* context-local facts. Likewise, automated cross-judge predictions (`auto_detected_labels`) are **strictly excluded** from annotator packets and adjudication.

| Phenomenon Observed in Candidate Response | `FI` | `FF` | `CI` | Reasoning & Tie-Break Rule |
|---|:---:|:---:|:---:|---|
| Real-world error on general domain fact while faithful to `{context}` (`"Staphylococcus aureus is a DNA virus..."` + accurate PubMed abstract summary) | **1** | 0 | **0** | Truth of viral vs. bacterial taxonomy is established real-world fact; supplied context wasn't distorted. |
| Misquotes passage-specific study/contract detail whose truth resides ONLY in `{context}` (`"The contract is governed by Texas law"` when contract specifies `New York`, or `"n=25"` vs `"n=250"`) | **0** | 0 | **1** | Do NOT double-penalize with `FI=1`; passage-local numbers/clauses belong strictly to **Faithfulness (`CI`)**. |
| Contradicts `{context}` conclusion AND mutates a general established biomedical/legal fact outside `{context}` | **1** | 0 | **1** | Both unfaithfulness to `{context}` (`CI=1`) and external factual falsehood (`FI=1`) exist in distinct claims. |
| Attributes an invented claim directly *to the provided document* (`"The contract imposes a 90-day exclusivity penalty"` when no exclusivity clause exists in `{context}`) | 0 | **0** | **1** | Document-grounded unfaithfulness (`CI=1`) per Faithfulness taxonomy; do not double-mark as `FF=1`. |
| Invents an *external* authority/trial/statute (`"As confirmed by the 2021 Geneva Cohort Study..."` or `"IRS Revenue Bulletin 44-9"`) while remaining faithful to `{context}` | 0 | **1** | **0** | Pure external fabrication (`FF=1`); supplied document itself was not misquoted (`CI=0`). |
| Invents an *external* trial/statute (`FF=1`) AND contradicts `{context}` (`CI=1`) in separate claims | 0 | **1** | **1** | Clean dual-label co-occurrence (`FF+CI`). |
| Response cites a real fact AND invents a fake clinical trial to support it | 0 | **1** | 0 | Real fact is consistent (`FI=0`); fake trial triggers `FF=1`. |

---

## 4. Concrete Annotated Domain Examples

1. **Medical (`PubMedQA`, Context-Grounded — `FI=1, FF=0, II=0, LI=0, CI=0`)**:
   - *Context*: Summarizes randomized trial finding oral magnesium did not reduce migraine frequency vs placebo (`p=0.38`).
   - *Response*: *"The randomized trial reported no statistically significant reduction in migraine frequency with oral magnesium compared to placebo (p=0.38). Magnesium acts clinically as a loop diuretic blocking sodium-potassium cotransporters in the renal ascending limb."*
   - *Labels*: `[1, 0, 0, 0, 0]` (`CI=0` because trial summary matches abstract verbatim; `FI=1` because magnesium is false-described as a loop diuretic).

2. **Legal (`CUAD-QA`, Context-Grounded — `FF=1, CI=1`)**:
   - *Context*: Distribution Agreement governed by laws of the State of Delaware; silence on international arbitration.
   - *Response*: *"The agreement specifies exclusive arbitration in Zurich under Article 19(c) of the 2018 UNCITRAL Commercial Enforcement Protocol, rather than state law."*
   - *Labels*: `[0, 1, 0, 0, 1]` (`CI=1` contradicts Delaware governing law / fabricated Zurich arbitration in contract; `FF=1` fabricates nonexistent `"2018 UNCITRAL Commercial Enforcement Protocol Article 19(c)"`).

3. **Finance (`FIQA`, No Context — `FI=1, II=1, LI=1`)**:
   - *Question*: *"How are short-term capital gains taxed when shares are held under 12 months?"*
   - *Response*: *"Long-term capital gains apply whenever assets are held over six months, qualifying for a flat 0% rate regardless of income bracket. However, because gains above zero always trigger ordinary income rates, taxpayers must report all long-term profits exclusively at maximum 37% short-term ordinary brackets."*
   - *Labels*: `[1, 0, 1, 1, 0]` (`II=1` pivots to long-term gains instead of asked short-term rules; `FI=1` claims >6 months = 0% flat rate; `LI=1` simultaneously asserts 0% rate and mandatory 37% ordinary bracket on the same profits).

---

## 5. Annotator Workflow, Calibration & Agreement Protocol

1. **Training & Calibration Round**: Before scoring any evaluation items, raters complete a **30-item calibration packet** covering all 5 labels, all 3 domains, Clean controls, and tricky `FI`/`FF`/`CI` boundaries, reviewing every disagreement together until pairwise Cohen's $\kappa \ge 0.75$ is reached on calibration.
2. **Independent Blind Coding**:
   - **Pilot (`N = 260`) & `test_synthetic_gold`**: At least **2 independent annotators** label every sample blindly (`human_observed_labels: {"rater_1": [...], "rater_2": [...]}`), recording brief span quotes for any `1` label and a `coherence_1_to_5` score.
   - **`test_natural` (`300`–`450` real LLM responses)**: **3 independent blind annotators** score every sample.
3. **Adjudication (`adjudicated_gold_labels`)**:
   - Any category bit where raters disagree (`1` vs `0`) triggers a dedicated **Adjudication Pass** by a third senior adjudicator (or 3-rater consensus discussion for `test_natural`), updating `adjudicated_gold_labels`, `annotation_explanation`, and setting `validation_status = "human_verified"` (or `"human_rejected"` if malformed/unreadable).
4. **Reported Agreement Metrics**:
   - Per-label and macro **Fleiss' $\kappa$** (when 3 raters) and **pairwise Cohen's $\kappa$** (2 raters).
   - **Per-category exact agreement (%)** and confusion pairs (especially `FI` vs `FF` vs `CI`).
   - **Target-Label Recall** (`P(gold_k=1 | intended_k=1)`) and **Unintended Spillover Rate** (`P(gold_k=1 | intended_k=0)`).

