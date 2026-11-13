# Multi-Label Hallucination Combination Matrix & Quotas (Phase 2)

Canonical label vector ordering: $\mathbf{y} = [\text{FI}, \text{FF}, \text{II}, \text{LI}, \text{CI}] \in \{0, 1\}^5$:
- **FI**: `factual_inconsistency` (real-world factual contradiction)
- **FF**: `factual_fabrication` (invented/unverifiable external entity, trial, statute, or fact)
- **II**: `instruction_inconsistency` (divergence from user's explicit question/directive)
- **LI**: `logical_inconsistency` (internal self-contradiction within the response)
- **CI**: `context_inconsistency` (unfaithfulness to supplied context $c$; defined **only when $c \neq \varnothing$**)

---

## 1. Exhaustive Audit of All $2^5 = 32$ Binary Combinations

Out of 32 mathematically possible vectors, **16 have $\text{CI} = 1$** and are structurally invalid whenever `has_context = False` (`MedQuAD`, `LegalQAv1`, `FIQA`). Even among structurally permissible vectors, high-cardinality combinations carrying both directive divergence (`II`) and multiple contradictory constraints (`LI`, `FI`, `FF`, `CI`) can cause semantic collapse in short answers (`< 3` clauses). We therefore classify every vector into one of four governance tiers: **Active**, **Restricted (Clause-Gated)**, **Conditional Stress (Pilot-Gated)**, or **Excluded / Structurally Invalid**.

### 1.1 Context-Independent / Context-Optional Vectors ($\text{CI} = 0$: 16 Vectors)

| Vector `[FI,FF,II,LI,CI]` | Pattern | $|y|_1$ | Tier | Governance Status & Rationale |
|---|---|---:|---|---|
| `[0,0,0,0,0]` | **Clean** | 0 | G0 | **Active** — Verified unmodified reference answers across all 5 sources (includes `864` context-present clean controls). |
| `[1,0,0,0,0]` | **FI** | 1 | G1 | **Active** — Single-label factual contradiction; includes `PubMedQA`/`CUAD-QA` non-CI controls. |
| `[0,1,0,0,0]` | **FF** | 1 | G1 | **Active** — Single-label invented external claim/source; includes `PubMedQA`/`CUAD-QA` non-CI controls. |
| `[0,0,1,0,0]` | **II** | 1 | G1 | **Active** — Single-label topical-shift instruction miss; includes `PubMedQA`/`CUAD-QA` non-CI controls. |
| `[0,0,0,1,0]` | **LI** | 1 | G1 | **Active** — Single-label internal contradiction; includes `PubMedQA`/`CUAD-QA` non-CI controls. |
| `[1,1,0,0,0]` | **FI+FF** | 2 | G2 | **Active** — Distinct spans: mutated real fact (`FI`) + invented external authority (`FF`). |
| `[1,0,0,1,0]` | **FI+LI** | 2 | G2 | **Active** — Real-world factual error plus mutually inconsistent statement inside response. |
| `[0,1,0,1,0]` | **FF+LI** | 2 | G2 | **Active** — Fabricated external claim plus internal contradiction in response logic. |
| `[1,0,1,0,0]` | **FI+II** | 2 | G2 | **Active** — Topical-shift miss (`II`) embedding a real-world factual falsehood (`FI`). |
| `[0,1,1,0,0]` | **FF+II** | 2 | G2 | **Active** — Topical-shift miss (`II`) supported by a fabricated external citation/entity (`FF`). |
| `[0,0,1,1,0]` | **II+LI** | 2 | G2 | **Active (No-Ctx Only)** — Answers wrong aspect (`II`) while contradicting itself across clauses (`LI`). |
| `[1,1,0,1,0]` | **FI+FF+LI** | 3 | G3 | **Restricted ($\ge 3$ clauses / $\ge 40$ words)** — Clean three-span realization across all 5 sources. |
| `[1,1,1,0,0]` | **FI+FF+II** | 3 | G3 | **Restricted ($\ge 3$ clauses, No-Ctx)** — Wrong aspect (`II`) containing a real-fact error (`FI`) + fake authority (`FF`). |
| `[1,0,1,1,0]` | **FI+II+LI** | 3 | G3 | **Restricted ($\ge 3$ clauses, No-Ctx)** — Wrong aspect (`II`) with factual falsehood (`FI`) and self-contradiction (`LI`). |
| `[0,1,1,1,0]` | **FF+II+LI** | 3 | G3 | **Restricted ($\ge 3$ clauses, No-Ctx)** — Wrong aspect (`II`) with fabrication (`FF`) and self-contradiction (`LI`). |
| `[1,1,1,1,0]` | **FI+FF+II+LI** | 4 | G4 | **Conditional Stress ($\ge 4$ clauses / $\ge 60$ words, No-Ctx)** — Pilot-gated: retained only if E0 human $\kappa \ge 0.65$ and spillover $< 20\%$. |

### 1.2 Context-Dependent Vectors ($\text{CI} = 1$: 16 Vectors)

> **Structural Constraint**: All 16 vectors below are **strictly invalid (`0` quota)** when `has_context = False` (`MedQuAD`, `LegalQAv1`, `FIQA`) and permitted **only** on `PubMedQA` and `CUAD-QA`.

| Vector `[FI,FF,II,LI,CI]` | Pattern | $|y|_1$ | Tier | Governance Status & Rationale |
|---|---|---:|---|---|
| `[0,0,0,0,1]` | **CI** | 1 | G1 | **Active** — Contradicts or unsupportedly extrapolates supplied context $c$ without external `FI`/`FF` noise. |
| `[1,0,0,0,1]` | **FI+CI** | 2 | G2 | **Active** — Contradicts supplied passage finding (`CI`) AND asserts false external domain fact (`FI`). |
| `[0,1,0,0,1]` | **FF+CI** | 2 | G2 | **Active** — Distorts passage (`CI`) AND adds fabricated external citation/trial/statute (`FF`). |
| `[0,0,0,1,1]` | **LI+CI** | 2 | G2 | **Active** — Contradicts passage (`CI`) AND contradicts an earlier clause inside the response (`LI`). |
| `[0,0,1,0,1]` | **II+CI** | 2 | G2 | **Active** — Misses exact ask (`II`) AND misstates supplied passage (`CI`). |
| `[1,1,0,0,1]` | **FI+FF+CI** | 3 | G3 | **Restricted ($\ge 3$ clauses / $\ge 40$ words)** — Clear 3-span realization across `PubMedQA`/`CUAD-QA`. |
| `[1,0,0,1,1]` | **FI+LI+CI** | 3 | G3 | **Restricted ($\ge 3$ clauses / $\ge 40$ words)** — Clear 3-span realization across `PubMedQA`/`CUAD-QA`. |
| `[0,1,0,1,1]` | **FF+LI+CI** | 3 | G3 | **Restricted ($\ge 3$ clauses / $\ge 40$ words)** — Clear 3-span realization across `PubMedQA`/`CUAD-QA`. |
| `[1,0,1,0,1]` | **FI+II+CI** | 3 | G3 | **Restricted ($\ge 3$ clauses / $\ge 40$ words)** — Included; requires topical-shift `II` alongside external `FI` and passage error `CI`. |
| `[0,1,1,0,1]` | **FF+II+CI** | 3 | G3 | **Excluded from Full Core** — High risk of collapsing `II`+`FF`+`CI` on short PubMedQA/CUAD snippets; merged/covered via G4 stress only if needed. |
| `[0,0,1,1,1]` | **II+LI+CI** | 3 | G3 | **Excluded from Full Core** — Without `FI`/`FF` anchors, `II` + `LI` + `CI` in brief grounded answers causes human ambiguity between directive mismatch vs passage distortion. |
| `[1,1,0,1,1]` | **FI+FF+LI+CI** | 4 | G4 | **Conditional Stress ($\ge 4$ clauses / $\ge 60$ words)** — All 4 content/context/logic defects without directive shift; highest-plausibility 4-label case. |
| `[1,1,1,0,1]` | **FI+FF+II+CI** | 4 | G4 | **Conditional Stress ($\ge 4$ clauses / $\ge 60$ words)** — Pilot-gated stress test. |
| `[1,0,1,1,1]` | **FI+II+LI+CI** | 4 | G4 | **Conditional Stress ($\ge 4$ clauses / $\ge 60$ words)** — Pilot-gated stress test. |
| `[0,1,1,1,1]` | **FF+II+LI+CI** | 4 | G4 | **Excluded (Semantic Collapse Risk)** — Combining `II`, `LI`, `FF`, and `CI` cleanly without `FI` produces noisy/contradictory boundaries in brief answers. |
| `[1,1,1,1,1]` | **FI+FF+II+LI+CI** | 5 | G5 | **Conditional Stress (`60` samples max, $\ge 5$ clauses / long-context)** — Hard ceiling; retained only if E0 confirms coherence $\ge 3.5/5$ and human verification passes. |

---

## 2. Explicit Generation Quotas (Non-Proportional Rebalancing)

Raw Phase 1 data is heavily skewed toward `PubMedQA` (`62.6%` of clean rows). We explicitly cap `PubMedQA` at `20.0%` and allocate balanced domain/context quotas across **12,000 target synthetic + clean records** (`train = 9,600`, `val = 1,200`, `test_synthetic = 1,200`). All quotas respect verified Phase 1 split capacities (zero group reuse needed on `legal_qa_v1` `test_synthetic` where `180 <= 298` available).

### 2.1 Domain, Dataset, Context, and Split Quotas (`N = 12,000`)

| Domain | Source Dataset | Context $c$ | Licensing / Release Track | Share | Total Quota | `train` (80%) | `val` (10%) | `test_synthetic` (10%) |
|---|---|---|---|---:|---:|---:|---:|---:|
| **Medical** (`40%`) | `medquad` | Absent (`False`) | `CC-BY-4.0` (Open w/ attrib.) | 20.0% | **2,400** | 1,920 | 240 | 240 |
| **Medical** (`40%`) | `pubmedqa` (`pqa_labeled` + `unlabeled`) | Present (`True`) | `MIT` (Open w/ PMID) | 20.0% | **2,400** | 1,920 | 240 | 240 |
| **Legal** (`35%`) | `legal_qa_v1` | Absent (`False`) | `unresolved` (**Local Only**) | 15.0% | **1,800** | 1,440 | 180 | 180 |
| **Legal** (`35%`) | `cuad_qa` | Present (`True`) | `CC-BY-4.0` (Zenodo `4595826`) | 20.0% | **2,400** | 1,920 | 240 | 240 |
| **Finance** (`25%`) | `fiqa` | Absent (`False`) | `upstream_caution` (**Local Only**) | 25.0% | **3,000** | 2,400 | 300 | 300 |
| **Totals** | **All 5 Sources** | **60% No-Ctx / 40% Ctx** | **7,200 Open / 4,800 Local** | **100.0%** | **12,000** | **9,600** | **1,200** | **1,200** |

*Separate `test_natural` allocation*: **`300`–`450`** human-annotated natural LLM responses drawn exclusively from `data/splits/source_test_natural_pool.jsonl` (`2,025` held-out clean source items), balanced `40%` Medical / `35%` Legal / `25%` Finance (`60%` No-Ctx / `40%` Ctx) — never mixed into `test_synthetic`.

### 2.2 Full-Scale Label-Combination Quotas (`12,000` Target)

| Tier | Combination | No-Ctx Quota (`7,200`) | Ctx-Present Quota (`4,800`) | Total (`12,000`) | Share |
|---|---|---:|---:|---:|---:|
| **G0** | `Clean` `[0,0,0,0,0]` | 1,296 | 864 | **2,160** | 18.0% |
| **G1** | `FI` `[1,0,0,0,0]` | 600 | 200 | **800** | 6.7% |
| **G1** | `FF` `[0,1,0,0,0]` | 600 | 200 | **800** | 6.7% |
| **G1** | `II` `[0,0,1,0,0]` | 500 | 160 | **660** | 5.5% |
| **G1** | `LI` `[0,0,0,1,0]` | 460 | 160 | **620** | 5.2% |
| **G1** | `CI` `[0,0,0,0,1]` | 0 | 720 | **720** | 6.0% |
| **G2** | `FI+FF` `[1,1,0,0,0]` | 564 | 120 | **684** | 5.7% |
| **G2** | `FI+LI` `[1,0,0,1,0]` | 460 | 84 | **544** | 4.5% |
| **G2** | `FF+LI` `[0,1,0,1,0]` | 380 | 72 | **452** | 3.8% |
| **G2** | `FI+II` `[1,0,1,0,0]` | 420 | 60 | **480** | 4.0% |
| **G2** | `FF+II` `[0,1,1,0,0]` | 420 | 60 | **480** | 4.0% |
| **G2** | `II+LI` `[0,0,1,1,0]` | 240 | 0 | **240** | 2.0% |
| **G2** | `FI+CI` `[1,0,0,0,1]` | 0 | 360 | **360** | 3.0% |
| **G2** | `FF+CI` `[0,1,0,0,1]` | 0 | 360 | **360** | 3.0% |
| **G2** | `LI+CI` `[0,0,0,1,1]` | 0 | 240 | **240** | 2.0% |
| **G2** | `II+CI` `[0,0,1,0,1]` | 0 | 240 | **240** | 2.0% |
| **G3** | `FI+FF+LI` `[1,1,0,1,0]` | 420 | 120 | **540** | 4.5% |
| **G3** | `FI+FF+II` `[1,1,1,0,0]` | 320 | 0 | **320** | 2.7% |
| **G3** | `FI+II+LI` `[1,0,1,1,0]` | 180 | 0 | **180** | 1.5% |
| **G3** | `FF+II+LI` `[0,1,1,1,0]` | 160 | 0 | **160** | 1.3% |
| **G3** | `FI+FF+CI` `[1,1,0,0,1]` | 0 | 220 | **220** | 1.8% |
| **G3** | `FI+LI+CI` `[1,0,0,1,1]` | 0 | 160 | **160** | 1.3% |
| **G3** | `FF+LI+CI` `[0,1,0,1,1]` | 0 | 120 | **120** | 1.0% |
| **G3** | `FI+II+CI` `[1,0,1,0,1]` | 0 | 100 | **100** | 0.8% |
| **G4*** | `FI+FF+II+LI` `[1,1,1,1,0]` | 180 | 0 | **180** | 1.5% |
| **G4*** | `FI+FF+LI+CI` `[1,1,0,1,1]` | 0 | 60 | **60** | 0.5% |
| **G4*** | `FI+FF+II+CI` `[1,1,1,0,1]` | 0 | 30 | **30** | 0.25% |
| **G4*** | `FI+II+LI+CI` `[1,0,1,1,1]` | 0 | 30 | **30** | 0.25% |
| **G5*** | `FI+FF+II+LI+CI` `[1,1,1,1,1]` | 0 | 60 | **60** | 0.5% |
| **Total** | **29 Active/Gated Patterns** | **7,200** | **4,800** | **12,000** | **100.0%** |

*\*G4/G5 quotas are conditional on Pilot E0 passing the stress-tier acceptance gate; if pruned, their 360 slots redistribute proportionally into G2/G3.*

- **Resulting Marginal Positives Across `12,000`**: `FI = 4,748` (`39.6%`), `FF = 4,466` (`37.2%`), `II = 3,160` (`26.3%`), `LI = 3,586` (`29.9%`), `CI = 2,700` (`22.5%` overall / `56.25%` within context-present items, leaving **`2,100` (`43.75%`) context-present items as strict `CI = 0` shortcut controls** across `864` Clean + `720` G1 + `396` G2 + `120` G3).

