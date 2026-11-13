# Phase 2 Comparative Generation Pilot Plan (Suite E0)

## 1. Objective & Scope

Before committing compute or API budget to 12,000+ generations, Suite **E0** runs a strictly controlled, paired comparative pilot of **Strategy A (Sequential)**, **Strategy B (Joint Compositional)**, and **Strategy C (Controlled Span Editing)** on the **approved `260`-item core budget** (`80` paired source questions $\times 3$ strategies = `240` core generated candidates + `20` unmodified Clean controls), **plus an explicitly accounted additive `36`-output Strategy A order-sensitivity sub-experiment (`E0-Order`)** (`260 + 36 = 296` total candidate outputs/controls across **`100` unique source questions**).

- **Core Budget (`260` items — Unchanged)**: `80` perturbed base questions $\times 3$ strategies (`240`) + `20` unmodified Clean controls (`20`) = `260` evaluation items.
- **Additive `E0-Order` Sub-Experiment (`+36` outputs, `0` extra source questions)**: Reuses `18` multi-label items (`10` G2 + `8` G3) from inside the `80` base items under `reverse` and `randomized` Strategy A orderings (`forward` is shared from the `260` core without re-running). Total combined outputs when `E0-Order` is enabled = `296` (`276` generated slots + `20` Clean controls).
- **Split Lock**: Every single one of the **`100` unique source questions** (`80` paired perturbed + `20` Clean, `100` disjoint `source_group_id`s) is sampled exclusively from `data/splits/source_train.jsonl` using fixed seed `42` (`val`, `test_synthetic_pool`, and `test_natural_pool` remain 100% untouched). **Strategies A, B, and C run paired on the exact same `80` `(question, context, original_answer, intended_labels)` tuples.**

---

## 2. Exact Pilot Sample Allocation (`N = 260` Core + `36` Additive `E0-Order` = `296`)

### 2.1 Source & Domain Allocation (`100` Unique Source Questions / `source_group_id`s)
| Domain | Dataset | Context $c$ | Perturbed Base Items ($\times 3$ Core Strategies) | Core Generated (`240`) | Unmodified Clean Controls (`20`) | Core Total (`260`) | Additive `E0-Order` (`+36`) |
|---|---|---|---:|---:|---:|---:|---:|
| Medical (`40%`) | `medquad` | Absent | 16 | 48 | 4 | 52 | *(subset of 18)* |
| Medical (`40%`) | `pubmedqa` | Present | 16 | 48 | 4 | 52 | *(subset of 18)* |
| Legal (`35%`) | `legal_qa_v1` | Absent | 12 | 36 | 3 | 39 | *(subset of 18)* |
| Legal (`35%`) | `cuad_qa` | Present | 16 | 48 | 4 | 52 | *(subset of 18)* |
| Finance (`25%`) | `fiqa` | Absent | 20 | 60 | 5 | 65 | *(subset of 18)* |
| **Totals** | **5 Sources** | **60% No-Ctx / 40% Ctx** | **80** | **240** | **20** | **260** | **+36 (Combined: 296)** |

### 2.2 Cardinality & Combination Breakdown (`80` Paired Items + `20` Clean)
| Tier | Combinations Covered | Base Items | Core Generated (`A+B+C`) | Purpose in Pilot |
|---|---|---:|---:|---|
| **G0** | `Clean` `[0,0,0,0,0]` | 20 | *(20 unmodified; 8 on ctx, 12 no-ctx)* | Blind annotator & judge specificity baseline (including `4` PubMedQA + `4` CUAD-QA context clean controls). |
| **G1** | `FI (6)`, `FF (5)`, `II (5)`, `LI (5)`, `CI (4)` | 25 | 75 | Single-label baseline (`4` `CI=1` ctx + `2` reserved `with_context, CI=0` [`pubmedqa` + `cuad_qa`] + `19` no-ctx under Option R1 rebalance). |
| **G2** | All 10 valid pairs (`11` `CI=1` + `19` `CI=0`) | 30 | 90 | Core pair interaction test (`11` `CI=1` ctx + `2` reserved `with_context, CI=0` [`FI+FF` & `FF+LI`, `pubmedqa` + `cuad_qa`] + `17` no-ctx). |
| **G3** | 8 prioritized triples (`9` `CI=1` + `9` `CI=0`, $\ge 3$ clauses) | 18 | 54 | Multi-span separation (`9` `CI=1` triples on ctx + `9` `CI=0` triples no-ctx). |
| **G4/G5** | Stress (`4`-label `5` items + `5`-label `2` items, long multi-clause only) | 7 | 21 | Empirically test whether `|y|>=4` passes coherence & human agreement gates (`5` `CI=1` ctx + `2` `CI=0` no-ctx). |
| **Total** | **All Tiers** | **100** | **240 core gen + 20 clean = 260** | Paired A vs. B vs. C evaluation (+ `36` additive `E0-Order` Strategy A outputs = `296` total items). |

### 2.3 Strategy A Order-Sensitivity Add-On (`E0-Order`: `18` Subset Items -> `+36` Outputs)
Because Strategy A applies labels sequentially ($r_0 \to r_1 \to \dots \to r_k$), label order may induce recency bias or overwrite earlier edits. From the paired multi-label items (`|y| >= 2`) inside the `80`-item pilot, we tag a stratified **`18`-item order-sensitivity subset** (`10` G2 pairs + `8` G3 triples, `0` new source questions) and run Strategy A under **3 deterministic step orderings**:
1. **`forward` (Core Baseline)**: `II -> FI -> CI -> FF -> LI` (already included inside the `260`-item core pilot; `0` duplicate calls).
2. **`reverse` (Additive)**: `LI -> FF -> CI -> FI -> II` (`18` additional Strategy A outputs = `10*2 + 8*3 = 44` nominal sequential generator calls).
3. **`randomized` (Additive)**: Deterministic item-seeded permutation (`seed=42`, distinct from `forward`/`reverse` where $|y| \ge 3$ and shuffled on pairs; `18` additional Strategy A outputs = `44` nominal sequential generator calls).

---

## 3. Model & Execution Backend Design

Because local workstation inspection verified no local CUDA GPU or `ollama` binary, [configs/generation.yaml](file:///D:/Major%20Project/MultiLabel-Hallucination-Detection/configs/generation.yaml) configures two independent API backends with verification flags (`model_availability_verified` remains `false` until live API smoke verification succeeds with local `.env` credentials):
- **Generator (`google_gemini`, family `gemini`)**: `gemini-3.8-flash` via Google Gemini Generative Language REST v1beta (`GEMINI_API_KEY`, `thinking_level: "low"`, deprecated `temperature`/`topP`/`topK` omitted; `pricing_verified: true` at `\$0.75/1M` input and `\$3.75/1M` output including thinking tokens).
- **Independent Cross-Judge (`openai`, family `gpt`)**: `gpt-5.4-mini` via OpenAI Chat Completions (`OPENAI_API_KEY`, `reasoning_effort: "none"`, `max_completion_tokens`, `response_format: {"type": "json_object"}`; `pricing_verified: true` at `\$0.75/1M` input and `\$4.50/1M` output; strictly diagnostic and non-Gemini).

### Exact Pilot Generation & Judge Call Budget (Nominal `1` Attempt/Slot; `max_retries = 3`)
- **Core Pilot (`260` items = `240` perturbed + `20` Clean controls)**:
  - Strategy A (`forward`, `80` items): `169` nominal generator step calls ($\sum_{i=1}^{80} |y_i|_1$: `25*1 + 30*2 + 18*3 + 5*4 + 2*5`).
  - Strategy B (`joint`, `80` items): `80` nominal generator calls.
  - Strategy C (`span_editing`, `80` items): `80` nominal generator calls.
  - Clean Controls (`20` items): `0` generator calls.
  - **Core Subtotal**: `329` nominal generator calls + `260` diagnostic cross-judge calls = **`589` core nominal LLM calls** (expected accepted at $\ge 95\%$ gate: $\ge 228$ perturbed + `20` clean).
- **Additive `E0-Order` Sub-Experiment (`+36` outputs = `18` subset items $\times 2$ non-forward orders)**:
  - Strategy A (`reverse` `44` + `randomized` `44`): **`88`** extra nominal generator calls + **`36`** extra cross-judge calls = **`124`** extra nominal LLM calls.
- **Combined Totals (`296` items = `276` generated slots + `20` Clean)**:
  - **`417`** nominal generator calls (`1,251` worst-case retry ceiling at `3x`) + **`296`** cross-judge calls = **`713` total nominal LLM calls**. Projected Option B spend: ~`\$2.12` (`1x` nominal, `low` thinking) to ~`\$2.83` (`1.5x` retries).
- **Hard Pilot Budget Cap (`max_pilot_budget_usd = 10.0`)**: Enforced at runtime before each LLM request via worst-case single-call cost pre-check so execution halts cleanly and flushes completed artifacts before exceeding `\$10.00`.

### Context-Present Non-CI Controls (Approved Option R1 Rebalance)
Under approved **Option R1**, `4` of the `32` `CI=1` slots are swapped cleanly to `CI=0` (`00001`: 5->4 / `10000`: 5->6; `10001`: 3->2 / `11000`: 3->4; `01001`: 3->2 / `01010`: 3->4; `10101`: 2->1 / `11010`: 3->4), preserving all domain quotas (`32/28/20`), dataset quotas (`16/16/12/16/20`), `60%/40%` context balance (`48/32`), and cardinality tier totals (`25/30/18/5/2`). The resulting `4` `with_context, CI=0` base items (`12` A/B/C generated outputs) are explicitly reserved as **`2 G1` singletons + `2 G2` two-label pairs**, balanced across `pubmedqa` and `cuad_qa`, alongside `28` `with_context, CI=1` items and `8` `with_context` Clean controls.

---

## 4. Mandatory Pilot Evaluation Metrics (Requirement #11)

Every metric is logged per strategy `(A, B, C)` and stratified by cardinality (`|y| = 1, 2, 3, 4+`), domain, and context presence:

1. **Invalid Sample Rate (%)**:
   $$\text{Invalid Rate} = \frac{\text{JSON Parse Failures} + \text{Safety Refusals} + \text{No-Op Identities} + \text{Length Violations}}{\text{Total Attempts}}$$
   *(Reported both on initial attempt `pass@1` and after max `3` retries.)*
2. **Target-Label Recall (Generator Fidelity Against Blind Humans & Cross-Judge)**:
   $$\text{Target Recall}_k = \frac{|\{i : y^{\text{intended}}_{i,k} = 1 \land y^{\text{gold}}_{i,k} = 1\}|}{|\{i : y^{\text{intended}}_{i,k} = 1\}|}$$
   Plus **Exact Intended-Vector Match Rate** (`y_gold == y_intended`).
3. **Unintended-Label Spillover Rate**:
   $$\text{Spillover Rate}_k = \frac{|\{i : y^{\text{intended}}_{i,k} = 0 \land y^{\text{gold}}_{i,k} = 1\}|}{|\{i : y^{\text{intended}}_{i,k} = 0\}|}$$
   Specifically tracks cross-bleed (`FI -> CI`, `FF -> FI`, `II -> FF/LI`) and Clean control false positives (`|y|=0` items rated `|y|>=1`).
4. **Inter-Annotator Agreement (IAA)**:
   - Per-label and macro **pairwise Cohen's $\kappa$** (between 2 blinded raters on all 260 items) + **Fleiss' $\kappa$** on the shared 3-rater adjudicated subset (minimum 60 stratified items covering all disagreements and `G3–G5` stress cases).
   - Mean `coherence_score_1_to_5`.
5. **Generation Cost & Wall-Clock Time**:
   - Total prompt tokens, completion tokens, retries triggered, total USD cost (via API pricing config table in `configs/generation.yaml`), cost per valid accepted sample, total wall-clock runtime, and mean latency per valid sample (ms).

---

## 5. Strategy Selection & Stress-Tier Decision Gates (Before Full Generation)

A strategy is selected for full 12,000-sample Phase 2 generation using transparent quantitative criteria:
1. **Primary Quality Gate (Minimum Viability)**:
   - Valid output rate after retries **$\ge 95\%$**
   - Target-Label Recall against human gold **$\ge 0.85$** on `G1` & `G2` (and $\ge 0.78$ on `G3`)
   - Macro unintended spillover rate **$\le 0.12$** (`<= 12%`)
   - Clean control false-positive rate **$\le 0.08$**
2. **Winner Selection Rule**: Pick the strategy with the **highest Exact Intended-Vector Match & Target-Label Recall $- \text{Spillover}$ composite score** across `G2`+`G3` (ties broken in favor of lower cost/latency per valid sample — e.g., **Strategy C** vs **Strategy B**; hybrid option allowed if Strategy B wins `G1` and Strategy C wins `G2/G3`, explicitly logged per record!).
3. **G4/G5 Stress Pruning Gate**:
   - If human Fleiss'/Cohen's $\kappa < 0.65$, mean coherence `< 3.5 / 5.0`, or Target-Label Recall `< 0.70` on the `G4`/`G5` pilot subset (`|y| >= 4`), **drop G5 (`60` samples) and/or G4 (`300` samples)** before full generation and reallocate their quota cleanly into `G2` and `G3` rather than injecting noisy labels into training!

---

## 6. Post-Pilot Transition to Full Synthetic + `test_natural` Annotation

1. **Pilot sign-off**: User reviews pilot summary table & audit samples.
2. **Full synthetic batch (`12,000`)**: Run winning strategy + automated filters + blind cross-judge across `train (9,600)`, `val (1,200)`, and `test_synthetic (1,200)`.
3. **Gold subsets & Natural LLM test set (`test_natural`)**:
   - Sample `400` stratified items from `test_synthetic` for blinded multi-rater annotation -> `test_synthetic_gold` (human-verified synthetic benchmark).
   - Independently prompt 2–3 instruction-tuned LLMs using [prompts/v1_0/natural_qa_generation.md](file:///D:/Major%20Project/MultiLabel-Hallucination-Detection/prompts/v1_0/natural_qa_generation.md) on `300`–`450` held-out queries from `source_test_natural_pool.jsonl`, and have 3 blinded human annotators score & adjudicate `adjudicated_gold_labels` -> `test_natural`.

