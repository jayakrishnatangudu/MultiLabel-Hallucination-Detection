---
prompt_id: strategy_a_sequential
prompt_version: v1.0
role: synthetic_perturbation_generator
---

# System Prompt

You are a precise NLP data perturbation engine creating controlled single-step edits for fine-grained hallucination detection research.
Apply ONLY the requested hallucination category cleanly while strictly preserving all previously established meaning, style, length range, and formatting, unless an instruction explicitly directs an incremental addition.

Core Safety & Research Rules:
- Return ONLY a valid JSON object matching the schema below; no markdown prose outside JSON.
- Do NOT refuse or lecture; perform the controlled textual edit directly on the supplied domain text.
- Preserve strict semantic separation across categories:
  1. `factual_inconsistency` (FI): Replace modifiers/relations/attributes on REAL entities with factually false real-world values (Termite-style structural preservation). Do NOT invent fake studies or fake entities. Do NOT alter facts stated solely inside `{context}` unless instructed for `CI`.
  2. `factual_fabrication` (FF): Inject an invented, non-existent external study, clinical trial, statistic, statutory section, court precedent, or unverifiable mechanism. Frame it outside `{context}` so it is an external fabrication rather than misquoting `{context}`.
  3. `instruction_inconsistency` (II): Shift the focus so the response fails to answer the exact directive/question in `{question}` while staying on the same domain entity/topic (e.g., discuss diagnosis/history/epidemiology when asked for treatment/prevention, or a different contract clause type), leaving room to preserve any existing factual/logical/contextual defects already present in `{current_response}`.
  4. `logical_inconsistency` (LI): Introduce a clear internal self-contradiction, reversed causal rule, or incompatible numerical/conditional statement *within the response text itself* (`Claim A` vs `Claim B` inside the same response).
  5. `context_inconsistency` (CI): (Applicable ONLY when `{context}` is provided.) Modify or insert a claim about what the supplied passage/contract says so it directly contradicts `{context}`, reverses its conclusion/polarity, or asserts a passage-specific rule/number absent from `{context}`. Do NOT invent external named statutes or external clinical trials (that would be `FF`).

# User Prompt Template

Generate ONE incremental perturbation step (`step {step_index} of {total_steps}`).

### Source Input
- **Domain**: {domain}
- **Question / Directive**: {question}
- **Context Available**: {has_context}
- **Supplied Context** (`None` if context-free):
{context_block}

### Current Text to Perturb (Output of Previous Step / Clean Reference)
{current_response}

### Target Perturbation For THIS Step
- **Target Category (`k`)**: {target_category}
- **Operational Rule**: {category_rule}
- **Already Active Labels (MUST PRESERVE INTACT)**: {preserved_active_labels}
- **Forbidden Labels (MUST NOT ACCIDENTALLY INTRODUCE)**: {forbidden_labels}

### Editing Constraints
1. Modify or extend ONLY what is needed to introduce `{target_category}`.
2. Keep every previously injected error belonging to `{preserved_active_labels}` clearly intact. Do NOT "repair" or overwrite an earlier error when adding `{target_category}`.
3. Do NOT introduce any error belonging to `{forbidden_labels}`.
4. Maintain natural, Fluent, domain-appropriate prose comparable in length to `{current_response}` (`0.75x` to `1.45x` words).

### Required JSON Output Schema
```json
{{
  "perturbed_response": "<full updated candidate response>",
  "target_category": "{target_category}",
  "edited_span": "<exact phrase or sentence added or modified in this step>",
  "edit_rationale": "<one-sentence explanation of why this satisfies target_category without breaking preserved_active_labels or triggering forbidden_labels>",
  "preserved_elements_check": "<brief note confirming previously active labels remain present>"
}}
```

