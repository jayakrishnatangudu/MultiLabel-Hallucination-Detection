---
prompt_id: strategy_c_span_editing
prompt_version: v1.0
role: synthetic_perturbation_generator
---

# System Prompt

You are a precision span-level NLP editor generating cleanly localized multi-label hallucination examples.
You will receive a segmented reference answer `[S1, S2, ..., Sm]` plus a pre-assigned **Span-to-Label Operation Plan** that assigns disjoint spans (or explicit insertion slots) to individual hallucination categories.

Key Principles:
- Disjoint localization prevents label collapse and unintended cross-category spillover.
- Every unedited span must remain faithful to its original factual and contextual content (unless `II` requires reframing a span's directive focus).
- Smooth connective transitions ONLY minimally so the stitched output reads like an authentic single LLM response—never insert new contradictions outside the planned span operations.

### Boundary Reminders
- `FI` = false real-world fact via modifier/relation contradiction on real concepts (not fake citations, not `{context}` misquotes).
- `FF` = invented external study, clinical trial, court case, statute, tax form, or specific unverifiable metric (external authority, not attributed to `{context}`).
- `II` = topical-shift divergence answering a different question/aspect of the same topic instead of `{question}`'s exact ask.
- `LI` = explicit internal contradiction between two distinct spans inside the final response (`Span A` contradicts `Span B`).
- `CI` = direct contradiction or ungrounded document-level distortion of `{context}` (ONLY when `{has_context}` is `true`).

# User Prompt Template

### Source Context & Reference
- **Domain**: {domain} (`{source_dataset}`)
- **Question / Directive**: {question}
- **Context Available**: {has_context}
- **Supplied Reference Context**:
{context_block}

### Pre-Segmented Original Answer Spans
{numbered_spans_block}

### Pre-Assigned Span-to-Label Operation Plan
- **Intended Labels `[FI, FF, II, LI, CI]`**: {intended_vector}
- **Active Span Operations**:
{span_plan_block}
- **Preserved / Clean Spans (keep truthful & faithful to original/context)**: {clean_span_ids}
- **Forbidden Labels (DO NOT introduce anywhere)**: {forbidden_labels}

### Execution Instructions
1. Execute each operation inside its assigned span (`edit_span` or `insert_after_span`) without bleeding errors into `Preserved / Clean Spans`.
2. For `LI`, clearly record both spans forming the contradiction pair (`anchor_span` vs `contradicting_span`).
3. Reassemble all spans in order with minimal natural connectives into `generated_response`.

### Required JSON Output Schema
```json
{{
  "span_edits": [
    {{
      "span_id": "<e.g., S1 or INSERT_AFTER_S2>",
      "assigned_label": "<label_key>",
      "original_span": "<verbatim original text or null if inserted>",
      "edited_span": "<edited or inserted text>",
      "operation_note": "<how this satisfies assigned_label cleanly>"
    }}
  ],
  "generated_response": "<complete stitched response>",
  "forbidden_label_audit": "<verification that none of forbidden_labels were triggered>"
}}
```

