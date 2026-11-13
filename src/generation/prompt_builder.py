"""Versioned prompt template loader and Strategy A/B/C/Judge prompt renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from src.generation.sampler import split_response_clauses
from src.utils.config import resolve_project_path
from src.utils.io import sha256_text


CATEGORY_OPERATIONAL_RULES: dict[str, str] = {
    "factual_inconsistency": (
        "Replace key factual modifiers/attributes/relations on REAL domain concepts with "
        "demonstrably false real-world facts (Termite-style structural preservation). "
        "Do NOT invent fake citations/studies, and if context is present, mutate external domain facts rather than misquoting local passage details."
    ),
    "factual_fabrication": (
        "Introduce an invented, non-existent external study, clinical trial, court case, "
        "regulatory code/form, institution, or unverifiable statistic/mechanism. Attribute it "
        "to an external fabricated authority rather than claiming it appears in any supplied context."
    ),
    "instruction_inconsistency": (
        "Shift the answer via topical shift so it responds about a related aspect/subtopic of the "
        "same subject entity while failing to answer the user's exact question or constraint, "
        "keeping coherent prose where other intended labels remain naturally manifest."
    ),
    "logical_inconsistency": (
        "Introduce a clear internal self-contradiction, incompatible numerical condition, or "
        "negated premise-to-conclusion inference across clauses inside the response itself."
    ),
    "context_inconsistency": (
        "(Context-present ONLY) Contradict explicit facts/conclusions in the supplied context or "
        "attribute an unsupported passage-specific rule/figure directly to the supplied document."
    ),
}

_SECTION_SYS = "# System Prompt"
_SECTION_USR = "# User Prompt Template"


@dataclass(frozen=True)
class LoadedPromptTemplate:
    """Immutable loaded prompt template with provenance SHA-256."""

    prompt_id: str
    prompt_version: str
    system_prompt: str
    user_template: str
    template_sha256: str


def load_prompt_template(path: str | Path) -> LoadedPromptTemplate:
    """Load and parse a versioned Markdown prompt template from prompts/."""
    resolved = resolve_project_path(path)
    raw = resolved.read_text(encoding="utf-8")
    sha = sha256_text(raw)

    prompt_id = resolved.stem
    prompt_version = "v1.0"
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, flags=re.DOTALL)
    body = raw
    if fm_match:
        fm_text = fm_match.group(1)
        body = raw[fm_match.end() :]
        for line in fm_text.splitlines():
            if line.startswith("prompt_id:"):
                prompt_id = line.split(":", 1)[1].strip()
            elif line.startswith("prompt_version:"):
                prompt_version = line.split(":", 1)[1].strip()

    if _SECTION_SYS not in body or _SECTION_USR not in body:
        raise ValueError(f"Prompt template missing required sections: {resolved}")

    sys_after = body.split(_SECTION_SYS, 1)[1]
    system_part, user_part = sys_after.split(_SECTION_USR, 1)
    return LoadedPromptTemplate(
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        system_prompt=system_part.strip(),
        user_template=user_part.strip(),
        template_sha256=sha,
    )


def format_context_block(context_text: str | None) -> str:
    """Format context safely for prompts."""
    if context_text and context_text.strip():
        return context_text.strip()
    return "None"


def render_strategy_a_step(
    template: LoadedPromptTemplate,
    *,
    domain: str,
    question: str,
    has_context: bool,
    prompt_context: str | None,
    current_response: str,
    target_category: str,
    step_index: int,
    total_steps: int,
    preserved_active_labels: list[str],
    forbidden_labels: list[str],
) -> tuple[str, str]:
    """Render system + user messages for one step of Strategy A (Sequential)."""
    sys_msg = template.system_prompt.format(
        context="supplied context" if has_context else "none",
        question=question,
        current_response=current_response,
    )
    usr_msg = template.user_template.format(
        step_index=step_index,
        total_steps=total_steps,
        domain=domain,
        question=question,
        has_context=str(has_context).lower(),
        context_block=format_context_block(prompt_context),
        current_response=current_response,
        target_category=target_category,
        category_rule=CATEGORY_OPERATIONAL_RULES[target_category],
        preserved_active_labels=", ".join(preserved_active_labels) if preserved_active_labels else "None",
        forbidden_labels=", ".join(forbidden_labels) if forbidden_labels else "None",
    )
    return sys_msg, usr_msg


def render_strategy_b_joint(
    template: LoadedPromptTemplate,
    *,
    domain: str,
    source_dataset: str,
    question: str,
    has_context: bool,
    prompt_context: str | None,
    original_answer: str,
    intended_bits: str,
    active_labels: list[str],
    forbidden_labels: list[str],
) -> tuple[str, str]:
    """Render system + user messages for Strategy B (Joint Compositional Prompting)."""
    sys_msg = template.system_prompt.format(
        context="supplied context" if has_context else "none",
        has_context=str(has_context).lower(),
        question=question,
    )
    active_rules = "\n".join(
        f"  - `{lbl}`: {CATEGORY_OPERATIONAL_RULES[lbl]}" for lbl in active_labels
    )
    usr_msg = template.user_template.format(
        domain=domain,
        source_dataset=source_dataset,
        question=question,
        has_context=str(has_context).lower(),
        context="supplied context" if has_context else "none",
        context_block=format_context_block(prompt_context),
        original_answer=original_answer,
        intended_vector=intended_bits,
        active_labels_with_rules=active_rules or "None",
        inactive_labels=", ".join(forbidden_labels) if forbidden_labels else "None",
    )
    return sys_msg, usr_msg


def build_span_assignment_plan(
    original_answer: str,
    active_labels: list[str],
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """
    Deterministically map active labels onto disjoint answer spans (or insertion slots).

    Returns (spans, span_operations, clean_span_ids).
    """
    spans = split_response_clauses(original_answer)
    if not spans:
        spans = [original_answer.strip()]

    n_spans = len(spans)
    ops: list[dict[str, str]] = []
    modified_span_ids: set[str] = set()

    # Deterministic priority so II modifies opening directive alignment if possible, LI targets last span
    ordered_labels = list(active_labels)
    for idx, lbl in enumerate(ordered_labels):
        if lbl == "logical_inconsistency" and n_spans >= 2:
            target_id = f"S{n_spans}"
            anchor_id = "S1"
            mode = "edit_span"
            note = f"{CATEGORY_OPERATIONAL_RULES[lbl]} Contradict {anchor_id} explicitly inside {target_id}."
        elif idx < n_spans:
            target_id = f"S{idx + 1}"
            mode = "edit_span"
            note = CATEGORY_OPERATIONAL_RULES[lbl]
        else:
            anchor = f"S{n_spans}"
            target_id = f"INSERT_AFTER_{anchor}"
            mode = "insert_clause"
            note = f"Append a new fluent clause after {anchor} realizing: {CATEGORY_OPERATIONAL_RULES[lbl]}"
        if mode == "edit_span":
            modified_span_ids.add(target_id)
        ops.append({"span_id": target_id, "mode": mode, "assigned_label": lbl, "rule": note})

    clean_ids = [f"S{i + 1}" for i in range(n_spans) if f"S{i + 1}" not in modified_span_ids]
    return spans, ops, clean_ids


def render_strategy_c_span_editing(
    template: LoadedPromptTemplate,
    *,
    domain: str,
    source_dataset: str,
    question: str,
    has_context: bool,
    prompt_context: str | None,
    original_answer: str,
    intended_bits: str,
    active_labels: list[str],
    forbidden_labels: list[str],
) -> tuple[str, str, list[dict[str, str]]]:
    """Render system + user messages for Strategy C (Controlled Span-Level Editing)."""
    spans, ops, clean_ids = build_span_assignment_plan(original_answer, active_labels)
    numbered_block = "\n".join(f"- [S{i + 1}] {sp}" for i, sp in enumerate(spans))
    plan_block = "\n".join(
        f"  - `{op['span_id']}` ({op['mode']}) -> `{op['assigned_label']}`: {op['rule']}"
        for op in ops
    )
    sys_msg = template.system_prompt.format(
        question=question,
        context="supplied context" if has_context else "none",
        has_context=str(has_context).lower(),
    )
    usr_msg = template.user_template.format(
        domain=domain,
        source_dataset=source_dataset,
        question=question,
        has_context=str(has_context).lower(),
        context_block=format_context_block(prompt_context),
        numbered_spans_block=numbered_block,
        intended_vector=intended_bits,
        span_plan_block=plan_block,
        clean_span_ids=", ".join(clean_ids) if clean_ids else "None",
        forbidden_labels=", ".join(forbidden_labels) if forbidden_labels else "None",
    )
    return sys_msg, usr_msg, ops


def render_cross_judge(
    template: LoadedPromptTemplate,
    *,
    domain: str,
    question: str,
    has_context: bool,
    prompt_context: str | None,
    reference_answer: str,
    candidate_response: str,
) -> tuple[str, str]:
    """Render strictly blind diagnostic cross-judge prompt (never exposes intended labels)."""
    sys_msg = template.system_prompt.format(
        context="supplied context" if has_context else "none",
        question=question,
        candidate_response=candidate_response,
        has_context=str(has_context).lower(),
    )
    usr_msg = template.user_template.format(
        domain=domain,
        question=question,
        has_context=str(has_context).lower(),
        context_block=format_context_block(prompt_context),
        reference_answer_block=reference_answer.strip(),
        candidate_response=candidate_response.strip(),
    )
    return sys_msg, usr_msg


def render_natural_qa(
    template: LoadedPromptTemplate,
    *,
    domain: str,
    question: str,
    has_context: bool,
    prompt_context: str | None,
) -> tuple[str, str]:
    """Render natural helpful QA prompt (no hallucination directives)."""
    ctx_suffix = " and strictly from the supplied reference context" if has_context else ""
    ctx_section = f"Reference Context:\n{prompt_context.strip()}\n\n" if has_context and prompt_context else ""
    sys_msg = template.system_prompt.format(
        domain=domain,
        context_instruction_suffix=ctx_suffix,
    )
    usr_msg = template.user_template.format(
        context_section=ctx_section,
        question=question,
    )
    return sys_msg, usr_msg
