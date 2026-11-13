# Versioned Prompt Registry

All synthetic generation, blind auditing, natural QA evaluation, and zero-shot baseline prompts live under versioned subdirectories (starting with `v1_0/`).

- Prompt edits **must** increment the directory version (`v1_0/` -> `v1_1/`) rather than silently overwriting prompts after pilot or full-scale runs begin.
- Every `MultiLabelRecord` stores `prompt_version` (e.g., `"v1.0"`) alongside the SHA-256 hash of the exact rendered template tracked in `configs/generation.yaml`.

