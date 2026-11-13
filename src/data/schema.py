"""Canonical dataset schemas and validation rules for Phase 1 and Phase 2+."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


HALLUCINATION_LABELS: tuple[str, ...] = (
    "factual_inconsistency",
    "factual_fabrication",
    "instruction_inconsistency",
    "logical_inconsistency",
    "context_inconsistency",
)

LicenseStatus = Literal["verified_open", "upstream_caution", "unresolved"]
DomainName = Literal["medical", "legal", "finance"]
SplitName = Literal["train", "val", "test_synthetic_pool", "test_natural_pool"]
ValidationStatus = Literal[
    "pending",
    "auto_passed",
    "auto_flagged",
    "rejected",
    "human_verified",
    "human_rejected",
]


class LicenseMetadata(BaseModel):
    """Source-specific licensing and attribution metadata carried on every record."""

    hf_card_license: str | None = None
    upstream_license: str
    license_status: LicenseStatus
    redistribution_permitted: bool
    attribution: str
    notes: str

    @field_validator("upstream_license", "attribution", "notes")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("License metadata strings must be non-empty.")
        return value.strip()


class ProvenanceMetadata(BaseModel):
    """Immutable raw source provenance recorded during acquisition and preprocessing."""

    source_url: str
    upstream_url: str
    dataset_version: str
    downloaded_at_utc: str
    raw_filename: str
    raw_file_sha256: str
    preprocessing_config_sha256: str


class SourceQARecord(BaseModel):
    """Canonical Phase 1 preprocessed question-answer record."""

    record_id: str
    source_dataset: str
    source_subset: str
    source_record_id: str
    source_document_id: str | None = None
    source_group_id: str
    domain: DomainName
    has_context: bool
    question: str
    context: str | None = None
    original_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceMetadata
    licensing: LicenseMetadata
    split: SplitName | None = None

    @field_validator("record_id", "source_dataset", "source_subset", "source_record_id", "source_group_id")
    @classmethod
    def _require_id_fields(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Identifier fields cannot be blank.")
        return value.strip()

    @field_validator("question", "original_answer")
    @classmethod
    def _require_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question and original_answer must be non-empty.")
        return cleaned

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "SourceQARecord":
        context_present = self.context is not None and bool(self.context.strip())
        if self.has_context != context_present:
            raise ValueError(
                f"has_context={self.has_context} mismatches context presence ({context_present})."
            )
        if self.source_dataset == "legal_qa_v1":
            if self.licensing.license_status != "unresolved" or self.licensing.redistribution_permitted:
                raise ValueError(
                    "legal_qa_v1 must always carry license_status='unresolved' and "
                    "redistribution_permitted=False."
                )
        return self


class LabelDictionary(BaseModel):
    """Exact five-category binary label representation."""

    factual_inconsistency: Literal[0, 1]
    factual_fabrication: Literal[0, 1]
    instruction_inconsistency: Literal[0, 1]
    logical_inconsistency: Literal[0, 1]
    context_inconsistency: Literal[0, 1]

    def to_vector(self) -> list[int]:
        """Return labels ordered canonically matching HALLUCINATION_LABELS."""
        return [getattr(self, key) for key in HALLUCINATION_LABELS]


class AutoDetectionTier(BaseModel):
    """Automated check and LLM cross-judge diagnostic signals (never gold ground truth)."""

    judge_model: str | None = None
    judge_provider: str | None = None
    judge_model_family: str | None = None
    predicted_labels: list[Literal[0, 1]] | None = None
    rule_checks_passed: bool
    target_missed: bool = False
    spillover_flagged: bool = False
    coherence_score_1_to_5: float | None = Field(default=None, ge=1.0, le=5.0)
    diagnostic_notes: str | None = None

    @field_validator("predicted_labels")
    @classmethod
    def _check_len_five(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(value) != 5:
            raise ValueError("predicted_labels must contain exactly 5 binary values.")
        return value


class MultiLabelRecord(BaseModel):
    """Canonical multi-label schema separating intended, auto, human, and gold tiers."""

    sample_id: str
    source_dataset: str
    source_subset: str | None = None
    source_record_id: str
    source_document_id: str | None = None
    source_group_id: str
    domain: DomainName
    data_origin: Literal["synthetic", "natural"]
    has_context: bool
    question: str
    context: str | None = None
    original_answer: str | None = None
    generated_answer: str
    generation_method: str
    generation_strategy: str | None = None
    generation_model: str | None = None
    prompt_version: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    intended_labels: LabelDictionary | None = None
    auto_detected_labels: AutoDetectionTier | None = None
    human_observed_labels: dict[str, list[Literal[0, 1]]] = Field(default_factory=dict)
    adjudicated_gold_labels: list[Literal[0, 1]] | None = None
    label_vector: list[Literal[0, 1]] | None = None
    validation_status: ValidationStatus = "pending"
    rejection_reasons: list[str] = Field(default_factory=list)
    attempted_count: int = Field(default=1, ge=1)
    annotation_explanation: str | None = None
    licensing: LicenseMetadata
    split: Literal["train", "val", "test_synthetic", "test_natural"] | None = None

    @model_validator(mode="after")
    def _validate_labels_and_tiers(self) -> "MultiLabelRecord":
        if self.generation_strategy is None:
            self.generation_strategy = self.generation_method
        context_present = self.context is not None and bool(self.context.strip())
        if self.has_context != context_present:
            raise ValueError(
                f"has_context={self.has_context} mismatches context presence ({context_present})."
            )
        if self.validation_status != "rejected" and not self.generated_answer.strip():
            raise ValueError("Accepted/pending records must have non-empty generated_answer.")
        if self.data_origin == "natural" and self.intended_labels is not None:
            raise ValueError("Natural LLM responses must have intended_labels=None.")
        if not self.has_context:
            if self.intended_labels is not None and self.intended_labels.context_inconsistency == 1:
                raise ValueError("context_inconsistency cannot be 1 when has_context=False.")
            if self.adjudicated_gold_labels is not None and self.adjudicated_gold_labels[4] == 1:
                raise ValueError("adjudicated_gold_labels CI cannot be 1 when has_context=False.")
            for rater_id, vec in self.human_observed_labels.items():
                if len(vec) == 5 and vec[4] == 1:
                    raise ValueError(f"Human label vector for {rater_id} cannot mark CI=1 without context.")
        for rater_id, vec in self.human_observed_labels.items():
            if len(vec) != 5:
                raise ValueError(f"Human label vector for {rater_id} must have length 5.")
        if self.adjudicated_gold_labels is not None and len(self.adjudicated_gold_labels) != 5:
            raise ValueError("adjudicated_gold_labels must have length 5.")
        if self.validation_status == "human_verified" and self.adjudicated_gold_labels is None:
            raise ValueError("human_verified records require adjudicated_gold_labels.")
        if self.intended_labels is not None and self.label_vector is not None:
            if self.label_vector != self.intended_labels.to_vector() and self.adjudicated_gold_labels is None:
                raise ValueError("label_vector must match intended_labels when gold labels are absent.")
        if self.label_vector is not None and len(self.label_vector) != 5:
            raise ValueError("label_vector must have length 5.")
        if self.source_dataset == "legal_qa_v1":
            if self.licensing.license_status != "unresolved" or self.licensing.redistribution_permitted:
                raise ValueError(
                    "legal_qa_v1 must always carry license_status='unresolved' and "
                    "redistribution_permitted=False."
                )
        return self


class GenerationAttemptRecord(BaseModel):
    """Per-attempt audit record preserving rejected outputs, retries, cost, and latency."""

    attempt_id: str
    sample_id: str
    base_item_id: str
    source_dataset: str
    source_subset: str | None = None
    source_record_id: str
    source_document_id: str | None = None
    source_group_id: str
    domain: DomainName
    split: Literal["train", "val", "test_synthetic", "test_natural"]
    generation_strategy: str
    step_order_name: str | None = None
    step_order_labels: list[str] | None = None
    generation_model: str
    prompt_version: str
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    attempt_number: int = Field(ge=1)
    intended_vector: str
    raw_output: str = ""
    extracted_response: str = ""
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    timestamp_utc: str

