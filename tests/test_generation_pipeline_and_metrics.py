"""End-to-end mock tests for backend verification, retries, rejections, blinding, IAA, and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src.data.schema import HALLUCINATION_LABELS
from src.generation.backend import (
    BackendHTTPRequestError,
    BackendNotConfiguredError,
    CONTROLLED_SMOKE_AUDIT_FIXTURE,
    GoogleGeminiClient,
    OpenAICompatibleClient,
    MockLLMClient,
    _redact_secrets,
    build_live_client,
    load_local_dotenv,
    probe_live_backends,
    verify_generation_and_judge_backends,
)
from src.generation.pipeline import (
    BLINDED_ANNOTATOR_FIELDS,
    apply_human_annotations,
    export_full_generation_plans,
    export_natural_generation_plan,
    run_full_generation,
    run_natural_generation,
    run_pilot_e0,
)
from src.generation.pilot_metrics import compute_pilot_e0_report
from src.generation.sampler import sample_full_generation_spec
from src.utils.config import load_yaml_config
from src.utils.io import read_jsonl
from tests.test_generation_sampler_and_prompts import _make_source_records_pool


def _make_verified_independent_cfg(cfg: dict) -> dict:
    return {
        **cfg,
        "backends": {
            **cfg["backends"],
            "synthetic_generator": {
                **cfg["backends"]["synthetic_generator"],
                "provider": "google_gemini",
                "model_family": "gemini",
                "model_id": "gemini-test-model-a",
                "model_availability_verified": True,
                "pricing_verified": True,
                "pricing_usd_per_1m_tokens": {"input": 0.10, "output": 0.40},
            },
            "cross_judge_auditor": {
                **cfg["backends"]["cross_judge_auditor"],
                "provider": "openai_compatible",
                "model_family": "qwen",
                "model_id": "qwen-test-judge-b",
                "default_api_base": "https://example.invalid/v1",
                "model_availability_verified": True,
                "pricing_verified": True,
                "pricing_usd_per_1m_tokens": {"input": 0.50, "output": 0.50},
            },
        },
    }


def test_backend_preflight_requires_gemini_verification_and_cross_family_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_yaml_config("configs/generation.yaml")
    for var in (
        "GEMINI_API_BASE",
        "GEMINI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "JUDGE_API_BASE",
        "JUDGE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    # 1. Default config blocks live launch until live GET /models verification sets model_availability_verified=true
    with pytest.raises(BackendNotConfiguredError, match="model_availability_verified=false"):
        verify_generation_and_judge_backends(cfg, check_live=False, use_dotenv=False)

    verified_cfg = _make_verified_independent_cfg(cfg)

    # 2. Missing credentials still block verified config
    with pytest.raises(BackendNotConfiguredError, match="credentials/endpoint not configured"):
        verify_generation_and_judge_backends(verified_cfg, check_live=False, use_dotenv=False)

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-judge-key")
    monkeypatch.setenv("JUDGE_API_BASE", "https://example.invalid/v1")

    # 3. Same model_family (or shared google_deepmind lineage gemini + gemma) must be rejected
    same_lineage_cfg = {
        **verified_cfg,
        "backends": {
            **verified_cfg["backends"],
            "cross_judge_auditor": {
                **verified_cfg["backends"]["cross_judge_auditor"],
                "model_family": "gemma",
                "model_id": "gemma-3-27b-it",
            },
        },
    }
    with pytest.raises(BackendNotConfiguredError, match="lineage independence violated"):
        verify_generation_and_judge_backends(same_lineage_cfg, check_live=False)

    # 4. Same upstream provider must be rejected even if model_id/family differ
    same_provider_cfg = {
        **verified_cfg,
        "backends": {
            **verified_cfg["backends"],
            "cross_judge_auditor": {
                **verified_cfg["backends"]["cross_judge_auditor"],
                "provider": "google_gemini",
                "model_family": "qwen",
                "model_id": "qwen-test-judge-b",
            },
        },
    }
    with pytest.raises(BackendNotConfiguredError, match="provider independence violated"):
        verify_generation_and_judge_backends(same_provider_cfg, check_live=False)

    # 5. Family aliasing (putting qwen model_family on a gemini model_id) must be caught
    alias_cfg = {
        **verified_cfg,
        "backends": {
            **verified_cfg["backends"],
            "cross_judge_auditor": {
                **verified_cfg["backends"]["cross_judge_auditor"],
                "model_family": "qwen",
                "model_id": "gemini-2.0-flash-alias",
            },
        },
    }
    with pytest.raises(BackendNotConfiguredError, match="conflicts with model_id"):
        verify_generation_and_judge_backends(alias_cfg, check_live=False)

    # 6. Fully distinct provider + model family + lineage + verified pricing passes check_live=False
    res = verify_generation_and_judge_backends(verified_cfg, check_live=False)
    assert res["verified"] is True
    assert res["cross_judge_independence"]["distinct_model_family"] is True
    assert res["cross_judge_independence"]["distinct_provider"] is True
    assert res["cross_judge_independence"]["diagnostic_only_never_gold"] is True


def test_google_gemini_client_payload_and_response_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [{"content": {"parts": [{"text": '{"generated_response": "ok"}'}]}}],
                "usageMetadata": {
                    "promptTokenCount": 120,
                    "candidatesTokenCount": 35,
                    "thoughtsTokenCount": 25,
                },
            }

    def _fake_post(url: str, headers: dict, json: dict, timeout: float) -> _FakeResp:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResp()

    monkeypatch.setattr("src.generation.backend.requests.post", _fake_post)
    client = GoogleGeminiClient(
        model_id="gemini-test-model",
        api_key="secret-gemini-key",
        pricing_usd_per_1m_tokens={"input": 0.10, "output": 0.40},
    )
    out = client.generate(
        system_prompt="System instruction",
        user_prompt="User prompt",
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
        seed=42,
    )
    assert captured["url"].endswith("/models/gemini-test-model:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "secret-gemini-key"
    assert captured["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert out.provider == "google_gemini"
    assert out.model_family == "gemini"
    assert out.input_tokens == 120 and out.output_tokens == 60
    assert out.cost_usd > 0.0


def test_pilot_e0_pipeline_retries_rejections_blinding_and_metrics(tmp_path: Path) -> None:
    cfg = load_yaml_config("configs/generation.yaml")
    train_pool = _make_source_records_pool("train")

    state = {"seen_first_b": False, "first_b_retried": False}

    def gen_responder(sys_p: str, usr_p: str, call_n: int, temp: float) -> str:
        # 1. Force one malformed JSON attempt followed by retry success on first Strategy B call
        if "Target Label Specification" in usr_p and not state["seen_first_b"]:
            state["seen_first_b"] = True
            return "NOT VALID JSON {"
        # 2. Force permanent safety refusal across all retries on Strategy C for criterion number 12
        if "Pre-Assigned Span-to-Label Operation Plan" in usr_p and "criteria number 12 " in usr_p:
            return json.dumps(
                {"generated_response": "I'm sorry, I cannot fabricate false domain claims for that query."}
            )
        # Normal clean JSON response with nontrivial edit
        base_resp = (
            "Altered Sentence one reverses the verified biological or legal mechanism inaccurately. "
            "Invented Study Alpha-2026 reported fabricated outcomes across external cohorts. "
            "Sentence three remains consistent unless contradicted by an incompatible final conclusion denying sentence one."
        )
        if "Generate ONE incremental perturbation step" in usr_p:
            return json.dumps(
                {
                    "perturbed_response": f"{base_resp} [step_call_{call_n}_temp_{temp}]",
                    "target_category": "factual_inconsistency",
                    "edited_span": "Altered Sentence one",
                    "edit_rationale": "Mock incremental edit.",
                    "preserved_elements_check": "ok",
                }
            )
        return json.dumps(
            {
                "generated_response": f"{base_resp} [call_{call_n}]",
                "per_label_modifications": {"mock": "edited"},
                "negative_constraints_verified": "checked",
            }
        )

    def judge_responder(sys_p: str, usr_p: str, call_n: int, temp: float) -> str:
        is_ctx_true = "- **Context Available**: true" in usr_p
        # Notice Clean controls produce unmodified reference sentences containing 'Sentence one explains'
        is_clean = "Sentence one explains core mechanism" in usr_p
        if is_clean:
            preds = {
                "factual_inconsistency": 0,
                "factual_fabrication": 0,
                "instruction_inconsistency": 0,
                "logical_inconsistency": 0,
                "context_inconsistency": 0,
            }
        else:
            preds = {
                "factual_inconsistency": 1,
                "factual_fabrication": 1,
                "instruction_inconsistency": 0,
                "logical_inconsistency": 1,
                "context_inconsistency": 1 if is_ctx_true else 0,
            }
        return json.dumps(
            {
                "labels": preds,
                "evidence": {},
                "coherence_score_1_to_5": 4,
                "diagnostic_notes": "Mock blind audit.",
            }
        )

    gen_client = MockLLMClient(
        model_id="mock-gemini-generator",
        provider="mock_google_gemini",
        model_family="gemini",
        responder=gen_responder,
    )
    judge_client = MockLLMClient(
        model_id="mock-independent-judge",
        provider="mock_non_gemini_provider",
        model_family="qwen",
        responder=judge_responder,
    )

    out_dir = tmp_path / "pilot_out"
    summary = run_pilot_e0(
        "configs/generation.yaml",
        generator_client=gen_client,
        judge_client=judge_client,
        output_dir=out_dir,
        source_records_override=train_pool,
    )

    # Check output files written and strict blinding in annotator packet
    assert (out_dir / "pilot_records.jsonl").exists()
    assert (out_dir / "rejected_samples.jsonl").exists()
    assert (out_dir / "generation_attempts.jsonl").exists()
    assert (out_dir / "blinded_annotation_packet.jsonl").exists()
    assert (out_dir / "blind_key_mapping.jsonl").exists()

    packet = read_jsonl(out_dir / "blinded_annotation_packet.jsonl")
    key_map = read_jsonl(out_dir / "blind_key_mapping.jsonl")
    rejected_rows = read_jsonl(out_dir / "rejected_samples.jsonl")
    attempts_rows = read_jsonl(out_dir / "generation_attempts.jsonl")

    assert len(packet) == len(key_map)
    for item in packet:
        assert set(item.keys()) == BLINDED_ANNOTATOR_FIELDS
        assert "intended_labels" not in item
        assert "generation_strategy" not in item
        assert "auto_detected_labels" not in item

    # Confirm exact rejected sample & retry attempt preservation
    assert len(rejected_rows) >= 1
    assert any("refusal_detected" in r for row in rejected_rows for r in row["rejection_reasons"])
    assert any(not a["accepted"] for a in attempts_rows)

    # Check paired alignment (80 slots per strategy attempted) & order-sensitivity 18 items
    p_metrics = summary["pilot_metrics"]
    assert p_metrics["overall_counts"]["clean_control_count"] == 20
    assert p_metrics["strategy_a_order_sensitivity"]["subset_size"] == 18
    for strat in ("strategy_a", "strategy_b", "strategy_c"):
        assert p_metrics["paired_strategy_comparison"][strat]["attempted_slots"] == 80
        assert p_metrics["paired_strategy_comparison"][strat]["total_cost_usd"] > 0.0
        assert p_metrics["paired_strategy_comparison"][strat]["total_generation_latency_ms"] > 0.0

    # Cross-judge must NEVER populate human_observed_labels or adjudicated_gold_labels
    from src.data.schema import GenerationAttemptRecord, MultiLabelRecord

    accepted_models = [MultiLabelRecord(**d) for d in read_jsonl(out_dir / "pilot_records.jsonl")]
    assert all(r.adjudicated_gold_labels is None for r in accepted_models)
    assert all(r.human_observed_labels == {} for r in accepted_models)
    assert all(
        r.auto_detected_labels is not None
        and r.auto_detected_labels.judge_provider == "mock_non_gemini_provider"
        and r.auto_detected_labels.judge_model_family == "qwen"
        for r in accepted_models
    )
    perturbed_acc = [r for r in accepted_models if r.generation_strategy != "clean_control"]
    assert all(
        r.generation_parameters.get("generation_provider") == "mock_google_gemini"
        and r.generation_parameters.get("generation_model_family") == "gemini"
        for r in perturbed_acc
    )

    # Simulate independent 2-rater human annotations + adjudication disagreeing intentionally with judge on one item
    rater_inputs = []
    adj_decisions = {}
    for row in key_map:
        tgt = row["intended_labels"] or [0, 0, 0, 0, 0]
        rater_inputs.append({"annotation_item_id": row["annotation_item_id"], "rater_id": "rater_1", "labels": tgt})
        # Make rater_2 disagree on first non-clean sample to verify adjudicator path
        r2_vec = list(tgt)
        if row["sample_id"].startswith("PILOT_BASE_000__strategy_a"):
            r2_vec[0] = 1 - r2_vec[0]
            adj_decisions[row["sample_id"]] = tgt
        rater_inputs.append({"annotation_item_id": row["annotation_item_id"], "rater_id": "rater_2", "labels": r2_vec})

    verified = apply_human_annotations(
        accepted_models,
        blind_key_rows=key_map,
        rater_annotations=rater_inputs,
        adjudicator_decisions=adj_decisions,
    )
    assert all(r.validation_status == "human_verified" for r in verified)

    rejected_models = [MultiLabelRecord(**d) for d in rejected_rows]
    attempt_models = [GenerationAttemptRecord(**d) for d in attempts_rows]
    post_human_report = compute_pilot_e0_report(
        accepted_records=verified,
        rejected_records=rejected_models,
        attempt_records=attempt_models,
    )
    iaa = post_human_report["inter_annotator_agreement"]
    assert iaa["annotated_multi_rater_items"] == len(verified)
    assert iaa["macro_cohens_kappa"] > 0.90
    assert iaa["macro_fleiss_kappa"] > 0.90
    assert post_human_report["paired_strategy_comparison"]["strategy_b"]["human_gold_fidelity"]["macro_target_recall"] == 1.0


def test_run_pilot_e0_stops_safely_before_exceeding_budget(tmp_path: Path) -> None:
    """Verify strict max_pilot_budget_usd enforcement halts before exceeding cap and flushes partial outputs."""
    import yaml

    base_cfg = load_yaml_config("configs/generation.yaml")
    tight_cap = 0.04
    base_cfg["pilot_e0"]["max_pilot_budget_usd"] = tight_cap
    base_cfg["pilot_e0"]["enforce_strict_budget_stop"] = True

    cfg_path = tmp_path / "tight_budget_gen.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    train_pool = _make_source_records_pool("train")

    def ok_gen(sys_p: str, usr_p: str, idx: int, temp: float) -> str:
        base_resp = (
            "Altered Sentence one reverses the verified biological or legal mechanism inaccurately. "
            "Invented Study Alpha-2026 reported fabricated outcomes across external cohorts. "
            "Sentence three remains consistent unless contradicted by an incompatible final conclusion."
        )
        if "Generate ONE incremental perturbation step" in usr_p:
            return json.dumps({"perturbed_response": f"{base_resp} [step_{idx}]"})
        return json.dumps({"generated_response": f"{base_resp} [gen_{idx}]"})

    def ok_judge(sys_p: str, usr_p: str, idx: int, temp: float) -> str:
        return json.dumps(
            {
                "labels": {"FI": 0, "FF": 0, "CI": 0, "CM": 0, "LC": 0},
                "coherence_score_1_to_5": 5,
                "evidence_notes": "mock",
            }
        )

    gen_client = MockLLMClient(
        model_id="gemini-mock-budget",
        provider="mock_google_gemini",
        model_family="gemini",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
        responder=ok_gen,
    )
    judge_client = MockLLMClient(
        model_id="gpt-mock-judge",
        provider="mock_openai",
        model_family="gpt",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        responder=ok_judge,
    )

    out_dir = tmp_path / "budget_pilot_out"
    summary = run_pilot_e0(
        cfg_path,
        generator_client=gen_client,
        judge_client=judge_client,
        output_dir=out_dir,
        source_records_override=train_pool,
    )

    guard = summary["budget_guard"]
    assert guard["max_pilot_budget_usd"] == tight_cap
    assert guard["enforce_strict_budget_stop"] is True
    assert guard["halted_safely"] is True
    assert guard["blocked_calls_prevented"] >= 1
    assert 0.0 < guard["cumulative_cost_usd"] <= tight_cap
    assert (out_dir / "pilot_e0_summary.json").exists()
    assert (out_dir / "pilot_records.jsonl").exists()
    completed = len(read_jsonl(out_dir / "pilot_records.jsonl")) + len(read_jsonl(out_dir / "rejected_samples.jsonl"))
    assert 1 <= completed < 296


def test_gemini_38_payload_omits_sampling_params_and_sets_thinking_level() -> None:
    """Gemini 3.8 Flash must never emit deprecated temperature/topP/topK params and forbids minimal thinking."""
    with pytest.raises(BackendNotConfiguredError, match="thinking_level='MINIMAL'"):
        GoogleGeminiClient(
            model_id="gemini-3.8-flash",
            api_key="dummy",
            pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
            thinking_level="minimal",
        )

    client = GoogleGeminiClient(
        model_id="gemini-3.8-flash",
        api_key="dummy",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
        thinking_level="low",
    )
    payload = client.build_request_payload(
        system_prompt="Sys",
        user_prompt="Usr",
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        seed=42,
    )
    gen_cfg = payload["generationConfig"]
    assert "temperature" not in gen_cfg
    assert "topP" not in gen_cfg
    assert "topK" not in gen_cfg
    assert gen_cfg["maxOutputTokens"] == 2048
    assert gen_cfg["responseMimeType"] == "application/json"
    assert gen_cfg["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_openai_gpt54_mini_payload_and_reasoning_effort() -> None:
    """OpenAI gpt-5.4-mini sends max_completion_tokens, reasoning_effort='none', and json_object mode."""
    client = OpenAICompatibleClient(
        model_id="gpt-5.4-mini",
        api_base="https://api.openai.com/v1",
        api_key="dummy",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        reasoning_effort="none",
        response_format_json_object=True,
    )
    body = client.build_request_payload(
        system_prompt="Judge sys",
        user_prompt="Judge usr",
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        seed=42,
    )
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "none"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0

    reasoning_client = OpenAICompatibleClient(
        model_id="gpt-5.4-mini",
        api_base="https://api.openai.com/v1",
        api_key="dummy",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        reasoning_effort="low",
    )
    body_r = reasoning_client.build_request_payload(
        system_prompt="Sys",
        user_prompt="Usr",
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
    )
    assert "temperature" not in body_r and "top_p" not in body_r


def test_probe_live_backends_and_dotenv_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify .env loading and end-to-end smoke verification without real keys or live traffic."""
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        'GEMINI_API_KEY="dotenv-gem-key"\nOPENAI_API_KEY=dotenv-oai-key\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already-set-env-wins")
    loaded = load_local_dotenv(dotenv_file)
    assert loaded == ["GEMINI_API_KEY"]
    import os

    assert os.getenv("GEMINI_API_KEY") == "dotenv-gem-key"
    assert os.getenv("OPENAI_API_KEY") == "already-set-env-wins"

    class _MockHTTPResp:
        def __init__(self, payload: dict, headers: dict | None = None) -> None:
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    def _fake_get(url: str, **kwargs) -> _MockHTTPResp:
        if "generativelanguage.googleapis.com" in url:
            return _MockHTTPResp(
                {
                    "models": [
                        {
                            "name": "models/gemini-3.8-flash",
                            "inputTokenLimit": 1048576,
                            "outputTokenLimit": 65536,
                        }
                    ]
                }
            )
        return _MockHTTPResp(
            {"data": [{"id": "gpt-5.4-mini", "owned_by": "system"}]},
            headers={"x-ratelimit-limit-requests": "5000", "x-ratelimit-limit-tokens": "2000000"},
        )

    def _fake_post(url: str, **kwargs) -> _MockHTTPResp:
        req_json = kwargs.get("json") or {}
        if "generativelanguage.googleapis.com" in url:
            assert "temperature" not in req_json["generationConfig"]
            assert "topP" not in req_json["generationConfig"]
            return _MockHTTPResp(
                {
                    "modelVersion": "gemini-3.8-flash",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "generated_response": (
                                                    "Oral aspirin tablet taken once with water is the first-line "
                                                    "emergency treatment for acute anaphylaxis because selective COX-2 inhibition "
                                                    "immediately reverses airway edema and distributive shock."
                                                ),
                                                "per_label_modifications": {
                                                    "factual_inconsistency": "Replaced intramuscular epinephrine with oral aspirin and inverted pharmacology."
                                                },
                                                "negative_constraints_verified": "No fabricated studies, internal contradictions, or topical shifts introduced.",
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 90,
                        "candidatesTokenCount": 30,
                        "thoughtsTokenCount": 18,
                    },
                }
            )
        assert req_json["reasoning_effort"] == "none"
        assert req_json["max_completion_tokens"] == 512
        return _MockHTTPResp(
            {
                "model": "gpt-5.4-mini-2026-03-17",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "labels": {
                                        "factual_inconsistency": 1,
                                        "factual_fabrication": 0,
                                        "instruction_inconsistency": 0,
                                        "logical_inconsistency": 0,
                                        "context_inconsistency": 0,
                                    },
                                    "evidence": {
                                        "factual_inconsistency": "Claims oral aspirin is first-line emergency anaphylaxis therapy instead of IM epinephrine.",
                                        "factual_fabrication": None,
                                        "instruction_inconsistency": None,
                                        "logical_inconsistency": None,
                                        "context_inconsistency": None,
                                    },
                                    "coherence_score_1_to_5": 5,
                                    "diagnostic_notes": "Clean single-label FI contradiction without spillover.",
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 42,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
            headers={"x-ratelimit-remaining-requests": "4999", "x-ratelimit-limit-tokens": "2000000"},
        )

    monkeypatch.setattr("src.generation.backend.requests.get", _fake_get)
    monkeypatch.setattr("src.generation.backend.requests.post", _fake_post)

    cfg = load_yaml_config("configs/generation.yaml")
    out_report = tmp_path / "smoke.json"
    rep = probe_live_backends(cfg, use_dotenv=False, write_report_path=out_report)
    assert out_report.exists()
    raw_report_text = out_report.read_text(encoding="utf-8")
    assert "dotenv-gem-key" not in raw_report_text
    assert "already-set-env-wins" not in raw_report_text
    assert rep["api_keys_redacted"] is True
    assert rep["synthetic_generator"]["configured_model_id"] == "gemini-3.8-flash"
    assert rep["synthetic_generator"]["thoughts_tokens"] == 18
    assert rep["synthetic_generator"]["output_tokens_billed"] == 48
    assert rep["synthetic_generator"]["stage1_validation"]["passed"] is True
    assert "temperature" not in rep["synthetic_generator"]["sample_request_payload_keys"]
    assert rep["cross_judge_auditor"]["response_model_id"] == "gpt-5.4-mini-2026-03-17"
    assert rep["cross_judge_auditor"]["reasoning_tokens"] == 0
    assert rep["cross_judge_auditor"]["response_rate_limit_headers"]["x-ratelimit-limit-tokens"] == "2000000"
    assert rep["canonical_schema_validation"] == {
        "source_qa_record_validated": True,
        "strategy_b_json_schema_validated": True,
        "generation_attempt_record_validated": True,
        "cross_judge_json_schema_validated": True,
        "auto_detection_tier_validated": True,
        "multilabel_record_validated": True,
        "multilabel_validation_status": "auto_passed",
        "predicted_labels_vector": [1, 0, 0, 0, 0],
    }
    assert rep["budget_guard_verification"]["max_pilot_budget_usd"] == 10.0
    assert rep["budget_guard_verification"]["enforce_strict_budget_stop"] is True
    assert rep["budget_guard_verification"]["boundary_over_cap_halted_safely"] is True
    assert rep["budget_guard_verification"]["boundary_blocked_calls_prevented"] == 1
    assert rep["total_smoke_cost_usd"] > 0.0


class _FakeStatusHTTPResp:
    """Minimal HTTP response stub supporting status codes, headers, JSON/text errors."""

    def __init__(
        self,
        *,
        status_code: int,
        payload: dict | None = None,
        text: str = "",
        headers: dict | None = None,
        reason: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}
        self.reason = reason or f"HTTP {status_code}"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests as _req

            raise _req.exceptions.HTTPError(f"{self.status_code} {self.reason}", response=self)

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


def test_http_retry_succeeds_after_transient_503_and_respects_retry_after(monkeypatch) -> None:
    """Transient 503 and 429 responses retry with bounded backoff/Retry-After and succeed."""
    sleep_delays: list[float] = []
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda sec: sleep_delays.append(sec))

    gem_calls = 0

    def _fake_gem_post(url, headers, json, timeout):
        nonlocal gem_calls
        gem_calls += 1
        if gem_calls == 1:
            return _FakeStatusHTTPResp(
                status_code=503,
                payload={"error": {"code": 503, "status": "UNAVAILABLE", "message": "Service overloaded"}},
            )
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "modelVersion": "gemini-3.8-flash",
                "candidates": [{"content": {"parts": [{"text": '{"generated_response": "Recovered."}'}]}}],
                "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 12, "thoughtsTokenCount": 5},
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _fake_gem_post)
    gem = GoogleGeminiClient(
        model_id="gemini-3.8-flash",
        api_key="AIzaDummyRetryTestKey0001",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
        max_retries=2,
        retry_base_delay_sec=1.0,
        retry_max_delay_sec=4.0,
    )
    resp = gem.generate(
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )
    assert gem_calls == 2
    assert sleep_delays == [1.0]
    assert "Recovered." in resp.raw_text
    assert resp.latency_ms >= 0.0

    # Verify Retry-After is respected and capped at retry_max_delay_sec on HTTP 429
    sleep_delays.clear()
    oai_calls = 0

    def _fake_oai_post(url, headers, json, timeout):
        nonlocal oai_calls
        oai_calls += 1
        if oai_calls == 1:
            return _FakeStatusHTTPResp(
                status_code=429,
                payload={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                headers={"Retry-After": "12", "x-ratelimit-remaining-requests": "0"},
            )
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "model": "gpt-5.4-mini-2026-03-17",
                "choices": [{"message": {"content": '{"labels": {"factual_inconsistency": 1}}'}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 20},
            },
            headers={"x-ratelimit-remaining-requests": "99"},
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _fake_oai_post)
    oai = OpenAICompatibleClient(
        model_id="gpt-5.4-mini",
        api_base="https://api.openai.com/v1",
        api_key="sk-DummyRetryOpenAI0001",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        max_retries=2,
        retry_base_delay_sec=1.0,
        retry_max_delay_sec=5.0,
    )
    oai_resp = oai.generate(
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )
    assert oai_calls == 2
    assert sleep_delays == [5.0]
    assert oai_resp.rate_limit_headers.get("x-ratelimit-remaining-requests") == "99"

    # Also verify GET /models retries transient 502 and recovers
    sleep_delays.clear()
    get_calls = 0

    def _fake_get(url, headers, timeout, params=None):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            return _FakeStatusHTTPResp(status_code=502, text="Bad Gateway")
        return _FakeStatusHTTPResp(status_code=200, payload={"data": [{"id": "gpt-5.4-mini"}]})

    monkeypatch.setattr("src.generation.backend.requests.get", _fake_get)
    avail = oai.verify_available_detailed()
    assert get_calls == 2
    assert sleep_delays == [1.0]
    assert avail["attempts"] == 2
    assert avail["available_ids"] == ["gpt-5.4-mini"]
    assert avail["matched_model"] == {"id": "gpt-5.4-mini"}


def test_http_retry_exhaustion_raises_sanitized_backend_error(monkeypatch) -> None:
    """Repeated 503 errors exhaust max_retries with exponential backoff and raise BackendHTTPRequestError."""
    sleep_delays: list[float] = []
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda sec: sleep_delays.append(sec))

    attempts = 0

    def _always_503(url, headers, json, timeout):
        nonlocal attempts
        attempts += 1
        return _FakeStatusHTTPResp(
            status_code=503,
            payload={"error": {"code": 503, "status": "UNAVAILABLE", "message": "Backend unavailable"}},
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _always_503)
    gem = GoogleGeminiClient(
        model_id="gemini-3.8-flash",
        api_key="AIzaDummyExhaustKey0001",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
        max_retries=2,
        retry_base_delay_sec=1.0,
        retry_max_delay_sec=8.0,
    )
    with pytest.raises(BackendHTTPRequestError) as exc_info:
        gem.generate(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
        )

    exc = exc_info.value
    assert attempts == 3
    assert sleep_delays == [1.0, 2.0]
    assert exc.status_code == 503
    assert exc.attempts == 3
    assert exc.retryable is True
    assert exc.provider == "google_gemini"
    assert exc.model_id == "gemini-3.8-flash"
    assert exc.to_dict()["status_code"] == 503
    assert exc.to_dict()["retryable"] is True


def test_no_retry_on_permanent_4xx_and_redacts_credentials(monkeypatch) -> None:
    """Permanent 400/401 errors must not retry and must scrub API keys, Bearer tokens, and ?key= params."""
    sleep_delays: list[float] = []
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda sec: sleep_delays.append(sec))

    oai_secret = "sk-test-secret-openai-99887766"
    gem_secret = "AIzaSyTestSecretGemini55443322"

    oai_attempts = 0

    def _always_401_oai(url, headers, json, timeout):
        nonlocal oai_attempts
        oai_attempts += 1
        return _FakeStatusHTTPResp(
            status_code=401,
            payload={
                "error": {
                    "type": "authentication_error",
                    "message": f"Invalid Authorization: Bearer {oai_secret} in header",
                }
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _always_401_oai)
    oai = OpenAICompatibleClient(
        model_id="gpt-5.4-mini",
        api_base="https://api.openai.com/v1",
        api_key=oai_secret,
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        max_retries=2,
    )
    with pytest.raises(BackendHTTPRequestError) as oai_exc:
        oai.generate(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
        )

    assert oai_attempts == 1
    assert sleep_delays == []
    assert oai_exc.value.status_code == 401
    assert oai_exc.value.attempts == 1
    assert oai_exc.value.retryable is False
    oai_dump = json.dumps(oai_exc.value.to_dict())
    assert oai_secret not in str(oai_exc.value)
    assert oai_secret not in oai_dump

    gem_attempts = 0

    def _always_400_gem(url, headers, json, timeout):
        nonlocal gem_attempts
        gem_attempts += 1
        return _FakeStatusHTTPResp(
            status_code=400,
            payload={
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": f"Bad request with x-goog-api-key={gem_secret} and https://host/?key={gem_secret}",
                }
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _always_400_gem)
    gem = GoogleGeminiClient(
        model_id="gemini-3.8-flash",
        api_key=gem_secret,
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
        max_retries=2,
    )
    with pytest.raises(BackendHTTPRequestError) as gem_exc:
        gem.generate(
            system_prompt="sys",
            user_prompt="usr",
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
        )

    assert gem_attempts == 1
    assert sleep_delays == []
    assert gem_exc.value.status_code == 400
    assert gem_exc.value.attempts == 1
    assert gem_exc.value.retryable is False
    gem_dump = json.dumps(gem_exc.value.to_dict())
    assert gem_secret not in str(gem_exc.value)
    assert gem_secret not in gem_dump


def test_probe_live_backends_gemini_failure_still_verifies_openai_with_fixture_and_budget_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """When Gemini fails with HTTP 503, OpenAI still runs via controlled fixture, saves partial report, and runs budget guard."""
    gem_secret = "AIzaSyFakeProbeSecretKey112233"
    oai_secret = "sk-fake-probe-openai-445566"
    monkeypatch.setenv("GEMINI_API_KEY", gem_secret)
    monkeypatch.setenv("OPENAI_API_KEY", oai_secret)
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda _sec: None)

    def _fake_get(url, headers, timeout, params=None):
        if "generativelanguage.googleapis.com" in url:
            return _FakeStatusHTTPResp(
                status_code=200,
                payload={"name": "models/gemini-3.8-flash", "inputTokenLimit": 1000000, "outputTokenLimit": 65536},
                headers={"x-ratelimit-limit-requests": "1000"},
            )
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={"id": "gpt-5.4-mini", "owned_by": "openai"},
            headers={"x-ratelimit-limit-requests": "5000", "x-ratelimit-remaining-requests": "4998"},
        )

    fixture_bits = str(CONTROLLED_SMOKE_AUDIT_FIXTURE["intended_bits"])
    fixture_labels = {
        lbl: int(bit) for lbl, bit in zip(HALLUCINATION_LABELS, fixture_bits)
    }
    judge_content = json.dumps(
        {
            "labels": fixture_labels,
            "evidence": {
                lbl: ("Controlled fixture contradicts IM epinephrine standard." if val else "")
                for lbl, val in fixture_labels.items()
            },
            "coherence_score_1_to_5": 5,
            "diagnostic_notes": "Fixture smoke audit",
        }
    )

    def _fake_post(url, headers, json, timeout):
        if "generativelanguage.googleapis.com" in url:
            return _FakeStatusHTTPResp(
                status_code=503,
                payload={
                    "error": {
                        "code": 503,
                        "status": "UNAVAILABLE",
                        "message": f"Model currently experiencing high demand (debug key={gem_secret})",
                    }
                },
            )
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "model": "gpt-5.4-mini-2026-03-17",
                "choices": [{"message": {"content": judge_content}}],
                "usage": {
                    "prompt_tokens": 115,
                    "completion_tokens": 45,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
            headers={"x-ratelimit-remaining-requests": "4997", "x-ratelimit-limit-tokens": "2000000"},
        )

    monkeypatch.setattr("src.generation.backend.requests.get", _fake_get)
    monkeypatch.setattr("src.generation.backend.requests.post", _fake_post)

    cfg = load_yaml_config("configs/generation.yaml")
    out_report = tmp_path / "partial_smoke.json"
    rep = probe_live_backends(cfg, use_dotenv=False, write_report_path=out_report)

    assert out_report.exists()
    raw_report = out_report.read_text(encoding="utf-8")
    assert gem_secret not in raw_report
    assert oai_secret not in raw_report
    assert rep["api_keys_redacted"] is True
    assert rep["overall_status"] == "partial_failure"

    assert rep["synthetic_generator"]["passed"] is False
    assert "gemini_models_availability" in rep["successful_checks"]
    assert "gemini_structured_json_generation" in rep["failed_checks"]
    assert rep["provider_errors"]["synthetic_generator"]["status_code"] == 503
    assert rep["provider_errors"]["synthetic_generator"]["attempts"] == 3
    assert gem_secret not in rep["provider_errors"]["synthetic_generator"]["error_message"]

    assert rep["cross_judge_auditor"]["passed"] is True
    assert rep["cross_judge_auditor"]["used_controlled_fixture"] is True
    assert rep["cross_judge_auditor"]["audit_input_source"] == "controlled_audit_fixture"
    assert rep["cross_judge_auditor"]["controlled_fixture"]["is_controlled_fixture"] is True
    assert rep["cross_judge_auditor"]["controlled_fixture"]["is_research_generated_sample"] is False
    assert "openai_models_availability" in rep["successful_checks"]
    assert "openai_structured_json_audit" in rep["successful_checks"]
    assert "openai_canonical_schema_validation" in rep["successful_checks"]

    assert rep["budget_guard_verification"]["passed"] is True
    assert rep["budget_guard_verification"]["max_pilot_budget_usd"] == 10.0
    assert rep["budget_guard_verification"]["boundary_over_cap_halted_safely"] is True
    assert rep["budget_guard_verification"]["boundary_request_rejected_before_api_call"] is True
    assert rep["budget_guard_verification"]["boundary_blocked_calls_prevented"] == 1
    assert "budget_guard_over_cap_boundary" in rep["successful_checks"]

    assert rep["timings_ms"]["total_probe_ms"] >= 0.0
    assert rep["rate_limit_summary"]["cross_judge_auditor"]["models_endpoint"]["x-ratelimit-limit-requests"] == "5000"


def test_probe_live_backends_both_fail_still_writes_report_and_runs_budget_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Even when both providers fail, probe_live_backends saves a sanitized report and runs the $10 budget guard."""
    gem_secret = "AIzaSyBothFailGemSecret7788"
    oai_secret = "sk-both-fail-openai-7788"
    monkeypatch.setenv("GEMINI_API_KEY", gem_secret)
    monkeypatch.setenv("OPENAI_API_KEY", oai_secret)
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda _sec: None)

    over_cap_post_called = False

    def _fail_get(url, headers, timeout, params=None):
        if "generativelanguage.googleapis.com" in url:
            return _FakeStatusHTTPResp(
                status_code=401,
                payload={"error": {"message": f"Invalid Gemini key {gem_secret}"}},
            )
        return _FakeStatusHTTPResp(
            status_code=403,
            payload={"error": {"message": f"OpenAI forbidden with Bearer {oai_secret}"}},
        )

    def _unexpected_post(url, headers, json, timeout):
        nonlocal over_cap_post_called
        over_cap_post_called = True
        raise AssertionError("No POST call should occur when /models fails and budget boundary blocks.")

    monkeypatch.setattr("src.generation.backend.requests.get", _fail_get)
    monkeypatch.setattr("src.generation.backend.requests.post", _unexpected_post)

    cfg = load_yaml_config("configs/generation.yaml")
    out_report = tmp_path / "both_failed.json"
    rep = probe_live_backends(cfg, use_dotenv=False, write_report_path=out_report)

    assert over_cap_post_called is False
    assert out_report.exists()
    raw_text = out_report.read_text(encoding="utf-8")
    assert gem_secret not in raw_text
    assert oai_secret not in raw_text
    assert rep["overall_status"] == "failed"
    assert rep["synthetic_generator"]["passed"] is False
    assert rep["cross_judge_auditor"]["passed"] is False
    assert rep["provider_errors"]["synthetic_generator"]["status_code"] == 401
    assert rep["provider_errors"]["cross_judge_auditor"]["status_code"] == 403
    assert rep["budget_guard_verification"]["passed"] is True
    assert rep["budget_guard_verification"]["boundary_over_cap_halted_safely"] is True
    assert rep["budget_guard_verification"]["boundary_request_rejected_before_api_call"] is True
    assert "budget_guard_over_cap_boundary" in rep["successful_checks"]


def test_post_read_timeout_does_not_retry_while_connect_timeout_and_get_retry(monkeypatch) -> None:
    """Prevent duplicate billed POST calls on ReadTimeout while retrying safe ConnectTimeout/GET/scrubbing RequestException."""
    sleep_delays: list[float] = []
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda sec: sleep_delays.append(sec))

    # 1) POST ReadTimeout must NOT retry
    post_attempts = 0

    def _post_read_timeout(url, headers, json, timeout):
        nonlocal post_attempts
        post_attempts += 1
        raise requests.exceptions.ReadTimeout("timed out reading response from server")

    monkeypatch.setattr("src.generation.backend.requests.post", _post_read_timeout)
    oai = OpenAICompatibleClient(
        model_id="gpt-5.4-mini",
        api_base="https://api.openai.com/v1",
        api_key="sk-ReadTimeoutSecretKey123",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        max_retries=2,
    )
    with pytest.raises(BackendHTTPRequestError) as rt_exc:
        oai.generate(system_prompt="sys", user_prompt="usr", temperature=0.0, top_p=1.0, max_tokens=128)
    assert post_attempts == 1
    assert sleep_delays == []
    assert rt_exc.value.retryable is False
    assert "duplicate execution" in str(rt_exc.value)

    # 2) POST ConnectTimeout (never connected) retries and recovers
    connect_attempts = 0

    def _post_connect_then_ok(url, headers, json, timeout):
        nonlocal connect_attempts
        connect_attempts += 1
        if connect_attempts == 1:
            raise requests.exceptions.ConnectTimeout("connect timed out")
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "model": "gpt-5.4-mini",
                "choices": [{"message": {"content": '{"labels": {"factual_inconsistency": 1}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _post_connect_then_ok)
    res = oai.generate(system_prompt="sys", user_prompt="usr", temperature=0.0, top_p=1.0, max_tokens=128)
    assert connect_attempts == 2
    assert sleep_delays == [1.0]
    assert res.input_tokens == 10

    # 3) GET /models ReadTimeout retries safely (idempotent)
    sleep_delays.clear()
    get_attempts = 0

    def _get_read_timeout_then_ok(url, headers, timeout, params=None):
        nonlocal get_attempts
        get_attempts += 1
        if get_attempts == 1:
            raise requests.exceptions.ReadTimeout("get timeout")
        return _FakeStatusHTTPResp(status_code=200, payload={"data": [{"id": "gpt-5.4-mini"}]})

    monkeypatch.setattr("src.generation.backend.requests.get", _get_read_timeout_then_ok)
    avail = oai.verify_available_detailed()
    assert get_attempts == 2
    assert sleep_delays == [1.0]
    assert avail["available_ids"] == ["gpt-5.4-mini"]

    # 4) Quoted JSON dict credentials + RequestException URL params are redacted
    def _post_chunked_err(url, headers, json, timeout):
        raise requests.exceptions.ChunkedEncodingError(
            'Stream error with {"x-goog-api-key": "custom-unprefixed-secret-abc"} and "authorization": "tok_999"'
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _post_chunked_err)
    gem = GoogleGeminiClient(
        model_id="gemini-3.8-flash",
        api_key="custom-unprefixed-secret-abc",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 3.75},
        max_retries=2,
    )
    with pytest.raises(BackendHTTPRequestError) as ce_exc:
        gem.generate(system_prompt="sys", user_prompt="usr", temperature=0.0, top_p=1.0, max_tokens=128)
    ce_dump = json.dumps(ce_exc.value.to_dict())
    assert "custom-unprefixed-secret-abc" not in ce_dump
    assert "tok_999" not in ce_dump


def test_probe_live_backends_distinguishes_schema_validation_from_provider_error_and_reconciles_spend(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """When HTTP 200 succeeds but schema validation fails, provider_errors stays empty, validation_errors captures it, and spend reconciles."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySchemaTestGem123456")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-schema-test-openai-123456")

    def _fake_get(url, headers, timeout, params=None):
        if "generativelanguage.googleapis.com" in url:
            return _FakeStatusHTTPResp(
                status_code=200,
                payload={"name": "models/gemini-3.8-flash"},
            )
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={"id": "gpt-5.4-mini"},
        )

    fixture_bits = str(CONTROLLED_SMOKE_AUDIT_FIXTURE["intended_bits"])
    fixture_labels = {lbl: int(bit) for lbl, bit in zip(HALLUCINATION_LABELS, fixture_bits)}
    judge_content = json.dumps(
        {
            "labels": fixture_labels,
            "evidence": {lbl: ("Fixture evidence." if val else "") for lbl, val in fixture_labels.items()},
            "coherence_score_1_to_5": 5,
            "diagnostic_notes": "Fixture audit ok",
        }
    )

    def _fake_post(url, headers, json, timeout):
        if "generativelanguage.googleapis.com" in url:
            # HTTP 200 OK with valid token billing, but missing required Strategy B keys!
            return _FakeStatusHTTPResp(
                status_code=200,
                payload={
                    "modelVersion": "gemini-3.8-flash",
                    "candidates": [{"content": {"parts": [{"text": '{"generated_response": "Incomplete JSON"}'}]}}],
                    "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 30, "thoughtsTokenCount": 14},
                },
                headers={"x-ratelimit-remaining-requests": "999"},
            )
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "model": "gpt-5.4-mini-2026-03-17",
                "choices": [{"message": {"content": judge_content}}],
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 40,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.get", _fake_get)
    monkeypatch.setattr("src.generation.backend.requests.post", _fake_post)

    cfg = load_yaml_config("configs/generation.yaml")
    out_report = tmp_path / "schema_vs_provider.json"
    rep = probe_live_backends(cfg, use_dotenv=False, write_report_path=out_report)

    assert rep["overall_status"] == "partial_failure"
    assert rep["provider_errors"] == {}
    assert "synthetic_generator" in rep["validation_errors"]
    assert rep["validation_errors"]["synthetic_generator"]["failure_category"] == "schema_validation_error"
    assert "gemini_structured_json_generation" in rep["successful_checks"]
    assert "gemini_canonical_schema_validation" in rep["failed_checks"]
    assert rep["synthetic_generator"]["generation_check"]["passed"] is True
    assert rep["synthetic_generator"]["thoughts_tokens"] == 14
    assert rep["synthetic_generator"]["cost_usd"] > 0.0
    assert rep["cross_judge_auditor"]["used_controlled_fixture"] is True
    assert rep["budget_guard_verification"]["live_smoke_spend_usd"] == pytest.approx(rep["total_smoke_cost_usd"])


def test_credit_balance_exhausted_429_does_not_retry_and_gsk_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Billing exhaustion HTTP 429 fails immediately on attempt 1 and Groq keys redact cleanly."""
    sleep_delays: list[float] = []
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda sec: sleep_delays.append(sec))

    calls = 0

    def _billing_429(url, headers, json, timeout, stream=False):
        nonlocal calls
        calls += 1
        return _FakeStatusHTTPResp(
            status_code=429,
            payload={
                "error": {
                    "message": "Your credit balance is exhausted.",
                    "type": "invalid_request_error",
                    "code": "credit_balance_exhausted",
                }
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _billing_429)
    oai = OpenAICompatibleClient(
        model_id="gpt-5.4-mini",
        api_base="https://api.openai.com/v1",
        api_key="sk-DummyBillingKey0001",
        pricing_usd_per_1m_tokens={"input": 0.75, "output": 4.50},
        max_retries=3,
    )
    with pytest.raises(BackendHTTPRequestError) as exc_info:
        oai.generate(system_prompt="sys", user_prompt="usr", temperature=0.0, top_p=1.0, max_tokens=128)
    assert calls == 1
    assert sleep_delays == []
    assert exc_info.value.retryable is False
    assert "***REDACTED***" in _redact_secrets("Authorization: Bearer gsk_SecretGroqToken1234567890")
    assert "***REDACTED_GROQ_KEY***" in _redact_secrets("key=gsk_SecretGroqToken1234567890")


def test_free_groq_config_wiring_and_gemini_sse_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate generation_free_groq.yaml wiring, preflight guard, Groq payload, and Gemini SSE."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaDummyFreeGemKey12345")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_DummyFreeGroqKey12345")

    cfg = load_yaml_config("configs/generation_free_groq.yaml")
    # Unverified config copy must still block preflight
    unverified_cfg = {
        **cfg,
        "backends": {
            **cfg["backends"],
            "synthetic_generator": {**cfg["backends"]["synthetic_generator"], "model_availability_verified": False},
        },
    }
    with pytest.raises(BackendNotConfiguredError, match="model_availability_verified=false"):
        verify_generation_and_judge_backends(unverified_cfg, use_dotenv=False, check_live=False)

    status = verify_generation_and_judge_backends(cfg, use_dotenv=False, check_live=False)
    assert status["verified"] is True
    assert status["cross_judge_independence"]["independence_verified"] is True

    # Zero-cost free-tier pricing must still trigger the $10 budget-guard boundary check
    from src.generation.backend import _run_budget_boundary_probe, init_budget_state

    boundary_res = _run_budget_boundary_probe(
        cfg,
        init_budget_state(cfg),
        type("_FreeProxy", (), {"pricing": {"input": 0.0, "output": 0.0}})(),
        system_prompt="s",
        user_prompt="u",
        max_output_tokens=256,
    )
    assert boundary_res["passed"] is True
    assert boundary_res["boundary_over_cap_halted_safely"] is True
    assert boundary_res["boundary_request_rejected_before_api_call"] is True

    posted_payloads: list[dict] = []

    class _FakeSSEResp:
        status_code = 200
        headers = {"x-ratelimit-remaining-requests": "19"}
        text = ""

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, decode_unicode=True):
            chunks = [
                {
                    "modelVersion": "gemini-3.8-flash",
                    "candidates": [{"content": {"parts": [{"text": '{"generated_response": '}]}}],
                    "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 5, "thoughtsTokenCount": 3},
                },
                {
                    "modelVersion": "gemini-3.8-flash",
                    "candidates": [{"content": {"parts": [{"text": '"Streamed ok"}'}]}}],
                    "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 11, "thoughtsTokenCount": 9},
                },
            ]
            for c in chunks:
                yield "data: " + json.dumps(c)

    def _fake_post(url, headers, json, timeout, stream=False, params=None):
        posted_payloads.append({"url": url, "json": json, "stream": stream, "params": params})
        if "generativelanguage.googleapis.com" in url:
            assert stream is True
            assert params == {"alt": "sse"}
            assert ":streamGenerateContent" in url
            return _FakeSSEResp()
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "model": "openai/gpt-oss-120b",
                "choices": [{"message": {"content": '{"labels": {"factual_inconsistency": 1}}'}}],
                "usage": {"prompt_tokens": 55, "completion_tokens": 22},
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.post", _fake_post)
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda _s: None)

    gen_client = build_live_client(cfg["backends"]["synthetic_generator"], use_dotenv=False)
    gen_resp = gen_client.generate(system_prompt="s", user_prompt="u", temperature=0.8, top_p=0.95, max_tokens=512)
    assert gen_resp.raw_text == '{"generated_response": "Streamed ok"}'
    assert gen_resp.thoughts_tokens == 9
    assert gen_resp.cost_usd == 0.0

    judge_client = build_live_client(cfg["backends"]["cross_judge_auditor"], use_dotenv=False)
    judge_resp = judge_client.generate(system_prompt="s", user_prompt="u", temperature=0.0, top_p=1.0, max_tokens=1536)
    assert judge_resp.cost_usd == 0.0
    groq_req = posted_payloads[-1]["json"]
    assert groq_req["max_tokens"] == 1536
    assert "max_completion_tokens" not in groq_req
    assert groq_req["reasoning_format"] == "hidden"


def test_pilot_resume_and_full_plus_natural_generation_offline(tmp_path: Path) -> None:
    """Verify bounded Pilot E0 resume recovery, full generation sampling/split isolation, and natural QA."""
    train_pool = _make_source_records_pool(split_name="train")
    val_pool = _make_source_records_pool(split_name="val")
    nat_pool = _make_source_records_pool(split_name="test_natural_pool")

    variant_a = (
        "Altered Sentence one introduces a clear distortion across domain claims. "
        "Invented Cohort Study 2026 fabricated unsupported citations. "
        "Sentence three concludes inconsistently with sentence one."
    )
    variant_b = (
        "Modified clause alpha replaces standard baseline guidelines with reversed thresholds. "
        "Unverified Archive Report X-99 invented non-existent clinical regulatory statutes. "
        "Therefore the closing statement directly contradicts earlier quantitative assertions."
    )

    def _gen_responder(sys_p: str, usr_p: str, call_n: int, temp: float) -> str:
        chosen_body = variant_a if call_n % 2 == 0 else variant_b
        if "Generate ONE incremental perturbation step" in usr_p:
            return json.dumps(
                {
                    "perturbed_response": f"{chosen_body} [step_{call_n}]",
                    "target_category": "factual_inconsistency",
                    "edited_span": "Altered Sentence one",
                    "edit_rationale": "Mock edit.",
                    "preserved_elements_check": "ok",
                }
            )
        return json.dumps(
            {
                "generated_response": f"{chosen_body} [call_{call_n}]",
                "per_label_modifications": {"mock": "edited"},
                "negative_constraints_verified": "checked",
            }
        )

    def _judge_responder(sys_p: str, usr_p: str, call_n: int, temp: float) -> str:
        is_clean = "Sentence one explains core mechanism" in usr_p
        val = 0 if is_clean else 1
        preds = {
            "factual_inconsistency": val,
            "factual_fabrication": 0,
            "instruction_inconsistency": 0,
            "logical_inconsistency": 0,
            "context_inconsistency": 0,
        }
        return json.dumps(
            {
                "labels": preds,
                "evidence": {},
                "coherence_score_1_to_5": 4,
                "diagnostic_notes": "Mock judge ok.",
            }
        )

    def _nat_responder(sys_p: str, usr_p: str, call_n: int, temp: float) -> str:
        return f"Natural domain answer response [call_{call_n}] addressing the question."

    # First partial run: 4 slots then interrupt
    pilot_dir = tmp_path / "pilot"
    gen_mock = MockLLMClient(
        model_id="gemini-3.8-flash",
        provider="mock_google_gemini",
        model_family="gemini",
        responder=_gen_responder,
    )
    judge_mock = MockLLMClient(
        model_id="openai/gpt-oss-120b",
        provider="mock_groq",
        model_family="gpt",
        responder=_judge_responder,
    )
    part1 = run_pilot_e0(
        "configs/generation.yaml",
        generator_client=gen_mock,
        judge_client=judge_mock,
        output_dir=pilot_dir,
        source_records_override=train_pool,
        resume=False,
        max_new_slots=4,
        checkpoint_every=2,
    )
    assert part1["new_slots_processed_this_run"] == 4
    # Resume with 3 more slots
    part2 = run_pilot_e0(
        "configs/generation.yaml",
        generator_client=gen_mock,
        judge_client=judge_mock,
        output_dir=pilot_dir,
        source_records_override=train_pool,
        resume=True,
        max_new_slots=3,
        checkpoint_every=2,
    )
    assert part2["new_slots_processed_this_run"] == 3
    recs_after_resume = read_jsonl(pilot_dir / "pilot_records.jsonl")
    assert len({r["sample_id"] for r in recs_after_resume}) == len(recs_after_resume)

    # Verify sample_full_generation_spec + run_full_generation split lock on val pool
    cfg = load_yaml_config("configs/generation.yaml")
    val_specs, val_sample_sum = sample_full_generation_spec(
        val_pool,
        cfg,
        split_name="val",
        total_target_override=20,
    )
    assert len(val_specs) == 20
    assert val_sample_sum["split_name"] == "val"
    full_dir = tmp_path / "full"
    full_sum = run_full_generation(
        "configs/generation.yaml",
        split_name="val",
        strategy_name="strategy_a",
        generator_client=gen_mock,
        judge_client=judge_mock,
        output_dir=full_dir,
        source_records_override=val_pool,
        total_target_override=20,
        checkpoint_every=5,
    )
    assert full_sum["split_name"] == "val"
    val_recs = read_jsonl(full_dir / "synthetic_val.jsonl")
    assert len(val_recs) == 20
    assert all(r["split"] == "val" for r in val_recs)

    # Verify natural QA leaves adjudicated_gold_labels=None
    nat_clients = {
        m["model_id"]: MockLLMClient(
            model_id=m["model_id"],
            provider=m.get("provider", "mock"),
            model_family=m["model_family"],
            responder=_nat_responder,
        )
        for m in cfg["backends"]["natural_qa_models"]
    }
    nat_sum = run_natural_generation(
        "configs/generation.yaml",
        respondent_clients=nat_clients,
        judge_client=judge_mock,
        output_dir=full_dir,
        source_records_override=nat_pool,
        max_new_slots=6,
    )
    assert nat_sum["new_slots_processed_this_run"] == 6
    nat_recs = read_jsonl(full_dir / "test_natural_unannotated.jsonl")
    assert len(nat_recs) == 6
    assert all(r["adjudicated_gold_labels"] is None and r["split"] == "test_natural" for r in nat_recs)


def test_export_full_and_natural_generation_plans(tmp_path: Path) -> None:
    """Verify deterministic export helpers write JSONL sampling plans without live LLM calls."""
    cfg = load_yaml_config("configs/generation_free_groq.yaml")
    pools = {
        "train": _make_source_records_pool("train"),
        "val": _make_source_records_pool("val"),
        "test_synthetic": _make_source_records_pool("test_synthetic_pool"),
    }
    full_dir = tmp_path / "full_plans"
    full_sum = export_full_generation_plans(
        cfg,
        strategy_name="strategy_a",
        total_target_override=12,
        output_dir=full_dir,
        source_pools_override=pools,
    )
    assert full_sum["total_planned_items"] == 36
    for split in ("train", "val", "test_synthetic"):
        rows = read_jsonl(full_dir / f"plan_{split}.jsonl")
        assert len(rows) == 12
        assert all("source_record_id" in r and "intended_bits" in r for r in rows)
    assert (full_dir / "full_sampling_summary.json").exists()

    nat_pool = _make_source_records_pool("test_natural_pool")
    nat_dir = tmp_path / "nat_plans"
    nat_sum = export_natural_generation_plan(
        cfg,
        output_dir=nat_dir,
        source_records_override=nat_pool,
    )
    nat_rows = read_jsonl(nat_dir / "test_natural_plan.jsonl")
    assert len(nat_rows) == nat_sum["exported_item_count"] > 0
    assert all("natural_item_id" in r and "respondent_model_spec" in r for r in nat_rows)
    assert (nat_dir / "natural_sampling_summary.json").exists()


def test_local_ollama_keyless_provider_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate keyless localhost Ollama provider wiring without leaking Authorization headers."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaDummyKeyForOllamaTest")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    cfg = load_yaml_config("configs/generation_local_ollama.yaml")
    status = verify_generation_and_judge_backends(cfg, use_dotenv=False, check_live=False)
    assert status["verified"] is True
    assert status["cross_judge_independence"]["independence_verified"] is True
    verified_cfg = cfg

    captured_headers: list[dict] = []

    def _fake_get(url, headers, timeout):
        captured_headers.append(dict(headers))
        return _FakeStatusHTTPResp(status_code=200, payload={"data": [{"id": "qwen3:4b-instruct"}]})

    def _fake_post(url, headers, json, timeout, stream=False, params=None):
        captured_headers.append(dict(headers))
        return _FakeStatusHTTPResp(
            status_code=200,
            payload={
                "model": "qwen3:4b-instruct",
                "choices": [{"message": {"content": '{"labels": {"factual_inconsistency": 1}}'}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            },
        )

    monkeypatch.setattr("src.generation.backend.requests.get", _fake_get)
    monkeypatch.setattr("src.generation.backend.requests.post", _fake_post)
    monkeypatch.setattr("src.generation.backend.time.sleep", lambda _s: None)

    judge_client = build_live_client(verified_cfg["backends"]["cross_judge_auditor"], use_dotenv=False)
    assert judge_client.verify_available() == ["qwen3:4b-instruct"]
    resp = judge_client.generate(system_prompt="s", user_prompt="u", temperature=0.0, top_p=1.0, max_tokens=512)
    assert resp.cost_usd == 0.0
    assert all("Authorization" not in h for h in captured_headers)





