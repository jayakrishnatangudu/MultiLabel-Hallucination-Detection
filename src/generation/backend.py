"""LLM generation/auditing backend abstractions, Gemini support, independence checks, and preflight verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Protocol

import requests

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


DEFAULT_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

TRANSIENT_HTTP_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
NON_RETRYABLE_BILLING_OR_QUOTA_CODES: frozenset[str] = frozenset(
    {
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing_not_active",
        "account_deactivated",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
    }
)
DEFAULT_HTTP_MAX_RETRIES: int = 2
DEFAULT_HTTP_RETRY_BASE_DELAY_SEC: float = 1.0
DEFAULT_HTTP_RETRY_MAX_DELAY_SEC: float = 8.0

_PLACEHOLDER_PREFIXES = ("PENDING_", "UNVERIFIED_", "TBD_", "SELECT_")

_SENSITIVE_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-=+/]{4,}", flags=re.I), r"\1***REDACTED***"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{10,}"), "***REDACTED_GEMINI_KEY***"),
    (re.compile(r"sk-(?:[A-Za-z0-9]+-)*[A-Za-z0-9\-_]{10,}"), "***REDACTED_OPENAI_KEY***"),
    (re.compile(r"gsk_[0-9A-Za-z\-_]{10,}"), "***REDACTED_GROQ_KEY***"),
    (
        re.compile(
            r"""((?:['"]?)(?:x-goog-api-key|x-api-key|authorization|api[-_]?key|access[-_]?token|bearer[-_]?token|openai[-_]?api[-_]?key|gemini[-_]?api[-_]?key)(?:['"]?)\s*[:=]\s*)(?:'[^']*'|\"[^\"]*\"|[^\s'\",}\]]+)""",
            flags=re.I,
        ),
        r"\1***REDACTED***",
    ),
    (re.compile(r"([?&]key=)[^&\s\"']+"), r"\1***REDACTED***"),
)

CONTROLLED_SMOKE_AUDIT_FIXTURE: dict[str, Any] = {
    "fixture_id": "CONTROLLED_AUDIT_FIXTURE_ANAPHYLAXIS_FI_001",
    "is_controlled_fixture": True,
    "is_research_generated_sample": False,
    "intended_bits": "10000",
    "candidate_response": (
        "Oral aspirin tablet taken once with water is the first-line emergency treatment "
        "for acute anaphylaxis because selective COX-2 inhibition rapidly resolves airway "
        "edema and reverses distributive shock."
    ),
    "fixture_description": (
        "Fixed controlled audit fixture asserting factual_inconsistency=1 (replaces IM epinephrine "
        "with oral aspirin / COX-2 inhibition) used solely for independent cross-judge smoke "
        "verification when Gemini generation is unavailable; NOT a generated research sample."
    ),
}

FAMILY_VENDOR_LINEAGE: dict[str, str] = {
    "gemini": "google_deepmind",
    "gemma": "google_deepmind",
    "gpt": "openai_proprietary",
    "o1": "openai_proprietary",
    "o3": "openai_proprietary",
    "o4": "openai_proprietary",
    "claude": "anthropic",
    "llama": "meta_llama",
    "qwen": "alibaba_qwen",
    "mistral": "mistral_ai",
    "mixtral": "mistral_ai",
    "ministral": "mistral_ai",
    "deepseek": "deepseek_ai",
    "phi": "microsoft",
    "command": "cohere",
}


class _RateLimitPacer:
    """Optional client-side RPM/TPM sliding-window pacer for strict free-tier APIs."""

    def __init__(self, *, min_interval_sec: float = 0.0, tpm_limit: int | None = None) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec or 0.0))
        self.tpm_limit = int(tpm_limit) if tpm_limit and int(tpm_limit) > 0 else None
        self._last_request_ts: float | None = None
        self._token_window: list[tuple[float, int]] = []

    def wait_before_call(self, estimated_tokens: int = 0) -> None:
        now = time.monotonic()
        if self.min_interval_sec > 0.0 and self._last_request_ts is not None:
            gap = (self._last_request_ts + self.min_interval_sec) - now
            if gap > 0.0:
                time.sleep(gap)
                now = time.monotonic()

        if self.tpm_limit is not None and estimated_tokens > 0:
            cutoff = now - 60.0
            self._token_window = [(ts, tok) for ts, tok in self._token_window if ts > cutoff]
            used = sum(tok for _, tok in self._token_window)
            while self._token_window and (used + estimated_tokens) > self.tpm_limit:
                oldest_ts, _ = self._token_window[0]
                sleep_for = max(0.1, (oldest_ts + 60.05) - time.monotonic())
                time.sleep(sleep_for)
                now = time.monotonic()
                cutoff = now - 60.0
                self._token_window = [(ts, tok) for ts, tok in self._token_window if ts > cutoff]
                used = sum(tok for _, tok in self._token_window)

        self._last_request_ts = time.monotonic()

    def record_tokens(self, tokens_used: int) -> None:
        if self.tpm_limit is not None and tokens_used > 0:
            now = time.monotonic()
            cutoff = now - 60.0
            self._token_window = [(ts, tok) for ts, tok in self._token_window if ts > cutoff]
            self._token_window.append((now, int(tokens_used)))


class BackendNotConfiguredError(RuntimeError):
    """Raised when generator/judge config, credentials, verification flags, or independence fail."""


class BackendHTTPRequestError(RuntimeError):
    """Raised when an HTTP API call fails permanently or exhausts bounded transient retries."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model_id: str,
        endpoint: str,
        status_code: int | None,
        attempts: int,
        retryable: bool,
        error_details: dict[str, Any] | None = None,
        rate_limit_headers: dict[str, str] | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model_id = model_id
        self.endpoint = endpoint
        self.status_code = status_code
        self.attempts = int(attempts)
        self.retryable = bool(retryable)
        self.error_message = message
        self.error_details = error_details or {}
        self.rate_limit_headers = rate_limit_headers or {}
        self.latency_ms = round(float(latency_ms), 2)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe, key-redacted dictionary representation for verification reports."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "attempts": self.attempts,
            "retryable": self.retryable,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "rate_limit_headers": self.rate_limit_headers,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LLMResponse:
    """Normalized LLM completion response with token, latency, and cost telemetry."""

    raw_text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_family: str | None = None
    thoughts_tokens: int = 0
    reasoning_tokens: int = 0
    response_model_id: str | None = None
    rate_limit_headers: dict[str, str] | None = None


class LLMClient(Protocol):
    """Protocol implemented by Gemini REST, OpenAI-compatible HTTP, and deterministic mock clients."""

    model_id: str
    provider: str
    model_family: str

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        ...


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    pricing_cfg: dict[str, Any],
) -> float:
    """Compute exact USD cost from token counts and configured per-1M token rates."""
    in_rate = float(pricing_cfg.get("input", 0.0) or 0.0)
    out_rate = float(pricing_cfg.get("output", 0.0) or 0.0)
    return round((input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate, 8)


def _redact_secrets(text: str, secrets: list[str] | None = None) -> str:
    """Scrub explicit secrets, auth headers, and API-key patterns from logs or serialized reports."""
    cleaned = str(text)
    for sec in secrets or []:
        s = str(sec).strip()
        if s and len(s) >= 4:
            cleaned = cleaned.replace(s, "***REDACTED***")
    for pattern, repl in _SENSITIVE_VALUE_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    return cleaned


def _redact_nested_value(value: Any, secrets: list[str] | None = None) -> Any:
    """Recursively scrub credentials across dicts, lists, and strings."""
    if isinstance(value, str):
        return _redact_secrets(value, secrets)
    if isinstance(value, dict):
        return {str(k): _redact_nested_value(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_nested_value(item, secrets) for item in value]
    return value


def _parse_retry_after_seconds(headers: Any, *, max_delay_sec: float) -> float | None:
    """Parse numeric Retry-After header conservatively, capped by max_delay_sec."""
    if not headers or not hasattr(headers, "items"):
        return None
    for k, v in headers.items():
        if str(k).lower() == "retry-after":
            try:
                val = float(str(v).strip())
                if val > 0.0:
                    return min(max_delay_sec, val)
            except ValueError:
                return None
    return None


def _extract_provider_error_info(
    resp: Any,
    *,
    secrets: list[str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Safely extract sanitized provider error fields and rate-limit headers without leaking secrets."""
    rl_headers = _extract_rate_limit_headers(getattr(resp, "headers", None))
    details: dict[str, Any] = {}
    summary = ""

    raw_json: Any = None
    try:
        raw_json = resp.json() if hasattr(resp, "json") else None
    except Exception:
        raw_json = None

    if isinstance(raw_json, dict):
        err_obj = raw_json.get("error") if isinstance(raw_json.get("error"), dict) else raw_json
        for field in ("code", "status", "type", "param"):
            if err_obj.get(field) is not None:
                details[field] = _redact_secrets(str(err_obj[field]), secrets)[:160]
        msg_val = err_obj.get("message") or raw_json.get("detail") or getattr(resp, "reason", None)
        if msg_val is not None:
            safe_msg = _redact_secrets(str(msg_val), secrets)[:500]
            details["message"] = safe_msg
            summary = safe_msg
    if not summary:
        raw_text = getattr(resp, "text", None)
        if isinstance(raw_text, str) and raw_text.strip():
            safe_txt = _redact_secrets(raw_text.strip(), secrets)[:400]
            details["raw_excerpt"] = safe_txt
            summary = safe_txt
    if not summary:
        reason = getattr(resp, "reason", None)
        summary = _redact_secrets(str(reason or "HTTP error"), secrets)
    return summary, details, rl_headers


def _is_non_retryable_billing_error(status_code: int, details: dict[str, Any]) -> bool:
    """Detect permanent billing/credit exhaustion errors returned with HTTP 429 or 402."""
    if status_code == 402:
        return True
    if status_code != 429:
        return False
    for key in ("code", "type", "status"):
        val = str(details.get(key, "")).strip().lower()
        if val in NON_RETRYABLE_BILLING_OR_QUOTA_CODES:
            return True
    return False


def _execute_http_with_retry(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float,
    provider: str,
    model_id: str,
    max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    base_delay_sec: float = DEFAULT_HTTP_RETRY_BASE_DELAY_SEC,
    max_delay_sec: float = DEFAULT_HTTP_RETRY_MAX_DELAY_SEC,
    secrets: list[str] | None = None,
    stream: bool = False,
) -> tuple[Any, float, int]:
    """
    Execute a GET/POST request with bounded exponential backoff strictly for transient
    HTTP failures (429 rate-limit, 500, 502, 503, 504) and network timeouts/connection resets.
    Never retries permanent 4xx errors (e.g., 400, 401, 403, 404) or 429 billing/credit exhaustion.
    """
    safe_url = _redact_secrets(url, secrets)
    retries_allowed = max(0, int(max_retries))
    base_d = max(0.0, float(base_delay_sec))
    max_d = max(base_d, float(max_delay_sec))
    t0 = time.perf_counter()

    for attempt_idx in range(retries_allowed + 1):
        attempt_num = attempt_idx + 1
        try:
            if method.upper() == "GET":
                get_kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout}
                if params is not None:
                    get_kwargs["params"] = params
                resp = requests.get(url, **get_kwargs)
            elif stream:
                post_kwargs: dict[str, Any] = {"headers": headers, "json": json_body, "timeout": timeout, "stream": True}
                if params is not None:
                    post_kwargs["params"] = params
                resp = requests.post(url, **post_kwargs)
            else:
                resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            safe_err = _redact_secrets(str(exc), secrets)
            is_read_timeout = isinstance(exc, requests.exceptions.ReadTimeout)
            is_transient_net = isinstance(
                exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
            )
            # Only retry ReadTimeout on idempotent GET; retrying ReadTimeout on POST can duplicate billed calls.
            can_retry = is_transient_net and not (is_read_timeout and method.upper() != "GET")
            if can_retry and attempt_idx < retries_allowed:
                delay = min(max_d, base_d * (2**attempt_idx))
                logger.warning(
                    "Transient network error for %s/%s (attempt %d/%d): %s; sleeping %.2fs",
                    provider,
                    model_id,
                    attempt_num,
                    retries_allowed + 1,
                    safe_err,
                    delay,
                )
                if delay > 0.0:
                    time.sleep(delay)
                continue
            reason = (
                "read timeout on non-idempotent POST (not retried to prevent duplicate execution)"
                if (is_read_timeout and method.upper() != "GET")
                else ("network error" if is_transient_net else "request error (not retried)")
            )
            raise BackendHTTPRequestError(
                f"{provider}/{model_id} {reason} after {attempt_num} attempt(s): {safe_err}",
                provider=provider,
                model_id=model_id,
                endpoint=safe_url,
                status_code=None,
                attempts=attempt_num,
                retryable=can_retry,
                error_details=_redact_nested_value(
                    {"exception_type": type(exc).__name__, "message": safe_err}, secrets
                ),
                latency_ms=elapsed_ms,
            ) from exc

        status_code = getattr(resp, "status_code", None)
        if not isinstance(status_code, int):
            try:
                resp.raise_for_status()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return resp, elapsed_ms, attempt_num
            except requests.exceptions.HTTPError as exc:
                err_resp = getattr(exc, "response", resp)
                status_code = getattr(err_resp, "status_code", None)
                if not isinstance(status_code, int):
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    safe_msg = _redact_secrets(str(exc), secrets)
                    raise BackendHTTPRequestError(
                        f"{provider}/{model_id} HTTP failure: {safe_msg}",
                        provider=provider,
                        model_id=model_id,
                        endpoint=safe_url,
                        status_code=None,
                        attempts=attempt_num,
                        retryable=False,
                        error_details={"message": safe_msg},
                        latency_ms=elapsed_ms,
                    ) from exc
                resp = err_resp

        if status_code < 400:
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return resp, elapsed_ms, attempt_num

        summary, details, rl_headers = _extract_provider_error_info(resp, secrets=secrets)
        is_billing_block = _is_non_retryable_billing_error(status_code, details)
        retryable = (status_code in TRANSIENT_HTTP_STATUS_CODES) and not is_billing_block
        if retryable and attempt_idx < retries_allowed:
            exp_delay = min(max_d, base_d * (2**attempt_idx))
            retry_after = _parse_retry_after_seconds(getattr(resp, "headers", None), max_delay_sec=max_d)
            delay = max(exp_delay, retry_after) if retry_after is not None else exp_delay
            logger.warning(
                "Transient HTTP %d from %s/%s (attempt %d/%d): %s; retrying in %.2fs",
                status_code,
                provider,
                model_id,
                attempt_num,
                retries_allowed + 1,
                summary,
                delay,
            )
            if delay > 0.0:
                time.sleep(delay)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        kind = "exhausted retries" if retryable else "permanent error (not retried)"
        raise BackendHTTPRequestError(
            f"{provider}/{model_id} HTTP {status_code} ({kind} after {attempt_num} attempt(s)): {summary}",
            provider=provider,
            model_id=model_id,
            endpoint=safe_url,
            status_code=status_code,
            attempts=attempt_num,
            retryable=retryable,
            error_details=details,
            rate_limit_headers=rl_headers,
            latency_ms=elapsed_ms,
        )

    raise RuntimeError("Unreachable state in _execute_http_with_retry")


def estimate_worst_case_call_cost_usd(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> float:
    """Upper-bound USD cost before issuing an API call so spend cannot cross max_pilot_budget_usd."""
    pricing_cfg = getattr(client, "pricing", None)
    if not isinstance(pricing_cfg, dict):
        return 0.0
    total_chars = len(system_prompt) + len(user_prompt)
    word_count = len(system_prompt.split()) + len(user_prompt.split())
    worst_in_tok = max(10, word_count, total_chars // 2) + 64
    worst_out_tok = max(5, int(max_output_tokens))
    return compute_cost_usd(worst_in_tok, worst_out_tok, pricing_cfg)


def init_budget_state(cfg: dict[str, Any]) -> dict[str, Any]:
    """Initialize runtime spend tracker and strict budget-stop state for Pilot E0."""
    pilot_cfg = cfg.get("pilot_e0", {})
    return {
        "max_pilot_budget_usd": float(pilot_cfg.get("max_pilot_budget_usd", 10.0)),
        "enforce_strict_budget_stop": bool(pilot_cfg.get("enforce_strict_budget_stop", True)),
        "cumulative_cost_usd": 0.0,
        "halted_safely": False,
        "halt_reason": None,
        "halted_at_sample_id": None,
        "blocked_calls_prevented": 0,
    }


def reserve_budget_for_call(
    budget_state: dict[str, Any] | None,
    *,
    client: Any,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    sample_id: str,
) -> bool:
    """Return True if the call fits cleanly under max_pilot_budget_usd; else mark safe halt."""
    if budget_state is None or not budget_state.get("enforce_strict_budget_stop", True):
        return True
    cap = float(budget_state["max_pilot_budget_usd"])
    spent = float(budget_state["cumulative_cost_usd"])
    est_next = estimate_worst_case_call_cost_usd(
        client, system_prompt, user_prompt, max_output_tokens
    )
    if spent + est_next > cap or spent >= cap:
        budget_state["halted_safely"] = True
        budget_state["blocked_calls_prevented"] = int(budget_state.get("blocked_calls_prevented", 0)) + 1
        budget_state["halted_at_sample_id"] = sample_id
        budget_state["halt_reason"] = (
            f"Stopped safely before {sample_id}: cumulative_cost_usd=${spent:.6f} + "
            f"est_max_next=${est_next:.6f} would exceed max_pilot_budget_usd=${cap:.6f}"
        )
        logger.warning(budget_state["halt_reason"])
        return False
    return True


def record_actual_spend(budget_state: dict[str, Any] | None, cost_usd: float) -> None:
    """Accumulate realized API cost immediately after an LLM call completes."""
    if budget_state is not None:
        budget_state["cumulative_cost_usd"] = round(
            float(budget_state["cumulative_cost_usd"]) + float(cost_usd), 8
        )


def load_local_dotenv(dotenv_path: Path | None = None) -> list[str]:
    """Load key=value pairs from local .env without overriding already-set environment variables."""
    target = dotenv_path if dotenv_path is not None else (PROJECT_ROOT / ".env")
    if not target.exists() or not target.is_file():
        return []
    loaded: list[str] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        elif "#" in val:
            val = val.split("#", 1)[0].strip()
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def _extract_rate_limit_headers(headers: Any) -> dict[str, str]:
    """Extract rate-limit and quota HTTP headers (case-insensitive) if returned by the server."""
    if not headers or not hasattr(headers, "items"):
        return {}
    out: dict[str, str] = {}
    for k, v in headers.items():
        lk = str(k).lower()
        if "ratelimit" in lk or lk in {"retry-after", "openai-processing-ms", "openai-organization"}:
            out[lk] = str(v)
    return out


def _should_omit_gemini_sampling_params(model_id: str, explicit_flag: bool | None) -> bool:
    """Gemini 3.5+/3.6+/3.8+/4 models deprecate temperature, topP, and topK."""
    if explicit_flag is not None:
        return bool(explicit_flag)
    lowered = model_id.strip().lower().removeprefix("models/")
    return lowered.startswith(("gemini-3.", "gemini-3-", "gemini-4"))


def _should_use_max_completion_tokens(model_id: str, explicit_flag: bool | None) -> bool:
    """OpenAI GPT-5/6/o-series models use max_completion_tokens on Chat Completions."""
    if explicit_flag is not None:
        return bool(explicit_flag)
    lowered = model_id.strip().lower()
    return lowered.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def _allows_optional_api_key(b_cfg: dict[str, Any]) -> bool:
    """Return True for local keyless runtimes such as Ollama on localhost."""
    provider = str(b_cfg.get("provider", "")).strip().lower()
    return provider in {"ollama", "local_ollama"} or bool(b_cfg.get("api_key_optional", False))


def _resolve_api_key(b_cfg: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve API key from primary api_key_env or any configured fallback env names."""
    candidates: list[str] = []
    primary = str(b_cfg.get("api_key_env", "")).strip()
    if primary:
        candidates.append(primary)
    for fb in b_cfg.get("api_key_fallback_envs", []) or []:
        s = str(fb).strip()
        if s and s not in candidates:
            candidates.append(s)
    for env_name in candidates:
        val = os.getenv(env_name, "").strip()
        if val:
            return val, env_name
    return "", (primary or (candidates[0] if candidates else None))


def infer_model_family_from_id(model_id: str) -> str | None:
    """Infer known canonical model family from model_id tokens to prevent family aliasing."""
    lowered = model_id.strip().lower()
    if not lowered or any(lowered.startswith(p.lower()) for p in _PLACEHOLDER_PREFIXES):
        return None
    for fam in (
        "gemini",
        "gemma",
        "claude",
        "qwen",
        "llama",
        "mixtral",
        "mistral",
        "deepseek",
        "gpt",
    ):
        if re.search(rf"(?:^|[/\-_:]){re.escape(fam)}(?:$|[/\-_:\d.])", lowered) or fam in lowered:
            return fam
    for reasoning_prefix in ("o1", "o3", "o4"):
        if re.search(rf"(?:^|[/\-_:]){reasoning_prefix}(?:$|[\-_:])", lowered):
            return reasoning_prefix
    return None


def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    cleaned = value.strip()
    if not cleaned:
        return True
    upper = cleaned.upper()
    return any(upper.startswith(pref) for pref in _PLACEHOLDER_PREFIXES)


class OpenAICompatibleClient:
    """HTTP OpenAI-compatible chat completion client with bounded retries and cost telemetry."""

    def __init__(
        self,
        *,
        model_id: str,
        api_base: str,
        api_key: str,
        pricing_usd_per_1m_tokens: dict[str, Any],
        timeout_sec: float = 90.0,
        provider: str = "openai_compatible",
        model_family: str = "unknown",
        reasoning_effort: str | None = None,
        reasoning_format: str | None = None,
        use_max_completion_tokens: bool | None = None,
        response_format_json_object: bool = False,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
        retry_base_delay_sec: float = DEFAULT_HTTP_RETRY_BASE_DELAY_SEC,
        retry_max_delay_sec: float = DEFAULT_HTTP_RETRY_MAX_DELAY_SEC,
        min_request_interval_sec: float = 0.0,
        tpm_limit: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.pricing = pricing_usd_per_1m_tokens
        self.timeout_sec = timeout_sec
        self.provider = provider
        self.model_family = model_family
        self.reasoning_effort = reasoning_effort.strip().lower() if reasoning_effort else None
        self.reasoning_format = reasoning_format.strip().lower() if reasoning_format else None
        self.use_max_completion_tokens = _should_use_max_completion_tokens(
            model_id, use_max_completion_tokens
        )
        self.response_format_json_object = bool(response_format_json_object)
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay_sec = max(0.0, float(retry_base_delay_sec))
        self.retry_max_delay_sec = max(self.retry_base_delay_sec, float(retry_max_delay_sec))
        self._pacer = _RateLimitPacer(
            min_interval_sec=min_request_interval_sec,
            tpm_limit=tpm_limit,
        )

    def _build_headers(self, *, content_type_json: bool = False) -> dict[str, str]:
        """Build HTTP headers, omitting Authorization for keyless local endpoints."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if content_type_json:
            headers["Content-Type"] = "application/json"
        return headers

    def _secrets_list(self) -> list[str]:
        return [self._api_key] if self._api_key else []

    def verify_available_detailed(self) -> dict[str, Any]:
        """Query GET /models with transient retry support and return metadata/headers."""
        resp, latency_ms, attempts = _execute_http_with_retry(
            method="GET",
            url=f"{self.api_base}/models",
            headers=self._build_headers(),
            timeout=15.0,
            provider=self.provider,
            model_id=self.model_id,
            max_retries=self.max_retries,
            base_delay_sec=self.retry_base_delay_sec,
            max_delay_sec=self.retry_max_delay_sec,
            secrets=self._secrets_list(),
        )
        payload = resp.json()
        items = [item for item in payload.get("data", []) if isinstance(item, dict)]
        available_ids = [str(item.get("id", "")) for item in items]
        if available_ids and self.model_id not in available_ids:
            raise BackendNotConfiguredError(
                f"Configured model {self.model_id!r} not advertised by {self.api_base}/models "
                f"(sample available: {available_ids[:10]})."
            )
        matched = next((item for item in items if str(item.get("id", "")) == self.model_id), None)
        return {
            "available_ids": available_ids,
            "matched_model": matched,
            "rate_limit_headers": _extract_rate_limit_headers(getattr(resp, "headers", None)),
            "latency_ms": round(latency_ms, 2),
            "attempts": attempts,
        }

    def verify_available(self) -> list[str]:
        """Query /models endpoint to confirm reachability and configured model ID existence."""
        return self.verify_available_detailed()["available_ids"]

    def build_request_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Construct an OpenAI Chat Completions payload compliant with reasoning_effort rules."""
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.use_max_completion_tokens:
            body["max_completion_tokens"] = int(max_tokens)
        else:
            body["max_tokens"] = int(max_tokens)
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_format is not None:
            body["reasoning_format"] = self.reasoning_format
        # Per OpenAI GPT-5+ docs, temperature/top_p are supported when reasoning_effort is 'none' or unset
        if self.reasoning_effort in (None, "none"):
            body["temperature"] = temperature
            body["top_p"] = top_p
        if self.response_format_json_object:
            body["response_format"] = {"type": "json_object"}
        if seed is not None:
            body["seed"] = int(seed)
        return body

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        body = self.build_request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        est_tok = max(1, (len(system_prompt) + len(user_prompt)) // 4) + int(max_tokens // 2)
        self._pacer.wait_before_call(est_tok)

        resp, latency_ms, _attempts = _execute_http_with_retry(
            method="POST",
            url=f"{self.api_base}/chat/completions",
            headers=self._build_headers(content_type_json=True),
            json_body=body,
            timeout=self.timeout_sec,
            provider=self.provider,
            model_id=self.model_id,
            max_retries=self.max_retries,
            base_delay_sec=self.retry_base_delay_sec,
            max_delay_sec=self.retry_max_delay_sec,
            secrets=self._secrets_list(),
        )
        data = resp.json()
        content = str(data["choices"][0]["message"]["content"])
        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        self._pacer.record_tokens(in_tok + out_tok)
        details = usage.get("completion_tokens_details") or {}
        reasoning_tok = int(details.get("reasoning_tokens", 0))
        return LLMResponse(
            raw_text=content,
            model_id=self.model_id,
            provider=self.provider,
            model_family=self.model_family,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            cost_usd=compute_cost_usd(in_tok, out_tok, self.pricing),
            reasoning_tokens=reasoning_tok,
            response_model_id=str(data.get("model")) if data.get("model") else None,
            rate_limit_headers=_extract_rate_limit_headers(getattr(resp, "headers", None)) or None,
        )


class GoogleGeminiClient:
    """Google Gemini native Generative Language REST v1beta client with bounded retries."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        pricing_usd_per_1m_tokens: dict[str, Any],
        api_base: str = DEFAULT_GEMINI_API_BASE,
        timeout_sec: float = 90.0,
        provider: str = "google_gemini",
        model_family: str = "gemini",
        thinking_level: str | None = None,
        omit_deprecated_sampling_params: bool | None = None,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
        retry_base_delay_sec: float = DEFAULT_HTTP_RETRY_BASE_DELAY_SEC,
        retry_max_delay_sec: float = DEFAULT_HTTP_RETRY_MAX_DELAY_SEC,
        use_streaming: bool = False,
        min_request_interval_sec: float = 0.0,
        tpm_limit: int | None = None,
    ) -> None:
        self.model_id = model_id.removeprefix("models/").strip()
        self.api_base = (api_base or DEFAULT_GEMINI_API_BASE).rstrip("/")
        self._api_key = api_key
        self.pricing = pricing_usd_per_1m_tokens
        self.timeout_sec = timeout_sec
        self.provider = provider
        self.model_family = model_family
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay_sec = max(0.0, float(retry_base_delay_sec))
        self.retry_max_delay_sec = max(self.retry_base_delay_sec, float(retry_max_delay_sec))
        self.use_streaming = bool(use_streaming)
        self._pacer = _RateLimitPacer(
            min_interval_sec=min_request_interval_sec,
            tpm_limit=tpm_limit,
        )
        self.omit_deprecated_sampling_params = _should_omit_gemini_sampling_params(
            self.model_id, omit_deprecated_sampling_params
        )
        self.thinking_level: str | None = None
        if thinking_level is not None and str(thinking_level).strip():
            norm_lvl = str(thinking_level).strip().upper()
            if "gemini-3.8" in self.model_id.lower() and norm_lvl == "MINIMAL":
                raise BackendNotConfiguredError(
                    "Gemini 3.8 models do not support thinking_level='MINIMAL'; use 'LOW', 'MEDIUM', or 'HIGH'."
                )
            if norm_lvl not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
                raise BackendNotConfiguredError(
                    f"Unsupported Gemini thinking_level={thinking_level!r}; expected LOW, MEDIUM, HIGH (or MINIMAL on supported lite models)."
                )
            self.thinking_level = norm_lvl

    def verify_available_detailed(self) -> dict[str, Any]:
        """Query Gemini GET /models endpoint (paginated, with transient retries) and return metadata."""
        items: list[dict[str, Any]] = []
        headers_out: dict[str, str] = {}
        total_latency_ms = 0.0
        total_attempts = 0
        page_token: str | None = None
        while True:
            resp, page_lat_ms, page_attempts = _execute_http_with_retry(
                method="GET",
                url=f"{self.api_base}/models",
                headers={"x-goog-api-key": self._api_key},
                params={"pageToken": page_token} if page_token else None,
                timeout=15.0,
                provider=self.provider,
                model_id=self.model_id,
                max_retries=self.max_retries,
                base_delay_sec=self.retry_base_delay_sec,
                max_delay_sec=self.retry_max_delay_sec,
                secrets=[self._api_key],
            )
            total_latency_ms += page_lat_ms
            total_attempts += page_attempts
            headers_out.update(_extract_rate_limit_headers(getattr(resp, "headers", None)))
            payload = resp.json()
            for item in payload.get("models", []):
                if isinstance(item, dict):
                    items.append(item)
            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        available_ids = [str(item.get("name", "")).removeprefix("models/") for item in items]
        if available_ids and self.model_id not in available_ids:
            raise BackendNotConfiguredError(
                f"Configured Gemini model {self.model_id!r} not advertised by {self.api_base}/models "
                f"(sample available: {available_ids[:10]})."
            )
        matched = next(
            (item for item in items if str(item.get("name", "")).removeprefix("models/") == self.model_id),
            None,
        )
        return {
            "available_ids": available_ids,
            "matched_model": matched,
            "rate_limit_headers": headers_out,
            "latency_ms": round(total_latency_ms, 2),
            "attempts": total_attempts,
        }

    def verify_available(self) -> list[str]:
        """Query Gemini GET /models endpoint to confirm API access and model availability."""
        return self.verify_available_detailed()["available_ids"]

    def build_request_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Construct the Gemini generateContent JSON request payload compliant with Gemini 3.8+."""
        gen_cfg: dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
        # Gemini 3.6 / 3.8+ deprecate temperature, topP, and topK; do not emit them on those models.
        if not self.omit_deprecated_sampling_params:
            gen_cfg["temperature"] = temperature
            gen_cfg["topP"] = top_p
        if self.thinking_level is not None:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        if seed is not None:
            gen_cfg["seed"] = int(seed)
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        body = self.build_request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        est_tok = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        self._pacer.wait_before_call(est_tok)

        if self.use_streaming:
            resp, latency_ms, _attempts = _execute_http_with_retry(
                method="POST",
                url=f"{self.api_base}/models/{self.model_id}:streamGenerateContent",
                params={"alt": "sse"},
                headers={
                    "x-goog-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json_body=body,
                timeout=self.timeout_sec,
                provider=self.provider,
                model_id=self.model_id,
                max_retries=self.max_retries,
                base_delay_sec=self.retry_base_delay_sec,
                max_delay_sec=self.retry_max_delay_sec,
                secrets=[self._api_key],
                stream=True,
            )
            parts: list[str] = []
            usage: dict[str, Any] = {}
            resp_model_ver: str | None = None
            t_stream_start = time.perf_counter()
            try:
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line_s = str(raw_line).strip()
                    if not line_s.startswith("data:"):
                        continue
                    payload_s = line_s[5:].strip()
                    if not payload_s or payload_s == "[DONE]":
                        continue
                    try:
                        frame = json.loads(payload_s)
                    except json.JSONDecodeError:
                        continue
                    cands = frame.get("candidates") or []
                    if cands:
                        c_obj = cands[0].get("content") or {}
                        for part in c_obj.get("parts") or []:
                            if isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                    if isinstance(frame.get("usageMetadata"), dict):
                        usage = frame["usageMetadata"]
                    if frame.get("modelVersion"):
                        resp_model_ver = str(frame["modelVersion"])
            except Exception as stream_err:
                safe_err = sanitize_sensitive_text(str(stream_err), secrets=[self._api_key])
                raise BackendHTTPRequestError(
                    f"{self.provider}/{self.model_id} stream read failure: {safe_err}",
                    provider=self.provider,
                    model_id=self.model_id,
                    endpoint=f"{self.api_base}/models/{self.model_id}:streamGenerateContent",
                    status_code=None,
                    attempts=1,
                    retryable=True,
                ) from stream_err
            latency_ms = max(latency_ms, (time.perf_counter() - t_stream_start) * 1000.0)
            raw_text = "".join(parts).strip()
            in_tok = int(usage.get("promptTokenCount", 0))
            cand_tok = int(usage.get("candidatesTokenCount", 0))
            thoughts_tok = int(usage.get("thoughtsTokenCount", 0))
            out_tok = cand_tok + thoughts_tok
            self._pacer.record_tokens(in_tok + out_tok)
            return LLMResponse(
                raw_text=raw_text,
                model_id=self.model_id,
                provider=self.provider,
                model_family=self.model_family,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=round(latency_ms, 2),
                cost_usd=compute_cost_usd(in_tok, out_tok, self.pricing),
                thoughts_tokens=thoughts_tok,
                response_model_id=resp_model_ver,
                rate_limit_headers=_extract_rate_limit_headers(getattr(resp, "headers", None)) or None,
            )

        resp, latency_ms, _attempts = _execute_http_with_retry(
            method="POST",
            url=f"{self.api_base}/models/{self.model_id}:generateContent",
            headers={
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            json_body=body,
            timeout=self.timeout_sec,
            provider=self.provider,
            model_id=self.model_id,
            max_retries=self.max_retries,
            base_delay_sec=self.retry_base_delay_sec,
            max_delay_sec=self.retry_max_delay_sec,
            secrets=[self._api_key],
        )
        data = resp.json()

        candidates = data.get("candidates") or []
        parts = []
        if candidates:
            content_obj = candidates[0].get("content") or {}
            for part in content_obj.get("parts") or []:
                if isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
        raw_text = "".join(parts).strip()

        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount", 0))
        cand_tok = int(usage.get("candidatesTokenCount", 0))
        thoughts_tok = int(usage.get("thoughtsTokenCount", 0))
        # Official Gemini API bills response candidates + thinking tokens at output token rate
        out_tok = cand_tok + thoughts_tok
        self._pacer.record_tokens(in_tok + out_tok)
        return LLMResponse(
            raw_text=raw_text,
            model_id=self.model_id,
            provider=self.provider,
            model_family=self.model_family,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            cost_usd=compute_cost_usd(in_tok, out_tok, self.pricing),
            thoughts_tokens=thoughts_tok,
            response_model_id=str(data.get("modelVersion")) if data.get("modelVersion") else None,
            rate_limit_headers=_extract_rate_limit_headers(getattr(resp, "headers", None)) or None,
        )


class MockLLMClient:
    """Deterministic programmable mock LLM backend for offline unit & pipeline integration tests."""

    def __init__(
        self,
        *,
        model_id: str,
        responder: Callable[[str, str, int, float], str],
        pricing_usd_per_1m_tokens: dict[str, Any] | None = None,
        provider: str = "mock_provider",
        model_family: str = "mock_family",
    ) -> None:
        self.model_id = model_id
        self._responder = responder
        self.pricing = pricing_usd_per_1m_tokens or {"input": 0.20, "output": 0.20}
        self.provider = provider
        self.model_family = model_family
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        t0 = time.perf_counter()
        raw_text = self._responder(system_prompt, user_prompt, self.call_count, temperature)
        latency_ms = max(1.0, (time.perf_counter() - t0) * 1000.0)
        in_tok = max(10, len(system_prompt.split()) + len(user_prompt.split()))
        out_tok = max(5, len(raw_text.split()))
        return LLMResponse(
            raw_text=raw_text,
            model_id=self.model_id,
            provider=self.provider,
            model_family=self.model_family,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency_ms, 2),
            cost_usd=compute_cost_usd(in_tok, out_tok, self.pricing),
        )


def _resolve_api_base(b_cfg: dict[str, Any]) -> str:
    api_base_env = str(b_cfg.get("api_base_env", "")).strip()
    env_val = os.getenv(api_base_env, "").strip() if api_base_env else ""
    if env_val:
        return env_val
    return str(b_cfg.get("default_api_base", "")).strip()


def _check_single_backend_config(
    role_name: str,
    b_cfg: dict[str, Any],
    check_live: bool,
    *,
    require_verified_flags: bool = True,
) -> dict[str, Any]:
    provider = str(b_cfg.get("provider", "")).strip()
    model_family = str(b_cfg.get("model_family", "")).strip().lower()
    model_id = str(b_cfg.get("model_id", "")).strip()
    api_base_env = str(b_cfg.get("api_base_env", "")).strip()
    pricing = b_cfg.get("pricing_usd_per_1m_tokens") or {}

    if _is_placeholder(provider):
        raise BackendNotConfiguredError(
            f"{role_name} provider is unverified/unconfigured ({provider!r})."
        )
    if _is_placeholder(model_family):
        raise BackendNotConfiguredError(
            f"{role_name} model_family is unverified/unconfigured ({model_family!r})."
        )
    if _is_placeholder(model_id):
        raise BackendNotConfiguredError(
            f"{role_name} model_id is unverified/unconfigured ({model_id!r}); "
            "explicitly select and verify an available model before live execution."
        )

    inferred_family = infer_model_family_from_id(model_id)
    if inferred_family is not None and inferred_family != model_family:
        raise BackendNotConfiguredError(
            f"{role_name} declared model_family={model_family!r} conflicts with model_id={model_id!r} "
            f"(inferred family: {inferred_family!r})."
        )

    if require_verified_flags:
        if not bool(b_cfg.get("model_availability_verified", False)):
            raise BackendNotConfiguredError(
                f"{role_name} has model_availability_verified=false for {model_id!r}; "
                "verify model availability before live execution."
            )
        if not bool(b_cfg.get("pricing_verified", False)):
            raise BackendNotConfiguredError(
                f"{role_name} has pricing_verified=false for {model_id!r}; "
                "verify per-1M token pricing rates before live execution."
            )

    in_rate = pricing.get("input")
    out_rate = pricing.get("output")
    if in_rate is None or out_rate is None:
        raise BackendNotConfiguredError(
            f"{role_name} pricing_usd_per_1m_tokens must have verified non-null numeric 'input' and 'output' rates."
        )
    if float(in_rate) < 0.0 or float(out_rate) < 0.0:
        raise BackendNotConfiguredError(f"{role_name} token pricing rates must be non-negative.")

    api_key, resolved_key_env = _resolve_api_key(b_cfg)
    api_base = _resolve_api_base(b_cfg)
    if not api_base or (not api_key and not _allows_optional_api_key(b_cfg)):
        raise BackendNotConfiguredError(
            f"{role_name} credentials/endpoint not configured: set {resolved_key_env!r}"
            + (f" and {api_base_env!r}" if not b_cfg.get("default_api_base") else "")
            + " in environment or .env before live execution."
        )

    available_models: list[str] = []
    live_meta: dict[str, Any] = {}
    if check_live:
        client = build_live_client(b_cfg, use_dotenv=False)
        try:
            live_meta = client.verify_available_detailed()
            available_models = live_meta["available_ids"]
        except Exception as exc:
            safe_msg = _redact_secrets(str(exc), [api_key])
            raise BackendNotConfiguredError(
                f"Live availability check failed for {role_name} ({provider}/{model_id} @ {api_base}): {safe_msg}"
            ) from exc

    return {
        "role": role_name,
        "provider": provider,
        "model_family": model_family,
        "model_id": model_id,
        "api_mode": b_cfg.get("api_mode", "openai_compatible"),
        "api_base_resolved": api_base,
        "api_key_env_used": resolved_key_env,
        "pricing_usd_per_1m_tokens": {"input": float(in_rate), "output": float(out_rate)},
        "model_availability_verified": bool(b_cfg.get("model_availability_verified", False)) or bool(check_live),
        "pricing_verified": bool(b_cfg.get("pricing_verified", False)),
        "live_checked": check_live,
        "advertised_models_count": len(available_models),
        "live_models_metadata": live_meta if check_live else None,
        "verified": True,
    }


def _check_cross_judge_independence(
    gen_cfg: dict[str, Any],
    judge_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate model-id, model-family, vendor-lineage, and provider independence between generator and judge.
    """
    gen_provider = str(gen_cfg.get("provider", "")).strip().lower()
    judge_provider = str(judge_cfg.get("provider", "")).strip().lower()
    gen_family = str(gen_cfg.get("model_family", "")).strip().lower()
    judge_family = str(judge_cfg.get("model_family", "")).strip().lower()
    gen_model = str(gen_cfg.get("model_id", "")).strip()
    judge_model = str(judge_cfg.get("model_id", "")).strip()

    if _is_placeholder(gen_provider) or _is_placeholder(judge_provider):
        raise BackendNotConfiguredError(
            "Both synthetic_generator and cross_judge_auditor must specify verified non-placeholder providers."
        )
    if _is_placeholder(gen_family) or _is_placeholder(judge_family):
        raise BackendNotConfiguredError(
            "Both synthetic_generator and cross_judge_auditor must specify verified non-placeholder model_family values."
        )

    if policy_cfg.get("require_distinct_model_id", True) and gen_model == judge_model:
        raise BackendNotConfiguredError(
            f"synthetic_generator and cross_judge_auditor must use distinct model_id values (got {gen_model!r})."
        )

    if policy_cfg.get("require_distinct_model_family", True):
        if gen_family == judge_family:
            raise BackendNotConfiguredError(
                f"Cross-judge family independence violated: both generator and judge belong to model_family={gen_family!r}."
            )
        gen_lineage = FAMILY_VENDOR_LINEAGE.get(gen_family, gen_family)
        judge_lineage = FAMILY_VENDOR_LINEAGE.get(judge_family, judge_family)
        if gen_lineage == judge_lineage:
            raise BackendNotConfiguredError(
                f"Cross-judge vendor-lineage independence violated: {gen_family!r} and {judge_family!r} "
                f"share lineage {gen_lineage!r}."
            )
        for group in policy_cfg.get("disallowed_same_vendor_families", []):
            norm_group = {str(x).strip().lower() for x in group}
            if gen_family in norm_group and judge_family in norm_group:
                raise BackendNotConfiguredError(
                    f"Cross-judge family independence violated: {gen_family!r} and {judge_family!r} "
                    f"are in the same restricted family group {sorted(norm_group)}."
                )

    if policy_cfg.get("require_distinct_provider", True) and gen_provider == judge_provider:
        raise BackendNotConfiguredError(
            f"Cross-judge provider independence violated: both generator and judge use provider={gen_provider!r}."
        )

    return {
        "distinct_model_id": gen_model != judge_model,
        "distinct_model_family": gen_family != judge_family,
        "distinct_vendor_lineage": FAMILY_VENDOR_LINEAGE.get(gen_family, gen_family)
        != FAMILY_VENDOR_LINEAGE.get(judge_family, judge_family),
        "distinct_provider": gen_provider != judge_provider,
        "diagnostic_only_never_gold": bool(policy_cfg.get("diagnostic_only_never_gold", True)),
        "independence_verified": True,
    }


def verify_generation_and_judge_backends(
    cfg: dict[str, Any],
    *,
    check_live: bool = True,
    use_dotenv: bool = True,
) -> dict[str, Any]:
    """
    Verify that both synthetic_generator and cross_judge_auditor are verified, configured with
    distinct providers and distinct model families, carry verified pricing, and respond live.
    """
    if use_dotenv:
        load_local_dotenv()
    backends_cfg = cfg["backends"]
    gen_cfg = backends_cfg["synthetic_generator"]
    judge_cfg = backends_cfg["cross_judge_auditor"]
    policy_cfg = backends_cfg.get("cross_judge_independence_policy", {})

    gen_status = _check_single_backend_config("synthetic_generator", gen_cfg, check_live=False)
    judge_status = _check_single_backend_config("cross_judge_auditor", judge_cfg, check_live=False)
    independence_status = _check_cross_judge_independence(gen_cfg, judge_cfg, policy_cfg)
    if check_live:
        gen_status = _check_single_backend_config("synthetic_generator", gen_cfg, check_live=True)
        judge_status = _check_single_backend_config("cross_judge_auditor", judge_cfg, check_live=True)

    return {
        "verified": True,
        "cross_judge_independence": independence_status,
        "synthetic_generator": gen_status,
        "cross_judge_auditor": judge_status,
    }


def build_live_client(
    b_cfg: dict[str, Any],
    *,
    use_dotenv: bool = True,
) -> GoogleGeminiClient | OpenAICompatibleClient:
    """Construct a GoogleGeminiClient or OpenAICompatibleClient from backend config."""
    if use_dotenv:
        load_local_dotenv()
    provider = str(b_cfg.get("provider", "openai_compatible")).strip().lower()
    api_mode = str(b_cfg.get("api_mode", "")).strip().lower()
    model_family = str(b_cfg.get("model_family", "unknown")).strip().lower()
    api_key, key_env_name = _resolve_api_key(b_cfg)
    api_base = _resolve_api_base(b_cfg)
    if not api_base or (not api_key and not _allows_optional_api_key(b_cfg)):
        raise BackendNotConfiguredError(
            f"Missing environment credentials/endpoint for provider={provider!r}: "
            f"{b_cfg.get('api_base_env')} / {key_env_name}"
        )

    dec = b_cfg.get("decoding") or {}
    max_retries = int(b_cfg.get("http_max_retries", DEFAULT_HTTP_MAX_RETRIES))
    base_delay = float(b_cfg.get("http_retry_base_delay_sec", DEFAULT_HTTP_RETRY_BASE_DELAY_SEC))
    max_delay = float(b_cfg.get("http_retry_max_delay_sec", DEFAULT_HTTP_RETRY_MAX_DELAY_SEC))

    if provider == "google_gemini" and api_mode != "openai_compatible":
        return GoogleGeminiClient(
            model_id=str(b_cfg["model_id"]),
            api_key=api_key,
            api_base=api_base,
            pricing_usd_per_1m_tokens=dict(b_cfg["pricing_usd_per_1m_tokens"]),
            timeout_sec=float(b_cfg.get("request_timeout_sec", 90.0)),
            provider=provider,
            model_family=model_family,
            thinking_level=dec.get("thinking_level"),
            omit_deprecated_sampling_params=dec.get("omit_deprecated_sampling_params"),
            max_retries=max_retries,
            retry_base_delay_sec=base_delay,
            retry_max_delay_sec=max_delay,
            use_streaming=bool(b_cfg.get("use_streaming", False)),
            min_request_interval_sec=float(b_cfg.get("min_request_interval_sec", 0.0)),
            tpm_limit=int(b_cfg["tpm_limit"]) if b_cfg.get("tpm_limit") else None,
        )

    resp_fmt = dec.get("response_format")
    json_mode = bool(dec.get("response_format_json_object", False)) or (
        isinstance(resp_fmt, dict) and str(resp_fmt.get("type", "")).lower() == "json_object"
    )
    return OpenAICompatibleClient(
        model_id=str(b_cfg["model_id"]),
        api_base=api_base,
        api_key=api_key,
        pricing_usd_per_1m_tokens=dict(b_cfg["pricing_usd_per_1m_tokens"]),
        timeout_sec=float(b_cfg.get("request_timeout_sec", 90.0)),
        provider=provider,
        model_family=model_family,
        reasoning_effort=dec.get("reasoning_effort"),
        reasoning_format=dec.get("reasoning_format"),
        use_max_completion_tokens=dec.get("use_max_completion_tokens"),
        response_format_json_object=json_mode,
        max_retries=max_retries,
        retry_base_delay_sec=base_delay,
        retry_max_delay_sec=max_delay,
        min_request_interval_sec=float(b_cfg.get("min_request_interval_sec", 0.0)),
        tpm_limit=int(b_cfg["tpm_limit"]) if b_cfg.get("tpm_limit") else None,
    )


def _parse_json_strict_or_fenced(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    return json.loads(cleaned)


class _SmokeSchemaValidationError(ValueError):
    """Raised when an API call returned HTTP < 400 but failed local JSON, Stage-1, or Pydantic validation."""


def _classify_exception_category(exc: Exception) -> str:
    """Classify probe failures cleanly into provider/API, configuration, budget, or schema validation."""
    from pydantic import ValidationError

    if isinstance(exc, (_SmokeSchemaValidationError, json.JSONDecodeError, ValidationError)):
        return "schema_validation_error"
    http_err = exc if isinstance(exc, BackendHTTPRequestError) else getattr(exc, "__cause__", None)
    if isinstance(http_err, BackendHTTPRequestError):
        return "provider_api_error"
    if isinstance(exc, BackendNotConfiguredError):
        return "configuration_error"
    msg_l = str(exc).lower()
    if "budget guard" in msg_l or "max_pilot_budget_usd" in msg_l:
        return "budget_guard_error"
    return "provider_api_error"


def _extract_exception_info(exc: Exception, secrets: list[str]) -> dict[str, Any]:
    """Convert any HTTP, configuration, or schema exception into a sanitized diagnostic dict."""
    category = _classify_exception_category(exc)
    http_err = exc if isinstance(exc, BackendHTTPRequestError) else getattr(exc, "__cause__", None)
    if isinstance(http_err, BackendHTTPRequestError):
        info = http_err.to_dict()
        info["error_type"] = type(exc).__name__
        info["failure_category"] = category
        return _redact_nested_value(info, secrets)
    return _redact_nested_value(
        {
            "error_type": type(exc).__name__,
            "failure_category": category,
            "status_code": None,
            "retryable": False,
            "error_message": str(exc),
        },
        secrets,
    )


def _build_smoke_source_record(now_iso: str) -> Any:
    """Construct the canonical medical SourceQARecord fixture used by live smoke checks."""
    from src.data.schema import LicenseMetadata, ProvenanceMetadata, SourceQARecord

    return SourceQARecord(
        record_id="smoke_medquad_001",
        source_dataset="medquad",
        source_subset="smoke_verification",
        source_record_id="smoke_001",
        source_document_id=None,
        source_group_id="grp_smoke_medquad_001",
        domain="medical",
        has_context=False,
        question="What is the first-line emergency medication and route for acute anaphylaxis?",
        context=None,
        original_answer=(
            "Intramuscular epinephrine injected into the mid-anterolateral thigh is the first-line "
            "emergency treatment for acute anaphylaxis because alpha-1 vasoconstriction and "
            "beta-2 bronchodilation rapidly reverse airway edema and distributive shock."
        ),
        metadata={"answer_words": 29, "question_words": 12},
        provenance=ProvenanceMetadata(
            source_url="https://smoke.local/medquad",
            upstream_url="https://smoke.local/medquad",
            dataset_version="smoke-v1",
            downloaded_at_utc=now_iso,
            raw_filename="smoke.json",
            raw_file_sha256="0" * 64,
            preprocessing_config_sha256="1" * 64,
        ),
        licensing=LicenseMetadata(
            hf_card_license=None,
            upstream_license="CC-BY-4.0",
            license_status="verified_open",
            redistribution_permitted=True,
            attribution="Smoke verification test item",
            notes="Synthetic smoke record for API verification only.",
        ),
        split="train",
    )


def _probe_single_gemini_smoke(
    *,
    cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
    smoke_src: Any,
    strategy_b_tpl: Any,
    budget_state: dict[str, Any],
    secrets_to_scrub: list[str],
    now_iso: str,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None, GoogleGeminiClient | OpenAICompatibleClient | None]:
    """Independently run Gemini /models availability + single-sample Strategy B generation smoke."""
    from src.data.schema import GenerationAttemptRecord, HALLUCINATION_LABELS
    from src.generation.prompt_builder import render_strategy_b_joint
    from src.generation.validators import validate_candidate_generation

    t0 = time.perf_counter()
    intended_bits = "10000"
    gen_dec = gen_cfg.get("decoding") or {}
    seed = int(cfg.get("random_seed", 42))
    gen_max_tok = min(1024, int(gen_dec.get("max_tokens", 2048)))
    _, key_env_name = _resolve_api_key(gen_cfg)

    section: dict[str, Any] = {
        "passed": False,
        "configured_model_id": str(gen_cfg.get("model_id", "")),
        "provider": str(gen_cfg.get("provider", "")),
        "api_base_resolved": _resolve_api_base(gen_cfg),
        "api_key_env_used": key_env_name,
        "models_check": {"passed": False},
        "generation_check": {"passed": False},
        "stage1_validation": {"passed": False},
        "generation_attempt_record_validated": False,
        "strategy_b_json_schema_validated": False,
        "input_tokens": 0,
        "output_tokens_billed": 0,
        "thoughts_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "total_latency_ms": 0.0,
        "models_endpoint_rate_headers": {},
        "response_rate_limit_headers": {},
        "error": None,
    }
    gen_client: GoogleGeminiClient | OpenAICompatibleClient | None = None
    attempt_params: dict[str, Any] | None = None
    extracted_response: str | None = None

    try:
        gen_avail = _check_single_backend_config(
            "synthetic_generator", gen_cfg, check_live=True, require_verified_flags=False
        )
        live_models = gen_avail.get("live_models_metadata") or {}
        section.update(
            {
                "api_base_resolved": gen_avail["api_base_resolved"],
                "api_key_env_used": gen_avail["api_key_env_used"],
                "advertised_models_count": gen_avail["advertised_models_count"],
                "matched_model_metadata": live_models.get("matched_model"),
                "models_endpoint_rate_headers": live_models.get("rate_limit_headers") or {},
                "models_check": {
                    "passed": True,
                    "advertised_models_count": gen_avail["advertised_models_count"],
                    "latency_ms": live_models.get("latency_ms", 0.0),
                    "attempts": live_models.get("attempts", 1),
                    "rate_limit_headers": live_models.get("rate_limit_headers") or {},
                },
            }
        )

        gen_client = build_live_client(gen_cfg, use_dotenv=False)
        gen_sys, gen_usr = render_strategy_b_joint(
            strategy_b_tpl,
            domain=smoke_src.domain,
            source_dataset=smoke_src.source_dataset,
            question=smoke_src.question,
            has_context=False,
            prompt_context=None,
            original_answer=smoke_src.original_answer,
            intended_bits=intended_bits,
            active_labels=["factual_inconsistency"],
            forbidden_labels=[lbl for lbl in HALLUCINATION_LABELS if lbl != "factual_inconsistency"],
        )
        section["sample_request_payload_keys"] = sorted(
            gen_client.build_request_payload(
                system_prompt=gen_sys,
                user_prompt=gen_usr,
                temperature=float(gen_dec.get("temperature", 0.7)),
                top_p=float(gen_dec.get("top_p", 0.95)),
                max_tokens=gen_max_tok,
                seed=seed,
            ).get("generationConfig", {}).keys()
        )
        if not reserve_budget_for_call(
            budget_state,
            client=gen_client,
            system_prompt=gen_sys,
            user_prompt=gen_usr,
            max_output_tokens=gen_max_tok,
            sample_id="SMOKE_GEMINI_001",
        ):
            raise RuntimeError(budget_state.get("halt_reason") or "Budget guard blocked smoke generator call.")

        gen_resp = gen_client.generate(
            system_prompt=gen_sys,
            user_prompt=gen_usr,
            temperature=float(gen_dec.get("temperature", 0.7)),
            top_p=float(gen_dec.get("top_p", 0.95)),
            max_tokens=gen_max_tok,
            seed=seed,
        )
        record_actual_spend(budget_state, gen_resp.cost_usd)
        # Persist API telemetry immediately so post-HTTP schema rejections retain exact usage & spend.
        section.update(
            {
                "response_model_id": gen_resp.response_model_id,
                "generation_check": {
                    "passed": True,
                    "response_model_id": gen_resp.response_model_id,
                    "latency_ms": gen_resp.latency_ms,
                },
                "input_tokens": gen_resp.input_tokens,
                "output_tokens_billed": gen_resp.output_tokens,
                "thoughts_tokens": gen_resp.thoughts_tokens,
                "cost_usd": gen_resp.cost_usd,
                "latency_ms": gen_resp.latency_ms,
                "response_rate_limit_headers": gen_resp.rate_limit_headers or {},
            }
        )

        gen_parsed = _parse_json_strict_or_fenced(gen_resp.raw_text)
        if (
            not isinstance(gen_parsed, dict)
            or not isinstance(gen_parsed.get("generated_response"), str)
            or not isinstance(gen_parsed.get("per_label_modifications"), dict)
            or "factual_inconsistency" not in gen_parsed["per_label_modifications"]
            or not isinstance(gen_parsed.get("negative_constraints_verified"), str)
        ):
            raise _SmokeSchemaValidationError(
                "Gemini smoke JSON failed canonical Strategy B schema keys."
            )

        stage1_val = validate_candidate_generation(
            raw_output=gen_resp.raw_text,
            response_field="generated_response",
            original_answer=smoke_src.original_answer,
            intended_bits=intended_bits,
            has_context=smoke_src.has_context,
            val_cfg=cfg["automated_validation"],
        )
        section["stage1_validation"] = {
            "passed": stage1_val.passed,
            "rejection_reasons": stage1_val.rejection_reasons,
            "normalized_edit_distance": stage1_val.normalized_edit_distance,
            "length_ratio": stage1_val.length_ratio,
        }
        if not stage1_val.passed:
            raise _SmokeSchemaValidationError(
                f"Gemini smoke candidate failed canonical Stage-1 validator: {stage1_val.rejection_reasons}"
            )

        attempt_params = {
            "generation_provider": gen_client.provider,
            "generation_model_family": gen_client.model_family,
            "thinking_level": getattr(gen_client, "thinking_level", None),
            "max_tokens": gen_max_tok,
        }
        attempt_record = GenerationAttemptRecord(
            attempt_id="SMOKE_SAMPLE_001__att_1",
            sample_id="SMOKE_SAMPLE_001",
            base_item_id="SMOKE_BASE_001",
            source_dataset=smoke_src.source_dataset,
            source_subset=smoke_src.source_subset,
            source_record_id=smoke_src.source_record_id,
            source_document_id=smoke_src.source_document_id,
            source_group_id=smoke_src.source_group_id,
            domain=smoke_src.domain,
            split="train",
            generation_strategy="strategy_b",
            generation_model=gen_client.model_id,
            prompt_version=strategy_b_tpl.prompt_version,
            generation_parameters=attempt_params,
            attempt_number=1,
            intended_vector=intended_bits,
            raw_output=gen_resp.raw_text,
            extracted_response=stage1_val.extracted_response,
            accepted=stage1_val.passed,
            rejection_reasons=stage1_val.rejection_reasons,
            latency_ms=gen_resp.latency_ms,
            input_tokens=gen_resp.input_tokens,
            output_tokens=gen_resp.output_tokens,
            cost_usd=gen_resp.cost_usd,
            timestamp_utc=now_iso,
        )
        extracted_response = stage1_val.extracted_response
        section.update(
            {
                "passed": True,
                "sample_json_parsed": gen_parsed,
                "strategy_b_json_schema_validated": True,
                "generation_attempt_record_validated": attempt_record.accepted,
            }
        )
    except Exception as exc:
        err_info = _extract_exception_info(exc, secrets_to_scrub)
        section["error"] = err_info
        section["failure_category"] = err_info.get("failure_category")
        if err_info.get("failure_category") == "schema_validation_error":
            section["validation_error"] = err_info
        else:
            section["provider_error"] = err_info
            section["response_rate_limit_headers"] = err_info.get("rate_limit_headers") or {}
            section["generation_check"] = {"passed": False, "error": err_info}

    section["total_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return section, extracted_response, attempt_params, gen_client


def _probe_single_openai_smoke(
    *,
    cfg: dict[str, Any],
    judge_cfg: dict[str, Any],
    smoke_src: Any,
    judge_tpl: Any,
    strategy_b_version: str,
    live_generated_response: str | None,
    attempt_params: dict[str, Any] | None,
    budget_state: dict[str, Any],
    secrets_to_scrub: list[str],
) -> tuple[dict[str, Any], OpenAICompatibleClient | GoogleGeminiClient | None]:
    """Independently verify OpenAI /models and cross-judge structured JSON audit (with controlled fixture fallback)."""
    from src.data.schema import HALLUCINATION_LABELS, LabelDictionary, MultiLabelRecord
    from src.generation.prompt_builder import render_cross_judge
    from src.generation.validators import audit_with_cross_judge

    t0 = time.perf_counter()
    judge_dec = judge_cfg.get("decoding") or {}
    seed = int(cfg.get("random_seed", 42))
    judge_max_tok = int(judge_dec.get("max_tokens", 512))
    _, key_env_name = _resolve_api_key(judge_cfg)

    used_controlled_fixture = live_generated_response is None
    if used_controlled_fixture:
        candidate_text = str(CONTROLLED_SMOKE_AUDIT_FIXTURE["candidate_response"])
        intended_bits = str(CONTROLLED_SMOKE_AUDIT_FIXTURE["intended_bits"])
        audit_source = "controlled_audit_fixture"
        fixture_info: dict[str, Any] | None = {
            k: v for k, v in CONTROLLED_SMOKE_AUDIT_FIXTURE.items() if k != "candidate_response"
        }
    else:
        candidate_text = str(live_generated_response)
        intended_bits = "10000"
        audit_source = "live_gemini_smoke_output"
        fixture_info = None

    section: dict[str, Any] = {
        "passed": False,
        "configured_model_id": str(judge_cfg.get("model_id", "")),
        "provider": str(judge_cfg.get("provider", "")),
        "api_base_resolved": _resolve_api_base(judge_cfg),
        "api_key_env_used": key_env_name,
        "audit_input_source": audit_source,
        "used_controlled_fixture": used_controlled_fixture,
        "controlled_fixture": fixture_info,
        "models_check": {"passed": False},
        "audit_check": {"passed": False},
        "cross_judge_json_schema_validated": False,
        "auto_detection_tier_validated": False,
        "multilabel_record_validated": False,
        "multilabel_validation_status": None,
        "predicted_labels_vector": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "total_latency_ms": 0.0,
        "models_endpoint_rate_headers": {},
        "response_rate_limit_headers": {},
        "error": None,
    }
    judge_client: OpenAICompatibleClient | GoogleGeminiClient | None = None

    try:
        judge_avail = _check_single_backend_config(
            "cross_judge_auditor", judge_cfg, check_live=True, require_verified_flags=False
        )
        live_models = judge_avail.get("live_models_metadata") or {}
        section.update(
            {
                "api_base_resolved": judge_avail["api_base_resolved"],
                "api_key_env_used": judge_avail["api_key_env_used"],
                "advertised_models_count": judge_avail["advertised_models_count"],
                "matched_model_metadata": live_models.get("matched_model"),
                "models_endpoint_rate_headers": live_models.get("rate_limit_headers") or {},
                "models_check": {
                    "passed": True,
                    "advertised_models_count": judge_avail["advertised_models_count"],
                    "latency_ms": live_models.get("latency_ms", 0.0),
                    "attempts": live_models.get("attempts", 1),
                    "rate_limit_headers": live_models.get("rate_limit_headers") or {},
                },
            }
        )

        judge_client = build_live_client(judge_cfg, use_dotenv=False)
        j_sys, j_usr = render_cross_judge(
            judge_tpl,
            domain=smoke_src.domain,
            question=smoke_src.question,
            has_context=False,
            prompt_context=None,
            reference_answer=smoke_src.original_answer,
            candidate_response=candidate_text,
        )
        section["sample_request_payload_keys"] = sorted(
            judge_client.build_request_payload(
                system_prompt=j_sys,
                user_prompt=j_usr,
                temperature=float(judge_dec.get("temperature", 0.0)),
                top_p=float(judge_dec.get("top_p", 1.0)),
                max_tokens=judge_max_tok,
                seed=seed,
            ).keys()
        )
        if not reserve_budget_for_call(
            budget_state,
            client=judge_client,
            system_prompt=j_sys,
            user_prompt=j_usr,
            max_output_tokens=judge_max_tok,
            sample_id="SMOKE_JUDGE_001",
        ):
            raise RuntimeError(budget_state.get("halt_reason") or "Budget guard blocked smoke judge call.")

        auto_tier, judge_resp, diag_flags = audit_with_cross_judge(
            judge_client=judge_client,
            judge_template=judge_tpl,
            judge_decoding=judge_dec,
            domain=smoke_src.domain,
            question=smoke_src.question,
            has_context=smoke_src.has_context,
            prompt_context=None,
            reference_answer=smoke_src.original_answer,
            candidate_response=candidate_text,
            intended_bits=intended_bits,
            seed=seed,
        )
        record_actual_spend(budget_state, judge_resp.cost_usd)
        # Persist API telemetry immediately so post-HTTP schema rejections retain exact usage & spend.
        section.update(
            {
                "response_model_id": judge_resp.response_model_id,
                "diagnostic_flags": diag_flags,
                "auto_detection_tier_validated": auto_tier.predicted_labels is not None,
                "predicted_labels_vector": auto_tier.predicted_labels,
                "audit_check": {
                    "passed": True,
                    "response_model_id": judge_resp.response_model_id,
                    "latency_ms": judge_resp.latency_ms,
                    "audit_input_source": audit_source,
                    "used_controlled_fixture": used_controlled_fixture,
                },
                "input_tokens": judge_resp.input_tokens,
                "output_tokens": judge_resp.output_tokens,
                "reasoning_tokens": judge_resp.reasoning_tokens,
                "cost_usd": judge_resp.cost_usd,
                "latency_ms": judge_resp.latency_ms,
                "response_rate_limit_headers": judge_resp.rate_limit_headers or {},
            }
        )

        judge_parsed = _parse_json_strict_or_fenced(judge_resp.raw_text)
        labels_obj = judge_parsed.get("labels") if isinstance(judge_parsed, dict) else None
        evidence_obj = judge_parsed.get("evidence") if isinstance(judge_parsed, dict) else None
        coh_raw = judge_parsed.get("coherence_score_1_to_5") if isinstance(judge_parsed, dict) else None
        if (
            not isinstance(labels_obj, dict)
            or not set(HALLUCINATION_LABELS).issubset(labels_obj.keys())
            or any(labels_obj[k] not in (0, 1) for k in HALLUCINATION_LABELS)
            or not isinstance(evidence_obj, dict)
            or not set(HALLUCINATION_LABELS).issubset(evidence_obj.keys())
            or not isinstance(coh_raw, (int, float))
            or not (1.0 <= float(coh_raw) <= 5.0)
            or not isinstance(judge_parsed.get("diagnostic_notes"), str)
        ):
            raise _SmokeSchemaValidationError(
                "Judge smoke JSON failed canonical cross_judge_auditor schema keys."
            )

        intended_dict = LabelDictionary(
            factual_inconsistency=1,
            factual_fabrication=0,
            instruction_inconsistency=0,
            logical_inconsistency=0,
            context_inconsistency=0,
        )
        gen_method = "controlled_fixture_smoke" if used_controlled_fixture else "strategy_b"
        gen_model_label = (
            str(CONTROLLED_SMOKE_AUDIT_FIXTURE["fixture_id"])
            if used_controlled_fixture
            else str(cfg["backends"]["synthetic_generator"]["model_id"])
        )
        gen_params = dict(attempt_params or {})
        if used_controlled_fixture:
            gen_params["controlled_audit_fixture"] = fixture_info
        multilabel_record = MultiLabelRecord(
            sample_id="SMOKE_FIXTURE_AUDIT_001" if used_controlled_fixture else "SMOKE_SAMPLE_001",
            source_dataset=smoke_src.source_dataset,
            source_subset=smoke_src.source_subset,
            source_record_id=smoke_src.source_record_id,
            source_document_id=smoke_src.source_document_id,
            source_group_id=smoke_src.source_group_id,
            domain=smoke_src.domain,
            data_origin="synthetic",
            has_context=smoke_src.has_context,
            question=smoke_src.question,
            context=smoke_src.context,
            original_answer=smoke_src.original_answer,
            generated_answer=candidate_text,
            generation_method=gen_method,
            generation_strategy=gen_method,
            generation_model=gen_model_label,
            prompt_version=strategy_b_version,
            generation_parameters=gen_params,
            intended_labels=intended_dict,
            auto_detected_labels=auto_tier,
            label_vector=intended_dict.to_vector(),
            validation_status=(
                "auto_flagged"
                if (auto_tier.target_missed or auto_tier.spillover_flagged or diag_flags)
                else "auto_passed"
            ),
            rejection_reasons=diag_flags,
            attempted_count=1,
            licensing=smoke_src.licensing,
            split="train",
        )
        section.update(
            {
                "passed": True,
                "sample_json_parsed": judge_parsed,
                "cross_judge_json_schema_validated": True,
                "multilabel_record_validated": True,
                "multilabel_validation_status": multilabel_record.validation_status,
            }
        )
    except Exception as exc:
        err_info = _extract_exception_info(exc, secrets_to_scrub)
        section["error"] = err_info
        section["failure_category"] = err_info.get("failure_category")
        if err_info.get("failure_category") == "schema_validation_error":
            section["validation_error"] = err_info
        else:
            section["provider_error"] = err_info
            section["response_rate_limit_headers"] = err_info.get("rate_limit_headers") or {}
            section["audit_check"] = {"passed": False, "error": err_info}

    section["total_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return section, judge_client


def _run_budget_boundary_probe(
    cfg: dict[str, Any],
    budget_state: dict[str, Any],
    client_or_pricing: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Verify the $10 runtime budget guard always blocks an over-budget request prior to API execution."""
    budget_cap = float(budget_state["max_pilot_budget_usd"])
    strict_stop = bool(budget_state["enforce_strict_budget_stop"])
    boundary_state = init_budget_state(cfg)
    est_next = estimate_worst_case_call_cost_usd(
        client_or_pricing, system_prompt, user_prompt, max_output_tokens
    )
    # On $0.00 free-tier pricing (`est_next == 0`), set spent == cap so `spent >= cap` still exercises the guard.
    boundary_state["cumulative_cost_usd"] = round(budget_cap - 1e-6, 6) if est_next > 0.0 else budget_cap
    boundary_allowed = reserve_budget_for_call(
        boundary_state,
        client=client_or_pricing,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_output_tokens=max_output_tokens,
        sample_id="BUDGET_GUARD_BOUNDARY_PROBE",
    )
    cap_ok = budget_cap <= 10.0 and strict_stop
    rejected_ok = (not boundary_allowed) and bool(boundary_state["halted_safely"]) and (
        int(boundary_state["blocked_calls_prevented"]) == 1
    )
    return {
        "passed": bool(cap_ok and rejected_ok),
        "max_pilot_budget_usd": budget_cap,
        "enforce_strict_budget_stop": strict_stop,
        "live_smoke_spend_usd": float(budget_state["cumulative_cost_usd"]),
        "live_smoke_remaining_budget_usd": round(budget_cap - float(budget_state["cumulative_cost_usd"]), 8),
        "boundary_over_cap_halted_safely": bool(boundary_state["halted_safely"]),
        "boundary_blocked_calls_prevented": int(boundary_state["blocked_calls_prevented"]),
        "boundary_request_rejected_before_api_call": not boundary_allowed,
        "boundary_halt_reason": boundary_state["halt_reason"],
    }


def _record_probe_section_checks(
    *,
    role_key: str,
    prefix: str,
    api_check_key: str,
    api_passed: bool,
    section: dict[str, Any],
    successful_checks: list[str],
    failed_checks: list[str],
    provider_errors: dict[str, Any],
    validation_errors: dict[str, Any],
) -> None:
    """Classify one backend smoke section into models, API, schema checks and error buckets."""
    models_check_name = f"{prefix}_models_availability"
    schema_check_name = f"{prefix}_canonical_schema_validation"
    if section["models_check"]["passed"]:
        successful_checks.append(models_check_name)
    else:
        failed_checks.append(models_check_name)

    if api_passed:
        successful_checks.append(api_check_key)
    else:
        failed_checks.append(api_check_key)

    if section["passed"]:
        successful_checks.append(schema_check_name)
    elif api_passed:
        failed_checks.append(schema_check_name)

    err_obj = section.get("error")
    if err_obj:
        if section.get("failure_category") == "schema_validation_error":
            validation_errors[role_key] = err_obj
        else:
            provider_errors[role_key] = err_obj


def probe_live_backends(
    cfg: dict[str, Any],
    *,
    use_dotenv: bool = True,
    write_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Independently verify Gemini and OpenAI backends (GET /models, structured JSON, canonical
    schema validation, and $10 budget guard), falling back to a controlled audit fixture for OpenAI
    if Gemini fails, and always writing a sanitized report without running Pilot E0.
    """
    from src.generation.prompt_builder import load_prompt_template
    from src.utils.io import utc_now_iso

    probe_t0 = time.perf_counter()
    if use_dotenv:
        load_local_dotenv()

    backends_cfg = cfg["backends"]
    gen_cfg = backends_cfg["synthetic_generator"]
    judge_cfg = backends_cfg["cross_judge_auditor"]
    policy_cfg = backends_cfg.get("cross_judge_independence_policy", {})

    gen_secret, _ = _resolve_api_key(gen_cfg)
    judge_secret, _ = _resolve_api_key(judge_cfg)
    secrets_to_scrub = [s for s in (gen_secret, judge_secret) if s]
    now_iso = utc_now_iso()

    successful_checks: list[str] = []
    failed_checks: list[str] = []
    provider_errors: dict[str, Any] = {}
    validation_errors: dict[str, Any] = {}

    try:
        independence = _check_cross_judge_independence(gen_cfg, judge_cfg, policy_cfg)
        successful_checks.append("cross_judge_independence")
    except Exception as exc:
        independence = {"independence_verified": False, "error": _extract_exception_info(exc, secrets_to_scrub)}
        failed_checks.append("cross_judge_independence")

    budget_state = init_budget_state(cfg)
    smoke_src = _build_smoke_source_record(now_iso)
    strategy_b_tpl = load_prompt_template(cfg["prompts"]["strategy_b_joint"])
    judge_tpl = load_prompt_template(cfg["prompts"]["cross_judge_auditor"])

    gen_section, extracted_response, attempt_params, gen_client = _probe_single_gemini_smoke(
        cfg=cfg,
        gen_cfg=gen_cfg,
        smoke_src=smoke_src,
        strategy_b_tpl=strategy_b_tpl,
        budget_state=budget_state,
        secrets_to_scrub=secrets_to_scrub,
        now_iso=now_iso,
    )
    _record_probe_section_checks(
        role_key="synthetic_generator",
        prefix="gemini",
        api_check_key="gemini_structured_json_generation",
        api_passed=bool(gen_section["generation_check"]["passed"]),
        section=gen_section,
        successful_checks=successful_checks,
        failed_checks=failed_checks,
        provider_errors=provider_errors,
        validation_errors=validation_errors,
    )

    judge_section, judge_client = _probe_single_openai_smoke(
        cfg=cfg,
        judge_cfg=judge_cfg,
        smoke_src=smoke_src,
        judge_tpl=judge_tpl,
        strategy_b_version=strategy_b_tpl.prompt_version,
        live_generated_response=extracted_response,
        attempt_params=attempt_params,
        budget_state=budget_state,
        secrets_to_scrub=secrets_to_scrub,
    )
    _record_probe_section_checks(
        role_key="cross_judge_auditor",
        prefix="openai",
        api_check_key="openai_structured_json_audit",
        api_passed=bool(judge_section["audit_check"]["passed"]),
        section=judge_section,
        successful_checks=successful_checks,
        failed_checks=failed_checks,
        provider_errors=provider_errors,
        validation_errors=validation_errors,
    )

    # Always execute $10 budget-guard over-cap boundary check even when a provider failed
    pricing_proxy = gen_client or judge_client or type(
        "_PricingProxy",
        (),
        {"pricing": dict(gen_cfg.get("pricing_usd_per_1m_tokens") or {"input": 0.75, "output": 3.75})},
    )()
    budget_guard_verification = _run_budget_boundary_probe(
        cfg,
        budget_state,
        pricing_proxy,
        system_prompt="Budget guard verification system prompt",
        user_prompt="Budget guard verification user prompt",
        max_output_tokens=min(1024, int((gen_cfg.get("decoding") or {}).get("max_tokens", 2048))),
    )
    if budget_guard_verification["passed"]:
        successful_checks.append("budget_guard_over_cap_boundary")
    else:
        failed_checks.append("budget_guard_over_cap_boundary")

    if not failed_checks:
        overall_status = "passed"
    elif gen_section["passed"] or judge_section["passed"]:
        overall_status = "partial_failure"
    else:
        overall_status = "failed"

    total_probe_ms = round((time.perf_counter() - probe_t0) * 1000.0, 2)
    report: dict[str, Any] = {
        "verification_timestamp_utc": now_iso,
        "overall_status": overall_status,
        "api_keys_redacted": True,
        "successful_checks": successful_checks,
        "failed_checks": failed_checks,
        "provider_errors": provider_errors,
        "validation_errors": validation_errors,
        "cross_judge_independence": independence,
        "retry_policy": {
            "transient_http_status_codes": sorted(TRANSIENT_HTTP_STATUS_CODES),
            "synthetic_generator": {
                "max_retries": int(gen_cfg.get("http_max_retries", DEFAULT_HTTP_MAX_RETRIES)),
                "retry_base_delay_sec": float(
                    gen_cfg.get("http_retry_base_delay_sec", DEFAULT_HTTP_RETRY_BASE_DELAY_SEC)
                ),
                "retry_max_delay_sec": float(
                    gen_cfg.get("http_retry_max_delay_sec", DEFAULT_HTTP_RETRY_MAX_DELAY_SEC)
                ),
            },
            "cross_judge_auditor": {
                "max_retries": int(judge_cfg.get("http_max_retries", DEFAULT_HTTP_MAX_RETRIES)),
                "retry_base_delay_sec": float(
                    judge_cfg.get("http_retry_base_delay_sec", DEFAULT_HTTP_RETRY_BASE_DELAY_SEC)
                ),
                "retry_max_delay_sec": float(
                    judge_cfg.get("http_retry_max_delay_sec", DEFAULT_HTTP_RETRY_MAX_DELAY_SEC)
                ),
            },
        },
        "synthetic_generator": gen_section,
        "cross_judge_auditor": judge_section,
        "canonical_schema_validation": {
            "source_qa_record_validated": True,
            "strategy_b_json_schema_validated": gen_section["strategy_b_json_schema_validated"],
            "generation_attempt_record_validated": gen_section["generation_attempt_record_validated"],
            "cross_judge_json_schema_validated": judge_section["cross_judge_json_schema_validated"],
            "auto_detection_tier_validated": judge_section["auto_detection_tier_validated"],
            "multilabel_record_validated": judge_section["multilabel_record_validated"],
            "multilabel_validation_status": judge_section["multilabel_validation_status"],
            "predicted_labels_vector": judge_section["predicted_labels_vector"],
        },
        "budget_guard_verification": budget_guard_verification,
        "timings_ms": {
            "total_probe_ms": total_probe_ms,
            "gemini_total_ms": gen_section["total_latency_ms"],
            "openai_total_ms": judge_section["total_latency_ms"],
        },
        "rate_limit_summary": {
            "synthetic_generator": {
                "models_endpoint": gen_section["models_endpoint_rate_headers"],
                "inference_endpoint": gen_section["response_rate_limit_headers"],
            },
            "cross_judge_auditor": {
                "models_endpoint": judge_section["models_endpoint_rate_headers"],
                "inference_endpoint": judge_section["response_rate_limit_headers"],
            },
        },
        "total_smoke_cost_usd": round(
            float(gen_section["cost_usd"]) + float(judge_section["cost_usd"]), 8
        ),
    }

    serialized = _redact_secrets(json.dumps(report, indent=2, ensure_ascii=False), secrets_to_scrub)
    redacted_report = json.loads(serialized)
    if write_report_path is not None:
        out_p = Path(write_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(serialized, encoding="utf-8")
    return redacted_report
