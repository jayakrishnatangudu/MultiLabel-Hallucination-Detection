"""Field-aware token budgeting and sliding-window context chunking for multi-label hallucination detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Protocol


DEFAULT_SPECIAL_TOKENS_OVERHEAD = 4
DEFAULT_MAX_LENGTH = 512
DEFAULT_RESPONSE_BUDGET = 288
DEFAULT_QUESTION_BUDGET = 96
DEFAULT_MIN_QUESTION_BUDGET = 32
DEFAULT_SLIDING_WINDOW_STRIDE = 128

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class TokenizerLike(Protocol):
    """Minimal HuggingFace-compatible tokenizer interface used by TokenBudgetAllocator."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        ...


class FallbackWhitespaceTokenizer:
    """
    Deterministic regex-based tokenizer used when HF AutoTokenizer is unavailable locally.

    Preserves exact piece count parity and lossless/clean string reconstruction for unit tests
    and TF-IDF / lightweight inspection paths without downloading remote HuggingFace weights.
    """

    def __init__(self) -> None:
        self._id_to_piece: dict[int, str] = {}
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(str(text or ""))

    def _piece_to_id(self, piece: str) -> int:
        # Reserve low IDs 0..999 for special tokens
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest()
        tid = 1000 + (int.from_bytes(digest, "big") % 100_000_000)
        # Disambiguate rare hash collisions within session
        while tid in self._id_to_piece and self._id_to_piece[tid] != piece:
            tid += 1
        self._id_to_piece[tid] = piece
        return tid

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._piece_to_id(tok) for tok in self.tokenize(text)]
        if add_special_tokens:
            return [self.cls_token_id, *ids, self.sep_token_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        pieces: list[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in specials:
                continue
            piece = self._id_to_piece.get(int(tid))
            if piece is not None:
                pieces.append(piece)
        out = ""
        for p in pieces:
            if not out:
                out = p
            elif re.match(r"^[,.;:!?%)\]}]+$", p):
                out += p
            elif out.endswith(("(", "[", "{", "/", "-")):
                out += p
            else:
                out += " " + p
        return out


@dataclass(frozen=True)
class TokenBudgetMetadata:
    """Mandatory per-sample truncation telemetry required by Section 4.1 of architecture.md."""

    question_tokens_orig: int
    question_tokens_kept: int
    question_truncated: bool
    response_tokens_orig: int
    response_tokens_kept: int
    response_truncated: bool
    context_tokens_orig: int
    context_tokens_kept: int
    context_truncated: bool
    total_sequence_tokens: int
    max_length_limit: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetedTokenizedSample:
    """Encoded `(question, response, context)` sequence with field-level truncation telemetry."""

    input_ids: list[int]
    attention_mask: list[int]
    formatted_text: str
    question_text_kept: str
    response_text_kept: str
    context_text_kept: str
    metadata: TokenBudgetMetadata
    chunk_index: int = 0
    total_chunks: int = 1


class TokenBudgetAllocator:
    """
    Field-aware encoder token budgeter that strictly protects candidate responses before context.

    Follows architecture.md Section 4.1 priority rules:
      1. Candidate Response (up to response_budget, or expanding into unused context capacity)
      2. Question Header (up to remaining space or question_budget; never below min_question_budget if possible)
      3. Reference Context (allocated whatever token budget remains up to max_length)
    """

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        response_budget: int = DEFAULT_RESPONSE_BUDGET,
        question_budget: int = DEFAULT_QUESTION_BUDGET,
        min_question_budget: int = DEFAULT_MIN_QUESTION_BUDGET,
        special_token_overhead: int = DEFAULT_SPECIAL_TOKENS_OVERHEAD,
    ) -> None:
        if max_length < 32:
            raise ValueError(f"max_length must be >= 32, got {max_length}")
        self.tokenizer: TokenizerLike = tokenizer or FallbackWhitespaceTokenizer()
        self.max_length = int(max_length)
        self.response_budget = int(response_budget)
        self.question_budget = int(question_budget)
        self.min_question_budget = int(min_question_budget)
        self.special_token_overhead = max(2, int(special_token_overhead))

        self.cls_id = int(getattr(self.tokenizer, "cls_token_id", None) or getattr(self.tokenizer, "bos_token_id", None) or 101)
        self.sep_id = int(getattr(self.tokenizer, "sep_token_id", None) or getattr(self.tokenizer, "eos_token_id", None) or 102)

        self._q_prefix_ids = self.tokenizer.encode("Question: ", add_special_tokens=False)
        self._r_prefix_ids = self.tokenizer.encode("\nResponse: ", add_special_tokens=False)
        self._c_prefix_ids = self.tokenizer.encode("\nContext: ", add_special_tokens=False)
        self._none_ctx_ids = self.tokenizer.encode("NONE", add_special_tokens=False)
        self._structural_ids_count = len(self._q_prefix_ids) + len(self._r_prefix_ids) + len(self._c_prefix_ids)

    @property
    def content_capacity(self) -> int:
        """Maximum content tokens available after [CLS]/[SEP] and field labels."""
        return max(12, self.max_length - 2 - self._structural_ids_count - max(0, self.special_token_overhead - 2))

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode_plain(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return str(self.tokenizer.decode(ids, skip_special_tokens=True)).strip()

    def _compute_field_allocations(
        self,
        q_len: int,
        r_len: int,
        c_len: int,
    ) -> tuple[int, int, int]:
        """Compute (q_keep, r_keep, c_keep) honoring response-first priority."""
        cap = self.content_capacity
        q_floor = min(q_len, min(self.min_question_budget, max(4, cap // 4)))

        # Priority 1: Candidate response gets up to response_budget, OR expands when context is small/absent
        if c_len == 0:
            max_resp_room = max(0, cap - len(self._none_ctx_ids) - min(q_len, self.question_budget))
            r_target = min(r_len, max(self.response_budget, max_resp_room))
        else:
            # Reserve at least q_floor for question header even on extreme response overflows
            r_target = min(r_len, max(0, cap - q_floor))
            if r_len <= self.response_budget:
                r_target = r_len
            elif r_len + min(q_len, self.question_budget) + c_len <= cap:
                r_target = r_len
            else:
                # Response exceeds nominal response_budget alongside non-empty context:
                # only allow exceeding response_budget if context & question fit, or up to cap - q_floor
                rem_after_q_and_ctx = cap - min(q_len, self.question_budget) - c_len
                r_target = min(r_len, max(min(r_len, cap - q_floor), rem_after_q_and_ctx))
                # Ensure if context is non-empty we only cap response after cap - q_floor
                r_target = min(r_target, max(0, cap - q_floor))

        rem_after_r = max(0, cap - r_target)

        # Priority 2: Question allocation (preserve short questions fully; expand if context small)
        if c_len == 0:
            q_max = max(0, rem_after_r - len(self._none_ctx_ids))
        else:
            q_max = min(rem_after_r, max(self.question_budget, rem_after_r - c_len))
        q_target = min(q_len, max(q_floor, q_max)) if rem_after_r >= q_floor else min(q_len, rem_after_r)
        # Reconcile if r_target + q_target slightly pushed past cap
        if r_target + q_target > cap:
            q_target = min(q_len, max(0, cap - r_target))

        # Priority 3: Context allocation gets remaining budget
        rem_for_c = max(0, cap - r_target - q_target)
        c_target = min(c_len, rem_for_c) if c_len > 0 else 0
        return q_target, r_target, c_target

    def _assemble_sample(
        self,
        *,
        q_ids_orig: list[int],
        r_ids_orig: list[int],
        c_ids_orig: list[int],
        q_kept_ids: list[int],
        r_kept_ids: list[int],
        c_kept_ids: list[int],
        has_context: bool,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> BudgetedTokenizedSample:
        ctx_body = c_kept_ids if (has_context and c_kept_ids) else (self._none_ctx_ids if not has_context else c_kept_ids)
        # Safety trim on contextual NONE sentinel if sequence is tiny
        max_body = self.max_length - 2 - self._structural_ids_count
        seq_body = [*q_kept_ids, *r_kept_ids, *ctx_body]
        if len(seq_body) > max_body and ctx_body == self._none_ctx_ids:
            ctx_body = ctx_body[: max(0, max_body - len(q_kept_ids) - len(r_kept_ids))]

        input_ids = [
            self.cls_id,
            *self._q_prefix_ids,
            *q_kept_ids,
            *self._r_prefix_ids,
            *r_kept_ids,
            *self._c_prefix_ids,
            *ctx_body,
            self.sep_id,
        ]
        # Hard guard so model_max_length is never exceeded under any tokenizer quirk
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length - 1] + [self.sep_id]

        attention_mask = [1] * len(input_ids)
        q_kept_str = self._decode_plain(q_kept_ids)
        r_kept_str = self._decode_plain(r_kept_ids)
        c_kept_str = self._decode_plain(c_kept_ids) if c_kept_ids else ("NONE" if not has_context else "")
        formatted = f"Question: {q_kept_str}\nResponse: {r_kept_str}\nContext: {c_kept_str}"

        meta = TokenBudgetMetadata(
            question_tokens_orig=len(q_ids_orig),
            question_tokens_kept=len(q_kept_ids),
            question_truncated=len(q_kept_ids) < len(q_ids_orig),
            response_tokens_orig=len(r_ids_orig),
            response_tokens_kept=len(r_kept_ids),
            response_truncated=len(r_kept_ids) < len(r_ids_orig),
            context_tokens_orig=len(c_ids_orig),
            context_tokens_kept=len(c_kept_ids),
            context_truncated=len(c_kept_ids) < len(c_ids_orig),
            total_sequence_tokens=len(input_ids),
            max_length_limit=self.max_length,
        )
        return BudgetedTokenizedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            formatted_text=formatted,
            question_text_kept=q_kept_str,
            response_text_kept=r_kept_str,
            context_text_kept=c_kept_str,
            metadata=meta,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

    def encode_sample(
        self,
        *,
        question: str,
        response: str,
        context: str | None = None,
    ) -> BudgetedTokenizedSample:
        """Tokenize `(question, response, context)` under field-aware budget rules."""
        q_ids = self._encode_plain(question)
        r_ids = self._encode_plain(response)
        c_clean = (context or "").strip()
        has_context = bool(c_clean)
        c_ids = self._encode_plain(c_clean) if has_context else []

        q_k, r_k, c_k = self._compute_field_allocations(len(q_ids), len(r_ids), len(c_ids))
        return self._assemble_sample(
            q_ids_orig=q_ids,
            r_ids_orig=r_ids,
            c_ids_orig=c_ids,
            q_kept_ids=q_ids[:q_k],
            r_kept_ids=r_ids[:r_k],
            c_kept_ids=c_ids[:c_k],
            has_context=has_context,
        )

    def build_sliding_window_chunks(
        self,
        *,
        question: str,
        response: str,
        context: str | None = None,
        stride: int = DEFAULT_SLIDING_WINDOW_STRIDE,
        max_chunks: int = 16,
    ) -> list[BudgetedTokenizedSample]:
        """
        Split overflowing context into overlapping chunks while repeating full `(question, response)`.

        Implements Suite A4 sliding-window aggregation input construction so zero context
        evidence is silently lost inside 512-token encoders.
        """
        q_ids = self._encode_plain(question)
        r_ids = self._encode_plain(response)
        c_clean = (context or "").strip()
        has_context = bool(c_clean)
        c_ids = self._encode_plain(c_clean) if has_context else []

        q_k, r_k, c_k = self._compute_field_allocations(len(q_ids), len(r_ids), len(c_ids))
        q_kept = q_ids[:q_k]
        r_kept = r_ids[:r_k]

        if not has_context or len(c_ids) <= c_k or c_k <= 0:
            single = self._assemble_sample(
                q_ids_orig=q_ids,
                r_ids_orig=r_ids,
                c_ids_orig=c_ids,
                q_kept_ids=q_kept,
                r_kept_ids=r_kept,
                c_kept_ids=c_ids[:c_k],
                has_context=has_context,
                chunk_index=0,
                total_chunks=1,
            )
            return [single]

        window_size = max(16, c_k)
        step = max(8, min(int(stride), window_size))
        slices: list[list[int]] = []
        pos = 0
        while pos < len(c_ids) and len(slices) < max_chunks:
            slices.append(c_ids[pos : pos + window_size])
            if pos + window_size >= len(c_ids):
                break
            pos += step

        total_chunks = len(slices)
        chunks: list[BudgetedTokenizedSample] = []
        for idx, c_slice in enumerate(slices):
            chunks.append(
                self._assemble_sample(
                    q_ids_orig=q_ids,
                    r_ids_orig=r_ids,
                    c_ids_orig=c_ids,
                    q_kept_ids=q_kept,
                    r_kept_ids=r_kept,
                    c_kept_ids=c_slice,
                    has_context=True,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                )
            )
        return chunks


def summarize_truncation_rates(metadata_list: list[TokenBudgetMetadata | dict[str, Any]]) -> dict[str, Any]:
    """Compute portfolio truncation rates across a split for mandatory experiment logging."""
    if not metadata_list:
        return {
            "count": 0,
            "question_truncated_rate": 0.0,
            "response_truncated_rate": 0.0,
            "context_truncated_rate": 0.0,
            "mean_sequence_tokens": 0.0,
        }
    rows = [m.to_dict() if isinstance(m, TokenBudgetMetadata) else dict(m) for m in metadata_list]
    n = len(rows)
    return {
        "count": n,
        "question_truncated_rate": round(sum(1 for r in rows if r.get("question_truncated")) / n, 6),
        "response_truncated_rate": round(sum(1 for r in rows if r.get("response_truncated")) / n, 6),
        "context_truncated_rate": round(sum(1 for r in rows if r.get("context_truncated")) / n, 6),
        "mean_sequence_tokens": round(sum(float(r.get("total_sequence_tokens", 0)) for r in rows) / n, 2),
    }


def export_split_token_budget_report(
    split_paths: dict[str, str | Any],
    *,
    max_lengths: list[int] | None = None,
    output_json: str | Any | None = None,
    max_records_per_split: int | None = None,
) -> dict[str, Any]:
    """
    Empirically measure TokenBudgetAllocator truncation rates on real Phase 1 JSONL splits.

    Reports overall split truncation rates plus per-source_dataset breakdown at each
    requested max_length (default [512, 2048]) using FallbackWhitespaceTokenizer.
    """
    from collections import defaultdict
    from pathlib import Path

    from src.utils.config import resolve_project_path
    from src.utils.io import read_jsonl, utc_now_iso, write_json

    lengths = [int(x) for x in (max_lengths or [512, 2048])]
    allocators = {max_len: TokenBudgetAllocator(max_length=max_len) for max_len in lengths}
    split_reports: dict[str, Any] = {}

    for split_name, raw_path in split_paths.items():
        path = resolve_project_path(raw_path)
        rows = read_jsonl(path)
        if max_records_per_split is not None:
            rows = rows[: int(max_records_per_split)]

        # Pre-compute piece counts once per record across all requested max_lengths
        tokenized_lengths: list[tuple[int, int, int, bool, str]] = []
        for row in rows:
            q_len = len(_TOKEN_RE.findall(str(row.get("question", ""))))
            r_len = len(_TOKEN_RE.findall(str(row.get("original_answer", row.get("generated_answer", "")))))
            c_clean = str(row.get("context") or "").strip() if row.get("has_context", bool(row.get("context"))) else ""
            has_context = bool(c_clean)
            c_len = len(_TOKEN_RE.findall(c_clean)) if has_context else 0
            ds = str(row.get("source_dataset", "unknown"))
            tokenized_lengths.append((q_len, r_len, c_len, has_context, ds))

        by_length: dict[str, Any] = {}
        for max_len, allocator in allocators.items():
            all_meta: list[TokenBudgetMetadata] = []
            per_dataset_meta: dict[str, list[TokenBudgetMetadata]] = defaultdict(list)
            chunk_counts: list[int] = []
            max_body = allocator.max_length - 2 - allocator._structural_ids_count
            none_len = len(allocator._none_ctx_ids)

            for q_len, r_len, c_len, has_context, ds in tokenized_lengths:
                q_k, r_k, c_k = allocator._compute_field_allocations(q_len, r_len, c_len)
                if has_context and c_k:
                    ctx_body_len = c_k
                elif not has_context:
                    ctx_body_len = min(none_len, max(0, max_body - q_k - r_k))
                else:
                    ctx_body_len = c_k
                total_seq = min(allocator.max_length, 2 + allocator._structural_ids_count + q_k + r_k + ctx_body_len)
                meta = TokenBudgetMetadata(
                    question_tokens_orig=q_len,
                    question_tokens_kept=q_k,
                    question_truncated=q_k < q_len,
                    response_tokens_orig=r_len,
                    response_tokens_kept=r_k,
                    response_truncated=r_k < r_len,
                    context_tokens_orig=c_len,
                    context_tokens_kept=c_k,
                    context_truncated=c_k < c_len,
                    total_sequence_tokens=total_seq,
                    max_length_limit=allocator.max_length,
                )
                all_meta.append(meta)
                per_dataset_meta[ds].append(meta)
                if max_len == 512:
                    if not has_context or c_len <= c_k or c_k <= 0:
                        chunk_counts.append(1)
                    else:
                        window_size = max(16, c_k)
                        step = max(8, min(DEFAULT_SLIDING_WINDOW_STRIDE, window_size))
                        n_chunks = 0
                        pos = 0
                        while pos < c_len and n_chunks < 16:
                            n_chunks += 1
                            if pos + window_size >= c_len:
                                break
                            pos += step
                        chunk_counts.append(n_chunks)

            length_entry: dict[str, Any] = {
                "overall": summarize_truncation_rates(all_meta),
                "by_source_dataset": {
                    ds: summarize_truncation_rates(metas)
                    for ds, metas in sorted(per_dataset_meta.items())
                },
            }
            if max_len == 512 and chunk_counts:
                length_entry["sliding_window_mean_chunks"] = round(sum(chunk_counts) / len(chunk_counts), 4)
                length_entry["sliding_window_multi_chunk_rate"] = round(
                    sum(1 for c in chunk_counts if c > 1) / len(chunk_counts),
                    6,
                )
            by_length[str(max_len)] = length_entry

        split_reports[split_name] = {
            "source_path": str(path),
            "record_count": len(rows),
            "max_lengths": by_length,
        }

    report = {
        "generated_at_utc": utc_now_iso(),
        "tokenizer": "FallbackWhitespaceTokenizer",
        "max_lengths": lengths,
        "splits": split_reports,
    }
    if output_json is not None:
        out_path = resolve_project_path(Path(output_json))
        write_json(out_path, report)
    return report


def _cli() -> None:
    import argparse

    from src.utils.config import load_yaml_config

    parser = argparse.ArgumentParser(description="Measure TokenBudgetAllocator truncation rates across Phase 1 splits.")
    parser.add_argument("--config", default="configs/generation.yaml", help="Config YAML with paths.source_splits.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test_synthetic_pool", "test_natural_pool"],
        help="Split keys from paths.source_splits to profile.",
    )
    parser.add_argument(
        "--max-lengths",
        nargs="+",
        type=int,
        default=[512, 2048],
        help="Sequence lengths to evaluate (default: 512 2048).",
    )
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        default=None,
        help="Optional cap on records read per split.",
    )
    parser.add_argument(
        "--output-json",
        default="results/phase1_token_budget_report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    source_splits = cfg["paths"]["source_splits"]
    selected = {k: source_splits[k] for k in args.splits}
    report = export_split_token_budget_report(
        selected,
        max_lengths=args.max_lengths,
        output_json=args.output_json,
        max_records_per_split=args.max_records_per_split,
    )
    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()


