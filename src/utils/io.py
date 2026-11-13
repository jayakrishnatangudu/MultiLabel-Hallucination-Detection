"""I/O, hashing, and timestamp helpers for reproducible data pipelines."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the hexadecimal SHA-256 digest of a file without modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute a short deterministic SHA-256 hex digest for a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    """Compute a deterministic SHA-256 digest of a JSON-serializable object."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return sha256_text(canonical)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Write a JSON-serializable object to disk, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
    return out


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any] | BaseModel]) -> Path:
    """Write an iterable of dicts or Pydantic models to a UTF-8 JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            obj = record.model_dump() if isinstance(record, BaseModel) else record
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a UTF-8 JSONL file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def summarize_lengths(values: Sequence[int]) -> dict[str, float | int]:
    """Return min, mean, median, p95, and max summary statistics for integer lengths."""
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    p95_idx = min(n - 1, int(0.95 * (n - 1)))
    return {
        "count": n,
        "min": ordered[0],
        "mean": round(sum(ordered) / n, 2),
        "median": round(median, 2),
        "p95": ordered[p95_idx],
        "max": ordered[-1],
    }

