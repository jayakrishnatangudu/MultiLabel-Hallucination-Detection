"""Configuration loading and path resolution utilities."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML config: {path}")
    return data


def resolve_project_path(relative_or_abs: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a relative path against the project root (or a custom base directory)."""
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    root = base_dir if base_dir is not None else PROJECT_ROOT
    return (root / path).resolve()

