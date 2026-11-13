"""Immutable raw dataset acquisition and provenance manifest generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
import requests

from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.io import read_json, sha256_file, utc_now_iso, write_json
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _download_stream(url: str, destination: Path, timeout: int = 120) -> None:
    """Download a remote binary resource to disk via streaming."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(destination)


def _fetch_hf_dataset_files(
    dataset_key: str,
    spec: dict[str, Any],
    dataset_raw_dir: Path,
    existing_manifest_entry: dict[str, Any] | None,
    hf_api: HfApi,
    force: bool = False,
) -> dict[str, Any]:
    """Download pinned Hugging Face Hub files into an immutable raw directory."""
    repo_id = spec["repo_id"]
    info = hf_api.dataset_info(repo_id=repo_id)
    commit_sha = str(info.sha)
    card_license = None
    if info.card_data is not None:
        card_license = getattr(info.card_data, "license", None)

    existing_files = {
        item["remote_path"]: item
        for item in (existing_manifest_entry or {}).get("files", [])
    }

    file_records: list[dict[str, Any]] = []
    for file_entry in spec["files"]:
        remote_path = file_entry["remote_path"]
        subset = file_entry["subset"]
        dest_path = dataset_raw_dir / remote_path
        download_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{commit_sha}/{remote_path}"

        if dest_path.exists() and not force and remote_path in existing_files:
            prev = existing_files[remote_path]
            current_sha = sha256_file(dest_path)
            if current_sha == prev.get("sha256"):
                logger.info(
                    "Preserving existing immutable raw file for %s/%s (sha256 verified).",
                    dataset_key,
                    remote_path,
                )
                file_records.append(prev)
                continue

        logger.info("Downloading %s -> %s (revision=%s)", repo_id, remote_path, commit_sha[:12])
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=commit_sha,
            local_dir=str(dataset_raw_dir),
        )
        checksum = sha256_file(dest_path)
        file_records.append(
            {
                "subset": subset,
                "remote_path": remote_path,
                "local_path": str(dest_path),
                "download_url": download_url,
                "downloaded_at_utc": utc_now_iso(),
                "size_bytes": dest_path.stat().st_size,
                "sha256": checksum,
            }
        )

    return {
        "dataset_key": dataset_key,
        "provider": spec["provider"],
        "repo_id": repo_id,
        "domain": spec["domain"],
        "has_context": bool(spec["has_context"]),
        "source_url": spec["source_url"],
        "upstream_url": spec["upstream_url"],
        "dataset_version": commit_sha,
        "observed_hf_card_license": card_license,
        "licensing": spec["licensing"],
        "files": file_records,
    }


def _fetch_github_zip_dataset(
    dataset_key: str,
    spec: dict[str, Any],
    dataset_raw_dir: Path,
    existing_manifest_entry: dict[str, Any] | None,
    force: bool = False,
) -> dict[str, Any]:
    """Download an immutable GitHub archive asset with commit provenance."""
    commit_resp = requests.get(spec["api_commit_url"], timeout=20)
    commit_resp.raise_for_status()
    commit_sha = str(commit_resp.json()["sha"])

    dest_path = dataset_raw_dir / "data.zip"
    existing_files = {
        item["remote_path"]: item
        for item in (existing_manifest_entry or {}).get("files", [])
    }

    if dest_path.exists() and not force and "data.zip" in existing_files:
        prev = existing_files["data.zip"]
        if sha256_file(dest_path) == prev.get("sha256"):
            logger.info("Preserving existing immutable archive for %s (sha256 verified).", dataset_key)
            return {
                "dataset_key": dataset_key,
                "provider": spec["provider"],
                "repo_id": spec["repo_id"],
                "domain": spec["domain"],
                "has_context": bool(spec["has_context"]),
                "source_url": spec["source_url"],
                "upstream_url": spec["upstream_url"],
                "dataset_version": existing_manifest_entry.get("dataset_version", commit_sha),
                "licensing": spec["licensing"],
                "archive_members": spec["archive_members"],
                "files": [prev],
            }

    pinned_url = f"https://github.com/{spec['repo_id']}/raw/{commit_sha}/data.zip"
    logger.info("Downloading %s archive from %s", dataset_key, pinned_url)
    _download_stream(pinned_url, dest_path)
    checksum = sha256_file(dest_path)

    return {
        "dataset_key": dataset_key,
        "provider": spec["provider"],
        "repo_id": spec["repo_id"],
        "domain": spec["domain"],
        "has_context": bool(spec["has_context"]),
        "source_url": spec["source_url"],
        "upstream_url": spec["upstream_url"],
        "dataset_version": commit_sha,
        "licensing": spec["licensing"],
        "archive_members": spec["archive_members"],
        "files": [
            {
                "subset": "archive",
                "remote_path": "data.zip",
                "local_path": str(dest_path),
                "download_url": pinned_url,
                "downloaded_at_utc": utc_now_iso(),
                "size_bytes": dest_path.stat().st_size,
                "sha256": checksum,
            }
        ],
    }


def download_all_datasets(config_path: str | Path, force: bool = False) -> dict[str, Any]:
    """Download all enabled raw datasets and write data/raw/manifest.json."""
    config = load_yaml_config(config_path)
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])
    manifest_path = resolve_project_path(config["paths"]["raw_manifest"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
    existing_datasets = existing_manifest.get("datasets", {})

    hf_api = HfApi()
    dataset_entries: dict[str, Any] = {}

    for dataset_key, spec in config["datasets"].items():
        if not spec.get("enabled", True):
            logger.info("Skipping disabled dataset: %s", dataset_key)
            continue
        dataset_raw_dir = raw_dir / dataset_key
        dataset_raw_dir.mkdir(parents=True, exist_ok=True)
        prev_entry = existing_datasets.get(dataset_key)

        provider = spec["provider"]
        if provider == "huggingface_hub":
            entry = _fetch_hf_dataset_files(
                dataset_key=dataset_key,
                spec=spec,
                dataset_raw_dir=dataset_raw_dir,
                existing_manifest_entry=prev_entry,
                hf_api=hf_api,
                force=force,
            )
        elif provider == "github_zip":
            entry = _fetch_github_zip_dataset(
                dataset_key=dataset_key,
                spec=spec,
                dataset_raw_dir=dataset_raw_dir,
                existing_manifest_entry=prev_entry,
                force=force,
            )
        else:
            raise ValueError(f"Unsupported dataset provider '{provider}' for {dataset_key}")

        dataset_entries[dataset_key] = entry

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": utc_now_iso(),
        "config_path": str(config_path),
        "datasets": dataset_entries,
    }
    write_json(manifest_path, manifest)
    logger.info("Wrote immutable raw provenance manifest: %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download immutable raw datasets for Phase 1.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if checksums match.")
    args = parser.parse_args()
    download_all_datasets(config_path=args.config, force=args.force)


if __name__ == "__main__":
    main()

