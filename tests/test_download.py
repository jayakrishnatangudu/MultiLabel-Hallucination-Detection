"""Unit tests for immutable raw dataset downloading and manifest provenance."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml

from src.data.download import download_all_datasets
from src.utils.io import read_json, sha256_file


def test_download_records_provenance_and_preserves_existing_files(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    manifest_path = raw_dir / "manifest.json"
    cfg_path = tmp_path / "dataset.yaml"

    config = {
        "random_seed": 42,
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "splits_dir": str(tmp_path / "data" / "splits"),
            "results_dir": str(tmp_path / "results"),
            "docs_dir": str(tmp_path / "docs"),
            "raw_manifest": str(manifest_path),
            "preprocessing_report": str(tmp_path / "results" / "pre.json"),
            "split_report": str(tmp_path / "results" / "split.json"),
            "markdown_report": str(tmp_path / "docs" / "report.md"),
        },
        "datasets": {
            "medquad": {
                "enabled": True,
                "domain": "medical",
                "has_context": False,
                "provider": "huggingface_hub",
                "repo_id": "mock/medquad",
                "files": [{"remote_path": "data.csv", "subset": "train"}],
                "source_url": "https://huggingface.co/datasets/mock/medquad",
                "upstream_url": "https://github.com/abachaa/MedQuAD",
                "licensing": {
                    "hf_card_license": None,
                    "upstream_license": "CC-BY-4.0",
                    "license_status": "upstream_caution",
                    "redistribution_permitted": True,
                    "attribution": "Test attribution",
                    "notes": "Test note",
                },
            },
            "cuad_qa": {
                "enabled": True,
                "domain": "legal",
                "has_context": True,
                "provider": "github_zip",
                "repo_id": "mock/cuad",
                "zip_url": "https://example.org/data.zip",
                "api_commit_url": "https://api.github.com/repos/mock/cuad/commits/main",
                "archive_members": [{"member_path": "train.json", "subset": "train"}],
                "source_url": "https://github.com/mock/cuad",
                "upstream_url": "https://www.atticusprojectai.org/cuad",
                "licensing": {
                    "hf_card_license": "cc-by-4.0",
                    "upstream_license": "CC-BY-4.0",
                    "license_status": "upstream_caution",
                    "redistribution_permitted": True,
                    "attribution": "CUAD attribution",
                    "notes": "CUAD note",
                },
            },
        },
    }
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    mock_api = MagicMock()
    mock_api.dataset_info.return_value = SimpleNamespace(
        sha="hf_sha_123456",
        card_data=SimpleNamespace(license=None),
    )
    monkeypatch.setattr("src.data.download.HfApi", lambda: mock_api)

    download_calls = {"hf": 0, "zip": 0}

    def fake_hf_hub_download(repo_id: str, filename: str, repo_type: str, revision: str, local_dir: str) -> str:
        download_calls["hf"] += 1
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Question,Answer\nWhat is X?,X is Y.\n", encoding="utf-8")
        return str(target)

    def fake_download_stream(url: str, destination: Path, timeout: int = 120) -> None:
        download_calls["zip"] += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"PK\x03\x04mockzip")

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"sha": "gh_sha_987654"}

    monkeypatch.setattr("src.data.download.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("src.data.download._download_stream", fake_download_stream)
    monkeypatch.setattr("src.data.download.requests.get", lambda url, timeout=20: FakeResp())

    manifest = download_all_datasets(cfg_path, force=False)
    assert download_calls == {"hf": 1, "zip": 1}
    assert manifest_path.exists()
    saved = read_json(manifest_path)
    assert saved["datasets"]["medquad"]["dataset_version"] == "hf_sha_123456"
    assert saved["datasets"]["cuad_qa"]["dataset_version"] == "gh_sha_987654"
    csv_path = raw_dir / "medquad" / "data.csv"
    assert saved["datasets"]["medquad"]["files"][0]["sha256"] == sha256_file(csv_path)

    # Second call without force must preserve immutable files without re-downloading
    manifest_second = download_all_datasets(cfg_path, force=False)
    assert download_calls == {"hf": 1, "zip": 1}
    assert manifest_second["datasets"]["medquad"]["files"][0]["sha256"] == sha256_file(csv_path)

