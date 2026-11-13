"""Turnkey Google Colab GPU & Free-Tier Pipeline Orchestrator for Phases 2-5."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_stage_commands(
    *,
    python_exe: str,
    generation_config: str = "configs/generation_free_groq.yaml",
    model_config: str = "configs/model.yaml",
    training_config: str = "configs/training.yaml",
    seeds: Sequence[int] = (42, 101, 202),
    max_pilot_slots: int | None = 20,
) -> dict[str, list[list[str]]]:
    """Build exact deterministic stage commands for Free-Tier generation & Colab GPU training."""
    seed_args = [str(s) for s in seeds]
    pilot_cmd = [
        python_exe,
        "-m",
        "src.generation.pipeline",
        "--config",
        generation_config,
        "--execute-pilot",
        "--resume",
    ]
    if max_pilot_slots is not None:
        pilot_cmd.extend(["--max-slots", str(max_pilot_slots)])

    return {
        "phase2_verify": [
            [
                python_exe,
                "-m",
                "src.generation.pipeline",
                "--config",
                generation_config,
                "--verify-live-smoke",
            ]
        ],
        "phase2_pilot": [pilot_cmd],
        "phase2_full": [
            [
                python_exe,
                "-m",
                "src.generation.pipeline",
                "--config",
                generation_config,
                "--run-full",
                "--split",
                split,
                "--strategy",
                "strategy_a",
                "--resume",
            ]
            for split in ("train", "val", "test_synthetic")
        ],
        "phase3_baselines": [
            [
                python_exe,
                "-m",
                "src.models.baseline",
                "--variant",
                variant,
                "--output-dir",
                f"results/baselines/e1_{variant}",
            ]
            for variant in ("ovr_logreg", "linear_svc", "classifier_chain")
        ],
        "phase4_train_eval": [
            [
                python_exe,
                "-m",
                "src.training.train",
                "--model-config",
                model_config,
                "--training-config",
                training_config,
                "--backbone",
                backbone,
                "--seeds",
                *seed_args,
            ]
            for backbone in ("distilbert", "roberta", "deberta_v3", "modernbert")
        ],
        "phase5_tables": [
            [
                python_exe,
                "-m",
                "src.experiments.ablation",
                "--model-config",
                model_config,
                "--training-config",
                training_config,
                "--output-json",
                "results/ablation_matrix.json",
            ],
            [
                python_exe,
                "-m",
                "src.experiments.comparison",
                "--output-dir",
                "results/tables",
            ],
        ],
    }


def inspect_runtime_readiness() -> dict[str, object]:
    """Inspect Torch/CUDA and free-tier env key presence without revealing secret values."""
    from src.generation.backend import load_local_dotenv

    load_local_dotenv()
    has_torch = False
    cuda_available = False
    gpu_name = None
    try:
        import torch

        has_torch = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "python_executable": sys.executable,
        "torch_installed": has_torch,
        "cuda_available": cuda_available,
        "cuda_device": gpu_name,
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY", "").strip()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Colab GPU & Free-Tier Generation/Training Orchestrator."
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "phase2_verify",
            "phase2_pilot",
            "phase2_full",
            "phase3_baselines",
            "phase4_train_eval",
            "phase5_tables",
            "all",
        ],
    )
    parser.add_argument("--generation-config", default="configs/generation_free_groq.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202])
    parser.add_argument("--max-pilot-slots", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and runtime status without running.")
    args = parser.parse_args()

    readiness = inspect_runtime_readiness()
    print("Runtime readiness:", readiness)

    cmds_by_stage = build_stage_commands(
        python_exe=sys.executable,
        generation_config=args.generation_config,
        model_config=args.model_config,
        training_config=args.training_config,
        seeds=args.seeds,
        max_pilot_slots=args.max_pilot_slots,
    )
    selected_stages = (
        list(cmds_by_stage.keys()) if args.stage == "all" else [args.stage]
    )

    for st_name in selected_stages:
        print(f"\n=== Stage: {st_name} ===")
        for cmd in cmds_by_stage[st_name]:
            print(" ->", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

