#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.compact_low_lr_continuation import (  # noqa: E402
    CONTINUATION_EPOCHS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_EPOCHS,
    EXPECTED_FINAL_LR,
    EXPECTED_INITIAL_LR,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_RECONSTRUCTION,
    EXPECTED_SEED,
    EXPECTED_WEIGHT_DECAY,
    load_checkpoint,
    load_yaml,
    now_utc,
    selection_thresholds,
    sha256_file,
    stable_hash,
    validate_continuation_config,
    validate_parent_checkpoint,
)


DEFAULT_PARENT = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/package_residual/"
    "feature_fusion_train40_source_v1_seed1/checkpoints/best.pt"
)
DEFAULT_CANONICAL_CONFIG = (
    REPO_ROOT
    / "configs/benchmark_v2_50family/training/"
    "package_residual_feature_fusion_v1.yaml"
)
DEFAULT_CONTINUATION_CONFIG = (
    REPO_ROOT
    / "configs/benchmark_v2_50family/training/"
    "package_residual_feature_fusion_continuation_v1.yaml"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/compact_low_lr_continuation"
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_experiment(
    *,
    data_root: Path,
    source_version: str,
    parent_path: Path,
    canonical_config_path: Path,
    continuation_config_path: Path,
    preflight_report: Path,
    out_root: Path,
    python: Path,
    device: str,
    workers: int,
    seed: int,
    write_artifacts: bool,
) -> dict[str, Any]:
    if seed != EXPECTED_SEED:
        raise ValueError(
            f"the bounded continuation fixes seed={EXPECTED_SEED}, got {seed}"
        )
    parent_path = parent_path.expanduser().resolve()
    parent = load_checkpoint(parent_path)
    compatibility = validate_parent_checkpoint(parent_path, parent)
    canonical_config = load_yaml(canonical_config_path)
    continuation_config = load_yaml(continuation_config_path)
    config_diff = validate_continuation_config(
        canonical_config,
        continuation_config,
    )
    lineage = parent["training_lineage"]
    if lineage.get("source_superposition_version") != source_version:
        raise ValueError(
            "requested source version differs from canonical parent: "
            f"{source_version} vs "
            f"{lineage.get('source_superposition_version')}"
        )
    data_root = data_root.expanduser().resolve()
    index_root = (
        data_root
        / "derived/indices/full_50x200/source_superposition"
        / source_version
        / "sample_split"
    )
    train_index = index_root / "train_index.csv"
    val_index = index_root / "val_index.csv"
    expected_hashes = {
        "train_index_sha256": lineage["train_index_sha256"],
        "internal_val_index_sha256": lineage["internal_val_index_sha256"],
    }
    actual_hashes: dict[str, str] = {}
    for key, path in (
        ("train_index_sha256", train_index),
        ("internal_val_index_sha256", val_index),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"continuation index is missing: {path}")
        actual_hashes[key] = sha256_file(path)
        if actual_hashes[key] != expected_hashes[key]:
            raise ValueError(
                f"{key} differs from canonical parent: "
                f"{actual_hashes[key]} vs {expected_hashes[key]}"
            )
    if not preflight_report.is_file():
        raise FileNotFoundError(
            f"Benchmark v2 preflight report is missing: {preflight_report}"
        )
    preflight = json.loads(preflight_report.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise ValueError("Benchmark v2 preflight report has not passed")
    out_root = out_root.expanduser().resolve()
    command = [
        str(python),
        "scripts/train_benchmark_v2_package_residual.py",
        "--data-root",
        str(data_root),
        "--source-version",
        source_version,
        "--config",
        str(continuation_config_path.expanduser().resolve()),
        "--preflight-report",
        str(preflight_report.expanduser().resolve()),
        "--out-dir",
        str(out_root),
        "--run-id",
        "compact_low_lr_continuation_seed1",
        "--train-family-count",
        "40",
        "--init-checkpoint",
        str(parent_path),
        "--device",
        device,
        "--workers",
        str(workers),
        "--seed",
        str(seed),
    ]
    manifest = {
        "schema_version": "compact_low_lr_continuation_manifest/1",
        "created_at_utc": now_utc(),
        "experiment": "bounded_low_learning_rate_continuation",
        "status": "prepared",
        "completed_weight_interpolation_experiment_modified": False,
        "parent_checkpoint": compatibility["parent_checkpoint"],
        "initialization": compatibility["training_state"],
        "canonical_configuration": {
            "path": str(canonical_config_path.expanduser().resolve()),
            "sha256": sha256_file(canonical_config_path),
        },
        "continuation_configuration": {
            "path": str(continuation_config_path.expanduser().resolve()),
            "sha256": sha256_file(continuation_config_path),
        },
        "source_superposition_version": source_version,
        "normalization": {
            "source": "canonical parent checkpoint and unchanged train split",
            "sha256": compatibility["normalization_sha256"],
        },
        "indices": {
            "train": str(train_index),
            "validation": str(val_index),
            **actual_hashes,
        },
        "model": {
            "architecture": compatibility["invariants"]["architecture"],
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "prediction_mode": compatibility["invariants"]["prediction_mode"],
            "physical_representation": compatibility["invariants"][
                "physical_representation"
            ],
            "reconstruction": EXPECTED_RECONSTRUCTION,
            "mean_correction_sign": 1,
            "centered_correction_sign": 1,
        },
        "training": {
            "epochs": EXPECTED_EPOCHS,
            "optimizer": "AdamW",
            "weight_decay": EXPECTED_WEIGHT_DECAY,
            "initial_lr": EXPECTED_INITIAL_LR,
            "scheduler": "CosineAnnealingLR",
            "final_lr": EXPECTED_FINAL_LR,
            "warmup": False,
            "ema": False,
            "swa": False,
            "batch_size": EXPECTED_BATCH_SIZE,
            "seed": EXPECTED_SEED,
            "checkpoint_epochs": list(CONTINUATION_EPOCHS),
        },
        "selection": {
            "protocols": [
                "known_family_sample_test",
                "primary_validation_families",
            ],
            "primary_test_used_for_selection": False,
            "thresholds": selection_thresholds(),
            "maximum_promoted_checkpoints": 1,
        },
        "preflight": {
            "path": str(preflight_report.expanduser().resolve()),
            "sha256": sha256_file(preflight_report),
        },
        "training_command": command,
    }
    manifest["definition_sha256"] = stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at_utc", "status", "definition_sha256"}
        }
    )
    if write_artifacts:
        out_root.mkdir(parents=True, exist_ok=True)
        if any((out_root / "checkpoints").glob("*.pt")):
            raise FileExistsError(
                "continuation checkpoints already exist; this experiment "
                "must not be resumed or overwritten"
            )
        write_json(out_root / "continuation_manifest.json", manifest)
        write_json(
            out_root / "initialization_compatibility_report.json",
            compatibility,
        )
        write_json(out_root / "training_config_diff.json", config_diff)
    return {
        "manifest": manifest,
        "compatibility": compatibility,
        "config_diff": config_diff,
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or launch the frozen compact low-LR continuation."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"),
        type=Path,
    )
    parser.add_argument(
        "--source-version",
        default="source_superposition_final_train40_source_v1",
    )
    parser.add_argument("--parent-checkpoint", default=DEFAULT_PARENT, type=Path)
    parser.add_argument(
        "--canonical-config",
        default=DEFAULT_CANONICAL_CONFIG,
        type=Path,
    )
    parser.add_argument(
        "--continuation-config",
        default=DEFAULT_CONTINUATION_CONFIG,
        type=Path,
    )
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--out-root", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=EXPECTED_SEED, type=int)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Write immutable pre-training artifacts but do not train.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write pre-training artifacts and launch the one fresh training run.",
    )
    args = parser.parse_args()
    if args.prepare and args.execute:
        raise SystemExit("--prepare and --execute are mutually exclusive")
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    result = prepare_experiment(
        data_root=args.data_root,
        source_version=args.source_version,
        parent_path=args.parent_checkpoint,
        canonical_config_path=args.canonical_config,
        continuation_config_path=args.continuation_config,
        preflight_report=args.preflight_report,
        out_root=args.out_root,
        python=args.python,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        write_artifacts=args.prepare or args.execute,
    )
    print(" ".join(result["command"]))
    if args.execute:
        subprocess.run(result["command"], cwd=REPO_ROOT, check=True)
    else:
        print(
            "Continuation definition validated; training was not launched."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
