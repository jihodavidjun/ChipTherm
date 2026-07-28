#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import (  # noqa: E402
    EXPECTED_PRIMARY_SPLIT,
    family_for_row,
    finalize_training_run,
    prepare_residual_scaling_indices,
    read_csv,
    sha256_file,
    write_json,
)


EXPERIMENTS = {
    "direct": {
        "architecture": "fno2d_direct_conditioned",
        "prediction_mode": "direct_temperature_fno",
        "physics_input": "none",
        "mean_head_mode": "direct_k",
        "target": "absolute_temperature_K",
    },
    "residual": {
        "architecture": "fno2d_residual_decomposed_conditioned",
        "prediction_mode": "residual_decomposed_fno",
        "physics_input": "source_superposition_v1",
        "mean_head_mode": "residual_resistance",
        "target": "source_superposition_residual_K",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a controlled Benchmark v2 direct or residual FNO."
    )
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise SystemExit("training preflight has not passed")

    root = args.data_root.expanduser().resolve()
    version_root = root / f"derived/indices/full_50x200/source_superposition/{args.source_version}"
    split_root = prepare_residual_scaling_indices(
        root,
        source_version=args.source_version,
        family_count=40,
        seed=int(preflight.get("determinism", {}).get("seed", 20260721)),
    )
    train_index = split_root / "train_index.csv"
    val_index = split_root / "val_index.csv"
    index_manifest = version_root / "index_manifest.json"
    if not all(path.is_file() for path in (train_index, val_index, index_manifest)):
        raise SystemExit("validated Benchmark v2 source-version indices are missing")
    counts = {"train": len(read_csv(train_index)), "val": len(read_csv(val_index))}
    if counts != {"train": 6400, "val": 800}:
        raise SystemExit(f"FNO experiments require canonical 6400/800 membership, got {counts}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected = EXPERIMENTS[args.experiment]
    validate_config(config, expected)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{args.experiment}_fno_train40_seed{args.seed}"
    lineage_path = out_dir / "training_lineage.json"
    lineage = {
        "schema_version": "benchmark_v2_fno_training_lineage/1",
        "run_id": run_id,
        "benchmark_id": "benchmark_v2_50family",
        "stage": "full_50x200",
        "experiment": args.experiment,
        "model_architecture": expected["architecture"],
        "prediction_mode": expected["prediction_mode"],
        "target": expected["target"],
        "source_superposition_used_as_model_input": args.experiment == "residual",
        "source_superposition_used_as_output_base": args.experiment == "residual",
        "source_superposition_version": args.source_version,
        "preflight_report_sha256": sha256_file(args.preflight_report),
        "source_version_index_manifest_sha256": sha256_file(index_manifest),
        "train_index_sha256": sha256_file(train_index),
        "internal_val_index_sha256": sha256_file(val_index),
        "optimization_family_uids": sorted(
            {family_for_row(row) for row in read_csv(train_index)}
        ),
        "checkpoint_selection_family_uids": sorted(
            {family_for_row(row) for row in read_csv(val_index)}
        ),
        "excluded_primary_val_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"]),
        "excluded_primary_test_family_uids": list(EXPECTED_PRIMARY_SPLIT["test"]),
        "primary_heldout_used_for_selection": False,
        "reconstruction": (
            "direct train-standardized absolute temperature"
            if args.experiment == "direct"
            else (
                "source_superposition_base_K + total_power_W * "
                "delta_R_eff_K_per_W + zero_mean_centered_field_K"
            )
        ),
        "resolved_training_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    write_json(lineage_path, lineage)

    epochs = 2 if args.smoke_test else int(config["epochs"])
    command = [
        sys.executable,
        "scripts/train_residual_cnn.py",
        "--train-index",
        str(train_index),
        "--val-index",
        str(val_index),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--base-channels",
        "32",
        "--model-architecture",
        str(config["model_architecture"]),
        "--prediction-mode",
        str(config["prediction_mode"]),
        "--metadata-conditioning",
        "--metadata-hidden-dim",
        str(config["metadata_hidden_dim"]),
        "--metadata-embedding-dim",
        str(config["metadata_embedding_dim"]),
        "--physics-input",
        str(config["physics_input"]),
        "--mean-head-mode",
        str(config["mean_head_mode"]),
        "--physical-representation",
        "dimensional",
        "--channel-routing-mode",
        "dimensional_baseline",
        "--fno-capacity-profile",
        str(config["fno_capacity_profile"]),
        "--fno-width",
        str(config["fno_width"]),
        "--fno-layers",
        str(config["fno_layers"]),
        "--fno-modes-x",
        str(config["fno_modes_x"]),
        "--fno-modes-y",
        str(config["fno_modes_y"]),
        "--fno-activation",
        str(config["fno_activation"]),
        "--fno-metadata-conditioning",
        str(config["fno_metadata_conditioning"]),
        "--fno-projection-channels",
        str(config["fno_projection_channels"]),
        "--scheduler",
        str(config["scheduler"]),
        "--early-stopping-patience",
        str(config["early_stopping_patience"]),
        "--checkpoint-frequency",
        str(config["checkpoint_frequency"]),
        "--lineage-manifest",
        str(lineage_path),
        "--device",
        args.device,
        "--num-workers",
        str(args.workers),
        "--seed",
        str(args.seed),
    ]
    if args.experiment == "direct":
        command.extend(["--direct-target-normalization", "train_standard"])
    else:
        command.extend(
            [
                "--lambda-final",
                str(config["lambda_final"]),
                "--lambda-mean",
                str(config["lambda_mean"]),
            ]
        )
    if args.resume:
        command.append("--resume")
    print(" ".join(command))
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    finalize_training_run(
        out_dir,
        lineage_path=lineage_path,
        resolved_config={"wrapper": vars(args), "training": config, "command": command},
    )
    return 0


def validate_config(config: dict[str, Any], expected: dict[str, str]) -> None:
    required = {
        "model_architecture": expected["architecture"],
        "prediction_mode": expected["prediction_mode"],
        "physics_input": expected["physics_input"],
        "mean_head_mode": expected["mean_head_mode"],
        "physical_representation": "dimensional",
        "fno_metadata_conditioning": "film",
        "graph_enabled": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid controlled FNO config: {mismatches}")
    if expected["prediction_mode"] == "direct_temperature_fno":
        if config.get("direct_target_normalization") != "train_standard":
            raise ValueError("direct FNO requires train-only target standardization")
    for key in (
        "fno_width",
        "fno_layers",
        "fno_modes_x",
        "fno_modes_y",
        "fno_projection_channels",
    ):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
