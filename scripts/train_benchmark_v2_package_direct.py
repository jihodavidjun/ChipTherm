#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

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


DIRECT_ARCHITECTURE = "miniunet_refine_conditioned_direct_temperature_feature_fusion"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the controlled Benchmark v2 direct-temperature feature-fusion CNN."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument(
        "--config",
        default=REPO_ROOT
        / "configs/benchmark_v2_50family/training/package_direct_temperature_feature_fusion_seed1.yaml",
        type=Path,
    )
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="direct_temperature_feature_fusion_train40_seed1")
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
    sample_split_root = prepare_residual_scaling_indices(
        root,
        source_version=args.source_version,
        family_count=40,
        seed=int(preflight.get("determinism", {}).get("seed", 20260721)),
    )
    train_index = sample_split_root / "train_index.csv"
    val_index = sample_split_root / "val_index.csv"
    index_manifest = version_root / "index_manifest.json"
    if not all(path.is_file() for path in (train_index, val_index, index_manifest)):
        raise SystemExit("validated Benchmark v2 indices are missing")
    counts = {"train": len(read_csv(train_index)), "val": len(read_csv(val_index))}
    if counts != {"train": 6400, "val": 800}:
        raise SystemExit(f"direct baseline requires the canonical 6400/800 split, got {counts}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_direct_config(config)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = out_dir / "training_lineage.json"
    lineage = {
        "schema_version": "benchmark_v2_package_direct_training_lineage/1",
        "run_id": args.run_id,
        "benchmark_id": "benchmark_v2_50family",
        "stage": "full_50x200",
        "prediction_mode": str(config["prediction_mode"]),
        "target": "absolute_temperature_K",
        "source_superposition_used_as_model_input": False,
        "source_superposition_used_as_output_base": False,
        "source_response_model_required_for_direct_inference": False,
        "source_superposition_version_for_split_alignment_and_comparison_only": args.source_version,
        "preflight_report_sha256": sha256_file(args.preflight_report),
        "source_version_index_manifest_sha256": sha256_file(index_manifest),
        "train_index_sha256": sha256_file(train_index),
        "internal_val_index_sha256": sha256_file(val_index),
        "optimization_family_uids": sorted({family_for_row(row) for row in read_csv(train_index)}),
        "checkpoint_selection_family_uids": sorted({family_for_row(row) for row in read_csv(val_index)}),
        "excluded_primary_val_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"]),
        "excluded_primary_test_family_uids": list(EXPECTED_PRIMARY_SPLIT["test"]),
        "primary_heldout_used_for_selection": False,
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
        str(config["base_channels"]),
        "--model-architecture",
        str(config["model_architecture"]),
        "--prediction-mode",
        str(config["prediction_mode"]),
        "--direct-target-normalization",
        str(config["direct_target_normalization"]),
        "--metadata-conditioning",
        "--metadata-hidden-dim",
        str(config["metadata_hidden_dim"]),
        "--metadata-embedding-dim",
        str(config["metadata_embedding_dim"]),
        "--refine-channels",
        str(config["refine_channels"]),
        "--refine-blocks",
        str(config["refine_blocks"]),
        "--physics-input",
        "none",
        "--physical-representation",
        str(config["physical_representation"]),
        "--channel-routing-mode",
        str(config["channel_routing_mode"]),
        "--global-hidden-channels",
        str(config["global_hidden_channels"]),
        "--global-pool-size",
        str(config["global_pool_size"]),
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


def validate_direct_config(config: dict[str, object]) -> None:
    expected = {
        "model_architecture": DIRECT_ARCHITECTURE,
        "prediction_mode": "direct_temperature",
        "physics_input": "none",
        "coarse_spatial_loss_enabled": False,
        "graph_enabled": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": config.get(key)}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"invalid controlled direct baseline config: {mismatches}")


if __name__ == "__main__":
    raise SystemExit(main())
