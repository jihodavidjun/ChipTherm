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

from chiptherm.benchmark_v2_training import (
    EXPECTED_PRIMARY_SPLIT,
    finalize_training_run,
    family_for_row,
    prepare_residual_scaling_indices,
    read_csv,
    sha256_file,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the final Benchmark v2 package residual CNN.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--config", default=REPO_ROOT / "configs/benchmark_v2_50family/training/package_residual_feature_fusion_v1.yaml", type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="feature_fusion_train40_source_v1_seed1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--train-family-count", default=40, choices=[5, 10, 20, 30, 40], type=int)
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
        family_count=args.train_family_count,
        seed=int(preflight.get("determinism", {}).get("seed", 20260721)),
    )
    index_root = version_root
    train_index = sample_split_root / "train_index.csv"
    val_index = sample_split_root / "val_index.csv"
    manifest = index_root / "index_manifest.json"
    if not all(path.is_file() for path in (train_index, val_index, manifest)):
        raise SystemExit("validated source-version residual indices are missing")
    if args.train_family_count == 40:
        counts = {"train": len(read_csv(train_index)), "val": len(read_csv(val_index))}
        if counts != {"train": 6400, "val": 800}:
            raise SystemExit(
                f"final residual optimization must use the 6400/800 train-family split, got {counts}"
            )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = out_dir / "training_lineage.json"
    lineage = {
        "schema_version": "benchmark_v2_package_residual_training_lineage/1",
        "run_id": args.run_id,
        "benchmark_id": "benchmark_v2_50family",
        "stage": "full_50x200",
        "preflight_report_sha256": sha256_file(args.preflight_report),
        "source_superposition_version": args.source_version,
        "source_version_index_manifest_sha256": sha256_file(manifest),
        "train_index_sha256": sha256_file(train_index),
        "internal_val_index_sha256": sha256_file(val_index),
        "optimization_family_uids": sorted({family_for_row(row) for row in read_csv(train_index)}),
        "checkpoint_selection_family_uids": sorted({family_for_row(row) for row in read_csv(val_index)}),
        "excluded_primary_val_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"]),
        "excluded_primary_test_family_uids": list(EXPECTED_PRIMARY_SPLIT["test"]),
        "primary_heldout_used_for_selection": False,
        "reconstruction": "source_superposition_base_K + total_power_W * delta_R_eff_K_per_W + zero_mean_centered_field_K",
        "resolved_training_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    write_json(lineage_path, lineage)
    epochs = 2 if args.smoke_test else int(config["epochs"])
    command = [
        sys.executable,
        "scripts/train_residual_cnn.py",
        "--train-index", str(train_index),
        "--val-index", str(val_index),
        "--out-dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--base-channels", str(config["base_channels"]),
        "--model-architecture", str(config["model_architecture"]),
        "--metadata-conditioning",
        "--metadata-hidden-dim", str(config["metadata_hidden_dim"]),
        "--metadata-embedding-dim", str(config["metadata_embedding_dim"]),
        "--refine-channels", str(config["refine_channels"]),
        "--refine-blocks", str(config["refine_blocks"]),
        "--physics-input", "source_superposition_v1",
        "--mean-head-mode", "residual_resistance",
        "--physical-representation", "dimensional",
        "--channel-routing-mode", "dimensional_baseline",
        "--lambda-final", str(config["lambda_final"]),
        "--lambda-mean", str(config["lambda_mean"]),
        "--global-hidden-channels", str(config["global_hidden_channels"]),
        "--global-pool-size", str(config["global_pool_size"]),
        "--scheduler", str(config["scheduler"]),
        "--early-stopping-patience", str(config["early_stopping_patience"]),
        "--checkpoint-frequency", str(config["checkpoint_frequency"]),
        "--lineage-manifest", str(lineage_path),
        "--device", args.device,
        "--num-workers", str(args.workers),
        "--seed", str(args.seed),
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
        resolved_config={
            "wrapper": vars(args),
            "training": config,
            "command": command,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
