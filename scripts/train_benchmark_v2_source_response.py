#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    assert_source_training_contract,
    finalize_training_run,
    prepare_source_scaling_indices,
    stable_json_hash,
    write_json,
    write_source_training_lineage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the split-safe Benchmark v2 source-response model.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--config", default=REPO_ROOT / "configs/benchmark_v2_50family/training/source_response_final_train40_v1.yaml", type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="final_train40_v1")
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
    prepare_source_scaling_indices(
        root,
        family_count=args.train_family_count,
        seed=int(preflight.get("determinism", {}).get("seed", 20260721)),
    )
    split_root = root / "derived/indices/full_50x200/source_response"
    if args.train_family_count != 40:
        split_root = split_root / f"scaling/train_{args.train_family_count}"
    train_index = split_root / "train_index.csv"
    val_index = split_root / "internal_val_index.csv"
    split_manifest = split_root / "split_manifest.json"
    contract = assert_source_training_contract(train_index, val_index, split_manifest)
    out_dir = args.out_dir.expanduser().resolve()
    if (out_dir / "approval.json").exists():
        raise SystemExit("approved source run is immutable; choose a new --run-id/--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = out_dir / "training_lineage.json"
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_lineage = write_source_training_lineage(
        lineage_path,
        contract=contract,
        preflight_report=args.preflight_report,
        run_id=args.run_id,
    )
    source_lineage["resolved_training_config_sha256"] = stable_json_hash(config)
    write_json(lineage_path, source_lineage)
    epochs = 2 if args.smoke_test else int(config["epochs"])
    command = [
        sys.executable,
        "scripts/train_source_response_model.py",
        "--train-index", str(train_index),
        "--val-index", str(val_index),
        "--data-root", str(root),
        "--out-dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(config["batch_size"]),
        "--packages-per-batch", str(config["packages_per_batch"]),
        "--lr", str(config["lr"]),
        "--base-channels", str(config["base_channels"]),
        "--depth", str(config["depth"]),
        "--lambda-source", str(config["lambda_source"]),
        "--lambda-package", str(config["lambda_package"]),
        "--package-loss-warmup-epochs", str(config["package_loss_warmup_epochs"]),
        "--lambda-source-mean", str(config["lambda_source_mean"]),
        "--device", args.device,
        "--num-workers", str(args.workers),
        "--seed", str(args.seed),
        "--lineage-manifest", str(lineage_path),
        "--early-stopping-patience", str(config["early_stopping_patience"]),
        "--checkpoint-frequency", str(config["checkpoint_frequency"]),
        "--scheduler", str(config["scheduler"]),
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
