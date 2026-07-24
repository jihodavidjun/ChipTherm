#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import (
    assert_preflight_immutability,
    prepare_source_version_residual_indices,
    require_approved_source_checkpoint,
)
from chiptherm.benchmark_v2_pipeline import loader_full_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly validate a versioned Benchmark v2 source-superposition artifact.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--approval-file", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--spot-check-count", default=50, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    require_approved_source_checkpoint(args.checkpoint, args.approval_file)
    assert_preflight_immutability(args.data_root, args.preflight_report)
    source_root = args.source_root.expanduser().resolve()
    input_root = source_root / "_input_splits"
    command = [
        sys.executable,
        "scripts/validate_source_superposition_base.py",
        "--train-index", str(input_root / "train_index.csv"),
        "--val-index", str(input_root / "val_index.csv"),
        "--test-index", str(input_root / "test_index.csv"),
        "--source-root", str(source_root),
        "--data-root", str(args.data_root.expanduser().resolve()),
        "--checkpoint", str(args.checkpoint),
        "--spot-check-count", str(args.spot_check_count),
        "--source-batch-size", str(args.source_batch_size),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    quality = [
        sys.executable,
        "scripts/evaluate_full_source_superposition_base.py",
        "--source-root", str(source_root),
        "--out-dir", str(args.out_dir.expanduser().resolve()),
        "--data-root", str(args.data_root.expanduser().resolve()),
    ]
    print(" ".join(command))
    print(" ".join(quality))
    if args.dry_run:
        return 0
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    loader_report = loader_full_audit(source_root / "combined_encoded_index.csv")
    validation_path = source_root / "validation_report.json"
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_payload["selected_version_loader_all"] = loader_report
    if loader_report.get("passed") is not True or loader_report.get("samples") != 10_000:
        validation_payload["ok"] = False
        validation_payload.setdefault("errors", []).append(
            f"selected source-version loader audit failed: {loader_report}"
        )
        validation_path.write_text(
            json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise ValueError(f"all-row source-version loader audit failed: {loader_report}")
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subprocess.run(quality, cwd=REPO_ROOT, check=True)
    assert_preflight_immutability(args.data_root, args.preflight_report)
    report = prepare_source_version_residual_indices(
        args.data_root,
        source_version_root=source_root,
    )
    print(f"Prepared residual indices: {report['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
