#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Benchmark v2 residual model on immutable protocols.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--error-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    root = args.data_root.expanduser().resolve()
    index_root = root / f"derived/indices/full_50x200/source_superposition/{args.source_version}"
    evaluations = {
        "known_family_sample_test": index_root / "sample_split/test_index.csv",
        "primary_validation_families": index_root / "family_split/val_index.csv",
        "primary_test_families": index_root / "family_split/test_index.csv",
    }
    for name, index in evaluations.items():
        command = [
            sys.executable,
            "scripts/evaluate_residual_cnn.py",
            "--checkpoint", str(args.checkpoint),
            "--index", str(index),
            "--out-dir", str(args.out_dir.expanduser().resolve() / name),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--num-workers", str(args.workers),
            "--measure-end-to-end",
        ]
        if args.profile_components:
            command.append("--profile-components")
        if args.save_predictions:
            command.append("--save-predictions")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        if args.error_analysis:
            analysis = [
                sys.executable,
                "scripts/analyze_residual_cnn_errors.py",
                "--checkpoint", str(args.checkpoint),
                "--index", str(index),
                "--out-dir", str(args.out_dir.expanduser().resolve() / name / "error_analysis"),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--num-workers", str(args.workers),
            ]
            print(" ".join(analysis))
            if not args.dry_run:
                subprocess.run(analysis, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
