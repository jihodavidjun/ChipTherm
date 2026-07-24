#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Benchmark v2 source model on split-safe and oracle indices.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    split_root = args.data_root.expanduser().resolve() / "derived/indices/full_50x200/source_response"
    splits = {
        "train": split_root / "train_index.csv",
        "internal_val": split_root / "internal_val_index.csv",
        "oracle_primary_val": split_root / "oracle_val_family_index.csv",
        "oracle_primary_test": split_root / "oracle_test_family_index.csv",
    }
    for name, index in splits.items():
        command = [
            sys.executable,
            "scripts/evaluate_source_response_model.py",
            "--checkpoint", str(args.checkpoint),
            "--source-index", str(index),
            "--data-root", str(args.data_root.expanduser().resolve()),
            "--out-dir", str(args.out_dir.expanduser().resolve() / name),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--num-workers", str(args.workers),
            "--profile-runtime",
        ]
        if args.save_predictions:
            command.append("--save-predictions")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
