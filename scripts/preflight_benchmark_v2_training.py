#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import run_training_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight accepted Benchmark v2 final training.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260721, type=int)
    parser.add_argument("--source-checkpoint", default=None, type=Path)
    parser.add_argument("--residual-checkpoint", default=None, type=Path)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    report = run_training_preflight(
        args.data_root,
        output_dir=args.out_dir,
        seed=args.seed,
        source_checkpoint=args.source_checkpoint,
        residual_checkpoint=args.residual_checkpoint,
    )
    print(f"Preflight passed: {report['passed']}")
    print(f"Report: {args.out_dir.resolve() / 'preflight_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
