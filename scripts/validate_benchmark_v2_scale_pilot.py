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

from chiptherm.benchmark_v2_pipeline import validate_scale_pilot_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly validate Benchmark v2's 10x50 scale pilot.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--residual-checkpoint", default=None, type=Path)
    parser.add_argument("--require-relocation", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    report = validate_scale_pilot_root(
        args.data_root,
        allow_dry_run=args.allow_dry_run,
        residual_checkpoint=args.residual_checkpoint,
        require_relocation=args.require_relocation,
    )
    print(f"Scale-pilot validation passed: {report['passed']}")
    for check in report["checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check.get('details', '')}")
    print(f"Recommendation: {report['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
