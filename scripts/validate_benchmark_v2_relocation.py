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

from chiptherm.benchmark_v2_pipeline import STAGE_SPECS, relocate_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Relocate and validate the Benchmark v2 pilot tree.")
    parser.add_argument("--source-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--residual-checkpoint", default=None, type=Path)
    parser.add_argument("--stage", default="pilot_5x10", choices=sorted(STAGE_SPECS))
    parser.add_argument("--link-bulk-arrays", action="store_true", help="Hard-link immutable bulk arrays when source and destination share a filesystem.")
    args = parser.parse_args()
    if args.source_root is None:
        raise SystemExit("--source-root or CHIPTHERM_V2_DATA_ROOT is required")
    report = relocate_pilot(
        args.source_root,
        args.destination_root,
        residual_checkpoint=args.residual_checkpoint,
        stage=args.stage,
        link_bulk_arrays=args.link_bulk_arrays,
    )
    print(f"Relocation passed: {report['passed']}")
    print(f"Files checked: {report['destination_file_count']}")
    print(f"Samples loaded: {report['loaded_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
