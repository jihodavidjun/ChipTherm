#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_pipeline import repair_pilot_portability


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair Benchmark v2 pilot indices/manifests in place without modifying thermal or array artifacts."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply atomic metadata-only repairs. Default is a dry-run report.")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    report = repair_pilot_portability(args.data_root, apply=bool(args.apply))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.apply and report["after"]["violation_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
