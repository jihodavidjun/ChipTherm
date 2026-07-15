#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_extension import validate_extension_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ChipTherm benchmark-extension root.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--require-hotspot-labels", action="store_true")
    args = parser.parse_args()
    report = validate_extension_root(args.root.resolve(), require_hotspot_labels=args.require_hotspot_labels)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
