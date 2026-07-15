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

from chiptherm.benchmark_extension import approve_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve a validated ChipTherm extension pilot for full generation.")
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-warnings", action="store_true")
    args = parser.parse_args()
    approval = approve_pilot(args.pilot_root.resolve(), args.out.resolve() if args.out else None, allow_warnings=args.allow_warnings)
    print(json.dumps(approval, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
