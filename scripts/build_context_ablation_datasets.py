#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_context_dataset import CONTEXT_SETS, build_context_dataset


DEFAULT_CONTEXT_SETS = [
    "total_power_only",
    "package_geometry",
    "occupancy_summary",
    "power_density_summary",
    "package_plus_power",
    "all_context",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all ChipTherm context-channel ablation datasets.")
    parser.add_argument("--base-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation", type=Path)
    parser.add_argument("--context-sets", nargs="+", default=DEFAULT_CONTEXT_SETS, choices=sorted(CONTEXT_SETS))
    args = parser.parse_args()

    base_root = args.base_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for context_set in args.context_sets:
        out_dir = out_root / context_set
        build_context_dataset(base_root=base_root, out_dir=out_dir, context_set=context_set)

    print("Context ablation dataset build complete")
    print(f"Context sets: {', '.join(args.context_sets)}")
    print(f"Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
