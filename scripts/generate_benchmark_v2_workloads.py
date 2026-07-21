#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_workloads import DEFAULT_SEED, load_family, write_workload_tree


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic Benchmark v2 pilot workloads.")
    parser.add_argument("--family-dir", default=REPO_ROOT / "configs/benchmark_v2_50family/families", type=Path)
    parser.add_argument("--family-uids", nargs="+", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(args.family_uids) != 5 or len(set(args.family_uids)) != 5:
        raise SystemExit("pilot workload generation requires exactly five unique --family-uids")
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {out_root}; pass --overwrite to replace it atomically")
    staging = out_root.parent / f".{out_root.name}.staging-{uuid.uuid4().hex}"
    try:
        families = [load_family(args.family_dir / f"{uid}.yaml") for uid in args.family_uids]
        manifest = write_workload_tree(families, staging, base_seed=int(args.seed))
        if manifest["workload_count"] != 50:
            raise RuntimeError(f"expected 50 workloads, got {manifest['workload_count']}")
        if out_root.exists():
            shutil.rmtree(out_root)
        staging.replace(out_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"Generated {manifest['workload_count']} workloads across {manifest['family_count']} families")
    print(f"Output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
