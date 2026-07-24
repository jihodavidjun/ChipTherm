#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import approve_source_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve and freeze a split-safe Benchmark v2 source checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-lineage", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--out-file", required=True, type=Path)
    parser.add_argument("--prototype-metrics", default=None, type=Path)
    parser.add_argument("--approve-with-caveats", action="store_true")
    args = parser.parse_args()
    approval = approve_source_checkpoint(
        args.checkpoint,
        lineage_path=args.training_lineage,
        evaluation_root=args.evaluation_root,
        output_path=args.out_file,
        allow_caveats=args.approve_with_caveats,
        prototype_metrics=args.prototype_metrics,
    )
    print(f"Approval: {approval['approval_status']}")
    print(f"Checkpoint SHA-256: {approval['checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
