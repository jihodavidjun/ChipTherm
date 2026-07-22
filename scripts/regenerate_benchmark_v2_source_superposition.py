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

from chiptherm.benchmark_v2_pipeline import (
    FULL_STAGE,
    PilotPaths,
    STAGE_SPECS,
    add_source_lineage_columns,
    canonicalize_stage_indices,
    create_builder_view,
    install_checkpoint,
    load_selection,
    promote_directory,
    run_checked,
    validate_source_checkpoint_lineage,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate a versioned full-stage source-superposition artifact without rerunning HotSpot."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-lineage", required=True, type=Path)
    parser.add_argument("--artifact-name", default="source_superposition_split_safe_v2")
    parser.add_argument("--package-batch-size", default=8, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default="regenerate-full-source-superposition")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    if "/" in args.artifact_name or args.artifact_name in {"source_superposition", "graphs"}:
        raise SystemExit("--artifact-name must be a new simple versioned name; the provisional artifact is never overwritten")

    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    selection = load_selection(STAGE_SPECS[FULL_STAGE].selection_path)
    lineage = validate_source_checkpoint_lineage(checkpoint, args.source_lineage.resolve(), selection)
    paths = PilotPaths(data_root, data_root / "staging", args.run_id, FULL_STAGE)
    destination = paths.derived(args.artifact_name)
    if destination.exists():
        if args.resume:
            print(f"Existing versioned artifact retained: {destination}")
            return 0
        raise FileExistsError(destination)

    portable_checkpoint = install_checkpoint(checkpoint, data_root, "source_response")
    graph_root = paths.derived("graphs")
    if not (graph_root / "combined_encoded_index.csv").is_file():
        raise FileNotFoundError(f"full-stage graph index is missing: {graph_root}")
    run_root = paths.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    graph_view = create_builder_view(graph_root, run_root / "source_base_input", data_root)
    stage = run_root / args.artifact_name
    run_checked(
        [
            sys.executable,
            "scripts/build_full_source_superposition_base.py",
            "--index", str(graph_view / "combined_encoded_index.csv"),
            "--checkpoint", str(portable_checkpoint),
            "--out-root", str(stage),
            "--package-batch-size", str(args.package_batch_size),
            "--source-batch-size", str(args.source_batch_size),
            "--device", args.device,
            "--resume",
        ],
        run_root,
    )
    canonicalize_stage_indices(stage, destination, data_root)
    add_source_lineage_columns(
        stage,
        lineage,
        portable_checkpoint,
        data_root,
        artifact_status="split_safe_source_checkpoint_dependent_candidate",
    )
    promote_directory(stage, destination, resume=False)
    print(f"Generated versioned source-superposition candidate: {destination}")
    print("The provisional full-stage artifact and final indices were not overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
