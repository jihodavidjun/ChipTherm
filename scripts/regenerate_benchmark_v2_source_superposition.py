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
    durable_stage_complete,
    install_checkpoint,
    load_selection,
    promote_directory,
    run_checked,
    validate_source_checkpoint_lineage,
)
from chiptherm.benchmark_v2_training import require_approved_source_checkpoint
from chiptherm.benchmark_v2_training import write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate a versioned full-stage source-superposition artifact without rerunning HotSpot."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-lineage", required=True, type=Path)
    parser.add_argument("--approval-file", required=True, type=Path)
    parser.add_argument("--artifact-name", default="source_superposition_final_train40_source_v1")
    parser.add_argument("--package-batch-size", default=8, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", default=0, type=int, help="Reserved; source inference is synchronously microbatched.")
    parser.add_argument("--run-id", default="regenerate-full-source-superposition")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    if "/" in args.artifact_name or args.artifact_name in {"source_superposition", "graphs"}:
        raise SystemExit("--artifact-name must be a new simple versioned name; the provisional artifact is never overwritten")

    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    approval = require_approved_source_checkpoint(checkpoint, args.approval_file.resolve())
    selection = load_selection(STAGE_SPECS[FULL_STAGE].selection_path)
    lineage = validate_source_checkpoint_lineage(checkpoint, args.source_lineage.resolve(), selection)
    if lineage.get("approval_status") not in {"APPROVED", "APPROVED WITH CAVEATS"}:
        raise ValueError("source lineage is not an approved checkpoint manifest")
    paths = PilotPaths(data_root, data_root / "staging", args.run_id, FULL_STAGE)
    destination = paths.derived(args.artifact_name)
    if destination.exists():
        if args.resume:
            if not durable_stage_complete(destination):
                raise ValueError(
                    f"existing source version is not a durable completed artifact: {destination}"
                )
            existing_manifest = json_load(destination / "manifest.json")
            if existing_manifest.get("source_checkpoint_sha256") != approval["checkpoint_sha256"]:
                raise ValueError("existing source version uses a different approved checkpoint")
            existing_lineage = json_load(destination / "source_checkpoint_lineage.json")
            if existing_lineage != lineage:
                raise ValueError("existing source version uses different source lineage")
            print(f"Existing versioned artifact retained: {destination}")
            return 0
        raise FileExistsError(destination)
    if args.dry_run:
        print(f"Approved checkpoint: {approval['checkpoint_sha256']}")
        print(f"Planned destination: {destination}")
        print("Requested packages: 10000; HotSpot calls: 0; provisional artifact overwrite: false")
        return 0

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
        artifact_status="approved_train40_source_checkpoint_dependent",
    )
    generation_manifest = json_load(stage / "manifest.json")
    split_summaries = generation_manifest.get("splits", {})
    generated = sum(
        int(item.get("regenerated_packages", 0)) for item in split_summaries.values()
    )
    reused = sum(int(item.get("reused_packages", 0)) for item in split_summaries.values())
    runtime = float(generation_manifest.get("generation_runtime_s", 0.0))
    write_json(
        stage / "regeneration_report.json",
        {
            "schema_version": "benchmark_v2_final_source_superposition_regeneration/1",
            "requested_packages": 10_000,
            "generated_packages": generated,
            "reused_packages": reused,
            "skipped_valid_packages": reused,
            "failed_packages": 0,
            "retries": 0,
            "total_source_predictions": int(
                generation_manifest.get("total_source_count", 0)
            ),
            "runtime_s": runtime,
            "throughput_packages_per_s": 10_000 / max(runtime, 1.0e-12),
            "storage_bytes_before_promotion": tree_size(stage),
            "checkpoint_sha256": approval["checkpoint_sha256"],
            "source_lineage_sha256": generation_manifest.get(
                "source_checkpoint_lineage_sha256"
            ),
            "hotspot_calls": 0,
            "provisional_artifact_mutated": False,
        },
    )
    promote_directory(stage, destination, resume=False)
    print(f"Generated versioned source-superposition candidate: {destination}")
    print("The provisional full-stage artifact and final indices were not overwritten.")
    return 0


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
