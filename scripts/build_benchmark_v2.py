#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_pipeline import PilotBuildOptions, build_pilot, load_selection


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the staged ChipTherm Benchmark v2 pilot.")
    parser.add_argument("--config", default=REPO_ROOT / "configs/benchmark_v2_50family/design_proposal.yaml", type=Path)
    parser.add_argument("--pilot-selection", default=REPO_ROOT / "configs/benchmark_v2_50family/pilot_5x10.yaml", type=Path)
    parser.add_argument("--family-dir", default=REPO_ROOT / "configs/benchmark_v2_50family/families", type=Path)
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--scratch-root", default=None, type=Path)
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    parser.add_argument("--stage", default="pilot_5x10", choices=["pilot_5x10"])
    parser.add_argument("--seed", default=20260721, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-parent-lock", default=REPO_ROOT / "configs/benchmark_v2_50family/dependency_lock.json", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--selected-families", nargs=5, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-hotspot-workdirs", action="store_true")
    parser.add_argument("--source-checkpoint", default=None, type=Path)
    parser.add_argument("--source-lineage", default=REPO_ROOT / "configs/benchmark_v2_50family/source_response_lineage_prototype_seed1.json", type=Path)
    parser.add_argument("--residual-checkpoint", default=None, type=Path)
    parser.add_argument("--source-device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    args = parser.parse_args()

    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    selection = load_selection(args.pilot_selection)
    selected = tuple(args.selected_families or [row["family_uid"] for row in selection["selected_families"]])
    data_root = args.data_root.expanduser().resolve()
    scratch_root = (args.scratch_root or data_root / "staging").expanduser().resolve()
    run_id = args.run_id or f"pilot-{args.seed}-{uuid.uuid4().hex[:10]}"
    options = PilotBuildOptions(
        config_path=args.config.resolve(),
        selection_path=args.pilot_selection.resolve(),
        family_dir=args.family_dir.resolve(),
        parent_lock_path=args.verify_parent_lock.resolve(),
        data_root=data_root,
        scratch_root=scratch_root,
        hotspot_home=args.hotspot_home.expanduser().resolve() if args.hotspot_home else None,
        config_template=args.config_template.resolve(),
        selected_families=selected,
        seed=int(args.seed),
        workers=int(args.workers),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        keep_hotspot_workdirs=bool(args.keep_hotspot_workdirs),
        run_id=run_id,
        source_checkpoint=args.source_checkpoint.expanduser().resolve() if args.source_checkpoint else None,
        source_lineage=args.source_lineage.resolve() if args.source_lineage else None,
        residual_checkpoint=args.residual_checkpoint.expanduser().resolve() if args.residual_checkpoint else None,
        source_device=args.source_device,
    )
    print(f"Benchmark: benchmark_v2_50family")
    print(f"Stage: {args.stage}")
    print(f"Data root: {data_root}")
    print(f"Scratch root: {scratch_root}")
    print(f"Run ID: {run_id}")
    print(f"Selected families: {', '.join(selected)}")
    report = build_pilot(options)
    print(f"Pilot status: {report['status']}")
    print(f"Workloads: {report['workload_count']}")
    print(f"Report: {data_root / 'canonical/manifests/pilot_5x10_validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
