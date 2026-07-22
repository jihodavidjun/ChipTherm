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

from chiptherm.benchmark_v2_pipeline import PilotBuildOptions, STAGE_SPECS, build_pilot, load_selection


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the staged ChipTherm Benchmark v2 pilot.")
    parser.add_argument("--config", default=REPO_ROOT / "configs/benchmark_v2_50family/design_proposal.yaml", type=Path)
    parser.add_argument("--pilot-selection", default=None, type=Path)
    parser.add_argument("--family-dir", default=REPO_ROOT / "configs/benchmark_v2_50family/families", type=Path)
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--scratch-root", default=None, type=Path)
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    parser.add_argument("--stage", default="pilot_5x10", choices=sorted(STAGE_SPECS))
    parser.add_argument("--seed", default=20260721, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-parent-lock", default=REPO_ROOT / "configs/benchmark_v2_50family/dependency_lock.json", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--selected-families", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-hotspot-workdirs", action="store_true")
    parser.add_argument("--source-checkpoint", default=None, type=Path)
    parser.add_argument("--source-lineage", default=REPO_ROOT / "configs/benchmark_v2_50family/source_response_lineage_prototype_seed1.json", type=Path)
    parser.add_argument("--residual-checkpoint", default=None, type=Path)
    parser.add_argument("--source-device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument(
        "--min-free-gb",
        default=100.0,
        type=float,
        help="Absolute GiB reserve that must remain after the projected build and at peak staging (default: 100).",
    )
    parser.add_argument(
        "--min-free-fraction",
        default=0.20,
        type=float,
        help=(
            "Fraction of the space currently free when the gate runs that must remain free after the build; "
            "the effective reserve is max(this amount, --min-free-gb), not a fraction of total capacity "
            "(default: 0.20)."
        ),
    )
    parser.add_argument(
        "--max-retained-gb",
        default=2000.0,
        type=float,
        help="Maximum projected total retained benchmark footprint in GiB (default: 2000).",
    )
    parser.add_argument(
        "--max-staging-gb",
        default=500.0,
        type=float,
        help="Maximum projected peak temporary staging footprint in GiB (default: 500).",
    )
    parser.add_argument("--override-storage-gate", action="store_true")
    parser.add_argument("--execution-families", nargs="*", default=None, help="Schedule only these families while retaining the full immutable stage identity.")
    parser.add_argument("--start-family", default=None, help="Inclusive execution-family lower bound, e.g. f011.")
    parser.add_argument("--end-family", default=None, help="Inclusive execution-family upper bound, e.g. f020.")
    parser.add_argument("--max-new-package-runs", default=None, type=int)
    parser.add_argument("--stop-after-current-family", action="store_true")
    args = parser.parse_args()

    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    selection_path = (args.pilot_selection or STAGE_SPECS[args.stage].selection_path).resolve()
    selection = load_selection(selection_path)
    selected = tuple(args.selected_families or [row["family_uid"] for row in selection["selected_families"]])
    execution_families = list(args.execution_families or [])
    if args.start_family or args.end_family:
        lower = args.start_family or selected[0]
        upper = args.end_family or selected[-1]
        ranged = [uid for uid in selected if lower <= uid <= upper]
        if not ranged:
            raise SystemExit(f"family range [{lower}, {upper}] selects no stage families")
        if execution_families and set(execution_families) != set(ranged):
            raise SystemExit("--execution-families conflicts with --start-family/--end-family")
        execution_families = ranged
    data_root = args.data_root.expanduser().resolve()
    scratch_root = (args.scratch_root or data_root / "staging").expanduser().resolve()
    run_id = args.run_id or f"pilot-{args.seed}-{uuid.uuid4().hex[:10]}"
    options = PilotBuildOptions(
        config_path=args.config.resolve(),
        selection_path=selection_path,
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
        stage=args.stage,
        min_free_gb=float(args.min_free_gb),
        min_free_fraction=float(args.min_free_fraction),
        max_retained_gb=float(args.max_retained_gb),
        max_staging_gb=float(args.max_staging_gb),
        override_storage_gate=bool(args.override_storage_gate),
        execution_family_uids=tuple(execution_families) if execution_families else None,
        max_new_package_runs=args.max_new_package_runs,
        stop_after_current_family=bool(args.stop_after_current_family),
    )
    print(f"Benchmark: benchmark_v2_50family")
    print(f"Stage: {args.stage}")
    print(f"Data root: {data_root}")
    print(f"Scratch root: {scratch_root}")
    print(f"Run ID: {run_id}")
    print(f"Selected families: {', '.join(selected)}")
    if execution_families:
        print(f"Execution families: {', '.join(execution_families)}")
    report = build_pilot(options)
    print(f"Pilot status: {report['status']}")
    print(f"Workloads: {report['workload_count']}")
    projection = report.get("resource_projection")
    if projection:
        summary = projection["capacity_summary_gib"]
        print("Storage gate:")
        print(f"  Current free: {summary['current_free']:.3f} GiB")
        print(f"  Projected new retained: {summary['projected_new_retained']:.3f} GiB")
        print(f"  Projected total retained: {summary['projected_total_retained']:.3f} GiB")
        print(f"  Projected peak staging: {summary['projected_peak_staging']:.3f} GiB")
        print(f"  Projected post-build free: {summary['projected_post_build_free']:.3f} GiB")
        print(f"  Projected peak-build free: {summary['projected_peak_build_free']:.3f} GiB")
        print(f"  Required absolute margin: {summary['required_absolute_margin']:.3f} GiB")
        print(f"  Required fractional margin: {summary['required_fractional_margin']:.3f} GiB")
        print(f"  Required effective margin: {summary['required_effective_margin']:.3f} GiB")
        failures = projection.get("failed_gate_conditions", [])
        print(f"  Failed gate conditions: {', '.join(failures) if failures else 'none'}")
        print(f"  Storage recommendation: {projection['recommendation']}")
    print(f"Final recommendation: {report['recommendation']}")
    print(f"Report: {data_root / 'canonical/manifests' / f'{args.stage}_validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
