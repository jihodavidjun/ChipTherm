#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_extension import (
    DEFAULT_CONFIG_PATH,
    estimate_storage,
    file_sha256,
    generate_sample,
    layout_statistics,
    load_extension_config,
    row_for_sample,
    select_cases,
    validate_sample_sources,
    verify_approval,
    write_audit_reports,
    write_indexes,
    write_sample_sources,
)
from chiptherm.parsers import parse_layer_grid
from chiptherm.paths import hotspot_home
from chiptherm.runner import build_hotspot_command, run_hotspot
from chiptherm.scenario import load_simulation_input
from chiptherm.writers import read_grid_shape, write_flp, write_hotspot_config, write_ptrace
from chiptherm.writers import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build controlled ChipTherm benchmark-extension source samples.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan generation without writing samples.")
    mode.add_argument("--pilot", action="store_true", help="Generate a pilot/smoke extension set.")
    mode.add_argument("--full", action="store_true", help="Generate the full approved extension set.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1")
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-case", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-hotspot-workdirs", action="store_true")
    parser.add_argument("--cleanup-hotspot-workdirs", action="store_true")
    parser.add_argument("--max-storage-gb", type=float, default=None)
    parser.add_argument("--approval-file", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--run-hotspot", action="store_true", help="Run full-package HotSpot labels for generated samples.")
    parser.add_argument("--hotspot-home", type=Path, default=None)
    parser.add_argument("--config-template", type=Path, default=REPO_ROOT / "configs/hotspot_base.config")
    args = parser.parse_args()

    config = load_extension_config(args.config)
    cases = select_cases(config, args.case_ids)
    samples_per_case = _samples_per_case(args)
    stage = _stage(args, samples_per_case)
    out_dir = (args.out_root / stage).resolve() if args.out_root.name != stage else args.out_root.resolve()
    total_samples = samples_per_case * len(cases)
    storage = estimate_storage(total_samples, include_hotspot_labels=False)

    if args.max_storage_gb is not None and storage["total_GB_for_requested_mode"] > args.max_storage_gb:
        raise SystemExit(
            f"estimated storage {storage['total_GB_for_requested_mode']:.3f} GB exceeds --max-storage-gb {args.max_storage_gb:.3f}"
        )
    if args.full:
        pilot_root = args.pilot_root or (args.out_root / "pilot")
        verify_approval(pilot_root.resolve(), args.approval_file.resolve() if args.approval_file else None)

    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": stage,
        "config": str(args.config.resolve()),
        "config_hash_sha256": file_sha256(args.config.resolve()),
        "out_dir": str(out_dir),
        "case_ids": [case["case_id"] for case in cases],
        "samples_per_case": samples_per_case,
        "total_samples": total_samples,
        "seed": args.seed,
        "storage_estimate": storage,
        "hotspot_labels": "full_package" if args.run_hotspot else "not_generated_by_this_stage",
    }

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(out_dir / "dry_run_manifest.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_stats = []
    validations = []
    for case in cases:
        case_dir = out_dir / case["case_id"]
        for sample_index in range(1, samples_per_case + 1):
            sample_uid = f"benchmark_extension_v1_{case['case_id']}_sample_{sample_index:06d}"
            sample_dir = case_dir / f"sample_{sample_index:06d}"
            if args.resume and (sample_dir / "source/scenario.yaml").exists():
                layout_path = sample_dir / "source/layout.json"
                power_path = sample_dir / "source/power.yaml"
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
                import yaml

                power = yaml.safe_load(power_path.read_text(encoding="utf-8")) or {}
                paths = {
                    "source_dir": sample_dir / "source",
                    "scenario_path": sample_dir / "source/scenario.yaml",
                    "layout_path": layout_path,
                    "power_path": power_path,
                    "package_path": sample_dir / "source/package.yaml",
                    "hotspot_path": sample_dir / "source/hotspot.yaml",
                    "benchmark_path": sample_dir / "source/benchmark.yaml",
                    "y_path": sample_dir / "parsed/temp_layer0.npy",
                }
            else:
                layout, power, benchmark = generate_sample(case, config["defaults"], sample_index, args.seed)
                paths = write_sample_sources(
                    sample_dir,
                    sample_uid,
                    layout,
                    power,
                    benchmark,
                    cleanup_hotspot_workdirs=args.cleanup_hotspot_workdirs and not args.keep_hotspot_workdirs,
                )
            stats = layout_statistics(layout, power)
            stats["sample_uid"] = sample_uid
            stats["case_id"] = case["case_id"]
            stats["split"] = case["split_role"]
            validation = validate_sample_sources(paths["scenario_path"], case)
            validations.append({"sample_uid": sample_uid, "passed": validation["passed"], "problems": validation["problems"]})
            hotspot_result = {"status": "not_run", "runtime_s": ""}
            if args.run_hotspot and validation["passed"]:
                hotspot_result = _run_full_package_hotspot(
                    paths["scenario_path"],
                    sample_dir,
                    hotspot_home_path=(args.hotspot_home or hotspot_home()).resolve(),
                    config_template=args.config_template.resolve(),
                    keep_hotspot_workdirs=args.keep_hotspot_workdirs,
                )
            sample_stats.append(stats)
            rows.append(
                row_for_sample(
                    sample_uid=sample_uid,
                    case=case,
                    paths=paths,
                    statistics=stats,
                    stage=stage,
                    hotspot_status=hotspot_result["status"],
                )
            )
            rows[-1]["hotspot_runtime_s"] = hotspot_result["runtime_s"]

    write_indexes(out_dir, rows)
    manifest = write_audit_reports(
        out_dir,
        rows,
        sample_stats,
        stage=stage,
        validation=validations,
        config_hash=file_sha256(args.config.resolve()),
    )
    manifest["runtime_s"] = time.perf_counter() - start
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"Generated ChipTherm extension {stage}: {len(rows)} samples")
    print(f"Output: {out_dir}")
    print(f"Validation passed: {manifest['validation']['passed']}")
    return 0 if manifest["validation"]["passed"] else 2


def _run_full_package_hotspot(
    scenario_path: Path,
    sample_dir: Path,
    *,
    hotspot_home_path: Path,
    config_template: Path,
    keep_hotspot_workdirs: bool,
) -> dict[str, str]:
    hotspot_dir = sample_dir / "hotspot"
    outputs_dir = sample_dir / "outputs"
    parsed_dir = sample_dir / "parsed"
    for path in (hotspot_dir, outputs_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)
    sim = load_simulation_input(scenario_path)
    start = time.perf_counter()
    flp_path = write_flp(sim.layout, hotspot_dir / "chiplet.flp")
    ptrace_path = write_ptrace(sim.layout, sim.power, hotspot_dir / "power.ptrace")
    config_path = write_hotspot_config(config_template, hotspot_dir / "hotspot.config", sim.package, sim.hotspot)
    rows, cols = read_grid_shape(config_path)
    block_steady_path = outputs_dir / "block.steady"
    grid_steady_path = outputs_dir / "grid.steady"
    command = build_hotspot_command(
        hotspot_home=str(hotspot_home_path),
        config_path=config_path.resolve(),
        flp_path=flp_path.resolve(),
        ptrace_path=ptrace_path.resolve(),
        steady_path=block_steady_path.resolve(),
        grid_steady_path=grid_steady_path.resolve(),
    )
    (sample_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    result = run_hotspot(command, cwd=hotspot_dir)
    (outputs_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (outputs_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    runtime = time.perf_counter() - start
    if result.returncode != 0:
        write_manifest(
            sample_dir / "manifest.json",
            {
                "schema_version": 1,
                "success": False,
                "hotspot_status": "failed",
                "return_code": result.returncode,
                "runtime_s": runtime,
                "command": command,
            },
        )
        return {"status": "failed", "runtime_s": f"{runtime:.8g}"}
    import numpy as np

    layer0 = parse_layer_grid(grid_steady_path, layer=0, rows=rows, cols=cols)
    np.save(parsed_dir / "temp_layer0.npy", layer0)
    write_manifest(
        sample_dir / "manifest.json",
        {
            "schema_version": 1,
            "success": True,
            "hotspot_status": "full_package_done",
            "runtime_s": runtime,
            "grid": {"rows": rows, "cols": cols},
            "temp_layer0": str((parsed_dir / "temp_layer0.npy").relative_to(sample_dir)),
            "temperature_K": {
                "min": float(layer0.min()),
                "max": float(layer0.max()),
                "mean": float(layer0.mean()),
            },
        },
    )
    if not keep_hotspot_workdirs:
        # Keep source, parsed target, stdout/stderr, and manifest. The generated
        # HotSpot inputs are reproducible from source files.
        for generated in (flp_path, ptrace_path, config_path):
            if generated.exists():
                generated.unlink()
    return {"status": "full_package_done", "runtime_s": f"{runtime:.8g}"}


def _samples_per_case(args: argparse.Namespace) -> int:
    if args.samples_per_case is not None:
        return args.samples_per_case
    if args.full:
        return 400
    return 50


def _stage(args: argparse.Namespace, samples_per_case: int) -> str:
    if args.dry_run:
        return "dry_run"
    if args.full:
        return "full"
    if samples_per_case <= 5:
        return "smoke"
    return "pilot"


if __name__ == "__main__":
    raise SystemExit(main())
