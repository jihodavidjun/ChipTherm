#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.encoder import active_power_map
from chiptherm.parsers import parse_block_temps, parse_layer_grid
from chiptherm.paths import hotspot_home as resolve_hotspot_home
from chiptherm.runner import build_hotspot_command, run_hotspot
from chiptherm.scenario import SimulationInput, load_simulation_input
from chiptherm.validate import _min_spacing_mm, _validate_layout
from chiptherm.writers import read_grid_shape, write_flp, write_hotspot_config, write_manifest, write_ptrace


@dataclass(frozen=True)
class SourceRun:
    name: str
    run_dir: Path
    temperature_path: Path
    runtime_s: float | None
    skipped: bool


class HotSpotRunError(RuntimeError):
    def __init__(self, message: str, *, run_dir: Path, returncode: int | None = None):
        super().__init__(message)
        self.run_dir = run_dir
        self.returncode = returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source-isolation superposition diagnostics for ChipTherm HotSpot samples.")
    parser.add_argument("--index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power/test_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/superposition_diagnostic", type=Path)
    parser.add_argument("--sample-uids", nargs="*", default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--samples-per-case", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--run-zero-power", action="store_true", help="Run one all-zero-power HotSpot baseline per selected sample.")
    parser.add_argument("--power-scale-test", nargs="*", type=float, default=None, help="Optional source scaling factors, e.g. 0.5 1.0 1.5.")
    parser.add_argument("--power-scale-source-index", default=0, type=int)
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip valid completed isolated runs.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate selected sample outputs.")
    parser.add_argument("--summary-only", action="store_true", help="Summarize existing metrics without running HotSpot.")
    parser.add_argument("--dry-run", action="store_true", help="Select samples and estimate run counts without running HotSpot.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed runs and continue with remaining samples.")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required to parse power.yaml")
    assert_safe_output_dir(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarize_existing(args.out_dir)
        return 0

    rows = read_rows(args.index)
    selected = select_rows(
        rows,
        sample_uids=args.sample_uids,
        cases=args.cases,
        samples_per_case=int(args.samples_per_case),
        seed=int(args.seed),
    )
    if not selected:
        raise SystemExit("No samples selected.")

    plan = estimate_hotspot_runs(selected, run_zero_power=bool(args.run_zero_power), power_scales=args.power_scale_test)
    print_selection_plan(selected, plan)
    if args.dry_run:
        write_run_manifest(args.out_dir, args, selected, plan, dry_run=True)
        return 0

    start = time.perf_counter()
    sample_records: list[dict[str, Any]] = []
    power_scaling_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in selected:
        try:
            sample_result = process_sample(
                row,
                out_dir=args.out_dir,
                hotspot_home=args.hotspot_home,
                config_template=args.config_template,
                run_zero_power=bool(args.run_zero_power),
                power_scales=args.power_scale_test or [],
                power_scale_source_index=int(args.power_scale_source_index),
                resume=bool(args.resume),
                overwrite=bool(args.overwrite),
            )
            sample_records.append(sample_result["sample_metrics"])
            power_scaling_records.extend(sample_result["power_scaling"])
        except Exception as exc:
            failure = {
                "sample_uid": row.get("sample_uid"),
                "case_id": row.get("case_id"),
                "error": str(exc),
                "type": type(exc).__name__,
            }
            if isinstance(exc, HotSpotRunError):
                failure["run_dir"] = str(exc.run_dir)
                failure["returncode"] = exc.returncode
            failures.append(failure)
            print(f"FAILED {row.get('sample_uid')}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    total_runtime_s = time.perf_counter() - start
    write_outputs(
        out_dir=args.out_dir,
        args=args,
        selected=selected,
        plan=plan,
        sample_records=sample_records,
        power_scaling_records=power_scaling_records,
        failures=failures,
        total_runtime_s=total_runtime_s,
    )
    if failures and not args.continue_on_error:
        raise SystemExit(1)
    print(f"Superposition diagnostic complete: {len(sample_records)} samples, {len(failures)} failures")
    print(f"Output: {args.out_dir}")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def select_rows(
    rows: list[dict[str, str]],
    *,
    sample_uids: list[str] | None,
    cases: list[str] | None,
    samples_per_case: int,
    seed: int,
) -> list[dict[str, str]]:
    by_uid = {row["sample_uid"]: row for row in rows}
    if sample_uids:
        missing = [uid for uid in sample_uids if uid not in by_uid]
        if missing:
            raise ValueError(f"sample_uid(s) not found in index: {', '.join(missing)}")
        return [by_uid[uid] for uid in sample_uids]
    if not cases:
        raise ValueError("Provide --sample-uids or --cases.")
    if samples_per_case <= 0:
        raise ValueError("--samples-per-case must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"]].append(row)
    for case_id in cases:
        candidates = sorted(rows_by_case.get(case_id, []), key=lambda item: item["sample_uid"])
        if not candidates:
            raise ValueError(f"case has no rows in index: {case_id}")
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_case])
    return selected


def estimate_hotspot_runs(
    rows: list[dict[str, str]],
    *,
    run_zero_power: bool,
    power_scales: list[float] | None,
) -> dict[str, Any]:
    isolated = 0
    chiplet_counts: dict[str, int] = {}
    for row in rows:
        source_dir = source_dir_for_row(row)
        layout = load_json(source_dir / "layout.json")
        count = len(layout.get("chiplets", []))
        chiplet_counts[row["sample_uid"]] = count
        isolated += count
    scale_count = 0
    if power_scales:
        scale_count = len(rows) * len(power_scales)
    return {
        "selected_samples": len(rows),
        "chiplet_counts": chiplet_counts,
        "isolated_runs": isolated,
        "full_power_runs": 0,
        "zero_power_runs": len(rows) if run_zero_power else 0,
        "power_scaling_runs": scale_count,
        "total_hotspot_runs": isolated + (len(rows) if run_zero_power else 0) + scale_count,
    }


def print_selection_plan(rows: list[dict[str, str]], plan: dict[str, Any]) -> None:
    print("Selected samples:")
    for row in rows:
        print(f"  {row['case_id']} {row['sample_uid']} chiplets={plan['chiplet_counts'][row['sample_uid']]}")
    print("Estimated HotSpot runs:")
    print(f"  isolated: {plan['isolated_runs']}")
    print(f"  full-power reused: {plan['full_power_runs']}")
    print(f"  zero-power: {plan['zero_power_runs']}")
    print(f"  power-scaling: {plan['power_scaling_runs']}")
    print(f"  total: {plan['total_hotspot_runs']}")


def process_sample(
    row: dict[str, str],
    *,
    out_dir: Path,
    hotspot_home: Path | None,
    config_template: Path,
    run_zero_power: bool,
    power_scales: list[float],
    power_scale_source_index: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source_dir = source_dir_for_row(row)
    layout = load_json(source_dir / "layout.json")
    package = load_yaml(source_dir / "package.yaml")
    power = load_yaml(source_dir / "power.yaml")
    hotspot = load_yaml(source_dir / "hotspot.yaml")
    chiplets = list(layout.get("chiplets", []))
    if not chiplets:
        raise ValueError(f"{source_dir / 'layout.json'} has no chiplets")
    powers = active_power_map(power)
    missing = [str(chiplet["name"]) for chiplet in chiplets if str(chiplet["name"]) not in powers]
    if missing:
        raise ValueError(f"power.yaml missing chiplet powers: {', '.join(missing)}")

    case_id = row["case_id"]
    sample_uid = row["sample_uid"]
    sample_dir = out_dir / case_id / sample_uid
    if overwrite and sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    full_temperature = np.load(row["y_path"]).astype(np.float64, copy=False)
    np.save(sample_dir / "full_temperature.npy", full_temperature.astype(np.float32))
    original_power_path = sample_dir / "original_power.json"
    original_power_path.write_text(json.dumps(powers, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ambient = float(package["ambient_K"])
    baseline = np.full_like(full_temperature, ambient, dtype=np.float64)
    zero_run: SourceRun | None = None
    if run_zero_power:
        zero_power = zero_power_map(powers)
        zero_run = run_power_case(
            source_dir=source_dir,
            output_run_dir=sample_dir / "runs" / "zero_power",
            modified_power=modified_power_yaml(power, zero_power),
            hotspot_home=hotspot_home,
            config_template=config_template,
            resume=resume,
            overwrite=False,
            expected_shape=full_temperature.shape,
        )
        baseline = np.load(zero_run.temperature_path).astype(np.float64, copy=False)
        np.save(sample_dir / "zero_temperature.npy", baseline.astype(np.float32))

    isolated_runs: list[SourceRun] = []
    isolated_fields: list[np.ndarray] = []
    per_source_records: list[dict[str, Any]] = []
    for index, chiplet in enumerate(chiplets):
        name = str(chiplet["name"])
        source_powers = isolated_power_map(powers, name, scale=1.0)
        run = run_power_case(
            source_dir=source_dir,
            output_run_dir=sample_dir / "runs" / f"source_{index:03d}_{safe_name(name)}",
            modified_power=modified_power_yaml(power, source_powers),
            hotspot_home=hotspot_home,
            config_template=config_template,
            resume=resume,
            overwrite=False,
            expected_shape=full_temperature.shape,
        )
        field = np.load(run.temperature_path).astype(np.float64, copy=False)
        np.save(sample_dir / f"source_{index:03d}_{safe_name(name)}_temperature.npy", field.astype(np.float32))
        isolated_runs.append(run)
        isolated_fields.append(field)
        per_source_records.append(
            {
                "source_index": index,
                "chiplet_name": name,
                "power_W": float(powers[name]),
                "temperature_path": str((sample_dir / f"source_{index:03d}_{safe_name(name)}_temperature.npy").relative_to(out_dir)),
                "runtime_s": run.runtime_s,
                "skipped": run.skipped,
            }
        )

    reconstruction = reconstruct_from_isolated(baseline, isolated_fields)
    error = reconstruction - full_temperature
    np.save(sample_dir / "reconstructed_temperature.npy", reconstruction.astype(np.float32))
    np.save(sample_dir / "error_map.npy", error.astype(np.float32))

    metrics = sample_metrics(
        reconstruction,
        full_temperature,
        row=row,
        layout=layout,
        hotspot=hotspot,
        x_path=Path(row["x_path"]),
        baseline=baseline,
        ambient_K=ambient,
        zero_run=zero_run,
        isolated_runs=isolated_runs,
    )
    metrics["sources"] = per_source_records
    metrics["full_temperature_path"] = row["y_path"]
    metrics["baseline_kind"] = "zero_power_hotspot" if zero_run is not None else "uniform_ambient"
    (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    draw_sample_panel(
        sample_dir / "visualization.png",
        row=row,
        layout=layout,
        full_temperature=full_temperature,
        reconstruction=reconstruction,
        error=error,
    )

    scale_records: list[dict[str, Any]] = []
    if power_scales:
        scale_records = run_power_scaling(
            row=row,
            source_dir=source_dir,
            sample_dir=sample_dir,
            out_dir=out_dir,
            chiplets=chiplets,
            powers=powers,
            base_power_yaml=power,
            baseline=baseline,
            reference_field=isolated_fields[power_scale_source_index],
            source_index=power_scale_source_index,
            scales=power_scales,
            hotspot_home=hotspot_home,
            config_template=config_template,
            resume=resume,
            expected_shape=full_temperature.shape,
        )

    return {"sample_metrics": flatten_sample_metrics(metrics), "power_scaling": scale_records}


def run_power_scaling(
    *,
    row: dict[str, str],
    source_dir: Path,
    sample_dir: Path,
    out_dir: Path,
    chiplets: list[dict[str, Any]],
    powers: dict[str, float],
    base_power_yaml: dict[str, Any],
    baseline: np.ndarray,
    reference_field: np.ndarray,
    source_index: int,
    scales: list[float],
    hotspot_home: Path | None,
    config_template: Path,
    resume: bool,
    expected_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    if source_index < 0 or source_index >= len(chiplets):
        raise ValueError(f"power_scale_source_index {source_index} is out of range for {len(chiplets)} chiplets")
    chiplet_name = str(chiplets[source_index]["name"])
    reference_delta = reference_field - baseline
    records: list[dict[str, Any]] = []
    for scale in scales:
        source_powers = isolated_power_map(powers, chiplet_name, scale=float(scale))
        run = run_power_case(
            source_dir=source_dir,
            output_run_dir=sample_dir / "runs" / f"scale_{scale:g}_source_{source_index:03d}_{safe_name(chiplet_name)}",
            modified_power=modified_power_yaml(base_power_yaml, source_powers),
            hotspot_home=hotspot_home,
            config_template=config_template,
            resume=resume,
            overwrite=False,
            expected_shape=expected_shape,
        )
        field = np.load(run.temperature_path).astype(np.float64, copy=False)
        np.save(sample_dir / f"scale_{scale:g}_source_{source_index:03d}_{safe_name(chiplet_name)}_temperature.npy", field.astype(np.float32))
        error = (field - baseline) - float(scale) * reference_delta
        metrics = field_metrics(field - baseline, float(scale) * reference_delta)
        record = {
            "case_id": row["case_id"],
            "sample_uid": row["sample_uid"],
            "source_index": source_index,
            "chiplet_name": chiplet_name,
            "scale": float(scale),
            "runtime_s": run.runtime_s,
            "skipped": run.skipped,
            "scaling_mae_K": metrics["mae_K"],
            "scaling_rmse_K": metrics["rmse_K"],
            "scaling_max_abs_error_K": metrics["max_abs_error_K"],
            "scaling_mean_signed_error_K": metrics["mean_signed_error_K"],
        }
        records.append(record)
        np.save(sample_dir / f"scale_{scale:g}_source_{source_index:03d}_{safe_name(chiplet_name)}_linearity_error.npy", error.astype(np.float32))
    return records


def run_power_case(
    *,
    source_dir: Path,
    output_run_dir: Path,
    modified_power: dict[str, Any],
    hotspot_home: Path | None,
    config_template: Path,
    resume: bool,
    overwrite: bool,
    expected_shape: tuple[int, int],
) -> SourceRun:
    temp_path = output_run_dir / "parsed" / "temp_layer0.npy"
    manifest_path = output_run_dir / "manifest.json"
    if resume and not overwrite and valid_run_output(temp_path, manifest_path, expected_shape):
        return SourceRun(output_run_dir.name, output_run_dir, temp_path, runtime_from_manifest(manifest_path), True)
    if overwrite and output_run_dir.exists():
        shutil.rmtree(output_run_dir)
    scenario_dir = output_run_dir / "input_source"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    copy_source_with_power(source_dir, scenario_dir, modified_power)

    try:
        run_hotspot_without_positive_power_validation(
            scenario_path=scenario_dir / "scenario.yaml",
            out_dir=output_run_dir,
            hotspot_home=hotspot_home,
            config_template=config_template,
        )
    except HotSpotRunError:
        raise
    except Exception as exc:
        raise HotSpotRunError(f"diagnostic HotSpot run failed: {exc}", run_dir=output_run_dir) from exc
    if not valid_run_output(temp_path, manifest_path, expected_shape):
        raise HotSpotRunError(f"HotSpot diagnostic run completed but output is invalid: {output_run_dir}", run_dir=output_run_dir)
    return SourceRun(output_run_dir.name, output_run_dir, temp_path, runtime_from_manifest(manifest_path), False)


def run_hotspot_without_positive_power_validation(
    *,
    scenario_path: Path,
    out_dir: Path,
    hotspot_home: Path | None,
    config_template: Path,
) -> None:
    total_start = time.perf_counter()
    runtime: dict[str, float] = {}
    source_dir = out_dir / "source"
    hotspot_dir = out_dir / "hotspot"
    outputs_dir = out_dir / "outputs"
    parsed_dir = out_dir / "parsed"
    for path in (source_dir, hotspot_dir, outputs_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    sim = load_simulation_input(scenario_path)
    runtime["load_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    validate_diagnostic_simulation_input(sim)
    runtime["diagnostic_validate_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    scenario_source_paths = [
        scenario_path,
        sim.scenario.layout_path.resolve(),
        sim.scenario.power_path.resolve(),
        sim.scenario.package_path.resolve(),
        sim.scenario.hotspot_path.resolve(),
    ]
    if sim.scenario.benchmark_path is not None:
        scenario_source_paths.append(sim.scenario.benchmark_path.resolve())
    for source_path in scenario_source_paths:
        shutil.copyfile(source_path, source_dir / source_path.name)

    flp_path = write_flp(sim.layout, hotspot_dir / "chiplet.flp")
    ptrace_path = write_ptrace(sim.layout, sim.power, hotspot_dir / "power.ptrace")
    config_path = write_hotspot_config(config_template, hotspot_dir / "hotspot.config", sim.package, sim.hotspot)
    rows, cols = read_grid_shape(config_path)
    runtime["write_inputs_s"] = time.perf_counter() - stage_start

    block_steady_path = outputs_dir / "block.steady"
    grid_steady_path = outputs_dir / "grid.steady"
    home = (hotspot_home or resolve_hotspot_home()).resolve()
    command = build_hotspot_command(
        hotspot_home=home,
        config_path=config_path.resolve(),
        flp_path=flp_path.resolve(),
        ptrace_path=ptrace_path.resolve(),
        steady_path=block_steady_path.resolve(),
        grid_steady_path=grid_steady_path.resolve(),
    )
    command_text = shlex.join(command)
    (out_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    (out_dir / "diagnostic_command.json").write_text(
        json.dumps({"command": command, "command_string": command_text}, indent=2) + "\n",
        encoding="utf-8",
    )

    stage_start = time.perf_counter()
    result = run_hotspot(command, cwd=hotspot_dir)
    runtime["hotspot_s"] = time.perf_counter() - stage_start
    (outputs_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (outputs_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (out_dir / "diagnostic_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (out_dir / "diagnostic_stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise HotSpotRunError(
            f"HotSpot failed with exit code {result.returncode}. See {outputs_dir / 'stderr.txt'}",
            run_dir=out_dir,
            returncode=result.returncode,
        )

    stage_start = time.perf_counter()
    layer0 = parse_layer_grid(grid_steady_path, layer=0, rows=rows, cols=cols)
    np.save(parsed_dir / "temp_layer0.npy", layer0)
    chiplet_names = tuple(chiplet.name for chiplet in sim.layout.chiplets)
    block_temps = parse_block_temps(block_steady_path, names=chiplet_names)
    (parsed_dir / "block_temps.json").write_text(json.dumps(block_temps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime["parse_s"] = time.perf_counter() - stage_start
    runtime["total_s"] = time.perf_counter() - total_start

    hottest_block, max_block_temp = max(block_temps.items(), key=lambda item: item[1])
    output_summary = {
        "temp_layer0_shape": list(layer0.shape),
        "temp_layer0_min_K": float(layer0.min()),
        "temp_layer0_max_K": float(layer0.max()),
        "temp_layer0_mean_K": float(layer0.mean()),
        "max_block_temperature_K": float(max_block_temp),
        "hottest_block": hottest_block,
        "grid_rows": rows,
        "grid_cols": cols,
    }
    write_manifest(
        out_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_name": sim.scenario.name,
            "runtime": runtime,
            "diagnostic_validation": "canonical positive-power checks bypassed; names, geometry, finite nonnegative powers, units, and required files still checked",
            "hotspot": {
                "home": str(home),
                "binary": command[0],
                "command": command,
                "command_string": command_text,
                "return_code": result.returncode,
            },
            "return_code": result.returncode,
            "grid": {"rows": rows, "cols": cols},
            "output_summary": output_summary,
            "sources": {path.name: sha256_file(path) for path in scenario_source_paths},
            "generated": {
                "flp": str(flp_path.relative_to(out_dir)),
                "ptrace": str(ptrace_path.relative_to(out_dir)),
                "config": str(config_path.relative_to(out_dir)),
                "block_steady": str(block_steady_path.relative_to(out_dir)),
                "grid_steady": str(grid_steady_path.relative_to(out_dir)),
                "temp_layer0": str((parsed_dir / "temp_layer0.npy").relative_to(out_dir)),
                "block_temps": str((parsed_dir / "block_temps.json").relative_to(out_dir)),
            },
        },
    )


def validate_diagnostic_simulation_input(sim: SimulationInput) -> None:
    errors: list[str] = []
    _validate_layout(sim.layout, errors, min_spacing_mm=_min_spacing_mm(sim))
    if sim.scenario.schema_version != 1:
        errors.append(f"scenario.schema_version must be 1, got {sim.scenario.schema_version}")
    if sim.package.schema_version != 1:
        errors.append(f"package.schema_version must be 1, got {sim.package.schema_version}")
    if sim.hotspot.schema_version != 1:
        errors.append(f"hotspot.schema_version must be 1, got {sim.hotspot.schema_version}")
    if sim.power.schema_version != 1:
        errors.append(f"power.schema_version must be 1, got {sim.power.schema_version}")
    if sim.power.units_power != "W":
        errors.append("power.units.power must be 'W'")
    if sim.power.mode != "fixed":
        errors.append("power.mode must be 'fixed'")
    if sim.power.active_workload != "nominal":
        errors.append("diagnostic isolated-source runs require active_workload='nominal'")
    if sim.power.workloads is None or "nominal" not in sim.power.workloads:
        errors.append("power.workloads.nominal is required")

    layout_names = {chiplet.name for chiplet in sim.layout.chiplets}
    chiplet_names = set(sim.power.chiplet_watts)
    nominal = sim.power.workloads.get("nominal", {}) if sim.power.workloads is not None else {}
    nominal_names = set(nominal)
    if chiplet_names != layout_names:
        errors.append(f"top-level power chiplet names must match layout names; missing={sorted(layout_names - chiplet_names)}, extra={sorted(chiplet_names - layout_names)}")
    if nominal_names != layout_names:
        errors.append(f"nominal workload names must match layout names; missing={sorted(layout_names - nominal_names)}, extra={sorted(nominal_names - layout_names)}")
    for name in sorted(layout_names):
        top_value = sim.power.chiplet_watts.get(name)
        nominal_value = nominal.get(name)
        if top_value is None or nominal_value is None:
            continue
        if not math.isfinite(top_value) or top_value < 0.0:
            errors.append(f"power.chiplets.{name} must be finite and nonnegative for diagnostics")
        if not math.isfinite(nominal_value) or nominal_value < 0.0:
            errors.append(f"power.workloads.nominal.{name} must be finite and nonnegative for diagnostics")
        if abs(float(top_value) - float(nominal_value)) > 1.0e-9:
            errors.append(f"power.chiplets.{name} must match power.workloads.nominal.{name}")

    for option, value in sim.package.options.items():
        if not math.isfinite(value) or value <= 0.0:
            errors.append(f"package option {option} must be positive and finite")
    if sim.hotspot.model_type != "grid":
        errors.append("hotspot.model_type must be 'grid'")
    if sim.hotspot.grid_rows <= 0 or sim.hotspot.grid_cols <= 0:
        errors.append("hotspot grid rows and cols must be positive")
    if sim.hotspot.grid_map_mode not in {"avg", "min", "max", "center"}:
        errors.append("hotspot.grid.map_mode must be one of avg/min/max/center")
    if sim.hotspot.leakage_used:
        errors.append("diagnostic expects leakage_used=false to test linear power superposition")
    if errors:
        raise ValueError("\n".join(errors))


def valid_run_output(temp_path: Path, manifest_path: Path, expected_shape: tuple[int, int]) -> bool:
    if not temp_path.exists() or not manifest_path.exists():
        return False
    try:
        arr = np.load(temp_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return tuple(arr.shape) == tuple(expected_shape) and np.isfinite(arr).all() and int(manifest.get("return_code", 0)) == 0


def runtime_from_manifest(manifest_path: Path) -> float | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("runtime", {}).get("hotspot_s")
        return float(value) if value is not None else None
    except Exception:
        return None


def copy_source_with_power(source_dir: Path, destination: Path, modified_power: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("scenario.yaml", "layout.json", "package.yaml", "hotspot.yaml", "benchmark.yaml"):
        source = source_dir / name
        if source.exists():
            shutil.copyfile(source, destination / name)
    (destination / "power.yaml").write_text(yaml.safe_dump(modified_power, sort_keys=False), encoding="utf-8")


def modified_power_yaml(original: dict[str, Any], chiplet_values: dict[str, float]) -> dict[str, Any]:
    data = json.loads(json.dumps(original))
    values = {str(name): float(value) for name, value in chiplet_values.items()}
    data["mode"] = data.get("mode", "fixed")
    data["active_workload"] = "nominal"
    data["chiplets"] = dict(values)
    workloads = data.get("workloads")
    if not isinstance(workloads, dict):
        workloads = {}
        data["workloads"] = workloads
    workloads["nominal"] = dict(values)
    return data


def zero_power_map(powers: dict[str, float]) -> dict[str, float]:
    return {name: 0.0 for name in powers}


def isolated_power_map(powers: dict[str, float], source_name: str, *, scale: float = 1.0) -> dict[str, float]:
    if source_name not in powers:
        raise KeyError(f"unknown source chiplet: {source_name}")
    return {name: (float(power) * float(scale) if name == source_name else 0.0) for name, power in powers.items()}


def reconstruct_from_isolated(baseline: np.ndarray, isolated_fields: list[np.ndarray]) -> np.ndarray:
    result = baseline.astype(np.float64, copy=True)
    for field in isolated_fields:
        result += field.astype(np.float64, copy=False) - baseline
    return result


def sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    row: dict[str, str],
    layout: dict[str, Any],
    hotspot: dict[str, Any],
    x_path: Path,
    baseline: np.ndarray,
    ambient_K: float,
    zero_run: SourceRun | None,
    isolated_runs: list[SourceRun],
) -> dict[str, Any]:
    package_size = layout["package"]["size"]
    width_mm = float(package_size["width"])
    height_mm = float(package_size["height"])
    x_tensor = np.load(x_path)
    occupancy = x_tensor[1] > 0.5 if x_tensor.ndim == 3 and x_tensor.shape[0] >= 2 else occupancy_from_layout(layout, target.shape)
    boundary = boundary_mask(occupancy)
    abs_error = np.abs(pred - target)
    base = field_metrics(pred, target)
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    target_hotspot = np.unravel_index(int(np.argmax(target)), target.shape)
    chiplet = chiplet_metrics(pred, target, layout, target.shape)
    runtimes = [run.runtime_s for run in isolated_runs if run.runtime_s is not None]
    zero_stats = None
    if zero_run is not None:
        zero_error = baseline - ambient_K
        zero_stats = {
            "zero_power_runtime_s": zero_run.runtime_s,
            "zero_power_mean_minus_ambient_K": float(zero_error.mean()),
            "zero_power_max_abs_minus_ambient_K": float(np.abs(zero_error).max()),
        }
    return {
        "case_id": row["case_id"],
        "sample_uid": row["sample_uid"],
        "num_chiplets": len(layout.get("chiplets", [])),
        "total_power_W": float(row.get("total_power_W") or sum(active_power_map(load_yaml(source_dir_for_row(row) / "power.yaml")).values())),
        "package_width_mm": width_mm,
        "package_height_mm": height_mm,
        "grid_rows": int(hotspot.get("grid", {}).get("rows", target.shape[0])),
        "grid_cols": int(hotspot.get("grid", {}).get("cols", target.shape[1])),
        "ambient_K": float(ambient_K),
        "full_grid": base,
        "occupied_mae_K": masked_mean(abs_error, occupancy),
        "unoccupied_mae_K": masked_mean(abs_error, ~occupancy),
        "boundary_mae_K": masked_mean(abs_error, boundary),
        "nonboundary_mae_K": masked_mean(abs_error, ~boundary),
        "hotspot_temp_error_K": float(pred[pred_hotspot] - target[target_hotspot]),
        "hotspot_location_error_cells": float(math.hypot(pred_hotspot[0] - target_hotspot[0], pred_hotspot[1] - target_hotspot[1])),
        "chiplet_mean_temperature_mae_K": chiplet["chiplet_mean_temperature_mae_K"],
        "chiplet_peak_temperature_mae_K": chiplet["chiplet_peak_temperature_mae_K"],
        "inter_chiplet_delta_T_mae_K": chiplet["inter_chiplet_delta_T_mae_K"],
        "isolated_hotspot_runtime_sum_s": float(sum(runtimes)) if runtimes else None,
        "isolated_hotspot_runtime_mean_s": float(np.mean(runtimes)) if runtimes else None,
        "zero_power": zero_stats,
    }


def field_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    return {
        "mae_K": float(np.abs(error).mean()),
        "rmse_K": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_K": float(np.abs(error).max()),
        "mean_signed_error_K": float(error.mean()),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return None
    return float(np.asarray(values)[mask].mean())


def boundary_mask(occupancy: np.ndarray) -> np.ndarray:
    occ = np.asarray(occupancy, dtype=bool)
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (occ != up) | (occ != down) | (occ != left) | (occ != right)


def occupancy_from_layout(layout: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    package = layout["package"]["size"]
    width_mm = float(package["width"])
    height_mm = float(package["height"])
    x_coords = (np.arange(cols, dtype=np.float64) + 0.5) / cols * width_mm
    y_coords = (np.arange(rows, dtype=np.float64) + 0.5) / rows * height_mm
    occupancy = np.zeros((rows, cols), dtype=bool)
    for chiplet in layout.get("chiplets", []):
        pos = chiplet["position"]
        size = chiplet["size"]
        left = float(pos["x"])
        bottom = float(pos["y"])
        right = left + float(size["width"])
        top = bottom + float(size["height"])
        mask = np.outer((y_coords >= bottom) & (y_coords < top), (x_coords >= left) & (x_coords < right))
        occupancy |= mask
    return occupancy


def chiplet_metrics(pred: np.ndarray, target: np.ndarray, layout: dict[str, Any], shape: tuple[int, int]) -> dict[str, float | None]:
    rows, cols = shape
    package = layout["package"]["size"]
    width_mm = float(package["width"])
    height_mm = float(package["height"])
    x_coords = (np.arange(cols, dtype=np.float64) + 0.5) / cols * width_mm
    y_coords = (np.arange(rows, dtype=np.float64) + 0.5) / rows * height_mm
    pred_means: list[float] = []
    target_means: list[float] = []
    pred_peaks: list[float] = []
    target_peaks: list[float] = []
    for chiplet in layout.get("chiplets", []):
        pos = chiplet["position"]
        size = chiplet["size"]
        left = float(pos["x"])
        bottom = float(pos["y"])
        chiplet_width = float(size["width"])
        chiplet_height = float(size["height"])
        mask = np.outer(
            (y_coords >= bottom) & (y_coords < bottom + chiplet_height),
            (x_coords >= left) & (x_coords < left + chiplet_width),
        )
        if not bool(mask.any()):
            center_x = left + 0.5 * chiplet_width
            center_y = bottom + 0.5 * chiplet_height
            col = int(np.clip(math.floor(center_x / max(width_mm, 1.0e-12) * cols), 0, cols - 1))
            row = int(np.clip(math.floor(center_y / max(height_mm, 1.0e-12) * rows), 0, rows - 1))
            mask[row, col] = True
        pred_means.append(float(pred[mask].mean()))
        target_means.append(float(target[mask].mean()))
        pred_peaks.append(float(pred[mask].max()))
        target_peaks.append(float(target[mask].max()))
    if not pred_means:
        return {
            "chiplet_mean_temperature_mae_K": None,
            "chiplet_peak_temperature_mae_K": None,
            "inter_chiplet_delta_T_mae_K": None,
        }
    pred_mean_arr = np.asarray(pred_means)
    target_mean_arr = np.asarray(target_means)
    delta_mae = None
    if len(pred_mean_arr) >= 2:
        pred_deltas = []
        target_deltas = []
        for i in range(len(pred_mean_arr)):
            for j in range(i + 1, len(pred_mean_arr)):
                pred_deltas.append(pred_mean_arr[i] - pred_mean_arr[j])
                target_deltas.append(target_mean_arr[i] - target_mean_arr[j])
        delta_mae = float(np.mean(np.abs(np.asarray(pred_deltas) - np.asarray(target_deltas))))
    return {
        "chiplet_mean_temperature_mae_K": float(np.mean(np.abs(pred_mean_arr - target_mean_arr))),
        "chiplet_peak_temperature_mae_K": float(np.mean(np.abs(np.asarray(pred_peaks) - np.asarray(target_peaks)))),
        "inter_chiplet_delta_T_mae_K": delta_mae,
    }


def flatten_sample_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "case_id": metrics["case_id"],
        "sample_uid": metrics["sample_uid"],
        "num_chiplets": metrics["num_chiplets"],
        "total_power_W": metrics["total_power_W"],
        "package_width_mm": metrics["package_width_mm"],
        "package_height_mm": metrics["package_height_mm"],
        "full_grid_mae_K": metrics["full_grid"]["mae_K"],
        "full_grid_rmse_K": metrics["full_grid"]["rmse_K"],
        "full_grid_max_abs_error_K": metrics["full_grid"]["max_abs_error_K"],
        "full_grid_mean_signed_error_K": metrics["full_grid"]["mean_signed_error_K"],
        "occupied_mae_K": metrics["occupied_mae_K"],
        "unoccupied_mae_K": metrics["unoccupied_mae_K"],
        "boundary_mae_K": metrics["boundary_mae_K"],
        "nonboundary_mae_K": metrics["nonboundary_mae_K"],
        "chiplet_mean_temperature_mae_K": metrics["chiplet_mean_temperature_mae_K"],
        "chiplet_peak_temperature_mae_K": metrics["chiplet_peak_temperature_mae_K"],
        "inter_chiplet_delta_T_mae_K": metrics["inter_chiplet_delta_T_mae_K"],
        "hotspot_temp_error_K": metrics["hotspot_temp_error_K"],
        "hotspot_location_error_cells": metrics["hotspot_location_error_cells"],
        "isolated_hotspot_runtime_sum_s": metrics["isolated_hotspot_runtime_sum_s"],
        "isolated_hotspot_runtime_mean_s": metrics["isolated_hotspot_runtime_mean_s"],
        "baseline_kind": metrics["baseline_kind"],
    }
    zero = metrics.get("zero_power")
    if zero:
        flat.update(zero)
    return flat


def draw_sample_panel(
    path: Path,
    *,
    row: dict[str, str],
    layout: dict[str, Any],
    full_temperature: np.ndarray,
    reconstruction: np.ndarray,
    error: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    power_map = np.load(row["x_path"])[0]
    abs_error = np.abs(error)
    vmin = float(min(full_temperature.min(), reconstruction.min()))
    vmax = float(max(full_temperature.max(), reconstruction.max()))
    err_lim = float(max(abs(error.min()), abs(error.max()), 1.0e-9))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    panels = [
        ("Power density", power_map, "inferno", None),
        ("Full HotSpot", full_temperature, "hot", (vmin, vmax)),
        ("Reconstructed", reconstruction, "hot", (vmin, vmax)),
        ("Signed error", error, "coolwarm", (-err_lim, err_lim)),
        ("Absolute error", abs_error, "magma", None),
    ]
    for ax, (title, data, cmap, limits) in zip(axes, panels):
        kwargs = {"cmap": cmap, "origin": "lower"}
        if limits is not None:
            kwargs["vmin"], kwargs["vmax"] = limits
        im = ax.imshow(data, **kwargs)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{row['case_id']} {row['sample_uid']} | chiplets={len(layout.get('chiplets', []))} | "
        f"MAE={np.abs(error).mean():.4f} K | max={np.abs(error).max():.4f} K"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_outputs(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    selected: list[dict[str, str]],
    plan: dict[str, Any],
    sample_records: list[dict[str, Any]],
    power_scaling_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    total_runtime_s: float,
) -> None:
    write_csv(out_dir / "superposition_by_sample.csv", sample_records)
    by_case = aggregate_by_case(sample_records)
    write_csv(out_dir / "superposition_by_case.csv", by_case)
    if power_scaling_records:
        write_csv(out_dir / "power_scaling_summary.csv", power_scaling_records)
    (out_dir / "power_scaling_summary.json").write_text(json.dumps(power_scaling_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index": str(args.index),
        "out_dir": str(out_dir),
        "selected_samples": [{"case_id": row["case_id"], "sample_uid": row["sample_uid"]} for row in selected],
        "run_plan": plan,
        "sample_count_completed": len(sample_records),
        "failure_count": len(failures),
        "failures": failures,
        "total_runtime_s": total_runtime_s,
        "overall": aggregate_records(sample_records),
        "interpretation": interpretation(aggregate_records(sample_records).get("full_grid_mae_K")),
        "notes": {
            "reconstruction": "baseline + sum_i(T_i - baseline)",
            "baseline": "zero-power HotSpot field when --run-zero-power is used, otherwise uniform ambient_K",
            "full_temperature": "existing y_path HotSpot target is reused; no full-power HotSpot rerun",
        },
    }
    (out_dir / "superposition_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_roadmap_report(out_dir / "roadmap_report.md", summary, by_case)


def summarize_existing(out_dir: Path) -> None:
    records = []
    for metrics_path in sorted(out_dir.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        records.append(flatten_sample_metrics(metrics))
    by_case = aggregate_by_case(records)
    write_csv(out_dir / "superposition_by_sample.csv", records)
    write_csv(out_dir / "superposition_by_case.csv", by_case)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_only": True,
        "sample_count_completed": len(records),
        "overall": aggregate_records(records),
        "interpretation": interpretation(aggregate_records(records).get("full_grid_mae_K")),
    }
    (out_dir / "superposition_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_roadmap_report(out_dir / "roadmap_report.md", summary, by_case)
    print(f"Summarized {len(records)} completed sample(s) from {out_dir}")


def aggregate_by_case(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    rows = []
    for case_id in sorted(grouped):
        row = {"case_id": case_id, "sample_count": len(grouped[case_id])}
        row.update(aggregate_records(grouped[case_id]))
        rows.append(row)
    return rows


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    keys = [
        "full_grid_mae_K",
        "full_grid_rmse_K",
        "full_grid_max_abs_error_K",
        "full_grid_mean_signed_error_K",
        "occupied_mae_K",
        "unoccupied_mae_K",
        "boundary_mae_K",
        "nonboundary_mae_K",
        "chiplet_mean_temperature_mae_K",
        "chiplet_peak_temperature_mae_K",
        "inter_chiplet_delta_T_mae_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "isolated_hotspot_runtime_sum_s",
        "isolated_hotspot_runtime_mean_s",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None and str(record.get(key)) != ""]
        result[key] = float(np.mean(values)) if values else None
    result["sample_count"] = len(records)
    worst = max(records, key=lambda item: float(item["full_grid_mae_K"]))
    result["worst_sample_uid"] = worst["sample_uid"]
    result["worst_case_id"] = worst["case_id"]
    result["worst_full_grid_mae_K"] = worst["full_grid_mae_K"]
    return result


def interpretation(mae: Any) -> str:
    if mae is None:
        return "No completed samples."
    value = float(mae)
    if value < 0.1:
        return "essentially exact linear superposition"
    if value < 0.5:
        return "very strong support for source-response learning"
    if value < 1.0:
        return "useful superposition, likely with a small nonlinear residual correction"
    if value < 2.0:
        return "partial superposition; investigate configuration or nonlinearity"
    return "superposition error is large; avoid a purely additive source-response model without correction"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_roadmap_report(path: Path, summary: dict[str, Any], by_case: list[dict[str, Any]]) -> None:
    overall = summary.get("overall", {})
    lines = [
        "# ChipTherm Superposition Diagnostic",
        "",
        f"Completed samples: {summary.get('sample_count_completed', 0)}",
        f"Overall full-grid MAE: {format_optional(overall.get('full_grid_mae_K'))} K",
        f"Overall RMSE: {format_optional(overall.get('full_grid_rmse_K'))} K",
        f"Interpretation: {summary.get('interpretation', 'n/a')}",
        "",
        "## Case Summary",
        "",
        "| case | samples | MAE K | RMSE K |",
        "|---|---:|---:|---:|",
    ]
    for row in by_case:
        lines.append(
            f"| {row['case_id']} | {row['sample_count']} | "
            f"{format_optional(row.get('full_grid_mae_K'))} | {format_optional(row.get('full_grid_rmse_K'))} |"
        )
    lines.extend(
        [
            "",
            "## Roadmap Implication",
            "",
            "- If the MAE is below 0.5 K, source-isolated labels are scientifically well motivated.",
            "- If the MAE is between 0.5 K and 2 K, an additive source-response model should keep a nonlinear residual corrector.",
            "- If the MAE is above 2 K, do not force a purely additive source decomposition without revisiting HotSpot configuration and physics assumptions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def write_run_manifest(out_dir: Path, args: argparse.Namespace, selected: list[dict[str, str]], plan: dict[str, Any], *, dry_run: bool) -> None:
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "index": str(args.index),
        "out_dir": str(out_dir),
        "selected_samples": [{"case_id": row["case_id"], "sample_uid": row["sample_uid"]} for row in selected],
        "run_plan": plan,
    }
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_dir_for_row(row: dict[str, str]) -> Path:
    case_id = row["case_id"]
    original = row.get("original_sample_uid") or row["sample_uid"]
    sample_name = original
    prefix = f"{case_id}_"
    if sample_name.startswith(prefix):
        sample_name = sample_name[len(prefix) :]
    return REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_name / "source"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def assert_safe_output_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    benchmarks = (REPO_ROOT / "data/runs/benchmarks").resolve()
    try:
        resolved.relative_to(benchmarks)
    except ValueError:
        return
    raise ValueError(f"Refusing to write superposition diagnostics inside canonical benchmark dataset root: {resolved}")


if __name__ == "__main__":
    raise SystemExit(main())
