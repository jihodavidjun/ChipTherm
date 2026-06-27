#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.layout import length_scale_to_m
from chiptherm.parsers import parse_block_temps, parse_layer_grid
from chiptherm.paths import hotspot_home
from chiptherm.runner import build_hotspot_command, run_hotspot
from chiptherm.scenario import load_simulation_input
from chiptherm.validate import validate_simulation_input
from chiptherm.writers import read_grid_shape, write_flp, write_hotspot_config, write_manifest, write_ptrace


POWER_RANGES_W = {
    "CPU": (40.0, 90.0),
    "GPU": (100.0, 220.0),
    "HBM": (6.0, 18.0),
    "IO": (5.0, 25.0),
    "NPU": (40.0, 120.0),
    "DRAM": (6.0, 25.0),
    "ANALOG": (2.0, 20.0),
    "MEMS": (1.0, 12.0),
}

POWER_DENSITY_RANGES_W_PER_MM2 = {
    "CPU": (0.8, 2.4),
    "GPU": (0.7, 2.2),
    "NPU": (0.6, 2.0),
    "HBM": (0.08, 0.25),
    "DRAM": (0.08, 0.35),
    "IO": (0.08, 0.45),
    "ANALOG": (0.05, 0.35),
    "MEMS": (0.03, 0.25),
}

WORKLOAD_MULTIPLIERS = {"idle": 0.25, "nominal": 1.0, "peak": 1.25}


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    success: bool
    run_dir: str
    reason: str | None
    runtime: dict[str, float]
    output_summary: dict[str, Any] | None
    sampled_power: dict[str, dict[str, float]]
    sampled_position: dict[str, dict[str, float]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a ChipTherm HotSpot dataset.")
    parser.add_argument("--base-scenario", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-retries", default=100, type=int)
    parser.add_argument("--perturb-radius-mm", default=3.0, type=float)
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be positive")

    total_start = time.perf_counter()
    base_scenario = args.base_scenario.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_bundle = _load_base_bundle(base_scenario)
    home = (args.hotspot_home or hotspot_home()).resolve()
    config_template = args.config_template.resolve()

    results: list[SampleResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_sample,
                sample_index,
                args.seed,
                str(out_dir),
                str(base_scenario),
                base_bundle,
                str(home),
                str(config_template),
                args.max_retries,
                args.perturb_radius_mm,
            )
            for sample_index in range(1, args.num_samples + 1)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.sample_id)
    dataset_manifest = _build_dataset_manifest(
        results=results,
        base_scenario=base_scenario,
        requested=args.num_samples,
        seed=args.seed,
        workers=args.workers,
        total_runtime_s=time.perf_counter() - total_start,
    )
    write_manifest(out_dir / "dataset_manifest.json", dataset_manifest)

    print("Dataset generation complete")
    print(f"Requested: {dataset_manifest['number_requested']}")
    print(f"Successful: {dataset_manifest['number_successful']}")
    print(f"Failed: {dataset_manifest['number_failed']}")
    print(f"Total runtime: {dataset_manifest['total_runtime_s']:.3f} s")
    print(f"Avg HotSpot runtime: {_format_optional_float(dataset_manifest['average_hotspot_runtime_s'])} s")
    print(f"Output: {out_dir}")
    return 0


def _run_sample(
    sample_index: int,
    seed: int,
    out_dir_text: str,
    base_scenario_text: str,
    base_bundle: dict[str, Any],
    hotspot_home_text: str,
    config_template_text: str,
    max_retries: int,
    perturb_radius_mm: float,
) -> SampleResult:
    sample_id = f"sample_{sample_index:06d}"
    run_dir = Path(out_dir_text) / sample_id
    source_dir = run_dir / "source"
    hotspot_dir = run_dir / "hotspot"
    outputs_dir = run_dir / "outputs"
    parsed_dir = run_dir / "parsed"
    for path in (source_dir, hotspot_dir, outputs_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed + sample_index)
    sampled_power: dict[str, dict[str, float]] = {}
    sampled_position: dict[str, dict[str, float]] = {}
    runtime: dict[str, float] = {}
    total_start = time.perf_counter()

    try:
        scenario_path = source_dir / "scenario.yaml"
        last_validation_error = "sample was not attempted"
        for attempt in range(1, max_retries + 1):
            layout_data, power_data = _sample_source_data(base_bundle, rng, perturb_radius_mm)
            _write_sample_sources(
                source_dir=source_dir,
                sample_id=sample_id,
                layout_data=layout_data,
                power_data=power_data,
                package_data=base_bundle["package"],
                hotspot_data=base_bundle["hotspot"],
                benchmark_data=base_bundle.get("benchmark"),
            )
            sampled_power = _sampled_power_by_type(layout_data, power_data)
            sampled_position = _sampled_position_by_type(layout_data)

            stage_start = time.perf_counter()
            sim = load_simulation_input(scenario_path)
            runtime["load_s"] = runtime.get("load_s", 0.0) + (time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            try:
                validate_simulation_input(sim)
            except Exception as exc:
                runtime["validate_s"] = runtime.get("validate_s", 0.0) + (time.perf_counter() - stage_start)
                last_validation_error = str(exc)
                continue
            runtime["validate_s"] = runtime.get("validate_s", 0.0) + (time.perf_counter() - stage_start)
            break
        else:
            runtime["total_s"] = time.perf_counter() - total_start
            _write_sample_failure_manifest(run_dir, sample_id, base_scenario_text, runtime, last_validation_error, None, None)
            return SampleResult(sample_id, False, str(run_dir), last_validation_error, runtime, None, sampled_power, sampled_position)

        stage_start = time.perf_counter()
        flp_path = write_flp(sim.layout, hotspot_dir / "chiplet.flp")
        ptrace_path = write_ptrace(sim.layout, sim.power, hotspot_dir / "power.ptrace")
        config_path = write_hotspot_config(config_template_text, hotspot_dir / "hotspot.config", sim.package, sim.hotspot)
        rows, cols = read_grid_shape(config_path)
        runtime["write_inputs_s"] = time.perf_counter() - stage_start

        block_steady_path = outputs_dir / "block.steady"
        grid_steady_path = outputs_dir / "grid.steady"
        command = build_hotspot_command(
            hotspot_home=hotspot_home_text,
            config_path=config_path.resolve(),
            flp_path=flp_path.resolve(),
            ptrace_path=ptrace_path.resolve(),
            steady_path=block_steady_path.resolve(),
            grid_steady_path=grid_steady_path.resolve(),
        )
        command_text = shlex.join(command)
        (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

        stage_start = time.perf_counter()
        result = run_hotspot(command, cwd=hotspot_dir)
        runtime["hotspot_s"] = time.perf_counter() - stage_start
        (outputs_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (outputs_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            runtime["total_s"] = time.perf_counter() - total_start
            reason = f"HotSpot failed with return code {result.returncode}"
            _write_sample_failure_manifest(run_dir, sample_id, base_scenario_text, runtime, reason, command, result.returncode)
            return SampleResult(sample_id, False, str(run_dir), reason, runtime, None, sampled_power, sampled_position)

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
        source_paths = [source_dir / name for name in ("scenario.yaml", "layout.json", "power.yaml", "package.yaml", "hotspot.yaml")]
        if (source_dir / "benchmark.yaml").exists():
            source_paths.append(source_dir / "benchmark.yaml")
        write_manifest(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scenario_name": sample_id,
                "base_scenario": base_scenario_text,
                "runtime": runtime,
                "hotspot": {
                    "home": hotspot_home_text,
                    "binary": command[0],
                    "git_commit": _git_commit(Path(hotspot_home_text)),
                    "command": command,
                    "command_string": command_text,
                    "return_code": result.returncode,
                },
                "grid": {"rows": rows, "cols": cols},
                "output_summary": output_summary,
                "sampled_power": sampled_power,
                "sampled_position": sampled_position,
                "sources": {path.name: _sha256(path) for path in source_paths},
                "generated": {
                    "flp": str(flp_path.relative_to(run_dir)),
                    "ptrace": str(ptrace_path.relative_to(run_dir)),
                    "config": str(config_path.relative_to(run_dir)),
                    "block_steady": str(block_steady_path.relative_to(run_dir)),
                    "grid_steady": str(grid_steady_path.relative_to(run_dir)),
                    "temp_layer0": str((parsed_dir / "temp_layer0.npy").relative_to(run_dir)),
                    "block_temps": str((parsed_dir / "block_temps.json").relative_to(run_dir)),
                },
            },
        )
        return SampleResult(sample_id, True, str(run_dir), None, runtime, output_summary, sampled_power, sampled_position)
    except Exception as exc:
        runtime["total_s"] = time.perf_counter() - total_start
        reason = str(exc)
        _write_sample_failure_manifest(run_dir, sample_id, base_scenario_text, runtime, reason, None, None)
        return SampleResult(sample_id, False, str(run_dir), reason, runtime, None, sampled_power, sampled_position)


def _load_base_bundle(base_scenario: Path) -> dict[str, Any]:
    sim = load_simulation_input(base_scenario)
    validate_simulation_input(sim)
    return {
        "scenario": _load_yaml(base_scenario),
        "layout": _load_json(sim.scenario.layout_path),
        "power": _load_yaml(sim.scenario.power_path),
        "package": _load_yaml(sim.scenario.package_path),
        "hotspot": _load_yaml(sim.scenario.hotspot_path),
        "benchmark": _load_yaml(sim.scenario.benchmark_path) if sim.scenario.benchmark_path is not None else None,
    }


def _sample_source_data(
    base_bundle: dict[str, Any],
    rng: random.Random,
    perturb_radius_mm: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout_data = copy.deepcopy(base_bundle["layout"])
    power_data = copy.deepcopy(base_bundle["power"])
    unit = layout_data.get("units", {}).get("length")
    perturb_radius = (perturb_radius_mm * 1e-3) / length_scale_to_m(str(unit))
    area_scale_to_mm2 = (length_scale_to_m(str(unit)) / 1e-3) ** 2
    package_size = layout_data["package"]["size"]

    power_data["mode"] = "fixed"
    power_data["chiplets"] = {}
    has_workloads = "workloads" in power_data
    if has_workloads:
        power_data["active_workload"] = power_data.get("active_workload", "nominal")
        power_data["workloads"] = {name: {} for name in WORKLOAD_MULTIPLIERS}

    for chiplet in layout_data["chiplets"]:
        chiplet_type = chiplet["type"]
        if chiplet_type not in POWER_RANGES_W:
            raise ValueError(f"unknown chiplet type {chiplet_type!r}")

        position = chiplet["position"]
        size = chiplet["size"]
        min_x = 0.0
        min_y = 0.0
        max_x = float(package_size["width"]) - float(size["width"])
        max_y = float(package_size["height"]) - float(size["height"])
        position["x"] = _clamp(float(position["x"]) + rng.uniform(-perturb_radius, perturb_radius), min_x, max_x)
        position["y"] = _clamp(float(position["y"]) + rng.uniform(-perturb_radius, perturb_radius), min_y, max_y)

        if has_workloads:
            area_mm2 = float(size["width"]) * float(size["height"]) * area_scale_to_mm2
            density_low, density_high = POWER_DENSITY_RANGES_W_PER_MM2[chiplet_type]
            nominal = area_mm2 * rng.uniform(density_low, density_high)
            for workload, multiplier in WORKLOAD_MULTIPLIERS.items():
                power_data["workloads"][workload][chiplet["name"]] = round(nominal * multiplier, 4)
            active = str(power_data["active_workload"])
            power_data["chiplets"][chiplet["name"]] = power_data["workloads"][active][chiplet["name"]]
        else:
            low, high = POWER_RANGES_W[chiplet_type]
            power_data["chiplets"][chiplet["name"]] = round(rng.uniform(low, high), 4)

    benchmark = base_bundle.get("benchmark")
    if benchmark is not None:
        min_spacing = float(benchmark.get("generation_constraints", {}).get("min_spacing_mm", 0.1))
        if not _repack_layout(layout_data, rng, min_spacing):
            raise ValueError("could not generate valid non-overlapping benchmark placement")

    return layout_data, power_data


def _write_sample_sources(
    *,
    source_dir: Path,
    sample_id: str,
    layout_data: dict[str, Any],
    power_data: dict[str, Any],
    package_data: dict[str, Any],
    hotspot_data: dict[str, Any],
    benchmark_data: dict[str, Any] | None,
) -> None:
    files = {
        "layout": "layout.json",
        "power": "power.yaml",
        "package": "package.yaml",
        "hotspot": "hotspot.yaml",
    }
    if benchmark_data is not None:
        files["benchmark"] = "benchmark.yaml"

    scenario_data = {
        "schema_version": 1,
        "name": sample_id,
        "description": "Generated ChipTherm dataset sample.",
        "files": files,
    }
    layout_data = copy.deepcopy(layout_data)
    layout_data["package"]["name"] = sample_id

    (source_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario_data, sort_keys=False), encoding="utf-8")
    (source_dir / "layout.json").write_text(json.dumps(layout_data, indent=2) + "\n", encoding="utf-8")
    (source_dir / "power.yaml").write_text(yaml.safe_dump(power_data, sort_keys=False), encoding="utf-8")
    (source_dir / "package.yaml").write_text(yaml.safe_dump(package_data, sort_keys=False), encoding="utf-8")
    (source_dir / "hotspot.yaml").write_text(yaml.safe_dump(hotspot_data, sort_keys=False), encoding="utf-8")
    if benchmark_data is not None:
        benchmark_data = copy.deepcopy(benchmark_data)
        benchmark_data["sample_id"] = sample_id
        (source_dir / "benchmark.yaml").write_text(yaml.safe_dump(benchmark_data, sort_keys=False), encoding="utf-8")


def _build_dataset_manifest(
    *,
    results: list[SampleResult],
    base_scenario: Path,
    requested: int,
    seed: int,
    workers: int,
    total_runtime_s: float,
) -> dict[str, Any]:
    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]
    hotspot_times = [result.runtime.get("hotspot_s") for result in successes if "hotspot_s" in result.runtime]
    sample_times = [result.runtime.get("total_s") for result in successes if "total_s" in result.runtime]
    layer_mins = [result.output_summary["temp_layer0_min_K"] for result in successes if result.output_summary]
    layer_maxs = [result.output_summary["temp_layer0_max_K"] for result in successes if result.output_summary]
    layer_means = [result.output_summary["temp_layer0_mean_K"] for result in successes if result.output_summary]

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_scenario": str(base_scenario),
        "number_requested": requested,
        "number_successful": len(successes),
        "number_failed": len(failures),
        "seed": seed,
        "workers": workers,
        "total_runtime_s": total_runtime_s,
        "average_hotspot_runtime_s": _mean_or_none(hotspot_times),
        "average_total_sample_runtime_s": _mean_or_none(sample_times),
        "temperature_K": {
            "min": min(layer_mins) if layer_mins else None,
            "max": max(layer_maxs) if layer_maxs else None,
            "mean_of_sample_means": _mean_or_none(layer_means),
        },
        "sampled_power_ranges_by_type_W": _ranges_by_type(result.sampled_power for result in successes),
        "sampled_position_ranges_by_type": _position_ranges_by_type(result.sampled_position for result in successes),
        "failed_samples": [{"sample_id": result.sample_id, "reason": result.reason} for result in failures],
        "samples": [
            {
                "sample_id": result.sample_id,
                "success": result.success,
                "run_dir": result.run_dir,
                "reason": result.reason,
                "runtime": result.runtime,
            }
            for result in results
        ],
        "verification": _verification_summary(results, base_scenario),
    }


def _verification_summary(results: list[SampleResult], base_scenario: Path) -> dict[str, Any]:
    sim = load_simulation_input(base_scenario)
    base_layout_hash = _sha256(sim.scenario.layout_path)
    base_power_hash = _sha256(sim.scenario.power_path)
    problems: list[str] = []

    for result in results:
        run_dir = Path(result.run_dir)
        if "examples" in run_dir.parts:
            problems.append(f"{result.sample_id}: run directory is under examples")
        if not result.success:
            continue
        layout_path = run_dir / "source" / "layout.json"
        power_path = run_dir / "source" / "power.yaml"
        temp_path = run_dir / "parsed" / "temp_layer0.npy"
        if not temp_path.exists():
            problems.append(f"{result.sample_id}: missing parsed/temp_layer0.npy")
        if _sha256(layout_path) == base_layout_hash:
            problems.append(f"{result.sample_id}: source/layout.json matches base layout")
        if _sha256(power_path) == base_power_hash:
            problems.append(f"{result.sample_id}: source/power.yaml matches base power")
        try:
            validate_simulation_input(load_simulation_input(run_dir / "source" / "scenario.yaml"))
        except Exception as exc:
            problems.append(f"{result.sample_id}: validation failed after generation: {exc}")

    return {
        "passed": not problems,
        "problems": problems,
    }


def _write_sample_failure_manifest(
    run_dir: Path,
    sample_id: str,
    base_scenario: str,
    runtime: dict[str, float],
    reason: str,
    command: list[str] | None,
    return_code: int | None,
) -> None:
    write_manifest(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_name": sample_id,
            "base_scenario": base_scenario,
            "success": False,
            "reason": reason,
            "runtime": runtime,
            "hotspot": {
                "command": command,
                "command_string": shlex.join(command) if command else None,
                "return_code": return_code,
            },
        },
    )


def _sampled_power_by_type(layout_data: dict[str, Any], power_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    by_type: dict[str, dict[str, float]] = {}
    for chiplet in layout_data["chiplets"]:
        by_type[chiplet["name"]] = {
            "type": chiplet["type"],
            "watts": float(power_data["chiplets"][chiplet["name"]]),
        }
    return by_type


def _sampled_position_by_type(layout_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    by_type: dict[str, dict[str, float]] = {}
    for chiplet in layout_data["chiplets"]:
        by_type[chiplet["name"]] = {
            "type": chiplet["type"],
            "x": float(chiplet["position"]["x"]),
            "y": float(chiplet["position"]["y"]),
        }
    return by_type


def _ranges_by_type(samples: Any) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for sample in samples:
        for item in sample.values():
            values.setdefault(item["type"], []).append(float(item["watts"]))
    return {key: {"min": min(vals), "max": max(vals)} for key, vals in values.items()}


def _position_ranges_by_type(samples: Any) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        for item in sample.values():
            bucket = values.setdefault(item["type"], {"x": [], "y": []})
            bucket["x"].append(float(item["x"]))
            bucket["y"].append(float(item["y"]))
    return {
        key: {
            "x_min": min(vals["x"]),
            "x_max": max(vals["x"]),
            "y_min": min(vals["y"]),
            "y_max": max(vals["y"]),
        }
        for key, vals in values.items()
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _repack_layout(layout_data: dict[str, Any], rng: random.Random, min_spacing: float) -> bool:
    package_size = layout_data["package"]["size"]
    width = float(package_size["width"])
    height = float(package_size["height"])
    chiplets = copy.deepcopy(layout_data["chiplets"])
    packed = _random_pack(chiplets, rng, width, height, min_spacing) or _shelf_pack(chiplets, width, height, min_spacing)
    if packed is None:
        return False

    by_name = {chiplet["name"]: chiplet for chiplet in packed}
    for chiplet in layout_data["chiplets"]:
        packed_chiplet = by_name[chiplet["name"]]
        chiplet["position"]["x"] = round(float(packed_chiplet["position"]["x"]), 6)
        chiplet["position"]["y"] = round(float(packed_chiplet["position"]["y"]), 6)
    return True


def _random_pack(
    chiplets: list[dict[str, Any]],
    rng: random.Random,
    width: float,
    height: float,
    min_spacing: float,
) -> list[dict[str, Any]] | None:
    placed: list[dict[str, Any]] = []
    for chiplet in sorted(chiplets, key=lambda item: item["size"]["width"] * item["size"]["height"], reverse=True):
        chiplet_width = float(chiplet["size"]["width"])
        chiplet_height = float(chiplet["size"]["height"])
        if chiplet_width > width or chiplet_height > height:
            return None
        for _ in range(500):
            candidate = copy.deepcopy(chiplet)
            candidate["position"]["x"] = rng.uniform(0.0, width - chiplet_width)
            candidate["position"]["y"] = rng.uniform(0.0, height - chiplet_height)
            if all(_spacing(candidate, other) >= min_spacing for other in placed):
                placed.append(candidate)
                break
        else:
            return None
    return placed


def _shelf_pack(
    chiplets: list[dict[str, Any]],
    width: float,
    height: float,
    min_spacing: float,
) -> list[dict[str, Any]] | None:
    placed: list[dict[str, Any]] = []
    x = 0.0
    y = 0.0
    shelf_height = 0.0
    for chiplet in sorted(chiplets, key=lambda item: item["size"]["height"], reverse=True):
        chiplet_width = float(chiplet["size"]["width"])
        chiplet_height = float(chiplet["size"]["height"])
        if chiplet_width > width or chiplet_height > height:
            return None
        if x > 0.0 and x + chiplet_width > width:
            x = 0.0
            y += shelf_height + min_spacing
            shelf_height = 0.0
        if y + chiplet_height > height:
            return None
        candidate = copy.deepcopy(chiplet)
        candidate["position"]["x"] = x
        candidate["position"]["y"] = y
        placed.append(candidate)
        x += chiplet_width + min_spacing
        shelf_height = max(shelf_height, chiplet_height)
    return placed


def _spacing(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_left = float(first["position"]["x"])
    first_bottom = float(first["position"]["y"])
    second_left = float(second["position"]["x"])
    second_bottom = float(second["position"]["y"])
    first_right = first_left + float(first["size"]["width"])
    first_top = first_bottom + float(first["size"]["height"])
    second_right = second_left + float(second["size"]["width"])
    second_top = second_bottom + float(second["size"]["height"])
    dx = max(first_left - second_right, second_left - first_right, 0.0)
    dy = max(first_bottom - second_top, second_bottom - first_top, 0.0)
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return (dx * dx + dy * dy) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
