#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.scenario import load_simulation_input
from chiptherm.validate import validate_simulation_input


CASES = [
    ("case01", "x32", 6, 3168, 42.0, 42.0, 0.40),
    ("case02", "x32", 6, 3520, 55.0, 52.0, 0.65),
    ("case03", "x32", 8, 8448, 39.0, 39.0, 0.60),
    ("case04", "x32", 11, 7040, 57.0, 59.0, 0.40),
    ("case05", "x32", 12, 7392, 37.0, 37.0, 0.35),
    ("case06", "x16", 20, 5632, 49.0, 53.0, 0.55),
    ("case07", "x16", 28, 2816, 30.0, 25.0, 0.55),
    ("case08", "x16", 36, 2948, 26.0, 23.0, 0.60),
    ("case09", "x16", 44, 7656, 59.0, 61.0, 0.45),
    ("case10", "x16", 61, 5280, 47.0, 47.0, 0.50),
]

UCIE = {
    "x16": {"cols": 12, "lanes": 16, "bump_pitch_um": 100, "pitch_x_um": 180, "pitch_y_um": 90},
    "x32": {"cols": 16, "lanes": 32, "bump_pitch_um": 25, "pitch_x_um": 27, "pitch_y_um": 42},
}

AREA_RANGES_MM2 = {
    "CPU": (50.0, 150.0),
    "GPU": (100.0, 400.0),
    "NPU": (60.0, 250.0),
    "HBM": (60.0, 100.0),
    "DRAM": (50.0, 120.0),
    "IO": (20.0, 100.0),
    "ANALOG": (10.0, 80.0),
    "MEMS": (5.0, 60.0),
}

ASPECT_RANGES = {
    "CPU": ((0.7, 1.4),),
    "GPU": ((0.7, 1.5),),
    "NPU": ((0.7, 1.5),),
    "HBM": ((0.6, 0.9), (1.1, 1.6)),
    "DRAM": ((0.6, 1.6),),
    "IO": ((0.5, 2.0),),
    "ANALOG": ((0.5, 2.0),),
    "MEMS": ((0.5, 2.0),),
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
MIN_SPACING_MM = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic ChipTherm benchmark cases.")
    parser.add_argument("--out-root", default=REPO_ROOT / "examples/benchmarks", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for case_index, row in enumerate(CASES, start=1):
        rng = random.Random(args.seed + case_index)
        case_id, bump_type, dies, nets, width_mm, height_mm, whitespace = row
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        layout, power, benchmark = _generate_case(
            rng=rng,
            case_id=case_id,
            case_index=case_index,
            bump_type=bump_type,
            dies=dies,
            nets=nets,
            width_mm=width_mm,
            height_mm=height_mm,
            whitespace=whitespace,
            seed=args.seed,
        )
        package = _package_yaml(width_mm, height_mm)
        hotspot = _hotspot_yaml()
        scenario = _scenario_yaml(case_id)

        _write_yaml(case_dir / "benchmark.yaml", benchmark)
        _write_yaml(case_dir / "scenario.yaml", scenario)
        (case_dir / "layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        _write_yaml(case_dir / "power.yaml", power)
        _write_yaml(case_dir / "package.yaml", package)
        _write_yaml(case_dir / "hotspot.yaml", hotspot)

        validate_simulation_input(load_simulation_input(case_dir / "scenario.yaml"))

    print(f"Created {len(CASES)} benchmark cases under {out_root}")
    return 0


def _generate_case(
    *,
    rng: random.Random,
    case_id: str,
    case_index: int,
    bump_type: str,
    dies: int,
    nets: int,
    width_mm: float,
    height_mm: float,
    whitespace: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target_area = width_mm * height_mm * (1.0 - whitespace)
    chiplets = _sample_chiplets(rng, case_index, dies)
    unscaled_area = sum(chiplet["area_mm2"] for chiplet in chiplets)
    scale = math.sqrt(target_area / unscaled_area)
    for chiplet in chiplets:
        chiplet["width"] *= scale
        chiplet["height"] *= scale
        chiplet["area_mm2"] = chiplet["width"] * chiplet["height"]

    placed_chiplets = None
    shrink = 1.0
    for _ in range(20):
        candidate = [
            {**chiplet, "width": chiplet["width"] * shrink, "height": chiplet["height"] * shrink}
            for chiplet in chiplets
        ]
        for chiplet in candidate:
            chiplet["area_mm2"] = chiplet["width"] * chiplet["height"]
        placed_chiplets = _random_pack(rng, candidate, width_mm, height_mm) or _shelf_pack(candidate, width_mm, height_mm)
        if placed_chiplets is not None:
            break
        shrink *= 0.97

    if placed_chiplets is None:
        raise RuntimeError(f"could not place {case_id}")

    actual_area = sum(chiplet["area_mm2"] for chiplet in placed_chiplets)
    actual_whitespace = 1.0 - actual_area / (width_mm * height_mm)
    layout_chiplets = []
    for chiplet in placed_chiplets:
        layout_chiplets.append(
            {
                "name": chiplet["name"],
                "type": chiplet["type"],
                "position": {"x": round(chiplet["x"], 6), "y": round(chiplet["y"], 6)},
                "size": {"width": round(chiplet["width"], 6), "height": round(chiplet["height"], 6)},
            }
        )

    layout = {
        "schema_version": 1,
        "units": {"length": "mm"},
        "package": {
            "name": case_id,
            "substrate": "silicon_interposer",
            "size": {"width": width_mm, "height": height_mm},
        },
        "chiplets": layout_chiplets,
    }
    power = _power_yaml(rng, layout_chiplets)
    benchmark = {
        "schema_version": 1,
        "case_id": case_id,
        "bump_type": bump_type,
        "dies": dies,
        "nets": nets,
        "interposer_width_mm": width_mm,
        "interposer_height_mm": height_mm,
        "whitespace": whitespace,
        "target_total_chiplet_area_mm2": round(target_area, 6),
        "actual_total_chiplet_area_mm2": round(actual_area, 6),
        "actual_whitespace": round(actual_whitespace, 6),
        "area_scaling_policy": "type-aware priors globally scaled to match target whitespace",
        "ucie": UCIE[bump_type],
        "generation_constraints": {
            "min_spacing_mm": MIN_SPACING_MM,
            "placement": "random packing with shelf fallback",
            "seed": seed,
        },
        "notes": "Synthetic benchmark family for ChipTherm research; not a commercial product replica.",
    }
    return layout, power, benchmark


def _sample_chiplets(rng: random.Random, case_index: int, dies: int) -> list[dict[str, Any]]:
    if case_index <= 5:
        pool = ["CPU", "GPU", "HBM", "DRAM", "IO"]
        required = ["CPU", "GPU", "IO"]
    else:
        pool = ["CPU", "GPU", "NPU", "HBM", "DRAM", "ANALOG", "MEMS", "IO"]
        required = ["CPU", "GPU", "NPU", "IO", "ANALOG", "MEMS"]

    types = required[: min(len(required), dies)]
    while len(types) < dies:
        types.append(rng.choice(pool))
    rng.shuffle(types)

    counts: dict[str, int] = {}
    chiplets = []
    for chiplet_type in types:
        index = counts.get(chiplet_type, 0)
        counts[chiplet_type] = index + 1
        area_low, area_high = AREA_RANGES_MM2[chiplet_type]
        aspect_options = ASPECT_RANGES[chiplet_type]
        aspect_low, aspect_high = rng.choice(aspect_options)
        area = rng.uniform(area_low, area_high)
        aspect = rng.uniform(aspect_low, aspect_high)
        width = math.sqrt(area * aspect)
        height = math.sqrt(area / aspect)
        chiplets.append(
            {
                "name": f"{chiplet_type}{index}",
                "type": chiplet_type,
                "width": width,
                "height": height,
                "area_mm2": width * height,
            }
        )
    return chiplets


def _random_pack(
    rng: random.Random,
    chiplets: list[dict[str, Any]],
    width_mm: float,
    height_mm: float,
    attempts_per_chiplet: int = 300,
) -> list[dict[str, Any]] | None:
    placed: list[dict[str, Any]] = []
    for chiplet in sorted(chiplets, key=lambda item: item["area_mm2"], reverse=True):
        if chiplet["width"] > width_mm or chiplet["height"] > height_mm:
            return None
        for _ in range(attempts_per_chiplet):
            candidate = dict(chiplet)
            candidate["x"] = rng.uniform(0.0, width_mm - candidate["width"])
            candidate["y"] = rng.uniform(0.0, height_mm - candidate["height"])
            if all(_spacing(candidate, other) >= MIN_SPACING_MM for other in placed):
                placed.append(candidate)
                break
        else:
            return None
    return placed


def _shelf_pack(chiplets: list[dict[str, Any]], width_mm: float, height_mm: float) -> list[dict[str, Any]] | None:
    placed: list[dict[str, Any]] = []
    x = 0.0
    y = 0.0
    shelf_height = 0.0
    for chiplet in sorted(chiplets, key=lambda item: item["height"], reverse=True):
        if chiplet["width"] > width_mm or chiplet["height"] > height_mm:
            return None
        if x > 0.0 and x + chiplet["width"] > width_mm:
            x = 0.0
            y += shelf_height + MIN_SPACING_MM
            shelf_height = 0.0
        if y + chiplet["height"] > height_mm:
            return None
        candidate = dict(chiplet)
        candidate["x"] = x
        candidate["y"] = y
        placed.append(candidate)
        x += chiplet["width"] + MIN_SPACING_MM
        shelf_height = max(shelf_height, chiplet["height"])
    return placed


def _spacing(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_right = first["x"] + first["width"]
    second_right = second["x"] + second["width"]
    first_top = first["y"] + first["height"]
    second_top = second["y"] + second["height"]
    dx = max(first["x"] - second_right, second["x"] - first_right, 0.0)
    dy = max(first["y"] - second_top, second["y"] - first_top, 0.0)
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _power_yaml(rng: random.Random, chiplets: list[dict[str, Any]]) -> dict[str, Any]:
    workloads = {name: {} for name in WORKLOAD_MULTIPLIERS}
    nominal = {}
    for chiplet in chiplets:
        area = chiplet["size"]["width"] * chiplet["size"]["height"]
        low, high = POWER_DENSITY_RANGES_W_PER_MM2[chiplet["type"]]
        nominal_power = area * rng.uniform(low, high)
        for workload, multiplier in WORKLOAD_MULTIPLIERS.items():
            workloads[workload][chiplet["name"]] = round(nominal_power * multiplier, 4)
        nominal[chiplet["name"]] = workloads["nominal"][chiplet["name"]]
    return {
        "schema_version": 1,
        "units": {"power": "W"},
        "mode": "fixed",
        "active_workload": "nominal",
        "workloads": workloads,
        "chiplets": nominal,
    }


def _scenario_yaml(case_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": case_id,
        "description": "Synthetic ChipTherm benchmark case.",
        "files": {
            "layout": "layout.json",
            "power": "power.yaml",
            "package": "package.yaml",
            "hotspot": "hotspot.yaml",
            "benchmark": "benchmark.yaml",
        },
    }


def _package_yaml(width_mm: float, height_mm: float) -> dict[str, Any]:
    required_side = max(width_mm, height_mm) / 1000.0
    spreader_side = max(0.055, required_side + 0.004)
    sink_side = max(0.065, required_side + 0.010)
    return {
        "schema_version": 1,
        "ambient_K": 318.15,
        "initial_temperature_K": 318.15,
        "chip": {
            "thickness_m": 0.00015,
            "thermal_conductivity_W_per_mK": 130.0,
            "volumetric_heat_capacity_J_per_m3K": 1630300,
        },
        "interface": {
            "thickness_m": 2.0e-05,
            "thermal_conductivity_W_per_mK": 4.0,
            "volumetric_heat_capacity_J_per_m3K": 4000000,
        },
        "spreader": {
            "side_m": round(spreader_side, 6),
            "thickness_m": 0.001,
            "thermal_conductivity_W_per_mK": 400.0,
            "volumetric_heat_capacity_J_per_m3K": 3550000,
        },
        "sink": {
            "side_m": round(sink_side, 6),
            "thickness_m": 0.0069,
            "thermal_conductivity_W_per_mK": 400.0,
            "volumetric_heat_capacity_J_per_m3K": 3550000,
            "convection_resistance_K_per_W": 0.12,
            "convection_capacitance_J_per_K": 140.4,
        },
    }


def _hotspot_yaml() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_type": "grid",
        "grid": {"rows": 64, "cols": 64, "map_mode": "avg"},
        "sampling_interval_s": 0.01,
        "base_processor_frequency_Hz": 3000000000,
        "leakage_used": False,
        "detailed_package": False,
        "secondary_path": False,
    }


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
