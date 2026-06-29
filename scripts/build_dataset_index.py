#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.scenario import load_simulation_input
from chiptherm.validate import validate_simulation_input


CSV_COLUMNS = [
    "sample_uid",
    "case_id",
    "sample_id",
    "sample_dir",
    "scenario_path",
    "benchmark_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "flp_path",
    "ptrace_path",
    "hotspot_config_path",
    "block_steady_path",
    "grid_steady_path",
    "temp_layer0_path",
    "block_temps_path",
    "manifest_path",
    "num_chiplets",
    "chiplet_types",
    "interposer_width_mm",
    "interposer_height_mm",
    "active_workload",
    "grid_rows",
    "grid_cols",
    "temp_min_K",
    "temp_max_K",
    "temp_mean_K",
    "hottest_block",
    "hottest_block_temp_K",
    "hotspot_runtime_s",
    "total_runtime_s",
    "validation_passed",
    "benchmark_bump_type",
    "benchmark_nets",
    "benchmark_whitespace",
]


REQUIRED_RELATIVE_FILES = {
    "scenario_path": "source/scenario.yaml",
    "layout_path": "source/layout.json",
    "power_path": "source/power.yaml",
    "package_path": "source/package.yaml",
    "hotspot_path": "source/hotspot.yaml",
    "flp_path": "hotspot/chiplet.flp",
    "ptrace_path": "hotspot/power.ptrace",
    "hotspot_config_path": "hotspot/hotspot.config",
    "block_steady_path": "outputs/block.steady",
    "grid_steady_path": "outputs/grid.steady",
    "temp_layer0_path": "parsed/temp_layer0.npy",
    "block_temps_path": "parsed/block_temps.json",
    "manifest_path": "manifest.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CSV/JSONL indexes for a ChipTherm dataset.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped = 0
    cases_found: set[str] = set()

    for sample_dir in _iter_sample_dirs(dataset_root):
        case_id = sample_dir.parent.name
        cases_found.add(case_id)
        record, reason = _build_record(dataset_root, case_id, sample_dir)
        if record is None:
            skipped += 1
            warnings.append(f"{_rel(dataset_root, sample_dir)}: {reason}")
            continue
        records.append(record)

    records.sort(key=lambda item: item["sample_uid"])
    csv_path = dataset_root / "dataset_index.csv"
    jsonl_path = dataset_root / "dataset_index.jsonl"
    _write_csv(csv_path, records)
    _write_jsonl(jsonl_path, records)

    temps = [float(record["temp_mean_K"]) for record in records if record.get("temp_mean_K") is not None]
    temp_min = min((float(record["temp_min_K"]) for record in records), default=None)
    temp_max = max((float(record["temp_max_K"]) for record in records), default=None)
    temp_mean = sum(temps) / len(temps) if temps else None

    for warning in warnings[:25]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(warnings) > 25:
        print(f"WARNING: {len(warnings) - 25} additional warnings omitted", file=sys.stderr)

    print("Dataset index build complete")
    print(f"Indexed samples: {len(records)}")
    print(f"Skipped samples: {skipped}")
    print(f"Cases found: {len(cases_found)}")
    if temp_min is not None and temp_max is not None and temp_mean is not None:
        print(f"Temperature min/max/mean: {temp_min:.2f} / {temp_max:.2f} / {temp_mean:.2f} K")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    return 0


def _iter_sample_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.glob("case*/sample_*") if path.is_dir())


def _build_record(dataset_root: Path, case_id: str, sample_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    paths = {key: sample_dir / rel for key, rel in REQUIRED_RELATIVE_FILES.items()}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        return None, f"missing required files: {', '.join(missing)}"

    manifest = _load_json(paths["manifest_path"])
    if manifest.get("success") is False:
        return None, f"manifest marks sample failed: {manifest.get('reason', 'no reason')}"
    if manifest.get("hotspot", {}).get("return_code") not in (0, None):
        return None, f"HotSpot return code is {manifest.get('hotspot', {}).get('return_code')}"
    if not isinstance(manifest.get("output_summary"), dict):
        return None, "manifest is missing output_summary"

    layout = _load_json(paths["layout_path"])
    power = _load_yaml(paths["power_path"])
    benchmark_path = sample_dir / "source" / "benchmark.yaml"
    benchmark = _load_yaml(benchmark_path) if benchmark_path.exists() else None
    block_temps = _load_json(paths["block_temps_path"])

    expected_shape = _expected_shape(manifest)
    try:
        layer0 = np.load(paths["temp_layer0_path"])
    except Exception as exc:
        return None, f"could not load temp_layer0.npy: {exc}"
    if expected_shape is not None and tuple(layer0.shape) != expected_shape:
        return None, f"temp_layer0.npy shape {tuple(layer0.shape)} != expected {expected_shape}"

    validation_passed = True
    validation_error = None
    try:
        validate_simulation_input(load_simulation_input(paths["scenario_path"]))
    except Exception as exc:
        validation_passed = False
        validation_error = str(exc)

    chiplets = layout.get("chiplets", [])
    chiplet_types = sorted({str(chiplet.get("type", "")) for chiplet in chiplets})
    package_size = layout.get("package", {}).get("size", {})
    output_summary = manifest["output_summary"]
    runtime = manifest.get("runtime", {})
    active_workload = power.get("active_workload")
    power_summary = _power_summary(layout, power)

    row = {
        "sample_uid": f"{case_id}_{sample_dir.name}",
        "case_id": case_id,
        "sample_id": sample_dir.name,
        "sample_dir": _rel(dataset_root, sample_dir),
        "scenario_path": _rel(dataset_root, paths["scenario_path"]),
        "benchmark_path": _rel(dataset_root, benchmark_path) if benchmark_path.exists() else "",
        "layout_path": _rel(dataset_root, paths["layout_path"]),
        "power_path": _rel(dataset_root, paths["power_path"]),
        "package_path": _rel(dataset_root, paths["package_path"]),
        "hotspot_path": _rel(dataset_root, paths["hotspot_path"]),
        "flp_path": _rel(dataset_root, paths["flp_path"]),
        "ptrace_path": _rel(dataset_root, paths["ptrace_path"]),
        "hotspot_config_path": _rel(dataset_root, paths["hotspot_config_path"]),
        "block_steady_path": _rel(dataset_root, paths["block_steady_path"]),
        "grid_steady_path": _rel(dataset_root, paths["grid_steady_path"]),
        "temp_layer0_path": _rel(dataset_root, paths["temp_layer0_path"]),
        "block_temps_path": _rel(dataset_root, paths["block_temps_path"]),
        "manifest_path": _rel(dataset_root, paths["manifest_path"]),
        "num_chiplets": len(chiplets),
        "chiplet_types": ",".join(chiplet_types),
        "interposer_width_mm": _float_or_none(package_size.get("width")),
        "interposer_height_mm": _float_or_none(package_size.get("height")),
        "active_workload": active_workload,
        "grid_rows": int(output_summary.get("grid_rows", manifest.get("grid", {}).get("rows", layer0.shape[0]))),
        "grid_cols": int(output_summary.get("grid_cols", manifest.get("grid", {}).get("cols", layer0.shape[1]))),
        "temp_min_K": float(output_summary.get("temp_layer0_min_K", layer0.min())),
        "temp_max_K": float(output_summary.get("temp_layer0_max_K", layer0.max())),
        "temp_mean_K": float(output_summary.get("temp_layer0_mean_K", layer0.mean())),
        "hottest_block": output_summary.get("hottest_block"),
        "hottest_block_temp_K": _float_or_none(output_summary.get("max_block_temperature_K")),
        "hotspot_runtime_s": _float_or_none(runtime.get("hotspot_s")),
        "total_runtime_s": _float_or_none(runtime.get("total_s")),
        "validation_passed": validation_passed,
        "benchmark_bump_type": benchmark.get("bump_type") if benchmark else None,
        "benchmark_nets": benchmark.get("nets") if benchmark else None,
        "benchmark_whitespace": benchmark.get("whitespace") if benchmark else None,
        "power_summary_by_type": power_summary,
        "block_temps": block_temps,
        "benchmark": benchmark,
    }
    if validation_error is not None:
        row["validation_error"] = validation_error
    return row, None


def _expected_shape(manifest: dict[str, Any]) -> tuple[int, int] | None:
    summary = manifest.get("output_summary", {})
    shape = summary.get("temp_layer0_shape")
    if isinstance(shape, list) and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    grid = manifest.get("grid", {})
    if "rows" in grid and "cols" in grid:
        return int(grid["rows"]), int(grid["cols"])
    return None


def _power_summary(layout: dict[str, Any], power: dict[str, Any]) -> dict[str, dict[str, float]]:
    chiplet_types = {chiplet["name"]: chiplet["type"] for chiplet in layout.get("chiplets", [])}
    powers = power.get("chiplets", {})
    by_type: dict[str, list[float]] = {}
    for name, watts in powers.items():
        chiplet_type = chiplet_types.get(name, "UNKNOWN")
        by_type.setdefault(chiplet_type, []).append(float(watts))
    return {
        chiplet_type: {
            "count": len(values),
            "min_W": min(values),
            "max_W": max(values),
            "mean_W": sum(values) / len(values),
            "total_W": sum(values),
        }
        for chiplet_type, values in sorted(by_type.items())
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in CSV_COLUMNS})


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _rel(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
