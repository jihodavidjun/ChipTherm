#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


FIELD_SPECS: list[tuple[str, str, str]] = [
    ("package_width_mm", "mm", "encoded X channel 9"),
    ("package_height_mm", "mm", "encoded X channel 10"),
    ("cell_size_x_mm", "mm", "encoded X channel 11"),
    ("cell_size_y_mm", "mm", "encoded X channel 12"),
    ("grid_rows", "cells", "X tensor shape"),
    ("grid_cols", "cells", "X tensor shape"),
    ("total_power_W", "W", "index column / power.yaml"),
    ("chiplet_count", "count", "layout.json chiplets"),
    ("occupied_fraction", "fraction", "X occupancy channel"),
    ("whitespace_fraction", "fraction", "layout/package geometry"),
    ("mean_power_density_W_per_mm2", "W/mm^2", "positive X power-density cells"),
    ("max_power_density_W_per_mm2", "W/mm^2", "X power-density channel"),
    ("mean_chiplet_area_mm2", "mm^2", "layout.json chiplet geometry"),
    ("max_chiplet_area_mm2", "mm^2", "layout.json chiplet geometry"),
    ("mean_chiplet_aspect_ratio", "ratio", "layout.json chiplet geometry"),
    ("ambient_K", "K", "package.yaml"),
    ("chip_thickness_m", "m", "package.yaml chip.thickness_m"),
    ("chip_conductivity_W_per_mK", "W/mK", "package.yaml chip.thermal_conductivity_W_per_mK"),
    ("interface_thickness_m", "m", "package.yaml interface.thickness_m"),
    ("interface_conductivity_W_per_mK", "W/mK", "package.yaml interface.thermal_conductivity_W_per_mK"),
    ("spreader_side_m", "m", "package.yaml spreader.side_m"),
    ("spreader_thickness_m", "m", "package.yaml spreader.thickness_m"),
    ("spreader_conductivity_W_per_mK", "W/mK", "package.yaml spreader.thermal_conductivity_W_per_mK"),
    ("sink_side_m", "m", "package.yaml sink.side_m"),
    ("sink_thickness_m", "m", "package.yaml sink.thickness_m"),
    ("sink_conductivity_W_per_mK", "W/mK", "package.yaml sink.thermal_conductivity_W_per_mK"),
    ("convection_resistance_K_per_W", "K/W", "package.yaml sink.convection_resistance_K_per_W"),
    ("detailed_package_enabled", "bool", "hotspot.yaml detailed_package"),
    ("secondary_path_enabled", "bool", "hotspot.yaml secondary_path"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact physical metadata features for ChipTherm samples.")
    parser.add_argument(
        "--dataset-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance/package_plus_power",
        type=Path,
    )
    parser.add_argument("--train-index", default=None, type=Path)
    parser.add_argument("--active-std-eps", default=1.0e-12, type=float)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    index_path = dataset_root / "combined_encoded_index.csv"
    if not index_path.exists():
        raise SystemExit(f"missing combined index: {index_path}")
    train_index = (args.train_index or dataset_root / "train_index.csv").expanduser().resolve()
    train_uids = {row["sample_uid"] for row in read_rows(train_index)}
    rows = read_rows(index_path)

    start = time.perf_counter()
    records = [metadata_record(row, dataset_root) for row in rows]
    runtime_s = time.perf_counter() - start

    fieldnames = ["sample_uid", "case_id", "split", *[name for name, _, _ in FIELD_SPECS]]
    write_csv(dataset_root / "metadata_features.csv", fieldnames, records)
    manifest = build_manifest(records, train_uids, dataset_root, runtime_s, args.active_std_eps)
    write_json(dataset_root / "metadata_manifest.json", manifest)
    print("Metadata feature build complete")
    print(f"Samples: {len(records)}")
    print(f"Active features: {len(manifest['active_features'])}")
    print(f"Build runtime: {runtime_s:.3f} s")
    print(f"Metadata CSV size: {(dataset_root / 'metadata_features.csv').stat().st_size} bytes")
    print(f"Output: {dataset_root}")
    return 0


def metadata_record(row: dict[str, str], dataset_root: Path) -> dict[str, Any]:
    x = np.load(resolve_path(row["x_path"], dataset_root), mmap_mode="r")
    layout_path, package_path, hotspot_path = source_paths_for_row(row)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    package = yaml.safe_load(package_path.read_text(encoding="utf-8")) or {}
    hotspot = yaml.safe_load(hotspot_path.read_text(encoding="utf-8")) or {}

    package_width = float(x[9, 0, 0])
    package_height = float(x[10, 0, 0])
    cell_size_x = float(x[11, 0, 0])
    cell_size_y = float(x[12, 0, 0])
    power_density = np.asarray(x[0], dtype=np.float64)
    occupancy = np.asarray(x[1], dtype=np.float64)
    positive_power = power_density[power_density > 0.0]
    chiplets = layout.get("chiplets", [])
    areas: list[float] = []
    aspects: list[float] = []
    for chiplet in chiplets:
        size = chiplet["size"]
        width = float(size["width"])
        height = float(size["height"])
        areas.append(width * height)
        aspects.append(width / height if height > 0 else 0.0)
    total_chiplet_area = float(sum(areas))
    package_area = max(package_width * package_height, 1.0e-12)
    return {
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "split": row.get("split", ""),
        "package_width_mm": package_width,
        "package_height_mm": package_height,
        "cell_size_x_mm": cell_size_x,
        "cell_size_y_mm": cell_size_y,
        "grid_rows": int(x.shape[1]),
        "grid_cols": int(x.shape[2]),
        "total_power_W": float(row.get("total_power_W") or float(x[8, 0, 0])),
        "chiplet_count": len(chiplets),
        "occupied_fraction": float(occupancy.mean()),
        "whitespace_fraction": float(max(0.0, 1.0 - total_chiplet_area / package_area)),
        "mean_power_density_W_per_mm2": float(positive_power.mean()) if positive_power.size else 0.0,
        "max_power_density_W_per_mm2": float(power_density.max()),
        "mean_chiplet_area_mm2": float(np.mean(areas)) if areas else 0.0,
        "max_chiplet_area_mm2": float(max(areas)) if areas else 0.0,
        "mean_chiplet_aspect_ratio": float(np.mean(aspects)) if aspects else 0.0,
        "ambient_K": float(package.get("ambient_K", 318.15)),
        "chip_thickness_m": nested_float(package, ["chip", "thickness_m"]),
        "chip_conductivity_W_per_mK": nested_float(package, ["chip", "thermal_conductivity_W_per_mK"]),
        "interface_thickness_m": nested_float(package, ["interface", "thickness_m"]),
        "interface_conductivity_W_per_mK": nested_float(package, ["interface", "thermal_conductivity_W_per_mK"]),
        "spreader_side_m": nested_float(package, ["spreader", "side_m"]),
        "spreader_thickness_m": nested_float(package, ["spreader", "thickness_m"]),
        "spreader_conductivity_W_per_mK": nested_float(package, ["spreader", "thermal_conductivity_W_per_mK"]),
        "sink_side_m": nested_float(package, ["sink", "side_m"]),
        "sink_thickness_m": nested_float(package, ["sink", "thickness_m"]),
        "sink_conductivity_W_per_mK": nested_float(package, ["sink", "thermal_conductivity_W_per_mK"]),
        "convection_resistance_K_per_W": nested_float(package, ["sink", "convection_resistance_K_per_W"]),
        "detailed_package_enabled": 1.0 if bool(hotspot.get("detailed_package", False)) else 0.0,
        "secondary_path_enabled": 1.0 if bool(hotspot.get("secondary_path", False)) else 0.0,
    }


def build_manifest(records: list[dict[str, Any]], train_uids: set[str], dataset_root: Path, runtime_s: float, std_eps: float) -> dict[str, Any]:
    train_records = [record for record in records if record["sample_uid"] in train_uids]
    stats: dict[str, dict[str, Any]] = {}
    active: list[str] = []
    constants: list[str] = []
    for name, unit, source in FIELD_SPECS:
        train_values = np.asarray([float(record[name]) for record in train_records], dtype=np.float64)
        all_values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        std = float(train_values.std())
        unique_count = int(len(set(float(value) for value in all_values)))
        stats[name] = {
            "unit": unit,
            "source": source,
            "train_mean": float(train_values.mean()),
            "train_std": std,
            "min": float(all_values.min()),
            "max": float(all_values.max()),
            "unique_count": unique_count,
            "active": bool(std > std_eps and unique_count > 1),
        }
        if stats[name]["active"]:
            active.append(name)
        else:
            constants.append(name)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": repo_relative(dataset_root),
        "num_samples": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "active_features": active,
        "constant_or_inactive_features": constants,
        "feature_stats": stats,
        "metadata_build_runtime_s": runtime_s,
        "metadata_csv": "metadata_features.csv",
        "selection_rule": f"Active iff train std > {std_eps} and value varies across samples. Case ID/package ID excluded.",
        "notes": [
            "No HotSpot temperature labels are used.",
            "Constant thermal-stack fields are recorded but excluded from the active vector for the current benchmark set.",
        ],
    }


def source_paths_for_row(row: dict[str, str]) -> tuple[Path, Path, Path]:
    original_uid = row.get("original_sample_uid", "")
    case_id = row["case_id"]
    prefix = f"{case_id}_"
    if not original_uid.startswith(prefix):
        raise SystemExit(f"{row['sample_uid']} original_sample_uid does not match case_id")
    sample_dir = original_uid[len(prefix) :]
    source_dir = REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_dir / "source"
    layout_path = source_dir / "layout.json"
    package_path = source_dir / "package.yaml"
    hotspot_path = source_dir / "hotspot.yaml"
    missing = [path for path in (layout_path, package_path, hotspot_path) if not path.exists()]
    if missing:
        raise SystemExit(f"{row['sample_uid']} missing source metadata: {missing}")
    return layout_path, package_path, hotspot_path


def nested_float(data: dict[str, Any], keys: list[str]) -> float:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return 0.0
        value = value[key]
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, base / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
