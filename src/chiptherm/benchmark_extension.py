from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .scenario import load_simulation_input
from .validate import validate_simulation_input
from .writers import write_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark_extension_v1" / "cases.yaml"

DEFAULT_PACKAGE = {
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
        "side_m": 0.09,
        "thickness_m": 0.001,
        "thermal_conductivity_W_per_mK": 400.0,
        "volumetric_heat_capacity_J_per_m3K": 3550000,
    },
    "sink": {
        "side_m": 0.1,
        "thickness_m": 0.0069,
        "thermal_conductivity_W_per_mK": 400.0,
        "volumetric_heat_capacity_J_per_m3K": 3550000,
        "convection_resistance_K_per_W": 0.12,
        "convection_capacitance_J_per_K": 140.4,
    },
}

DEFAULT_HOTSPOT = {
    "schema_version": 1,
    "model_type": "grid",
    "grid": {"rows": 64, "cols": 64, "map_mode": "avg"},
    "sampling_interval_s": 0.01,
    "base_processor_frequency_Hz": 3000000000,
    "leakage_used": False,
    "detailed_package": False,
    "secondary_path": False,
}


@dataclass(frozen=True)
class ExtensionSample:
    row: dict[str, Any]
    statistics: dict[str, Any]
    validation: dict[str, Any]


def load_extension_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("benchmark extension config must define exactly ten cases")
    seen = set()
    for case in cases:
        case_id = str(case.get("case_id"))
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id}")
        seen.add(case_id)
        if sum(int(v) for v in case.get("composition", {}).values()) != int(case["die_count"]):
            raise ValueError(f"{case_id}: composition does not sum to die_count")
    return data


def select_cases(config: dict[str, Any], case_ids: list[str] | None) -> list[dict[str, Any]]:
    cases = list(config["cases"])
    if case_ids is None:
        return cases
    wanted = set(case_ids)
    selected = [case for case in cases if case["case_id"] in wanted]
    missing = wanted - {case["case_id"] for case in selected}
    if missing:
        raise ValueError(f"unknown case ids: {', '.join(sorted(missing))}")
    return selected


def generate_sample(case: dict[str, Any], defaults: dict[str, Any], sample_index: int, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed + _stable_int(case["case_id"]) * 1000003 + sample_index)
    width_mm, height_mm = [float(v) for v in case["interposer_mm"]]
    whitespace_low, whitespace_high = [float(v) for v in case["whitespace_range"]]
    target_whitespace = rng.uniform(whitespace_low, whitespace_high)
    chiplets = _make_chiplets(case, defaults, rng, width_mm, height_mm, target_whitespace)
    _place_chiplets(chiplets, width_mm, height_mm, str(case["placement_regime"]), rng, float(defaults.get("min_spacing_mm", 0.5)))
    powers = _make_power(case, defaults, chiplets, rng)
    sample_uid = f"benchmark_extension_v1_{case['case_id']}_sample_{sample_index:06d}"
    layout = {
        "schema_version": 1,
        "units": {"length": "mm"},
        "package": {
            "name": sample_uid,
            "substrate": "silicon_interposer",
            "size": {"width": round(width_mm, 6), "height": round(height_mm, 6)},
        },
        "chiplets": chiplets,
    }
    power = _power_yaml(powers, defaults)
    benchmark = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "split_role": case["split_role"],
        "bump_type": case.get("bump_type", "x16"),
        "dies": int(case["die_count"]),
        "nets": rng.randint(int(case["nets_range"][0]), int(case["nets_range"][1])),
        "interposer_width_mm": width_mm,
        "interposer_height_mm": height_mm,
        "target_whitespace": target_whitespace,
        "actual_whitespace": layout_statistics(layout, power)["whitespace_fraction"],
        "placement_regime": case["placement_regime"],
        "generation_constraints": {"min_spacing_mm": float(defaults.get("min_spacing_mm", 0.5))},
        "purpose": case.get("purpose", ""),
        "sample_id": sample_uid,
    }
    return layout, power, benchmark


def write_sample_sources(
    sample_dir: Path,
    sample_uid: str,
    layout: dict[str, Any],
    power: dict[str, Any],
    benchmark: dict[str, Any],
    *,
    cleanup_hotspot_workdirs: bool = False,
) -> dict[str, Path]:
    source_dir = sample_dir / "source"
    hotspot_dir = sample_dir / "hotspot"
    parsed_dir = sample_dir / "parsed"
    for path in (source_dir, hotspot_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)
    if cleanup_hotspot_workdirs:
        for child in hotspot_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    package = _package_for_layout(layout)
    hotspot = dict(DEFAULT_HOTSPOT)
    scenario = {
        "schema_version": 1,
        "name": sample_uid,
        "description": "Generated ChipTherm benchmark extension sample.",
        "files": {
            "layout": "layout.json",
            "power": "power.yaml",
            "package": "package.yaml",
            "hotspot": "hotspot.yaml",
            "benchmark": "benchmark.yaml",
        },
    }
    paths = {
        "source_dir": source_dir,
        "scenario_path": source_dir / "scenario.yaml",
        "layout_path": source_dir / "layout.json",
        "power_path": source_dir / "power.yaml",
        "package_path": source_dir / "package.yaml",
        "hotspot_path": source_dir / "hotspot.yaml",
        "benchmark_path": source_dir / "benchmark.yaml",
        "y_path": sample_dir / "parsed" / "temp_layer0.npy",
    }
    paths["scenario_path"].write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
    paths["layout_path"].write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    paths["power_path"].write_text(yaml.safe_dump(power, sort_keys=False), encoding="utf-8")
    paths["package_path"].write_text(yaml.safe_dump(package, sort_keys=False), encoding="utf-8")
    paths["hotspot_path"].write_text(yaml.safe_dump(hotspot, sort_keys=False), encoding="utf-8")
    paths["benchmark_path"].write_text(yaml.safe_dump(benchmark, sort_keys=False), encoding="utf-8")
    return paths


def validate_sample_sources(scenario_path: Path, expected_case: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    try:
        sim = load_simulation_input(scenario_path)
        validate_simulation_input(sim)
    except Exception as exc:
        problems.append(str(exc))
        sim = None
    if sim is not None:
        if len(sim.layout.chiplets) != int(expected_case["die_count"]):
            problems.append(f"die count mismatch: expected {expected_case['die_count']}, got {len(sim.layout.chiplets)}")
        stats = layout_statistics(_json(sim.scenario.layout_path), _yaml(sim.scenario.power_path))
        low, high = expected_case["whitespace_range"]
        if not (float(low) - 1e-6 <= stats["whitespace_fraction"] <= float(high) + 1e-6):
            problems.append(f"whitespace {stats['whitespace_fraction']:.4f} outside target range [{low}, {high}]")
    return {"passed": not problems, "problems": problems}


def row_for_sample(
    *,
    sample_uid: str,
    case: dict[str, Any],
    paths: dict[str, Path],
    statistics: dict[str, Any],
    stage: str,
    hotspot_status: str,
) -> dict[str, Any]:
    split = str(case["split_role"])
    y_path = paths["y_path"] if paths["y_path"].exists() else ""
    return {
        "sample_uid": sample_uid,
        "original_sample_uid": sample_uid,
        "case_id": case["case_id"],
        "dataset_source": f"benchmark_extension_v1_{stage}",
        "split": split,
        "source_dir": str(paths["source_dir"]),
        "scenario_path": str(paths["scenario_path"]),
        "layout_path": str(paths["layout_path"]),
        "power_path": str(paths["power_path"]),
        "package_path": str(paths["package_path"]),
        "hotspot_path": str(paths["hotspot_path"]),
        "benchmark_path": str(paths["benchmark_path"]),
        "x_path": "",
        "y_path": str(y_path),
        "prediction_path": "",
        "residual_path": "",
        "hotspot_runtime_s": "",
        "physics_runtime_s": "",
        "hotspot_status": hotspot_status,
        "num_chiplets": statistics["chiplet_count"],
        "total_power_W": f"{statistics['total_power_W']:.8g}",
        "package_width_mm": f"{statistics['package_width_mm']:.8g}",
        "package_height_mm": f"{statistics['package_height_mm']:.8g}",
        "whitespace_fraction": f"{statistics['whitespace_fraction']:.8g}",
        "mean_power_density_W_per_mm2": f"{statistics['mean_power_density_W_per_mm2']:.8g}",
        "max_power_density_W_per_mm2": f"{statistics['max_power_density_W_per_mm2']:.8g}",
        "layout_hash": statistics["layout_hash"],
    }


def write_indexes(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    fieldnames = _index_fieldnames(rows)
    for name, subset in {
        "train": [row for row in rows if row["split"] == "train"],
        "val": [row for row in rows if row["split"] == "val"],
        "test": [row for row in rows if row["split"] == "test"],
        "all_extension": rows,
        "combined_encoded": rows,
    }.items():
        path = out_dir / f"{name}_index.csv"
        _write_csv(path, subset, fieldnames)
        paths[name] = path
    _write_jsonl(out_dir / "combined_encoded_index.jsonl", rows)
    return paths


def layout_statistics(layout: dict[str, Any], power: dict[str, Any]) -> dict[str, Any]:
    width = float(layout["package"]["size"]["width"])
    height = float(layout["package"]["size"]["height"])
    rects = []
    areas = []
    powers = []
    aspect_ratios = []
    edge_flags = []
    corner_flags = []
    densities = []
    type_counts: dict[str, int] = {}
    edge_threshold = 0.1 * min(width, height)
    for chiplet in layout["chiplets"]:
        x = float(chiplet["position"]["x"])
        y = float(chiplet["position"]["y"])
        w = float(chiplet["size"]["width"])
        h = float(chiplet["size"]["height"])
        area = w * h
        watts = float(power["chiplets"][chiplet["name"]])
        rects.append((x, y, w, h))
        areas.append(area)
        powers.append(watts)
        aspect_ratios.append(max(w, h) / max(min(w, h), 1e-12))
        densities.append(watts / max(area, 1e-12))
        type_counts[chiplet["type"]] = type_counts.get(chiplet["type"], 0) + 1
        d_left, d_right, d_bottom, d_top = x, width - (x + w), y, height - (y + h)
        edge_flags.append(min(d_left, d_right, d_bottom, d_top) <= edge_threshold)
        corner_flags.append(min(d_left, d_right) <= edge_threshold and min(d_bottom, d_top) <= edge_threshold)
    center_distances = []
    edge_distances = []
    for i, first in enumerate(rects):
        for second in rects[i + 1:]:
            cx1, cy1 = first[0] + first[2] / 2.0, first[1] + first[3] / 2.0
            cx2, cy2 = second[0] + second[2] / 2.0, second[1] + second[3] / 2.0
            center_distances.append(math.hypot(cx1 - cx2, cy1 - cy2))
            edge_distances.append(_edge_spacing_rect(first, second))
    nearest = []
    for i, first in enumerate(rects):
        values = [_edge_spacing_rect(first, second) for j, second in enumerate(rects) if i != j]
        if values:
            nearest.append(min(values))
    total_area = sum(areas)
    total_power = sum(powers)
    return {
        "chiplet_count": len(rects),
        "package_width_mm": width,
        "package_height_mm": height,
        "package_area_mm2": width * height,
        "package_aspect_ratio": max(width, height) / min(width, height),
        "occupied_area_mm2": total_area,
        "occupied_fraction": total_area / (width * height),
        "whitespace_fraction": 1.0 - total_area / (width * height),
        "total_power_W": total_power,
        "mean_power_W": _mean(powers),
        "max_power_W": max(powers),
        "power_std_W": _std(powers),
        "max_chiplet_power_fraction": max(powers) / max(total_power, 1e-12),
        "mean_power_density_W_per_mm2": _mean(densities),
        "max_power_density_W_per_mm2": max(densities),
        "mean_chiplet_area_mm2": _mean(areas),
        "max_chiplet_area_mm2": max(areas),
        "mean_chiplet_aspect_ratio": _mean(aspect_ratios),
        "min_pairwise_center_distance_mm": min(center_distances) if center_distances else 0.0,
        "mean_pairwise_center_distance_mm": _mean(center_distances) if center_distances else 0.0,
        "min_pairwise_edge_distance_mm": min(edge_distances) if edge_distances else 0.0,
        "mean_nearest_edge_distance_mm": _mean(nearest) if nearest else 0.0,
        "fraction_chiplets_near_edge": sum(edge_flags) / len(edge_flags),
        "fraction_chiplets_near_corner": sum(corner_flags) / len(corner_flags),
        "thermal_crowding_W_per_mm_mean": _thermal_crowding(rects, powers),
        "type_counts": type_counts,
        "layout_hash": layout_hash(layout),
    }


def write_audit_reports(out_dir: Path, rows: list[dict[str, Any]], sample_stats: list[dict[str, Any]], *, stage: str, validation: list[dict[str, Any]], config_hash: str) -> dict[str, Any]:
    case_stats = case_statistics(sample_stats)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "config_hash_sha256": config_hash,
        "sample_count": len(rows),
        "case_count": len(case_stats),
        "split_counts": _counts(row["split"] for row in rows),
        "case_counts": _counts(row["case_id"] for row in rows),
        "hotspot_status_counts": _counts(row["hotspot_status"] for row in rows),
        "validation": {
            "passed": all(item["passed"] for item in validation),
            "failed_samples": [item for item in validation if not item["passed"]],
        },
        "case_statistics": case_stats,
        "storage_estimate": estimate_storage(len(rows), include_hotspot_labels=False),
    }
    write_manifest(out_dir / "manifest.json", manifest)
    write_manifest(out_dir / f"{stage}_audit_report.json", manifest)
    _write_case_stats_csv(out_dir / "case_statistics.csv", case_stats)
    _write_sample_stats_csv(out_dir / "sample_statistics.csv", sample_stats)
    _write_report_md(out_dir / f"{stage}_audit_report.md", manifest)
    return manifest


def case_statistics(sample_stats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in sample_stats:
        by_case.setdefault(str(item["case_id"]), []).append(item)
    out: dict[str, dict[str, float]] = {}
    keys = [
        "chiplet_count",
        "package_width_mm",
        "package_height_mm",
        "whitespace_fraction",
        "total_power_W",
        "mean_power_density_W_per_mm2",
        "max_power_density_W_per_mm2",
        "min_pairwise_edge_distance_mm",
        "mean_nearest_edge_distance_mm",
        "fraction_chiplets_near_edge",
        "fraction_chiplets_near_corner",
    ]
    for case_id, items in by_case.items():
        values: dict[str, float] = {"count": float(len(items))}
        for key in keys:
            vals = [float(item[key]) for item in items]
            values[f"{key}_mean"] = _mean(vals)
            values[f"{key}_std"] = _std(vals)
            values[f"{key}_min"] = min(vals)
            values[f"{key}_max"] = max(vals)
        out[case_id] = values
    return out


def validate_extension_root(out_dir: Path, *, require_hotspot_labels: bool = False) -> dict[str, Any]:
    rows = read_index(out_dir / "all_extension_index.csv")
    problems: list[str] = []
    warnings: list[str] = []
    if not rows:
        problems.append("all_extension_index.csv is empty or missing")
    seen_uids = set()
    layout_hashes = set()
    for row in rows:
        uid = row["sample_uid"]
        if uid in seen_uids:
            problems.append(f"duplicate sample_uid {uid}")
        seen_uids.add(uid)
        for field in ("scenario_path", "layout_path", "power_path", "package_path", "hotspot_path", "benchmark_path"):
            if not Path(row[field]).exists():
                problems.append(f"{uid}: missing {field} {row[field]}")
        if row.get("x_path"):
            warnings.append(f"{uid}: x_path is populated before canonical encoding")
        if row.get("residual_path") or row.get("prediction_path"):
            warnings.append(f"{uid}: prediction/residual path is populated unexpectedly")
        if require_hotspot_labels and not row.get("y_path"):
            problems.append(f"{uid}: missing required HotSpot label y_path")
        if row.get("y_path") and not Path(row["y_path"]).exists():
            problems.append(f"{uid}: y_path does not exist")
        if row.get("layout_hash") in layout_hashes:
            problems.append(f"{uid}: duplicate layout hash {row.get('layout_hash')}")
        layout_hashes.add(row.get("layout_hash"))
        try:
            sim = load_simulation_input(row["scenario_path"])
            validate_simulation_input(sim)
        except Exception as exc:
            problems.append(f"{uid}: canonical validation failed: {exc}")
    split_cases = {split: sorted({row["case_id"] for row in rows if row["split"] == split}) for split in ("train", "val", "test")}
    if any(case in split_cases["train"] for case in split_cases["val"] + split_cases["test"]):
        problems.append("family-level split overlap detected")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not problems,
        "problems": problems,
        "warnings": warnings,
        "sample_count": len(rows),
        "split_counts": _counts(row["split"] for row in rows),
        "case_counts": _counts(row["case_id"] for row in rows),
        "split_cases": split_cases,
        "require_hotspot_labels": require_hotspot_labels,
    }
    write_manifest(out_dir / "validation_report.json", report)
    _write_validation_md(out_dir / "validation_report.md", report)
    return report


def approve_pilot(pilot_root: Path, approval_path: Path | None = None, *, allow_warnings: bool = False) -> dict[str, Any]:
    validation_path = pilot_root / "validation_report.json"
    manifest_path = pilot_root / "manifest.json"
    if not validation_path.exists():
        raise ValueError(f"missing validation report: {validation_path}")
    if not manifest_path.exists():
        raise ValueError(f"missing manifest: {manifest_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise ValueError("pilot validation did not pass")
    if validation.get("warnings") and not allow_warnings:
        raise ValueError("pilot validation has warnings; pass --allow-warnings to approve")
    approval = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved": True,
        "pilot_root": str(pilot_root),
        "manifest_sha256": file_sha256(manifest_path),
        "validation_report_sha256": file_sha256(validation_path),
    }
    path = approval_path or (pilot_root / "pilot_approval.json")
    write_manifest(path, approval)
    return approval


def verify_approval(pilot_root: Path, approval_file: Path | None = None) -> dict[str, Any]:
    approval_path = approval_file or (pilot_root / "pilot_approval.json")
    if not approval_path.exists():
        raise ValueError(f"full generation requires pilot approval: {approval_path}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not approval.get("approved"):
        raise ValueError("pilot approval file is not approved")
    manifest_path = pilot_root / "manifest.json"
    validation_path = pilot_root / "validation_report.json"
    if approval.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("pilot manifest hash mismatch")
    if approval.get("validation_report_sha256") != file_sha256(validation_path):
        raise ValueError("pilot validation report hash mismatch")
    return approval


def estimate_storage(sample_count: int, *, include_hotspot_labels: bool) -> dict[str, Any]:
    source_per_sample_mb = 0.03
    y_per_sample_mb = 64 * 64 * 4 / (1024 * 1024)
    x_per_sample_mb = 34 * 64 * 64 * 4 / (1024 * 1024)
    hotspot_workdir_mb = 0.2
    total_mb = sample_count * source_per_sample_mb
    if include_hotspot_labels:
        total_mb += sample_count * (y_per_sample_mb + hotspot_workdir_mb)
    return {
        "sample_count": sample_count,
        "source_files_MB": sample_count * source_per_sample_mb,
        "label_tensors_MB_if_generated": sample_count * y_per_sample_mb,
        "future_encoded_X_MB_estimate": sample_count * x_per_sample_mb,
        "hotspot_workdirs_MB_if_kept": sample_count * hotspot_workdir_mb,
        "total_MB_for_requested_mode": total_mb,
        "total_GB_for_requested_mode": total_mb / 1024.0,
    }


def read_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_hash(layout: dict[str, Any]) -> str:
    payload = json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_chiplets(case: dict[str, Any], defaults: dict[str, Any], rng: random.Random, width_mm: float, height_mm: float, target_whitespace: float) -> list[dict[str, Any]]:
    ranges = defaults["chiplet_type_size_ranges_mm"]
    specs: list[tuple[str, str, float, float]] = []
    for chiplet_type, count in case["composition"].items():
        low, high = [float(v) for v in ranges[chiplet_type]]
        for idx in range(int(count)):
            aspect = rng.uniform(0.75, 1.35)
            side = rng.uniform(low, high)
            w = side * math.sqrt(aspect)
            h = side / math.sqrt(aspect)
            specs.append((chiplet_type, f"{chiplet_type}{idx}", w, h))
    rng.shuffle(specs)
    raw_area = sum(w * h for _, _, w, h in specs)
    target_area = width_mm * height_mm * (1.0 - target_whitespace)
    scale = math.sqrt(target_area / raw_area)
    chiplets = []
    for chiplet_type, name, w, h in specs:
        max_w = 0.35 * width_mm
        max_h = 0.35 * height_mm
        ww = min(max(w * scale, 0.8), max_w)
        hh = min(max(h * scale, 0.8), max_h)
        chiplets.append(
            {
                "name": name,
                "type": chiplet_type,
                "position": {"x": 0.0, "y": 0.0},
                "size": {"width": round(ww, 6), "height": round(hh, 6)},
            }
        )
    return chiplets


def _place_chiplets(chiplets: list[dict[str, Any]], width_mm: float, height_mm: float, regime: str, rng: random.Random, min_spacing_mm: float) -> None:
    base_order = sorted(chiplets, key=lambda c: float(c["size"]["width"]) * float(c["size"]["height"]), reverse=True)
    for attempt_round in range(12):
        order = list(base_order)
        if attempt_round:
            # Keep larger dies early in general, but vary tie ordering enough to
            # avoid seed-fragile local packing failures while preserving the
            # declared benchmark spacing.
            decorated = [(-float(c["size"]["width"]) * float(c["size"]["height"]), rng.random(), c) for c in order]
            order = [item[2] for item in sorted(decorated)]
        placed: list[tuple[float, float, float, float]] = []
        ok = True
        for chiplet in order:
            w = float(chiplet["size"]["width"])
            h = float(chiplet["size"]["height"])
            found = False
            for _ in range(10000):
                x, y = _candidate_position(regime, rng, width_mm, height_mm, w, h)
                rect = (x, y, w, h)
                if all(_edge_spacing_rect(rect, other) + 1e-9 >= min_spacing_mm for other in placed):
                    chiplet["position"] = {"x": round(x, 6), "y": round(y, 6)}
                    placed.append(rect)
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            return
    raise ValueError("could not place chiplets without overlap")


def _candidate_position(regime: str, rng: random.Random, width: float, height: float, w: float, h: float) -> tuple[float, float]:
    max_x = max(width - w, 0.0)
    max_y = max(height - h, 0.0)
    if regime in {"horizontal_spread", "elongated_edge_mixed"}:
        x = rng.choice([rng.uniform(0, max_x * 0.25), rng.uniform(max_x * 0.35, max_x * 0.65), rng.uniform(max_x * 0.75, max_x)])
        y = rng.uniform(0, max_y)
    elif regime == "vertical_clusters":
        x = rng.uniform(0, max_x)
        y = rng.choice([rng.uniform(0, max_y * 0.30), rng.uniform(max_y * 0.45, max_y * 0.70), rng.uniform(max_y * 0.75, max_y)])
    elif regime in {"edge_heavy", "corner_hotspot"} and rng.random() < 0.65:
        edge = rng.choice(["left", "right", "bottom", "top"])
        margin_x = min(max_x, 0.15 * width)
        margin_y = min(max_y, 0.15 * height)
        if edge == "left":
            x, y = rng.uniform(0, margin_x), rng.uniform(0, max_y)
        elif edge == "right":
            x, y = rng.uniform(max(0, max_x - margin_x), max_x), rng.uniform(0, max_y)
        elif edge == "bottom":
            x, y = rng.uniform(0, max_x), rng.uniform(0, margin_y)
        else:
            x, y = rng.uniform(0, max_x), rng.uniform(max(0, max_y - margin_y), max_y)
    elif regime in {"mixed_clusters", "dense_mixed", "compact_high_count"}:
        centers = [(0.30 * width, 0.30 * height), (0.70 * width, 0.35 * height), (0.52 * width, 0.70 * height)]
        cx, cy = rng.choice(centers)
        x = min(max(rng.gauss(cx - w / 2.0, 0.18 * width), 0.0), max_x)
        y = min(max(rng.gauss(cy - h / 2.0, 0.18 * height), 0.0), max_y)
    elif regime == "asymmetric_power":
        x = min(max(rng.gauss(0.45 * width, 0.28 * width) - w / 2.0, 0.0), max_x)
        y = min(max(rng.gauss(0.55 * height, 0.28 * height) - h / 2.0, 0.0), max_y)
    else:
        x = rng.uniform(0, max_x)
        y = rng.uniform(0, max_y)
    return x, y


def _make_power(case: dict[str, Any], defaults: dict[str, Any], chiplets: list[dict[str, Any]], rng: random.Random) -> dict[str, float]:
    density_ranges = defaults["chiplet_type_power_density_ranges_W_per_mm2"]
    validator_caps = {
        "CPU": 3.0,
        "GPU": 3.0,
        "NPU": 3.0,
        "HBM": 0.35,
        "DRAM": 0.5,
        "IO": 0.6,
        "ANALOG": 0.7,
        "MEMS": 0.5,
    }
    scale = float(case.get("power_density_scale", 1.0))
    powers = {}
    for chiplet in chiplets:
        low, high = [float(v) for v in density_ranges[chiplet["type"]]]
        density = rng.uniform(low, high) * scale
        if str(case["placement_regime"]) in {"asymmetric_power", "corner_hotspot"} and chiplet["type"] in {"GPU", "NPU"} and rng.random() < 0.45:
            density *= 1.35
        density = min(density, validator_caps[chiplet["type"]] * 0.98)
        area = float(chiplet["size"]["width"]) * float(chiplet["size"]["height"])
        powers[chiplet["name"]] = round(max(density * area, 1e-4), 4)
    return powers


def _power_yaml(powers: dict[str, float], defaults: dict[str, Any]) -> dict[str, Any]:
    multipliers = defaults["power_workload_multipliers"]
    workloads = {
        workload: {name: round(power * float(mult), 4) for name, power in powers.items()} for workload, mult in multipliers.items()
    }
    return {
        "schema_version": 1,
        "units": {"power": "W"},
        "mode": "fixed",
        "active_workload": "nominal",
        "workloads": workloads,
        "chiplets": workloads["nominal"],
    }


def _package_for_layout(layout: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(DEFAULT_PACKAGE))
    width_m = float(layout["package"]["size"]["width"]) * 1e-3
    height_m = float(layout["package"]["size"]["height"]) * 1e-3
    side = max(width_m, height_m) + 0.02
    data["spreader"]["side_m"] = round(side, 6)
    data["sink"]["side_m"] = round(side + 0.01, 6)
    return data


def _edge_spacing_rect(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    dx = max(ax - (bx + bw), bx - (ax + aw), 0.0)
    dy = max(ay - (by + bh), by - (ay + ah), 0.0)
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _thermal_crowding(rects: list[tuple[float, float, float, float]], powers: list[float]) -> float:
    vals = []
    for rect in rects:
        cx = rect[0] + rect[2] / 2.0
        cy = rect[1] + rect[3] / 2.0
        total = 0.0
        for other, power in zip(rects, powers):
            ox = other[0] + other[2] / 2.0
            oy = other[1] + other[3] / 2.0
            total += power / math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2 + 1.0)
        vals.append(total)
    return _mean(vals) if vals else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def _index_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    base = [
        "sample_uid",
        "original_sample_uid",
        "case_id",
        "dataset_source",
        "split",
        "source_dir",
        "scenario_path",
        "layout_path",
        "power_path",
        "package_path",
        "hotspot_path",
        "benchmark_path",
        "x_path",
        "y_path",
        "prediction_path",
        "residual_path",
        "hotspot_runtime_s",
        "physics_runtime_s",
        "hotspot_status",
        "num_chiplets",
        "total_power_W",
        "package_width_mm",
        "package_height_mm",
        "whitespace_fraction",
        "mean_power_density_W_per_mm2",
        "max_power_density_W_per_mm2",
        "layout_hash",
    ]
    extra = sorted({key for row in rows for key in row} - set(base))
    return base + extra


def _write_case_stats_csv(path: Path, case_stats: dict[str, dict[str, float]]) -> None:
    rows = [{"case_id": case_id, **stats} for case_id, stats in sorted(case_stats.items())]
    fieldnames = ["case_id"] + sorted({key for row in rows for key in row if key != "case_id"})
    _write_csv(path, rows, fieldnames)


def _write_sample_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "type_counts"}
        item["type_counts_json"] = json.dumps(row.get("type_counts", {}), sort_keys=True)
        flat.append(item)
    fieldnames = sorted({key for row in flat for key in row})
    _write_csv(path, flat, fieldnames)


def _write_report_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# ChipTherm Benchmark Extension Audit",
        "",
        f"Stage: `{manifest['stage']}`",
        f"Samples: {manifest['sample_count']}",
        f"Validation passed: {manifest['validation']['passed']}",
        "",
        "## Splits",
        "",
    ]
    for split, count in sorted(manifest["split_counts"].items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Cases", ""])
    for case_id, count in sorted(manifest["case_counts"].items()):
        lines.append(f"- {case_id}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ChipTherm Benchmark Extension Validation",
        "",
        f"Passed: {report['passed']}",
        f"Samples: {report['sample_count']}",
        "",
        "## Problems",
        "",
    ]
    lines += [f"- {item}" for item in report["problems"]] or ["- none"]
    lines.extend(["", "## Warnings", ""])
    lines += [f"- {item}" for item in report["warnings"]] or ["- none"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
