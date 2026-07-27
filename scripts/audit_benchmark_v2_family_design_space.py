#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2 import TYPE_ORDER, compute_layout_descriptors, validate_family_spec  # noqa: E402
from chiptherm.benchmark_v2_training import EXPECTED_PRIMARY_SPLIT  # noqa: E402


EPS = 1.0e-12
DEFAULT_FAMILY_ROOT = REPO_ROOT / "configs/benchmark_v2_50family/families"
DEFAULT_WORKLOAD_SPEC = REPO_ROOT / "configs/benchmark_v2_50family/full_50x200_workload_cells.yaml"
IDENTITY_COLUMNS = {
    "family_uid",
    "split",
    "primary_category",
    "placement_style",
    "secondary_tags",
    "family_config_path",
    "substrate",
    "material_and_cooling_variant",
}
DISTANCE_EXCLUDE = {
    "workload_count",
    "grid_rows",
    "grid_cols",
    "hotspot_grid_rows",
    "hotspot_grid_cols",
}
REDUNDANCY_RMS_DISTANCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class TrainStandardizer:
    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    train_family_uids: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Formal repository-first design-space audit for the Benchmark v2 50-family dataset."
    )
    parser.add_argument("--family-config-root", type=Path, default=DEFAULT_FAMILY_ROOT)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument("--source-version", default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nearest-neighbor-k", type=int, default=5)
    args = parser.parse_args()
    if args.nearest_neighbor_k < 1:
        raise SystemExit("--nearest-neighbor-k must be at least 1")
    result = run_audit(
        family_config_root=args.family_config_root.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve() if args.data_root else None,
        source_version=args.source_version,
        split_manifest=args.split_manifest.expanduser().resolve() if args.split_manifest else None,
        out_dir=args.out_dir.expanduser().resolve(),
        nearest_neighbor_k=args.nearest_neighbor_k,
    )
    print("Benchmark v2 family design-space audit complete")
    print(f"Families: {result['family_count']}")
    print(f"Active distance descriptors: {result['active_distance_descriptor_count']}")
    print(f"Redundant pairs: {result['redundant_pair_count']}")
    print(f"Recommendation: {result['recommendation']['code']}")
    print(f"Output: {args.out_dir}")
    return 0


def run_audit(
    *,
    family_config_root: Path,
    data_root: Path | None,
    source_version: str | None,
    split_manifest: Path | None,
    out_dir: Path,
    nearest_neighbor_k: int,
) -> dict[str, Any]:
    family_specs = load_family_specs(family_config_root)
    split_by_uid = load_split_assignment(split_manifest, family_specs)
    workload_rows, workload_source = load_workload_rows(data_root, source_version)
    workload_summary, shared_template = summarize_workloads(
        family_specs,
        workload_rows,
        DEFAULT_WORKLOAD_SPEC,
    )
    records = [
        extract_family_descriptor(
            spec,
            config_path=path,
            split=split_by_uid[str(spec["family_uid"])],
            workload=workload_summary.get(str(spec["family_uid"]), {}),
        )
        for path, spec in family_specs
    ]
    numeric_names = numeric_descriptor_names(records)
    summary_rows = summarize_numeric_descriptors(records, numeric_names)
    standardizer = fit_train_standardizer(records, numeric_names)
    matrix, ordered_uids = standardized_matrix(records, standardizer)
    nearest_rows, redundant_pairs = nearest_family_rows(
        records,
        matrix,
        ordered_uids,
        k=nearest_neighbor_k,
        redundancy_threshold=REDUNDANCY_RMS_DISTANCE_THRESHOLD,
    )
    clusters = deterministic_kmeans(
        matrix,
        ordered_uids,
        cluster_count=max(2, round(math.sqrt(len(standardizer.train_family_uids)))),
        fit_uids=standardizer.train_family_uids,
    )
    cluster_rows = build_cluster_rows(records, clusters, matrix, ordered_uids)
    split_coverage = build_split_coverage(records, numeric_names, standardizer, nearest_rows, cluster_rows)
    pca = fit_train_pca(records, standardizer)
    gap_analysis = build_gap_analysis(
        records,
        summary_rows,
        workload_summary,
        shared_template=shared_template,
        redundant_pairs=redundant_pairs,
        clusters=cluster_rows,
    )
    recommendation = generate_recommendation(
        family_count=len(records),
        redundant_pairs=redundant_pairs,
        gap_analysis=gap_analysis,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "family_design_space.csv", records)
    write_csv(out_dir / "descriptor_summary.csv", summary_rows)
    write_csv(out_dir / "nearest_family_pairs.csv", nearest_rows)
    write_csv(out_dir / "family_clusters.csv", cluster_rows)
    write_csv(out_dir / "split_coverage_summary.csv", split_coverage)
    write_csv(
        out_dir / "workload_coverage_summary.csv",
        [dict(family_uid=uid, **values) for uid, values in sorted(workload_summary.items())],
    )
    audit_payload = {
        "schema_version": "benchmark_v2_family_design_space_audit/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "family_count": len(records),
        "split_counts": dict(Counter(str(row["split"]) for row in records)),
        "family_config_root": portable_or_absolute(family_config_root),
        "data_root_used": data_root is not None,
        "source_version": source_version,
        "workload_source": workload_source,
        "workload_template_shared_across_families": shared_template,
        "numeric_descriptor_count": len(numeric_names),
        "active_distance_descriptor_count": len(standardizer.names),
        "active_distance_descriptors": list(standardizer.names),
        "excluded_constant_or_nonstructural_descriptors": sorted(set(numeric_names) - set(standardizer.names)),
        "distance_definition": (
            "root-mean-square Euclidean distance after standardization by the 40 training-family "
            "means and standard deviations; constant training descriptors are excluded"
        ),
        "redundancy_threshold": REDUNDANCY_RMS_DISTANCE_THRESHOLD,
        "redundant_pair_count": len(redundant_pairs),
        "redundant_pairs": redundant_pairs,
        "gap_analysis": gap_analysis,
        "recommendation": recommendation,
        "limitations": [
            "Descriptor proximity is descriptive and is not evidence of equal thermal response.",
            "Repository-only mode cannot report realized total-power or active-chiplet distributions.",
            "All material, cooling, layer-stack, and HotSpot boundary-condition fields are audited as configured; unavailable fields are not invented.",
            "No HotSpot target or learned-model output enters descriptors, scaling, PCA, distances, clustering, or recommendations.",
        ],
    }
    write_json(out_dir / "benchmark_gap_analysis.json", audit_payload)
    write_report(
        out_dir / "benchmark_audit_report.md",
        records=records,
        summaries=summary_rows,
        nearest_rows=nearest_rows,
        cluster_rows=cluster_rows,
        gap_analysis=gap_analysis,
        recommendation=recommendation,
        workload_source=workload_source,
    )
    write_extension_spec(
        out_dir / "proposed_benchmark_extension_spec.md",
        recommendation=recommendation,
        records=records,
        gap_analysis=gap_analysis,
    )
    write_figures(
        out_dir=out_dir,
        records=records,
        standardizer=standardizer,
        pca=pca,
        nearest_rows=nearest_rows,
        split_coverage=split_coverage,
    )
    return audit_payload


def load_family_specs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob("f[0-9][0-9][0-9].yaml"))
    if len(paths) != 50:
        raise ValueError(f"expected exactly 50 family YAML files under {root}, found {len(paths)}")
    output: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for path in paths:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        uid = str(spec.get("family_uid", ""))
        if not uid or uid in seen:
            raise ValueError(f"missing or duplicate family_uid in {path}: {uid!r}")
        problems = validate_family_spec(spec)
        if problems:
            raise ValueError(f"invalid canonical family {uid}: {'; '.join(problems)}")
        seen.add(uid)
        output.append((path, spec))
    return output


def load_split_assignment(
    split_manifest: Path | None,
    family_specs: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, str]:
    assignments = {str(spec["family_uid"]): str(spec["primary_split"]) for _, spec in family_specs}
    if split_manifest is None:
        return assignments
    payload = load_structured(split_manifest)
    split_payload = payload.get("primary_family_split", payload)
    overridden: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for uid in split_payload.get(split, []):
            uid = str(uid)
            if uid in overridden:
                raise ValueError(f"family {uid} appears in multiple splits in {split_manifest}")
            overridden[uid] = split
    if set(overridden) != set(assignments):
        missing = sorted(set(assignments) - set(overridden))
        extra = sorted(set(overridden) - set(assignments))
        raise ValueError(f"split manifest does not cover canonical families exactly; missing={missing}, extra={extra}")
    return overridden


def extract_family_descriptor(
    spec: Mapping[str, Any],
    *,
    config_path: Path,
    split: str,
    workload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    structure = require_mapping(spec, "fixed_structure")
    layout = require_mapping(structure, "layout")
    package = require_mapping(require_mapping(layout, "package"), "size")
    chiplets = list(layout.get("chiplets", []))
    if not chiplets:
        raise ValueError(f"{spec.get('family_uid')} has no chiplets")
    width = float(package["width"])
    height = float(package["height"])
    grid = require_mapping(structure, "grid")
    rows, cols = int(grid["rows"]), int(grid["cols"])
    canonical = compute_layout_descriptors(dict(layout))
    rects = [chiplet_rectangle(item) for item in chiplets]
    chiplet_widths = np.asarray([rect[2] for rect in rects], dtype=np.float64)
    chiplet_heights = np.asarray([rect[3] for rect in rects], dtype=np.float64)
    areas = chiplet_widths * chiplet_heights
    aspects = np.maximum(chiplet_widths, chiplet_heights) / np.minimum(chiplet_widths, chiplet_heights)
    centers = np.asarray([[x + w / 2.0, y + h / 2.0] for x, y, w, h in rects], dtype=np.float64)
    boundaries = np.asarray([rectangle_boundary_clearance(rect, width, height) for rect in rects])
    gaps = [rectangle_gap(first, second) for index, first in enumerate(rects) for second in rects[index + 1 :]]
    center_distances = [
        float(np.linalg.norm(centers[index] - centers[other]))
        for index in range(len(centers))
        for other in range(index + 1, len(centers))
    ]
    type_counts = Counter(str(item.get("type", "OTHER")) for item in chiplets)
    centroid = np.mean(centers, axis=0)
    normalized_centers = np.column_stack((centers[:, 0] / width, centers[:, 1] / height))
    centered_norm = normalized_centers - np.asarray([0.5, 0.5])
    edge_heavy_score = float(np.mean(np.max(np.abs(centered_norm), axis=1) / 0.5))
    dispersion = float(np.sqrt(np.var(normalized_centers[:, 0]) + np.var(normalized_centers[:, 1])))
    centroid_offset = float(
        math.hypot((centroid[0] - width / 2.0) / width, (centroid[1] - height / 2.0) / height)
    )
    reflection_asymmetry = reflection_asymmetry_score(normalized_centers)
    near_boundary_threshold = 0.10 * min(width, height)
    thermal = require_mapping(structure, "thermal_stack")
    hotspot = require_mapping(structure, "hotspot")

    record: dict[str, Any] = {
        "family_uid": str(spec["family_uid"]),
        "split": split,
        "primary_category": str(spec.get("primary_category", "")),
        "placement_style": str(spec.get("placement_style", "")),
        "secondary_tags": ";".join(str(value) for value in spec.get("secondary_tags", [])),
        "family_config_path": portable_or_absolute(config_path),
        "substrate": str(require_mapping(layout, "package").get("substrate", "")),
        "material_and_cooling_variant": str(structure.get("material_and_cooling_variant", "")),
        "workload_count": int((workload or {}).get("workload_count", 200)),
        "package_width_mm": width,
        "package_height_mm": height,
        "package_area_mm2": width * height,
        "package_aspect_ratio": max(width, height) / min(width, height),
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_dx_mm": width / cols,
        "grid_dy_mm": height / rows,
        "grid_cell_anisotropy_ratio": max(width / cols, height / rows) / min(width / cols, height / rows),
        "chiplet_count": len(chiplets),
        "distinct_chiplet_type_count": len(type_counts),
        "total_chiplet_area_mm2": float(areas.sum()),
        "occupied_area_ratio": float(areas.sum() / (width * height)),
        "whitespace_ratio": float(1.0 - areas.sum() / (width * height)),
        "chiplet_width_min_mm": float(chiplet_widths.min()),
        "chiplet_width_mean_mm": float(chiplet_widths.mean()),
        "chiplet_width_max_mm": float(chiplet_widths.max()),
        "chiplet_height_min_mm": float(chiplet_heights.min()),
        "chiplet_height_mean_mm": float(chiplet_heights.mean()),
        "chiplet_height_max_mm": float(chiplet_heights.max()),
        "chiplet_area_min_mm2": float(areas.min()),
        "chiplet_area_mean_mm2": float(areas.mean()),
        "chiplet_area_max_mm2": float(areas.max()),
        "chiplet_area_coefficient_of_variation": coefficient_of_variation(areas),
        "chiplet_width_coefficient_of_variation": coefficient_of_variation(chiplet_widths),
        "chiplet_height_coefficient_of_variation": coefficient_of_variation(chiplet_heights),
        "chiplet_aspect_ratio_min": float(aspects.min()),
        "chiplet_aspect_ratio_mean": float(aspects.mean()),
        "chiplet_aspect_ratio_std": float(aspects.std()),
        "chiplet_aspect_ratio_max": float(aspects.max()),
        "chiplet_size_heterogeneity_index": float(
            0.5 * (coefficient_of_variation(areas) + coefficient_of_variation(aspects))
        ),
        "minimum_rectangle_gap_mm": float(min(gaps) if gaps else 0.0),
        "mean_rectangle_gap_mm": float(np.mean(gaps) if gaps else 0.0),
        "minimum_center_spacing_mm": float(min(center_distances) if center_distances else 0.0),
        "mean_center_spacing_mm": float(np.mean(center_distances) if center_distances else 0.0),
        "minimum_boundary_clearance_mm": float(boundaries.min()),
        "mean_boundary_clearance_mm": float(boundaries.mean()),
        "near_boundary_fraction": float(np.mean(boundaries <= near_boundary_threshold)),
        "near_boundary_threshold_mm": near_boundary_threshold,
        "edge_heavy_placement_score": edge_heavy_score,
        "center_heavy_placement_score": 1.0 - edge_heavy_score,
        "normalized_spatial_dispersion": dispersion,
        "placement_centroid_x_fraction": float(centroid[0] / width),
        "placement_centroid_y_fraction": float(centroid[1] / height),
        "placement_centroid_offset": centroid_offset,
        "reflection_asymmetry_score": reflection_asymmetry,
    }
    for chiplet_type in TYPE_ORDER:
        count = int(type_counts.get(chiplet_type, 0))
        record[f"type_{chiplet_type}_count"] = count
        record[f"type_{chiplet_type}_fraction"] = count / len(chiplets)
    add_thermal_context(record, thermal, hotspot)
    for key, value in (workload or {}).items():
        if key != "workload_count":
            record[f"workload_{key}"] = value
    validate_against_canonical_descriptors(record, canonical)
    ensure_finite_numeric(record, str(spec["family_uid"]))
    return record


def chiplet_rectangle(chiplet: Mapping[str, Any]) -> tuple[float, float, float, float]:
    position = require_mapping(chiplet, "position")
    size = require_mapping(chiplet, "size")
    return float(position["x"]), float(position["y"]), float(size["width"]), float(size["height"])


def rectangle_gap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    dx = max(ax - (bx + bw), bx - (ax + aw), 0.0)
    dy = max(ay - (by + bh), by - (ay + ah), 0.0)
    return math.hypot(dx, dy)


def rectangle_boundary_clearance(
    rectangle: tuple[float, float, float, float],
    package_width: float,
    package_height: float,
) -> float:
    x, y, width, height = rectangle
    return min(x, y, package_width - x - width, package_height - y - height)


def reflection_asymmetry_score(normalized_centers: np.ndarray) -> float:
    if len(normalized_centers) == 0:
        return 0.0
    scores: list[float] = []
    for reflected in (
        np.column_stack((1.0 - normalized_centers[:, 0], normalized_centers[:, 1])),
        np.column_stack((normalized_centers[:, 0], 1.0 - normalized_centers[:, 1])),
    ):
        nearest = [
            float(np.min(np.linalg.norm(normalized_centers - point[None, :], axis=1)))
            for point in reflected
        ]
        scores.append(float(np.mean(nearest) / math.sqrt(2.0)))
    return float(np.mean(scores))


def add_thermal_context(
    record: dict[str, Any],
    thermal: Mapping[str, Any],
    hotspot: Mapping[str, Any],
) -> None:
    record["ambient_temperature_K"] = float(thermal["ambient_K"])
    record["initial_temperature_K"] = float(thermal["initial_temperature_K"])
    for layer in ("chip", "interface", "spreader", "sink"):
        payload = require_mapping(thermal, layer)
        for key, value in payload.items():
            if isinstance(value, (int, float, bool)):
                record[f"{layer}_{key}"] = float(value)
    hotspot_grid = require_mapping(hotspot, "grid")
    record.update(
        {
            "hotspot_grid_rows": int(hotspot_grid["rows"]),
            "hotspot_grid_cols": int(hotspot_grid["cols"]),
            "hotspot_sampling_interval_s": float(hotspot["sampling_interval_s"]),
            "hotspot_base_processor_frequency_Hz": float(hotspot["base_processor_frequency_Hz"]),
            "hotspot_leakage_used": int(bool(hotspot["leakage_used"])),
            "hotspot_detailed_package": int(bool(hotspot["detailed_package"])),
            "hotspot_secondary_path": int(bool(hotspot["secondary_path"])),
        }
    )


def validate_against_canonical_descriptors(record: Mapping[str, Any], canonical: Mapping[str, Any]) -> None:
    checks = {
        "package_width_mm": "package_width_mm",
        "package_height_mm": "package_height_mm",
        "package_area_mm2": "package_area_mm2",
        "chiplet_count": "chiplet_count",
        "total_chiplet_area_mm2": "occupied_area_mm2",
        "occupied_area_ratio": "occupied_fraction",
        "minimum_rectangle_gap_mm": "minimum_chiplet_gap_mm",
        "minimum_boundary_clearance_mm": "minimum_package_edge_clearance_mm",
    }
    for extracted, stored in checks.items():
        if not math.isclose(float(record[extracted]), float(canonical[stored]), rel_tol=1.0e-9, abs_tol=1.0e-8):
            raise ValueError(f"descriptor extraction mismatch for {extracted} versus canonical {stored}")


def load_workload_rows(
    data_root: Path | None,
    source_version: str | None,
) -> tuple[list[dict[str, str]], str]:
    if data_root is None:
        return [], "repository_workload_template_only"
    coverage = data_root / "canonical/manifests/full_50x200_workload_coverage.csv"
    if coverage.is_file():
        return read_csv(coverage), data_root_relative_or_absolute(coverage, data_root)
    candidates: list[Path] = []
    if source_version:
        root = data_root / "derived/indices/full_50x200/source_superposition" / source_version / "family_split"
        candidates.extend(root / f"{split}_index.csv" for split in ("train", "val", "test"))
    rows: list[dict[str, str]] = []
    existing = [path for path in candidates if path.is_file()]
    for path in existing:
        rows.extend(read_csv(path))
    if rows:
        deduplicated = {str(row.get("sample_uid", index)): row for index, row in enumerate(rows)}
        return list(deduplicated.values()), ";".join(data_root_relative_or_absolute(path, data_root) for path in existing)
    return [], "repository_workload_template_only_data_root_had_no_supported_index"


def summarize_workloads(
    family_specs: Sequence[tuple[Path, Mapping[str, Any]]],
    workload_rows: Sequence[Mapping[str, str]],
    workload_spec_path: Path,
) -> tuple[dict[str, dict[str, Any]], bool]:
    family_uids = [str(spec["family_uid"]) for _, spec in family_specs]
    if not workload_rows:
        workload_spec = yaml.safe_load(workload_spec_path.read_text(encoding="utf-8"))
        power_keys = [str(item["key"]) for item in workload_spec.get("power_regimes", [])]
        topology_keys = [str(item) for item in workload_spec.get("topology_regimes", [])]
        summary = {}
        for uid in family_uids:
            row: dict[str, Any] = {
                "workload_count": int(workload_spec.get("cell_count", 200)),
                "realized_power_statistics_available": 0,
                "shared_template_coverage": 1,
            }
            for key in power_keys:
                row[f"power_regime_{key}_count"] = len(topology_keys)
            for key in topology_keys:
                row[f"topology_regime_{key}_count"] = len(power_keys)
            summary[uid] = row
        return summary, True

    by_family: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in workload_rows:
        uid = str(row.get("family_uid") or row.get("case_id") or "")
        if uid:
            by_family[uid].append(row)
    output: dict[str, dict[str, Any]] = {}
    cell_sets: list[set[str]] = []
    for uid in family_uids:
        rows = by_family.get(uid, [])
        total_power = finite_column(rows, ("total_package_power_W", "total_power_W"))
        active_counts = finite_column(rows, ("active_chiplet_count",))
        if not active_counts:
            fractions = finite_column(rows, ("active_chiplet_fraction",))
            chiplet_count = int(next(spec for _, spec in family_specs if spec["family_uid"] == uid)["descriptors"]["chiplet_count"])
            active_counts = [value * chiplet_count for value in fractions]
        power_counts = Counter(str(row.get("power_regime", "")) for row in rows if row.get("power_regime"))
        topology_counts = Counter(str(row.get("topology_regime", "")) for row in rows if row.get("topology_regime"))
        cells = {
            str(row.get("workload_cell") or f"{row.get('power_regime', '')}|{row.get('topology_regime', '')}")
            for row in rows
        }
        cell_sets.append(cells)
        summary: dict[str, Any] = {
            "workload_count": len(rows),
            "realized_power_statistics_available": int(bool(total_power)),
            "total_power_min_W": min(total_power) if total_power else "",
            "total_power_mean_W": float(np.mean(total_power)) if total_power else "",
            "total_power_std_W": float(np.std(total_power)) if total_power else "",
            "total_power_max_W": max(total_power) if total_power else "",
            "active_chiplet_count_min": min(active_counts) if active_counts else "",
            "active_chiplet_count_max": max(active_counts) if active_counts else "",
        }
        for key, count in sorted(power_counts.items()):
            summary[f"power_regime_{sanitize_name(key)}_count"] = count
        for key, count in sorted(topology_counts.items()):
            summary[f"topology_regime_{sanitize_name(key)}_count"] = count
        summary.update(workload_category_counts(rows, active_counts))
        output[uid] = summary
    shared = bool(cell_sets) and all(cells == cell_sets[0] for cells in cell_sets[1:])
    for summary in output.values():
        summary["shared_template_coverage"] = int(shared)
    return output, shared


def workload_category_counts(
    rows: Sequence[Mapping[str, str]],
    active_counts: Sequence[float],
) -> dict[str, int]:
    topologies = [str(row.get("topology_regime", "")).lower() for row in rows]
    return {
        "sparse_activity_count": sum("sparse" in value or "single_source" in value for value in topologies),
        "dense_activity_count": sum("dense" in value for value in topologies),
        "balanced_count": sum("balanced" in value or "symmetric" in value for value in topologies),
        "skewed_or_dominant_count": sum("dominant" in value or "asymmetric" in value for value in topologies),
        "interacting_source_count": sum(
            any(token in value for token in ("interaction", "two_source", "three_source", "cross_type"))
            for value in topologies
        ),
        "active_count_observation_count": len(active_counts),
    }


def numeric_descriptor_names(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    common = set(records[0])
    for record in records[1:]:
        common &= set(record)
    names = []
    for name in sorted(common - IDENTITY_COLUMNS):
        values = [record[name] for record in records]
        if all(isinstance(value, (int, float, bool, np.number)) and np.isfinite(float(value)) for value in values):
            names.append(name)
    return tuple(names)


def summarize_numeric_descriptors(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in names:
        values = np.asarray([float(row[name]) for row in records], dtype=np.float64)
        row: dict[str, Any] = {
            "descriptor": name,
            "min": float(values.min()),
            "q1": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "q3": float(np.quantile(values, 0.75)),
            "max": float(values.max()),
            "std": float(values.std()),
            "unique_values": unique_numeric_count(values),
        }
        row["variation_class"] = variation_class(int(row["unique_values"]))
        train_values = np.asarray([float(item[name]) for item in records if item["split"] == "train"])
        row["train_min"] = float(train_values.min())
        row["train_max"] = float(train_values.max())
        for split in ("val", "test"):
            split_values = np.asarray([float(item[name]) for item in records if item["split"] == split])
            row[f"{split}_min"] = float(split_values.min())
            row[f"{split}_max"] = float(split_values.max())
            outside = (split_values < train_values.min() - 1.0e-9) | (split_values > train_values.max() + 1.0e-9)
            row[f"{split}_outside_train_count"] = int(outside.sum())
        output.append(row)
    return output


def fit_train_standardizer(
    records: Sequence[Mapping[str, Any]],
    candidate_names: Sequence[str],
) -> TrainStandardizer:
    train = sorted((row for row in records if row["split"] == "train"), key=lambda row: str(row["family_uid"]))
    if not train:
        raise ValueError("cannot fit descriptor standardization without training families")
    active = []
    for name in candidate_names:
        values = np.asarray([float(row[name]) for row in train], dtype=np.float64)
        if float(values.std()) > EPS and name not in DISTANCE_EXCLUDE and not name.endswith("_threshold_mm"):
            active.append(name)
    if not active:
        raise ValueError("no varying training descriptors are available for distance analysis")
    matrix = np.asarray([[float(row[name]) for name in active] for row in train], dtype=np.float64)
    scale = matrix.std(axis=0)
    if not np.isfinite(matrix).all() or np.any(scale <= EPS):
        raise ValueError("non-finite or constant descriptors leaked into train standardizer")
    return TrainStandardizer(
        names=tuple(active),
        mean=matrix.mean(axis=0),
        scale=scale,
        minimum=matrix.min(axis=0),
        maximum=matrix.max(axis=0),
        train_family_uids=tuple(str(row["family_uid"]) for row in train),
    )


def standardized_matrix(
    records: Sequence[Mapping[str, Any]],
    standardizer: TrainStandardizer,
) -> tuple[np.ndarray, tuple[str, ...]]:
    ordered = sorted(records, key=lambda row: str(row["family_uid"]))
    raw = np.asarray([[float(row[name]) for name in standardizer.names] for row in ordered], dtype=np.float64)
    matrix = (raw - standardizer.mean) / standardizer.scale
    if not np.isfinite(matrix).all():
        raise ValueError("standardized family descriptor matrix contains non-finite values")
    return matrix, tuple(str(row["family_uid"]) for row in ordered)


def rms_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def nearest_family_rows(
    records: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    ordered_uids: Sequence[str],
    *,
    k: int,
    redundancy_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_uid = {str(row["family_uid"]): row for row in records}
    train_indices = [index for index, uid in enumerate(ordered_uids) if by_uid[str(uid)]["split"] == "train"]
    output: list[dict[str, Any]] = []
    unique_redundant: dict[tuple[str, str], dict[str, Any]] = {}
    for index, uid in enumerate(ordered_uids):
        candidates = sorted(
            (
                (rms_distance(matrix[index], matrix[other]), str(other_uid))
                for other, other_uid in enumerate(ordered_uids)
                if other != index
            ),
            key=lambda item: (item[0], item[1]),
        )
        training_candidates = sorted(
            (
                (rms_distance(matrix[index], matrix[other]), str(ordered_uids[other]))
                for other in train_indices
                if other != index
            ),
            key=lambda item: (item[0], item[1]),
        )
        nearest_train_distance, nearest_train_uid = training_candidates[0]
        for rank, (distance, neighbor) in enumerate(candidates[: min(k, len(candidates))], start=1):
            redundant = distance <= redundancy_threshold
            row = {
                "family_uid": uid,
                "split": by_uid[uid]["split"],
                "neighbor_rank": rank,
                "neighbor_family_uid": neighbor,
                "neighbor_split": by_uid[neighbor]["split"],
                "rms_standardized_distance": distance,
                "same_primary_category": int(by_uid[uid]["primary_category"] == by_uid[neighbor]["primary_category"]),
                "nearly_redundant": int(redundant),
                "nearest_training_family_uid": nearest_train_uid,
                "nearest_training_rms_standardized_distance": nearest_train_distance,
            }
            output.append(row)
            if redundant:
                pair = tuple(sorted((uid, neighbor)))
                unique_redundant[pair] = {
                    "family_a": pair[0],
                    "family_b": pair[1],
                    "rms_standardized_distance": distance,
                    "same_primary_category": row["same_primary_category"],
                }
    redundant_rows = sorted(
        unique_redundant.values(),
        key=lambda row: (float(row["rms_standardized_distance"]), row["family_a"], row["family_b"]),
    )
    return output, redundant_rows


def deterministic_kmeans(
    matrix: np.ndarray,
    ordered_uids: Sequence[str],
    *,
    cluster_count: int,
    fit_uids: Sequence[str] | None = None,
    max_iterations: int = 100,
) -> dict[str, int]:
    fit_uid_set = set(str(uid) for uid in (fit_uids or ordered_uids))
    fit_indices = [index for index, uid in enumerate(ordered_uids) if str(uid) in fit_uid_set]
    fit_matrix = matrix[fit_indices]
    if not 1 <= cluster_count <= len(fit_matrix):
        raise ValueError("invalid deterministic cluster count")
    center_indices = [0]
    while len(center_indices) < cluster_count:
        distances = np.asarray(
            [
                min(rms_distance(fit_matrix[index], fit_matrix[center]) for center in center_indices)
                for index in range(len(fit_matrix))
            ]
        )
        for selected in center_indices:
            distances[selected] = -1.0
        center_indices.append(int(np.argmax(distances)))
    centers = fit_matrix[center_indices].copy()
    fit_labels = np.zeros(len(fit_matrix), dtype=np.int64)
    for _ in range(max_iterations):
        distance_matrix = np.asarray(
            [[rms_distance(row, center) for center in centers] for row in fit_matrix],
            dtype=np.float64,
        )
        new_labels = np.argmin(distance_matrix, axis=1)
        if np.array_equal(new_labels, fit_labels) and _ > 0:
            break
        fit_labels = new_labels
        for cluster in range(cluster_count):
            members = fit_matrix[fit_labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    labels = np.argmin(
        np.asarray([[rms_distance(row, center) for center in centers] for row in matrix]),
        axis=1,
    )
    return {str(uid): int(label) + 1 for uid, label in zip(ordered_uids, labels, strict=True)}


def build_cluster_rows(
    records: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, int],
    matrix: np.ndarray,
    ordered_uids: Sequence[str],
) -> list[dict[str, Any]]:
    by_uid = {str(row["family_uid"]): row for row in records}
    counts = Counter(clusters.values())
    split_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for uid, cluster in clusters.items():
        split_counts[cluster][str(by_uid[uid]["split"])] += 1
    output = []
    for uid in ordered_uids:
        cluster = clusters[uid]
        output.append(
            {
                "family_uid": uid,
                "split": by_uid[uid]["split"],
                "primary_category": by_uid[uid]["primary_category"],
                "cluster_id": cluster,
                "cluster_size": counts[cluster],
                "cluster_train_count": split_counts[cluster]["train"],
                "cluster_val_count": split_counts[cluster]["val"],
                "cluster_test_count": split_counts[cluster]["test"],
            }
        )
    return output


def build_split_coverage(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    standardizer: TrainStandardizer,
    nearest_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    nearest_first = {str(row["family_uid"]): row for row in nearest_rows if int(row["neighbor_rank"]) == 1}
    cluster_by_uid = {str(row["family_uid"]): row for row in cluster_rows}
    output: list[dict[str, Any]] = []
    for name in names:
        train = np.asarray([float(row[name]) for row in records if row["split"] == "train"])
        row: dict[str, Any] = {
            "descriptor": name,
            "train_min": float(train.min()),
            "train_max": float(train.max()),
            "train_unique": unique_numeric_count(train),
        }
        for split in ("val", "test"):
            values = np.asarray([float(item[name]) for item in records if item["split"] == split])
            outside = (values < train.min() - 1.0e-9) | (values > train.max() + 1.0e-9)
            row[f"{split}_min"] = float(values.min())
            row[f"{split}_max"] = float(values.max())
            row[f"{split}_outside_train_count"] = int(outside.sum())
            row[f"{split}_inside_train_fraction"] = float(1.0 - outside.mean())
        output.append(row)
    for record in records:
        if record["split"] == "train":
            continue
        uid = str(record["family_uid"])
        raw = np.asarray([float(record[name]) for name in standardizer.names])
        z = (raw - standardizer.mean) / standardizer.scale
        nearest = nearest_first[uid]
        output.append(
            {
                "descriptor": f"joint_combination::{uid}",
                "heldout_split": record["split"],
                "nearest_family_uid": nearest["nearest_training_family_uid"],
                "nearest_rms_standardized_distance": nearest["nearest_training_rms_standardized_distance"],
                "max_abs_train_zscore": float(np.max(np.abs(z))),
                "outside_train_range_count": int(
                    ((raw < standardizer.minimum - 1.0e-9) | (raw > standardizer.maximum + 1.0e-9)).sum()
                ),
                "cluster_id": cluster_by_uid[uid]["cluster_id"],
                "cluster_train_count": cluster_by_uid[uid]["cluster_train_count"],
            }
        )
    return output


def fit_train_pca(
    records: Sequence[Mapping[str, Any]],
    standardizer: TrainStandardizer,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: str(row["family_uid"]))
    raw = np.asarray([[float(row[name]) for name in standardizer.names] for row in ordered])
    standardized = (raw - standardizer.mean) / standardizer.scale
    train_mask = np.asarray([row["split"] == "train" for row in ordered])
    train = standardized[train_mask]
    train_center = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - train_center, full_matrices=False)
    components = vt[:2].copy()
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    coordinates = (standardized - train_center) @ components.T
    variance = singular * singular
    ratios = variance[:2] / max(float(variance.sum()), EPS)
    return {
        "family_uids": [str(row["family_uid"]) for row in ordered],
        "splits": [str(row["split"]) for row in ordered],
        "coordinates": coordinates,
        "explained_variance_ratio": ratios,
    }


def build_gap_analysis(
    records: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    workload_summary: Mapping[str, Mapping[str, Any]],
    *,
    shared_template: bool,
    redundant_pairs: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_name = {str(row["descriptor"]): row for row in summaries}

    def category(names: Sequence[str], *, unsupported: bool = False) -> str:
        if unsupported:
            return "unsupported_by_current_benchmark"
        unique = min(int(by_name[name]["unique_values"]) for name in names if name in by_name)
        if unique <= 1:
            return "fixed_not_varied"
        if unique <= 3:
            return "weakly_covered"
        if unique <= 5:
            return "partially_covered"
        return "well_covered"

    dimensions = {
        "chiplet_count_diversity": category(["chiplet_count"]),
        "package_size_diversity": category(["package_width_mm", "package_height_mm", "package_area_mm2"]),
        "package_aspect_ratio_diversity": category(["package_aspect_ratio"]),
        "chiplet_size_and_aspect_diversity": category(
            ["chiplet_area_mean_mm2", "chiplet_aspect_ratio_mean", "chiplet_size_heterogeneity_index"]
        ),
        "occupied_area_diversity": category(["occupied_area_ratio"]),
        "boundary_and_spacing_diversity": category(
            ["minimum_boundary_clearance_mm", "minimum_rectangle_gap_mm", "edge_heavy_placement_score"]
        ),
        "heterogeneous_chiplet_types": category(["distinct_chiplet_type_count"]),
        "package_material_variation": category(
            ["chip_thermal_conductivity_W_per_mK", "interface_thermal_conductivity_W_per_mK"]
        ),
        "cooling_boundary_condition_variation": category(
            ["sink_convection_resistance_K_per_W", "hotspot_secondary_path"]
        ),
        "layer_stack_variation": category(["chip_thickness_m", "interface_thickness_m", "sink_thickness_m"]),
        "workload_template_diversity": "well_covered" if shared_template and workload_summary else "partially_covered",
        "active_passive_static_chiplet_labels": category([], unsupported=True),
        "multi_layer_heat_source_placement": category([], unsupported=True),
        "interposer_specific_thermal_conductivity": category([], unsupported=True),
    }
    fixed = sorted(name for name, status in dimensions.items() if status == "fixed_not_varied")
    weak = sorted(
        name
        for name, status in dimensions.items()
        if status in {"weakly_covered", "unsupported_by_current_benchmark"}
    )
    cluster_counts = Counter(int(row["cluster_id"]) for row in clusters)
    train_clusters = {
        int(row["cluster_id"]) for row in clusters if row["split"] == "train"
    }
    heldout_without_train_cluster = sorted(
        str(row["family_uid"])
        for row in clusters
        if row["split"] != "train" and int(row["cluster_id"]) not in train_clusters
    )
    return {
        "dimension_coverage": dimensions,
        "fixed_dimensions": fixed,
        "weak_or_unsupported_dimensions": weak,
        "redundant_pair_count": len(redundant_pairs),
        "redundant_family_uids": sorted(
            {str(row["family_a"]) for row in redundant_pairs} | {str(row["family_b"]) for row in redundant_pairs}
        ),
        "cluster_count": len(cluster_counts),
        "cluster_sizes": {str(key): value for key, value in sorted(cluster_counts.items())},
        "underpopulated_clusters": [key for key, value in sorted(cluster_counts.items()) if value <= 2],
        "overrepresented_clusters": [key for key, value in sorted(cluster_counts.items()) if value >= 2 * np.median(list(cluster_counts.values()))],
        "heldout_families_without_training_cluster": heldout_without_train_cluster,
        "marginal_vs_joint_interpretation": (
            "Marginal train-range coverage is reported per descriptor. Joint coverage is assessed separately "
            "through train-standardized nearest-neighbor distance and cluster occupancy."
        ),
    }


def generate_recommendation(
    *,
    family_count: int,
    redundant_pairs: Sequence[Mapping[str, Any]],
    gap_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    redundant_uids = set(gap_analysis.get("redundant_family_uids", []))
    critical_fixed = {
        "package_material_variation",
        "cooling_boundary_condition_variation",
        "layer_stack_variation",
    } & set(gap_analysis.get("fixed_dimensions", []))
    unsupported = {
        name
        for name, status in gap_analysis.get("dimension_coverage", {}).items()
        if status == "unsupported_by_current_benchmark"
    }
    if family_count < 30:
        code = "D"
        title = "A larger benchmark expansion is justified"
    elif critical_fixed or unsupported:
        code = "C"
        title = "Preserve the existing 50 and add a small structured challenge extension"
    elif len(redundant_uids) >= max(10, round(0.25 * family_count)):
        code = "B"
        title = "Keep 50 total but replace redundant families in a new benchmark version"
    else:
        code = "A"
        title = "Keep the existing 50 families unchanged"
    return {
        "code": code,
        "title": title,
        "numerical_family_count_sufficient": family_count >= 40,
        "effective_design_space_coverage_sufficient": code == "A",
        "redundant_pair_count": len(redundant_pairs),
        "redundant_family_count": len(redundant_uids),
        "missing_physical_dimensions": sorted(critical_fixed | unsupported),
        "current_result_validity": (
            "Current results remain valid for Benchmark v2 as defined. A versioned extension must be reported "
            "as an additional protocol and must not rewrite or silently replace the accepted 50-family benchmark."
        ),
        "smallest_next_dataset_change": (
            "No extension justified."
            if code == "A"
            else "Add a versioned, small factorial challenge set spanning material/stack and cooling axes while retaining all existing families."
        ),
    }


def write_report(
    path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    nearest_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    gap_analysis: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    workload_source: str,
) -> None:
    fixed = [row["descriptor"] for row in summaries if row["variation_class"] == "fixed"]
    low_unique = [
        f"{row['descriptor']} ({row['unique_values']})"
        for row in summaries
        if row["variation_class"] == "low_cardinality"
    ]
    heldout_joint = [
        row for row in nearest_rows if int(row["neighbor_rank"]) == 1 and row["split"] != "train"
    ]
    heldout_joint.sort(key=lambda row: float(row["rms_standardized_distance"]), reverse=True)
    cluster_counts = Counter(int(row["cluster_id"]) for row in cluster_rows)
    lines = [
        "# Benchmark v2 50-Family Design-Space Audit",
        "",
        "## Scope",
        "",
        f"- Families audited: {len(records)}.",
        f"- Split: {dict(Counter(str(row['split']) for row in records))}.",
        f"- Workload evidence: `{workload_source}`.",
        "- No HotSpot target, model prediction, or learned error metric is used.",
        "",
        "## Coverage",
        "",
    ]
    for dimension, status in gap_analysis["dimension_coverage"].items():
        lines.append(f"- `{dimension}`: **{status.replace('_', ' ')}**.")
    lines.extend(
        [
            "",
            "## Fixed And Low-Cardinality Axes",
            "",
            f"- Fixed numeric descriptors: {', '.join(f'`{name}`' for name in fixed) or 'none'}.",
            f"- Two-to-three-value descriptors: {', '.join(f'`{name}`' for name in low_unique) or 'none'}.",
            "",
            "## Joint Coverage",
            "",
            (
                "- Held-out nearest-training distances: "
                + "; ".join(
                    f"`{row['family_uid']}` -> `{row['nearest_training_family_uid']}` "
                    f"({float(row['nearest_training_rms_standardized_distance']):.3f})"
                    for row in heldout_joint
                )
                + "."
            ),
            f"- Deterministic cluster sizes: {dict(sorted(cluster_counts.items()))}.",
            f"- Nearly redundant pairs: {gap_analysis['redundant_pair_count']}.",
            "- Distances describe representation proximity, not causal or thermal equivalence.",
            "",
            "## Recommendation",
            "",
            f"**{recommendation['code']}. {recommendation['title']}**",
            "",
            f"- Fifty families numerically sufficient: `{recommendation['numerical_family_count_sufficient']}`.",
            f"- Effective coverage sufficient: `{recommendation['effective_design_space_coverage_sufficient']}`.",
            f"- Families implicated in redundant pairs: {recommendation['redundant_family_count']}.",
            f"- Missing/fixed physical axes: {', '.join(recommendation['missing_physical_dimensions']) or 'none'}.",
            f"- Smallest next change: {recommendation['smallest_next_dataset_change']}",
            f"- Result validity: {recommendation['current_result_validity']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_extension_spec(
    path: Path,
    *,
    recommendation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    gap_analysis: Mapping[str, Any],
) -> None:
    if recommendation["code"] == "A":
        path.write_text("# Proposed Benchmark Extension\n\nNo extension justified.\n", encoding="utf-8")
        return
    chiplet_counts = np.asarray([float(row["chiplet_count"]) for row in records])
    package_areas = np.asarray([float(row["package_area_mm2"]) for row in records])
    occupied = np.asarray([float(row["occupied_area_ratio"]) for row in records])
    gaps = np.asarray([float(row["minimum_rectangle_gap_mm"]) for row in records])
    boundary = np.asarray([float(row["minimum_boundary_clearance_mm"]) for row in records])
    lines = [
        "# Proposed Benchmark v2 Challenge Extension Specification",
        "",
        "This is a coverage specification only. It does not define new family YAML files and does not replace the accepted 50 families.",
        "",
        "## Structural Targets",
        "",
        f"- Chiplet-count strata spanning current `{chiplet_counts.min():.0f}-{chiplet_counts.max():.0f}` plus one validated higher-count challenge bin.",
        f"- Package-area strata spanning current `{package_areas.min():.0f}-{package_areas.max():.0f} mm2`, crossed with low/medium/high aspect ratio.",
        f"- Occupied-area bins: below `{np.quantile(occupied, 0.25):.2f}`, central, and above `{np.quantile(occupied, 0.75):.2f}`.",
        f"- Minimum spacing bins around current quartiles `{np.quantile(gaps, 0.25):.2f}` and `{np.quantile(gaps, 0.75):.2f} mm`.",
        f"- Boundary-clearance bins around current quartiles `{np.quantile(boundary, 0.25):.2f}` and `{np.quantile(boundary, 0.75):.2f} mm`.",
        "- Placement categories: center-heavy, edge-heavy, corner-heavy, clustered, distributed, and asymmetric.",
        "- Type composition categories: compute-memory, compute-accelerator, IO/analog/MEMS, and broadly heterogeneous.",
        "",
        "## Missing Physics Targets",
        "",
        "- At least three physically validated package/interface conductivity-thickness combinations.",
        "- At least three cooling strengths represented through supported sink/convection parameters.",
        "- A small crossed subset separating material/stack effects from cooling effects.",
        "- Include these only after model inputs represent the varying quantities and HotSpot validation confirms identical semantics.",
        "",
        "## Design Rules",
        "",
        "- Use a compact factorial or space-filling design; do not clone any held-out family.",
        "- Preserve the original 50-family split and report extension performance separately.",
        "- Match the existing 200-cell workload template unless a workload-specific gap is independently demonstrated.",
        f"- Current weak/fixed axes motivating this specification: {', '.join(gap_analysis['weak_or_unsupported_dimensions'] + gap_analysis['fixed_dimensions'])}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(
    *,
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    standardizer: TrainStandardizer,
    pca: Mapping[str, Any],
    nearest_rows: Sequence[Mapping[str, Any]],
    split_coverage: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        write_figures_with_pillow(
            out_dir=out_dir,
            records=records,
            standardizer=standardizer,
            pca=pca,
            nearest_rows=nearest_rows,
            split_coverage=split_coverage,
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"train": "#2563eb", "val": "#d97706", "test": "#dc2626"}

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for split in ("train", "val", "test"):
        ax.hist(
            [float(row["chiplet_count"]) for row in records if row["split"] == split],
            bins=12,
            alpha=0.55,
            label=split,
            color=colors[split],
        )
    ax.set(xlabel="Chiplet count", ylabel="Families", title="Chiplet-count coverage")
    ax.legend()
    save(fig, "chiplet_count_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for split in ("train", "val", "test"):
        selected = [row for row in records if row["split"] == split]
        ax.scatter(
            [row["package_width_mm"] for row in selected],
            [row["package_height_mm"] for row in selected],
            label=split,
            color=colors[split],
        )
    ax.set(xlabel="Package width (mm)", ylabel="Package height (mm)", title="Package-size coverage")
    ax.legend()
    save(fig, "package_size_coverage.png")

    histogram_figure(
        plt,
        records,
        "occupied_area_ratio",
        "Occupied-area ratio",
        colors,
        out_dir / "occupied_area_distribution.png",
    )
    histogram_figure(
        plt,
        records,
        "minimum_boundary_clearance_mm",
        "Minimum boundary clearance (mm)",
        colors,
        out_dir / "boundary_clearance_distribution.png",
    )
    histogram_figure(
        plt,
        records,
        "minimum_rectangle_gap_mm",
        "Minimum rectangle gap (mm)",
        colors,
        out_dir / "spacing_distribution.png",
    )

    varying = list(standardizer.names[: min(28, len(standardizer.names))])
    corr = np.corrcoef(
        np.asarray([[float(row[name]) for name in varying] for row in records], dtype=np.float64),
        rowvar=False,
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(varying)), labels=varying, rotation=90, fontsize=6)
    ax.set_yticks(range(len(varying)), labels=varying, fontsize=6)
    ax.set_title("Descriptor correlation matrix (first varying descriptors)")
    fig.colorbar(image, ax=ax, fraction=0.03)
    save(fig, "descriptor_correlation_matrix.png")

    coordinates = np.asarray(pca["coordinates"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in ("train", "val", "test"):
        indices = [index for index, value in enumerate(pca["splits"]) if value == split]
        ax.scatter(coordinates[indices, 0], coordinates[indices, 1], label=split, color=colors[split])
        for index in indices:
            ax.annotate(pca["family_uids"][index], coordinates[index], fontsize=6, alpha=0.8)
    ratio = np.asarray(pca["explained_variance_ratio"]) * 100.0
    ax.set(
        xlabel=f"Train-fit PC1 ({ratio[0]:.1f}%)",
        ylabel=f"Train-fit PC2 ({ratio[1]:.1f}%)",
        title="Family descriptor embedding (visualization only)",
    )
    ax.legend()
    save(fig, "family_descriptor_embedding.png")

    scalar_rows = [row for row in split_coverage if not str(row["descriptor"]).startswith("joint_combination::")]
    outside = np.asarray(
        [[float(row.get(f"{split}_outside_train_count", 0)) for row in scalar_rows] for split in ("val", "test")]
    )
    top = np.argsort(outside.sum(axis=0))[::-1][: min(20, outside.shape[1])]
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(outside[:, top], aspect="auto", cmap="Reds")
    ax.set_yticks([0, 1], labels=["val", "test"])
    ax.set_xticks(range(len(top)), labels=[scalar_rows[index]["descriptor"] for index in top], rotation=90, fontsize=7)
    ax.set_title("Held-out families outside training marginal ranges")
    fig.colorbar(image, ax=ax, fraction=0.03)
    save(fig, "train_val_test_coverage.png")

    nearest = [float(row["rms_standardized_distance"]) for row in nearest_rows if int(row["neighbor_rank"]) == 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nearest, bins=12, color="#475569", alpha=0.85)
    ax.axvline(REDUNDANCY_RMS_DISTANCE_THRESHOLD, color="#dc2626", linestyle="--", label="redundancy threshold")
    ax.set(xlabel="Nearest-neighbor RMS standardized distance", ylabel="Families", title="Nearest-family distance")
    ax.legend()
    save(fig, "nearest_neighbor_distance_distribution.png")


def write_figures_with_pillow(
    *,
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    standardizer: TrainStandardizer,
    pca: Mapping[str, Any],
    nearest_rows: Sequence[Mapping[str, Any]],
    split_coverage: Sequence[Mapping[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    palette = {"train": (37, 99, 235), "val": (217, 119, 6), "test": (220, 38, 38)}
    font = ImageFont.load_default()

    def canvas(title: str, width: int = 920, height: int = 620) -> tuple[Any, Any]:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), title, fill=(20, 25, 35), font=font)
        draw.rectangle((70, 55, width - 30, height - 65), outline=(100, 110, 125), width=1)
        return image, draw

    def histogram(name: str, title: str, output: str, bins: int = 12) -> None:
        image, draw = canvas(title)
        values = np.asarray([float(row[name]) for row in records], dtype=np.float64)
        minimum, maximum = float(values.min()), float(values.max())
        if maximum <= minimum:
            maximum = minimum + 1.0
        edges = np.linspace(minimum, maximum, bins + 1)
        counts = {
            split: np.histogram(
                [float(row[name]) for row in records if row["split"] == split],
                bins=edges,
            )[0]
            for split in ("train", "val", "test")
        }
        max_count = max(int(value.max()) for value in counts.values()) or 1
        left, top, right, bottom = 70, 55, 890, 555
        group_width = (right - left) / bins
        for bin_index in range(bins):
            for split_index, split in enumerate(("train", "val", "test")):
                bar_width = group_width / 3.5
                x0 = left + bin_index * group_width + split_index * bar_width
                x1 = x0 + bar_width
                height = float(counts[split][bin_index]) / max_count * (bottom - top - 20)
                draw.rectangle((x0, bottom - height, x1, bottom), fill=palette[split])
        draw.text((70, 570), f"range {minimum:.3g} to {maximum:.3g}", fill=(30, 35, 45), font=font)
        draw_legend(draw, palette, x=690, y=70, font=font)
        image.save(out_dir / output)

    histogram("chiplet_count", "Chiplet-count coverage", "chiplet_count_distribution.png")
    histogram("occupied_area_ratio", "Occupied-area ratio coverage", "occupied_area_distribution.png")
    histogram(
        "minimum_boundary_clearance_mm",
        "Minimum boundary-clearance coverage",
        "boundary_clearance_distribution.png",
    )
    histogram("minimum_rectangle_gap_mm", "Minimum rectangle-gap coverage", "spacing_distribution.png")

    image, draw = canvas("Package-size coverage")
    draw_split_scatter(
        draw,
        records,
        x_name="package_width_mm",
        y_name="package_height_mm",
        palette=palette,
        bounds=(70, 55, 890, 555),
        font=font,
    )
    image.save(out_dir / "package_size_coverage.png")

    varying = list(standardizer.names[: min(28, len(standardizer.names))])
    correlation = np.corrcoef(
        np.asarray([[float(row[name]) for name in varying] for row in records], dtype=np.float64),
        rowvar=False,
    )
    image, draw = canvas("Descriptor correlation matrix", 980, 760)
    draw_heatmap(draw, correlation, bounds=(190, 70, 900, 700), low=-1.0, high=1.0)
    for index, name in enumerate(varying):
        position = 70 + (index + 0.5) * 630 / len(varying)
        draw.text((8, position - 4), name[:27], fill=(20, 25, 35), font=font)
    image.save(out_dir / "descriptor_correlation_matrix.png")

    coordinates = np.asarray(pca["coordinates"], dtype=np.float64)
    image, draw = canvas("Family descriptor PCA (visualization only)")
    draw_labeled_scatter(
        draw,
        coordinates,
        pca["family_uids"],
        pca["splits"],
        palette,
        bounds=(70, 55, 890, 555),
        font=font,
    )
    image.save(out_dir / "family_descriptor_embedding.png")

    scalar_rows = [row for row in split_coverage if not str(row["descriptor"]).startswith("joint_combination::")]
    outside = np.asarray(
        [[float(row.get(f"{split}_outside_train_count", 0)) for row in scalar_rows] for split in ("val", "test")],
        dtype=np.float64,
    )
    top_indices = np.argsort(outside.sum(axis=0))[::-1][: min(20, outside.shape[1])]
    image, draw = canvas("Held-out marginal range violations", 1050, 520)
    draw_heatmap(draw, outside[:, top_indices], bounds=(160, 70, 1000, 260), low=0.0, high=max(float(outside.max()), 1.0))
    draw.text((90, 110), "validation", fill=(20, 25, 35), font=font)
    draw.text((90, 205), "test", fill=(20, 25, 35), font=font)
    for offset, index in enumerate(top_indices):
        x = 160 + (offset + 0.5) * 840 / len(top_indices)
        draw.text((x - 4, 280), str(scalar_rows[index]["descriptor"])[:18], fill=(20, 25, 35), font=font)
    image.save(out_dir / "train_val_test_coverage.png")

    nearest_values = [
        float(row["rms_standardized_distance"]) for row in nearest_rows if int(row["neighbor_rank"]) == 1
    ]
    image, draw = canvas("Nearest-neighbor RMS standardized distance")
    values = np.asarray(nearest_values, dtype=np.float64)
    edges = np.linspace(float(values.min()), float(values.max()) + EPS, 13)
    counts, _ = np.histogram(values, bins=edges)
    maximum = max(int(counts.max()), 1)
    for index, count in enumerate(counts):
        x0 = 70 + index * 820 / len(counts)
        x1 = 70 + (index + 0.8) * 820 / len(counts)
        height = float(count) / maximum * 470
        draw.rectangle((x0, 555 - height, x1, 555), fill=(71, 85, 105))
    threshold_x = 70 + (
        (REDUNDANCY_RMS_DISTANCE_THRESHOLD - float(values.min()))
        / max(float(values.max() - values.min()), EPS)
        * 820
    )
    draw.line((threshold_x, 55, threshold_x, 555), fill=(220, 38, 38), width=2)
    image.save(out_dir / "nearest_neighbor_distance_distribution.png")


def draw_legend(draw: Any, palette: Mapping[str, tuple[int, int, int]], *, x: int, y: int, font: Any) -> None:
    for index, (name, color) in enumerate(palette.items()):
        offset = y + index * 20
        draw.rectangle((x, offset, x + 12, offset + 12), fill=color)
        draw.text((x + 18, offset), name, fill=(20, 25, 35), font=font)


def draw_split_scatter(
    draw: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    x_name: str,
    y_name: str,
    palette: Mapping[str, tuple[int, int, int]],
    bounds: tuple[int, int, int, int],
    font: Any,
) -> None:
    xs = np.asarray([float(row[x_name]) for row in records])
    ys = np.asarray([float(row[y_name]) for row in records])
    left, top, right, bottom = bounds
    for row, x_value, y_value in zip(records, xs, ys, strict=True):
        x = left + (x_value - xs.min()) / max(float(np.ptp(xs)), EPS) * (right - left)
        y = bottom - (y_value - ys.min()) / max(float(np.ptp(ys)), EPS) * (bottom - top)
        color = palette[str(row["split"])]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x + 5, y - 4), str(row["family_uid"]), fill=(40, 45, 55), font=font)
    draw_legend(draw, palette, x=710, y=70, font=font)


def draw_labeled_scatter(
    draw: Any,
    coordinates: np.ndarray,
    labels: Sequence[str],
    splits: Sequence[str],
    palette: Mapping[str, tuple[int, int, int]],
    *,
    bounds: tuple[int, int, int, int],
    font: Any,
) -> None:
    left, top, right, bottom = bounds
    x_values, y_values = coordinates[:, 0], coordinates[:, 1]
    for label, split, x_value, y_value in zip(labels, splits, x_values, y_values, strict=True):
        x = left + (x_value - x_values.min()) / max(float(np.ptp(x_values)), EPS) * (right - left)
        y = bottom - (y_value - y_values.min()) / max(float(np.ptp(y_values)), EPS) * (bottom - top)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=palette[str(split)])
        draw.text((x + 5, y - 4), str(label), fill=(35, 40, 50), font=font)
    draw_legend(draw, palette, x=710, y=70, font=font)


def draw_heatmap(
    draw: Any,
    matrix: np.ndarray,
    *,
    bounds: tuple[int, int, int, int],
    low: float,
    high: float,
) -> None:
    left, top, right, bottom = bounds
    rows, columns = matrix.shape
    for row in range(rows):
        for column in range(columns):
            value = (float(matrix[row, column]) - low) / max(high - low, EPS)
            value = min(max(value, 0.0), 1.0)
            color = (
                int(40 + 215 * value),
                int(80 + 150 * (1.0 - abs(value - 0.5) * 2.0)),
                int(255 - 215 * value),
            )
            x0 = left + column * (right - left) / columns
            x1 = left + (column + 1) * (right - left) / columns
            y0 = top + row * (bottom - top) / rows
            y1 = top + (row + 1) * (bottom - top) / rows
            draw.rectangle((x0, y0, x1, y1), fill=color)


def histogram_figure(
    plt: Any,
    records: Sequence[Mapping[str, Any]],
    name: str,
    label: str,
    colors: Mapping[str, str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for split in ("train", "val", "test"):
        ax.hist(
            [float(row[name]) for row in records if row["split"] == split],
            bins=12,
            alpha=0.55,
            label=split,
            color=colors[split],
        )
    ax.set(xlabel=label, ylabel="Families", title=f"{label} coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"required mapping {key!r} is unavailable")
    return value


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if abs(mean) > EPS else 0.0


def finite_column(rows: Sequence[Mapping[str, str]], names: Sequence[str]) -> list[float]:
    output = []
    for row in rows:
        for name in names:
            value = row.get(name, "")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                output.append(numeric)
                break
    return output


def variation_class(unique_count: int) -> str:
    if unique_count <= 1:
        return "fixed"
    if unique_count <= 3:
        return "low_cardinality"
    return "varying"


def unique_numeric_count(values: np.ndarray) -> int:
    return int(len(np.unique(np.round(values.astype(np.float64), decimals=10))))


def ensure_finite_numeric(record: Mapping[str, Any], uid: str) -> None:
    for name, value in record.items():
        if isinstance(value, (int, float, bool, np.number)) and not np.isfinite(float(value)):
            raise ValueError(f"{uid}: descriptor {name} is non-finite")


def sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def load_structured(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def portable_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def data_root_relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
