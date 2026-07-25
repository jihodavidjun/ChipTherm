#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import EXPECTED_PRIMARY_SPLIT  # noqa: E402


DEFAULT_SOURCE_VERSION = "source_superposition_final_train40_source_v1"
TYPE_NAMES = ("CPU", "GPU", "NPU", "HBM", "DRAM", "IO", "ANALOG", "MEMS", "OTHER")
EPS = 1.0e-12
SOURCE_AGGREGATES = ("mean", "std", "q10", "q50", "q90")
ERROR_COLUMNS = {
    "source_mae_K": ("source_superposition_mae_K", "physics_baseline_mae_K"),
    "final_mae_K": ("final_cnn_mae_K", "mae_K", "final_mae_K"),
    "mean_correction_mae_K": (
        "absolute_mean_correction_error_K",
        "mean_head_abs_error_K",
        "mean_correction_mae_K",
    ),
    "centered_spatial_mae_K": ("centered_spatial_mae_K", "centered_field_mae_K"),
    "peak_error_K": ("peak_temperature_abs_error_K", "hotspot_temperature_abs_error_K"),
}


@dataclass(frozen=True)
class DescriptorSpec:
    name: str
    group: str
    unit: str
    formula: str
    source: str


@dataclass(frozen=True)
class Standardizer:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    fit_family_uids: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline inference-only package-family OOD descriptor analysis for Benchmark v2."
    )
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument("--source-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--residual-decomposition-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--family-root",
        type=Path,
        default=REPO_ROOT / "configs/benchmark_v2_50family/families",
        help="Root containing immutable f001.yaml ... f050.yaml family definitions.",
    )
    parser.add_argument(
        "--known-family-error-csv",
        type=Path,
        default=None,
        help="Optional known-family per-sample evaluation CSV. It is used only for response comparison, never distances.",
    )
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--mahalanobis-regularization", default=0.10, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    if not 0.0 < args.mahalanobis_regularization <= 1.0:
        raise SystemExit("--mahalanobis-regularization must be in (0, 1]")

    result = analyze_family_ood(
        data_root=args.data_root.expanduser().resolve(),
        source_version=args.source_version,
        residual_decomposition_csv=args.residual_decomposition_csv.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        family_root=args.family_root.expanduser().resolve(),
        known_family_error_csv=(
            args.known_family_error_csv.expanduser().resolve()
            if args.known_family_error_csv is not None
            else None
        ),
        top_k=args.top_k,
        mahalanobis_regularization=args.mahalanobis_regularization,
        seed=args.seed,
    )
    print("Benchmark v2 family OOD analysis complete")
    family_count = sum(len(items) for items in result["family_uids"].values())
    print(f"Families: {family_count}; descriptors: {result['descriptor_count']}")
    print(f"f044 diagnosis: {result['family_diagnoses']['f044']['classification']}")
    print(f"Output: {args.out_dir}")
    return 0


def analyze_family_ood(
    *,
    data_root: Path,
    source_version: str,
    residual_decomposition_csv: Path,
    out_dir: Path,
    family_root: Path,
    known_family_error_csv: Path | None,
    top_k: int,
    mahalanobis_regularization: float,
    seed: int,
) -> dict[str, Any]:
    del seed  # All operations are deterministic and contain no random sampling.
    out_dir.mkdir(parents=True, exist_ok=True)
    index_root = (
        data_root
        / "derived/indices/full_50x200/source_superposition"
        / source_version
        / "family_split"
    )
    split_indices = {
        "train": index_root / "train_index.csv",
        "val": index_root / "val_index.csv",
        "test": index_root / "test_index.csv",
    }
    split_rows = {split: read_csv_required(path) for split, path in split_indices.items()}
    grouped_rows = validate_and_group_rows(split_rows)
    metadata_rows, metadata_names, metadata_units, metadata_sources = load_metadata_sidecars(
        data_root=data_root,
        index_root=index_root.parent,
        split_indices=list(split_indices.values()),
    )
    occupancy_channel, feature_manifest_path = load_occupancy_channel(
        data_root=data_root,
        index_root=index_root.parent,
        split_indices=list(split_indices.values()),
    )

    records: list[dict[str, Any]] = []
    schema: dict[str, DescriptorSpec] = {}
    family_sources: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        for family_uid in sorted(grouped_rows[split]):
            definition_path = family_root / f"{family_uid}.yaml"
            if not definition_path.is_file():
                raise FileNotFoundError(f"missing family definition for {family_uid}: {definition_path}")
            definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
            descriptor, specs, sources = build_family_descriptor(
                family_uid=family_uid,
                split=split,
                rows=grouped_rows[split][family_uid],
                family_definition=definition,
                metadata_rows=metadata_rows,
                metadata_names=metadata_names,
                metadata_units=metadata_units,
                data_root=data_root,
                index_path=split_indices[split],
                occupancy_channel=occupancy_channel,
            )
            for name, spec in specs.items():
                if name in schema and schema[name] != spec:
                    raise ValueError(f"descriptor schema changed across families for {name}")
                schema[name] = spec
            records.append(
                {
                    "family_uid": family_uid,
                    "split": split,
                    "primary_category": str(definition.get("primary_category", "")),
                    "placement_style": str(definition.get("placement_style", "")),
                    **descriptor,
                }
            )
            family_sources[family_uid] = sources

    descriptor_names = tuple(schema)
    ensure_finite_descriptor_records(records, descriptor_names)
    train_uids = tuple(EXPECTED_PRIMARY_SPLIT["train"])
    heldout_uids = tuple(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"])
    standardizer = fit_train_standardizer(records, descriptor_names, train_uids)
    distance_result = compute_ood_distances(
        records,
        descriptor_names,
        train_uids,
        heldout_uids,
        regularization=mahalanobis_regularization,
        top_k=top_k,
        standardizer=standardizer,
    )
    nearest_rows = distance_result["nearest_rows"]
    zscore_rows = build_feature_zscore_rows(records, descriptor_names, heldout_uids, standardizer, schema)

    heldout_error_rows = read_csv_required(residual_decomposition_csv)
    heldout_errors = aggregate_error_labels(heldout_error_rows, required_families=heldout_uids)
    known_error_path = known_family_error_csv or discover_known_family_error_csv(residual_decomposition_csv)
    known_errors: dict[str, dict[str, float]] = {}
    if known_error_path is not None and known_error_path.is_file():
        known_errors = aggregate_error_labels(read_csv_required(known_error_path), required_families=())
    error_by_family = {**known_errors, **heldout_errors}

    correlations = build_ood_error_correlations(distance_result["family_scores"], heldout_errors)
    pca = fit_train_pca(
        records=records,
        descriptor_names=descriptor_names,
        train_uids=train_uids,
        standardizer=standardizer,
    )
    diagnoses = build_family_diagnoses(
        records=records,
        descriptor_names=descriptor_names,
        schema=schema,
        distance_result=distance_result,
        errors=error_by_family,
        heldout_errors=heldout_errors,
        train_uids=train_uids,
        heldout_uids=heldout_uids,
        focus_uids=("f044", "f041"),
    )
    high_low = compare_high_low_heldout(
        records=records,
        descriptor_names=descriptor_names,
        heldout_uids=heldout_uids,
        heldout_errors=heldout_errors,
        standardizer=standardizer,
    )
    direction_assessment = assess_research_directions(
        distance_result=distance_result,
        heldout_errors=heldout_errors,
        diagnoses=diagnoses,
        correlations=correlations,
    )

    write_csv(out_dir / "family_descriptors.csv", records)
    write_csv(out_dir / "heldout_nearest_train_families.csv", nearest_rows)
    write_csv(out_dir / "heldout_feature_zscores.csv", zscore_rows)
    write_csv(out_dir / "ood_error_correlations.csv", correlations)
    write_plots(
        out_dir=out_dir,
        records=records,
        descriptor_names=descriptor_names,
        schema=schema,
        pca=pca,
        distance_result=distance_result,
        errors=heldout_errors,
        diagnoses=diagnoses,
    )

    summary = {
        "schema_version": "benchmark_v2_family_ood_analysis/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "source_version": source_version,
        "indices": {split: portable_path(path, data_root) for split, path in split_indices.items()},
        "residual_decomposition_csv": portable_or_absolute(residual_decomposition_csv, data_root),
        "known_family_error_csv": (
            portable_or_absolute(known_error_path, data_root) if known_error_path and known_error_path.is_file() else None
        ),
        "family_uids": {
            "train": list(train_uids),
            "heldout_validation": list(EXPECTED_PRIMARY_SPLIT["val"]),
            "heldout_test": list(EXPECTED_PRIMARY_SPLIT["test"]),
        },
        "descriptor_count": len(descriptor_names),
        "descriptor_names": list(descriptor_names),
        "descriptor_schema": {
            name: {
                "group": spec.group,
                "unit": spec.unit,
                "formula": spec.formula,
                "source": spec.source,
            }
            for name, spec in schema.items()
        },
        "unavailable_descriptor_notes": [
            "Benchmark v2 family YAMLs do not define a separate substrate or interposer thermal layer; the "
            "silicon-interposer identity and physical package/interposer dimensions are recorded without inventing "
            "conductivity or thickness values.",
            "Only fields present for every family in fixed_structure.layout, thermal_stack, hotspot, compact model "
            "metadata, and source-superposition artifacts are active descriptors.",
        ],
        "descriptor_sources_by_family": family_sources,
        "metadata_feature_names": metadata_names,
        "metadata_feature_units": metadata_units,
        "metadata_sources": [portable_or_absolute(path, data_root) for path in metadata_sources],
        "feature_manifest": portable_or_absolute(feature_manifest_path, data_root),
        "standardization": standardizer_to_json(standardizer),
        "mahalanobis": {
            "regularization": mahalanobis_regularization,
            "definition": "(1-lambda)*cov(train z) + lambda*I; pair distances use its pseudoinverse",
        },
        "distance_thresholds": distance_result["thresholds"],
        "family_ood_scores": distance_result["family_scores"],
        "family_errors": error_by_family,
        "family_diagnoses": diagnoses,
        "high_vs_low_heldout": high_low,
        "ood_error_correlations": correlations,
        "pca": {
            "fit_family_uids": list(train_uids),
            "explained_variance_ratio": pca["explained_variance_ratio"],
        },
        "direction_assessment": direction_assessment,
        "leakage_safeguards": [
            "No y_path or HotSpot target tensor is opened while constructing descriptors.",
            "Residual-decomposition error labels are joined only after descriptor scaling, PCA, and OOD distances are fixed.",
            "Standardization, covariance regularization, nearest-neighbor thresholds, and PCA use only the 40 training families.",
            "Source-superposition maps are inference-time model inputs and are aggregated over all 200 workloads per family.",
            "Held-out error labels never enter descriptor imputation, feature selection, scaling, distance, or PCA.",
        ],
        "statistical_caveat": (
            "OOD/error correlations use only the ten held-out families and are exploratory, not causal evidence."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "family_ood_report.md", summary, nearest_rows)
    return summary


def validate_and_group_rows(
    split_rows: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for split in ("train", "val", "test"):
        rows = split_rows[split]
        by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
        seen: set[str] = set()
        for source_row in rows:
            row = dict(source_row)
            uid = require_text(row, "sample_uid")
            family = family_uid_for_row(row)
            if uid in seen:
                raise ValueError(f"duplicate sample_uid in {split} index: {uid}")
            seen.add(uid)
            by_family[family].append(row)
        expected = set(EXPECTED_PRIMARY_SPLIT[split])
        if set(by_family) != expected:
            raise ValueError(
                f"{split} family set mismatch: expected={sorted(expected)} actual={sorted(by_family)}"
            )
        counts = {family: len(items) for family, items in by_family.items()}
        bad_counts = {family: count for family, count in counts.items() if count != 200}
        if bad_counts:
            raise ValueError(f"{split} does not contain exactly 200 workloads per family: {bad_counts}")
        grouped[split] = dict(by_family)
    return grouped


def build_family_descriptor(
    *,
    family_uid: str,
    split: str,
    rows: Sequence[Mapping[str, str]],
    family_definition: Mapping[str, Any],
    metadata_rows: Mapping[str, Mapping[str, float]],
    metadata_names: Sequence[str],
    metadata_units: Mapping[str, str],
    data_root: Path,
    index_path: Path,
    occupancy_channel: int,
) -> tuple[dict[str, float], dict[str, DescriptorSpec], dict[str, str]]:
    if str(family_definition.get("family_uid")) != family_uid:
        raise ValueError(f"family definition UID mismatch for {family_uid}")
    if str(family_definition.get("primary_split")) != split:
        raise ValueError(f"family definition split mismatch for {family_uid}")
    descriptor: dict[str, float] = {}
    specs: dict[str, DescriptorSpec] = {}

    def add(name: str, value: float, *, group: str, unit: str, formula: str, source: str) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{family_uid} descriptor {name} is non-finite")
        descriptor[name] = numeric
        specs[name] = DescriptorSpec(name, group, unit, formula, source)

    structure = family_definition["fixed_structure"]
    layout = structure["layout"]
    package = layout["package"]["size"]
    width = float(package["width"])
    height = float(package["height"])
    chips = list(layout["chiplets"])
    if not chips:
        raise ValueError(f"{family_uid} has no chiplets")
    chip_array = geometry_array(chips)
    x, y, chip_width, chip_height = chip_array.T
    area = chip_width * chip_height
    center_x = x + 0.5 * chip_width
    center_y = y + 0.5 * chip_height
    package_area = width * height
    characteristic = math.sqrt(package_area)
    boundary = np.minimum.reduce((x, y, width - x - chip_width, height - y - chip_height))
    pairwise = pairwise_distances(center_x, center_y)
    edge_threshold = 0.05 * characteristic
    near_x_edge = np.minimum(x, width - x - chip_width) <= edge_threshold
    near_y_edge = np.minimum(y, height - y - chip_height) <= edge_threshold
    types = [canonical_type(str(chip.get("type", "OTHER"))) for chip in chips]

    geometry_source = "family YAML fixed_structure.layout"
    add("package_width_mm", width, group="global", unit="mm", formula="package width", source=geometry_source)
    add("package_height_mm", height, group="global", unit="mm", formula="package height", source=geometry_source)
    add("package_area_mm2", package_area, group="global", unit="mm^2", formula="width*height", source=geometry_source)
    add(
        "package_aspect_ratio",
        max(width, height) / min(width, height),
        group="global",
        unit="ratio",
        formula="max(width,height)/min(width,height)",
        source=geometry_source,
    )
    add("chiplet_count", len(chips), group="global", unit="count", formula="number of layout chiplets", source=geometry_source)
    add("total_chiplet_area_mm2", area.sum(), group="global", unit="mm^2", formula="sum chiplet rectangle area", source=geometry_source)
    add("occupied_area_fraction", area.sum() / package_area, group="global", unit="fraction", formula="sum chiplet area/package area", source=geometry_source)
    add_stats(add, "chiplet_area_mm2", area, group="global", unit="mm^2", source=geometry_source)
    add_stats(add, "chiplet_width_mm", chip_width, group="spatial", unit="mm", source=geometry_source)
    add_stats(add, "chiplet_height_mm", chip_height, group="spatial", unit="mm", source=geometry_source)
    add_stats(add, "chiplet_aspect_ratio", chip_width / chip_height, group="spatial", unit="ratio", source=geometry_source)
    add("placement_centroid_x_fraction", center_x.mean() / width, group="spatial", unit="fraction", formula="mean chiplet center x/package width", source=geometry_source)
    add("placement_centroid_y_fraction", center_y.mean() / height, group="spatial", unit="fraction", formula="mean chiplet center y/package height", source=geometry_source)
    add("placement_spread_x_fraction", center_x.std() / width, group="spatial", unit="fraction", formula="std chiplet center x/package width", source=geometry_source)
    add("placement_spread_y_fraction", center_y.std() / height, group="spatial", unit="fraction", formula="std chiplet center y/package height", source=geometry_source)
    add("pairwise_center_distance_mm_min", pairwise.min() if pairwise.size else 0.0, group="spatial", unit="mm", formula="minimum distinct chiplet-center distance", source=geometry_source)
    add("pairwise_center_distance_mm_mean", pairwise.mean() if pairwise.size else 0.0, group="spatial", unit="mm", formula="mean distinct chiplet-center distance", source=geometry_source)
    add("pairwise_center_distance_normalized_mean", pairwise.mean() / characteristic if pairwise.size else 0.0, group="spatial", unit="ratio", formula="mean center distance/sqrt(package area)", source=geometry_source)
    add("boundary_clearance_mm_min", boundary.min(), group="spatial", unit="mm", formula="minimum rectangle clearance to any package edge", source=geometry_source)
    add("boundary_clearance_mm_mean", boundary.mean(), group="spatial", unit="mm", formula="mean rectangle clearance to nearest package edge", source=geometry_source)
    add("edge_occupancy_fraction", np.mean(near_x_edge | near_y_edge), group="spatial", unit="fraction", formula="fraction chiplets within 5% sqrt(package area) of any edge", source=geometry_source)
    add("corner_occupancy_fraction", np.mean(near_x_edge & near_y_edge), group="spatial", unit="fraction", formula="fraction chiplets simultaneously near orthogonal edges", source=geometry_source)
    for type_name in TYPE_NAMES:
        count = types.count(type_name)
        add(f"type_{type_name.lower()}_count", count, group="global", unit="count", formula=f"number of {type_name} chiplets", source=geometry_source)
        add(f"type_{type_name.lower()}_fraction", count / len(chips), group="global", unit="fraction", formula=f"{type_name} count/chiplet count", source=geometry_source)

    add_thermal_descriptors(add, structure, layout)

    sorted_rows = sorted(rows, key=lambda item: require_text(item, "sample_uid"))
    missing_metadata = [
        require_text(row, "sample_uid")
        for row in sorted_rows
        if require_text(row, "sample_uid") not in metadata_rows
    ]
    if missing_metadata:
        raise ValueError(f"{family_uid} metadata sidecar is missing rows: {missing_metadata[:5]}")
    for name in metadata_names:
        values = np.asarray(
            [float(metadata_rows[require_text(row, "sample_uid")][name]) for row in sorted_rows],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{family_uid} metadata feature {name} contains non-finite values")
        add(
            f"metadata_{name}_family_mean",
            values.mean(),
            group=metadata_group(name),
            unit=metadata_units[name],
            formula=f"mean across 200 workloads of compact metadata feature {name}",
            source="metadata_features.csv and metadata_manifest.json",
        )
        if metadata_varies_by_workload(name):
            for suffix, value in (
                ("std", values.std()),
                ("q10", np.quantile(values, 0.10)),
                ("q90", np.quantile(values, 0.90)),
            ):
                add(
                    f"metadata_{name}_family_{suffix}",
                    value,
                    group=metadata_group(name),
                    unit=metadata_units[name],
                    formula=f"{suffix} across 200 workloads of metadata feature {name}",
                    source="metadata_features.csv and metadata_manifest.json",
                )

    first_x_path = resolve_data_path(require_text(sorted_rows[0], "x_path"), data_root, index_path)
    x_tensor = np.load(first_x_path, mmap_mode="r")
    if x_tensor.ndim != 3 or occupancy_channel >= x_tensor.shape[0]:
        raise ValueError(
            f"{family_uid} X tensor cannot provide occupancy channel {occupancy_channel}: {first_x_path} {x_tensor.shape}"
        )
    occupancy = np.asarray(x_tensor[occupancy_channel] > 0.5, dtype=bool)
    if occupancy.shape != (64, 64):
        raise ValueError(f"{family_uid} occupancy shape must be 64x64, got {occupancy.shape}")
    source_stats: dict[str, list[float]] = defaultdict(list)
    for row in sorted_rows:
        base_logical = require_text(row, "source_superposition_base_path")
        base_path = resolve_data_path(base_logical, data_root, index_path)
        base = np.asarray(np.load(base_path), dtype=np.float64)
        if base.shape != (64, 64):
            raise ValueError(f"{require_text(row, 'sample_uid')} source base shape is {base.shape}, expected (64, 64)")
        if not np.isfinite(base).all():
            raise ValueError(f"{require_text(row, 'sample_uid')} source base contains NaN/Inf")
        values = source_map_statistics(base, occupancy)
        for name, value in values.items():
            source_stats[name].append(float(value))
    for stat_name in sorted(source_stats):
        values = np.asarray(source_stats[stat_name], dtype=np.float64)
        stat_group, unit, formula = source_stat_spec(stat_name)
        for aggregate in SOURCE_AGGREGATES:
            add(
                f"source_base_{stat_name}_workload_{aggregate}",
                aggregate_value(values, aggregate),
                group=stat_group,
                unit=unit,
                formula=f"{aggregate} across 200 workloads of ({formula})",
                source="source_superposition_base_path; no HotSpot target read",
            )

    return descriptor, specs, {
        "family_definition": portable_or_absolute(Path(family_definition.get("_source_path", "")), data_root)
        if family_definition.get("_source_path")
        else f"configs/benchmark_v2_50family/families/{family_uid}.yaml",
        "first_x_path": portable_path(first_x_path, data_root),
        "source_base_path_column": "source_superposition_base_path",
    }


def add_thermal_descriptors(add: Any, structure: Mapping[str, Any], layout: Mapping[str, Any]) -> None:
    stack = structure["thermal_stack"]
    source = "family YAML fixed_structure.thermal_stack"
    add("ambient_K", stack["ambient_K"], group="global", unit="K", formula="thermal-stack ambient temperature", source=source)
    add(
        "substrate_silicon_interposer_flag",
        1.0 if str(layout["package"].get("substrate", "")).lower() == "silicon_interposer" else 0.0,
        group="global",
        unit="bool",
        formula="1 iff layout package substrate is silicon_interposer",
        source="family YAML fixed_structure.layout.package.substrate",
    )
    layer_keys = {
        "chip": ("thickness_m", "thermal_conductivity_W_per_mK", "volumetric_heat_capacity_J_per_m3K"),
        "interface": ("thickness_m", "thermal_conductivity_W_per_mK", "volumetric_heat_capacity_J_per_m3K"),
        "spreader": ("side_m", "thickness_m", "thermal_conductivity_W_per_mK", "volumetric_heat_capacity_J_per_m3K"),
        "sink": (
            "side_m",
            "thickness_m",
            "thermal_conductivity_W_per_mK",
            "volumetric_heat_capacity_J_per_m3K",
            "convection_resistance_K_per_W",
            "convection_capacitance_J_per_K",
        ),
    }
    unit_by_suffix = {
        "side_m": "m",
        "thickness_m": "m",
        "thermal_conductivity_W_per_mK": "W/(m K)",
        "volumetric_heat_capacity_J_per_m3K": "J/(m^3 K)",
        "convection_resistance_K_per_W": "K/W",
        "convection_capacitance_J_per_K": "J/K",
    }
    for layer, keys in layer_keys.items():
        values = stack.get(layer)
        if not isinstance(values, Mapping):
            raise ValueError(f"required thermal-stack layer {layer} is unavailable")
        for key in keys:
            if key not in values:
                raise ValueError(f"required thermal-stack descriptor {layer}.{key} is unavailable")
            name = key.replace("thermal_", "").replace("volumetric_", "")
            add(
                f"{layer}_{name}",
                values[key],
                group="global",
                unit=unit_by_suffix[key],
                formula=f"{layer}.{key}",
                source=source,
            )
    hotspot = structure["hotspot"]
    add("grid_rows", hotspot["grid"]["rows"], group="global", unit="cells", formula="HotSpot grid rows", source="family YAML fixed_structure.hotspot")
    add("grid_cols", hotspot["grid"]["cols"], group="global", unit="cells", formula="HotSpot grid columns", source="family YAML fixed_structure.hotspot")
    add("sampling_interval_s", hotspot["sampling_interval_s"], group="global", unit="s", formula="HotSpot sampling interval", source="family YAML fixed_structure.hotspot")
    add("base_processor_frequency_Hz", hotspot["base_processor_frequency_Hz"], group="global", unit="Hz", formula="HotSpot base processor frequency", source="family YAML fixed_structure.hotspot")
    for key in ("leakage_used", "detailed_package", "secondary_path"):
        add(
            f"hotspot_{key}",
            float(bool(hotspot.get(key, False))),
            group="global",
            unit="bool",
            formula=f"HotSpot {key} flag",
            source="family YAML fixed_structure.hotspot",
        )


def source_map_statistics(base: np.ndarray, occupancy: np.ndarray) -> dict[str, float]:
    centered = base - float(base.mean())
    gradient_y, gradient_x = np.gradient(base)
    gradient_sq = gradient_x * gradient_x + gradient_y * gradient_y
    spectrum = np.fft.rfft2(centered, norm="ortho")
    energy = np.abs(spectrum) ** 2
    fy = np.fft.fftfreq(base.shape[0])[:, None]
    fx = np.fft.rfftfreq(base.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    total_energy = float(energy.sum())
    low_fraction = float(energy[radius <= 0.125].sum() / max(total_energy, EPS))
    high_fraction = float(energy[radius >= 0.25].sum() / max(total_energy, EPS))
    boundary = np.zeros(base.shape, dtype=bool)
    border = max(1, int(round(0.0625 * min(base.shape))))
    boundary[:border] = True
    boundary[-border:] = True
    boundary[:, :border] = True
    boundary[:, -border:] = True
    interior = ~boundary
    occupied_mean = float(base[occupancy].mean()) if occupancy.any() else float(base.mean())
    unoccupied_mean = float(base[~occupancy].mean()) if (~occupancy).any() else float(base.mean())
    return {
        "mean_K": float(base.mean()),
        "std_K": float(base.std()),
        "min_K": float(base.min()),
        "max_K": float(base.max()),
        "peak_above_mean_K": float(base.max() - base.mean()),
        "centered_rms_K": float(np.sqrt(np.mean(centered * centered))),
        "centered_abs_mean_K": float(np.mean(np.abs(centered))),
        "low_frequency_energy_fraction": low_fraction,
        "high_frequency_energy_fraction": high_fraction,
        "gradient_abs_mean_K_per_cell": float(np.mean(np.sqrt(gradient_sq))),
        "gradient_energy_K2_per_cell2": float(np.mean(gradient_sq)),
        "boundary_minus_interior_mean_K": float(base[boundary].mean() - base[interior].mean()),
        "occupied_minus_unoccupied_mean_K": float(occupied_mean - unoccupied_mean),
    }


def source_stat_spec(name: str) -> tuple[str, str, str]:
    if name in {"mean_K", "min_K", "max_K"}:
        return "global", "K", f"source-superposition map {name.removesuffix('_K')}"
    if name in {"std_K", "peak_above_mean_K", "centered_rms_K", "centered_abs_mean_K"}:
        return "spatial", "K", f"source-superposition map {name.removesuffix('_K')}"
    if name.endswith("_fraction"):
        return "spatial", "fraction", name.replace("_", " ")
    if name.endswith("_K_per_cell"):
        return "spatial", "K/cell", name.replace("_", " ")
    if name.endswith("_K2_per_cell2"):
        return "spatial", "K^2/cell^2", name.replace("_", " ")
    return "spatial", "K", name.replace("_", " ")


def fit_train_standardizer(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    train_family_uids: Sequence[str],
) -> Standardizer:
    by_uid = {str(row["family_uid"]): row for row in records}
    train_matrix = np.asarray(
        [[float(by_uid[uid][name]) for name in feature_names] for uid in train_family_uids],
        dtype=np.float64,
    )
    if not np.isfinite(train_matrix).all():
        raise ValueError("training descriptor matrix contains non-finite values")
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    positive = std[std > 1.0e-12]
    floor = max(1.0e-12, (float(np.median(positive)) * 1.0e-6) if positive.size else 1.0)
    scale = np.where(std > floor, std, floor)
    return Standardizer(
        feature_names=tuple(feature_names),
        mean=mean,
        std=std,
        scale=scale,
        minimum=train_matrix.min(axis=0),
        maximum=train_matrix.max(axis=0),
        fit_family_uids=tuple(train_family_uids),
    )


def compute_ood_distances(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    train_family_uids: Sequence[str],
    heldout_family_uids: Sequence[str],
    *,
    regularization: float,
    top_k: int,
    standardizer: Standardizer | None = None,
) -> dict[str, Any]:
    by_uid = {str(row["family_uid"]): row for row in records}
    scaler = standardizer or fit_train_standardizer(records, feature_names, train_family_uids)
    matrix = np.asarray(
        [[float(by_uid[uid][name]) for name in feature_names] for uid in [*train_family_uids, *heldout_family_uids]],
        dtype=np.float64,
    )
    z = (matrix - scaler.mean) / scaler.scale
    if not np.isfinite(z).all():
        raise ValueError("standardized descriptor matrix contains non-finite values")
    train_z = z[: len(train_family_uids)]
    heldout_z = z[len(train_family_uids) :]
    covariance = np.cov(train_z, rowvar=False, ddof=1)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    covariance_regularized = (1.0 - regularization) * covariance + regularization * np.eye(len(feature_names))
    inverse_covariance = np.linalg.pinv(covariance_regularized, hermitian=True)
    nearest_rows: list[dict[str, Any]] = []
    family_scores: dict[str, dict[str, float]] = {}

    train_nearest_euclidean: list[float] = []
    train_global_rms: list[float] = []
    for index, vector in enumerate(train_z):
        others = np.delete(train_z, index, axis=0)
        train_nearest_euclidean.append(float(np.linalg.norm(others - vector, axis=1).min()))
        train_global_rms.append(float(np.sqrt(np.mean(vector * vector))))
    thresholds = {
        "train_leave_one_out_nearest_euclidean_q95": float(np.quantile(train_nearest_euclidean, 0.95)),
        "train_standardized_rms_q95": float(np.quantile(train_global_rms, 0.95)),
    }
    train_centroid = train_z.mean(axis=0)
    for heldout_uid, vector in zip(heldout_family_uids, heldout_z):
        delta = train_z - vector
        euclidean = np.linalg.norm(delta, axis=1)
        mahalanobis = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inverse_covariance, delta), 0.0))
        order = np.argsort(euclidean, kind="stable")
        for rank, train_index in enumerate(order[: min(top_k, len(order))], start=1):
            nearest_rows.append(
                {
                    "heldout_family_uid": heldout_uid,
                    "heldout_split": str(by_uid[heldout_uid]["split"]),
                    "rank": rank,
                    "train_family_uid": train_family_uids[int(train_index)],
                    "euclidean_distance": float(euclidean[train_index]),
                    "mahalanobis_distance": float(mahalanobis[train_index]),
                }
            )
        center_delta = vector - train_centroid
        family_scores[heldout_uid] = {
            "nearest_euclidean_distance": float(euclidean.min()),
            "nearest_mahalanobis_distance": float(mahalanobis.min()),
            "centroid_euclidean_distance": float(np.linalg.norm(center_delta)),
            "centroid_mahalanobis_distance": float(
                np.sqrt(max(float(center_delta @ inverse_covariance @ center_delta), 0.0))
            ),
            "standardized_rms": float(np.sqrt(np.mean(vector * vector))),
            "range_violation_count": int(
                np.sum((matrix[len(train_family_uids) + list(heldout_family_uids).index(heldout_uid)] < scaler.minimum)
                       | (matrix[len(train_family_uids) + list(heldout_family_uids).index(heldout_uid)] > scaler.maximum))
            ),
            "max_abs_zscore": float(np.max(np.abs(vector))),
        }
    return {
        "standardizer": scaler,
        "inverse_covariance": inverse_covariance,
        "standardized_by_uid": {
            uid: vector
            for uid, vector in zip([*train_family_uids, *heldout_family_uids], z)
        },
        "nearest_rows": nearest_rows,
        "family_scores": family_scores,
        "thresholds": thresholds,
    }


def build_feature_zscore_rows(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    heldout_uids: Sequence[str],
    standardizer: Standardizer,
    schema: Mapping[str, DescriptorSpec],
) -> list[dict[str, Any]]:
    by_uid = {str(row["family_uid"]): row for row in records}
    rows: list[dict[str, Any]] = []
    for uid in heldout_uids:
        values = np.asarray([float(by_uid[uid][name]) for name in feature_names], dtype=np.float64)
        z = (values - standardizer.mean) / standardizer.scale
        for index, name in enumerate(feature_names):
            below = values[index] < standardizer.minimum[index]
            above = values[index] > standardizer.maximum[index]
            train_values = np.asarray(
                [float(by_uid[train_uid][name]) for train_uid in standardizer.fit_family_uids],
                dtype=np.float64,
            )
            percentile = 100.0 * (
                float(np.sum(train_values < values[index])) + 0.5 * float(np.sum(train_values == values[index]))
            ) / len(train_values)
            rows.append(
                {
                    "family_uid": uid,
                    "split": str(by_uid[uid]["split"]),
                    "feature_name": name,
                    "feature_group": schema[name].group,
                    "unit": schema[name].unit,
                    "value": float(values[index]),
                    "train_mean": float(standardizer.mean[index]),
                    "train_std": float(standardizer.std[index]),
                    "zscore": float(z[index]),
                    "abs_zscore": float(abs(z[index])),
                    "train_min": float(standardizer.minimum[index]),
                    "train_max": float(standardizer.maximum[index]),
                    "train_percentile": percentile,
                    "range_violation": bool(below or above),
                    "range_violation_direction": "below" if below else ("above" if above else ""),
                }
            )
    return rows


def aggregate_error_labels(
    rows: Sequence[Mapping[str, str]],
    *,
    required_families: Sequence[str],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        family = family_uid_for_row(row)
        grouped[family].append(row)
    missing = sorted(set(required_families) - set(grouped))
    if missing:
        raise ValueError(f"error CSV is missing required held-out families: {missing}")
    output: dict[str, dict[str, float]] = {}
    for family, family_rows in sorted(grouped.items()):
        metrics: dict[str, float] = {"num_error_samples": float(len(family_rows))}
        for canonical, aliases in ERROR_COLUMNS.items():
            column = first_available_column(family_rows, aliases)
            if column is None:
                if family in required_families:
                    raise ValueError(
                        f"error CSV lacks {canonical}; expected one of {aliases}; available={sorted(family_rows[0])}"
                    )
                continue
            values = np.asarray([float(row[column]) for row in family_rows], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{family} error label {column} contains non-finite values")
            metrics[canonical] = float(values.mean())
        output[family] = metrics
    return output


def build_ood_error_correlations(
    family_scores: Mapping[str, Mapping[str, float]],
    heldout_errors: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    uids = sorted(set(family_scores) & set(heldout_errors))
    rows: list[dict[str, Any]] = []
    if len(uids) < 3:
        raise ValueError("at least three held-out families are required for exploratory correlations")
    score_names = sorted(next(iter(family_scores.values())))
    error_names = [name for name in ERROR_COLUMNS if all(name in heldout_errors[uid] for uid in uids)]
    for score_name in score_names:
        x = np.asarray([float(family_scores[uid][score_name]) for uid in uids], dtype=np.float64)
        for error_name in error_names:
            y = np.asarray([float(heldout_errors[uid][error_name]) for uid in uids], dtype=np.float64)
            rows.append(
                {
                    "ood_score": score_name,
                    "error_metric": error_name,
                    "heldout_family_count": len(uids),
                    "pearson_correlation": finite_correlation(x, y),
                    "spearman_correlation": finite_correlation(rankdata(x), rankdata(y)),
                    "interpretation": "exploratory_only_n_equals_10",
                }
            )
    return rows


def fit_train_pca(
    *,
    records: Sequence[Mapping[str, Any]],
    descriptor_names: Sequence[str],
    train_uids: Sequence[str],
    standardizer: Standardizer,
) -> dict[str, Any]:
    by_uid = {str(row["family_uid"]): row for row in records}
    ordered_uids = [str(row["family_uid"]) for row in records]
    matrix = np.asarray(
        [[float(by_uid[uid][name]) for name in descriptor_names] for uid in ordered_uids],
        dtype=np.float64,
    )
    z = (matrix - standardizer.mean) / standardizer.scale
    train_indices = [ordered_uids.index(uid) for uid in train_uids]
    train = z[train_indices]
    center = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - center, full_matrices=False)
    components = vt[:2].copy()
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1.0
    projected = (z - center) @ components.T
    denominator = max(float(np.sum(singular * singular)), EPS)
    explained = ((singular[:2] * singular[:2]) / denominator).tolist()
    return {
        "family_uids": ordered_uids,
        "coordinates": projected,
        "components": components,
        "explained_variance_ratio": [float(value) for value in explained],
    }


def build_family_diagnoses(
    *,
    records: Sequence[Mapping[str, Any]],
    descriptor_names: Sequence[str],
    schema: Mapping[str, DescriptorSpec],
    distance_result: Mapping[str, Any],
    errors: Mapping[str, Mapping[str, float]],
    heldout_errors: Mapping[str, Mapping[str, float]],
    train_uids: Sequence[str],
    heldout_uids: Sequence[str],
    focus_uids: Sequence[str],
) -> dict[str, Any]:
    del records, descriptor_names, heldout_uids
    standardized = distance_result["standardized_by_uid"]
    train_group_rms: dict[str, list[float]] = {"global": [], "spatial": []}
    group_indices = {
        group: np.asarray([index for index, name in enumerate(schema) if schema[name].group == group], dtype=np.int64)
        for group in ("global", "spatial")
    }
    for uid in train_uids:
        vector = standardized[uid]
        for group, indices in group_indices.items():
            train_group_rms[group].append(float(np.sqrt(np.mean(vector[indices] ** 2))))
    group_thresholds = {
        group: float(np.quantile(values, 0.95))
        for group, values in train_group_rms.items()
    }
    nearest_by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in distance_result["nearest_rows"]:
        nearest_by_uid[str(row["heldout_family_uid"])].append(row)
    result: dict[str, Any] = {}
    for uid in focus_uids:
        vector = standardized[uid]
        group_scores = {
            group: float(np.sqrt(np.mean(vector[indices] ** 2)))
            for group, indices in group_indices.items()
        }
        global_ood = group_scores["global"] > group_thresholds["global"]
        spatial_ood = group_scores["spatial"] > group_thresholds["spatial"]
        no_close = (
            distance_result["family_scores"][uid]["nearest_euclidean_distance"]
            > distance_result["thresholds"]["train_leave_one_out_nearest_euclidean_q95"]
        )
        top_features = sorted(
            (
                {"feature_name": name, "zscore": float(vector[index]), "group": schema[name].group}
                for index, name in enumerate(schema)
            ),
            key=lambda item: (-abs(item["zscore"]), item["feature_name"]),
        )[:15]
        nearest = nearest_by_uid[uid]
        neighbor_errors = [
            errors[str(row["train_family_uid"])]["final_mae_K"]
            for row in nearest
            if str(row["train_family_uid"]) in errors and "final_mae_K" in errors[str(row["train_family_uid"])]
        ]
        response_mismatch: bool | None = None
        if neighbor_errors and uid in heldout_errors:
            response_mismatch = heldout_errors[uid]["final_mae_K"] > (
                float(np.mean(neighbor_errors)) + max(0.5, 0.5 * float(np.mean(neighbor_errors)))
            )
        if global_ood and spatial_ood:
            classification = "globally_and_spatially_ood"
        elif global_ood:
            classification = "mainly_globally_ood"
        elif spatial_ood:
            classification = "mainly_spatially_ood"
        elif no_close:
            classification = "no_close_training_neighbor_without_single_group_dominance"
        elif response_mismatch:
            classification = "close_descriptor_neighbor_but_different_thermal_response"
        else:
            classification = "descriptor_in_distribution"
        result[uid] = {
            "classification": classification,
            "no_close_training_neighbor": no_close,
            "global_ood": global_ood,
            "spatial_ood": spatial_ood,
            "global_group_rms_z": group_scores["global"],
            "spatial_group_rms_z": group_scores["spatial"],
            "global_train_q95": group_thresholds["global"],
            "spatial_train_q95": group_thresholds["spatial"],
            "close_neighbor_different_response": response_mismatch,
            "nearest_training_families": nearest,
            "top_distinguishing_features": top_features,
        }
    return result


def compare_high_low_heldout(
    *,
    records: Sequence[Mapping[str, Any]],
    descriptor_names: Sequence[str],
    heldout_uids: Sequence[str],
    heldout_errors: Mapping[str, Mapping[str, float]],
    standardizer: Standardizer,
) -> dict[str, Any]:
    ordered = sorted(heldout_uids, key=lambda uid: (heldout_errors[uid]["final_mae_K"], uid))
    low = ordered[: len(ordered) // 2]
    high = ordered[len(ordered) // 2 :]
    by_uid = {str(row["family_uid"]): row for row in records}

    def matrix(uids: Sequence[str]) -> np.ndarray:
        raw = np.asarray([[float(by_uid[uid][name]) for name in descriptor_names] for uid in uids], dtype=np.float64)
        return (raw - standardizer.mean) / standardizer.scale

    difference = matrix(high).mean(axis=0) - matrix(low).mean(axis=0)
    features = sorted(
        (
            {
                "feature_name": name,
                "high_minus_low_mean_z": float(difference[index]),
                "absolute_difference": float(abs(difference[index])),
            }
            for index, name in enumerate(descriptor_names)
        ),
        key=lambda item: (-item["absolute_difference"], item["feature_name"]),
    )
    return {
        "high_error_family_uids": high,
        "low_error_family_uids": low,
        "top_distinguishing_features": features[:20],
    }


def assess_research_directions(
    *,
    distance_result: Mapping[str, Any],
    heldout_errors: Mapping[str, Mapping[str, float]],
    diagnoses: Mapping[str, Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_corr = [
        row for row in correlations
        if row["ood_score"] == "nearest_euclidean_distance" and row["error_metric"] == "final_mae_K"
    ][0]
    ood_error_association = float(final_corr["spearman_correlation"])
    f044 = diagnoses["f044"]
    f044_errors = heldout_errors["f044"]
    centered_dominates = f044_errors["centered_spatial_mae_K"] > f044_errors["mean_correction_mae_K"]
    source_to_final_gain = f044_errors["source_mae_K"] - f044_errors["final_mae_K"]
    targeted = bool(f044["no_close_training_neighbor"] or ood_error_association >= 0.4)
    mean_head = bool(not centered_dominates and f044_errors["mean_correction_mae_K"] >= 1.0)
    gating = bool(ood_error_association >= 0.4 and source_to_final_gain <= 0.0)
    retrieval = bool(
        not f044["no_close_training_neighbor"]
        and f044.get("close_neighbor_different_response") is not False
    )
    return {
        "targeted_dataset_expansion": {
            "supported": targeted,
            "reason": (
                "f044 lacks a close training neighbor or held-out error rises with descriptor OOD distance."
                if targeted else
                "f044 is not clearly isolated in descriptor space and distance/error association is weak."
            ),
        },
        "stronger_global_calibrated_mean_head": {
            "supported": mean_head,
            "reason": (
                "Mean-correction error is the dominant f044 component."
                if mean_head else
                "f044 centered-spatial error exceeds mean-correction error."
            ),
        },
        "ood_aware_gating": {
            "supported": gating,
            "reason": (
                "OOD score tracks error and the CNN does not improve f044 over its source baseline."
                if gating else
                "The evidence does not show both strong OOD/error association and harmful residual correction."
            ),
        },
        "retrieval_conditioned_residual_modeling": {
            "supported": retrieval,
            "reason": (
                "f044 has a descriptor-near training neighborhood that can serve as a retrieval diagnostic."
                if retrieval else
                "f044 has no sufficiently close descriptor neighbor, weakening retrieval as the first response."
            ),
        },
        "none_of_the_above": {
            "supported": not any((targeted, mean_head, gating, retrieval)),
            "reason": "Selected only when the measured descriptor/error evidence supports none of the proposed directions.",
        },
        "caveat": "All correlations use ten held-out families and are exploratory.",
    }


def write_plots(
    *,
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    descriptor_names: Sequence[str],
    schema: Mapping[str, DescriptorSpec],
    pca: Mapping[str, Any],
    distance_result: Mapping[str, Any],
    errors: Mapping[str, Mapping[str, float]],
    diagnoses: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to produce the requested OOD plots") from exc
    del ImageFont
    split_by_uid = {str(row["family_uid"]): str(row["split"]) for row in records}
    colors = {"train": "#386cb0", "val": "#fdb462", "test": "#ef3b2c"}
    draw_scatter_plot(
        Image=Image,
        ImageDraw=ImageDraw,
        path=out_dir / "descriptor_embedding.png",
        title="Train-fitted PCA of inference-time family descriptors",
        points=[
            (
                float(pca["coordinates"][index, 0]),
                float(pca["coordinates"][index, 1]),
                uid,
                colors[split_by_uid[uid]],
            )
            for index, uid in enumerate(pca["family_uids"])
        ],
        x_label=f"PC1 ({pca['explained_variance_ratio'][0] * 100:.1f}%)",
        y_label=f"PC2 ({pca['explained_variance_ratio'][1] * 100:.1f}%)",
    )
    heldout = sorted(errors)
    draw_multi_error_scatter(
        Image=Image,
        ImageDraw=ImageDraw,
        path=out_dir / "ood_distance_vs_error.png",
        title="Held-out descriptor OOD distance versus residual-error components",
        family_uids=heldout,
        distances={
            uid: float(distance_result["family_scores"][uid]["nearest_euclidean_distance"])
            for uid in heldout
        },
        errors=errors,
        split_by_uid=split_by_uid,
        colors=colors,
    )
    f044_features = diagnoses["f044"]["top_distinguishing_features"][:15]
    draw_bar_plot(
        Image=Image,
        ImageDraw=ImageDraw,
        path=out_dir / "f044_feature_zscores.png",
        title="f044 largest train-relative feature z-scores",
        labels=[item["feature_name"] for item in f044_features],
        values=[float(item["zscore"]) for item in f044_features],
        value_label="z-score",
    )
    nearest = diagnoses["f044"]["nearest_training_families"]
    selected = ["f044", *[str(item["train_family_uid"]) for item in nearest]]
    standardized = distance_result["standardized_by_uid"]
    top_names = [item["feature_name"] for item in f044_features[:8]]
    indices = [list(descriptor_names).index(name) for name in top_names]
    draw_grouped_comparison(
        Image=Image,
        ImageDraw=ImageDraw,
        path=out_dir / "nearest_family_comparison.png",
        title="f044 and top-5 nearest training families on distinguishing features",
        family_uids=selected,
        feature_names=top_names,
        values=np.asarray([[standardized[uid][index] for index in indices] for uid in selected]),
        schema=schema,
    )


def draw_scatter_plot(*, Image: Any, ImageDraw: Any, path: Path, title: str, points: Sequence[tuple[float, float, str, str]], x_label: str, y_label: str) -> None:
    width, height = 1100, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = (100, 70, 60, 90)
    left, top, right, bottom = margin[0], margin[1], width - margin[2], height - margin[3]
    x_values = np.asarray([point[0] for point in points], dtype=np.float64)
    y_values = np.asarray([point[1] for point in points], dtype=np.float64)
    x_min, x_max = padded_range(x_values)
    y_min, y_max = padded_range(y_values)
    draw.text((left, 22), title, fill="black")
    draw.line((left, bottom, right, bottom), fill="#444444", width=2)
    draw.line((left, top, left, bottom), fill="#444444", width=2)
    for x, y, label, color in points:
        px = left + (x - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        radius = 6 if label not in {"f044", "f041"} else 9
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="black")
        if label in {"f044", "f041"}:
            draw.text((px + 10, py - 8), label, fill="black")
    draw.text(((left + right) // 2 - 100, height - 35), x_label, fill="black")
    draw.text((10, top), y_label, fill="black")
    draw.text((left, bottom + 12), f"{x_min:.2f}", fill="#555555")
    draw.text((right - 45, bottom + 12), f"{x_max:.2f}", fill="#555555")
    draw.text((left - 72, bottom - 7), f"{y_min:.2f}", fill="#555555")
    draw.text((left - 72, top - 7), f"{y_max:.2f}", fill="#555555")
    image.save(path)


def draw_multi_error_scatter(
    *,
    Image: Any,
    ImageDraw: Any,
    path: Path,
    title: str,
    family_uids: Sequence[str],
    distances: Mapping[str, float],
    errors: Mapping[str, Mapping[str, float]],
    split_by_uid: Mapping[str, str],
    colors: Mapping[str, str],
) -> None:
    width, height = 1300, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), title, fill="black")
    panels = (
        ("source_mae_K", "Source-superposition MAE"),
        ("mean_correction_mae_K", "Mean-correction MAE"),
        ("centered_spatial_mae_K", "Centered-spatial MAE"),
        ("final_mae_K", "Final CNN MAE"),
    )
    for panel_index, (metric, label) in enumerate(panels):
        column = panel_index % 2
        row = panel_index // 2
        left = 90 + column * 630
        top = 70 + row * 420
        right = left + 540
        bottom = top + 320
        x_values = np.asarray([distances[uid] for uid in family_uids], dtype=np.float64)
        y_values = np.asarray([errors[uid][metric] for uid in family_uids], dtype=np.float64)
        x_min, x_max = padded_range(x_values)
        y_min, y_max = padded_range(y_values)
        draw.text((left, top - 22), label, fill="black")
        draw.line((left, bottom, right, bottom), fill="#444444", width=2)
        draw.line((left, top, left, bottom), fill="#444444", width=2)
        for uid in family_uids:
            x = distances[uid]
            y = errors[uid][metric]
            px = left + (x - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
            radius = 8 if uid in {"f044", "f041"} else 5
            color = colors[split_by_uid[uid]]
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="black")
            if uid in {"f044", "f041"}:
                draw.text((px + 9, py - 8), uid, fill="black")
        draw.text((left + 120, bottom + 20), "nearest train-family distance", fill="#555555")
        draw.text((left - 78, top), "MAE (K)", fill="#555555")
    image.save(path)


def draw_bar_plot(*, Image: Any, ImageDraw: Any, path: Path, title: str, labels: Sequence[str], values: Sequence[float], value_label: str) -> None:
    width, row_height = 1300, 34
    height = 100 + row_height * len(labels)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill="black")
    axis_x = 620
    max_abs = max(max(abs(value) for value in values), 1.0)
    draw.line((axis_x, 60, axis_x, height - 20), fill="#444444", width=2)
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 68 + index * row_height
        draw.text((20, y), label[:82], fill="black")
        length = 560 * abs(value) / max_abs
        end = axis_x + length if value >= 0 else axis_x - length
        draw.rectangle((min(axis_x, end), y, max(axis_x, end), y + 18), fill="#ef3b2c" if value >= 0 else "#386cb0")
        draw.text((end + (5 if value >= 0 else -55), y), f"{value:.2f}", fill="black")
    draw.text((axis_x - 45, height - 18), value_label, fill="#555555")
    image.save(path)


def draw_grouped_comparison(
    *,
    Image: Any,
    ImageDraw: Any,
    path: Path,
    title: str,
    family_uids: Sequence[str],
    feature_names: Sequence[str],
    values: np.ndarray,
    schema: Mapping[str, DescriptorSpec],
) -> None:
    width, height = 1450, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill="black")
    left, top, right, bottom = 100, 80, width - 40, height - 180
    max_abs = max(float(np.max(np.abs(values))), 1.0)
    zero_y = (top + bottom) // 2
    draw.line((left, zero_y, right, zero_y), fill="#444444", width=2)
    group_width = (right - left) / len(feature_names)
    bar_width = max(4.0, group_width / (len(family_uids) + 2))
    palette = ("#ef3b2c", "#386cb0", "#7fc97f", "#beaed4", "#fdc086", "#ffff99")
    for feature_index, feature in enumerate(feature_names):
        center_x = left + (feature_index + 0.5) * group_width
        for family_index, uid in enumerate(family_uids):
            value = float(values[family_index, feature_index])
            x0 = center_x + (family_index - len(family_uids) / 2) * bar_width
            y = zero_y - value / max_abs * (bottom - top) * 0.45
            draw.rectangle((x0, min(zero_y, y), x0 + bar_width * 0.8, max(zero_y, y)), fill=palette[family_index % len(palette)])
        short = feature.replace("source_base_", "").replace("_workload_mean", "")
        draw.text((center_x - group_width * 0.45, bottom + 12), short[:22], fill="black")
        draw.text((center_x - group_width * 0.45, bottom + 30), f"[{schema[feature].group}]", fill="#555555")
    for index, uid in enumerate(family_uids):
        x = 120 + index * 150
        draw.rectangle((x, height - 70, x + 18, height - 52), fill=palette[index % len(palette)])
        draw.text((x + 24, height - 70), uid, fill="black")
    image.save(path)


def write_report(path: Path, summary: Mapping[str, Any], nearest_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Benchmark v2 Family OOD Analysis",
        "",
        "Descriptors use only package definitions, compact model metadata, context occupancy, and source-superposition maps available at inference time. HotSpot-derived errors are joined only after train-only scaling and distance computation.",
        "",
        "## Focus Families",
        "",
    ]
    for uid in ("f044", "f041"):
        diagnosis = summary["family_diagnoses"][uid]
        error = summary["family_errors"][uid]
        lines.extend(
            [
                f"### {uid}",
                "",
                f"- Classification: `{diagnosis['classification']}`",
                f"- Nearest-training Euclidean distance: {summary['family_ood_scores'][uid]['nearest_euclidean_distance']:.3f}",
                f"- Final CNN MAE: {error['final_mae_K']:.3f} K",
                f"- Mean-correction MAE: {error['mean_correction_mae_K']:.3f} K",
                f"- Centered-spatial MAE: {error['centered_spatial_mae_K']:.3f} K",
                f"- Global/spatial RMS z-score: {diagnosis['global_group_rms_z']:.3f} / {diagnosis['spatial_group_rms_z']:.3f}",
                "- Largest descriptor deviations: "
                + ", ".join(
                    f"`{item['feature_name']}` ({item['zscore']:+.2f})"
                    for item in diagnosis["top_distinguishing_features"][:6]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## f044 Nearest Training Families",
            "",
            "| Rank | Family | Euclidean | Mahalanobis | Source MAE | Mean MAE | Centered MAE | Final MAE |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in nearest_rows:
        if row["heldout_family_uid"] == "f044":
            neighbor_error = summary["family_errors"].get(row["train_family_uid"], {})

            def metric(name: str) -> str:
                value = neighbor_error.get(name)
                return f"{float(value):.3f}" if value is not None else "n/a"

            lines.append(
                f"| {row['rank']} | {row['train_family_uid']} | {float(row['euclidean_distance']):.3f} | "
                f"{float(row['mahalanobis_distance']):.3f} | {metric('source_mae_K')} | "
                f"{metric('mean_correction_mae_K')} | {metric('centered_spatial_mae_K')} | "
                f"{metric('final_mae_K')} |"
            )
    lines.extend(["", "## Direction Assessment", ""])
    for key, item in summary["direction_assessment"].items():
        if key == "caveat":
            continue
        lines.append(f"- **{key}**: {'supported' if item['supported'] else 'not supported'}; {item['reason']}")
    lines.extend(
        [
            "",
            "## Statistical Caution",
            "",
            summary["statistical_caveat"],
            "",
            "The plots and correlations diagnose coverage; they do not establish causal effects or authorize held-out-target-conditioned retrieval.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_metadata_sidecars(
    *,
    data_root: Path,
    index_root: Path,
    split_indices: Sequence[Path],
) -> tuple[dict[str, dict[str, float]], list[str], dict[str, str], list[Path]]:
    manifest_path = find_sidecar("metadata_manifest.json", data_root, index_root, split_indices)
    table_path = find_sidecar("metadata_features.csv", data_root, index_root, split_indices)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest.get("active_features", []))
    if not names:
        raise ValueError(f"metadata manifest has no active_features: {manifest_path}")
    feature_stats = manifest.get("feature_stats", {})
    units = {
        name: str(feature_stats.get(name, {}).get("unit", "unspecified"))
        for name in names
    }
    rows: dict[str, dict[str, float]] = {}
    for row in read_csv_required(table_path):
        uid = require_text(row, "sample_uid")
        values: dict[str, float] = {}
        for name in names:
            if name not in row or row[name] == "":
                raise ValueError(f"metadata row {uid} lacks active feature {name}")
            values[name] = float(row[name])
        rows[uid] = values
    return rows, names, units, [manifest_path, table_path]


def load_occupancy_channel(
    *,
    data_root: Path,
    index_root: Path,
    split_indices: Sequence[Path],
) -> tuple[int, Path]:
    manifest_path = find_sidecar("feature_manifest.json", data_root, index_root, split_indices)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest.get("channel_names", []))
    if not names:
        features = manifest.get("features", [])
        names = [str(item["name"]) for item in features if isinstance(item, Mapping) and "name" in item]
    aliases = ("occupancy_mask", "occupancy", "chiplet_occupancy_mask")
    for alias in aliases:
        if alias in names:
            return names.index(alias), manifest_path
    raise ValueError(f"feature manifest lacks an occupancy channel; available={names}")


def find_sidecar(name: str, data_root: Path, index_root: Path, split_indices: Sequence[Path]) -> Path:
    candidates = [
        index_root / name,
        data_root / "derived/stages/full_50x200/context_33ch" / name,
        data_root / "derived/stages/full_50x200/metadata" / name,
    ]
    for index in split_indices:
        candidates.extend((index.parent / name, index.parent.parent / name, index.parent.parent.parent / name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"required sidecar {name} not found; checked: {[str(path) for path in candidates]}")


def resolve_data_path(logical_path: str, data_root: Path, index_path: Path) -> Path:
    path = Path(logical_path).expanduser()
    candidates = [path] if path.is_absolute() else [data_root / path, REPO_ROOT / path, index_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve logical path {logical_path!r} against data root {data_root}; "
        f"index={index_path}; candidates={[str(candidate) for candidate in candidates]}"
    )


def geometry_array(chips: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(chip["position"]["x"]),
                float(chip["position"]["y"]),
                float(chip["size"]["width"]),
                float(chip["size"]["height"]),
            ]
            for chip in chips
        ],
        dtype=np.float64,
    )


def pairwise_distances(center_x: np.ndarray, center_y: np.ndarray) -> np.ndarray:
    if len(center_x) < 2:
        return np.empty(0, dtype=np.float64)
    dx = center_x[:, None] - center_x[None, :]
    dy = center_y[:, None] - center_y[None, :]
    upper = np.triu_indices(len(center_x), k=1)
    return np.sqrt(dx[upper] ** 2 + dy[upper] ** 2)


def canonical_type(value: str) -> str:
    upper = value.upper()
    if upper in TYPE_NAMES:
        return upper
    if upper in {"MEMORY", "RAM"}:
        return "DRAM"
    return "OTHER"


def add_stats(add: Any, prefix: str, values: np.ndarray, *, group: str, unit: str, source: str) -> None:
    for suffix, value in (
        ("mean", values.mean()),
        ("std", values.std()),
        ("min", values.min()),
        ("max", values.max()),
    ):
        add(
            f"{prefix}_{suffix}",
            value,
            group=group,
            unit=unit,
            formula=f"{suffix} over chiplets",
            source=source,
        )


def aggregate_value(values: np.ndarray, aggregate: str) -> float:
    if aggregate == "mean":
        return float(values.mean())
    if aggregate == "std":
        return float(values.std())
    if aggregate == "q10":
        return float(np.quantile(values, 0.10))
    if aggregate == "q50":
        return float(np.quantile(values, 0.50))
    if aggregate == "q90":
        return float(np.quantile(values, 0.90))
    raise KeyError(aggregate)


def metadata_varies_by_workload(name: str) -> bool:
    return name in {
        "total_power_W",
        "mean_power_density_W_per_mm2",
        "max_power_density_W_per_mm2",
    }


def metadata_group(name: str) -> str:
    return "spatial" if name in {"occupied_fraction", "whitespace_fraction"} else "global"


def family_uid_for_row(row: Mapping[str, str]) -> str:
    value = str(row.get("family_uid") or row.get("case_id") or "").strip()
    if not value:
        raise ValueError(f"row lacks family_uid/case_id; available={sorted(row)}")
    return value


def require_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"required field {key!r} is blank; available={sorted(row)}")
    return value


def ensure_finite_descriptor_records(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> None:
    if len({str(row["family_uid"]) for row in records}) != len(records):
        raise ValueError("family descriptor records contain duplicate family_uid values")
    for row in records:
        missing = [name for name in names if name not in row]
        if missing:
            raise ValueError(f"{row['family_uid']} lacks descriptor columns: {missing[:10]}")
        values = np.asarray([float(row[name]) for name in names], dtype=np.float64)
        if not np.isfinite(values).all():
            bad = [name for name, finite in zip(names, np.isfinite(values)) if not finite]
            raise ValueError(f"{row['family_uid']} has non-finite descriptors: {bad}")


def first_available_column(rows: Sequence[Mapping[str, str]], aliases: Sequence[str]) -> str | None:
    for name in aliases:
        if all(name in row and str(row[name]).strip() != "" for row in rows):
            return name
    return None


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() <= EPS or right.std() <= EPS:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def discover_known_family_error_csv(decomposition_csv: Path) -> Path | None:
    candidates = [
        decomposition_csv.parent.parent / "evaluation/known_family_sample_test/metrics_by_sample.csv",
        decomposition_csv.parent.parent / "evaluation/known_family_sample_test/per_sample_metrics.csv",
        decomposition_csv.parent.parent / "known_family_sample_test/metrics_by_sample.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def standardizer_to_json(standardizer: Standardizer) -> dict[str, Any]:
    constant = [
        name for name, std in zip(standardizer.feature_names, standardizer.std) if std <= 1.0e-12
    ]
    return {
        "fit_on": "40 primary training families only",
        "fit_family_uids": list(standardizer.fit_family_uids),
        "feature_names": list(standardizer.feature_names),
        "mean": standardizer.mean.tolist(),
        "std": standardizer.std.tolist(),
        "scale_with_floor": standardizer.scale.tolist(),
        "train_min": standardizer.minimum.tolist(),
        "train_max": standardizer.maximum.tolist(),
        "constant_train_features": constant,
    }


def padded_range(values: np.ndarray) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        return minimum - 1.0, maximum + 1.0
    padding = 0.08 * (maximum - minimum)
    return minimum - padding, maximum + padding


def portable_path(path: Path, data_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(data_root).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(resolved)


def portable_or_absolute(path: Path, data_root: Path) -> str:
    return portable_path(path, data_root)


def read_csv_required(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
