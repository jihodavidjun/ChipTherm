#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_VERSION = "source_superposition_final_train40_source_v1"
FAMILIES = ("f043", "f044")
GRID_SHAPE = (64, 64)
EPS = 1.0e-12
DEFAULT_BOUNDARY_BAND_MM = 4.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline matched-workload physical comparison of Benchmark v2 families f043 and f044."
    )
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument("--source-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--residual-decomposition-csv", required=True, type=Path)
    parser.add_argument("--f043-prediction-root", required=True, type=Path)
    parser.add_argument("--f044-prediction-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--family-root",
        type=Path,
        default=REPO_ROOT / "configs/benchmark_v2_50family/families",
    )
    parser.add_argument("--boundary-band-mm", default=DEFAULT_BOUNDARY_BAND_MM, type=float)
    parser.add_argument(
        "--decomposition-tolerance-K",
        default=0.05,
        type=float,
        help="Maximum allowed discrepancy when an existing decomposition row is available.",
    )
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    if args.boundary_band_mm <= 0:
        raise SystemExit("--boundary-band-mm must be positive")

    summary = analyze_physical_comparison(
        data_root=args.data_root.expanduser().resolve(),
        source_version=args.source_version,
        decomposition_csv=args.residual_decomposition_csv.expanduser().resolve(),
        prediction_roots={
            "f043": args.f043_prediction_root.expanduser().resolve(),
            "f044": args.f044_prediction_root.expanduser().resolve(),
        },
        family_root=args.family_root.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        boundary_band_mm=float(args.boundary_band_mm),
        decomposition_tolerance_K=float(args.decomposition_tolerance_K),
    )
    print("Benchmark v2 f043/f044 physical comparison complete")
    print(f"Matched workloads: {summary['matched_workload_count']}")
    print(f"f043/f044 final MAE: {summary['families']['f043']['final_temperature_mae_K']['mean']:.4f} / "
          f"{summary['families']['f044']['final_temperature_mae_K']['mean']:.4f} K")
    print(f"Output: {args.out_dir}")
    return 0


def analyze_physical_comparison(
    *,
    data_root: Path,
    source_version: str,
    decomposition_csv: Path,
    prediction_roots: Mapping[str, Path],
    family_root: Path,
    out_dir: Path,
    boundary_band_mm: float,
    decomposition_tolerance_K: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_index_root = (
        data_root
        / "derived/indices/full_50x200/source_superposition"
        / source_version
        / "family_split"
    )
    index_paths = {
        "f043": source_index_root / "train_index.csv",
        "f044": source_index_root / "test_index.csv",
    }
    family_rows = {
        family: select_family_rows(read_csv(index_paths[family]), family)
        for family in FAMILIES
    }
    matched_workloads = match_workloads(family_rows["f043"], family_rows["f044"], expected_count=200)
    definitions = {
        family: load_family_definition(family_root / f"{family}.yaml", family)
        for family in FAMILIES
    }
    geometry = {
        family: family_geometry(definitions[family])
        for family in FAMILIES
    }
    decomposition = load_decomposition_rows(decomposition_csv)
    audit_saved_prediction_coverage(family_rows, prediction_roots)

    records: list[dict[str, Any]] = []
    arrays_by_key: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    prediction_usage: dict[str, int] = defaultdict(int)
    decomposition_checks: list[dict[str, Any]] = []
    for workload_uid in matched_workloads:
        for family in FAMILIES:
            row = family_rows_by_workload(family_rows[family])[workload_uid]
            source = load_map(
                resolve_data_path(required_text(row, "source_superposition_base_path"), data_root, index_paths[family]),
                f"{family}/{workload_uid} source-superposition map",
            )
            target_path = resolve_target_path(row, data_root, index_paths[family])
            target = load_map(target_path, f"{family}/{workload_uid} HotSpot target")
            prediction, prediction_path, prediction_kind = load_saved_prediction(
                prediction_root=prediction_roots[family],
                family=family,
                sample_uid=required_text(row, "sample_uid"),
                source=source,
            )
            prediction_usage[prediction_kind] += 1
            metrics = compute_workload_metrics(
                family=family,
                workload_uid=workload_uid,
                row=row,
                source=source,
                target=target,
                prediction=prediction,
                geometry=geometry[family],
                boundary_band_mm=boundary_band_mm,
            )
            metrics.update(
                {
                    "source_map_path": portable_path(
                        resolve_data_path(required_text(row, "source_superposition_base_path"), data_root, index_paths[family]),
                        data_root,
                    ),
                    "target_path": portable_path(target_path, data_root),
                    "saved_prediction_path": str(prediction_path),
                    "saved_prediction_kind": prediction_kind,
                }
            )
            decomposition_row = decomposition.get((family, workload_uid))
            if decomposition_row is not None:
                check = compare_decomposition_row(metrics, decomposition_row)
                decomposition_checks.append(check)
                if check["max_abs_difference_K"] > decomposition_tolerance_K:
                    raise ValueError(
                        f"computed maps disagree with residual decomposition for {family}/{workload_uid}: {check}"
                    )
                metrics["decomposition_row_available"] = True
                metrics["decomposition_max_abs_difference_K"] = check["max_abs_difference_K"]
            else:
                metrics["decomposition_row_available"] = False
                metrics["decomposition_max_abs_difference_K"] = ""
            records.append(metrics)
            arrays_by_key[(family, workload_uid)] = {
                "source": source,
                "target": target,
                "true_residual": target - source,
                "predicted_residual": prediction - source,
                "final_error": prediction - target,
            }

    records.sort(key=lambda item: (workload_sort_key(str(item["workload_uid"])), str(item["family_uid"])))
    family_summary = {
        family: aggregate_family([record for record in records if record["family_uid"] == family])
        for family in FAMILIES
    }
    matched_summary = aggregate_matched_differences(records, matched_workloads)
    hypotheses = evaluate_hypotheses(family_summary, matched_summary)
    representative = select_representative_workloads(records, family="f044")
    write_representative_maps(
        out_dir=out_dir / "representative_maps",
        selected=representative,
        arrays_by_key=arrays_by_key,
    )
    write_directional_gradient_plot(out_dir / "directional_gradient_comparison.png", records)
    write_directional_frequency_plot(out_dir / "directional_frequency_comparison.png", records)
    write_matched_error_plot(out_dir / "matched_workload_error_comparison.png", records, matched_workloads)

    write_csv(out_dir / "per_workload_comparison.csv", records)
    summary = {
        "schema_version": "benchmark_v2_f043_f044_physical_comparison/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": source_version,
        "matched_workload_count": len(matched_workloads),
        "matched_workload_uids": matched_workloads,
        "families": family_summary,
        "matched_f044_minus_f043": matched_summary,
        "hypothesis_tests": hypotheses,
        "representative_workloads": representative,
        "geometry": geometry,
        "definitions": {
            "source_temperature_mae_K": "mean(abs(source_superposition_K - HotSpot_K))",
            "true_residual_abs_mean_K": "mean(abs(HotSpot_K - source_superposition_K))",
            "predicted_residual_abs_mean_K": "mean(abs(final_prediction_K - source_superposition_K))",
            "predicted_residual_error_mae_K": "mean(abs(predicted_residual_K - true_residual_K))",
            "final_temperature_mae_K": "mean(abs(final_prediction_K - HotSpot_K))",
            "gradient_energy": "mean((dT/daxis)^2), using physical dx_mm and dy_mm",
            "directional_low_frequency_energy": (
                "fraction of centered-map FFT energy in the first two nonzero physical axis modes; "
                "x band has |fx|<=2/package_width and |fy|<=1/package_height, with x nonzero; y is symmetric"
            ),
            "boundary_contrast": (
                f"mean in cells within {boundary_band_mm:g} mm of the named package boundary minus mean in "
                "cells farther than that distance from every edge"
            ),
            "chiplet_spacing": "Euclidean gap between non-overlapping chiplet rectangles in physical millimeters",
        },
        "prediction_usage": dict(prediction_usage),
        "decomposition_cross_check": {
            "available_rows": len(decomposition_checks),
            "missing_rows": len(records) - len(decomposition_checks),
            "maximum_abs_difference_K": max(
                (float(item["max_abs_difference_K"]) for item in decomposition_checks),
                default=None,
            ),
            "tolerance_K": decomposition_tolerance_K,
            "note": "The CSV is an independent consistency check; all physical metrics are recomputed from saved maps.",
        },
        "input_contract": {
            "data_root": str(data_root),
            "indices": {family: portable_path(path, data_root) for family, path in index_paths.items()},
            "family_definitions": {
                family: str(family_root / f"{family}.yaml")
                for family in FAMILIES
            },
            "prediction_roots": {family: str(path) for family, path in prediction_roots.items()},
            "residual_decomposition_csv": str(decomposition_csv),
            "no_inference_performed": True,
            "no_hotspot_target_generated": True,
        },
        "interpretation_caveat": (
            "These matched-workload associations separate directional/global error components but do not establish "
            "causality because package geometry differs jointly between two fixed families."
        ),
    }
    write_json(out_dir / "family_summary.json", summary)
    write_report(out_dir / "f043_f044_physical_report.md", summary)
    return summary


def compute_workload_metrics(
    *,
    family: str,
    workload_uid: str,
    row: Mapping[str, str],
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    geometry: Mapping[str, Any],
    boundary_band_mm: float,
) -> dict[str, Any]:
    dx_mm = float(geometry["dx_mm"])
    dy_mm = float(geometry["dy_mm"])
    true_residual = target - source
    predicted_residual = prediction - source
    residual_error = predicted_residual - true_residual
    true_mean = float(true_residual.mean())
    predicted_mean = float(predicted_residual.mean())
    true_centered = true_residual - true_mean
    predicted_centered = predicted_residual - predicted_mean
    centered_error = predicted_centered - true_centered
    source_gradient = directional_gradient_metrics(source, dx_mm, dy_mm)
    target_gradient = directional_gradient_metrics(target, dx_mm, dy_mm)
    final_gradient = directional_gradient_metrics(prediction, dx_mm, dy_mm)
    true_gradient = directional_gradient_metrics(true_residual, dx_mm, dy_mm)
    predicted_gradient = directional_gradient_metrics(predicted_residual, dx_mm, dy_mm)
    error_gradient = directional_gradient_metrics(residual_error, dx_mm, dy_mm)
    source_frequency = directional_low_frequency_metrics(
        source,
        float(geometry["package_width_mm"]),
        float(geometry["package_height_mm"]),
    )
    target_frequency = directional_low_frequency_metrics(
        target,
        float(geometry["package_width_mm"]),
        float(geometry["package_height_mm"]),
    )
    final_frequency = directional_low_frequency_metrics(
        prediction,
        float(geometry["package_width_mm"]),
        float(geometry["package_height_mm"]),
    )
    true_frequency = directional_low_frequency_metrics(
        true_residual,
        float(geometry["package_width_mm"]),
        float(geometry["package_height_mm"]),
    )
    predicted_frequency = directional_low_frequency_metrics(
        predicted_residual,
        float(geometry["package_width_mm"]),
        float(geometry["package_height_mm"]),
    )
    error_frequency = directional_low_frequency_metrics(
        residual_error,
        float(geometry["package_width_mm"]),
        float(geometry["package_height_mm"]),
    )
    source_boundary = boundary_contrasts(
        source,
        width_mm=float(geometry["package_width_mm"]),
        height_mm=float(geometry["package_height_mm"]),
        boundary_band_mm=boundary_band_mm,
    )
    target_boundary = boundary_contrasts(
        target,
        width_mm=float(geometry["package_width_mm"]),
        height_mm=float(geometry["package_height_mm"]),
        boundary_band_mm=boundary_band_mm,
    )
    final_boundary = boundary_contrasts(
        prediction,
        width_mm=float(geometry["package_width_mm"]),
        height_mm=float(geometry["package_height_mm"]),
        boundary_band_mm=boundary_band_mm,
    )
    true_boundary = boundary_contrasts(
        true_residual,
        width_mm=float(geometry["package_width_mm"]),
        height_mm=float(geometry["package_height_mm"]),
        boundary_band_mm=boundary_band_mm,
    )
    predicted_boundary = boundary_contrasts(
        predicted_residual,
        width_mm=float(geometry["package_width_mm"]),
        height_mm=float(geometry["package_height_mm"]),
        boundary_band_mm=boundary_band_mm,
    )
    return {
        "family_uid": family,
        "sample_uid": required_text(row, "sample_uid"),
        "workload_uid": workload_uid,
        "workload_regime": str(
            row.get("broad_stratum")
            or row.get("power_regime")
            or row.get("workload_stratum")
            or row.get("workload_cell")
            or "unknown"
        ),
        "workload_cell": str(row.get("workload_cell", "")),
        "power_regime": str(row.get("power_regime", "")),
        "topology_regime": str(row.get("topology_regime", "")),
        "package_width_mm": float(geometry["package_width_mm"]),
        "package_height_mm": float(geometry["package_height_mm"]),
        "package_aspect_ratio": float(geometry["package_aspect_ratio"]),
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "cell_size_anisotropy_ratio": max(dx_mm, dy_mm) / min(dx_mm, dy_mm),
        "source_temperature_mae_K": float(np.mean(np.abs(source - target))),
        "true_residual_abs_mean_K": float(np.mean(np.abs(true_residual))),
        "predicted_residual_abs_mean_K": float(np.mean(np.abs(predicted_residual))),
        "predicted_residual_error_mae_K": float(np.mean(np.abs(residual_error))),
        "final_temperature_mae_K": float(np.mean(np.abs(prediction - target))),
        "true_residual_mean_K": true_mean,
        "predicted_residual_mean_K": predicted_mean,
        "residual_mean_error_K": predicted_mean - true_mean,
        "absolute_residual_mean_error_K": abs(predicted_mean - true_mean),
        "centered_spatial_mae_K": float(np.mean(np.abs(centered_error))),
        "source_x_gradient_energy_K2_per_mm2": source_gradient["x_energy"],
        "source_y_gradient_energy_K2_per_mm2": source_gradient["y_energy"],
        "source_x_y_gradient_energy_ratio": source_gradient["x_y_ratio"],
        "target_x_gradient_energy_K2_per_mm2": target_gradient["x_energy"],
        "target_y_gradient_energy_K2_per_mm2": target_gradient["y_energy"],
        "target_x_y_gradient_energy_ratio": target_gradient["x_y_ratio"],
        "final_prediction_x_gradient_energy_K2_per_mm2": final_gradient["x_energy"],
        "final_prediction_y_gradient_energy_K2_per_mm2": final_gradient["y_energy"],
        "final_prediction_x_y_gradient_energy_ratio": final_gradient["x_y_ratio"],
        "true_residual_x_gradient_energy_K2_per_mm2": true_gradient["x_energy"],
        "true_residual_y_gradient_energy_K2_per_mm2": true_gradient["y_energy"],
        "true_residual_x_y_gradient_energy_ratio": true_gradient["x_y_ratio"],
        "predicted_residual_x_gradient_energy_K2_per_mm2": predicted_gradient["x_energy"],
        "predicted_residual_y_gradient_energy_K2_per_mm2": predicted_gradient["y_energy"],
        "predicted_residual_x_y_gradient_energy_ratio": predicted_gradient["x_y_ratio"],
        "residual_error_x_gradient_energy_K2_per_mm2": error_gradient["x_energy"],
        "residual_error_y_gradient_energy_K2_per_mm2": error_gradient["y_energy"],
        "residual_error_x_y_gradient_energy_ratio": error_gradient["x_y_ratio"],
        "source_boundary_minus_interior_K": source_boundary["all"],
        "target_boundary_minus_interior_K": target_boundary["all"],
        "final_prediction_boundary_minus_interior_K": final_boundary["all"],
        "true_residual_boundary_minus_interior_K": true_boundary["all"],
        "predicted_residual_boundary_minus_interior_K": predicted_boundary["all"],
        "boundary_contrast_error_K": predicted_boundary["all"] - true_boundary["all"],
        "true_residual_x_boundary_minus_interior_K": true_boundary["x"],
        "predicted_residual_x_boundary_minus_interior_K": predicted_boundary["x"],
        "x_boundary_contrast_error_K": predicted_boundary["x"] - true_boundary["x"],
        "true_residual_y_boundary_minus_interior_K": true_boundary["y"],
        "predicted_residual_y_boundary_minus_interior_K": predicted_boundary["y"],
        "y_boundary_contrast_error_K": predicted_boundary["y"] - true_boundary["y"],
        "source_x_low_frequency_energy_fraction": source_frequency["x_fraction"],
        "source_y_low_frequency_energy_fraction": source_frequency["y_fraction"],
        "target_x_low_frequency_energy_fraction": target_frequency["x_fraction"],
        "target_y_low_frequency_energy_fraction": target_frequency["y_fraction"],
        "final_prediction_x_low_frequency_energy_fraction": final_frequency["x_fraction"],
        "final_prediction_y_low_frequency_energy_fraction": final_frequency["y_fraction"],
        "true_residual_x_low_frequency_energy_fraction": true_frequency["x_fraction"],
        "true_residual_y_low_frequency_energy_fraction": true_frequency["y_fraction"],
        "true_residual_x_y_low_frequency_energy_ratio": true_frequency["x_y_ratio"],
        "predicted_residual_x_low_frequency_energy_fraction": predicted_frequency["x_fraction"],
        "predicted_residual_y_low_frequency_energy_fraction": predicted_frequency["y_fraction"],
        "predicted_residual_x_y_low_frequency_energy_ratio": predicted_frequency["x_y_ratio"],
        "residual_error_x_low_frequency_energy_fraction": error_frequency["x_fraction"],
        "residual_error_y_low_frequency_energy_fraction": error_frequency["y_fraction"],
        "chiplet_pairwise_center_distance_mm_min": float(geometry["chiplet_pairwise_center_distance_mm_min"]),
        "chiplet_pairwise_center_distance_mm_mean": float(geometry["chiplet_pairwise_center_distance_mm_mean"]),
        "chiplet_rectangle_gap_mm_min": float(geometry["chiplet_rectangle_gap_mm_min"]),
        "chiplet_rectangle_gap_mm_mean": float(geometry["chiplet_rectangle_gap_mm_mean"]),
        "chiplet_boundary_distance_mm_min": float(geometry["chiplet_boundary_distance_mm_min"]),
        "chiplet_boundary_distance_mm_mean": float(geometry["chiplet_boundary_distance_mm_mean"]),
    }


def directional_gradient_metrics(field: np.ndarray, dx_mm: float, dy_mm: float) -> dict[str, float]:
    gradient_y, gradient_x = np.gradient(validated_map(field, "gradient field"), dy_mm, dx_mm)
    x_energy = float(np.mean(gradient_x * gradient_x))
    y_energy = float(np.mean(gradient_y * gradient_y))
    return {
        "x_energy": x_energy,
        "y_energy": y_energy,
        "x_y_ratio": x_energy / max(y_energy, EPS),
    }


def directional_low_frequency_metrics(
    field: np.ndarray,
    package_width_mm: float,
    package_height_mm: float,
) -> dict[str, float]:
    centered = validated_map(field, "frequency field") - float(np.mean(field))
    spectrum = np.fft.fft2(centered, norm="ortho")
    energy = np.abs(spectrum) ** 2
    fx = np.fft.fftfreq(centered.shape[1], d=package_width_mm / centered.shape[1])[None, :]
    fy = np.fft.fftfreq(centered.shape[0], d=package_height_mm / centered.shape[0])[:, None]
    x_nonzero = np.abs(fx) > EPS
    y_nonzero = np.abs(fy) > EPS
    x_band = x_nonzero & (np.abs(fx) <= (2.0 / package_width_mm + EPS)) & (np.abs(fy) <= (1.0 / package_height_mm + EPS))
    y_band = y_nonzero & (np.abs(fy) <= (2.0 / package_height_mm + EPS)) & (np.abs(fx) <= (1.0 / package_width_mm + EPS))
    total = float(energy.sum())
    x_fraction = float(energy[x_band].sum() / max(total, EPS))
    y_fraction = float(energy[y_band].sum() / max(total, EPS))
    return {
        "x_fraction": x_fraction,
        "y_fraction": y_fraction,
        "x_y_ratio": x_fraction / max(y_fraction, EPS),
    }


def boundary_contrasts(
    field: np.ndarray,
    *,
    width_mm: float,
    height_mm: float,
    boundary_band_mm: float,
) -> dict[str, float]:
    array = validated_map(field, "boundary field")
    x_centers = (np.arange(array.shape[1], dtype=np.float64) + 0.5) * width_mm / array.shape[1]
    y_centers = (np.arange(array.shape[0], dtype=np.float64) + 0.5) * height_mm / array.shape[0]
    x_edge_1d = np.minimum(x_centers, width_mm - x_centers) <= boundary_band_mm
    y_edge_1d = np.minimum(y_centers, height_mm - y_centers) <= boundary_band_mm
    x_edge = np.broadcast_to(x_edge_1d[None, :], array.shape)
    y_edge = np.broadcast_to(y_edge_1d[:, None], array.shape)
    interior = ~(x_edge | y_edge)
    if not interior.any():
        raise ValueError(
            f"boundary band {boundary_band_mm} mm leaves no interior for package {width_mm}x{height_mm} mm"
        )
    interior_mean = float(array[interior].mean())
    return {
        "all": float(array[x_edge | y_edge].mean() - interior_mean),
        "x": float(array[x_edge].mean() - interior_mean),
        "y": float(array[y_edge].mean() - interior_mean),
    }


def family_geometry(definition: Mapping[str, Any]) -> dict[str, Any]:
    structure = definition["fixed_structure"]
    layout = structure["layout"]
    package = layout["package"]["size"]
    width = float(package["width"])
    height = float(package["height"])
    chiplets = list(layout["chiplets"])
    rectangles = np.asarray(
        [
            [
                float(chip["position"]["x"]),
                float(chip["position"]["y"]),
                float(chip["size"]["width"]),
                float(chip["size"]["height"]),
            ]
            for chip in chiplets
        ],
        dtype=np.float64,
    )
    centers = rectangles[:, :2] + 0.5 * rectangles[:, 2:]
    center_distances: list[float] = []
    rectangle_gaps: list[float] = []
    for left in range(len(rectangles)):
        for right in range(left + 1, len(rectangles)):
            center_distances.append(float(np.linalg.norm(centers[left] - centers[right])))
            rectangle_gaps.append(rectangle_gap(rectangles[left], rectangles[right]))
    boundary = np.minimum.reduce(
        (
            rectangles[:, 0],
            rectangles[:, 1],
            width - rectangles[:, 0] - rectangles[:, 2],
            height - rectangles[:, 1] - rectangles[:, 3],
        )
    )
    if np.any(boundary < -1.0e-6):
        raise ValueError(f"{definition['family_uid']} has chiplets outside its package")
    return {
        "package_width_mm": width,
        "package_height_mm": height,
        "package_aspect_ratio": max(width, height) / min(width, height),
        "dx_mm": width / GRID_SHAPE[1],
        "dy_mm": height / GRID_SHAPE[0],
        "cell_size_anisotropy_ratio": max(width, height) / min(width, height),
        "chiplet_count": len(chiplets),
        "chiplet_pairwise_center_distance_mm_min": min(center_distances) if center_distances else 0.0,
        "chiplet_pairwise_center_distance_mm_mean": float(np.mean(center_distances)) if center_distances else 0.0,
        "chiplet_rectangle_gap_mm_min": min(rectangle_gaps) if rectangle_gaps else 0.0,
        "chiplet_rectangle_gap_mm_mean": float(np.mean(rectangle_gaps)) if rectangle_gaps else 0.0,
        "chiplet_boundary_distance_mm_min": float(boundary.min()),
        "chiplet_boundary_distance_mm_mean": float(boundary.mean()),
    }


def rectangle_gap(left: np.ndarray, right: np.ndarray) -> float:
    left_x0, left_y0, left_w, left_h = [float(value) for value in left]
    right_x0, right_y0, right_w, right_h = [float(value) for value in right]
    x_gap = max(left_x0 - (right_x0 + right_w), right_x0 - (left_x0 + left_w), 0.0)
    y_gap = max(left_y0 - (right_y0 + right_h), right_y0 - (left_y0 + left_h), 0.0)
    return float(math.hypot(x_gap, y_gap))


def aggregate_family(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_names = [
        name
        for name in records[0]
        if name not in {
            "family_uid",
            "sample_uid",
            "workload_uid",
            "workload_regime",
            "workload_cell",
            "power_regime",
            "topology_regime",
            "source_map_path",
            "target_path",
            "saved_prediction_path",
            "saved_prediction_kind",
            "decomposition_row_available",
            "decomposition_max_abs_difference_K",
        }
        and isinstance(records[0][name], (int, float, np.number))
    ]
    output: dict[str, Any] = {"workload_count": len(records)}
    for name in numeric_names:
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    return output


def aggregate_matched_differences(
    records: Sequence[Mapping[str, Any]],
    workload_uids: Sequence[str],
) -> dict[str, Any]:
    by_key = {(str(row["family_uid"]), str(row["workload_uid"])): row for row in records}
    metric_names = (
        "source_temperature_mae_K",
        "final_temperature_mae_K",
        "absolute_residual_mean_error_K",
        "centered_spatial_mae_K",
        "residual_error_x_gradient_energy_K2_per_mm2",
        "residual_error_y_gradient_energy_K2_per_mm2",
        "x_boundary_contrast_error_K",
        "y_boundary_contrast_error_K",
    )
    output: dict[str, Any] = {}
    for name in metric_names:
        use_magnitude = name in {"x_boundary_contrast_error_K", "y_boundary_contrast_error_K"}
        differences = np.asarray(
            [
                (
                    abs(float(by_key[("f044", workload)][name]))
                    - abs(float(by_key[("f043", workload)][name]))
                    if use_magnitude
                    else float(by_key[("f044", workload)][name])
                    - float(by_key[("f043", workload)][name])
                )
                for workload in workload_uids
            ],
            dtype=np.float64,
        )
        output[name] = {
            "mean": float(differences.mean()),
            "median": float(np.median(differences)),
            "positive_fraction": float(np.mean(differences > 0)),
            "min": float(differences.min()),
            "max": float(differences.max()),
        }
    return output


def evaluate_hypotheses(
    families: Mapping[str, Mapping[str, Any]],
    matched: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    f043 = families["f043"]
    f044 = families["f044"]
    f044_worse = f044["final_temperature_mae_K"]["mean"] > f043["final_temperature_mae_K"]["mean"]
    anisotropy_ratio = (
        f044["cell_size_anisotropy_ratio"]["mean"]
        / max(f043["cell_size_anisotropy_ratio"]["mean"], EPS)
    )
    aspect_ratio = (
        f044["package_aspect_ratio"]["mean"]
        / max(f043["package_aspect_ratio"]["mean"], EPS)
    )
    boundary_error_043 = (
        abs(f043["x_boundary_contrast_error_K"]["mean"])
        + abs(f043["y_boundary_contrast_error_K"]["mean"])
    )
    boundary_error_044 = (
        abs(f044["x_boundary_contrast_error_K"]["mean"])
        + abs(f044["y_boundary_contrast_error_K"]["mean"])
    )
    mean_signed = f044["residual_mean_error_K"]["mean"]
    mean_ratio = (
        f044["absolute_residual_mean_error_K"]["mean"]
        / max(f043["absolute_residual_mean_error_K"]["mean"], EPS)
    )
    centered_ratio = (
        f044["centered_spatial_mae_K"]["mean"]
        / max(f043["centered_spatial_mae_K"]["mean"], EPS)
    )
    return {
        "physical_cell_size_anisotropy": {
            "associated": bool(f044_worse and anisotropy_ratio >= 1.25),
            "f044_to_f043_anisotropy_ratio": anisotropy_ratio,
            "evidence": "Family-level association only; dx and dy are fixed within each family.",
        },
        "package_aspect_ratio": {
            "associated": bool(f044_worse and aspect_ratio >= 1.25),
            "f044_to_f043_aspect_ratio": aspect_ratio,
            "evidence": "Family-level association only; aspect ratio and cell anisotropy are collinear on a fixed 64x64 grid.",
        },
        "directional_boundary_response": {
            "associated": bool(f044_worse and boundary_error_044 >= 1.25 * max(boundary_error_043, EPS)),
            "f043_mean_absolute_directional_contrast_error_K": boundary_error_043,
            "f044_mean_absolute_directional_contrast_error_K": boundary_error_044,
            "matched_f044_worse_fraction_x": matched["x_boundary_contrast_error_K"]["positive_fraction"],
            "matched_f044_worse_fraction_y": matched["y_boundary_contrast_error_K"]["positive_fraction"],
        },
        "global_mean_undercorrection": {
            "associated": bool(mean_signed < -0.25 and mean_ratio >= 1.25),
            "f044_mean_signed_residual_mean_error_K": mean_signed,
            "f044_to_f043_absolute_mean_error_ratio": mean_ratio,
            "definition": "negative signed error means predicted residual mean is too low",
        },
        "centered_spatial_mismatch": {
            "associated": bool(centered_ratio >= 1.25),
            "f044_to_f043_centered_spatial_mae_ratio": centered_ratio,
            "matched_f044_worse_fraction": matched["centered_spatial_mae_K"]["positive_fraction"],
        },
        "caveat": (
            "Thresholds flag associations for diagnosis, not causal tests. f043 and f044 differ in multiple fixed "
            "geometric variables, while the 200 paired workloads provide repeated loading conditions."
        ),
    }


def compare_decomposition_row(
    metrics: Mapping[str, Any],
    decomposition_row: Mapping[str, str],
) -> dict[str, Any]:
    comparisons = {
        "source_superposition_mae_K": float(metrics["source_temperature_mae_K"]),
        "final_cnn_mae_K": float(metrics["final_temperature_mae_K"]),
        "true_residual_mean_K": float(metrics["true_residual_mean_K"]),
        "predicted_scalar_mean_correction_K": float(metrics["predicted_residual_mean_K"]),
        "centered_spatial_mae_K": float(metrics["centered_spatial_mae_K"]),
    }
    differences = {
        name: abs(value - float(decomposition_row[name]))
        for name, value in comparisons.items()
        if name in decomposition_row and str(decomposition_row[name]).strip() != ""
    }
    return {
        "family_uid": metrics["family_uid"],
        "workload_uid": metrics["workload_uid"],
        "compared_fields": len(differences),
        "max_abs_difference_K": max(differences.values(), default=0.0),
        "field_differences_K": differences,
    }


def select_representative_workloads(
    records: Sequence[Mapping[str, Any]],
    *,
    family: str,
) -> dict[str, str]:
    family_records = sorted(
        (record for record in records if record["family_uid"] == family),
        key=lambda item: (float(item["final_temperature_mae_K"]), workload_sort_key(str(item["workload_uid"]))),
    )
    return {
        "easy": str(family_records[0]["workload_uid"]),
        "median": str(family_records[(len(family_records) - 1) // 2]["workload_uid"]),
        "worst": str(family_records[-1]["workload_uid"]),
    }


def write_representative_maps(
    *,
    out_dir: Path,
    selected: Mapping[str, str],
    arrays_by_key: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> None:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    for category, workload_uid in selected.items():
        family_arrays = {family: arrays_by_key[(family, workload_uid)] for family in FAMILIES}
        source_target_values = np.concatenate(
            [family_arrays[family][name].ravel() for family in FAMILIES for name in ("source", "target")]
        )
        residual_limit = max(
            float(np.max(np.abs(family_arrays[family][name])))
            for family in FAMILIES
            for name in ("true_residual", "predicted_residual")
        )
        error_limit = max(
            float(np.max(np.abs(family_arrays[family]["final_error"])))
            for family in FAMILIES
        )
        columns = ("source", "target", "true_residual", "predicted_residual", "final_error")
        tile = 230
        image = Image.new("RGB", (tile * len(columns), 70 + tile * len(FAMILIES)), "white")
        draw = ImageDraw.Draw(image)
        draw.text((15, 15), f"{category}: matched workload {workload_uid}", fill="black")
        for column, name in enumerate(columns):
            draw.text((column * tile + 10, 45), name.replace("_", " "), fill="black")
        for row, family in enumerate(FAMILIES):
            draw.text((5, 75 + row * tile), family, fill="black")
            for column, name in enumerate(columns):
                array = family_arrays[family][name]
                if name in {"source", "target"}:
                    colored = heatmap_image(array, float(source_target_values.min()), float(source_target_values.max()), diverging=False)
                elif name == "final_error":
                    colored = heatmap_image(array, -error_limit, error_limit, diverging=True)
                else:
                    colored = heatmap_image(array, -residual_limit, residual_limit, diverging=True)
                colored = colored.resize((tile - 25, tile - 25), resample=Image.Resampling.NEAREST)
                image.paste(colored, (column * tile + 20, 85 + row * tile))
        image.save(out_dir / f"{category}_{workload_uid}.png")


def heatmap_image(array: np.ndarray, minimum: float, maximum: float, *, diverging: bool) -> Any:
    from PIL import Image

    normalized = np.clip((array - minimum) / max(maximum - minimum, EPS), 0.0, 1.0)
    if diverging:
        red = np.where(normalized >= 0.5, 255.0, 510.0 * normalized)
        blue = np.where(normalized <= 0.5, 255.0, 510.0 * (1.0 - normalized))
        green = 255.0 - 1.5 * np.abs(normalized - 0.5) * 255.0
    else:
        red = 255.0 * np.clip(1.5 * normalized - 0.25, 0.0, 1.0)
        green = 255.0 * np.clip(1.5 * normalized, 0.0, 1.0)
        blue = 255.0 * np.clip(1.25 - 1.5 * normalized, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def write_directional_gradient_plot(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    series = {
        family: {
            "true x": [float(row["true_residual_x_gradient_energy_K2_per_mm2"]) for row in records if row["family_uid"] == family],
            "true y": [float(row["true_residual_y_gradient_energy_K2_per_mm2"]) for row in records if row["family_uid"] == family],
            "error x": [float(row["residual_error_x_gradient_energy_K2_per_mm2"]) for row in records if row["family_uid"] == family],
            "error y": [float(row["residual_error_y_gradient_energy_K2_per_mm2"]) for row in records if row["family_uid"] == family],
        }
        for family in FAMILIES
    }
    write_box_summary_plot(path, "Directional residual gradient energy", series, "K^2/mm^2")


def write_directional_frequency_plot(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    series = {
        family: {
            "true x": [float(row["true_residual_x_low_frequency_energy_fraction"]) for row in records if row["family_uid"] == family],
            "true y": [float(row["true_residual_y_low_frequency_energy_fraction"]) for row in records if row["family_uid"] == family],
            "pred x": [float(row["predicted_residual_x_low_frequency_energy_fraction"]) for row in records if row["family_uid"] == family],
            "pred y": [float(row["predicted_residual_y_low_frequency_energy_fraction"]) for row in records if row["family_uid"] == family],
        }
        for family in FAMILIES
    }
    write_box_summary_plot(path, "Directional low-frequency residual energy", series, "energy fraction")


def write_box_summary_plot(
    path: Path,
    title: str,
    series: Mapping[str, Mapping[str, Sequence[float]]],
    y_label: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), title, fill="black")
    left, top, right, bottom = 100, 70, width - 40, height - 100
    all_values = np.asarray(
        [value for family in FAMILIES for values in series[family].values() for value in values],
        dtype=np.float64,
    )
    y_min = min(0.0, float(all_values.min()))
    y_max = max(float(all_values.max()), EPS)
    y_max *= 1.08
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    labels = list(next(iter(series.values())))
    palette = {"f043": "#386cb0", "f044": "#ef3b2c"}
    group_width = (right - left) / len(labels)
    for label_index, label in enumerate(labels):
        center = left + (label_index + 0.5) * group_width
        for family_index, family in enumerate(FAMILIES):
            values = np.asarray(series[family][label], dtype=np.float64)
            q10, q25, median, q75, q90 = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90))
            x = center + (-35 if family_index == 0 else 35)

            def py(value: float) -> float:
                return bottom - (value - y_min) / max(y_max - y_min, EPS) * (bottom - top)

            draw.line((x, py(q10), x, py(q90)), fill=palette[family], width=3)
            draw.rectangle((x - 20, py(q75), x + 20, py(q25)), outline=palette[family], width=3)
            draw.line((x - 20, py(median), x + 20, py(median)), fill=palette[family], width=3)
        draw.text((center - 45, bottom + 15), label, fill="black")
    draw.text((10, top), y_label, fill="black")
    draw.rectangle((right - 180, top + 10, right - 165, top + 25), fill=palette["f043"])
    draw.text((right - 155, top + 10), "f043", fill="black")
    draw.rectangle((right - 100, top + 10, right - 85, top + 25), fill=palette["f044"])
    draw.text((right - 75, top + 10), "f044", fill="black")
    image.save(path)


def write_matched_error_plot(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    workload_uids: Sequence[str],
) -> None:
    from PIL import Image, ImageDraw

    by_key = {(str(row["family_uid"]), str(row["workload_uid"])): row for row in records}
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Matched workload final-temperature MAE", fill="black")
    left, top, right, bottom = 80, 70, width - 40, height - 90
    values = np.asarray(
        [float(by_key[(family, workload)]["final_temperature_mae_K"]) for family in FAMILIES for workload in workload_uids],
        dtype=np.float64,
    )
    maximum = max(float(values.max()) * 1.05, EPS)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    for index, workload in enumerate(workload_uids):
        x = left + index / max(len(workload_uids) - 1, 1) * (right - left)
        f043 = float(by_key[("f043", workload)]["final_temperature_mae_K"])
        f044 = float(by_key[("f044", workload)]["final_temperature_mae_K"])
        y043 = bottom - f043 / maximum * (bottom - top)
        y044 = bottom - f044 / maximum * (bottom - top)
        draw.line((x, y043, x, y044), fill="#bbbbbb", width=1)
        draw.point((x, y043), fill="#386cb0")
        draw.point((x, y044), fill="#ef3b2c")
    draw.text((left, bottom + 15), "w001", fill="#555555")
    draw.text((right - 35, bottom + 15), "w200", fill="#555555")
    draw.text((10, top), "MAE (K)", fill="black")
    image.save(path)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    f043 = summary["families"]["f043"]
    f044 = summary["families"]["f044"]
    lines = [
        "# Benchmark v2 f043 vs f044 Physical Comparison",
        "",
        f"Matched workloads: **{summary['matched_workload_count']}**.",
        "",
        "## Family Geometry",
        "",
        "| Family | Width (mm) | Height (mm) | Aspect | dx (mm) | dy (mm) | Min chiplet gap (mm) | Mean boundary clearance (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        geometry = summary["geometry"][family]
        lines.append(
            f"| {family} | {geometry['package_width_mm']:.3f} | {geometry['package_height_mm']:.3f} | "
            f"{geometry['package_aspect_ratio']:.3f} | {geometry['dx_mm']:.4f} | {geometry['dy_mm']:.4f} | "
            f"{geometry['chiplet_rectangle_gap_mm_min']:.3f} | {geometry['chiplet_boundary_distance_mm_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Error Decomposition",
            "",
            "| Family | Source MAE | Final MAE | Mean-correction error | Centered-spatial MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, values in (("f043", f043), ("f044", f044)):
        lines.append(
            f"| {family} | {values['source_temperature_mae_K']['mean']:.3f} K | "
            f"{values['final_temperature_mae_K']['mean']:.3f} K | "
            f"{values['absolute_residual_mean_error_K']['mean']:.3f} K | "
            f"{values['centered_spatial_mae_K']['mean']:.3f} K |"
        )
    lines.extend(["", "## Explicit Hypothesis Checks", ""])
    for name, item in summary["hypothesis_tests"].items():
        if name == "caveat":
            continue
        lines.append(f"- **{name}**: {'associated' if item['associated'] else 'not associated by the preset criterion'}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["hypothesis_tests"]["caveat"],
            "",
            "The residual-decomposition CSV is used only as a numerical cross-check. All directional and physical metrics are recomputed from cached source, target, and final-prediction maps.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def match_workloads(
    f043_rows: Sequence[Mapping[str, str]],
    f044_rows: Sequence[Mapping[str, str]],
    *,
    expected_count: int,
) -> list[str]:
    left = family_rows_by_workload(f043_rows)
    right = family_rows_by_workload(f044_rows)
    missing_f043 = sorted(set(right) - set(left), key=workload_sort_key)
    missing_f044 = sorted(set(left) - set(right), key=workload_sort_key)
    if missing_f043 or missing_f044 or len(left) != expected_count or len(right) != expected_count:
        raise ValueError(
            "f043/f044 workload matching failed: "
            f"f043={len(left)} f044={len(right)} expected={expected_count} "
            f"missing_from_f043={missing_f043[:10]} missing_from_f044={missing_f044[:10]}"
        )
    return sorted(left, key=workload_sort_key)


def family_rows_by_workload(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for row in rows:
        workload_uid = normalized_workload_uid(row)
        if workload_uid in output:
            raise ValueError(f"duplicate workload_uid {workload_uid}")
        output[workload_uid] = row
    return output


def normalized_workload_uid(row: Mapping[str, str]) -> str:
    value = str(row.get("workload_uid", "")).strip()
    if value:
        return value
    sample_uid = required_text(row, "sample_uid")
    prefix = f"{str(row.get('family_uid') or row.get('case_id'))}_"
    if sample_uid.startswith(prefix):
        return sample_uid[len(prefix) :]
    raise ValueError(f"cannot derive workload_uid from row {sample_uid}")


def workload_sort_key(value: str) -> tuple[int, str]:
    prefix = value.split("_", 1)[0]
    if prefix.startswith("w") and prefix[1:].isdigit():
        return int(prefix[1:]), value
    return 10**9, value


def select_family_rows(rows: Sequence[Mapping[str, str]], family: str) -> list[dict[str, str]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("family_uid") or row.get("case_id")) == family
    ]
    if len(selected) != 200:
        raise ValueError(f"{family} index selection expected 200 rows, found {len(selected)}")
    return selected


def load_family_definition(path: Path, family: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if str(payload.get("family_uid")) != family:
        raise ValueError(f"family YAML mismatch: expected {family}, found {payload.get('family_uid')}")
    return payload


def load_decomposition_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        family = str(row.get("family_uid") or row.get("case_id") or "")
        if family not in FAMILIES:
            continue
        workload = normalized_workload_uid(row)
        key = (family, workload)
        if key in output:
            raise ValueError(f"duplicate decomposition row for {family}/{workload}")
        output[key] = row
    if not any(family == "f044" for family, _ in output):
        raise ValueError("residual-decomposition CSV contains no f044 rows")
    return output


def load_saved_prediction(
    *,
    prediction_root: Path,
    family: str,
    sample_uid: str,
    source: np.ndarray,
) -> tuple[np.ndarray, Path, str]:
    path, kind = find_saved_prediction_file(prediction_root, family, sample_uid)
    array = load_map(path, f"{family}/{sample_uid} saved {kind}")
    prediction = array if kind == "final_temperature" else source + array
    return validated_map(prediction, "reconstructed final prediction"), path.resolve(), kind


def find_saved_prediction_file(
    prediction_root: Path,
    family: str,
    sample_uid: str,
) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    roots = (
        prediction_root,
        prediction_root / "predictions",
        prediction_root / "predicted_residuals",
    )
    final_names = (
        f"{sample_uid}_tpred.npy",
        f"{sample_uid}_prediction.npy",
        f"{sample_uid}_temperature_pred.npy",
    )
    residual_names = (
        f"{sample_uid}_residual_pred.npy",
        f"{sample_uid}_predicted_residual.npy",
    )
    for root in roots:
        for name in final_names:
            candidates.extend(((root / family / name, "final_temperature"), (root / name, "final_temperature")))
        for name in residual_names:
            candidates.extend(((root / family / name, "predicted_residual"), (root / name, "predicted_residual")))
    for path, kind in candidates:
        if path.is_file():
            return path, kind
    raise FileNotFoundError(
        f"saved final prediction is missing for {family}/{sample_uid}; root={prediction_root}; "
        f"checked={[str(path) for path, _ in candidates]}"
    )


def audit_saved_prediction_coverage(
    family_rows: Mapping[str, Sequence[Mapping[str, str]]],
    prediction_roots: Mapping[str, Path],
) -> None:
    failures: dict[str, list[str]] = defaultdict(list)
    for family in FAMILIES:
        for row in family_rows[family]:
            uid = required_text(row, "sample_uid")
            try:
                find_saved_prediction_file(prediction_roots[family], family, uid)
            except FileNotFoundError:
                failures[family].append(uid)
    if failures:
        details = "; ".join(
            f"{family}: missing {len(uids)}/200, first={uids[:10]}, root={prediction_roots[family]}"
            for family, uids in sorted(failures.items())
        )
        raise FileNotFoundError(
            "the matched 200-workload analysis requires cached predictions for every row; "
            f"{details}. No checkpoint inference is performed by this script."
        )


def resolve_target_path(row: Mapping[str, str], data_root: Path, index_path: Path) -> Path:
    keys = ("y_path", "target_path", "temperature_path", "final_temperature")
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return resolve_data_path(value, data_root, index_path)
    raise ValueError(f"row {row.get('sample_uid')} has no target path; expected one of {keys}; available={sorted(row)}")


def resolve_data_path(logical_path: str, data_root: Path, index_path: Path) -> Path:
    path = Path(logical_path).expanduser()
    candidates = [path] if path.is_absolute() else [data_root / path, REPO_ROOT / path, index_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve {logical_path!r} using data_root={data_root}; candidates={[str(item) for item in candidates]}"
    )


def validated_map(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != GRID_SHAPE:
        raise ValueError(f"{name} must have shape {GRID_SHAPE}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def load_map(path: Path, name: str) -> np.ndarray:
    return validated_map(np.load(path), name)


def required_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"required field {key} is blank; available={sorted(row)}")
    return value


def portable_path(path: Path, data_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(data_root).as_posix()
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
