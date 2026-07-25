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


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_SHAPE = (64, 64)
CANONICAL_COARSE_SIZE = 8
DEFAULT_SOURCE_VERSION = "source_superposition_final_train40_source_v1"
VARIANTS = (
    "baseline_final",
    "oracle_mean",
    "oracle_centered",
    "oracle_low_frequency",
    "oracle_high_frequency",
    "oracle_mean_and_low_frequency",
    "optimal_centered_scale",
    "full_oracle",
)
METRICS = (
    "temperature_mae_K",
    "temperature_rmse_K",
    "peak_temperature_mae_K",
    "boundary_region_mae_K",
    "interior_region_mae_K",
)
EPS = 1.0e-12


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline residual-component oracle analysis for Benchmark v2 held-out families."
    )
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument("--source-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--residual-decomposition-csv", required=True, type=Path)
    parser.add_argument("--validation-prediction-root", required=True, type=Path)
    parser.add_argument("--test-prediction-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--validation-index", type=Path, default=None)
    parser.add_argument("--test-index", type=Path, default=None)
    parser.add_argument("--boundary-width-cells", default=4, type=int)
    parser.add_argument("--decomposition-tolerance-K", default=0.05, type=float)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    if args.boundary_width_cells < 1 or args.boundary_width_cells >= 32:
        raise SystemExit("--boundary-width-cells must be in [1, 31]")

    summary = analyze_oracle_components(
        data_root=args.data_root.expanduser().resolve(),
        source_version=args.source_version,
        decomposition_csv=args.residual_decomposition_csv.expanduser().resolve(),
        prediction_roots={
            "heldout_validation": args.validation_prediction_root.expanduser().resolve(),
            "heldout_test": args.test_prediction_root.expanduser().resolve(),
        },
        out_dir=args.out_dir.expanduser().resolve(),
        index_overrides={
            "heldout_validation": (
                args.validation_index.expanduser().resolve()
                if args.validation_index is not None
                else None
            ),
            "heldout_test": (
                args.test_index.expanduser().resolve()
                if args.test_index is not None
                else None
            ),
        },
        boundary_width_cells=int(args.boundary_width_cells),
        decomposition_tolerance_K=float(args.decomposition_tolerance_K),
    )
    print("Benchmark v2 oracle residual-component analysis complete")
    print(f"Samples: {summary['sample_count']}")
    print(f"Recommendation: {summary['recommendation']['choice']}")
    print(f"Output: {args.out_dir}")
    return 0


def analyze_oracle_components(
    *,
    data_root: Path,
    source_version: str,
    decomposition_csv: Path,
    prediction_roots: Mapping[str, Path],
    out_dir: Path,
    index_overrides: Mapping[str, Path | None],
    boundary_width_cells: int,
    decomposition_tolerance_K: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_root = (
        data_root
        / "derived/indices/full_50x200/source_superposition"
        / source_version
        / "family_split"
    )
    index_paths = {
        "heldout_validation": index_overrides["heldout_validation"] or index_root / "val_index.csv",
        "heldout_test": index_overrides["heldout_test"] or index_root / "test_index.csv",
    }
    protocol_rows = {
        protocol: read_csv(path)
        for protocol, path in index_paths.items()
    }
    expected_counts = {"heldout_validation": 1000, "heldout_test": 1000}
    for protocol, rows in protocol_rows.items():
        if len(rows) != expected_counts[protocol]:
            raise ValueError(f"{protocol} index expected {expected_counts[protocol]} rows, found {len(rows)}")
    decomposition_rows = load_decomposition_rows(decomposition_csv)
    audit_prediction_coverage(protocol_rows, prediction_roots)

    boundary_mask = make_boundary_mask(GRID_SHAPE, boundary_width_cells)
    sample_rows: list[dict[str, Any]] = []
    component_arrays: dict[str, dict[str, np.ndarray]] = {}
    decomposition_checks: list[dict[str, Any]] = []
    for protocol in ("heldout_validation", "heldout_test"):
        index_path = index_paths[protocol]
        for row in sorted(protocol_rows[protocol], key=lambda item: required_text(item, "sample_uid")):
            sample_uid = required_text(row, "sample_uid")
            family_uid = str(row.get("family_uid") or row.get("case_id") or "").strip()
            if not family_uid:
                raise ValueError(f"{sample_uid} lacks family_uid/case_id")
            source = load_map(
                resolve_data_path(required_text(row, "source_superposition_base_path"), data_root, index_path),
                f"{sample_uid} source-superposition base",
            )
            target = load_map(resolve_target_path(row, data_root, index_path), f"{sample_uid} HotSpot target")
            final_prediction, prediction_path = load_saved_final_prediction(
                prediction_roots[protocol],
                family_uid,
                sample_uid,
                source,
            )
            result = analyze_sample(
                source=source,
                target=target,
                final_prediction=final_prediction,
                boundary_mask=boundary_mask,
                coarse_size=CANONICAL_COARSE_SIZE,
            )
            workload_uid = normalized_workload_uid(row)
            decomposition_key = (family_uid, workload_uid)
            if decomposition_key not in decomposition_rows:
                raise ValueError(f"residual decomposition is missing {family_uid}/{workload_uid}")
            cross_check = cross_check_decomposition(result, decomposition_rows[decomposition_key])
            decomposition_checks.append(
                {"sample_uid": sample_uid, "family_uid": family_uid, "workload_uid": workload_uid, **cross_check}
            )
            if cross_check["max_abs_difference_K"] > decomposition_tolerance_K:
                raise ValueError(
                    f"baseline/decomposition mismatch for {sample_uid}: {cross_check}; "
                    f"tolerance={decomposition_tolerance_K} K"
                )
            sample_record = flatten_sample_result(
                result=result,
                protocol=protocol,
                row=row,
                prediction_path=prediction_path,
                decomposition_max_difference=cross_check["max_abs_difference_K"],
            )
            sample_rows.append(sample_record)
            if family_uid == "f044":
                component_arrays[sample_uid] = {
                    "source": source,
                    "target": target,
                    "true_low": result["components"]["true_low"],
                    "predicted_low": result["components"]["predicted_low"],
                    "low_error": result["components"]["predicted_low"] - result["components"]["true_low"],
                    "true_high": result["components"]["true_high"],
                    "predicted_high": result["components"]["predicted_high"],
                    "high_error": result["components"]["predicted_high"] - result["components"]["true_high"],
                }

    sample_rows.sort(key=lambda item: (str(item["protocol"]), str(item["family_uid"]), str(item["sample_uid"])))
    per_family_rows = aggregate_family_rows(sample_rows)
    aggregate_summary = build_aggregate_summary(sample_rows)
    f044_analysis = build_f044_analysis(sample_rows)
    recommendation = make_recommendation(aggregate_summary, f044_analysis)
    representatives = select_f044_representatives(sample_rows)

    write_csv(out_dir / "per_sample_oracle_components.csv", sample_rows)
    write_csv(out_dir / "per_family_oracle_components.csv", per_family_rows)
    write_f044_component_maps(
        out_dir / "representative_f044_components",
        representatives,
        component_arrays,
    )
    write_f044_waterfall(out_dir / "f044_oracle_waterfall.png", f044_analysis)
    write_oracle_family_plot(out_dir / "oracle_mae_by_family.png", per_family_rows)
    write_low_high_family_plot(out_dir / "low_vs_high_error_by_family.png", per_family_rows)

    summary = {
        "schema_version": "benchmark_v2_oracle_residual_components/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(sample_rows),
        "source_version": source_version,
        "canonical_decomposition": {
            "coarse_size": [CANONICAL_COARSE_SIZE, CANONICAL_COARSE_SIZE],
            "downsample": "nonoverlapping 8x8 area average from 64x64 to 8x8",
            "upsample": "deterministic bilinear resize to 64x64 with half-pixel coordinates",
            "low_frequency": "bilinear_upsample(area_average(centered)); then subtract spatial mean",
            "high_frequency": "centered - low_frequency",
            "identity": "low_frequency + high_frequency == centered within floating-point tolerance",
        },
        "reconstruction_formulas": {
            "baseline_final": "source + predicted_mean + predicted_low + predicted_high",
            "oracle_mean": "source + true_mean + predicted_low + predicted_high",
            "oracle_centered": "source + predicted_mean + true_low + true_high",
            "oracle_low_frequency": "source + predicted_mean + true_low + predicted_high",
            "oracle_high_frequency": "source + predicted_mean + predicted_low + true_high",
            "oracle_mean_and_low_frequency": "source + true_mean + true_low + predicted_high",
            "optimal_centered_scale": "source + true_mean + alpha*predicted_centered; alpha minimizes per-sample squared error",
            "full_oracle": "source + true_mean + true_low + true_high",
            "sign_note": (
                "Residual components are added to the source base. Subtracting true_low/predicted_high would invert "
                "the residual and is not an oracle replacement under target=source+residual."
            ),
        },
        "metrics": {
            "temperature_map": "grid-cell MAE/RMSE against HotSpot",
            "peak_temperature": "absolute difference between predicted and true map maxima",
            "boundary_region": f"{boundary_width_cells}-cell perimeter band",
            "interior_region": f"cells outside the {boundary_width_cells}-cell perimeter band",
            "improvement": "baseline_final MAE minus reconstruction MAE; positive is better",
        },
        "aggregates": aggregate_summary,
        "f044": f044_analysis,
        "f041": aggregate_summary["by_family"].get("f041"),
        "representative_f044_samples": representatives,
        "recommendation": recommendation,
        "decomposition_cross_check": {
            "rows_checked": len(decomposition_checks),
            "maximum_abs_difference_K": max(
                float(item["max_abs_difference_K"]) for item in decomposition_checks
            ),
            "tolerance_K": decomposition_tolerance_K,
        },
        "inputs": {
            "data_root": str(data_root),
            "indices": {protocol: portable_path(path, data_root) for protocol, path in index_paths.items()},
            "prediction_roots": {protocol: str(path) for protocol, path in prediction_roots.items()},
            "residual_decomposition_csv": str(decomposition_csv),
        },
        "oracle_scope": (
            "Every substituted true component uses held-out HotSpot targets and is an explicitly labeled oracle "
            "diagnostic. These results are upper bounds, not deployable inference metrics and not evidence of "
            "target-free prediction."
        ),
        "execution_contract": {
            "checkpoint_inference_performed": False,
            "hotspot_run_performed": False,
            "prediction_artifacts_modified": False,
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "oracle_residual_component_report.md", summary)
    return summary


def analyze_sample(
    *,
    source: np.ndarray,
    target: np.ndarray,
    final_prediction: np.ndarray,
    boundary_mask: np.ndarray,
    coarse_size: int = CANONICAL_COARSE_SIZE,
) -> dict[str, Any]:
    source = validated_map(source, "source")
    target = validated_map(target, "target")
    final_prediction = validated_map(final_prediction, "final prediction")
    true_residual = target - source
    predicted_residual = final_prediction - source
    true_mean = float(true_residual.mean())
    predicted_mean = float(predicted_residual.mean())
    true_centered = true_residual - true_mean
    predicted_centered = predicted_residual - predicted_mean
    true_low, true_high = decompose_centered(true_centered, coarse_size)
    predicted_low, predicted_high = decompose_centered(predicted_centered, coarse_size)
    alpha = optimal_centered_scale(predicted_centered, true_centered)
    reconstructions = reconstruct_oracles(
        source=source,
        true_mean=true_mean,
        predicted_mean=predicted_mean,
        true_low=true_low,
        true_high=true_high,
        predicted_low=predicted_low,
        predicted_high=predicted_high,
        optimal_alpha=alpha,
    )
    metrics = {
        name: reconstruction_metrics(prediction, target, boundary_mask)
        for name, prediction in reconstructions.items()
    }
    baseline_mae = metrics["baseline_final"]["temperature_mae_K"]
    for name in VARIANTS:
        metrics[name]["mae_improvement_vs_baseline_K"] = (
            baseline_mae - metrics[name]["temperature_mae_K"]
        )

    low_error = predicted_low - true_low
    high_error = predicted_high - true_high
    centered_error = predicted_centered - true_centered
    low_energy = float(np.mean(low_error * low_error))
    high_energy = float(np.mean(high_error * high_error))
    centered_energy = float(np.mean(centered_error * centered_error))
    cross_energy = float(2.0 * np.mean(low_error * high_error))
    component_sum = max(low_energy + high_energy, EPS)
    interior_mask = ~boundary_mask
    low_boundary_mse = float(np.mean(low_error[boundary_mask] ** 2))
    low_interior_mse = float(np.mean(low_error[interior_mask] ** 2))
    return {
        "source_temperature_mae_K": float(np.mean(np.abs(source - target))),
        "true_mean_K": true_mean,
        "predicted_mean_K": predicted_mean,
        "mean_error_K": predicted_mean - true_mean,
        "optimal_centered_alpha": alpha,
        "metrics": metrics,
        "component_energy": {
            "low_error_mse_K2": low_energy,
            "high_error_mse_K2": high_energy,
            "centered_error_mse_K2": centered_energy,
            "cross_term_K2": cross_energy,
            "low_fraction_of_low_plus_high": low_energy / component_sum,
            "high_fraction_of_low_plus_high": high_energy / component_sum,
            "low_fraction_of_total_centered_error": low_energy / max(centered_energy, EPS),
            "high_fraction_of_total_centered_error": high_energy / max(centered_energy, EPS),
            "cross_fraction_of_total_centered_error": cross_energy / max(centered_energy, EPS),
            "low_error_boundary_mse_K2": low_boundary_mse,
            "low_error_interior_mse_K2": low_interior_mse,
            "low_error_boundary_to_interior_mse_ratio": low_boundary_mse / max(low_interior_mse, EPS),
        },
        "components": {
            "true_residual": true_residual,
            "predicted_residual": predicted_residual,
            "true_centered": true_centered,
            "predicted_centered": predicted_centered,
            "true_low": true_low,
            "true_high": true_high,
            "predicted_low": predicted_low,
            "predicted_high": predicted_high,
        },
        "reconstructions": reconstructions,
    }


def decompose_centered(centered: np.ndarray, coarse_size: int = CANONICAL_COARSE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    array = validated_map(centered, "centered residual")
    if 64 % coarse_size:
        raise ValueError(f"coarse size {coarse_size} must divide 64 exactly")
    block = 64 // coarse_size
    coarse = array.reshape(coarse_size, block, coarse_size, block).mean(axis=(1, 3))
    low = bilinear_resize(coarse, GRID_SHAPE)
    low = low - float(low.mean())
    high = array - low
    if not np.allclose(low + high, array, atol=1.0e-12, rtol=0.0):
        raise AssertionError("low/high residual decomposition lost reconstruction identity")
    return low, high


def bilinear_resize(array: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(array, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"bilinear resize expects 2D input, got {source.shape}")
    out_h, out_w = output_shape
    in_h, in_w = source.shape
    y = (np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w - 0.5
    y0_raw = np.floor(y).astype(np.int64)
    x0_raw = np.floor(x).astype(np.int64)
    y1_raw = y0_raw + 1
    x1_raw = x0_raw + 1
    wy = y - y0_raw
    wx = x - x0_raw
    y0 = np.clip(y0_raw, 0, in_h - 1)
    y1 = np.clip(y1_raw, 0, in_h - 1)
    x0 = np.clip(x0_raw, 0, in_w - 1)
    x1 = np.clip(x1_raw, 0, in_w - 1)
    top = source[y0[:, None], x0[None, :]] * (1.0 - wx)[None, :] + source[y0[:, None], x1[None, :]] * wx[None, :]
    bottom = source[y1[:, None], x0[None, :]] * (1.0 - wx)[None, :] + source[y1[:, None], x1[None, :]] * wx[None, :]
    return top * (1.0 - wy)[:, None] + bottom * wy[:, None]


def optimal_centered_scale(predicted_centered: np.ndarray, true_centered: np.ndarray) -> float:
    denominator = float(np.sum(predicted_centered * predicted_centered))
    if denominator <= EPS:
        return 0.0
    return float(np.sum(predicted_centered * true_centered) / denominator)


def reconstruct_oracles(
    *,
    source: np.ndarray,
    true_mean: float,
    predicted_mean: float,
    true_low: np.ndarray,
    true_high: np.ndarray,
    predicted_low: np.ndarray,
    predicted_high: np.ndarray,
    optimal_alpha: float,
) -> dict[str, np.ndarray]:
    predicted_centered = predicted_low + predicted_high
    true_centered = true_low + true_high
    return {
        "baseline_final": source + predicted_mean + predicted_centered,
        "oracle_mean": source + true_mean + predicted_centered,
        "oracle_centered": source + predicted_mean + true_centered,
        "oracle_low_frequency": source + predicted_mean + true_low + predicted_high,
        "oracle_high_frequency": source + predicted_mean + predicted_low + true_high,
        "oracle_mean_and_low_frequency": source + true_mean + true_low + predicted_high,
        "optimal_centered_scale": source + true_mean + optimal_alpha * predicted_centered,
        "full_oracle": source + true_mean + true_centered,
    }


def reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    boundary_mask: np.ndarray,
) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "temperature_mae_K": float(absolute.mean()),
        "temperature_rmse_K": float(np.sqrt(np.mean(error * error))),
        "peak_temperature_mae_K": float(abs(float(prediction.max()) - float(target.max()))),
        "boundary_region_mae_K": float(absolute[boundary_mask].mean()),
        "interior_region_mae_K": float(absolute[~boundary_mask].mean()),
    }


def flatten_sample_result(
    *,
    result: Mapping[str, Any],
    protocol: str,
    row: Mapping[str, str],
    prediction_path: Path,
    decomposition_max_difference: float,
) -> dict[str, Any]:
    family = str(row.get("family_uid") or row.get("case_id"))
    record: dict[str, Any] = {
        "protocol": protocol,
        "split": str(row.get("split", "")),
        "sample_uid": required_text(row, "sample_uid"),
        "family_uid": family,
        "workload_uid": normalized_workload_uid(row),
        "workload_regime": str(
            row.get("broad_stratum")
            or row.get("power_regime")
            or row.get("workload_stratum")
            or row.get("workload_cell")
            or "unknown"
        ),
        "power_regime": str(row.get("power_regime", "")),
        "topology_regime": str(row.get("topology_regime", "")),
        "true_residual_mean_K": float(result["true_mean_K"]),
        "predicted_residual_mean_K": float(result["predicted_mean_K"]),
        "source_temperature_mae_K": float(result["source_temperature_mae_K"]),
        "residual_mean_error_K": float(result["mean_error_K"]),
        "absolute_residual_mean_error_K": abs(float(result["mean_error_K"])),
        "optimal_centered_alpha": float(result["optimal_centered_alpha"]),
        "saved_prediction_path": str(prediction_path),
        "decomposition_max_abs_difference_K": decomposition_max_difference,
    }
    for name, value in result["component_energy"].items():
        record[name] = float(value)
    for variant in VARIANTS:
        for metric, value in result["metrics"][variant].items():
            record[f"{variant}_{metric}"] = float(value)
    return record


def cross_check_decomposition(
    result: Mapping[str, Any],
    row: Mapping[str, str],
) -> dict[str, Any]:
    baseline = result["metrics"]["baseline_final"]
    comparisons = {
        "source_superposition_mae_K": result["source_temperature_mae_K"],
        "final_cnn_mae_K": baseline["temperature_mae_K"],
        "true_residual_mean_K": result["true_mean_K"],
        "predicted_scalar_mean_correction_K": result["predicted_mean_K"],
        "centered_spatial_mae_K": float(
            np.mean(
                np.abs(
                    result["components"]["predicted_centered"]
                    - result["components"]["true_centered"]
                )
            )
        ),
    }
    differences = {
        key: abs(float(value) - float(row[key]))
        for key, value in comparisons.items()
        if key in row and str(row[key]).strip()
    }
    return {
        "compared_field_count": len(differences),
        "max_abs_difference_K": max(differences.values(), default=0.0),
        "field_differences_K": differences,
    }


def aggregate_family_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(str(row["protocol"]), str(row["family_uid"]))].append(row)
    output: list[dict[str, Any]] = []
    for (protocol, family), rows in sorted(grouped.items()):
        for variant in VARIANTS:
            aggregate = aggregate_variant(rows, variant)
            output.append(
                {
                    "protocol": protocol,
                    "family_uid": family,
                    "sample_count": len(rows),
                    "reconstruction": variant,
                    **aggregate,
                }
            )
    return output


def aggregate_variant(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in METRICS:
        column = f"{variant}_{metric}"
        output[metric] = float(np.mean([float(row[column]) for row in rows]))
    output["mae_improvement_vs_baseline_K"] = float(
        np.mean([float(row[f"{variant}_mae_improvement_vs_baseline_K"]) for row in rows])
    )
    return output


def build_aggregate_summary(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_protocol: dict[str, Any] = {}
    for protocol in ("heldout_validation", "heldout_test"):
        rows = [row for row in sample_rows if row["protocol"] == protocol]
        by_protocol[protocol] = {variant: aggregate_variant(rows, variant) for variant in VARIANTS}
    by_family: dict[str, Any] = {}
    for family in sorted({str(row["family_uid"]) for row in sample_rows}):
        rows = [row for row in sample_rows if row["family_uid"] == family]
        by_family[family] = {
            "sample_count": len(rows),
            "protocol": str(rows[0]["protocol"]),
            "variants": {variant: aggregate_variant(rows, variant) for variant in VARIANTS},
        }
    return {
        "overall": {
            variant: aggregate_variant(sample_rows, variant)
            for variant in VARIANTS
        },
        "by_protocol": by_protocol,
        "by_family": by_family,
        "by_workload_regime": aggregate_by_field(sample_rows, "workload_regime"),
        "by_power_regime": aggregate_by_field(sample_rows, "power_regime"),
    }


def aggregate_by_field(
    sample_rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row.get(field, "") or "unknown")].append(row)
    return {
        value: {
            "sample_count": len(rows),
            "variants": {variant: aggregate_variant(rows, variant) for variant in VARIANTS},
        }
        for value, rows in sorted(grouped.items())
    }


def build_f044_analysis(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in sample_rows if row["family_uid"] == "f044"]
    variants = {variant: aggregate_variant(rows, variant) for variant in VARIANTS}
    baseline = variants["baseline_final"]["temperature_mae_K"]
    removable = {
        "mean_only_K": baseline - variants["oracle_mean"]["temperature_mae_K"],
        "low_frequency_only_K": baseline - variants["oracle_low_frequency"]["temperature_mae_K"],
        "high_frequency_only_K": baseline - variants["oracle_high_frequency"]["temperature_mae_K"],
        "mean_and_low_frequency_K": baseline
        - variants["oracle_mean_and_low_frequency"]["temperature_mae_K"],
    }
    low_fraction = float(np.mean([float(row["low_fraction_of_low_plus_high"]) for row in rows]))
    high_fraction = float(np.mean([float(row["high_fraction_of_low_plus_high"]) for row in rows]))
    boundary_ratio = float(
        np.mean([float(row["low_error_boundary_to_interior_mse_ratio"]) for row in rows])
    )
    return {
        "sample_count": len(rows),
        "variants": variants,
        "removable_mae": removable,
        "low_error_energy_fraction_of_component_sum": low_fraction,
        "high_error_energy_fraction_of_component_sum": high_fraction,
        "low_error_boundary_to_interior_mse_ratio": boundary_ratio,
        "low_frequency_error_concentrated_near_boundaries": boundary_ratio > 1.2,
        "energy_note": (
            "Low/high components are complementary but not orthogonal after bilinear upsampling. Reported low/high "
            "fractions use low_energy/(low_energy+high_energy); cross-term diagnostics remain in the per-sample CSV."
        ),
    }


def make_recommendation(
    aggregate_summary: Mapping[str, Any],
    f044: Mapping[str, Any],
) -> dict[str, Any]:
    overall = aggregate_summary["overall"]
    baseline = overall["baseline_final"]["temperature_mae_K"]
    gains = {
        "improve scalar mean head only": baseline - overall["oracle_mean"]["temperature_mae_K"],
        "add coarse package-scale residual head": baseline
        - overall["oracle_low_frequency"]["temperature_mae_K"],
        "improve fine residual head": baseline
        - overall["oracle_high_frequency"]["temperature_mae_K"],
        "combine mean and coarse heads": baseline
        - overall["oracle_mean_and_low_frequency"]["temperature_mae_K"],
    }
    best_single = max(
        gains["improve scalar mean head only"],
        gains["add coarse package-scale residual head"],
        gains["improve fine residual head"],
    )
    combined = gains["combine mean and coarse heads"]
    material_threshold = max(0.05, 0.05 * baseline)
    if max(gains.values()) < material_threshold:
        choice = "no architecture change justified"
    elif (
        combined >= 1.10 * max(best_single, EPS)
        and gains["improve scalar mean head only"] > 0.05
        and gains["add coarse package-scale residual head"] > 0.05
    ):
        choice = "combine mean and coarse heads"
    else:
        choice = max(
            (
                "improve scalar mean head only",
                "add coarse package-scale residual head",
                "improve fine residual head",
            ),
            key=lambda name: gains[name],
        )
    return {
        "choice": choice,
        "overall_oracle_mae_gains_K": gains,
        "f044_removable_mae_K": dict(f044["removable_mae"]),
        "rule": (
            "Choose no change below max(0.05 K, 5% baseline); choose combined when it exceeds the best single "
            "component by 10% and both mean/coarse gains exceed 0.05 K; otherwise choose the largest single gain."
        ),
        "oracle_warning": "Recommendation ranks where capacity could help; it does not make oracle targets deployable.",
    }


def select_f044_representatives(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    rows = sorted(
        (row for row in sample_rows if row["family_uid"] == "f044"),
        key=lambda row: (
            float(row["baseline_final_temperature_mae_K"]),
            str(row["sample_uid"]),
        ),
    )
    return {
        "easy": str(rows[0]["sample_uid"]),
        "median": str(rows[(len(rows) - 1) // 2]["sample_uid"]),
        "worst": str(rows[-1]["sample_uid"]),
    }


def write_f044_component_maps(
    out_dir: Path,
    representatives: Mapping[str, str],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ("true_low", "predicted_low", "low_error", "true_high", "predicted_high", "high_error")
    for category, sample_uid in representatives.items():
        sample = arrays[sample_uid]
        low_limit = max(float(np.max(np.abs(sample[name]))) for name in ("true_low", "predicted_low", "low_error"))
        high_limit = max(float(np.max(np.abs(sample[name]))) for name in ("true_high", "predicted_high", "high_error"))
        tile = 190
        image = Image.new("RGB", (tile * len(columns), tile + 75), "white")
        draw = ImageDraw.Draw(image)
        draw.text((15, 15), f"{category}: {sample_uid}", fill="black")
        for index, name in enumerate(columns):
            draw.text((index * tile + 8, 45), name.replace("_", " "), fill="black")
            limit = low_limit if "low" in name else high_limit
            heatmap = diverging_heatmap(sample[name], limit)
            heatmap = heatmap.resize((tile - 18, tile - 18), resample=Image.Resampling.NEAREST)
            image.paste(heatmap, (index * tile + 9, 68))
        image.save(out_dir / f"{category}_{sample_uid}.png")


def diverging_heatmap(array: np.ndarray, limit: float) -> Any:
    from PIL import Image

    normalized = np.clip(0.5 + 0.5 * array / max(limit, EPS), 0.0, 1.0)
    red = np.where(normalized >= 0.5, 255.0, 510.0 * normalized)
    blue = np.where(normalized <= 0.5, 255.0, 510.0 * (1.0 - normalized))
    green = 255.0 - 1.5 * np.abs(normalized - 0.5) * 255.0
    return Image.fromarray(np.stack((red, green, blue), axis=-1).astype(np.uint8), mode="RGB")


def write_f044_waterfall(path: Path, f044: Mapping[str, Any]) -> None:
    labels = (
        "baseline_final",
        "oracle_mean",
        "oracle_low_frequency",
        "oracle_high_frequency",
        "oracle_mean_and_low_frequency",
        "full_oracle",
    )
    values = [float(f044["variants"][name]["temperature_mae_K"]) for name in labels]
    write_bar_plot(path, "f044 oracle residual-component MAE", labels, values, "MAE (K)")


def write_oracle_family_plot(path: Path, family_rows: Sequence[Mapping[str, Any]]) -> None:
    variants = ("baseline_final", "oracle_mean", "oracle_low_frequency", "oracle_high_frequency", "oracle_mean_and_low_frequency")
    families = sorted({str(row["family_uid"]) for row in family_rows})
    lookup = {
        (str(row["family_uid"]), str(row["reconstruction"])): float(row["temperature_mae_K"])
        for row in family_rows
    }
    write_grouped_family_plot(
        path,
        "Oracle MAE by held-out family",
        families,
        variants,
        lookup,
        "MAE (K)",
    )


def write_low_high_family_plot(path: Path, family_rows: Sequence[Mapping[str, Any]]) -> None:
    families = sorted({str(row["family_uid"]) for row in family_rows})
    lookup_rows = {
        (str(row["family_uid"]), str(row["reconstruction"])): row
        for row in family_rows
    }
    lookup: dict[tuple[str, str], float] = {}
    variants = ("low_frequency_removable", "high_frequency_removable")
    for family in families:
        baseline = float(lookup_rows[(family, "baseline_final")]["temperature_mae_K"])
        lookup[(family, variants[0])] = baseline - float(
            lookup_rows[(family, "oracle_low_frequency")]["temperature_mae_K"]
        )
        lookup[(family, variants[1])] = baseline - float(
            lookup_rows[(family, "oracle_high_frequency")]["temperature_mae_K"]
        )
    write_grouped_family_plot(
        path,
        "Removable low- versus high-frequency MAE by family",
        families,
        variants,
        lookup,
        "MAE improvement (K)",
    )


def write_bar_plot(
    path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1150, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), title, fill="black")
    left, top, right, bottom = 80, 70, width - 35, height - 120
    maximum = max(max(values), EPS) * 1.08
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    width_per = (right - left) / len(labels)
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = left + (index + 0.15) * width_per
        x1 = left + (index + 0.85) * width_per
        y = bottom - value / maximum * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill="#386cb0" if index == 0 else "#7fc97f")
        draw.text((x0, y - 18), f"{value:.3f}", fill="black")
        draw.text((x0, bottom + 12), label.replace("_", " ")[:24], fill="black")
    draw.text((8, top), y_label, fill="black")
    image.save(path)


def write_grouped_family_plot(
    path: Path,
    title: str,
    families: Sequence[str],
    variants: Sequence[str],
    lookup: Mapping[tuple[str, str], float],
    y_label: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1450, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), title, fill="black")
    left, top, right, bottom = 80, 70, width - 35, height - 125
    values = [float(lookup[(family, variant)]) for family in families for variant in variants]
    minimum = min(0.0, min(values))
    maximum = max(max(values), EPS)
    padding = 0.08 * max(maximum - minimum, EPS)
    minimum -= padding
    maximum += padding
    zero_y = bottom - (0.0 - minimum) / (maximum - minimum) * (bottom - top)
    draw.line((left, zero_y, right, zero_y), fill="#333333", width=2)
    palette = ("#386cb0", "#7fc97f", "#fdc086", "#beaed4", "#ef3b2c")
    family_width = (right - left) / len(families)
    bar_width = family_width / (len(variants) + 1)
    for family_index, family in enumerate(families):
        for variant_index, variant in enumerate(variants):
            value = float(lookup[(family, variant)])
            x0 = left + family_index * family_width + (variant_index + 0.3) * bar_width
            x1 = x0 + 0.8 * bar_width
            y = bottom - (value - minimum) / (maximum - minimum) * (bottom - top)
            draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill=palette[variant_index % len(palette)])
        draw.text((left + family_index * family_width + 10, bottom + 15), family, fill="black")
    for index, variant in enumerate(variants):
        x = left + index * 250
        draw.rectangle((x, height - 55, x + 14, height - 41), fill=palette[index % len(palette)])
        draw.text((x + 20, height - 57), variant.replace("_", " ")[:30], fill="black")
    draw.text((8, top), y_label, fill="black")
    image.save(path)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    overall = summary["aggregates"]["overall"]
    f044 = summary["f044"]
    lines = [
        "# Benchmark v2 Oracle Residual-Component Analysis",
        "",
        "This is an explicitly target-leaking oracle diagnostic. It estimates component-wise headroom and is not a deployable result.",
        "",
        "## Overall Held-Out Headroom",
        "",
        "| Reconstruction | MAE | Improvement |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        metrics = overall[variant]
        lines.append(
            f"| {variant} | {metrics['temperature_mae_K']:.4f} K | "
            f"{metrics['mae_improvement_vs_baseline_K']:.4f} K |"
        )
    lines.extend(
        [
            "",
            "## f044",
            "",
            f"- Mean-only removable MAE: {f044['removable_mae']['mean_only_K']:.4f} K",
            f"- Low-frequency-only removable MAE: {f044['removable_mae']['low_frequency_only_K']:.4f} K",
            f"- High-frequency-only removable MAE: {f044['removable_mae']['high_frequency_only_K']:.4f} K",
            f"- Mean plus low-frequency removable MAE: {f044['removable_mae']['mean_and_low_frequency_K']:.4f} K",
            f"- Low/high component error-energy fractions: {f044['low_error_energy_fraction_of_component_sum']:.3f} / "
            f"{f044['high_error_energy_fraction_of_component_sum']:.3f}",
            f"- Low-frequency boundary/interior MSE ratio: {f044['low_error_boundary_to_interior_mse_ratio']:.3f}",
            "",
            "## Recommendation",
            "",
            f"**{summary['recommendation']['choice']}**",
            "",
            summary["recommendation"]["rule"],
            "",
            "## Methodological Note",
            "",
            summary["oracle_scope"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_boundary_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:width] = True
    mask[-width:] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def load_decomposition_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        family = str(row.get("family_uid") or row.get("case_id") or "").strip()
        workload = normalized_workload_uid(row)
        key = (family, workload)
        if key in output:
            raise ValueError(f"duplicate residual-decomposition row: {key}")
        output[key] = row
    if len(output) != 2000:
        raise ValueError(f"residual-decomposition CSV expected 2000 unique rows, found {len(output)}")
    return output


def audit_prediction_coverage(
    protocol_rows: Mapping[str, Sequence[Mapping[str, str]]],
    prediction_roots: Mapping[str, Path],
) -> None:
    failures: dict[str, list[str]] = defaultdict(list)
    for protocol, rows in protocol_rows.items():
        for row in rows:
            family = str(row.get("family_uid") or row.get("case_id") or "").strip()
            uid = required_text(row, "sample_uid")
            try:
                find_prediction_file(prediction_roots[protocol], family, uid)
            except FileNotFoundError:
                failures[protocol].append(uid)
    if failures:
        detail = "; ".join(
            f"{protocol}: missing {len(uids)}, first={uids[:10]}, root={prediction_roots[protocol]}"
            for protocol, uids in sorted(failures.items())
        )
        raise FileNotFoundError(
            f"saved prediction coverage is incomplete: {detail}. This offline analyzer never runs checkpoint inference."
        )


def load_saved_final_prediction(
    root: Path,
    family: str,
    sample_uid: str,
    source: np.ndarray,
) -> tuple[np.ndarray, Path]:
    path, kind = find_prediction_file(root, family, sample_uid)
    array = load_map(path, f"{sample_uid} saved {kind}")
    final = array if kind == "final_temperature" else source + array
    return validated_map(final, f"{sample_uid} reconstructed final prediction"), path.resolve()


def find_prediction_file(root: Path, family: str, sample_uid: str) -> tuple[Path, str]:
    roots = (root, root / "predictions", root / "predicted_residuals")
    names = (
        (f"{sample_uid}_tpred.npy", "final_temperature"),
        (f"{sample_uid}_prediction.npy", "final_temperature"),
        (f"{sample_uid}_temperature_pred.npy", "final_temperature"),
        (f"{sample_uid}_residual_pred.npy", "predicted_residual"),
        (f"{sample_uid}_predicted_residual.npy", "predicted_residual"),
    )
    candidates: list[tuple[Path, str]] = []
    for candidate_root in roots:
        for name, kind in names:
            candidates.extend(((candidate_root / family / name, kind), (candidate_root / name, kind)))
    for path, kind in candidates:
        if path.is_file():
            return path, kind
    raise FileNotFoundError(f"no cached prediction for {family}/{sample_uid} under {root}")


def resolve_target_path(row: Mapping[str, str], data_root: Path, index_path: Path) -> Path:
    for key in ("y_path", "target_path", "temperature_path", "final_temperature"):
        value = str(row.get(key, "")).strip()
        if value:
            return resolve_data_path(value, data_root, index_path)
    raise ValueError(
        f"{row.get('sample_uid')} has no target path; available columns={sorted(row)}"
    )


def resolve_data_path(logical: str, data_root: Path, index_path: Path) -> Path:
    path = Path(logical).expanduser()
    candidates = [path] if path.is_absolute() else [data_root / path, REPO_ROOT / path, index_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve logical path {logical!r} against data_root={data_root}; "
        f"candidates={[str(candidate) for candidate in candidates]}"
    )


def normalized_workload_uid(row: Mapping[str, str]) -> str:
    value = str(row.get("workload_uid", "")).strip()
    if value:
        return value
    sample_uid = required_text(row, "sample_uid")
    family = str(row.get("family_uid") or row.get("case_id") or "").strip()
    prefix = f"{family}_"
    if family and sample_uid.startswith(prefix):
        return sample_uid[len(prefix) :]
    raise ValueError(f"cannot derive workload_uid for {sample_uid}")


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
        raise ValueError(f"required field {key!r} is blank; available={sorted(row)}")
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
