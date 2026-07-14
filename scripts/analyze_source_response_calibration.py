#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponseDataset,
    SourceResponseNormalizationStats,
    normalize_source_input,
    source_response_collate,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, count_parameters, predict_source_rise  # noqa: E402
from scripts.run_superposition_diagnostic import chiplet_metrics, field_metrics, load_json  # noqa: E402


EPSILON = 1.0e-12


@dataclass
class PackagePrediction:
    sample_uid: str
    case_id: str
    ambient_K: float
    pred: np.ndarray
    true: np.ndarray
    oracle_source_sum: np.ndarray
    layout_path: str
    num_sources: int
    total_power_W: float

    @property
    def delta_pred(self) -> np.ndarray:
        return self.pred - self.ambient_K

    @property
    def delta_true(self) -> np.ndarray:
        return self.true - self.ambient_K


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze source-response package calibration oracles.")
    parser.add_argument("--checkpoint", required=False, type=Path)
    parser.add_argument("--val-source-index", required=False, type=Path)
    parser.add_argument("--test-source-index", required=False, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--save-calibrated-predictions", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Regenerate calibration_report.md from existing CSV/JSON outputs.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        regenerate_report(out_dir)
        return 0
    if args.checkpoint is None or args.val_source_index is None or args.test_source_index is None:
        raise SystemExit("--checkpoint, --val-source-index, and --test-source-index are required unless --summary-only is used")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(checkpoint["normalization"])
    model = build_source_response_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    start = time.perf_counter()
    val_packages = reconstruct_packages(model, stats, args.val_source_index, args.batch_size, args.num_workers, device)
    test_packages = reconstruct_packages(model, stats, args.test_source_index, args.batch_size, args.num_workers, device)
    runtime_s = time.perf_counter() - start

    val_global = fit_all_global_calibrations(val_packages)
    val_case = fit_case_calibrations(val_packages)
    analysis = analyze_test_packages(test_packages, val_global, val_case)
    analysis["metadata"] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "val_source_index": str(args.val_source_index.resolve()),
        "test_source_index": str(args.test_source_index.resolve()),
        "parameter_count": count_parameters(model),
        "num_val_packages": len(val_packages),
        "num_test_packages": len(test_packages),
        "runtime_s": runtime_s,
        "calibration_warning": "Per-sample modes are oracle diagnostics fitted with each test sample target; validation-fitted modes use validation packages only.",
    }
    analysis["validation_fitted_parameters"] = {
        "global": val_global,
        "per_case": val_case,
    }
    write_outputs(out_dir, analysis)
    if args.save_calibrated_predictions:
        save_selected_predictions(out_dir, test_packages, analysis["sample_records"])
    write_report(out_dir, analysis)
    print("Source-response calibration analysis complete")
    raw = analysis["overall_metrics"]["raw"]["mae_K"]
    print(f"Raw package MAE: {raw:.4f} K")
    for mode in ("per_sample_mean_bias_oracle", "per_sample_gain_oracle", "per_sample_gain_offset_oracle", "true_mean_pred_centered_oracle"):
        print(f"{mode}: MAE={analysis['overall_metrics'][mode]['mae_K']:.4f} K")
    return 0


@torch.no_grad()
def reconstruct_packages(
    model: torch.nn.Module,
    stats: SourceResponseNormalizationStats,
    source_index: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[PackagePrediction]:
    dataset = SourceResponseDataset(source_index, power_floor_W=stats.power_floor_W)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=source_response_collate,
    )
    groups: dict[str, dict[str, Any]] = {}
    for batch in loader:
        x = normalize_source_input(batch["x"].to(device), stats)
        source_power = batch["source_power_W"].to(device)
        pred_unit = unnormalize_source_prediction(model(x), stats)
        pred_rise = predict_source_rise(pred_unit, source_power).detach().cpu().numpy()
        true_rise = batch["target_rise"].detach().cpu().numpy()
        full = batch["full_temperature"].detach().cpu().numpy()
        ambient = batch["ambient_K"].detach().cpu().numpy()
        for i, meta in enumerate(batch["metadata"]):
            uid = str(meta["original_sample_uid"])
            group = groups.setdefault(
                uid,
                {
                    "case_id": str(meta["case_id"]),
                    "ambient_K": float(ambient[i]),
                    "pred_sum": np.zeros_like(pred_rise[i], dtype=np.float64),
                    "true_source_sum": np.zeros_like(true_rise[i], dtype=np.float64),
                    "full_temperature": full[i].astype(np.float64),
                    "layout_path": str(meta["layout_path"]),
                    "num_sources": 0,
                    "num_chiplets": int(float(meta["num_chiplets"])),
                    "total_power_W": 0.0,
                },
            )
            group["pred_sum"] += pred_rise[i]
            group["true_source_sum"] += true_rise[i]
            group["num_sources"] += 1
            group["total_power_W"] += float(meta["source_power_W"])
    packages: list[PackagePrediction] = []
    for uid, group in sorted(groups.items()):
        if int(group["num_sources"]) != int(group["num_chiplets"]):
            continue
        ambient_K = float(group["ambient_K"])
        packages.append(
            PackagePrediction(
                sample_uid=uid,
                case_id=str(group["case_id"]),
                ambient_K=ambient_K,
                pred=ambient_K + group["pred_sum"],
                true=group["full_temperature"],
                oracle_source_sum=ambient_K + group["true_source_sum"],
                layout_path=str(group["layout_path"]),
                num_sources=int(group["num_sources"]),
                total_power_W=float(group["total_power_W"]),
            )
        )
    return packages


def fit_all_global_calibrations(packages: list[PackagePrediction]) -> dict[str, Any]:
    return {
        "offset": fit_offset(packages),
        "gain": fit_gain(packages),
        "gain_offset": fit_gain_offset(packages),
    }


def fit_case_calibrations(packages: list[PackagePrediction]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[PackagePrediction]] = defaultdict(list)
    for package in packages:
        grouped[package.case_id].append(package)
    return {case: fit_all_global_calibrations(items) for case, items in sorted(grouped.items())}


def fit_offset(packages: list[PackagePrediction]) -> dict[str, Any]:
    residuals = [package.true - package.pred for package in packages]
    if not residuals:
        return {"b_K": 0.0, "fallback": "empty"}
    return {"b_K": float(np.mean(np.concatenate([item.reshape(-1) for item in residuals]))), "fallback": None}


def fit_gain(packages: list[PackagePrediction]) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    for package in packages:
        dp = package.delta_pred.reshape(-1)
        dt = package.delta_true.reshape(-1)
        numerator += float(np.dot(dp, dt))
        denominator += float(np.dot(dp, dp))
    if abs(denominator) < EPSILON:
        return {"a": 1.0, "fallback": "zero_denominator"}
    return {"a": float(numerator / denominator), "fallback": None}


def fit_gain_offset(packages: list[PackagePrediction]) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for package in packages:
        xs.append(package.delta_pred.reshape(-1))
        ys.append(package.delta_true.reshape(-1))
    if not xs:
        return {"a": 1.0, "b_K": 0.0, "fallback": "empty"}
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if float(np.var(x)) < EPSILON:
        return {"a": 0.0, "b_K": float(np.mean(y)), "fallback": "constant_delta_pred"}
    design = np.stack([x, np.ones_like(x)], axis=1)
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"a": float(params[0]), "b_K": float(params[1]), "fallback": None}


def fit_offset_sample(package: PackagePrediction) -> dict[str, Any]:
    return {"b_K": float(np.mean(package.true - package.pred)), "fallback": None}


def fit_gain_sample(package: PackagePrediction) -> dict[str, Any]:
    return fit_gain([package])


def fit_gain_offset_sample(package: PackagePrediction) -> dict[str, Any]:
    return fit_gain_offset([package])


def apply_offset(package: PackagePrediction, params: dict[str, Any]) -> np.ndarray:
    return package.pred + float(params.get("b_K", 0.0))


def apply_gain(package: PackagePrediction, params: dict[str, Any]) -> np.ndarray:
    return package.ambient_K + float(params.get("a", 1.0)) * package.delta_pred


def apply_gain_offset(package: PackagePrediction, params: dict[str, Any]) -> np.ndarray:
    return package.ambient_K + float(params.get("a", 1.0)) * package.delta_pred + float(params.get("b_K", 0.0))


def true_mean_pred_centered(package: PackagePrediction) -> np.ndarray:
    delta_pred = package.delta_pred
    delta_true = package.delta_true
    return package.ambient_K + float(np.mean(delta_true)) + (delta_pred - float(np.mean(delta_pred)))


def pred_mean_true_centered(package: PackagePrediction) -> np.ndarray:
    delta_pred = package.delta_pred
    delta_true = package.delta_true
    return package.ambient_K + float(np.mean(delta_pred)) + (delta_true - float(np.mean(delta_true)))


def analyze_test_packages(
    packages: list[PackagePrediction],
    global_params: dict[str, Any],
    case_params: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mode_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parameter_records: list[dict[str, Any]] = []
    for package in packages:
        sample_params = {
            "offset": fit_offset_sample(package),
            "gain": fit_gain_sample(package),
            "gain_offset": fit_gain_offset_sample(package),
        }
        case_fit = case_params.get(package.case_id, global_params)
        modes: dict[str, np.ndarray] = {
            "raw": package.pred,
            "per_sample_mean_bias_oracle": apply_offset(package, sample_params["offset"]),
            "global_val_mean_bias": apply_offset(package, global_params["offset"]),
            "per_sample_gain_oracle": apply_gain(package, sample_params["gain"]),
            "global_val_gain": apply_gain(package, global_params["gain"]),
            "per_case_val_gain": apply_gain(package, case_fit["gain"]),
            "per_sample_gain_offset_oracle": apply_gain_offset(package, sample_params["gain_offset"]),
            "global_val_gain_offset": apply_gain_offset(package, global_params["gain_offset"]),
            "per_case_val_gain_offset": apply_gain_offset(package, case_fit["gain_offset"]),
            "true_mean_pred_centered_oracle": true_mean_pred_centered(package),
            "pred_mean_true_centered_oracle": pred_mean_true_centered(package),
            "oracle_source_sum": package.oracle_source_sum,
        }
        raw_mae = None
        for mode, pred in modes.items():
            record = metrics_for_prediction(package, mode, pred)
            if mode == "raw":
                raw_mae = record["mae_K"]
            record["raw_mae_K"] = raw_mae
            record["mae_improvement_K"] = None if raw_mae is None else raw_mae - record["mae_K"]
            record["fraction_removed"] = None if raw_mae is None or raw_mae <= 0.0 else (raw_mae - record["mae_K"]) / raw_mae
            mode_records[mode].append(record)
        parameter_records.append(
            {
                "sample_uid": package.sample_uid,
                "case_id": package.case_id,
                "num_sources": package.num_sources,
                "total_power_W": package.total_power_W,
                "oracle_mean_bias_b_K": sample_params["offset"]["b_K"],
                "oracle_gain_a": sample_params["gain"]["a"],
                "oracle_gain_fallback": sample_params["gain"]["fallback"],
                "oracle_gain_offset_a": sample_params["gain_offset"]["a"],
                "oracle_gain_offset_b_K": sample_params["gain_offset"]["b_K"],
                "oracle_gain_offset_fallback": sample_params["gain_offset"]["fallback"],
                "global_val_gain_a": global_params["gain"]["a"],
                "global_val_gain_offset_a": global_params["gain_offset"]["a"],
                "global_val_gain_offset_b_K": global_params["gain_offset"]["b_K"],
                "per_case_val_gain_a": case_fit["gain"]["a"],
                "per_case_val_gain_offset_a": case_fit["gain_offset"]["a"],
                "per_case_val_gain_offset_b_K": case_fit["gain_offset"]["b_K"],
            }
        )
    overall = {mode: summarize_records(records) for mode, records in sorted(mode_records.items())}
    by_case = summarize_by_case(mode_records)
    parameter_case = parameters_by_case(parameter_records)
    return {
        "overall_metrics": overall,
        "metrics_by_case": by_case,
        "sample_records": [record for mode in sorted(mode_records) for record in mode_records[mode]],
        "parameter_records": parameter_records,
        "parameter_case_records": parameter_case,
        "parameter_diagnostics": parameter_diagnostics(parameter_records),
        "decision_interpretation": decision_interpretation(overall),
    }


def metrics_for_prediction(package: PackagePrediction, mode: str, pred: np.ndarray) -> dict[str, Any]:
    metrics = field_metrics(pred, package.true)
    delta_pred = pred - package.ambient_K
    delta_true = package.delta_true
    mu_pred = float(np.mean(delta_pred))
    mu_true = float(np.mean(delta_true))
    centered_pred = delta_pred - mu_pred
    centered_true = delta_true - mu_true
    centered_error = centered_pred - centered_true
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    true_hotspot = np.unravel_index(int(np.argmax(package.true)), package.true.shape)
    layout_path = Path(package.layout_path)
    resolved_layout = layout_path if layout_path.is_absolute() else REPO_ROOT / layout_path
    if resolved_layout.exists():
        layout = load_json(resolved_layout)
        chip = chiplet_metrics(pred, package.true, layout, package.true.shape)
    else:
        chip = {
            "chiplet_mean_temperature_mae_K": None,
            "chiplet_peak_temperature_mae_K": None,
            "inter_chiplet_delta_T_mae_K": None,
        }
    return {
        "mode": mode,
        "sample_uid": package.sample_uid,
        "case_id": package.case_id,
        "num_sources": package.num_sources,
        "total_power_W": package.total_power_W,
        "mae_K": metrics["mae_K"],
        "rmse_K": metrics["rmse_K"],
        "max_abs_error_K": metrics["max_abs_error_K"],
        "mean_signed_error_K": metrics["mean_signed_error_K"],
        "mean_rise_abs_error_K": abs(mu_pred - mu_true),
        "centered_field_mae_K": float(np.mean(np.abs(centered_error))),
        "centered_field_rmse_K": float(np.sqrt(np.mean(centered_error * centered_error))),
        "chiplet_mean_temperature_mae_K": chip["chiplet_mean_temperature_mae_K"],
        "chiplet_peak_temperature_mae_K": chip["chiplet_peak_temperature_mae_K"],
        "inter_chiplet_delta_T_mae_K": chip["inter_chiplet_delta_T_mae_K"],
        "hotspot_temp_error_K": float(pred[pred_hotspot] - package.true[true_hotspot]),
        "hotspot_location_error_cells": float(np.hypot(pred_hotspot[0] - true_hotspot[0], pred_hotspot[1] - true_hotspot[1])),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_packages": 0}
    keys = [
        "mae_K",
        "rmse_K",
        "max_abs_error_K",
        "mean_signed_error_K",
        "mean_rise_abs_error_K",
        "centered_field_mae_K",
        "centered_field_rmse_K",
        "chiplet_mean_temperature_mae_K",
        "chiplet_peak_temperature_mae_K",
        "inter_chiplet_delta_T_mae_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "mae_improvement_K",
        "fraction_removed",
    ]
    out: dict[str, Any] = {"num_packages": len(records)}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def summarize_by_case(mode_records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, records in sorted(mode_records.items()):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record["case_id"])].append(record)
        for case_id, items in sorted(grouped.items()):
            summary = summarize_records(items)
            summary.update({"mode": mode, "case_id": case_id})
            rows.append(summary)
    return rows


def parameters_by_case(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    rows: list[dict[str, Any]] = []
    for case_id, items in sorted(grouped.items()):
        row = {"case_id": case_id, "num_samples": len(items)}
        for key in ("oracle_mean_bias_b_K", "oracle_gain_a", "oracle_gain_offset_a", "oracle_gain_offset_b_K"):
            values = np.asarray([float(item[key]) for item in items], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
            row[f"{key}_min"] = float(np.min(values))
            row[f"{key}_max"] = float(np.max(values))
        rows.append(row)
    return rows


def parameter_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "oracle_gain_distribution": describe([r["oracle_gain_a"] for r in records]),
        "oracle_gain_offset_a_distribution": describe([r["oracle_gain_offset_a"] for r in records]),
        "oracle_gain_offset_b_distribution_K": describe([r["oracle_gain_offset_b_K"] for r in records]),
        "oracle_offset_distribution_K": describe([r["oracle_mean_bias_b_K"] for r in records]),
        "oracle_gain_vs_total_power_spearman": spearman([r["oracle_gain_a"] for r in records], [r["total_power_W"] for r in records]),
        "oracle_offset_vs_source_count_spearman": spearman([r["oracle_mean_bias_b_K"] for r in records], [r["num_sources"] for r in records]),
        "oracle_offset_vs_total_power_spearman": spearman([r["oracle_mean_bias_b_K"] for r in records], [r["total_power_W"] for r in records]),
    }


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if float(np.std(a_arr)) == 0.0 or float(np.std(b_arr)) == 0.0:
        return None
    return float(np.corrcoef(rankdata(a_arr), rankdata(b_arr))[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def decision_interpretation(overall: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = float(overall.get("raw", {}).get("mae_K", 0.0) or 0.0)
    def mae(mode: str) -> float | None:
        value = overall.get(mode, {}).get("mae_K")
        return float(value) if value is not None else None
    return {
        "raw_mae_K": raw,
        "mean_bias_bottleneck": mae("per_sample_mean_bias_oracle") is not None and mae("per_sample_mean_bias_oracle") < 2.5,
        "gain_bottleneck": mae("per_sample_gain_oracle") is not None and mae("per_sample_gain_oracle") < 2.5,
        "gain_offset_strong_upper_bound": mae("per_sample_gain_offset_oracle") is not None and mae("per_sample_gain_offset_oracle") < 2.0,
        "true_mean_oracle_implies_mean_rise_problem": mae("true_mean_pred_centered_oracle") is not None and mae("true_mean_pred_centered_oracle") < 2.5,
        "simple_calibration_insufficient": mae("per_sample_gain_offset_oracle") is not None and mae("per_sample_gain_offset_oracle") > 3.0,
        "global_val_beats_cnn_gnn_baseline": mae("global_val_gain_offset") is not None and mae("global_val_gain_offset") < 2.638,
        "per_case_val_beats_cnn_gnn_baseline": mae("per_case_val_gain_offset") is not None and mae("per_case_val_gain_offset") < 2.638,
    }


def write_outputs(out_dir: Path, analysis: dict[str, Any]) -> None:
    summary = {
        "metadata": analysis["metadata"],
        "overall_metrics": analysis["overall_metrics"],
        "parameter_diagnostics": analysis["parameter_diagnostics"],
        "validation_fitted_parameters": analysis["validation_fitted_parameters"],
        "decision_interpretation": analysis["decision_interpretation"],
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "calibration_metrics_by_case.csv", analysis["metrics_by_case"])
    write_csv(out_dir / "calibration_metrics_by_sample.csv", analysis["sample_records"])
    write_csv(out_dir / "calibration_parameters_by_sample.csv", analysis["parameter_records"])
    write_csv(out_dir / "calibration_parameters_by_case.csv", analysis["parameter_case_records"])


def write_report(out_dir: Path, analysis: dict[str, Any]) -> None:
    overall = analysis["overall_metrics"]
    lines = [
        "# Source-Response Calibration Analysis",
        "",
        "This report separates deployable validation-fitted calibrations from per-sample oracle diagnostics.",
        "",
        "## Overall Metrics",
        "",
        "| Mode | MAE K | RMSE K | Mean-rise MAE K | Centered MAE K | Fraction Removed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in sorted(overall.items()):
        lines.append(
            f"| {mode} | {fmt(metrics.get('mae_K'))} | {fmt(metrics.get('rmse_K'))} | "
            f"{fmt(metrics.get('mean_rise_abs_error_K'))} | {fmt(metrics.get('centered_field_mae_K'))} | "
            f"{fmt(metrics.get('fraction_removed'))} |"
        )
    decision = analysis["decision_interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Mean-bias bottleneck flag: `{decision['mean_bias_bottleneck']}`",
            f"- Gain bottleneck flag: `{decision['gain_bottleneck']}`",
            f"- Gain+offset oracle below 2 K: `{decision['gain_offset_strong_upper_bound']}`",
            f"- True-mean oracle suggests mean-rise problem: `{decision['true_mean_oracle_implies_mean_rise_problem']}`",
            f"- Simple calibration insufficient flag: `{decision['simple_calibration_insufficient']}`",
            f"- Global validation gain+offset beats 2.638 K baseline: `{decision['global_val_beats_cnn_gnn_baseline']}`",
            f"- Per-case validation gain+offset beats 2.638 K baseline: `{decision['per_case_val_beats_cnn_gnn_baseline']}`",
            "",
            "Questions:",
            "",
            "- Is error mostly global mean bias? Check `per_sample_mean_bias_oracle` and validation-fitted offset rows.",
            "- Is error mostly multiplicative gain? Check `per_sample_gain_oracle`, `global_val_gain`, and `per_case_val_gain`.",
            "- Is centered shape accurate? Compare `true_mean_pred_centered_oracle` against raw.",
            "- Would a small calibration head likely help? Yes only if validation-fitted global/per-case modes remove substantial error.",
            "- Is a residual CNN/GNN still needed? Yes if per-sample gain+offset remains high or centered-field MAE dominates.",
        ]
    )
    (out_dir / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def regenerate_report(out_dir: Path) -> None:
    summary_path = out_dir / "calibration_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = {
        "overall_metrics": summary["overall_metrics"],
        "decision_interpretation": summary.get("decision_interpretation", decision_interpretation(summary["overall_metrics"])),
    }
    write_report(out_dir, analysis)
    print(f"Regenerated {out_dir / 'calibration_report.md'}")


def save_selected_predictions(out_dir: Path, packages: list[PackagePrediction], sample_records: list[dict[str, Any]]) -> None:
    pred_dir = out_dir / "selected_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    raw_records = [record for record in sample_records if record["mode"] == "raw"]
    if not raw_records:
        return
    ordered = sorted(raw_records, key=lambda item: float(item["mae_K"]))
    selected = ordered[:3] + ordered[-3:]
    package_by_uid = {package.sample_uid: package for package in packages}
    for record in selected:
        package = package_by_uid[record["sample_uid"]]
        params = fit_gain_offset_sample(package)
        np.save(pred_dir / f"{package.sample_uid}_raw_prediction.npy", package.pred.astype(np.float32))
        np.save(pred_dir / f"{package.sample_uid}_gain_offset_oracle_prediction.npy", apply_gain_offset(package, params).astype(np.float32))
        np.save(pred_dir / f"{package.sample_uid}_target.npy", package.true.astype(np.float32))
        np.save(pred_dir / f"{package.sample_uid}_raw_error.npy", (package.pred - package.true).astype(np.float32))


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
        writer.writerows(records)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
