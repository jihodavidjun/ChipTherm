#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiptherm.ml.source_response_dataset import (
    SourceResponseDataset,
    SourceResponseNormalizationStats,
    normalize_source_input,
    source_response_collate,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, count_parameters, predict_source_rise
from scripts.run_superposition_diagnostic import chiplet_metrics, field_metrics, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a ChipTherm source-response model.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--clamp-unit-response-min", default=None, type=float, help="Optional evaluation-only lower clamp in K/W. Default: disabled.")
    args = parser.parse_args()

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(checkpoint["normalization"])
    model = build_source_response_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SourceResponseDataset(args.source_index, power_floor_W=float(checkpoint["model_config"].get("power_floor_W", stats.power_floor_W)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=source_response_collate)
    results = evaluate(
        model,
        loader,
        stats,
        device,
        save_predictions=args.save_predictions,
        out_dir=out_dir,
        clamp_unit_response_min=args.clamp_unit_response_min,
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "source_index": str(args.source_index.resolve()),
        "model_config": checkpoint["model_config"],
        "parameter_count": count_parameters(model),
        "source_level": results["source_level"],
        "package_reconstruction": results["package_reconstruction"],
        "oracle_reconstruction": results["oracle_reconstruction"],
        "package_bias_summary": results["package_bias_summary"],
        "prediction_stats": results["prediction_stats"],
        "evaluation_options": {"clamp_unit_response_min": args.clamp_unit_response_min},
        "runtime": results["runtime"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "source_metrics.csv", results["source_records"])
    write_csv(out_dir / "metrics_by_case.csv", results["case_records"])
    write_csv(out_dir / "package_bias_diagnostics.csv", results["package_bias_records"])
    (out_dir / "package_bias_summary.json").write_text(
        json.dumps(results["package_bias_summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Source-response evaluation complete")
    print(f"Sources: {results['source_level']['num_sources']}")
    print(f"Packages: {results['package_reconstruction']['num_packages']}")
    print(f"Source physical MAE/RMSE: {results['source_level']['physical_mae_K']:.4f} / {results['source_level']['physical_rmse_K']:.4f} K")
    print(f"Package MAE/RMSE: {results['package_reconstruction']['mae_K']:.4f} / {results['package_reconstruction']['rmse_K']:.4f} K")
    print(
        "Predicted K/W mean/std/min/max: "
        f"{results['prediction_stats']['pred_unit_K_per_W']['mean']:.6f} / "
        f"{results['prediction_stats']['pred_unit_K_per_W']['std']:.6f} / "
        f"{results['prediction_stats']['pred_unit_K_per_W']['min']:.6f} / "
        f"{results['prediction_stats']['pred_unit_K_per_W']['max']:.6f}"
    )
    print(f"Negative K/W fraction: {results['prediction_stats']['negative_unit_response_fraction_used']:.4f}")
    if args.profile_runtime:
        print(f"Runtime/source: {results['runtime']['seconds_per_source']:.6f} s")
        print(f"Runtime/package: {results['runtime']['seconds_per_package']:.6f} s")
    return 0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: SourceResponseNormalizationStats,
    device: torch.device,
    *,
    save_predictions: bool,
    out_dir: Path,
    clamp_unit_response_min: float | None,
) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    source_unit_errors: list[np.ndarray] = []
    source_physical_errors: list[np.ndarray] = []
    pred_unit_values: list[np.ndarray] = []
    pred_rise_values: list[np.ndarray] = []
    negative_count_raw = 0
    negative_count_used = 0
    prediction_count = 0
    start = time.perf_counter()
    for batch in loader:
        x = normalize_source_input(batch["x"].to(device), stats)
        power = batch["source_power_W"].to(device)
        pred_normalized = model(x)
        pred_unit_raw = unnormalize_source_prediction(pred_normalized, stats)
        pred_unit = pred_unit_raw
        if clamp_unit_response_min is not None:
            pred_unit = torch.clamp(pred_unit, min=float(clamp_unit_response_min))
        pred_rise = predict_source_rise(pred_unit, power)
        target_unit = batch["target_unit"].to(device)
        target_rise = batch["target_rise"].to(device)
        source_error = pred_rise - target_rise
        unit_error = (pred_unit - target_unit).detach().cpu().numpy()
        physical_error = source_error.detach().cpu().numpy()
        source_unit_errors.append(unit_error)
        source_physical_errors.append(physical_error)
        pred_unit_values.append(pred_unit.detach().cpu().numpy().reshape(-1))
        pred_rise_values.append(pred_rise.detach().cpu().numpy().reshape(-1))
        negative_count_raw += int((pred_unit_raw < 0.0).sum().item())
        negative_count_used += int((pred_unit < 0.0).sum().item())
        prediction_count += int(pred_unit.numel())
        pred_np = pred_rise.detach().cpu().numpy()
        target_np = target_rise.detach().cpu().numpy()
        full_np = batch["full_temperature"].detach().cpu().numpy()
        ambient_np = batch["ambient_K"].detach().cpu().numpy()
        for i, meta in enumerate(batch["metadata"]):
            source_metrics = field_metrics(pred_np[i], target_np[i])
            source_records.append(
                {
                    "source_response_uid": meta["source_response_uid"],
                    "original_sample_uid": meta["original_sample_uid"],
                    "case_id": meta["case_id"],
                    "source_index": meta["source_index"],
                    "source_name": meta["source_name"],
                    "source_power_W": meta["source_power_W"],
                    "physical_mae_K": source_metrics["mae_K"],
                    "physical_rmse_K": source_metrics["rmse_K"],
                    "unit_mae_K_per_W": float(np.abs(unit_error[i]).mean()),
                    "unit_rmse_K_per_W": float(np.sqrt(np.mean(unit_error[i] * unit_error[i]))),
                }
            )
            uid = str(meta["original_sample_uid"])
            group = groups.setdefault(
                uid,
                {
                    "case_id": meta["case_id"],
                    "ambient_K": float(ambient_np[i]),
                    "pred_sum": np.zeros_like(pred_np[i], dtype=np.float64),
                    "target_sum": np.zeros_like(target_np[i], dtype=np.float64),
                    "full_temperature": full_np[i].astype(np.float64),
                    "layout_path": meta["layout_path"],
                    "num_chiplets": int(float(meta["num_chiplets"])),
                    "num_sources": 0,
                    "total_power_W": 0.0,
                    "source_signed_mean_errors": [],
                    "source_abs_mean_errors": [],
                },
            )
            group["pred_sum"] += pred_np[i]
            group["target_sum"] += target_np[i]
            group["num_sources"] += 1
            group["total_power_W"] += float(meta["source_power_W"])
            source_error_i = source_error[i].detach().cpu().numpy()
            group["source_signed_mean_errors"].append(float(np.mean(source_error_i)))
            group["source_abs_mean_errors"].append(float(np.mean(np.abs(source_error_i))))
    elapsed = time.perf_counter() - start
    packages = package_records(groups, save_predictions=save_predictions, out_dir=out_dir)
    case_records = case_summary(packages)
    source_unit = np.concatenate([e.reshape(-1) for e in source_unit_errors])
    source_physical = np.concatenate([e.reshape(-1) for e in source_physical_errors])
    package_summary = summarize_package_records(packages, prefix="")
    oracle_summary = summarize_package_records(packages, prefix="oracle_")
    prediction_stats = {
        "pred_unit_K_per_W": describe_values(np.concatenate(pred_unit_values)),
        "pred_source_rise_K": describe_values(np.concatenate(pred_rise_values)),
        "negative_unit_response_fraction_raw": float(negative_count_raw / max(prediction_count, 1)),
        "negative_unit_response_fraction_used": float(negative_count_used / max(prediction_count, 1)),
    }
    return {
        "source_level": {
            "num_sources": len(source_records),
            "unit_mae_K_per_W": float(np.abs(source_unit).mean()),
            "unit_rmse_K_per_W": float(np.sqrt(np.mean(source_unit * source_unit))),
            "physical_mae_K": float(np.abs(source_physical).mean()),
            "physical_rmse_K": float(np.sqrt(np.mean(source_physical * source_physical))),
        },
        "package_reconstruction": package_summary,
        "oracle_reconstruction": oracle_summary,
        "source_records": source_records,
        "case_records": case_records,
        "package_bias_records": packages,
        "package_bias_summary": package_bias_summary(packages),
        "prediction_stats": prediction_stats,
        "runtime": {
            "total_forward_seconds": elapsed,
            "seconds_per_source": elapsed / max(len(source_records), 1),
            "seconds_per_package": elapsed / max(package_summary["num_packages"], 1),
        },
    }


def describe_values(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def package_records(groups: dict[str, dict[str, Any]], *, save_predictions: bool, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pred_dir = out_dir / "predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for uid, group in groups.items():
        if int(group["num_sources"]) != int(group["num_chiplets"]):
            continue
        pred_temp = float(group["ambient_K"]) + group["pred_sum"]
        oracle_temp = float(group["ambient_K"]) + group["target_sum"]
        full = group["full_temperature"]
        metrics = field_metrics(pred_temp, full)
        oracle = field_metrics(oracle_temp, full)
        package_error = pred_temp - full
        summed_source_error = group["pred_sum"] - group["target_sum"]
        source_signed = np.asarray(group["source_signed_mean_errors"], dtype=np.float64)
        source_abs = np.asarray(group["source_abs_mean_errors"], dtype=np.float64)
        pred_hotspot = np.unravel_index(int(np.argmax(pred_temp)), pred_temp.shape)
        target_hotspot = np.unravel_index(int(np.argmax(full)), full.shape)
        layout_path = Path(group["layout_path"])
        layout = load_json(layout_path if layout_path.is_absolute() else REPO_ROOT / layout_path)
        chip = chiplet_metrics(pred_temp, full, layout, full.shape)
        record = {
            "original_sample_uid": uid,
            "case_id": group["case_id"],
            "num_sources": group["num_sources"],
            "total_power_W": float(group["total_power_W"]),
            "mae_K": metrics["mae_K"],
            "rmse_K": metrics["rmse_K"],
            "max_abs_error_K": metrics["max_abs_error_K"],
            "mean_signed_error_K": metrics["mean_signed_error_K"],
            "summed_source_mean_signed_error_K": float(np.mean(summed_source_error)),
            "mean_source_signed_error_K": float(np.mean(source_signed)) if source_signed.size else 0.0,
            "mean_source_abs_error_K": float(np.mean(source_abs)) if source_abs.size else 0.0,
            "positive_source_bias_fraction": float(np.mean(source_signed > 0.0)) if source_signed.size else 0.0,
            "negative_source_bias_fraction": float(np.mean(source_signed < 0.0)) if source_signed.size else 0.0,
            "hotspot_temp_error_K": float(pred_temp[pred_hotspot] - full[target_hotspot]),
            "hotspot_location_error_cells": float(np.hypot(pred_hotspot[0] - target_hotspot[0], pred_hotspot[1] - target_hotspot[1])),
            "chiplet_mean_temperature_mae_K": chip["chiplet_mean_temperature_mae_K"],
            "chiplet_peak_temperature_mae_K": chip["chiplet_peak_temperature_mae_K"],
            "inter_chiplet_delta_T_mae_K": chip["inter_chiplet_delta_T_mae_K"],
            "oracle_mae_K": oracle["mae_K"],
            "oracle_rmse_K": oracle["rmse_K"],
            "num_sources": group["num_sources"],
        }
        records.append(record)
        if save_predictions:
            np.save(pred_dir / f"{uid}_temperature_pred.npy", pred_temp.astype(np.float32))
    return records


def summarize_package_records(records: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    if not records:
        return {"num_packages": 0, "mae_K": None, "rmse_K": None}
    key = f"{prefix}mae_K"
    rmse_key = f"{prefix}rmse_K"
    return {
        "num_packages": len(records),
        "mae_K": float(np.mean([r[key] for r in records])),
        "rmse_K": float(np.mean([r[rmse_key] for r in records])),
        "mean_signed_error_K": float(np.mean([r[f"{prefix}mean_signed_error_K"] for r in records])) if f"{prefix}mean_signed_error_K" in records[0] else None,
    }


def case_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["case_id"]].append(record)
    rows = []
    for case_id, items in sorted(grouped.items()):
        rows.append(
            {
                "case_id": case_id,
                "num_packages": len(items),
                "mae_K": float(np.mean([item["mae_K"] for item in items])),
                "rmse_K": float(np.mean([item["rmse_K"] for item in items])),
                "oracle_mae_K": float(np.mean([item["oracle_mae_K"] for item in items])),
                "chiplet_mean_temperature_mae_K": mean_optional(item["chiplet_mean_temperature_mae_K"] for item in items),
                "chiplet_peak_temperature_mae_K": mean_optional(item["chiplet_peak_temperature_mae_K"] for item in items),
                "inter_chiplet_delta_T_mae_K": mean_optional(item["inter_chiplet_delta_T_mae_K"] for item in items),
            }
        )
    return rows


def mean_optional(values: Any) -> float | None:
    numeric = [float(v) for v in values if v is not None]
    return float(np.mean(numeric)) if numeric else None


def package_bias_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "num_packages": len(records),
        "package_mae_vs_source_count_spearman": spearman([r["mae_K"] for r in records], [r["num_sources"] for r in records]),
        "package_mae_vs_total_power_spearman": spearman([r["mae_K"] for r in records], [r["total_power_W"] for r in records]),
        "package_signed_bias_vs_source_count_spearman": spearman([r["mean_signed_error_K"] for r in records], [r["num_sources"] for r in records]),
        "package_signed_bias_vs_mean_source_signed_bias_spearman": spearman(
            [r["mean_signed_error_K"] for r in records],
            [r["mean_source_signed_error_K"] for r in records],
        ),
        "mean_positive_source_bias_fraction": float(np.mean([r["positive_source_bias_fraction"] for r in records])),
        "mean_negative_source_bias_fraction": float(np.mean([r["negative_source_bias_fraction"] for r in records])),
    }


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if float(np.std(a_arr)) == 0.0 or float(np.std(b_arr)) == 0.0:
        return None
    a_rank = rankdata(a_arr)
    b_rank = rankdata(b_arr)
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    mean_ranks = sums / counts
    return mean_ranks[inverse]


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
