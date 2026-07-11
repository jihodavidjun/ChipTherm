#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.models import build_model  # noqa: E402
from chiptherm.ml.normalization import (  # noqa: E402
    NormalizationStats,
    build_metadata_input,
    build_model_input,
)
from analyze_residual_cnn_errors import architecture_info, predict_temperature  # noqa: E402


RECURRING_SAMPLE_HINTS = [
    "training_set_4k_extra_case02_sample_000216",
    "training_set_4k_extra_case02_sample_000365",
    "training_set_4k_extra_case02_sample_000183",
    "training_set_4k_extra_case02_sample_000327",
    "training_set_4k_extra_case01_sample_000155",
    "training_set_4k_extra_case01_sample_000181",
]

BASE_DESCRIPTOR_COLUMNS = [
    "package_width_mm",
    "package_height_mm",
    "package_area_mm2",
    "cell_size_x_mm",
    "cell_size_y_mm",
    "total_power_W",
    "chiplet_count",
    "occupied_fraction",
    "whitespace_fraction",
    "mean_power_density_W_per_mm2",
    "max_power_density_W_per_mm2",
    "mean_chiplet_area_mm2",
    "max_chiplet_area_mm2",
    "mean_chiplet_aspect_ratio",
    "power_variance_across_chiplets",
    "max_chiplet_power_fraction",
    "minimum_pairwise_chiplet_distance_mm",
    "mean_pairwise_chiplet_distance_mm",
    "fraction_chiplets_near_package_edges",
    "thermal_crowding_mean",
    "thermal_crowding_max",
    "thermal_crowding_std",
    "physics_v1_mae_K",
    "physics_v1_mean_signed_error_K",
    "physics_v1_hotspot_location_error_cells",
    "true_temperature_mean_K",
    "true_temperature_range_K",
    "true_temperature_std_K",
    "true_temperature_gradient_magnitude_K_per_cell",
]

TARGET_COLUMNS = [
    "seed0_final_mae_K",
    "seed1_final_mae_K",
    "mean_final_mae_K",
    "ensemble_final_mae_K",
    "seed0_centered_field_mae_K",
    "seed1_centered_field_mae_K",
    "seed0_mean_rise_abs_error_K",
    "seed1_mean_rise_abs_error_K",
    "seed0_hotspot_top_5pct_mae_K",
    "seed1_hotspot_top_5pct_mae_K",
    "seed0_hotspot_location_error_cells",
    "seed1_hotspot_location_error_cells",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose persistent hard cases for ChipTherm decomposed models.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--seed0-checkpoint", required=True, type=Path)
    parser.add_argument("--seed1-checkpoint", required=True, type=Path)
    parser.add_argument("--seed0-analysis", default=None, type=Path)
    parser.add_argument("--seed1-analysis", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()

    device = select_device(args.device)
    dataset_root = args.dataset_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    plots_dir = out_dir / "plots"
    panels_dir = out_dir / "sample_panels"
    plots_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        split: read_rows(dataset_root / f"{split}_index.csv")
        for split in ("train", "val", "test")
    }
    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    metadata_table = read_metadata_table(dataset_root / "metadata_features.csv")

    print("Computing descriptors for train/val/test samples")
    descriptors = [sample_descriptors(row, metadata_table[row["sample_uid"]], dataset_root) for row in all_rows]
    descriptor_by_uid = {record["sample_uid"]: record for record in descriptors}

    print("Running seed-0 and seed-1 checkpoint inference on test split")
    seed0_predictions = predict_split(args.seed0_checkpoint, dataset_root / "test_index.csv", device, args.batch_size, args.num_workers)
    seed1_predictions = predict_split(args.seed1_checkpoint, dataset_root / "test_index.csv", device, args.batch_size, args.num_workers)

    test_records = []
    for row in rows_by_split["test"]:
        uid = row["sample_uid"]
        descriptor = descriptor_by_uid[uid]
        combined = {
            **descriptor,
            **prediction_metrics_for_uid(uid, seed0_predictions, prefix="seed0"),
            **prediction_metrics_for_uid(uid, seed1_predictions, prefix="seed1"),
        }
        y = seed0_predictions[uid]["target"]
        ensemble = 0.5 * (seed0_predictions[uid]["prediction"] + seed1_predictions[uid]["prediction"])
        combined["ensemble_final_mae_K"] = mae(ensemble, y)
        combined["ensemble_final_rmse_K"] = rmse(ensemble, y)
        combined["mean_final_mae_K"] = 0.5 * (combined["seed0_final_mae_K"] + combined["seed1_final_mae_K"])
        combined["seed_final_mae_abs_diff_K"] = abs(combined["seed0_final_mae_K"] - combined["seed1_final_mae_K"])
        test_records.append(combined)

    case_stats = case_statistics(descriptors)
    case02_rows = case02_comparison_rows(descriptors)
    correlations = correlation_rows(test_records)
    hard_samples = hard_sample_rows(test_records, descriptor_by_uid, seed0_predictions, seed1_predictions, dataset_root)
    seed_comparison = seed_comparison_rows(test_records)

    write_csv(out_dir / "case_statistics.csv", case_stats)
    write_csv(out_dir / "case02_vs_others.csv", case02_rows)
    write_csv(out_dir / "sample_error_correlations.csv", correlations)
    write_csv(out_dir / "hard_samples.csv", hard_samples)
    write_csv(out_dir / "seed_comparison.csv", seed_comparison)
    make_plots(plots_dir, test_records, case02_rows)
    make_sample_panels(panels_dir, hard_samples, seed0_predictions, seed1_predictions)

    summary = build_summary(args, descriptors, test_records, case02_rows, correlations, hard_samples, seed_comparison)
    write_json(out_dir / "summary.json", summary)

    print("Hard-case analysis complete")
    print(f"Test samples: {len(test_records)}")
    print(f"Seed MAE correlation: {summary['seed_comparison']['spearman_seed0_seed1_final_mae']:.3f}")
    print(f"Seed0/Seed1/Ensemble MAE: {summary['overall']['seed0_mae_K']:.3f} / {summary['overall']['seed1_mae_K']:.3f} / {summary['overall']['ensemble_mae_K']:.3f} K")
    print(f"Case02 mean final MAE: {summary['case02']['mean_final_mae_K']:.3f} K")
    print(f"Output: {out_dir}")
    return 0


def sample_descriptors(row: dict[str, str], metadata: dict[str, float], dataset_root: Path) -> dict[str, Any]:
    x_path = resolve_path(row["x_path"], dataset_root)
    y_path = resolve_path(row["y_path"], dataset_root)
    physics_path = resolve_path(row["prediction_path"], dataset_root)
    x = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.float32, copy=False)
    physics = np.load(physics_path).astype(np.float32, copy=False)
    layout, power_data, package, hotspot = load_source_artifacts(row)
    chiplets = chiplet_records(layout, power_data)
    powers = np.asarray([chiplet["power_W"] for chiplet in chiplets], dtype=np.float64)
    areas = np.asarray([chiplet["width_mm"] * chiplet["height_mm"] for chiplet in chiplets], dtype=np.float64)
    aspects = np.asarray([chiplet["width_mm"] / chiplet["height_mm"] for chiplet in chiplets], dtype=np.float64)
    pairwise_edge, pairwise_center = pairwise_distances(chiplets)
    edge_fraction = fraction_near_edges(chiplets, metadata["package_width_mm"], metadata["package_height_mm"])
    physics_error = physics - y
    pred_hotspot = np.unravel_index(int(np.argmax(physics)), physics.shape)
    target_hotspot = np.unravel_index(int(np.argmax(y)), y.shape)
    gy, gx = np.gradient(y.astype(np.float64))
    crowding = x[32].astype(np.float64) if x.shape[0] > 32 else np.zeros_like(y, dtype=np.float64)
    raster_power_W = float((x[0].astype(np.float64) * metadata["cell_size_x_mm"] * metadata["cell_size_y_mm"]).sum())
    layout_power_W = float(powers.sum())
    occupancy_area_fraction = float(x[1].mean())
    layout_occupied_fraction = float(areas.sum() / max(metadata["package_width_mm"] * metadata["package_height_mm"], 1.0e-12))
    return {
        "sample_uid": row["sample_uid"],
        "original_sample_uid": row.get("original_sample_uid", ""),
        "case_id": row["case_id"],
        "split": row.get("split", ""),
        "dataset_source": row.get("dataset_source", ""),
        "package_width_mm": metadata["package_width_mm"],
        "package_height_mm": metadata["package_height_mm"],
        "package_area_mm2": metadata["package_width_mm"] * metadata["package_height_mm"],
        "cell_size_x_mm": metadata["cell_size_x_mm"],
        "cell_size_y_mm": metadata["cell_size_y_mm"],
        "total_power_W": metadata["total_power_W"],
        "chiplet_count": metadata["chiplet_count"],
        "occupied_fraction": metadata["occupied_fraction"],
        "whitespace_fraction": metadata["whitespace_fraction"],
        "mean_power_density_W_per_mm2": metadata["mean_power_density_W_per_mm2"],
        "max_power_density_W_per_mm2": metadata["max_power_density_W_per_mm2"],
        "mean_chiplet_area_mm2": metadata["mean_chiplet_area_mm2"],
        "max_chiplet_area_mm2": metadata["max_chiplet_area_mm2"],
        "mean_chiplet_aspect_ratio": metadata["mean_chiplet_aspect_ratio"],
        "power_variance_across_chiplets": float(powers.var()) if powers.size else 0.0,
        "max_chiplet_power_fraction": float(powers.max() / powers.sum()) if powers.size and powers.sum() > 0 else 0.0,
        "minimum_pairwise_chiplet_distance_mm": float(pairwise_edge.min()) if pairwise_edge.size else 0.0,
        "mean_pairwise_chiplet_distance_mm": float(pairwise_center.mean()) if pairwise_center.size else 0.0,
        "fraction_chiplets_near_package_edges": edge_fraction,
        "thermal_crowding_mean": float(crowding.mean()),
        "thermal_crowding_max": float(crowding.max()),
        "thermal_crowding_std": float(crowding.std()),
        "physics_v1_mae_K": mae(physics, y),
        "physics_v1_mean_signed_error_K": float(physics_error.mean()),
        "physics_v1_hotspot_location_error_cells": float(math.hypot(pred_hotspot[0] - target_hotspot[0], pred_hotspot[1] - target_hotspot[1])),
        "true_temperature_mean_K": float(y.mean()),
        "true_temperature_range_K": float(y.max() - y.min()),
        "true_temperature_std_K": float(y.std()),
        "true_temperature_gradient_magnitude_K_per_cell": float(np.sqrt(gx * gx + gy * gy).mean()),
        "layout_power_W": layout_power_W,
        "raster_power_W": raster_power_W,
        "power_mismatch_W": abs(layout_power_W - metadata["total_power_W"]),
        "raster_power_mismatch_W": abs(raster_power_W - metadata["total_power_W"]),
        "occupancy_fraction_mismatch": abs(occupancy_area_fraction - layout_occupied_fraction),
        "metadata_hash": row_hash(row),
        "has_chiplet_overlap": has_overlap(chiplets),
        "has_chiplet_clipping": has_clipping(chiplets, metadata["package_width_mm"], metadata["package_height_mm"]),
        "detailed_package_enabled": bool(hotspot.get("detailed_package", False)),
        "secondary_path_enabled": bool(hotspot.get("secondary_path", False)),
        "ambient_K": float(package.get("ambient_K", 318.15)),
    }


def predict_split(
    checkpoint_path: Path,
    index_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    stats = NormalizationStats(**checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    info = architecture_info(checkpoint["model_config"])
    physics_input_mode = str(info.get("physics_input_mode", "v1"))
    dataset = ChipThermDataset(index_path, target="residual", return_metadata=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    result: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            physics = batch["physics"].to(device, non_blocking=True)
            temperature = batch["temperature"].to(device, non_blocking=True)
            ambient = batch["ambient_K"].to(device, non_blocking=True).float()
            model_input = build_model_input(x, physics, stats, physics_input_mode=physics_input_mode)
            metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
            if metadata_input is not None:
                metadata_input = metadata_input.to(device, non_blocking=True)
            pred = predict_temperature(model, model_input, physics, ambient, metadata_input, stats, info)
            batch_size_actual = int(x.shape[0])
            sample_uids = metadata_values(batch["metadata"], "sample_uid", batch_size_actual)
            case_ids = metadata_values(batch["metadata"], "case_id", batch_size_actual)
            for index, uid in enumerate(sample_uids):
                entry = {
                    "sample_uid": str(uid),
                    "case_id": str(case_ids[index]),
                    "prediction": pred["temperature"][index].detach().cpu().numpy().astype(np.float32),
                    "target": temperature[index].detach().cpu().numpy().astype(np.float32),
                    "physics": physics[index].detach().cpu().numpy().astype(np.float32),
                }
                if "mean_rise" in pred:
                    entry["mean_rise"] = float(pred["mean_rise"][index].detach().cpu().item())
                    entry["centered_field"] = pred["centered_field"][index].detach().cpu().numpy().astype(np.float32)
                result[str(uid)] = entry
    return result


def prediction_metrics_for_uid(uid: str, predictions: dict[str, dict[str, Any]], prefix: str) -> dict[str, float]:
    entry = predictions[uid]
    pred = entry["prediction"]
    y = entry["target"]
    error = pred - y
    true_hotspot = np.unravel_index(int(np.argmax(y)), y.shape)
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    hotspot5 = top_fraction_mask(y, 0.05)
    centered_target = y - float(y.mean())
    if "centered_field" in entry:
        centered = entry["centered_field"]
        mean_rise_error = float(entry["mean_rise"] - float(y.mean() - 318.15))
    else:
        centered = pred - float(pred.mean())
        mean_rise_error = float(pred.mean() - y.mean())
    return {
        f"{prefix}_final_mae_K": mae(pred, y),
        f"{prefix}_final_rmse_K": rmse(pred, y),
        f"{prefix}_mean_signed_error_K": float(error.mean()),
        f"{prefix}_mean_rise_error_K": mean_rise_error,
        f"{prefix}_mean_rise_abs_error_K": abs(mean_rise_error),
        f"{prefix}_centered_field_mae_K": mae(centered, centered_target),
        f"{prefix}_centered_field_rmse_K": rmse(centered, centered_target),
        f"{prefix}_hotspot_top_5pct_mae_K": float(np.abs(error)[hotspot5].mean()),
        f"{prefix}_hotspot_location_error_cells": float(math.hypot(pred_hotspot[0] - true_hotspot[0], pred_hotspot[1] - true_hotspot[1])),
    }


def case_statistics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("all", "train", "val", "test"):
        split_records = records if split == "all" else [record for record in records if record["split"] == split]
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in split_records:
            by_case[record["case_id"]].append(record)
        for case_id, items in sorted(by_case.items()):
            row: dict[str, Any] = {"case_id": case_id, "split": split, "num_samples": len(items)}
            for column in BASE_DESCRIPTOR_COLUMNS:
                values = np.asarray([float(item[column]) for item in items], dtype=np.float64)
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std())
            output.append(row)
    return output


def case02_comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_case_means: dict[str, dict[str, float]] = {}
    for case_id in sorted({record["case_id"] for record in records}):
        case_records = [record for record in records if record["case_id"] == case_id]
        all_case_means[case_id] = {
            column: float(np.mean([float(record[column]) for record in case_records]))
            for column in BASE_DESCRIPTOR_COLUMNS
        }
    rows: list[dict[str, Any]] = []
    train_case02 = [record for record in records if record["case_id"] == "case02" and record["split"] == "train"]
    test_case02 = [record for record in records if record["case_id"] == "case02" and record["split"] == "test"]
    for column in BASE_DESCRIPTOR_COLUMNS:
        case02_value = all_case_means["case02"][column]
        other_values = np.asarray([means[column] for case, means in all_case_means.items() if case != "case02"], dtype=np.float64)
        all_values = np.asarray([means[column] for means in all_case_means.values()], dtype=np.float64)
        z = (case02_value - float(all_values.mean())) / max(float(all_values.std()), 1.0e-12)
        train_mean = float(np.mean([float(record[column]) for record in train_case02]))
        test_mean = float(np.mean([float(record[column]) for record in test_case02]))
        rows.append(
            {
                "descriptor": column,
                "case02_mean": case02_value,
                "others_mean": float(other_values.mean()),
                "others_min": float(other_values.min()),
                "others_max": float(other_values.max()),
                "global_case_mean": float(all_values.mean()),
                "global_case_std": float(all_values.std()),
                "case02_zscore_across_case_means": z,
                "case02_outside_other_case_range": bool(case02_value < other_values.min() or case02_value > other_values.max()),
                "case02_train_mean": train_mean,
                "case02_test_mean": test_mean,
                "case02_test_minus_train": test_mean - train_mean,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["case02_zscore_across_case_means"])), reverse=True)


def correlation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        if target not in records[0]:
            continue
        for descriptor in BASE_DESCRIPTOR_COLUMNS:
            rows.append(correlation_row(records, target, descriptor, "global"))
        for case_id in sorted({record["case_id"] for record in records}):
            case_records = [record for record in records if record["case_id"] == case_id]
            if len(case_records) >= 5:
                for descriptor in BASE_DESCRIPTOR_COLUMNS:
                    rows.append(correlation_row(case_records, target, descriptor, case_id))
    return rows


def correlation_row(records: list[dict[str, Any]], target: str, descriptor: str, scope: str) -> dict[str, Any]:
    x = np.asarray([float(record[descriptor]) for record in records], dtype=np.float64)
    y = np.asarray([float(record[target]) for record in records], dtype=np.float64)
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        rho, pvalue = float("nan"), float("nan")
    else:
        rho, pvalue = spearmanr(x, y)
    return {
        "scope": scope,
        "target": target,
        "descriptor": descriptor,
        "spearman_rho": float(rho),
        "p_value": float(pvalue),
        "num_samples": len(records),
    }


def hard_sample_rows(
    test_records: list[dict[str, Any]],
    descriptor_by_uid: dict[str, dict[str, Any]],
    seed0_predictions: dict[str, dict[str, Any]],
    seed1_predictions: dict[str, dict[str, Any]],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    selected_uids: set[str] = set()
    for hint in RECURRING_SAMPLE_HINTS:
        selected_uids.update(record["sample_uid"] for record in test_records if hint in record["sample_uid"])
    for key in ("mean_final_mae_K", "seed0_mean_rise_abs_error_K", "seed1_mean_rise_abs_error_K", "seed0_centered_field_mae_K", "seed1_centered_field_mae_K", "seed0_hotspot_top_5pct_mae_K", "seed1_hotspot_top_5pct_mae_K"):
        selected_uids.add(max(test_records, key=lambda record: float(record[key]))["sample_uid"])
    rows: list[dict[str, Any]] = []
    test_by_uid = {record["sample_uid"]: record for record in test_records}
    for uid in sorted(selected_uids):
        record = test_by_uid[uid]
        descriptor = descriptor_by_uid[uid]
        layout, power_data, package, hotspot = load_source_artifacts(record)
        chiplets = chiplet_records(layout, power_data)
        rows.append(
            {
                **{column: record.get(column) for column in ["sample_uid", "case_id", "dataset_source", *BASE_DESCRIPTOR_COLUMNS]},
                "selection_reason": selection_reason(uid, record, test_records),
                "chiplet_summary": chiplet_summary(chiplets),
                "seed0_final_mae_K": record["seed0_final_mae_K"],
                "seed1_final_mae_K": record["seed1_final_mae_K"],
                "ensemble_final_mae_K": record["ensemble_final_mae_K"],
                "seed0_mean_rise_error_K": record["seed0_mean_rise_error_K"],
                "seed1_mean_rise_error_K": record["seed1_mean_rise_error_K"],
                "seed0_centered_field_mae_K": record["seed0_centered_field_mae_K"],
                "seed1_centered_field_mae_K": record["seed1_centered_field_mae_K"],
                "seed0_hotspot_top_5pct_mae_K": record["seed0_hotspot_top_5pct_mae_K"],
                "seed1_hotspot_top_5pct_mae_K": record["seed1_hotspot_top_5pct_mae_K"],
                "power_mismatch_W": descriptor["power_mismatch_W"],
                "raster_power_mismatch_W": descriptor["raster_power_mismatch_W"],
                "occupancy_fraction_mismatch": descriptor["occupancy_fraction_mismatch"],
                "has_chiplet_overlap": descriptor["has_chiplet_overlap"],
                "has_chiplet_clipping": descriptor["has_chiplet_clipping"],
                "detailed_package_enabled": descriptor["detailed_package_enabled"],
                "secondary_path_enabled": descriptor["secondary_path_enabled"],
                "max_seed0_error_location": max_error_location(seed0_predictions[uid]["prediction"], seed0_predictions[uid]["target"]),
                "max_seed1_error_location": max_error_location(seed1_predictions[uid]["prediction"], seed1_predictions[uid]["target"]),
                "error_concentration": error_concentration(record),
            }
        )
    return rows


def selection_reason(uid: str, record: dict[str, Any], records: list[dict[str, Any]]) -> str:
    reasons = []
    if any(hint in uid for hint in RECURRING_SAMPLE_HINTS):
        reasons.append("prompt_recurring")
    for key, label in [
        ("mean_final_mae_K", "worst_final"),
        ("seed0_mean_rise_abs_error_K", "worst_seed0_mean"),
        ("seed1_mean_rise_abs_error_K", "worst_seed1_mean"),
        ("seed0_centered_field_mae_K", "worst_seed0_centered"),
        ("seed1_centered_field_mae_K", "worst_seed1_centered"),
        ("seed0_hotspot_top_5pct_mae_K", "worst_seed0_hotspot"),
        ("seed1_hotspot_top_5pct_mae_K", "worst_seed1_hotspot"),
    ]:
        if uid == max(records, key=lambda item: float(item[key]))["sample_uid"]:
            reasons.append(label)
    return "+".join(reasons)


def seed_comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in sorted(records, key=lambda item: item["sample_uid"]):
        rows.append(
            {
                "sample_uid": record["sample_uid"],
                "case_id": record["case_id"],
                "seed0_final_mae_K": record["seed0_final_mae_K"],
                "seed1_final_mae_K": record["seed1_final_mae_K"],
                "seed1_minus_seed0_final_mae_K": record["seed1_final_mae_K"] - record["seed0_final_mae_K"],
                "ensemble_final_mae_K": record["ensemble_final_mae_K"],
                "bad_in_both_seeds": bool(record["seed0_final_mae_K"] >= 6.0 and record["seed1_final_mae_K"] >= 6.0),
            }
        )
    return rows


def build_summary(
    args: argparse.Namespace,
    descriptors: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    case02_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    hard_samples: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seed0 = np.asarray([record["seed0_final_mae_K"] for record in test_records], dtype=np.float64)
    seed1 = np.asarray([record["seed1_final_mae_K"] for record in test_records], dtype=np.float64)
    ensemble = np.asarray([record["ensemble_final_mae_K"] for record in test_records], dtype=np.float64)
    mean_final = np.asarray([record["mean_final_mae_K"] for record in test_records], dtype=np.float64)
    case02 = [record for record in test_records if record["case_id"] == "case02"]
    seed_rho = float(spearmanr(seed0, seed1).statistic)
    bad_both = [row for row in seed_rows if row["bad_in_both_seeds"]]
    top_case02_unusual = case02_rows[:8]
    top_corrs = sorted(
        [row for row in correlations if row["scope"] == "global" and row["target"] == "mean_final_mae_K" and not math.isnan(row["spearman_rho"])],
        key=lambda row: abs(float(row["spearman_rho"])),
        reverse=True,
    )[:10]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "seed0_checkpoint": str(args.seed0_checkpoint.resolve()),
        "seed1_checkpoint": str(args.seed1_checkpoint.resolve()),
        "num_samples": {"all_splits": len(descriptors), "test": len(test_records)},
        "overall": {
            "seed0_mae_K": float(seed0.mean()),
            "seed1_mae_K": float(seed1.mean()),
            "ensemble_mae_K": float(ensemble.mean()),
            "mean_two_seed_mae_K": float(mean_final.mean()),
        },
        "case02": {
            "num_test_samples": len(case02),
            "mean_final_mae_K": float(np.mean([record["mean_final_mae_K"] for record in case02])),
            "seed0_final_mae_K": float(np.mean([record["seed0_final_mae_K"] for record in case02])),
            "seed1_final_mae_K": float(np.mean([record["seed1_final_mae_K"] for record in case02])),
            "seed0_centered_field_mae_K": float(np.mean([record["seed0_centered_field_mae_K"] for record in case02])),
            "seed1_centered_field_mae_K": float(np.mean([record["seed1_centered_field_mae_K"] for record in case02])),
        },
        "seed_comparison": {
            "spearman_seed0_seed1_final_mae": seed_rho,
            "bad_in_both_seed_count_threshold_6K": len(bad_both),
            "most_improved_seed1_vs_seed0": sorted(seed_rows, key=lambda row: float(row["seed1_minus_seed0_final_mae_K"]))[:10],
            "most_worsened_seed1_vs_seed0": sorted(seed_rows, key=lambda row: float(row["seed1_minus_seed0_final_mae_K"]), reverse=True)[:10],
        },
        "case02_most_unusual_descriptors": top_case02_unusual,
        "top_global_error_correlations": top_corrs,
        "hard_sample_count": len(hard_samples),
        "ranked_diagnosis": diagnose(top_case02_unusual, top_corrs, hard_samples, bad_both, case02),
    }


def diagnose(case02_rows: list[dict[str, Any]], correlations: list[dict[str, Any]], hard_samples: list[dict[str, Any]], bad_both: list[dict[str, Any]], case02_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    top_unusual = [row["descriptor"] for row in case02_rows[:5]]
    top_corr = [row["descriptor"] for row in correlations[:5]]
    case02_centered = np.mean([record["seed0_centered_field_mae_K"] + record["seed1_centered_field_mae_K"] for record in case02_records]) / 2.0
    case02_mean = np.mean([record["seed0_mean_rise_abs_error_K"] + record["seed1_mean_rise_abs_error_K"] for record in case02_records]) / 2.0
    return [
        {
            "rank": "primary",
            "diagnosis": "case02 difficulty is dominated by spatial/centered-field error rather than common-mode mean-rise error",
            "evidence": f"case02 average centered-field MAE {case02_centered:.3f} K vs mean-rise abs error {case02_mean:.3f} K",
        },
        {
            "rank": "secondary",
            "diagnosis": "case02 is distributionally unusual in package/sparsity/source-scale descriptors",
            "evidence": f"largest case02 z-score descriptors: {', '.join(top_unusual)}",
        },
        {
            "rank": "tertiary",
            "diagnosis": "recurring outliers are mostly systematic, not seed-specific optimization accidents",
            "evidence": f"{len(bad_both)} samples exceed 6 K final MAE in both seeds; top global correlations include {', '.join(top_corr)}",
        },
    ]


def make_plots(out_dir: Path, records: list[dict[str, Any]], case02_rows: list[dict[str, Any]]) -> None:
    scatter(records, "package_area_mm2", "mean_final_mae_K", out_dir / "centered_field_mae_vs_package_area.png", "Error vs package area")
    scatter(records, "occupied_fraction", "mean_final_mae_K", out_dir / "centered_field_mae_vs_occupied_fraction.png", "Error vs occupied fraction")
    scatter(records, "total_power_W", "mean_final_mae_K", out_dir / "centered_field_mae_vs_total_power.png", "Error vs total power")
    scatter(records, "max_power_density_W_per_mm2", "mean_final_mae_K", out_dir / "centered_field_mae_vs_max_power_density.png", "Error vs max power density")
    scatter(records, "physics_v1_mae_K", "mean_final_mae_K", out_dir / "error_vs_physics_v1_mae.png", "Error vs physics_v1 MAE")
    scatter(records, "seed0_final_mae_K", "seed1_final_mae_K", out_dir / "seed0_vs_seed1_per_sample_mae.png", "Seed0 vs seed1 MAE")
    draw_case02_zbars(case02_rows, out_dir / "case02_descriptor_zscores.png")


def make_sample_panels(
    out_dir: Path,
    hard_samples: list[dict[str, Any]],
    seed0_predictions: dict[str, dict[str, Any]],
    seed1_predictions: dict[str, dict[str, Any]],
) -> None:
    for row in hard_samples:
        uid = row["sample_uid"]
        if uid not in seed0_predictions:
            continue
        y = seed0_predictions[uid]["target"]
        physics = seed0_predictions[uid]["physics"]
        pred0 = seed0_predictions[uid]["prediction"]
        pred1 = seed1_predictions[uid]["prediction"]
        arrays = [
            ("HotSpot", y),
            ("Physics v1", physics),
            ("Seed0 pred", pred0),
            ("Seed0 err", pred0 - y),
            ("Seed1 pred", pred1),
            ("Seed1 err", pred1 - y),
        ]
        save_panel(arrays, out_dir / f"{sanitize(uid)}.png", uid)


def scatter(records: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str) -> None:
    w, h = 900, 620
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    plot = (80, 70, 840, 540)
    draw.rectangle(plot, outline=(30, 30, 30))
    xs = np.asarray([float(record[x_key]) for record in records], dtype=np.float64)
    ys = np.asarray([float(record[y_key]) for record in records], dtype=np.float64)
    xmin, xmax = padded(xs)
    ymin, ymax = padded(ys)
    colors = case_colors()
    for record, x, y in zip(records, xs, ys):
        px = int(plot[0] + (x - xmin) / (xmax - xmin) * (plot[2] - plot[0]))
        py = int(plot[3] - (y - ymin) / (ymax - ymin) * (plot[3] - plot[1]))
        color = colors.get(record["case_id"], (80, 80, 80))
        radius = 4 if record["case_id"] == "case02" else 2
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    draw.text((25, 20), title, fill=(20, 20, 20), font=font())
    draw.text((plot[0], plot[3] + 18), x_key, fill=(20, 20, 20), font=font())
    draw.text((20, plot[1]), y_key, fill=(20, 20, 20), font=font())
    image.save(path)


def draw_case02_zbars(case02_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in case02_rows if abs(float(row["case02_zscore_across_case_means"])) > 0.5][:14]
    w, h = 1000, 620
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), "Case02 descriptor z-scores", fill=(20, 20, 20), font=font())
    if not rows:
        image.save(path)
        return
    max_abs = max(abs(float(row["case02_zscore_across_case_means"])) for row in rows)
    x0 = 500
    for i, row in enumerate(rows):
        y = 70 + i * 36
        z = float(row["case02_zscore_across_case_means"])
        length = int(360 * abs(z) / max_abs)
        if z >= 0:
            draw.rectangle((x0, y, x0 + length, y + 18), fill=(180, 70, 60))
        else:
            draw.rectangle((x0 - length, y, x0, y + 18), fill=(70, 110, 180))
        draw.text((20, y), str(row["descriptor"])[:50], fill=(20, 20, 20), font=font())
        draw.text((x0 + 370, y), f"{z:.2f}", fill=(20, 20, 20), font=font())
    draw.line((x0, 60, x0, h - 30), fill=(30, 30, 30))
    image.save(path)

def save_panel(arrays: list[tuple[str, np.ndarray]], path: Path, title: str) -> None:
    panel = 150
    image = Image.new("RGB", (panel * len(arrays), panel + 55), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 5), title[:100], fill=(20, 20, 20), font=font())
    temp_arrays = [array for name, array in arrays if "err" not in name.lower()]
    tmin = min(float(array.min()) for array in temp_arrays)
    tmax = max(float(array.max()) for array in temp_arrays)
    for i, (name, array) in enumerate(arrays):
        if "err" in name.lower():
            lim = max(float(abs(array).max()), 1.0)
            img = array_image(array, -lim, lim, diverging=True)
        else:
            img = array_image(array, tmin, tmax, diverging=False)
        image.paste(img.resize((panel, panel), Image.Resampling.BILINEAR), (i * panel, 45))
        draw.text((i * panel + 4, 28), name, fill=(20, 20, 20), font=font())
    image.save(path)


def array_image(array: np.ndarray, vmin: float, vmax: float, *, diverging: bool) -> Image.Image:
    t = np.clip((array.astype(np.float64) - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
    if diverging:
        rgb = np.zeros(t.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = np.clip(255 * t, 0, 255)
        rgb[..., 1] = np.clip(255 * (1.0 - np.abs(t - 0.5) * 2.0), 0, 255)
        rgb[..., 2] = np.clip(255 * (1.0 - t), 0, 255)
    else:
        rgb = np.zeros(t.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = np.clip(255 * t, 0, 255)
        rgb[..., 1] = np.clip(180 * np.sqrt(t), 0, 255)
        rgb[..., 2] = np.clip(255 * (1.0 - t), 0, 255)
    return Image.fromarray(rgb)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=to_jsonable) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_metadata_table(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        result = {}
        for row in reader:
            result[row["sample_uid"]] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"sample_uid", "case_id", "split"}
            }
        return result


def load_source_artifacts(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_id = row["case_id"]
    original_uid = row.get("original_sample_uid", "")
    prefix = f"{case_id}_"
    if not original_uid.startswith(prefix):
        raise ValueError(f"{row['sample_uid']} original_sample_uid mismatch")
    sample_dir = original_uid[len(prefix) :]
    source_dir = REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_dir / "source"
    layout = json.loads((source_dir / "layout.json").read_text(encoding="utf-8"))
    power = yaml.safe_load((source_dir / "power.yaml").read_text(encoding="utf-8")) or {}
    package = yaml.safe_load((source_dir / "package.yaml").read_text(encoding="utf-8")) or {}
    hotspot = yaml.safe_load((source_dir / "hotspot.yaml").read_text(encoding="utf-8")) or {}
    return layout, power, package, hotspot


def chiplet_records(layout: dict[str, Any], power_data: dict[str, Any]) -> list[dict[str, float]]:
    powers = active_power_map(power_data)
    chiplets = []
    for item in layout.get("chiplets", []):
        name = str(item["name"])
        pos = item["position"]
        size = item["size"]
        chiplets.append(
            {
                "name": name,
                "x_mm": float(pos["x"]),
                "y_mm": float(pos["y"]),
                "width_mm": float(size["width"]),
                "height_mm": float(size["height"]),
                "power_W": float(powers[name]),
            }
        )
    return chiplets


def active_power_map(power_data: dict[str, Any]) -> dict[str, float]:
    active = power_data.get("active_workload")
    workloads = power_data.get("workloads") or {}
    if active and active in workloads:
        return {str(k): float(v) for k, v in workloads[active].items()}
    return {str(k): float(v) for k, v in power_data.get("chiplets", {}).items()}


def pairwise_distances(chiplets: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    edge_distances = []
    center_distances = []
    for i in range(len(chiplets)):
        a = chiplets[i]
        ax0, ax1 = a["x_mm"], a["x_mm"] + a["width_mm"]
        ay0, ay1 = a["y_mm"], a["y_mm"] + a["height_mm"]
        acx, acy = a["x_mm"] + a["width_mm"] / 2.0, a["y_mm"] + a["height_mm"] / 2.0
        for j in range(i + 1, len(chiplets)):
            b = chiplets[j]
            bx0, bx1 = b["x_mm"], b["x_mm"] + b["width_mm"]
            by0, by1 = b["y_mm"], b["y_mm"] + b["height_mm"]
            bcx, bcy = b["x_mm"] + b["width_mm"] / 2.0, b["y_mm"] + b["height_mm"] / 2.0
            gap_x = max(0.0, bx0 - ax1, ax0 - bx1)
            gap_y = max(0.0, by0 - ay1, ay0 - by1)
            edge_distances.append(math.hypot(gap_x, gap_y))
            center_distances.append(math.hypot(acx - bcx, acy - bcy))
    return np.asarray(edge_distances, dtype=np.float64), np.asarray(center_distances, dtype=np.float64)


def fraction_near_edges(chiplets: list[dict[str, float]], width: float, height: float) -> float:
    if not chiplets:
        return 0.0
    count = 0
    threshold = 2.0
    for c in chiplets:
        dist = min(c["x_mm"], c["y_mm"], width - (c["x_mm"] + c["width_mm"]), height - (c["y_mm"] + c["height_mm"]))
        if dist <= threshold:
            count += 1
    return count / len(chiplets)


def has_overlap(chiplets: list[dict[str, float]]) -> bool:
    for i in range(len(chiplets)):
        a = chiplets[i]
        for j in range(i + 1, len(chiplets)):
            b = chiplets[j]
            if not (
                a["x_mm"] + a["width_mm"] <= b["x_mm"]
                or b["x_mm"] + b["width_mm"] <= a["x_mm"]
                or a["y_mm"] + a["height_mm"] <= b["y_mm"]
                or b["y_mm"] + b["height_mm"] <= a["y_mm"]
            ):
                return True
    return False


def has_clipping(chiplets: list[dict[str, float]], width: float, height: float) -> bool:
    return any(c["x_mm"] < 0 or c["y_mm"] < 0 or c["x_mm"] + c["width_mm"] > width or c["y_mm"] + c["height_mm"] > height for c in chiplets)


def chiplet_summary(chiplets: list[dict[str, float]]) -> str:
    parts = []
    for chiplet in chiplets:
        parts.append(
            f"{chiplet['name']}:{chiplet['width_mm']:.1f}x{chiplet['height_mm']:.1f}mm,{chiplet['power_W']:.1f}W"
        )
    return "; ".join(parts)


def error_concentration(record: dict[str, Any]) -> str:
    candidates = {
        "occupied": record.get("occupied_mae_K"),
        "unoccupied": record.get("unoccupied_mae_K"),
        "hotspot_top5": max(record.get("seed0_hotspot_top_5pct_mae_K", 0), record.get("seed1_hotspot_top_5pct_mae_K", 0)),
        "centered_field": max(record.get("seed0_centered_field_mae_K", 0), record.get("seed1_centered_field_mae_K", 0)),
    }
    return max(candidates, key=lambda key: float(candidates[key] or 0.0))


def max_error_location(pred: np.ndarray, target: np.ndarray) -> str:
    error = np.abs(pred - target)
    row, col = np.unravel_index(int(np.argmax(error)), error.shape)
    return f"row={row},col={col},abs_error_K={float(error[row, col]):.3f}"


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    error = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(error * error)))


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    flat = values.reshape(-1)
    k = max(1, int(math.ceil(flat.size * fraction)))
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[idx] = True
    return mask.reshape(values.shape)


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    for candidate in (REPO_ROOT / path, base / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def row_hash(row: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for key in ("sample_uid", "x_path", "y_path", "prediction_path"):
        digest.update(str(row.get(key, "")).encode("utf-8"))
    return digest.hexdigest()


def padded(values: np.ndarray) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.05, 1.0e-6)
    return lo - pad, hi + pad


def case_colors() -> dict[str, tuple[int, int, int]]:
    return {
        "case01": (76, 114, 176),
        "case02": (196, 78, 82),
        "case03": (85, 168, 104),
        "case04": (129, 114, 179),
        "case05": (147, 120, 96),
        "case06": (218, 139, 195),
        "case07": (140, 140, 140),
        "case08": (204, 185, 116),
        "case09": (100, 181, 205),
        "case10": (221, 132, 82),
    }


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU for diagnostics")
        return torch.device("cpu")
    return device


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    raise SystemExit(main())
