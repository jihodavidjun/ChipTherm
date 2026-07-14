#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate, collate_graphs
from chiptherm.ml.graph_models import move_graph_to_device, normalize_graph_batch
from chiptherm.ml.models import build_model
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input, unnormalize_residual


SAMPLE_COLUMNS = [
    "dataset_index",
    "sample_uid",
    "case_id",
    "dataset_source",
    "total_power_W",
    "hotspot_mean_K",
    "hotspot_max_K",
    "physics_mae_K",
    "physics_rmse_K",
    "physics_max_abs_error_K",
    "physics_mean_signed_error_K",
    "cnn_mae_K",
    "cnn_rmse_K",
    "cnn_max_abs_error_K",
    "cnn_mean_signed_error_K",
    "mae_improvement_percent",
    "rmse_improvement_percent",
    "hotspot_temp_error_K",
    "hotspot_location_error_cells",
    "occupied_mae_K",
    "unoccupied_mae_K",
    "boundary_mae_K",
    "non_boundary_mae_K",
    "hotspot_top_1pct_mae_K",
    "hotspot_top_5pct_mae_K",
    "hotspot_top_10pct_mae_K",
    "power_top_5pct_mae_K",
    "power_top_10pct_mae_K",
    "mean_rise_error_K",
    "mean_rise_abs_error_K",
    "centered_field_mae_K",
    "centered_field_rmse_K",
    "mean_bias_removed_mae_K",
    "mean_bias_removed_rmse_K",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze residual CNN error modes for ChipTherm.")
    parser.add_argument("--checkpoint", default=REPO_ROOT / "outputs/residual_cnn_v2_base32/checkpoints/best.pt", type=Path)
    parser.add_argument("--index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/test_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/residual_cnn_v2_base32/error_analysis", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = NormalizationStats(**checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model_info = architecture_info(checkpoint["model_config"])

    dataset = ChipThermDataset(args.index, target="residual", return_metadata=True, return_graph=bool(model_info.get("graph_enabled")))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=chiptherm_collate if bool(model_info.get("graph_enabled")) else None,
    )

    records, all_cnn_errors, regional_sums = analyze(model, loader, stats, device, model_info)
    by_case = aggregate_by_case(records)
    selected = select_samples(records, seed=args.seed)
    summary = build_summary(args, records, by_case, regional_sums, selected)

    write_json(out_dir / "summary.json", summary)
    write_sample_metrics(out_dir / "sample_metrics.csv", records)
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case)
    write_plots(out_dir, records, by_case, all_cnn_errors, regional_sums)
    write_sample_panels(out_dir / "samples", dataset, model, stats, device, selected, model_info)

    overall = summary["overall"]
    worst_case_id = max(by_case.items(), key=lambda item: item[1]["cnn_mae_K"])[0]
    print("Residual CNN error analysis complete")
    print(f"Samples: {len(records)}")
    print(f"Overall CNN MAE/RMSE: {overall['cnn_mae_K']:.3f} / {overall['cnn_rmse_K']:.3f} K")
    print(f"Worst case by MAE: {worst_case_id} ({by_case[worst_case_id]['cnn_mae_K']:.3f} K)")
    print(
        "Occupied vs unoccupied MAE: "
        f"{overall['occupied_mae_K']:.3f} / {overall['unoccupied_mae_K']:.3f} K"
    )
    print(
        "Boundary vs non-boundary MAE: "
        f"{overall['boundary_mae_K']:.3f} / {overall['non_boundary_mae_K']:.3f} K"
    )
    print(
        "Hotspot-region MAE top 1/5/10%: "
        f"{overall['hotspot_top_1pct_mae_K']:.3f} / "
        f"{overall['hotspot_top_5pct_mae_K']:.3f} / "
        f"{overall['hotspot_top_10pct_mae_K']:.3f} K"
    )
    print("Top 5 worst samples:")
    for record in sorted(records, key=lambda item: item["cnn_mae_K"], reverse=True)[:5]:
        print(f"  {record['sample_uid']} {record['case_id']} MAE={record['cnn_mae_K']:.3f} K")
    print(f"Output: {out_dir}")
    return 0


def architecture_info(model_config: dict[str, Any]) -> dict[str, Any]:
    architecture = str(model_config.get("architecture", "miniunet"))
    physics_input_mode = str(model_config.get("physics_input_mode", "v1"))
    if physics_input_mode not in {"v1", "none", "gated_v1", "source_superposition_v1"}:
        raise ValueError(f"unsupported physics_input_mode: {physics_input_mode}")
    return {
        "architecture": architecture,
        "conditioned": architecture in {
            "miniunet_refine_conditioned",
            "miniunet_refine_conditioned_decomposed",
            "miniunet_refine_conditioned_decomposed_graph",
            "miniunet_refine_conditioned_decomposed_pairwise",
            "miniunet_refine_conditioned_decomposed_pairwise_basis",
        },
        "decomposed": architecture in {
            "miniunet_refine_decomposed",
            "miniunet_refine_conditioned_decomposed",
            "miniunet_refine_conditioned_decomposed_graph",
            "miniunet_refine_conditioned_decomposed_pairwise",
            "miniunet_refine_conditioned_decomposed_pairwise_basis",
        },
        "graph_enabled": architecture in {
            "miniunet_refine_conditioned_decomposed_graph",
            "miniunet_refine_conditioned_decomposed_pairwise",
            "miniunet_refine_conditioned_decomposed_pairwise_basis",
        },
        "graph_normalization": model_config.get("graph_normalization"),
        "metadata_dim": int(model_config.get("metadata_dim", 0) or 0),
        "physics_input_mode": physics_input_mode,
    }


@torch.no_grad()
def analyze(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    stats: NormalizationStats,
    device: torch.device,
    model_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, RunningRegion]]:
    records: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []
    regional_sums = {
        "occupied": RunningRegion(),
        "unoccupied": RunningRegion(),
        "boundary": RunningRegion(),
        "non_boundary": RunningRegion(),
        "hotspot_top_1pct": RunningRegion(),
        "hotspot_top_5pct": RunningRegion(),
        "hotspot_top_10pct": RunningRegion(),
        "power_top_5pct": RunningRegion(),
        "power_top_10pct": RunningRegion(),
    }
    dataset_offset = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        model_input = build_model_input(x, physics, stats, physics_input_mode=str(model_info.get("physics_input_mode", "v1")))
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        graph_batch = prepare_graph_batch(batch, bool(model_info.get("graph_enabled")), model_info.get("graph_normalization"), device)
        prediction = predict_temperature(model, model_input, physics, ambient, metadata_input, graph_batch, stats, model_info)
        pred_temperature = prediction["temperature"]

        x_np = x.detach().cpu().numpy()
        physics_np = physics.detach().cpu().numpy()
        temperature_np = temperature.detach().cpu().numpy()
        pred_np = pred_temperature.detach().cpu().numpy()
        mean_rise_pred_np = to_numpy_or_none(prediction.get("mean_rise"))
        centered_pred_np = to_numpy_or_none(prediction.get("centered_field"))
        metadata = batch["metadata"]
        batch_size = int(x_np.shape[0])
        sample_uids = metadata_values(metadata, "sample_uid", batch_size)
        case_ids = metadata_values(metadata, "case_id", batch_size)
        dataset_sources = metadata_values(metadata, "dataset_source", batch_size)
        total_powers = optional_float_values(metadata_values(metadata, "total_power_W", batch_size))

        for i in range(batch_size):
            x_i = x_np[i]
            y = temperature_np[i]
            phys = physics_np[i]
            pred = pred_np[i]
            cnn_error = pred - y
            physics_error = phys - y
            abs_error = np.abs(cnn_error)
            all_errors.append(cnn_error.reshape(-1).astype(np.float32, copy=True))

            occupancy = x_i[1] > 0.5
            boundary = boundary_mask_4_neighbor(occupancy)
            non_boundary = ~boundary
            hotspot_top_1 = top_fraction_mask(y, 0.01)
            hotspot_top_5 = top_fraction_mask(y, 0.05)
            hotspot_top_10 = top_fraction_mask(y, 0.10)
            power_positive = x_i[0] > 0.0
            power_top_5 = top_fraction_mask(x_i[0], 0.05, allowed=power_positive)
            power_top_10 = top_fraction_mask(x_i[0], 0.10, allowed=power_positive)

            region_masks = {
                "occupied": occupancy,
                "unoccupied": ~occupancy,
                "boundary": boundary,
                "non_boundary": non_boundary,
                "hotspot_top_1pct": hotspot_top_1,
                "hotspot_top_5pct": hotspot_top_5,
                "hotspot_top_10pct": hotspot_top_10,
                "power_top_5pct": power_top_5,
                "power_top_10pct": power_top_10,
            }
            for name, mask in region_masks.items():
                regional_sums[name].update(abs_error, mask)

            physics_stats = error_stats(phys, y)
            cnn_stats = error_stats(pred, y)
            ambient_i = float(ambient.detach().cpu().numpy()[i])
            mean_rise_target = float((y - ambient_i).mean())
            centered_target = y - float(y.mean())
            if mean_rise_pred_np is not None and centered_pred_np is not None:
                mean_rise_error = float(mean_rise_pred_np[i] - mean_rise_target)
                centered_stats = error_stats(centered_pred_np[i], centered_target)
                mean_bias_removed_stats = centered_stats
            else:
                mean_rise_error = None
                centered_stats = {}
                mean_bias_removed_stats = {}
            pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
            true_hotspot = np.unravel_index(int(np.argmax(y)), y.shape)
            record = {
                "dataset_index": dataset_offset + i,
                "sample_uid": str(sample_uids[i]),
                "case_id": str(case_ids[i]),
                "dataset_source": str(dataset_sources[i]),
                "total_power_W": total_powers[i] if i < len(total_powers) else None,
                "hotspot_mean_K": float(y.mean()),
                "hotspot_max_K": float(y.max()),
                "physics_mae_K": physics_stats["mae_K"],
                "physics_rmse_K": physics_stats["rmse_K"],
                "physics_max_abs_error_K": physics_stats["max_abs_error_K"],
                "physics_mean_signed_error_K": physics_stats["mean_signed_error_K"],
                "cnn_mae_K": cnn_stats["mae_K"],
                "cnn_rmse_K": cnn_stats["rmse_K"],
                "cnn_max_abs_error_K": cnn_stats["max_abs_error_K"],
                "cnn_mean_signed_error_K": cnn_stats["mean_signed_error_K"],
                "mae_improvement_percent": percent_improvement(physics_stats["mae_K"], cnn_stats["mae_K"]),
                "rmse_improvement_percent": percent_improvement(physics_stats["rmse_K"], cnn_stats["rmse_K"]),
                "hotspot_temp_error_K": float(pred[pred_hotspot] - y[true_hotspot]),
                "hotspot_location_error_cells": float(
                    ((pred_hotspot[0] - true_hotspot[0]) ** 2 + (pred_hotspot[1] - true_hotspot[1]) ** 2) ** 0.5
                ),
                "occupied_mae_K": masked_mae(abs_error, occupancy),
                "unoccupied_mae_K": masked_mae(abs_error, ~occupancy),
                "chiplet_boundary_mae_K": masked_mae(abs_error, boundary),
                "boundary_mae_K": masked_mae(abs_error, boundary),
                "non_boundary_mae_K": masked_mae(abs_error, non_boundary),
                "hotspot_top_1pct_mae_K": masked_mae(abs_error, hotspot_top_1),
                "hotspot_top_5pct_mae_K": masked_mae(abs_error, hotspot_top_5),
                "hotspot_top_10pct_mae_K": masked_mae(abs_error, hotspot_top_10),
                "power_top_5pct_mae_K": masked_mae(abs_error, power_top_5),
                "power_top_10pct_mae_K": masked_mae(abs_error, power_top_10),
                "mean_rise_error_K": mean_rise_error,
                "mean_rise_abs_error_K": abs(mean_rise_error) if mean_rise_error is not None else None,
                "centered_field_mae_K": centered_stats.get("mae_K"),
                "centered_field_rmse_K": centered_stats.get("rmse_K"),
                "mean_bias_removed_mae_K": mean_bias_removed_stats.get("mae_K"),
                "mean_bias_removed_rmse_K": mean_bias_removed_stats.get("rmse_K"),
            }
            records.append(record)
        dataset_offset += batch_size

    return records, np.concatenate(all_errors).astype(np.float64, copy=False), regional_sums


def predict_temperature(
    model: nn.Module,
    model_input: torch.Tensor,
    physics: torch.Tensor,
    ambient: torch.Tensor,
    metadata_input: torch.Tensor | None,
    graph_batch: dict[str, torch.Tensor] | None,
    stats: NormalizationStats,
    model_info: dict[str, Any],
) -> dict[str, torch.Tensor]:
    conditioned = bool(model_info["conditioned"])
    decomposed = bool(model_info["decomposed"])
    if conditioned and metadata_input is None:
        raise ValueError("conditioned checkpoint requires metadata tensor; build metadata_features.csv first")
    if decomposed:
        if bool(model_info.get("graph_enabled")):
            outputs = model(model_input, metadata_input, graph_batch)
        else:
            outputs = model(model_input, metadata_input) if conditioned else model(model_input)
        centered = outputs["centered_field"]
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        temperature = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered
        return {
            "temperature": temperature,
            "residual": temperature - physics,
            "mean_rise": outputs["mean_rise"],
            "centered_field": centered,
        }
    if hasattr(model, "forward_components"):
        if conditioned:
            pred_norm, _coarse_norm, _detail_norm = model.forward_components(model_input, metadata_input)
        else:
            pred_norm, _coarse_norm, _detail_norm = model.forward_components(model_input)
    else:
        pred_norm = model(model_input, metadata_input) if conditioned else model(model_input)
    pred_residual = unnormalize_residual(pred_norm.squeeze(1), stats)
    return {"temperature": physics + pred_residual, "residual": pred_residual}


def prepare_graph_batch(
    batch: dict[str, Any],
    graph_enabled: bool,
    graph_stats: Any | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if not graph_enabled:
        return None
    graph = batch.get("graph")
    if graph is None:
        raise ValueError("graph-enabled checkpoint requires graph_path artifacts in the analysis index")
    graph = move_graph_to_device(graph, device)
    return normalize_graph_batch(graph, graph_stats)


def to_numpy_or_none(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().numpy()


def error_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - target.astype(np.float64)
    abs_error = np.abs(error)
    return {
        "mae_K": float(abs_error.mean()),
        "rmse_K": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_K": float(abs_error.max()),
        "mean_signed_error_K": float(error.mean()),
    }


def boundary_mask_4_neighbor(occupancy: np.ndarray) -> np.ndarray:
    occ = occupancy.astype(bool)
    up = np.zeros_like(occ)
    down = np.zeros_like(occ)
    left = np.zeros_like(occ)
    right = np.zeros_like(occ)
    up[1:, :] = occ[:-1, :]
    down[:-1, :] = occ[1:, :]
    left[:, 1:] = occ[:, :-1]
    right[:, :-1] = occ[:, 1:]
    neighbor_count = up.astype(int) + down.astype(int) + left.astype(int) + right.astype(int)
    occupied_adjacent_empty = occ & (neighbor_count < 4)
    empty_adjacent_occupied = (~occ) & (neighbor_count > 0)
    return occupied_adjacent_empty | empty_adjacent_occupied


def top_fraction_mask(values: np.ndarray, fraction: float, allowed: np.ndarray | None = None) -> np.ndarray:
    if allowed is None:
        allowed = np.ones_like(values, dtype=bool)
    else:
        allowed = allowed.astype(bool)
    indices = np.flatnonzero(allowed.reshape(-1))
    if len(indices) == 0:
        return np.zeros_like(values, dtype=bool)
    count = max(1, int(math.ceil(len(indices) * fraction)))
    flat = values.reshape(-1)
    selected = indices[np.argpartition(flat[indices], -count)[-count:]]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[selected] = True
    return mask.reshape(values.shape)


def masked_mae(abs_error: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(abs_error[mask].mean())


class RunningRegion:
    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.count = 0

    def update(self, abs_error: np.ndarray, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        self.sum_abs += float(abs_error[mask].sum())
        self.count += int(mask.sum())

    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.sum_abs / self.count


def aggregate_by_case(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    return {case_id: aggregate_records(items) for case_id, items in sorted(by_case.items())}


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    keys_mean = [
        "physics_mae_K",
        "physics_rmse_K",
        "physics_mean_signed_error_K",
        "cnn_mae_K",
        "cnn_rmse_K",
        "cnn_mean_signed_error_K",
        "mae_improvement_percent",
        "rmse_improvement_percent",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "occupied_mae_K",
        "unoccupied_mae_K",
        "boundary_mae_K",
        "non_boundary_mae_K",
        "hotspot_top_1pct_mae_K",
        "hotspot_top_5pct_mae_K",
        "hotspot_top_10pct_mae_K",
        "power_top_5pct_mae_K",
        "power_top_10pct_mae_K",
        "mean_rise_abs_error_K",
        "centered_field_mae_K",
        "centered_field_rmse_K",
        "mean_bias_removed_mae_K",
        "mean_bias_removed_rmse_K",
    ]
    result: dict[str, float] = {"num_samples": float(len(records))}
    for key in keys_mean:
        result[key] = mean_optional(record.get(key) for record in records)
    result["physics_max_abs_error_K"] = max(float(record["physics_max_abs_error_K"]) for record in records)
    result["cnn_max_abs_error_K"] = max(float(record["cnn_max_abs_error_K"]) for record in records)
    result["hotspot_mean_K"] = mean_optional(record.get("hotspot_mean_K") for record in records)
    result["hotspot_max_K"] = mean_optional(record.get("hotspot_max_K") for record in records)
    result["total_power_W"] = mean_optional(record.get("total_power_W") for record in records)
    return result


def build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    by_case: dict[str, dict[str, float]],
    regional_sums: dict[str, RunningRegion],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = aggregate_records(records)
    overall.update(
        {
            "occupied_mae_K": regional_sums["occupied"].mean(),
            "unoccupied_mae_K": regional_sums["unoccupied"].mean(),
            "boundary_mae_K": regional_sums["boundary"].mean(),
            "non_boundary_mae_K": regional_sums["non_boundary"].mean(),
            "hotspot_top_1pct_mae_K": regional_sums["hotspot_top_1pct"].mean(),
            "hotspot_top_5pct_mae_K": regional_sums["hotspot_top_5pct"].mean(),
            "hotspot_top_10pct_mae_K": regional_sums["hotspot_top_10pct"].mean(),
            "power_top_5pct_mae_K": regional_sums["power_top_5pct"].mean(),
            "power_top_10pct_mae_K": regional_sums["power_top_10pct"].mean(),
        }
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "index": str(args.index.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "num_samples": len(records),
        "overall": overall,
        "metrics_by_case": by_case,
        "selected_samples": [
            {
                "criteria": record["criteria"],
                "sample_uid": record["sample_uid"],
                "case_id": record["case_id"],
                "cnn_mae_K": record["cnn_mae_K"],
                "cnn_rmse_K": record["cnn_rmse_K"],
                "mean_rise_abs_error_K": record.get("mean_rise_abs_error_K"),
                "centered_field_mae_K": record.get("centered_field_mae_K"),
            }
            for record in selected
        ],
        "analysis_notes": {
            "boundary_definition": "4-neighbor occupancy transition: occupied adjacent to empty, or empty adjacent to occupied.",
            "top_power_regions": "Top fraction is computed among positive power-density cells.",
            "cnn_error_sign": "CNN signed error is T_pred - HotSpot.",
        },
    }


def select_samples(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda item: item["cnn_mae_K"])
    selected: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any], criteria: str) -> None:
        item = selected.get(record["sample_uid"])
        if item is None:
            item = dict(record)
            item["criteria"] = criteria
            selected[record["sample_uid"]] = item
        else:
            item["criteria"] += f"+{criteria}"

    add(sorted_records[0], "best")
    add(sorted_records[len(sorted_records) // 2], "median")
    add(sorted_records[-1], "worst")
    add(sorted_records[-1], "worst_final_temperature")
    mean_records = [record for record in records if record.get("mean_rise_abs_error_K") is not None]
    if mean_records:
        add(max(mean_records, key=lambda item: float(item["mean_rise_abs_error_K"])), "worst_mean_rise")
    centered_records = [record for record in records if record.get("centered_field_mae_K") is not None]
    if centered_records:
        add(max(centered_records, key=lambda item: float(item["centered_field_mae_K"])), "worst_centered_field")
    add(max(records, key=lambda item: float(item["hotspot_top_5pct_mae_K"])), "worst_hotspot_region")
    case02_records = [record for record in records if record["case_id"] == "case02"]
    if case02_records:
        add(max(case02_records, key=lambda item: float(item["cnn_mae_K"])), "worst_case02")
    rng = random.Random(seed)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    for case_id in sorted(by_case):
        add(rng.choice(by_case[case_id]), f"random_{case_id}")
    return sorted(selected.values(), key=lambda item: (item["criteria"], item["case_id"], item["sample_uid"]))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sample_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in SAMPLE_COLUMNS})


def write_case_metrics(path: Path, by_case: dict[str, dict[str, float]]) -> None:
    columns = [
        "case_id",
        "num_samples",
        "physics_mae_K",
        "physics_rmse_K",
        "cnn_mae_K",
        "cnn_rmse_K",
        "cnn_max_abs_error_K",
        "cnn_mean_signed_error_K",
        "mae_improvement_percent",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "occupied_mae_K",
        "unoccupied_mae_K",
        "boundary_mae_K",
        "non_boundary_mae_K",
        "hotspot_top_1pct_mae_K",
        "hotspot_top_5pct_mae_K",
        "hotspot_top_10pct_mae_K",
        "power_top_5pct_mae_K",
        "power_top_10pct_mae_K",
        "mean_rise_abs_error_K",
        "centered_field_mae_K",
        "centered_field_rmse_K",
        "mean_bias_removed_mae_K",
        "mean_bias_removed_rmse_K",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(by_case.items()):
            row = {"case_id": case_id}
            row.update({column: metrics.get(column) for column in columns if column != "case_id"})
            writer.writerow(row)


def write_plots(
    out_dir: Path,
    records: list[dict[str, Any]],
    by_case: dict[str, dict[str, float]],
    all_cnn_errors: np.ndarray,
    regional_sums: dict[str, RunningRegion],
) -> None:
    draw_histogram(all_cnn_errors, out_dir / "error_histogram.png", title="CNN Signed Error Histogram: T_pred - HotSpot")
    draw_bar_chart({case: metrics["cnn_mae_K"] for case, metrics in by_case.items()}, out_dir / "mae_by_case.png", "CNN MAE by Case", "MAE (K)")
    draw_bar_chart({case: metrics["cnn_rmse_K"] for case, metrics in by_case.items()}, out_dir / "rmse_by_case.png", "CNN RMSE by Case", "RMSE (K)")
    draw_scatter(records, "total_power_W", "cnn_mae_K", out_dir / "mae_vs_total_power.png", "MAE vs Total Power", "Total power (W)", "MAE (K)")
    draw_scatter(records, "hotspot_max_K", "cnn_mae_K", out_dir / "mae_vs_hotspot_max.png", "MAE vs HotSpot Max", "HotSpot max (K)", "MAE (K)")
    draw_scatter(records, "physics_mae_K", "cnn_mae_K", out_dir / "mae_vs_physics_mae.png", "CNN MAE vs Physics MAE", "Physics MAE (K)", "CNN MAE (K)")
    draw_scatter(records, "hotspot_max_K", "hotspot_temp_error_K", out_dir / "hotspot_pred_vs_true.png", "Hotspot Temp Error vs True Hotspot", "True hotspot (K)", "Pred hotspot - true hotspot (K)")
    draw_scatter(records, "hotspot_mean_K", "cnn_mean_signed_error_K", out_dir / "mean_temp_pred_vs_true.png", "Mean Temp Error vs True Mean Temp", "True mean temp (K)", "Mean signed error (K)")
    draw_two_bar(
        "Occupied vs Unoccupied MAE",
        {"occupied": regional_sums["occupied"].mean(), "unoccupied": regional_sums["unoccupied"].mean()},
        out_dir / "occupied_vs_unoccupied_mae.png",
    )
    draw_two_bar(
        "Boundary vs Non-Boundary MAE",
        {"boundary": regional_sums["boundary"].mean(), "non-boundary": regional_sums["non_boundary"].mean()},
        out_dir / "boundary_vs_nonboundary_mae.png",
    )


@torch.no_grad()
def write_sample_panels(
    samples_dir: Path,
    dataset: ChipThermDataset,
    model: nn.Module,
    stats: NormalizationStats,
    device: torch.device,
    selected: list[dict[str, Any]],
    model_info: dict[str, Any],
) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        sample = dataset[int(record["dataset_index"])]
        x = sample["x"].unsqueeze(0).to(device)
        physics = sample["physics"].unsqueeze(0).to(device)
        ambient = sample["ambient_K"].view(1).to(device)
        y = sample["temperature"].cpu().numpy()
        model_input = build_model_input(x, physics, stats, physics_input_mode=str(model_info.get("physics_input_mode", "v1")))
        metadata_input = None
        if "metadata_vector" in sample:
            metadata_input = build_metadata_input(sample["metadata_vector"].unsqueeze(0), stats)
            if metadata_input is not None:
                metadata_input = metadata_input.to(device)
        graph_batch = None
        if bool(model_info.get("graph_enabled")):
            graph_batch = collate_graphs([sample["graph"]])
            graph_batch = prepare_graph_batch({"graph": graph_batch}, True, model_info.get("graph_normalization"), device)
        pred = predict_temperature(model, model_input, physics, ambient, metadata_input, graph_batch, stats, model_info)["temperature"]
        draw_sample_panel(
            sample,
            pred.squeeze(0).detach().cpu().numpy(),
            record,
            samples_dir / f"{sanitize(record['criteria'])}.png",
        )


def draw_sample_panel(sample: dict[str, Any], pred: np.ndarray, record: dict[str, Any], path: Path) -> None:
    x = sample["x"].cpu().numpy()
    y = sample["temperature"].cpu().numpy()
    physics = sample["physics"].cpu().numpy()
    error = pred - y
    abs_error = np.abs(error)
    power = x[0]
    occupancy = x[1]
    temp_min = float(min(y.min(), physics.min(), pred.min()))
    temp_max = float(max(y.max(), physics.max(), pred.max()))
    signed_abs = float(max(abs(error.min()), abs(error.max()), 1.0))
    panels = [
        ("Power density", power, (float(power.min()), float(power.max())), "power"),
        ("Occupancy", occupancy, (0.0, 1.0), "gray"),
        ("HotSpot", y, (temp_min, temp_max), "thermal"),
        ("Physics", physics, (temp_min, temp_max), "thermal"),
        ("CNN prediction", pred, (temp_min, temp_max), "thermal"),
        ("CNN signed error", error, (-signed_abs, signed_abs), "diverging"),
        ("CNN absolute error", abs_error, (0.0, float(max(abs_error.max(), 1.0))), "error"),
    ]
    panel_w = 205
    panel_h = 250
    margin = 22
    header_h = 88
    image = new_canvas(margin * 2 + panel_w * len(panels), header_h + panel_h + 35)
    draw = ImageDraw.Draw(image)
    font = default_font()
    title = (
        f"{record['criteria']} | {record['sample_uid']} | {record['case_id']} | "
        f"MAE {record['cnn_mae_K']:.2f} K | RMSE {record['cnn_rmse_K']:.2f} K"
    )
    draw.text((margin, 22), title, fill=(20, 20, 20), font=font)
    for i, (label, array, limits, cmap) in enumerate(panels):
        x0 = margin + i * panel_w
        y0 = header_h
        draw.text((x0, y0 - 22), label, fill=(20, 20, 20), font=font)
        heatmap = array_to_image(array, limits[0], limits[1], cmap).resize((165, 165), Image.Resampling.BILINEAR)
        image.paste(heatmap, (x0, y0))
        draw_colorbar(draw, image, (x0 + 172, y0, x0 + 187, y0 + 165), limits, cmap)
    image.save(path)


def draw_histogram(values: np.ndarray, path: Path, *, title: str) -> None:
    hist, edges = np.histogram(values, bins=100)
    image = new_canvas(1100, 720)
    draw = ImageDraw.Draw(image)
    plot = (90, 90, 1040, 610)
    draw_title(draw, title)
    draw_axes(draw, plot)
    max_count = max(int(hist.max()), 1)
    for i, count in enumerate(hist):
        x0 = plot[0] + int(i * (plot[2] - plot[0]) / len(hist))
        x1 = plot[0] + int((i + 1) * (plot[2] - plot[0]) / len(hist)) - 1
        y0 = plot[3] - int(count * (plot[3] - plot[1]) / max_count)
        draw.rectangle((x0, y0, x1, plot[3]), fill=(76, 114, 176))
    font = default_font()
    draw.text((plot[0], plot[3] + 18), f"{edges[0]:.2f} K", fill=(20, 20, 20), font=font)
    draw.text((plot[2] - 85, plot[3] + 18), f"{edges[-1]:.2f} K", fill=(20, 20, 20), font=font)
    image.save(path)


def draw_bar_chart(values: dict[str, float], path: Path, title: str, ylabel: str) -> None:
    image = new_canvas(1100, 720)
    draw = ImageDraw.Draw(image)
    font = default_font()
    plot = (90, 90, 1040, 610)
    draw_title(draw, title)
    draw_axes(draw, plot)
    labels = list(values)
    vals = [float(values[label]) for label in labels]
    max_value = max(vals) if vals else 1.0
    gap = 12
    width = max(12, int((plot[2] - plot[0] - gap * (len(vals) + 1)) / max(len(vals), 1)))
    for i, (label, value) in enumerate(zip(labels, vals)):
        x0 = plot[0] + gap + i * (width + gap)
        y0 = plot[3] - int(value / max_value * (plot[3] - plot[1]))
        draw.rectangle((x0, y0, x0 + width, plot[3]), fill=(221, 132, 82))
        draw.text((x0, plot[3] + 18), label, fill=(20, 20, 20), font=font)
        draw.text((x0, y0 - 18), f"{value:.1f}", fill=(20, 20, 20), font=font)
    draw.text((18, 90), ylabel, fill=(20, 20, 20), font=font)
    image.save(path)


def draw_two_bar(title: str, values: dict[str, float | None], path: Path) -> None:
    draw_bar_chart({key: float(value or 0.0) for key, value in values.items()}, path, title, "MAE (K)")


def draw_scatter(records: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    points = [(float(record[x_key]), float(record[y_key]), record["case_id"]) for record in records if record.get(x_key) is not None and record.get(y_key) is not None]
    image = new_canvas(1100, 720)
    draw = ImageDraw.Draw(image)
    font = default_font()
    plot = (100, 90, 1040, 610)
    draw_title(draw, title)
    draw_axes(draw, plot)
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = padded_range(xs)
        ymin, ymax = padded_range(ys)
        colors = case_colors()
        for x, y, case_id in points:
            px = scale_value(x, xmin, xmax, plot[0], plot[2])
            py = scale_value(y, ymin, ymax, plot[3], plot[1])
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=colors.get(str(case_id), (80, 80, 80)))
        draw.text((plot[0], plot[3] + 8), f"{xmin:.1f}", fill=(20, 20, 20), font=font)
        draw.text((plot[2] - 55, plot[3] + 8), f"{xmax:.1f}", fill=(20, 20, 20), font=font)
        draw.text((plot[0] - 75, plot[3] - 8), f"{ymin:.1f}", fill=(20, 20, 20), font=font)
        draw.text((plot[0] - 75, plot[1] - 8), f"{ymax:.1f}", fill=(20, 20, 20), font=font)
    draw.text((plot[0], plot[3] + 28), xlabel, fill=(20, 20, 20), font=font)
    draw.text((18, 90), ylabel, fill=(20, 20, 20), font=font)
    image.save(path)


def array_to_image(array: np.ndarray, vmin: float, vmax: float, cmap: str) -> Image.Image:
    if vmax <= vmin:
        vmax = vmin + 1.0
    t = np.clip((array.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    return Image.fromarray(colormap(t, cmap).astype(np.uint8))


def colormap(t: np.ndarray, cmap: str) -> np.ndarray:
    if cmap == "gray":
        v = (t * 255.0)[..., None]
        return np.repeat(v, 3, axis=-1)
    if cmap == "diverging":
        blue = np.array([58, 108, 178], dtype=np.float64)
        white = np.array([246, 246, 246], dtype=np.float64)
        red = np.array([190, 64, 54], dtype=np.float64)
        rgb = np.empty(t.shape + (3,), dtype=np.float64)
        low = t <= 0.5
        rgb[low] = lerp(blue, white, (t[low] / 0.5)[..., None])
        rgb[~low] = lerp(white, red, ((t[~low] - 0.5) / 0.5)[..., None])
        return rgb
    if cmap == "thermal":
        return multi_lerp(t, [(42, 72, 160), (70, 170, 210), (250, 220, 90), (190, 45, 35)])
    if cmap == "error":
        return multi_lerp(t, [(255, 255, 245), (245, 170, 70), (165, 35, 35)])
    return multi_lerp(t, [(245, 245, 245), (230, 190, 80), (180, 55, 35)])


def multi_lerp(t: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    anchors = np.array(colors, dtype=np.float64)
    scaled = np.clip(t, 0.0, 1.0) * (len(colors) - 1)
    idx = np.minimum(np.floor(scaled).astype(int), len(colors) - 2)
    frac = (scaled - idx)[..., None]
    return lerp(anchors[idx], anchors[idx + 1], frac)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def draw_colorbar(draw: ImageDraw.ImageDraw, image: Image.Image, box: tuple[int, int, int, int], limits: tuple[float, float], cmap: str) -> None:
    x0, y0, x1, y1 = box
    values = np.linspace(1.0, 0.0, max(y1 - y0, 1)).reshape(-1, 1)
    bar = Image.fromarray(colormap(values, cmap).astype(np.uint8).repeat(max(x1 - x0, 1), axis=1))
    image.paste(bar, (x0, y0))
    font = default_font()
    draw.rectangle(box, outline=(30, 30, 30), width=1)
    draw.text((x1 + 4, y0 - 4), f"{limits[1]:.1f}", fill=(20, 20, 20), font=font)
    draw.text((x1 + 4, y1 - 10), f"{limits[0]:.1f}", fill=(20, 20, 20), font=font)
    if limits[0] < 0.0 < limits[1]:
        zero_y = int(scale_value(0.0, limits[0], limits[1], y1, y0))
        draw.line((x0, zero_y, x1 + 4, zero_y), fill=(20, 20, 20), width=1)


def new_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(250, 250, 247))


def draw_title(draw: ImageDraw.ImageDraw, title: str) -> None:
    draw.text((40, 32), title, fill=(20, 20, 20), font=default_font())


def draw_axes(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int]) -> None:
    draw.rectangle(plot, outline=(35, 35, 35), width=2)


def scale_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> int:
    if src_max <= src_min:
        return int((dst_min + dst_max) / 2)
    t = (value - src_min) / (src_max - src_min)
    return int(dst_min + t * (dst_max - dst_min))


def padded_range(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    pad = max((hi - lo) * 0.05, 1.0)
    return lo - pad, hi + pad


def case_colors() -> dict[str, tuple[int, int, int]]:
    palette = [
        (76, 114, 176),
        (221, 132, 82),
        (85, 168, 104),
        (196, 78, 82),
        (129, 114, 179),
        (147, 120, 96),
        (218, 139, 195),
        (140, 140, 140),
        (204, 185, 116),
        (100, 181, 205),
    ]
    return {f"case{i:02d}": color for i, color in enumerate(palette, start=1)}


def default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def mean_optional(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None and value != ""]
    return float(sum(numeric) / len(numeric)) if numeric else float("nan")


def percent_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0
    return float((baseline - candidate) / baseline * 100.0)


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def optional_float_values(values: list[Any]) -> list[float | None]:
    result: list[float | None] = []
    for value in values:
        if value is None or value == "":
            result.append(None)
        else:
            result.append(float(value))
    return result


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested but is not available")
    return device


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    raise SystemExit(main())
