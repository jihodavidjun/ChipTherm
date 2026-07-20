#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.graph_models import chiplet_cell_weights
from chiptherm.ml.models import build_model, count_parameters
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input

from analyze_residual_cnn_errors import architecture_info
from evaluate_residual_cnn import (
    call_model,
    decomposed_targets,
    load_checkpoint,
    metadata_values,
    physics_input_channel_count,
    prepare_graph_batch,
    reconstruct_decomposed_temperature,
    select_device,
)


DEFAULT_DISTANCE_BINS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf")]
DEFAULT_GRADIENT_QUANTILES = [0.33, 0.66, 0.90]
DEFAULT_FREQUENCY_CUTOFFS = [0.08, 0.20]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research diagnostic for spatial ChipTherm temperature-field errors."
    )
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--index", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--boundary-width-cells", default=1, type=int)
    parser.add_argument("--package-edge-width-cells", default=3, type=int)
    parser.add_argument("--hotspot-radius-cells", default=3, type=int)
    parser.add_argument("--gradient-quantiles", default=DEFAULT_GRADIENT_QUANTILES, nargs="+", type=float)
    parser.add_argument("--top-k-worst", default=20, type=int)
    parser.add_argument("--frequency-cutoffs", default=DEFAULT_FREQUENCY_CUTOFFS, nargs="+", type=float)
    parser.add_argument("--max-samples", default=None, type=int, help="Optional lightweight smoke-test limit.")
    parser.add_argument("--save-sample-panels", action="store_true", help="Save PNG panels for worst samples.")
    parser.add_argument("--compare-audit-dir", default=None, type=Path)
    parser.add_argument("--comparison-out", default=None, type=Path)
    args = parser.parse_args()

    if args.compare_audit_dir is not None:
        out_path = args.comparison_out or args.out_dir / "comparison_report.md"
        write_comparison_report(args.compare_audit_dir, args.out_dir, out_path)
        print(f"Comparison report written: {out_path}")
        return 0
    if args.checkpoint is None or args.index is None:
        raise SystemExit("--checkpoint and --index are required unless --compare-audit-dir is used")

    validate_quantiles(args.gradient_quantiles)
    validate_cutoffs(args.frequency_cutoffs)
    out_dir = args.out_dir.resolve()
    create_output_tree(out_dir)

    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = NormalizationStats(**checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_info = architecture_info(checkpoint["model_config"])
    physical_representation = str(model_info["physical_representation"])
    graph_model_required = bool(model_info.get("graph_enabled"))
    physics_input_mode = str(model_info["physics_input_mode"])
    mean_head_mode = str(model_info["mean_head_mode"])
    graph_stats = model_info.get("graph_normalization")

    dataset = ChipThermDataset(
        args.index,
        target="residual",
        return_metadata=True,
        return_graph=True,
        physical_representation=physical_representation,
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise SystemExit("--max-samples must be positive")
        dataset.rows = dataset.rows[: args.max_samples]

    dataset_input_channels = int(dataset[0]["x"].shape[0])
    actual_input_channels = dataset_input_channels + physics_input_channel_count(physics_input_mode)
    expected_input_channels = int(checkpoint["model_config"].get("input_channels", actual_input_channels))
    if actual_input_channels != expected_input_channels:
        raise SystemExit(
            f"checkpoint expects {expected_input_channels} model input channels, "
            f"but dataset provides {actual_input_channels} channels with physics_input_mode={physics_input_mode}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=chiptherm_collate,
    )

    gradient_thresholds = collect_gradient_thresholds(dataset, args.gradient_quantiles)
    result = audit(
        model,
        loader,
        stats,
        device,
        args=args,
        model_info=model_info,
        graph_model_required=graph_model_required,
        graph_stats=graph_stats,
        gradient_thresholds=gradient_thresholds,
    )
    write_outputs(
        out_dir,
        result,
        args=args,
        checkpoint=checkpoint,
        model_info=model_info,
        dataset=dataset,
        parameter_count=count_parameters(model),
        gradient_thresholds=gradient_thresholds,
    )

    print("Spatial error audit complete")
    print(f"Samples: {result.sample_count}")
    print(
        "Final MAE/RMSE: "
        f"{result.global_final.mae():.3f} / {result.global_final.rmse():.3f} K"
    )
    if result.global_centered.count:
        print(
            "Centered-field MAE/RMSE: "
            f"{result.global_centered.mae():.3f} / {result.global_centered.rmse():.3f} K"
        )
    worst_case = max(result.case_stats.items(), key=lambda item: item[1].final.mae())[0]
    print(f"Worst case by final MAE: {worst_case} ({result.case_stats[worst_case].final.mae():.3f} K)")
    print(f"Output: {out_dir}")
    return 0


@dataclass
class ErrorAccumulator:
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_signed: float = 0.0
    count: int = 0
    sample_count: int = 0

    def update(self, error: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is None:
            values = error.reshape(-1).astype(np.float64, copy=False)
        else:
            mask = mask.astype(bool, copy=False)
            if not np.any(mask):
                return
            values = error[mask].reshape(-1).astype(np.float64, copy=False)
        self.sum_abs += float(np.abs(values).sum())
        self.sum_sq += float(np.square(values).sum())
        self.sum_signed += float(values.sum())
        self.count += int(values.size)
        self.sample_count += 1

    def mae(self) -> float:
        return self.sum_abs / self.count if self.count else float("nan")

    def rmse(self) -> float:
        return math.sqrt(self.sum_sq / self.count) if self.count else float("nan")

    def mean_signed(self) -> float:
        return self.sum_signed / self.count if self.count else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mae_K": self.mae(),
            "rmse_K": self.rmse(),
            "mean_signed_error_K": self.mean_signed(),
            "pixel_count": self.count,
            "sample_count": self.sample_count,
        }


@dataclass
class MapAccumulator:
    sum_signed_error: np.ndarray | None = None
    sum_abs_error: np.ndarray | None = None
    sum_sq_error: np.ndarray | None = None
    sum_true_centered: np.ndarray | None = None
    sum_pred_centered: np.ndarray | None = None
    sum_centered_diff: np.ndarray | None = None
    sum_true_temperature: np.ndarray | None = None
    sum_pred_temperature: np.ndarray | None = None
    true_hotspot_density: np.ndarray | None = None
    pred_hotspot_density: np.ndarray | None = None
    count: int = 0

    def update(
        self,
        *,
        final_error: np.ndarray,
        true_temp: np.ndarray,
        pred_temp: np.ndarray,
        true_centered: np.ndarray,
        pred_centered: np.ndarray,
    ) -> None:
        if self.sum_signed_error is None:
            shape = final_error.shape
            self.sum_signed_error = np.zeros(shape, dtype=np.float64)
            self.sum_abs_error = np.zeros(shape, dtype=np.float64)
            self.sum_sq_error = np.zeros(shape, dtype=np.float64)
            self.sum_true_centered = np.zeros(shape, dtype=np.float64)
            self.sum_pred_centered = np.zeros(shape, dtype=np.float64)
            self.sum_centered_diff = np.zeros(shape, dtype=np.float64)
            self.sum_true_temperature = np.zeros(shape, dtype=np.float64)
            self.sum_pred_temperature = np.zeros(shape, dtype=np.float64)
            self.true_hotspot_density = np.zeros(shape, dtype=np.float64)
            self.pred_hotspot_density = np.zeros(shape, dtype=np.float64)
        assert self.sum_signed_error is not None
        assert self.sum_abs_error is not None
        assert self.sum_sq_error is not None
        assert self.sum_true_centered is not None
        assert self.sum_pred_centered is not None
        assert self.sum_centered_diff is not None
        assert self.sum_true_temperature is not None
        assert self.sum_pred_temperature is not None
        assert self.true_hotspot_density is not None
        assert self.pred_hotspot_density is not None
        self.sum_signed_error += final_error
        self.sum_abs_error += np.abs(final_error)
        self.sum_sq_error += np.square(final_error)
        self.sum_true_centered += true_centered
        self.sum_pred_centered += pred_centered
        self.sum_centered_diff += pred_centered - true_centered
        self.sum_true_temperature += true_temp
        self.sum_pred_temperature += pred_temp
        self.true_hotspot_density[np.unravel_index(int(np.argmax(true_temp)), true_temp.shape)] += 1.0
        self.pred_hotspot_density[np.unravel_index(int(np.argmax(pred_temp)), pred_temp.shape)] += 1.0
        self.count += 1

    def mean_maps(self) -> dict[str, np.ndarray]:
        if self.count == 0 or self.sum_signed_error is None:
            return {}
        denom = float(self.count)
        assert self.sum_abs_error is not None
        assert self.sum_sq_error is not None
        assert self.sum_true_centered is not None
        assert self.sum_pred_centered is not None
        assert self.sum_centered_diff is not None
        assert self.sum_true_temperature is not None
        assert self.sum_pred_temperature is not None
        assert self.true_hotspot_density is not None
        assert self.pred_hotspot_density is not None
        return {
            "mean_signed_final_error_K": self.sum_signed_error / denom,
            "mean_absolute_final_error_K": self.sum_abs_error / denom,
            "rmse_final_error_K": np.sqrt(self.sum_sq_error / denom),
            "mean_true_centered_field_K": self.sum_true_centered / denom,
            "mean_predicted_centered_field_K": self.sum_pred_centered / denom,
            "mean_centered_field_difference_K": self.sum_centered_diff / denom,
            "mean_true_temperature_K": self.sum_true_temperature / denom,
            "mean_predicted_temperature_K": self.sum_pred_temperature / denom,
            "true_hotspot_density": self.true_hotspot_density / denom,
            "predicted_hotspot_density": self.pred_hotspot_density / denom,
        }


@dataclass
class CaseStats:
    final: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    centered: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    mean_rise: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    region_stats: dict[str, ErrorAccumulator] = field(default_factory=lambda: defaultdict(ErrorAccumulator))
    hotspot_temp_abs_error: list[float] = field(default_factory=list)
    hotspot_location_error: list[float] = field(default_factory=list)
    maps: MapAccumulator = field(default_factory=MapAccumulator)


@dataclass
class AuditResult:
    sample_count: int = 0
    global_final: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    global_centered: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    global_mean_rise: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    region_stats: dict[str, ErrorAccumulator] = field(default_factory=lambda: defaultdict(ErrorAccumulator))
    partition_stats: dict[str, ErrorAccumulator] = field(default_factory=lambda: defaultdict(ErrorAccumulator))
    distance_stats: dict[str, list[ErrorAccumulator]] = field(default_factory=dict)
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    chiplet_records: list[dict[str, Any]] = field(default_factory=list)
    case_stats: dict[str, CaseStats] = field(default_factory=lambda: defaultdict(CaseStats))
    frequency_stats: dict[str, ErrorAccumulator] = field(default_factory=lambda: defaultdict(ErrorAccumulator))
    frequency_energy: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    frequency_radial_energy: np.ndarray | None = None
    frequency_radial_count: np.ndarray | None = None
    worst_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    worst_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)


@torch.no_grad()
def audit(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    stats: NormalizationStats,
    device: torch.device,
    *,
    args: argparse.Namespace,
    model_info: dict[str, Any],
    graph_model_required: bool,
    graph_stats: Any | None,
    gradient_thresholds: list[float],
) -> AuditResult:
    result = AuditResult()
    distance_names = [
        "nearest_chiplet_boundary_cells",
        "nearest_occupied_cell_cells",
        "package_edge_cells",
        "true_hotspot_cells",
        "predicted_hotspot_cells",
    ]
    result.distance_stats = {
        name: [ErrorAccumulator() for _ in range(len(DEFAULT_DISTANCE_BINS) - 1)]
        for name in distance_names
    }
    radial_bins = np.linspace(0.0, 0.5 * math.sqrt(2.0), 65)
    result.frequency_radial_energy = np.zeros(len(radial_bins) - 1, dtype=np.float64)
    result.frequency_radial_count = np.zeros(len(radial_bins) - 1, dtype=np.float64)

    dataset_offset = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        physics_v1 = batch.get("physics_v1")
        if physics_v1 is not None:
            physics_v1 = physics_v1.to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        total_power = batch["total_power_W"].to(device, non_blocking=True).float()
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        if bool(model_info["conditioned"]) and metadata_input is None:
            raise ValueError("conditioned checkpoint requires metadata tensor")
        graph_batch = prepare_graph_batch(batch, graph_model_required, graph_stats, device)
        model_input = build_model_input(
            x,
            physics,
            stats,
            physics_input_mode=str(model_info["physics_input_mode"]),
            physics_v1=physics_v1,
        )
        outputs = call_model(
            model,
            model_input,
            metadata_input,
            graph_batch,
            conditioned=bool(model_info["conditioned"]),
            graph_enabled=graph_model_required,
            total_power_W=total_power,
        )
        if not isinstance(outputs, dict):
            raise ValueError("audit_spatial_errors requires decomposed checkpoints with dict outputs")
        targets = decomposed_targets(
            temperature,
            ambient,
            physics,
            total_power,
            mean_head_mode=str(model_info["mean_head_mode"]),
        )
        pred_temp = reconstruct_decomposed_temperature(
            outputs,
            ambient,
            physics,
            mean_head_mode=str(model_info["mean_head_mode"]),
        )
        centered_pred = outputs["centered_field"] - outputs["centered_field"].mean(dim=(-2, -1), keepdim=True)
        centered_target = targets["centered_field_K"]
        centered_mean_abs = centered_pred.mean(dim=(-2, -1)).abs().detach().cpu().numpy()
        if not np.all(centered_mean_abs < 1e-3):
            raise ValueError(f"centered prediction has non-zero mean: max {float(centered_mean_abs.max())}")

        batch_size = int(x.shape[0])
        metadata = batch["metadata"]
        sample_uids = metadata_values(metadata, "sample_uid", batch_size)
        case_ids = metadata_values(metadata, "case_id", batch_size)
        dataset_sources = metadata_values(metadata, "dataset_source", batch_size)
        graph_cpu = batch.get("graph")

        x_np = x.detach().cpu().numpy()
        physics_np = physics.detach().cpu().numpy()
        temp_np = temperature.detach().cpu().numpy()
        pred_np = pred_temp.detach().cpu().numpy()
        centered_pred_np = centered_pred.detach().cpu().numpy()
        centered_target_np = centered_target.detach().cpu().numpy()
        mean_pred_np = outputs["mean_rise"].detach().cpu().numpy()
        mean_target_np = targets["mean_correction_K"].detach().cpu().numpy()
        total_power_np = total_power.detach().cpu().numpy()

        for i in range(batch_size):
            case_id = str(case_ids[i])
            sample_uid = str(sample_uids[i])
            y = temp_np[i].astype(np.float64)
            pred = pred_np[i].astype(np.float64)
            phys = physics_np[i].astype(np.float64)
            final_error = pred - y
            centered_error = centered_pred_np[i].astype(np.float64) - centered_target_np[i].astype(np.float64)
            mean_error = float(mean_pred_np[i] - mean_target_np[i])
            occupancy = x_np[i, 1] > 0.5 if x_np.shape[1] > 1 else np.zeros_like(y, dtype=bool)
            masks = build_region_masks(
                occupancy,
                y,
                pred,
                args.boundary_width_cells,
                args.package_edge_width_cells,
                args.hotspot_radius_cells,
                gradient_thresholds,
            )
            validate_masks(masks, y.shape)

            result.global_final.update(final_error)
            result.global_centered.update(centered_error)
            result.global_mean_rise.update(np.asarray([mean_error], dtype=np.float64))
            case_stats = result.case_stats[case_id]
            case_stats.final.update(final_error)
            case_stats.centered.update(centered_error)
            case_stats.mean_rise.update(np.asarray([mean_error], dtype=np.float64))
            for region_name, mask in masks.independent.items():
                result.region_stats[region_name].update(final_error, mask)
                case_stats.region_stats[region_name].update(final_error, mask)
            for region_name, mask in masks.partition.items():
                result.partition_stats[region_name].update(final_error, mask)
            for dist_name, dist_map in distance_maps(masks, occupancy).items():
                update_distance_bins(result.distance_stats[dist_name], final_error, dist_map, DEFAULT_DISTANCE_BINS)

            true_hotspot = np.unravel_index(int(np.argmax(y)), y.shape)
            pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
            hotspot_temp_error = float(pred[pred_hotspot] - y[true_hotspot])
            hotspot_location_error = float(
                math.hypot(float(pred_hotspot[0] - true_hotspot[0]), float(pred_hotspot[1] - true_hotspot[1]))
            )
            case_stats.hotspot_temp_abs_error.append(abs(hotspot_temp_error))
            case_stats.hotspot_location_error.append(hotspot_location_error)
            case_stats.maps.update(
                final_error=final_error,
                true_temp=y,
                pred_temp=pred,
                true_centered=centered_target_np[i].astype(np.float64),
                pred_centered=centered_pred_np[i].astype(np.float64),
            )
            update_frequency_stats(
                result,
                centered_error,
                args.frequency_cutoffs,
                radial_bins,
            )

            graph_slice = graph_for_sample(graph_cpu, i) if graph_cpu is not None else None
            chiplet_stats = chiplet_descriptors(graph_slice, pred, y) if graph_slice is not None else []
            nearest_spacing = nearest_chiplet_spacing_mm(graph_slice) if graph_slice is not None else None
            chiplet_count = int(len(chiplet_stats)) if chiplet_stats else metadata_value(metadata, "num_chiplets", i, default=-1)
            sample_record = {
                "dataset_index": dataset_offset + i,
                "sample_uid": sample_uid,
                "case_id": case_id,
                "dataset_source": str(dataset_sources[i]),
                "final_mae_K": float(np.abs(final_error).mean()),
                "final_rmse_K": float(np.sqrt(np.mean(np.square(final_error)))),
                "final_mean_signed_error_K": float(final_error.mean()),
                "centered_field_mae_K": float(np.abs(centered_error).mean()),
                "centered_field_rmse_K": float(np.sqrt(np.mean(np.square(centered_error)))),
                "mean_rise_error_K": mean_error,
                "mean_rise_abs_error_K": abs(mean_error),
                "source_base_mae_K": float(np.abs(phys - y).mean()),
                "hotspot_temperature_error_K": hotspot_temp_error,
                "hotspot_temperature_abs_error_K": abs(hotspot_temp_error),
                "hotspot_location_error_cells": hotspot_location_error,
                "max_abs_pixel_error_K": float(np.abs(final_error).max()),
                "occupied_mae_K": masked_mae(final_error, masks.independent["occupied"]),
                "unoccupied_mae_K": masked_mae(final_error, masks.independent["unoccupied"]),
                "chiplet_interior_mae_K": masked_mae(final_error, masks.independent["chiplet_interior"]),
                "chiplet_boundary_band_mae_K": masked_mae(final_error, masks.independent["chiplet_boundary_band"]),
                "package_edge_mae_K": masked_mae(final_error, masks.independent["package_edge_band"]),
                "package_corner_mae_K": masked_mae(final_error, masks.independent["package_corners"]),
                "true_hotspot_neighborhood_mae_K": masked_mae(final_error, masks.independent["true_hotspot_neighborhood"]),
                "pred_hotspot_neighborhood_mae_K": masked_mae(final_error, masks.independent["predicted_hotspot_neighborhood"]),
                "high_gradient_mae_K": masked_mae(final_error, masks.independent["high_gradient"]),
                "total_power_W": float(total_power_np[i]),
                "chiplet_count": chiplet_count,
                "nearest_neighbor_chiplet_distance_mm": nearest_spacing,
                "true_mean_temperature_K": float(y.mean()),
                "true_peak_temperature_K": float(y.max()),
                "true_temperature_range_K": float(y.max() - y.min()),
                "true_spatial_std_K": float(y.std()),
            }
            sample_record.update(sample_descriptors_from_metadata(metadata, i))
            sample_record.update(sample_descriptors_from_channels(x_np[i], loader.dataset.channel_names))
            result.sample_records.append(sample_record)

            for chip_index, item in enumerate(chiplet_stats):
                chip_row = {
                    "sample_uid": sample_uid,
                    "case_id": case_id,
                    "chiplet_index": chip_index,
                    **item,
                }
                result.chiplet_records.append(chip_row)

            result.worst_payloads[sample_uid] = {
                "sample_uid": sample_uid,
                "case_id": case_id,
                "summary": sample_record,
                "arrays": {
                    "true_temperature_K": y.astype(np.float32),
                    "source_superposition_base_K": phys.astype(np.float32),
                    "predicted_temperature_K": pred.astype(np.float32),
                    "signed_error_K": final_error.astype(np.float32),
                    "absolute_error_K": np.abs(final_error).astype(np.float32),
                    "true_centered_field_K": centered_target_np[i].astype(np.float32),
                    "predicted_centered_field_K": centered_pred_np[i].astype(np.float32),
                    "occupancy_mask": occupancy.astype(np.float32),
                },
            }
        dataset_offset += batch_size
        result.sample_count += batch_size

    result.worst_samples = select_worst_samples(result.sample_records, args.top_k_worst)
    return result


@dataclass
class MaskBundle:
    independent: dict[str, np.ndarray]
    partition: dict[str, np.ndarray]
    boundary_core: np.ndarray
    true_hotspot: tuple[int, int]
    predicted_hotspot: tuple[int, int]


def build_region_masks(
    occupancy: np.ndarray,
    true_temperature: np.ndarray,
    pred_temperature: np.ndarray,
    boundary_width: int,
    edge_width: int,
    hotspot_radius: int,
    gradient_thresholds: list[float],
) -> MaskBundle:
    occupancy = occupancy.astype(bool)
    h, w = occupancy.shape
    boundary_core = boundary_mask(occupancy, max(1, boundary_width))
    eroded_occ = erode_bool(occupancy, max(0, boundary_width))
    package_edge = package_edge_mask(h, w, edge_width)
    corners = package_corner_mask(h, w, edge_width)
    true_hotspot = np.unravel_index(int(np.argmax(true_temperature)), true_temperature.shape)
    pred_hotspot = np.unravel_index(int(np.argmax(pred_temperature)), pred_temperature.shape)
    true_hotspot_mask = disk_mask(h, w, true_hotspot, hotspot_radius)
    pred_hotspot_mask = disk_mask(h, w, pred_hotspot, hotspot_radius)
    grad = gradient_magnitude(true_temperature)
    q1, q2, q3 = gradient_thresholds
    low_gradient = grad <= q1
    mid_gradient = (grad > q1) & (grad <= q2)
    high_gradient = grad > q3
    upper_mid_gradient = (grad > q2) & (grad <= q3)

    independent = {
        "full_grid": np.ones((h, w), dtype=bool),
        "occupied": occupancy,
        "unoccupied": ~occupancy,
        "chiplet_interior": occupancy & eroded_occ,
        "chiplet_boundary_band": boundary_core,
        "non_boundary": ~boundary_core,
        "package_edge_band": package_edge,
        "package_corners": corners,
        "true_hotspot_neighborhood": true_hotspot_mask,
        "predicted_hotspot_neighborhood": pred_hotspot_mask,
        "low_gradient": low_gradient,
        "medium_gradient": mid_gradient,
        "upper_mid_gradient": upper_mid_gradient,
        "high_gradient": high_gradient,
    }

    assigned = np.zeros((h, w), dtype=bool)
    partition: dict[str, np.ndarray] = {}
    for name, mask in [
        ("package_corners", corners),
        ("package_edge_non_corner", package_edge & ~corners),
        ("chiplet_boundary_band", boundary_core),
        ("chiplet_interior", occupancy & ~boundary_core),
        ("background_other", ~occupancy),
    ]:
        part = mask & ~assigned
        partition[name] = part
        assigned |= part
    partition["unassigned"] = ~assigned
    return MaskBundle(independent=independent, partition=partition, boundary_core=boundary_core, true_hotspot=true_hotspot, predicted_hotspot=pred_hotspot)


def validate_masks(masks: MaskBundle, shape: tuple[int, int]) -> None:
    for name, mask in {**masks.independent, **masks.partition}.items():
        if mask.shape != shape:
            raise ValueError(f"mask {name} has shape {mask.shape}, expected {shape}")
    occupied = masks.independent["occupied"]
    unoccupied = masks.independent["unoccupied"]
    if not np.all(occupied | unoccupied):
        raise ValueError("occupied + unoccupied does not cover the grid")
    partition_cover = np.zeros(shape, dtype=bool)
    for mask in masks.partition.values():
        partition_cover |= mask
    if not np.all(partition_cover):
        raise ValueError("mutually exclusive partition does not cover the grid")


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            out |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return out


def erode_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            out &= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return out


def boundary_mask(occupancy: np.ndarray, width: int) -> np.ndarray:
    return dilate_bool(occupancy, width) & ~erode_bool(occupancy, width)


def package_edge_mask(height: int, width: int, edge_width: int) -> np.ndarray:
    edge_width = max(0, int(edge_width))
    mask = np.zeros((height, width), dtype=bool)
    if edge_width <= 0:
        return mask
    mask[:edge_width, :] = True
    mask[-edge_width:, :] = True
    mask[:, :edge_width] = True
    mask[:, -edge_width:] = True
    return mask


def package_corner_mask(height: int, width: int, edge_width: int) -> np.ndarray:
    edge_width = max(0, int(edge_width))
    mask = np.zeros((height, width), dtype=bool)
    if edge_width <= 0:
        return mask
    mask[:edge_width, :edge_width] = True
    mask[:edge_width, -edge_width:] = True
    mask[-edge_width:, :edge_width] = True
    mask[-edge_width:, -edge_width:] = True
    return mask


def disk_mask(height: int, width: int, center: tuple[int, int], radius: int) -> np.ndarray:
    rr, cc = np.ogrid[:height, :width]
    return (rr - int(center[0])) ** 2 + (cc - int(center[1])) ** 2 <= int(radius) ** 2


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(field.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def collect_gradient_thresholds(dataset: ChipThermDataset, quantiles: list[float]) -> list[float]:
    values: list[np.ndarray] = []
    for row in dataset.rows:
        y = np.load(dataset._resolve_path(row["y_path"])).astype(np.float64, copy=False)
        values.append(gradient_magnitude(y).reshape(-1))
    flat = np.concatenate(values) if values else np.asarray([0.0])
    return [float(np.quantile(flat, q)) for q in quantiles]


def distance_maps(masks: MaskBundle, occupancy: np.ndarray) -> dict[str, np.ndarray]:
    h, w = occupancy.shape
    rr, cc = np.indices((h, w), dtype=np.float64)
    edge_distance = np.minimum.reduce([rr, cc, h - 1 - rr, w - 1 - cc])
    true_hotspot_distance = np.sqrt((rr - masks.true_hotspot[0]) ** 2 + (cc - masks.true_hotspot[1]) ** 2)
    pred_hotspot_distance = np.sqrt((rr - masks.predicted_hotspot[0]) ** 2 + (cc - masks.predicted_hotspot[1]) ** 2)
    return {
        "nearest_chiplet_boundary_cells": distance_to_mask(masks.boundary_core),
        "nearest_occupied_cell_cells": distance_to_mask(occupancy),
        "package_edge_cells": edge_distance,
        "true_hotspot_cells": true_hotspot_distance,
        "predicted_hotspot_cells": pred_hotspot_distance,
    }


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    targets = np.argwhere(mask.astype(bool))
    if targets.size == 0:
        return np.full((h, w), np.inf, dtype=np.float64)
    coords = np.indices((h, w)).reshape(2, -1).T.astype(np.float64)
    best = np.full(coords.shape[0], np.inf, dtype=np.float64)
    chunk = 512
    target_float = targets.astype(np.float64)
    for start in range(0, coords.shape[0], chunk):
        diff = coords[start : start + chunk, None, :] - target_float[None, :, :]
        best[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return best.reshape(h, w)


def update_distance_bins(
    accs: list[ErrorAccumulator],
    error: np.ndarray,
    distance_map: np.ndarray,
    bins: list[float],
) -> None:
    for idx in range(len(bins) - 1):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (distance_map >= lo) & (distance_map < hi)
        accs[idx].update(error, mask)


def update_frequency_stats(
    result: AuditResult,
    centered_error: np.ndarray,
    cutoffs: list[float],
    radial_bins: np.ndarray,
) -> None:
    h, w = centered_error.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    fft = np.fft.fft2(centered_error.astype(np.float64))
    power = np.abs(fft) ** 2
    bands = {
        "low_frequency": radius < cutoffs[0],
        "mid_frequency": (radius >= cutoffs[0]) & (radius < cutoffs[1]),
        "high_frequency": radius >= cutoffs[1],
    }
    total_energy = float(power.sum())
    for name, mask in bands.items():
        result.frequency_energy[name] += float(power[mask].sum())
        filtered = np.fft.ifft2(fft * mask).real
        result.frequency_stats[name].update(filtered)
    if total_energy > 0.0:
        bin_ids = np.digitize(radius.reshape(-1), radial_bins) - 1
        assert result.frequency_radial_energy is not None
        assert result.frequency_radial_count is not None
        flat_power = power.reshape(-1)
        for idx in range(len(radial_bins) - 1):
            selected = bin_ids == idx
            result.frequency_radial_energy[idx] += float(flat_power[selected].sum())
            result.frequency_radial_count[idx] += int(selected.sum())


def graph_for_sample(graph: dict[str, torch.Tensor], sample_index: int) -> dict[str, torch.Tensor] | None:
    node_batch = graph["node_batch"]
    node_mask = node_batch == sample_index
    if not bool(node_mask.any()):
        return None
    node_ids = torch.nonzero(node_mask, as_tuple=False).reshape(-1)
    old_to_new = {int(old): new for new, old in enumerate(node_ids.tolist())}
    edge_index = graph["edge_index"]
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    selected_edges = edge_index[:, edge_mask].clone()
    for old, new in old_to_new.items():
        selected_edges[selected_edges == old] = new
    return {
        "node_features": graph["node_features"][node_mask],
        "edge_index": selected_edges,
        "edge_features": graph["edge_features"][edge_mask],
        "chiplet_rects": graph["chiplet_rects"][node_mask],
        "package_size": graph["package_size"][sample_index : sample_index + 1],
        "node_batch": torch.zeros(int(node_mask.sum().item()), dtype=torch.long),
    }


def nearest_chiplet_spacing_mm(graph: dict[str, torch.Tensor] | None) -> float | None:
    if graph is None or int(graph["chiplet_rects"].shape[0]) < 2:
        return None
    rects = graph["chiplet_rects"].float().numpy()
    centers = np.column_stack([rects[:, 0] + 0.5 * rects[:, 2], rects[:, 1] + 0.5 * rects[:, 3]])
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist += np.eye(dist.shape[0]) * 1.0e9
    return float(np.min(dist))


def chiplet_descriptors(graph: dict[str, torch.Tensor] | None, pred: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    if graph is None:
        return []
    field_pred = torch.from_numpy(pred.astype(np.float32))[None, ...]
    field_target = torch.from_numpy(target.astype(np.float32))[None, ...]
    weights, counts = chiplet_cell_weights(graph, height=int(pred.shape[0]), width=int(pred.shape[1]), dtype=torch.float32)
    node_pred = field_pred.index_select(0, graph["node_batch"].long())
    node_target = field_target.index_select(0, graph["node_batch"].long())
    mean_err = ((node_pred - node_target) * weights).sum(dim=(-2, -1)) / counts
    peak_pred = node_pred.masked_fill(weights <= 0.0, -torch.inf).amax(dim=(-2, -1))
    peak_target = node_target.masked_fill(weights <= 0.0, -torch.inf).amax(dim=(-2, -1))
    rects = graph["chiplet_rects"].float().numpy()
    rows: list[dict[str, Any]] = []
    for idx in range(rects.shape[0]):
        width = float(rects[idx, 2])
        height = float(rects[idx, 3])
        area = width * height
        rows.append(
            {
                "chiplet_x_mm": float(rects[idx, 0]),
                "chiplet_y_mm": float(rects[idx, 1]),
                "chiplet_width_mm": width,
                "chiplet_height_mm": height,
                "chiplet_area_mm2": area,
                "chiplet_aspect_ratio": width / height if height > 0.0 else None,
                "chiplet_mean_abs_error_K": abs(float(mean_err[idx].item())),
                "chiplet_mean_signed_error_K": float(mean_err[idx].item()),
                "chiplet_peak_abs_error_K": abs(float(peak_pred[idx].item() - peak_target[idx].item())),
            }
        )
    return rows


def masked_mae(error: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.abs(error[mask]).mean())


def sample_descriptors_from_metadata(metadata: dict[str, Any], sample_index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    features = metadata.get("metadata_features") if isinstance(metadata, dict) else None
    if not isinstance(features, dict):
        return out
    for name, values in features.items():
        out[f"metadata_{name}"] = tensor_or_sequence_value(values, sample_index)
    return out


def sample_descriptors_from_channels(x: np.ndarray, channel_names: list[str]) -> dict[str, Any]:
    wanted = [
        "package_width_mm",
        "package_height_mm",
        "total_power_W",
        "occupied_area_fraction",
        "occupied_fraction",
        "whitespace_fraction",
        "mean_power_density_W_per_mm2",
        "max_power_density_W_per_mm2",
        "minimum_distance_to_package_edge_mm",
        "thermal_crowding_W_per_mm",
    ]
    out: dict[str, Any] = {}
    channel_map = {name: idx for idx, name in enumerate(channel_names)}
    for name in wanted:
        if name in channel_map and channel_map[name] < x.shape[0]:
            values = x[channel_map[name]]
            out[name] = float(np.mean(values))
            out[f"{name}_max"] = float(np.max(values))
    if "power_density_W_per_mm2" in channel_map:
        power = x[channel_map["power_density_W_per_mm2"]]
        positive = power > 0.0
        if np.any(positive):
            out["active_mean_power_density_W_per_mm2"] = float(power[positive].mean())
            out["active_max_power_density_W_per_mm2"] = float(power[positive].max())
    if "occupancy_mask" in channel_map:
        occ = x[channel_map["occupancy_mask"]] > 0.5
        out["raster_occupied_fraction"] = float(occ.mean())
        out["raster_whitespace_fraction"] = float(1.0 - occ.mean())
    return out


def metadata_value(metadata: dict[str, Any], key: str, sample_index: int, *, default: Any = None) -> Any:
    if key not in metadata:
        return default
    return tensor_or_sequence_value(metadata[key], sample_index)


def tensor_or_sequence_value(values: Any, sample_index: int) -> Any:
    if torch.is_tensor(values):
        item = values[sample_index]
        return float(item.item()) if item.ndim == 0 and item.dtype.is_floating_point else int(item.item()) if item.ndim == 0 else item.tolist()
    if isinstance(values, np.ndarray):
        return values[sample_index].item() if values[sample_index].shape == () else values[sample_index].tolist()
    if isinstance(values, (list, tuple)):
        return values[sample_index]
    return values


def select_worst_samples(records: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    criteria = {
        "final_full_field_mae": "final_mae_K",
        "final_rmse": "final_rmse_K",
        "centered_field_mae": "centered_field_mae_K",
        "occupied_region_mae": "occupied_mae_K",
        "boundary_region_mae": "chiplet_boundary_band_mae_K",
        "package_edge_mae": "package_edge_mae_K",
        "hotspot_temperature_abs_error": "hotspot_temperature_abs_error_K",
        "hotspot_location_error": "hotspot_location_error_cells",
        "maximum_absolute_pixel_error": "max_abs_pixel_error_K",
        "mean_rise_abs_error": "mean_rise_abs_error_K",
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    for name, key in criteria.items():
        valid = [record for record in records if record.get(key) is not None and np.isfinite(float(record[key]))]
        valid.sort(key=lambda item: float(item[key]), reverse=True)
        rows: list[dict[str, Any]] = []
        for rank, record in enumerate(valid[:top_k], start=1):
            row = {
                "rank": rank,
                "criterion": name,
                "criterion_value": float(record[key]),
                "sample_uid": record["sample_uid"],
                "case_id": record["case_id"],
                "final_mae_K": record["final_mae_K"],
                "centered_field_mae_K": record["centered_field_mae_K"],
                "mean_rise_abs_error_K": record["mean_rise_abs_error_K"],
            }
            rows.append(row)
        selected[name] = rows
    return selected


def write_outputs(
    out_dir: Path,
    result: AuditResult,
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    model_info: dict[str, Any],
    dataset: ChipThermDataset,
    parameter_count: int,
    gradient_thresholds: list[float],
) -> None:
    summary = build_summary(result, args, checkpoint, model_info, parameter_count, gradient_thresholds)
    write_json(out_dir / "audit_summary.json", summary)
    write_json(out_dir / "run_config.json", summary["run_config"])
    write_region_metrics(out_dir / "region_metrics.csv", result.region_stats)
    write_region_metrics(out_dir / "partition_metrics.csv", result.partition_stats)
    write_case_metrics(out_dir / "metrics_by_case.csv", result.case_stats)
    write_json(out_dir / "metrics_by_case.json", {case: case_stats_to_dict(stats) for case, stats in result.case_stats.items()})
    write_records_csv(out_dir / "sample_metrics.csv", result.sample_records)
    write_records_csv(out_dir / "chiplet_metrics.csv", result.chiplet_records)
    write_distance_bins(out_dir / "distance_bins", result.distance_stats, DEFAULT_DISTANCE_BINS)
    write_condition_bins(out_dir / "condition_bins" / "sample_condition_bins.csv", result.sample_records, "final_mae_K")
    write_condition_bins(out_dir / "condition_bins" / "chiplet_condition_bins.csv", result.chiplet_records, "chiplet_mean_abs_error_K")
    write_frequency_outputs(out_dir / "frequency_analysis", result, args.frequency_cutoffs)
    write_family_maps(out_dir / "family_maps", result.case_stats)
    write_worst_samples(out_dir / "worst_samples", result, save_panels=bool(args.save_sample_panels))
    write_report(out_dir / "audit_report.md", summary, result)


def build_summary(
    result: AuditResult,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    model_info: dict[str, Any],
    parameter_count: int,
    gradient_thresholds: list[float],
) -> dict[str, Any]:
    by_case = {case: case_stats_to_dict(stats) for case, stats in sorted(result.case_stats.items())}
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "index": str(args.index.resolve()),
        "num_samples": result.sample_count,
        "run_config": {
            "checkpoint": str(args.checkpoint),
            "index": str(args.index),
            "out_dir": str(args.out_dir),
            "batch_size": args.batch_size,
            "device": args.device,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "boundary_width_cells": args.boundary_width_cells,
            "package_edge_width_cells": args.package_edge_width_cells,
            "hotspot_radius_cells": args.hotspot_radius_cells,
            "gradient_quantiles": args.gradient_quantiles,
            "gradient_thresholds_K_per_cell": gradient_thresholds,
            "frequency_cutoffs_normalized_cycles_per_pixel": args.frequency_cutoffs,
            "top_k_worst": args.top_k_worst,
        },
        "model": {
            "config": checkpoint.get("model_config", {}),
            "architecture_info": model_info,
            "parameter_count": parameter_count,
        },
        "overall": {
            "final_temperature": result.global_final.as_dict(),
            "centered_field": result.global_centered.as_dict(),
            "mean_rise": result.global_mean_rise.as_dict(),
        },
        "region_metrics": {name: acc.as_dict() for name, acc in sorted(result.region_stats.items())},
        "partition_metrics": {name: acc.as_dict() for name, acc in sorted(result.partition_stats.items())},
        "metrics_by_case": by_case,
        "worst_samples": result.worst_samples,
        "notes": {
            "error_sign": "prediction - HotSpot target",
            "mask_overlap": "region_metrics masks are independent and may overlap; partition_metrics is mutually exclusive.",
            "frequency_method": "FFT of centered-field error. Frequency radius is normalized cycles per pixel.",
            "checkpoint_reconstruction": "Model/config/statistics are reconstructed from checkpoint metadata.",
        },
    }


def case_stats_to_dict(stats: CaseStats) -> dict[str, Any]:
    return {
        "final_temperature": stats.final.as_dict(),
        "centered_field": stats.centered.as_dict(),
        "mean_rise": stats.mean_rise.as_dict(),
        "hotspot_temperature_abs_error_K": summarize_list(stats.hotspot_temp_abs_error),
        "hotspot_location_error_cells": summarize_list(stats.hotspot_location_error),
        "regions": {name: acc.as_dict() for name, acc in sorted(stats.region_stats.items())},
    }


def summarize_list(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def write_region_metrics(path: Path, stats: dict[str, ErrorAccumulator]) -> None:
    rows = []
    for region, acc in sorted(stats.items()):
        row = {"region": region}
        row.update(acc.as_dict())
        rows.append(row)
    write_records_csv(path, rows)


def write_case_metrics(path: Path, case_stats: dict[str, CaseStats]) -> None:
    rows = []
    for case, stats in sorted(case_stats.items()):
        row = {
            "case_id": case,
            "num_samples": stats.final.sample_count,
            "final_mae_K": stats.final.mae(),
            "final_rmse_K": stats.final.rmse(),
            "final_mean_signed_error_K": stats.final.mean_signed(),
            "centered_mae_K": stats.centered.mae(),
            "centered_rmse_K": stats.centered.rmse(),
            "mean_rise_mae_K": stats.mean_rise.mae(),
            "hotspot_temperature_abs_error_mean_K": summarize_list(stats.hotspot_temp_abs_error)["mean"],
            "hotspot_location_error_mean_cells": summarize_list(stats.hotspot_location_error)["mean"],
        }
        for region in [
            "occupied",
            "unoccupied",
            "chiplet_interior",
            "chiplet_boundary_band",
            "package_edge_band",
            "package_corners",
            "true_hotspot_neighborhood",
            "high_gradient",
        ]:
            row[f"{region}_mae_K"] = stats.region_stats[region].mae()
        rows.append(row)
    write_records_csv(path, rows)


def write_distance_bins(path: Path, stats: dict[str, list[ErrorAccumulator]], bins: list[float]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, accs in sorted(stats.items()):
        rows = []
        for idx, acc in enumerate(accs):
            row = {
                "bin_index": idx,
                "distance_min_cells": bins[idx],
                "distance_max_cells": bins[idx + 1],
            }
            row.update(acc.as_dict())
            rows.append(row)
        write_records_csv(path / f"{name}.csv", rows)


def write_condition_bins(path: Path, records: list[dict[str, Any]], metric_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        write_records_csv(path, [])
        return
    candidate_keys = [
        key for key in records[0].keys()
        if key not in {"sample_uid", "case_id", "dataset_source"}
        and key != metric_key
        and all(is_number(record.get(key)) or record.get(key) is None for record in records)
    ]
    rows: list[dict[str, Any]] = []
    for key in candidate_keys:
        values = np.asarray([float(record[key]) for record in records if is_number(record.get(key))], dtype=np.float64)
        if values.size < 4 or np.nanstd(values) == 0.0:
            continue
        quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        for idx in range(4):
            selected = [
                record for record in records
                if is_number(record.get(key))
                and is_number(record.get(metric_key))
                and float(record[key]) >= quantiles[idx]
                and float(record[key]) < quantiles[idx + 1]
            ]
            if not selected:
                continue
            errors = np.asarray([float(record[metric_key]) for record in selected], dtype=np.float64)
            vals = np.asarray([float(record[key]) for record in selected], dtype=np.float64)
            rows.append(
                {
                    "descriptor": key,
                    "bin_index": idx,
                    "descriptor_min": float(vals.min()),
                    "descriptor_max": float(vals.max()),
                    "descriptor_mean": float(vals.mean()),
                    "count": int(errors.size),
                    f"{metric_key}_mean": float(errors.mean()),
                    f"{metric_key}_median": float(np.median(errors)),
                    f"{metric_key}_max": float(errors.max()),
                }
            )
    write_records_csv(path, rows)


def write_frequency_outputs(path: Path, result: AuditResult, cutoffs: list[float]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    total_energy = sum(result.frequency_energy.values())
    rows = []
    for band, acc in sorted(result.frequency_stats.items()):
        energy = result.frequency_energy.get(band, 0.0)
        row = {
            "band": band,
            "energy": energy,
            "energy_fraction": energy / total_energy if total_energy > 0.0 else None,
            "component_mae_K": acc.mae(),
            "component_rmse_K": acc.rmse(),
            "cutoff_low_normalized": cutoffs[0],
            "cutoff_high_normalized": cutoffs[1],
        }
        rows.append(row)
    write_records_csv(path / "frequency_bands.csv", rows)
    radial_energy = result.frequency_radial_energy
    radial_count = result.frequency_radial_count
    if radial_energy is not None and radial_count is not None:
        radial_bins = np.linspace(0.0, 0.5 * math.sqrt(2.0), len(radial_energy) + 1)
        cumulative = np.cumsum(radial_energy)
        denom = float(radial_energy.sum())
        radial_rows = []
        for idx, energy in enumerate(radial_energy):
            radial_rows.append(
                {
                    "bin_index": idx,
                    "frequency_min": float(radial_bins[idx]),
                    "frequency_max": float(radial_bins[idx + 1]),
                    "energy": float(energy),
                    "energy_fraction": float(energy / denom) if denom > 0.0 else None,
                    "cumulative_energy_fraction": float(cumulative[idx] / denom) if denom > 0.0 else None,
                    "mode_count": int(radial_count[idx]),
                }
            )
        write_records_csv(path / "radial_power_spectrum.csv", radial_rows)


def write_family_maps(path: Path, case_stats: dict[str, CaseStats]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for case, stats in sorted(case_stats.items()):
        case_dir = path / case
        case_dir.mkdir(parents=True, exist_ok=True)
        maps = stats.maps.mean_maps()
        for name, array in maps.items():
            np.save(case_dir / f"{name}.npy", array.astype(np.float32))
            save_heatmap_png(case_dir / f"{name}.png", array, signed="error" in name or "difference" in name)


def write_worst_samples(path: Path, result: AuditResult, *, save_panels: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    selected_uids: set[str] = set()
    for criterion, rows in sorted(result.worst_samples.items()):
        criterion_dir = path / criterion
        criterion_dir.mkdir(parents=True, exist_ok=True)
        write_records_csv(criterion_dir / "ranking.csv", rows)
        for row in rows:
            flat_rows.append(row)
            selected_uids.add(str(row["sample_uid"]))
    write_records_csv(path / "worst_sample_rankings.csv", flat_rows)
    for sample_uid in sorted(selected_uids):
        payload = result.worst_payloads.get(sample_uid)
        if payload is None:
            continue
        sample_dir = path / sample_uid
        sample_dir.mkdir(parents=True, exist_ok=True)
        arrays = payload["arrays"]
        np.savez_compressed(sample_dir / "maps.npz", **arrays)
        write_json(sample_dir / "summary.json", payload["summary"])
        if save_panels:
            save_sample_panel(sample_dir / "panel.png", arrays, payload["summary"])


def write_report(path: Path, summary: dict[str, Any], result: AuditResult) -> None:
    overall = summary["overall"]
    final = overall["final_temperature"]
    centered = overall["centered_field"]
    mean_rise = overall["mean_rise"]
    regions = summary["region_metrics"]
    freq_rows = []
    total_energy = sum(result.frequency_energy.values())
    for band, energy in sorted(result.frequency_energy.items()):
        freq_rows.append(f"- {band}: {energy / total_energy:.3f} energy fraction" if total_energy > 0 else f"- {band}: n/a")
    hardest = sorted(summary["metrics_by_case"].items(), key=lambda item: item[1]["final_temperature"]["mae_K"], reverse=True)[:5]
    lines = [
        "# ChipTherm Spatial Error Audit",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Index: `{summary['index']}`",
        f"Samples: {summary['num_samples']}",
        "",
        "## Main Metrics",
        "",
        f"- Final temperature MAE/RMSE: {final['mae_K']:.4f} / {final['rmse_K']:.4f} K",
        f"- Centered-field MAE/RMSE: {centered['mae_K']:.4f} / {centered['rmse_K']:.4f} K",
        f"- Mean-response MAE: {mean_rise['mae_K']:.4f} K",
        "",
        "## Region Signals",
        "",
    ]
    for region in [
        "occupied",
        "unoccupied",
        "chiplet_interior",
        "chiplet_boundary_band",
        "package_edge_band",
        "package_corners",
        "true_hotspot_neighborhood",
        "high_gradient",
    ]:
        if region in regions:
            lines.append(f"- {region}: MAE {regions[region]['mae_K']:.4f} K over {regions[region]['pixel_count']} pixels")
    lines.extend(["", "## Frequency Signals", "", *freq_rows, "", "## Hardest Families", ""])
    for case, metrics in hardest:
        lines.append(f"- {case}: final MAE {metrics['final_temperature']['mae_K']:.4f} K")
    lines.extend(
        [
            "",
            "## Diagnostic Interpretation Template",
            "",
            "- Compare mean-response MAE against centered-field MAE to decide whether the dominant error is scalar response or spatial shape.",
            "- Compare chiplet interior, boundary, package-edge, and hotspot-neighborhood MAE before choosing boundary, gradient, or hotspot losses.",
            "- Compare low/mid/high FFT error energy before increasing local refinement or receptive field.",
            "- Treat condition-bin correlations as diagnostic associations only; they are not causal proof.",
            "",
            "## Recommended Next Experiments",
            "",
            "Fill this section after running both all-family and held-out-family audits. Do not choose a new architecture unless one error mode is consistently dominant.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(first_dir: Path, second_dir: Path, out_path: Path) -> None:
    first = json.loads((first_dir / "audit_summary.json").read_text(encoding="utf-8"))
    second = json.loads((second_dir / "audit_summary.json").read_text(encoding="utf-8"))
    rows = []
    for label, path in [
        ("final_temperature", ["overall", "final_temperature", "mae_K"]),
        ("centered_field", ["overall", "centered_field", "mae_K"]),
        ("mean_rise", ["overall", "mean_rise", "mae_K"]),
    ]:
        a = nested_get(first, path)
        b = nested_get(second, path)
        rows.append((label, a, b, None if a is None or b is None else b - a))
    for region, metrics in second.get("region_metrics", {}).items():
        a = first.get("region_metrics", {}).get(region, {}).get("mae_K")
        b = metrics.get("mae_K")
        rows.append((f"region:{region}", a, b, None if a is None or b is None else b - a))
    rows.sort(key=lambda item: -abs(item[3]) if item[3] is not None else 0.0)
    lines = [
        "# Spatial Error Audit Comparison",
        "",
        f"Reference A: `{first_dir}`",
        f"Reference B: `{second_dir}`",
        "",
        "| metric | A MAE | B MAE | B - A |",
        "|---|---:|---:|---:|",
    ]
    for name, a, b, delta in rows:
        lines.append(f"| {name} | {fmt(a)} | {fmt(b)} | {fmt(delta)} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def save_heatmap_png(path: Path, array: np.ndarray, *, signed: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(4, 3.5), dpi=160)
    if signed:
        vmax = float(np.nanpercentile(np.abs(array), 99.0))
        vmax = vmax if vmax > 0.0 else 1.0
        plt.imshow(array, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    else:
        plt.imshow(array, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_sample_panel(path: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [
        "true_temperature_K",
        "source_superposition_base_K",
        "predicted_temperature_K",
        "signed_error_K",
        "absolute_error_K",
        "occupancy_mask",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=160)
    for ax, name in zip(axes.reshape(-1), names):
        data = arrays[name]
        if "signed_error" in name:
            vmax = float(np.nanpercentile(np.abs(data), 99.0)) or 1.0
            im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap="magma")
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{summary['sample_uid']} {summary['case_id']} MAE={summary['final_mae_K']:.3f} K", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in record.items()})


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def create_output_tree(out_dir: Path) -> None:
    for name in ["distance_bins", "condition_bins", "frequency_analysis", "family_maps", "worst_samples"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def validate_quantiles(values: list[float]) -> None:
    if len(values) != 3:
        raise SystemExit("--gradient-quantiles must provide exactly three values")
    if sorted(values) != list(values) or values[0] <= 0.0 or values[-1] >= 1.0:
        raise SystemExit("--gradient-quantiles must be sorted values strictly between 0 and 1")


def validate_cutoffs(values: list[float]) -> None:
    if len(values) != 2:
        raise SystemExit("--frequency-cutoffs must provide exactly two values")
    if values[0] <= 0.0 or values[0] >= values[1] or values[1] >= 0.5 * math.sqrt(2.0):
        raise SystemExit("--frequency-cutoffs must be two increasing normalized frequencies")


def is_number(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
