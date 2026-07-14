#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.graph_models import move_graph_to_device, normalize_graph_batch  # noqa: E402
from chiptherm.ml.metrics import error_metric_summary  # noqa: E402
from chiptherm.ml.models import build_model  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input  # noqa: E402
from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponseDataset,
    SourceResponseNormalizationStats,
    normalize_source_input,
    source_response_collate,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, predict_source_rise  # noqa: E402


CASE_IDS = ("case01", "case02")
DISTANCE_BINS_MM = (0.0, 2.0, 5.0, 10.0, 20.0, float("inf"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Case01-vs-case02 source/generalization diagnostic.")
    parser.add_argument("--checkpoint", default=REPO_ROOT / "outputs/source_superposition_full/source_superposition_cnn_gnn_seed1/checkpoints/best.pt", type=Path)
    parser.add_argument("--index", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full/test_index.csv", type=Path)
    parser.add_argument("--source-checkpoint", default=REPO_ROOT / "outputs/source_response_operator_v1/prototype_seed1/checkpoints/best.pt", type=Path)
    parser.add_argument("--source-index", default=REPO_ROOT / "data/runs/derived/source_response_v1/test_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/case01_case02_generalization", type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-samples-per-case", default=None, type=int)
    parser.add_argument("--skip-source-level", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_rows = run_package_analysis(args, device)
    source_rows = [] if args.skip_source_level else run_source_analysis(args, device)
    distance_rows = flatten_distance_rows(package_rows)
    correlations = compute_correlations(package_rows, source_rows)
    matched_pairs = compute_matched_pairs(package_rows)
    summary = build_summary(args, package_rows, source_rows, correlations, matched_pairs)

    write_csv(out_dir / "case01_case02_sample_metrics.csv", package_rows)
    write_csv(out_dir / "case01_case02_source_metrics.csv", source_rows)
    write_csv(out_dir / "case01_case02_distance_error.csv", distance_rows)
    write_csv(out_dir / "case01_case02_correlations.csv", correlations)
    write_csv(out_dir / "case01_case02_matched_pairs.csv", matched_pairs)
    write_json(out_dir / "case01_case02_summary.json", summary)
    write_report(out_dir / "case01_case02_report.md", summary)
    write_plots(out_dir, package_rows, source_rows, distance_rows)

    print("Case01/case02 diagnostic complete")
    print(f"Package samples: {len(package_rows)}")
    print(f"Source rows with isolated labels: {len(source_rows)}")
    print(f"Decision: {summary['decision']['recommendation']}")
    print(f"Output: {out_dir}")
    return 0


@torch.no_grad()
def run_package_analysis(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = NormalizationStats(**checkpoint["normalization"])
    config = checkpoint["model_config"]
    physics_mode = str(config.get("physics_input_mode", "v1"))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    graph_enabled = str(config.get("architecture", "")) in {
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
    }
    conditioned = "conditioned" in str(config.get("architecture", ""))
    graph_stats = config.get("graph_normalization")

    dataset = ChipThermDataset(args.index, target="residual", return_metadata=True, return_graph=graph_enabled)
    filtered = [row for row in dataset.rows if row["case_id"] in CASE_IDS]
    if args.max_samples_per_case is not None:
        limited: list[dict[str, str]] = []
        counts = defaultdict(int)
        for row in filtered:
            if counts[row["case_id"]] < int(args.max_samples_per_case):
                limited.append(row)
                counts[row["case_id"]] += 1
        filtered = limited
    dataset.rows = filtered
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=chiptherm_collate if graph_enabled else None,
    )

    records: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        base = batch["physics"].to(device, non_blocking=True)
        physics_v1 = batch.get("physics_v1")
        if physics_v1 is not None:
            physics_v1 = physics_v1.to(device, non_blocking=True)
        y = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        graph_batch = None
        if graph_enabled:
            graph_batch = normalize_graph_batch(move_graph_to_device(batch["graph"], device), graph_stats)
        model_input = build_model_input(x, base, stats, physics_input_mode=physics_mode, physics_v1=physics_v1)
        if graph_enabled:
            outputs = model(model_input, metadata_input, graph_batch, return_diagnostics=True, ambient=ambient)
        elif conditioned:
            outputs = model(model_input, metadata_input)
        else:
            outputs = model(model_input)
        final = reconstruct_temperature(outputs, ambient)
        cnn_centered = outputs.get("cnn_centered_field", outputs.get("centered_field"))
        if cnn_centered is not None:
            cnn_only = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + cnn_centered
        else:
            cnn_only = final
        case_ids = metadata_values(batch["metadata"], "case_id", int(x.shape[0]))
        sample_uids = metadata_values(batch["metadata"], "sample_uid", int(x.shape[0]))
        for i, sample_uid in enumerate(sample_uids):
            row = filtered[len(records)]
            layout = load_json(resolve_path(row.get("source_layout_path") or source_layout_path_from_row(row)))
            power = load_yaml(resolve_path(row.get("source_power_path") or source_power_path_from_row(row)))
            package_features = package_feature_summary(layout, power)
            y_np = y[i].detach().cpu().numpy()
            base_np = base[i].detach().cpu().numpy()
            cnn_np = cnn_only[i].detach().cpu().numpy()
            final_np = final[i].detach().cpu().numpy()
            distance_map = distance_to_nearest_chiplet_mm(layout, y_np.shape)
            rec = {
                "sample_uid": str(sample_uid),
                "case_id": str(case_ids[i]),
                **package_features,
                **stage_metrics("source_base", base_np, y_np),
                **stage_metrics("cnn_only", cnn_np, y_np),
                **stage_metrics("final", final_np, y_np),
                "cnn_improvement_over_source_base_mae_K": stage_mae(base_np, y_np) - stage_mae(cnn_np, y_np),
                "gnn_improvement_over_cnn_mae_K": stage_mae(cnn_np, y_np) - stage_mae(final_np, y_np),
                "mean_rise_error_K": float((final_np.mean() - y_np.mean())),
                "centered_field_mae_K": stage_mae(final_np - final_np.mean(), y_np - y_np.mean()),
                "hotspot_temp_error_K": hotspot_temp_error(final_np, y_np),
                "hotspot_location_error_cells": hotspot_location_error(final_np, y_np),
                "occupied_mae_K": masked_mae(np.abs(final_np - y_np), x[i, 1].detach().cpu().numpy() > 0.5),
                "unoccupied_mae_K": masked_mae(np.abs(final_np - y_np), x[i, 1].detach().cpu().numpy() <= 0.5),
            }
            rec.update(distance_bin_errors(final_np, y_np, distance_map))
            records.append(rec)
    return records


@torch.no_grad()
def run_source_analysis(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    if not args.source_index.exists() or not args.source_checkpoint.exists():
        return []
    checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(checkpoint["normalization"])
    model = build_source_response_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = SourceResponseDataset(args.source_index)
    dataset.rows = [row for row in dataset.rows if row["case_id"] in CASE_IDS]
    loader = DataLoader(dataset, batch_size=args.source_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=source_response_collate)
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(device)
        power = batch["source_power_W"].to(device)
        pred_unit = unnormalize_source_prediction(model(normalize_source_input(x, stats)), stats)
        pred_rise = predict_source_rise(pred_unit, power)
        target_rise = batch["target_rise"].to(device)
        target_unit = batch["target_unit"].to(device)
        pred_np = pred_rise.detach().cpu().numpy()
        target_np = target_rise.detach().cpu().numpy()
        unit_err = (pred_unit - target_unit).detach().cpu().numpy()
        for i, meta in enumerate(batch["metadata"]):
            layout = load_json(resolve_path(meta["layout_path"]))
            source_index = int(meta["source_index"])
            chiplet = layout["chiplets"][source_index]
            dist = distance_to_source_center_mm(layout, source_index, target_np[i].shape)
            rows.append(
                {
                    "source_response_uid": meta["source_response_uid"],
                    "original_sample_uid": meta["original_sample_uid"],
                    "case_id": meta["case_id"],
                    "source_index": source_index,
                    "source_name": meta["source_name"],
                    "source_power_W": float(meta["source_power_W"]),
                    "source_area_mm2": float(meta["source_area_mm2"]),
                    "source_power_density_W_per_mm2": float(meta["source_power_density_W_per_mm2"]),
                    **source_metric_summary(pred_np[i], target_np[i], unit_err[i], dist),
                    "source_width_mm": float(chiplet["size"]["width"]),
                    "source_height_mm": float(chiplet["size"]["height"]),
                    "source_min_edge_distance_mm": source_min_edge_distance(layout, source_index),
                }
            )
    return rows


def reconstruct_temperature(outputs: dict[str, torch.Tensor], ambient: torch.Tensor) -> torch.Tensor:
    centered = outputs["centered_field"]
    centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
    return ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered


def package_feature_summary(layout: dict[str, Any], power: dict[str, Any]) -> dict[str, float]:
    width = float(layout["package"]["size"]["width"])
    height = float(layout["package"]["size"]["height"])
    area = width * height
    diag = math.hypot(width, height)
    chiplets = layout["chiplets"]
    powers = active_powers(power)
    rects = [chiplet_rect(c) for c in chiplets]
    chip_areas = np.asarray([r[2] * r[3] for r in rects], dtype=np.float64)
    chip_powers = np.asarray([powers[str(c["name"])] for c in chiplets], dtype=np.float64)
    centers = np.asarray([[r[0] + r[2] / 2.0, r[1] + r[3] / 2.0] for r in rects], dtype=np.float64)
    pairwise = pairwise_distances(centers)
    edge_dists = []
    for x, y, w, h in rects:
        edge_dists.append(min(x, y, width - (x + w), height - (y + h)))
    occupied = float(chip_areas.sum() / max(area, 1.0e-12))
    dispersion = float(np.mean(np.linalg.norm(centers - centers.mean(axis=0), axis=1))) if len(centers) else 0.0
    weights = chip_powers / max(float(chip_powers.sum()), 1.0e-12)
    weighted_center = np.sum(centers * weights[:, None], axis=0)
    weighted_dispersion = float(np.sum(np.linalg.norm(centers - weighted_center, axis=1) * weights))
    return {
        "package_width_mm": width,
        "package_height_mm": height,
        "package_area_mm2": area,
        "package_diagonal_mm": diag,
        "chiplet_count": float(len(chiplets)),
        "occupied_fraction": occupied,
        "whitespace_fraction": 1.0 - occupied,
        "total_power_W": float(chip_powers.sum()),
        "mean_chiplet_power_W": float(chip_powers.mean()),
        "max_chiplet_power_W": float(chip_powers.max()),
        "mean_power_density_W_per_mm2": float(np.mean(chip_powers / np.maximum(chip_areas, 1.0e-12))),
        "max_power_density_W_per_mm2": float(np.max(chip_powers / np.maximum(chip_areas, 1.0e-12))),
        "pairwise_distance_mean_mm": float(pairwise.mean()) if pairwise.size else 0.0,
        "pairwise_distance_max_mm": float(pairwise.max()) if pairwise.size else 0.0,
        "pairwise_distance_mean_norm_diag": float(pairwise.mean() / diag) if pairwise.size and diag else 0.0,
        "pairwise_distance_max_norm_diag": float(pairwise.max() / diag) if pairwise.size and diag else 0.0,
        "nearest_neighbor_distance_mm": float(nearest_neighbor_distance(pairwise)),
        "chiplet_edge_distance_mean_mm": float(np.mean(edge_dists)),
        "chiplet_edge_distance_min_mm": float(np.min(edge_dists)),
        "layout_dispersion_mm": dispersion,
        "power_weighted_layout_dispersion_mm": weighted_dispersion,
        "graph_edge_count": float(len(chiplets) * max(len(chiplets) - 1, 0)),
    }


def stage_metrics(prefix: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    metric = error_metric_summary(pred, target).to_dict()
    return {
        f"{prefix}_mae_K": metric["mae_K"],
        f"{prefix}_rmse_K": metric["rmse_K"],
        f"{prefix}_global_pixel_rmse_K": metric["global_pixel_rmse_K"],
        f"{prefix}_mean_sample_rmse_K": metric["mean_sample_rmse_K"],
        f"{prefix}_max_abs_error_K": metric["max_abs_error_K"],
        f"{prefix}_mean_signed_error_K": metric["mean_signed_error_K"],
    }


def source_metric_summary(pred: np.ndarray, target: np.ndarray, unit_error: np.ndarray, dist: np.ndarray) -> dict[str, float]:
    metric = error_metric_summary(pred, target).to_dict()
    abs_err = np.abs(pred - target)
    pred_centroid = response_centroid(pred)
    target_centroid = response_centroid(target)
    pred_spread = response_spread_radius(pred, pred_centroid)
    target_spread = response_spread_radius(target, target_centroid)
    return {
        "source_physical_mae_K": metric["mae_K"],
        "source_physical_rmse_K": metric["rmse_K"],
        "source_unit_mae_K_per_W": float(np.mean(np.abs(unit_error))),
        "source_unit_rmse_K_per_W": float(np.sqrt(np.mean(unit_error * unit_error))),
        "source_peak_rise_error_K": float(pred.max() - target.max()),
        "source_centroid_displacement_cells": float(math.hypot(pred_centroid[0] - target_centroid[0], pred_centroid[1] - target_centroid[1])),
        "source_spread_radius_error_cells": float(pred_spread - target_spread),
        "source_near_mae_K": masked_mae(abs_err, dist <= 5.0),
        "source_far_mae_K": masked_mae(abs_err, dist > 10.0),
    }


def distance_bin_errors(pred: np.ndarray, target: np.ndarray, distance_map: np.ndarray) -> dict[str, float | None]:
    abs_error = np.abs(pred - target)
    result: dict[str, float | None] = {}
    for low, high in zip(DISTANCE_BINS_MM[:-1], DISTANCE_BINS_MM[1:]):
        label = f"distance_{format_bin(low)}_{format_bin(high)}_mae_K"
        mask = (distance_map >= low) & (distance_map < high)
        result[label] = masked_mae(abs_error, mask)
    return result


def flatten_distance_rows(package_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in package_rows:
        for low, high in zip(DISTANCE_BINS_MM[:-1], DISTANCE_BINS_MM[1:]):
            key = f"distance_{format_bin(low)}_{format_bin(high)}_mae_K"
            rows.append({"sample_uid": row["sample_uid"], "case_id": row["case_id"], "bin_low_mm": low, "bin_high_mm": high, "final_mae_K": row.get(key)})
    return rows


def compute_correlations(package_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "package_width_mm",
        "package_area_mm2",
        "whitespace_fraction",
        "pairwise_distance_max_norm_diag",
        "pairwise_distance_max_mm",
        "nearest_neighbor_distance_mm",
        "chiplet_edge_distance_min_mm",
        "total_power_W",
        "max_power_density_W_per_mm2",
        "source_base_mae_K",
    ]
    targets = ["final_mae_K", "centered_field_mae_K", "hotspot_location_error_cells"]
    rows: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        items = [row for row in package_rows if row["case_id"] == case_id]
        for feature in features:
            for target in targets:
                rows.append(correlation_row("package", case_id, feature, target, items))
    for feature in ("source_power_density_W_per_mm2", "source_min_edge_distance_mm", "source_far_mae_K", "source_area_mm2"):
        for target in ("source_physical_mae_K", "source_far_mae_K", "source_spread_radius_error_cells"):
            for case_id in CASE_IDS:
                items = [row for row in source_rows if row["case_id"] == case_id]
                rows.append(correlation_row("source", case_id, feature, target, items))
    return rows


def compute_matched_pairs(package_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case01 = [row for row in package_rows if row["case_id"] == "case01"]
    case02 = [row for row in package_rows if row["case_id"] == "case02"]
    features = ["total_power_W", "occupied_fraction", "max_power_density_W_per_mm2", "pairwise_distance_max_norm_diag"]
    all_rows = package_rows
    means = {f: np.mean([row[f] for row in all_rows]) for f in features}
    stds = {f: max(float(np.std([row[f] for row in all_rows])), 1.0e-8) for f in features}
    pairs = []
    used: set[str] = set()
    for left in case01:
        best = None
        best_dist = float("inf")
        for right in case02:
            if right["sample_uid"] in used:
                continue
            dist = math.sqrt(sum(((left[f] - right[f]) / stds[f]) ** 2 for f in features))
            if dist < best_dist:
                best = right
                best_dist = dist
        if best is not None:
            used.add(best["sample_uid"])
            pairs.append(
                {
                    "case01_sample_uid": left["sample_uid"],
                    "case02_sample_uid": best["sample_uid"],
                    "match_distance_z": best_dist,
                    "case01_final_mae_K": left["final_mae_K"],
                    "case02_final_mae_K": best["final_mae_K"],
                    "case02_minus_case01_final_mae_K": best["final_mae_K"] - left["final_mae_K"],
                    "case01_source_base_mae_K": left["source_base_mae_K"],
                    "case02_source_base_mae_K": best["source_base_mae_K"],
                    "case02_minus_case01_source_base_mae_K": best["source_base_mae_K"] - left["source_base_mae_K"],
                }
            )
    return pairs


def build_summary(
    args: argparse.Namespace,
    package_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    matched_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    by_case = {}
    for case_id in CASE_IDS:
        items = [row for row in package_rows if row["case_id"] == case_id]
        source_items = [row for row in source_rows if row["case_id"] == case_id]
        by_case[case_id] = {
            "num_package_samples": len(items),
            "num_source_rows": len(source_items),
            "final_mae_K": mean(row["final_mae_K"] for row in items),
            "centered_field_mae_K": mean(row["centered_field_mae_K"] for row in items),
            "source_base_mae_K": mean(row["source_base_mae_K"] for row in items),
            "cnn_only_mae_K": mean(row["cnn_only_mae_K"] for row in items),
            "cnn_improvement_over_source_base_mae_K": mean(row["cnn_improvement_over_source_base_mae_K"] for row in items),
            "gnn_improvement_over_cnn_mae_K": mean(row["gnn_improvement_over_cnn_mae_K"] for row in items),
            "mean_pairwise_distance_max_norm_diag": mean(row["pairwise_distance_max_norm_diag"] for row in items),
            "source_physical_mae_K": mean(row["source_physical_mae_K"] for row in source_items) if source_items else None,
            "source_far_mae_K": mean(row["source_far_mae_K"] for row in source_items) if source_items else None,
        }
    decision = choose_decision(by_case, correlations, matched_pairs)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "index": str(args.index.resolve()),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_index": str(args.source_index.resolve()),
        "cases": by_case,
        "source_level_coverage": source_coverage(source_rows),
        "decision": decision,
    }


def choose_decision(by_case: dict[str, Any], correlations: list[dict[str, Any]], matched_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    c1 = by_case.get("case01", {})
    c2 = by_case.get("case02", {})
    source_gap = nullable_diff(c2.get("source_physical_mae_K"), c1.get("source_physical_mae_K"))
    base_gap = nullable_diff(c2.get("source_base_mae_K"), c1.get("source_base_mae_K"))
    final_gap = nullable_diff(c2.get("final_mae_K"), c1.get("final_mae_K"))
    improvement_gap = nullable_diff(c1.get("cnn_improvement_over_source_base_mae_K"), c2.get("cnn_improvement_over_source_base_mae_K"))
    if source_gap is not None and source_gap > 1.0 and base_gap is not None and base_gap > 1.0:
        recommendation = "targeted source-label enrichment"
        diagnosis = "A: source-response supervision/coverage bottleneck"
    elif base_gap is not None and base_gap < 1.0 and final_gap is not None and final_gap > 2.0:
        recommendation = "new long-range package residual model"
        diagnosis = "B: package-level residual architecture bottleneck"
    elif improvement_gap is not None and improvement_gap > 1.0:
        recommendation = "package-aware calibration or complete-source-group residual training"
        diagnosis = "D: accumulated-source-error or hard residual correction bottleneck"
    else:
        recommendation = "richer package-diverse source dataset"
        diagnosis = "C: distribution/generalization bottleneck"
    return {
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "case02_minus_case01_source_base_mae_K": base_gap,
        "case02_minus_case01_final_mae_K": final_gap,
        "case02_minus_case01_source_physical_mae_K": source_gap,
        "case01_minus_case02_cnn_improvement_K": improvement_gap,
        "caution": "Decision is diagnostic and correlation-based; do not claim causality without targeted follow-up.",
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    c1 = summary["cases"].get("case01", {})
    c2 = summary["cases"].get("case02", {})
    decision = summary["decision"]
    lines = [
        "# Case01 vs Case02 Generalization Diagnostic",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        "",
        "| Metric | case01 | case02 |",
        "|---|---:|---:|",
        f"| Package samples | {c1.get('num_package_samples', 0)} | {c2.get('num_package_samples', 0)} |",
        f"| Source rows with isolated labels | {c1.get('num_source_rows', 0)} | {c2.get('num_source_rows', 0)} |",
        f"| Source-base MAE K | {fmt(c1.get('source_base_mae_K'))} | {fmt(c2.get('source_base_mae_K'))} |",
        f"| CNN-only MAE K | {fmt(c1.get('cnn_only_mae_K'))} | {fmt(c2.get('cnn_only_mae_K'))} |",
        f"| Final MAE K | {fmt(c1.get('final_mae_K'))} | {fmt(c2.get('final_mae_K'))} |",
        f"| Centered-field MAE K | {fmt(c1.get('centered_field_mae_K'))} | {fmt(c2.get('centered_field_mae_K'))} |",
        f"| Source physical MAE K | {fmt(c1.get('source_physical_mae_K'))} | {fmt(c2.get('source_physical_mae_K'))} |",
        "",
        f"Decision: **{decision['recommendation']}**",
        "",
        f"Diagnosis: {decision['diagnosis']}",
        "",
        decision["caution"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(out_dir: Path, package_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], distance_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    scatter_plot(package_rows, "source_base_mae_K", "final_mae_K", plots / "final_vs_source_base_mae.png", "Final MAE vs Source-Base MAE")
    scatter_plot(package_rows, "pairwise_distance_max_norm_diag", "final_mae_K", plots / "final_mae_vs_normalized_spacing.png", "Final MAE vs Normalized Max Spacing")
    distance_plot(distance_rows, plots / "distance_error_by_case.png")
    if source_rows:
        scatter_plot(source_rows, "source_far_mae_K", "source_physical_mae_K", plots / "source_far_vs_total_mae.png", "Source Far-Field MAE vs Source MAE")


def scatter_plot(rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    for case_id, color in (("case01", "tab:blue"), ("case02", "tab:red")):
        items = [r for r in rows if r.get("case_id") == case_id and r.get(x_key) is not None and r.get(y_key) is not None]
        plt.scatter([r[x_key] for r in items], [r[y_key] for r in items], label=case_id, alpha=0.75, s=18, c=color)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def distance_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    labels = []
    for low, high in zip(DISTANCE_BINS_MM[:-1], DISTANCE_BINS_MM[1:]):
        label = f"{format_bin(low)}-{format_bin(high)}"
        labels.append(label)
    xs = np.arange(len(labels))
    for case_id, color in (("case01", "tab:blue"), ("case02", "tab:red")):
        values = []
        for low, high in zip(DISTANCE_BINS_MM[:-1], DISTANCE_BINS_MM[1:]):
            vals = [r["final_mae_K"] for r in rows if r["case_id"] == case_id and r["bin_low_mm"] == low and r["final_mae_K"] is not None]
            values.append(mean(vals))
        plt.plot(xs, values, marker="o", label=case_id, color=color)
    plt.xticks(xs, labels, rotation=25)
    plt.ylabel("Final MAE (K)")
    plt.xlabel("Distance to nearest chiplet (mm)")
    plt.title("Error vs Distance From Powered Chiplets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def correlation_row(level: str, case_id: str, feature: str, target: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(float(r[feature]), float(r[target])) for r in rows if r.get(feature) is not None and r.get(target) is not None]
    xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return {
        "level": level,
        "case_id": case_id,
        "feature": feature,
        "target": target,
        "n": len(pairs),
        "pearson": pearson(xs, ys),
        "spearman": pearson(rankdata(xs), rankdata(ys)) if len(pairs) >= 3 else None,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def source_coverage(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case = defaultdict(set)
    for row in source_rows:
        by_case[row["case_id"]].add(row["original_sample_uid"])
    return {case: {"packages": len(uids), "sources": sum(1 for row in source_rows if row["case_id"] == case)} for case, uids in sorted(by_case.items())}


def active_powers(power: dict[str, Any]) -> dict[str, float]:
    workload = power.get("active_workload", "nominal")
    workloads = power.get("workloads") or {}
    if workload in workloads:
        return {str(k): float(v) for k, v in workloads[workload].items()}
    return {str(k): float(v) for k, v in power.get("chiplets", {}).items()}


def chiplet_rect(chiplet: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(chiplet["position"]["x"]),
        float(chiplet["position"]["y"]),
        float(chiplet["size"]["width"]),
        float(chiplet["size"]["height"]),
    )


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=np.float64)
    values = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            values.append(float(np.linalg.norm(points[i] - points[j])))
    return np.asarray(values, dtype=np.float64)


def nearest_neighbor_distance(pairwise: np.ndarray) -> float:
    if pairwise.size == 0:
        return 0.0
    return float(pairwise.min())


def distance_to_nearest_chiplet_mm(layout: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    package_w = float(layout["package"]["size"]["width"])
    package_h = float(layout["package"]["size"]["height"])
    xs = (np.arange(width, dtype=np.float64) + 0.5) * package_w / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) * package_h / height
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = np.full(shape, np.inf, dtype=np.float64)
    for chiplet in layout["chiplets"]:
        x, y, w, h = chiplet_rect(chiplet)
        dx = np.maximum(np.maximum(x - grid_x, 0.0), grid_x - (x + w))
        dy = np.maximum(np.maximum(y - grid_y, 0.0), grid_y - (y + h))
        distance = np.minimum(distance, np.sqrt(dx * dx + dy * dy))
    return distance


def distance_to_source_center_mm(layout: dict[str, Any], source_index: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    package_w = float(layout["package"]["size"]["width"])
    package_h = float(layout["package"]["size"]["height"])
    xs = (np.arange(width, dtype=np.float64) + 0.5) * package_w / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) * package_h / height
    grid_x, grid_y = np.meshgrid(xs, ys)
    x, y, w, h = chiplet_rect(layout["chiplets"][source_index])
    cx, cy = x + w / 2.0, y + h / 2.0
    return np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)


def source_min_edge_distance(layout: dict[str, Any], source_index: int) -> float:
    x, y, w, h = chiplet_rect(layout["chiplets"][source_index])
    package_w = float(layout["package"]["size"]["width"])
    package_h = float(layout["package"]["size"]["height"])
    return float(min(x, y, package_w - (x + w), package_h - (y + h)))


def response_centroid(response: np.ndarray) -> tuple[float, float]:
    weights = np.maximum(response.astype(np.float64), 0.0)
    total = float(weights.sum())
    if total <= 1.0e-12:
        idx = np.unravel_index(int(np.argmax(response)), response.shape)
        return float(idx[0]), float(idx[1])
    rows, cols = np.indices(response.shape)
    return float((rows * weights).sum() / total), float((cols * weights).sum() / total)


def response_spread_radius(response: np.ndarray, centroid: tuple[float, float]) -> float:
    weights = np.maximum(response.astype(np.float64), 0.0)
    total = float(weights.sum())
    if total <= 1.0e-12:
        return 0.0
    rows, cols = np.indices(response.shape)
    return float(np.sqrt((((rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2) * weights).sum() / total))


def hotspot_temp_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_idx = np.unravel_index(int(np.argmax(pred)), pred.shape)
    target_idx = np.unravel_index(int(np.argmax(target)), target.shape)
    return float(pred[pred_idx] - target[target_idx])


def hotspot_location_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_idx = np.unravel_index(int(np.argmax(pred)), pred.shape)
    target_idx = np.unravel_index(int(np.argmax(target)), target.shape)
    return float(math.hypot(pred_idx[0] - target_idx[0], pred_idx[1] - target_idx[1]))


def stage_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def masked_mae(abs_error: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(abs_error[mask].mean())


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def source_layout_path_from_row(row: dict[str, str]) -> str:
    return str(REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / row["case_id"] / row["original_sample_uid"].replace(f"{row['case_id']}_", "") / "source/layout.json")


def source_power_path_from_row(row: dict[str, str]) -> str:
    return source_layout_path_from_row(row).replace("layout.json", "power.yaml")


def mean(values: Any) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def nullable_diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def format_bin(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return str(value).replace(".", "p")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
