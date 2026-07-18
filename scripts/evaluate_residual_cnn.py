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
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.graph_models import chiplet_metric_values, move_graph_to_device, normalize_graph_batch
from chiptherm.ml.models import build_model, count_parameters
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input, unnormalize_residual


def physics_input_channel_count(mode: str) -> int:
    if mode in {"v1", "gated_v1", "source_superposition_v1"}:
        return 1
    if mode == "source_superposition_plus_physics_v1":
        return 2
    if mode == "none":
        return 0
    raise ValueError(f"unsupported physics input mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained ChipTherm residual CNN.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/test_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/residual_cnn_v1/test_eval", type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--measure-end-to-end", action="store_true")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--disable-graph-correction", action="store_true")
    parser.add_argument("--graph-correction-scale", default=1.0, type=float)
    parser.add_argument("--disable-global-correction", action="store_true")
    parser.add_argument("--global-correction-scale", default=1.0, type=float)
    parser.add_argument("--disable-global-fusion", action="store_true")
    parser.add_argument("--disable-global-fusion-16", action="store_true")
    parser.add_argument("--disable-global-fusion-32", action="store_true")
    parser.add_argument("--disable-global-fusion-64", action="store_true")
    parser.add_argument("--graph-rasterizer-mode", default=None, choices=["vectorized", "legacy"])
    args = parser.parse_args()
    if args.disable_graph_correction:
        args.graph_correction_scale = 0.0
    if args.disable_global_correction:
        args.global_correction_scale = 0.0
    disabled_fusion_scales = global_fusion_disabled_scales(args)

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = NormalizationStats(**checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    if args.graph_rasterizer_mode is not None and hasattr(model, "graph_rasterizer_mode"):
        model.graph_rasterizer_mode = args.graph_rasterizer_mode
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    architecture = str(checkpoint["model_config"].get("architecture", "miniunet"))
    graph_enabled = architecture in {
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
    }
    decomposed = architecture in {
        "miniunet_refine_decomposed",
        "miniunet_refine_conditioned_decomposed",
        "miniunet_refine_conditioned_decomposed_global",
        "miniunet_refine_conditioned_decomposed_feature_fusion",
        "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
    }
    conditioned = architecture in {
        "miniunet_refine_conditioned",
        "miniunet_refine_conditioned_decomposed",
        "miniunet_refine_conditioned_decomposed_global",
        "miniunet_refine_conditioned_decomposed_feature_fusion",
        "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
    }
    graph_stats = checkpoint["model_config"].get("graph_normalization")
    physics_input_mode = str(checkpoint["model_config"].get("physics_input_mode", "v1"))
    mean_head_mode = str(checkpoint["model_config"].get("mean_head_mode", "direct_k"))
    if mean_head_mode not in {"direct_k", "residual_resistance"}:
        raise SystemExit(f"unsupported checkpoint mean_head_mode: {mean_head_mode}")
    if physics_input_mode not in {
        "v1",
        "none",
        "gated_v1",
        "source_superposition_v1",
        "source_superposition_plus_physics_v1",
    }:
        raise SystemExit(f"unsupported checkpoint physics_input_mode: {physics_input_mode}")

    dataset = ChipThermDataset(args.index, target="residual", return_metadata=True, return_graph=graph_enabled)
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
        collate_fn=chiptherm_collate if graph_enabled else None,
    )

    metrics, by_case, runtime_s, hotspot_runtime_s, physics_runtime_s, gate_runtime_s = evaluate(
        model,
        loader,
        stats,
        device,
        measure_end_to_end=args.measure_end_to_end,
        save_predictions=args.save_predictions,
        out_dir=out_dir,
        decomposed=decomposed,
        conditioned=conditioned,
        physics_input_mode=physics_input_mode,
        mean_head_mode=mean_head_mode,
        graph_enabled=graph_enabled,
        graph_stats=graph_stats,
        graph_correction_scale=args.graph_correction_scale,
        global_correction_scale=args.global_correction_scale,
        disabled_fusion_scales=disabled_fusion_scales,
    )
    cnn_runtime_per_sample = runtime_s / max(metrics["num_samples"], 1)
    gate_runtime_per_sample = gate_runtime_s / max(metrics["num_samples"], 1) if gate_runtime_s > 0.0 else None
    cnn_side_speedup = hotspot_runtime_s / cnn_runtime_per_sample if hotspot_runtime_s and cnn_runtime_per_sample else None
    end_to_end_runtime_per_sample = None
    end_to_end_speedup = None
    timing_note = (
        "CNN-side timing uses loaded T_phys and includes normalization/input assembly, CNN forward, "
        "residual unnormalization, and final temperature reconstruction. Disk I/O is excluded."
    )
    if args.measure_end_to_end:
        if physics_input_mode == "none":
            physics_runtime_s = 0.0
            end_to_end_runtime_per_sample = cnn_runtime_per_sample
            end_to_end_speedup = (
                hotspot_runtime_s / end_to_end_runtime_per_sample
                if hotspot_runtime_s and end_to_end_runtime_per_sample
                else None
            )
            timing_note += (
                " End-to-end timing equals CNN-side timing because checkpoint physics_input_mode=none; "
                "physics_v1 is loaded only for reference metrics."
            )
        elif physics_input_mode in {"source_superposition_v1", "source_superposition_plus_physics_v1"}:
            end_to_end_runtime_per_sample = cnn_runtime_per_sample
            end_to_end_speedup = (
                hotspot_runtime_s / end_to_end_runtime_per_sample
                if hotspot_runtime_s and end_to_end_runtime_per_sample
                else None
            )
            timing_note += (
                " End-to-end timing here is cached-source-base runtime only; "
                "uncached source-response package inference must be added separately. "
                "For source_superposition_plus_physics_v1, physics-v1 is a preloaded auxiliary channel only."
            )
        elif physics_runtime_s is None:
            timing_note += " End-to-end timing requested, but physics_runtime_s metadata was unavailable."
        else:
            end_to_end_runtime_per_sample = physics_runtime_s + cnn_runtime_per_sample
            end_to_end_speedup = (
                hotspot_runtime_s / end_to_end_runtime_per_sample
                if hotspot_runtime_s and end_to_end_runtime_per_sample
                else None
            )
            timing_note += (
                " End-to-end timing is estimated as mean metadata physics_runtime_s plus CNN-side runtime; "
                "physics is not recomputed in this script."
            )
    physics_mae = metrics["physics_baseline"]["mae_K"]
    final_mae = metrics["cnn_final_temperature"]["mae_K"]
    physics_rmse = metrics["physics_baseline"]["rmse_K"]
    final_rmse = metrics["cnn_final_temperature"]["rmse_K"]
    improvement = {
        "mae_percent": percent_improvement(physics_mae, final_mae),
        "rmse_percent": percent_improvement(physics_rmse, final_rmse),
    }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "index": str(args.index.resolve()),
        "model": {
            "config": checkpoint["model_config"],
            "physics_input_mode": physics_input_mode,
            "mean_head_mode": mean_head_mode,
            "parameter_count": count_parameters(model),
        },
        "num_samples": metrics["num_samples"],
        "inference_runtime_total_s": runtime_s,
        "inference_runtime_per_sample_s": cnn_runtime_per_sample,
        "hotspot_runtime_reference_s": hotspot_runtime_s,
        "estimated_speedup_vs_hotspot": cnn_side_speedup,
        "runtime": {
            "hotspot_runtime_reference_s": hotspot_runtime_s,
            "cnn_runtime_per_sample_s": cnn_runtime_per_sample,
            "physics_runtime_per_sample_s": physics_runtime_s,
            "gating_overhead_per_sample_s": gate_runtime_per_sample,
            "end_to_end_runtime_per_sample_s": end_to_end_runtime_per_sample,
            "cnn_side_speedup_vs_hotspot": cnn_side_speedup,
            "end_to_end_speedup_vs_hotspot": end_to_end_speedup,
            "timing_note": timing_note,
        },
        "physics_baseline": metrics["physics_baseline"],
        "cnn_final_temperature": metrics["cnn_final_temperature"],
        "cnn_residual": metrics["cnn_residual"],
        "coarse_final_temperature": metrics.get("coarse_final_temperature"),
        "mean_rise": metrics.get("mean_rise"),
        "delta_R_eff": metrics.get("delta_R_eff"),
        "worse_than_physics_baseline_fraction": metrics.get("worse_than_physics_baseline_fraction"),
        "centered_field": metrics.get("centered_field"),
        "mean_bias_removed": metrics.get("mean_bias_removed"),
        "chiplet_mean_temperature": metrics.get("chiplet_mean_temperature"),
        "chiplet_peak_temperature": metrics.get("chiplet_peak_temperature"),
        "inter_chiplet_delta_T": metrics.get("inter_chiplet_delta_T"),
        "physics_gate": metrics.get("physics_gate"),
        "graph_correction_abs_mean": metrics.get("graph_correction_abs_mean"),
        "global_correction_abs_mean": metrics.get("global_correction_abs_mean"),
        "global_correction_abs_max": metrics.get("global_correction_abs_max"),
        "global_correction_rms": metrics.get("global_correction_rms"),
        "global_correction_spatial_std": metrics.get("global_correction_spatial_std"),
        "global_correction_low_frequency_energy": metrics.get("global_correction_low_frequency_energy"),
        "global_fusion_enabled": metrics.get("global_fusion_enabled"),
        "global_feature_abs_mean": metrics.get("global_feature_abs_mean"),
        "pairwise_k": metrics.get("pairwise_k"),
        "pairwise_contribution": metrics.get("pairwise_contribution"),
        "pairwise_self": metrics.get("pairwise_self"),
        "pairwise_node_correction": metrics.get("pairwise_node_correction"),
        "pairwise_basis_coeff": metrics.get("pairwise_basis_coeff"),
        "pairwise_basis_weighted_coeff": metrics.get("pairwise_basis_weighted_coeff"),
        "cnn_only_final_temperature": metrics.get("cnn_only_final_temperature"),
        "cnn_only_centered_field": metrics.get("cnn_only_centered_field"),
        "graph_improvement": metrics.get("graph_improvement"),
        "improvement_vs_physics_baseline": improvement,
    }
    component_profile = None
    if args.profile_components and graph_enabled:
        component_profile = profile_components(
            model,
            loader,
            stats,
            device,
            conditioned=conditioned,
            physics_input_mode=physics_input_mode,
            mean_head_mode=mean_head_mode,
            graph_stats=graph_stats,
            graph_correction_scale=args.graph_correction_scale,
            global_correction_scale=args.global_correction_scale,
            disabled_fusion_scales=disabled_fusion_scales,
        )
        payload["component_runtime"] = component_profile
        (out_dir / "component_runtime.json").write_text(json.dumps(component_profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case)

    print("Residual CNN evaluation complete")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Physics input mode: {physics_input_mode}")
    print(f"Mean head mode: {mean_head_mode}")
    print(f"CNN-side inference runtime/sample: {cnn_runtime_per_sample:.6f} s")
    if args.measure_end_to_end:
        if physics_runtime_s is None:
            print("Physics runtime/sample: n/a (metadata physics_runtime_s missing)")
            print("End-to-end estimated runtime/sample: n/a")
        else:
            print(f"Physics runtime/sample: {physics_runtime_s:.6f} s")
            print(f"End-to-end estimated runtime/sample: {end_to_end_runtime_per_sample:.6f} s")
    print(f"HotSpot runtime reference: {hotspot_runtime_s:.6f} s" if hotspot_runtime_s else "HotSpot runtime reference: n/a")
    print(f"CNN-side speedup: {cnn_side_speedup:.1f}x" if cnn_side_speedup else "CNN-side speedup: n/a")
    if args.measure_end_to_end:
        print(f"End-to-end speedup: {end_to_end_speedup:.1f}x" if end_to_end_speedup else "End-to-end speedup: n/a")
    print(f"Physics MAE/RMSE: {physics_mae:.3f} / {physics_rmse:.3f} K")
    if metrics.get("coarse_final_temperature"):
        coarse = metrics["coarse_final_temperature"]
        print(f"Coarse-only MAE/RMSE: {coarse['mae_K']:.3f} / {coarse['rmse_K']:.3f} K")
    if metrics.get("physics_gate"):
        gate = metrics["physics_gate"]
        print(
            "Physics gate alpha mean/std/min/max: "
            f"{gate['mean']:.3f} / {gate['std']:.3f} / {gate['min']:.3f} / {gate['max']:.3f}"
        )
        if gate_runtime_per_sample is not None:
            print(f"Estimated gate overhead/sample: {gate_runtime_per_sample:.9f} s")
    if metrics.get("graph_correction_abs_mean"):
        graph_summary = metrics["graph_correction_abs_mean"]
        print(
            "Graph correction abs mean/std/min/max: "
            f"{graph_summary['mean']:.3f} / {graph_summary['std']:.3f} / "
            f"{graph_summary['min']:.3f} / {graph_summary['max']:.3f} K"
        )
        if metrics.get("cnn_only_final_temperature"):
            cnn_only = metrics["cnn_only_final_temperature"]
            graph_improvement = metrics.get("graph_improvement", {})
            print(f"CNN-only MAE/RMSE: {cnn_only['mae_K']:.3f} / {cnn_only['rmse_K']:.3f} K")
            print(f"Graph MAE improvement: {graph_improvement.get('mae_K', float('nan')):.3f} K")
    if metrics.get("global_correction_abs_mean"):
        global_summary = metrics["global_correction_abs_mean"]
        print(
            "Global correction abs mean/std/min/max: "
            f"{global_summary['mean']:.3f} / {global_summary['std']:.3f} / "
            f"{global_summary['min']:.3f} / {global_summary['max']:.3f} K"
        )
    if metrics.get("global_fusion_enabled"):
        fusion = metrics["global_fusion_enabled"]
        status = ", ".join(f"{scale}:{summary['mean']:.0f}" for scale, summary in sorted(fusion.items()))
        print(f"Global feature fusion enabled by scale: {status}")
    if metrics.get("chiplet_mean_temperature"):
        chip_mean = metrics["chiplet_mean_temperature"]
        chip_peak = metrics.get("chiplet_peak_temperature", {})
        delta = metrics.get("inter_chiplet_delta_T", {})
        print(f"Chiplet mean-temp MAE: {chip_mean['mae_K']:.3f} K")
        if chip_peak:
            print(f"Chiplet peak-temp MAE: {chip_peak['mae_K']:.3f} K")
        if delta:
            print(f"Inter-chiplet delta-T MAE: {delta['mean']:.3f} K")
    if metrics.get("pairwise_k"):
        print(
            "Pairwise K mean/std/abs_mean: "
            f"{metrics['pairwise_k']['mean']:.6f} / {metrics['pairwise_k']['std']:.6f} / "
            f"{metrics['pairwise_k']['abs_mean']:.6f}"
        )
    if metrics.get("pairwise_basis_coeff"):
        print(
            "Pairwise basis coeff mean/std/abs_mean: "
            f"{metrics['pairwise_basis_coeff']['mean']:.6f} / "
            f"{metrics['pairwise_basis_coeff']['std']:.6f} / "
            f"{metrics['pairwise_basis_coeff']['abs_mean']:.6f}"
        )
    if metrics.get("physics_v1_auxiliary"):
        aux = metrics["physics_v1_auxiliary"]
        print(f"Physics-v1 auxiliary raw MAE/RMSE: {aux['mae_K']:.3f} / {aux['rmse_K']:.3f} K")
    if metrics.get("delta_R_eff"):
        delta_r = metrics["delta_R_eff"]
        print(f"Delta R_eff MAE/RMSE: {delta_r['mae_K_per_W']:.6f} / {delta_r['rmse_K_per_W']:.6f} K/W")
    if metrics.get("worse_than_physics_baseline_fraction") is not None:
        print(f"Fraction worse than physics/base: {metrics['worse_than_physics_baseline_fraction']:.3f}")
    print(f"CNN final MAE/RMSE: {final_mae:.3f} / {final_rmse:.3f} K")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Improvement: MAE {improvement['mae_percent']:.2f}% / RMSE {improvement['rmse_percent']:.2f}%")
    print(f"Output: {out_dir}")
    return 0


def global_fusion_disabled_scales(args: argparse.Namespace) -> tuple[str, ...]:
    if args.disable_global_fusion:
        return ("all",)
    disabled: list[str] = []
    if args.disable_global_fusion_16:
        disabled.append("16")
    if args.disable_global_fusion_32:
        disabled.append("32")
    if args.disable_global_fusion_64:
        disabled.append("64")
    return tuple(disabled)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    stats: NormalizationStats,
    device: torch.device,
    *,
    measure_end_to_end: bool,
    save_predictions: bool,
    out_dir: Path,
    decomposed: bool = False,
    conditioned: bool = False,
    physics_input_mode: str = "v1",
    graph_enabled: bool = False,
    graph_stats: Any | None = None,
    graph_correction_scale: float = 1.0,
    global_correction_scale: float = 1.0,
    disabled_fusion_scales: tuple[str, ...] = (),
    mean_head_mode: str = "direct_k",
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]], float, float | None, float | None, float]:
    residual_acc = MetricAccumulator()
    final_acc = MetricAccumulator()
    cnn_only_final_acc = MetricAccumulator()
    physics_acc = MetricAccumulator()
    physics_v1_acc = MetricAccumulator()
    coarse_final_acc = MetricAccumulator()
    mean_acc = ScalarMetricAccumulator()
    delta_R_acc = ScalarMetricAccumulator()
    centered_acc = MetricAccumulator()
    cnn_only_centered_acc = MetricAccumulator()
    mean_bias_removed_acc = MetricAccumulator()
    gate_acc = ScalarSummaryAccumulator()
    graph_correction_acc = ScalarSummaryAccumulator()
    graph_correction_max_acc = ScalarSummaryAccumulator()
    graph_correction_rms_acc = ScalarSummaryAccumulator()
    graph_correction_std_acc = ScalarSummaryAccumulator()
    global_correction_acc = ScalarSummaryAccumulator()
    global_correction_max_acc = ScalarSummaryAccumulator()
    global_correction_rms_acc = ScalarSummaryAccumulator()
    global_correction_std_acc = ScalarSummaryAccumulator()
    global_correction_low_freq_acc = ScalarSummaryAccumulator()
    global_fusion_enabled_acc = {scale: ScalarSummaryAccumulator() for scale in ("16", "32", "64")}
    global_feature_abs_acc = {scale: ScalarSummaryAccumulator() for scale in ("16", "32", "64")}
    cnn_centered_abs_acc = ScalarSummaryAccumulator()
    final_centered_abs_acc = ScalarSummaryAccumulator()
    chiplet_mean_acc = ScalarMetricAccumulator()
    chiplet_peak_acc = ScalarMetricAccumulator()
    chiplet_delta_acc = ScalarSummaryAccumulator()
    pairwise_k_acc = ScalarSummaryAccumulator()
    pairwise_contribution_acc = ScalarSummaryAccumulator()
    pairwise_self_acc = ScalarSummaryAccumulator()
    pairwise_node_acc = ScalarSummaryAccumulator()
    basis_coeff_acc = VectorSummaryAccumulator()
    basis_weighted_coeff_acc = VectorSummaryAccumulator()
    graph_rows: list[dict[str, Any]] = []
    chiplet_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    has_coarse_prediction = False
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            "cnn_residual": MetricAccumulator(),
            "cnn_final_temperature": MetricAccumulator(),
            "cnn_only_final_temperature": MetricAccumulator(),
            "physics_baseline": MetricAccumulator(),
            "physics_v1_auxiliary": MetricAccumulator(),
            "coarse_final_temperature": MetricAccumulator(),
        }
    )
    hotspot_runtimes: list[float] = []
    physics_runtimes: list[float] = []
    inference_runtime_s = 0.0
    gate_runtime_s = 0.0
    num_samples = 0
    worse_than_physics_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        physics_v1 = batch.get("physics_v1")
        if physics_v1 is not None:
            physics_v1 = physics_v1.to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        total_power = batch["total_power_W"].to(device, non_blocking=True).float()
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        graph_batch = prepare_graph_batch(batch, graph_enabled, graph_stats, device)
        if getattr(model, "physics_gate", None) is not None and getattr(model, "metadata_encoder", None) is not None and metadata_input is not None:
            synchronize(device)
            gate_start = time.perf_counter()
            _gate_alpha = model.physics_gate(model.metadata_encoder(metadata_input))
            synchronize(device)
            gate_runtime_s += time.perf_counter() - gate_start

        synchronize(device)
        start = time.perf_counter()
        model_input = build_model_input(
            x,
            physics,
            stats,
            physics_input_mode=physics_input_mode,
            physics_v1=physics_v1,
        )
        coarse_norm = None
        alpha = None
        if decomposed:
            outputs = call_model(
                model,
                model_input,
                metadata_input,
                graph_batch,
                conditioned=conditioned,
                graph_enabled=graph_enabled,
                graph_correction_scale=graph_correction_scale,
                global_correction_scale=global_correction_scale,
                disabled_fusion_scales=disabled_fusion_scales,
                total_power_W=total_power,
            )
            pred_temperature = reconstruct_decomposed_temperature(outputs, ambient, physics, mean_head_mode=mean_head_mode)
            pred_residual = pred_temperature - physics
            targets = decomposed_targets(temperature, ambient, physics, total_power, mean_head_mode=mean_head_mode)
            centered_pred = outputs["centered_field"]
            centered_target = targets["centered_field_K"]
            mean_target = targets["mean_correction_K"]
            mean_acc.update(outputs["mean_rise"], mean_target)
            if "delta_R_eff" in outputs:
                delta_R_acc.update(outputs["delta_R_eff"], targets["delta_R_eff_K_per_W"])
            centered_acc.update(centered_pred, centered_target)
            mean_bias_removed_acc.update(centered_pred, centered_target)
            alpha = outputs.get("physics_gate_alpha")
            if alpha is not None:
                gate_acc.update(alpha)
            graph_correction = outputs.get("graph_correction_field")
            if graph_correction is not None:
                graph_abs = graph_correction.abs()
                graph_correction_acc.update(graph_abs.mean(dim=(-2, -1)))
                graph_correction_max_acc.update(graph_abs.amax(dim=(-2, -1)))
                graph_correction_rms_acc.update(torch.sqrt(torch.mean(graph_correction * graph_correction, dim=(-2, -1))))
                graph_correction_std_acc.update(graph_correction.std(dim=(-2, -1)))
                cnn_centered = outputs["cnn_centered_field"]
                cnn_centered_abs_acc.update(cnn_centered.abs().mean(dim=(-2, -1)))
                final_centered_abs_acc.update(outputs["centered_field"].abs().mean(dim=(-2, -1)))
                if mean_head_mode == "residual_resistance":
                    cnn_only_temperature = physics + outputs["mean_rise"][:, None, None] + cnn_centered
                else:
                    cnn_only_temperature = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + cnn_centered
                cnn_only_centered = cnn_centered - cnn_centered.mean(dim=(-2, -1), keepdim=True)
                cnn_only_final_acc.update(cnn_only_temperature, temperature)
                cnn_only_centered_acc.update(cnn_only_centered, centered_target)
            global_correction = outputs.get("global_correction_field")
            if global_correction is not None:
                global_abs = global_correction.abs()
                global_correction_acc.update(global_abs.mean(dim=(-2, -1)))
                global_correction_max_acc.update(global_abs.amax(dim=(-2, -1)))
                global_correction_rms_acc.update(torch.sqrt(torch.mean(global_correction * global_correction, dim=(-2, -1))))
                global_correction_std_acc.update(global_correction.std(dim=(-2, -1)))
                if "global_correction_low_frequency_energy" in outputs:
                    global_correction_low_freq_acc.update(outputs["global_correction_low_frequency_energy"])
            for scale in ("16", "32", "64"):
                enabled_value = outputs.get(f"global_fusion_enabled_{scale}")
                if enabled_value is not None:
                    global_fusion_enabled_acc[scale].update(enabled_value)
                feature_value = outputs.get(f"global_feature_{scale}_abs_mean")
                if feature_value is not None:
                    global_feature_abs_acc[scale].update(feature_value)
            update_pairwise_summaries(
                outputs,
                pairwise_k_acc,
                pairwise_contribution_acc,
                pairwise_self_acc,
                pairwise_node_acc,
                basis_coeff_acc,
                basis_weighted_coeff_acc,
            )
            coarse_temperature = None
        elif hasattr(model, "forward_components"):
            if conditioned:
                pred_norm, coarse_norm, _detail_norm = model.forward_components(model_input, metadata_input)
            else:
                pred_norm, coarse_norm, _detail_norm = model.forward_components(model_input)
            pred_residual = unnormalize_residual(pred_norm.squeeze(1), stats)
            pred_temperature = physics + pred_residual
        else:
            pred_norm = model(model_input, metadata_input) if conditioned else model(model_input)
            pred_residual = unnormalize_residual(pred_norm.squeeze(1), stats)
            pred_temperature = physics + pred_residual
        if not decomposed:
            if coarse_norm is not None:
                coarse_residual = unnormalize_residual(coarse_norm.squeeze(1), stats)
                coarse_temperature = physics + coarse_residual
                has_coarse_prediction = True
            else:
                coarse_temperature = None
        synchronize(device)
        inference_runtime_s += time.perf_counter() - start

        batch_size = int(x.shape[0])
        num_samples += batch_size
        final_sample_mae = (pred_temperature - temperature).abs().reshape(batch_size, -1).mean(dim=1)
        physics_sample_mae = (physics - temperature).abs().reshape(batch_size, -1).mean(dim=1)
        worse_than_physics_count += int((final_sample_mae > physics_sample_mae).sum().item())
        case_ids = metadata_values(batch["metadata"], "case_id", batch_size)
        sample_uids = metadata_values(batch["metadata"], "sample_uid", batch_size)
        hotspot_runtimes.extend(
            value
            for value in optional_float_values(metadata_values(batch["metadata"], "hotspot_runtime_s", batch_size))
        )
        if measure_end_to_end:
            physics_runtimes.extend(
                value
                for value in optional_float_values(metadata_values(batch["metadata"], "physics_runtime_s", batch_size))
            )

        residual_acc.update(pred_residual, residual)
        final_acc.update(pred_temperature, temperature)
        physics_acc.update(physics, temperature)
        if physics_v1 is not None:
            physics_v1_acc.update(physics_v1, temperature)
        chiplet_metrics = None
        if graph_enabled and graph_batch is not None:
            chiplet_metrics = chiplet_metric_values(pred_temperature, temperature, graph_batch)
            chiplet_mean_acc.update(chiplet_metrics["pred_mean"], chiplet_metrics["target_mean"])
            chiplet_peak_acc.update(chiplet_metrics["pred_peak"], chiplet_metrics["target_peak"])
            chiplet_delta_acc.update(chiplet_metrics["delta_mae"].reshape(1))
        if coarse_temperature is not None:
            coarse_final_acc.update(coarse_temperature, temperature)
        for index, case_id in enumerate(case_ids):
            case_metrics = by_case[str(case_id)]
            case_metrics["cnn_residual"].update(pred_residual[index : index + 1], residual[index : index + 1])
            case_metrics["cnn_final_temperature"].update(pred_temperature[index : index + 1], temperature[index : index + 1])
            case_metrics["physics_baseline"].update(physics[index : index + 1], temperature[index : index + 1])
            if physics_v1 is not None:
                case_metrics["physics_v1_auxiliary"].update(physics_v1[index : index + 1], temperature[index : index + 1])
            if decomposed and "cnn_only_temperature" in locals():
                case_metrics["cnn_only_final_temperature"].update(
                    cnn_only_temperature[index : index + 1],
                    temperature[index : index + 1],
                )
            if coarse_temperature is not None:
                case_metrics["coarse_final_temperature"].update(coarse_temperature[index : index + 1], temperature[index : index + 1])
        if chiplet_metrics is not None:
            append_chiplet_rows(chiplet_rows, sample_uids, case_ids, chiplet_metrics, graph_batch)
            update_chiplet_case_metrics(by_case, case_ids, chiplet_metrics, graph_batch)

        if save_predictions:
            save_batch_predictions(out_dir, sample_uids, case_ids, pred_temperature, pred_residual)
        if decomposed and "cnn_only_temperature" in locals() and "graph_correction" in locals() and graph_correction is not None:
            append_graph_rows(
                graph_rows,
                batch,
                sample_uids,
                case_ids,
                cnn_only_temperature,
                pred_temperature,
                temperature,
                graph_correction,
            )
            del cnn_only_temperature
        if decomposed and "alpha" in locals() and alpha is not None:
            append_gate_rows(
                gate_rows,
                batch,
                sample_uids,
                case_ids,
                alpha,
                pred_temperature,
                temperature,
                physics,
                centered_pred if "centered_pred" in locals() else None,
                centered_target if "centered_target" in locals() else None,
                outputs.get("mean_rise") if "outputs" in locals() else None,
                mean_target if "mean_target" in locals() else None,
            )
        alpha = None

    metrics = {
        "num_samples": num_samples,
        "cnn_residual": residual_acc.compute(),
        "cnn_final_temperature": final_acc.compute(),
        "physics_baseline": physics_acc.compute(),
    }
    physics_v1_summary = physics_v1_acc.compute()
    if physics_v1_summary:
        metrics["physics_v1_auxiliary"] = physics_v1_summary
    if has_coarse_prediction:
        metrics["coarse_final_temperature"] = coarse_final_acc.compute()
    if decomposed:
        metrics["mean_rise"] = mean_acc.compute()
        delta_summary = delta_R_acc.compute()
        if delta_summary:
            metrics["delta_R_eff"] = rename_scalar_metric_units(delta_summary, "K_per_W")
        metrics["centered_field"] = centered_acc.compute()
        metrics["mean_bias_removed"] = mean_bias_removed_acc.compute()
        metrics["worse_than_physics_baseline_fraction"] = worse_than_physics_count / max(num_samples, 1)
    chiplet_mean_summary = chiplet_mean_acc.compute()
    if chiplet_mean_summary:
        metrics["chiplet_mean_temperature"] = chiplet_mean_summary
        metrics["chiplet_peak_temperature"] = chiplet_peak_acc.compute()
        metrics["inter_chiplet_delta_T"] = chiplet_delta_acc.compute()
        write_chiplet_outputs(out_dir, metrics, by_case, chiplet_rows)
    cnn_only_summary = cnn_only_final_acc.compute()
    if cnn_only_summary:
        metrics["cnn_only_final_temperature"] = cnn_only_summary
        metrics["cnn_only_centered_field"] = cnn_only_centered_acc.compute()
        metrics["graph_improvement"] = {
            "mae_K": cnn_only_summary["mae_K"] - metrics["cnn_final_temperature"]["mae_K"],
            "rmse_K": cnn_only_summary["rmse_K"] - metrics["cnn_final_temperature"]["rmse_K"],
            "graph_correction_scale": float(graph_correction_scale),
        }
    graph_summary = graph_correction_acc.compute()
    if graph_summary:
        metrics["graph_correction_abs_mean"] = graph_summary
        metrics["graph_correction_abs_max"] = graph_correction_max_acc.compute()
        metrics["graph_correction_rms"] = graph_correction_rms_acc.compute()
        metrics["graph_correction_spatial_std"] = graph_correction_std_acc.compute()
        metrics["cnn_centered_field_abs_mean"] = cnn_centered_abs_acc.compute()
        metrics["final_centered_field_abs_mean"] = final_centered_abs_acc.compute()
        denominator = max(float(metrics["cnn_centered_field_abs_mean"]["mean"]), 1.0e-8)
        metrics["graph_to_cnn_ratio"] = float(metrics["graph_correction_abs_mean"]["mean"] / denominator)
        write_graph_contribution_rows(out_dir / "graph_contribution_by_sample.csv", graph_rows)
    global_summary = global_correction_acc.compute()
    if global_summary:
        metrics["global_correction_abs_mean"] = global_summary
        metrics["global_correction_abs_max"] = global_correction_max_acc.compute()
        metrics["global_correction_rms"] = global_correction_rms_acc.compute()
        metrics["global_correction_spatial_std"] = global_correction_std_acc.compute()
        low_freq_summary = global_correction_low_freq_acc.compute()
        if low_freq_summary:
            metrics["global_correction_low_frequency_energy"] = low_freq_summary
    fusion_summary = {
        scale: global_fusion_enabled_acc[scale].compute()
        for scale in ("16", "32", "64")
        if global_fusion_enabled_acc[scale].compute()
    }
    if fusion_summary:
        metrics["global_fusion_enabled"] = fusion_summary
        metrics["global_feature_abs_mean"] = {
            scale: global_feature_abs_acc[scale].compute()
            for scale in ("16", "32", "64")
            if global_feature_abs_acc[scale].compute()
        }
    pairwise_summary = pairwise_k_acc.compute()
    if pairwise_summary:
        metrics["pairwise_k"] = pairwise_summary
        metrics["pairwise_contribution"] = pairwise_contribution_acc.compute()
        metrics["pairwise_self"] = pairwise_self_acc.compute()
        metrics["pairwise_node_correction"] = pairwise_node_acc.compute()
    basis_summary = basis_coeff_acc.compute()
    if basis_summary:
        metrics["pairwise_basis_coeff"] = basis_summary
        metrics["pairwise_basis_weighted_coeff"] = basis_weighted_coeff_acc.compute()
    gate_summary = gate_acc.compute()
    if gate_summary:
        metrics["physics_gate"] = gate_summary
        write_gate_values(out_dir / "gate_values.csv", gate_rows)
        write_gate_summary(out_dir / "gate_summary.json", gate_rows, gate_summary)
    case_payload = {
        case_id: {
            name: accumulator.compute()
            for name, accumulator in sorted(accs.items())
        }
        for case_id, accs in sorted(by_case.items())
    }
    if graph_summary:
        write_graph_contribution_by_case(out_dir / "graph_contribution_by_case.csv", case_payload)
        write_graph_contribution_summary(out_dir / "graph_contribution_summary.json", metrics, case_payload)
    hotspot_runtime_s = float(sum(hotspot_runtimes) / len(hotspot_runtimes)) if hotspot_runtimes else None
    physics_runtime_s = float(sum(physics_runtimes) / len(physics_runtimes)) if physics_runtimes else None
    return metrics, case_payload, inference_runtime_s, hotspot_runtime_s, physics_runtime_s, gate_runtime_s


def decomposed_targets(
    temperature: torch.Tensor,
    ambient: torch.Tensor,
    physics: torch.Tensor,
    total_power: torch.Tensor,
    *,
    mean_head_mode: str,
) -> dict[str, torch.Tensor]:
    if mean_head_mode == "residual_resistance":
        total_power_flat = total_power.to(device=temperature.device, dtype=temperature.dtype).view(-1)
        if torch.any(total_power_flat <= 0.0):
            raise ValueError("residual_resistance target requires strictly positive total_power_W")
        residual = temperature - physics
        mean_correction = residual.mean(dim=(-2, -1))
        centered = residual - mean_correction[:, None, None]
        return {
            "mean_correction_K": mean_correction,
            "centered_field_K": centered,
            "delta_R_eff_K_per_W": mean_correction / total_power_flat,
        }
    if mean_head_mode != "direct_k":
        raise ValueError(f"unsupported mean_head_mode: {mean_head_mode}")
    mean_rise = (temperature - ambient[:, None, None]).mean(dim=(-2, -1))
    centered = temperature - temperature.mean(dim=(-2, -1), keepdim=True)
    return {
        "mean_correction_K": mean_rise,
        "centered_field_K": centered,
        "delta_R_eff_K_per_W": torch.zeros_like(mean_rise),
    }


def reconstruct_decomposed_temperature(
    outputs: dict[str, torch.Tensor],
    ambient: torch.Tensor,
    physics: torch.Tensor | None = None,
    *,
    mean_head_mode: str = "direct_k",
) -> torch.Tensor:
    centered = outputs["centered_field"]
    centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
    if mean_head_mode == "residual_resistance":
        if physics is None:
            raise ValueError("residual_resistance reconstruction requires the physics/base tensor")
        return physics + outputs["mean_rise"][:, None, None] + centered
    if mean_head_mode != "direct_k":
        raise ValueError(f"unsupported mean_head_mode: {mean_head_mode}")
    return ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered


def rename_scalar_metric_units(metrics: dict[str, float], suffix: str) -> dict[str, float]:
    renamed: dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith("_K"):
            renamed[f"{key[:-2]}_{suffix}"] = value
        else:
            renamed[key] = value
    return renamed


@torch.no_grad()
def profile_components(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    stats: NormalizationStats,
    device: torch.device,
    *,
    conditioned: bool,
    physics_input_mode: str,
    graph_stats: Any | None,
    graph_correction_scale: float,
    mean_head_mode: str,
    global_correction_scale: float = 1.0,
    disabled_fusion_scales: tuple[str, ...] = (),
    warmup_batches: int = 2,
    profile_batches: int = 20,
) -> dict[str, Any]:
    if not hasattr(model, "forward_profile"):
        return {"error": "checkpoint model does not expose forward_profile"}
    timing_values: dict[str, list[float]] = defaultdict(list)
    nodes_per_sample: list[float] = []
    edges_per_sample: list[float] = []
    max_nodes = 0
    max_edges = 0

    def sync() -> None:
        synchronize(device)

    measured = 0
    for batch_index, batch in enumerate(loader):
        sync()
        prep_start = time.perf_counter()
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        total_power = batch["total_power_W"].to(device, non_blocking=True).float()
        physics_v1 = batch.get("physics_v1")
        if physics_v1 is not None:
            physics_v1 = physics_v1.to(device, non_blocking=True)
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is None and conditioned:
            raise ValueError("conditioned graph model requires metadata")
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        graph_batch = prepare_graph_batch(batch, True, graph_stats, device)
        model_input = build_model_input(
            x,
            physics,
            stats,
            physics_input_mode=physics_input_mode,
            physics_v1=physics_v1,
        )
        sync()
        prep_time = time.perf_counter() - prep_start
        graph = graph_batch or {}
        num_graphs = int(graph["num_graphs"].item())
        node_counts = torch.bincount(graph["node_batch"].detach().cpu(), minlength=num_graphs).numpy()
        edge_count = int(graph["edge_features"].shape[0])
        nodes_per_sample.extend(float(value) for value in node_counts.tolist())
        edges_per_sample.append(edge_count / max(num_graphs, 1))
        max_nodes = max(max_nodes, int(node_counts.max()) if len(node_counts) else 0)
        max_edges = max(max_edges, edge_count)

        sync()
        forward_start = time.perf_counter()
        profile_kwargs: dict[str, Any] = {
            "synchronize": sync,
            "graph_correction_scale": graph_correction_scale,
        }
        if getattr(model, "global_branch_enabled", False):
            profile_kwargs["global_correction_scale"] = global_correction_scale
        if getattr(model, "feature_fusion_enabled", False):
            profile_kwargs["disabled_fusion_scales"] = disabled_fusion_scales
        if mean_head_mode == "residual_resistance":
            profile_kwargs["total_power_W"] = total_power
        _outputs, component_times = model.forward_profile(model_input, metadata_input, graph_batch, **profile_kwargs)
        sync()
        total_time = time.perf_counter() - forward_start
        if batch_index >= warmup_batches:
            timing_values["batch_preparation_s"].append(prep_time)
            timing_values["total_forward_s"].append(total_time)
            for key, value in component_times.items():
                timing_values[key].append(float(value))
            measured += 1
        if measured >= profile_batches:
            break
    architecture = str(getattr(model, "architecture", ""))
    pairwise = architecture in {
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
    }
    return {
        "warmup_batches": warmup_batches,
        "profiled_batches": measured,
        "timings": {key: summarize_times(values) for key, values in sorted(timing_values.items())},
        "graph_size": {
            "average_nodes_per_sample": float(np.mean(nodes_per_sample)) if nodes_per_sample else None,
            "average_directed_edges_per_sample": float(np.mean(edges_per_sample)) if edges_per_sample else None,
            "maximum_nodes_per_sample": max_nodes,
            "maximum_directed_edges_per_batch": max_edges,
        },
        "rasterizer_audit": {
            "status": "vectorized over graphs, nodes, and grid cells; uses a small loop over raster-channel chunks to avoid materializing [N, C, H, W].",
            "gradient_path": (
                "pairwise node corrections are multiplied by differentiable halo weights and added directly to the centered field."
                if architecture == "miniunet_refine_conditioned_decomposed_pairwise"
                else "pairwise basis coefficients weight differentiable directional basis maps and are added directly to the centered field."
                if architecture == "miniunet_refine_conditioned_decomposed_pairwise_basis"
                else "node_raster_head outputs are multiplied by differentiable halo weights, then consumed by convolutional fusion."
            ),
            "optimization_opportunity": "Caching static node-to-grid weights helps repeated fixed-geometry inference; a custom kernel could fuse weight construction and accumulation if PyTorch memory traffic remains limiting.",
        },
    }


def summarize_times(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(array.mean()),
        "median_s": float(np.median(array)),
        "p95_s": float(np.percentile(array, 95)),
    }


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
        raise ValueError("graph-enabled checkpoint requires graph_path artifacts in the evaluation index")
    graph = move_graph_to_device(graph, device)
    return normalize_graph_batch(graph, graph_stats)


def call_model(
    model: nn.Module,
    model_input: torch.Tensor,
    metadata_input: torch.Tensor | None,
    graph_batch: dict[str, torch.Tensor] | None,
    *,
    conditioned: bool,
    graph_enabled: bool,
    graph_correction_scale: float = 1.0,
    global_correction_scale: float = 1.0,
    disabled_fusion_scales: tuple[str, ...] = (),
    total_power_W: torch.Tensor | None = None,
) -> Any:
    if graph_enabled:
        kwargs = {
            "return_diagnostics": True,
            "graph_correction_scale": graph_correction_scale,
        }
        if getattr(model, "global_branch_enabled", False):
            kwargs["global_correction_scale"] = global_correction_scale
        if getattr(model, "feature_fusion_enabled", False):
            kwargs["disabled_fusion_scales"] = disabled_fusion_scales
        return model(model_input, metadata_input, graph_batch, **kwargs)
    if conditioned:
        if getattr(model, "architecture", "") == "miniunet_refine_conditioned_decomposed_feature_fusion":
            return model(
                model_input,
                metadata_input,
                return_diagnostics=True,
                disabled_fusion_scales=disabled_fusion_scales,
                total_power_W=total_power_W,
            )
        if hasattr(model, "global_branch"):
            return model(
                model_input,
                metadata_input,
                return_diagnostics=True,
                global_correction_scale=global_correction_scale,
            )
        return model(model_input, metadata_input)
    return model(model_input)


def append_gate_rows(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    sample_uids: list[Any],
    case_ids: list[Any],
    alpha: torch.Tensor,
    pred_temperature: torch.Tensor,
    temperature: torch.Tensor,
    physics: torch.Tensor,
    centered_pred: torch.Tensor | None,
    centered_target: torch.Tensor | None,
    mean_rise: torch.Tensor | None,
    mean_target: torch.Tensor | None,
) -> None:
    alpha_cpu = alpha.detach().float().reshape(-1).cpu()
    pred_cpu = pred_temperature.detach().float().cpu()
    temp_cpu = temperature.detach().float().cpu()
    physics_cpu = physics.detach().float().cpu()
    centered_pred_cpu = centered_pred.detach().float().cpu() if centered_pred is not None else None
    centered_target_cpu = centered_target.detach().float().cpu() if centered_target is not None else None
    mean_rise_cpu = mean_rise.detach().float().cpu() if mean_rise is not None else None
    mean_target_cpu = mean_target.detach().float().cpu() if mean_target is not None else None
    metadata = batch["metadata"]
    feature_map = metadata.get("metadata_features", {}) if isinstance(metadata, dict) else {}
    for index, alpha_value in enumerate(alpha_cpu.tolist()):
        final_error = pred_cpu[index] - temp_cpu[index]
        physics_error = physics_cpu[index] - temp_cpu[index]
        row = {
            "sample_uid": str(sample_uids[index]),
            "case_id": str(case_ids[index]),
            "alpha": float(alpha_value),
            "final_mae_K": float(final_error.abs().mean().item()),
            "centered_field_mae_K": "",
            "mean_rise_abs_error_K": "",
            "physics_v1_mae_K": float(physics_error.abs().mean().item()),
            "occupied_fraction": metadata_feature_value(feature_map, "occupied_fraction", index),
            "whitespace_fraction": metadata_feature_value(feature_map, "whitespace_fraction", index),
            "total_power_W": metadata_feature_value(feature_map, "total_power_W", index),
            "mean_power_density_W_per_mm2": metadata_feature_value(feature_map, "mean_power_density_W_per_mm2", index),
            "minimum_pairwise_chiplet_distance_mm": "",
        }
        if centered_pred_cpu is not None and centered_target_cpu is not None:
            row["centered_field_mae_K"] = float((centered_pred_cpu[index] - centered_target_cpu[index]).abs().mean().item())
        if mean_rise_cpu is not None and mean_target_cpu is not None:
            row["mean_rise_abs_error_K"] = float(abs(mean_rise_cpu[index].item() - mean_target_cpu[index].item()))
        rows.append(row)


def append_graph_rows(
    rows: list[dict[str, Any]],
    batch: dict[str, Any],
    sample_uids: list[Any],
    case_ids: list[Any],
    cnn_only_temperature: torch.Tensor,
    fused_temperature: torch.Tensor,
    temperature: torch.Tensor,
    graph_correction: torch.Tensor,
) -> None:
    cnn_cpu = cnn_only_temperature.detach().float().cpu()
    fused_cpu = fused_temperature.detach().float().cpu()
    temp_cpu = temperature.detach().float().cpu()
    graph_cpu = graph_correction.detach().float().cpu()
    metadata = batch["metadata"]
    feature_map = metadata.get("metadata_features", {}) if isinstance(metadata, dict) else {}
    batch_size = int(cnn_cpu.shape[0])
    chiplet_counts = metadata_values(metadata, "num_chiplets", batch_size)
    for index in range(batch_size):
        cnn_error = cnn_cpu[index] - temp_cpu[index]
        fused_error = fused_cpu[index] - temp_cpu[index]
        graph_item = graph_cpu[index]
        cnn_mae = float(cnn_error.abs().mean().item())
        fused_mae = float(fused_error.abs().mean().item())
        rows.append(
            {
                "sample_uid": str(sample_uids[index]),
                "case_id": str(case_ids[index]),
                "cnn_only_mae_K": cnn_mae,
                "fused_mae_K": fused_mae,
                "mae_improvement_K": cnn_mae - fused_mae,
                "graph_correction_abs_mean_K": float(graph_item.abs().mean().item()),
                "graph_correction_abs_max_K": float(graph_item.abs().max().item()),
                "graph_correction_rms_K": float(torch.sqrt(torch.mean(graph_item * graph_item)).item()),
                "chiplet_count": chiplet_counts[index] if index < len(chiplet_counts) else "",
                "occupied_fraction": metadata_feature_value(feature_map, "occupied_fraction", index),
                "whitespace_fraction": metadata_feature_value(feature_map, "whitespace_fraction", index),
                "total_power_W": metadata_feature_value(feature_map, "total_power_W", index),
                "minimum_pairwise_chiplet_distance_mm": metadata_feature_value(feature_map, "minimum_pairwise_chiplet_distance_mm", index),
            }
        )


def write_graph_contribution_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_graph_contribution_by_case(path: Path, case_payload: dict[str, dict[str, dict[str, float]]]) -> None:
    columns = [
        "case_id",
        "cnn_only_mae_K",
        "cnn_only_rmse_K",
        "fused_mae_K",
        "fused_rmse_K",
        "mae_improvement_K",
        "rmse_improvement_K",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_payload.items()):
            cnn_only = metrics.get("cnn_only_final_temperature", {})
            fused = metrics.get("cnn_final_temperature", {})
            if not cnn_only or not fused:
                continue
            writer.writerow(
                {
                    "case_id": case_id,
                    "cnn_only_mae_K": cnn_only["mae_K"],
                    "cnn_only_rmse_K": cnn_only["rmse_K"],
                    "fused_mae_K": fused["mae_K"],
                    "fused_rmse_K": fused["rmse_K"],
                    "mae_improvement_K": cnn_only["mae_K"] - fused["mae_K"],
                    "rmse_improvement_K": cnn_only["rmse_K"] - fused["rmse_K"],
                }
            )


def write_graph_contribution_summary(
    path: Path,
    metrics: dict[str, Any],
    case_payload: dict[str, dict[str, dict[str, float]]],
) -> None:
    case02 = case_payload.get("case02", {})
    payload = {
        "overall": metrics.get("graph_improvement"),
        "graph_correction_abs_mean": metrics.get("graph_correction_abs_mean"),
        "graph_correction_abs_max": metrics.get("graph_correction_abs_max"),
        "graph_correction_rms": metrics.get("graph_correction_rms"),
        "graph_correction_spatial_std": metrics.get("graph_correction_spatial_std"),
        "graph_to_cnn_ratio": metrics.get("graph_to_cnn_ratio"),
        "case02": {
            "cnn_only": case02.get("cnn_only_final_temperature"),
            "fused": case02.get("cnn_final_temperature"),
        },
        "notes": "Positive MAE improvement means fused CNN-GNN is better than CNN-only within the same checkpoint.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_pairwise_summaries(
    outputs: dict[str, torch.Tensor],
    k_acc: ScalarSummaryAccumulator,
    contribution_acc: ScalarSummaryAccumulator,
    self_acc: ScalarSummaryAccumulator,
    node_acc: ScalarSummaryAccumulator,
    basis_coeff_acc: VectorSummaryAccumulator | None = None,
    basis_weighted_coeff_acc: VectorSummaryAccumulator | None = None,
) -> None:
    if "pairwise_k_values" in outputs:
        k_acc.update(outputs["pairwise_k_values"])
        contribution_acc.update(outputs["pairwise_contributions"])
        self_acc.update(outputs["pairwise_self_corrections"])
        node_acc.update(outputs["pairwise_node_corrections"])
    if "pairwise_basis_coefficients" in outputs:
        if basis_coeff_acc is not None:
            basis_coeff_acc.update(outputs["pairwise_basis_coefficients"])
        if basis_weighted_coeff_acc is not None:
            basis_weighted_coeff_acc.update(outputs["pairwise_basis_weighted_coefficients"])


def append_chiplet_rows(
    rows: list[dict[str, Any]],
    sample_uids: list[Any],
    case_ids: list[Any],
    chiplet_metrics: dict[str, torch.Tensor],
    graph_batch: dict[str, torch.Tensor],
) -> None:
    node_batch = graph_batch["node_batch"].detach().cpu().long()
    pred_mean = chiplet_metrics["pred_mean"].detach().cpu()
    target_mean = chiplet_metrics["target_mean"].detach().cpu()
    pred_peak = chiplet_metrics["pred_peak"].detach().cpu()
    target_peak = chiplet_metrics["target_peak"].detach().cpu()
    for graph_index, sample_uid in enumerate(sample_uids):
        node_indices = torch.nonzero(node_batch == graph_index, as_tuple=False).reshape(-1)
        if node_indices.numel() == 0:
            continue
        mean_error = (pred_mean.index_select(0, node_indices) - target_mean.index_select(0, node_indices)).abs()
        peak_error = (pred_peak.index_select(0, node_indices) - target_peak.index_select(0, node_indices)).abs()
        delta_mae = inter_chiplet_delta_mae_for_indices(pred_mean, target_mean, node_indices)
        rows.append(
            {
                "sample_uid": str(sample_uid),
                "case_id": str(case_ids[graph_index]),
                "chiplet_count": int(node_indices.numel()),
                "chiplet_mean_temperature_mae_K": float(mean_error.mean().item()),
                "chiplet_peak_temperature_mae_K": float(peak_error.mean().item()),
                "inter_chiplet_delta_T_mae_K": float(delta_mae.item()),
            }
        )


def update_chiplet_case_metrics(
    by_case: dict[str, dict[str, Any]],
    case_ids: list[Any],
    chiplet_metrics: dict[str, torch.Tensor],
    graph_batch: dict[str, torch.Tensor],
) -> None:
    node_batch = graph_batch["node_batch"].detach().cpu().long()
    pred_mean = chiplet_metrics["pred_mean"].detach().cpu()
    target_mean = chiplet_metrics["target_mean"].detach().cpu()
    pred_peak = chiplet_metrics["pred_peak"].detach().cpu()
    target_peak = chiplet_metrics["target_peak"].detach().cpu()
    for graph_index, case_id_value in enumerate(case_ids):
        case_id = str(case_id_value)
        accs = by_case[case_id]
        accs.setdefault("chiplet_mean_temperature", ScalarMetricAccumulator())
        accs.setdefault("chiplet_peak_temperature", ScalarMetricAccumulator())
        accs.setdefault("inter_chiplet_delta_T", ScalarSummaryAccumulator())
        node_indices = torch.nonzero(node_batch == graph_index, as_tuple=False).reshape(-1)
        if node_indices.numel() == 0:
            continue
        accs["chiplet_mean_temperature"].update(pred_mean.index_select(0, node_indices), target_mean.index_select(0, node_indices))
        accs["chiplet_peak_temperature"].update(pred_peak.index_select(0, node_indices), target_peak.index_select(0, node_indices))
        if node_indices.numel() >= 2:
            accs["inter_chiplet_delta_T"].update(inter_chiplet_delta_mae_for_indices(pred_mean, target_mean, node_indices).reshape(1))


def inter_chiplet_delta_mae_for_indices(pred_mean: torch.Tensor, target_mean: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    if int(node_indices.numel()) < 2:
        return pred_mean.new_tensor(0.0)
    pred = pred_mean.index_select(0, node_indices)
    target = target_mean.index_select(0, node_indices)
    pairs = torch.triu_indices(int(node_indices.numel()), int(node_indices.numel()), offset=1)
    pred_delta = pred[pairs[0]] - pred[pairs[1]]
    target_delta = target[pairs[0]] - target[pairs[1]]
    return (pred_delta - target_delta).abs().mean()


def write_chiplet_outputs(
    out_dir: Path,
    metrics: dict[str, Any],
    by_case: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    summary = {
        "chiplet_mean_temperature": metrics.get("chiplet_mean_temperature"),
        "chiplet_peak_temperature": metrics.get("chiplet_peak_temperature"),
        "inter_chiplet_delta_T": metrics.get("inter_chiplet_delta_T"),
        "cell_center_convention": "(row/col + 0.5) scaled by package height/width, matching graph rasterizer.",
        "rectangle_rule": "A grid cell belongs to a chiplet when its center is inside the closed chiplet rectangle [x0, x0+w] x [y0, y0+h].",
        "empty_mask_handling": "If a tiny chiplet has no cell centers inside its rectangle, the nearest grid cell to the chiplet center is assigned.",
    }
    (out_dir / "chiplet_metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = ["case_id", "chiplet_mean_temperature_mae_K", "chiplet_peak_temperature_mae_K", "inter_chiplet_delta_T_mae_K"]
    with (out_dir / "chiplet_metrics_by_case.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, accs in sorted(by_case.items()):
            chip_mean = accs.get("chiplet_mean_temperature")
            chip_peak = accs.get("chiplet_peak_temperature")
            chip_delta = accs.get("inter_chiplet_delta_T")
            writer.writerow(
                {
                    "case_id": case_id,
                    "chiplet_mean_temperature_mae_K": chip_mean.compute().get("mae_K", "") if chip_mean else "",
                    "chiplet_peak_temperature_mae_K": chip_peak.compute().get("mae_K", "") if chip_peak else "",
                    "inter_chiplet_delta_T_mae_K": chip_delta.compute().get("mean", "") if chip_delta else "",
                }
            )
    if rows:
        with (out_dir / "chiplet_metrics_by_sample.csv").open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def metadata_feature_value(feature_map: Any, name: str, index: int) -> float | str:
    if not isinstance(feature_map, dict) or name not in feature_map:
        return ""
    value = feature_map[name]
    if torch.is_tensor(value):
        return float(value[index].item())
    if isinstance(value, (list, tuple)):
        return float(value[index])
    return float(value)


def write_gate_values(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_gate_summary(path: Path, rows: list[dict[str, Any]], overall: dict[str, float]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    case_summary = {
        case_id: {
            "num_samples": len(items),
            "alpha_mean": mean_float(item["alpha"] for item in items),
            "alpha_std": std_float(item["alpha"] for item in items),
            "final_mae_K": mean_float(item["final_mae_K"] for item in items),
            "centered_field_mae_K": mean_float(item["centered_field_mae_K"] for item in items),
            "physics_v1_mae_K": mean_float(item["physics_v1_mae_K"] for item in items),
        }
        for case_id, items in sorted(by_case.items())
    }
    correlations = {}
    for key in (
        "final_mae_K",
        "centered_field_mae_K",
        "physics_v1_mae_K",
        "occupied_fraction",
        "whitespace_fraction",
        "total_power_W",
        "mean_power_density_W_per_mm2",
    ):
        correlations[f"alpha_vs_{key}"] = pearson_corr(rows, "alpha", key)
    payload = {
        "overall": overall,
        "by_case": case_summary,
        "correlations": correlations,
        "notes": "minimum_pairwise_chiplet_distance_mm is left blank here because it is not present in compact metadata_features.csv.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case_path = path.with_name("gate_by_case.csv")
    with case_path.open("w", encoding="utf-8", newline="") as fp:
        columns = ["case_id", "num_samples", "alpha_mean", "alpha_std", "final_mae_K", "centered_field_mae_K", "physics_v1_mae_K"]
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, summary in case_summary.items():
            writer.writerow({"case_id": case_id, **summary})


def numeric_values(values: Any) -> list[float]:
    result = []
    for value in values:
        if value == "" or value is None:
            continue
        result.append(float(value))
    return result


def mean_float(values: Any) -> float | None:
    data = numeric_values(values)
    if not data:
        return None
    return float(np.mean(np.asarray(data, dtype=np.float64)))


def std_float(values: Any) -> float | None:
    data = numeric_values(values)
    if not data:
        return None
    return float(np.std(np.asarray(data, dtype=np.float64)))


def pearson_corr(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    pairs = [(row[x_key], row[y_key]) for row in rows if row.get(x_key) not in {"", None} and row.get(y_key) not in {"", None}]
    if len(pairs) < 2:
        return None
    x = np.asarray([float(pair[0]) for pair in pairs], dtype=np.float64)
    y = np.asarray([float(pair[1]) for pair in pairs], dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class MetricAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_cells = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
        self.sum_sample_rmse = 0.0
        self.max_abs = 0.0
        self.hotspot_temp_error_sum = 0.0
        self.hotspot_location_error_sum = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_cpu = pred.detach().float().cpu()
        target_cpu = target.detach().float().cpu()
        error = pred_cpu - target_cpu
        abs_error = error.abs()
        self.num_samples += int(pred_cpu.shape[0])
        self.num_cells += int(error.numel())
        self.sum_abs += float(abs_error.sum().item())
        self.sum_sq += float((error * error).sum().item())
        self.sum_signed += float(error.sum().item())
        self.sum_sample_rmse += float(torch.sqrt(torch.mean(error.reshape(error.shape[0], -1) ** 2, dim=1)).sum().item())
        self.max_abs = max(self.max_abs, float(abs_error.max().item()))
        for pred_item, target_item in zip(pred_cpu, target_cpu):
            pred_flat = pred_item.reshape(-1)
            target_flat = target_item.reshape(-1)
            pred_idx = int(torch.argmax(pred_flat).item())
            target_idx = int(torch.argmax(target_flat).item())
            pred_row, pred_col = divmod(pred_idx, pred_item.shape[-1])
            target_row, target_col = divmod(target_idx, target_item.shape[-1])
            self.hotspot_temp_error_sum += float(pred_flat[pred_idx].item() - target_flat[target_idx].item())
            self.hotspot_location_error_sum += float(((pred_row - target_row) ** 2 + (pred_col - target_col) ** 2) ** 0.5)

    def compute(self) -> dict[str, float]:
        if self.num_cells == 0:
            return {}
        global_pixel_rmse = (self.sum_sq / self.num_cells) ** 0.5
        return {
            "num_samples": float(self.num_samples),
            "num_cells": float(self.num_cells),
            "mae_K": self.sum_abs / self.num_cells,
            "global_pixel_rmse_K": global_pixel_rmse,
            "mean_sample_rmse_K": self.sum_sample_rmse / max(self.num_samples, 1),
            "rmse_K": global_pixel_rmse,
            "max_abs_error_K": self.max_abs,
            "mean_signed_error_K": self.sum_signed / self.num_cells,
            "hotspot_temp_error_K": self.hotspot_temp_error_sum / max(self.num_samples, 1),
            "hotspot_location_error_cells": self.hotspot_location_error_sum / max(self.num_samples, 1),
        }


class ScalarMetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        error = pred.detach().float().cpu() - target.detach().float().cpu()
        self.count += int(error.numel())
        self.sum_abs += float(error.abs().sum().item())
        self.sum_sq += float((error * error).sum().item())
        self.sum_signed += float(error.sum().item())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "mae_K": self.sum_abs / self.count,
            "rmse_K": (self.sum_sq / self.count) ** 0.5,
            "mean_signed_error_K": self.sum_signed / self.count,
        }


class ScalarSummaryAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: torch.Tensor) -> None:
        self.values.extend(float(item) for item in value.detach().float().reshape(-1).cpu().tolist())

    def compute(self) -> dict[str, float]:
        if not self.values:
            return {}
        array = np.asarray(self.values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "abs_mean": float(np.abs(array).mean()),
        }


class VectorSummaryAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total: torch.Tensor | None = None
        self.total_abs: torch.Tensor | None = None
        self.total_sq: torch.Tensor | None = None
        self.positive: torch.Tensor | None = None
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        data = value.detach().float().reshape(-1, value.shape[-1]).cpu()
        if data.numel() == 0:
            return
        if self.total is None:
            dim = int(data.shape[1])
            self.total = torch.zeros(dim, dtype=torch.float64)
            self.total_abs = torch.zeros(dim, dtype=torch.float64)
            self.total_sq = torch.zeros(dim, dtype=torch.float64)
            self.positive = torch.zeros(dim, dtype=torch.float64)
            self.minimum = torch.full((dim,), float("inf"), dtype=torch.float64)
            self.maximum = torch.full((dim,), -float("inf"), dtype=torch.float64)
        data64 = data.double()
        self.count += int(data64.shape[0])
        self.total += data64.sum(dim=0)
        self.total_abs += data64.abs().sum(dim=0)
        self.total_sq += (data64 * data64).sum(dim=0)
        self.positive += (data64 > 0.0).double().sum(dim=0)
        self.minimum = torch.minimum(self.minimum, data64.min(dim=0).values)
        self.maximum = torch.maximum(self.maximum, data64.max(dim=0).values)

    def compute(self) -> dict[str, Any]:
        if self.count == 0 or self.total is None:
            return {}
        mean = self.total / float(self.count)
        abs_mean = self.total_abs / float(self.count)
        variance = torch.clamp(self.total_sq / float(self.count) - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        positive_fraction = self.positive / float(self.count)
        by_basis = []
        for index in range(int(mean.numel())):
            by_basis.append(
                {
                    "basis_index": index,
                    "mean": float(mean[index].item()),
                    "std": float(std[index].item()),
                    "abs_mean": float(abs_mean[index].item()),
                    "min": float(self.minimum[index].item()),
                    "max": float(self.maximum[index].item()),
                    "positive_fraction": float(positive_fraction[index].item()),
                    "negative_fraction": float(1.0 - positive_fraction[index].item()),
                }
            )
        return {
            "mean": float(mean.mean().item()),
            "std": float(std.mean().item()),
            "abs_mean": float(abs_mean.mean().item()),
            "min": float(self.minimum.min().item()),
            "max": float(self.maximum.max().item()),
            "by_basis": by_basis,
        }


def save_batch_predictions(
    out_dir: Path,
    sample_uids: list[Any],
    case_ids: list[Any],
    pred_temperature: torch.Tensor,
    pred_residual: torch.Tensor,
) -> None:
    pred_temperature_cpu = pred_temperature.detach().float().cpu().numpy().astype(np.float32, copy=False)
    pred_residual_cpu = pred_residual.detach().float().cpu().numpy().astype(np.float32, copy=False)
    for index, sample_uid in enumerate(sample_uids):
        case_id = str(case_ids[index])
        case_dir = out_dir / "predictions" / case_id
        residual_dir = out_dir / "predicted_residuals" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        residual_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / f"{sample_uid}_tpred.npy", pred_temperature_cpu[index])
        np.save(residual_dir / f"{sample_uid}_residual_pred.npy", pred_residual_cpu[index])


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    columns = [
        "case_id",
        "physics_mae_K",
        "physics_rmse_K",
        "physics_v1_auxiliary_mae_K",
        "physics_v1_auxiliary_rmse_K",
        "cnn_final_mae_K",
        "cnn_final_rmse_K",
        "cnn_only_final_mae_K",
        "cnn_only_final_rmse_K",
        "graph_delta_mae_K",
        "cnn_final_max_abs_error_K",
        "cnn_final_mean_signed_error_K",
        "cnn_hotspot_temp_error_K",
        "cnn_hotspot_location_error_cells",
        "coarse_final_mae_K",
        "coarse_final_rmse_K",
        "chiplet_mean_mae_K",
        "chiplet_peak_mae_K",
        "inter_chiplet_delta_mae_K",
        "mae_improvement_percent",
        "rmse_improvement_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            physics = metrics["physics_baseline"]
            physics_v1 = metrics.get("physics_v1_auxiliary", {})
            final = metrics["cnn_final_temperature"]
            cnn_only = metrics.get("cnn_only_final_temperature", {})
            coarse = metrics.get("coarse_final_temperature", {})
            chiplet_mean = metrics.get("chiplet_mean_temperature", {})
            chiplet_peak = metrics.get("chiplet_peak_temperature", {})
            chiplet_delta = metrics.get("inter_chiplet_delta_T", {})
            writer.writerow(
                {
                    "case_id": case_id,
                    "physics_mae_K": physics["mae_K"],
                    "physics_rmse_K": physics["rmse_K"],
                    "physics_v1_auxiliary_mae_K": physics_v1.get("mae_K", ""),
                    "physics_v1_auxiliary_rmse_K": physics_v1.get("rmse_K", ""),
                    "cnn_final_mae_K": final["mae_K"],
                    "cnn_final_rmse_K": final["rmse_K"],
                    "cnn_only_final_mae_K": cnn_only.get("mae_K", ""),
                    "cnn_only_final_rmse_K": cnn_only.get("rmse_K", ""),
                    "graph_delta_mae_K": (cnn_only.get("mae_K", 0.0) - final["mae_K"]) if cnn_only else "",
                    "cnn_final_max_abs_error_K": final["max_abs_error_K"],
                    "cnn_final_mean_signed_error_K": final["mean_signed_error_K"],
                    "cnn_hotspot_temp_error_K": final["hotspot_temp_error_K"],
                    "cnn_hotspot_location_error_cells": final["hotspot_location_error_cells"],
                    "coarse_final_mae_K": coarse.get("mae_K", ""),
                    "coarse_final_rmse_K": coarse.get("rmse_K", ""),
                    "chiplet_mean_mae_K": chiplet_mean.get("mae_K", ""),
                    "chiplet_peak_mae_K": chiplet_peak.get("mae_K", ""),
                    "inter_chiplet_delta_mae_K": chiplet_delta.get("mean", ""),
                    "mae_improvement_percent": percent_improvement(physics["mae_K"], final["mae_K"]),
                    "rmse_improvement_percent": percent_improvement(physics["rmse_K"], final["rmse_K"]),
                }
            )


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def optional_float_values(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        result.append(float(value))
    return result


def percent_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0
    return float((baseline - candidate) / baseline * 100.0)


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    raise SystemExit(main())
