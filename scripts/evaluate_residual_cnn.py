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

from chiptherm.ml.dataset import ChipThermDataset
from chiptherm.ml.models import build_model, count_parameters
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input, unnormalize_residual


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
    args = parser.parse_args()

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = NormalizationStats(**checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    architecture = str(checkpoint["model_config"].get("architecture", "miniunet"))
    decomposed = architecture in {"miniunet_refine_decomposed", "miniunet_refine_conditioned_decomposed"}
    conditioned = architecture in {"miniunet_refine_conditioned", "miniunet_refine_conditioned_decomposed"}

    dataset = ChipThermDataset(args.index, target="residual", return_metadata=True)
    actual_input_channels = int(dataset[0]["x"].shape[0]) + 1
    expected_input_channels = int(checkpoint["model_config"].get("input_channels", actual_input_channels))
    if actual_input_channels != expected_input_channels:
        raise SystemExit(
            f"checkpoint expects {expected_input_channels} model input channels, "
            f"but dataset provides {actual_input_channels} channels including physics"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metrics, by_case, runtime_s, hotspot_runtime_s, physics_runtime_s = evaluate(
        model,
        loader,
        stats,
        device,
        measure_end_to_end=args.measure_end_to_end,
        save_predictions=args.save_predictions,
        out_dir=out_dir,
        decomposed=decomposed,
        conditioned=conditioned,
    )
    cnn_runtime_per_sample = runtime_s / max(metrics["num_samples"], 1)
    cnn_side_speedup = hotspot_runtime_s / cnn_runtime_per_sample if hotspot_runtime_s and cnn_runtime_per_sample else None
    end_to_end_runtime_per_sample = None
    end_to_end_speedup = None
    timing_note = (
        "CNN-side timing uses loaded T_phys and includes normalization/input assembly, CNN forward, "
        "residual unnormalization, and final temperature reconstruction. Disk I/O is excluded."
    )
    if args.measure_end_to_end:
        if physics_runtime_s is None:
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
        "centered_field": metrics.get("centered_field"),
        "mean_bias_removed": metrics.get("mean_bias_removed"),
        "improvement_vs_physics_baseline": improvement,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case)

    print("Residual CNN evaluation complete")
    print(f"Samples: {metrics['num_samples']}")
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
    print(f"CNN final MAE/RMSE: {final_mae:.3f} / {final_rmse:.3f} K")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Improvement: MAE {improvement['mae_percent']:.2f}% / RMSE {improvement['rmse_percent']:.2f}%")
    print(f"Output: {out_dir}")
    return 0


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
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]], float, float | None, float | None]:
    residual_acc = MetricAccumulator()
    final_acc = MetricAccumulator()
    physics_acc = MetricAccumulator()
    coarse_final_acc = MetricAccumulator()
    mean_acc = ScalarMetricAccumulator()
    centered_acc = MetricAccumulator()
    mean_bias_removed_acc = MetricAccumulator()
    has_coarse_prediction = False
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            "cnn_residual": MetricAccumulator(),
            "cnn_final_temperature": MetricAccumulator(),
            "physics_baseline": MetricAccumulator(),
            "coarse_final_temperature": MetricAccumulator(),
        }
    )
    hotspot_runtimes: list[float] = []
    physics_runtimes: list[float] = []
    inference_runtime_s = 0.0
    num_samples = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)

        synchronize(device)
        start = time.perf_counter()
        model_input = build_model_input(x, physics, stats)
        coarse_norm = None
        if decomposed:
            outputs = model(model_input, metadata_input) if conditioned else model(model_input)
            pred_temperature = reconstruct_decomposed_temperature(outputs, ambient)
            pred_residual = pred_temperature - physics
            centered_pred = pred_temperature - pred_temperature.mean(dim=(-2, -1), keepdim=True)
            centered_target = temperature - temperature.mean(dim=(-2, -1), keepdim=True)
            mean_target = (temperature - ambient[:, None, None]).mean(dim=(-2, -1))
            mean_acc.update(outputs["mean_rise"], mean_target)
            centered_acc.update(centered_pred, centered_target)
            mean_bias_removed_acc.update(centered_pred, centered_target)
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
        if coarse_temperature is not None:
            coarse_final_acc.update(coarse_temperature, temperature)
        for index, case_id in enumerate(case_ids):
            case_metrics = by_case[str(case_id)]
            case_metrics["cnn_residual"].update(pred_residual[index : index + 1], residual[index : index + 1])
            case_metrics["cnn_final_temperature"].update(pred_temperature[index : index + 1], temperature[index : index + 1])
            case_metrics["physics_baseline"].update(physics[index : index + 1], temperature[index : index + 1])
            if coarse_temperature is not None:
                case_metrics["coarse_final_temperature"].update(coarse_temperature[index : index + 1], temperature[index : index + 1])

        if save_predictions:
            save_batch_predictions(out_dir, sample_uids, case_ids, pred_temperature, pred_residual)

    metrics = {
        "num_samples": num_samples,
        "cnn_residual": residual_acc.compute(),
        "cnn_final_temperature": final_acc.compute(),
        "physics_baseline": physics_acc.compute(),
    }
    if has_coarse_prediction:
        metrics["coarse_final_temperature"] = coarse_final_acc.compute()
    if decomposed:
        metrics["mean_rise"] = mean_acc.compute()
        metrics["centered_field"] = centered_acc.compute()
        metrics["mean_bias_removed"] = mean_bias_removed_acc.compute()
    case_payload = {
        case_id: {
            name: accumulator.compute()
            for name, accumulator in sorted(accs.items())
        }
        for case_id, accs in sorted(by_case.items())
    }
    hotspot_runtime_s = float(sum(hotspot_runtimes) / len(hotspot_runtimes)) if hotspot_runtimes else None
    physics_runtime_s = float(sum(physics_runtimes) / len(physics_runtimes)) if physics_runtimes else None
    return metrics, case_payload, inference_runtime_s, hotspot_runtime_s, physics_runtime_s


def reconstruct_decomposed_temperature(outputs: dict[str, torch.Tensor], ambient: torch.Tensor) -> torch.Tensor:
    centered = outputs["centered_field"]
    centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
    return ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered


class MetricAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_cells = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
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
        return {
            "num_samples": float(self.num_samples),
            "mae_K": self.sum_abs / self.num_cells,
            "rmse_K": (self.sum_sq / self.num_cells) ** 0.5,
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
        "cnn_final_mae_K",
        "cnn_final_rmse_K",
        "cnn_final_max_abs_error_K",
        "cnn_final_mean_signed_error_K",
        "cnn_hotspot_temp_error_K",
        "cnn_hotspot_location_error_cells",
        "coarse_final_mae_K",
        "coarse_final_rmse_K",
        "mae_improvement_percent",
        "rmse_improvement_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            physics = metrics["physics_baseline"]
            final = metrics["cnn_final_temperature"]
            coarse = metrics.get("coarse_final_temperature", {})
            writer.writerow(
                {
                    "case_id": case_id,
                    "physics_mae_K": physics["mae_K"],
                    "physics_rmse_K": physics["rmse_K"],
                    "cnn_final_mae_K": final["mae_K"],
                    "cnn_final_rmse_K": final["rmse_K"],
                    "cnn_final_max_abs_error_K": final["max_abs_error_K"],
                    "cnn_final_mean_signed_error_K": final["mean_signed_error_K"],
                    "cnn_hotspot_temp_error_K": final["hotspot_temp_error_K"],
                    "cnn_hotspot_location_error_cells": final["hotspot_location_error_cells"],
                    "coarse_final_mae_K": coarse.get("mae_K", ""),
                    "coarse_final_rmse_K": coarse.get("rmse_K", ""),
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
