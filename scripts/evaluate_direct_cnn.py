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
from chiptherm.ml.models import build_model


EPSILON = 1.0e-8


@dataclass(frozen=True)
class DirectNormalizationStats:
    schema_version: int
    input_channels: int
    power_density_mean: float
    power_density_std: float
    context_channel_indices: list[int]
    context_channel_means: list[float]
    context_channel_stds: list[float]
    temperature_mean: float
    temperature_std: float
    num_samples: int
    num_grid_cells: int
    notes: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained direct ChipTherm temperature CNN.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--index",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power/test_index.csv",
        type=Path,
    )
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/direct_cnn_package_plus_power_base32/test_eval", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = DirectNormalizationStats(**checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ChipThermDataset(args.index, target="temperature", return_metadata=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metrics, by_case, runtime_s, hotspot_runtime_s = evaluate(
        model,
        loader,
        stats,
        device,
        save_predictions=args.save_predictions,
        out_dir=out_dir,
    )
    runtime_per_sample = runtime_s / max(metrics["num_samples"], 1)
    speedup = hotspot_runtime_s / runtime_per_sample if hotspot_runtime_s and runtime_per_sample else None
    runtime_note = (
        "Direct CNN timing includes input normalization, CNN forward, and temperature unnormalization. "
        "It does not compute or use T_phys. Disk I/O is excluded from the timed region."
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "index": str(args.index.resolve()),
        "num_samples": metrics["num_samples"],
        "inference_runtime_total_s": runtime_s,
        "inference_runtime_per_sample_s": runtime_per_sample,
        "hotspot_runtime_reference_s": hotspot_runtime_s,
        "estimated_speedup_vs_hotspot": speedup,
        "runtime": {
            "hotspot_runtime_reference_s": hotspot_runtime_s,
            "direct_cnn_runtime_per_sample_s": runtime_per_sample,
            "estimated_speedup_vs_hotspot": speedup,
            "timing_note": runtime_note,
        },
        "direct_cnn_temperature": metrics["direct_cnn_temperature"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case)

    temp_metrics = metrics["direct_cnn_temperature"]
    print("Direct CNN evaluation complete")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Direct CNN inference runtime/sample: {runtime_per_sample:.6f} s")
    print(f"HotSpot runtime reference: {hotspot_runtime_s:.6f} s" if hotspot_runtime_s else "HotSpot runtime reference: n/a")
    print(f"Speedup vs HotSpot: {speedup:.1f}x" if speedup else "Speedup vs HotSpot: n/a")
    print(f"Temperature MAE/RMSE: {temp_metrics['mae_K']:.3f} / {temp_metrics['rmse_K']:.3f} K")
    print(f"Max abs error: {temp_metrics['max_abs_error_K']:.3f} K")
    print(f"Mean signed error: {temp_metrics['mean_signed_error_K']:.3f} K")
    print(f"Hotspot temp/location error: {temp_metrics['hotspot_temp_error_K']:.3f} K / {temp_metrics['hotspot_location_error_cells']:.3f} cells")
    print(f"Output: {out_dir}")
    return 0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    stats: DirectNormalizationStats,
    device: torch.device,
    *,
    save_predictions: bool,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], float, float | None]:
    acc = MetricAccumulator()
    by_case: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    hotspot_runtimes: list[float] = []
    inference_runtime_s = 0.0
    num_samples = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)

        synchronize(device)
        start = time.perf_counter()
        model_input = normalize_x(x, stats)
        pred_norm = model(model_input)
        pred_temperature = unnormalize_temperature(pred_norm.squeeze(1), stats)
        synchronize(device)
        inference_runtime_s += time.perf_counter() - start

        batch_size = int(x.shape[0])
        num_samples += batch_size
        case_ids = metadata_values(batch["metadata"], "case_id", batch_size)
        sample_uids = metadata_values(batch["metadata"], "sample_uid", batch_size)
        hotspot_runtimes.extend(optional_float_values(metadata_values(batch["metadata"], "hotspot_runtime_s", batch_size)))

        acc.update(pred_temperature, temperature)
        for index, case_id in enumerate(case_ids):
            by_case[str(case_id)].update(pred_temperature[index : index + 1], temperature[index : index + 1])

        if save_predictions:
            save_batch_predictions(out_dir, sample_uids, case_ids, pred_temperature)

    metrics = {
        "num_samples": num_samples,
        "direct_cnn_temperature": acc.compute(),
    }
    case_payload = {case_id: case_acc.compute() for case_id, case_acc in sorted(by_case.items())}
    hotspot_runtime_s = float(sum(hotspot_runtimes) / len(hotspot_runtimes)) if hotspot_runtimes else None
    return metrics, case_payload, inference_runtime_s, hotspot_runtime_s


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


def normalize_x(x: torch.Tensor, stats: DirectNormalizationStats) -> torch.Tensor:
    x_norm = x.float().clone()
    x_norm[:, 0] = normalize_tensor(x_norm[:, 0], stats.power_density_mean, stats.power_density_std)
    for channel, mean, std in zip(stats.context_channel_indices, stats.context_channel_means, stats.context_channel_stds):
        if int(channel) < x_norm.shape[1]:
            x_norm[:, int(channel)] = normalize_tensor(x_norm[:, int(channel)], mean, std)
    return x_norm


def unnormalize_temperature(temperature_norm: torch.Tensor, stats: DirectNormalizationStats) -> torch.Tensor:
    return temperature_norm.float() * float(stats.temperature_std) + float(stats.temperature_mean)


def normalize_tensor(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (value - float(mean)) / max(float(std), EPSILON)


def save_batch_predictions(
    out_dir: Path,
    sample_uids: list[Any],
    case_ids: list[Any],
    pred_temperature: torch.Tensor,
) -> None:
    pred_temperature_cpu = pred_temperature.detach().float().cpu().numpy().astype(np.float32, copy=False)
    for index, sample_uid in enumerate(sample_uids):
        case_id = str(case_ids[index])
        case_dir = out_dir / "predictions" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / f"{sample_uid}_temperature_pred.npy", pred_temperature_cpu[index])


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, float]]) -> None:
    columns = [
        "case_id",
        "num_samples",
        "mae_K",
        "rmse_K",
        "max_abs_error_K",
        "mean_signed_error_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            row = {"case_id": case_id}
            row.update({column: metrics.get(column, "") for column in columns if column != "case_id"})
            writer.writerow(row)


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
