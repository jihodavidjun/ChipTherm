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

from chiptherm.ml.dual_physics import (  # noqa: E402
    DualPhysicsDataset,
    DualPhysicsNormalizationStats,
    build_dual_physics_model_input,
    unnormalize_residual_v1,
)
from chiptherm.ml.models import build_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained dual-physics ChipTherm residual CNN.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--physics-v1-index", required=True, type=Path)
    parser.add_argument("--physics-v2-index", required=True, type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/residual_cnn_dual_physics_package_plus_power_base32/test_eval_e2e", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--measure-end-to-end", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-batches", default=None, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    stats = DualPhysicsNormalizationStats.from_dict(checkpoint["normalization"])
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = DualPhysicsDataset(args.physics_v1_index, args.physics_v2_index)
    dataset_input_channels = int(dataset[0]["x"].shape[0])
    expected_model_channels = dataset_input_channels + 2
    if int(checkpoint["model_config"]["input_channels"]) != expected_model_channels:
        raise SystemExit(
            f"checkpoint input channels {checkpoint['model_config']['input_channels']} "
            f"do not match dataset channels {dataset_input_channels}+2"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    metrics, by_case, runtime_s, hotspot_runtime_s, physics_v1_runtime_s, physics_v2_runtime_s = evaluate(
        model,
        loader,
        stats,
        device,
        save_predictions=args.save_predictions,
        out_dir=out_dir,
        max_batches=args.max_batches,
    )
    cnn_runtime_per_sample = runtime_s / max(metrics["num_samples"], 1)
    cnn_side_speedup = hotspot_runtime_s / cnn_runtime_per_sample if hotspot_runtime_s and cnn_runtime_per_sample else None
    end_to_end_runtime_per_sample = None
    end_to_end_speedup = None
    timing_note = (
        "CNN-side timing includes dual input normalization, CNN forward, residual unnormalization, "
        "and final temperature reconstruction. Disk I/O is excluded."
    )
    if args.measure_end_to_end:
        if physics_v1_runtime_s is None or physics_v2_runtime_s is None:
            timing_note += " End-to-end timing requested, but one or both physics runtime metadata fields are unavailable."
        else:
            end_to_end_runtime_per_sample = physics_v1_runtime_s + physics_v2_runtime_s + cnn_runtime_per_sample
            end_to_end_speedup = hotspot_runtime_s / end_to_end_runtime_per_sample if hotspot_runtime_s else None
            timing_note += (
                " End-to-end timing is estimated as physics_v1 runtime + physics_v2 runtime + CNN-side runtime; "
                "physics is not recomputed in this script."
            )

    final_mae = metrics["cnn_final_temperature"]["mae_K"]
    final_rmse = metrics["cnn_final_temperature"]["rmse_K"]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "physics_v1_index": str(args.physics_v1_index.resolve()),
        "physics_v2_index": str(args.physics_v2_index.resolve()),
        "num_samples": metrics["num_samples"],
        "inference_runtime_total_s": runtime_s,
        "inference_runtime_per_sample_s": cnn_runtime_per_sample,
        "hotspot_runtime_reference_s": hotspot_runtime_s,
        "estimated_speedup_vs_hotspot": cnn_side_speedup,
        "runtime": {
            "hotspot_runtime_reference_s": hotspot_runtime_s,
            "cnn_runtime_per_sample_s": cnn_runtime_per_sample,
            "physics_v1_runtime_per_sample_s": physics_v1_runtime_s,
            "physics_v2_runtime_per_sample_s": physics_v2_runtime_s,
            "dual_physics_end_to_end_runtime_per_sample_s": end_to_end_runtime_per_sample,
            "cnn_side_speedup_vs_hotspot": cnn_side_speedup,
            "dual_physics_end_to_end_speedup_vs_hotspot": end_to_end_speedup,
            "timing_note": timing_note,
        },
        "physics_v1_baseline": metrics["physics_v1_baseline"],
        "physics_v2_auxiliary": metrics["physics_v2_auxiliary"],
        "cnn_residual_v1": metrics["cnn_residual_v1"],
        "cnn_final_temperature": metrics["cnn_final_temperature"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case)

    print("Dual-physics residual CNN evaluation complete")
    print(f"Samples: {metrics['num_samples']}")
    print(f"CNN-side inference runtime/sample: {cnn_runtime_per_sample:.6f} s")
    if args.measure_end_to_end:
        print(f"physics_v1 runtime/sample: {physics_v1_runtime_s:.6f} s" if physics_v1_runtime_s else "physics_v1 runtime/sample: n/a")
        print(f"physics_v2 runtime/sample: {physics_v2_runtime_s:.6f} s" if physics_v2_runtime_s else "physics_v2 runtime/sample: n/a")
        print(f"Dual end-to-end runtime/sample: {end_to_end_runtime_per_sample:.6f} s" if end_to_end_runtime_per_sample else "Dual end-to-end runtime/sample: n/a")
    print(f"HotSpot runtime reference: {hotspot_runtime_s:.6f} s" if hotspot_runtime_s else "HotSpot runtime reference: n/a")
    print(f"CNN-side speedup: {cnn_side_speedup:.1f}x" if cnn_side_speedup else "CNN-side speedup: n/a")
    if args.measure_end_to_end:
        print(f"Dual end-to-end speedup: {end_to_end_speedup:.1f}x" if end_to_end_speedup else "Dual end-to-end speedup: n/a")
    print(f"CNN final MAE/RMSE: {final_mae:.3f} / {final_rmse:.3f} K")
    print(f"Output: {out_dir}")
    return 0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    stats: DualPhysicsNormalizationStats,
    device: torch.device,
    *,
    save_predictions: bool,
    out_dir: Path,
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]], float, float | None, float | None, float | None]:
    residual_acc = MetricAccumulator()
    final_acc = MetricAccumulator()
    physics_v1_acc = MetricAccumulator()
    physics_v2_acc = MetricAccumulator()
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            "cnn_residual_v1": MetricAccumulator(),
            "cnn_final_temperature": MetricAccumulator(),
            "physics_v1_baseline": MetricAccumulator(),
            "physics_v2_auxiliary": MetricAccumulator(),
        }
    )
    hotspot_runtimes: list[float] = []
    physics_v1_runtimes: list[float] = []
    physics_v2_runtimes: list[float] = []
    inference_runtime_s = 0.0
    num_samples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        physics_v1 = batch["physics_v1"].to(device, non_blocking=True)
        physics_v2 = batch["physics_v2"].to(device, non_blocking=True)
        residual_v1 = batch["residual_v1"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        synchronize(device)
        start = time.perf_counter()
        model_input = build_dual_physics_model_input(x, physics_v1, physics_v2, stats)
        pred_norm = model(model_input)
        pred_residual = unnormalize_residual_v1(pred_norm.squeeze(1), stats)
        pred_temperature = physics_v1 + pred_residual
        synchronize(device)
        inference_runtime_s += time.perf_counter() - start

        batch_size = int(x.shape[0])
        num_samples += batch_size
        metadata = batch["metadata"]
        case_ids = metadata_values(metadata, "case_id", batch_size)
        sample_uids = metadata_values(metadata, "sample_uid", batch_size)
        hotspot_runtimes.extend(optional_float_values(metadata_values(metadata, "hotspot_runtime_s", batch_size)))
        physics_v1_runtimes.extend(optional_float_values(metadata_values(metadata, "physics_v1_runtime_s", batch_size)))
        physics_v2_runtimes.extend(optional_float_values(metadata_values(metadata, "physics_v2_runtime_s", batch_size)))

        residual_acc.update(pred_residual, residual_v1)
        final_acc.update(pred_temperature, temperature)
        physics_v1_acc.update(physics_v1, temperature)
        physics_v2_acc.update(physics_v2, temperature)
        for index, case_id in enumerate(case_ids):
            accs = by_case[str(case_id)]
            accs["cnn_residual_v1"].update(pred_residual[index : index + 1], residual_v1[index : index + 1])
            accs["cnn_final_temperature"].update(pred_temperature[index : index + 1], temperature[index : index + 1])
            accs["physics_v1_baseline"].update(physics_v1[index : index + 1], temperature[index : index + 1])
            accs["physics_v2_auxiliary"].update(physics_v2[index : index + 1], temperature[index : index + 1])
        if save_predictions:
            save_batch_predictions(out_dir, sample_uids, case_ids, pred_temperature, pred_residual)

    metrics = {
        "num_samples": num_samples,
        "cnn_residual_v1": residual_acc.compute(),
        "cnn_final_temperature": final_acc.compute(),
        "physics_v1_baseline": physics_v1_acc.compute(),
        "physics_v2_auxiliary": physics_v2_acc.compute(),
    }
    case_payload = {
        case_id: {name: accumulator.compute() for name, accumulator in sorted(accs.items())}
        for case_id, accs in sorted(by_case.items())
    }
    return (
        metrics,
        case_payload,
        inference_runtime_s,
        mean_or_none(hotspot_runtimes),
        mean_or_none(physics_v1_runtimes),
        mean_or_none(physics_v2_runtimes),
    )


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


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    columns = [
        "case_id",
        "physics_v1_mae_K",
        "physics_v2_mae_K",
        "cnn_final_mae_K",
        "cnn_final_rmse_K",
        "cnn_final_max_abs_error_K",
        "cnn_final_mean_signed_error_K",
        "cnn_hotspot_temp_error_K",
        "cnn_hotspot_location_error_cells",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            final = metrics["cnn_final_temperature"]
            writer.writerow({
                "case_id": case_id,
                "physics_v1_mae_K": metrics["physics_v1_baseline"]["mae_K"],
                "physics_v2_mae_K": metrics["physics_v2_auxiliary"]["mae_K"],
                "cnn_final_mae_K": final["mae_K"],
                "cnn_final_rmse_K": final["rmse_K"],
                "cnn_final_max_abs_error_K": final["max_abs_error_K"],
                "cnn_final_mean_signed_error_K": final["mean_signed_error_K"],
                "cnn_hotspot_temp_error_K": final["hotspot_temp_error_K"],
                "cnn_hotspot_location_error_cells": final["hotspot_location_error_cells"],
            })


def save_batch_predictions(out_dir: Path, sample_uids: list[Any], case_ids: list[Any], pred_temperature: torch.Tensor, pred_residual: torch.Tensor) -> None:
    pred_temperature_cpu = pred_temperature.detach().float().cpu().numpy().astype(np.float32, copy=False)
    pred_residual_cpu = pred_residual.detach().float().cpu().numpy().astype(np.float32, copy=False)
    for index, sample_uid in enumerate(sample_uids):
        case_id = str(case_ids[index])
        pred_dir = out_dir / "predictions" / case_id
        residual_dir = out_dir / "predicted_residuals" / case_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        residual_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / f"{sample_uid}_tpred.npy", pred_temperature_cpu[index])
        np.save(residual_dir / f"{sample_uid}_residual_v1_pred.npy", pred_residual_cpu[index])


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


def mean_or_none(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


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
