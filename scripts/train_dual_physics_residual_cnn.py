#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
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
    compute_dual_physics_normalization_stats,
    normalize_residual_v1,
    unnormalize_residual_v1,
)
from chiptherm.ml.models import MiniUNet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train dual-physics ChipTherm residual mini-UNet.")
    parser.add_argument("--physics-v1-train-index", required=True, type=Path)
    parser.add_argument("--physics-v2-train-index", required=True, type=Path)
    parser.add_argument("--physics-v1-val-index", required=True, type=Path)
    parser.add_argument("--physics-v2-val-index", required=True, type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/residual_cnn_dual_physics_package_plus_power_base32", type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1.0e-3, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-train-batches", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-batches", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-normalization-batches", default=None, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DualPhysicsDataset(args.physics_v1_train_index, args.physics_v2_train_index)
    val_dataset = DualPhysicsDataset(args.physics_v1_val_index, args.physics_v2_val_index)
    sample = train_dataset[0]
    dataset_input_channels = int(sample["x"].shape[0])
    model_input_channels = dataset_input_channels + 2

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
    stats = compute_dual_physics_normalization_stats(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_normalization_batches,
    )
    if model_input_channels != stats.input_channels + 2:
        raise SystemExit(f"checkpoint input-channel sanity failed: model={model_input_channels}, stats={stats.input_channels}+2")

    config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics_v1_train_index": str(args.physics_v1_train_index.resolve()),
        "physics_v2_train_index": str(args.physics_v2_train_index.resolve()),
        "physics_v1_val_index": str(args.physics_v1_val_index.resolve()),
        "physics_v2_val_index": str(args.physics_v2_val_index.resolve()),
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "device": str(device),
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_normalization_batches": args.max_normalization_batches,
        "model": {
            "name": "MiniUNet",
            "input_channels": model_input_channels,
            "dataset_input_channels": dataset_input_channels,
            "output_channels": 1,
            "base_channels": args.base_channels,
            "depth": args.depth,
        },
        "loss": "SmoothL1Loss on normalized residual_v1 = HotSpot - physics_v1",
        "target": "residual_v1",
        "final_prediction": "physics_v1 + predicted_residual_v1",
        "physics_v2_role": "auxiliary input feature only; not residual anchor",
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "normalization.json", stats.to_dict())

    model = MiniUNet(input_channels=model_input_channels, output_channels=1, base_channels=args.base_channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    log_path = out_dir / "train_log.csv"
    init_train_log(log_path)
    best_val_mae = float("inf")
    best_metrics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            stats,
            device,
            max_batches=args.max_train_batches,
        )
        val_metrics, val_by_case = evaluate_model(
            model,
            val_loader,
            criterion,
            stats,
            device,
            max_batches=args.max_val_batches,
        )
        epoch_runtime_s = time.perf_counter() - epoch_start
        val_final_mae = float(val_metrics["final_temperature"]["mae_K"])
        is_best = val_final_mae < best_val_mae
        if is_best:
            best_val_mae = val_final_mae
            best_metrics = {"epoch": epoch, "metrics": val_metrics, "metrics_by_case": val_by_case}
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, config, stats, val_metrics, best=True)
        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, config, stats, val_metrics, best=is_best)
        append_train_log(log_path, epoch, train_loss, val_metrics, epoch_runtime_s, is_best, optimizer.param_groups[0]["lr"])
        write_json(out_dir / "val_metrics.json", best_metrics or {"epoch": epoch, "metrics": val_metrics, "metrics_by_case": val_by_case})
        write_case_metrics(out_dir / "val_metrics_by_case.csv", (best_metrics or {"metrics_by_case": val_by_case})["metrics_by_case"])
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} "
            f"val_mae={val_final_mae:.3f}K val_rmse={val_metrics['final_temperature']['rmse_K']:.3f}K "
            f"{'best' if is_best else ''}"
        )

    print("Dual-physics residual CNN training complete")
    print(f"Best validation final-temperature MAE: {best_val_mae:.3f} K")
    print(f"Output: {out_dir}")
    return 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    stats: DualPhysicsNormalizationStats,
    device: torch.device,
    *,
    max_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        physics_v1 = batch["physics_v1"].to(device, non_blocking=True)
        physics_v2 = batch["physics_v2"].to(device, non_blocking=True)
        residual_v1 = batch["residual_v1"].to(device, non_blocking=True)
        model_input = build_dual_physics_model_input(x, physics_v1, physics_v2, stats)
        target = normalize_residual_v1(residual_v1, stats).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        pred = model(model_input)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    stats: DualPhysicsNormalizationStats,
    device: torch.device,
    *,
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    model.eval()
    residual_acc = MetricAccumulator()
    final_acc = MetricAccumulator()
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(lambda: {"residual": MetricAccumulator(), "final_temperature": MetricAccumulator()})
    total_loss = 0.0
    total_samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        physics_v1 = batch["physics_v1"].to(device, non_blocking=True)
        physics_v2 = batch["physics_v2"].to(device, non_blocking=True)
        residual_v1 = batch["residual_v1"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        model_input = build_dual_physics_model_input(x, physics_v1, physics_v2, stats)
        target_norm = normalize_residual_v1(residual_v1, stats).unsqueeze(1)
        pred_norm = model(model_input)
        loss = criterion(pred_norm, target_norm)
        pred_residual = unnormalize_residual_v1(pred_norm.squeeze(1), stats)
        pred_temperature = physics_v1 + pred_residual
        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        residual_acc.update(pred_residual, residual_v1)
        final_acc.update(pred_temperature, temperature)
        case_ids = metadata_values(batch["metadata"], "case_id", batch_size)
        for index, case_id in enumerate(case_ids):
            by_case[str(case_id)]["residual"].update(pred_residual[index : index + 1], residual_v1[index : index + 1])
            by_case[str(case_id)]["final_temperature"].update(pred_temperature[index : index + 1], temperature[index : index + 1])
    metrics = {
        "normalized_residual_loss": total_loss / max(total_samples, 1),
        "residual": residual_acc.compute(),
        "final_temperature": final_acc.compute(),
    }
    case_metrics = {
        case_id: {"residual": accs["residual"].compute(), "final_temperature": accs["final_temperature"].compute()}
        for case_id, accs in sorted(by_case.items())
    }
    return metrics, case_metrics


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


def save_checkpoint(path: Path, model: MiniUNet, optimizer: torch.optim.Optimizer, epoch: int, config: dict[str, Any], stats: DualPhysicsNormalizationStats, metrics: dict[str, Any], *, best: bool) -> None:
    torch.save(
        {
            "schema_version": 1,
            "epoch": epoch,
            "best": best,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config(),
            "training_config": config,
            "normalization": stats.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def init_train_log(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerow(["epoch", "lr", "train_loss", "val_loss", "val_residual_mae_K", "val_final_mae_K", "val_final_rmse_K", "val_hotspot_temp_error_K", "val_hotspot_location_error_cells", "epoch_runtime_s", "is_best"])


def append_train_log(path: Path, epoch: int, train_loss: float, val_metrics: dict[str, Any], epoch_runtime_s: float, is_best: bool, current_lr: float) -> None:
    with path.open("a", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerow([
            epoch,
            current_lr,
            train_loss,
            val_metrics["normalized_residual_loss"],
            val_metrics["residual"]["mae_K"],
            val_metrics["final_temperature"]["mae_K"],
            val_metrics["final_temperature"]["rmse_K"],
            val_metrics["final_temperature"]["hotspot_temp_error_K"],
            val_metrics["final_temperature"]["hotspot_location_error_cells"],
            epoch_runtime_s,
            int(is_best),
        ])


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    columns = ["case_id", "residual_mae_K", "residual_rmse_K", "final_temperature_mae_K", "final_temperature_rmse_K", "final_temperature_max_abs_error_K", "final_temperature_mean_signed_error_K", "hotspot_temp_error_K", "hotspot_location_error_cells"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            final = metrics["final_temperature"]
            residual = metrics["residual"]
            writer.writerow({
                "case_id": case_id,
                "residual_mae_K": residual["mae_K"],
                "residual_rmse_K": residual["rmse_K"],
                "final_temperature_mae_K": final["mae_K"],
                "final_temperature_rmse_K": final["rmse_K"],
                "final_temperature_max_abs_error_K": final["max_abs_error_K"],
                "final_temperature_mean_signed_error_K": final["mean_signed_error_K"],
                "hotspot_temp_error_K": final["hotspot_temp_error_K"],
                "hotspot_location_error_cells": final["hotspot_location_error_cells"],
            })


def make_loader(dataset: DualPhysicsDataset, batch_size: int, *, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader[dict[str, Any]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == "cuda")


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
