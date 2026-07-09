#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset
from chiptherm.ml.models import MiniUNet


EPSILON = 1.0e-8


@dataclass(frozen=True)
class DirectNormalizationStats:
    schema_version: int
    input_channels: int
    power_density_mean: float
    power_density_std: float
    context_channel_indices: tuple[int, ...]
    context_channel_means: tuple[float, ...]
    context_channel_stds: tuple[float, ...]
    temperature_mean: float
    temperature_std: float
    num_samples: int
    num_grid_cells: int
    notes: str = "Computed from train split only. Masks and normalized coordinate channels are not normalized."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_channel_indices"] = list(self.context_channel_indices)
        data["context_channel_means"] = list(self.context_channel_means)
        data["context_channel_stds"] = list(self.context_channel_stds)
        return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Train direct ChipTherm temperature MiniUNet baseline.")
    parser.add_argument("--train-index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power/train_index.csv", type=Path)
    parser.add_argument("--val-index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power/val_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/direct_cnn_package_plus_power_base32", type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1.0e-3, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--scheduler", default="none", choices=["none", "plateau", "cosine"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ChipThermDataset(args.train_index, target="temperature", return_metadata=True)
    val_dataset = ChipThermDataset(args.val_index, target="temperature", return_metadata=True)
    input_channels = int(train_dataset[0]["x"].shape[0])
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)

    config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_index": str(args.train_index.resolve()),
        "val_index": str(args.val_index.resolve()),
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "device": str(device),
        "num_workers": args.num_workers,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "model": {
            "name": "MiniUNet",
            "input_channels": input_channels,
            "output_channels": 1,
            "base_channels": args.base_channels,
            "depth": args.depth,
        },
        "loss": "SmoothL1Loss on normalized HotSpot temperature",
        "target": "temperature = HotSpot Layer 0 map",
        "runtime_note": "Direct CNN does not compute or use T_phys.",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats = compute_direct_normalization_stats(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    (out_dir / "normalization.json").write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model = MiniUNet(input_channels=input_channels, output_channels=1, base_channels=args.base_channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)
    criterion = nn.SmoothL1Loss()

    log_path = out_dir / "train_log.csv"
    init_train_log(log_path)
    best_val_mae = float("inf")
    best_payload: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, stats, device)
        val_metrics, val_by_case = evaluate_model(model, val_loader, criterion, stats, device)
        epoch_runtime_s = time.perf_counter() - epoch_start
        val_mae = float(val_metrics["temperature"]["mae_K"])
        step_scheduler(args.scheduler, scheduler, val_mae)
        current_lr = float(optimizer.param_groups[0]["lr"])
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_payload = {"epoch": epoch, "metrics": val_metrics, "metrics_by_case": val_by_case}
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, config, stats, val_metrics, best=True)

        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, config, stats, val_metrics, best=is_best)
        append_train_log(log_path, epoch, current_lr, train_loss, val_metrics, epoch_runtime_s, is_best)
        write_json(out_dir / "val_metrics.json", best_payload or {"epoch": epoch, "metrics": val_metrics, "metrics_by_case": val_by_case})
        write_case_metrics(out_dir / "val_metrics_by_case.csv", (best_payload or {"metrics_by_case": val_by_case})["metrics_by_case"])
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} lr={current_lr:.3e} "
            f"val_mae={val_mae:.3f}K val_rmse={val_metrics['temperature']['rmse_K']:.3f}K "
            f"{'best' if is_best else ''}"
        )

    print("Direct CNN training complete")
    print(f"Best validation MAE: {best_val_mae:.3f} K")
    print(f"Output: {out_dir}")
    return 0


def compute_direct_normalization_stats(dataset: Dataset[Any], *, batch_size: int, num_workers: int) -> DirectNormalizationStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    power_acc = RunningMoments()
    temp_acc = RunningMoments()
    context_accs: list[RunningMoments] | None = None
    input_channels: int | None = None
    num_samples = 0
    for batch in loader:
        x = batch["x"].float()
        temperature = batch["temperature"].float()
        if input_channels is None:
            input_channels = int(x.shape[1])
            context_accs = [RunningMoments() for _ in range(max(input_channels - 8, 0))]
        elif input_channels != int(x.shape[1]):
            raise ValueError(f"inconsistent input channel count: expected {input_channels}, got {x.shape[1]}")
        power_acc.update(x[:, 0])
        if context_accs:
            for offset, acc in enumerate(context_accs, start=8):
                acc.update(x[:, offset])
        temp_acc.update(temperature)
        num_samples += int(x.shape[0])
    context_accs = context_accs or []
    return DirectNormalizationStats(
        schema_version=1,
        input_channels=int(input_channels or 8),
        power_density_mean=power_acc.mean,
        power_density_std=power_acc.std,
        context_channel_indices=tuple(range(8, int(input_channels or 8))),
        context_channel_means=tuple(acc.mean for acc in context_accs),
        context_channel_stds=tuple(acc.std for acc in context_accs),
        temperature_mean=temp_acc.mean,
        temperature_std=temp_acc.std,
        num_samples=num_samples,
        num_grid_cells=int(temp_acc.count),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    stats: DirectNormalizationStats,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        model_input = normalize_x(x, stats)
        target = normalize_temperature(temperature, stats).unsqueeze(1)
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
    stats: DirectNormalizationStats,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    model.eval()
    acc = MetricAccumulator()
    by_case: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        model_input = normalize_x(x, stats)
        target = normalize_temperature(temperature, stats).unsqueeze(1)
        pred_norm = model(model_input)
        loss = criterion(pred_norm, target)
        pred_temperature = unnormalize_temperature(pred_norm.squeeze(1), stats)
        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        acc.update(pred_temperature, temperature)
        case_ids = metadata_values(batch["metadata"], "case_id", batch_size)
        for index, case_id in enumerate(case_ids):
            by_case[str(case_id)].update(pred_temperature[index : index + 1], temperature[index : index + 1])
    return (
        {
            "normalized_temperature_loss": total_loss / max(total_samples, 1),
            "temperature": acc.compute(),
        },
        {case_id: case_acc.compute() for case_id, case_acc in sorted(by_case.items())},
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
        return {
            "num_samples": float(self.num_samples),
            "mae_K": self.sum_abs / max(self.num_cells, 1),
            "rmse_K": (self.sum_sq / max(self.num_cells, 1)) ** 0.5,
            "max_abs_error_K": self.max_abs,
            "mean_signed_error_K": self.sum_signed / max(self.num_cells, 1),
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


def normalize_temperature(temperature: torch.Tensor, stats: DirectNormalizationStats) -> torch.Tensor:
    return normalize_tensor(temperature.float(), stats.temperature_mean, stats.temperature_std)


def unnormalize_temperature(temperature_norm: torch.Tensor, stats: DirectNormalizationStats) -> torch.Tensor:
    return temperature_norm.float() * float(stats.temperature_std) + float(stats.temperature_mean)


def normalize_tensor(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (value - float(mean)) / max(float(std), EPSILON)


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        data = tensor.detach().double()
        self.count += int(data.numel())
        self.total += float(data.sum().item())
        self.total_sq += float((data * data).sum().item())

    @property
    def mean(self) -> float:
        return float(self.total / self.count) if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 1.0
        variance = max(self.total_sq / self.count - self.mean * self.mean, EPSILON)
        return float(variance**0.5)


def make_loader(dataset: ChipThermDataset, batch_size: int, *, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader[dict[str, Any]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == "cuda")


def save_checkpoint(
    path: Path,
    model: MiniUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    stats: DirectNormalizationStats,
    metrics: dict[str, Any],
    *,
    best: bool,
) -> None:
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
        writer = csv.writer(fp)
        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "val_loss",
            "val_mae_K",
            "val_rmse_K",
            "val_max_abs_error_K",
            "val_mean_signed_error_K",
            "val_hotspot_temp_error_K",
            "val_hotspot_location_error_cells",
            "epoch_runtime_s",
            "is_best",
        ])


def append_train_log(path: Path, epoch: int, lr: float, train_loss: float, val_metrics: dict[str, Any], epoch_runtime_s: float, is_best: bool) -> None:
    metrics = val_metrics["temperature"]
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            epoch,
            lr,
            train_loss,
            val_metrics["normalized_temperature_loss"],
            metrics["mae_K"],
            metrics["rmse_K"],
            metrics["max_abs_error_K"],
            metrics["mean_signed_error_K"],
            metrics["hotspot_temp_error_K"],
            metrics["hotspot_location_error_cells"],
            epoch_runtime_s,
            int(is_best),
        ])


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, threshold=1.0e-4)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=1.0e-6)
    raise ValueError(f"unsupported scheduler: {scheduler_name}")


def step_scheduler(
    scheduler_name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    val_mae: float,
) -> None:
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(val_mae)
    else:
        scheduler.step()


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


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
