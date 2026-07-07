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

from chiptherm.ml.dataset import ChipThermDataset
from chiptherm.ml.models import MiniUNet
from chiptherm.ml.normalization import (
    NormalizationStats,
    build_model_input,
    compute_normalization_stats,
    normalize_residual,
    save_normalization_stats,
    unnormalize_residual,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ChipTherm residual mini-UNet.")
    parser.add_argument("--train-index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/train_index.csv", type=Path)
    parser.add_argument("--val-index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/val_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/residual_cnn_v1", type=Path)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1.0e-3, type=float)
    parser.add_argument("--base-channels", default=16, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--scheduler", default="none", choices=["none", "plateau", "cosine"])
    parser.add_argument("--temp-loss-weight", default=0.0, type=float)
    parser.add_argument("--hotspot-loss-weight", default=0.0, type=float)
    parser.add_argument("--hotspot-top-frac", default=0.05, type=float)
    args = parser.parse_args()
    if args.temp_loss_weight < 0.0:
        raise SystemExit("--temp-loss-weight must be non-negative")
    if args.hotspot_loss_weight < 0.0:
        raise SystemExit("--hotspot-loss-weight must be non-negative")
    if not 0.0 < args.hotspot_top_frac <= 1.0:
        raise SystemExit("--hotspot-top-frac must be in the interval (0, 1]")

    set_seed(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ChipThermDataset(args.train_index, target="residual", return_metadata=True)
    val_dataset = ChipThermDataset(args.val_index, target="residual", return_metadata=True)
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
        "temp_loss_weight": args.temp_loss_weight,
        "temp_loss_scaling": "temperature L1 loss in Kelvin divided by train residual_std before weighting",
        "hotspot_loss_weight": args.hotspot_loss_weight,
        "hotspot_top_frac": args.hotspot_top_frac,
        "hotspot_loss_scaling": "hotspot L1 loss in Kelvin over top ground-truth HotSpot cells divided by train residual_std before weighting",
        "model": {
            "name": "MiniUNet",
            "input_channels": 9,
            "output_channels": 1,
            "base_channels": args.base_channels,
            "depth": args.depth,
        },
        "loss": (
            "SmoothL1Loss on normalized residual plus optional temp_loss_weight * L1(T_pred, HotSpot) / residual_std "
            "plus optional hotspot_loss_weight * L1(T_pred, HotSpot on top HotSpot cells) / residual_std"
        ),
        "target": "residual = HotSpot - PhysicsBaseline",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats = compute_normalization_stats(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    save_normalization_stats(stats, out_dir / "normalization.json")

    model = MiniUNet(input_channels=9, output_channels=1, base_channels=args.base_channels, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)
    criterion = nn.SmoothL1Loss()
    temp_criterion = nn.L1Loss()

    log_path = out_dir / "train_log.csv"
    init_train_log(log_path)
    best_val_mae = float("inf")
    best_metrics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            temp_criterion,
            stats,
            device,
            temp_loss_weight=args.temp_loss_weight,
            hotspot_loss_weight=args.hotspot_loss_weight,
            hotspot_top_frac=args.hotspot_top_frac,
        )
        val_metrics, val_by_case = evaluate_model(model, val_loader, criterion, stats, device)
        epoch_runtime_s = time.perf_counter() - epoch_start
        val_final_mae = float(val_metrics["final_temperature"]["mae_K"])
        step_scheduler(args.scheduler, scheduler, val_final_mae)
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_final_mae < best_val_mae
        if is_best:
            best_val_mae = val_final_mae
            best_metrics = {
                "epoch": epoch,
                "metrics": val_metrics,
                "metrics_by_case": val_by_case,
            }
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, config, stats, val_metrics, best=True)

        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, config, stats, val_metrics, best=is_best)
        append_train_log(log_path, epoch, train_losses, val_metrics, epoch_runtime_s, is_best, current_lr)
        write_metrics(out_dir / "val_metrics.json", best_metrics or {"epoch": epoch, "metrics": val_metrics, "metrics_by_case": val_by_case})
        write_case_metrics(out_dir / "val_metrics_by_case.csv", (best_metrics or {"metrics_by_case": val_by_case})["metrics_by_case"])

        print(
            f"epoch {epoch:03d} train_loss={train_losses['total_loss']:.6f} lr={current_lr:.3e} "
            f"val_mae={val_final_mae:.3f}K val_rmse={val_metrics['final_temperature']['rmse_K']:.3f}K "
            f"{'best' if is_best else ''}"
        )

    print("Residual CNN training complete")
    print(f"Best validation final-temperature MAE: {best_val_mae:.3f} K")
    print(f"Output: {out_dir}")
    return 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    temp_criterion: nn.Module,
    stats: NormalizationStats,
    device: torch.device,
    *,
    temp_loss_weight: float,
    hotspot_loss_weight: float,
    hotspot_top_frac: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    residual_loss_total = 0.0
    temp_loss_scaled_total = 0.0
    temp_loss_K_total = 0.0
    hotspot_loss_scaled_total = 0.0
    hotspot_loss_K_total = 0.0
    total_samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        model_input = build_model_input(x, physics, stats)
        target = normalize_residual(residual, stats).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(model_input)
        residual_loss = criterion(pred, target)
        if temp_loss_weight > 0.0 or hotspot_loss_weight > 0.0:
            pred_residual_K = unnormalize_residual(pred.squeeze(1), stats)
            pred_temperature = physics + pred_residual_K
        if temp_loss_weight > 0.0:
            temp_loss_K = temp_criterion(pred_temperature, temperature)
            temp_loss_scaled = temp_loss_K / max(float(stats.residual_std), 1.0e-8)
        else:
            temp_loss_K = pred.new_tensor(0.0)
            temp_loss_scaled = pred.new_tensor(0.0)
        if hotspot_loss_weight > 0.0:
            hotspot_loss_K = hotspot_l1_loss(pred_temperature, temperature, hotspot_top_frac)
            hotspot_loss_scaled = hotspot_loss_K / max(float(stats.residual_std), 1.0e-8)
        else:
            hotspot_loss_K = pred.new_tensor(0.0)
            hotspot_loss_scaled = pred.new_tensor(0.0)
        loss = (
            residual_loss
            + float(temp_loss_weight) * temp_loss_scaled
            + float(hotspot_loss_weight) * hotspot_loss_scaled
        )
        loss.backward()
        optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        residual_loss_total += float(residual_loss.item()) * batch_size
        temp_loss_scaled_total += float(temp_loss_scaled.item()) * batch_size
        temp_loss_K_total += float(temp_loss_K.item()) * batch_size
        hotspot_loss_scaled_total += float(hotspot_loss_scaled.item()) * batch_size
        hotspot_loss_K_total += float(hotspot_loss_K.item()) * batch_size
        total_samples += batch_size
    denominator = max(total_samples, 1)
    return {
        "total_loss": total_loss / denominator,
        "residual_loss": residual_loss_total / denominator,
        "temp_loss_scaled": temp_loss_scaled_total / denominator,
        "temp_loss_K": temp_loss_K_total / denominator,
        "hotspot_loss_scaled": hotspot_loss_scaled_total / denominator,
        "hotspot_loss_K": hotspot_loss_K_total / denominator,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    stats: NormalizationStats,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    model.eval()
    residual_acc = MetricAccumulator()
    final_acc = MetricAccumulator()
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(lambda: {"residual": MetricAccumulator(), "final_temperature": MetricAccumulator()})
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        model_input = build_model_input(x, physics, stats)
        target_norm = normalize_residual(residual, stats).unsqueeze(1)
        pred_norm = model(model_input)
        loss = criterion(pred_norm, target_norm)
        pred_residual = unnormalize_residual(pred_norm.squeeze(1), stats)
        pred_temperature = physics + pred_residual
        case_ids = metadata_values(batch["metadata"], "case_id", int(x.shape[0]))

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        residual_acc.update(pred_residual, residual)
        final_acc.update(pred_temperature, temperature)
        for index, case_id in enumerate(case_ids):
            by_case[str(case_id)]["residual"].update(pred_residual[index : index + 1], residual[index : index + 1])
            by_case[str(case_id)]["final_temperature"].update(pred_temperature[index : index + 1], temperature[index : index + 1])

    metrics = {
        "normalized_residual_loss": total_loss / max(total_samples, 1),
        "residual": residual_acc.compute(),
        "final_temperature": final_acc.compute(),
    }
    case_metrics = {
        case_id: {
            "residual": accs["residual"].compute(),
            "final_temperature": accs["final_temperature"].compute(),
        }
        for case_id, accs in sorted(by_case.items())
    }
    return metrics, case_metrics


def hotspot_l1_loss(pred_temperature: torch.Tensor, temperature: torch.Tensor, top_frac: float) -> torch.Tensor:
    if pred_temperature.shape != temperature.shape:
        raise ValueError(f"pred_temperature shape {pred_temperature.shape} does not match temperature shape {temperature.shape}")
    batch_size = int(temperature.shape[0])
    flat_temperature = temperature.reshape(batch_size, -1)
    flat_error = torch.abs(pred_temperature - temperature).reshape(batch_size, -1)
    num_cells = int(flat_temperature.shape[1])
    k = max(1, int(np.ceil(num_cells * float(top_frac))))
    top_indices = torch.topk(flat_temperature, k=k, dim=1, largest=True, sorted=False).indices
    hotspot_error = torch.gather(flat_error, dim=1, index=top_indices)
    return hotspot_error.mean()


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


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def make_loader(
    dataset: ChipThermDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def save_checkpoint(
    path: Path,
    model: MiniUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    stats: NormalizationStats,
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
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_residual_loss",
                "train_temp_loss_scaled",
                "train_temp_loss_K",
                "train_hotspot_loss_scaled",
                "train_hotspot_loss_K",
                "val_loss",
                "val_residual_mae_K",
                "val_residual_rmse_K",
                "val_final_mae_K",
                "val_final_rmse_K",
                "val_hotspot_temp_error_K",
                "val_hotspot_location_error_cells",
                "epoch_runtime_s",
                "is_best",
            ]
        )


def append_train_log(
    path: Path,
    epoch: int,
    train_losses: dict[str, float],
    val_metrics: dict[str, Any],
    epoch_runtime_s: float,
    is_best: bool,
    current_lr: float,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                epoch,
                current_lr,
                train_losses["total_loss"],
                train_losses["residual_loss"],
                train_losses["temp_loss_scaled"],
                train_losses["temp_loss_K"],
                train_losses["hotspot_loss_scaled"],
                train_losses["hotspot_loss_K"],
                val_metrics["normalized_residual_loss"],
                val_metrics["residual"]["mae_K"],
                val_metrics["residual"]["rmse_K"],
                val_metrics["final_temperature"]["mae_K"],
                val_metrics["final_temperature"]["rmse_K"],
                val_metrics["final_temperature"]["hotspot_temp_error_K"],
                val_metrics["final_temperature"]["hotspot_location_error_cells"],
                epoch_runtime_s,
                int(is_best),
            ]
        )


def make_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1.0e-4,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=1.0e-6,
        )
    raise ValueError(f"unsupported scheduler: {scheduler_name}")


def step_scheduler(
    scheduler_name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    val_final_mae: float,
) -> None:
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(val_final_mae)
    else:
        scheduler.step()


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    columns = [
        "case_id",
        "residual_mae_K",
        "residual_rmse_K",
        "final_temperature_mae_K",
        "final_temperature_rmse_K",
        "final_temperature_max_abs_error_K",
        "final_temperature_mean_signed_error_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            final = metrics["final_temperature"]
            residual = metrics["residual"]
            writer.writerow(
                {
                    "case_id": case_id,
                    "residual_mae_K": residual["mae_K"],
                    "residual_rmse_K": residual["rmse_K"],
                    "final_temperature_mae_K": final["mae_K"],
                    "final_temperature_rmse_K": final["rmse_K"],
                    "final_temperature_max_abs_error_K": final["max_abs_error_K"],
                    "final_temperature_mean_signed_error_K": final["mean_signed_error_K"],
                    "hotspot_temp_error_K": final["hotspot_temp_error_K"],
                    "hotspot_location_error_cells": final["hotspot_location_error_cells"],
                }
            )


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
