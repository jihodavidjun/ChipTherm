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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiptherm.ml.source_response_dataset import (
    SourceResponseDataset,
    compute_source_response_normalization,
    normalize_source_input,
    normalize_source_target_unit,
    save_source_response_normalization,
    source_response_collate,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import (
    build_source_response_model,
    count_parameters,
    predict_source_rise,
)
from scripts.run_superposition_diagnostic import chiplet_metrics, field_metrics, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ChipTherm source-response operator.")
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1.0e-3, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--power-floor-W", default=1.0e-6, type=float)
    parser.add_argument("--low-power-warning-W", default=1.0, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SourceResponseDataset(args.train_index, power_floor_W=args.power_floor_W)
    val_dataset = SourceResponseDataset(args.val_index, power_floor_W=args.power_floor_W)
    stats = compute_source_response_normalization(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    save_source_response_normalization(stats, out_dir / "source_response_normalization.json")
    power_diagnostics = source_power_diagnostics(train_dataset, val_dataset, args.low_power_warning_W)
    (out_dir / "source_power_diagnostics.json").write_text(json.dumps(power_diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model_config = {
        "architecture": "source_response_operator_v1",
        "input_channels": len(train_dataset.channel_names),
        "channel_names": list(train_dataset.channel_names),
        "base_channels": args.base_channels,
        "depth": args.depth,
        "output_mode": "linear_normalized",
        "target_normalization_mode": stats.target_normalization_mode,
        "target_unit_mean_K_per_W": stats.target_unit_mean_K_per_W,
        "target_unit_std_K_per_W": stats.target_unit_std_K_per_W,
        "power_floor_W": args.power_floor_W,
    }
    model = build_source_response_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_index": str(args.train_index),
        "val_index": str(args.val_index),
        "model_config": model_config,
        "optimizer": "AdamW",
        "loss": "SmoothL1 on train-standardized unit response K/W; best checkpoint selected by validation package full-grid MAE",
        "parameter_count": count_parameters(model),
        "args": json_safe(vars(args)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    init_log(out_dir / "train_log.csv")

    best_val_package_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, stats, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, stats, device)
        package_mae = val_metrics["package_reconstruction"]["mae_K"]
        if package_mae is None:
            raise SystemExit("validation package-level MAE unavailable; ensure val source index has all source chiplets per original sample")
        is_best = package_mae < best_val_package_mae
        if is_best:
            best_val_package_mae = float(package_mae)
        payload = checkpoint_payload(model, model_config, stats, epoch, val_metrics, args)
        torch.save(payload, checkpoints_dir / "last.pt")
        if is_best:
            torch.save(payload, checkpoints_dir / "best.pt")
            best_payload = payload
        append_log(out_dir / "train_log.csv", epoch, train_loss, val_metrics, time.perf_counter() - start, is_best)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} "
            f"val_source_K={val_metrics['source_physical']['mae_K']:.4f} "
            f"val_source_K_per_W={val_metrics['source_unit']['mae_K_per_W']:.6f} "
            f"pred_K_per_W_mean={val_metrics['prediction_stats']['pred_unit_K_per_W']['mean']:.6f} "
            f"pred_K_per_W_std={val_metrics['prediction_stats']['pred_unit_K_per_W']['std']:.6f} "
            f"neg_frac={val_metrics['prediction_stats']['negative_unit_response_fraction']:.4f} "
            f"val_package_mae={package_mae:.4f} best={best_val_package_mae:.4f}"
        )
    if best_payload is not None:
        (out_dir / "val_metrics.json").write_text(json.dumps(best_payload["val_metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best validation package reconstructed MAE: {best_val_package_mae:.4f} K")
    return 0


def train_one_epoch(model: nn.Module, loader: DataLoader, stats: Any, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        x = normalize_source_input(batch["x"].to(device), stats)
        target = normalize_source_target_unit(batch["target_unit"].to(device), stats)
        optimizer.zero_grad(set_to_none=True)
        pred_normalized = model(x)
        loss = criterion(pred_normalized, target)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(x.shape[0])
        count += int(x.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, stats: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    source_unit_errors: list[np.ndarray] = []
    source_physical_errors: list[np.ndarray] = []
    pred_unit_values: list[np.ndarray] = []
    pred_rise_values: list[np.ndarray] = []
    negative_count = 0
    prediction_count = 0
    groups: dict[str, dict[str, Any]] = {}
    for batch in loader:
        x = normalize_source_input(batch["x"].to(device), stats)
        source_power = batch["source_power_W"].to(device)
        pred_normalized = model(x)
        pred_unit = unnormalize_source_prediction(pred_normalized, stats)
        pred_rise = predict_source_rise(pred_unit, source_power)
        target_unit = batch["target_unit"].to(device)
        target_rise = batch["target_rise"].to(device)
        source_unit_errors.append((pred_unit - target_unit).detach().cpu().numpy())
        source_physical_errors.append((pred_rise - target_rise).detach().cpu().numpy())
        pred_unit_values.append(pred_unit.detach().cpu().numpy().reshape(-1))
        pred_rise_values.append(pred_rise.detach().cpu().numpy().reshape(-1))
        negative_count += int((pred_unit < 0.0).sum().item())
        prediction_count += int(pred_unit.numel())
        pred_np = pred_rise.detach().cpu().numpy()
        target_np = target_rise.detach().cpu().numpy()
        full_np = batch["full_temperature"].detach().cpu().numpy()
        ambient_np = batch["ambient_K"].detach().cpu().numpy()
        for i, meta in enumerate(batch["metadata"]):
            uid = str(meta["original_sample_uid"])
            group = groups.setdefault(
                uid,
                {
                    "case_id": meta["case_id"],
                    "ambient_K": float(ambient_np[i]),
                    "pred_sum": np.zeros_like(pred_np[i], dtype=np.float64),
                    "target_sum": np.zeros_like(target_np[i], dtype=np.float64),
                    "full_temperature": full_np[i].astype(np.float64),
                    "layout_path": meta["layout_path"],
                    "num_chiplets": int(float(meta["num_chiplets"])),
                    "num_sources": 0,
                },
            )
            group["pred_sum"] += pred_np[i]
            group["target_sum"] += target_np[i]
            group["num_sources"] += 1
    source_unit = aggregate_error(np.concatenate([e.reshape(-1) for e in source_unit_errors]), suffix="K_per_W")
    source_physical = aggregate_error(np.concatenate([e.reshape(-1) for e in source_physical_errors]), suffix="K")
    package = package_metrics(groups)
    prediction_stats = {
        "pred_unit_K_per_W": describe_values(np.concatenate(pred_unit_values), warning=None),
        "pred_source_rise_K": describe_values(np.concatenate(pred_rise_values), warning=None),
        "negative_unit_response_fraction": float(negative_count / max(prediction_count, 1)),
    }
    return {
        "source_unit": source_unit,
        "source_physical": source_physical,
        "package_reconstruction": package["overall"],
        "package_by_case": package["by_case"],
        "prediction_stats": prediction_stats,
    }


def package_metrics(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for uid, group in groups.items():
        if int(group["num_sources"]) != int(group["num_chiplets"]):
            continue
        pred_temp = float(group["ambient_K"]) + group["pred_sum"]
        full = group["full_temperature"]
        base = field_metrics(pred_temp, full)
        layout = load_json((REPO_ROOT / group["layout_path"]).resolve() if not Path(group["layout_path"]).is_absolute() else Path(group["layout_path"]))
        chip = chiplet_metrics(pred_temp, full, layout, full.shape)
        record = {
            "original_sample_uid": uid,
            "case_id": group["case_id"],
            "mae_K": base["mae_K"],
            "rmse_K": base["rmse_K"],
            "max_abs_error_K": base["max_abs_error_K"],
            "chiplet_mean_temperature_mae_K": chip["chiplet_mean_temperature_mae_K"],
            "chiplet_peak_temperature_mae_K": chip["chiplet_peak_temperature_mae_K"],
            "inter_chiplet_delta_T_mae_K": chip["inter_chiplet_delta_T_mae_K"],
        }
        records.append(record)
        by_case[str(group["case_id"])].append(record)
    if not records:
        return {"overall": {"mae_K": None, "rmse_K": None, "num_packages": 0}, "by_case": {}}
    overall = {
        "mae_K": float(np.mean([r["mae_K"] for r in records])),
        "rmse_K": float(np.mean([r["rmse_K"] for r in records])),
        "max_abs_error_K": float(np.max([r["max_abs_error_K"] for r in records])),
        "chiplet_mean_temperature_mae_K": mean_optional(r["chiplet_mean_temperature_mae_K"] for r in records),
        "chiplet_peak_temperature_mae_K": mean_optional(r["chiplet_peak_temperature_mae_K"] for r in records),
        "inter_chiplet_delta_T_mae_K": mean_optional(r["inter_chiplet_delta_T_mae_K"] for r in records),
        "num_packages": len(records),
    }
    case_payload = {
        case: {
            "mae_K": float(np.mean([r["mae_K"] for r in items])),
            "rmse_K": float(np.mean([r["rmse_K"] for r in items])),
            "num_packages": len(items),
        }
        for case, items in sorted(by_case.items())
    }
    return {"overall": overall, "by_case": case_payload}


def aggregate_error(error: np.ndarray, *, suffix: str) -> dict[str, float]:
    return {
        f"mae_{suffix}": float(np.abs(error).mean()),
        f"rmse_{suffix}": float(np.sqrt(np.mean(error * error))),
        f"max_abs_error_{suffix}": float(np.abs(error).max()),
        f"mean_signed_error_{suffix}": float(error.mean()),
    }


def mean_optional(values: Any) -> float | None:
    numeric = [float(v) for v in values if v is not None]
    return float(np.mean(numeric)) if numeric else None


def checkpoint_payload(model: nn.Module, model_config: dict[str, Any], stats: Any, epoch: int, val_metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "normalization": stats.to_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }


def source_power_diagnostics(train_dataset: SourceResponseDataset, val_dataset: SourceResponseDataset, warning: float) -> dict[str, Any]:
    return {
        "train": describe_source_targets(train_dataset, warning),
        "val": describe_source_targets(val_dataset, warning),
        "low_power_warning_W": float(warning),
    }


def describe_source_targets(dataset: SourceResponseDataset, warning: float) -> dict[str, Any]:
    powers: list[float] = []
    rise_abs_values: list[np.ndarray] = []
    unit_abs_values: list[np.ndarray] = []
    for index, row in enumerate(dataset.rows):
        source_power = float(row["source_power_W"])
        powers.append(source_power)
        target_rise = np.load(dataset.resolve_row_path(row["target_rise_path"])).astype(np.float32, copy=False)
        rise_abs = np.abs(target_rise.astype(np.float64, copy=False))
        unit_abs = rise_abs / max(source_power, dataset.power_floor_W)
        rise_abs_values.append(rise_abs.reshape(-1))
        unit_abs_values.append(unit_abs.reshape(-1))
    return {
        "source_power_W": describe_values(np.asarray(powers, dtype=np.float64), warning=warning),
        "source_temperature_rise_abs_K": describe_values(np.concatenate(rise_abs_values), warning=None),
        "unit_response_abs_K_per_W": describe_values(np.concatenate(unit_abs_values), warning=None),
    }


def describe_values(values: np.ndarray, warning: float | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }
    if warning is not None:
        payload["num_below_warning"] = int((values < float(warning)).sum())
    return payload


def describe_power(values: np.ndarray, warning: float) -> dict[str, Any]:
    return {
        "min_W": float(np.min(values)),
        "p01_W": float(np.percentile(values, 1)),
        "p05_W": float(np.percentile(values, 5)),
        "p50_W": float(np.percentile(values, 50)),
        "p95_W": float(np.percentile(values, 95)),
        "max_W": float(np.max(values)),
        "num_below_warning": int((values < float(warning)).sum()),
    }


def make_loader(dataset: SourceResponseDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=source_response_collate)


def init_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(
            [
                "epoch",
                "train_loss_normalized",
                "val_source_mae_K_per_W",
                "val_source_mae_K",
                "val_package_mae_K",
                "val_package_rmse_K",
                "pred_K_per_W_mean",
                "pred_K_per_W_std",
                "pred_K_per_W_min",
                "pred_K_per_W_max",
                "pred_source_rise_K_mean",
                "pred_source_rise_K_std",
                "pred_source_rise_K_min",
                "pred_source_rise_K_max",
                "negative_prediction_fraction",
                "epoch_runtime_s",
                "is_best",
            ]
        )


def append_log(path: Path, epoch: int, train_loss: float, val_metrics: dict[str, Any], runtime: float, is_best: bool) -> None:
    pred_unit = val_metrics["prediction_stats"]["pred_unit_K_per_W"]
    pred_rise = val_metrics["prediction_stats"]["pred_source_rise_K"]
    with path.open("a", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow([
            epoch,
            train_loss,
            val_metrics["source_unit"]["mae_K_per_W"],
            val_metrics["source_physical"]["mae_K"],
            val_metrics["package_reconstruction"]["mae_K"],
            val_metrics["package_reconstruction"]["rmse_K"],
            pred_unit["mean"],
            pred_unit["std"],
            pred_unit["min"],
            pred_unit["max"],
            pred_rise["mean"],
            pred_rise["std"],
            pred_rise["min"],
            pred_rise["max"],
            val_metrics["prediction_stats"]["negative_unit_response_fraction"],
            runtime,
            int(is_best),
        ])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    return torch.device(requested)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
