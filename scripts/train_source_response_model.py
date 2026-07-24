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
    SourceResponsePackageDataset,
    SourceResponseDataset,
    compute_source_response_normalization,
    normalize_source_input,
    normalize_source_target_unit,
    save_source_response_normalization,
    source_response_package_collate,
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
    parser.add_argument("--data-root", default=None, type=Path, help="Explicit root for portable root-relative index paths.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--packages-per-batch", default=1, type=int)
    parser.add_argument("--lr", default=1.0e-3, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--power-floor-W", default=1.0e-6, type=float)
    parser.add_argument("--low-power-warning-W", default=1.0, type=float)
    parser.add_argument("--lambda-source", default=1.0, type=float)
    parser.add_argument("--lambda-package", default=0.0, type=float)
    parser.add_argument("--package-loss-warmup-epochs", default=0, type=int)
    parser.add_argument("--lambda-source-mean", default=0.0, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lineage-manifest", default=None, type=Path)
    parser.add_argument("--early-stopping-patience", default=0, type=int)
    parser.add_argument("--checkpoint-frequency", default=10, type=int)
    parser.add_argument("--scheduler", default="none", choices=["none", "plateau", "cosine"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SourceResponseDataset(
        args.train_index,
        power_floor_W=args.power_floor_W,
        data_root=args.data_root,
    )
    val_dataset = SourceResponseDataset(
        args.val_index,
        power_floor_W=args.power_floor_W,
        data_root=args.data_root,
    )
    train_package_dataset = SourceResponsePackageDataset(
        args.train_index,
        power_floor_W=args.power_floor_W,
        require_complete=True,
        data_root=args.data_root,
    )
    stats = compute_source_response_normalization(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    if not args.resume:
        save_source_response_normalization(stats, out_dir / "source_response_normalization.json")
    power_diagnostics = source_power_diagnostics(train_dataset, val_dataset, args.low_power_warning_W)
    if not args.resume:
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
        "packages_per_batch": args.packages_per_batch,
        "loss_config": {
            "source_loss": "SmoothL1 on train-standardized unit response K/W",
            "package_loss": "SmoothL1 on reconstructed full temperature in Kelvin",
            "lambda_source": args.lambda_source,
            "lambda_package": args.lambda_package,
            "package_loss_warmup_epochs": args.package_loss_warmup_epochs,
            "lambda_source_mean": args.lambda_source_mean,
        },
    }
    resume_signature = {
        key: json_safe(vars(args))[key]
        for key in (
            "train_index",
            "val_index",
            "data_root",
            "batch_size",
            "packages_per_batch",
            "lr",
            "base_channels",
            "depth",
            "power_floor_W",
            "lambda_source",
            "lambda_package",
            "package_loss_warmup_epochs",
            "lambda_source_mean",
            "seed",
            "scheduler",
            "lineage_manifest",
        )
    }
    model = build_source_response_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)
    criterion = nn.SmoothL1Loss()
    lineage = load_json(args.lineage_manifest) if args.lineage_manifest else None

    train_loader = make_package_loader(train_package_dataset, args.packages_per_batch, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_index": str(args.train_index),
        "val_index": str(args.val_index),
        "model_config": model_config,
        "optimizer": "AdamW",
        "loss": "lambda_source*SmoothL1(normalized source K/W) + active_lambda_package*SmoothL1(package K) + optional source mean loss; best checkpoint selected by validation package full-grid MAE",
        "parameter_count": count_parameters(model),
        "args": json_safe(vars(args)),
        "resume_signature": resume_signature,
    }
    if not args.resume:
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.resume:
        init_log(out_dir / "train_log.csv")

    best_val_package_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    start_epoch = 1
    epochs_without_improvement = 0
    last_checkpoint = checkpoints_dir / "last.pt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise SystemExit(f"--resume requested but checkpoint is missing: {last_checkpoint}")
        resumed = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if resumed.get("model_config") != model_config:
            raise SystemExit("resume checkpoint model configuration differs from requested training")
        if resumed.get("normalization") != stats.to_dict():
            raise SystemExit("resume checkpoint normalization differs from current train-only statistics")
        if lineage is not None and resumed.get("training_lineage") != lineage:
            raise SystemExit("resume checkpoint lineage differs from requested lineage")
        if resumed.get("resume_signature") != resume_signature:
            raise SystemExit("resume checkpoint training recipe differs from requested training")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if scheduler is not None and resumed.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        best_val_package_mae = float(resumed.get("best_val_package_mae", float("inf")))
        epochs_without_improvement = int(resumed.get("epochs_without_improvement", 0))
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        active_package_weight = package_loss_weight(args.lambda_package, args.package_loss_warmup_epochs, epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            stats,
            criterion,
            optimizer,
            device,
            lambda_source=args.lambda_source,
            lambda_package=active_package_weight,
            lambda_source_mean=args.lambda_source_mean,
        )
        val_metrics = evaluate_model(model, val_loader, stats, device, data_root=args.data_root)
        write_records_csv(out_dir / "package_bias_diagnostics.csv", val_metrics["package_bias_records"])
        (out_dir / "package_bias_summary.json").write_text(
            json.dumps(val_metrics["package_bias_summary"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        package_mae = val_metrics["package_reconstruction"]["mae_K"]
        if package_mae is None:
            raise SystemExit("validation package-level MAE unavailable; ensure val source index has all source chiplets per original sample")
        is_best = package_mae < best_val_package_mae
        if is_best:
            best_val_package_mae = float(package_mae)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        step_scheduler(args.scheduler, scheduler, float(package_mae))
        payload = checkpoint_payload(
            model,
            model_config,
            stats,
            epoch,
            val_metrics,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_package_mae=best_val_package_mae,
            epochs_without_improvement=epochs_without_improvement,
            training_lineage=lineage,
            resume_signature=resume_signature,
        )
        torch.save(payload, checkpoints_dir / "last.pt")
        if args.checkpoint_frequency > 0 and epoch % args.checkpoint_frequency == 0:
            torch.save(payload, checkpoints_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(payload, checkpoints_dir / "best.pt")
            best_payload = payload
        append_log(out_dir / "train_log.csv", epoch, train_metrics, val_metrics, time.perf_counter() - start, is_best, active_package_weight, optimizer)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['total_loss']:.6f} "
            f"source_loss={train_metrics['source_loss']:.6f} "
            f"package_loss={train_metrics['package_loss']:.6f} "
            f"pkg_w={active_package_weight:.4f} "
            f"val_source_K={val_metrics['source_physical']['mae_K']:.4f} "
            f"val_source_K_per_W={val_metrics['source_unit']['mae_K_per_W']:.6f} "
            f"val_pkg_bias={val_metrics['package_reconstruction'].get('mean_signed_error_K', float('nan')):.4f} "
            f"pred_K_per_W_mean={val_metrics['prediction_stats']['pred_unit_K_per_W']['mean']:.6f} "
            f"pred_K_per_W_std={val_metrics['prediction_stats']['pred_unit_K_per_W']['std']:.6f} "
            f"neg_frac={val_metrics['prediction_stats']['negative_unit_response_fraction']:.4f} "
            f"val_package_mae={package_mae:.4f} best={best_val_package_mae:.4f}"
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without validation improvement")
            break
    if best_payload is not None:
        (out_dir / "val_metrics.json").write_text(json.dumps(best_payload["val_metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best validation package reconstructed MAE: {best_val_package_mae:.4f} K")
    return 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    stats: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_source: float,
    lambda_package: float,
    lambda_source_mean: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "total_loss": 0.0,
        "source_loss": 0.0,
        "package_loss": 0.0,
        "source_mean_loss": 0.0,
        "packages": 0.0,
        "sources": 0.0,
        "max_sources_per_batch": 0.0,
    }
    for batch in loader:
        x = normalize_source_input(batch["x"].to(device), stats)
        target = normalize_source_target_unit(batch["target_unit"].to(device), stats)
        target_rise = batch["target_rise"].to(device)
        source_power = batch["source_power_W"].to(device)
        source_to_package = batch["source_to_package_index"].to(device)
        package_ambient = batch["package_ambient_K"].to(device)
        package_target = batch["package_full_temperature"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_normalized = model(x)
        source_loss = criterion(pred_normalized, target)
        pred_unit = unnormalize_source_prediction(pred_normalized, stats)
        pred_rise = predict_source_rise(pred_unit, source_power)
        pred_sum = segment_sum_fields(pred_rise, source_to_package, int(package_target.shape[0]))
        pred_temp = package_ambient[:, None, None] + pred_sum
        package_loss = criterion(pred_temp, package_target)
        if lambda_source_mean > 0.0:
            source_mean_loss = criterion(pred_rise.mean(dim=(-2, -1)), target_rise.mean(dim=(-2, -1)))
        else:
            source_mean_loss = pred_normalized.new_tensor(0.0)
        loss = float(lambda_source) * source_loss + float(lambda_package) * package_loss + float(lambda_source_mean) * source_mean_loss
        loss.backward()
        optimizer.step()
        packages = int(package_target.shape[0])
        sources = int(x.shape[0])
        totals["total_loss"] += float(loss.item()) * packages
        totals["source_loss"] += float(source_loss.item()) * packages
        totals["package_loss"] += float(package_loss.item()) * packages
        totals["source_mean_loss"] += float(source_mean_loss.item()) * packages
        totals["packages"] += packages
        totals["sources"] += sources
        totals["max_sources_per_batch"] = max(totals["max_sources_per_batch"], float(sources))
    package_count = max(totals["packages"], 1.0)
    return {
        "total_loss": totals["total_loss"] / package_count,
        "source_loss": totals["source_loss"] / package_count,
        "package_loss": totals["package_loss"] / package_count,
        "source_mean_loss": totals["source_mean_loss"] / package_count,
        "packages_per_batch": float(getattr(loader, "batch_size", 0) or 0),
        "effective_sources_per_step": totals["sources"] / max(len(loader), 1),
        "max_sources_per_batch": totals["max_sources_per_batch"],
    }


def segment_sum_fields(values: torch.Tensor, source_to_package: torch.Tensor, num_packages: int) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"values must have shape [sources,H,W], got {tuple(values.shape)}")
    result = values.new_zeros((int(num_packages), int(values.shape[-2]), int(values.shape[-1])))
    result.index_add_(0, source_to_package.long(), values)
    return result


def package_loss_weight(lambda_package: float, warmup_epochs: int, epoch: int) -> float:
    if lambda_package <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return float(lambda_package)
    progress = max(0.0, min(1.0, (float(epoch) - 1.0) / float(warmup_epochs)))
    return float(lambda_package) * progress


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    stats: Any,
    device: torch.device,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
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
        source_error = pred_rise - target_rise
        source_unit_errors.append((pred_unit - target_unit).detach().cpu().numpy())
        source_physical_errors.append(source_error.detach().cpu().numpy())
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
                    "total_power_W": 0.0,
                    "source_signed_mean_errors": [],
                    "source_abs_mean_errors": [],
                },
            )
            group["pred_sum"] += pred_np[i]
            group["target_sum"] += target_np[i]
            group["num_sources"] += 1
            group["total_power_W"] += float(meta["source_power_W"])
            source_error_i = source_error[i].detach().cpu().numpy()
            group["source_signed_mean_errors"].append(float(np.mean(source_error_i)))
            group["source_abs_mean_errors"].append(float(np.mean(np.abs(source_error_i))))
    source_unit = aggregate_error(np.concatenate([e.reshape(-1) for e in source_unit_errors]), suffix="K_per_W")
    source_physical = aggregate_error(np.concatenate([e.reshape(-1) for e in source_physical_errors]), suffix="K")
    package = package_metrics(groups, data_root=data_root)
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
        "package_bias_records": package["records"],
        "package_bias_summary": package["bias_summary"],
        "prediction_stats": prediction_stats,
    }


def package_metrics(groups: dict[str, dict[str, Any]], data_root=None) -> dict[str, Any]:
    records = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for uid, group in groups.items():
        if int(group["num_sources"]) != int(group["num_chiplets"]):
            continue
        pred_temp = float(group["ambient_K"]) + group["pred_sum"]
        full = group["full_temperature"]
        base = field_metrics(pred_temp, full)
        package_error = pred_temp - full
        summed_source_error = group["pred_sum"] - group["target_sum"]
        source_signed = np.asarray(group["source_signed_mean_errors"], dtype=np.float64)
        source_abs = np.asarray(group["source_abs_mean_errors"], dtype=np.float64)
        layout_path = Path(group["layout_path"])
        if not layout_path.is_absolute():
            layout_path = (
                Path(data_root).expanduser().resolve() / layout_path
                if data_root is not None
                else REPO_ROOT / layout_path
            )
        layout = load_json(layout_path)
        chip = chiplet_metrics(pred_temp, full, layout, full.shape)
        record = {
            "original_sample_uid": uid,
            "case_id": group["case_id"],
            "num_sources": int(group["num_sources"]),
            "total_power_W": float(group["total_power_W"]),
            "mae_K": base["mae_K"],
            "rmse_K": base["rmse_K"],
            "max_abs_error_K": base["max_abs_error_K"],
            "mean_signed_error_K": float(np.mean(package_error)),
            "summed_source_mean_signed_error_K": float(np.mean(summed_source_error)),
            "mean_source_signed_error_K": float(np.mean(source_signed)) if source_signed.size else 0.0,
            "mean_source_abs_error_K": float(np.mean(source_abs)) if source_abs.size else 0.0,
            "positive_source_bias_fraction": float(np.mean(source_signed > 0.0)) if source_signed.size else 0.0,
            "negative_source_bias_fraction": float(np.mean(source_signed < 0.0)) if source_signed.size else 0.0,
            "chiplet_mean_temperature_mae_K": chip["chiplet_mean_temperature_mae_K"],
            "chiplet_peak_temperature_mae_K": chip["chiplet_peak_temperature_mae_K"],
            "inter_chiplet_delta_T_mae_K": chip["inter_chiplet_delta_T_mae_K"],
        }
        records.append(record)
        by_case[str(group["case_id"])].append(record)
    if not records:
        return {
            "overall": {"mae_K": None, "rmse_K": None, "num_packages": 0},
            "by_case": {},
            "records": [],
            "bias_summary": {},
        }
    overall = {
        "mae_K": float(np.mean([r["mae_K"] for r in records])),
        "rmse_K": float(np.mean([r["rmse_K"] for r in records])),
        "max_abs_error_K": float(np.max([r["max_abs_error_K"] for r in records])),
        "mean_signed_error_K": float(np.mean([r["mean_signed_error_K"] for r in records])),
        "mean_abs_signed_error_K": float(np.mean([abs(r["mean_signed_error_K"]) for r in records])),
        "chiplet_mean_temperature_mae_K": mean_optional(r["chiplet_mean_temperature_mae_K"] for r in records),
        "chiplet_peak_temperature_mae_K": mean_optional(r["chiplet_peak_temperature_mae_K"] for r in records),
        "inter_chiplet_delta_T_mae_K": mean_optional(r["inter_chiplet_delta_T_mae_K"] for r in records),
        "num_packages": len(records),
    }
    case_payload = {
        case: {
            "mae_K": float(np.mean([r["mae_K"] for r in items])),
            "rmse_K": float(np.mean([r["rmse_K"] for r in items])),
            "mean_signed_error_K": float(np.mean([r["mean_signed_error_K"] for r in items])),
            "num_packages": len(items),
        }
        for case, items in sorted(by_case.items())
    }
    return {"overall": overall, "by_case": case_payload, "records": records, "bias_summary": package_bias_summary(records)}


def package_bias_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "num_packages": len(records),
        "package_mae_vs_source_count_spearman": spearman([r["mae_K"] for r in records], [r["num_sources"] for r in records]),
        "package_mae_vs_total_power_spearman": spearman([r["mae_K"] for r in records], [r["total_power_W"] for r in records]),
        "package_signed_bias_vs_source_count_spearman": spearman([r["mean_signed_error_K"] for r in records], [r["num_sources"] for r in records]),
        "package_signed_bias_vs_mean_source_signed_bias_spearman": spearman(
            [r["mean_signed_error_K"] for r in records],
            [r["mean_source_signed_error_K"] for r in records],
        ),
        "mean_positive_source_bias_fraction": float(np.mean([r["positive_source_bias_fraction"] for r in records])),
        "mean_negative_source_bias_fraction": float(np.mean([r["negative_source_bias_fraction"] for r in records])),
    }


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if float(np.std(a_arr)) == 0.0 or float(np.std(b_arr)) == 0.0:
        return None
    a_rank = rankdata(a_arr)
    b_rank = rankdata(b_arr)
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique_values
    sums = np.bincount(inverse, weights=ranks)
    mean_ranks = sums / counts
    return mean_ranks[inverse]


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


def checkpoint_payload(
    model: nn.Module,
    model_config: dict[str, Any],
    stats: Any,
    epoch: int,
    val_metrics: dict[str, Any],
    args: argparse.Namespace,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_package_mae: float,
    epochs_without_improvement: int,
    training_lineage: dict[str, Any] | None,
    resume_signature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "normalization": stats.to_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
        "best_val_package_mae": best_val_package_mae,
        "epochs_without_improvement": epochs_without_improvement,
        "training_lineage": training_lineage,
        "resume_signature": resume_signature,
    }


def make_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int) -> Any:
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    raise ValueError(name)


def step_scheduler(name: str, scheduler: Any, metric: float) -> None:
    if scheduler is None:
        return
    if name == "plateau":
        scheduler.step(metric)
    else:
        scheduler.step()


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


def make_package_loader(dataset: SourceResponsePackageDataset, packages_per_batch: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=packages_per_batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=source_response_package_collate,
    )


def init_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(
            [
                "epoch",
                "train_total_loss",
                "train_source_loss",
                "train_package_loss",
                "train_source_mean_loss",
                "active_lambda_package",
                "packages_per_batch",
                "effective_sources_per_step",
                "max_sources_per_batch",
                "val_source_mae_K_per_W",
                "val_source_mae_K",
                "val_package_mae_K",
                "val_package_rmse_K",
                "val_package_mean_signed_error_K",
                "val_package_mean_abs_signed_error_K",
                "package_mae_vs_source_count_spearman",
                "package_mae_vs_total_power_spearman",
                "package_signed_bias_vs_source_count_spearman",
                "package_signed_bias_vs_mean_source_signed_bias_spearman",
                "pred_K_per_W_mean",
                "pred_K_per_W_std",
                "pred_K_per_W_min",
                "pred_K_per_W_max",
                "pred_source_rise_K_mean",
                "pred_source_rise_K_std",
                "pred_source_rise_K_min",
                "pred_source_rise_K_max",
                "negative_prediction_fraction",
                "learning_rate",
                "epoch_runtime_s",
                "is_best",
            ]
        )


def append_log(
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, Any],
    runtime: float,
    is_best: bool,
    active_package_weight: float,
    optimizer: torch.optim.Optimizer,
) -> None:
    pred_unit = val_metrics["prediction_stats"]["pred_unit_K_per_W"]
    pred_rise = val_metrics["prediction_stats"]["pred_source_rise_K"]
    bias = val_metrics.get("package_bias_summary", {})
    package = val_metrics["package_reconstruction"]
    with path.open("a", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow([
            epoch,
            train_metrics["total_loss"],
            train_metrics["source_loss"],
            train_metrics["package_loss"],
            train_metrics["source_mean_loss"],
            active_package_weight,
            train_metrics["packages_per_batch"],
            train_metrics["effective_sources_per_step"],
            train_metrics["max_sources_per_batch"],
            val_metrics["source_unit"]["mae_K_per_W"],
            val_metrics["source_physical"]["mae_K"],
            package["mae_K"],
            package["rmse_K"],
            package.get("mean_signed_error_K"),
            package.get("mean_abs_signed_error_K"),
            bias.get("package_mae_vs_source_count_spearman"),
            bias.get("package_mae_vs_total_power_spearman"),
            bias.get("package_signed_bias_vs_source_count_spearman"),
            bias.get("package_signed_bias_vs_mean_source_signed_bias_spearman"),
            pred_unit["mean"],
            pred_unit["std"],
            pred_unit["min"],
            pred_unit["max"],
            pred_rise["mean"],
            pred_rise["std"],
            pred_rise["min"],
            pred_rise["max"],
            val_metrics["prediction_stats"]["negative_unit_response_fraction"],
            optimizer.param_groups[0]["lr"],
            runtime,
            int(is_best),
        ])


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


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
