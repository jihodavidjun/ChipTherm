#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.graph_models import chiplet_metric_values  # noqa: E402
from chiptherm.ml.integrated_inference import (  # noqa: E402
    IntegratedChipThermModel,
    resolve_path,
    rows_from_batch_metadata,
)


HOTSPOT_REFERENCE_S = 4.943711
WARNING_TOLERANCE_K = 0.01
HARD_TOLERANCE_K = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate integrated uncached ChipTherm inference.")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--residual-checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--package-batch-size", nargs="+", default=[8], type=int)
    parser.add_argument("--source-batch-size", nargs="+", default=[64], type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--warmup-batches", default=0, type=int)
    parser.add_argument("--timed-batches", default=None, type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compare-cached-index", default=None, type=Path)
    parser.add_argument("--sample-selection", default="stratified", choices=["stratified", "prefix"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--inference-mode", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-source", action="store_true")
    parser.add_argument("--compile-package-cnn", action="store_true")
    parser.add_argument("--compile-graph", action="store_true")
    parser.add_argument("--device-summation", action="store_true")
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--non-blocking-transfer", action="store_true")
    args = parser.parse_args()

    for source_batch_size in args.source_batch_size:
        if source_batch_size <= 0:
            raise SystemExit("--source-batch-size values must be positive")
    for batch_size in args.package_batch_size:
        if batch_size <= 0:
            raise SystemExit("--package-batch-size values must be positive")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    integrated = IntegratedChipThermModel(
        source_checkpoint=args.source_checkpoint,
        residual_checkpoint=args.residual_checkpoint,
        device=device,
        deterministic=args.deterministic,
        precision=args.precision,
        channels_last=args.channels_last,
        compile_source=args.compile_source,
        compile_package_cnn=args.compile_package_cnn,
        compile_graph=args.compile_graph,
        device_summation=args.device_summation,
        non_blocking_transfer=args.non_blocking_transfer,
    )
    canonical_rows_all = read_rows(args.index)
    canonical_rows = select_rows(canonical_rows_all, max_samples=args.max_samples, mode=args.sample_selection)
    cached_rows = read_rows(args.compare_cached_index) if args.compare_cached_index else None
    if cached_rows is not None:
        cached_rows = cached_rows_for_selected(cached_rows, canonical_rows)
        validate_cached_rows(canonical_rows, cached_rows, expected_source_hash=integrated.source_checkpoint_sha256)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runtime_rows: list[dict[str, Any]] = []
    primary_payload: dict[str, Any] | None = None
    for package_batch_size in args.package_batch_size:
        for source_batch_size in args.source_batch_size:
            payload = evaluate_once(
                integrated=integrated,
                index=args.index,
                cached_rows=cached_rows,
                selected_rows=canonical_rows,
                out_dir=out_dir,
                package_batch_size=package_batch_size,
                source_batch_size=source_batch_size,
                device=device,
                num_workers=args.num_workers,
                profile_components=args.profile_components,
                save_predictions=args.save_predictions and primary_payload is None,
                max_samples=args.max_samples,
                sample_selection=args.sample_selection,
                warmup_batches=args.warmup_batches,
                timed_batches=args.timed_batches if (len(args.package_batch_size) > 1 or len(args.source_batch_size) > 1) else None,
                inference_mode=args.inference_mode,
                pinned_memory=args.pinned_memory,
            )
            runtime_rows.append(runtime_summary_row(package_batch_size, source_batch_size, payload))
            if primary_payload is None:
                primary_payload = payload

    assert primary_payload is not None
    write_csv(out_dir / "integrated_runtime_by_batch_size.csv", runtime_rows)
    write_csv(out_dir / "runtime_sweep.csv", runtime_rows)
    write_csv(out_dir / "integrated_runtime_components.csv", runtime_component_rows(primary_payload))
    write_csv(out_dir / "optimization_comparison.csv", [optimization_comparison_row(args, primary_payload)])
    (out_dir / "integrated_metrics.json").write_text(json.dumps(primary_payload["metrics_payload"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_case_metrics(out_dir / "integrated_metrics_by_case.csv", primary_payload["metrics_by_case"])
    (out_dir / "integrated_runtime.json").write_text(json.dumps(primary_payload["runtime"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "cached_vs_uncached_runtime.json").write_text(
        json.dumps(primary_payload["cached_vs_uncached_runtime"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "numerical_equivalence.json").write_text(json.dumps(primary_payload["equivalence"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "integrated_manifest.json").write_text(json.dumps(primary_payload["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "integrated_inference_manifest.json").write_text(json.dumps(primary_payload["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "integrated_report.md", primary_payload)

    if args.compare_cached_index and primary_payload["equivalence"].get("ok") is False:
        (out_dir / "FAILED_NUMERICAL_EQUIVALENCE").write_text(
            json.dumps(primary_payload["equivalence"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            f"cached-vs-uncached numerical equivalence exceeded hard tolerance "
            f"{HARD_TOLERANCE_K} K; see {out_dir / 'numerical_equivalence.json'}"
        )

    final = primary_payload["metrics_payload"]["final_temperature"]
    print("Integrated ChipTherm evaluation complete")
    print(f"Samples: {primary_payload['num_samples']}")
    print(f"Final MAE/RMSE: {final['mae_K']:.3f} / {final['rmse_K']:.3f} K")
    print(f"Integrated uncached runtime/package: {primary_payload['runtime']['runtime_per_package_s']:.6f} s")
    print(f"Integrated uncached speedup vs HotSpot: {primary_payload['runtime']['speedup_vs_hotspot']:.1f}x")
    print(f"Output: {out_dir}")
    return 0


def evaluate_once(
    *,
    integrated: IntegratedChipThermModel,
    index: Path,
    cached_rows: list[dict[str, str]] | None,
    selected_rows: list[dict[str, str]],
    out_dir: Path,
    package_batch_size: int,
    source_batch_size: int,
    device: torch.device,
    num_workers: int,
    profile_components: bool,
    save_predictions: bool,
    max_samples: int | None,
    sample_selection: str,
    warmup_batches: int,
    timed_batches: int | None,
    inference_mode: bool,
    pinned_memory: bool,
) -> dict[str, Any]:
    dataset = ChipThermDataset(index, target="residual", return_metadata=True, return_graph=integrated.graph_enabled)
    dataset.rows = list(selected_rows)
    loader = DataLoader(
        dataset,
        batch_size=package_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pinned_memory and device.type == "cuda"),
        collate_fn=chiptherm_collate if integrated.graph_enabled else None,
    )
    final_acc = MetricAccumulator()
    cnn_only_acc = MetricAccumulator()
    base_acc = MetricAccumulator()
    mean_acc = ScalarMetricAccumulator()
    centered_acc = MetricAccumulator()
    chip_mean_acc = ScalarAverage()
    chip_peak_acc = ScalarAverage()
    chip_delta_acc = ScalarAverage()
    source_counts: list[int] = []
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            "final_temperature": MetricAccumulator(),
            "cnn_only_temperature": MetricAccumulator(),
            "source_superposition_base": MetricAccumulator(),
        }
    )
    case_sample_counts: dict[str, int] = defaultdict(int)
    case_source_counts: dict[str, int] = defaultdict(int)
    equivalence = EquivalenceAccumulator()
    runtime = RuntimeAccumulator()
    prediction_dir = out_dir / "predictions"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    loader_iter = iter(loader)
    sample_offset = 0
    measured_batches = 0
    total_batches = 0
    while True:
        if timed_batches is not None and measured_batches >= timed_batches:
            break
        try:
            data_start = time.perf_counter()
            batch = next(loader_iter)
            data_loading_s = time.perf_counter() - data_start
        except StopIteration:
            break
        batch_size = int(batch["x"].shape[0])
        batch_rows = rows_from_batch_metadata(batch["metadata"], batch_size)
        batch_start = time.perf_counter()
        context = torch.inference_mode() if inference_mode else nullcontext()
        with context:
            result = integrated.predict_batch(
                batch,
                batch_rows,
                source_batch_size=source_batch_size,
                profile_components=profile_components,
            )
        total_latency_s = time.perf_counter() - batch_start + data_loading_s
        if total_batches >= warmup_batches:
            timings = dict(result["timings"])
            timings["data_loading_batch_preparation_s"] = data_loading_s
            timings["raw_input_to_output_latency_s"] = total_latency_s
            runtime.update(timings, batch_size, sum(result["source_counts"]))
            measured_batches += 1
        source_counts.extend(int(value) for value in result["source_counts"])

        temperature = result["temperature"]
        final = result["final_temperature_K"]
        cnn_only = result["cnn_only_temperature_K"]
        base = result["source_superposition_base_K"]
        outputs = result["outputs"]
        ambient = result["ambient_K"]
        centered_pred = final - final.mean(dim=(-2, -1), keepdim=True)
        centered_target = temperature - temperature.mean(dim=(-2, -1), keepdim=True)
        mean_target = (temperature - ambient[:, None, None]).mean(dim=(-2, -1))

        final_acc.update(final, temperature)
        cnn_only_acc.update(cnn_only, temperature)
        base_acc.update(base, temperature)
        mean_acc.update(outputs["mean_rise"], mean_target)
        centered_acc.update(centered_pred, centered_target)
        if integrated.graph_enabled and result["graph_batch"] is not None:
            chip = chiplet_metric_values(final, temperature, result["graph_batch"])
            chip_mean_acc.update(chip["mean_abs_error"])
            chip_peak_acc.update(chip["peak_abs_error"])
            chip_delta_acc.update(chip["delta_mae"])

        case_ids = metadata_values(batch["metadata"], "case_id", batch_size)
        sample_uids = metadata_values(batch["metadata"], "sample_uid", batch_size)
        for i, case_id in enumerate(case_ids):
            key = str(case_id)
            case_sample_counts[key] += 1
            case_source_counts[key] += int(result["source_counts"][i])
            by_case[key]["final_temperature"].update(final[i : i + 1], temperature[i : i + 1])
            by_case[key]["cnn_only_temperature"].update(cnn_only[i : i + 1], temperature[i : i + 1])
            by_case[key]["source_superposition_base"].update(base[i : i + 1], temperature[i : i + 1])
            if save_predictions:
                case_dir = prediction_dir / key
                case_dir.mkdir(parents=True, exist_ok=True)
                np.save(case_dir / f"{sample_uids[i]}_final_temperature.npy", final[i].detach().cpu().numpy().astype(np.float32))
                np.save(case_dir / f"{sample_uids[i]}_source_base.npy", base[i].detach().cpu().numpy().astype(np.float32))

        if cached_rows is not None:
            cached_batch = cached_rows[sample_offset : sample_offset + batch_size]
            compare_cached_batch(integrated, batch, cached_batch, result, equivalence)
        sample_offset += batch_size
        total_batches += 1

    metrics_payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index": str(index.resolve()),
        "num_samples": final_acc.num_samples,
        "final_temperature": final_acc.compute(),
        "cnn_only_temperature": cnn_only_acc.compute(),
        "source_superposition_base": base_acc.compute(),
        "mean_rise": mean_acc.compute(),
        "centered_field": centered_acc.compute(),
        "chiplet_mean_temperature": chip_mean_acc.compute("mae_K"),
        "chiplet_peak_temperature": chip_peak_acc.compute("mae_K"),
        "inter_chiplet_delta_T": chip_delta_acc.compute("mae_K"),
        "parameter_counts": {
            "source_response": integrated.manifest().get("source_parameter_count"),
            "residual_package_model": integrated.manifest().get("residual_parameter_count"),
        },
    }
    runtime_payload = runtime.compute()
    add_gpu_model_runtime(runtime_payload)
    if device.type == "cuda":
        runtime_payload["peak_cuda_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        runtime_payload["peak_cuda_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    runtime_payload["hotspot_reference_s_per_package"] = HOTSPOT_REFERENCE_S
    runtime_payload["speedup_vs_hotspot"] = (
        HOTSPOT_REFERENCE_S / runtime_payload["runtime_per_package_s"]
        if runtime_payload.get("runtime_per_package_s")
        else None
    )
    cached_runtime = {
        "cached_path_note": "Cached source-base runtime is measured by scripts/evaluate_residual_cnn.py; integrated runtime here includes source-base generation.",
        "integrated_uncached_runtime_per_package_s": runtime_payload.get("runtime_per_package_s"),
        "hotspot_reference_s_per_package": HOTSPOT_REFERENCE_S,
        "integrated_uncached_speedup_vs_hotspot": runtime_payload.get("speedup_vs_hotspot"),
    }
    manifest = integrated.manifest()
    manifest.update(
        {
            "package_batch_size": package_batch_size,
            "source_batch_size": source_batch_size,
            "profile_components": profile_components,
            "max_samples": max_samples,
            "sample_selection": sample_selection if max_samples is not None else "full",
            "selected_sample_count": len(selected_rows),
            "warmup_batches": warmup_batches,
            "timed_batches": timed_batches,
            "warning_tolerance_K": WARNING_TOLERANCE_K,
            "hard_tolerance_K": HARD_TOLERANCE_K,
            "inference_mode": inference_mode,
            "pinned_memory": pinned_memory,
        }
    )
    source_counts_array = np.asarray(source_counts, dtype=np.float64) if source_counts else np.asarray([], dtype=np.float64)
    metrics_payload["source_counts_per_package"] = {
        "mean": float(source_counts_array.mean()) if source_counts_array.size else None,
        "min": float(source_counts_array.min()) if source_counts_array.size else None,
        "max": float(source_counts_array.max()) if source_counts_array.size else None,
        "p50": float(np.percentile(source_counts_array, 50)) if source_counts_array.size else None,
        "p95": float(np.percentile(source_counts_array, 95)) if source_counts_array.size else None,
    }
    metrics_payload["case_sample_counts"] = dict(sorted(case_sample_counts.items()))
    metrics_payload["case_mean_source_counts"] = {
        case_id: float(case_source_counts[case_id] / max(count, 1))
        for case_id, count in sorted(case_sample_counts.items())
    }
    return {
        "num_samples": final_acc.num_samples,
        "metrics_payload": metrics_payload,
        "metrics_by_case": {
            case_id: {name: acc.compute() for name, acc in sorted(accs.items())}
            for case_id, accs in sorted(by_case.items())
        },
        "runtime": runtime_payload,
        "cached_vs_uncached_runtime": cached_runtime,
        "equivalence": equivalence.compute(),
        "manifest": manifest,
    }


def compare_cached_batch(
    integrated: IntegratedChipThermModel,
    batch: dict[str, Any],
    cached_rows: list[dict[str, str]],
    integrated_result: dict[str, Any],
    equivalence: "EquivalenceAccumulator",
) -> None:
    source_base_values = []
    for row in cached_rows:
        value = row.get("source_superposition_base_path")
        if not value:
            raise ValueError(f"cached row {row.get('sample_uid')} is missing source_superposition_base_path")
        source_base_values.append(np.load(resolve_path(value)).astype(np.float32, copy=False))
    cached_base = torch.from_numpy(np.stack(source_base_values).astype(np.float32, copy=False))
    cached_result = integrated.residual_from_base(batch, cached_base)
    equivalence.update("source_superposition_base_K", integrated_result["source_superposition_base_K"], cached_base.to(integrated.device))
    left_components = integrated_result.get("components", {})
    right_components = cached_result.get("components", {})
    component_pairs = {
        "cnn_mean_rise_correction_K": ("cnn_mean_rise", "cnn_mean_rise"),
        "cnn_centered_field_correction_K": ("cnn_centered_field", "cnn_centered_field"),
        "graph_mean_rise_correction_K": ("graph_mean_delta", "graph_mean_delta"),
        "graph_centered_field_correction_K": ("graph_correction_field", "graph_correction_field"),
        "residual_mean_rise_total_K": ("mean_rise", "mean_rise"),
        "residual_centered_field_total_K": ("centered_field", "centered_field"),
    }
    for label, (left_key, right_key) in component_pairs.items():
        left = left_components.get(left_key)
        right = right_components.get(right_key)
        if left is not None and right is not None:
            equivalence.update(label, left, right)
    equivalence.update("cnn_only_temperature_K", integrated_result["cnn_only_temperature_K"], cached_result["cnn_only_temperature_K"])
    graph_left = integrated_result.get("graph_correction_K")
    graph_right = cached_result.get("graph_correction_K")
    if graph_left is not None and graph_right is not None:
        equivalence.update("graph_correction_K", graph_left, graph_right)
    equivalence.update("final_temperature_K", integrated_result["final_temperature_K"], cached_result["final_temperature_K"])


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


class ScalarAverage:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, tensor: torch.Tensor) -> None:
        self.values.extend(float(value) for value in tensor.detach().float().reshape(-1).cpu().tolist())

    def compute(self, name: str) -> dict[str, float]:
        if not self.values:
            return {}
        arr = np.asarray(self.values, dtype=np.float64)
        return {name: float(arr.mean()), "count": float(arr.size)}


class RuntimeAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)
        self.total_packages = 0
        self.total_sources = 0

    def update(self, timings: dict[str, float], packages: int, sources: int) -> None:
        self.total_packages += int(packages)
        self.total_sources += int(sources)
        for key, value in timings.items():
            self.values[key].append(float(value))

    def compute(self) -> dict[str, Any]:
        total_latency = sum(self.values.get("raw_input_to_output_latency_s", []))
        result: dict[str, Any] = {
            "num_packages": self.total_packages,
            "num_sources": self.total_sources,
            "runtime_total_s": total_latency,
            "runtime_per_package_s": total_latency / max(self.total_packages, 1),
            "runtime_per_source_s": total_latency / max(self.total_sources, 1),
            "throughput_packages_per_s": self.total_packages / total_latency if total_latency > 0 else None,
            "batch_latency_s": summarize_runtime_values(self.values.get("raw_input_to_output_latency_s", [])),
            "components": {},
        }
        for key, values in sorted(self.values.items()):
            arr = np.asarray(values, dtype=np.float64)
            result["components"][key] = {
                "total_s": float(arr.sum()),
                "mean_s_per_batch": float(arr.mean()),
                "median_s_per_batch": float(np.median(arr)),
                "p95_s_per_batch": float(np.percentile(arr, 95)),
                "p99_s_per_batch": float(np.percentile(arr, 99)),
                "mean_s_per_package": float(arr.sum() / max(self.total_packages, 1)),
            }
        return result


class EquivalenceAccumulator:
    def __init__(self) -> None:
        self.items: dict[str, DifferenceAccumulator] = defaultdict(DifferenceAccumulator)

    def update(self, name: str, left: torch.Tensor, right: torch.Tensor) -> None:
        self.items[name].update(left, right)

    def compute(self) -> dict[str, Any]:
        values = {
            name: {
                "max_abs_diff_K": metric.compute().get("max_abs_diff_K", 0.0),
                "mean_abs_diff_K": metric.compute().get("mean_abs_diff_K", 0.0),
                "rmse_diff_K": metric.compute().get("rmse_diff_K", 0.0),
                "warning": metric.compute().get("max_abs_diff_K", 0.0) > WARNING_TOLERANCE_K,
                "ok": metric.compute().get("max_abs_diff_K", 0.0) <= HARD_TOLERANCE_K,
            }
            for name, metric in sorted(self.items.items())
        }
        return {
            "warning_tolerance_K": WARNING_TOLERANCE_K,
            "hard_tolerance_K": HARD_TOLERANCE_K,
            "comparisons": values,
            "ok": all(item["ok"] for item in values.values()) if values else None,
            "warning_count": sum(1 for item in values.values() if item["warning"]),
        }


class DifferenceAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0

    def update(self, left: torch.Tensor, right: torch.Tensor) -> None:
        left_cpu = left.detach().float().cpu()
        right_cpu = right.detach().float().cpu()
        if tuple(left_cpu.shape) != tuple(right_cpu.shape):
            raise ValueError(f"equivalence shape mismatch: {tuple(left_cpu.shape)} != {tuple(right_cpu.shape)}")
        diff = left_cpu - right_cpu
        abs_diff = diff.abs()
        self.count += int(diff.numel())
        self.sum_abs += float(abs_diff.sum().item())
        self.sum_sq += float((diff * diff).sum().item())
        self.max_abs = max(self.max_abs, float(abs_diff.max().item()))

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "max_abs_diff_K": self.max_abs,
            "mean_abs_diff_K": self.sum_abs / self.count,
            "rmse_diff_K": (self.sum_sq / self.count) ** 0.5,
        }


def runtime_summary_row(package_batch_size: int, source_batch_size: int, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload["runtime"]
    metrics = payload["metrics_payload"]
    row = {
        "package_batch_size": package_batch_size,
        "source_batch_size": source_batch_size,
        "num_packages": runtime.get("num_packages"),
        "num_sources": runtime.get("num_sources"),
        "mean_sources_per_package": (
            float(runtime.get("num_sources", 0)) / max(float(runtime.get("num_packages", 0)), 1.0)
            if runtime.get("num_packages")
            else None
        ),
        "case_sample_counts_json": json.dumps(metrics.get("case_sample_counts", {}), sort_keys=True),
        "case_mean_source_counts_json": json.dumps(metrics.get("case_mean_source_counts", {}), sort_keys=True),
        "runtime_per_package_s": runtime.get("runtime_per_package_s"),
        "runtime_per_source_s": runtime.get("runtime_per_source_s"),
        "throughput_packages_per_s": runtime.get("throughput_packages_per_s"),
        "gpu_model_runtime_per_package_s": runtime.get("gpu_model_runtime_per_package_s"),
        "gpu_model_throughput_packages_per_s": runtime.get("gpu_model_throughput_packages_per_s"),
        "speedup_vs_hotspot": runtime.get("speedup_vs_hotspot"),
        "peak_cuda_memory_allocated_bytes": runtime.get("peak_cuda_memory_allocated_bytes"),
    }
    for name, component in sorted(runtime.get("components", {}).items()):
        row[f"{name}_per_package_s"] = component.get("mean_s_per_package")
    return row


def runtime_component_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, component in sorted(payload["runtime"].get("components", {}).items()):
        row = {"component": name}
        row.update(component)
        rows.append(row)
    return rows


def optimization_comparison_row(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    final = payload["metrics_payload"].get("final_temperature", {})
    equivalence = payload.get("equivalence", {})
    max_diff = None
    if equivalence.get("comparisons"):
        max_diff = max(float(item.get("max_abs_diff_K", 0.0)) for item in equivalence["comparisons"].values())
    return {
        "precision": args.precision,
        "inference_mode": bool(args.inference_mode),
        "channels_last": bool(args.channels_last),
        "compile_source": bool(args.compile_source),
        "compile_package_cnn": bool(args.compile_package_cnn),
        "compile_graph": bool(args.compile_graph),
        "device_summation": bool(args.device_summation),
        "pinned_memory": bool(args.pinned_memory),
        "non_blocking_transfer": bool(args.non_blocking_transfer),
        "package_batch_size": args.package_batch_size[0],
        "source_batch_size": args.source_batch_size[0],
        "final_mae_K": final.get("mae_K"),
        "runtime_per_package_s": payload["runtime"].get("runtime_per_package_s"),
        "speedup_vs_hotspot": payload["runtime"].get("speedup_vs_hotspot"),
        "max_equivalence_diff_K": max_diff,
        "equivalence_ok": equivalence.get("ok"),
    }


def summarize_runtime_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "p99_s": float(np.percentile(arr, 99)),
    }


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def validate_cached_rows(
    canonical_rows: list[dict[str, str]],
    cached_rows: list[dict[str, str]],
    *,
    expected_source_hash: str | None = None,
) -> None:
    if len(canonical_rows) != len(cached_rows):
        raise SystemExit(f"cached index row count mismatch: canonical={len(canonical_rows)} cached={len(cached_rows)}")
    for index, (canonical, cached) in enumerate(zip(canonical_rows, cached_rows, strict=True)):
        if canonical["sample_uid"] != cached["sample_uid"]:
            raise SystemExit(
                f"cached index row order mismatch at row {index}: {canonical['sample_uid']} != {cached['sample_uid']}"
            )
        if not cached.get("source_superposition_base_path"):
            raise SystemExit(f"cached row {index} missing source_superposition_base_path")
        if cached.get("source_base_mode") and cached["source_base_mode"] != "source_superposition_v1":
            raise SystemExit(f"cached row {index} has incompatible source_base_mode={cached['source_base_mode']}")
        if cached.get("source_base_shape") and cached["source_base_shape"] != "64x64":
            raise SystemExit(f"cached row {index} has incompatible source_base_shape={cached['source_base_shape']}")
        if cached.get("source_base_dtype") and cached["source_base_dtype"] != "float32":
            raise SystemExit(f"cached row {index} has incompatible source_base_dtype={cached['source_base_dtype']}")
        if cached.get("source_count") and canonical.get("num_chiplets"):
            if int(float(cached["source_count"])) != int(float(canonical["num_chiplets"])):
                raise SystemExit(
                    f"cached row {index} source_count mismatch: "
                    f"{cached['source_count']} != canonical num_chiplets {canonical['num_chiplets']}"
                )
        cached_hash = cached.get("source_checkpoint_sha256")
        if expected_source_hash and cached_hash and cached_hash != expected_source_hash:
            raise SystemExit(
                f"cached source checkpoint hash mismatch at row {index}: "
                f"{cached_hash} != {expected_source_hash}"
            )


def select_rows(rows: list[dict[str, str]], *, max_samples: int | None, mode: str) -> list[dict[str, str]]:
    if max_samples is None or int(max_samples) >= len(rows):
        return list(rows)
    limit = int(max_samples)
    if limit <= 0:
        raise SystemExit("--max-samples must be positive")
    if mode == "prefix":
        return list(rows[:limit])
    if mode != "stratified":
        raise SystemExit(f"unsupported sample selection mode: {mode}")
    original_order = {str(row["sample_uid"]): index for index, row in enumerate(rows)}
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id", ""))].append(row)
    cases = sorted(by_case)
    base = limit // max(len(cases), 1)
    remainder = limit % max(len(cases), 1)
    selected: list[dict[str, str]] = []
    for case_index, case_id in enumerate(cases):
        target = base + (1 if case_index < remainder else 0)
        if target <= 0:
            continue
        case_rows = sorted(
            by_case[case_id],
            key=lambda row: (
                int(float(row.get("num_chiplets") or row.get("source_count") or 0)),
                str(row.get("sample_uid", "")),
            ),
        )
        selected.extend(evenly_spaced(case_rows, min(target, len(case_rows))))
    return sorted(selected, key=lambda row: original_order[str(row["sample_uid"])])


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = np.linspace(0, len(rows) - 1, num=count)
    selected_indices = []
    seen = set()
    for value in indices:
        index = int(round(float(value)))
        while index in seen and index + 1 < len(rows):
            index += 1
        while index in seen and index - 1 >= 0:
            index -= 1
        if index not in seen:
            seen.add(index)
            selected_indices.append(index)
    return [rows[index] for index in selected_indices[:count]]


def cached_rows_for_selected(cached_rows: list[dict[str, str]], selected_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_uid = {str(row["sample_uid"]): row for row in cached_rows}
    result: list[dict[str, str]] = []
    for row in selected_rows:
        uid = str(row["sample_uid"])
        if uid not in by_uid:
            raise SystemExit(f"cached comparison index is missing selected sample_uid {uid}")
        result.append(by_uid[uid])
    return result


def read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


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


def add_gpu_model_runtime(runtime_payload: dict[str, Any]) -> None:
    components = runtime_payload.get("components", {})
    gpu_keys = (
        "source_host_to_device_transfer_s",
        "source_input_normalization_s",
        "source_response_model_inference_s",
        "source_kw_denormalization_s",
        "source_power_scaling_s",
        "source_segment_sum_s",
        "ambient_base_reconstruction_s",
        "residual_input_assembly_s",
        "residual_total_forward_s",
        "final_reconstruction_s",
    )
    total = 0.0
    present = []
    for key in gpu_keys:
        component = components.get(key)
        if component:
            total += float(component.get("total_s", 0.0))
            present.append(key)
    packages = float(runtime_payload.get("num_packages") or 0.0)
    runtime_payload["gpu_model_runtime_total_s"] = total
    runtime_payload["gpu_model_runtime_per_package_s"] = total / max(packages, 1.0) if packages else None
    runtime_payload["gpu_model_throughput_packages_per_s"] = packages / total if total > 0 else None
    runtime_payload["gpu_model_runtime_components"] = present
    runtime_payload["gpu_model_runtime_note"] = (
        "This is model-side tensor work after DataLoader timing and excludes package YAML/JSON parsing and source raster construction. "
        "With host summation it still includes source_segment_sum_s because it is part of the model assembly path."
    )


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    rows: list[dict[str, Any]] = []
    for case_id, metrics in sorted(case_metrics.items()):
        final = metrics.get("final_temperature", {})
        cnn = metrics.get("cnn_only_temperature", {})
        base = metrics.get("source_superposition_base", {})
        rows.append(
            {
                "case_id": case_id,
                "final_mae_K": final.get("mae_K", ""),
                "final_rmse_K": final.get("rmse_K", ""),
                "cnn_only_mae_K": cnn.get("mae_K", ""),
                "source_base_mae_K": base.get("mae_K", ""),
                "hotspot_location_error_cells": final.get("hotspot_location_error_cells", ""),
            }
        )
    write_csv(path, rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics_payload"]
    runtime = payload["runtime"]
    equivalence = payload["equivalence"]
    lines = [
        "# Integrated ChipTherm Inference Report",
        "",
        f"Samples: {payload['num_samples']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Final MAE K | {metrics['final_temperature'].get('mae_K', float('nan')):.4f} |",
        f"| Final RMSE K | {metrics['final_temperature'].get('rmse_K', float('nan')):.4f} |",
        f"| Chiplet mean MAE K | {metrics.get('chiplet_mean_temperature', {}).get('mae_K', float('nan')):.4f} |",
        f"| Chiplet peak MAE K | {metrics.get('chiplet_peak_temperature', {}).get('mae_K', float('nan')):.4f} |",
        f"| Inter-chiplet delta-T MAE K | {metrics.get('inter_chiplet_delta_T', {}).get('mae_K', float('nan')):.4f} |",
        f"| Integrated runtime/package s | {runtime.get('runtime_per_package_s', float('nan')):.6f} |",
        f"| Speedup vs HotSpot | {runtime.get('speedup_vs_hotspot', float('nan')):.1f}x |",
        "",
        f"Numerical equivalence ok: `{equivalence.get('ok')}`",
        "",
        "Cached source-base runtime is not the uncached deployment runtime; this integrated report includes source-base generation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
