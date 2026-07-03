#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.physics_baseline import PhysicsBaselineConfig, aggregate_metrics, predict_temperature, sample_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed analytical ChipTherm physics baseline.")
    parser.add_argument("--encoded-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ambient-K", default=318.15, type=float)
    parser.add_argument("--sigmas", nargs="+", default=[1.5, 4.0, 10.0], type=float)
    parser.add_argument("--weights", nargs="+", default=[20.0, 35.0, 60.0], type=float)
    parser.add_argument("--global-R-eff-K-per-W", default=0.03, type=float)
    parser.add_argument("--hotspot-runtime-s", default=None, type=float)
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--save-residuals", action="store_true")
    args = parser.parse_args()

    if len(args.sigmas) != len(args.weights):
        raise SystemExit("--sigmas and --weights must have the same number of values")

    encoded_index = args.encoded_index.resolve()
    encoded_root = encoded_index.parent
    out_dir = args.out_dir.resolve()
    predictions_dir = out_dir / "predictions"
    residuals_dir = out_dir / "optional_residuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    if args.save_residuals:
        residuals_dir.mkdir(parents=True, exist_ok=True)

    config = PhysicsBaselineConfig(
        ambient_K=args.ambient_K,
        global_R_eff_K_per_W=args.global_R_eff_K_per_W,
        sigmas_cells=tuple(args.sigmas),
        weights_K_per_W_per_mm2=tuple(args.weights),
    )
    rows = _read_rows(encoded_index, args.max_samples)
    jsonl_metadata = _read_companion_jsonl_metadata(encoded_index)
    dataset_root = encoded_index.parent.parent if encoded_index.parent.name == "encoded" else encoded_index.parent

    total_power_by_uid: dict[str, float] = {}
    total_power_sources: Counter[str] = Counter()
    for row in rows:
        x_for_power = np.load(encoded_root / row["x_path"], mmap_mode="r")
        total_power, source = _resolve_total_power_W(row, x_for_power, jsonl_metadata, dataset_root)
        total_power_by_uid[row["sample_uid"]] = total_power
        total_power_sources[source] += 1

    fallback_used = any(source.startswith("raster_integration") for source in total_power_sources)
    baseline_config = config.to_dict()
    baseline_config["encoded_index"] = str(encoded_index)
    baseline_config["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    baseline_config["save_predictions"] = not args.no_save_predictions
    baseline_config["save_residuals"] = bool(args.save_residuals)
    baseline_config["total_power_source"] = dict(total_power_sources)
    baseline_config["total_power_fallback_used"] = fallback_used
    (out_dir / "baseline_config.json").write_text(json.dumps(baseline_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics: list[dict[str, float]] = []
    by_case: dict[str, list[dict[str, float]]] = defaultdict(list)
    runtimes: list[float] = []
    total_start = time.perf_counter()

    for row in rows:
        x = np.load(encoded_root / row["x_path"])
        y = np.load(encoded_root / row["y_path"])
        start = time.perf_counter()
        pred = predict_temperature(x, config, total_power_W=total_power_by_uid[row["sample_uid"]])
        runtimes.append(time.perf_counter() - start)
        item_metrics = sample_metrics(pred, y)
        item_metrics["baseline_runtime_s"] = runtimes[-1]
        metrics.append(item_metrics)
        by_case[row["case_id"]].append(item_metrics)

        if not args.no_save_predictions:
            case_dir = predictions_dir / row["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            np.save(case_dir / f"{row['sample_uid']}_tphys.npy", pred.astype(np.float32, copy=False))
        if args.save_residuals:
            case_dir = residuals_dir / row["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            np.save(case_dir / f"{row['sample_uid']}_residual.npy", (y - pred).astype(np.float32, copy=False))

    total_runtime = time.perf_counter() - total_start
    global_metrics = aggregate_metrics(metrics)
    avg_baseline_runtime = float(sum(runtimes) / len(runtimes)) if runtimes else None
    hotspot_runtime = args.hotspot_runtime_s
    if hotspot_runtime is None:
        hotspot_runtime = _infer_hotspot_runtime(encoded_index)
    speedup = (hotspot_runtime / avg_baseline_runtime) if hotspot_runtime and avg_baseline_runtime else None

    metrics_json = {
        "schema_version": 1,
        "encoded_index": str(encoded_index),
        "num_samples": len(metrics),
        "total_eval_runtime_s": total_runtime,
        "baseline_runtime_per_sample_s": avg_baseline_runtime,
        "hotspot_runtime_reference_s": hotspot_runtime,
        "estimated_speedup_vs_hotspot": speedup,
        "global_R_eff_K_per_W": args.global_R_eff_K_per_W,
        "total_power_source": dict(total_power_sources),
        "total_power_fallback_used": fallback_used,
        "global": global_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_case_metrics(out_dir / "metrics_by_case.csv", by_case, hotspot_runtime)

    print("Physics baseline evaluation complete")
    print(f"Samples: {len(metrics)}")
    print(f"Baseline runtime/sample: {avg_baseline_runtime:.6f} s" if avg_baseline_runtime else "Baseline runtime/sample: n/a")
    print(f"HotSpot runtime reference: {hotspot_runtime:.6f} s" if hotspot_runtime else "HotSpot runtime reference: n/a")
    print(f"Estimated speedup: {speedup:.1f}x" if speedup else "Estimated speedup: n/a")
    print(f"Global R_eff: {args.global_R_eff_K_per_W:.6f} K/W")
    print(f"Total power source: {dict(total_power_sources)}")
    if global_metrics:
        print(f"MAE/RMSE/max abs: {global_metrics['mae_K']:.3f} / {global_metrics['rmse_K']:.3f} / {global_metrics['max_abs_error_K']:.3f} K")
        print(f"Hotspot temp/location error: {global_metrics['hotspot_temp_error_K']:.3f} K / {global_metrics['hotspot_location_error_cells']:.3f} cells")
    print(f"Output: {out_dir}")
    return 0


def _read_rows(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if max_samples is not None:
        return rows[:max_samples]
    return rows


def _read_companion_jsonl_metadata(encoded_index: Path) -> dict[str, dict[str, Any]]:
    jsonl_path = encoded_index.with_suffix(".jsonl")
    if not jsonl_path.exists():
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_uid = record.get("sample_uid")
            if sample_uid:
                metadata[str(sample_uid)] = record
    return metadata


def _resolve_total_power_W(
    row: dict[str, str],
    x: np.ndarray,
    jsonl_metadata: dict[str, dict[str, Any]],
    dataset_root: Path,
) -> tuple[float, str]:
    record = jsonl_metadata.get(row["sample_uid"])
    chiplets = record.get("encoding", {}).get("metadata", {}).get("chiplets", []) if record else []
    if chiplets:
        total_power = sum(float(chiplet["power_W"]) for chiplet in chiplets)
        return total_power, "encoded_index_jsonl_chiplet_metadata"

    width_mm, height_mm = _package_size_from_source_layout(row, dataset_root)
    cell_area_mm2 = (width_mm / x.shape[2]) * (height_mm / x.shape[1])
    power_density = x[0].astype(np.float64, copy=False)
    occupancy = x[1].astype(np.float64, copy=False) if x.shape[0] > 1 else 1.0
    total_power = float(np.sum(power_density * occupancy) * cell_area_mm2)
    return total_power, "raster_integration_from_power_density"


def _package_size_from_source_layout(row: dict[str, str], dataset_root: Path) -> tuple[float, float]:
    original_temp_path = Path(row["original_temp_path"])
    sample_dir = original_temp_path.parent.parent
    layout_path = dataset_root / sample_dir / "source" / "layout.json"
    with layout_path.open("r", encoding="utf-8") as fp:
        layout = json.load(fp)
    size = layout["package"]["size"]
    return float(size["width"]), float(size["height"])


def _write_case_metrics(path: Path, by_case: dict[str, list[dict[str, float]]], hotspot_runtime: float | None) -> None:
    columns = [
        "case_id",
        "num_samples",
        "mae_K",
        "rmse_K",
        "max_abs_error_K",
        "mean_signed_error_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "baseline_runtime_per_sample_s",
        "estimated_speedup_vs_hotspot",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id in sorted(by_case):
            metrics = by_case[case_id]
            aggregated = aggregate_metrics(metrics)
            runtime = sum(item["baseline_runtime_s"] for item in metrics) / len(metrics)
            speedup = hotspot_runtime / runtime if hotspot_runtime else None
            writer.writerow(
                {
                    "case_id": case_id,
                    "num_samples": len(metrics),
                    "mae_K": aggregated["mae_K"],
                    "rmse_K": aggregated["rmse_K"],
                    "max_abs_error_K": aggregated["max_abs_error_K"],
                    "mean_signed_error_K": aggregated["mean_signed_error_K"],
                    "hotspot_temp_error_K": aggregated["hotspot_temp_error_K"],
                    "hotspot_location_error_cells": aggregated["hotspot_location_error_cells"],
                    "baseline_runtime_per_sample_s": runtime,
                    "estimated_speedup_vs_hotspot": speedup,
                }
            )


def _infer_hotspot_runtime(encoded_index: Path) -> float | None:
    dataset_root = encoded_index.parent.parent if encoded_index.parent.name == "encoded" else encoded_index.parent
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = manifest.get("average_hotspot_runtime_s")
    return float(value) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
