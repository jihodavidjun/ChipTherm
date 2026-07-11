#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate compact analytical physics candidate predictions.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-samples", default=None, type=int)
    args = parser.parse_args()

    index = args.index.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_index(index)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    records: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    spectra: list[np.ndarray] = []
    residual_values: list[float] = []
    runtimes: list[float] = []
    hotspot_runtimes: list[float] = []

    for row in rows:
        y = np.load(resolve_path(row["y_path"], index.parent)).astype(np.float32, copy=False)
        pred = np.load(resolve_path(row["prediction_path"], index.parent)).astype(np.float32, copy=False)
        residual = y - pred
        metrics = sample_metrics(pred, y)
        residual_stats = residual_complexity(residual)
        spectrum = radial_spectrum(residual)
        spectra.append(spectrum)
        residual_values.extend([float(residual.mean()), float(residual.std())])
        runtime = optional_float(row.get("physics_runtime_s"))
        hotspot_runtime = optional_float(row.get("hotspot_runtime_s"))
        if runtime is not None:
            runtimes.append(runtime)
        if hotspot_runtime is not None:
            hotspot_runtimes.append(hotspot_runtime)
        record = {
            "sample_uid": row["sample_uid"],
            "case_id": row["case_id"],
            "physics_runtime_s": runtime,
            "hotspot_runtime_s": hotspot_runtime,
            **metrics,
            **residual_stats,
        }
        records.append(record)
        by_case[row["case_id"]].append(record)

    overall = aggregate_records(records)
    case_payload = {case_id: aggregate_records(case_records) for case_id, case_records in sorted(by_case.items())}
    mean_runtime = float(np.mean(runtimes)) if runtimes else None
    hotspot_runtime = float(np.mean(hotspot_runtimes)) if hotspot_runtimes else None
    speedup = hotspot_runtime / mean_runtime if hotspot_runtime and mean_runtime else None
    average_spectrum = np.mean(np.stack(spectra, axis=0), axis=0) if spectra else np.zeros(1)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index": str(index),
        "num_samples": len(records),
        "physics_only": overall,
        "per_case": case_payload,
        "case02": case_payload.get("case02"),
        "runtime": {
            "physics_runtime_per_sample_s": mean_runtime,
            "hotspot_runtime_reference_s": hotspot_runtime,
            "speedup_vs_hotspot": speedup,
            "runtime_note": "Physics runtime is read from candidate generation metadata and excludes disk I/O.",
        },
        "residual_spectrum": {
            "average_radial_fft_energy": average_spectrum.tolist(),
            "bands": spectrum_bands_from_average(average_spectrum),
        },
        "notes": {
            "residual_sign": "HotSpot - physics_candidate",
            "fft": "Residual FFT is computed after subtracting residual mean.",
        },
    }
    write_json(out_dir / "summary.json", payload)
    write_sample_metrics(out_dir / "sample_metrics.csv", records)
    write_case_metrics(out_dir / "metrics_by_case.csv", case_payload)

    print("Candidate physics evaluation complete")
    print(f"Samples: {len(records)}")
    print(f"MAE/RMSE: {overall['mae_K']:.3f} / {overall['rmse_K']:.3f} K")
    print(f"Case02 MAE: {case_payload.get('case02', {}).get('mae_K', float('nan')):.3f} K")
    print(f"Runtime/sample: {mean_runtime:.6f} s" if mean_runtime else "Runtime/sample: n/a")
    print(f"Speedup vs HotSpot: {speedup:.1f}x" if speedup else "Speedup vs HotSpot: n/a")
    print(f"Output: {out_dir}")
    return 0


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows


def sample_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.shape != target.shape:
        raise SystemExit(f"prediction shape {pred.shape} does not match target {target.shape}")
    error = pred.astype(np.float64) - target.astype(np.float64)
    abs_error = np.abs(error)
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    target_hotspot = np.unravel_index(int(np.argmax(target)), target.shape)
    top_metrics = hotspot_region_metrics(error, target)
    return {
        "mae_K": float(abs_error.mean()),
        "rmse_K": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_K": float(abs_error.max()),
        "mean_signed_error_K": float(error.mean()),
        "hotspot_temp_error_K": float(pred[pred_hotspot] - target[target_hotspot]),
        "hotspot_location_error_cells": float(np.hypot(pred_hotspot[0] - target_hotspot[0], pred_hotspot[1] - target_hotspot[1])),
        **top_metrics,
    }


def hotspot_region_metrics(error: np.ndarray, target: np.ndarray) -> dict[str, float]:
    flat_target = target.reshape(-1)
    flat_abs = np.abs(error).reshape(-1)
    result = {}
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(flat_target.size * frac)))
        indices = np.argpartition(flat_target, -k)[-k:]
        result[f"hotspot_top_{int(frac * 100)}pct_mae_K"] = float(flat_abs[indices].mean())
    return result


def residual_complexity(residual: np.ndarray) -> dict[str, float]:
    residual64 = residual.astype(np.float64, copy=False)
    gy, gx = np.gradient(residual64)
    lap = (
        np.roll(residual64, 1, axis=0)
        + np.roll(residual64, -1, axis=0)
        + np.roll(residual64, 1, axis=1)
        + np.roll(residual64, -1, axis=1)
        - 4.0 * residual64
    )
    bands = spectrum_bands(radial_spectrum(residual64))
    return {
        "residual_mean_K": float(residual64.mean()),
        "residual_std_K": float(residual64.std()),
        "residual_mean_abs_K": float(np.abs(residual64).mean()),
        "mean_gradient_magnitude_K_per_cell": float(np.sqrt(gx * gx + gy * gy).mean()),
        "avg_abs_grad_x_K_per_cell": float(np.abs(gx).mean()),
        "avg_abs_grad_y_K_per_cell": float(np.abs(gy).mean()),
        "avg_abs_laplacian_K_per_cell2": float(np.abs(lap).mean()),
        "low_freq_energy_frac": bands["low_freq_energy_frac"],
        "mid_freq_energy_frac": bands["mid_freq_energy_frac"],
        "high_freq_energy_frac": bands["high_freq_energy_frac"],
    }


def radial_spectrum(residual: np.ndarray) -> np.ndarray:
    centered = residual.astype(np.float64, copy=False) - float(residual.mean())
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    rows, cols = spectrum.shape
    yy, xx = np.indices((rows, cols))
    radius = np.sqrt((yy - rows // 2) ** 2 + (xx - cols // 2) ** 2).astype(np.int32)
    max_radius = int(radius.max())
    radial = np.bincount(radius.reshape(-1), weights=spectrum.reshape(-1), minlength=max_radius + 1)
    counts = np.bincount(radius.reshape(-1), minlength=max_radius + 1)
    return radial / np.maximum(counts, 1)


def spectrum_bands(radial: np.ndarray) -> dict[str, float]:
    total = float(radial.sum())
    if total <= 0.0:
        return {"low_freq_energy_frac": 0.0, "mid_freq_energy_frac": 0.0, "high_freq_energy_frac": 0.0}
    n = len(radial)
    low = float(radial[: max(1, n // 8)].sum() / total)
    mid = float(radial[max(1, n // 8) : max(2, n // 3)].sum() / total)
    high = max(0.0, 1.0 - low - mid)
    return {"low_freq_energy_frac": low, "mid_freq_energy_frac": mid, "high_freq_energy_frac": high}


def spectrum_bands_from_average(radial: np.ndarray) -> dict[str, float]:
    bands = spectrum_bands(radial)
    return bands


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    mean_keys = [
        "mae_K",
        "rmse_K",
        "mean_signed_error_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "hotspot_top_1pct_mae_K",
        "hotspot_top_5pct_mae_K",
        "hotspot_top_10pct_mae_K",
        "residual_mean_K",
        "residual_std_K",
        "residual_mean_abs_K",
        "mean_gradient_magnitude_K_per_cell",
        "avg_abs_grad_x_K_per_cell",
        "avg_abs_grad_y_K_per_cell",
        "avg_abs_laplacian_K_per_cell2",
        "low_freq_energy_frac",
        "mid_freq_energy_frac",
        "high_freq_energy_frac",
    ]
    payload = {"num_samples": float(len(records)), "max_abs_error_K": float(max(record["max_abs_error_K"] for record in records))}
    for key in mean_keys:
        payload[key] = float(np.mean([record[key] for record in records]))
    runtimes = [record["physics_runtime_s"] for record in records if record.get("physics_runtime_s") is not None]
    if runtimes:
        payload["physics_runtime_per_sample_s"] = float(np.mean(runtimes))
    return payload


def write_sample_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    columns = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_case_metrics(path: Path, case_payload: dict[str, dict[str, float]]) -> None:
    if not case_payload:
        return
    columns = ["case_id", *next(iter(case_payload.values())).keys()]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_payload.items()):
            writer.writerow({"case_id": case_id, **metrics})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, base / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
