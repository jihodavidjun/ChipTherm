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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_superposition_diagnostic import chiplet_metrics, field_metrics, load_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate matched physics-v1 and source-superposition base maps.")
    parser.add_argument("--physics-index", required=True, type=Path)
    parser.add_argument("--source-base-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    physics_rows = read_rows(args.physics_index)
    source_rows = read_rows(args.source_base_index)
    source_by_uid = {row["sample_uid"]: row for row in source_rows}
    physics_by_uid = {row["sample_uid"]: row for row in physics_rows}
    if set(source_by_uid) != set(physics_by_uid):
        raise SystemExit("physics and source-base indices do not contain identical sample_uid sets")

    records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    for uid in sorted(source_by_uid):
        for mode, row in (("physics_v1", physics_by_uid[uid]), ("source_superposition_v1", source_by_uid[uid])):
            target = np.load(resolve_path(row["y_path"])).astype(np.float64)
            base = np.load(resolve_path(row["prediction_path"])).astype(np.float64)
            residual = target - base
            metrics = base_metrics(row, mode, base, target)
            records.append(metrics)
            residual_records.append(residual_stats(row, mode, residual))
    overall = summarize(records)
    by_case = summarize_by_case(records)
    residual_summary = summarize_residuals(residual_records)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics_index": str(args.physics_index.resolve()),
        "source_base_index": str(args.source_base_index.resolve()),
        "overall": overall,
        "residual_statistics": residual_summary,
    }
    (out_dir / "base_comparison_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "base_metrics_by_case.csv", by_case)
    write_csv(out_dir / "base_metrics_by_sample.csv", records)
    write_csv(out_dir / "residual_statistics.csv", residual_records)
    write_report(out_dir, payload)
    print("Base comparison complete")
    for mode, metrics in overall.items():
        print(f"{mode}: MAE={metrics['mae_K']:.4f} RMSE={metrics['rmse_K']:.4f} centered={metrics['centered_field_mae_K']:.4f}")
    return 0


def base_metrics(row: dict[str, str], mode: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    metrics = field_metrics(pred, target)
    ambient = ambient_for_row(row)
    delta_pred = pred - ambient
    delta_true = target - ambient
    mean_error = float(np.mean(delta_pred) - np.mean(delta_true))
    centered_error = (delta_pred - np.mean(delta_pred)) - (delta_true - np.mean(delta_true))
    hotspot_pred = np.unravel_index(int(np.argmax(pred)), pred.shape)
    hotspot_true = np.unravel_index(int(np.argmax(target)), target.shape)
    layout_path = source_layout_path(row)
    chip = chiplet_metrics(pred, target, load_json(layout_path), target.shape) if layout_path.exists() else {}
    return {
        "mode": mode,
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "mae_K": metrics["mae_K"],
        "rmse_K": metrics["rmse_K"],
        "max_abs_error_K": metrics["max_abs_error_K"],
        "mean_signed_error_K": metrics["mean_signed_error_K"],
        "mean_rise_abs_error_K": abs(mean_error),
        "centered_field_mae_K": float(np.mean(np.abs(centered_error))),
        "centered_field_rmse_K": float(np.sqrt(np.mean(centered_error * centered_error))),
        "chiplet_mean_temperature_mae_K": chip.get("chiplet_mean_temperature_mae_K"),
        "chiplet_peak_temperature_mae_K": chip.get("chiplet_peak_temperature_mae_K"),
        "inter_chiplet_delta_T_mae_K": chip.get("inter_chiplet_delta_T_mae_K"),
        "hotspot_temp_error_K": float(pred[hotspot_pred] - target[hotspot_true]),
        "hotspot_location_error_cells": float(np.hypot(hotspot_pred[0] - hotspot_true[0], hotspot_pred[1] - hotspot_true[1])),
    }


def residual_stats(row: dict[str, str], mode: str, residual: np.ndarray) -> dict[str, Any]:
    gy, gx = np.gradient(residual)
    fft = np.fft.rfft2(residual - np.mean(residual))
    power = np.abs(fft) ** 2
    total = float(power.sum())
    h, w = residual.shape
    yy = np.fft.fftfreq(h)[:, None]
    xx = np.fft.rfftfreq(w)[None, :]
    radius = np.sqrt(xx * xx + yy * yy)
    low = float(power[radius <= 0.1].sum() / total) if total > 0 else 0.0
    high = float(power[radius >= 0.25].sum() / total) if total > 0 else 0.0
    return {
        "mode": mode,
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "residual_mean_K": float(np.mean(residual)),
        "residual_std_K": float(np.std(residual)),
        "residual_mae_K": float(np.mean(np.abs(residual))),
        "residual_rmse_K": float(np.sqrt(np.mean(residual * residual))),
        "gradient_abs_mean_K_per_cell": float(np.mean(np.sqrt(gx * gx + gy * gy))),
        "low_frequency_energy_fraction": low,
        "high_frequency_energy_fraction": high,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    return {mode: summarize_records(items) for mode, items in sorted(grouped.items())}


def summarize_by_case(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["mode"], record["case_id"])].append(record)
    rows: list[dict[str, Any]] = []
    for (mode, case_id), items in sorted(grouped.items()):
        row = summarize_records(items)
        row.update({"mode": mode, "case_id": case_id})
        rows.append(row)
    return rows


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mae_K",
        "rmse_K",
        "max_abs_error_K",
        "mean_signed_error_K",
        "mean_rise_abs_error_K",
        "centered_field_mae_K",
        "centered_field_rmse_K",
        "chiplet_mean_temperature_mae_K",
        "chiplet_peak_temperature_mae_K",
        "inter_chiplet_delta_T_mae_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
    ]
    out: dict[str, Any] = {"num_samples": len(records)}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def summarize_residuals(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    keys = [
        "residual_mean_K",
        "residual_std_K",
        "residual_mae_K",
        "residual_rmse_K",
        "gradient_abs_mean_K_per_cell",
        "low_frequency_energy_fraction",
        "high_frequency_energy_fraction",
    ]
    return {
        mode: {key: float(np.mean([item[key] for item in items])) for key in keys}
        for mode, items in sorted(grouped.items())
    }


def source_layout_path(row: dict[str, str]) -> Path:
    if row.get("layout_path"):
        return resolve_path(row["layout_path"])
    x_path = resolve_path(row["x_path"])
    # encoded paths live under derived dataset trees; source layout is available via graph/source manifests only in some rows.
    source_dir = row.get("source_dir") or row.get("layout_dir") or ""
    if source_dir:
        return resolve_path(source_dir) / "layout.json"
    graph_path = row.get("graph_path", "")
    if graph_path:
        # fall back to no chiplet metrics if exact layout is not directly encoded.
        return Path("__missing_layout__")
    candidate = x_path.parent / "layout.json"
    return candidate


def ambient_for_row(row: dict[str, str]) -> float:
    value = row.get("ambient_K")
    return float(value) if value not in {None, ""} else 318.15


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
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


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source Superposition Base Comparison",
        "",
        "Matched subset base-map comparison before residual training.",
        "",
        "| Mode | MAE K | RMSE K | Mean-rise MAE K | Centered MAE K |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, metrics in sorted(payload["overall"].items()):
        lines.append(
            f"| {mode} | {metrics['mae_K']:.4f} | {metrics['rmse_K']:.4f} | "
            f"{metrics['mean_rise_abs_error_K']:.4f} | {metrics['centered_field_mae_K']:.4f} |"
        )
    lines.extend(["", "Residual statistics are in `residual_statistics.csv`."])
    (out_dir / "base_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
