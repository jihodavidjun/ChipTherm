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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_superposition_diagnostic import chiplet_metrics, field_metrics, load_json  # noqa: E402


SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full canonical source-superposition base-map quality.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary-only", action="store_true", help="Regenerate summary/report from existing by-sample CSV if present.")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = out_dir / "base_quality_by_sample.csv"
    if args.summary_only:
        if not sample_csv.exists():
            raise SystemExit(f"--summary-only requested but {sample_csv} does not exist")
        records = read_rows(sample_csv)
    else:
        records = []
        for split in SPLITS:
            index_path = source_root / f"{split}_index.csv"
            for row in read_rows(index_path):
                target = np.load(resolve_path(row["y_path"], index_path.parent)).astype(np.float64)
                base = np.load(resolve_path(row["source_superposition_base_path"], index_path.parent)).astype(np.float64)
                records.append(base_metrics(row, split, base, target))
        write_csv(sample_csv, records)

    by_case = summarize_by_case(records)
    overall = summarize_overall(records)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "overall": overall,
    }
    (out_dir / "base_quality_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "base_quality_by_case.csv", by_case)
    write_report(out_dir, payload)
    print("Full source-superposition base-quality evaluation complete")
    for split, metrics in sorted(overall.items()):
        print(f"{split}: MAE={metrics['mae_K']:.4f} RMSE={metrics['rmse_K']:.4f} centered={metrics['centered_field_mae_K']:.4f}")
    print(f"Output: {out_dir}")
    return 0


def base_metrics(row: dict[str, str], split: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    metrics = field_metrics(pred, target)
    ambient = ambient_for_row(row)
    delta_pred = pred - ambient
    delta_true = target - ambient
    centered_pred = delta_pred - np.mean(delta_pred)
    centered_true = delta_true - np.mean(delta_true)
    centered_error = centered_pred - centered_true
    hotspot_pred = np.unravel_index(int(np.argmax(pred)), pred.shape)
    hotspot_true = np.unravel_index(int(np.argmax(target)), target.shape)
    chip = {}
    layout_path = source_layout_path(row)
    if layout_path.exists():
        chip = chiplet_metrics(pred, target, load_json(layout_path), target.shape)
    return {
        "split": split,
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "mae_K": metrics["mae_K"],
        "rmse_K": metrics["rmse_K"],
        "max_abs_error_K": metrics["max_abs_error_K"],
        "mean_signed_error_K": metrics["mean_signed_error_K"],
        "mean_rise_abs_error_K": abs(float(np.mean(delta_pred) - np.mean(delta_true))),
        "centered_field_mae_K": float(np.mean(np.abs(centered_error))),
        "centered_field_rmse_K": float(np.sqrt(np.mean(centered_error * centered_error))),
        "chiplet_mean_temperature_mae_K": chip.get("chiplet_mean_temperature_mae_K"),
        "chiplet_peak_temperature_mae_K": chip.get("chiplet_peak_temperature_mae_K"),
        "inter_chiplet_delta_T_mae_K": chip.get("inter_chiplet_delta_T_mae_K"),
        "hotspot_temp_error_K": float(pred[hotspot_pred] - target[hotspot_true]),
        "hotspot_location_error_cells": float(np.hypot(hotspot_pred[0] - hotspot_true[0], hotspot_pred[1] - hotspot_true[1])),
        "source_superposition_base_path": row.get("source_superposition_base_path", ""),
    }


def summarize_overall(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["split"])].append(record)
        grouped["all"].append(record)
    return {name: summarize_records(items) for name, items in sorted(grouped.items())}


def summarize_by_case(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["split"]), str(record["case_id"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (split, case_id), items in sorted(grouped.items()):
        summary = summarize_records(items)
        summary.update({"split": split, "case_id": case_id})
        rows.append(summary)
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
    output: dict[str, Any] = {"num_samples": len(records)}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) not in {None, ""}]
        output[key] = float(np.mean(values)) if values else None
    return output


def source_layout_path(row: dict[str, str]) -> Path:
    for key in ("source_layout_path", "layout_path"):
        if row.get(key):
            return resolve_path(row[key])
    case_id = row["case_id"]
    original = row.get("original_sample_uid") or row["sample_uid"]
    sample_name = original[len(case_id) + 1 :] if original.startswith(f"{case_id}_") else original
    return REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_name / "source" / "layout.json"


def ambient_for_row(row: dict[str, str]) -> float:
    value = row.get("ambient_K")
    return float(value) if value not in {None, ""} else 318.15


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def resolve_path(path_value: str | Path, base: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path]
    if base is not None:
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Full Source-Superposition Base Quality",
        "",
        "| Split | Samples | MAE K | RMSE K | Mean-rise MAE K | Centered MAE K | Hotspot loc err cells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in sorted(payload["overall"].items()):
        lines.append(
            f"| {split} | {metrics['num_samples']} | {metrics['mae_K']:.4f} | {metrics['rmse_K']:.4f} | "
            f"{metrics['mean_rise_abs_error_K']:.4f} | {metrics['centered_field_mae_K']:.4f} | "
            f"{metrics['hotspot_location_error_cells']:.4f} |"
        )
    lines.append("")
    (out_dir / "base_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
