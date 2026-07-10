#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare physics_v1 and physics_v2 predictions on the same samples.")
    parser.add_argument("--physics-v1-index", required=True, type=Path, help="Index whose prediction_path points to physics_v1.")
    parser.add_argument("--physics-v2-index", required=True, type=Path, help="Index whose prediction_path points to physics_v2.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--label-v1", default="physics_v1")
    parser.add_argument("--label-v2", default="physics_v2")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    v1_index = args.physics_v1_index.expanduser().resolve()
    v2_index = args.physics_v2_index.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_rows, _ = read_index(v1_index)
    v2_rows, _ = read_index(v2_index)
    v2_by_uid = {row["sample_uid"]: row for row in v2_rows}
    missing = [row["sample_uid"] for row in v1_rows if row["sample_uid"] not in v2_by_uid]
    if missing:
        raise SystemExit(f"physics_v2 index is missing {len(missing)} sample_uids, first: {missing[0]}")

    rows: list[dict[str, Any]] = []
    metrics_by_version = {
        args.label_v1: MetricAccumulator(),
        args.label_v2: MetricAccumulator(),
    }
    case_metrics: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            args.label_v1: MetricAccumulator(),
            args.label_v2: MetricAccumulator(),
        }
    )
    v1_runtimes: list[float] = []
    v2_runtimes: list[float] = []
    hotspot_runtimes: list[float] = []

    total_start = time.perf_counter()
    for v1_row in v1_rows:
        v2_row = v2_by_uid[v1_row["sample_uid"]]
        y = np.load(resolve_path(v1_row["y_path"], v1_index.parent)).astype(np.float32, copy=False)
        pred_v1 = np.load(resolve_path(v1_row["prediction_path"], v1_index.parent)).astype(np.float32, copy=False)
        pred_v2 = np.load(resolve_path(v2_row["prediction_path"], v2_index.parent)).astype(np.float32, copy=False)
        if pred_v1.shape != y.shape or pred_v2.shape != y.shape:
            raise SystemExit(f"{v1_row['sample_uid']} shape mismatch: y={y.shape}, v1={pred_v1.shape}, v2={pred_v2.shape}")

        item_v1 = sample_metrics(pred_v1, y)
        item_v2 = sample_metrics(pred_v2, y)
        metrics_by_version[args.label_v1].update_from_metrics(item_v1)
        metrics_by_version[args.label_v2].update_from_metrics(item_v2)
        case_id = v1_row["case_id"]
        case_metrics[case_id][args.label_v1].update_from_metrics(item_v1)
        case_metrics[case_id][args.label_v2].update_from_metrics(item_v2)
        v1_runtimes.extend(optional_float_list([v1_row.get("physics_runtime_s")]))
        v2_runtimes.extend(optional_float_list([v2_row.get("physics_runtime_s")]))
        hotspot_runtimes.extend(optional_float_list([v1_row.get("hotspot_runtime_s")]))
        rows.append(
            {
                "sample_uid": v1_row["sample_uid"],
                "case_id": case_id,
                "v1_mae_K": item_v1["mae_K"],
                "v2_mae_K": item_v2["mae_K"],
                "v1_rmse_K": item_v1["rmse_K"],
                "v2_rmse_K": item_v2["rmse_K"],
                "mae_improvement_percent": percent_improvement(item_v1["mae_K"], item_v2["mae_K"]),
                "rmse_improvement_percent": percent_improvement(item_v1["rmse_K"], item_v2["rmse_K"]),
            }
        )
    total_runtime_s = time.perf_counter() - total_start

    global_metrics = {
        label: accumulator.compute()
        for label, accumulator in metrics_by_version.items()
    }
    by_case_payload = {
        case_id: {
            label: accumulator.compute()
            for label, accumulator in sorted(version_accs.items())
        }
        for case_id, version_accs in sorted(case_metrics.items())
    }
    avg_hotspot_runtime = mean_or_none(hotspot_runtimes)
    avg_v1_runtime = mean_or_none(v1_runtimes)
    avg_v2_runtime = mean_or_none(v2_runtimes)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics_v1_index": str(v1_index),
        "physics_v2_index": str(v2_index),
        "num_samples": len(v1_rows),
        "total_eval_runtime_s": total_runtime_s,
        "labels": {
            "v1": args.label_v1,
            "v2": args.label_v2,
        },
        "runtime": {
            "hotspot_runtime_reference_s": avg_hotspot_runtime,
            f"{args.label_v1}_runtime_per_sample_s": avg_v1_runtime,
            f"{args.label_v2}_runtime_per_sample_s": avg_v2_runtime,
            f"{args.label_v1}_speedup_vs_hotspot": avg_hotspot_runtime / avg_v1_runtime if avg_hotspot_runtime and avg_v1_runtime else None,
            f"{args.label_v2}_speedup_vs_hotspot": avg_hotspot_runtime / avg_v2_runtime if avg_hotspot_runtime and avg_v2_runtime else None,
            "timing_note": "Runtimes are read from index metadata. Evaluation disk I/O is not included as inference time.",
        },
        "global": global_metrics,
        "improvement_v2_over_v1": {
            "mae_percent": percent_improvement(global_metrics[args.label_v1]["mae_K"], global_metrics[args.label_v2]["mae_K"]),
            "rmse_percent": percent_improvement(global_metrics[args.label_v1]["rmse_K"], global_metrics[args.label_v2]["rmse_K"]),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case_payload, args.label_v1, args.label_v2)
    write_sample_metrics(out_dir / "sample_metrics.csv", rows)
    if args.save_plots:
        write_plots(out_dir, rows, by_case_payload, args.label_v1, args.label_v2)

    print("Physics version comparison complete")
    print(f"Samples: {len(v1_rows)}")
    print(f"{args.label_v1} MAE/RMSE: {global_metrics[args.label_v1]['mae_K']:.3f} / {global_metrics[args.label_v1]['rmse_K']:.3f} K")
    print(f"{args.label_v2} MAE/RMSE: {global_metrics[args.label_v2]['mae_K']:.3f} / {global_metrics[args.label_v2]['rmse_K']:.3f} K")
    print(f"Improvement: MAE {summary['improvement_v2_over_v1']['mae_percent']:.2f}% / RMSE {summary['improvement_v2_over_v1']['rmse_percent']:.2f}%")
    print(f"{args.label_v1} runtime/sample: {avg_v1_runtime:.6f} s" if avg_v1_runtime else f"{args.label_v1} runtime/sample: n/a")
    print(f"{args.label_v2} runtime/sample: {avg_v2_runtime:.6f} s" if avg_v2_runtime else f"{args.label_v2} runtime/sample: n/a")
    print(f"Output: {out_dir}")
    return 0


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.mae_sum = 0.0
        self.rmse_sq_sum = 0.0
        self.max_abs = 0.0
        self.mean_signed_sum = 0.0
        self.hotspot_temp_error_sum = 0.0
        self.hotspot_location_error_sum = 0.0

    def update_from_metrics(self, metrics: dict[str, float]) -> None:
        self.count += 1
        self.mae_sum += float(metrics["mae_K"])
        self.rmse_sq_sum += float(metrics["rmse_K"]) ** 2
        self.max_abs = max(self.max_abs, float(metrics["max_abs_error_K"]))
        self.mean_signed_sum += float(metrics["mean_signed_error_K"])
        self.hotspot_temp_error_sum += float(metrics["hotspot_temp_error_K"])
        self.hotspot_location_error_sum += float(metrics["hotspot_location_error_cells"])

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "num_samples": float(self.count),
            "mae_K": self.mae_sum / self.count,
            "rmse_K": (self.rmse_sq_sum / self.count) ** 0.5,
            "max_abs_error_K": self.max_abs,
            "mean_signed_error_K": self.mean_signed_sum / self.count,
            "hotspot_temp_error_K": self.hotspot_temp_error_sum / self.count,
            "hotspot_location_error_cells": self.hotspot_location_error_sum / self.count,
        }


def sample_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - target.astype(np.float64)
    abs_error = np.abs(error)
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    target_hotspot = np.unravel_index(int(np.argmax(target)), target.shape)
    row_error = float(pred_hotspot[0] - target_hotspot[0])
    col_error = float(pred_hotspot[1] - target_hotspot[1])
    return {
        "mae_K": float(abs_error.mean()),
        "rmse_K": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_K": float(abs_error.max()),
        "mean_signed_error_K": float(error.mean()),
        "hotspot_temp_error_K": float(pred[pred_hotspot] - target[target_hotspot]),
        "hotspot_location_error_cells": float(np.hypot(row_error, col_error)),
    }


def read_index(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise SystemExit(f"{path} has no header")
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows, fieldnames


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path, base / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_case_metrics(path: Path, by_case: dict[str, dict[str, dict[str, float]]], label_v1: str, label_v2: str) -> None:
    columns = [
        "case_id",
        f"{label_v1}_mae_K",
        f"{label_v1}_rmse_K",
        f"{label_v1}_mean_signed_error_K",
        f"{label_v1}_hotspot_location_error_cells",
        f"{label_v2}_mae_K",
        f"{label_v2}_rmse_K",
        f"{label_v2}_mean_signed_error_K",
        f"{label_v2}_hotspot_location_error_cells",
        "mae_improvement_percent",
        "rmse_improvement_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(by_case.items()):
            v1 = metrics[label_v1]
            v2 = metrics[label_v2]
            writer.writerow(
                {
                    "case_id": case_id,
                    f"{label_v1}_mae_K": v1["mae_K"],
                    f"{label_v1}_rmse_K": v1["rmse_K"],
                    f"{label_v1}_mean_signed_error_K": v1["mean_signed_error_K"],
                    f"{label_v1}_hotspot_location_error_cells": v1["hotspot_location_error_cells"],
                    f"{label_v2}_mae_K": v2["mae_K"],
                    f"{label_v2}_rmse_K": v2["rmse_K"],
                    f"{label_v2}_mean_signed_error_K": v2["mean_signed_error_K"],
                    f"{label_v2}_hotspot_location_error_cells": v2["hotspot_location_error_cells"],
                    "mae_improvement_percent": percent_improvement(v1["mae_K"], v2["mae_K"]),
                    "rmse_improvement_percent": percent_improvement(v1["rmse_K"], v2["rmse_K"]),
                }
            )


def write_sample_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(out_dir: Path, rows: list[dict[str, Any]], by_case: dict[str, dict[str, dict[str, float]]], label_v1: str, label_v2: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plots")
        return
    v1_mae = [float(row["v1_mae_K"]) for row in rows]
    v2_mae = [float(row["v2_mae_K"]) for row in rows]
    plt.figure(figsize=(6, 5))
    plt.scatter(v1_mae, v2_mae, s=10, alpha=0.6)
    limit = max(max(v1_mae), max(v2_mae))
    plt.plot([0, limit], [0, limit], "k--", linewidth=1)
    plt.xlabel(f"{label_v1} sample MAE (K)")
    plt.ylabel(f"{label_v2} sample MAE (K)")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_mae_v1_vs_v2.png", dpi=160)
    plt.close()

    cases = sorted(by_case)
    x = np.arange(len(cases))
    width = 0.38
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, [by_case[case][label_v1]["mae_K"] for case in cases], width, label=label_v1)
    plt.bar(x + width / 2, [by_case[case][label_v2]["mae_K"] for case in cases], width, label=label_v2)
    plt.xticks(x, cases, rotation=45)
    plt.ylabel("MAE (K)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_case_v1_vs_v2.png", dpi=160)
    plt.close()


def optional_float_list(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        result.append(float(value))
    return result


def mean_or_none(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def percent_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0
    return float((baseline - candidate) / baseline * 100.0)


if __name__ == "__main__":
    raise SystemExit(main())
