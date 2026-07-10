#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

VERSION_LABELS = ("v1", "v2")
HOTSPOT_FRACTIONS = (0.01, 0.05, 0.10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze physics_v1 vs physics_v2 residual target complexity.")
    parser.add_argument(
        "--physics-v1-index",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power/test_index.csv",
        type=Path,
    )
    parser.add_argument(
        "--physics-v2-index",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_physics_v2/package_plus_power/test_index.csv",
        type=Path,
    )
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/physics_v1_vs_v2_residual_analysis", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-random-per-case", default=1, type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    v1_index = args.physics_v1_index.expanduser().resolve()
    v2_index = args.physics_v2_index.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_rows = read_index(v1_index)
    v2_rows = read_index(v2_index)
    v2_by_uid = {row["sample_uid"]: row for row in v2_rows}
    missing = [row["sample_uid"] for row in v1_rows if row["sample_uid"] not in v2_by_uid]
    if missing:
        raise SystemExit(f"physics_v2 index is missing {len(missing)} matched sample_uids; first missing: {missing[0]}")

    records: list[dict[str, Any]] = []
    radial_sum = {"v1": None, "v2": None}
    radial_count = 0
    gradient_hist_values = {"v1": [], "v2": []}

    for v1_row in v1_rows:
        v2_row = v2_by_uid[v1_row["sample_uid"]]
        y = np.load(resolve_path(v1_row["y_path"], v1_index.parent)).astype(np.float32, copy=False)
        p1 = np.load(resolve_path(v1_row["prediction_path"], v1_index.parent)).astype(np.float32, copy=False)
        p2 = np.load(resolve_path(v2_row["prediction_path"], v2_index.parent)).astype(np.float32, copy=False)
        if y.shape != p1.shape or y.shape != p2.shape:
            raise SystemExit(f"{v1_row['sample_uid']} shape mismatch: y={y.shape}, v1={p1.shape}, v2={p2.shape}")

        residuals = {"v1": y - p1, "v2": y - p2}
        record: dict[str, Any] = {
            "sample_uid": v1_row["sample_uid"],
            "case_id": v1_row["case_id"],
            "dataset_source": v1_row.get("dataset_source", ""),
            "hotspot_mean_K": float(y.mean()),
            "hotspot_max_K": float(y.max()),
            "total_power_W": optional_float(v1_row.get("total_power_W")),
        }
        for label, residual in residuals.items():
            metrics = residual_complexity_metrics(residual, y)
            for key, value in metrics.items():
                if key == "radial_spectrum":
                    spectrum = np.asarray(value, dtype=np.float64)
                    radial_sum[label] = spectrum.copy() if radial_sum[label] is None else radial_sum[label] + spectrum
                elif key == "gradient_magnitude_values":
                    gradient_hist_values[label].append(np.asarray(value, dtype=np.float32))
                else:
                    record[f"{label}_{key}"] = value
        radial_count += 1
        record["v2_minus_v1_mae_K"] = record["v2_mae_K"] - record["v1_mae_K"]
        record["v2_minus_v1_high_freq_energy_frac"] = record["v2_high_freq_energy_frac"] - record["v1_high_freq_energy_frac"]
        records.append(record)

    radial_average = {
        label: (radial_sum[label] / max(radial_count, 1)).tolist() if radial_sum[label] is not None else []
        for label in VERSION_LABELS
    }
    by_case = aggregate_by_case(records)
    selected = select_representative_samples(records, max_random_per_case=args.max_random_per_case, seed=args.seed)
    summary = build_summary(args, records, by_case, radial_average, selected)

    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "sample_metrics.csv", records)
    write_case_metrics(out_dir / "metrics_by_case.csv", by_case)
    write_csv(out_dir / "selected_samples.csv", selected)

    generated_plots: list[str] = []
    if not args.no_plots:
        generated_plots.extend(write_plots(out_dir, records, by_case, radial_average, gradient_hist_values))
        generated_plots.extend(write_sample_panels(out_dir / "samples", selected, v1_index, v2_index, v2_by_uid))

    print("Physics residual version analysis complete")
    print(f"Samples: {len(records)}")
    print(f"v1 residual MAE/RMSE: {summary['global']['v1']['mae_K']:.3f} / {summary['global']['v1']['rmse_K']:.3f} K")
    print(f"v2 residual MAE/RMSE: {summary['global']['v2']['mae_K']:.3f} / {summary['global']['v2']['rmse_K']:.3f} K")
    print(
        "High-frequency energy fraction v1/v2: "
        f"{summary['global']['v1']['high_freq_energy_frac']:.4f} / "
        f"{summary['global']['v2']['high_freq_energy_frac']:.4f}"
    )
    if "case02" in summary["per_case"]:
        case02 = summary["per_case"]["case02"]
        print(f"case02 v1/v2 residual MAE: {case02['v1']['mae_K']:.3f} / {case02['v2']['mae_K']:.3f} K")
    print("Selected samples:")
    for item in selected:
        print(f"  {item['selection']}: {item['sample_uid']} ({item['case_id']})")
    if generated_plots:
        print("Generated plots:")
        for path in generated_plots:
            print(f"  {path}")
    print(f"Output: {out_dir}")
    return 0


def residual_complexity_metrics(residual: np.ndarray, temperature: np.ndarray) -> dict[str, Any]:
    residual64 = residual.astype(np.float64, copy=False)
    abs_residual = np.abs(residual64)
    gx, gy, grad_mag = gradients(residual64)
    lap = laplacian(residual64)
    spectrum, low, mid, high = radial_fft_spectrum_and_bands(residual64)
    corr_len = estimate_correlation_length_cells(residual64)
    metrics: dict[str, Any] = {
        "mae_K": float(abs_residual.mean()),
        "rmse_K": float(np.sqrt(np.mean(residual64 * residual64))),
        "mean_signed_K": float(residual64.mean()),
        "std_K": float(residual64.std()),
        "max_abs_K": float(abs_residual.max()),
        "avg_abs_grad_x_K_per_cell": float(np.abs(gx).mean()),
        "avg_abs_grad_y_K_per_cell": float(np.abs(gy).mean()),
        "total_gradient_magnitude": float(grad_mag.sum()),
        "mean_gradient_magnitude_K_per_cell": float(grad_mag.mean()),
        "avg_abs_laplacian_K_per_cell2": float(np.abs(lap).mean()),
        "low_freq_energy_frac": low,
        "mid_freq_energy_frac": mid,
        "high_freq_energy_frac": high,
        "estimated_corr_length_cells": corr_len,
        "radial_spectrum": spectrum,
        "gradient_magnitude_values": grad_mag.reshape(-1),
    }
    for fraction in HOTSPOT_FRACTIONS:
        mask = top_fraction_mask(temperature, fraction)
        key = f"hotspot_top_{int(round(fraction * 100))}pct_residual_mae_K"
        metrics[key] = masked_mae(abs_residual, mask)
    return metrics


def gradients(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.gradient(array)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, grad_mag


def laplacian(array: np.ndarray) -> np.ndarray:
    padded = np.pad(array, 1, mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * array
    )


def radial_fft_spectrum_and_bands(array: np.ndarray, bins: int = 32) -> tuple[np.ndarray, float, float, float]:
    centered = array.astype(np.float64, copy=False) - float(array.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    rows, cols = array.shape
    yy = np.arange(rows) - rows // 2
    xx = np.arange(cols) - cols // 2
    radius = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    radius_norm = radius / max(float(radius.max()), 1.0)
    total = float(power.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float64), 0.0, 0.0, 0.0
    radial = np.zeros(bins, dtype=np.float64)
    bin_ids = np.minimum((radius_norm * bins).astype(int), bins - 1)
    for bin_index in range(bins):
        radial[bin_index] = float(power[bin_ids == bin_index].sum()) / total
    low = float(power[radius_norm < 0.15].sum() / total)
    mid = float(power[(radius_norm >= 0.15) & (radius_norm < 0.35)].sum() / total)
    high = float(power[radius_norm >= 0.35].sum() / total)
    return radial, low, mid, high


def estimate_correlation_length_cells(array: np.ndarray) -> float:
    centered = array.astype(np.float64, copy=False) - float(array.mean())
    denom = float(np.sum(centered * centered))
    if denom <= 0.0:
        return 0.0
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(centered)) ** 2).real / denom
    autocorr = np.fft.fftshift(autocorr)
    rows, cols = array.shape
    yy = np.arange(rows) - rows // 2
    xx = np.arange(cols) - cols // 2
    radius = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    max_radius = int(radius.max())
    threshold = math.exp(-1.0)
    for r in range(1, max_radius + 1):
        mask = (radius >= r - 0.5) & (radius < r + 0.5)
        if np.any(mask) and float(autocorr[mask].mean()) <= threshold:
            return float(r)
    return float(max_radius)


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    flat = values.reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    threshold = np.partition(flat, flat.size - count)[flat.size - count]
    return values >= threshold


def masked_mae(abs_residual: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(abs_residual[mask].mean())


def aggregate_by_case(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    return {case_id: aggregate_records(items) for case_id, items in sorted(grouped.items())}


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    scalar_keys = [
        "mae_K",
        "rmse_K",
        "mean_signed_K",
        "std_K",
        "max_abs_K",
        "avg_abs_grad_x_K_per_cell",
        "avg_abs_grad_y_K_per_cell",
        "total_gradient_magnitude",
        "mean_gradient_magnitude_K_per_cell",
        "avg_abs_laplacian_K_per_cell2",
        "low_freq_energy_frac",
        "mid_freq_energy_frac",
        "high_freq_energy_frac",
        "estimated_corr_length_cells",
        "hotspot_top_1pct_residual_mae_K",
        "hotspot_top_5pct_residual_mae_K",
        "hotspot_top_10pct_residual_mae_K",
    ]
    for label in VERSION_LABELS:
        result[label] = {"num_samples": float(len(records))}
        for key in scalar_keys:
            values = [float(record[f"{label}_{key}"]) for record in records if record.get(f"{label}_{key}") not in {None, ""}]
            if not values:
                continue
            if key == "max_abs_K":
                result[label][key] = float(max(values))
            else:
                result[label][key] = float(sum(values) / len(values))
    return result


def build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    by_case: dict[str, dict[str, dict[str, float]]],
    radial_average: dict[str, list[float]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    global_summary = aggregate_records(records)
    correlations = {}
    for label in VERSION_LABELS:
        correlations[label] = {
            "residual_mae_vs_high_freq_energy": pearson_corr(
                [float(record[f"{label}_mae_K"]) for record in records],
                [float(record[f"{label}_high_freq_energy_frac"]) for record in records],
            ),
            "residual_mae_vs_laplacian": pearson_corr(
                [float(record[f"{label}_mae_K"]) for record in records],
                [float(record[f"{label}_avg_abs_laplacian_K_per_cell2"]) for record in records],
            ),
        }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics_v1_index": str(args.physics_v1_index.expanduser().resolve()),
        "physics_v2_index": str(args.physics_v2_index.expanduser().resolve()),
        "num_samples": len(records),
        "global": global_summary,
        "per_case": by_case,
        "case02": by_case.get("case02"),
        "correlations": correlations,
        "average_radial_fft_spectrum": radial_average,
        "selected_samples": selected,
        "notes": [
            "Residuals are computed as HotSpot temperature minus each physics prediction.",
            "FFT energy bands are computed after subtracting the residual mean.",
            "Low/mid/high radial frequency bands use normalized radius thresholds <0.15, 0.15-0.35, and >=0.35.",
        ],
    }


def select_representative_samples(
    records: list[dict[str, Any]],
    *,
    max_random_per_case: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any], selection: str) -> None:
        existing = selected.get(record["sample_uid"])
        if existing is None:
            selected[record["sample_uid"]] = {
                "selection": selection,
                "sample_uid": record["sample_uid"],
                "case_id": record["case_id"],
                "v1_mae_K": record["v1_mae_K"],
                "v2_mae_K": record["v2_mae_K"],
                "v1_high_freq_energy_frac": record["v1_high_freq_energy_frac"],
                "v2_high_freq_energy_frac": record["v2_high_freq_energy_frac"],
            }
        else:
            existing["selection"] = f"{existing['selection']}+{selection}"

    sorted_v2 = sorted(records, key=lambda item: float(item["v2_mae_K"]))
    add(sorted_v2[len(sorted_v2) // 2], "median_v2_mae")
    add(sorted_v2[-1], "worst_v2_mae")
    case02_records = [record for record in records if record["case_id"] == "case02"]
    if case02_records:
        add(max(case02_records, key=lambda item: float(item["v2_mae_K"])), "worst_case02_v2_mae")

    rng = random.Random(seed)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[str(record["case_id"])].append(record)
    for case_id in sorted(by_case):
        count = min(max_random_per_case, len(by_case[case_id]))
        for record in rng.sample(by_case[case_id], count):
            add(record, f"random_{case_id}")
    return sorted(selected.values(), key=lambda item: (item["selection"], item["case_id"], item["sample_uid"]))


def write_plots(
    out_dir: Path,
    records: list[dict[str, Any]],
    by_case: dict[str, dict[str, dict[str, float]]],
    radial_average: dict[str, list[float]],
    gradient_hist_values: dict[str, list[np.ndarray]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plots")
        return []

    paths: list[str] = []
    bins = np.arange(len(radial_average["v1"]))
    plt.figure(figsize=(7, 4))
    plt.plot(bins, radial_average["v1"], label="physics_v1 residual")
    plt.plot(bins, radial_average["v2"], label="physics_v2 residual")
    plt.xlabel("Radial FFT bin")
    plt.ylabel("Mean energy fraction")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "average_radial_fft_spectrum.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))

    cases = sorted(by_case)
    x = np.arange(len(cases))
    width = 0.38
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, [by_case[case]["v1"]["high_freq_energy_frac"] for case in cases], width, label="v1")
    plt.bar(x + width / 2, [by_case[case]["v2"]["high_freq_energy_frac"] for case in cases], width, label="v2")
    plt.xticks(x, cases, rotation=45)
    plt.ylabel("High-frequency energy fraction")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "high_frequency_energy_by_case.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(6, 5))
    plt.scatter([record["v1_high_freq_energy_frac"] for record in records], [record["v1_mae_K"] for record in records], s=12, alpha=0.55, label="v1")
    plt.scatter([record["v2_high_freq_energy_frac"] for record in records], [record["v2_mae_K"] for record in records], s=12, alpha=0.55, label="v2")
    plt.xlabel("High-frequency energy fraction")
    plt.ylabel("Residual MAE (K)")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "residual_mae_vs_high_frequency_energy.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(7, 4))
    for label in VERSION_LABELS:
        values = np.concatenate(gradient_hist_values[label]) if gradient_hist_values[label] else np.array([])
        if values.size:
            plt.hist(values, bins=80, alpha=0.5, density=True, label=label)
    plt.xlabel("Residual gradient magnitude (K/cell)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "residual_gradient_magnitude_histogram.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))
    return paths


def write_sample_panels(
    samples_dir: Path,
    selected: list[dict[str, Any]],
    v1_index: Path,
    v2_index: Path,
    v2_by_uid: dict[str, dict[str, str]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception:
        print("matplotlib unavailable; skipping sample panels")
        return []

    samples_dir.mkdir(parents=True, exist_ok=True)
    v1_by_uid = {row["sample_uid"]: row for row in read_index(v1_index)}
    paths: list[str] = []
    for item in selected:
        uid = item["sample_uid"]
        v1_row = v1_by_uid[uid]
        v2_row = v2_by_uid[uid]
        y = np.load(resolve_path(v1_row["y_path"], v1_index.parent)).astype(np.float32, copy=False)
        p1 = np.load(resolve_path(v1_row["prediction_path"], v1_index.parent)).astype(np.float32, copy=False)
        p2 = np.load(resolve_path(v2_row["prediction_path"], v2_index.parent)).astype(np.float32, copy=False)
        r1 = y - p1
        r2 = y - p2
        _, _, g1 = gradients(r1)
        _, _, g2 = gradients(r2)

        temp_min = float(min(y.min(), p1.min(), p2.min()))
        temp_max = float(max(y.max(), p1.max(), p2.max()))
        residual_limit = float(max(np.abs(r1).max(), np.abs(r2).max(), 1.0))
        grad_max = float(max(g1.max(), g2.max(), 1.0))
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        panels = [
            ("HotSpot Y", y, "inferno", temp_min, temp_max, None),
            ("Physics v1", p1, "inferno", temp_min, temp_max, None),
            ("Physics v2", p2, "inferno", temp_min, temp_max, None),
            ("v2 - v1 prediction", p2 - p1, "coolwarm", -residual_limit, residual_limit, TwoSlopeNorm(vcenter=0.0, vmin=-residual_limit, vmax=residual_limit)),
            ("Residual v1", r1, "coolwarm", -residual_limit, residual_limit, TwoSlopeNorm(vcenter=0.0, vmin=-residual_limit, vmax=residual_limit)),
            ("Residual v2", r2, "coolwarm", -residual_limit, residual_limit, TwoSlopeNorm(vcenter=0.0, vmin=-residual_limit, vmax=residual_limit)),
            ("|grad residual v1|", g1, "magma", 0.0, grad_max, None),
            ("|grad residual v2|", g2, "magma", 0.0, grad_max, None),
        ]
        for ax, (title, image, cmap, vmin, vmax, norm) in zip(axes.ravel(), panels):
            im = ax.imshow(image, cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{uid} {item['case_id']} | {item['selection']} | "
            f"v1 MAE {float(item['v1_mae_K']):.2f} K, v2 MAE {float(item['v2_mae_K']):.2f} K",
            fontsize=11,
        )
        fig.tight_layout()
        path = samples_dir / f"{sanitize_filename(uid)}_{sanitize_filename(item['selection'])}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows


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
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_case_metrics(path: Path, by_case: dict[str, dict[str, dict[str, float]]]) -> None:
    rows: list[dict[str, Any]] = []
    for case_id, metrics in sorted(by_case.items()):
        row: dict[str, Any] = {"case_id": case_id}
        for label in VERSION_LABELS:
            for key, value in metrics[label].items():
                row[f"{label}_{key}"] = value
        rows.append(row)
    write_csv(path, rows)


def pearson_corr(a: list[float], b: list[float]) -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
