#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset


DEFAULT_DATASET_ROOT = REPO_ROOT / "data/runs/benchmarks/dataset_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ChipTherm physics-baseline residual maps.")
    parser.add_argument("--index", default=None, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=DEFAULT_DATASET_ROOT / "residual_analysis", type=Path)
    parser.add_argument("--max-random-per-case", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-png", dest="save_png", action="store_true", default=True)
    parser.add_argument("--no-save-png", dest="save_png", action="store_false")
    parser.add_argument("--save-summary-only", action="store_true")
    args = parser.parse_args()

    index_csv = args.index or (DEFAULT_DATASET_ROOT / f"{args.split}_index.csv")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ChipThermDataset(index_csv, target="residual", return_metadata=True)
    if Path(index_csv).name == "combined_encoded_index.csv":
        dataset.rows = [row for row in dataset.rows if row.get("split") == args.split]
        if not dataset.rows:
            raise SystemExit(f"no rows with split={args.split!r} in {index_csv}")

    analysis = analyze_dataset(dataset)
    selected = select_representative_samples(analysis["sample_metrics"], max_random_per_case=args.max_random_per_case, seed=args.seed)
    analysis["selected_samples"] = selected
    analysis["selection_criteria"] = {
        "best": "lowest per-sample residual MAE",
        "median": "middle per-sample residual MAE after sorting",
        "worst": "highest per-sample residual MAE",
        "random_per_case": f"{args.max_random_per_case} random sample(s) per case using seed {args.seed}",
    }
    summary = build_summary(dataset, analysis, index_csv=index_csv, out_dir=out_dir)

    write_summary(out_dir / "summary.json", summary)
    write_selected_samples(out_dir / "selected_samples.csv", selected)

    generated_figures: list[str] = []
    if args.save_png and not args.save_summary_only:
        generated_figures.extend(write_dataset_figures(out_dir, analysis))
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            sample = dataset[item["dataset_index"]]
            path = samples_dir / f"{sanitize_filename(item['sample_uid'])}_{sanitize_filename(item['criteria'])}.png"
            draw_sample_figure(sample, item, path)
            generated_figures.append(str(path))

    print("Residual analysis complete")
    print(f"Samples: {summary['dataset']['num_samples']}")
    print(
        "Overall residual mean/std/min/max: "
        f"{summary['overall']['mean']:.3f} / {summary['overall']['std']:.3f} / "
        f"{summary['overall']['min']:.3f} / {summary['overall']['max']:.3f} K"
    )
    print(f"Overall MAE/RMSE: {summary['overall']['mae']:.3f} / {summary['overall']['rmse']:.3f} K")
    print("Per-case MAE:")
    for case_id, stats in summary["per_case"].items():
        print(f"  {case_id}: {stats['mae']:.3f} K")
    print("Selected samples:")
    for item in selected:
        print(f"  {item['criteria']}: {item['sample_uid']}")
    if generated_figures:
        print("Generated figures:")
        for path in generated_figures:
            print(f"  {path}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Selected samples CSV: {out_dir / 'selected_samples.csv'}")
    return 0


def analyze_dataset(dataset: ChipThermDataset) -> dict[str, Any]:
    residual_chunks: list[np.ndarray] = []
    sample_metrics: list[dict[str, Any]] = []
    case_acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for index in range(len(dataset)):
        sample = dataset[index]
        residual = sample["residual"].detach().cpu().numpy().astype(np.float64, copy=False)
        temperature = sample["temperature"].detach().cpu().numpy().astype(np.float64, copy=False)
        physics = sample["physics"].detach().cpu().numpy().astype(np.float64, copy=False)
        metadata = sample["metadata"]

        residual_chunks.append(residual.reshape(-1).astype(np.float32, copy=True))
        abs_residual = np.abs(residual)
        mae = float(abs_residual.mean())
        rmse = float(np.sqrt(np.mean(residual * residual)))
        metric = {
            "dataset_index": index,
            "sample_uid": metadata["sample_uid"],
            "original_sample_uid": metadata.get("original_sample_uid"),
            "case_id": metadata["case_id"],
            "dataset_source": metadata["dataset_source"],
            "mae": mae,
            "rmse": rmse,
            "mean_residual": float(residual.mean()),
            "std_residual": float(residual.std()),
            "mean_hotspot_temperature": float(temperature.mean()),
            "mean_physics_temperature": float(physics.mean()),
            "max_abs_error": float(abs_residual.max()),
            "total_power_W": metadata.get("total_power_W"),
            "num_chiplets": metadata.get("num_chiplets"),
        }
        sample_metrics.append(metric)

        case_id = metadata["case_id"]
        acc = case_acc[case_id]
        acc["samples"] += 1.0
        acc["cells"] += float(residual.size)
        acc["sum_residual"] += float(residual.sum())
        acc["sum_residual_sq"] += float(np.sum(residual * residual))
        acc["sum_abs_residual"] += float(abs_residual.sum())
        acc["sum_hotspot_mean"] += float(temperature.mean())
        acc["sum_physics_mean"] += float(physics.mean())

    all_residuals = np.concatenate(residual_chunks).astype(np.float64, copy=False)
    overall = {
        "mean": float(all_residuals.mean()),
        "std": float(all_residuals.std()),
        "min": float(all_residuals.min()),
        "max": float(all_residuals.max()),
        "mae": float(np.abs(all_residuals).mean()),
        "rmse": float(np.sqrt(np.mean(all_residuals * all_residuals))),
        "mean_absolute_value": float(np.abs(all_residuals).mean()),
        "percentiles": {
            "5": float(np.percentile(all_residuals, 5)),
            "25": float(np.percentile(all_residuals, 25)),
            "50": float(np.percentile(all_residuals, 50)),
            "75": float(np.percentile(all_residuals, 75)),
            "95": float(np.percentile(all_residuals, 95)),
        },
    }
    per_case = {}
    for case_id in sorted(case_acc):
        acc = case_acc[case_id]
        cells = acc["cells"]
        samples = acc["samples"]
        mean_residual = acc["sum_residual"] / cells
        mean_sq = acc["sum_residual_sq"] / cells
        per_case[case_id] = {
            "num_samples": int(samples),
            "mae": float(acc["sum_abs_residual"] / cells),
            "rmse": float(math.sqrt(mean_sq)),
            "mean_residual": float(mean_residual),
            "std_residual": float(math.sqrt(max(mean_sq - mean_residual * mean_residual, 0.0))),
            "mean_hotspot_temperature": float(acc["sum_hotspot_mean"] / samples),
            "mean_physics_temperature": float(acc["sum_physics_mean"] / samples),
        }
    return {
        "overall": overall,
        "per_case": per_case,
        "sample_metrics": sample_metrics,
        "all_residuals": all_residuals,
    }


def select_representative_samples(sample_metrics: list[dict[str, Any]], *, max_random_per_case: int, seed: int) -> list[dict[str, Any]]:
    sorted_by_mae = sorted(sample_metrics, key=lambda item: item["mae"])
    selected: dict[str, dict[str, Any]] = {}

    def add(metric: dict[str, Any], criterion: str) -> None:
        key = metric["sample_uid"]
        item = selected.get(key)
        if item is None:
            item = dict(metric)
            item["criteria"] = criterion
            selected[key] = item
        else:
            item["criteria"] = f"{item['criteria']}+{criterion}"

    add(sorted_by_mae[0], "best_mae")
    add(sorted_by_mae[len(sorted_by_mae) // 2], "median_mae")
    add(sorted_by_mae[-1], "worst_mae")

    rng = random.Random(seed)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in sample_metrics:
        by_case[metric["case_id"]].append(metric)
    for case_id in sorted(by_case):
        count = min(max_random_per_case, len(by_case[case_id]))
        for metric in rng.sample(by_case[case_id], count):
            add(metric, f"random_{case_id}")
    return sorted(selected.values(), key=lambda item: (item["criteria"], item["case_id"], item["sample_uid"]))


def build_summary(dataset: ChipThermDataset, analysis: dict[str, Any], *, index_csv: Path, out_dir: Path) -> dict[str, Any]:
    dataset_stats = dataset.statistics()
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_csv": str(Path(index_csv).resolve()),
        "out_dir": str(out_dir),
        "dataset": dataset_stats,
        "overall": analysis["overall"],
        "per_case": analysis["per_case"],
        "selected_samples": [
            {key: value for key, value in item.items() if key != "dataset_index"}
            for item in analysis["selected_samples"]
        ],
        "selection_criteria": analysis["selection_criteria"],
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_selected_samples(path: Path, selected: list[dict[str, Any]]) -> None:
    columns = [
        "criteria",
        "sample_uid",
        "original_sample_uid",
        "case_id",
        "dataset_source",
        "mae",
        "rmse",
        "mean_residual",
        "std_residual",
        "total_power_W",
        "num_chiplets",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for item in selected:
            writer.writerow({column: item.get(column) for column in columns})


def write_dataset_figures(out_dir: Path, analysis: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    residuals = analysis["all_residuals"]
    sample_metrics = analysis["sample_metrics"]
    per_case = analysis["per_case"]

    histogram_path = out_dir / "residual_histogram.png"
    draw_histogram(residuals, histogram_path, title="Residual Histogram: HotSpot - Physics")
    paths.append(str(histogram_path))

    mae_path = out_dir / "mae_by_case.png"
    draw_bar_chart({case: stats["mae"] for case, stats in per_case.items()}, mae_path, title="Residual MAE by Case", ylabel="MAE (K)")
    paths.append(str(mae_path))

    rmse_path = out_dir / "rmse_by_case.png"
    draw_bar_chart({case: stats["rmse"] for case, stats in per_case.items()}, rmse_path, title="Residual RMSE by Case", ylabel="RMSE (K)")
    paths.append(str(rmse_path))

    scatter_path = out_dir / "physics_vs_hotspot_mean.png"
    draw_scatter(
        [(item["mean_physics_temperature"], item["mean_hotspot_temperature"], item["case_id"]) for item in sample_metrics],
        scatter_path,
        title="Physics Mean Temperature vs HotSpot Mean Temperature",
        xlabel="Physics mean temperature (K)",
        ylabel="HotSpot mean temperature (K)",
    )
    paths.append(str(scatter_path))
    return paths


def draw_histogram(values: np.ndarray, path: Path, *, title: str) -> None:
    hist, edges = np.histogram(values, bins=100)
    image = new_canvas(1100, 720)
    draw = ImageDraw.Draw(image)
    font = default_font()
    plot = (90, 90, 1040, 610)
    draw_title(draw, title, image.width)
    draw_axes(draw, plot)
    max_count = max(int(hist.max()), 1)
    width = plot[2] - plot[0]
    height = plot[3] - plot[1]
    for i, count in enumerate(hist):
        x0 = plot[0] + int(i * width / len(hist))
        x1 = plot[0] + int((i + 1) * width / len(hist)) - 1
        y0 = plot[3] - int(count * height / max_count)
        draw.rectangle((x0, y0, x1, plot[3]), fill=(76, 114, 176))
    draw.text((plot[0], plot[3] + 20), f"{edges[0]:.1f} K", fill=(20, 20, 20), font=font)
    draw.text((plot[2] - 80, plot[3] + 20), f"{edges[-1]:.1f} K", fill=(20, 20, 20), font=font)
    draw.text((plot[0], plot[1] - 28), f"max bin count: {max_count}", fill=(20, 20, 20), font=font)
    image.save(path)


def draw_bar_chart(values: dict[str, float], path: Path, *, title: str, ylabel: str) -> None:
    image = new_canvas(1100, 720)
    draw = ImageDraw.Draw(image)
    font = default_font()
    plot = (90, 90, 1040, 610)
    draw_title(draw, title, image.width)
    draw_axes(draw, plot)
    labels = list(values)
    vals = [float(values[label]) for label in labels]
    max_value = max(vals) if vals else 1.0
    bar_gap = 12
    bar_width = max(12, int((plot[2] - plot[0] - bar_gap * (len(vals) + 1)) / max(len(vals), 1)))
    for i, (label, value) in enumerate(zip(labels, vals)):
        x0 = plot[0] + bar_gap + i * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y0 = plot[3] - int((value / max_value) * (plot[3] - plot[1]))
        draw.rectangle((x0, y0, x1, plot[3]), fill=(221, 132, 82))
        draw.text((x0, plot[3] + 18), label, fill=(20, 20, 20), font=font)
        draw.text((x0, y0 - 18), f"{value:.1f}", fill=(20, 20, 20), font=font)
    draw.text((18, 90), ylabel, fill=(20, 20, 20), font=font)
    image.save(path)


def draw_scatter(points: list[tuple[float, float, str]], path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    image = new_canvas(1100, 720)
    draw = ImageDraw.Draw(image)
    font = default_font()
    plot = (100, 90, 1040, 610)
    draw_title(draw, title, image.width)
    draw_axes(draw, plot)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = max((xmax - xmin) * 0.05, 1.0)
    pad_y = max((ymax - ymin) * 0.05, 1.0)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    colors = case_colors()
    for x, y, case_id in points:
        px = scale_value(x, xmin, xmax, plot[0], plot[2])
        py = scale_value(y, ymin, ymax, plot[3], plot[1])
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=colors.get(case_id, (80, 80, 80)))
    draw.text((plot[0], plot[3] + 28), xlabel, fill=(20, 20, 20), font=font)
    draw.text((18, 90), ylabel, fill=(20, 20, 20), font=font)
    draw.text((plot[0], plot[3] + 8), f"{xmin:.1f}", fill=(20, 20, 20), font=font)
    draw.text((plot[2] - 55, plot[3] + 8), f"{xmax:.1f}", fill=(20, 20, 20), font=font)
    draw.text((plot[0] - 72, plot[3] - 8), f"{ymin:.1f}", fill=(20, 20, 20), font=font)
    draw.text((plot[0] - 72, plot[1] - 8), f"{ymax:.1f}", fill=(20, 20, 20), font=font)
    image.save(path)


def draw_sample_figure(sample: dict[str, Any], item: dict[str, Any], path: Path) -> None:
    x = sample["x"].detach().cpu().numpy()
    temperature = sample["temperature"].detach().cpu().numpy()
    physics = sample["physics"].detach().cpu().numpy()
    residual = sample["residual"].detach().cpu().numpy()
    abs_error = np.abs(residual)
    power = x[0]

    temp_min = float(min(temperature.min(), physics.min()))
    temp_max = float(max(temperature.max(), physics.max()))
    resid_abs = float(max(abs(residual.min()), abs(residual.max()), 1.0))
    abs_max = float(max(abs_error.max(), 1.0))

    panels = [
        ("Power density input", power, (float(power.min()), float(power.max())), "power"),
        ("HotSpot temperature", temperature, (temp_min, temp_max), "thermal"),
        ("Physics baseline", physics, (temp_min, temp_max), "thermal"),
        ("Residual", residual, (-resid_abs, resid_abs), "diverging"),
        ("Absolute error", abs_error, (0.0, abs_max), "error"),
    ]

    panel_w = 250
    panel_h = 310
    margin_x = 24
    header_h = 92
    image = new_canvas(margin_x * 2 + panel_w * len(panels), header_h + panel_h + 40)
    draw = ImageDraw.Draw(image)
    font = default_font()
    title = (
        f"{item['sample_uid']} | {item['case_id']} | "
        f"MAE {item['mae']:.2f} K | RMSE {item['rmse']:.2f} K | "
        f"Power {float(item['total_power_W']):.1f} W"
    )
    draw.text((margin_x, 20), title, fill=(20, 20, 20), font=font)

    for i, (name, array, limits, cmap) in enumerate(panels):
        x0 = margin_x + i * panel_w
        y0 = header_h
        draw.text((x0, y0 - 24), name, fill=(20, 20, 20), font=font)
        heatmap = array_to_image(array, vmin=limits[0], vmax=limits[1], cmap=cmap).resize((205, 205), Image.Resampling.BILINEAR)
        image.paste(heatmap, (x0, y0))
        draw_colorbar(draw, image, (x0 + 214, y0, x0 + 234, y0 + 205), limits, cmap)
    image.save(path)


def array_to_image(array: np.ndarray, *, vmin: float, vmax: float, cmap: str) -> Image.Image:
    arr = np.asarray(array, dtype=np.float64)
    if vmax <= vmin:
        vmax = vmin + 1.0
    t = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = colormap(t, cmap)
    return Image.fromarray(rgb.astype(np.uint8))


def colormap(t: np.ndarray, cmap: str) -> np.ndarray:
    if cmap == "diverging":
        blue = np.array([58, 108, 178], dtype=np.float64)
        white = np.array([246, 246, 246], dtype=np.float64)
        red = np.array([190, 64, 54], dtype=np.float64)
        rgb = np.empty(t.shape + (3,), dtype=np.float64)
        low = t <= 0.5
        rgb[low] = lerp(blue, white, (t[low] / 0.5)[..., None])
        rgb[~low] = lerp(white, red, ((t[~low] - 0.5) / 0.5)[..., None])
        return rgb
    if cmap == "thermal":
        return multi_lerp(t, [(42, 72, 160), (70, 170, 210), (250, 220, 90), (190, 45, 35)])
    if cmap == "error":
        return multi_lerp(t, [(255, 255, 245), (245, 170, 70), (165, 35, 35)])
    return multi_lerp(t, [(245, 245, 245), (230, 190, 80), (180, 55, 35)])


def multi_lerp(t: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    anchors = np.array(colors, dtype=np.float64)
    scaled = np.clip(t, 0.0, 1.0) * (len(colors) - 1)
    idx = np.minimum(np.floor(scaled).astype(int), len(colors) - 2)
    frac = (scaled - idx)[..., None]
    return lerp(anchors[idx], anchors[idx + 1], frac)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def draw_colorbar(draw: ImageDraw.ImageDraw, image: Image.Image, box: tuple[int, int, int, int], limits: tuple[float, float], cmap: str) -> None:
    x0, y0, x1, y1 = box
    values = np.linspace(1.0, 0.0, max(y1 - y0, 1)).reshape(-1, 1)
    bar = Image.fromarray(colormap(values, cmap).astype(np.uint8).repeat(max(x1 - x0, 1), axis=1))
    image.paste(bar, (x0, y0))
    font = default_font()
    draw.rectangle(box, outline=(30, 30, 30), width=1)
    draw.text((x1 + 4, y0 - 4), f"{limits[1]:.1f}", fill=(20, 20, 20), font=font)
    draw.text((x1 + 4, y1 - 10), f"{limits[0]:.1f}", fill=(20, 20, 20), font=font)
    if limits[0] < 0.0 < limits[1]:
        zero_y = int(scale_value(0.0, limits[0], limits[1], y1, y0))
        draw.line((x0, zero_y, x1 + 4, zero_y), fill=(20, 20, 20), width=1)


def new_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(250, 250, 247))


def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int) -> None:
    draw.text((40, 32), title, fill=(20, 20, 20), font=default_font())


def draw_axes(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int]) -> None:
    draw.rectangle(plot, outline=(35, 35, 35), width=2)


def scale_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> int:
    if src_max <= src_min:
        return int((dst_min + dst_max) / 2)
    t = (value - src_min) / (src_max - src_min)
    return int(dst_min + t * (dst_max - dst_min))


def case_colors() -> dict[str, tuple[int, int, int]]:
    palette = [
        (76, 114, 176),
        (221, 132, 82),
        (85, 168, 104),
        (196, 78, 82),
        (129, 114, 179),
        (147, 120, 96),
        (218, 139, 195),
        (140, 140, 140),
        (204, 185, 116),
        (100, 181, 205),
    ]
    return {f"case{i:02d}": color for i, color in enumerate(palette, start=1)}


def default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    raise SystemExit(main())
