#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
    "primary_test_families",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare direct-temperature, source-only, and source-plus-residual Benchmark v2 results."
    )
    parser.add_argument("--direct-eval-root", required=True, type=Path)
    parser.add_argument("--source-baseline-dir", default=None, type=Path)
    parser.add_argument("--residual-eval-root", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_root = args.direct_eval_root.expanduser().resolve()
    source_by_uid = load_source_rows(args.source_baseline_dir)
    headline: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        direct_dir = direct_root / protocol
        direct_metrics = load_json(direct_dir / "metrics.json")
        direct_samples = read_csv(direct_dir / "metrics_by_sample.csv")
        ensure_direct_checkpoint_metrics(direct_metrics)
        direct_summary = model_summary("direct_temperature_feature_fusion", protocol, direct_samples)
        direct_final = direct_metrics.get("cnn_final_temperature", {})
        direct_summary["mae_K"] = direct_final.get("mae_K", direct_summary["mae_K"])
        direct_summary["rmse_K"] = direct_final.get("rmse_K", direct_summary["rmse_K"])
        direct_summary["max_abs_error_K"] = direct_final.get(
            "max_abs_error_K", direct_summary["max_abs_error_K"]
        )
        direct_summary["parameter_count"] = direct_metrics.get("model", {}).get("parameter_count")
        direct_summary["inference_runtime_per_sample_s"] = direct_metrics.get(
            "inference_runtime_per_sample_s"
        )
        direct_summary["worse_than_source_fraction"] = direct_metrics.get(
            "worse_than_physics_baseline_fraction"
        )
        headline.append(direct_summary)

        source_samples = [source_by_uid[row["sample_uid"]] for row in direct_samples if row["sample_uid"] in source_by_uid]
        if source_by_uid:
            if len(source_samples) != len(direct_samples):
                raise ValueError(
                    f"source baseline alignment failed for {protocol}: "
                    f"{len(source_samples)}/{len(direct_samples)}"
                )
            headline.append(model_summary("source_superposition_only", protocol, source_samples))

        residual_samples: list[dict[str, str]] = []
        if args.residual_eval_root is not None:
            residual_path = (
                args.residual_eval_root.expanduser().resolve()
                / protocol
                / "metrics_by_sample.csv"
            )
            residual_samples = read_csv(residual_path)
            headline.append(
                model_summary("source_superposition_plus_residual_feature_fusion", protocol, residual_samples)
            )
        family_rows.extend(
            compare_by_family(
                protocol,
                direct_samples,
                source_samples=source_samples,
                residual_samples=residual_samples,
            )
        )

    write_csv(out_dir / "direct_temperature_baseline_metrics.csv", headline)
    write_csv(out_dir / "direct_vs_decomposed_by_family.csv", family_rows)
    write_report(out_dir / "direct_vs_decomposed_report.md", headline, family_rows)
    write_plots(out_dir, headline, family_rows)
    print(f"Direct-temperature comparison report: {out_dir}")
    return 0


def ensure_direct_checkpoint_metrics(metrics: dict[str, Any]) -> None:
    mode = metrics.get("model", {}).get("prediction_mode")
    if mode not in {"direct_temperature", "direct_temperature_source_conditioned"}:
        raise ValueError(f"expected direct-temperature evaluation metrics, found prediction_mode={mode!r}")


def load_source_rows(root: Path | None) -> dict[str, dict[str, str]]:
    if root is None:
        return {}
    path = root.expanduser().resolve() / "base_quality_by_sample.csv"
    return {str(row["sample_uid"]): row for row in read_csv(path)}


def model_summary(model: str, protocol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "protocol": protocol,
        "num_samples": len(rows),
        "mae_K": mean(rows, ("mae_K",)),
        "rmse_K": rms(rows, ("rmse_K",)),
        "peak_temperature_mae_K": mean(
            rows, ("peak_temperature_abs_error_K", "hotspot_temp_error_K")
        ),
        "max_abs_error_K": maximum(rows, ("max_abs_error_K",)),
        "hotspot_top1pct_mae_K": mean(rows, ("hotspot_top1pct_mae_K",)),
        "occupied_region_mae_K": mean(rows, ("occupied_region_mae_K",)),
        "unoccupied_region_mae_K": mean(rows, ("unoccupied_region_mae_K",)),
        "boundary_region_mae_K": mean(rows, ("boundary_region_mae_K",)),
        "non_boundary_region_mae_K": mean(rows, ("non_boundary_region_mae_K",)),
    }


def compare_by_family(
    protocol: str,
    direct_rows: list[dict[str, str]],
    *,
    source_samples: list[dict[str, str]],
    residual_samples: list[dict[str, str]],
) -> list[dict[str, Any]]:
    direct = group_by_family(direct_rows)
    source = group_by_family(source_samples)
    residual = group_by_family(residual_samples)
    output = []
    for family in sorted(direct):
        direct_mae = mean(direct[family], ("mae_K",))
        source_mae = mean(source.get(family, []), ("mae_K",))
        residual_mae = mean(residual.get(family, []), ("mae_K",))
        output.append(
            {
                "protocol": protocol,
                "family_uid": family,
                "num_samples": len(direct[family]),
                "source_superposition_mae_K": source_mae,
                "direct_temperature_mae_K": direct_mae,
                "source_plus_residual_mae_K": residual_mae,
                "direct_minus_source_mae_K": difference(direct_mae, source_mae),
                "direct_minus_residual_mae_K": difference(direct_mae, residual_mae),
            }
        )
    return output


def group_by_family(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family_uid") or row.get("case_id") or "")
        if family:
            output[family].append(row)
    return output


def difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def mean(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
        if values:
            return float(np.mean(np.asarray(values, dtype=np.float64)))
    return None


def maximum(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
        if values:
            return max(values)
    return None


def rms(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
        if values:
            array = np.asarray(values, dtype=np.float64)
            return float(np.sqrt(np.mean(array * array)))
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty comparison table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    headline: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Direct Temperature vs Source-Superposition Residual",
        "",
        "The direct CNN receives no source-superposition map and emits an absolute-temperature map.",
        "Source-superposition values in this report are comparison metrics only.",
        "",
        "| Model | Protocol | N | MAE K | RMSE K | Peak MAE K |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f"| {row['model']} | {row['protocol']} | {row['num_samples']} | "
            f"{fmt(row.get('mae_K'))} | {fmt(row.get('rmse_K'))} | "
            f"{fmt(row.get('peak_temperature_mae_K'))} |"
        )
    lines.extend(
        [
            "",
            "Positive `direct_minus_residual_mae_K` means the decomposed source-plus-residual model is better.",
            "",
            "| Protocol | Family | Direct MAE K | Source MAE K | Residual MAE K | Direct - residual K |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| {row['protocol']} | {row['family_uid']} | "
            f"{fmt(row.get('direct_temperature_mae_K'))} | "
            f"{fmt(row.get('source_superposition_mae_K'))} | "
            f"{fmt(row.get('source_plus_residual_mae_K'))} | "
            f"{fmt(row.get('direct_minus_residual_mae_K'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(
    out_dir: Path,
    headline: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        write_plots_pillow(out_dir, headline, family_rows)
        return
    protocol = "primary_test_families"
    selected = [row for row in headline if row["protocol"] == protocol]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([row["model"] for row in selected], [row["mae_K"] for row in selected])
    ax.set_ylabel("MAE (K)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "direct_vs_decomposed_mae.png", dpi=160)
    plt.close(fig)

    selected_families = [row for row in family_rows if row["protocol"] == protocol]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(selected_families))
    ax.plot(x, [row["direct_temperature_mae_K"] for row in selected_families], marker="o", label="direct")
    if any(row["source_plus_residual_mae_K"] is not None for row in selected_families):
        ax.plot(x, [row["source_plus_residual_mae_K"] for row in selected_families], marker="o", label="source + residual")
    ax.set_xticks(x, [row["family_uid"] for row in selected_families])
    ax.set_ylabel("MAE (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_family_direct_vs_decomposed.png", dpi=160)
    plt.close(fig)


def write_plots_pillow(
    out_dir: Path,
    headline: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw

    protocol = "primary_test_families"
    selected = [row for row in headline if row["protocol"] == protocol]
    draw_bar_png(
        out_dir / "direct_vs_decomposed_mae.png",
        [(str(row["model"]), float(row["mae_K"])) for row in selected],
        Image,
        ImageDraw,
    )
    selected_families = [row for row in family_rows if row["protocol"] == protocol]
    values = [
        (str(row["family_uid"]), float(row["direct_temperature_mae_K"]))
        for row in selected_families
    ]
    draw_bar_png(out_dir / "per_family_direct_vs_decomposed.png", values, Image, ImageDraw)


def draw_bar_png(path: Path, values: list[tuple[str, float]], image_module: Any, draw_module: Any) -> None:
    image = image_module.new("RGB", (1000, 560), "white")
    draw = draw_module.Draw(image)
    draw.text((30, 20), path.stem.replace("_", " "), fill="black")
    maximum = max((value for _, value in values), default=1.0)
    width = 900 / max(len(values), 1)
    for index, (label, value) in enumerate(values):
        x0 = 60 + index * width
        x1 = x0 + width * 0.72
        height = value / max(maximum, 1.0e-12) * 400
        draw.rectangle((x0, 480 - height, x1, 480), fill=(48, 112, 160))
        draw.text((x0, 490), label[:18], fill="black")
    image.save(path)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
