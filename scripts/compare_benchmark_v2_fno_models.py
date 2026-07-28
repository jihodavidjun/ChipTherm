#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
    "primary_test_families",
)

MODEL_ARGUMENTS = {
    "direct_cnn": "direct_cnn_root",
    "direct_fno": "direct_fno_root",
    "direct_ufno": "direct_ufno_root",
    "residual_cnn": "residual_cnn_root",
    "residual_fno": "residual_fno_root",
    "residual_ufno": "residual_ufno_root",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the controlled Benchmark v2 CNN/FNO 2x2.")
    parser.add_argument("--direct-cnn-root", type=Path)
    parser.add_argument("--direct-fno-root", type=Path)
    parser.add_argument("--direct-ufno-root", type=Path)
    parser.add_argument("--source-only-root", type=Path)
    parser.add_argument("--residual-cnn-root", type=Path)
    parser.add_argument("--residual-fno-root", type=Path)
    parser.add_argument("--residual-ufno-root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = {
        name: getattr(args, argument).expanduser().resolve()
        for name, argument in MODEL_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }
    if not roots:
        raise SystemExit("provide at least one explicit model evaluation root")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    headline, families = aggregate_comparison(roots)
    if args.source_only_root is not None:
        source_by_uid = load_source_baseline(args.source_only_root.expanduser().resolve())
        anchor_root = roots.get("direct_fno") or roots.get("direct_cnn") or next(iter(roots.values()))
        source_headline, source_families = align_source_baseline(source_by_uid, anchor_root)
        headline.extend(source_headline)
        families.extend(source_families)
    write_csv(out_dir / "fno_model_comparison.csv", headline)
    write_csv(out_dir / "fno_model_comparison_by_family.csv", families)
    write_report(out_dir / "fno_comparison_report.md", headline, families)
    write_plots(out_dir, headline, families)
    print(f"FNO comparison: {out_dir}")
    return 0


def aggregate_comparison(
    roots: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    headline: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    expected_modes = {
        "direct_cnn": {"direct_temperature", ""},
        "direct_fno": {"direct_temperature_fno"},
        "direct_ufno": {"direct_temperature_ufno"},
        "residual_cnn": {"residual_decomposed", ""},
        "residual_fno": {"residual_decomposed_fno"},
        "residual_ufno": {"residual_decomposed_ufno"},
    }
    for model_name, root in roots.items():
        for protocol in PROTOCOLS:
            protocol_root = root / protocol
            metrics = load_json(protocol_root / "metrics.json")
            mode = str(metrics.get("model", {}).get("prediction_mode", ""))
            if mode not in expected_modes[model_name]:
                raise ValueError(
                    f"{model_name} has incompatible prediction_mode={mode!r} in {protocol}"
                )
            final = metrics.get("cnn_final_temperature") or metrics.get("final_temperature") or {}
            runtime = metrics.get("runtime", {})
            row = {
                "model": model_name,
                "formulation": "direct" if model_name.startswith("direct") else "residual",
                "backbone": (
                    "ufno"
                    if model_name.endswith("ufno")
                    else ("fno" if model_name.endswith("fno") else "cnn")
                ),
                "protocol": protocol,
                "num_samples": metrics.get("num_samples"),
                "mae_K": final.get("mae_K"),
                "rmse_K": final.get("rmse_K"),
                "max_abs_error_K": final.get("max_abs_error_K"),
                "peak_temperature_mae_K": nested_metric(
                    metrics, "peak_temperature", "mae_K"
                ),
                "hotspot_location_error_cells": nested_metric(
                    metrics, "hotspot_location", "mean_cells"
                ),
                "boundary_region_mae_K": nested_metric(
                    metrics, "region_mae", "boundary_region_mae_K"
                ),
                "occupied_region_mae_K": nested_metric(
                    metrics, "region_mae", "occupied_region_mae_K"
                ),
                "worse_than_source_fraction": metrics.get(
                    "worse_than_physics_baseline_fraction"
                ),
                "parameter_count": metrics.get("model", {}).get("parameter_count"),
                "runtime_per_sample_s": metrics.get("inference_runtime_per_sample_s"),
                "throughput_samples_per_s": metrics.get("throughput_samples_per_s"),
                "peak_gpu_memory_bytes": metrics.get("peak_gpu_memory_bytes")
                or runtime.get("peak_gpu_memory_bytes"),
            }
            headline.append(row)
            for family_row in read_csv(protocol_root / "metrics_by_case.csv"):
                family = str(
                    family_row.get("case")
                    or family_row.get("case_id")
                    or family_row.get("family_uid")
                    or ""
                )
                families.append(
                    {
                        "model": model_name,
                        "formulation": row["formulation"],
                        "backbone": row["backbone"],
                        "protocol": protocol,
                        "family_uid": family,
                        "num_samples": numeric(
                            family_row.get("num_samples") or family_row.get("count")
                        ),
                        "mae_K": first_numeric(
                            family_row,
                            "final_temperature_mae_K",
                            "cnn_final_mae_K",
                            "mae_K",
                            "final_mae_K",
                        ),
                        "rmse_K": first_numeric(
                            family_row,
                            "final_temperature_rmse_K",
                            "cnn_final_rmse_K",
                            "rmse_K",
                            "final_rmse_K",
                        ),
                    }
                )
                if families[-1]["mae_K"] is None:
                    raise ValueError(
                        "learned-model per-family MAE is missing: "
                        f"model={model_name}, protocol={protocol}, family={family}, "
                        f"available_columns={sorted(family_row)}"
                    )
    return headline, families


def load_source_baseline(root: Path) -> dict[str, dict[str, str]]:
    path = root / "base_quality_by_sample.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"source-only root must contain base_quality_by_sample.csv: {root}"
        )
    rows = read_csv(path)
    return {str(row["sample_uid"]): row for row in rows}


def align_source_baseline(
    source_by_uid: dict[str, dict[str, str]],
    anchor_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    headline = []
    families = []
    for protocol in PROTOCOLS:
        anchor = read_csv(anchor_root / protocol / "metrics_by_sample.csv")
        selected: list[dict[str, str]] = []
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in anchor:
            uid = str(row.get("sample_uid") or "")
            if uid not in source_by_uid:
                raise ValueError(f"source-only baseline is missing {uid} for {protocol}")
            source = dict(source_by_uid[uid])
            family = str(row.get("family_uid") or row.get("case_id") or "")
            source["family_uid"] = family
            selected.append(source)
            grouped[family].append(source)
        headline.append(
            {
                "model": "source_only",
                "formulation": "source_only",
                "backbone": "none",
                "protocol": protocol,
                "num_samples": len(selected),
                "mae_K": mean_column(selected, "mae_K"),
                "rmse_K": rms_column(selected, "rmse_K"),
                "max_abs_error_K": max_column(selected, "max_abs_error_K"),
            }
        )
        for family, family_rows in sorted(grouped.items()):
            families.append(
                {
                    "model": "source_only",
                    "formulation": "source_only",
                    "backbone": "none",
                    "protocol": protocol,
                    "family_uid": family,
                    "num_samples": len(family_rows),
                    "mae_K": mean_column(family_rows, "mae_K"),
                    "rmse_K": rms_column(family_rows, "rmse_K"),
                }
            )
    return headline, families


def write_report(
    path: Path,
    headline: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> None:
    lines = [
        "# Benchmark v2 FNO Comparison",
        "",
        "This report separates formulation effects (direct versus source-residual) "
        "from backbone effects (CNN versus FNO).",
        "",
        "| Model | Formulation | Backbone | Protocol | MAE K | RMSE K | Params | Runtime ms |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f"| {row['model']} | {row['formulation']} | {row['backbone']} | "
            f"{row['protocol']} | {fmt(row.get('mae_K'))} | {fmt(row.get('rmse_K'))} | "
            f"{fmt_int(row.get('parameter_count'))} | "
            f"{fmt_ms(row.get('runtime_per_sample_s'))} |"
        )
    test_rows = [
        row
        for row in headline
        if row["protocol"] == "primary_test_families" and row.get("mae_K") is not None
    ]
    if test_rows:
        best = min(test_rows, key=lambda row: float(row["mae_K"]))
        lines.extend(
            [
                "",
                f"Best held-out-test model: **{best['model']}** at {float(best['mae_K']):.4f} K MAE.",
            ]
        )
    lines.extend(
        [
            "",
            "Per-family values are available in `fno_model_comparison_by_family.csv`.",
            f"Rows: {len(families)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(
    out_dir: Path,
    headline: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    models = sorted({str(row["model"]) for row in headline})
    protocols = list(PROTOCOLS)
    x = np.arange(len(protocols))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(models), 1)
    for index, model in enumerate(models):
        values = [
            value_for(headline, model=model, protocol=protocol, key="mae_K")
            for protocol in protocols
        ]
        ax.bar(x + (index - (len(models) - 1) / 2) * width, values, width, label=model)
    ax.set_xticks(x, protocols, rotation=15)
    ax.set_ylabel("MAE (K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fno_protocol_mae.png", dpi=160)
    plt.close(fig)

    selected = [row for row in families if row["protocol"] == "primary_test_families"]
    family_names = sorted({str(row["family_uid"]) for row in selected})
    fig, ax = plt.subplots(figsize=(11, 5))
    for model in models:
        values = [
            value_for(selected, model=model, family_uid=family, key="mae_K")
            for family in family_names
        ]
        ax.plot(family_names, values, marker="o", label=model)
    ax.set_ylabel("MAE (K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fno_per_family_test_mae.png", dpi=160)
    plt.close(fig)

    test_rows = [row for row in headline if row["protocol"] == "primary_test_families"]
    plot_scatter(
        out_dir / "fno_accuracy_runtime_tradeoff.png",
        test_rows,
        "runtime_per_sample_s",
        "mae_K",
        "Runtime per sample (s)",
        "MAE (K)",
        plt,
    )
    plot_scatter(
        out_dir / "fno_parameter_efficiency.png",
        test_rows,
        "parameter_count",
        "mae_K",
        "Parameters",
        "MAE (K)",
        plt,
    )


def plot_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    plt: Any,
) -> None:
    usable = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None]
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in usable:
        ax.scatter(float(row[x_key]), float(row[y_key]))
        ax.annotate(str(row["model"]), (float(row[x_key]), float(row[y_key])))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def nested_metric(payload: dict[str, Any], first: str, second: str) -> Any:
    value = payload.get(first)
    return value.get(second) if isinstance(value, dict) else None


def value_for(rows: list[dict[str, Any]], *, key: str, **match: str) -> float:
    for row in rows:
        if all(str(row.get(name)) == value for name, value in match.items()):
            value = row.get(key)
            return float(value) if value is not None else float("nan")
    return float("nan")


def numeric(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def first_numeric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if row.get(key) not in {None, ""}:
            return float(row[key])
    return None


def mean_column(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return float(np.mean(values)) if values else None


def rms_column(rows: list[dict[str, str]], key: str) -> float | None:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) not in {None, ""}],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def max_column(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return max(values) if values else None


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
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def fmt_int(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{1000.0 * float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
