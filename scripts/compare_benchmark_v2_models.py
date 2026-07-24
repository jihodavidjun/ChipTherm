#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import gnn_promotion_decision, write_csv, write_json


PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
    "primary_test_families",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create split-correct Benchmark v2 baseline and optional-GNN comparison reports."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-baseline-dir", required=True, type=Path)
    parser.add_argument("--cnn-eval-root", required=True, type=Path)
    parser.add_argument("--context-cnn-eval-root", default=None, type=Path)
    parser.add_argument("--provisional-cnn-eval-root", default=None, type=Path)
    parser.add_argument("--gnn-eval-root", default=None, type=Path)
    parser.add_argument("--gnn-runtime-overhead-fraction", default=0.0, type=float)
    parser.add_argument("--gnn-memory-overhead-fraction", default=0.0, type=float)
    parser.add_argument("--bootstrap-samples", default=2000, type=int)
    parser.add_argument("--seed", default=20260721, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv(args.source_baseline_dir / "base_quality_by_sample.csv")
    source_by_uid = {str(row["sample_uid"]): row for row in source_rows}
    model_roots = {
        "context_cnn_without_source_superposition": args.context_cnn_eval_root,
        "provisional_source_plus_residual_cnn": args.provisional_cnn_eval_root,
        "final_source_plus_residual_cnn": args.cnn_eval_root,
        "final_source_plus_residual_cnn_gnn": args.gnn_eval_root,
    }
    headline: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        index_path = protocol_index(data_root, args.source_version, protocol)
        index_rows = read_csv(index_path)
        uids = [str(row["sample_uid"]) for row in index_rows]
        ambient_records = ambient_baseline_records(index_rows, data_root)
        headline.append(summary_row("ambient_constant", protocol, ambient_records))
        selected_source = [source_by_uid[uid] for uid in uids if uid in source_by_uid]
        if len(selected_source) != len(uids):
            raise ValueError(
                f"source baseline does not align with {protocol}: {len(selected_source)}/{len(uids)}"
            )
        headline.append(summary_row("final_source_superposition_only", protocol, selected_source))

        protocol_models: dict[str, list[dict[str, str]]] = {}
        for model_name, root in model_roots.items():
            if root is None:
                continue
            sample_path = root.expanduser().resolve() / protocol / "metrics_by_sample.csv"
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            records = read_csv(sample_path)
            protocol_models[model_name] = records
            headline.append(summary_row(model_name, protocol, records))
            ranking_rows.extend(rank_samples(model_name, protocol, records))
            stratum_rows.extend(stratify_records(model_name, protocol, records, index_rows))

        final_rows = protocol_models.get("final_source_plus_residual_cnn")
        if final_rows:
            paired[f"{protocol}:source_to_cnn"] = paired_improvement(
                selected_source, final_rows, args.bootstrap_samples, args.seed
            )
        gnn_rows = protocol_models.get("final_source_plus_residual_cnn_gnn")
        if final_rows and gnn_rows:
            paired[f"{protocol}:cnn_to_gnn"] = paired_improvement(
                final_rows, gnn_rows, args.bootstrap_samples, args.seed
            )
            if protocol == "primary_test_families":
                graph_rows_path = (
                    args.gnn_eval_root.expanduser().resolve()
                    / protocol
                    / "graph_contribution_by_sample.csv"
                )
                if not graph_rows_path.is_file():
                    raise FileNotFoundError(
                        "matched GNN promotion requires graph_contribution_by_sample.csv "
                        f"from the integrated graph checkpoint: {graph_rows_path}"
                    )
                matched_graph_rows = read_csv(graph_rows_path)
                matched_cnn = [
                    {
                        "sample_uid": row["sample_uid"],
                        "family_uid": row["case_id"],
                        "mae_K": row["cnn_only_mae_K"],
                        "rmse_K": row["cnn_only_rmse_K"],
                        "peak_temperature_abs_error_K": row[
                            "cnn_only_peak_temperature_abs_error_K"
                        ],
                    }
                    for row in matched_graph_rows
                ]
                matched_gnn = [
                    {
                        "sample_uid": row["sample_uid"],
                        "family_uid": row["case_id"],
                        "mae_K": row["fused_mae_K"],
                        "rmse_K": row["fused_rmse_K"],
                        "peak_temperature_abs_error_K": row[
                            "fused_peak_temperature_abs_error_K"
                        ],
                    }
                    for row in matched_graph_rows
                ]
                paired["gnn_promotion"] = gnn_promotion_decision(
                    matched_cnn,
                    matched_gnn,
                    runtime_overhead_fraction=args.gnn_runtime_overhead_fraction,
                    memory_overhead_fraction=args.gnn_memory_overhead_fraction,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                )

    write_csv(out_dir / "headline_metrics.csv", headline)
    write_csv(out_dir / "sample_rankings.csv", ranking_rows)
    write_csv(out_dir / "metrics_by_stratum.csv", stratum_rows)
    write_json(out_dir / "paired_comparisons.json", paired)
    write_report(out_dir, headline, paired)
    print(f"Comparison report: {out_dir / 'comparison_report.md'}")
    return 0


def protocol_index(data_root: Path, version: str, protocol: str) -> Path:
    root = data_root / f"derived/indices/full_50x200/source_superposition/{version}"
    if protocol == "known_family_sample_test":
        return root / "sample_split/test_index.csv"
    if protocol == "primary_validation_families":
        return root / "family_split/val_index.csv"
    return root / "family_split/test_index.csv"


def ambient_baseline_records(rows: list[dict[str, str]], data_root: Path) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        target = np.load(resolve_path(row["y_path"], data_root)).astype(np.float64)
        ambient = float(row.get("ambient_K") or 318.15)
        error = ambient - target
        output.append(
            {
                "sample_uid": row["sample_uid"],
                "family_uid": row.get("family_uid") or row.get("case_id"),
                "mae_K": float(np.mean(np.abs(error))),
                "rmse_K": float(np.sqrt(np.mean(error * error))),
                "peak_temperature_abs_error_K": float(abs(ambient - np.max(target))),
            }
        )
    return output


def summary_row(model: str, protocol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "protocol": protocol,
        "num_samples": len(rows),
        "mae_K": mean(rows, "mae_K"),
        "mean_sample_rmse_K": mean(rows, "rmse_K"),
        "peak_temperature_mae_K": mean_first(
            rows, ("peak_temperature_abs_error_K", "hotspot_temp_error_K")
        ),
        "max_abs_error_K": maximum(rows, "max_abs_error_K"),
        "centered_field_mae_K": mean(rows, "centered_field_mae_K"),
        "hotspot_location_error_cells": mean(rows, "hotspot_location_error_cells"),
        "occupied_region_mae_K": mean(rows, "occupied_region_mae_K"),
        "boundary_region_mae_K": mean(rows, "boundary_region_mae_K"),
        "hotspot_top1pct_mae_K": mean(rows, "hotspot_top1pct_mae_K"),
        "low_frequency_error_energy_fraction": mean(
            rows, "low_frequency_error_energy_fraction"
        ),
        "gradient_error_abs_mean_K_per_cell": mean(
            rows, "gradient_error_abs_mean_K_per_cell"
        ),
        "high_gradient_region_mae_K": mean(rows, "high_gradient_region_mae_K"),
    }


def paired_improvement(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_uid = {str(row.get("sample_uid") or row.get("original_sample_uid")): row for row in baseline}
    candidate_by_uid = {str(row.get("sample_uid") or row.get("original_sample_uid")): row for row in candidate}
    common = sorted(set(baseline_by_uid) & set(candidate_by_uid))
    if not common:
        raise ValueError("paired model comparison has no common sample_uid")
    improvements = np.asarray(
        [
            float(baseline_by_uid[uid]["mae_K"]) - float(candidate_by_uid[uid]["mae_K"])
            for uid in common
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(improvements), size=(max(bootstrap_samples, 1), len(improvements)))
    boot = improvements[indices].mean(axis=1)
    return {
        "matched_samples": len(common),
        "mean_mae_improvement_K": float(np.mean(improvements)),
        "median_mae_improvement_K": float(np.median(improvements)),
        "improved_sample_fraction": float(np.mean(improvements > 0.0)),
        "paired_bootstrap_95pct_CI_K": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
    }


def rank_samples(model: str, protocol: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["mae_K"]))
    chosen: list[tuple[str, dict[str, str]]] = []
    chosen.extend(("best", row) for row in ordered[:20])
    chosen.append(("median", ordered[len(ordered) // 2]))
    chosen.extend(("worst", row) for row in ordered[-50:])
    return [
        {
            "model": model,
            "protocol": protocol,
            "rank_category": category,
            **row,
        }
        for category, row in chosen
    ]


def stratify_records(
    model: str,
    protocol: str,
    records: list[dict[str, str]],
    index_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metadata = {str(row["sample_uid"]): row for row in index_rows}
    joined = [(row, metadata.get(str(row["sample_uid"]), {})) for row in records]
    outputs: list[dict[str, Any]] = []
    categorical = ("family_uid", "family_archetype", "workload_topology")
    for field in categorical:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for metric, meta in joined:
            value = meta.get(field) or (metric.get("family_uid") if field == "family_uid" else "")
            if value:
                groups[str(value)].append(metric)
        for value, items in sorted(groups.items()):
            outputs.append(stratum_row(model, protocol, field, value, items))
    numeric = (
        "chiplet_count",
        "occupied_fraction",
        "total_power_W",
        "active_chiplet_fraction",
        "dominant_source_share",
        "target_peak_temperature_K",
    )
    for field in numeric:
        available = [
            (metric, float(meta[field]))
            for metric, meta in joined
            if meta.get(field) not in {None, ""}
        ]
        if len(available) < 4:
            continue
        values = np.asarray([value for _, value in available], dtype=np.float64)
        boundaries = np.quantile(values, [0.25, 0.5, 0.75])
        groups = defaultdict(list)
        for metric, value in available:
            bucket = int(np.searchsorted(boundaries, value, side="right"))
            groups[f"Q{bucket + 1}"].append(metric)
        for value, items in sorted(groups.items()):
            outputs.append(stratum_row(model, protocol, field, value, items))
    return outputs


def stratum_row(
    model: str,
    protocol: str,
    field: str,
    value: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "model": model,
        "protocol": protocol,
        "stratum": field,
        "value": value,
        "num_samples": len(rows),
        "mae_K": mean(rows, "mae_K"),
        "rmse_K": mean(rows, "rmse_K"),
        "peak_temperature_mae_K": mean_first(rows, ("peak_temperature_abs_error_K",)),
    }


def mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return float(np.mean(values)) if values else None


def mean_first(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mean(rows, key)
        if value is not None:
            return value
    return None


def maximum(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return max(values) if values else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_root / path


def write_report(out_dir: Path, headline: list[dict[str, Any]], paired: dict[str, Any]) -> None:
    lines = [
        "# Benchmark v2 Final Model Comparison",
        "",
        "Primary held-out-family and secondary known-family results are deliberately separated.",
        "",
        "| Model | Protocol | N | MAE K | Mean sample RMSE K | Peak MAE K |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f"| {row['model']} | {row['protocol']} | {row['num_samples']} | "
            f"{format_value(row['mae_K'])} | {format_value(row['mean_sample_rmse_K'])} | "
            f"{format_value(row['peak_temperature_mae_K'])} |"
        )
    recommendation = paired.get("gnn_promotion", {}).get(
        "recommendation", "GNN NOT EVALUATED; OMIT GNN FROM PRIMARY MODEL"
    )
    lines.extend(["", f"**GNN recommendation:** {recommendation}", ""])
    (out_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def format_value(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
