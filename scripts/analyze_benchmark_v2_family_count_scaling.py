#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_family_scaling import (  # noqa: E402
    FAMILY_COUNTS,
    RUN_IDS,
    aggregate_sample_metrics,
)
from scripts.analyze_benchmark_v2_zero_shot import locate_protocol_dir  # noqa: E402


PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
    "primary_test_families",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate completed Benchmark v2 family-count scaling endpoints."
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument(
        "--canonical-train40-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1",
    )
    parser.add_argument(
        "--definition-dir",
        type=Path,
        default=REPO_ROOT / "outputs/benchmark_v2_50family/family_count_scaling_summary",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/benchmark_v2_50family/family_count_scaling_summary",
    )
    parser.add_argument("--include-primary-test", action="store_true")
    parser.add_argument("--require-validation-complete", action="store_true")
    args = parser.parse_args()
    summary = analyze_scaling(
        experiment_root=args.experiment_root.expanduser().resolve(),
        canonical_train40_root=args.canonical_train40_root.expanduser().resolve(),
        definition_dir=args.definition_dir.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        include_primary_test=args.include_primary_test,
        require_validation_complete=args.require_validation_complete,
    )
    print("Family-count scaling endpoints available:", summary["complete_family_counts"])
    print("Primary test included:", summary["primary_test_included"])
    return 0


def analyze_scaling(
    *,
    experiment_root: Path,
    canonical_train40_root: Path,
    definition_dir: Path,
    out_dir: Path,
    include_primary_test: bool,
    require_validation_complete: bool,
) -> dict[str, Any]:
    equivalence = read_json(definition_dir / "train40_reuse_equivalence.json")
    if equivalence.get("canonical_train40_reusable") is not True:
        raise ValueError("canonical train40 endpoint cannot be reused")
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        count: (
            canonical_train40_root if count == 40 else experiment_root / RUN_IDS[count]
        )
        for count in FAMILY_COUNTS
    }
    requested_protocols = list(PROTOCOLS[:2])
    if include_primary_test:
        requested_protocols.append(PROTOCOLS[2])
    metrics_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for count, root in roots.items():
        training = training_statistics(root, family_count=count)
        for protocol in requested_protocols:
            try:
                protocol_dir = locate_protocol_dir(root, protocol)
            except FileNotFoundError:
                missing.append(f"train{count}:{protocol}")
                continue
            sample_path = protocol_dir / "metrics_by_sample.csv"
            if not sample_path.is_file():
                missing.append(f"train{count}:{protocol}")
                continue
            rows = read_csv(sample_path)
            aggregate = aggregate_extended(rows)
            metrics_path = protocol_dir / "metrics.json"
            metrics = read_json(metrics_path)
            aggregate["runtime_per_sample_s"] = float(
                metrics["inference_runtime_per_sample_s"]
            )
            aggregate["parameter_count"] = int(metrics["model"]["parameter_count"])
            metrics_rows.append(
                {
                    "family_count": count,
                    "run_id": RUN_IDS.get(count, "feature_fusion_train40_source_v1_seed1"),
                    "endpoint_type": "canonical_reference" if count == 40 else "new_run",
                    "protocol": protocol,
                    "sample_count": len(rows),
                    **aggregate,
                    **training,
                    "metrics_path": str(sample_path),
                    "aggregate_metrics_path": str(metrics_path),
                }
            )
            family_rows.extend(aggregate_families(count, protocol, rows))
    if require_validation_complete:
        required = {
            f"train{count}:{protocol}"
            for count in FAMILY_COUNTS
            for protocol in PROTOCOLS[:2]
        }
        absent = sorted(required & set(missing))
        if absent:
            raise ValueError(f"validation scaling endpoints are incomplete: {absent}")
    write_csv(out_dir / "family_count_scaling_metrics.csv", metrics_rows)
    write_csv(out_dir / "family_count_scaling_per_family.csv", family_rows)
    interpretation = interpret(metrics_rows, family_rows, include_primary_test)
    summary = {
        "schema_version": "benchmark_v2_family_count_scaling_summary/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_family_counts": sorted({int(row["family_count"]) for row in metrics_rows}),
        "primary_test_included": include_primary_test,
        "missing_results": sorted(missing),
        "canonical_train40_reused": True,
        "canonical_checkpoint": equivalence["canonical"]["checkpoint_path"],
        "interpretation": interpretation,
    }
    (out_dir / "family_count_scaling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(out_dir / "family_count_scaling_report.md", summary, metrics_rows)
    write_plots(out_dir, metrics_rows, family_rows)
    return summary


def aggregate_extended(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    base = aggregate_sample_metrics(rows)

    def mean(name: str) -> float:
        values = [float(row[name]) for row in rows if str(row.get(name, "")).strip()]
        return float(np.mean(values)) if values else float("nan")

    source = mean("physics_baseline_mae_K")
    final = base["micro_mae_K"]
    return {
        **base,
        "centered_field_mae_K": mean("centered_field_mae_K"),
        "mean_rise_mae_K": mean("mean_head_abs_error_K"),
        "hotspot_top1pct_mae_K": mean("hotspot_top1pct_mae_K"),
        "boundary_mae_K": mean("boundary_region_mae_K"),
        "source_baseline_mae_K": source,
        "source_improvement_K": source - final,
        "fraction_worse_than_source": float(
            np.mean(
                [
                    float(row["mae_K"]) > float(row["physics_baseline_mae_K"])
                    for row in rows
                ]
            )
        ),
        "runtime_per_sample_s": float("nan"),
        "parameter_count": float("nan"),
    }


def aggregate_families(
    family_count: int, protocol: str, rows: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family_uid") or row.get("case_id"))].append(row)
    output = []
    for family, items in sorted(grouped.items()):
        aggregate = aggregate_extended(items)
        output.append(
            {
                "family_count": family_count,
                "protocol": protocol,
                "family_uid": family,
                "sample_count": len(items),
                **aggregate,
            }
        )
    return output


def training_statistics(root: Path, *, family_count: int) -> dict[str, Any]:
    log_path = root / "train_log.csv"
    if not log_path.is_file():
        return {
            "samples_per_epoch": 160 * family_count,
            "optimizer_updates_per_epoch": math.ceil(160 * family_count / 64),
            "total_optimizer_updates": "",
            "completed_epochs": "",
            "early_stopping_epoch": "",
            "training_runtime_s": "",
        }
    rows = read_csv(log_path)
    completed = max(int(row["epoch"]) for row in rows)
    updates = math.ceil(160 * family_count / 64)
    return {
        "samples_per_epoch": 160 * family_count,
        "optimizer_updates_per_epoch": updates,
        "total_optimizer_updates": updates * completed,
        "completed_epochs": completed,
        "early_stopping_epoch": completed if completed < 100 else "",
        "training_runtime_s": float(
            sum(float(row.get("epoch_runtime_s") or 0.0) for row in rows)
        ),
    }


def interpret(
    metrics_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    include_primary_test: bool,
) -> dict[str, Any]:
    validation = sorted(
        (
            row
            for row in metrics_rows
            if row["protocol"] == "primary_validation_families"
        ),
        key=lambda row: int(row["family_count"]),
    )
    result: dict[str, Any] = {
        "validation_curve_K": {
            str(row["family_count"]): row["micro_mae_K"] for row in validation
        },
        "validation_material_improvement": "pending" if len(validation) < 4 else None,
        "saturation_assessment": "pending" if len(validation) < 4 else None,
        "centered_vs_mean_assessment": "pending" if len(validation) < 4 else None,
        "source_baseline_error_correlation": "pending",
        "next_step": "pending until train10/20/30 validation results exist",
    }
    if len(validation) == 4:
        values = [float(row["micro_mae_K"]) for row in validation]
        gains = [values[index - 1] - values[index] for index in range(1, 4)]
        centered = [float(row["centered_field_mae_K"]) for row in validation]
        mean = [float(row["mean_rise_mae_K"]) for row in validation]
        result["validation_material_improvement"] = gains
        result["saturation_assessment"] = (
            "saturating_before_40" if gains[-1] < 0.05 * max(values[0], 1e-12) else "not_saturated"
        )
        result["centered_vs_mean_assessment"] = {
            "centered_error_reduction_train10_to_train40_K": centered[0] - centered[-1],
            "mean_error_reduction_train10_to_train40_K": mean[0] - mean[-1],
            "centered_decreases_more_slowly": (
                centered[0] - centered[-1] < mean[0] - mean[-1]
            ),
        }
        result["next_step"] = (
            "more_training_family_diversity"
            if result["saturation_assessment"] == "not_saturated"
            else "inspect_source_quality_and_family_specific_adaptation"
        )
    reference_families = [
        row
        for row in family_rows
        if int(row["family_count"]) == 40
        and row["protocol"]
        in (
            "primary_validation_families",
            "primary_test_families",
        )
    ]
    if len(reference_families) >= 5:
        source = np.asarray(
            [float(row["source_baseline_mae_K"]) for row in reference_families]
        )
        final = np.asarray([float(row["micro_mae_K"]) for row in reference_families])
        result["source_baseline_error_correlation"] = {
            "family_count": len(reference_families),
            "pearson": float(np.corrcoef(source, final)[0, 1]),
            "scope": "canonical train40 held-out families available to this analysis",
        }
    if include_primary_test:
        f044 = [
            row
            for row in family_rows
            if row["protocol"] == "primary_test_families" and row["family_uid"] == "f044"
        ]
        result["f044_curve_K"] = {
            str(row["family_count"]): row["micro_mae_K"] for row in f044
        }
    return result


def write_plots(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_specs = (
        ("known_family_sample_test", "micro_mae_K", "known_family_mae_vs_family_count.png"),
        ("primary_validation_families", "micro_mae_K", "heldout_validation_mae_vs_family_count.png"),
        ("primary_test_families", "micro_mae_K", "heldout_test_mae_vs_family_count.png"),
        ("primary_validation_families", "worst_family_mae_K", "worst_family_mae_vs_family_count.png"),
        ("primary_validation_families", "hotspot_top1pct_mae_K", "hotspot_mae_vs_family_count.png"),
        ("known_family_sample_test", "training_runtime_s", "training_time_vs_family_count.png"),
    )
    for protocol, metric, filename in plot_specs:
        selected = sorted(
            (row for row in rows if row["protocol"] == protocol and row.get(metric) not in ("", None)),
            key=lambda row: int(row["family_count"]),
        )
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        ax.plot(
            [int(row["family_count"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker="o",
        )
        ax.set_xlabel("Training package families")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)
    f044 = sorted(
        (
            row
            for row in family_rows
            if row["protocol"] == "primary_test_families" and row["family_uid"] == "f044"
        ),
        key=lambda row: int(row["family_count"]),
    )
    if f044:
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        ax.plot(
            [int(row["family_count"]) for row in f044],
            [float(row["micro_mae_K"]) for row in f044],
            marker="o",
        )
        ax.set_xlabel("Training package families")
        ax.set_ylabel("f044 MAE (K)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "f044_mae_vs_family_count.png", dpi=180)
        plt.close(fig)
    validation = sorted(
        (row for row in rows if row["protocol"] == "primary_validation_families"),
        key=lambda row: int(row["family_count"]),
    )
    if validation:
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        counts = [int(row["family_count"]) for row in validation]
        ax.plot(
            counts,
            [float(row["centered_field_mae_K"]) for row in validation],
            marker="o",
            label="Centered field",
        )
        ax.plot(
            counts,
            [float(row["mean_rise_mae_K"]) for row in validation],
            marker="o",
            label="Mean correction",
        )
        ax.set_xlabel("Training package families")
        ax.set_ylabel("MAE (K)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "centered_and_mean_error_vs_family_count.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    lines = [
        "# Benchmark v2 Family-Count Scaling",
        "",
        f"- Canonical train40 reused: **{summary['canonical_train40_reused']}**",
        f"- Primary test included: **{summary['primary_test_included']}**",
        f"- Complete endpoints: {summary['complete_family_counts']}",
        "",
        "| Families | Protocol | Micro MAE (K) | Macro family MAE (K) | Centered MAE (K) | Mean MAE (K) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family_count']} | {row['protocol']} | {float(row['micro_mae_K']):.4f} | "
            f"{float(row['macro_family_mae_K']):.4f} | {float(row['centered_field_mae_K']):.4f} | "
            f"{float(row['mean_rise_mae_K']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Primary-test results are excluded until explicitly enabled. Interpretive conclusions "
            "remain pending until all validation endpoints exist.",
            "",
            "## Interpretation",
            "",
            "```json",
            json.dumps(summary["interpretation"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
