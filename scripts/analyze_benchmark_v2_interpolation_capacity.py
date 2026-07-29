#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.benchmark_v2_interpolation_capacity import (  # noqa: E402
    CANONICAL_RUN_ID,
    RUN_IDS,
    aggregate_sample_rows,
    interpolation_decision_gate,
)
from scripts.analyze_benchmark_v2_zero_shot import locate_protocol_dir  # noqa: E402


REFERENCE_ROOTS = {
    "canonical_cnn": REPO_ROOT
    / "outputs/benchmark_v2_50family/package_residual"
    / CANONICAL_RUN_ID,
    "fno": REPO_ROOT
    / "outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1",
    "ufno": REPO_ROOT
    / "outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1",
    "sau_fno": REPO_ROOT
    / "outputs/benchmark_v2_50family/sau_fno/residual_sau_fno_decomposed_train40_seed1",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze bounded CNN interpolation capacity without using primary test for selection."
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/interpolation_capacity_summary",
    )
    parser.add_argument("--include-primary-test", action="store_true")
    parser.add_argument("--require-cosine-validation", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        experiment_root=args.experiment_root.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        include_primary_test=args.include_primary_test,
        require_cosine_validation=args.require_cosine_validation,
    )
    print("Available models:", ", ".join(summary["available_models"]))
    print("Primary test included:", summary["primary_test_included"])
    return 0


def analyze(
    *,
    experiment_root: Path,
    out_dir: Path,
    include_primary_test: bool,
    require_cosine_validation: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, tuple[Path, str]] = {
        name: (root, "reference") for name, root in REFERENCE_ROOTS.items()
    }
    models.update(
        {
            "cnn_cosine_ema": (
                experiment_root / RUN_IDS["cosine_ema"],
                "ema",
            ),
            "cnn_cosine_raw": (
                experiment_root / RUN_IDS["cosine_ema"],
                "raw",
            ),
            "cnn_param_matched": (
                experiment_root / RUN_IDS["param_matched"],
                "ema",
            ),
            "cnn_param_matched_raw": (
                experiment_root / RUN_IDS["param_matched"],
                "raw",
            ),
        }
    )
    protocols = ["known_family_sample_test", "primary_validation_families"]
    if include_primary_test:
        protocols.append("primary_test_families")
    metric_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, str]] = []
    for model_name, (root, weights) in models.items():
        for protocol in protocols:
            protocol_dir = resolve_protocol(root, protocol, weights)
            if protocol_dir is None:
                continue
            sample_path = protocol_dir / "metrics_by_sample.csv"
            metrics_path = protocol_dir / "metrics.json"
            if not sample_path.is_file() or not metrics_path.is_file():
                continue
            rows = read_csv(sample_path)
            metrics = read_json(metrics_path)
            aggregate = aggregate_sample_rows(rows)
            metric_rows.append(
                {
                    "model": model_name,
                    "weights": weights,
                    "protocol": protocol,
                    "sample_count": len(rows),
                    **aggregate,
                    "runtime_per_sample_s": float(
                        metrics["inference_runtime_per_sample_s"]
                    ),
                    "parameter_count": int(metrics["model"]["parameter_count"]),
                    "metrics_path": str(metrics_path),
                }
            )
            family_rows.extend(per_family(model_name, weights, protocol, rows))
            inventory.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "weights": weights,
                    "selected_path": str(protocol_dir),
                }
            )
    cosine_required = {
        ("cnn_cosine_ema", "known_family_sample_test"),
        ("cnn_cosine_ema", "primary_validation_families"),
    }
    available = {(row["model"], row["protocol"]) for row in metric_rows}
    if require_cosine_validation and not cosine_required <= available:
        raise ValueError(
            f"cosine EMA validation is incomplete: {sorted(cosine_required - available)}"
        )

    write_csv(out_dir / "interpolation_capacity_metrics.csv", metric_rows)
    write_csv(out_dir / "interpolation_capacity_per_family.csv", family_rows)
    write_csv(out_dir / "artifact_inventory.csv", inventory)
    gate = build_gate(metric_rows)
    write_json(out_dir / "decision_gate.json", gate)
    summary = {
        "schema_version": "benchmark_v2_interpolation_capacity_summary/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_models": sorted({row["model"] for row in metric_rows}),
        "primary_test_included": include_primary_test,
        "selection_protocols": [
            "known_family_sample_test",
            "primary_validation_families",
        ],
        "primary_test_used_for_selection": False,
        "decision_gate": gate,
        "interpretation": interpret(metric_rows, gate),
    }
    write_json(out_dir / "interpolation_capacity_summary.json", summary)
    write_report(
        out_dir / "interpolation_capacity_report.md",
        summary,
        metric_rows,
    )
    write_plots(out_dir, metric_rows)
    return summary


def resolve_protocol(root: Path, protocol: str, weights: str) -> Path | None:
    if weights == "ema":
        candidates = (
            root / "evaluation_selection_ema" / protocol,
            root / "evaluation_primary_test_ema" / protocol,
        )
        return next((path for path in candidates if path.is_dir()), None)
    if weights == "raw":
        candidates = (
            root / "evaluation_selection_raw" / protocol,
            root / "evaluation_primary_test_raw" / protocol,
        )
        return next((path for path in candidates if path.is_dir()), None)
    try:
        return locate_protocol_dir(root, protocol)
    except FileNotFoundError:
        return None


def build_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {
        (str(row["model"]), str(row["protocol"])): row
        for row in rows
    }
    keys = (
        ("canonical_cnn", "known_family_sample_test"),
        ("canonical_cnn", "primary_validation_families"),
        ("cnn_cosine_ema", "known_family_sample_test"),
        ("cnn_cosine_ema", "primary_validation_families"),
    )
    if any(key not in lookup for key in keys):
        return {
            "status": "pending_cosine_ema_validation",
            "recommend_param_matched_training": None,
            "primary_test_used": False,
        }
    gate = interpolation_decision_gate(
        canonical_known_mae_K=float(lookup[keys[0]]["micro_mae_K"]),
        canonical_validation_mae_K=float(lookup[keys[1]]["micro_mae_K"]),
        candidate_known_mae_K=float(lookup[keys[2]]["micro_mae_K"]),
        candidate_validation_mae_K=float(lookup[keys[3]]["micro_mae_K"]),
    )
    gate["status"] = "complete"
    return gate


def per_family(
    model: str,
    weights: str,
    protocol: str,
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family_uid") or row.get("case_id"))].append(row)
    return [
        {
            "model": model,
            "weights": weights,
            "protocol": protocol,
            "family_uid": family,
            "sample_count": len(items),
            **aggregate_sample_rows(items),
        }
        for family, items in sorted(grouped.items())
    ]


def interpret(
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "canonical_undertrained": "pending",
        "cosine_ema_effect": "pending",
        "additional_capacity_needed": "pending",
        "inductive_bias_assessment": "pending",
    }
    if gate.get("status") == "complete":
        result["canonical_undertrained"] = (
            "supported" if gate["primary_success"] else "not_established"
        )
        result["cosine_ema_effect"] = {
            "known_relative_improvement_fraction": gate[
                "known_relative_improvement_fraction"
            ],
            "validation_delta_K": gate["validation_delta_K"],
        }
        result["additional_capacity_needed"] = (
            "not_immediately"
            if not gate["recommend_param_matched_training"]
            else "run_single_parameter_matched_variant"
        )
    raw = {
        (str(row["model"]), str(row["protocol"])): row for row in rows
    }
    for protocol in ("known_family_sample_test", "primary_validation_families"):
        ema_key = ("cnn_cosine_ema", protocol)
        raw_key = ("cnn_cosine_raw", protocol)
        if ema_key in raw and raw_key in raw:
            result.setdefault("ema_vs_raw", {})[protocol] = (
                float(raw[raw_key]["micro_mae_K"])
                - float(raw[ema_key]["micro_mae_K"])
            )
    return result


def write_plots(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    validation_rows = [
        row
        for row in rows
        if row["model"] not in {"cnn_cosine_raw", "cnn_param_matched_raw"}
    ]
    for protocol, filename in (
        ("known_family_sample_test", "known_family_mae_comparison.png"),
        (
            "primary_validation_families",
            "heldout_validation_mae_comparison.png",
        ),
    ):
        selected = [row for row in validation_rows if row["protocol"] == protocol]
        bar_plot(selected, "micro_mae_K", filename, out_dir, "MAE (K)")
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in validation_rows:
        paired[str(row["model"])][str(row["protocol"])] = float(
            row["micro_mae_K"]
        )
    names = [
        name
        for name, values in paired.items()
        if {"known_family_sample_test", "primary_validation_families"} <= values.keys()
    ]
    if names:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for name in names:
            ax.scatter(
                paired[name]["known_family_sample_test"],
                paired[name]["primary_validation_families"],
                label=name,
            )
        ax.set_xlabel("Known-family MAE (K)")
        ax.set_ylabel("Held-out validation MAE (K)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "interpolation_vs_generalization.png", dpi=180)
        plt.close(fig)
    scatter_plot(
        validation_rows,
        "runtime_per_sample_s",
        "micro_mae_K",
        out_dir / "runtime_vs_mae.png",
        "Runtime/sample (s)",
    )
    scatter_plot(
        validation_rows,
        "parameter_count",
        "micro_mae_K",
        out_dir / "parameter_count_vs_mae.png",
        "Parameters",
    )
    known = [
        row
        for row in validation_rows
        if row["protocol"] == "known_family_sample_test"
    ]
    if known:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        positions = np.arange(len(known))
        ax.bar(
            positions - 0.18,
            [float(row["centered_field_mae_K"]) for row in known],
            width=0.36,
            label="Centered",
        )
        ax.bar(
            positions + 0.18,
            [float(row["mean_correction_mae_K"]) for row in known],
            width=0.36,
            label="Mean",
        )
        ax.set_xticks(positions, [str(row["model"]) for row in known], rotation=25)
        ax.set_ylabel("MAE (K)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "centered_and_mean_error_comparison.png", dpi=180)
        plt.close(fig)


def bar_plot(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    filename: str,
    out_dir: Path,
    ylabel: str,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [str(row["model"]) for row in rows],
        [float(row[metric]) for row in rows],
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def scatter_plot(
    rows: Sequence[Mapping[str, Any]],
    x_name: str,
    y_name: str,
    path: Path,
    xlabel: str,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for row in rows:
        ax.scatter(float(row[x_name]), float(row[y_name]), label=str(row["model"]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE (K)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Benchmark v2 CNN Interpolation Capacity",
        "",
        "| Model | Weights | Protocol | MAE (K) | RMSE (K) | Parameters | Runtime/sample (s) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['weights']} | {row['protocol']} | "
            f"{float(row['micro_mae_K']):.5f} | {float(row['micro_rmse_K']):.5f} | "
            f"{int(row['parameter_count'])} | {float(row['runtime_per_sample_s']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Primary-test results are never used by the decision gate.",
            "",
            "## Decision",
            "",
            "```json",
            json.dumps(summary["decision_gate"], indent=2, sort_keys=True),
            "```",
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
