#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import compare_benchmark_v2_fno_models as base  # noqa: E402


MODEL_ARGUMENTS = {
    "direct_cnn": "direct_cnn_root",
    "residual_cnn": "residual_cnn_root",
    "direct_fno": "direct_fno_root",
    "residual_fno": "residual_fno_root",
    "direct_ufno": "direct_ufno_root",
    "residual_ufno": "residual_ufno_root",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare controlled Benchmark v2 CNN, FNO, and U-FNO experiments."
    )
    parser.add_argument("--source-only-root", type=Path)
    for argument in MODEL_ARGUMENTS.values():
        parser.add_argument(f"--{argument.replace('_', '-')}", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = {
        name: getattr(args, argument).expanduser().resolve()
        for name, argument in MODEL_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }
    if not roots:
        raise SystemExit("provide at least one explicit learned-model evaluation root")
    headline, families = base.aggregate_comparison(roots)
    if args.source_only_root is not None:
        source = base.load_source_baseline(args.source_only_root.expanduser().resolve())
        anchor = next(iter(roots.values()))
        source_headline, source_families = base.align_source_baseline(source, anchor)
        headline.extend(source_headline)
        families.extend(source_families)
    enrich_effects(headline)
    mark_pareto(headline)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(out_dir / "operator_model_comparison.csv", headline)
    base.write_csv(out_dir / "operator_model_comparison_by_family.csv", families)
    write_report(out_dir / "operator_comparison_report.md", headline)
    write_plots(out_dir, headline, families)
    print(f"Operator comparison: {out_dir}")
    return 0


def enrich_effects(rows: list[dict[str, Any]]) -> None:
    lookup = {
        (str(row["model"]), str(row["protocol"])): row
        for row in rows
        if row.get("mae_K") is not None
    }
    for row in rows:
        protocol = str(row["protocol"])
        backbone = str(row["backbone"])
        formulation = str(row["formulation"])
        row["decomposition_gain_K"] = None
        row["local_multiscale_gain_K"] = None
        if formulation == "residual":
            direct = lookup.get((f"direct_{backbone}", protocol))
            if direct is not None and row.get("mae_K") is not None:
                row["decomposition_gain_K"] = float(direct["mae_K"]) - float(row["mae_K"])
        if backbone == "ufno":
            plain = lookup.get((f"{formulation}_fno", protocol))
            if plain is not None and row.get("mae_K") is not None:
                row["local_multiscale_gain_K"] = float(plain["mae_K"]) - float(row["mae_K"])


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["pareto_accuracy_runtime"] = False
    for protocol in base.PROTOCOLS:
        candidates = [
            row
            for row in rows
            if row["protocol"] == protocol
            and row.get("mae_K") is not None
            and row.get("runtime_per_sample_s") is not None
        ]
        for row in candidates:
            dominated = any(
                float(other["mae_K"]) <= float(row["mae_K"])
                and float(other["runtime_per_sample_s"])
                <= float(row["runtime_per_sample_s"])
                and (
                    float(other["mae_K"]) < float(row["mae_K"])
                    or float(other["runtime_per_sample_s"])
                    < float(row["runtime_per_sample_s"])
                )
                for other in candidates
            )
            row["pareto_accuracy_runtime"] = not dominated


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark v2 Operator Comparison",
        "",
        "Positive gains mean the residual formulation or U-Net augmentation reduced MAE.",
        "",
        "| Model | Protocol | MAE K | RMSE K | Decomposition gain K | U-Net gain K | Params | Runtime ms | Pareto |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['protocol']} | {base.fmt(row.get('mae_K'))} | "
            f"{base.fmt(row.get('rmse_K'))} | {base.fmt(row.get('decomposition_gain_K'))} | "
            f"{base.fmt(row.get('local_multiscale_gain_K'))} | "
            f"{base.fmt_int(row.get('parameter_count'))} | "
            f"{base.fmt_ms(row.get('runtime_per_sample_s'))} | "
            f"{'yes' if row.get('pareto_accuracy_runtime') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- `decomposition_gain = direct_MAE - residual_MAE` for a matched backbone.",
            "- `U-Net gain = FNO_MAE - U-FNO_MAE` for a matched formulation.",
            "- Held-out primary-test metrics are descriptive only and are not used to select a model.",
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
    learned = [row for row in headline if row["model"] != "source_only"]
    models = sorted({str(row["model"]) for row in learned})
    protocols = list(base.PROTOCOLS)
    x = np.arange(len(protocols))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, model in enumerate(models):
        values = [
            base.value_for(learned, model=model, protocol=protocol, key="mae_K")
            for protocol in protocols
        ]
        ax.bar(x + (index - (len(models) - 1) / 2) * width, values, width, label=model)
    ax.set_xticks(x, protocols, rotation=15)
    ax.set_ylabel("MAE (K)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "operator_protocol_mae.png", dpi=160)
    plt.close(fig)

    test_families = [
        row for row in families if row["protocol"] == "primary_test_families"
    ]
    names = sorted({str(row["family_uid"]) for row in test_families})
    fig, ax = plt.subplots(figsize=(11, 5))
    for model in sorted({str(row["model"]) for row in test_families}):
        values = [
            base.value_for(
                test_families, model=model, family_uid=family, key="mae_K"
            )
            for family in names
        ]
        ax.plot(names, values, marker="o", label=model)
    ax.set_ylabel("MAE (K)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "operator_per_family_test_mae.png", dpi=160)
    plt.close(fig)

    test_rows = [
        row for row in learned if row["protocol"] == "primary_test_families"
    ]
    base.plot_scatter(
        out_dir / "operator_accuracy_runtime_tradeoff.png",
        test_rows,
        "runtime_per_sample_s",
        "mae_K",
        "Runtime per sample (s)",
        "MAE (K)",
        plt,
    )
    base.plot_scatter(
        out_dir / "operator_parameter_efficiency.png",
        test_rows,
        "parameter_count",
        "mae_K",
        "Parameters",
        "MAE (K)",
        plt,
    )
    plot_gain(
        out_dir / "operator_formulation_gain.png",
        test_rows,
        "decomposition_gain_K",
        "Residual decomposition gain (K)",
        plt,
    )
    plot_gain(
        out_dir / "operator_backbone_gain.png",
        test_rows,
        "local_multiscale_gain_K",
        "U-Net augmentation gain (K)",
        plt,
    )


def plot_gain(
    path: Path,
    rows: list[dict[str, Any]],
    key: str,
    label: str,
    plt: Any,
) -> None:
    usable = [row for row in rows if row.get(key) is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(row["model"]) for row in usable], [float(row[key]) for row in usable])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
