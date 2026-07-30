#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
    stable_hash,
)
from scripts.analyze_benchmark_v2_zero_shot import locate_protocol_dir  # noqa: E402


PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
    "primary_test_families",
)
VALIDATION_PROTOCOLS = PROTOCOLS[:2]
CNN_MATRIX_NAMES = (
    "canonical_small_constant",
    "small_cosine_ema_epoch100",
    "small_cosine_ema_epoch150",
    "wide_constant_epoch100",
    "wide_cosine_ema_epoch100",
    "wide_cosine_ema_epoch150",
)
OPERATOR_NAMES = ("fno", "ufno", "sau_fno")
REFERENCE_ROOTS = {
    "canonical_small_constant": REPO_ROOT
    / "outputs/benchmark_v2_50family/package_residual"
    / CANONICAL_RUN_ID,
    "fno": REPO_ROOT
    / "outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1",
    "ufno": REPO_ROOT
    / "outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1",
    "sau_fno": REPO_ROOT
    / "outputs/benchmark_v2_50family/sau_fno/residual_sau_fno_decomposed_train40_seed1",
}


def experiment_specs(experiment_root: Path) -> list[dict[str, Any]]:
    small = experiment_root / RUN_IDS["small_cosine_ema"]
    wide_constant = experiment_root / RUN_IDS["param_matched_constant"]
    wide_cosine = experiment_root / RUN_IDS["param_matched_cosine_ema"]
    specs = [
        spec("canonical_small_constant", REFERENCE_ROOTS["canonical_small_constant"], 100, "constant", "none", "raw", "cnn", None),
        spec("small_cosine_ema_epoch100", small, 100, "cosine_ema", "cosine", "ema", "cnn", "evaluation_epoch0100_ema"),
        spec("small_cosine_ema_epoch150", small, 150, "cosine_ema", "cosine", "ema", "cnn", "evaluation_epoch0150_ema"),
        spec("wide_constant_epoch100", wide_constant, 100, "constant", "none", "raw", "cnn", "evaluation_epoch0100_raw"),
        spec("wide_cosine_ema_epoch100", wide_cosine, 100, "cosine_ema", "cosine", "ema", "cnn", "evaluation_epoch0100_ema"),
        spec("wide_cosine_ema_epoch150", wide_cosine, 150, "cosine_ema", "cosine", "ema", "cnn", "evaluation_epoch0150_ema"),
        spec("wide_cosine_raw_epoch100", wide_cosine, 100, "cosine_ema", "cosine", "raw", "diagnostic", "evaluation_epoch0100_raw"),
        spec("wide_cosine_raw_epoch150", wide_cosine, 150, "cosine_ema", "cosine", "raw", "diagnostic", "evaluation_epoch0150_raw"),
        spec("fno", REFERENCE_ROOTS["fno"], 100, "frozen_operator", "operator_recipe", "raw", "fno", None),
        spec("ufno", REFERENCE_ROOTS["ufno"], 100, "frozen_operator", "operator_recipe", "raw", "ufno", None),
        spec("sau_fno", REFERENCE_ROOTS["sau_fno"], 100, "frozen_operator", "operator_recipe", "raw", "sau_fno", None),
    ]
    for item in specs:
        if item["model"].startswith(("wide_constant", "wide_cosine")):
            item["expected_parameter_count"] = 3_919_642
        elif item["model"].startswith(("canonical_small", "small_cosine")):
            item["expected_parameter_count"] = 2_188_803
    return specs


def spec(
    name: str,
    root: Path,
    epoch: int,
    recipe: str,
    scheduler: str,
    weights: str,
    architecture_family: str,
    evaluation_dir: str | None,
) -> dict[str, Any]:
    return {
        "model": name,
        "root": root,
        "epoch": epoch,
        "checkpoint": f"epoch_{epoch:04d}.pt" if evaluation_dir else "best.pt",
        "training_recipe": recipe,
        "scheduler": scheduler,
        "weights": weights,
        "architecture_family": architecture_family,
        "evaluation_dir": evaluation_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen two-factor CNN interpolation-capacity matrix."
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/interpolation_capacity_summary",
    )
    parser.add_argument("--freeze-validation", action="store_true")
    parser.add_argument("--include-primary-test", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        experiment_root=args.experiment_root.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        freeze_validation=args.freeze_validation,
        include_primary_test=args.include_primary_test,
    )
    print(f"Validation status: {summary['validation_status']}")
    print(f"Primary test included: {summary['primary_test_included']}")
    return 0


def analyze(
    *,
    experiment_root: Path,
    out_dir: Path,
    freeze_validation: bool = False,
    include_primary_test: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "validation_decision_gate.json"
    frozen_gate = read_json(gate_path) if gate_path.is_file() else None
    if include_primary_test and (
        frozen_gate is None or frozen_gate.get("status") != "frozen"
    ):
        raise ValueError(
            "--include-primary-test requires a frozen validation_decision_gate.json"
        )

    protocols = list(VALIDATION_PROTOCOLS)
    if include_primary_test:
        protocols.append("primary_test_families")
    metric_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for model_spec in experiment_specs(experiment_root):
        for protocol in protocols:
            protocol_dir = resolve_protocol(model_spec, protocol)
            record = {
                "model": model_spec["model"],
                "protocol": protocol,
                "selected_path": str(protocol_dir) if protocol_dir else "",
                "status": "missing",
            }
            if protocol_dir is None:
                inventory.append(record)
                continue
            sample_path = protocol_dir / "metrics_by_sample.csv"
            metrics_path = protocol_dir / "metrics.json"
            if not sample_path.is_file() or not metrics_path.is_file():
                inventory.append(record)
                continue
            sample_rows = read_csv(sample_path)
            metrics = read_json(metrics_path)
            aggregate = aggregate_sample_rows(sample_rows)
            parameter_count = parameter_count_from_metrics(metrics)
            expected_parameters = model_spec.get("expected_parameter_count")
            if (
                expected_parameters is not None
                and parameter_count != expected_parameters
            ):
                raise ValueError(
                    f"{model_spec['model']} parameter count {parameter_count} "
                    f"does not match frozen study count {expected_parameters}"
                )
            row = {
                **{key: value for key, value in model_spec.items() if key != "root"},
                "protocol": protocol,
                "sample_count": len(sample_rows),
                **aggregate,
                "runtime_per_sample_s": runtime_from_metrics(metrics),
                "parameter_count": parameter_count,
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "sample_metrics_sha256": sha256_file(sample_path),
            }
            metric_rows.append(row)
            family_rows.extend(
                per_family(
                    model_spec["model"],
                    model_spec["weights"],
                    protocol,
                    sample_rows,
                )
            )
            record.update(
                status="available",
                metrics_path=str(metrics_path),
                metrics_sha256=row["metrics_sha256"],
            )
            inventory.append(record)

    validation_payload = build_validation_gate(metric_rows)
    current_fingerprint = validation_payload.get("input_fingerprint")
    if frozen_gate is not None and frozen_gate.get("status") == "frozen":
        if current_fingerprint != frozen_gate.get("input_fingerprint"):
            raise ValueError(
                "validation artifacts changed after freeze; refusing to include or "
                "overwrite the frozen interpretation"
            )
        validation_payload = frozen_gate
    elif freeze_validation:
        if validation_payload["status"] != "ready_to_freeze":
            raise ValueError(
                "cannot freeze validation interpretation until all six CNN settings "
                "have known-family and held-out-validation metrics"
            )
        validation_payload["status"] = "frozen"
        validation_payload["frozen_at_utc"] = now()
    write_json(gate_path, validation_payload)

    primary_gate = build_primary_gate(
        include_primary_test=include_primary_test,
        validation_gate=validation_payload,
        metric_rows=metric_rows,
    )
    write_json(out_dir / "primary_test_gate.json", primary_gate)
    effects = compute_two_factor_effects(metric_rows)
    final_rows = build_final_comparison(metric_rows, experiment_specs(experiment_root))
    write_csv(out_dir / "interpolation_capacity_metrics.csv", metric_rows)
    write_csv(out_dir / "interpolation_capacity_per_family.csv", family_rows)
    write_csv(out_dir / "artifact_inventory.csv", inventory)
    write_csv(out_dir / "final_model_comparison.csv", final_rows)
    write_json(out_dir / "two_factor_effects.json", effects)
    summary = {
        "schema_version": "benchmark_v2_interpolation_capacity_summary/2",
        "created_at_utc": now(),
        "matrix": list(CNN_MATRIX_NAMES),
        "available_models": sorted({row["model"] for row in metric_rows}),
        "pending_models": sorted(
            model
            for model in CNN_MATRIX_NAMES
            if not all(
                any(
                    row["model"] == model and row["protocol"] == protocol
                    for row in metric_rows
                )
                for protocol in VALIDATION_PROTOCOLS
            )
        ),
        "validation_status": validation_payload["status"],
        "primary_test_included": include_primary_test,
        "primary_test_used_for_selection": False,
        "predetermined_before_primary_test": list(CNN_MATRIX_NAMES),
        "two_factor_effects": effects,
        "validation_interpretation": validation_payload.get("interpretation", {}),
    }
    write_json(out_dir / "interpolation_capacity_summary.json", summary)
    write_report(out_dir / "interpolation_capacity_report.md", summary, final_rows)
    write_plots(out_dir, metric_rows, family_rows)
    return summary


def resolve_protocol(model_spec: Mapping[str, Any], protocol: str) -> Path | None:
    root = Path(model_spec["root"])
    evaluation_dir = model_spec.get("evaluation_dir")
    if evaluation_dir:
        candidate = root / str(evaluation_dir) / protocol
        return candidate if candidate.is_dir() else None
    try:
        return locate_protocol_dir(root, protocol)
    except (FileNotFoundError, ValueError):
        return None


def build_validation_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {(row["model"], row["protocol"]): row for row in rows}
    required = [
        (model, protocol)
        for model in CNN_MATRIX_NAMES
        for protocol in VALIDATION_PROTOCOLS
    ]
    missing = [list(key) for key in required if key not in lookup]
    selected = [lookup[key] for key in required if key in lookup]
    fingerprint = stable_hash(
        [
            {
                "model": row["model"],
                "protocol": row["protocol"],
                "metrics_sha256": row["metrics_sha256"],
                "sample_metrics_sha256": row["sample_metrics_sha256"],
            }
            for row in selected
        ]
    )
    payload: dict[str, Any] = {
        "schema_version": "benchmark_v2_interpolation_validation_gate/1",
        "status": "pending" if missing else "ready_to_freeze",
        "required_entries": [list(key) for key in required],
        "missing_entries": missing,
        "input_fingerprint": fingerprint,
        "selection_protocols": list(VALIDATION_PROTOCOLS),
        "primary_test_used": False,
        "predetermined_primary_test_entries": list(CNN_MATRIX_NAMES),
    }
    if not missing:
        payload["interpretation"] = validation_interpretation(lookup)
        payload["validation_metrics"] = {
            model: {
                protocol: float(lookup[(model, protocol)]["micro_mae_K"])
                for protocol in VALIDATION_PROTOCOLS
            }
            for model in CNN_MATRIX_NAMES
        }
    return payload


def validation_interpretation(
    lookup: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    effects = compute_two_factor_effects(list(lookup.values()))
    known_best = min(
        CNN_MATRIX_NAMES,
        key=lambda model: float(
            lookup[(model, "known_family_sample_test")]["micro_mae_K"]
        ),
    )
    validation_best = min(
        CNN_MATRIX_NAMES,
        key=lambda model: float(
            lookup[(model, "primary_validation_families")]["micro_mae_K"]
        ),
    )
    efficiency = min(
        CNN_MATRIX_NAMES,
        key=lambda model: (
            float(lookup[(model, "known_family_sample_test")]["micro_mae_K"])
            * int(lookup[(model, "known_family_sample_test")]["parameter_count"])
        ),
    )
    recipe_validation_deltas = {
        "small_epoch100_K": float(
            lookup[
                ("small_cosine_ema_epoch100", "primary_validation_families")
            ]["micro_mae_K"]
        )
        - float(
            lookup[
                ("canonical_small_constant", "primary_validation_families")
            ]["micro_mae_K"]
        ),
        "wide_epoch100_K": float(
            lookup[
                ("wide_cosine_ema_epoch100", "primary_validation_families")
            ]["micro_mae_K"]
        )
        - float(
            lookup[
                ("wide_constant_epoch100", "primary_validation_families")
            ]["micro_mae_K"]
        ),
    }
    operator_known = {
        operator: float(
            lookup[(operator, "known_family_sample_test")]["micro_mae_K"]
        )
        for operator in OPERATOR_NAMES
        if (operator, "known_family_sample_test") in lookup
    }
    wide_known = min(
        float(
            lookup[(model, "known_family_sample_test")]["micro_mae_K"]
        )
        for model in (
            "wide_constant_epoch100",
            "wide_cosine_ema_epoch100",
            "wide_cosine_ema_epoch150",
        )
    )
    return {
        "best_known_family_interpolation": known_best,
        "best_heldout_validation": validation_best,
        "best_parameter_efficiency_proxy": efficiency,
        "practical_tradeoff": pareto_models(lookup),
        "width_and_recipe_effects": effects,
        "cosine_ema_validation_deltas_K": recipe_validation_deltas,
        "cosine_ema_systematically_worsens_validation_at_epoch100": all(
            delta > 0.0 for delta in recipe_validation_deltas.values()
        ),
        "best_wide_known_family_mae_K": wide_known,
        "known_family_operator_gap_K": {
            operator: wide_known - value
            for operator, value in operator_known.items()
        },
        "operator_gap_is_reported_but_not_used_to_select_cnn_training": True,
    }


def pareto_models(
    lookup: Mapping[tuple[str, str], Mapping[str, Any]]
) -> list[str]:
    points = {
        model: (
            float(lookup[(model, "known_family_sample_test")]["micro_mae_K"]),
            float(lookup[(model, "primary_validation_families")]["micro_mae_K"]),
        )
        for model in CNN_MATRIX_NAMES
    }
    return [
        model
        for model, point in points.items()
        if not any(
            other != model
            and other_point[0] <= point[0]
            and other_point[1] <= point[1]
            and other_point != point
            for other, other_point in points.items()
        )
    ]


def build_primary_gate(
    *,
    include_primary_test: bool,
    validation_gate: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    available = sorted(
        row["model"]
        for row in metric_rows
        if row["protocol"] == "primary_test_families"
    )
    return {
        "schema_version": "benchmark_v2_interpolation_primary_test_gate/1",
        "status": "included" if include_primary_test else "closed",
        "validation_gate_status": validation_gate["status"],
        "validation_input_fingerprint": validation_gate.get("input_fingerprint"),
        "primary_test_used_for_selection": False,
        "predetermined_entries": list(CNN_MATRIX_NAMES),
        "available_primary_test_models": available,
    }


def compute_two_factor_effects(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {(row["model"], row["protocol"]): row for row in rows}
    output: dict[str, Any] = {
        "sign_convention": "negative MAE delta means the second factor improves",
        "epoch100": {},
        "bounded_epoch150": {},
    }
    names = {
        "small_constant": "canonical_small_constant",
        "small_cosine": "small_cosine_ema_epoch100",
        "wide_constant": "wide_constant_epoch100",
        "wide_cosine": "wide_cosine_ema_epoch100",
    }
    for protocol in PROTOCOLS:
        if all((model, protocol) in lookup for model in names.values()):
            value = {
                key: float(lookup[(model, protocol)]["micro_mae_K"])
                for key, model in names.items()
            }
            output["epoch100"][protocol] = {
                "width_effect_constant_K": value["wide_constant"]
                - value["small_constant"],
                "width_effect_cosine_ema_K": value["wide_cosine"]
                - value["small_cosine"],
                "recipe_effect_small_K": value["small_cosine"]
                - value["small_constant"],
                "recipe_effect_wide_K": value["wide_cosine"]
                - value["wide_constant"],
                "width_recipe_interaction_K": (
                    value["wide_cosine"] - value["small_cosine"]
                )
                - (value["wide_constant"] - value["small_constant"]),
            }
        bounded = (
            "small_cosine_ema_epoch150",
            "wide_cosine_ema_epoch150",
        )
        if all((model, protocol) in lookup for model in bounded):
            output["bounded_epoch150"][protocol] = {
                "width_effect_cosine_ema_K": float(
                    lookup[(bounded[1], protocol)]["micro_mae_K"]
                )
                - float(lookup[(bounded[0], protocol)]["micro_mae_K"])
            }
    return output


def build_final_comparison(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["protocol"]): row for row in rows}
    output = []
    for model_spec in specs:
        name = model_spec["model"]
        if name.startswith("wide_cosine_raw"):
            continue
        available = {
            protocol: lookup.get((name, protocol)) for protocol in PROTOCOLS
        }
        parameter_count = next(
            (
                int(row["parameter_count"])
                for row in available.values()
                if row is not None
            ),
            "",
        )
        training = read_training_state(model_spec)
        known = available["known_family_sample_test"]
        validation = available["primary_validation_families"]
        primary = available["primary_test_families"]
        output.append(
            {
                "model_name": name,
                "architecture_family": model_spec["architecture_family"],
                "parameter_count": parameter_count,
                "epoch_checkpoint": model_spec["checkpoint"],
                "training_recipe": model_spec["training_recipe"],
                "scheduler": model_spec["scheduler"],
                "ema_or_raw": model_spec["weights"],
                "optimizer_steps": training["optimizer_steps"],
                "wall_clock_training_time_s": training["wall_clock_training_time_s"],
                "known_family_mae_K": metric(known, "micro_mae_K"),
                "heldout_validation_mae_K": metric(validation, "micro_mae_K"),
                "heldout_primary_test_mae_K": metric(primary, "micro_mae_K"),
                "known_family_runtime_per_sample_s": metric(
                    known, "runtime_per_sample_s"
                ),
                "heldout_runtime_per_sample_s": metric(
                    primary or validation, "runtime_per_sample_s"
                ),
                "fraction_worse_than_source": metric(
                    known, "fraction_worse_than_source"
                ),
                "centered_field_mae_K": metric(known, "centered_field_mae_K"),
                "mean_correction_mae_K": metric(
                    known, "mean_correction_mae_K"
                ),
                "hotspot_top1pct_mae_K": metric(
                    known, "hotspot_top1pct_mae_K"
                ),
            }
        )
    return output


def read_training_state(model_spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(model_spec["root"])
    epoch = int(model_spec["epoch"])
    log_path = root / "train_log.csv"
    wall_time: float | str = ""
    if log_path.is_file():
        rows = read_csv(log_path)
        times = [
            float(row["epoch_runtime_s"])
            for row in rows
            if int(row["epoch"]) <= epoch and row.get("epoch_runtime_s")
        ]
        wall_time = sum(times) if times else ""
    optimizer_steps: int | str = ""
    checkpoint = root / "checkpoints" / str(model_spec["checkpoint"])
    if checkpoint.is_file():
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            optimizer_steps = int(payload.get("global_optimizer_step", -1))
        except (ImportError, OSError, RuntimeError, ValueError):
            optimizer_steps = ""
    return {
        "optimizer_steps": optimizer_steps,
        "wall_clock_training_time_s": wall_time,
    }


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


def metric(row: Mapping[str, Any] | None, name: str) -> Any:
    return "" if row is None else row.get(name, "")


def parameter_count_from_metrics(metrics: Mapping[str, Any]) -> int:
    return int(metrics.get("model", {}).get("parameter_count", 0))


def runtime_from_metrics(metrics: Mapping[str, Any]) -> float:
    return float(
        metrics.get(
            "inference_runtime_per_sample_s",
            metrics.get("runtime_per_sample_s", float("nan")),
        )
    )


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
    primary_rows = [
        row
        for row in rows
        if row["model"] in (*CNN_MATRIX_NAMES, *OPERATOR_NAMES)
    ]
    protocol_bar(primary_rows, "known_family_sample_test", out_dir / "known_family_mae_comparison.png")
    protocol_bar(primary_rows, "primary_validation_families", out_dir / "heldout_validation_mae_comparison.png")
    protocol_bar(primary_rows, "primary_test_families", out_dir / "heldout_primary_test_mae_comparison.png")
    interaction_plot(primary_rows, "known_family_sample_test", out_dir / "width_recipe_interaction_known_family.png")
    interaction_plot(primary_rows, "primary_validation_families", out_dir / "width_recipe_interaction_validation.png")
    interaction_plot(primary_rows, "primary_test_families", out_dir / "width_recipe_interaction_primary_test.png")
    paired_scatter(primary_rows, "known_family_sample_test", "primary_validation_families", out_dir / "interpolation_vs_validation.png")
    paired_scatter(primary_rows, "known_family_sample_test", "primary_test_families", out_dir / "interpolation_vs_primary_test.png")
    simple_scatter(primary_rows, "known_family_sample_test", "parameter_count", out_dir / "parameter_count_vs_known_family_mae.png")
    simple_scatter(primary_rows, "primary_test_families", "parameter_count", out_dir / "parameter_count_vs_primary_test_mae.png")
    simple_scatter(primary_rows, "known_family_sample_test", "runtime_per_sample_s", out_dir / "runtime_vs_known_family_mae.png")
    component_plot(primary_rows, out_dir / "centered_and_mean_error_comparison.png")
    family_plot(family_rows, "primary_validation_families", out_dir / "per_family_validation_comparison.png")
    family_plot(family_rows, "primary_test_families", out_dir / "per_family_primary_test_comparison.png")
    plt.close("all")


def protocol_bar(rows: Sequence[Mapping[str, Any]], protocol: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["protocol"] == protocol]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([row["model"] for row in selected], [row["micro_mae_K"] for row in selected])
    ax.set_ylabel("MAE (K)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def interaction_plot(rows: Sequence[Mapping[str, Any]], protocol: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    lookup = {(row["model"], row["protocol"]): row for row in rows}
    groups = (
        ("constant", "canonical_small_constant", "wide_constant_epoch100"),
        ("cosine+EMA", "small_cosine_ema_epoch100", "wide_cosine_ema_epoch100"),
    )
    if not all((name, protocol) in lookup for _, *names in groups for name in names):
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, small, wide in groups:
        ax.plot(
            [2.188803, 3.919642],
            [lookup[(small, protocol)]["micro_mae_K"], lookup[(wide, protocol)]["micro_mae_K"]],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("MAE (K)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def paired_scatter(rows: Sequence[Mapping[str, Any]], x_protocol: str, y_protocol: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    lookup = {(row["model"], row["protocol"]): row for row in rows}
    names = sorted(
        {row["model"] for row in rows}
        & {name for name, protocol in lookup if protocol == x_protocol}
        & {name for name, protocol in lookup if protocol == y_protocol}
    )
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for name in names:
        ax.scatter(lookup[(name, x_protocol)]["micro_mae_K"], lookup[(name, y_protocol)]["micro_mae_K"], label=name)
    ax.set_xlabel("Known-family MAE (K)")
    ax.set_ylabel("Held-out MAE (K)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def simple_scatter(rows: Sequence[Mapping[str, Any]], protocol: str, x_name: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["protocol"] == protocol]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for row in selected:
        ax.scatter(row[x_name], row["micro_mae_K"], label=row["model"])
    ax.set_xlabel(x_name)
    ax.set_ylabel("MAE (K)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def component_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["protocol"] == "known_family_sample_test"]
    if not selected:
        return
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, [row["centered_field_mae_K"] for row in selected], 0.36, label="centered")
    ax.bar(x + 0.18, [row["mean_correction_mae_K"] for row in selected], 0.36, label="mean")
    ax.set_xticks(x, [row["model"] for row in selected], rotation=35)
    ax.set_ylabel("MAE (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def family_plot(rows: Sequence[Mapping[str, Any]], protocol: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["protocol"] == protocol and row["model"] in CNN_MATRIX_NAMES]
    if not selected:
        return
    families = sorted({row["family_uid"] for row in selected})
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in CNN_MATRIX_NAMES:
        by_family = {row["family_uid"]: row["micro_mae_K"] for row in selected if row["model"] == model}
        if by_family:
            ax.plot(families, [by_family.get(family, np.nan) for family in families], marker="o", label=model)
    ax.set_ylabel("Family MAE (K)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Benchmark v2 CNN Interpolation-Capacity Study",
        "",
        "The six CNN settings were predetermined. Validation interpretation uses only "
        "known-family sample test and held-out validation families. Primary test is "
        "never used for training, checkpoint selection, or validation interpretation.",
        "",
        "| Model | Params | Checkpoint | Recipe | Known MAE | Validation MAE | Primary-test MAE |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['parameter_count']} | {row['epoch_checkpoint']} | "
            f"{row['training_recipe']} | {format_cell(row['known_family_mae_K'])} | "
            f"{format_cell(row['heldout_validation_mae_K'])} | "
            f"{format_cell(row['heldout_primary_test_mae_K'])} |"
        )
    lines.extend(
        [
            "",
            "## Fairness Views",
            "",
            "- Equal epoch: four epoch-100 CNN cells plus frozen 100-epoch operators. Equal epochs are not claimed to be equal compute.",
            "- Parameter matched: both wide CNN recipes versus U-FNO and SAU-FNO.",
            "- Best bounded recipe: epoch-150 small/wide cosine+EMA, canonical CNN, and frozen operators.",
            "",
            f"Validation gate status: **{summary['validation_status']}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    return "pending" if value in ("", None) else f"{float(value):.5f}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
