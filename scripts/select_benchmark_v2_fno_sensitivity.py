#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.models import build_model, count_parameters  # noqa: E402


BASELINE_CONFIG = (
    REPO_ROOT
    / "configs/benchmark_v2_50family/training/package_residual_fno_decomposed_seed1.yaml"
)
SELECTION_PROTOCOL = "primary_validation_families"
REFERENCE_PROTOCOL = "known_family_sample_test"
ALLOWED_SENSITIVITY_KEYS = {"fno_width", "fno_modes_x", "fno_modes_y"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight and select controlled plain-FNO sensitivity variants."
    )
    parser.add_argument("--baseline-config", default=BASELINE_CONFIG, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=CONFIG_YAML",
        help="Named config; repeat for the baseline and each sensitivity variant.",
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        metavar="NAME=EVAL_ROOT",
        help=(
            "Optional named evaluation root. When provided for every variant, validation-only "
            "selection is performed."
        ),
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    variants = parse_named_paths(args.variant)
    evaluations = parse_named_paths(args.evaluation)
    if evaluations and set(evaluations) != set(variants):
        raise SystemExit(
            "--evaluation names must exactly match --variant names; "
            f"variants={sorted(variants)}, evaluations={sorted(evaluations)}"
        )

    baseline = load_yaml(args.baseline_config)
    capacity_rows = build_capacity_rows(
        variants,
        baseline=baseline,
        batch_size=args.batch_size,
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fno_sensitivity_capacity.csv", capacity_rows)

    validation_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] | None = None
    if evaluations:
        validation_rows = load_validation_rows(evaluations, capacity_rows)
        write_csv(out_dir / "fno_sensitivity_validation.csv", validation_rows)
        selection = select_variant(validation_rows)
        (out_dir / "fno_sensitivity_selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "selected_variant.txt").write_text(
            str(selection["selected_variant"]) + "\n",
            encoding="utf-8",
        )
    report = render_report(capacity_rows, validation_rows, selection)
    (out_dir / "fno_sensitivity_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def build_capacity_rows(
    variants: dict[str, Path],
    *,
    baseline: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in variants.items():
        config = load_yaml(path)
        validate_sensitivity_config(config, baseline, name=name)
        model = build_model(
            {
                "architecture": config["model_architecture"],
                "input_channels": 34,
                "metadata_dim": 15,
                "metadata_hidden_dim": config["metadata_hidden_dim"],
                "metadata_embedding_dim": config["metadata_embedding_dim"],
                "fno_capacity_profile": config["fno_capacity_profile"],
                "fno_width": config["fno_width"],
                "fno_layers": config["fno_layers"],
                "fno_modes_x": config["fno_modes_x"],
                "fno_modes_y": config["fno_modes_y"],
                "fno_activation": config["fno_activation"],
                "fno_projection_channels": config["fno_projection_channels"],
            }
        )
        parameters = count_parameters(model)
        memory = estimate_memory(config, batch_size=batch_size, parameter_count=parameters)
        rows.append(
            {
                "variant": name,
                "config": str(path.expanduser().resolve()),
                "fno_width": int(config["fno_width"]),
                "fno_modes_x": int(config["fno_modes_x"]),
                "fno_modes_y": int(config["fno_modes_y"]),
                "fno_layers": int(config["fno_layers"]),
                "projection_channels": int(config["fno_projection_channels"]),
                "batch_size": int(config["batch_size"]),
                "parameter_count": parameters,
                **memory,
            }
        )
    return sorted(rows, key=lambda row: str(row["variant"]))


def validate_sensitivity_config(
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    name: str,
) -> None:
    if set(config) != set(baseline):
        raise ValueError(
            f"{name} config schema differs from baseline: "
            f"missing={sorted(set(baseline) - set(config))}, "
            f"extra={sorted(set(config) - set(baseline))}"
        )
    changed = {
        key
        for key in baseline
        if config[key] != baseline[key]
    }
    unexpected = changed - ALLOWED_SENSITIVITY_KEYS
    if unexpected:
        raise ValueError(
            f"{name} changes settings outside controlled width/mode sensitivity: "
            f"{sorted(unexpected)}"
        )
    if int(config["fno_layers"]) != 4:
        raise ValueError(f"{name} must retain four FNO layers")
    if int(config["fno_modes_x"]) != int(config["fno_modes_y"]):
        raise ValueError(f"{name} must use equal x/y retained modes")
    if config["prediction_mode"] != "residual_decomposed_fno":
        raise ValueError(f"{name} is not a residual-decomposed FNO")
    if config["physics_input"] != "source_superposition_v1":
        raise ValueError(f"{name} does not use the canonical source-superposition base")


def estimate_memory(
    config: dict[str, Any],
    *,
    batch_size: int,
    parameter_count: int,
) -> dict[str, int]:
    width = int(config["fno_width"])
    layers = int(config["fno_layers"])
    height = width_pixels = 64
    spatial_bytes = batch_size * width * height * width_pixels * 4
    rfft_complex_bytes = batch_size * width * height * (width_pixels // 2 + 1) * 8
    forward_activation_bytes = spatial_bytes * (layers + 2) + 2 * rfft_complex_bytes
    # Lower-bound estimate: retained forward activations plus gradients and FP32 AdamW state.
    training_lower_bound_bytes = 3 * forward_activation_bytes + 16 * parameter_count
    return {
        "estimated_forward_activation_bytes": int(forward_activation_bytes),
        "estimated_forward_activation_MiB": int(round(forward_activation_bytes / 2**20)),
        "estimated_fp32_adam_training_lower_bound_bytes": int(training_lower_bound_bytes),
        "estimated_fp32_adam_training_lower_bound_MiB": int(
            round(training_lower_bound_bytes / 2**20)
        ),
    }


def load_validation_rows(
    evaluations: dict[str, Path],
    capacity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    capacity = {str(row["variant"]): row for row in capacity_rows}
    rows: list[dict[str, Any]] = []
    for name, root in evaluations.items():
        validation_path = root / SELECTION_PROTOCOL / "metrics.json"
        reference_path = root / REFERENCE_PROTOCOL / "metrics.json"
        validation = load_json(validation_path)
        reference = load_json(reference_path)
        ensure_residual_fno_metrics(validation, name=name, protocol=SELECTION_PROTOCOL)
        ensure_residual_fno_metrics(reference, name=name, protocol=REFERENCE_PROTOCOL)
        validation_final = final_metrics(validation)
        reference_final = final_metrics(reference)
        runtime = validation.get("inference_runtime_per_sample_s")
        if runtime is None:
            raise ValueError(f"{name}/{SELECTION_PROTOCOL} is missing runtime per sample")
        parameters = validation.get("model", {}).get("parameter_count")
        expected_parameters = int(capacity[name]["parameter_count"])
        if parameters is not None and int(parameters) != expected_parameters:
            raise ValueError(
                f"{name} evaluation parameter count {parameters} does not match "
                f"config-derived count {expected_parameters}"
            )
        rows.append(
            {
                "variant": name,
                "known_family_mae_K": float(reference_final["mae_K"]),
                "known_family_rmse_K": float(reference_final["rmse_K"]),
                "validation_family_mae_K": float(validation_final["mae_K"]),
                "validation_family_rmse_K": float(validation_final["rmse_K"]),
                "validation_runtime_per_sample_s": float(runtime),
                "parameter_count": expected_parameters,
                "selection_uses_primary_test": False,
                "selection_protocol": SELECTION_PROTOCOL,
            }
        )
    return sorted(rows, key=lambda row: str(row["variant"]))


def select_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty sensitivity table")
    ranking = sorted(
        rows,
        key=lambda row: (
            float(row["validation_family_mae_K"]),
            float(row["validation_family_rmse_K"]),
            float(row["validation_runtime_per_sample_s"]),
            int(row["parameter_count"]),
            str(row["variant"]),
        ),
    )
    return {
        "schema_version": "benchmark_v2_fno_sensitivity_selection/1",
        "selected_variant": ranking[0]["variant"],
        "selection_protocol": SELECTION_PROTOCOL,
        "primary_criterion": "held-out validation-family MAE",
        "secondary_criterion": "held-out validation-family RMSE",
        "tie_breakers": ["runtime per sample", "parameter count", "variant name"],
        "primary_test_family_metrics_used": False,
        "ranking": ranking,
    }


def ensure_residual_fno_metrics(
    metrics: dict[str, Any],
    *,
    name: str,
    protocol: str,
) -> None:
    mode = metrics.get("model", {}).get("prediction_mode")
    architecture = metrics.get("model", {}).get("config", {}).get("architecture")
    if mode != "residual_decomposed_fno":
        raise ValueError(f"{name}/{protocol} has incompatible prediction_mode={mode!r}")
    if architecture != "fno2d_residual_decomposed_conditioned":
        raise ValueError(f"{name}/{protocol} has incompatible architecture={architecture!r}")


def final_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    final = metrics.get("cnn_final_temperature") or metrics.get("final_temperature")
    if not isinstance(final, dict) or final.get("mae_K") is None or final.get("rmse_K") is None:
        raise ValueError("evaluation metrics are missing final-temperature MAE/RMSE")
    return final


def render_report(
    capacity_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    selection: dict[str, Any] | None,
) -> str:
    lines = [
        "# Plain-FNO Sensitivity Study",
        "",
        "All variants use the canonical source-superposition residual formulation and differ "
        "only in width and retained Fourier modes.",
        "",
        "## Capacity Preflight",
        "",
        "| Variant | Width | Modes | Layers | Parameters | Forward MiB | Training lower-bound MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in capacity_rows:
        lines.append(
            f"| {row['variant']} | {row['fno_width']} | {row['fno_modes_x']} | "
            f"{row['fno_layers']} | {row['parameter_count']:,} | "
            f"{row['estimated_forward_activation_MiB']} | "
            f"{row['estimated_fp32_adam_training_lower_bound_MiB']} |"
        )
    if validation_rows:
        lines.extend(
            [
                "",
                "## Validation Selection",
                "",
                "| Variant | Known MAE K | Validation MAE K | Validation RMSE K | Runtime ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in validation_rows:
            lines.append(
                f"| {row['variant']} | {row['known_family_mae_K']:.4f} | "
                f"{row['validation_family_mae_K']:.4f} | "
                f"{row['validation_family_rmse_K']:.4f} | "
                f"{1000.0 * row['validation_runtime_per_sample_s']:.3f} |"
            )
        assert selection is not None
        lines.extend(
            [
                "",
                f"Selected: **{selection['selected_variant']}**.",
                "",
                "Primary test-family metrics were not read or used during selection.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No evaluation roots were supplied. This is a pre-training capacity report only.",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in output:
            raise ValueError(f"invalid or duplicate named path: {value!r}")
        output[name] = Path(raw_path).expanduser().resolve()
    return output


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
