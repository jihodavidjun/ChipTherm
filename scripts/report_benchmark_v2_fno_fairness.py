#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit controlled Benchmark v2 FNO fairness.")
    parser.add_argument("--direct-train-index", required=True, type=Path)
    parser.add_argument("--residual-train-index", required=True, type=Path)
    parser.add_argument("--direct-config", required=True, type=Path)
    parser.add_argument("--residual-config", required=True, type=Path)
    parser.add_argument("--direct-cnn-checkpoint", type=Path)
    parser.add_argument("--residual-cnn-checkpoint", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build_fairness_report(
        args.direct_train_index,
        args.residual_train_index,
        args.direct_config,
        args.residual_config,
        batch_size=args.batch_size,
        direct_cnn_checkpoint=args.direct_cnn_checkpoint,
        residual_cnn_checkpoint=args.residual_cnn_checkpoint,
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fno_fairness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "fno_fairness_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    if not all(report["checks"].values()):
        raise SystemExit("FNO fairness audit failed")
    return 0


def build_fairness_report(
    direct_index: Path,
    residual_index: Path,
    direct_config_path: Path,
    residual_config_path: Path,
    *,
    batch_size: int,
    direct_cnn_checkpoint: Path | None = None,
    residual_cnn_checkpoint: Path | None = None,
) -> dict[str, Any]:
    direct_rows = read_csv(direct_index)
    residual_rows = read_csv(residual_index)
    direct_uids = [sample_uid(row) for row in direct_rows]
    residual_uids = [sample_uid(row) for row in residual_rows]
    direct_dataset = ChipThermDataset(
        direct_index,
        target="temperature",
        return_metadata=True,
        physical_representation="dimensional",
    )
    residual_dataset = ChipThermDataset(
        residual_index,
        target="residual",
        return_metadata=True,
        physical_representation="dimensional",
    )
    direct_config = yaml.safe_load(direct_config_path.read_text(encoding="utf-8"))
    residual_config = yaml.safe_load(residual_config_path.read_text(encoding="utf-8"))
    direct_channels = list(direct_dataset.channel_names)
    residual_channels = list(residual_dataset.channel_names)
    direct_fno = build_fno(direct_config, len(direct_channels), len(direct_dataset.metadata_feature_names))
    residual_fno = build_fno(
        residual_config,
        len(residual_channels) + 1,
        len(residual_dataset.metadata_feature_names),
    )
    direct_parameters = count_parameters(direct_fno)
    residual_parameters = count_parameters(residual_fno)
    direct_cnn_parameters = checkpoint_parameter_count(direct_cnn_checkpoint)
    residual_cnn_parameters = checkpoint_parameter_count(residual_cnn_checkpoint)
    source_tokens = ("source_superposition", "isolated_source", "source_response")
    checks = {
        "identical_train_sample_membership": direct_uids == residual_uids,
        "direct_has_33_spatial_channels": len(direct_channels) == 33,
        "residual_has_33_plus_source_channels": len(residual_channels) == 33,
        "direct_excludes_source_derived_channels": not any(
            any(token in name.lower() for token in source_tokens) for name in direct_channels
        ),
        "direct_physics_input_is_none": direct_config.get("physics_input") == "none",
        "direct_target_normalization_is_train_only": (
            direct_config.get("direct_target_normalization") == "train_standard"
        ),
        "residual_uses_source_base": (
            residual_config.get("physics_input") == "source_superposition_v1"
        ),
        "residual_uses_resistance_mean": (
            residual_config.get("mean_head_mode") == "residual_resistance"
        ),
        "metadata_dimension_is_15": (
            len(direct_dataset.metadata_feature_names) == 15
            and len(residual_dataset.metadata_feature_names) == 15
        ),
    }
    return {
        "schema_version": "benchmark_v2_fno_fairness/1",
        "checks": checks,
        "sample_counts": {"direct": len(direct_rows), "residual": len(residual_rows)},
        "stage1": {
            "prediction_mode": "direct_temperature_fno",
            "spatial_input_channels": direct_channels,
            "model_input_channels": len(direct_channels),
            "excluded_source_inputs": list(source_tokens),
            "target": "T_norm = (HotSpot_K - train_mean_K) / train_std_K",
            "target_normalization_fit_scope": "train split only",
            "parameters": direct_parameters,
            "estimated_activation_memory_bytes_per_batch": activation_estimate(
                direct_config, batch_size
            ),
        },
        "stage2": {
            "prediction_mode": "residual_decomposed_fno",
            "spatial_input_channels": residual_channels + ["source_superposition_base_K"],
            "model_input_channels": len(residual_channels) + 1,
            "reconstruction": (
                "source_superposition_base_K + total_power_W * "
                "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
            ),
            "parameters": residual_parameters,
            "estimated_activation_memory_bytes_per_batch": activation_estimate(
                residual_config, batch_size
            ),
        },
        "cnn_reference_parameters": {
            "direct": direct_cnn_parameters,
            "residual": residual_cnn_parameters,
        },
        "parameter_differences_percent": {
            "direct_fno_vs_cnn": percent_difference(direct_parameters, direct_cnn_parameters),
            "residual_fno_vs_cnn": percent_difference(
                residual_parameters, residual_cnn_parameters
            ),
        },
        "expected_training_time": (
            "Measure on GT using epoch_runtime_s; FFT runtime is hardware dependent. "
            "Use the same 100 epochs, batch size, scheduler, and early stopping as CNN controls."
        ),
    }


def build_fno(config: dict[str, Any], input_channels: int, metadata_dim: int) -> torch.nn.Module:
    payload = {
        "architecture": config["model_architecture"],
        "input_channels": input_channels,
        "metadata_dim": metadata_dim,
        "metadata_hidden_dim": config["metadata_hidden_dim"],
        "metadata_embedding_dim": config["metadata_embedding_dim"],
        "fno_capacity_profile": config["fno_capacity_profile"],
        "fno_width": config["fno_width"],
        "fno_layers": config["fno_layers"],
        "fno_modes_x": config["fno_modes_x"],
        "fno_modes_y": config["fno_modes_y"],
        "fno_activation": config["fno_activation"],
        "fno_projection_channels": config["fno_projection_channels"],
        "target_normalization_mode": config.get("direct_target_normalization", "not_applicable"),
        "target_std_K": 1.0,
    }
    return build_model(payload)


def activation_estimate(config: dict[str, Any], batch_size: int) -> int:
    width = int(config["fno_width"])
    layers = int(config["fno_layers"])
    # Forward feature tensors only; training autograd and FFT workspaces add implementation-dependent memory.
    return int(batch_size * width * 64 * 64 * (layers + 2) * 4)


def checkpoint_parameter_count(path: Path | None) -> int | None:
    if path is None:
        return None
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    config_count = checkpoint.get("model_config", {}).get("total_parameters")
    if config_count is not None:
        return int(config_count)
    return int(sum(value.numel() for value in checkpoint["model_state_dict"].values()))


def percent_difference(value: int, reference: int | None) -> float | None:
    if reference in {None, 0}:
        return None
    return 100.0 * (value - int(reference)) / int(reference)


def sample_uid(row: dict[str, str]) -> str:
    value = row.get("sample_uid") or row.get("uid")
    if not value:
        raise ValueError("index row is missing sample_uid")
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_markdown(report: dict[str, Any]) -> str:
    stage1 = report["stage1"]
    stage2 = report["stage2"]
    lines = [
        "# Benchmark v2 FNO Fairness",
        "",
        f"- Stage 1 samples: {report['sample_counts']['direct']}",
        f"- Stage 2 samples: {report['sample_counts']['residual']}",
        f"- Direct FNO parameters: {stage1['parameters']:,}",
        f"- Residual FNO parameters: {stage2['parameters']:,}",
        f"- Stage 1 model inputs: {stage1['model_input_channels']}",
        f"- Stage 2 model inputs: {stage2['model_input_channels']}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in report["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Reconstruction",
            "",
            stage2["reconstruction"],
            "",
            "Stage 1 source-superposition data is used only to align immutable sample membership "
            "and may be used as an offline evaluation reference; it is not a model input.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
