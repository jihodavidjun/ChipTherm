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

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.ufno_models import UFNO_REFERENCE_COMMIT  # noqa: E402


CORRESPONDENCE_PATH = REPO_ROOT / "docs/ufno_architecture_correspondence.md"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the controlled Benchmark v2 U-FNO comparison."
    )
    parser.add_argument("--direct-train-index", required=True, type=Path)
    parser.add_argument("--residual-train-index", required=True, type=Path)
    parser.add_argument("--direct-fno-config", required=True, type=Path)
    parser.add_argument("--residual-fno-config", required=True, type=Path)
    parser.add_argument("--direct-ufno-config", required=True, type=Path)
    parser.add_argument("--residual-ufno-config", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--direct-cnn-checkpoint", type=Path)
    parser.add_argument("--residual-cnn-checkpoint", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build_fairness_report(
        direct_index=args.direct_train_index,
        residual_index=args.residual_train_index,
        direct_fno_config=load_yaml(args.direct_fno_config),
        residual_fno_config=load_yaml(args.residual_fno_config),
        direct_ufno_config=load_yaml(args.direct_ufno_config),
        residual_ufno_config=load_yaml(args.residual_ufno_config),
        batch_size=args.batch_size,
        direct_cnn_checkpoint=args.direct_cnn_checkpoint,
        residual_cnn_checkpoint=args.residual_cnn_checkpoint,
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ufno_fairness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "ufno_fairness_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    if not all(report["checks"].values()):
        raise SystemExit("U-FNO fairness audit failed")
    return 0


def build_fairness_report(
    *,
    direct_index: Path,
    residual_index: Path,
    direct_fno_config: dict[str, Any],
    residual_fno_config: dict[str, Any],
    direct_ufno_config: dict[str, Any],
    residual_ufno_config: dict[str, Any],
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
    direct_channels = list(direct_dataset.channel_names)
    residual_channels = list(residual_dataset.channel_names)
    metadata_names_direct = list(direct_dataset.metadata_feature_names)
    metadata_names_residual = list(residual_dataset.metadata_feature_names)
    direct_fno = build_operator(
        direct_fno_config, len(direct_channels), len(metadata_names_direct)
    )
    residual_fno = build_operator(
        residual_fno_config, len(residual_channels) + 1, len(metadata_names_residual)
    )
    direct_ufno = build_operator(
        direct_ufno_config, len(direct_channels), len(metadata_names_direct)
    )
    residual_ufno = build_operator(
        residual_ufno_config, len(residual_channels) + 1, len(metadata_names_residual)
    )
    source_tokens = ("source_superposition", "isolated_source", "source_response")
    checks = {
        "identical_direct_residual_sample_membership": direct_uids == residual_uids,
        "exact_33_direct_channels": len(direct_channels) == 33,
        "direct_excludes_source_derived_inputs": not any(
            any(token in name.lower() for token in source_tokens)
            for name in direct_channels
        ),
        "residual_source_base_included_once": (
            len(residual_channels) == 33
            and residual_ufno_config.get("physics_input") == "source_superposition_v1"
        ),
        "metadata_features_identical": (
            metadata_names_direct == metadata_names_residual
            and len(metadata_names_direct) == 15
        ),
        "direct_target_contract_identical": (
            direct_fno_config.get("direct_target_normalization")
            == direct_ufno_config.get("direct_target_normalization")
            == "train_standard"
        ),
        "residual_reconstruction_identical": (
            residual_fno_config.get("mean_head_mode")
            == residual_ufno_config.get("mean_head_mode")
            == "residual_resistance"
        ),
        "losses_identical_by_formulation": (
            direct_fno_config.get("loss") == direct_ufno_config.get("loss")
            and float(residual_fno_config.get("lambda_final", -1))
            == float(residual_ufno_config.get("lambda_final", -2))
            and float(residual_fno_config.get("lambda_mean", -1))
            == float(residual_ufno_config.get("lambda_mean", -2))
        ),
        "controlled_width_modes_projection": all(
            int(config[key]) == expected
            for config in (direct_ufno_config, residual_ufno_config)
            for key, expected in (
                ("fno_width", 32),
                ("fno_modes_x", 12),
                ("fno_modes_y", 12),
                ("fno_projection_channels", 64),
            )
        ),
        "published_branch_placement": all(
            int(config["fno_layers"]) == 6
            and list(config["ufno_unet_branch_indices"]) == [3, 4, 5]
            and int(config["ufno_unet_depth"]) == 3
            for config in (direct_ufno_config, residual_ufno_config)
        ),
        "reference_commit_recorded": all(
            config.get("ufno_reference_commit") == UFNO_REFERENCE_COMMIT
            for config in (direct_ufno_config, residual_ufno_config)
        ),
        "architecture_correspondence_exists": CORRESPONDENCE_PATH.is_file(),
        "no_target_derived_descriptor": not any(
            "target" in name.lower() or "hotspot" in name.lower()
            for name in direct_channels + metadata_names_direct
        ),
    }
    parameter_counts = {
        "direct_fno": count_parameters(direct_fno),
        "direct_ufno": count_parameters(direct_ufno),
        "residual_fno": count_parameters(residual_fno),
        "residual_ufno": count_parameters(residual_ufno),
        "direct_cnn": checkpoint_parameter_count(direct_cnn_checkpoint),
        "residual_cnn": checkpoint_parameter_count(residual_cnn_checkpoint),
    }
    memory = {
        "direct_fno": activation_estimate(direct_fno_config, batch_size, unet_branches=0),
        "direct_ufno": activation_estimate(
            direct_ufno_config, batch_size, unet_branches=3
        ),
        "residual_fno": activation_estimate(
            residual_fno_config, batch_size, unet_branches=0
        ),
        "residual_ufno": activation_estimate(
            residual_ufno_config, batch_size, unet_branches=3
        ),
    }
    return {
        "schema_version": "benchmark_v2_ufno_fairness/1",
        "reference": {
            "commit": UFNO_REFERENCE_COMMIT,
            "correspondence_report": str(CORRESPONDENCE_PATH.relative_to(REPO_ROOT)),
            "adaptation": "task-adapted published U-FNO",
        },
        "checks": checks,
        "sample_counts": {"direct": len(direct_rows), "residual": len(residual_rows)},
        "spatial_channels": {
            "direct": direct_channels,
            "residual_pre_base": residual_channels,
            "residual_effective": residual_channels + ["source_superposition_base_K"],
        },
        "metadata_feature_names": metadata_names_direct,
        "parameter_counts": parameter_counts,
        "parameter_increase_percent": {
            "direct_ufno_over_fno": percent_increase(
                parameter_counts["direct_ufno"], parameter_counts["direct_fno"]
            ),
            "residual_ufno_over_fno": percent_increase(
                parameter_counts["residual_ufno"], parameter_counts["residual_fno"]
            ),
            "direct_ufno_over_cnn": optional_percent_increase(
                parameter_counts["direct_ufno"], parameter_counts["direct_cnn"]
            ),
            "residual_ufno_over_cnn": optional_percent_increase(
                parameter_counts["residual_ufno"], parameter_counts["residual_cnn"]
            ),
        },
        "peak_activation_memory_estimate_bytes": memory,
        "runtime_expectation": (
            "U-FNO adds three three-level convolutional encoder-decoder branches. "
            "Measure runtime and peak CUDA memory on GT; the static estimate excludes "
            "FFT workspaces and autograd allocator overhead."
        ),
        "reconstruction": (
            "source_superposition_base_K + total_power_W * "
            "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
        ),
    }


def build_operator(
    config: dict[str, Any], input_channels: int, metadata_dim: int
):
    payload = {
        "architecture": config["model_architecture"],
        "input_channels": input_channels,
        "output_channels": 1,
        "metadata_dim": metadata_dim,
        "metadata_hidden_dim": config["metadata_hidden_dim"],
        "metadata_embedding_dim": config["metadata_embedding_dim"],
        "fno_capacity_profile": config.get("fno_capacity_profile", "fno_small"),
        "fno_width": config["fno_width"],
        "fno_layers": config["fno_layers"],
        "fno_modes_x": config["fno_modes_x"],
        "fno_modes_y": config["fno_modes_y"],
        "fno_activation": config["fno_activation"],
        "fno_projection_channels": config["fno_projection_channels"],
        "target_normalization_mode": config.get(
            "direct_target_normalization", "train_standard"
        ),
        "target_std_K": 1.0,
        "ufno_adaptation_profile": config.get(
            "ufno_adaptation_profile", "ufno_published_adapted"
        ),
        "ufno_unet_branch_indices": config.get(
            "ufno_unet_branch_indices", [3, 4, 5]
        ),
        "ufno_unet_depth": config.get("ufno_unet_depth", 3),
        "ufno_unet_dropout": config.get("ufno_unet_dropout", 0.0),
        "ufno_domain_padding": config.get("ufno_domain_padding", 8),
        "ufno_padding_mode": config.get("ufno_padding_mode", "published_mixed"),
    }
    return build_model(payload)


def activation_estimate(
    config: dict[str, Any],
    batch_size: int,
    *,
    unet_branches: int,
) -> int:
    width = int(config["fno_width"])
    layers = int(config["fno_layers"])
    full = batch_size * width * 72 * 72
    operator_features = full * (layers + 2)
    # Per U-Net branch: encoder and decoder outputs at 1/2, 1/4, and 1/8
    # resolution plus concatenation tensors. This is a forward-activation estimate.
    unet_features = unet_branches * full * (2.0 + 1.0 + 0.5 + 0.25)
    return int((operator_features + unet_features) * 4)


def percent_increase(value: int, reference: int) -> float:
    return 100.0 * (value - reference) / reference


def optional_percent_increase(value: int, reference: int | None) -> float | None:
    return None if reference in {None, 0} else percent_increase(value, int(reference))


def checkpoint_parameter_count(path: Path | None) -> int | None:
    if path is None:
        return None
    checkpoint = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    configured = checkpoint.get("model_config", {}).get("total_parameters")
    if configured is not None:
        return int(configured)
    return int(sum(value.numel() for value in checkpoint["model_state_dict"].values()))


def sample_uid(row: dict[str, str]) -> str:
    uid = row.get("sample_uid") or row.get("uid")
    if not uid:
        raise ValueError("index row is missing sample_uid")
    return str(uid)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["parameter_counts"]
    lines = [
        "# Benchmark v2 U-FNO Fairness",
        "",
        f"- Reference commit: `{report['reference']['commit']}`",
        f"- Direct samples: {report['sample_counts']['direct']}",
        f"- Residual samples: {report['sample_counts']['residual']}",
        f"- Direct FNO parameters: {counts['direct_fno']:,}",
        f"- Direct U-FNO parameters: {counts['direct_ufno']:,}",
        f"- Residual FNO parameters: {counts['residual_fno']:,}",
        f"- Residual U-FNO parameters: {counts['residual_ufno']:,}",
        f"- Direct CNN parameters: {counts['direct_cnn'] or 'not supplied'}",
        f"- Residual CNN parameters: {counts['residual_cnn'] or 'not supplied'}",
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
            "## Controlled difference",
            "",
            "Width, retained modes, projection width, metadata FiLM, data membership, "
            "normalization, targets, losses, and reconstruction are matched. U-FNO uses "
            "the published three plain Fourier plus three U-Fourier topology; the "
            "three U-Net branches are the intended architectural difference.",
            "",
            "Residual reconstruction:",
            "",
            f"`{report['reconstruction']}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
