#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
from chiptherm.ml.sau_fno_models import (  # noqa: E402
    SAU_FNO_ADAPTATION_PROFILE,
    attention_memory_estimate,
)
from chiptherm.ml.ufno_models import UFNO_REFERENCE_COMMIT  # noqa: E402


CONFIG_KEYS = (
    "direct_fno",
    "residual_fno",
    "direct_ufno",
    "residual_ufno",
    "direct_sau_fno",
    "residual_sau_fno",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the controlled Benchmark v2 FNO/U-FNO/SAU-FNO comparison."
    )
    parser.add_argument("--direct-train-index", required=True, type=Path)
    parser.add_argument("--residual-train-index", required=True, type=Path)
    for key in CONFIG_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}-config", required=True, type=Path)
    parser.add_argument("--direct-cnn-checkpoint", type=Path)
    parser.add_argument("--residual-cnn-checkpoint", type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    configs = {
        key: load_yaml(getattr(args, f"{key}_config")) for key in CONFIG_KEYS
    }
    report = build_fairness_report(
        direct_index=args.direct_train_index,
        residual_index=args.residual_train_index,
        configs=configs,
        batch_size=args.batch_size,
        direct_cnn_checkpoint=args.direct_cnn_checkpoint,
        residual_cnn_checkpoint=args.residual_cnn_checkpoint,
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "sau_fno_fairness_report.json", report)
    (out_dir / "sau_fno_fairness_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    if not all(report["checks"].values()):
        raise SystemExit("SAU-FNO fairness audit failed")
    return 0


def build_fairness_report(
    *,
    direct_index: Path,
    residual_index: Path,
    configs: dict[str, dict[str, Any]],
    batch_size: int,
    direct_cnn_checkpoint: Path | None = None,
    residual_cnn_checkpoint: Path | None = None,
) -> dict[str, Any]:
    direct_rows = read_csv(direct_index)
    residual_rows = read_csv(residual_index)
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
    direct_metadata = list(direct_dataset.metadata_feature_names)
    residual_metadata = list(residual_dataset.metadata_feature_names)
    models = {
        name: build_operator(
            config,
            input_channels=(
                len(direct_channels)
                if name.startswith("direct")
                else len(residual_channels) + 1
            ),
            metadata_dim=len(direct_metadata),
        )
        for name, config in configs.items()
    }
    direct_configs = [configs[name] for name in CONFIG_KEYS if name.startswith("direct")]
    residual_configs = [
        configs[name] for name in CONFIG_KEYS if name.startswith("residual")
    ]
    controlled_operator_keys = (
        "epochs",
        "batch_size",
        "lr",
        "metadata_hidden_dim",
        "metadata_embedding_dim",
        "fno_width",
        "fno_modes_x",
        "fno_modes_y",
        "fno_projection_channels",
        "scheduler",
        "early_stopping_patience",
        "checkpoint_frequency",
        "seed",
    )
    checks = {
        "identical_train_index_membership": [
            sample_uid(row) for row in direct_rows
        ]
        == [sample_uid(row) for row in residual_rows],
        "identical_train_index_content_hash": sha256_file(direct_index)
        == sha256_file(residual_index),
        "exact_33_direct_channels": len(direct_channels) == 33,
        "exact_34_residual_model_channels": len(residual_channels) + 1 == 34,
        "metadata_schema_identical": direct_metadata == residual_metadata
        and len(direct_metadata) == 15,
        "controlled_operator_hyperparameters": all(
            config.get(key) == configs["direct_ufno"].get(key)
            for config in configs.values()
            for key in controlled_operator_keys
        ),
        "direct_normalization_identical": all(
            config.get("direct_target_normalization") == "train_standard"
            for config in direct_configs
        ),
        "residual_target_identical": all(
            config.get("target") == "source_superposition_residual_K"
            and config.get("physics_input") == "source_superposition_v1"
            and config.get("mean_head_mode") == "residual_resistance"
            for config in residual_configs
        ),
        "residual_signs_additive": all(
            int(config.get("mean_correction_sign", 1)) == 1
            and int(config.get("centered_correction_sign", 1)) == 1
            for config in residual_configs
        ),
        "ufno_topology_identical": all(
            int(config.get("fno_layers", 0)) == 6
            and list(config.get("ufno_unet_branch_indices", [])) == [3, 4, 5]
            and int(config.get("ufno_unet_depth", 0)) == 3
            and int(config.get("ufno_domain_padding", -1)) == 8
            and config.get("ufno_padding_mode") == "published_mixed"
            for name, config in configs.items()
            if "ufno" in name or "sau_fno" in name
        ),
        "sau_attention_only_controlled_addition": all(
            config.get("sau_fno_adaptation_profile")
            == SAU_FNO_ADAPTATION_PROFILE
            and config.get("sau_attention_enabled") is True
            and config.get("sau_attention_placement") == "after_final_ufourier_block"
            and config.get("sau_attention_type")
            == "single_head_unscaled_spatial_self_attention"
            and int(config.get("sau_attention_dim", 0))
            == int(config.get("fno_width", -1))
            and int(config.get("sau_number_of_heads", 0)) == 1
            for name, config in configs.items()
            if "sau_fno" in name
        ),
        "reference_commit_recorded": all(
            config.get("ufno_reference_commit") == UFNO_REFERENCE_COMMIT
            for name, config in configs.items()
            if "ufno" in name or "sau_fno" in name
        ),
    }
    parameter_counts = {name: count_parameters(model) for name, model in models.items()}
    parameter_counts.update(
        {
            "direct_cnn": checkpoint_parameter_count(direct_cnn_checkpoint),
            "residual_cnn": checkpoint_parameter_count(residual_cnn_checkpoint),
        }
    )
    fp32_attention = attention_memory_estimate(
        height=64, width=64, batch_size=batch_size, element_size_bytes=4
    )
    fp16_attention = attention_memory_estimate(
        height=64, width=64, batch_size=batch_size, element_size_bytes=2
    )
    return {
        "schema_version": "benchmark_v2_sau_fno_fairness/1",
        "checks": checks,
        "sample_counts": {"direct": len(direct_rows), "residual": len(residual_rows)},
        "spatial_channels": {
            "direct": direct_channels,
            "residual_pre_base": residual_channels,
            "residual_effective": residual_channels + ["source_superposition_base_K"],
        },
        "metadata_feature_names": direct_metadata,
        "parameter_counts": parameter_counts,
        "attention_memory": {
            "unpadded_tokens": 64 * 64,
            "padded_operator_tokens": 72 * 72,
            "attention_runs_after_crop": True,
            "fp32_explicit_matrix": fp32_attention,
            "fp16_or_bf16_explicit_matrix": fp16_attention,
            "estimate_scope": (
                "attention score matrix only; excludes Q/K/value, U-FNO activations, "
                "autograd state, FFT workspaces, allocator overhead, and optimizer state"
            ),
            "implementation": (
                "exact unscaled single-head torch scaled_dot_product_attention; "
                "CUDA may select a memory-efficient exact backend"
            ),
            "batch_64_status": (
                "not guaranteed; run one-batch A6000 peak-memory preflight before training"
            ),
        },
        "controlled_difference": (
            "Each SAU-FNO is its matched U-FNO plus one Q/K/value 1x1-projected "
            "single-head spatial attention operation after block 5 and padding crop."
        ),
        "reconstruction": (
            "source_superposition_base_K + total_power_W * "
            "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
        ),
    }


def build_operator(
    config: dict[str, Any], *, input_channels: int, metadata_dim: int
) -> torch.nn.Module:
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
        "sau_fno_adaptation_profile": config.get(
            "sau_fno_adaptation_profile", SAU_FNO_ADAPTATION_PROFILE
        ),
        "sau_attention_dim": config.get(
            "sau_attention_dim", config.get("fno_width", 32)
        ),
    }
    return build_model(payload)


def checkpoint_parameter_count(path: Path | None) -> int | None:
    if path is None:
        return None
    checkpoint = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    configured = checkpoint.get("model_config", {}).get("total_parameters")
    if configured is not None:
        return int(configured)
    return int(
        sum(value.numel() for value in checkpoint["model_state_dict"].values())
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark v2 SAU-FNO Fairness",
        "",
        f"- Samples: {report['sample_counts']['direct']}",
        f"- Unpadded attention tokens: {report['attention_memory']['unpadded_tokens']:,}",
        f"- Padded U-FNO operator tokens: {report['attention_memory']['padded_operator_tokens']:,}",
        "- Attention is applied after the existing padding crop.",
        "- Batch 64 requires an RTX A6000 one-batch peak-memory preflight.",
        "",
        "## Parameter Counts",
        "",
    ]
    lines.extend(
        f"- `{name}`: {value:,}" if value is not None else f"- `{name}`: not supplied"
        for name, value in report["parameter_counts"].items()
    )
    lines.extend(["", "## Checks", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in report["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Controlled Difference",
            "",
            report["controlled_difference"],
            "",
            f"Residual reconstruction: `{report['reconstruction']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def sample_uid(row: dict[str, str]) -> str:
    uid = row.get("sample_uid") or row.get("uid")
    if not uid:
        raise ValueError("index row is missing sample_uid")
    return str(uid)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
