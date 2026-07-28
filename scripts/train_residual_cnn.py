#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, DIMENSIONLESS_V1_TRANSFORMS, DIMENSIONLESS_V2_TRANSFORMS, chiptherm_collate
from chiptherm.ml.graph_models import (
    chiplet_mean_loss,
    chiplet_metric_values,
    compute_graph_normalization_stats,
    move_graph_to_device,
    normalize_graph_batch,
)
from chiptherm.ml.fno_models import FNO_CAPACITY_PROFILES
from chiptherm.ml.models import build_model, count_parameters
from chiptherm.ml.normalization import (
    DirectTemperatureTargetStats,
    NormalizationStats,
    build_metadata_input,
    build_model_input,
    compute_direct_temperature_target_stats,
    compute_normalization_stats,
    normalize_direct_temperature,
    normalize_residual,
    save_normalization_stats,
    unnormalize_direct_temperature,
    unnormalize_residual,
)


COARSE_SPATIAL_LOSS_DEFAULTS = {
    "coarse_spatial_loss_enabled": False,
    "coarse_spatial_loss_weight": 0.0,
    "coarse_spatial_loss_size": 8,
    "coarse_spatial_loss_type": "l1",
}
SUPPORTED_COARSE_SPATIAL_LOSS_SIZES = (8, 16)


def validate_coarse_spatial_loss_config(
    *,
    enabled: bool,
    weight: float,
    size: int,
    loss_type: str,
) -> None:
    if weight < 0.0:
        raise ValueError("coarse_spatial_loss_weight must be non-negative")
    if size not in SUPPORTED_COARSE_SPATIAL_LOSS_SIZES:
        raise ValueError(
            "coarse_spatial_loss_size must be one of "
            f"{SUPPORTED_COARSE_SPATIAL_LOSS_SIZES}, got {size}"
        )
    if loss_type != "l1":
        raise ValueError(f"coarse_spatial_loss_type must be 'l1', got {loss_type!r}")


def coarse_spatial_components(
    pred_centered: torch.Tensor,
    true_centered: torch.Tensor,
    *,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_coarse_spatial_loss_config(enabled=True, weight=0.0, size=size, loss_type="l1")
    if pred_centered.shape != true_centered.shape:
        raise ValueError(
            "predicted and target centered fields must have identical shapes, "
            f"got {tuple(pred_centered.shape)} and {tuple(true_centered.shape)}"
        )
    if pred_centered.ndim == 3:
        pred_4d = pred_centered.unsqueeze(1)
        true_4d = true_centered.unsqueeze(1)
    elif pred_centered.ndim == 4 and pred_centered.shape[1] == 1:
        pred_4d = pred_centered
        true_4d = true_centered
    else:
        raise ValueError(
            "centered fields must have shape [B,H,W] or [B,1,H,W], "
            f"got {tuple(pred_centered.shape)}"
        )
    if pred_4d.shape[-2] < size or pred_4d.shape[-1] < size:
        raise ValueError(
            f"coarse spatial size {size} exceeds input shape {tuple(pred_4d.shape[-2:])}"
        )
    pred_coarse = F.adaptive_avg_pool2d(pred_4d, output_size=(size, size))
    true_coarse = F.adaptive_avg_pool2d(true_4d, output_size=(size, size))
    pred_coarse = pred_coarse - pred_coarse.mean(dim=(-2, -1), keepdim=True)
    true_coarse = true_coarse - true_coarse.mean(dim=(-2, -1), keepdim=True)
    return pred_coarse, true_coarse


def compute_decomposed_training_losses(
    *,
    pred_temperature: torch.Tensor,
    true_temperature: torch.Tensor,
    pred_mean: torch.Tensor,
    true_mean: torch.Tensor,
    pred_centered: torch.Tensor,
    true_centered: torch.Tensor,
    lambda_final: float,
    lambda_mean: float,
    coarse_spatial_loss_enabled: bool = False,
    coarse_spatial_loss_weight: float = 0.0,
    coarse_spatial_loss_size: int = 8,
    coarse_spatial_loss_type: str = "l1",
) -> dict[str, torch.Tensor]:
    validate_coarse_spatial_loss_config(
        enabled=coarse_spatial_loss_enabled,
        weight=coarse_spatial_loss_weight,
        size=coarse_spatial_loss_size,
        loss_type=coarse_spatial_loss_type,
    )
    final_map_loss = F.l1_loss(pred_temperature, true_temperature)
    mean_loss = F.l1_loss(pred_mean, true_mean)
    centered_spatial_loss = F.l1_loss(pred_centered, true_centered)
    existing_total_loss = float(lambda_final) * final_map_loss + float(lambda_mean) * mean_loss
    coarse_spatial_loss = existing_total_loss.new_zeros(())
    if coarse_spatial_loss_enabled:
        pred_coarse, true_coarse = coarse_spatial_components(
            pred_centered,
            true_centered,
            size=coarse_spatial_loss_size,
        )
        coarse_spatial_loss = F.l1_loss(pred_coarse, true_coarse)
    weighted_coarse_spatial_loss = float(coarse_spatial_loss_weight) * coarse_spatial_loss
    total_loss = (
        existing_total_loss + weighted_coarse_spatial_loss
        if coarse_spatial_loss_enabled and coarse_spatial_loss_weight > 0.0
        else existing_total_loss
    )
    return {
        "total_loss": total_loss,
        "final_map_loss_K": final_map_loss,
        "mean_loss_K": mean_loss,
        "centered_spatial_loss_K": centered_spatial_loss,
        "coarse_spatial_loss_K": coarse_spatial_loss,
        "weighted_coarse_spatial_loss": weighted_coarse_spatial_loss,
    }


def build_resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    optional_defaults = {
        "prediction_mode": resolve_prediction_mode(
            "auto",
            str((config.get("model") or {}).get("architecture", "miniunet")),
        ),
        "direct_target_normalization_mode": "none",
    }
    return {
        key: config.get(key, optional_defaults[key]) if key in optional_defaults else config[key]
        for key in (
            "train_index",
            "val_index",
            "batch_size",
            "lr",
            "physics_input_mode",
            "physical_representation",
            "prediction_mode",
            "direct_target_normalization_mode",
            "mean_head_mode",
            "scheduler",
            "temp_loss_weight",
            "hotspot_loss_weight",
            "hotspot_top_frac",
            "lambda_final",
            "lambda_mean",
            "coarse_spatial_loss_enabled",
            "coarse_spatial_loss_weight",
            "coarse_spatial_loss_size",
            "coarse_spatial_loss_type",
            "lambda_graph",
            "lambda_chiplet_mean",
            "seed",
            "model",
        )
    }


def normalize_resume_signature(signature: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(signature or {})
    for key, value in COARSE_SPATIAL_LOSS_DEFAULTS.items():
        normalized.setdefault(key, value)
    architecture = str((normalized.get("model") or {}).get("architecture", "miniunet"))
    normalized.setdefault("prediction_mode", resolve_prediction_mode("auto", architecture))
    normalized.setdefault("direct_target_normalization_mode", "none")
    return normalized


def physics_input_channel_count(mode: str) -> int:
    if mode in {"v1", "gated_v1", "source_superposition_v1"}:
        return 1
    if mode == "source_superposition_plus_physics_v1":
        return 2
    if mode == "none":
        return 0
    raise ValueError(f"unsupported physics input mode: {mode}")


DIRECT_ARCHITECTURE = "miniunet_refine_conditioned_direct_temperature_feature_fusion"
DIRECT_FNO_ARCHITECTURE = "fno2d_direct_conditioned"
RESIDUAL_FNO_ARCHITECTURE = "fno2d_residual_decomposed_conditioned"
DIRECT_UFNO_ARCHITECTURE = "ufno2d_direct_conditioned"
RESIDUAL_UFNO_ARCHITECTURE = "ufno2d_residual_decomposed_conditioned"
DIRECT_SAU_FNO_ARCHITECTURE = "sau_fno2d_direct_conditioned"
RESIDUAL_SAU_FNO_ARCHITECTURE = "sau_fno2d_residual_decomposed_conditioned"
DIRECT_PREDICTION_MODES = {
    "direct_temperature",
    "direct_temperature_source_conditioned",
    "direct_temperature_fno",
    "direct_temperature_ufno",
    "direct_temperature_sau_fno",
}


def resolve_prediction_mode(requested: str, architecture: str) -> str:
    if requested != "auto":
        return requested
    if architecture == DIRECT_FNO_ARCHITECTURE:
        return "direct_temperature_fno"
    if architecture == RESIDUAL_FNO_ARCHITECTURE:
        return "residual_decomposed_fno"
    if architecture == DIRECT_UFNO_ARCHITECTURE:
        return "direct_temperature_ufno"
    if architecture == RESIDUAL_UFNO_ARCHITECTURE:
        return "residual_decomposed_ufno"
    if architecture == DIRECT_SAU_FNO_ARCHITECTURE:
        return "direct_temperature_sau_fno"
    if architecture == RESIDUAL_SAU_FNO_ARCHITECTURE:
        return "residual_decomposed_sau_fno"
    if architecture == DIRECT_ARCHITECTURE:
        return "direct_temperature"
    if "decomposed" in architecture:
        return "residual_decomposed"
    return "residual"


def validate_prediction_mode(
    prediction_mode: str,
    architecture: str,
    physics_input_mode: str,
) -> None:
    if prediction_mode in DIRECT_PREDICTION_MODES:
        expected_architecture = (
            DIRECT_FNO_ARCHITECTURE
            if prediction_mode == "direct_temperature_fno"
            else (
                DIRECT_UFNO_ARCHITECTURE
                if prediction_mode == "direct_temperature_ufno"
                else (
                    DIRECT_SAU_FNO_ARCHITECTURE
                    if prediction_mode == "direct_temperature_sau_fno"
                    else DIRECT_ARCHITECTURE
                )
            )
        )
        if architecture != expected_architecture:
            raise ValueError(
                f"prediction_mode={prediction_mode} requires architecture={expected_architecture}"
            )
        required_physics = (
            "source_superposition_v1"
            if prediction_mode == "direct_temperature_source_conditioned"
            else "none"
        )
        if physics_input_mode != required_physics:
            raise ValueError(
                f"prediction_mode={prediction_mode} requires physics_input_mode={required_physics}"
            )
        return
    if architecture in {
        DIRECT_ARCHITECTURE,
        DIRECT_FNO_ARCHITECTURE,
        DIRECT_UFNO_ARCHITECTURE,
        DIRECT_SAU_FNO_ARCHITECTURE,
    }:
        raise ValueError(f"architecture={architecture} requires its direct prediction mode")
    if architecture == RESIDUAL_FNO_ARCHITECTURE:
        if prediction_mode != "residual_decomposed_fno":
            raise ValueError(
                f"architecture={RESIDUAL_FNO_ARCHITECTURE} requires "
                "prediction_mode=residual_decomposed_fno"
            )
        if physics_input_mode != "source_superposition_v1":
            raise ValueError(
                "residual_decomposed_fno requires physics_input_mode=source_superposition_v1"
            )
        return
    if architecture == RESIDUAL_UFNO_ARCHITECTURE:
        if prediction_mode != "residual_decomposed_ufno":
            raise ValueError(
                f"architecture={RESIDUAL_UFNO_ARCHITECTURE} requires "
                "prediction_mode=residual_decomposed_ufno"
            )
        if physics_input_mode != "source_superposition_v1":
            raise ValueError(
                "residual_decomposed_ufno requires "
                "physics_input_mode=source_superposition_v1"
            )
        return
    if architecture == RESIDUAL_SAU_FNO_ARCHITECTURE:
        if prediction_mode != "residual_decomposed_sau_fno":
            raise ValueError(
                f"architecture={RESIDUAL_SAU_FNO_ARCHITECTURE} requires "
                "prediction_mode=residual_decomposed_sau_fno"
            )
        if physics_input_mode != "source_superposition_v1":
            raise ValueError(
                "residual_decomposed_sau_fno requires "
                "physics_input_mode=source_superposition_v1"
            )
        return
    if prediction_mode not in {"residual", "residual_decomposed"}:
        raise ValueError(f"unsupported prediction_mode: {prediction_mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ChipTherm residual mini-UNet.")
    parser.add_argument("--train-index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/train_index.csv", type=Path)
    parser.add_argument("--val-index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/val_index.csv", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "outputs/residual_cnn_v1", type=Path)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1.0e-3, type=float)
    parser.add_argument("--base-channels", default=16, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument(
        "--model-architecture",
        default="miniunet",
        choices=[
            "miniunet",
            "miniunet_refine",
            "miniunet_refine_conditioned",
            "miniunet_refine_decomposed",
            "miniunet_refine_conditioned_decomposed",
            "miniunet_refine_conditioned_decomposed_global",
            "miniunet_refine_conditioned_decomposed_feature_fusion",
            "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
            "miniunet_refine_conditioned_direct_temperature_feature_fusion",
            "miniunet_refine_conditioned_decomposed_graph",
            "miniunet_refine_conditioned_decomposed_global_graph",
            "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
            "miniunet_refine_conditioned_decomposed_pairwise",
            "miniunet_refine_conditioned_decomposed_pairwise_basis",
            DIRECT_FNO_ARCHITECTURE,
            RESIDUAL_FNO_ARCHITECTURE,
            DIRECT_UFNO_ARCHITECTURE,
            RESIDUAL_UFNO_ARCHITECTURE,
            DIRECT_SAU_FNO_ARCHITECTURE,
            RESIDUAL_SAU_FNO_ARCHITECTURE,
        ],
    )
    parser.add_argument("--refine-channels", default=32, type=int)
    parser.add_argument("--refine-blocks", default=4, type=int)
    parser.add_argument("--metadata-conditioning", action="store_true")
    parser.add_argument("--metadata-hidden-dim", default=64, type=int)
    parser.add_argument("--metadata-embedding-dim", default=64, type=int)
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=[
            "auto",
            "residual",
            "residual_decomposed",
            "direct_temperature",
            "direct_temperature_source_conditioned",
            "direct_temperature_fno",
            "residual_decomposed_fno",
            "direct_temperature_ufno",
            "residual_decomposed_ufno",
            "direct_temperature_sau_fno",
            "residual_decomposed_sau_fno",
        ],
        help="Checkpoint-visible output semantics. auto preserves legacy architecture behavior.",
    )
    parser.add_argument(
        "--fno-capacity-profile",
        default="fno_small",
        choices=["fno_small", "fno_standard"],
    )
    parser.add_argument("--fno-width", default=None, type=int)
    parser.add_argument("--fno-layers", default=None, type=int)
    parser.add_argument("--fno-modes-x", default=None, type=int)
    parser.add_argument("--fno-modes-y", default=None, type=int)
    parser.add_argument("--fno-activation", default="gelu", choices=["gelu"])
    parser.add_argument(
        "--fno-metadata-conditioning",
        default="film",
        choices=["film"],
    )
    parser.add_argument("--fno-projection-channels", default=None, type=int)
    parser.add_argument(
        "--ufno-adaptation-profile",
        default="ufno_published_adapted",
        choices=["ufno_published_adapted"],
    )
    parser.add_argument(
        "--ufno-unet-branch-indices",
        nargs="+",
        default=[3, 4, 5],
        type=int,
    )
    parser.add_argument("--ufno-unet-depth", default=3, type=int)
    parser.add_argument("--ufno-unet-dropout", default=0.0, type=float)
    parser.add_argument("--ufno-domain-padding", default=8, type=int)
    parser.add_argument(
        "--ufno-padding-mode",
        default="published_mixed",
        choices=["published_mixed", "none"],
    )
    parser.add_argument(
        "--sau-fno-adaptation-profile",
        default="sau_fno_paper_adapted",
        choices=["sau_fno_paper_adapted"],
    )
    parser.add_argument("--sau-attention-dim", default=32, type=int)
    parser.add_argument(
        "--direct-target-normalization",
        default="none",
        choices=["none", "train_standard"],
        help="Training-only absolute-temperature target representation for direct-temperature models.",
    )
    parser.add_argument(
        "--mean-head-mode",
        default="direct_k",
        choices=["direct_k", "residual_resistance"],
        help="Scalar decomposed mean head. direct_k preserves existing behavior; residual_resistance predicts normalized delta_R_eff in K/W and reconstructs mean correction with total_power_W.",
    )
    parser.add_argument(
        "--physical-representation",
        default="dimensional",
        choices=["dimensional", "dimensionless_v1", "dimensionless_v2"],
        help="Input physical representation. dimensionless_v1 applies aggressive physical ratios; dimensionless_v2 applies geometry-only package-relative ratios before train-only standardization.",
    )
    parser.add_argument(
        "--physics-input",
        default="v1",
        choices=["v1", "none", "gated_v1", "source_superposition_v1", "source_superposition_plus_physics_v1"],
        help="Spatial base input mode. 'v1' appends normalized physics_v1; 'source_superposition_v1' appends normalized source-superposition base; 'source_superposition_plus_physics_v1' appends source-superposition base plus preserved physics_v1 as an auxiliary channel; 'none' uses normalized X only; 'gated_v1' metadata-gates normalized physics_v1.",
    )
    parser.add_argument("--physics-gate-hidden-dim", default=32, type=int)
    parser.add_argument("--physics-gate-init", default=0.9, type=float)
    parser.add_argument("--physics-gate-regularization", default=0.0, type=float)
    parser.add_argument("--graph-hidden-dim", default=96, type=int)
    parser.add_argument("--graph-edge-hidden-dim", default=64, type=int)
    parser.add_argument("--graph-layers", default=4, type=int)
    parser.add_argument("--graph-message-aggregation", default="sum", choices=["sum", "mean"])
    parser.add_argument("--graph-raster-channels", default=16, type=int)
    parser.add_argument("--graph-halo-decay-mm", default=4.0, type=float)
    parser.add_argument("--graph-rasterizer-mode", default="vectorized", choices=["vectorized", "legacy"])
    parser.add_argument("--graph-use-edge-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--graph-mean-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow graph branch to add a scalar package-mean correction. "
            "Use --no-graph-mean-correction for matched frozen-CNN correction experiments "
            "where final output is exactly T_cnn + graph_correction_K."
        ),
    )
    parser.add_argument("--lambda-graph", default=0.0, type=float)
    parser.add_argument("--global-hidden-channels", default=32, type=int)
    parser.add_argument("--global-blocks", default=3, type=int)
    parser.add_argument("--global-pool-size", default=8, type=int)
    parser.add_argument(
        "--global-branch-channels",
        nargs="*",
        default=None,
        help="Optional explicit global branch channel names or integer model-input indices. Defaults to a compact physical subset.",
    )
    parser.add_argument(
        "--channel-routing-mode",
        default="auto",
        choices=["auto", "dimensional_baseline"],
        help=(
            "Controls default refinement/global channel selection. "
            "'auto' uses the manifest-named physical feature policy; "
            "'dimensional_baseline' reproduces the successful dimensional source-superposition "
            "residual-resistance routing: base raster channels for refinement and "
            "power/occupancy/coordinates plus source/base map for global fusion."
        ),
    )
    parser.add_argument("--pairwise-hidden-dim", default=96, type=int)
    parser.add_argument("--pairwise-layers", default=3, type=int)
    parser.add_argument("--pairwise-basis-rank", default=8, type=int)
    parser.add_argument("--pairwise-basis-hidden-dim", default=96, type=int)
    parser.add_argument("--pairwise-basis-layers", default=3, type=int)
    parser.add_argument("--pairwise-basis-halo-decay-mm", default=4.0, type=float)
    parser.add_argument("--pairwise-basis-edge-chunk-size", default=512, type=int)
    parser.add_argument("--lambda-chiplet-mean", default=0.0, type=float)
    parser.add_argument("--freeze-cnn", action="store_true")
    parser.add_argument("--init-checkpoint", default=None, type=Path, help="Optional checkpoint used to initialize matching model or CNN-submodule weights.")
    parser.add_argument("--lambda-final", default=1.0, type=float)
    parser.add_argument("--lambda-mean", default=0.1, type=float)
    parser.add_argument(
        "--coarse-spatial-loss-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add training-only coarse supervision to the existing centered spatial prediction.",
    )
    parser.add_argument("--coarse-spatial-loss-weight", default=0.0, type=float)
    parser.add_argument("--coarse-spatial-loss-size", default=8, type=int)
    parser.add_argument("--coarse-spatial-loss-type", default="l1", choices=["l1"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--scheduler", default="none", choices=["none", "plateau", "cosine"])
    parser.add_argument("--temp-loss-weight", default=0.0, type=float)
    parser.add_argument("--hotspot-loss-weight", default=0.0, type=float)
    parser.add_argument("--hotspot-top-frac", default=0.05, type=float)
    parser.add_argument(
        "--train-mae-every",
        default=5,
        type=int,
        help="Compute train-set final-temperature MAE every N epochs. Use 1 for every epoch, 0 to disable.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stopping-patience", default=0, type=int)
    parser.add_argument("--checkpoint-frequency", default=10, type=int)
    parser.add_argument("--lineage-manifest", default=None, type=Path)
    args = parser.parse_args()
    fno_profile = FNO_CAPACITY_PROFILES[args.fno_capacity_profile]
    fno_layers_was_default = args.fno_layers is None
    for argument, profile_key in (
        ("fno_width", "width"),
        ("fno_layers", "layers"),
        ("fno_modes_x", "modes_x"),
        ("fno_modes_y", "modes_y"),
        ("fno_projection_channels", "projection_channels"),
    ):
        if getattr(args, argument) is None:
            setattr(args, argument, int(fno_profile[profile_key]))
    if (
        args.model_architecture
        in {
            DIRECT_UFNO_ARCHITECTURE,
            RESIDUAL_UFNO_ARCHITECTURE,
            DIRECT_SAU_FNO_ARCHITECTURE,
            RESIDUAL_SAU_FNO_ARCHITECTURE,
        }
        and fno_layers_was_default
    ):
        args.fno_layers = 6
    if args.temp_loss_weight < 0.0:
        raise SystemExit("--temp-loss-weight must be non-negative")
    if args.hotspot_loss_weight < 0.0:
        raise SystemExit("--hotspot-loss-weight must be non-negative")
    if not 0.0 < args.hotspot_top_frac <= 1.0:
        raise SystemExit("--hotspot-top-frac must be in the interval (0, 1]")
    if args.train_mae_every < 0:
        raise SystemExit("--train-mae-every must be non-negative")
    if not 0.0 < args.physics_gate_init < 1.0:
        raise SystemExit("--physics-gate-init must be in the interval (0, 1)")
    if args.physics_gate_regularization < 0.0:
        raise SystemExit("--physics-gate-regularization must be non-negative")
    if args.lambda_graph < 0.0:
        raise SystemExit("--lambda-graph must be non-negative")
    if args.global_hidden_channels <= 0:
        raise SystemExit("--global-hidden-channels must be positive")
    if args.global_blocks < 0:
        raise SystemExit("--global-blocks must be non-negative")
    if args.global_pool_size <= 1:
        raise SystemExit("--global-pool-size must be greater than one")
    if args.lambda_chiplet_mean < 0.0:
        raise SystemExit("--lambda-chiplet-mean must be non-negative")
    if args.pairwise_hidden_dim <= 0:
        raise SystemExit("--pairwise-hidden-dim must be positive")
    if args.pairwise_layers <= 0:
        raise SystemExit("--pairwise-layers must be positive")
    if not 1 <= args.pairwise_basis_rank <= 8:
        raise SystemExit("--pairwise-basis-rank must be in [1, 8]")
    if args.pairwise_basis_hidden_dim <= 0:
        raise SystemExit("--pairwise-basis-hidden-dim must be positive")
    if args.pairwise_basis_layers <= 0:
        raise SystemExit("--pairwise-basis-layers must be positive")
    if args.pairwise_basis_halo_decay_mm <= 0.0:
        raise SystemExit("--pairwise-basis-halo-decay-mm must be positive")
    if args.pairwise_basis_edge_chunk_size <= 0:
        raise SystemExit("--pairwise-basis-edge-chunk-size must be positive")
    try:
        validate_coarse_spatial_loss_config(
            enabled=args.coarse_spatial_loss_enabled,
            weight=args.coarse_spatial_loss_weight,
            size=args.coarse_spatial_loss_size,
            loss_type=args.coarse_spatial_loss_type,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    is_global_arch = args.model_architecture in {
        "miniunet_refine_conditioned_decomposed_global",
        "miniunet_refine_conditioned_decomposed_global_graph",
    }
    is_feature_fusion_arch = args.model_architecture in {
        "miniunet_refine_conditioned_decomposed_feature_fusion",
        "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
        DIRECT_ARCHITECTURE,
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
    }
    is_fno_arch = args.model_architecture in {
        DIRECT_FNO_ARCHITECTURE,
        RESIDUAL_FNO_ARCHITECTURE,
    }
    is_ufno_arch = args.model_architecture in {
        DIRECT_UFNO_ARCHITECTURE,
        RESIDUAL_UFNO_ARCHITECTURE,
        DIRECT_SAU_FNO_ARCHITECTURE,
        RESIDUAL_SAU_FNO_ARCHITECTURE,
    }
    is_sau_fno_arch = args.model_architecture in {
        DIRECT_SAU_FNO_ARCHITECTURE,
        RESIDUAL_SAU_FNO_ARCHITECTURE,
    }
    is_operator_arch = is_fno_arch or is_ufno_arch
    is_direct_arch = args.model_architecture in {
        DIRECT_ARCHITECTURE,
        DIRECT_FNO_ARCHITECTURE,
        DIRECT_UFNO_ARCHITECTURE,
        DIRECT_SAU_FNO_ARCHITECTURE,
    }
    is_generic_graph_arch = args.model_architecture in {
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
    }
    is_pairwise_arch = args.model_architecture == "miniunet_refine_conditioned_decomposed_pairwise"
    is_pairwise_basis_arch = args.model_architecture == "miniunet_refine_conditioned_decomposed_pairwise_basis"
    is_graph_arch = is_generic_graph_arch or is_pairwise_arch or is_pairwise_basis_arch
    is_conditioned_arch = args.model_architecture in {
        "miniunet_refine_conditioned",
        "miniunet_refine_conditioned_decomposed",
        "miniunet_refine_conditioned_decomposed_global",
        "miniunet_refine_conditioned_decomposed_feature_fusion",
        "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
        DIRECT_ARCHITECTURE,
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
        DIRECT_FNO_ARCHITECTURE,
        RESIDUAL_FNO_ARCHITECTURE,
        DIRECT_UFNO_ARCHITECTURE,
        RESIDUAL_UFNO_ARCHITECTURE,
        DIRECT_SAU_FNO_ARCHITECTURE,
        RESIDUAL_SAU_FNO_ARCHITECTURE,
    }
    is_decomposed_arch = args.model_architecture in {
        "miniunet_refine_decomposed",
        "miniunet_refine_conditioned_decomposed",
        "miniunet_refine_conditioned_decomposed_global",
        "miniunet_refine_conditioned_decomposed_feature_fusion",
        "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
        "miniunet_refine_conditioned_decomposed_pairwise",
        "miniunet_refine_conditioned_decomposed_pairwise_basis",
        RESIDUAL_FNO_ARCHITECTURE,
        RESIDUAL_UFNO_ARCHITECTURE,
        RESIDUAL_SAU_FNO_ARCHITECTURE,
    }
    if args.physics_input == "gated_v1" and not is_conditioned_arch:
        raise SystemExit("--physics-input gated_v1 requires a metadata-conditioned architecture")
    if args.metadata_conditioning and not is_conditioned_arch:
        print("--metadata-conditioning requested; using architecture-selected behavior only.")
    if args.model_architecture == "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean":
        args.model_architecture = "miniunet_refine_conditioned_decomposed_feature_fusion"
        args.mean_head_mode = "residual_resistance"
    prediction_mode = resolve_prediction_mode(args.prediction_mode, args.model_architecture)
    try:
        validate_prediction_mode(prediction_mode, args.model_architecture, args.physics_input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.mean_head_mode == "residual_resistance" and not is_decomposed_arch:
        raise SystemExit("--mean-head-mode residual_resistance requires a decomposed architecture")
    if args.coarse_spatial_loss_enabled and not is_decomposed_arch:
        raise SystemExit("--coarse-spatial-loss-enabled requires a decomposed architecture")
    if is_direct_arch and args.coarse_spatial_loss_enabled:
        raise SystemExit("direct-temperature baseline does not support coarse spatial loss")
    if is_operator_arch:
        for name in ("fno_width", "fno_layers", "fno_modes_x", "fno_modes_y", "fno_projection_channels"):
            if int(getattr(args, name)) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.model_architecture == DIRECT_FNO_ARCHITECTURE:
        if args.direct_target_normalization != "train_standard":
            raise SystemExit("direct_temperature_fno requires --direct-target-normalization train_standard")
        if args.physics_input != "none":
            raise SystemExit("direct_temperature_fno must exclude the source/base input")
    if args.model_architecture == RESIDUAL_FNO_ARCHITECTURE:
        if args.mean_head_mode != "residual_resistance":
            raise SystemExit("residual_decomposed_fno requires --mean-head-mode residual_resistance")
        if args.physics_input != "source_superposition_v1":
            raise SystemExit(
                "residual_decomposed_fno requires --physics-input source_superposition_v1"
            )
    if is_ufno_arch:
        if args.fno_layers != 6 or tuple(args.ufno_unet_branch_indices) != (3, 4, 5):
            raise SystemExit(
                "ufno_published_adapted requires --fno-layers 6 and "
                "--ufno-unet-branch-indices 3 4 5"
            )
        if args.ufno_unet_depth != 3:
            raise SystemExit("ufno_published_adapted requires --ufno-unet-depth 3")
        if args.ufno_domain_padding < 0:
            raise SystemExit("--ufno-domain-padding must be non-negative")
        if not 0.0 <= args.ufno_unet_dropout < 1.0:
            raise SystemExit("--ufno-unet-dropout must be in [0, 1)")
    if is_sau_fno_arch and args.sau_attention_dim != args.fno_width:
        raise SystemExit(
            "sau_fno_paper_adapted requires --sau-attention-dim to equal --fno-width"
        )
    if args.model_architecture == DIRECT_UFNO_ARCHITECTURE:
        if args.direct_target_normalization != "train_standard":
            raise SystemExit(
                "direct_temperature_ufno requires --direct-target-normalization train_standard"
            )
        if args.physics_input != "none":
            raise SystemExit("direct_temperature_ufno must exclude the source/base input")
    if args.model_architecture == RESIDUAL_UFNO_ARCHITECTURE:
        if args.mean_head_mode != "residual_resistance":
            raise SystemExit(
                "residual_decomposed_ufno requires --mean-head-mode residual_resistance"
            )
        if args.physics_input != "source_superposition_v1":
            raise SystemExit(
                "residual_decomposed_ufno requires "
                "--physics-input source_superposition_v1"
            )
    if args.model_architecture == DIRECT_SAU_FNO_ARCHITECTURE:
        if args.direct_target_normalization != "train_standard":
            raise SystemExit(
                "direct_temperature_sau_fno requires "
                "--direct-target-normalization train_standard"
            )
        if args.physics_input != "none":
            raise SystemExit(
                "direct_temperature_sau_fno must exclude the source/base input"
            )
    if args.model_architecture == RESIDUAL_SAU_FNO_ARCHITECTURE:
        if args.mean_head_mode != "residual_resistance":
            raise SystemExit(
                "residual_decomposed_sau_fno requires "
                "--mean-head-mode residual_resistance"
            )
        if args.physics_input != "source_superposition_v1":
            raise SystemExit(
                "residual_decomposed_sau_fno requires "
                "--physics-input source_superposition_v1"
            )

    set_seed(args.seed)
    device = select_device(args.device)
    out_dir = args.out_dir.resolve()
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dataset_target = "temperature" if is_direct_arch else "residual"

    train_dataset = ChipThermDataset(
        args.train_index,
        target=dataset_target,
        return_metadata=True,
        return_graph=is_graph_arch,
        physical_representation=args.physical_representation,
    )
    val_dataset = ChipThermDataset(
        args.val_index,
        target=dataset_target,
        return_metadata=True,
        return_graph=is_graph_arch,
        physical_representation=args.physical_representation,
    )
    dataset_input_channels = int(train_dataset[0]["x"].shape[0])
    model_input_channels = dataset_input_channels + physics_input_channel_count(args.physics_input)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, device=device, graph_enabled=is_graph_arch)
    train_eval_loader = make_loader(train_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device, graph_enabled=is_graph_arch)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, device=device, graph_enabled=is_graph_arch)

    stats_dataset = (
        ChipThermDataset(
            args.train_index,
            target=dataset_target,
            return_metadata=True,
            return_graph=False,
            physical_representation=args.physical_representation,
        )
        if is_graph_arch
        else train_dataset
    )
    stats = compute_normalization_stats(stats_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    graph_stats = compute_graph_normalization_stats(train_dataset) if is_graph_arch else None
    delta_R_stats = compute_delta_R_eff_target_stats(train_eval_loader) if args.mean_head_mode == "residual_resistance" else None
    direct_target_stats = (
        compute_direct_temperature_target_stats(
            stats_dataset,
            mode=args.direct_target_normalization,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if is_direct_arch
        else None
    )
    metadata_dim = len(stats.metadata_feature_names)
    if is_conditioned_arch and metadata_dim <= 0:
        raise SystemExit("metadata-conditioned architecture requires metadata_features.csv/metadata_manifest.json")
    refinement_channel_indices, refinement_channel_names = refinement_channels_for_dataset(
        dataset_input_channels,
        stats,
        enabled=args.model_architecture != "miniunet" and not is_operator_arch,
        routing_mode=args.channel_routing_mode,
    )
    global_channel_indices, global_channel_names = global_branch_channels_for_dataset(
        dataset_input_channels,
        stats,
        physics_input_mode=args.physics_input,
        enabled=is_global_arch or is_feature_fusion_arch,
        requested=args.global_branch_channels,
        routing_mode=args.channel_routing_mode,
    )
    model_config: dict[str, Any] = {
        "architecture": args.model_architecture,
        "name": "MiniUNetWithRefinement" if args.model_architecture == "miniunet_refine" else "MiniUNet",
        "input_channels": model_input_channels,
        "dataset_input_channels": dataset_input_channels,
        "physics_input_mode": args.physics_input,
        "prediction_mode": prediction_mode,
        "physical_representation": args.physical_representation,
        "channel_routing_mode": args.channel_routing_mode,
        "dimensionless_v1_transforms": DIMENSIONLESS_V1_TRANSFORMS if args.physical_representation == "dimensionless_v1" else {},
        "dimensionless_v2_transforms": DIMENSIONLESS_V2_TRANSFORMS if args.physical_representation == "dimensionless_v2" else {},
        "dimensionless_v1_characteristic_length": "L_char_mm = sqrt(package_width_mm * package_height_mm)",
        "dimensionless_v1_characteristic_power_density": "total_power_W / occupied_area_mm2, with occupied_area from occupancy cells * cell_size_x_mm * cell_size_y_mm",
        "dimensionless_v2_characteristic_length": "L_char_mm = sqrt(package_width_mm * package_height_mm)",
        "dimensionless_v2_note": "Geometry-only hybrid: power, thermal, fixed-radius physics, package width/height, cell size, and source base remain dimensional.",
        "model_input_channels": model_input_channels,
        "physics_gate_hidden_dim": args.physics_gate_hidden_dim,
        "physics_gate_init": args.physics_gate_init,
        "mean_head_mode": args.mean_head_mode,
        "target_name": "absolute_temperature_K" if is_direct_arch else "residual",
        "target_normalization_mode": (
            direct_target_stats.mode if direct_target_stats is not None else "not_applicable"
        ),
        "target_mean_K": direct_target_stats.mean_K if direct_target_stats is not None else 0.0,
        "target_std_K": direct_target_stats.std_K if direct_target_stats is not None else 1.0,
        "output_channels": 1,
        "base_channels": args.base_channels,
        "depth": args.depth,
    }
    if is_operator_arch:
        model_config.update(
            {
                "fno_capacity_profile": args.fno_capacity_profile,
                "fno_width": args.fno_width,
                "fno_layers": args.fno_layers,
                "fno_modes_x": args.fno_modes_x,
                "fno_modes_y": args.fno_modes_y,
                "fno_activation": args.fno_activation,
                "fno_metadata_conditioning": args.fno_metadata_conditioning,
                "fno_projection_channels": args.fno_projection_channels,
            }
        )
        if is_ufno_arch:
            model_config.update(
                {
                    "ufno_reference_commit": (
                        "8315fd7b5bd75282b7efe42ee6b8de86543d13cc"
                    ),
                    "ufno_adaptation_profile": args.ufno_adaptation_profile,
                    "ufno_unet_branch_indices": list(args.ufno_unet_branch_indices),
                    "ufno_unet_depth": args.ufno_unet_depth,
                    "ufno_unet_dropout": args.ufno_unet_dropout,
                    "ufno_domain_padding": args.ufno_domain_padding,
                    "ufno_padding_mode": args.ufno_padding_mode,
                    "ufno_branch_fusion": "add",
                }
            )
        if is_sau_fno_arch:
            model_config.update(
                {
                    "sau_fno_adaptation_profile": args.sau_fno_adaptation_profile,
                    "sau_attention_enabled": True,
                    "sau_attention_dim": args.sau_attention_dim,
                    "sau_attention_placement": (
                        "after_final_ufourier_activation_after_padding_crop"
                    ),
                    "sau_attention_type": (
                        "single_head_unscaled_spatial_self_attention"
                    ),
                    "sau_number_of_heads": 1,
                    "sau_softmax_axis": "key_token_axis_-1",
                    "sau_fusion_rule": "softmax(Q @ K^T, dim=-1) @ W_h(V)",
                }
            )
    elif args.model_architecture != "miniunet":
        model_config.update(
            {
                "refine_channels": args.refine_channels,
                "refine_blocks": args.refine_blocks,
                "refinement_channel_indices": list(refinement_channel_indices),
                "refinement_channel_names": list(refinement_channel_names),
            }
        )
    if is_conditioned_arch or is_decomposed_arch:
        model_config.update(
            {
                "metadata_dim": metadata_dim,
                "metadata_feature_names": list(stats.metadata_feature_names),
                "metadata_hidden_dim": args.metadata_hidden_dim,
                "metadata_embedding_dim": args.metadata_embedding_dim,
            }
        )
    if is_graph_arch:
        sample_graph = train_dataset[0].get("graph")
        if sample_graph is None:
            raise SystemExit("graph architecture requires graph_path artifacts in the training index")
        model_config.update(
            {
                "graph_enabled": True,
                "graph_node_feature_dim": int(sample_graph["node_features"].shape[1]),
                "graph_edge_feature_dim": int(sample_graph["edge_features"].shape[1]),
                "graph_hidden_dim": args.graph_hidden_dim,
                "graph_edge_hidden_dim": args.graph_edge_hidden_dim,
                "graph_layers": args.graph_layers,
                "graph_message_aggregation": args.graph_message_aggregation,
                "graph_raster_channels": args.graph_raster_channels,
                "graph_halo_decay_mm": args.graph_halo_decay_mm,
                "graph_rasterizer_mode": args.graph_rasterizer_mode,
                "graph_use_edge_features": args.graph_use_edge_features,
                "graph_mean_correction_enabled": args.graph_mean_correction,
                "freeze_cnn": args.freeze_cnn,
                "graph_node_feature_names": list(getattr(train_dataset, "graph_node_feature_names", ()) or []),
                "graph_edge_feature_names": list(getattr(train_dataset, "graph_edge_feature_names", ()) or []),
                "graph_normalization": graph_stats.to_dict() if graph_stats is not None else None,
            }
        )
    if is_pairwise_arch:
        model_config.update(
            {
                "pairwise_enabled": True,
                "pairwise_hidden_dim": args.pairwise_hidden_dim,
                "pairwise_layers": args.pairwise_layers,
                "source_power_feature_index": 6,
            }
        )
    if is_pairwise_basis_arch:
        model_config.update(
            {
                "pairwise_basis_enabled": True,
                "pairwise_basis_rank": args.pairwise_basis_rank,
                "pairwise_basis_hidden_dim": args.pairwise_basis_hidden_dim,
                "pairwise_basis_layers": args.pairwise_basis_layers,
                "pairwise_basis_halo_decay_mm": args.pairwise_basis_halo_decay_mm,
                "pairwise_basis_edge_chunk_size": args.pairwise_basis_edge_chunk_size,
                "source_power_feature_index": 6,
            }
        )
    if is_global_arch or is_feature_fusion_arch:
        model_config.update(
            {
                "global_branch_enabled": is_global_arch,
                "feature_fusion_enabled": is_feature_fusion_arch,
                "global_branch_channel_indices": list(global_channel_indices),
                "global_branch_channel_names": list(global_channel_names),
                "global_hidden_channels": args.global_hidden_channels,
                "global_blocks": args.global_blocks,
                "global_pool_size": args.global_pool_size,
            }
        )
    if args.mean_head_mode == "residual_resistance":
        assert delta_R_stats is not None
        model_config.update(
            {
                "delta_R_eff_target_mean_K_per_W": delta_R_stats["mean_K_per_W"],
                "delta_R_eff_target_std_K_per_W": delta_R_stats["std_K_per_W"],
                "delta_R_eff_target_units": "K/W",
                "delta_R_eff_target_normalization": "raw_head * train_std + train_mean; statistics fit on train split only",
            }
        )
        if not args.resume:
            (out_dir / "delta_R_eff_normalization.json").write_text(
                json.dumps(delta_R_stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_index": str(args.train_index.resolve()),
        "val_index": str(args.val_index.resolve()),
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "model_architecture": args.model_architecture,
        "physics_input_mode": args.physics_input,
        "prediction_mode": prediction_mode,
        "direct_target_normalization_mode": args.direct_target_normalization,
        "direct_temperature_target_normalization": (
            direct_target_stats.to_dict() if direct_target_stats is not None else None
        ),
        "physical_representation": args.physical_representation,
        "dimensionless_v1_transforms": DIMENSIONLESS_V1_TRANSFORMS if args.physical_representation == "dimensionless_v1" else {},
        "dimensionless_v2_transforms": DIMENSIONLESS_V2_TRANSFORMS if args.physical_representation == "dimensionless_v2" else {},
        "dimensionless_v1_characteristic_length": "L_char_mm = sqrt(package_width_mm * package_height_mm)",
        "dimensionless_v1_characteristic_power_density": "total_power_W / occupied_area_mm2",
        "dimensionless_v2_characteristic_length": "L_char_mm = sqrt(package_width_mm * package_height_mm)",
        "dimensionless_v2_note": "Geometry-only hybrid; all power/thermal/package-scale channels remain dimensional.",
        "model_input_channels": model_input_channels,
        "dataset_input_channels": dataset_input_channels,
        "physics_gate_hidden_dim": args.physics_gate_hidden_dim,
        "mean_head_mode": args.mean_head_mode,
        "delta_R_eff_target_normalization": delta_R_stats,
        "physics_gate_init": args.physics_gate_init,
        "physics_gate_regularization": args.physics_gate_regularization,
        "graph_hidden_dim": args.graph_hidden_dim,
        "graph_edge_hidden_dim": args.graph_edge_hidden_dim,
        "graph_layers": args.graph_layers,
        "graph_message_aggregation": args.graph_message_aggregation,
        "graph_raster_channels": args.graph_raster_channels,
        "graph_halo_decay_mm": args.graph_halo_decay_mm,
        "graph_rasterizer_mode": args.graph_rasterizer_mode,
        "graph_use_edge_features": args.graph_use_edge_features,
        "graph_mean_correction_enabled": args.graph_mean_correction,
        "lambda_graph": args.lambda_graph,
        "global_hidden_channels": args.global_hidden_channels,
        "global_blocks": args.global_blocks,
        "global_pool_size": args.global_pool_size,
        "global_branch_channels": args.global_branch_channels,
        "sau_fno_adaptation_profile": args.sau_fno_adaptation_profile,
        "sau_attention_dim": args.sau_attention_dim,
        "channel_routing_mode": args.channel_routing_mode,
        "pairwise_hidden_dim": args.pairwise_hidden_dim,
        "pairwise_layers": args.pairwise_layers,
        "pairwise_basis_rank": args.pairwise_basis_rank,
        "pairwise_basis_hidden_dim": args.pairwise_basis_hidden_dim,
        "pairwise_basis_layers": args.pairwise_basis_layers,
        "pairwise_basis_halo_decay_mm": args.pairwise_basis_halo_decay_mm,
        "pairwise_basis_edge_chunk_size": args.pairwise_basis_edge_chunk_size,
        "lambda_chiplet_mean": args.lambda_chiplet_mean,
        "freeze_cnn": args.freeze_cnn,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "refine_channels": args.refine_channels,
        "refine_blocks": args.refine_blocks,
        "device": str(device),
        "num_workers": args.num_workers,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "temp_loss_weight": args.temp_loss_weight,
        "temp_loss_scaling": "temperature L1 loss in Kelvin divided by train residual_std before weighting",
        "hotspot_loss_weight": args.hotspot_loss_weight,
        "hotspot_top_frac": args.hotspot_top_frac,
        "hotspot_loss_scaling": "hotspot L1 loss in Kelvin over top ground-truth HotSpot cells divided by train residual_std before weighting",
        "train_mae_every": args.train_mae_every,
        "resume": args.resume,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_frequency": args.checkpoint_frequency,
        "lineage_manifest": str(args.lineage_manifest.resolve()) if args.lineage_manifest else None,
        "lambda_final": args.lambda_final,
        "lambda_mean": args.lambda_mean,
        "coarse_spatial_loss_enabled": args.coarse_spatial_loss_enabled,
        "coarse_spatial_loss_weight": args.coarse_spatial_loss_weight,
        "coarse_spatial_loss_size": args.coarse_spatial_loss_size,
        "coarse_spatial_loss_type": args.coarse_spatial_loss_type,
        "target_decomposition": is_decomposed_arch,
        "metadata_conditioning": is_conditioned_arch,
        "graph_enabled": is_graph_arch,
        "global_branch_enabled": is_global_arch,
        "feature_fusion_enabled": is_feature_fusion_arch,
        "pairwise_enabled": is_pairwise_arch,
        "pairwise_basis_enabled": is_pairwise_basis_arch,
        "model": model_config,
        "loss": (
            "L1Loss between direct model output and absolute-temperature target representation"
            if is_direct_arch
            else (
                "SmoothL1Loss on normalized residual plus optional temp_loss_weight * L1(T_pred, HotSpot) / residual_std "
                "plus optional hotspot_loss_weight * L1(T_pred, HotSpot on top HotSpot cells) / residual_std"
            )
        ),
        "target": (
            "absolute_temperature_K = HotSpot target map"
            if is_direct_arch
            else "residual = HotSpot - PhysicsBaseline"
        ),
    }
    config["resume_signature"] = build_resume_signature(config)
    model = build_model(model_config).to(device)
    if args.resume and args.init_checkpoint:
        raise SystemExit("--resume and --init-checkpoint are mutually exclusive")
    init_summary = load_initial_checkpoint(model, args.init_checkpoint, device) if args.init_checkpoint else None
    config["model"] = model.config() if hasattr(model, "config") else model_config
    config["model"].update(
        {
            "physics_input_mode": args.physics_input,
            "prediction_mode": prediction_mode,
            "physical_representation": args.physical_representation,
            "channel_routing_mode": args.channel_routing_mode,
            "dimensionless_v1_transforms": DIMENSIONLESS_V1_TRANSFORMS if args.physical_representation == "dimensionless_v1" else {},
            "dimensionless_v2_transforms": DIMENSIONLESS_V2_TRANSFORMS if args.physical_representation == "dimensionless_v2" else {},
            "dimensionless_v1_characteristic_length": "L_char_mm = sqrt(package_width_mm * package_height_mm)",
            "dimensionless_v1_characteristic_power_density": "total_power_W / occupied_area_mm2",
            "dimensionless_v2_characteristic_length": "L_char_mm = sqrt(package_width_mm * package_height_mm)",
            "dimensionless_v2_note": "Geometry-only hybrid; all power/thermal/package-scale channels remain dimensional.",
            "model_input_channels": model_input_channels,
            "dataset_input_channels": dataset_input_channels,
            "physics_gate_hidden_dim": args.physics_gate_hidden_dim,
            "physics_gate_init": args.physics_gate_init,
            "graph_normalization": graph_stats.to_dict() if graph_stats is not None else None,
            "lambda_chiplet_mean": args.lambda_chiplet_mean,
            "graph_node_feature_names": list(getattr(train_dataset, "graph_node_feature_names", ()) or []),
            "graph_edge_feature_names": list(getattr(train_dataset, "graph_edge_feature_names", ()) or []),
            "graph_mean_correction_enabled": args.graph_mean_correction,
        }
    )
    if is_global_arch or is_feature_fusion_arch:
        config["model"].update(
            {
                "global_branch_enabled": is_global_arch,
                "feature_fusion_enabled": is_feature_fusion_arch,
                "global_branch_channel_indices": list(global_channel_indices),
                "global_branch_channel_names": list(global_channel_names),
                "global_hidden_channels": args.global_hidden_channels,
                "global_blocks": args.global_blocks,
                "global_pool_size": args.global_pool_size,
            }
        )
    if not args.resume:
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_normalization_stats(stats, out_dir / "normalization.json")
        if direct_target_stats is not None:
            (out_dir / "direct_temperature_normalization.json").write_text(
                json.dumps(direct_target_stats.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)
    criterion = nn.SmoothL1Loss()
    temp_criterion = nn.L1Loss()

    print(f"Model architecture: {args.model_architecture}")
    print(f"Prediction mode: {prediction_mode}")
    print(f"Physics input mode: {args.physics_input}")
    print(f"Physical representation: {args.physical_representation}")
    print(f"Channel routing mode: {args.channel_routing_mode}")
    print(f"Mean head mode: {args.mean_head_mode}")
    if delta_R_stats is not None:
        print(
            "Delta R_eff train target mean/std/min/max: "
            f"{delta_R_stats['mean_K_per_W']:.6f} / {delta_R_stats['std_K_per_W']:.6f} / "
            f"{delta_R_stats['min_K_per_W']:.6f} / {delta_R_stats['max_K_per_W']:.6f} K/W"
        )
    if direct_target_stats is not None:
        print(
            "Direct target normalization mode/mean/std/min/max: "
            f"{direct_target_stats.mode} / {direct_target_stats.mean_K:.6f} / "
            f"{direct_target_stats.std_K:.6f} / {direct_target_stats.min_K:.6f} / "
            f"{direct_target_stats.max_K:.6f} K"
        )
    print(f"Model input channels: {model_input_channels}")
    if args.model_architecture == "miniunet_refine":
        print(f"Refinement channels: {list(refinement_channel_indices)}")
        print(f"Refinement channel names: {', '.join(refinement_channel_names)}")
    if is_global_arch or is_feature_fusion_arch:
        print(f"Global physical channels: {list(global_channel_indices)}")
        print(f"Global physical channel names: {', '.join(global_channel_names)}")
    if init_summary:
        print(f"Initialized from {args.init_checkpoint}: {init_summary}")
    if is_graph_arch:
        print(f"Graph mean correction enabled: {args.graph_mean_correction}")
    print(f"Trainable parameters: {count_parameters(model)}")

    log_path = out_dir / "train_log.csv"
    if not args.resume:
        init_train_log(log_path)
    else:
        ensure_train_log_schema(log_path)
    best_val_mae = float("inf")
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    start_epoch = 1
    last_completed_epoch = 0
    last_train_losses: dict[str, float] | None = None
    training_lineage = (
        json.loads(args.lineage_manifest.read_text(encoding="utf-8"))
        if args.lineage_manifest
        else None
    )
    if args.resume:
        last_path = checkpoints_dir / "last.pt"
        if not last_path.is_file():
            raise SystemExit(f"--resume requested but checkpoint is missing: {last_path}")
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        if resumed.get("model_config") != config["model"]:
            raise SystemExit("resume checkpoint model configuration differs from requested model")
        if resumed.get("normalization") != stats.to_dict():
            raise SystemExit("resume checkpoint normalization differs from current train-only statistics")
        if training_lineage is not None and resumed.get("training_lineage") != training_lineage:
            raise SystemExit("resume checkpoint lineage differs from requested lineage")
        resumed_signature = normalize_resume_signature(
            resumed.get("training_config", {}).get("resume_signature")
        )
        if resumed_signature != normalize_resume_signature(config["resume_signature"]):
            raise SystemExit("resume checkpoint training recipe differs from requested training")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if scheduler is not None and resumed.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        last_completed_epoch = int(resumed["epoch"])
        best_val_mae = float(resumed.get("best_val_mae_K", float("inf")))
        epochs_without_improvement = int(resumed.get("epochs_without_improvement", 0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            temp_criterion,
            stats,
            device,
            temp_loss_weight=args.temp_loss_weight,
            hotspot_loss_weight=args.hotspot_loss_weight,
            hotspot_top_frac=args.hotspot_top_frac,
            decomposed=is_decomposed_arch,
            conditioned=is_conditioned_arch,
            lambda_final=args.lambda_final,
            lambda_mean=args.lambda_mean,
            coarse_spatial_loss_enabled=args.coarse_spatial_loss_enabled,
            coarse_spatial_loss_weight=args.coarse_spatial_loss_weight,
            coarse_spatial_loss_size=args.coarse_spatial_loss_size,
            coarse_spatial_loss_type=args.coarse_spatial_loss_type,
            physics_input_mode=args.physics_input,
            physics_gate_regularization=args.physics_gate_regularization,
            physics_gate_init=args.physics_gate_init,
            graph_enabled=is_graph_arch,
            graph_stats=graph_stats,
            lambda_graph=args.lambda_graph,
            lambda_chiplet_mean=args.lambda_chiplet_mean,
            mean_head_mode=args.mean_head_mode,
            prediction_mode=prediction_mode,
            direct_target_stats=direct_target_stats,
        )
        last_train_losses = train_losses
        last_completed_epoch = epoch
        val_metrics, val_by_case = evaluate_model(
            model,
            val_loader,
            criterion,
            stats,
            device,
            decomposed=is_decomposed_arch,
            conditioned=is_conditioned_arch,
            lambda_final=args.lambda_final,
            lambda_mean=args.lambda_mean,
            physics_input_mode=args.physics_input,
            graph_enabled=is_graph_arch,
            graph_stats=graph_stats,
            lambda_chiplet_mean=args.lambda_chiplet_mean,
            mean_head_mode=args.mean_head_mode,
            prediction_mode=prediction_mode,
            direct_target_stats=direct_target_stats,
        )
        train_final_mae_K: float | None = None
        if should_compute_train_mae(epoch, args.epochs, args.train_mae_every):
            train_metrics, _ = evaluate_model(
                model,
                train_eval_loader,
                criterion,
                stats,
                device,
                decomposed=is_decomposed_arch,
                conditioned=is_conditioned_arch,
                lambda_final=args.lambda_final,
                lambda_mean=args.lambda_mean,
                physics_input_mode=args.physics_input,
                graph_enabled=is_graph_arch,
                graph_stats=graph_stats,
                lambda_chiplet_mean=args.lambda_chiplet_mean,
                mean_head_mode=args.mean_head_mode,
                prediction_mode=prediction_mode,
                direct_target_stats=direct_target_stats,
            )
            train_final_mae_K = float(train_metrics["final_temperature"]["mae_K"])
        epoch_runtime_s = time.perf_counter() - epoch_start
        val_final_mae = float(val_metrics["final_temperature"]["mae_K"])
        step_scheduler(args.scheduler, scheduler, val_final_mae)
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_final_mae < best_val_mae
        if is_best:
            best_val_mae = val_final_mae
            epochs_without_improvement = 0
            best_metrics = {
                "epoch": epoch,
                "metrics": val_metrics,
                "metrics_by_case": val_by_case,
            }
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                stats,
                val_metrics,
                best=True,
                best_val_mae=best_val_mae,
                epochs_without_improvement=epochs_without_improvement,
                training_lineage=training_lineage,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            stats,
            val_metrics,
            best=is_best,
            best_val_mae=best_val_mae,
            epochs_without_improvement=epochs_without_improvement,
            training_lineage=training_lineage,
        )
        if args.checkpoint_frequency > 0 and epoch % args.checkpoint_frequency == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                stats,
                val_metrics,
                best=is_best,
                best_val_mae=best_val_mae,
                epochs_without_improvement=epochs_without_improvement,
                training_lineage=training_lineage,
            )
        append_train_log(
            log_path,
            epoch,
            train_losses,
            train_final_mae_K,
            val_metrics,
            epoch_runtime_s,
            is_best,
            current_lr,
            args.physical_representation,
        )
        write_training_history_json(log_path, out_dir / "training_history.json")
        write_metrics(out_dir / "val_metrics.json", best_metrics or {"epoch": epoch, "metrics": val_metrics, "metrics_by_case": val_by_case})
        write_case_metrics(out_dir / "val_metrics_by_case.csv", (best_metrics or {"metrics_by_case": val_by_case})["metrics_by_case"])

        direct_loss_log = (
            f"direct_map={train_losses['direct_map_loss']:.6f} "
            if is_direct_arch
            else ""
        )
        print(
            f"epoch {epoch:03d} train_loss={train_losses['total_loss']:.6f} lr={current_lr:.3e} "
            f"{direct_loss_log}"
            f"coarse={train_losses['coarse_spatial_loss_K']:.6f}K "
            f"weighted_coarse={train_losses['weighted_coarse_spatial_loss']:.6f} "
            f"train_final_mae={format_optional_mae(train_final_mae_K)} "
            f"val_mae={val_final_mae:.3f}K val_rmse={val_metrics['final_temperature']['rmse_K']:.3f}K "
            f"gate={format_gate_summary(val_metrics.get('physics_gate'))} "
            f"{format_graph_epoch_summary(val_metrics, val_by_case)} "
            f"{'best' if is_best else ''}"
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without validation improvement")
            break

    write_metrics(
        out_dir / "training_summary.json",
        {
            "best_validation_final_temperature_mae_K": best_val_mae,
            "last_completed_epoch": last_completed_epoch,
            "final_training_losses": last_train_losses,
            "parameter_count": count_parameters(model),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "coarse_spatial_loss": {
                "enabled": args.coarse_spatial_loss_enabled,
                "weight": args.coarse_spatial_loss_weight,
                "size": args.coarse_spatial_loss_size,
                "type": args.coarse_spatial_loss_type,
            },
        },
    )
    print("Residual CNN training complete")
    print(f"Best validation final-temperature MAE: {best_val_mae:.3f} K")
    print(f"Output: {out_dir}")
    return 0


BASE_CHANNEL_NAMES = (
    "power_density_W_per_mm2",
    "occupancy_mask",
    "CPU_mask",
    "GPU_or_NPU_mask",
    "memory_mask",
    "IO_or_ANALOG_or_MEMS_mask",
    "normalized_x_coordinate",
    "normalized_y_coordinate",
)


def refinement_channels_for_dataset(
    dataset_input_channels: int,
    stats: NormalizationStats,
    *,
    enabled: bool,
    routing_mode: str = "auto",
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not enabled:
        return (), ()
    names = dataset_channel_names(dataset_input_channels, stats)
    if routing_mode == "dimensional_baseline":
        selected = tuple(range(min(8, dataset_input_channels)))
        if not selected:
            raise SystemExit("dimensional_baseline routing could not identify base raster refinement channels")
        return selected, tuple(names[index] for index in selected)
    if routing_mode != "auto":
        raise SystemExit(f"unsupported channel routing mode: {routing_mode}")
    selected: list[int] = []
    selected_names: list[str] = []
    for index, name in enumerate(names):
        if is_refinement_channel(index, name):
            selected.append(index)
            selected_names.append(name)
    if not selected:
        raise SystemExit("miniunet_refine could not identify any full-resolution refinement input channels")
    return tuple(selected), tuple(selected_names)


def is_refinement_channel(index: int, name: str) -> bool:
    if index < 8:
        return True
    if name.startswith("finite_source_"):
        return True
    if name.startswith("enclosed_power_"):
        return True
    if name == "minimum_distance_to_package_edge_mm":
        return True
    if name in {
        "chiplet_total_power_W",
        "chiplet_width_mm",
        "chiplet_height_mm",
        "chiplet_area_mm2",
        "chiplet_aspect_ratio",
        "chiplet_power_density_W_per_mm2",
        "thermal_crowding_W_per_mm",
    }:
        return True
    return False


def dataset_channel_names(dataset_input_channels: int, stats: NormalizationStats) -> list[str]:
    names = list(BASE_CHANNEL_NAMES[: min(8, dataset_input_channels)])
    context_names = list(getattr(stats, "context_channel_names", ()) or ())
    for offset in range(8, dataset_input_channels):
        context_index = offset - 8
        if context_index < len(context_names):
            names.append(str(context_names[context_index]))
        else:
            names.append(f"channel_{offset}")
    return names


def model_input_channel_names(dataset_input_channels: int, stats: NormalizationStats, physics_input_mode: str) -> list[str]:
    names = dataset_channel_names(dataset_input_channels, stats)
    if physics_input_mode == "source_superposition_v1":
        names.append("source_superposition_base_K")
    elif physics_input_mode == "source_superposition_plus_physics_v1":
        names.extend(["source_superposition_base_K", "physics_v1_temperature_K"])
    elif physics_input_mode in {"v1", "gated_v1"}:
        names.append("physics_v1_temperature_K")
    elif physics_input_mode == "none":
        pass
    else:
        raise ValueError(f"unsupported physics_input_mode: {physics_input_mode}")
    return names


def global_branch_channels_for_dataset(
    dataset_input_channels: int,
    stats: NormalizationStats,
    *,
    physics_input_mode: str,
    enabled: bool,
    requested: list[str] | None,
    routing_mode: str = "auto",
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not enabled:
        return (), ()
    names = model_input_channel_names(dataset_input_channels, stats, physics_input_mode)
    selected: list[int] = []
    if requested:
        name_to_index = {name: index for index, name in enumerate(names)}
        for item in requested:
            if str(item).isdigit():
                index = int(item)
            else:
                if item not in name_to_index:
                    raise SystemExit(f"unknown --global-branch-channels item {item!r}; available names include {names}")
                index = name_to_index[item]
            if index < 0 or index >= len(names):
                raise SystemExit(f"global branch channel index {index} out of range for {len(names)} model input channels")
            if index not in selected:
                selected.append(index)
    elif routing_mode == "dimensional_baseline":
        target_names = [
            "power_density_W_per_mm2",
            "occupancy_mask",
            "normalized_x_coordinate",
            "normalized_y_coordinate",
        ]
        if physics_input_mode in {"source_superposition_v1", "source_superposition_plus_physics_v1"}:
            target_names.append("source_superposition_base_K")
        elif physics_input_mode in {"v1", "gated_v1"}:
            target_names.append("physics_v1_temperature_K")
        name_to_index = {name: index for index, name in enumerate(names)}
        missing = [name for name in target_names if name not in name_to_index]
        if missing:
            raise SystemExit(
                "dimensional_baseline routing could not identify global channels: "
                + ", ".join(missing)
            )
        selected = [name_to_index[name] for name in target_names]
    else:
        if routing_mode != "auto":
            raise SystemExit(f"unsupported channel routing mode: {routing_mode}")
        for index, name in enumerate(names):
            if is_global_branch_channel(name):
                selected.append(index)
        if physics_input_mode == "source_superposition_plus_physics_v1":
            physics_v1_index = len(names) - 1
            if physics_v1_index in selected:
                selected.remove(physics_v1_index)
    if not selected:
        raise SystemExit("global branch could not identify any compact physical input channels")
    return tuple(selected), tuple(names[index] for index in selected)


def is_global_branch_channel(name: str) -> bool:
    return name in {
        "power_density_W_per_mm2",
        "occupancy_mask",
        "normalized_x_coordinate",
        "normalized_y_coordinate",
        "minimum_distance_to_package_edge_mm",
        "finite_source_L2mm",
        "finite_source_L4mm",
        "enclosed_power_R8mm_W",
        "enclosed_power_R16mm_W",
        "source_superposition_base_K",
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    temp_criterion: nn.Module,
    stats: NormalizationStats,
    device: torch.device,
    *,
    temp_loss_weight: float,
    hotspot_loss_weight: float,
    hotspot_top_frac: float,
    decomposed: bool,
    conditioned: bool,
    lambda_final: float,
    lambda_mean: float,
    coarse_spatial_loss_enabled: bool,
    coarse_spatial_loss_weight: float,
    coarse_spatial_loss_size: int,
    coarse_spatial_loss_type: str,
    physics_input_mode: str,
    physics_gate_regularization: float,
    physics_gate_init: float,
    graph_enabled: bool = False,
    graph_stats: Any | None = None,
    lambda_graph: float = 0.0,
    lambda_chiplet_mean: float = 0.0,
    mean_head_mode: str = "direct_k",
    prediction_mode: str = "residual",
    direct_target_stats: DirectTemperatureTargetStats | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    residual_loss_total = 0.0
    temp_loss_scaled_total = 0.0
    temp_loss_K_total = 0.0
    final_map_loss_total = 0.0
    mean_loss_total = 0.0
    centered_spatial_loss_total = 0.0
    coarse_spatial_loss_total = 0.0
    weighted_coarse_spatial_loss_total = 0.0
    hotspot_loss_scaled_total = 0.0
    hotspot_loss_K_total = 0.0
    gate_regularization_total = 0.0
    graph_regularization_total = 0.0
    graph_correction_total = 0.0
    global_correction_total = 0.0
    chiplet_mean_loss_total = 0.0
    gate_acc = ScalarSummaryAccumulator()
    pairwise_k_acc = ScalarSummaryAccumulator()
    pairwise_contribution_acc = ScalarSummaryAccumulator()
    pairwise_self_acc = ScalarSummaryAccumulator()
    pairwise_node_acc = ScalarSummaryAccumulator()
    basis_coeff_acc = VectorSummaryAccumulator()
    basis_weighted_coeff_acc = VectorSummaryAccumulator()
    total_samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        physics_v1 = batch.get("physics_v1")
        if physics_v1 is not None:
            physics_v1 = physics_v1.to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        total_power = batch["total_power_W"].to(device, non_blocking=True).float()
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        graph_batch = prepare_graph_batch(batch, graph_enabled, graph_stats, device)
        model_input = build_model_input(
            x,
            physics,
            stats,
            physics_input_mode=physics_input_mode,
            physics_v1=physics_v1,
        )
        batch_size = int(x.shape[0])

        optimizer.zero_grad(set_to_none=True)
        if prediction_mode in DIRECT_PREDICTION_MODES:
            if direct_target_stats is None:
                raise ValueError("direct-temperature training requires train-fitted target statistics")
            pred_direct = call_model(
                model,
                model_input,
                metadata_input,
                graph_batch,
                conditioned=conditioned,
                graph_enabled=False,
            )
            if not torch.is_tensor(pred_direct) or pred_direct.ndim != 4 or pred_direct.shape[1] != 1:
                raise ValueError(
                    "direct-temperature model must return [B,1,H,W], "
                    f"got {type(pred_direct).__name__} {getattr(pred_direct, 'shape', None)}"
                )
            target_direct = normalize_direct_temperature(
                temperature,
                direct_target_stats,
            ).unsqueeze(1)
            direct_map_loss = F.l1_loss(pred_direct, target_direct)
            pred_temperature = unnormalize_direct_temperature(
                pred_direct.squeeze(1),
                direct_target_stats,
            )
            final_loss = F.l1_loss(pred_temperature, temperature)
            residual_loss = direct_map_loss
            temp_loss_K = final_loss
            temp_loss_scaled = direct_map_loss
            hotspot_loss_K = pred_direct.new_zeros(())
            hotspot_loss_scaled = pred_direct.new_zeros(())
            mean_loss = pred_direct.new_zeros(())
            centered_spatial_loss = pred_direct.new_zeros(())
            coarse_spatial_loss = pred_direct.new_zeros(())
            weighted_coarse_spatial_loss = pred_direct.new_zeros(())
            loss = direct_map_loss
        elif decomposed:
            outputs = call_model(
                model,
                model_input,
                metadata_input,
                graph_batch,
                conditioned=conditioned,
                graph_enabled=graph_enabled,
                total_power_W=total_power,
            )
            pred_temperature = reconstruct_decomposed_temperature(outputs, ambient, physics, mean_head_mode=mean_head_mode)
            targets = decomposed_targets(temperature, ambient, physics, total_power, mean_head_mode=mean_head_mode)
            mean_target = targets["mean_correction_K"]
            decomposed_losses = compute_decomposed_training_losses(
                pred_temperature=pred_temperature,
                true_temperature=temperature,
                pred_mean=outputs["mean_rise"],
                true_mean=mean_target,
                pred_centered=outputs["centered_field"],
                true_centered=targets["centered_field_K"],
                lambda_final=lambda_final,
                lambda_mean=lambda_mean,
                coarse_spatial_loss_enabled=coarse_spatial_loss_enabled,
                coarse_spatial_loss_weight=coarse_spatial_loss_weight,
                coarse_spatial_loss_size=coarse_spatial_loss_size,
                coarse_spatial_loss_type=coarse_spatial_loss_type,
            )
            final_loss = decomposed_losses["final_map_loss_K"]
            mean_loss = decomposed_losses["mean_loss_K"]
            centered_spatial_loss = decomposed_losses["centered_spatial_loss_K"]
            coarse_spatial_loss = decomposed_losses["coarse_spatial_loss_K"]
            weighted_coarse_spatial_loss = decomposed_losses["weighted_coarse_spatial_loss"]
            residual_loss = final_loss
            temp_loss_K = final_loss
            temp_loss_scaled = mean_loss
            hotspot_loss_K = pred_temperature.new_tensor(0.0)
            hotspot_loss_scaled = pred_temperature.new_tensor(0.0)
            loss = decomposed_losses["total_loss"]
            if lambda_chiplet_mean > 0.0:
                if graph_batch is None:
                    raise ValueError("--lambda-chiplet-mean requires graph-enabled training data")
                chiplet_loss = chiplet_mean_loss(pred_temperature, temperature, graph_batch)
                loss = loss + float(lambda_chiplet_mean) * chiplet_loss
                chiplet_mean_loss_total += float(chiplet_loss.item()) * batch_size
            alpha = outputs.get("physics_gate_alpha")
            if alpha is not None:
                gate_acc.update(alpha)
                if physics_gate_regularization > 0.0:
                    gate_reg = torch.mean((alpha - float(physics_gate_init)) ** 2)
                    loss = loss + float(physics_gate_regularization) * gate_reg
                    gate_regularization_total += float(gate_reg.item()) * batch_size
            graph_correction = outputs.get("graph_correction_field")
            if graph_correction is not None:
                graph_mag = torch.mean(torch.abs(graph_correction))
                graph_correction_total += float(graph_mag.item()) * batch_size
                if lambda_graph > 0.0:
                    loss = loss + float(lambda_graph) * graph_mag
                    graph_regularization_total += float(graph_mag.item()) * batch_size
            global_correction = outputs.get("global_correction_field")
            if global_correction is not None:
                global_correction_total += float(torch.mean(torch.abs(global_correction)).item()) * batch_size
            update_pairwise_summaries(
                outputs,
                pairwise_k_acc,
                pairwise_contribution_acc,
                pairwise_self_acc,
                pairwise_node_acc,
                basis_coeff_acc,
                basis_weighted_coeff_acc,
            )
        else:
            target = normalize_residual(residual, stats).unsqueeze(1)
            pred = call_model(model, model_input, metadata_input, graph_batch, conditioned=conditioned, graph_enabled=graph_enabled)
            residual_loss = criterion(pred, target)
            if temp_loss_weight > 0.0 or hotspot_loss_weight > 0.0:
                pred_residual_K = unnormalize_residual(pred.squeeze(1), stats)
                pred_temperature = physics + pred_residual_K
            if temp_loss_weight > 0.0:
                temp_loss_K = temp_criterion(pred_temperature, temperature)
                temp_loss_scaled = temp_loss_K / max(float(stats.residual_std), 1.0e-8)
            else:
                temp_loss_K = pred.new_tensor(0.0)
                temp_loss_scaled = pred.new_tensor(0.0)
            if hotspot_loss_weight > 0.0:
                hotspot_loss_K = hotspot_l1_loss(pred_temperature, temperature, hotspot_top_frac)
                hotspot_loss_scaled = hotspot_loss_K / max(float(stats.residual_std), 1.0e-8)
            else:
                hotspot_loss_K = pred.new_tensor(0.0)
                hotspot_loss_scaled = pred.new_tensor(0.0)
            loss = (
                residual_loss
                + float(temp_loss_weight) * temp_loss_scaled
                + float(hotspot_loss_weight) * hotspot_loss_scaled
            )
            final_loss = residual_loss
            mean_loss = pred.new_tensor(0.0)
            centered_spatial_loss = pred.new_tensor(0.0)
            coarse_spatial_loss = pred.new_tensor(0.0)
            weighted_coarse_spatial_loss = pred.new_tensor(0.0)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_size
        residual_loss_total += float(residual_loss.item()) * batch_size
        temp_loss_scaled_total += float(temp_loss_scaled.item()) * batch_size
        temp_loss_K_total += float(temp_loss_K.item()) * batch_size
        final_map_loss_total += float(final_loss.item()) * batch_size
        mean_loss_total += float(mean_loss.item()) * batch_size
        centered_spatial_loss_total += float(centered_spatial_loss.item()) * batch_size
        coarse_spatial_loss_total += float(coarse_spatial_loss.item()) * batch_size
        weighted_coarse_spatial_loss_total += float(weighted_coarse_spatial_loss.item()) * batch_size
        hotspot_loss_scaled_total += float(hotspot_loss_scaled.item()) * batch_size
        hotspot_loss_K_total += float(hotspot_loss_K.item()) * batch_size
        total_samples += batch_size
    denominator = max(total_samples, 1)
    return {
        "total_loss": total_loss / denominator,
        "direct_map_loss": residual_loss_total / denominator
        if prediction_mode in DIRECT_PREDICTION_MODES
        else 0.0,
        "residual_loss": residual_loss_total / denominator,
        "temp_loss_scaled": temp_loss_scaled_total / denominator,
        "temp_loss_K": temp_loss_K_total / denominator,
        "final_map_loss_K": final_map_loss_total / denominator,
        "mean_loss_K": mean_loss_total / denominator,
        "centered_spatial_loss_K": centered_spatial_loss_total / denominator,
        "coarse_spatial_loss_K": coarse_spatial_loss_total / denominator,
        "weighted_coarse_spatial_loss": weighted_coarse_spatial_loss_total / denominator,
        "hotspot_loss_scaled": hotspot_loss_scaled_total / denominator,
        "hotspot_loss_K": hotspot_loss_K_total / denominator,
        "gate_regularization": gate_regularization_total / denominator,
        "graph_regularization": graph_regularization_total / denominator,
        "graph_correction_abs_mean": graph_correction_total / denominator,
        "global_correction_abs_mean": global_correction_total / denominator,
        "chiplet_mean_loss_K": chiplet_mean_loss_total / denominator,
        **gate_acc.prefixed("gate_alpha"),
        **pairwise_k_acc.prefixed("pairwise_k"),
        **pairwise_contribution_acc.prefixed("pairwise_contribution"),
        **pairwise_self_acc.prefixed("pairwise_self"),
        **pairwise_node_acc.prefixed("pairwise_node_correction"),
        **basis_coeff_acc.prefixed("pairwise_basis_coeff"),
        **basis_weighted_coeff_acc.prefixed("pairwise_basis_weighted_coeff"),
    }


def should_compute_train_mae(epoch: int, epochs: int, cadence: int) -> bool:
    return cadence > 0 and (epoch == 1 or epoch == epochs or epoch % cadence == 0)


def format_optional_mae(value: float | None) -> str:
    if value is None:
        return "skip"
    return f"{value:.3f}K"


def format_gate_summary(summary: dict[str, float] | None) -> str:
    if not summary:
        return "n/a"
    return f"{summary['mean']:.3f}/{summary['std']:.3f}"


def format_graph_epoch_summary(metrics: dict[str, Any], by_case: dict[str, dict[str, dict[str, float]]]) -> str:
    if "graph_correction_abs_mean" not in metrics and "global_correction_abs_mean" not in metrics:
        return "graph/global=n/a"
    graph_abs = metrics.get("graph_correction_abs_mean", {}).get("mean", float("nan"))
    graph_ratio = metrics.get("graph_to_cnn_ratio", float("nan"))
    cnn_only = metrics.get("cnn_only_final_temperature", {}).get("mae_K")
    fused = metrics.get("final_temperature", {}).get("mae_K")
    case02 = by_case.get("case02", {})
    case02_cnn = case02.get("cnn_only_final_temperature", {}).get("mae_K")
    case02_fused = case02.get("final_temperature", {}).get("mae_K")
    parts = []
    if "global_correction_abs_mean" in metrics:
        parts.append(f"global_abs={metrics['global_correction_abs_mean']['mean']:.3f}")
    if "graph_correction_abs_mean" in metrics:
        parts.extend([f"graph_abs={graph_abs:.3f}", f"graph_ratio={graph_ratio:.3f}"])
    if cnn_only is not None and fused is not None:
        parts.append(f"cnn_only={cnn_only:.3f}K")
        parts.append(f"fused={fused:.3f}K")
        parts.append(f"delta={cnn_only - fused:.3f}K")
    if case02_cnn is not None and case02_fused is not None:
        parts.append(f"case02_delta={case02_cnn - case02_fused:.3f}K")
    chiplet_mean = metrics.get("chiplet_mean_temperature", {}).get("mae_K")
    if chiplet_mean is not None:
        parts.append(f"chiplet_mean={chiplet_mean:.3f}K")
    pairwise_k = metrics.get("pairwise_k", {}).get("abs_mean")
    if pairwise_k is not None:
        parts.append(f"K_abs={pairwise_k:.4f}")
    return " ".join(parts)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    stats: NormalizationStats,
    device: torch.device,
    *,
    decomposed: bool = False,
    conditioned: bool = False,
    lambda_final: float = 1.0,
    lambda_mean: float = 0.1,
    physics_input_mode: str = "v1",
    graph_enabled: bool = False,
    graph_stats: Any | None = None,
    lambda_chiplet_mean: float = 0.0,
    mean_head_mode: str = "direct_k",
    prediction_mode: str = "residual",
    direct_target_stats: DirectTemperatureTargetStats | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    model.eval()
    residual_acc = MetricAccumulator()
    final_acc = MetricAccumulator()
    cnn_only_final_acc = MetricAccumulator()
    centered_acc = MetricAccumulator()
    cnn_only_centered_acc = MetricAccumulator()
    mean_acc = ScalarMetricAccumulator()
    delta_R_acc = ScalarMetricAccumulator()
    mean_bias_removed_acc = MetricAccumulator()
    gate_acc = ScalarSummaryAccumulator()
    graph_correction_acc = ScalarSummaryAccumulator()
    graph_correction_max_acc = ScalarSummaryAccumulator()
    graph_correction_rms_acc = ScalarSummaryAccumulator()
    graph_correction_std_acc = ScalarSummaryAccumulator()
    global_correction_acc = ScalarSummaryAccumulator()
    global_correction_max_acc = ScalarSummaryAccumulator()
    global_correction_rms_acc = ScalarSummaryAccumulator()
    global_correction_std_acc = ScalarSummaryAccumulator()
    global_correction_low_freq_acc = ScalarSummaryAccumulator()
    cnn_centered_abs_acc = ScalarSummaryAccumulator()
    final_centered_abs_acc = ScalarSummaryAccumulator()
    chiplet_mean_acc = ScalarMetricAccumulator()
    chiplet_peak_acc = ScalarMetricAccumulator()
    chiplet_delta_acc = ScalarSummaryAccumulator()
    pairwise_k_acc = ScalarSummaryAccumulator()
    pairwise_contribution_acc = ScalarSummaryAccumulator()
    pairwise_self_acc = ScalarSummaryAccumulator()
    pairwise_node_acc = ScalarSummaryAccumulator()
    basis_coeff_acc = VectorSummaryAccumulator()
    basis_weighted_coeff_acc = VectorSummaryAccumulator()
    by_case: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {
            "residual": MetricAccumulator(),
            "final_temperature": MetricAccumulator(),
            "cnn_only_final_temperature": MetricAccumulator(),
        }
    )
    chiplet_by_case: dict[str, dict[str, ScalarMetricAccumulator | ScalarSummaryAccumulator]] = defaultdict(
        lambda: {
            "chiplet_mean_temperature": ScalarMetricAccumulator(),
            "chiplet_peak_temperature": ScalarMetricAccumulator(),
            "inter_chiplet_delta_T": ScalarSummaryAccumulator(),
        }
    )
    total_loss = 0.0
    total_samples = 0
    worse_than_physics_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        physics_v1 = batch.get("physics_v1")
        if physics_v1 is not None:
            physics_v1 = physics_v1.to(device, non_blocking=True)
        residual = batch["residual"].to(device, non_blocking=True)
        temperature = batch["temperature"].to(device, non_blocking=True)
        ambient = batch["ambient_K"].to(device, non_blocking=True).float()
        total_power = batch["total_power_W"].to(device, non_blocking=True).float()
        metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(device, non_blocking=True)
        graph_batch = prepare_graph_batch(batch, graph_enabled, graph_stats, device)
        model_input = build_model_input(
            x,
            physics,
            stats,
            physics_input_mode=physics_input_mode,
            physics_v1=physics_v1,
        )
        if prediction_mode in DIRECT_PREDICTION_MODES:
            if direct_target_stats is None:
                raise ValueError("direct-temperature evaluation requires train-fitted target statistics")
            pred_direct = call_model(
                model,
                model_input,
                metadata_input,
                graph_batch,
                conditioned=conditioned,
                graph_enabled=False,
            )
            if not torch.is_tensor(pred_direct) or pred_direct.ndim != 4 or pred_direct.shape[1] != 1:
                raise ValueError("direct-temperature model must return [B,1,H,W]")
            target_direct = normalize_direct_temperature(
                temperature,
                direct_target_stats,
            ).unsqueeze(1)
            loss = F.l1_loss(pred_direct, target_direct)
            pred_temperature = unnormalize_direct_temperature(
                pred_direct.squeeze(1),
                direct_target_stats,
            )
            pred_residual = pred_temperature - physics
        elif decomposed:
            outputs = call_model(
                model,
                model_input,
                metadata_input,
                graph_batch,
                conditioned=conditioned,
                graph_enabled=graph_enabled,
                total_power_W=total_power,
            )
            pred_temperature = reconstruct_decomposed_temperature(outputs, ambient, physics, mean_head_mode=mean_head_mode)
            pred_residual = pred_temperature - physics
            targets = decomposed_targets(temperature, ambient, physics, total_power, mean_head_mode=mean_head_mode)
            mean_target = targets["mean_correction_K"]
            final_loss = torch.nn.functional.smooth_l1_loss(pred_temperature, temperature)
            mean_loss = torch.nn.functional.smooth_l1_loss(outputs["mean_rise"], mean_target)
            loss = float(lambda_final) * final_loss + float(lambda_mean) * mean_loss
            centered_pred = outputs["centered_field"]
            centered_target = targets["centered_field_K"]
            mean_acc.update(outputs["mean_rise"], mean_target)
            if "delta_R_eff" in outputs:
                delta_R_acc.update(outputs["delta_R_eff"], targets["delta_R_eff_K_per_W"])
            centered_acc.update(centered_pred, centered_target)
            mean_bias_removed_acc.update(centered_pred, centered_target)
            alpha = outputs.get("physics_gate_alpha")
            if alpha is not None:
                gate_acc.update(alpha)
            graph_correction = outputs.get("graph_correction_field")
            if graph_correction is not None:
                graph_abs = graph_correction.abs()
                graph_correction_acc.update(graph_abs.mean(dim=(-2, -1)))
                graph_correction_max_acc.update(graph_abs.amax(dim=(-2, -1)))
                graph_correction_rms_acc.update(torch.sqrt(torch.mean(graph_correction * graph_correction, dim=(-2, -1))))
                graph_correction_std_acc.update(graph_correction.std(dim=(-2, -1)))
                cnn_centered = outputs["cnn_centered_field"]
                cnn_centered_abs_acc.update(cnn_centered.abs().mean(dim=(-2, -1)))
                final_centered_abs_acc.update(outputs["centered_field"].abs().mean(dim=(-2, -1)))
                cnn_mean_rise = outputs.get("cnn_mean_rise", outputs["mean_rise"])
                if mean_head_mode == "residual_resistance":
                    cnn_only_temperature = physics + cnn_mean_rise[:, None, None] + cnn_centered
                else:
                    cnn_only_temperature = ambient[:, None, None] + cnn_mean_rise[:, None, None] + cnn_centered
                cnn_only_centered = cnn_centered
                cnn_only_final_acc.update(cnn_only_temperature, temperature)
                cnn_only_centered_acc.update(cnn_only_centered, centered_target)
            global_correction = outputs.get("global_correction_field")
            if global_correction is not None:
                global_abs = global_correction.abs()
                global_correction_acc.update(global_abs.mean(dim=(-2, -1)))
                global_correction_max_acc.update(global_abs.amax(dim=(-2, -1)))
                global_correction_rms_acc.update(torch.sqrt(torch.mean(global_correction * global_correction, dim=(-2, -1))))
                global_correction_std_acc.update(global_correction.std(dim=(-2, -1)))
                if "global_correction_low_frequency_energy" in outputs:
                    global_correction_low_freq_acc.update(outputs["global_correction_low_frequency_energy"])
            update_pairwise_summaries(
                outputs,
                pairwise_k_acc,
                pairwise_contribution_acc,
                pairwise_self_acc,
                pairwise_node_acc,
                basis_coeff_acc,
                basis_weighted_coeff_acc,
            )
        else:
            target_norm = normalize_residual(residual, stats).unsqueeze(1)
            pred_norm = model(model_input, metadata_input) if conditioned else model(model_input)
            loss = criterion(pred_norm, target_norm)
            pred_residual = unnormalize_residual(pred_norm.squeeze(1), stats)
            pred_temperature = physics + pred_residual
        case_ids = metadata_values(batch["metadata"], "case_id", int(x.shape[0]))

        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        final_sample_mae = (pred_temperature - temperature).abs().reshape(batch_size, -1).mean(dim=1)
        physics_sample_mae = (physics - temperature).abs().reshape(batch_size, -1).mean(dim=1)
        worse_than_physics_count += int((final_sample_mae > physics_sample_mae).sum().item())
        residual_acc.update(pred_residual, residual)
        final_acc.update(pred_temperature, temperature)
        chiplet_metrics = None
        if graph_enabled and graph_batch is not None:
            chiplet_metrics = chiplet_metric_values(pred_temperature, temperature, graph_batch)
            chiplet_mean_acc.update(chiplet_metrics["pred_mean"], chiplet_metrics["target_mean"])
            chiplet_peak_acc.update(chiplet_metrics["pred_peak"], chiplet_metrics["target_peak"])
            chiplet_delta_acc.update(chiplet_metrics["delta_mae"].reshape(1))
        for index, case_id in enumerate(case_ids):
            by_case[str(case_id)]["residual"].update(pred_residual[index : index + 1], residual[index : index + 1])
            by_case[str(case_id)]["final_temperature"].update(pred_temperature[index : index + 1], temperature[index : index + 1])
            if decomposed and "cnn_only_temperature" in locals():
                by_case[str(case_id)]["cnn_only_final_temperature"].update(
                    cnn_only_temperature[index : index + 1],
                    temperature[index : index + 1],
                )
        if chiplet_metrics is not None:
            update_chiplet_case_metrics(chiplet_by_case, case_ids, chiplet_metrics, graph_batch)
        if "cnn_only_temperature" in locals():
            del cnn_only_temperature

    metrics = {
        "normalized_residual_loss": total_loss / max(total_samples, 1),
        "direct_map_loss": (
            total_loss / max(total_samples, 1)
            if prediction_mode in DIRECT_PREDICTION_MODES
            else None
        ),
        "prediction_mode": prediction_mode,
        "residual": residual_acc.compute(),
        "final_temperature": final_acc.compute(),
        "worse_than_physics_baseline_fraction": worse_than_physics_count / max(total_samples, 1),
    }
    if decomposed:
        metrics["mean_rise"] = mean_acc.compute()
        delta_summary = delta_R_acc.compute()
        if delta_summary:
            metrics["delta_R_eff"] = rename_scalar_metric_units(delta_summary, "K_per_W")
        metrics["centered_field"] = centered_acc.compute()
        metrics["mean_bias_removed"] = mean_bias_removed_acc.compute()
    cnn_only_summary = cnn_only_final_acc.compute()
    if cnn_only_summary:
        metrics["cnn_only_final_temperature"] = cnn_only_summary
        metrics["cnn_only_centered_field"] = cnn_only_centered_acc.compute()
        metrics["graph_delta_val_mae_K"] = cnn_only_summary["mae_K"] - metrics["final_temperature"]["mae_K"]
    graph_summary = graph_correction_acc.compute()
    if graph_summary:
        metrics["graph_correction_abs_mean"] = graph_summary
        metrics["graph_correction_abs_max"] = graph_correction_max_acc.compute()
        metrics["graph_correction_rms"] = graph_correction_rms_acc.compute()
        metrics["graph_correction_spatial_std"] = graph_correction_std_acc.compute()
        metrics["cnn_centered_field_abs_mean"] = cnn_centered_abs_acc.compute()
        metrics["final_centered_field_abs_mean"] = final_centered_abs_acc.compute()
        denominator = max(float(metrics["cnn_centered_field_abs_mean"]["mean"]), 1.0e-8)
        metrics["graph_to_cnn_ratio"] = float(metrics["graph_correction_abs_mean"]["mean"] / denominator)
    global_summary = global_correction_acc.compute()
    if global_summary:
        metrics["global_correction_abs_mean"] = global_summary
        metrics["global_correction_abs_max"] = global_correction_max_acc.compute()
        metrics["global_correction_rms"] = global_correction_rms_acc.compute()
        metrics["global_correction_spatial_std"] = global_correction_std_acc.compute()
        low_freq_summary = global_correction_low_freq_acc.compute()
        if low_freq_summary:
            metrics["global_correction_low_frequency_energy"] = low_freq_summary
    chiplet_mean_summary = chiplet_mean_acc.compute()
    if chiplet_mean_summary:
        metrics["chiplet_mean_temperature"] = chiplet_mean_summary
        metrics["chiplet_peak_temperature"] = chiplet_peak_acc.compute()
        metrics["inter_chiplet_delta_T"] = chiplet_delta_acc.compute()
    pairwise_summary = pairwise_k_acc.compute()
    if pairwise_summary:
        metrics["pairwise_k"] = pairwise_summary
        metrics["pairwise_contribution"] = pairwise_contribution_acc.compute()
        metrics["pairwise_self"] = pairwise_self_acc.compute()
        metrics["pairwise_node_correction"] = pairwise_node_acc.compute()
    basis_summary = basis_coeff_acc.compute()
    if basis_summary:
        metrics["pairwise_basis_coeff"] = basis_summary
        metrics["pairwise_basis_weighted_coeff"] = basis_weighted_coeff_acc.compute()
    gate_summary = gate_acc.compute()
    if gate_summary:
        metrics["physics_gate"] = gate_summary
    case_metrics = {
        case_id: {
            "residual": accs["residual"].compute(),
            "final_temperature": accs["final_temperature"].compute(),
            "cnn_only_final_temperature": accs["cnn_only_final_temperature"].compute(),
            "chiplet_mean_temperature": chiplet_by_case[case_id]["chiplet_mean_temperature"].compute(),
            "chiplet_peak_temperature": chiplet_by_case[case_id]["chiplet_peak_temperature"].compute(),
            "inter_chiplet_delta_T": chiplet_by_case[case_id]["inter_chiplet_delta_T"].compute(),
        }
        for case_id, accs in sorted(by_case.items())
    }
    if "case02" in case_metrics and case_metrics["case02"].get("cnn_only_final_temperature"):
        cnn_only_case02 = case_metrics["case02"]["cnn_only_final_temperature"]
        fused_case02 = case_metrics["case02"]["final_temperature"]
        if cnn_only_case02 and fused_case02:
            metrics["case02_cnn_only_mae_K"] = cnn_only_case02["mae_K"]
            metrics["case02_fused_mae_K"] = fused_case02["mae_K"]
            metrics["case02_graph_delta_mae_K"] = cnn_only_case02["mae_K"] - fused_case02["mae_K"]
    return metrics, case_metrics


@torch.no_grad()
def compute_delta_R_eff_target_stats(loader: DataLoader[dict[str, Any]]) -> dict[str, Any]:
    """Fit train-only scalar target statistics for residual effective resistance."""
    values: list[torch.Tensor] = []
    total_power_values: list[torch.Tensor] = []
    for batch in loader:
        temperature = batch["temperature"].float()
        physics = batch["physics"].float()
        total_power = batch["total_power_W"].float().view(-1)
        if torch.any(total_power <= 0.0):
            bad = torch.nonzero(total_power <= 0.0, as_tuple=False).flatten().tolist()
            raise ValueError(f"residual_resistance target requires strictly positive total_power_W; bad batch indices={bad}")
        mean_residual = (temperature - physics).mean(dim=(-2, -1))
        values.append(mean_residual / total_power)
        total_power_values.append(total_power)
    if not values:
        raise ValueError("cannot fit delta_R_eff target normalization on an empty training loader")
    stacked = torch.cat(values).double()
    power = torch.cat(total_power_values).double()
    std = float(stacked.std(unbiased=False).item())
    if not np.isfinite(std) or std <= 1.0e-12:
        std = 1.0
    return {
        "target_name": "delta_R_eff_true_K_per_W",
        "units": "K/W",
        "normalization_mode": "train_split_standardization",
        "fit_scope": "train split only",
        "count": int(stacked.numel()),
        "mean_K_per_W": float(stacked.mean().item()),
        "std_K_per_W": std,
        "min_K_per_W": float(stacked.min().item()),
        "max_K_per_W": float(stacked.max().item()),
        "total_power_min_W": float(power.min().item()),
        "total_power_max_W": float(power.max().item()),
        "total_power_mean_W": float(power.mean().item()),
        "formula": "mean(HotSpot_K - source_superposition_base_K) / total_power_W",
    }


def decomposed_targets(
    temperature: torch.Tensor,
    ambient: torch.Tensor,
    physics: torch.Tensor,
    total_power: torch.Tensor,
    *,
    mean_head_mode: str,
) -> dict[str, torch.Tensor]:
    if mean_head_mode == "residual_resistance":
        total_power_flat = total_power.to(device=temperature.device, dtype=temperature.dtype).view(-1)
        if torch.any(total_power_flat <= 0.0):
            raise ValueError("residual_resistance target requires strictly positive total_power_W")
        residual = temperature - physics
        mean_correction = residual.mean(dim=(-2, -1))
        centered = residual - mean_correction[:, None, None]
        return {
            "mean_correction_K": mean_correction,
            "centered_field_K": centered,
            "delta_R_eff_K_per_W": mean_correction / total_power_flat,
        }
    if mean_head_mode != "direct_k":
        raise ValueError(f"unsupported mean_head_mode: {mean_head_mode}")
    mean_rise = (temperature - ambient[:, None, None]).mean(dim=(-2, -1))
    centered = temperature - temperature.mean(dim=(-2, -1), keepdim=True)
    return {
        "mean_correction_K": mean_rise,
        "centered_field_K": centered,
        "delta_R_eff_K_per_W": torch.zeros_like(mean_rise),
    }


def reconstruct_decomposed_temperature(
    outputs: dict[str, torch.Tensor],
    ambient: torch.Tensor,
    physics: torch.Tensor | None = None,
    *,
    mean_head_mode: str = "direct_k",
) -> torch.Tensor:
    centered = outputs["centered_field"]
    centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
    if mean_head_mode == "residual_resistance":
        if physics is None:
            raise ValueError("residual_resistance reconstruction requires the physics/base tensor")
        return physics + outputs["mean_rise"][:, None, None] + centered
    if mean_head_mode != "direct_k":
        raise ValueError(f"unsupported mean_head_mode: {mean_head_mode}")
    return ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered


def rename_scalar_metric_units(metrics: dict[str, float], suffix: str) -> dict[str, float]:
    renamed: dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith("_K"):
            renamed[f"{key[:-2]}_{suffix}"] = value
        else:
            renamed[key] = value
    return renamed


def prepare_graph_batch(
    batch: dict[str, Any],
    graph_enabled: bool,
    graph_stats: Any | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if not graph_enabled:
        return None
    graph = batch.get("graph")
    if graph is None:
        raise ValueError("graph-enabled model requires graph data from the dataset")
    graph = move_graph_to_device(graph, device)
    return normalize_graph_batch(graph, graph_stats)


def call_model(
    model: nn.Module,
    model_input: torch.Tensor,
    metadata_input: torch.Tensor | None,
    graph_batch: dict[str, torch.Tensor] | None,
    *,
    conditioned: bool,
    graph_enabled: bool,
    total_power_W: torch.Tensor | None = None,
) -> Any:
    if graph_enabled:
        kwargs: dict[str, Any] = {}
        if getattr(model, "mean_head_mode", "direct_k") == "residual_resistance":
            kwargs["total_power_W"] = total_power_W
        return model(model_input, metadata_input, graph_batch, **kwargs)
    if conditioned:
        if getattr(model, "mean_head_mode", "direct_k") == "residual_resistance":
            return model(model_input, metadata_input, total_power_W=total_power_W)
        return model(model_input, metadata_input)
    return model(model_input)


def update_pairwise_summaries(
    outputs: dict[str, torch.Tensor],
    k_acc: ScalarSummaryAccumulator,
    contribution_acc: ScalarSummaryAccumulator,
    self_acc: ScalarSummaryAccumulator,
    node_acc: ScalarSummaryAccumulator,
    basis_coeff_acc: VectorSummaryAccumulator | None = None,
    basis_weighted_coeff_acc: VectorSummaryAccumulator | None = None,
) -> None:
    if "pairwise_k_values" in outputs:
        k_acc.update(outputs["pairwise_k_values"])
        contribution_acc.update(outputs["pairwise_contributions"])
        self_acc.update(outputs["pairwise_self_corrections"])
        node_acc.update(outputs["pairwise_node_corrections"])
    if "pairwise_basis_coefficients" in outputs:
        if basis_coeff_acc is not None:
            basis_coeff_acc.update(outputs["pairwise_basis_coefficients"])
        if basis_weighted_coeff_acc is not None:
            basis_weighted_coeff_acc.update(outputs["pairwise_basis_weighted_coefficients"])


def update_chiplet_case_metrics(
    by_case: dict[str, dict[str, ScalarMetricAccumulator | ScalarSummaryAccumulator]],
    case_ids: list[Any],
    chiplet_metrics: dict[str, torch.Tensor],
    graph_batch: dict[str, torch.Tensor],
) -> None:
    node_batch = graph_batch["node_batch"].detach().cpu().long()
    pred_mean = chiplet_metrics["pred_mean"].detach().cpu()
    target_mean = chiplet_metrics["target_mean"].detach().cpu()
    pred_peak = chiplet_metrics["pred_peak"].detach().cpu()
    target_peak = chiplet_metrics["target_peak"].detach().cpu()
    num_graphs = len(case_ids)
    for graph_index in range(num_graphs):
        case_id = str(case_ids[graph_index])
        node_indices = torch.nonzero(node_batch == graph_index, as_tuple=False).reshape(-1)
        if node_indices.numel() == 0:
            continue
        by_case[case_id]["chiplet_mean_temperature"].update(
            pred_mean.index_select(0, node_indices),
            target_mean.index_select(0, node_indices),
        )
        by_case[case_id]["chiplet_peak_temperature"].update(
            pred_peak.index_select(0, node_indices),
            target_peak.index_select(0, node_indices),
        )
        if node_indices.numel() >= 2:
            pred = pred_mean.index_select(0, node_indices)
            target = target_mean.index_select(0, node_indices)
            pairs = torch.triu_indices(int(node_indices.numel()), int(node_indices.numel()), offset=1)
            pred_delta = pred[pairs[0]] - pred[pairs[1]]
            target_delta = target[pairs[0]] - target[pairs[1]]
            by_case[case_id]["inter_chiplet_delta_T"].update((pred_delta - target_delta).abs())


def hotspot_l1_loss(pred_temperature: torch.Tensor, temperature: torch.Tensor, top_frac: float) -> torch.Tensor:
    if pred_temperature.shape != temperature.shape:
        raise ValueError(f"pred_temperature shape {pred_temperature.shape} does not match temperature shape {temperature.shape}")
    batch_size = int(temperature.shape[0])
    flat_temperature = temperature.reshape(batch_size, -1)
    flat_error = torch.abs(pred_temperature - temperature).reshape(batch_size, -1)
    num_cells = int(flat_temperature.shape[1])
    k = max(1, int(np.ceil(num_cells * float(top_frac))))
    top_indices = torch.topk(flat_temperature, k=k, dim=1, largest=True, sorted=False).indices
    hotspot_error = torch.gather(flat_error, dim=1, index=top_indices)
    return hotspot_error.mean()


class MetricAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_cells = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
        self.max_abs = 0.0
        self.hotspot_temp_error_sum = 0.0
        self.hotspot_location_error_sum = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_cpu = pred.detach().float().cpu()
        target_cpu = target.detach().float().cpu()
        error = pred_cpu - target_cpu
        abs_error = error.abs()
        self.num_samples += int(pred_cpu.shape[0])
        self.num_cells += int(error.numel())
        self.sum_abs += float(abs_error.sum().item())
        self.sum_sq += float((error * error).sum().item())
        self.sum_signed += float(error.sum().item())
        self.max_abs = max(self.max_abs, float(abs_error.max().item()))
        for pred_item, target_item in zip(pred_cpu, target_cpu):
            pred_flat = pred_item.reshape(-1)
            target_flat = target_item.reshape(-1)
            pred_idx = int(torch.argmax(pred_flat).item())
            target_idx = int(torch.argmax(target_flat).item())
            pred_row, pred_col = divmod(pred_idx, pred_item.shape[-1])
            target_row, target_col = divmod(target_idx, target_item.shape[-1])
            self.hotspot_temp_error_sum += float(pred_flat[pred_idx].item() - target_flat[target_idx].item())
            self.hotspot_location_error_sum += float(((pred_row - target_row) ** 2 + (pred_col - target_col) ** 2) ** 0.5)

    def compute(self) -> dict[str, float]:
        if self.num_cells == 0:
            return {}
        return {
            "num_samples": float(self.num_samples),
            "mae_K": self.sum_abs / self.num_cells,
            "rmse_K": (self.sum_sq / self.num_cells) ** 0.5,
            "max_abs_error_K": self.max_abs,
            "mean_signed_error_K": self.sum_signed / self.num_cells,
            "hotspot_temp_error_K": self.hotspot_temp_error_sum / max(self.num_samples, 1),
            "hotspot_location_error_cells": self.hotspot_location_error_sum / max(self.num_samples, 1),
        }


class ScalarMetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        error = pred.detach().float().cpu() - target.detach().float().cpu()
        self.count += int(error.numel())
        self.sum_abs += float(error.abs().sum().item())
        self.sum_sq += float((error * error).sum().item())
        self.sum_signed += float(error.sum().item())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "mae_K": self.sum_abs / self.count,
            "rmse_K": (self.sum_sq / self.count) ** 0.5,
            "mean_signed_error_K": self.sum_signed / self.count,
        }


class ScalarSummaryAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: torch.Tensor) -> None:
        data = value.detach().float().reshape(-1).cpu().tolist()
        self.values.extend(float(item) for item in data)

    def compute(self) -> dict[str, float]:
        if not self.values:
            return {}
        array = np.asarray(self.values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "abs_mean": float(np.abs(array).mean()),
        }

    def prefixed(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in self.compute().items()}


class VectorSummaryAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total: torch.Tensor | None = None
        self.total_abs: torch.Tensor | None = None
        self.total_sq: torch.Tensor | None = None
        self.positive: torch.Tensor | None = None
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        data = value.detach().float().reshape(-1, value.shape[-1]).cpu()
        if data.numel() == 0:
            return
        if self.total is None:
            dim = int(data.shape[1])
            self.total = torch.zeros(dim, dtype=torch.float64)
            self.total_abs = torch.zeros(dim, dtype=torch.float64)
            self.total_sq = torch.zeros(dim, dtype=torch.float64)
            self.positive = torch.zeros(dim, dtype=torch.float64)
            self.minimum = torch.full((dim,), float("inf"), dtype=torch.float64)
            self.maximum = torch.full((dim,), -float("inf"), dtype=torch.float64)
        data64 = data.double()
        self.count += int(data64.shape[0])
        self.total += data64.sum(dim=0)
        self.total_abs += data64.abs().sum(dim=0)
        self.total_sq += (data64 * data64).sum(dim=0)
        self.positive += (data64 > 0.0).double().sum(dim=0)
        self.minimum = torch.minimum(self.minimum, data64.min(dim=0).values)
        self.maximum = torch.maximum(self.maximum, data64.max(dim=0).values)

    def compute(self) -> dict[str, Any]:
        if self.count == 0 or self.total is None:
            return {}
        mean = self.total / float(self.count)
        abs_mean = self.total_abs / float(self.count)
        variance = torch.clamp(self.total_sq / float(self.count) - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        positive_fraction = self.positive / float(self.count)
        by_basis = []
        for index in range(int(mean.numel())):
            by_basis.append(
                {
                    "basis_index": index,
                    "mean": float(mean[index].item()),
                    "std": float(std[index].item()),
                    "abs_mean": float(abs_mean[index].item()),
                    "min": float(self.minimum[index].item()),
                    "max": float(self.maximum[index].item()),
                    "positive_fraction": float(positive_fraction[index].item()),
                    "negative_fraction": float(1.0 - positive_fraction[index].item()),
                }
            )
        return {
            "mean": float(mean.mean().item()),
            "std": float(std.mean().item()),
            "abs_mean": float(abs_mean.mean().item()),
            "min": float(self.minimum.min().item()),
            "max": float(self.maximum.max().item()),
            "by_basis": by_basis,
        }

    def prefixed(self, prefix: str) -> dict[str, float]:
        summary = self.compute()
        if not summary:
            return {}
        return {
            f"{prefix}_mean": float(summary["mean"]),
            f"{prefix}_std": float(summary["std"]),
            f"{prefix}_abs_mean": float(summary["abs_mean"]),
            f"{prefix}_min": float(summary["min"]),
            f"{prefix}_max": float(summary["max"]),
        }


def metadata_values(metadata: dict[str, Any], key: str, batch_size: int) -> list[Any]:
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value for _ in range(batch_size)]


def make_loader(
    dataset: ChipThermDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    graph_enabled: bool = False,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=chiptherm_collate if graph_enabled else None,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: dict[str, Any],
    stats: NormalizationStats,
    metrics: dict[str, Any],
    *,
    best: bool,
    best_val_mae: float,
    epochs_without_improvement: int,
    training_lineage: dict[str, Any] | None,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "epoch": epoch,
            "best": best,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": config.get("model", model.config()),
            "training_config": config,
            "normalization": stats.to_dict(),
            "metrics": metrics,
            "best_val_mae_K": best_val_mae,
            "epochs_without_improvement": epochs_without_improvement,
            "training_lineage": training_lineage,
        },
        path,
    )


def load_initial_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, int]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    direct_matches = 0
    cnn_matches = 0
    skipped = 0
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            mapped[key] = value
            direct_matches += 1
            continue
        wrapped_key = f"cnn_model.{key}"
        if wrapped_key in target_state and tuple(target_state[wrapped_key].shape) == tuple(value.shape):
            mapped[wrapped_key] = value
            cnn_matches += 1
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return {
        "direct_tensors_loaded": direct_matches,
        "cnn_submodule_tensors_loaded": cnn_matches,
        "source_tensors_skipped": skipped,
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
    }


def train_log_columns() -> list[str]:
    return [
                "epoch",
                "physical_representation",
                "lr",
                "train_loss",
                "train_direct_map_loss",
                "train_final_map_loss_K",
                "train_mean_loss_K",
                "train_centered_spatial_loss_K",
                "train_coarse_spatial_loss_K",
                "train_weighted_coarse_spatial_loss",
                "train_residual_loss",
                "train_temp_loss_scaled",
                "train_temp_loss_K",
                "train_hotspot_loss_scaled",
                "train_hotspot_loss_K",
                "train_gate_regularization",
                "train_gate_alpha_mean",
                "train_gate_alpha_std",
                "train_gate_alpha_min",
                "train_gate_alpha_max",
                "train_graph_regularization",
                "train_graph_correction_abs_mean",
                "train_global_correction_abs_mean",
                "train_chiplet_mean_loss_K",
                "train_pairwise_k_abs_mean",
                "train_pairwise_contribution_abs_mean",
                "train_pairwise_self_abs_mean",
                "train_pairwise_node_correction_abs_mean",
                "train_pairwise_basis_coeff_abs_mean",
                "train_pairwise_basis_weighted_coeff_abs_mean",
                "train_final_temperature_mae_K",
                "val_loss",
                "val_direct_map_loss",
                "val_residual_mae_K",
                "val_residual_rmse_K",
                "val_final_mae_K",
                "val_final_rmse_K",
                "val_mean_correction_mae_K",
                "val_delta_R_eff_mae_K_per_W",
                "val_delta_R_eff_rmse_K_per_W",
                "val_worse_than_physics_fraction",
                "val_cnn_only_mae_K",
                "val_cnn_only_rmse_K",
                "val_graph_improvement_K",
                "val_case02_fused_mae_K",
                "val_case02_cnn_only_mae_K",
                "val_case02_graph_improvement_K",
                "val_hotspot_temp_error_K",
                "val_hotspot_location_error_cells",
                "val_gate_alpha_mean",
                "val_gate_alpha_std",
                "val_gate_alpha_min",
                "val_gate_alpha_max",
                "val_graph_correction_abs_mean",
                "val_graph_correction_abs_max",
                "val_graph_correction_rms",
                "val_graph_correction_spatial_std",
                "val_graph_to_cnn_ratio",
                "val_global_correction_abs_mean",
                "val_global_correction_abs_max",
                "val_global_correction_rms",
                "val_global_correction_spatial_std",
                "val_global_correction_low_frequency_energy",
                "val_chiplet_mean_mae_K",
                "val_chiplet_peak_mae_K",
                "val_inter_chiplet_delta_mae_K",
                "val_pairwise_k_abs_mean",
                "val_pairwise_contribution_abs_mean",
                "val_pairwise_self_abs_mean",
                "val_pairwise_node_correction_abs_mean",
                "val_pairwise_basis_coeff_abs_mean",
                "val_pairwise_basis_weighted_coeff_abs_mean",
                "epoch_runtime_s",
                "is_best",
            ]


def init_train_log(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerow(train_log_columns())


def ensure_train_log_schema(path: Path) -> None:
    if not path.is_file():
        init_train_log(path)
        return
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    expected = train_log_columns()
    if fieldnames == expected:
        return
    unknown = sorted(set(fieldnames) - set(expected))
    if unknown:
        raise ValueError(f"cannot migrate training log with unknown columns: {unknown}")
    temporary = path.with_suffix(path.suffix + ".schema_upgrade.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=expected)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in expected})
    temporary.replace(path)


def append_train_log(
    path: Path,
    epoch: int,
    train_losses: dict[str, float],
    train_final_mae_K: float | None,
    val_metrics: dict[str, Any],
    epoch_runtime_s: float,
    is_best: bool,
    current_lr: float,
    physical_representation: str = "dimensional",
) -> None:
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                epoch,
                physical_representation,
                current_lr,
                train_losses["total_loss"],
                train_losses.get("direct_map_loss", ""),
                train_losses["final_map_loss_K"],
                train_losses["mean_loss_K"],
                train_losses["centered_spatial_loss_K"],
                train_losses["coarse_spatial_loss_K"],
                train_losses["weighted_coarse_spatial_loss"],
                train_losses["residual_loss"],
                train_losses["temp_loss_scaled"],
                train_losses["temp_loss_K"],
                train_losses["hotspot_loss_scaled"],
                train_losses["hotspot_loss_K"],
                train_losses["gate_regularization"],
                train_losses.get("gate_alpha_mean", ""),
                train_losses.get("gate_alpha_std", ""),
                train_losses.get("gate_alpha_min", ""),
                train_losses.get("gate_alpha_max", ""),
                train_losses.get("graph_regularization", ""),
                train_losses.get("graph_correction_abs_mean", ""),
                train_losses.get("global_correction_abs_mean", ""),
                train_losses.get("chiplet_mean_loss_K", ""),
                train_losses.get("pairwise_k_abs_mean", ""),
                train_losses.get("pairwise_contribution_abs_mean", ""),
                train_losses.get("pairwise_self_abs_mean", ""),
                train_losses.get("pairwise_node_correction_abs_mean", ""),
                train_losses.get("pairwise_basis_coeff_abs_mean", ""),
                train_losses.get("pairwise_basis_weighted_coeff_abs_mean", ""),
                "" if train_final_mae_K is None else train_final_mae_K,
                val_metrics["normalized_residual_loss"],
                val_metrics.get("direct_map_loss", ""),
                val_metrics["residual"]["mae_K"],
                val_metrics["residual"]["rmse_K"],
                val_metrics["final_temperature"]["mae_K"],
                val_metrics["final_temperature"]["rmse_K"],
                val_metrics.get("mean_rise", {}).get("mae_K", ""),
                val_metrics.get("delta_R_eff", {}).get("mae_K_per_W", ""),
                val_metrics.get("delta_R_eff", {}).get("rmse_K_per_W", ""),
                val_metrics.get("worse_than_physics_baseline_fraction", ""),
                val_metrics.get("cnn_only_final_temperature", {}).get("mae_K", ""),
                val_metrics.get("cnn_only_final_temperature", {}).get("rmse_K", ""),
                val_metrics.get("graph_delta_val_mae_K", ""),
                val_metrics.get("case02_fused_mae_K", ""),
                val_metrics.get("case02_cnn_only_mae_K", ""),
                val_metrics.get("case02_graph_delta_mae_K", ""),
                val_metrics["final_temperature"]["hotspot_temp_error_K"],
                val_metrics["final_temperature"]["hotspot_location_error_cells"],
                val_metrics.get("physics_gate", {}).get("mean", ""),
                val_metrics.get("physics_gate", {}).get("std", ""),
                val_metrics.get("physics_gate", {}).get("min", ""),
                val_metrics.get("physics_gate", {}).get("max", ""),
                val_metrics.get("graph_correction_abs_mean", {}).get("mean", ""),
                val_metrics.get("graph_correction_abs_max", {}).get("mean", ""),
                val_metrics.get("graph_correction_rms", {}).get("mean", ""),
                val_metrics.get("graph_correction_spatial_std", {}).get("mean", ""),
                val_metrics.get("graph_to_cnn_ratio", ""),
                val_metrics.get("global_correction_abs_mean", {}).get("mean", ""),
                val_metrics.get("global_correction_abs_max", {}).get("mean", ""),
                val_metrics.get("global_correction_rms", {}).get("mean", ""),
                val_metrics.get("global_correction_spatial_std", {}).get("mean", ""),
                val_metrics.get("global_correction_low_frequency_energy", {}).get("mean", ""),
                val_metrics.get("chiplet_mean_temperature", {}).get("mae_K", ""),
                val_metrics.get("chiplet_peak_temperature", {}).get("mae_K", ""),
                val_metrics.get("inter_chiplet_delta_T", {}).get("mean", ""),
                val_metrics.get("pairwise_k", {}).get("abs_mean", ""),
                val_metrics.get("pairwise_contribution", {}).get("abs_mean", ""),
                val_metrics.get("pairwise_self", {}).get("abs_mean", ""),
                val_metrics.get("pairwise_node_correction", {}).get("abs_mean", ""),
                val_metrics.get("pairwise_basis_coeff", {}).get("abs_mean", ""),
                val_metrics.get("pairwise_basis_weighted_coeff", {}).get("abs_mean", ""),
                epoch_runtime_s,
                int(is_best),
            ]
        )


def write_training_history_json(csv_path: Path, json_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    json_path.write_text(
        json.dumps({"schema_version": 1, "epochs": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1.0e-4,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=1.0e-6,
        )
    raise ValueError(f"unsupported scheduler: {scheduler_name}")


def step_scheduler(
    scheduler_name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    val_final_mae: float,
) -> None:
    if scheduler is None:
        return
    if scheduler_name == "plateau":
        scheduler.step(val_final_mae)
    else:
        scheduler.step()


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_case_metrics(path: Path, case_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    columns = [
        "case_id",
        "residual_mae_K",
        "residual_rmse_K",
        "final_temperature_mae_K",
        "final_temperature_rmse_K",
        "cnn_only_final_temperature_mae_K",
        "cnn_only_final_temperature_rmse_K",
        "graph_delta_mae_K",
        "final_temperature_max_abs_error_K",
        "final_temperature_mean_signed_error_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
        "chiplet_mean_mae_K",
        "chiplet_peak_mae_K",
        "inter_chiplet_delta_mae_K",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for case_id, metrics in sorted(case_metrics.items()):
            final = metrics["final_temperature"]
            residual = metrics["residual"]
            cnn_only = metrics.get("cnn_only_final_temperature", {})
            chiplet_mean = metrics.get("chiplet_mean_temperature", {})
            chiplet_peak = metrics.get("chiplet_peak_temperature", {})
            chiplet_delta = metrics.get("inter_chiplet_delta_T", {})
            writer.writerow(
                {
                    "case_id": case_id,
                    "residual_mae_K": residual["mae_K"],
                    "residual_rmse_K": residual["rmse_K"],
                    "final_temperature_mae_K": final["mae_K"],
                    "final_temperature_rmse_K": final["rmse_K"],
                    "cnn_only_final_temperature_mae_K": cnn_only.get("mae_K", ""),
                    "cnn_only_final_temperature_rmse_K": cnn_only.get("rmse_K", ""),
                    "graph_delta_mae_K": (cnn_only.get("mae_K", 0.0) - final["mae_K"]) if cnn_only else "",
                    "final_temperature_max_abs_error_K": final["max_abs_error_K"],
                    "final_temperature_mean_signed_error_K": final["mean_signed_error_K"],
                    "hotspot_temp_error_K": final["hotspot_temp_error_K"],
                    "hotspot_location_error_cells": final["hotspot_location_error_cells"],
                    "chiplet_mean_mae_K": chiplet_mean.get("mae_K", ""),
                    "chiplet_peak_mae_K": chiplet_peak.get("mae_K", ""),
                    "inter_chiplet_delta_mae_K": chiplet_delta.get("mean", ""),
                }
            )


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
