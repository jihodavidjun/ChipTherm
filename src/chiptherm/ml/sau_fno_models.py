from __future__ import annotations

import time
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .fno_models import FNOProjectionHead, _count_parameters
from .ufno_models import (
    PUBLISHED_UFNO_BRANCH_INDICES,
    UFNO_ADAPTATION_PROFILE,
    UFNO_REFERENCE_COMMIT,
    ConditionedDirectUFNO2d,
    ConditionedResidualDecomposedUFNO2d,
    UFNO2dBackbone,
)


DIRECT_SAU_FNO_ARCHITECTURE = "sau_fno2d_direct_conditioned"
RESIDUAL_SAU_FNO_ARCHITECTURE = "sau_fno2d_residual_decomposed_conditioned"
SAU_FNO_ARCHITECTURES = {
    DIRECT_SAU_FNO_ARCHITECTURE,
    RESIDUAL_SAU_FNO_ARCHITECTURE,
}
SAU_FNO_ADAPTATION_PROFILE = "sau_fno_paper_adapted"
SAU_FNO_REFERENCE = (
    "Zhen Huang et al., Self-Attention to Operator Learning-based "
    "3D-IC Thermal Simulation, DAC 2025, arXiv:2510.15968v1"
)


class SAUAttention2d(nn.Module):
    """Paper-adapted, single-head spatial attention over a 2D operator field."""

    def __init__(self, channels: int, attention_dim: int | None = None) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("SAUAttention2d channels must be positive")
        resolved_dim = channels if attention_dim is None else int(attention_dim)
        if resolved_dim != channels:
            raise ValueError(
                "the controlled SAU-FNO profile requires Q, K, and value dimensions "
                "to equal the U-FNO feature width"
            )
        self.channels = int(channels)
        self.attention_dim = resolved_dim
        self.query = nn.Conv2d(self.channels, self.attention_dim, kernel_size=1)
        self.key = nn.Conv2d(self.channels, self.attention_dim, kernel_size=1)
        self.value = nn.Conv2d(self.channels, self.attention_dim, kernel_size=1)

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"SAUAttention2d expects [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        # Spatial cells are tokens; channels are the embedding dimension.
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2).transpose(1, 2)
        value = self.value(x).flatten(2).transpose(1, 2)
        return query, key, value

    @staticmethod
    def attention_weights(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if query.shape != key.shape or query.ndim != 3:
            raise ValueError("query and key must have identical [B,N,D] shapes")
        # Equation (9) normalizes over key positions k for every query i.
        return torch.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.project_qkv(x)
        # scale=1.0 preserves the paper's unscaled s_ij = Q_i^T K_j. PyTorch's
        # exact SDPA contract permits memory-efficient CUDA kernels without
        # changing the single-head softmax attention equation.
        attended = F.scaled_dot_product_attention(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        ).squeeze(1)
        batch, _, height, width = x.shape
        return attended.transpose(1, 2).reshape(
            batch, self.attention_dim, height, width
        )


class SAUFNO2dBackbone(UFNO2dBackbone):
    """Audited U-FNO backbone plus one attention block after its final block."""

    def __init__(self, *args: Any, attention_dim: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attention = SAUAttention2d(self.width, attention_dim)
        self.attention_placement = "after_final_ufourier_activation_after_padding_crop"

    def forward_features(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None,
        *,
        disable_unet: bool = False,
        disable_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, embedding = super().forward_features(
            x, metadata, disable_unet=disable_unet
        )
        if not disable_attention:
            features = self.attention(features)
        return features, embedding

    def forward_features_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None,
        *,
        synchronize: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        synchronize()
        start = time.perf_counter()
        features, embedding = super().forward_features(x, metadata)
        synchronize()
        backbone_time = time.perf_counter() - start
        start = time.perf_counter()
        features = self.attention(features)
        synchronize()
        attention_time = time.perf_counter() - start
        return features, embedding, {
            "ufno_backbone_s": backbone_time,
            "sau_attention_s": attention_time,
        }


class ConditionedDirectSAUFNO2d(ConditionedDirectUFNO2d):
    architecture = DIRECT_SAU_FNO_ARCHITECTURE
    prediction_mode = "direct_temperature_sau_fno"
    physics_input_mode = "none"

    def __init__(
        self,
        input_channels: int = 33,
        output_channels: int = 1,
        metadata_dim: int = 15,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        width: int = 32,
        layers: int = 6,
        modes_x: int = 12,
        modes_y: int = 12,
        activation: str = "gelu",
        projection_channels: int = 64,
        capacity_profile: str = "fno_small",
        adaptation_profile: str = SAU_FNO_ADAPTATION_PROFILE,
        unet_branch_indices: Sequence[int] = PUBLISHED_UFNO_BRANCH_INDICES,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        domain_padding: int = 8,
        padding_mode: str = "published_mixed",
        attention_dim: int | None = None,
        target_normalization_mode: str = "train_standard",
        target_mean_K: float = 0.0,
        target_std_K: float = 1.0,
    ) -> None:
        if adaptation_profile != SAU_FNO_ADAPTATION_PROFILE:
            raise ValueError(
                f"unsupported SAU-FNO adaptation profile: {adaptation_profile}"
            )
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            width=width,
            layers=layers,
            modes_x=modes_x,
            modes_y=modes_y,
            activation=activation,
            projection_channels=projection_channels,
            capacity_profile=capacity_profile,
            adaptation_profile=UFNO_ADAPTATION_PROFILE,
            unet_branch_indices=unet_branch_indices,
            unet_depth=unet_depth,
            unet_dropout=unet_dropout,
            domain_padding=domain_padding,
            padding_mode=padding_mode,
            target_normalization_mode=target_normalization_mode,
            target_mean_K=target_mean_K,
            target_std_K=target_std_K,
        )
        self.adaptation_profile = adaptation_profile
        self.backbone = SAUFNO2dBackbone(
            self.input_channels,
            self.metadata_dim,
            width=self.width,
            layers=self.layers,
            modes_x=self.modes_x,
            modes_y=self.modes_y,
            metadata_hidden_dim=self.metadata_hidden_dim,
            metadata_embedding_dim=self.metadata_embedding_dim,
            activation=self.activation,
            unet_branch_indices=unet_branch_indices,
            unet_depth=unet_depth,
            unet_dropout=unet_dropout,
            domain_padding=domain_padding,
            padding_mode=padding_mode,
            attention_dim=attention_dim,
        )
        self.projection = FNOProjectionHead(
            self.width, self.projection_channels, self.output_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        disable_unet: bool = False,
        disable_attention: bool = False,
    ) -> torch.Tensor:
        features, _ = self.backbone.forward_features(
            x,
            metadata,
            disable_unet=disable_unet,
            disable_attention=disable_attention,
        )
        return self.projection(features)

    def config(self) -> dict[str, Any]:
        return _sau_fno_config(
            self, reconstruction="train-standardized absolute temperature"
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        synchronize: Any,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        features, _, timings = self.backbone.forward_features_profile(
            x, metadata, synchronize=synchronize
        )
        synchronize()
        start = time.perf_counter()
        output = self.projection(features)
        synchronize()
        timings["projection_head_s"] = time.perf_counter() - start
        return output, timings


class ConditionedResidualDecomposedSAUFNO2d(ConditionedResidualDecomposedUFNO2d):
    architecture = RESIDUAL_SAU_FNO_ARCHITECTURE
    prediction_mode = "residual_decomposed_sau_fno"
    physics_input_mode = "source_superposition_v1"
    mean_head_mode = "residual_resistance"

    def __init__(
        self,
        input_channels: int = 34,
        output_channels: int = 1,
        metadata_dim: int = 15,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        width: int = 32,
        layers: int = 6,
        modes_x: int = 12,
        modes_y: int = 12,
        activation: str = "gelu",
        projection_channels: int = 64,
        capacity_profile: str = "fno_small",
        adaptation_profile: str = SAU_FNO_ADAPTATION_PROFILE,
        unet_branch_indices: Sequence[int] = PUBLISHED_UFNO_BRANCH_INDICES,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        domain_padding: int = 8,
        padding_mode: str = "published_mixed",
        attention_dim: int | None = None,
        delta_R_eff_mean_K_per_W: float = 0.0,
        delta_R_eff_std_K_per_W: float = 1.0,
    ) -> None:
        if adaptation_profile != SAU_FNO_ADAPTATION_PROFILE:
            raise ValueError(
                f"unsupported SAU-FNO adaptation profile: {adaptation_profile}"
            )
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            width=width,
            layers=layers,
            modes_x=modes_x,
            modes_y=modes_y,
            activation=activation,
            projection_channels=projection_channels,
            capacity_profile=capacity_profile,
            adaptation_profile=UFNO_ADAPTATION_PROFILE,
            unet_branch_indices=unet_branch_indices,
            unet_depth=unet_depth,
            unet_dropout=unet_dropout,
            domain_padding=domain_padding,
            padding_mode=padding_mode,
            delta_R_eff_mean_K_per_W=delta_R_eff_mean_K_per_W,
            delta_R_eff_std_K_per_W=delta_R_eff_std_K_per_W,
        )
        self.adaptation_profile = adaptation_profile
        self.backbone = SAUFNO2dBackbone(
            self.input_channels,
            self.metadata_dim,
            width=self.width,
            layers=self.layers,
            modes_x=self.modes_x,
            modes_y=self.modes_y,
            metadata_hidden_dim=self.metadata_hidden_dim,
            metadata_embedding_dim=self.metadata_embedding_dim,
            activation=self.activation,
            unet_branch_indices=unet_branch_indices,
            unet_depth=unet_depth,
            unet_dropout=unet_dropout,
            domain_padding=domain_padding,
            padding_mode=padding_mode,
            attention_dim=attention_dim,
        )
        self.centered_projection = FNOProjectionHead(
            self.width, self.projection_channels, self.output_channels
        )
        self.mean_head = nn.Sequential(
            nn.Linear(self.width + self.metadata_embedding_dim, self.metadata_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.metadata_hidden_dim, 1),
        )

    def forward_components(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        total_power_W: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        disable_unet: bool = False,
        disable_attention: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if total_power_W is None:
            raise ValueError("residual SAU-FNO requires total_power_W")
        features, embedding = self.backbone.forward_features(
            x,
            metadata,
            disable_unet=disable_unet,
            disable_attention=disable_attention,
        )
        raw_centered = self.centered_projection(features)
        centered = raw_centered - raw_centered.mean(dim=(-2, -1), keepdim=True)
        pooled = features.mean(dim=(-2, -1))
        mean_head_raw = self.mean_head(torch.cat([pooled, embedding], dim=1)).squeeze(1)
        delta_r = (
            mean_head_raw * mean_head_raw.new_tensor(self.delta_R_eff_std_K_per_W)
            + mean_head_raw.new_tensor(self.delta_R_eff_mean_K_per_W)
        )
        total_power = total_power_W.to(
            device=delta_r.device, dtype=delta_r.dtype
        ).view(-1)
        if total_power.shape != delta_r.shape:
            raise ValueError(
                "total_power_W must contain one value per sample, "
                f"got {tuple(total_power.shape)}"
            )
        if not torch.isfinite(total_power).all() or torch.any(total_power <= 0.0):
            raise ValueError(
                "residual SAU-FNO requires finite, strictly positive total_power_W"
            )
        mean_rise = total_power * delta_r
        output = {
            "mean_rise": mean_rise,
            "mean_head_raw": mean_head_raw,
            "delta_R_eff": delta_r,
            "centered_field": centered.squeeze(1),
            "coarse_centered_field": centered.squeeze(1),
            "detail_field": torch.zeros_like(centered.squeeze(1)),
        }
        if return_diagnostics:
            output["sau_fno_feature_abs_mean"] = features.abs().mean(dim=(1, 2, 3))
        return output

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        total_power_W: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return self.forward_components(
            x,
            metadata,
            total_power_W=total_power_W,
            return_diagnostics=return_diagnostics,
            **kwargs,
        )

    def config(self) -> dict[str, Any]:
        return _sau_fno_config(
            self,
            reconstruction=(
                "source_superposition_base_K + total_power_W * "
                "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
            ),
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        total_power_W: torch.Tensor | None = None,
        synchronize: Any,
        **_: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        if total_power_W is None:
            raise ValueError("residual SAU-FNO requires total_power_W")
        features, embedding, timings = self.backbone.forward_features_profile(
            x, metadata, synchronize=synchronize
        )
        synchronize()
        start = time.perf_counter()
        raw_centered = self.centered_projection(features)
        centered = raw_centered - raw_centered.mean(dim=(-2, -1), keepdim=True)
        synchronize()
        timings["centered_projection_s"] = time.perf_counter() - start

        start = time.perf_counter()
        pooled = features.mean(dim=(-2, -1))
        mean_head_raw = self.mean_head(torch.cat([pooled, embedding], dim=1)).squeeze(1)
        delta_r = (
            mean_head_raw * mean_head_raw.new_tensor(self.delta_R_eff_std_K_per_W)
            + mean_head_raw.new_tensor(self.delta_R_eff_mean_K_per_W)
        )
        total_power = total_power_W.to(
            device=delta_r.device, dtype=delta_r.dtype
        ).view(-1)
        if total_power.shape != delta_r.shape:
            raise ValueError(
                "total_power_W must contain one value per sample, "
                f"got {tuple(total_power.shape)}"
            )
        if not torch.isfinite(total_power).all() or torch.any(total_power <= 0.0):
            raise ValueError(
                "residual SAU-FNO requires finite, strictly positive total_power_W"
            )
        mean_rise = total_power * delta_r
        synchronize()
        timings["mean_head_s"] = time.perf_counter() - start
        return {
            "mean_rise": mean_rise,
            "mean_head_raw": mean_head_raw,
            "delta_R_eff": delta_r,
            "centered_field": centered.squeeze(1),
            "coarse_centered_field": centered.squeeze(1),
            "detail_field": torch.zeros_like(centered.squeeze(1)),
        }, timings


def attention_memory_estimate(
    *,
    height: int,
    width: int,
    batch_size: int,
    element_size_bytes: int,
) -> dict[str, int]:
    if min(height, width, batch_size, element_size_bytes) <= 0:
        raise ValueError("attention memory dimensions must be positive")
    tokens = int(height) * int(width)
    elements_per_sample = tokens * tokens
    return {
        "tokens": tokens,
        "attention_matrix_elements_per_sample": elements_per_sample,
        "attention_matrix_bytes_per_sample": elements_per_sample
        * int(element_size_bytes),
        "attention_matrix_bytes_per_batch": elements_per_sample
        * int(element_size_bytes)
        * int(batch_size),
    }


def _sau_fno_config(
    model: ConditionedDirectSAUFNO2d | ConditionedResidualDecomposedSAUFNO2d,
    *,
    reconstruction: str,
) -> dict[str, Any]:
    backbone = model.backbone
    payload: dict[str, Any] = {
        "architecture": model.architecture,
        "prediction_mode": model.prediction_mode,
        "input_channels": model.input_channels,
        "output_channels": model.output_channels,
        "physics_input_mode": model.physics_input_mode,
        "metadata_dim": model.metadata_dim,
        "metadata_hidden_dim": model.metadata_hidden_dim,
        "metadata_embedding_dim": model.metadata_embedding_dim,
        "metadata_conditioning_mode": "film",
        "ufno_reference_commit": UFNO_REFERENCE_COMMIT,
        "ufno_adaptation_profile": UFNO_ADAPTATION_PROFILE,
        "sau_fno_adaptation_profile": model.adaptation_profile,
        "sau_fno_reference": SAU_FNO_REFERENCE,
        "sau_fno_independent_implementation": True,
        "fno_width": model.width,
        "fno_layers": model.layers,
        "fno_modes_x": model.modes_x,
        "fno_modes_y": model.modes_y,
        "fno_activation": model.activation,
        "fno_projection_channels": model.projection_channels,
        "fno_capacity_profile": model.capacity_profile,
        "ufno_unet_branch_indices": list(backbone.unet_branch_indices),
        "ufno_unet_depth": backbone.unet_depth,
        "ufno_unet_channel_progression": [model.width] * backbone.unet_depth,
        "ufno_unet_dropout": backbone.unet_dropout,
        "ufno_domain_padding": backbone.domain_padding,
        "ufno_padding_mode": backbone.padding_mode,
        "ufno_branch_fusion": "add",
        "sau_attention_enabled": True,
        "sau_attention_placement": backbone.attention_placement,
        "sau_attention_type": "single_head_unscaled_spatial_self_attention",
        "sau_query_dimension": backbone.attention.attention_dim,
        "sau_key_dimension": backbone.attention.attention_dim,
        "sau_value_dimension": backbone.attention.attention_dim,
        "sau_number_of_heads": 1,
        "sau_softmax_axis": "key_token_axis_-1",
        "sau_fusion_rule": "softmax(Q @ K^T, dim=-1) @ W_h(V)",
        "sau_residual_connection": False,
        "sau_normalization": "none",
        "sau_transformer_mlp": False,
        "sau_attention_applied_after_padding_crop": True,
        "reconstruction": reconstruction,
        "parameter_count": _count_parameters(model),
        "total_parameters": _count_parameters(model),
    }
    if isinstance(model, ConditionedDirectSAUFNO2d):
        payload.update(
            {
                "target_normalization_mode": model.target_normalization_mode,
                "target_mean_K": model.target_mean_K,
                "target_std_K": model.target_std_K,
            }
        )
    else:
        payload.update(
            {
                "mean_head_mode": model.mean_head_mode,
                "delta_R_eff_target_mean_K_per_W": model.delta_R_eff_mean_K_per_W,
                "delta_R_eff_target_std_K_per_W": model.delta_R_eff_std_K_per_W,
                "delta_R_eff_target_units": "K/W",
                "residual_target": "HotSpot_K - source_superposition_base_K",
                "mean_correction_sign": 1,
                "centered_correction_sign": 1,
            }
        )
    return payload
