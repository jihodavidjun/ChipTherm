from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .fno_models import (
    FNOMetadataEncoder,
    FNOProjectionHead,
    SpectralConv2d,
    _count_parameters,
)


DIRECT_UFNO_ARCHITECTURE = "ufno2d_direct_conditioned"
RESIDUAL_UFNO_ARCHITECTURE = "ufno2d_residual_decomposed_conditioned"
UFNO_ARCHITECTURES = {DIRECT_UFNO_ARCHITECTURE, RESIDUAL_UFNO_ARCHITECTURE}
UFNO_REFERENCE_COMMIT = "8315fd7b5bd75282b7efe42ee6b8de86543d13cc"
UFNO_ADAPTATION_PROFILE = "ufno_published_adapted"
PUBLISHED_UFNO_BRANCH_INDICES = (3, 4, 5)


class MiniUNet2d(nn.Module):
    """Two-dimensional adaptation of the published three-level U-FNO U-Net path."""

    def __init__(
        self,
        channels: int,
        *,
        depth: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("MiniUNet2d channels must be positive")
        if depth != 3:
            raise ValueError("the published-adapted MiniUNet2d requires depth=3")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("MiniUNet2d kernel_size must be positive and odd")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("MiniUNet2d dropout must be in [0, 1)")
        self.channels = int(channels)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.down1 = self._conv(stride=2)
        self.down2 = self._conv(stride=2)
        self.refine2 = self._conv(stride=1)
        self.down3 = self._conv(stride=2)
        self.refine3 = self._conv(stride=1)
        self.up2 = self._deconv(self.channels)
        self.up1 = self._deconv(2 * self.channels)
        self.up0 = self._deconv(2 * self.channels)
        self.output = nn.Conv2d(
            2 * self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )

    def _conv(self, *, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=(self.kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.dropout),
        )

    def _deconv(self, input_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"MiniUNet2d expects [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError(
                "MiniUNet2d spatial dimensions must be divisible by 8, "
                f"got {tuple(x.shape[-2:])}"
            )
        down1 = self.down1(x)
        down2 = self.refine2(self.down2(down1))
        down3 = self.refine3(self.down3(down2))
        up2 = self.up2(down3)
        up1 = self.up1(torch.cat([down2, up2], dim=1))
        up0 = self.up0(torch.cat([down1, up1], dim=1))
        return self.output(torch.cat([x, up0], dim=1))


class UFNO2dBlock(nn.Module):
    """Fourier + pointwise + optional published mini U-Net branch."""

    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
        metadata_embedding_dim: int,
        *,
        use_unet: bool,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation != "gelu":
            raise ValueError(f"unsupported task-adapted U-FNO activation: {activation}")
        self.width = int(width)
        self.use_unet = bool(use_unet)
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.unet = (
            MiniUNet2d(width, depth=unet_depth, dropout=unet_dropout)
            if self.use_unet
            else None
        )
        self.film = nn.Linear(metadata_embedding_dim, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.activation = nn.GELU()

    def branch_sum(
        self,
        x: torch.Tensor,
        *,
        disable_unet: bool = False,
    ) -> torch.Tensor:
        combined = self.spectral(x) + self.pointwise(x)
        if self.unet is not None and not disable_unet:
            combined = combined + self.unet(x)
        return combined

    def forward(
        self,
        x: torch.Tensor,
        metadata_embedding: torch.Tensor,
        *,
        disable_unet: bool = False,
    ) -> torch.Tensor:
        combined = self.branch_sum(x, disable_unet=disable_unet)
        gamma, beta = self.film(metadata_embedding).chunk(2, dim=1)
        combined = combined * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.activation(combined)


class UFNO2dBackbone(nn.Module):
    """Published six-block U-FNO topology adapted from 3D transient to 2D steady state."""

    def __init__(
        self,
        input_channels: int,
        metadata_dim: int,
        *,
        width: int = 32,
        layers: int = 6,
        modes_x: int = 12,
        modes_y: int = 12,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        activation: str = "gelu",
        unet_branch_indices: Sequence[int] = PUBLISHED_UFNO_BRANCH_INDICES,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        domain_padding: int = 8,
        padding_mode: str = "published_mixed",
    ) -> None:
        super().__init__()
        if min(input_channels, width, layers, modes_x, modes_y) <= 0:
            raise ValueError("U-FNO input, width, layers, and modes must be positive")
        branch_indices = tuple(int(index) for index in unet_branch_indices)
        if layers != 6 or branch_indices != PUBLISHED_UFNO_BRANCH_INDICES:
            raise ValueError(
                "ufno_published_adapted requires six blocks with U-Net branches at (3, 4, 5)"
            )
        if domain_padding < 0:
            raise ValueError("U-FNO domain padding must be non-negative")
        if padding_mode not in {"published_mixed", "none"}:
            raise ValueError(f"unsupported U-FNO padding mode: {padding_mode}")
        self.input_channels = int(input_channels)
        self.metadata_dim = int(metadata_dim)
        self.width = int(width)
        self.layers = int(layers)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.activation_name = str(activation)
        self.unet_branch_indices = branch_indices
        self.unet_depth = int(unet_depth)
        self.unet_dropout = float(unet_dropout)
        self.domain_padding = int(domain_padding)
        self.padding_mode = str(padding_mode)
        self.lift = nn.Conv2d(self.input_channels, self.width, kernel_size=1)
        self.metadata_encoder = FNOMetadataEncoder(
            self.metadata_dim,
            self.metadata_hidden_dim,
            self.metadata_embedding_dim,
        )
        self.blocks = nn.ModuleList(
            [
                UFNO2dBlock(
                    self.width,
                    self.modes_x,
                    self.modes_y,
                    self.metadata_embedding_dim,
                    use_unet=index in self.unet_branch_indices,
                    unet_depth=self.unet_depth,
                    unet_dropout=self.unet_dropout,
                    activation=self.activation_name,
                )
                for index in range(self.layers)
            ]
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.domain_padding == 0 or self.padding_mode == "none":
            return x
        # The reference pads the positive second spatial axis by replication,
        # then the positive first spatial axis by zero, and crops both afterward.
        x = F.pad(x, (0, self.domain_padding, 0, 0), mode="replicate")
        return F.pad(x, (0, 0, 0, self.domain_padding), mode="constant", value=0.0)

    def _crop(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x[..., :height, :width]

    def forward_features(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None,
        *,
        disable_unet: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if metadata is None:
            raise ValueError("conditioned U-FNO requires metadata tensor")
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"U-FNO expects [B,{self.input_channels},H,W], got {tuple(x.shape)}"
            )
        if metadata.shape != (x.shape[0], self.metadata_dim):
            raise ValueError(
                f"U-FNO metadata must have shape {(x.shape[0], self.metadata_dim)}, "
                f"got {tuple(metadata.shape)}"
            )
        height, width = x.shape[-2:]
        padded = self._pad(x)
        if padded.shape[-2] % 8 or padded.shape[-1] % 8:
            raise ValueError(
                "padded U-FNO dimensions must be divisible by 8 for the published U-Net, "
                f"got {tuple(padded.shape[-2:])}"
            )
        embedding = self.metadata_encoder(metadata)
        features = self.lift(padded)
        for block in self.blocks:
            features = block(features, embedding, disable_unet=disable_unet)
        return self._crop(features, height, width), embedding


class ConditionedDirectUFNO2d(nn.Module):
    architecture = DIRECT_UFNO_ARCHITECTURE
    prediction_mode = "direct_temperature_ufno"
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
        adaptation_profile: str = UFNO_ADAPTATION_PROFILE,
        unet_branch_indices: Sequence[int] = PUBLISHED_UFNO_BRANCH_INDICES,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        domain_padding: int = 8,
        padding_mode: str = "published_mixed",
        target_normalization_mode: str = "train_standard",
        target_mean_K: float = 0.0,
        target_std_K: float = 1.0,
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("direct-temperature U-FNO requires one output channel")
        if adaptation_profile != UFNO_ADAPTATION_PROFILE:
            raise ValueError(f"unsupported U-FNO adaptation profile: {adaptation_profile}")
        if target_normalization_mode != "train_standard" or target_std_K <= 0.0:
            raise ValueError("direct U-FNO requires positive train-standard target normalization")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.width = int(width)
        self.layers = int(layers)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.activation = str(activation)
        self.projection_channels = int(projection_channels)
        self.capacity_profile = str(capacity_profile)
        self.adaptation_profile = str(adaptation_profile)
        self.target_normalization_mode = str(target_normalization_mode)
        self.target_mean_K = float(target_mean_K)
        self.target_std_K = float(target_std_K)
        self.backbone = UFNO2dBackbone(
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
    ) -> torch.Tensor:
        features, _ = self.backbone.forward_features(
            x, metadata, disable_unet=disable_unet
        )
        return self.projection(features)

    def config(self) -> dict[str, Any]:
        return _ufno_config(self, reconstruction="train-standardized absolute temperature")


class ConditionedResidualDecomposedUFNO2d(nn.Module):
    architecture = RESIDUAL_UFNO_ARCHITECTURE
    prediction_mode = "residual_decomposed_ufno"
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
        adaptation_profile: str = UFNO_ADAPTATION_PROFILE,
        unet_branch_indices: Sequence[int] = PUBLISHED_UFNO_BRANCH_INDICES,
        unet_depth: int = 3,
        unet_dropout: float = 0.0,
        domain_padding: int = 8,
        padding_mode: str = "published_mixed",
        delta_R_eff_mean_K_per_W: float = 0.0,
        delta_R_eff_std_K_per_W: float = 1.0,
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("residual U-FNO requires one centered-field output channel")
        if adaptation_profile != UFNO_ADAPTATION_PROFILE:
            raise ValueError(f"unsupported U-FNO adaptation profile: {adaptation_profile}")
        if delta_R_eff_std_K_per_W <= 0.0:
            raise ValueError("delta_R_eff_std_K_per_W must be positive")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.width = int(width)
        self.layers = int(layers)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.activation = str(activation)
        self.projection_channels = int(projection_channels)
        self.capacity_profile = str(capacity_profile)
        self.adaptation_profile = str(adaptation_profile)
        self.delta_R_eff_mean_K_per_W = float(delta_R_eff_mean_K_per_W)
        self.delta_R_eff_std_K_per_W = float(delta_R_eff_std_K_per_W)
        self.backbone = UFNO2dBackbone(
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
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if total_power_W is None:
            raise ValueError("residual U-FNO requires total_power_W")
        features, embedding = self.backbone.forward_features(
            x, metadata, disable_unet=disable_unet
        )
        raw_centered = self.centered_projection(features)
        centered = raw_centered - raw_centered.mean(dim=(-2, -1), keepdim=True)
        pooled = features.mean(dim=(-2, -1))
        mean_head_raw = self.mean_head(torch.cat([pooled, embedding], dim=1)).squeeze(1)
        delta_r = (
            mean_head_raw * mean_head_raw.new_tensor(self.delta_R_eff_std_K_per_W)
            + mean_head_raw.new_tensor(self.delta_R_eff_mean_K_per_W)
        )
        total_power = total_power_W.to(device=delta_r.device, dtype=delta_r.dtype).view(-1)
        if total_power.shape != delta_r.shape:
            raise ValueError(
                f"total_power_W must contain one value per sample, got {tuple(total_power.shape)}"
            )
        if not torch.isfinite(total_power).all() or torch.any(total_power <= 0.0):
            raise ValueError("residual U-FNO requires finite, strictly positive total_power_W")
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
            output["ufno_feature_abs_mean"] = features.abs().mean(dim=(1, 2, 3))
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
        return _ufno_config(
            self,
            reconstruction=(
                "source_superposition_base_K + total_power_W * "
                "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
            ),
        )


def _ufno_config(
    model: ConditionedDirectUFNO2d | ConditionedResidualDecomposedUFNO2d,
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
        "ufno_adaptation_profile": model.adaptation_profile,
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
        "reconstruction": reconstruction,
        "parameter_count": _count_parameters(model),
        "total_parameters": _count_parameters(model),
    }
    if isinstance(model, ConditionedDirectUFNO2d):
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
