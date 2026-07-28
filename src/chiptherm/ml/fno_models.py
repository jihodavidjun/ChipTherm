from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


DIRECT_FNO_ARCHITECTURE = "fno2d_direct_conditioned"
RESIDUAL_FNO_ARCHITECTURE = "fno2d_residual_decomposed_conditioned"
FNO_ARCHITECTURES = {DIRECT_FNO_ARCHITECTURE, RESIDUAL_FNO_ARCHITECTURE}

FNO_CAPACITY_PROFILES: dict[str, dict[str, int]] = {
    "fno_small": {
        "width": 32,
        "layers": 4,
        "modes_x": 12,
        "modes_y": 12,
        "projection_channels": 64,
    },
    "fno_standard": {
        "width": 48,
        "layers": 4,
        "modes_x": 16,
        "modes_y": 16,
        "projection_channels": 96,
    },
}


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class SpectralConv2d(nn.Module):
    """Real-valued 2D spectral convolution over retained rFFT modes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes_x, modes_y) <= 0:
            raise ValueError("spectral convolution dimensions and modes must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        shape = (self.in_channels, self.out_channels, self.modes_x, self.modes_y, 2)
        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        self.weight_positive = nn.Parameter(scale * torch.randn(shape))
        self.weight_negative = nn.Parameter(scale * torch.randn(shape))

    @staticmethod
    def _multiply(
        values: torch.Tensor,
        weights_as_real: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.view_as_complex(weights_as_real.contiguous())
        return torch.einsum("bixy,ioxy->boxy", values, weights)

    def retained_modes(self, height: int, rfft_width: int) -> tuple[int, int]:
        # Positive and negative x bands must not overlap.
        modes_x = min(self.modes_x, max(height // 2, 1))
        modes_y = min(self.modes_y, rfft_width)
        return modes_x, modes_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SpectralConv2d expects [B,C,H,W], got {tuple(x.shape)}")
        batch, _, height, width = x.shape
        x = x.contiguous()
        spectrum = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros(
            batch,
            self.out_channels,
            height,
            spectrum.shape[-1],
            dtype=spectrum.dtype,
            device=x.device,
        )
        modes_x, modes_y = self.retained_modes(height, spectrum.shape[-1])
        output[:, :, :modes_x, :modes_y] = self._multiply(
            spectrum[:, :, :modes_x, :modes_y],
            self.weight_positive[:, :, :modes_x, :modes_y],
        )
        output[:, :, -modes_x:, :modes_y] = self._multiply(
            spectrum[:, :, -modes_x:, :modes_y],
            self.weight_negative[:, :, :modes_x, :modes_y],
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")

    def config(self) -> dict[str, int]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "modes_x": self.modes_x,
            "modes_y": self.modes_y,
        }


class FNOMetadataEncoder(nn.Module):
    def __init__(self, metadata_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if min(metadata_dim, hidden_dim, embedding_dim) <= 0:
            raise ValueError("conditioned FNO requires positive metadata dimensions")
        self.network = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.SiLU(),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        if metadata.ndim != 2:
            raise ValueError(f"metadata must have shape [B,M], got {tuple(metadata.shape)}")
        return self.network(metadata)


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
        metadata_embedding_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation != "gelu":
            raise ValueError(f"unsupported FNO activation: {activation}")
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.film = nn.Linear(metadata_embedding_dim, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, metadata_embedding: torch.Tensor) -> torch.Tensor:
        combined = self.spectral(x) + self.pointwise(x)
        gamma, beta = self.film(metadata_embedding).chunk(2, dim=1)
        combined = combined * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.activation(combined)


class FNO2dBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        metadata_dim: int,
        *,
        width: int = 32,
        layers: int = 4,
        modes_x: int = 12,
        modes_y: int = 12,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if min(input_channels, width, layers, modes_x, modes_y) <= 0:
            raise ValueError("FNO input, width, layers, and modes must be positive")
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.layers = int(layers)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.activation_name = str(activation)
        self.lift = nn.Conv2d(self.input_channels, self.width, kernel_size=1)
        self.metadata_encoder = FNOMetadataEncoder(
            self.metadata_dim,
            self.metadata_hidden_dim,
            self.metadata_embedding_dim,
        )
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    self.width,
                    self.modes_x,
                    self.modes_y,
                    self.metadata_embedding_dim,
                    activation=self.activation_name,
                )
                for _ in range(self.layers)
            ]
        )

    def forward_features(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if metadata is None:
            raise ValueError("conditioned FNO requires metadata tensor")
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"FNO expects [B,{self.input_channels},H,W], got {tuple(x.shape)}"
            )
        embedding = self.metadata_encoder(metadata)
        features = self.lift(x)
        for block in self.blocks:
            features = block(features, embedding)
        return features, embedding


class FNOProjectionHead(nn.Module):
    def __init__(self, width: int, hidden_channels: int, output_channels: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(width, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class DirectTemperatureFNO2d(nn.Module):
    """Conditioned FNO predicting train-standardized absolute temperature."""

    architecture = DIRECT_FNO_ARCHITECTURE
    prediction_mode = "direct_temperature_fno"
    physics_input_mode = "none"

    def __init__(
        self,
        input_channels: int = 33,
        output_channels: int = 1,
        metadata_dim: int = 15,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        width: int = 32,
        layers: int = 4,
        modes_x: int = 12,
        modes_y: int = 12,
        activation: str = "gelu",
        projection_channels: int = 64,
        capacity_profile: str = "fno_small",
        target_normalization_mode: str = "train_standard",
        target_mean_K: float = 0.0,
        target_std_K: float = 1.0,
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("direct-temperature FNO requires one output channel")
        if target_normalization_mode != "train_standard":
            raise ValueError("direct-temperature FNO requires train_standard target normalization")
        if target_std_K <= 0.0:
            raise ValueError("target_std_K must be positive")
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
        self.target_normalization_mode = str(target_normalization_mode)
        self.target_mean_K = float(target_mean_K)
        self.target_std_K = float(target_std_K)
        self.backbone = FNO2dBackbone(
            self.input_channels,
            self.metadata_dim,
            width=self.width,
            layers=self.layers,
            modes_x=self.modes_x,
            modes_y=self.modes_y,
            metadata_hidden_dim=self.metadata_hidden_dim,
            metadata_embedding_dim=self.metadata_embedding_dim,
            activation=self.activation,
        )
        self.projection = FNOProjectionHead(
            self.width,
            self.projection_channels,
            self.output_channels,
        )

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        features, _ = self.backbone.forward_features(x, metadata)
        return self.projection(features)

    def config(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "prediction_mode": self.prediction_mode,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "physics_input_mode": self.physics_input_mode,
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "metadata_conditioning_mode": "film",
            "fno_capacity_profile": self.capacity_profile,
            "fno_width": self.width,
            "fno_layers": self.layers,
            "fno_modes_x": self.modes_x,
            "fno_modes_y": self.modes_y,
            "fno_activation": self.activation,
            "fno_projection_channels": self.projection_channels,
            "target_normalization_mode": self.target_normalization_mode,
            "target_mean_K": self.target_mean_K,
            "target_std_K": self.target_std_K,
            "parameter_count": _count_parameters(self),
            "total_parameters": _count_parameters(self),
        }


class ResidualDecomposedFNO2d(nn.Module):
    """Conditioned FNO with resistance-mean and zero-mean residual heads."""

    architecture = RESIDUAL_FNO_ARCHITECTURE
    prediction_mode = "residual_decomposed_fno"
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
        layers: int = 4,
        modes_x: int = 12,
        modes_y: int = 12,
        activation: str = "gelu",
        projection_channels: int = 64,
        capacity_profile: str = "fno_small",
        delta_R_eff_mean_K_per_W: float = 0.0,
        delta_R_eff_std_K_per_W: float = 1.0,
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("residual FNO requires one centered-field output channel")
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
        self.delta_R_eff_mean_K_per_W = float(delta_R_eff_mean_K_per_W)
        self.delta_R_eff_std_K_per_W = float(delta_R_eff_std_K_per_W)
        self.backbone = FNO2dBackbone(
            self.input_channels,
            self.metadata_dim,
            width=self.width,
            layers=self.layers,
            modes_x=self.modes_x,
            modes_y=self.modes_y,
            metadata_hidden_dim=self.metadata_hidden_dim,
            metadata_embedding_dim=self.metadata_embedding_dim,
            activation=self.activation,
        )
        self.centered_projection = FNOProjectionHead(
            self.width,
            self.projection_channels,
            self.output_channels,
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
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if total_power_W is None:
            raise ValueError("residual FNO requires total_power_W")
        features, embedding = self.backbone.forward_features(x, metadata)
        raw_centered = self.centered_projection(features)
        centered = raw_centered - raw_centered.mean(dim=(-2, -1), keepdim=True)
        pooled = features.mean(dim=(-2, -1))
        mean_head_raw = self.mean_head(torch.cat([pooled, embedding], dim=1)).squeeze(1)
        delta_r = (
            mean_head_raw * mean_head_raw.new_tensor(self.delta_R_eff_std_K_per_W)
            + mean_head_raw.new_tensor(self.delta_R_eff_mean_K_per_W)
        )
        total_power = total_power_W.to(device=delta_r.device, dtype=delta_r.dtype).view(-1)
        if torch.any(total_power <= 0.0):
            raise ValueError("residual FNO requires strictly positive total_power_W")
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
            output["fno_feature_abs_mean"] = features.abs().mean(dim=(1, 2, 3))
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
        return {
            "architecture": self.architecture,
            "prediction_mode": self.prediction_mode,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "physics_input_mode": self.physics_input_mode,
            "mean_head_mode": self.mean_head_mode,
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "metadata_conditioning_mode": "film",
            "fno_capacity_profile": self.capacity_profile,
            "fno_width": self.width,
            "fno_layers": self.layers,
            "fno_modes_x": self.modes_x,
            "fno_modes_y": self.modes_y,
            "fno_activation": self.activation,
            "fno_projection_channels": self.projection_channels,
            "delta_R_eff_target_mean_K_per_W": self.delta_R_eff_mean_K_per_W,
            "delta_R_eff_target_std_K_per_W": self.delta_R_eff_std_K_per_W,
            "delta_R_eff_target_units": "K/W",
            "delta_R_eff_target_normalization": (
                "raw_head * train_std + train_mean; statistics fit on train split only"
            ),
            "reconstruction": (
                "source_superposition_base_K + total_power_W * "
                "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
            ),
            "parameter_count": _count_parameters(self),
            "total_parameters": _count_parameters(self),
        }

