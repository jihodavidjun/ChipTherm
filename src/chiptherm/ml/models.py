from __future__ import annotations

import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from .graph_models import (
    ChipletMessagePassingGNN,
    PairwiseBasisThermalOperator,
    PairwiseThermalImpedanceOperator,
    rasterize_node_values,
)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MiniUNet(nn.Module):
    """Small UNet-style model for 64x64 residual thermal-map correction."""

    def __init__(
        self,
        input_channels: int = 9,
        output_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("MiniUNet depth must be at least 2")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.architecture = "miniunet"

        channels = [base_channels * (2**i) for i in range(depth)]
        encoders: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in channels:
            encoders.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        decoders: list[nn.Module] = []
        decoder_channels = list(reversed(channels[:-1]))
        current_ch = channels[-1]
        for skip_ch in decoder_channels:
            decoders.append(ConvBlock(current_ch + skip_ch, skip_ch))
            current_ch = skip_ch
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(current_ch, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for index, encoder in enumerate(self.encoders):
            h = encoder(h)
            if index < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)
        return self.head(h)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        h = x
        for index, encoder in enumerate(self.encoders):
            h = encoder(h)
            if index < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)
        return h, skips

    def config(self) -> dict[str, int]:
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
        }


class RefinementResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.relu(self.conv1(x))
        h = self.conv2(h)
        return self.relu(h + residual)


class FullResolutionRefinementCNN(nn.Module):
    def __init__(self, input_channels: int, refine_channels: int = 32, refine_blocks: int = 4) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("refinement input_channels must be positive")
        if refine_channels <= 0:
            raise ValueError("refine_channels must be positive")
        if refine_blocks < 0:
            raise ValueError("refine_blocks must be non-negative")
        self.input_channels = input_channels
        self.refine_channels = refine_channels
        self.refine_blocks = refine_blocks
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, refine_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[RefinementResidualBlock(refine_channels) for _ in range(refine_blocks)])
        self.output_projection = nn.Conv2d(refine_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        h = self.blocks(h)
        return self.output_projection(h)


class MiniUNetWithRefinement(nn.Module):
    """MiniUNet coarse residual predictor plus a full-resolution local correction branch."""

    def __init__(
        self,
        input_channels: int = 18,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("MiniUNetWithRefinement currently supports one output channel")
        if not refinement_channel_indices:
            raise ValueError("MiniUNetWithRefinement requires refinement_channel_indices")
        indices = tuple(int(index) for index in refinement_channel_indices)
        if min(indices) < 0 or max(indices) >= input_channels:
            raise ValueError(f"refinement_channel_indices {indices} out of range for {input_channels} input channels")

        self.architecture = "miniunet_refine"
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.refine_channels = refine_channels
        self.refine_blocks = refine_blocks
        self.refinement_channel_indices = indices
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.coarse_model = MiniUNet(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            depth=depth,
        )
        refinement_input_channels = len(indices) + output_channels
        self.refinement_model = FullResolutionRefinementCNN(
            input_channels=refinement_input_channels,
            refine_channels=refine_channels,
            refine_blocks=refine_blocks,
        )

    def forward_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse = self.coarse_model(x)
        selected = x[:, list(self.refinement_channel_indices), :, :]
        detail_input = torch.cat([selected, coarse], dim=1)
        detail = self.refinement_model(detail_input)
        final = coarse + detail
        return final, coarse, detail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        final, _, _ = self.forward_components(x)
        return final

    def config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "coarse_parameters": count_parameters(self.coarse_model),
            "refinement_parameters": count_parameters(self.refinement_model),
            "total_parameters": count_parameters(self),
        }


class MetadataEncoder(nn.Module):
    def __init__(self, metadata_dim: int, hidden_dim: int = 64, embedding_dim: int = 64) -> None:
        super().__init__()
        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive for metadata conditioning")
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.SiLU(),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata.float())


class FiLM(nn.Module):
    def __init__(self, embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embedding_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:channels].fill_(1.0)

    def forward(self, feature: torch.Tensor, embedding: torch.Tensor | None) -> torch.Tensor:
        if embedding is None:
            return feature
        gamma_beta = self.proj(embedding)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return feature * gamma[:, :, None, None] + beta[:, :, None, None]


class PhysicsReliabilityGate(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 32, init_alpha: float = 0.9) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("physics gate hidden_dim must be positive")
        if not 0.0 < float(init_alpha) < 1.0:
            raise ValueError("physics gate init_alpha must be in the interval (0, 1)")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.init_alpha = float(init_alpha)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, math.log(self.init_alpha / (1.0 - self.init_alpha)))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(embedding)).view(-1, 1, 1, 1)


class FiLMMiniUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        metadata_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("FiLMMiniUNet depth must be at least 2")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.metadata_embedding_dim = metadata_embedding_dim
        channels = [base_channels * (2**i) for i in range(depth)]
        encoders: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in channels:
            encoders.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_film = FiLM(metadata_embedding_dim, channels[-1])
        decoder_channels = list(reversed(channels[:-1]))
        current_ch = channels[-1]
        decoders: list[nn.Module] = []
        films: list[nn.Module] = []
        for skip_ch in decoder_channels:
            decoders.append(ConvBlock(current_ch + skip_ch, skip_ch))
            films.append(FiLM(metadata_embedding_dim, skip_ch))
            current_ch = skip_ch
        self.decoders = nn.ModuleList(decoders)
        self.decoder_films = nn.ModuleList(films)
        self.head = nn.Conv2d(current_ch, output_channels, kernel_size=1)

    def forward_features(self, x: torch.Tensor, metadata_embedding: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        h = x
        for index, encoder in enumerate(self.encoders):
            h = encoder(h)
            if index < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)
        h = self.bottleneck_film(h, metadata_embedding)
        return h, skips

    def decode(self, h: torch.Tensor, skips: list[torch.Tensor], metadata_embedding: torch.Tensor | None = None) -> torch.Tensor:
        for decoder, film, skip in zip(self.decoders, self.decoder_films, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)
            h = film(h, metadata_embedding)
        return self.head(h)

    def forward(self, x: torch.Tensor, metadata_embedding: torch.Tensor | None = None) -> torch.Tensor:
        h, skips = self.forward_features(x, metadata_embedding)
        return self.decode(h, skips, metadata_embedding)


class ConditionedFullResolutionRefinementCNN(FullResolutionRefinementCNN):
    def __init__(self, input_channels: int, refine_channels: int = 32, refine_blocks: int = 4, metadata_embedding_dim: int = 64) -> None:
        super().__init__(input_channels, refine_channels=refine_channels, refine_blocks=refine_blocks)
        self.film = FiLM(metadata_embedding_dim, refine_channels)

    def forward(self, x: torch.Tensor, metadata_embedding: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_projection(x)
        h = self.film(h, metadata_embedding)
        h = self.blocks(h)
        return self.output_projection(h)


class ConditionedMiniUNetWithRefinement(nn.Module):
    def __init__(
        self,
        input_channels: int = 34,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 1,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in refinement_channel_indices)
        if not indices:
            raise ValueError("ConditionedMiniUNetWithRefinement requires refinement_channel_indices")
        self.architecture = "miniunet_refine_conditioned"
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.refine_channels = refine_channels
        self.refine_blocks = refine_blocks
        self.refinement_channel_indices = indices
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = metadata_dim
        self.metadata_hidden_dim = metadata_hidden_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.metadata_encoder = MetadataEncoder(metadata_dim, metadata_hidden_dim, metadata_embedding_dim)
        self.coarse_model = FiLMMiniUNet(input_channels, output_channels, base_channels, depth, metadata_embedding_dim)
        self.refinement_model = ConditionedFullResolutionRefinementCNN(
            len(indices) + output_channels,
            refine_channels=refine_channels,
            refine_blocks=refine_blocks,
            metadata_embedding_dim=metadata_embedding_dim,
        )

    def forward_components(
        self, x: torch.Tensor, metadata: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if metadata is None:
            raise ValueError("conditioned model requires metadata tensor")
        embedding = self.metadata_encoder(metadata)
        coarse = self.coarse_model(x, embedding)
        selected = x[:, list(self.refinement_channel_indices), :, :]
        detail = self.refinement_model(torch.cat([selected, coarse], dim=1), embedding)
        final = coarse + detail
        return final, coarse, detail

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        final, _, _ = self.forward_components(x, metadata)
        return final

    def config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "coarse_parameters": count_parameters(self.coarse_model),
            "refinement_parameters": count_parameters(self.refinement_model),
            "metadata_parameters": count_parameters(self.metadata_encoder),
            "total_parameters": count_parameters(self),
        }


class DecomposedMiniUNetWithRefinement(nn.Module):
    def __init__(
        self,
        input_channels: int = 34,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        conditioned: bool = False,
        physics_input_mode: str = "v1",
        physics_gate_hidden_dim: int = 32,
        physics_gate_init: float = 0.9,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in refinement_channel_indices)
        if not indices:
            raise ValueError("DecomposedMiniUNetWithRefinement requires refinement_channel_indices")
        self.conditioned = bool(conditioned)
        self.architecture = "miniunet_refine_conditioned_decomposed" if self.conditioned else "miniunet_refine_decomposed"
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.refine_channels = refine_channels
        self.refine_blocks = refine_blocks
        self.refinement_channel_indices = indices
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = metadata_hidden_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.physics_input_mode = str(physics_input_mode)
        if self.physics_input_mode not in {
            "v1",
            "none",
            "gated_v1",
            "source_superposition_v1",
            "source_superposition_plus_physics_v1",
        }:
            raise ValueError(f"unsupported physics_input_mode: {self.physics_input_mode}")
        if self.physics_input_mode == "gated_v1" and not self.conditioned:
            raise ValueError("gated_v1 requires a metadata-conditioned decomposed model")
        self.physics_gate_hidden_dim = int(physics_gate_hidden_dim)
        self.physics_gate_init = float(physics_gate_init)
        if self.conditioned:
            self.metadata_encoder = MetadataEncoder(self.metadata_dim, metadata_hidden_dim, metadata_embedding_dim)
            self.physics_gate = (
                PhysicsReliabilityGate(metadata_embedding_dim, self.physics_gate_hidden_dim, self.physics_gate_init)
                if self.physics_input_mode == "gated_v1"
                else None
            )
            self.coarse_model = FiLMMiniUNet(input_channels, output_channels, base_channels, depth, metadata_embedding_dim)
            self.refinement_model = ConditionedFullResolutionRefinementCNN(
                len(indices) + output_channels,
                refine_channels=refine_channels,
                refine_blocks=refine_blocks,
                metadata_embedding_dim=metadata_embedding_dim,
            )
            mean_input_dim = metadata_embedding_dim + input_channels
        else:
            self.metadata_encoder = None
            self.physics_gate = None
            self.coarse_model = MiniUNet(input_channels, output_channels, base_channels, depth)
            self.refinement_model = FullResolutionRefinementCNN(
                len(indices) + output_channels,
                refine_channels=refine_channels,
                refine_blocks=refine_blocks,
            )
            mean_input_dim = input_channels
        self.mean_head = nn.Sequential(
            nn.Linear(mean_input_dim, metadata_hidden_dim),
            nn.SiLU(),
            nn.Linear(metadata_hidden_dim, 1),
        )

    def forward_components(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embedding = None
        alpha = None
        if self.conditioned:
            if metadata is None:
                raise ValueError("conditioned decomposed model requires metadata tensor")
            embedding = self.metadata_encoder(metadata)
            if self.physics_gate is not None:
                if x.shape[1] < 1:
                    raise ValueError("gated_v1 requires a physics input channel")
                alpha = self.physics_gate(embedding)
                x = torch.cat([x[:, :-1], x[:, -1:] * alpha], dim=1)
        if isinstance(self.coarse_model, FiLMMiniUNet):
            coarse = self.coarse_model(x, embedding)
        else:
            coarse = self.coarse_model(x)
        selected = x[:, list(self.refinement_channel_indices), :, :]
        detail_input = torch.cat([selected, coarse], dim=1)
        if isinstance(self.refinement_model, ConditionedFullResolutionRefinementCNN):
            detail = self.refinement_model(detail_input, embedding)
        else:
            detail = self.refinement_model(detail_input)
        centered = coarse + detail
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        pooled = x.mean(dim=(-2, -1))
        mean_input = torch.cat([embedding, pooled], dim=1) if embedding is not None else pooled
        mean_rise = self.mean_head(mean_input).squeeze(1)
        output = {
            "mean_rise": mean_rise,
            "centered_field": centered.squeeze(1),
            "coarse_centered_field": (coarse - coarse.mean(dim=(-2, -1), keepdim=True)).squeeze(1),
            "detail_field": detail.squeeze(1),
        }
        if alpha is not None:
            output["physics_gate_alpha"] = alpha.view(-1)
        return output

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward_components(x, metadata)

    def config(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "conditioned": self.conditioned,
            "physics_input_mode": self.physics_input_mode,
            "physics_gate_hidden_dim": self.physics_gate_hidden_dim,
            "physics_gate_init": self.physics_gate_init,
            "physics_gate_parameter_count": count_parameters(self.physics_gate) if self.physics_gate is not None else 0,
            "metadata_parameters": count_parameters(self.metadata_encoder) if self.metadata_encoder is not None else 0,
            "total_parameters": count_parameters(self),
        }


class GlobalCenteredFieldBranch(nn.Module):
    """Explicit low-resolution, zero-mean package-scale correction branch."""

    def __init__(
        self,
        input_channels: int,
        channel_indices: tuple[int, ...] | list[int],
        channel_names: tuple[str, ...] | list[str] = (),
        hidden_channels: int = 32,
        blocks: int = 3,
        pool_size: int = 8,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in channel_indices)
        if not indices:
            raise ValueError("GlobalCenteredFieldBranch requires at least one input channel")
        if min(indices) < 0 or max(indices) >= int(input_channels):
            raise ValueError(f"global branch indices {indices} out of range for {input_channels} input channels")
        if hidden_channels <= 0:
            raise ValueError("global branch hidden_channels must be positive")
        if blocks < 0:
            raise ValueError("global branch blocks must be non-negative")
        if pool_size <= 1:
            raise ValueError("global branch pool_size must be greater than one")
        self.input_channels = int(input_channels)
        self.channel_indices = indices
        self.channel_names = tuple(str(name) for name in channel_names)
        self.hidden_channels = int(hidden_channels)
        self.blocks = int(blocks)
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.input_projection = nn.Sequential(
            nn.Conv2d(len(indices), self.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[RefinementResidualBlock(self.hidden_channels) for _ in range(self.blocks)]
        )
        self.output_projection = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        selected = x[:, list(self.channel_indices), :, :]
        pooled = self.pool(selected)
        h = self.input_projection(pooled)
        h = self.residual_blocks(h)
        coarse = self.output_projection(h)
        upsampled = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        centered = upsampled - upsampled.mean(dim=(-2, -1), keepdim=True)
        return centered * float(scale)

    def config(self) -> dict[str, object]:
        return {
            "global_branch_input_channels": self.input_channels,
            "global_branch_channel_indices": list(self.channel_indices),
            "global_branch_channel_names": list(self.channel_names),
            "global_branch_hidden_channels": self.hidden_channels,
            "global_branch_blocks": self.blocks,
            "global_branch_pool_size": self.pool_size,
            "global_branch_zero_initialized": True,
            "global_branch_parameters": count_parameters(self),
        }


class FusionResidualDeltaBlock(nn.Module):
    """Small residual delta block with zero-initialized output for identity fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class GlobalFeatureFusionBlock(nn.Module):
    """Inject package-scale global features into a local decoder feature map."""

    def __init__(self, local_channels: int, global_channels: int) -> None:
        super().__init__()
        self.local_channels = int(local_channels)
        self.global_channels = int(global_channels)
        self.mix = nn.Sequential(
            nn.Conv2d(self.local_channels + self.global_channels, self.local_channels, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = FusionResidualDeltaBlock(self.local_channels)

    def forward(self, local: torch.Tensor, global_feature: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return local
        if global_feature.shape[-2:] != local.shape[-2:]:
            global_feature = F.interpolate(global_feature, size=local.shape[-2:], mode="bilinear", align_corners=False)
        h = self.mix(torch.cat([local, global_feature], dim=1))
        return local + self.delta(h)


class GlobalContextFeatureEncoder(nn.Module):
    """Aggressively pooled global context branch that emits decoder-scale features."""

    def __init__(
        self,
        input_channels: int,
        channel_indices: tuple[int, ...] | list[int],
        channel_names: tuple[str, ...] | list[str] = (),
        hidden_channels: int = 32,
        output_channels_by_scale: dict[str, int] | None = None,
        pool_size: int = 8,
        context_blocks: int = 3,
    ) -> None:
        super().__init__()
        indices = tuple(int(index) for index in channel_indices)
        if not indices:
            raise ValueError("GlobalContextFeatureEncoder requires at least one input channel")
        if min(indices) < 0 or max(indices) >= int(input_channels):
            raise ValueError(f"global feature indices {indices} out of range for {input_channels} input channels")
        if hidden_channels <= 0:
            raise ValueError("global feature hidden_channels must be positive")
        if pool_size != 8:
            raise ValueError("feature-fusion global encoder currently expects pool_size=8 for 64x64 inputs")
        if context_blocks < 0:
            raise ValueError("global feature context_blocks must be non-negative")
        output_channels_by_scale = output_channels_by_scale or {}
        for scale in ("16", "32", "64"):
            if scale not in output_channels_by_scale:
                raise ValueError(f"missing output channel count for global fusion scale {scale}")
        self.input_channels = int(input_channels)
        self.channel_indices = indices
        self.channel_names = tuple(str(name) for name in channel_names)
        self.hidden_channels = int(hidden_channels)
        self.pool_size = int(pool_size)
        self.context_blocks = int(context_blocks)
        self.output_channels_by_scale = {str(key): int(value) for key, value in output_channels_by_scale.items()}
        self.stem = nn.Sequential(
            nn.Conv2d(len(indices), hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.down32 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down16 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down8 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.context = nn.Sequential(*[RefinementResidualBlock(hidden_channels * 4) for _ in range(context_blocks)])
        self.proj16 = nn.Conv2d(hidden_channels * 4, self.output_channels_by_scale["16"], kernel_size=1)
        self.proj32 = nn.Conv2d(hidden_channels * 4, self.output_channels_by_scale["32"], kernel_size=1)
        self.proj64 = nn.Conv2d(hidden_channels * 4, self.output_channels_by_scale["64"], kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        selected = x[:, list(self.channel_indices), :, :]
        h64 = self.stem(selected)
        h32 = self.down32(h64)
        h16 = self.down16(h32)
        h8 = self.context(self.down8(h16))
        return {
            "16": self.proj16(F.interpolate(h8, size=(16, 16), mode="bilinear", align_corners=False)),
            "32": self.proj32(F.interpolate(h8, size=(32, 32), mode="bilinear", align_corners=False)),
            "64": self.proj64(F.interpolate(h8, size=(64, 64), mode="bilinear", align_corners=False)),
            "8": h8,
        }

    def config(self) -> dict[str, object]:
        return {
            "global_branch_input_channels": self.input_channels,
            "global_branch_channel_indices": list(self.channel_indices),
            "global_branch_channel_names": list(self.channel_names),
            "global_branch_hidden_channels": self.hidden_channels,
            "global_branch_pool_size": self.pool_size,
            "global_branch_context_blocks": self.context_blocks,
            "global_branch_output_channels_by_scale": dict(self.output_channels_by_scale),
            "global_encoder_parameters": count_parameters(self),
        }


class FeatureFusionFiLMMiniUNet(nn.Module):
    """MiniUNet decoder with package-scale global features fused at 16/32/64."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        metadata_embedding_dim: int = 64,
        global_branch_channel_indices: tuple[int, ...] | list[int] = (),
        global_branch_channel_names: tuple[str, ...] | list[str] = (),
        global_hidden_channels: int = 32,
        global_pool_size: int = 8,
        global_context_blocks: int = 3,
    ) -> None:
        super().__init__()
        if depth != 3:
            raise ValueError("feature-fusion MiniUNet currently supports depth=3 (64->32->16 decoder fusion)")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        channels = [base_channels * (2**i) for i in range(depth)]
        self.channels = tuple(channels)
        encoders: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in channels:
            encoders.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_film = FiLM(metadata_embedding_dim, channels[-1])
        self.global_encoder = GlobalContextFeatureEncoder(
            input_channels=input_channels,
            channel_indices=global_branch_channel_indices,
            channel_names=global_branch_channel_names,
            hidden_channels=global_hidden_channels,
            output_channels_by_scale={"16": channels[-1], "32": channels[1], "64": channels[0]},
            pool_size=global_pool_size,
            context_blocks=global_context_blocks,
        )
        self.fuse16 = GlobalFeatureFusionBlock(channels[-1], channels[-1])
        self.decoder32 = ConvBlock(channels[-1] + channels[1], channels[1])
        self.decoder32_film = FiLM(metadata_embedding_dim, channels[1])
        self.fuse32 = GlobalFeatureFusionBlock(channels[1], channels[1])
        self.decoder64 = ConvBlock(channels[1] + channels[0], channels[0])
        self.decoder64_film = FiLM(metadata_embedding_dim, channels[0])
        self.fuse64 = GlobalFeatureFusionBlock(channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], output_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        metadata_embedding: torch.Tensor | None = None,
        *,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        disabled = {str(scale) for scale in disabled_fusion_scales}
        skips: list[torch.Tensor] = []
        h = x
        for index, encoder in enumerate(self.encoders):
            h = encoder(h)
            if index < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)
        h = self.bottleneck_film(h, metadata_embedding)
        global_features = self.global_encoder(x)
        h = self.fuse16(h, global_features["16"], enabled="16" not in disabled and "all" not in disabled)
        skip32, skip64 = skips[-1], skips[0]
        h = F.interpolate(h, size=skip32.shape[-2:], mode="bilinear", align_corners=False)
        h = self.decoder32(torch.cat([h, skip32], dim=1))
        h = self.decoder32_film(h, metadata_embedding)
        h = self.fuse32(h, global_features["32"], enabled="32" not in disabled and "all" not in disabled)
        h = F.interpolate(h, size=skip64.shape[-2:], mode="bilinear", align_corners=False)
        h = self.decoder64(torch.cat([h, skip64], dim=1))
        h = self.decoder64_film(h, metadata_embedding)
        h = self.fuse64(h, global_features["64"], enabled="64" not in disabled and "all" not in disabled)
        output = self.head(h)
        if return_features:
            return output, global_features
        return output

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata_embedding: torch.Tensor | None = None,
        *,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        synchronize: object | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]], dict[str, float]]:
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        disabled = {str(scale) for scale in disabled_fusion_scales}
        skips: list[torch.Tensor] = []
        h = x
        start = tic()
        for index, encoder in enumerate(self.encoders):
            h = encoder(h)
            if index < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)
        h = self.bottleneck_film(h, metadata_embedding)
        toc("feature_fusion_local_encoder_bottleneck_s", start)

        start = tic()
        global_features = self.global_encoder(x)
        toc("feature_fusion_global_encoder_s", start)

        start = tic()
        h = self.fuse16(h, global_features["16"], enabled="16" not in disabled and "all" not in disabled)
        toc("feature_fusion_16_s", start)

        skip32, skip64 = skips[-1], skips[0]
        start = tic()
        h = F.interpolate(h, size=skip32.shape[-2:], mode="bilinear", align_corners=False)
        h = self.decoder32(torch.cat([h, skip32], dim=1))
        h = self.decoder32_film(h, metadata_embedding)
        toc("feature_fusion_decoder32_s", start)

        start = tic()
        h = self.fuse32(h, global_features["32"], enabled="32" not in disabled and "all" not in disabled)
        toc("feature_fusion_32_s", start)

        start = tic()
        h = F.interpolate(h, size=skip64.shape[-2:], mode="bilinear", align_corners=False)
        h = self.decoder64(torch.cat([h, skip64], dim=1))
        h = self.decoder64_film(h, metadata_embedding)
        toc("feature_fusion_decoder64_s", start)

        start = tic()
        h = self.fuse64(h, global_features["64"], enabled="64" not in disabled and "all" not in disabled)
        toc("feature_fusion_64_s", start)

        start = tic()
        output = self.head(h)
        toc("feature_fusion_centered_head_s", start)

        if return_features:
            return (output, global_features), timings
        return output, timings

    def config(self) -> dict[str, object]:
        return {
            "feature_fusion_channels": list(self.channels),
            "feature_fusion_parameters": count_parameters(self),
            **self.global_encoder.config(),
        }


def low_frequency_energy_fraction(field: torch.Tensor, pool_size: int = 8) -> torch.Tensor:
    """Cheap per-sample low-frequency energy proxy from pooled/re-expanded fields."""

    if field.ndim != 3:
        raise ValueError(f"expected [B,H,W] field, got shape {tuple(field.shape)}")
    pooled = F.adaptive_avg_pool2d(field.unsqueeze(1), (pool_size, pool_size))
    low = F.interpolate(pooled, size=field.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    total = torch.mean(field * field, dim=(-2, -1))
    low_energy = torch.mean(low * low, dim=(-2, -1))
    return low_energy / (total + 1.0e-12)


class DecomposedMiniUNetWithGlobalBranch(nn.Module):
    """Conditioned decomposed CNN plus an explicit low-frequency centered-field branch."""

    def __init__(
        self,
        input_channels: int = 34,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        physics_input_mode: str = "source_superposition_v1",
        physics_gate_hidden_dim: int = 32,
        physics_gate_init: float = 0.9,
        global_branch_channel_indices: tuple[int, ...] | list[int] = (),
        global_branch_channel_names: tuple[str, ...] | list[str] = (),
        global_hidden_channels: int = 32,
        global_blocks: int = 3,
        global_pool_size: int = 8,
    ) -> None:
        super().__init__()
        self.architecture = "miniunet_refine_conditioned_decomposed_global"
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.refine_channels = int(refine_channels)
        self.refine_blocks = int(refine_blocks)
        self.refinement_channel_indices = tuple(int(index) for index in refinement_channel_indices)
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.physics_input_mode = str(physics_input_mode)
        self.global_hidden_channels = int(global_hidden_channels)
        self.global_blocks = int(global_blocks)
        self.global_pool_size = int(global_pool_size)
        self.cnn_model = DecomposedMiniUNetWithRefinement(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            depth=depth,
            refine_channels=refine_channels,
            refine_blocks=refine_blocks,
            refinement_channel_indices=refinement_channel_indices,
            refinement_channel_names=refinement_channel_names,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conditioned=True,
            physics_input_mode=physics_input_mode,
            physics_gate_hidden_dim=physics_gate_hidden_dim,
            physics_gate_init=physics_gate_init,
        )
        self.global_branch = GlobalCenteredFieldBranch(
            input_channels=input_channels,
            channel_indices=global_branch_channel_indices,
            channel_names=global_branch_channel_names,
            hidden_channels=global_hidden_channels,
            blocks=global_blocks,
            pool_size=global_pool_size,
        )

    def forward_components(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
        global_correction_scale: float = 1.0,
        ambient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        local_outputs = self.cnn_model(x, metadata)
        local_centered = local_outputs["centered_field"]
        global_centered = self.global_branch(x, scale=global_correction_scale)
        combined_before_projection = local_centered + global_centered
        combined = combined_before_projection - combined_before_projection.mean(dim=(-2, -1), keepdim=True)
        outputs = dict(local_outputs)
        outputs["local_centered_field"] = local_centered
        outputs["global_centered_field"] = global_centered
        outputs["global_correction_field"] = global_centered
        outputs["centered_before_zero_mean"] = combined_before_projection
        outputs["centered_field"] = combined
        outputs["global_correction_abs_mean"] = global_centered.abs().mean(dim=(-2, -1))
        outputs["global_correction_abs_max"] = global_centered.abs().amax(dim=(-2, -1))
        outputs["global_correction_rms"] = torch.sqrt(torch.mean(global_centered * global_centered, dim=(-2, -1)))
        outputs["global_correction_spatial_std"] = global_centered.std(dim=(-2, -1))
        outputs["global_correction_low_frequency_energy"] = low_frequency_energy_fraction(
            global_centered,
            pool_size=self.global_pool_size,
        )
        outputs["global_scale"] = global_centered.new_full((global_centered.shape[0],), float(global_correction_scale))
        if return_diagnostics and ambient is not None:
            outputs["final_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + combined
            outputs["local_only_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + local_centered
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
        global_correction_scale: float = 1.0,
        ambient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_components(
            x,
            metadata,
            return_diagnostics=return_diagnostics,
            global_correction_scale=global_correction_scale,
            ambient=ambient,
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        synchronize: object | None = None,
        graph_correction_scale: float = 1.0,
        global_correction_scale: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        del graph, graph_correction_scale
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        start = tic()
        local_outputs = self.cnn_model(x, metadata)
        toc("local_cnn_branch_s", start)
        start = tic()
        global_centered = self.global_branch(x, scale=global_correction_scale)
        toc("global_branch_s", start)
        start = tic()
        local_centered = local_outputs["centered_field"]
        centered = local_centered + global_centered
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        outputs = dict(local_outputs)
        outputs["local_centered_field"] = local_centered
        outputs["global_centered_field"] = global_centered
        outputs["global_correction_field"] = global_centered
        outputs["centered_field"] = centered
        toc("global_fusion_s", start)
        return outputs, timings

    def config(self) -> dict[str, object]:
        global_config = self.global_branch.config()
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "conditioned": True,
            "physics_input_mode": self.physics_input_mode,
            "global_branch_enabled": True,
            "cnn_parameter_count": count_parameters(self.cnn_model),
            "global_branch_parameter_count": count_parameters(self.global_branch),
            "total_parameters": count_parameters(self),
            **global_config,
        }


class DecomposedMiniUNetWithFeatureFusion(nn.Module):
    """Conditioned decomposed CNN with package-scale features fused into the decoder."""

    def __init__(
        self,
        input_channels: int = 34,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        physics_input_mode: str = "source_superposition_v1",
        physics_gate_hidden_dim: int = 32,
        physics_gate_init: float = 0.9,
        global_branch_channel_indices: tuple[int, ...] | list[int] = (),
        global_branch_channel_names: tuple[str, ...] | list[str] = (),
        global_hidden_channels: int = 32,
        global_pool_size: int = 8,
        global_context_blocks: int = 3,
        mean_head_mode: str = "direct_k",
        delta_R_eff_mean_K_per_W: float = 0.0,
        delta_R_eff_std_K_per_W: float = 1.0,
    ) -> None:
        super().__init__()
        if mean_head_mode not in {"direct_k", "residual_resistance"}:
            raise ValueError(f"unsupported mean_head_mode: {mean_head_mode}")
        if delta_R_eff_std_K_per_W <= 0.0:
            raise ValueError("delta_R_eff_std_K_per_W must be positive")
        indices = tuple(int(index) for index in refinement_channel_indices)
        if not indices:
            raise ValueError("DecomposedMiniUNetWithFeatureFusion requires refinement_channel_indices")
        self.architecture = "miniunet_refine_conditioned_decomposed_feature_fusion"
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.refine_channels = int(refine_channels)
        self.refine_blocks = int(refine_blocks)
        self.refinement_channel_indices = indices
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.physics_input_mode = str(physics_input_mode)
        self.physics_gate_hidden_dim = int(physics_gate_hidden_dim)
        self.physics_gate_init = float(physics_gate_init)
        self.global_hidden_channels = int(global_hidden_channels)
        self.global_pool_size = int(global_pool_size)
        self.global_context_blocks = int(global_context_blocks)
        self.mean_head_mode = str(mean_head_mode)
        self.delta_R_eff_mean_K_per_W = float(delta_R_eff_mean_K_per_W)
        self.delta_R_eff_std_K_per_W = float(delta_R_eff_std_K_per_W)
        self.metadata_encoder = MetadataEncoder(metadata_dim, metadata_hidden_dim, metadata_embedding_dim)
        self.physics_gate = (
            PhysicsReliabilityGate(metadata_embedding_dim, self.physics_gate_hidden_dim, self.physics_gate_init)
            if self.physics_input_mode == "gated_v1"
            else None
        )
        self.coarse_model = FeatureFusionFiLMMiniUNet(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            depth=depth,
            metadata_embedding_dim=metadata_embedding_dim,
            global_branch_channel_indices=global_branch_channel_indices,
            global_branch_channel_names=global_branch_channel_names,
            global_hidden_channels=global_hidden_channels,
            global_pool_size=global_pool_size,
            global_context_blocks=global_context_blocks,
        )
        self.refinement_model = ConditionedFullResolutionRefinementCNN(
            len(indices) + output_channels,
            refine_channels=refine_channels,
            refine_blocks=refine_blocks,
            metadata_embedding_dim=metadata_embedding_dim,
        )
        self.mean_head = nn.Sequential(
            nn.Linear(metadata_embedding_dim + input_channels, metadata_hidden_dim),
            nn.SiLU(),
            nn.Linear(metadata_hidden_dim, 1),
        )

    def forward_components(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        ambient: torch.Tensor | None = None,
        total_power_W: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if metadata is None:
            raise ValueError("feature-fusion decomposed model requires metadata tensor")
        embedding = self.metadata_encoder(metadata)
        alpha = None
        if self.physics_gate is not None:
            if x.shape[1] < 1:
                raise ValueError("gated_v1 requires a physics input channel")
            alpha = self.physics_gate(embedding)
            x = torch.cat([x[:, :-1], x[:, -1:] * alpha], dim=1)
        coarse_result = self.coarse_model(
            x,
            embedding,
            disabled_fusion_scales=disabled_fusion_scales,
            return_features=return_diagnostics,
        )
        if return_diagnostics:
            coarse, global_features = coarse_result
        else:
            coarse = coarse_result
            global_features = {}
        selected = x[:, list(self.refinement_channel_indices), :, :]
        detail = self.refinement_model(torch.cat([selected, coarse], dim=1), embedding)
        centered = coarse + detail
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        pooled = x.mean(dim=(-2, -1))
        mean_head_raw = self.mean_head(torch.cat([embedding, pooled], dim=1)).squeeze(1)
        mean_rise, delta_r_eff = self._mean_outputs(mean_head_raw, total_power_W)
        output = {
            "mean_rise": mean_rise,
            "mean_head_raw": mean_head_raw,
            "centered_field": centered.squeeze(1),
            "coarse_centered_field": (coarse - coarse.mean(dim=(-2, -1), keepdim=True)).squeeze(1),
            "detail_field": detail.squeeze(1),
        }
        if delta_r_eff is not None:
            output["delta_R_eff"] = delta_r_eff
        if alpha is not None:
            output["physics_gate_alpha"] = alpha.view(-1)
        if return_diagnostics:
            disabled = {str(scale) for scale in disabled_fusion_scales}
            output["global_fusion_enabled_16"] = centered.new_full((centered.shape[0],), 0.0 if "all" in disabled or "16" in disabled else 1.0)
            output["global_fusion_enabled_32"] = centered.new_full((centered.shape[0],), 0.0 if "all" in disabled or "32" in disabled else 1.0)
            output["global_fusion_enabled_64"] = centered.new_full((centered.shape[0],), 0.0 if "all" in disabled or "64" in disabled else 1.0)
            for scale, feature in global_features.items():
                if scale in {"16", "32", "64"}:
                    output[f"global_feature_{scale}_abs_mean"] = feature.abs().mean(dim=(1, 2, 3))
            if ambient is not None and self.mean_head_mode == "direct_k":
                output["final_temperature"] = ambient[:, None, None] + mean_rise[:, None, None] + output["centered_field"]
        return output

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        ambient: torch.Tensor | None = None,
        total_power_W: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_components(
            x,
            metadata,
            return_diagnostics=return_diagnostics,
            disabled_fusion_scales=disabled_fusion_scales,
            ambient=ambient,
            total_power_W=total_power_W,
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        synchronize: object | None = None,
        graph_correction_scale: float = 1.0,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        total_power_W: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        del graph, graph_correction_scale
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        if metadata is None:
            raise ValueError("feature-fusion decomposed model requires metadata tensor")
        start = tic()
        embedding = self.metadata_encoder(metadata)
        toc("metadata_encoder_s", start)
        start_total = tic()
        coarse_result, coarse_timings = self.coarse_model.forward_profile(
            x,
            embedding,
            disabled_fusion_scales=disabled_fusion_scales,
            synchronize=synchronize,
            return_features=True,
        )
        timings.update(coarse_timings)
        coarse, global_features = coarse_result
        start = tic()
        selected = x[:, list(self.refinement_channel_indices), :, :]
        detail = self.refinement_model(torch.cat([selected, coarse], dim=1), embedding)
        centered = coarse + detail
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        toc("feature_fusion_refinement_s", start)
        start = tic()
        pooled = x.mean(dim=(-2, -1))
        mean_head_raw = self.mean_head(torch.cat([embedding, pooled], dim=1)).squeeze(1)
        mean_rise, delta_r_eff = self._mean_outputs(mean_head_raw, total_power_W)
        toc("feature_fusion_mean_head_s", start)
        outputs = {
            "mean_rise": mean_rise,
            "mean_head_raw": mean_head_raw,
            "centered_field": centered.squeeze(1),
            "coarse_centered_field": (coarse - coarse.mean(dim=(-2, -1), keepdim=True)).squeeze(1),
            "detail_field": detail.squeeze(1),
        }
        if delta_r_eff is not None:
            outputs["delta_R_eff"] = delta_r_eff
        disabled = {str(scale) for scale in disabled_fusion_scales}
        outputs["global_fusion_enabled_16"] = centered.new_full((centered.shape[0],), 0.0 if "all" in disabled or "16" in disabled else 1.0)
        outputs["global_fusion_enabled_32"] = centered.new_full((centered.shape[0],), 0.0 if "all" in disabled or "32" in disabled else 1.0)
        outputs["global_fusion_enabled_64"] = centered.new_full((centered.shape[0],), 0.0 if "all" in disabled or "64" in disabled else 1.0)
        for scale, feature in global_features.items():
            if scale in {"16", "32", "64"}:
                outputs[f"global_feature_{scale}_abs_mean"] = feature.abs().mean(dim=(1, 2, 3))
        toc("feature_fusion_cnn_branch_s", start_total)
        return outputs, timings

    def _mean_outputs(
        self,
        raw: torch.Tensor,
        total_power_W: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mean_head_mode == "direct_k":
            return raw, None
        if total_power_W is None:
            raise ValueError("residual_resistance mean head requires total_power_W")
        total_power = total_power_W.to(device=raw.device, dtype=raw.dtype).view(-1)
        if torch.any(total_power <= 0.0):
            raise ValueError("residual_resistance mean head requires strictly positive total_power_W")
        delta_r = raw * raw.new_tensor(self.delta_R_eff_std_K_per_W) + raw.new_tensor(self.delta_R_eff_mean_K_per_W)
        return total_power * delta_r, delta_r

    def config(self) -> dict[str, object]:
        coarse_config = self.coarse_model.config()
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "conditioned": True,
            "physics_input_mode": self.physics_input_mode,
            "mean_head_mode": self.mean_head_mode,
            "delta_R_eff_target_mean_K_per_W": self.delta_R_eff_mean_K_per_W,
            "delta_R_eff_target_std_K_per_W": self.delta_R_eff_std_K_per_W,
            "delta_R_eff_target_units": "K/W",
            "delta_R_eff_target_normalization": "raw_head * train_std + train_mean; statistics fit on train split only",
            "feature_fusion_enabled": True,
            "metadata_parameters": count_parameters(self.metadata_encoder),
            "coarse_parameters": count_parameters(self.coarse_model),
            "refinement_parameters": count_parameters(self.refinement_model),
            "feature_fusion_parameter_count": count_parameters(self.coarse_model.global_encoder)
            + count_parameters(self.coarse_model.fuse16)
            + count_parameters(self.coarse_model.fuse32)
            + count_parameters(self.coarse_model.fuse64),
            "total_parameters": count_parameters(self),
            **coarse_config,
        }


class DecomposedMiniUNetWithGraph(nn.Module):
    """Conditioned decomposed CNN with a parallel chiplet interaction GNN."""

    def __init__(
        self,
        input_channels: int = 34,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        physics_input_mode: str = "v1",
        physics_gate_hidden_dim: int = 32,
        physics_gate_init: float = 0.9,
        graph_node_feature_dim: int = 24,
        graph_edge_feature_dim: int = 15,
        graph_hidden_dim: int = 96,
        graph_edge_hidden_dim: int = 64,
        graph_layers: int = 4,
        graph_message_aggregation: str = "sum",
        graph_raster_channels: int = 16,
        graph_halo_decay_mm: float = 4.0,
        graph_use_edge_features: bool = True,
        graph_rasterizer_mode: str = "vectorized",
        freeze_cnn: bool = False,
        architecture: str = "miniunet_refine_conditioned_decomposed_graph",
        global_branch_channel_indices: tuple[int, ...] | list[int] = (),
        global_branch_channel_names: tuple[str, ...] | list[str] = (),
        global_hidden_channels: int = 32,
        global_blocks: int = 3,
        global_pool_size: int = 8,
        mean_head_mode: str = "direct_k",
        delta_R_eff_mean_K_per_W: float = 0.0,
        delta_R_eff_std_K_per_W: float = 1.0,
        graph_mean_correction_enabled: bool = True,
    ) -> None:
        super().__init__()
        if mean_head_mode not in {"direct_k", "residual_resistance"}:
            raise ValueError(f"unsupported mean_head_mode: {mean_head_mode}")
        self.architecture = str(architecture)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.refine_channels = refine_channels
        self.refine_blocks = refine_blocks
        self.refinement_channel_indices = tuple(int(index) for index in refinement_channel_indices)
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.physics_input_mode = str(physics_input_mode)
        self.graph_node_feature_dim = int(graph_node_feature_dim)
        self.graph_edge_feature_dim = int(graph_edge_feature_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_edge_hidden_dim = int(graph_edge_hidden_dim)
        self.graph_layers = int(graph_layers)
        self.graph_message_aggregation = str(graph_message_aggregation)
        self.graph_raster_channels = int(graph_raster_channels)
        self.graph_halo_decay_mm = float(graph_halo_decay_mm)
        self.graph_use_edge_features = bool(graph_use_edge_features)
        self.graph_rasterizer_mode = str(graph_rasterizer_mode)
        self.freeze_cnn = bool(freeze_cnn)
        self.mean_head_mode = str(mean_head_mode)
        self.delta_R_eff_mean_K_per_W = float(delta_R_eff_mean_K_per_W)
        self.delta_R_eff_std_K_per_W = float(delta_R_eff_std_K_per_W)
        self.graph_mean_correction_enabled = bool(graph_mean_correction_enabled)
        self.global_branch_enabled = self.architecture == "miniunet_refine_conditioned_decomposed_global_graph"
        self.feature_fusion_enabled = self.architecture == "miniunet_refine_conditioned_decomposed_feature_fusion_graph"
        if self.feature_fusion_enabled:
            self.cnn_model = DecomposedMiniUNetWithFeatureFusion(
                input_channels=input_channels,
                output_channels=output_channels,
                base_channels=base_channels,
                depth=depth,
                refine_channels=refine_channels,
                refine_blocks=refine_blocks,
                refinement_channel_indices=refinement_channel_indices,
                refinement_channel_names=refinement_channel_names,
                metadata_dim=metadata_dim,
                metadata_hidden_dim=metadata_hidden_dim,
                metadata_embedding_dim=metadata_embedding_dim,
                physics_input_mode=physics_input_mode,
                physics_gate_hidden_dim=physics_gate_hidden_dim,
                physics_gate_init=physics_gate_init,
                global_branch_channel_indices=global_branch_channel_indices,
                global_branch_channel_names=global_branch_channel_names,
                global_hidden_channels=global_hidden_channels,
                global_pool_size=global_pool_size,
                global_context_blocks=global_blocks,
                mean_head_mode=mean_head_mode,
                delta_R_eff_mean_K_per_W=delta_R_eff_mean_K_per_W,
                delta_R_eff_std_K_per_W=delta_R_eff_std_K_per_W,
            )
        elif self.global_branch_enabled:
            self.cnn_model = DecomposedMiniUNetWithGlobalBranch(
                input_channels=input_channels,
                output_channels=output_channels,
                base_channels=base_channels,
                depth=depth,
                refine_channels=refine_channels,
                refine_blocks=refine_blocks,
                refinement_channel_indices=refinement_channel_indices,
                refinement_channel_names=refinement_channel_names,
                metadata_dim=metadata_dim,
                metadata_hidden_dim=metadata_hidden_dim,
                metadata_embedding_dim=metadata_embedding_dim,
                physics_input_mode=physics_input_mode,
                physics_gate_hidden_dim=physics_gate_hidden_dim,
                physics_gate_init=physics_gate_init,
                global_branch_channel_indices=global_branch_channel_indices,
                global_branch_channel_names=global_branch_channel_names,
                global_hidden_channels=global_hidden_channels,
                global_blocks=global_blocks,
                global_pool_size=global_pool_size,
            )
        else:
            self.cnn_model = DecomposedMiniUNetWithRefinement(
                input_channels=input_channels,
                output_channels=output_channels,
                base_channels=base_channels,
                depth=depth,
                refine_channels=refine_channels,
                refine_blocks=refine_blocks,
                refinement_channel_indices=refinement_channel_indices,
                refinement_channel_names=refinement_channel_names,
                metadata_dim=metadata_dim,
                metadata_hidden_dim=metadata_hidden_dim,
                metadata_embedding_dim=metadata_embedding_dim,
                conditioned=True,
                physics_input_mode=physics_input_mode,
                physics_gate_hidden_dim=physics_gate_hidden_dim,
                physics_gate_init=physics_gate_init,
            )
        self.graph_model = ChipletMessagePassingGNN(
            node_feature_dim=graph_node_feature_dim,
            edge_feature_dim=graph_edge_feature_dim,
            hidden_dim=graph_hidden_dim,
            edge_hidden_dim=graph_edge_hidden_dim,
            layers=graph_layers,
            aggregation=graph_message_aggregation,
            raster_channels=graph_raster_channels,
            use_edge_features=graph_use_edge_features,
        )
        self.fusion_head = nn.Sequential(
            nn.Conv2d(graph_raster_channels + 1, refine_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(refine_channels, 1, kernel_size=3, padding=1),
        )
        final_conv = self.fusion_head[-1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)
        self.graph_mean_head = nn.Linear(graph_hidden_dim, 1)
        nn.init.zeros_(self.graph_mean_head.weight)
        nn.init.zeros_(self.graph_mean_head.bias)
        if self.freeze_cnn:
            for parameter in self.cnn_model.parameters():
                parameter.requires_grad_(False)

    def forward_components(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        return_diagnostics: bool = False,
        graph_correction_scale: float = 1.0,
        global_correction_scale: float = 1.0,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        ambient: torch.Tensor | None = None,
        total_power_W: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if graph is None:
            raise ValueError("graph architecture requires graph batch")
        if self.feature_fusion_enabled:
            cnn_outputs = self.cnn_model(
                x,
                metadata,
                return_diagnostics=return_diagnostics,
                disabled_fusion_scales=disabled_fusion_scales,
                total_power_W=total_power_W,
            )
        elif self.global_branch_enabled:
            cnn_outputs = self.cnn_model(x, metadata, global_correction_scale=global_correction_scale)
        else:
            cnn_outputs = self.cnn_model(x, metadata)
        graph_outputs = self.graph_model(graph, return_diagnostics=return_diagnostics)
        graph_maps = rasterize_node_values(
            graph_outputs["node_raster_values"],
            graph,
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            halo_decay_mm=self.graph_halo_decay_mm,
            mode=self.graph_rasterizer_mode,
        )
        cnn_centered = cnn_outputs["centered_field"]
        correction = self.fusion_head(torch.cat([cnn_centered.unsqueeze(1), graph_maps], dim=1)).squeeze(1)
        correction = correction - correction.mean(dim=(-2, -1), keepdim=True)
        scaled_correction = correction * float(graph_correction_scale)
        centered_before_projection = cnn_centered + scaled_correction
        centered = centered_before_projection
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        if self.graph_mean_correction_enabled:
            mean_delta = self.graph_mean_head(graph_outputs["graph_embedding"]).squeeze(1)
        else:
            mean_delta = graph_outputs["graph_embedding"].new_zeros(graph_outputs["graph_embedding"].shape[0])
        outputs = dict(cnn_outputs)
        outputs["cnn_mean_rise"] = cnn_outputs["mean_rise"]
        outputs["cnn_centered_field"] = cnn_centered
        outputs["graph_correction_field"] = correction
        outputs["scaled_graph_correction_field"] = scaled_correction
        outputs["centered_before_zero_mean"] = centered_before_projection
        outputs["graph_correction_abs_mean"] = correction.abs().mean(dim=(-2, -1))
        outputs["graph_correction_abs_max"] = correction.abs().amax(dim=(-2, -1))
        outputs["graph_mean_delta"] = mean_delta
        outputs["graph_mean_correction_enabled"] = mean_delta.new_full(mean_delta.shape, 1.0 if self.graph_mean_correction_enabled else 0.0)
        outputs["mean_rise"] = cnn_outputs["mean_rise"] + mean_delta
        outputs["centered_field"] = centered
        if return_diagnostics:
            outputs["final_centered_field"] = centered
            outputs["graph_raster_features"] = graph_maps
            outputs["node_embeddings"] = graph_outputs["node_embeddings"]
            outputs["global_graph_embedding"] = graph_outputs["graph_embedding"]
            if ambient is not None and self.mean_head_mode == "direct_k":
                outputs["final_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered
                outputs["cnn_only_temperature"] = ambient[:, None, None] + outputs["cnn_mean_rise"][:, None, None] + cnn_centered
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        return_diagnostics: bool = False,
        graph_correction_scale: float = 1.0,
        global_correction_scale: float = 1.0,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        ambient: torch.Tensor | None = None,
        total_power_W: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_components(
            x,
            metadata,
            graph,
            return_diagnostics=return_diagnostics,
            graph_correction_scale=graph_correction_scale,
            global_correction_scale=global_correction_scale,
            disabled_fusion_scales=disabled_fusion_scales,
            ambient=ambient,
            total_power_W=total_power_W,
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        graph: dict[str, torch.Tensor],
        *,
        synchronize: object | None = None,
        graph_correction_scale: float = 1.0,
        global_correction_scale: float = 1.0,
        disabled_fusion_scales: tuple[str, ...] | list[str] = (),
        total_power_W: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        start = tic()
        if self.feature_fusion_enabled:
            cnn_outputs, cnn_timings = self.cnn_model.forward_profile(
                x,
                metadata,
                synchronize=synchronize,
                disabled_fusion_scales=disabled_fusion_scales,
                total_power_W=total_power_W,
            )
            timings.update(cnn_timings)
        elif self.global_branch_enabled:
            cnn_outputs = self.cnn_model(x, metadata, global_correction_scale=global_correction_scale)
        else:
            cnn_outputs = self.cnn_model(x, metadata)
        toc("cnn_branch_s", start)
        graph_outputs, graph_timings = self.graph_model.forward_profile(graph, synchronize=synchronize)
        timings.update(graph_timings)
        start = tic()
        graph_maps = rasterize_node_values(
            graph_outputs["node_raster_values"],
            graph,
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            halo_decay_mm=self.graph_halo_decay_mm,
            mode=self.graph_rasterizer_mode,
        )
        toc("graph_rasterization_s", start)
        start = tic()
        cnn_centered = cnn_outputs["centered_field"]
        correction = self.fusion_head(torch.cat([cnn_centered.unsqueeze(1), graph_maps], dim=1)).squeeze(1)
        correction = correction - correction.mean(dim=(-2, -1), keepdim=True)
        centered = cnn_centered + correction * float(graph_correction_scale)
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        toc("fusion_head_s", start)
        start = tic()
        if self.graph_mean_correction_enabled:
            mean_delta = self.graph_mean_head(graph_outputs["graph_embedding"]).squeeze(1)
        else:
            mean_delta = graph_outputs["graph_embedding"].new_zeros(graph_outputs["graph_embedding"].shape[0])
        toc("graph_mean_head_s", start)
        start = tic()
        outputs = dict(cnn_outputs)
        outputs["cnn_mean_rise"] = cnn_outputs["mean_rise"]
        outputs["cnn_centered_field"] = cnn_centered
        outputs["graph_correction_field"] = correction
        outputs["scaled_graph_correction_field"] = correction * float(graph_correction_scale)
        outputs["centered_field"] = centered
        outputs["graph_mean_delta"] = mean_delta
        outputs["graph_mean_correction_enabled"] = mean_delta.new_full(mean_delta.shape, 1.0 if self.graph_mean_correction_enabled else 0.0)
        outputs["mean_rise"] = cnn_outputs["mean_rise"] + mean_delta
        outputs["graph_raster_features"] = graph_maps
        outputs["node_embeddings"] = graph_outputs["node_embeddings"]
        outputs["global_graph_embedding"] = graph_outputs["graph_embedding"]
        toc("graph_final_reconstruction_s", start)
        return outputs, timings

    def config(self) -> dict[str, object]:
        graph_parameters = count_parameters(self.graph_model)
        fusion_parameters = count_parameters(self.fusion_head) + count_parameters(self.graph_mean_head)
        payload = {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "conditioned": True,
            "physics_input_mode": self.physics_input_mode,
            "mean_head_mode": self.mean_head_mode,
            "delta_R_eff_target_mean_K_per_W": self.delta_R_eff_mean_K_per_W,
            "delta_R_eff_target_std_K_per_W": self.delta_R_eff_std_K_per_W,
            "delta_R_eff_target_units": "K/W",
            "graph_enabled": True,
            "graph_mean_correction_enabled": self.graph_mean_correction_enabled,
            "global_branch_enabled": self.global_branch_enabled,
            "feature_fusion_enabled": self.feature_fusion_enabled,
            "graph_node_feature_dim": self.graph_node_feature_dim,
            "graph_edge_feature_dim": self.graph_edge_feature_dim,
            "graph_hidden_dim": self.graph_hidden_dim,
            "graph_edge_hidden_dim": self.graph_edge_hidden_dim,
            "graph_layers": self.graph_layers,
            "graph_message_aggregation": self.graph_message_aggregation,
            "graph_raster_channels": self.graph_raster_channels,
            "graph_halo_decay_mm": self.graph_halo_decay_mm,
            "graph_use_edge_features": self.graph_use_edge_features,
            "graph_rasterizer_mode": self.graph_rasterizer_mode,
            "freeze_cnn": self.freeze_cnn,
            "cnn_parameter_count": count_parameters(self.cnn_model),
            "global_branch_parameter_count": count_parameters(getattr(self.cnn_model, "global_branch", None))
            if self.global_branch_enabled
            else 0,
            "graph_parameter_count": graph_parameters,
            "fusion_parameter_count": fusion_parameters,
            "total_parameters": count_parameters(self),
        }
        if self.global_branch_enabled and hasattr(self.cnn_model, "global_branch"):
            payload.update(self.cnn_model.global_branch.config())
        if self.feature_fusion_enabled and hasattr(self.cnn_model, "coarse_model"):
            payload.update(self.cnn_model.coarse_model.config())
        return payload


class DecomposedMiniUNetWithPairwiseOperator(nn.Module):
    """Frozen/conditioned decomposed CNN with explicit pairwise chiplet correction."""

    def __init__(
        self,
        input_channels: int = 14,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        physics_input_mode: str = "v1",
        physics_gate_hidden_dim: int = 32,
        physics_gate_init: float = 0.9,
        graph_node_feature_dim: int = 24,
        graph_edge_feature_dim: int = 15,
        pairwise_hidden_dim: int = 96,
        pairwise_layers: int = 3,
        graph_halo_decay_mm: float = 4.0,
        graph_rasterizer_mode: str = "vectorized",
        freeze_cnn: bool = True,
        source_power_feature_index: int = 6,
    ) -> None:
        super().__init__()
        self.architecture = "miniunet_refine_conditioned_decomposed_pairwise"
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.refine_channels = int(refine_channels)
        self.refine_blocks = int(refine_blocks)
        self.refinement_channel_indices = tuple(int(index) for index in refinement_channel_indices)
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.physics_input_mode = str(physics_input_mode)
        self.graph_node_feature_dim = int(graph_node_feature_dim)
        self.graph_edge_feature_dim = int(graph_edge_feature_dim)
        self.pairwise_hidden_dim = int(pairwise_hidden_dim)
        self.pairwise_layers = int(pairwise_layers)
        self.graph_halo_decay_mm = float(graph_halo_decay_mm)
        self.graph_rasterizer_mode = str(graph_rasterizer_mode)
        self.freeze_cnn = bool(freeze_cnn)
        self.source_power_feature_index = int(source_power_feature_index)
        self.cnn_model = DecomposedMiniUNetWithRefinement(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            depth=depth,
            refine_channels=refine_channels,
            refine_blocks=refine_blocks,
            refinement_channel_indices=refinement_channel_indices,
            refinement_channel_names=refinement_channel_names,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conditioned=True,
            physics_input_mode=physics_input_mode,
            physics_gate_hidden_dim=physics_gate_hidden_dim,
            physics_gate_init=physics_gate_init,
        )
        self.pairwise_operator = PairwiseThermalImpedanceOperator(
            node_feature_dim=graph_node_feature_dim,
            edge_feature_dim=graph_edge_feature_dim,
            metadata_dim=metadata_dim,
            hidden_dim=pairwise_hidden_dim,
            layers=pairwise_layers,
            source_power_feature_index=source_power_feature_index,
        )
        if self.freeze_cnn:
            for parameter in self.cnn_model.parameters():
                parameter.requires_grad_(False)

    def forward_components(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        return_diagnostics: bool = False,
        graph_correction_scale: float = 1.0,
        ambient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if graph is None:
            raise ValueError("pairwise architecture requires graph batch")
        cnn_outputs = self.cnn_model(x, metadata)
        pairwise_outputs = self.pairwise_operator(graph, metadata, return_diagnostics=return_diagnostics)
        node_values = pairwise_outputs["node_corrections"].unsqueeze(1)
        operator_map = rasterize_node_values(
            node_values,
            graph,
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            halo_decay_mm=self.graph_halo_decay_mm,
            mode=self.graph_rasterizer_mode,
        ).squeeze(1)
        operator_map = operator_map - operator_map.mean(dim=(-2, -1), keepdim=True)
        scaled_correction = operator_map * float(graph_correction_scale)
        cnn_centered = cnn_outputs["centered_field"]
        centered_before_projection = cnn_centered + scaled_correction
        centered = centered_before_projection - centered_before_projection.mean(dim=(-2, -1), keepdim=True)
        outputs = dict(cnn_outputs)
        outputs["cnn_centered_field"] = cnn_centered
        outputs["graph_correction_field"] = operator_map
        outputs["scaled_graph_correction_field"] = scaled_correction
        outputs["centered_before_zero_mean"] = centered_before_projection
        outputs["centered_field"] = centered
        outputs["pairwise_k_values"] = pairwise_outputs["k_values"]
        outputs["pairwise_contributions"] = pairwise_outputs["pairwise_contributions"]
        outputs["pairwise_node_sums"] = pairwise_outputs["pairwise_node_sums"]
        outputs["pairwise_self_corrections"] = pairwise_outputs["self_corrections"]
        outputs["pairwise_node_corrections"] = pairwise_outputs["node_corrections"]
        outputs["graph_correction_abs_mean"] = operator_map.abs().mean(dim=(-2, -1))
        outputs["graph_correction_abs_max"] = operator_map.abs().amax(dim=(-2, -1))
        if return_diagnostics:
            outputs["final_centered_field"] = centered
            outputs["node_corrections"] = pairwise_outputs["node_corrections"]
            outputs["source_target_transfer_K"] = pairwise_outputs["k_values"]
            if ambient is not None:
                outputs["final_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered
                outputs["cnn_only_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + cnn_centered
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        return_diagnostics: bool = False,
        graph_correction_scale: float = 1.0,
        ambient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_components(
            x,
            metadata,
            graph,
            return_diagnostics=return_diagnostics,
            graph_correction_scale=graph_correction_scale,
            ambient=ambient,
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        graph: dict[str, torch.Tensor],
        *,
        synchronize: object | None = None,
        graph_correction_scale: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        start = tic()
        cnn_outputs = self.cnn_model(x, metadata)
        toc("cnn_branch_s", start)
        start = tic()
        pairwise_outputs = self.pairwise_operator(graph, metadata, return_diagnostics=True)
        toc("pairwise_operator_s", start)
        start = tic()
        operator_map = rasterize_node_values(
            pairwise_outputs["node_corrections"].unsqueeze(1),
            graph,
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            halo_decay_mm=self.graph_halo_decay_mm,
            mode=self.graph_rasterizer_mode,
        ).squeeze(1)
        toc("graph_rasterization_s", start)
        start = tic()
        operator_map = operator_map - operator_map.mean(dim=(-2, -1), keepdim=True)
        cnn_centered = cnn_outputs["centered_field"]
        centered = cnn_centered + operator_map * float(graph_correction_scale)
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        outputs = dict(cnn_outputs)
        outputs["cnn_centered_field"] = cnn_centered
        outputs["graph_correction_field"] = operator_map
        outputs["scaled_graph_correction_field"] = operator_map * float(graph_correction_scale)
        outputs["centered_field"] = centered
        outputs["pairwise_k_values"] = pairwise_outputs["k_values"]
        outputs["pairwise_contributions"] = pairwise_outputs["pairwise_contributions"]
        outputs["pairwise_node_sums"] = pairwise_outputs["pairwise_node_sums"]
        outputs["pairwise_self_corrections"] = pairwise_outputs["self_corrections"]
        outputs["pairwise_node_corrections"] = pairwise_outputs["node_corrections"]
        toc("fusion_head_s", start)
        return outputs, timings

    def config(self) -> dict[str, object]:
        pairwise_parameters = count_parameters(self.pairwise_operator)
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "conditioned": True,
            "physics_input_mode": self.physics_input_mode,
            "graph_enabled": True,
            "pairwise_enabled": True,
            "graph_node_feature_dim": self.graph_node_feature_dim,
            "graph_edge_feature_dim": self.graph_edge_feature_dim,
            "pairwise_hidden_dim": self.pairwise_hidden_dim,
            "pairwise_layers": self.pairwise_layers,
            "source_power_feature_index": self.source_power_feature_index,
            "graph_halo_decay_mm": self.graph_halo_decay_mm,
            "graph_rasterizer_mode": self.graph_rasterizer_mode,
            "freeze_cnn": self.freeze_cnn,
            "cnn_parameter_count": count_parameters(self.cnn_model),
            "pairwise_parameter_count": pairwise_parameters,
            "total_parameters": count_parameters(self),
        }


class DecomposedMiniUNetWithPairwiseBasisOperator(nn.Module):
    """Frozen decomposed CNN plus explicit low-rank directional pairwise field."""

    def __init__(
        self,
        input_channels: int = 14,
        output_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        refine_channels: int = 32,
        refine_blocks: int = 4,
        refinement_channel_indices: tuple[int, ...] | list[int] = (),
        refinement_channel_names: tuple[str, ...] | list[str] = (),
        metadata_dim: int = 0,
        metadata_hidden_dim: int = 64,
        metadata_embedding_dim: int = 64,
        physics_input_mode: str = "v1",
        physics_gate_hidden_dim: int = 32,
        physics_gate_init: float = 0.9,
        graph_node_feature_dim: int = 24,
        graph_edge_feature_dim: int = 15,
        pairwise_basis_rank: int = 8,
        pairwise_basis_hidden_dim: int = 96,
        pairwise_basis_layers: int = 3,
        pairwise_basis_halo_decay_mm: float = 4.0,
        pairwise_basis_edge_chunk_size: int = 512,
        graph_rasterizer_mode: str = "vectorized",
        freeze_cnn: bool = True,
        source_power_feature_index: int = 6,
    ) -> None:
        super().__init__()
        self.architecture = "miniunet_refine_conditioned_decomposed_pairwise_basis"
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.refine_channels = int(refine_channels)
        self.refine_blocks = int(refine_blocks)
        self.refinement_channel_indices = tuple(int(index) for index in refinement_channel_indices)
        self.refinement_channel_names = tuple(str(name) for name in refinement_channel_names)
        self.metadata_dim = int(metadata_dim)
        self.metadata_hidden_dim = int(metadata_hidden_dim)
        self.metadata_embedding_dim = int(metadata_embedding_dim)
        self.physics_input_mode = str(physics_input_mode)
        self.graph_node_feature_dim = int(graph_node_feature_dim)
        self.graph_edge_feature_dim = int(graph_edge_feature_dim)
        self.pairwise_basis_rank = int(pairwise_basis_rank)
        self.pairwise_basis_hidden_dim = int(pairwise_basis_hidden_dim)
        self.pairwise_basis_layers = int(pairwise_basis_layers)
        self.pairwise_basis_halo_decay_mm = float(pairwise_basis_halo_decay_mm)
        self.pairwise_basis_edge_chunk_size = int(pairwise_basis_edge_chunk_size)
        self.graph_rasterizer_mode = str(graph_rasterizer_mode)
        self.freeze_cnn = bool(freeze_cnn)
        self.source_power_feature_index = int(source_power_feature_index)
        self.cnn_model = DecomposedMiniUNetWithRefinement(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            depth=depth,
            refine_channels=refine_channels,
            refine_blocks=refine_blocks,
            refinement_channel_indices=refinement_channel_indices,
            refinement_channel_names=refinement_channel_names,
            metadata_dim=metadata_dim,
            metadata_hidden_dim=metadata_hidden_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conditioned=True,
            physics_input_mode=physics_input_mode,
            physics_gate_hidden_dim=physics_gate_hidden_dim,
            physics_gate_init=physics_gate_init,
        )
        self.basis_operator = PairwiseBasisThermalOperator(
            node_feature_dim=graph_node_feature_dim,
            edge_feature_dim=graph_edge_feature_dim,
            metadata_dim=metadata_dim,
            basis_rank=pairwise_basis_rank,
            hidden_dim=pairwise_basis_hidden_dim,
            layers=pairwise_basis_layers,
            source_power_feature_index=source_power_feature_index,
            edge_chunk_size=pairwise_basis_edge_chunk_size,
        )
        if self.freeze_cnn:
            for parameter in self.cnn_model.parameters():
                parameter.requires_grad_(False)

    @property
    def basis_names(self) -> tuple[str, ...]:
        return self.basis_operator.basis_names

    def forward_components(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        return_diagnostics: bool = False,
        graph_correction_scale: float = 1.0,
        ambient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if graph is None:
            raise ValueError("pairwise-basis architecture requires graph batch")
        cnn_outputs = self.cnn_model(x, metadata)
        basis_outputs = self.basis_operator(
            graph,
            metadata,
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            halo_decay_mm=self.pairwise_basis_halo_decay_mm,
            return_diagnostics=return_diagnostics,
        )
        operator_map = basis_outputs["field"]
        operator_map = operator_map - operator_map.mean(dim=(-2, -1), keepdim=True)
        scaled_correction = operator_map * float(graph_correction_scale)
        cnn_centered = cnn_outputs["centered_field"]
        centered_before_projection = cnn_centered + scaled_correction
        centered = centered_before_projection - centered_before_projection.mean(dim=(-2, -1), keepdim=True)
        outputs = dict(cnn_outputs)
        outputs["cnn_centered_field"] = cnn_centered
        outputs["graph_correction_field"] = operator_map
        outputs["scaled_graph_correction_field"] = scaled_correction
        outputs["centered_before_zero_mean"] = centered_before_projection
        outputs["centered_field"] = centered
        outputs["pairwise_basis_coefficients"] = basis_outputs["coefficients"]
        outputs["pairwise_basis_weighted_coefficients"] = basis_outputs["weighted_coefficients"]
        outputs["pairwise_basis_names"] = self.basis_names
        outputs["graph_correction_abs_mean"] = operator_map.abs().mean(dim=(-2, -1))
        outputs["graph_correction_abs_max"] = operator_map.abs().amax(dim=(-2, -1))
        if return_diagnostics:
            outputs["final_centered_field"] = centered
            outputs["source_target_basis_coefficients"] = basis_outputs["coefficients"]
            if ambient is not None:
                outputs["final_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered
                outputs["cnn_only_temperature"] = ambient[:, None, None] + outputs["mean_rise"][:, None, None] + cnn_centered
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
        *,
        return_diagnostics: bool = False,
        graph_correction_scale: float = 1.0,
        ambient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.forward_components(
            x,
            metadata,
            graph,
            return_diagnostics=return_diagnostics,
            graph_correction_scale=graph_correction_scale,
            ambient=ambient,
        )

    def forward_profile(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        graph: dict[str, torch.Tensor],
        *,
        synchronize: object | None = None,
        graph_correction_scale: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        start = tic()
        cnn_outputs = self.cnn_model(x, metadata)
        toc("cnn_branch_s", start)
        start = tic()
        basis_outputs = self.basis_operator(
            graph,
            metadata,
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            halo_decay_mm=self.pairwise_basis_halo_decay_mm,
            return_diagnostics=True,
        )
        toc("pairwise_basis_operator_s", start)
        start = tic()
        operator_map = basis_outputs["field"] - basis_outputs["field"].mean(dim=(-2, -1), keepdim=True)
        cnn_centered = cnn_outputs["centered_field"]
        centered = cnn_centered + operator_map * float(graph_correction_scale)
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        outputs = dict(cnn_outputs)
        outputs["cnn_centered_field"] = cnn_centered
        outputs["graph_correction_field"] = operator_map
        outputs["scaled_graph_correction_field"] = operator_map * float(graph_correction_scale)
        outputs["centered_field"] = centered
        outputs["pairwise_basis_coefficients"] = basis_outputs["coefficients"]
        outputs["pairwise_basis_weighted_coefficients"] = basis_outputs["weighted_coefficients"]
        outputs["pairwise_basis_names"] = self.basis_names
        toc("fusion_head_s", start)
        return outputs, timings

    def config(self) -> dict[str, object]:
        basis_parameters = count_parameters(self.basis_operator)
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "refine_channels": self.refine_channels,
            "refine_blocks": self.refine_blocks,
            "refinement_channel_indices": list(self.refinement_channel_indices),
            "refinement_channel_names": list(self.refinement_channel_names),
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_embedding_dim": self.metadata_embedding_dim,
            "conditioned": True,
            "physics_input_mode": self.physics_input_mode,
            "graph_enabled": True,
            "pairwise_basis_enabled": True,
            "graph_node_feature_dim": self.graph_node_feature_dim,
            "graph_edge_feature_dim": self.graph_edge_feature_dim,
            "pairwise_basis_rank": self.pairwise_basis_rank,
            "pairwise_basis_hidden_dim": self.pairwise_basis_hidden_dim,
            "pairwise_basis_layers": self.pairwise_basis_layers,
            "pairwise_basis_halo_decay_mm": self.pairwise_basis_halo_decay_mm,
            "pairwise_basis_edge_chunk_size": self.pairwise_basis_edge_chunk_size,
            "pairwise_basis_names": list(self.basis_names),
            "source_power_feature_index": self.source_power_feature_index,
            "graph_rasterizer_mode": self.graph_rasterizer_mode,
            "freeze_cnn": self.freeze_cnn,
            "cnn_parameter_count": count_parameters(self.cnn_model),
            "pairwise_basis_parameter_count": basis_parameters,
            "total_parameters": count_parameters(self),
        }


def build_model(config: dict[str, object]) -> nn.Module:
    architecture = str(config.get("architecture") or config.get("name") or "miniunet").lower()
    if architecture == "miniunet":
        return MiniUNet(
            input_channels=int(config.get("input_channels", 9)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 16)),
            depth=int(config.get("depth", 3)),
        )
    if architecture == "miniunet_refine":
        return MiniUNetWithRefinement(
            input_channels=int(config.get("input_channels", 18)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
        )
    if architecture == "miniunet_refine_conditioned":
        return ConditionedMiniUNetWithRefinement(
            input_channels=int(config.get("input_channels", 34)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 1)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
        )
    if architecture in {"miniunet_refine_decomposed", "miniunet_refine_conditioned_decomposed"}:
        return DecomposedMiniUNetWithRefinement(
            input_channels=int(config.get("input_channels", 34)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 0)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
            conditioned=architecture == "miniunet_refine_conditioned_decomposed",
            physics_input_mode=str(config.get("physics_input_mode", "v1")),
            physics_gate_hidden_dim=int(config.get("physics_gate_hidden_dim", 32)),
            physics_gate_init=float(config.get("physics_gate_init", 0.9)),
        )
    if architecture == "miniunet_refine_conditioned_decomposed_global":
        return DecomposedMiniUNetWithGlobalBranch(
            input_channels=int(config.get("input_channels", 34)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 0)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
            physics_input_mode=str(config.get("physics_input_mode", "source_superposition_v1")),
            physics_gate_hidden_dim=int(config.get("physics_gate_hidden_dim", 32)),
            physics_gate_init=float(config.get("physics_gate_init", 0.9)),
            global_branch_channel_indices=tuple(int(index) for index in config.get("global_branch_channel_indices", ())),
            global_branch_channel_names=tuple(str(name) for name in config.get("global_branch_channel_names", ())),
            global_hidden_channels=int(config.get("global_hidden_channels", config.get("global_branch_hidden_channels", 32))),
            global_blocks=int(config.get("global_blocks", config.get("global_branch_blocks", 3))),
            global_pool_size=int(config.get("global_pool_size", config.get("global_branch_pool_size", 8))),
        )
    if architecture in {
        "miniunet_refine_conditioned_decomposed_feature_fusion",
        "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
    }:
        return DecomposedMiniUNetWithFeatureFusion(
            input_channels=int(config.get("input_channels", 34)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 0)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
            physics_input_mode=str(config.get("physics_input_mode", "source_superposition_v1")),
            physics_gate_hidden_dim=int(config.get("physics_gate_hidden_dim", 32)),
            physics_gate_init=float(config.get("physics_gate_init", 0.9)),
            global_branch_channel_indices=tuple(int(index) for index in config.get("global_branch_channel_indices", ())),
            global_branch_channel_names=tuple(str(name) for name in config.get("global_branch_channel_names", ())),
            global_hidden_channels=int(config.get("global_hidden_channels", config.get("global_branch_hidden_channels", 32))),
            global_pool_size=int(config.get("global_pool_size", config.get("global_branch_pool_size", 8))),
            global_context_blocks=int(config.get("global_blocks", config.get("global_branch_context_blocks", 3))),
            mean_head_mode=str(
                config.get(
                    "mean_head_mode",
                    "residual_resistance"
                    if architecture == "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean"
                    else "direct_k",
                )
            ),
            delta_R_eff_mean_K_per_W=float(
                config.get("delta_R_eff_target_mean_K_per_W", config.get("delta_R_eff_mean_K_per_W", 0.0))
            ),
            delta_R_eff_std_K_per_W=float(
                config.get("delta_R_eff_target_std_K_per_W", config.get("delta_R_eff_std_K_per_W", 1.0))
            ),
        )
    if architecture in {
        "miniunet_refine_conditioned_decomposed_graph",
        "miniunet_refine_conditioned_decomposed_global_graph",
        "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
    }:
        return DecomposedMiniUNetWithGraph(
            input_channels=int(config.get("input_channels", 34)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 0)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
            physics_input_mode=str(config.get("physics_input_mode", "v1")),
            physics_gate_hidden_dim=int(config.get("physics_gate_hidden_dim", 32)),
            physics_gate_init=float(config.get("physics_gate_init", 0.9)),
            graph_node_feature_dim=int(config.get("graph_node_feature_dim", 24)),
            graph_edge_feature_dim=int(config.get("graph_edge_feature_dim", 15)),
            graph_hidden_dim=int(config.get("graph_hidden_dim", 96)),
            graph_edge_hidden_dim=int(config.get("graph_edge_hidden_dim", 64)),
            graph_layers=int(config.get("graph_layers", 4)),
            graph_message_aggregation=str(config.get("graph_message_aggregation", "sum")),
            graph_raster_channels=int(config.get("graph_raster_channels", 16)),
            graph_halo_decay_mm=float(config.get("graph_halo_decay_mm", 4.0)),
            graph_use_edge_features=bool(config.get("graph_use_edge_features", True)),
            graph_rasterizer_mode=str(config.get("graph_rasterizer_mode", "vectorized")),
            freeze_cnn=bool(config.get("freeze_cnn", False)),
            architecture=architecture,
            global_branch_channel_indices=tuple(int(index) for index in config.get("global_branch_channel_indices", ())),
            global_branch_channel_names=tuple(str(name) for name in config.get("global_branch_channel_names", ())),
            global_hidden_channels=int(config.get("global_hidden_channels", config.get("global_branch_hidden_channels", 32))),
            global_blocks=int(config.get("global_blocks", config.get("global_branch_blocks", 3))),
            global_pool_size=int(config.get("global_pool_size", config.get("global_branch_pool_size", 8))),
            mean_head_mode=str(config.get("mean_head_mode", "direct_k")),
            delta_R_eff_mean_K_per_W=float(
                config.get("delta_R_eff_target_mean_K_per_W", config.get("delta_R_eff_mean_K_per_W", 0.0))
            ),
            delta_R_eff_std_K_per_W=float(
                config.get("delta_R_eff_target_std_K_per_W", config.get("delta_R_eff_std_K_per_W", 1.0))
            ),
            graph_mean_correction_enabled=bool(config.get("graph_mean_correction_enabled", True)),
        )
    if architecture == "miniunet_refine_conditioned_decomposed_pairwise":
        return DecomposedMiniUNetWithPairwiseOperator(
            input_channels=int(config.get("input_channels", 14)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 0)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
            physics_input_mode=str(config.get("physics_input_mode", "v1")),
            physics_gate_hidden_dim=int(config.get("physics_gate_hidden_dim", 32)),
            physics_gate_init=float(config.get("physics_gate_init", 0.9)),
            graph_node_feature_dim=int(config.get("graph_node_feature_dim", 24)),
            graph_edge_feature_dim=int(config.get("graph_edge_feature_dim", 15)),
            pairwise_hidden_dim=int(config.get("pairwise_hidden_dim", 96)),
            pairwise_layers=int(config.get("pairwise_layers", 3)),
            graph_halo_decay_mm=float(config.get("graph_halo_decay_mm", 4.0)),
            graph_rasterizer_mode=str(config.get("graph_rasterizer_mode", "vectorized")),
            freeze_cnn=bool(config.get("freeze_cnn", True)),
            source_power_feature_index=int(config.get("source_power_feature_index", 6)),
        )
    if architecture == "miniunet_refine_conditioned_decomposed_pairwise_basis":
        return DecomposedMiniUNetWithPairwiseBasisOperator(
            input_channels=int(config.get("input_channels", 14)),
            output_channels=int(config.get("output_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            refine_channels=int(config.get("refine_channels", 32)),
            refine_blocks=int(config.get("refine_blocks", 4)),
            refinement_channel_indices=tuple(int(index) for index in config.get("refinement_channel_indices", ())),
            refinement_channel_names=tuple(str(name) for name in config.get("refinement_channel_names", ())),
            metadata_dim=int(config.get("metadata_dim", 0)),
            metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
            metadata_embedding_dim=int(config.get("metadata_embedding_dim", 64)),
            physics_input_mode=str(config.get("physics_input_mode", "v1")),
            physics_gate_hidden_dim=int(config.get("physics_gate_hidden_dim", 32)),
            physics_gate_init=float(config.get("physics_gate_init", 0.9)),
            graph_node_feature_dim=int(config.get("graph_node_feature_dim", 24)),
            graph_edge_feature_dim=int(config.get("graph_edge_feature_dim", 15)),
            pairwise_basis_rank=int(config.get("pairwise_basis_rank", 8)),
            pairwise_basis_hidden_dim=int(config.get("pairwise_basis_hidden_dim", 96)),
            pairwise_basis_layers=int(config.get("pairwise_basis_layers", 3)),
            pairwise_basis_halo_decay_mm=float(config.get("pairwise_basis_halo_decay_mm", config.get("graph_halo_decay_mm", 4.0))),
            pairwise_basis_edge_chunk_size=int(config.get("pairwise_basis_edge_chunk_size", 512)),
            graph_rasterizer_mode=str(config.get("graph_rasterizer_mode", "vectorized")),
            freeze_cnn=bool(config.get("freeze_cnn", True)),
            source_power_feature_index=int(config.get("source_power_feature_index", 6)),
        )
    raise ValueError(f"unsupported model architecture: {architecture}")


def build_miniunet(config: dict[str, int]) -> MiniUNet:
    return MiniUNet(
        input_channels=int(config.get("input_channels", 9)),
        output_channels=int(config.get("output_channels", 1)),
        base_channels=int(config.get("base_channels", 16)),
        depth=int(config.get("depth", 3)),
    )
