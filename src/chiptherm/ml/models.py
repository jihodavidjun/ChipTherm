from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
    raise ValueError(f"unsupported model architecture: {architecture}")


def build_miniunet(config: dict[str, int]) -> MiniUNet:
    return MiniUNet(
        input_channels=int(config.get("input_channels", 9)),
        output_channels=int(config.get("output_channels", 1)),
        base_channels=int(config.get("base_channels", 16)),
        depth=int(config.get("depth", 3)),
    )
