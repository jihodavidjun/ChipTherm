from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
        }


def build_model(config: dict[str, int]) -> MiniUNet:
    return MiniUNet(
        input_channels=int(config.get("input_channels", 9)),
        output_channels=int(config.get("output_channels", 1)),
        base_channels=int(config.get("base_channels", 16)),
        depth=int(config.get("depth", 3)),
    )
