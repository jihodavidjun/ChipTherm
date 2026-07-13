from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse_softplus value must be positive")
    if value > 20.0:
        return float(value)
    return float(math.log(math.expm1(value)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceResponseOperatorV1(nn.Module):
    """Raster source-response operator that predicts nonnegative unit response K/W."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        output_init_K_per_W: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if output_init_K_per_W <= 0.0:
            raise ValueError("output_init_K_per_W must be positive")
        self.architecture = "source_response_operator_v1"
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.output_init_K_per_W = float(output_init_K_per_W)

        channels = [base_channels * (2**index) for index in range(depth)]
        encoders: list[nn.Module] = []
        current = input_channels
        for out_channels in channels:
            encoders.append(ConvBlock(current, out_channels))
            current = out_channels
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        decoders: list[nn.Module] = []
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            decoders.append(ConvBlock(current + skip_channels, skip_channels))
            current = skip_channels
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(current, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, inverse_softplus(self.output_init_K_per_W))

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.head(h).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.forward_raw(x))

    def config(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "input_channels": self.input_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "output_init_K_per_W": self.output_init_K_per_W,
            "parameter_count": count_parameters(self),
        }


def build_source_response_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "source_response_operator_v1")).lower()
    if architecture != "source_response_operator_v1":
        raise ValueError(f"unsupported source-response architecture: {architecture}")
    return SourceResponseOperatorV1(
        input_channels=int(config["input_channels"]),
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 3)),
        output_init_K_per_W=float(config.get("output_init_K_per_W", 1.0e-4)),
    )


def predict_source_rise(unit_response_K_per_W: torch.Tensor, source_power_W: torch.Tensor) -> torch.Tensor:
    if source_power_W.ndim == 1:
        source_power_W = source_power_W[:, None, None]
    return unit_response_K_per_W * source_power_W.float()


def segment_sum_by_sample(values: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"values must have shape [N,H,W], got {tuple(values.shape)}")
    out = values.new_zeros((int(num_groups), int(values.shape[-2]), int(values.shape[-1])))
    out.index_add_(0, group_ids.long(), values)
    return out
