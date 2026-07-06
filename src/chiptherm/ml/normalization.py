from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


EPSILON = 1.0e-8


@dataclass(frozen=True)
class NormalizationStats:
    schema_version: int
    power_density_mean: float
    power_density_std: float
    physics_mean: float
    physics_std: float
    residual_mean: float
    residual_std: float
    num_samples: int
    num_grid_cells: int
    notes: str = "Computed from train split only. Masks and normalized coordinate channels are not normalized."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_normalization_stats(
    dataset: Dataset[Any],
    *,
    batch_size: int = 16,
    num_workers: int = 0,
) -> NormalizationStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    power_acc = RunningMoments()
    physics_acc = RunningMoments()
    residual_acc = RunningMoments()
    num_samples = 0

    for batch in loader:
        x = batch["x"].float()
        physics = batch["physics"].float()
        residual = batch["residual"].float()
        power_acc.update(x[:, 0])
        physics_acc.update(physics)
        residual_acc.update(residual)
        num_samples += int(x.shape[0])

    num_grid_cells = int(power_acc.count)
    return NormalizationStats(
        schema_version=1,
        power_density_mean=power_acc.mean,
        power_density_std=power_acc.std,
        physics_mean=physics_acc.mean,
        physics_std=physics_acc.std,
        residual_mean=residual_acc.mean,
        residual_std=residual_acc.std,
        num_samples=num_samples,
        num_grid_cells=num_grid_cells,
    )


def save_normalization_stats(stats: NormalizationStats, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizationStats(**data)


def build_model_input(x: torch.Tensor, physics: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    x_norm = x.float().clone()
    x_norm[:, 0] = normalize_tensor(x_norm[:, 0], stats.power_density_mean, stats.power_density_std)
    physics_norm = normalize_tensor(physics.float(), stats.physics_mean, stats.physics_std).unsqueeze(1)
    return torch.cat([x_norm, physics_norm], dim=1)


def normalize_residual(residual: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return normalize_tensor(residual.float(), stats.residual_mean, stats.residual_std)


def unnormalize_residual(residual_norm: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return residual_norm.float() * float(stats.residual_std) + float(stats.residual_mean)


def normalize_tensor(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (value - float(mean)) / max(float(std), EPSILON)


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        data = tensor.detach().double()
        self.count += int(data.numel())
        self.total += float(data.sum().item())
        self.total_sq += float((data * data).sum().item())

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.total / self.count)

    @property
    def std(self) -> float:
        if self.count == 0:
            return 1.0
        variance = max(self.total_sq / self.count - self.mean * self.mean, EPSILON)
        return float(variance**0.5)
