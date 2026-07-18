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
    input_channels: int = 8
    context_channel_indices: tuple[int, ...] = ()
    context_channel_names: tuple[str, ...] = ()
    context_channel_means: tuple[float, ...] = ()
    context_channel_stds: tuple[float, ...] = ()
    metadata_feature_names: tuple[str, ...] = ()
    metadata_means: tuple[float, ...] = ()
    metadata_stds: tuple[float, ...] = ()
    auxiliary_physics_v1_mean: float | None = None
    auxiliary_physics_v1_std: float | None = None
    notes: str = "Computed from train split only. Masks and normalized coordinate channels are not normalized."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_channel_indices"] = list(self.context_channel_indices)
        data["context_channel_names"] = list(self.context_channel_names)
        data["context_channel_means"] = list(self.context_channel_means)
        data["context_channel_stds"] = list(self.context_channel_stds)
        data["metadata_feature_names"] = list(self.metadata_feature_names)
        data["metadata_means"] = list(self.metadata_means)
        data["metadata_stds"] = list(self.metadata_stds)
        return data


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
    auxiliary_physics_v1_acc = RunningMoments()
    saw_auxiliary_physics_v1 = False
    metadata_accs: list[RunningMoments] | None = None
    metadata_feature_names: tuple[str, ...] = ()
    context_accs: list[RunningMoments] | None = None
    input_channels: int | None = None
    num_samples = 0

    for batch in loader:
        x = batch["x"].float()
        physics = batch["physics"].float()
        auxiliary_physics_v1 = batch.get("physics_v1")
        residual = batch["residual"].float()
        metadata_vector = batch.get("metadata_vector")
        if input_channels is None:
            input_channels = int(x.shape[1])
            context_accs = [RunningMoments() for _ in range(max(input_channels - 8, 0))]
            if metadata_vector is not None:
                metadata_feature_names = tuple(getattr(dataset, "metadata_feature_names", ()) or ())
                metadata_accs = [RunningMoments() for _ in range(int(metadata_vector.shape[1]))]
        elif input_channels != int(x.shape[1]):
            raise ValueError(f"inconsistent input channel count: expected {input_channels}, got {x.shape[1]}")
        power_acc.update(x[:, 0])
        if context_accs:
            for offset, acc in enumerate(context_accs, start=8):
                acc.update(x[:, offset])
        physics_acc.update(physics)
        if auxiliary_physics_v1 is not None:
            auxiliary_physics_v1_acc.update(auxiliary_physics_v1.float())
            saw_auxiliary_physics_v1 = True
        residual_acc.update(residual)
        if metadata_vector is not None and metadata_accs is not None:
            for index, acc in enumerate(metadata_accs):
                acc.update(metadata_vector[:, index])
        num_samples += int(x.shape[0])

    num_grid_cells = int(power_acc.count)
    context_accs = context_accs or []
    metadata_accs = metadata_accs or []
    context_indices = tuple(range(8, int(input_channels or 8)))
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
        input_channels=int(input_channels or 8),
        context_channel_indices=context_indices,
        context_channel_names=context_channel_names(dataset, context_indices),
        context_channel_means=tuple(acc.mean for acc in context_accs),
        context_channel_stds=tuple(acc.std for acc in context_accs),
        metadata_feature_names=metadata_feature_names,
        metadata_means=tuple(acc.mean for acc in metadata_accs),
        metadata_stds=tuple(acc.std for acc in metadata_accs),
        auxiliary_physics_v1_mean=auxiliary_physics_v1_acc.mean if saw_auxiliary_physics_v1 else None,
        auxiliary_physics_v1_std=auxiliary_physics_v1_acc.std if saw_auxiliary_physics_v1 else None,
    )


def save_normalization_stats(stats: NormalizationStats, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "context_channel_indices",
        "context_channel_names",
        "context_channel_means",
        "context_channel_stds",
        "metadata_feature_names",
        "metadata_means",
        "metadata_stds",
    ):
        if key in data:
            data[key] = tuple(data[key])
    return NormalizationStats(**data)


def context_channel_names(dataset: Dataset[Any], indices: tuple[int, ...]) -> tuple[str, ...]:
    dataset_names = getattr(dataset, "channel_names", None)
    if isinstance(dataset_names, list) and len(dataset_names) > max(indices, default=-1):
        return tuple(str(dataset_names[index]) for index in indices)
    index_csv = getattr(dataset, "index_csv", None)
    if index_csv is None:
        return tuple(f"channel_{index}" for index in indices)
    manifest_candidates = [
        Path(index_csv).parent / "feature_manifest.json",
        Path(index_csv).parent.parent / "feature_manifest.json",
        Path(index_csv).parent / "context_manifest.json",
        Path(index_csv).parent.parent / "context_manifest.json",
    ]
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), manifest_candidates[0])
    if not manifest_path.exists():
        return tuple(f"channel_{index}" for index in indices)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return tuple(f"channel_{index}" for index in indices)
    channel_names = manifest.get("channel_names")
    if isinstance(channel_names, list) and len(channel_names) > max(indices, default=-1):
        return tuple(str(channel_names[index]) for index in indices)
    context_channels = manifest.get("context_channels")
    if isinstance(context_channels, list) and len(context_channels) >= len(indices):
        return tuple(str(name) for name in context_channels[: len(indices)])
    return tuple(f"channel_{index}" for index in indices)


def build_model_input(
    x: torch.Tensor,
    physics: torch.Tensor,
    stats: NormalizationStats,
    *,
    physics_input_mode: str = "v1",
    physics_v1: torch.Tensor | None = None,
) -> torch.Tensor:
    if physics_input_mode not in {
        "v1",
        "none",
        "gated_v1",
        "source_superposition_v1",
        "source_superposition_plus_physics_v1",
    }:
        raise ValueError(f"unsupported physics_input_mode: {physics_input_mode}")
    x_norm = x.float().clone()
    x_norm[:, 0] = normalize_tensor(x_norm[:, 0], stats.power_density_mean, stats.power_density_std)
    for channel, mean, std in zip(stats.context_channel_indices, stats.context_channel_means, stats.context_channel_stds):
        if int(channel) < x_norm.shape[1]:
            x_norm[:, int(channel)] = normalize_tensor(x_norm[:, int(channel)], float(mean), float(std))
    if physics_input_mode == "none":
        return x_norm
    physics_norm = normalize_tensor(physics.float(), stats.physics_mean, stats.physics_std).unsqueeze(1)
    if physics_input_mode != "source_superposition_plus_physics_v1":
        return torch.cat([x_norm, physics_norm], dim=1)
    if physics_v1 is None:
        raise ValueError("physics_input_mode=source_superposition_plus_physics_v1 requires physics_v1 tensor")
    if stats.auxiliary_physics_v1_mean is None or stats.auxiliary_physics_v1_std is None:
        raise ValueError("normalization stats are missing auxiliary physics-v1 mean/std")
    physics_v1_norm = normalize_tensor(
        physics_v1.float(),
        float(stats.auxiliary_physics_v1_mean),
        float(stats.auxiliary_physics_v1_std),
    ).unsqueeze(1)
    return torch.cat([x_norm, physics_norm, physics_v1_norm], dim=1)


def build_metadata_input(metadata_vector: torch.Tensor | None, stats: NormalizationStats) -> torch.Tensor | None:
    if metadata_vector is None:
        return None
    value = metadata_vector.float().clone()
    if not stats.metadata_means:
        return value
    means = torch.tensor(stats.metadata_means, dtype=value.dtype, device=value.device).view(1, -1)
    stds = torch.tensor([max(float(std), EPSILON) for std in stats.metadata_stds], dtype=value.dtype, device=value.device).view(1, -1)
    return (value - means) / stds


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
