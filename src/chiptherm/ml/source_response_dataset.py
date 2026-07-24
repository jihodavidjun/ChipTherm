from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
EPSILON = 1.0e-8


SOURCE_RESPONSE_CHANNEL_NAMES = (
    "occupancy_mask",
    "CPU_mask",
    "GPU_or_NPU_mask",
    "memory_mask",
    "IO_or_ANALOG_or_MEMS_mask",
    "normalized_x_coordinate",
    "normalized_y_coordinate",
    "source_mask",
    "source_power_density_W_per_mm2",
    "source_dx_mm",
    "source_dy_mm",
    "source_radius_mm",
    "distance_to_left_edge_mm",
    "distance_to_right_edge_mm",
    "distance_to_bottom_edge_mm",
    "distance_to_top_edge_mm",
    "minimum_distance_to_package_edge_mm",
)

NORMALIZED_CHANNELS = (
    "source_power_density_W_per_mm2",
    "source_dx_mm",
    "source_dy_mm",
    "source_radius_mm",
    "distance_to_left_edge_mm",
    "distance_to_right_edge_mm",
    "distance_to_bottom_edge_mm",
    "distance_to_top_edge_mm",
    "minimum_distance_to_package_edge_mm",
)


@dataclass(frozen=True)
class SourceResponseNormalizationStats:
    schema_version: int
    channel_names: tuple[str, ...]
    channel_means: tuple[float, ...]
    channel_stds: tuple[float, ...]
    normalized_channel_indices: tuple[int, ...]
    target_unit_mean_K_per_W: float
    target_unit_std_K_per_W: float
    source_power_min_W: float
    source_power_p01_W: float
    source_power_p05_W: float
    source_power_p50_W: float
    source_power_p95_W: float
    source_power_max_W: float
    target_rise_abs_max_K: float
    target_unit_abs_max_K_per_W: float
    power_floor_W: float
    num_sources: int
    target_normalization_mode: str = "standardized_unit_response_K_per_W"
    notes: str = "Computed from train source-response split only. Binary masks and normalized coordinates are not standardized."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("channel_names", "channel_means", "channel_stds", "normalized_channel_indices"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceResponseNormalizationStats":
        payload = dict(data)
        for key in ("channel_names", "channel_means", "channel_stds", "normalized_channel_indices"):
            payload[key] = tuple(payload.get(key, ()))
        payload.setdefault("target_normalization_mode", "standardized_unit_response_K_per_W")
        return cls(**payload)


class SourceResponseDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        *,
        power_floor_W: float = 1.0e-6,
        return_metadata: bool = True,
        data_root: str | Path | None = None,
    ) -> None:
        self.index_csv = Path(index_csv).expanduser().resolve()
        if not self.index_csv.exists():
            raise FileNotFoundError(self.index_csv)
        self.rows = read_rows(self.index_csv)
        if not self.rows:
            raise ValueError(f"{self.index_csv} contains no source-response rows")
        self.power_floor_W = float(power_floor_W)
        self.return_metadata = bool(return_metadata)
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else None
        self.channel_names = SOURCE_RESPONSE_CHANNEL_NAMES

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        x = np.load(resolve_path(row["original_x_path"], self.index_csv.parent, self.data_root)).astype(np.float32, copy=False)
        target_rise = np.load(resolve_path(row["target_rise_path"], self.index_csv.parent, self.data_root)).astype(np.float32, copy=False)
        layout = load_json(resolve_path(row["layout_path"], self.index_csv.parent, self.data_root))
        source_index = int(row["source_index"])
        source_power_W = float(row["source_power_W"])
        source_input = build_source_input(x, layout, source_index, source_power_W)
        unit_target = target_rise / max(source_power_W, self.power_floor_W)
        full_temperature_path = row.get("full_temperature_path") or row.get("original_y_path")
        sample: dict[str, Any] = {
            "x": torch.from_numpy(source_input),
            "target_rise": torch.from_numpy(target_rise),
            "target_unit": torch.from_numpy(unit_target.astype(np.float32, copy=False)),
            "source_power_W": torch.tensor(source_power_W, dtype=torch.float32),
            "ambient_K": torch.tensor(float(row["ambient_K"]), dtype=torch.float32),
            "full_temperature": torch.from_numpy(
                np.load(resolve_path(full_temperature_path, self.index_csv.parent, self.data_root)).astype(np.float32, copy=False)
            ),
        }
        if self.return_metadata:
            sample["metadata"] = dict(row)
        return sample

    def original_sample_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(self.rows):
            groups.setdefault(row["original_sample_uid"], []).append(index)
        return groups

    def resolve_row_path(self, path_value: str | None) -> Path:
        return resolve_path(path_value, self.index_csv.parent, self.data_root)


class SourceResponsePackageDataset(Dataset):
    """Package-grouped view where each item contains all source rows for one sample."""

    def __init__(
        self,
        index_csv: str | Path,
        *,
        power_floor_W: float = 1.0e-6,
        require_complete: bool = True,
        data_root: str | Path | None = None,
    ) -> None:
        self.source_dataset = SourceResponseDataset(
            index_csv,
            power_floor_W=power_floor_W,
            return_metadata=True,
            data_root=data_root,
        )
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(self.source_dataset.rows):
            groups.setdefault(row["original_sample_uid"], []).append(index)
        self.package_uids: list[str] = []
        self.group_indices: list[list[int]] = []
        for uid, indices in sorted(groups.items()):
            ordered = sorted(indices, key=lambda item: int(float(self.source_dataset.rows[item]["source_index"])))
            first = self.source_dataset.rows[ordered[0]]
            expected = int(float(first["num_chiplets"]))
            if require_complete and len(ordered) != expected:
                raise ValueError(
                    f"package {uid} has {len(ordered)} source rows but expected {expected}; "
                    "package-level loss requires complete source groups"
                )
            self.package_uids.append(uid)
            self.group_indices.append(ordered)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.source_dataset.channel_names

    @property
    def rows(self) -> list[dict[str, str]]:
        return self.source_dataset.rows

    @property
    def power_floor_W(self) -> float:
        return self.source_dataset.power_floor_W

    def __len__(self) -> int:
        return len(self.group_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_indices = self.group_indices[int(index)]
        sources = [self.source_dataset[source_index] for source_index in source_indices]
        first_meta = sources[0]["metadata"]
        source_powers = torch.stack([source["source_power_W"] for source in sources])
        return {
            "original_sample_uid": str(first_meta["original_sample_uid"]),
            "case_id": str(first_meta["case_id"]),
            "sources": sources,
            "num_sources": len(sources),
            "total_power_W": torch.sum(source_powers),
            "ambient_K": sources[0]["ambient_K"],
            "full_temperature": sources[0]["full_temperature"],
        }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def resolve_path(path_value: str | None, base: Path, data_root: Path | None = None) -> Path:
    if path_value is None:
        raise ValueError("path value is missing")
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if data_root is not None:
        resolved = data_root / path
        if not resolved.exists():
            raise FileNotFoundError(
                f"source-response path does not exist: logical={path_value!r}, "
                f"data_root={data_root}, resolved={resolved}"
            )
        return resolved
    for candidate in (Path.cwd() / path, REPO_ROOT / path, base / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def build_source_input(original_x: np.ndarray, layout: dict[str, Any], source_index: int, source_power_W: float) -> np.ndarray:
    if original_x.ndim != 3 or original_x.shape[0] < 8:
        raise ValueError(f"original_x must have at least 8 channels, got {original_x.shape}")
    rows, cols = int(original_x.shape[-2]), int(original_x.shape[-1])
    if source_index < 0 or source_index >= len(layout.get("chiplets", [])):
        raise IndexError(f"source_index {source_index} out of range")
    package_size = layout["package"]["size"]
    width_mm = float(package_size["width"])
    height_mm = float(package_size["height"])
    x_coords = (np.arange(cols, dtype=np.float32) + 0.5) / float(cols) * width_mm
    y_coords = (np.arange(rows, dtype=np.float32) + 0.5) / float(rows) * height_mm
    xx, yy = np.meshgrid(x_coords, y_coords)

    source = layout["chiplets"][source_index]
    pos = source["position"]
    size = source["size"]
    left = float(pos["x"])
    bottom = float(pos["y"])
    source_width = float(size["width"])
    source_height = float(size["height"])
    source_area = max(source_width * source_height, EPSILON)
    source_mask = (
        (xx >= left)
        & (xx < left + source_width)
        & (yy >= bottom)
        & (yy < bottom + source_height)
    ).astype(np.float32)
    center_x = left + 0.5 * source_width
    center_y = bottom + 0.5 * source_height
    dx = xx - center_x
    dy = yy - center_y
    radius = np.sqrt(dx * dx + dy * dy)
    edge_left = xx
    edge_right = width_mm - xx
    edge_bottom = yy
    edge_top = height_mm - yy
    min_edge = np.minimum.reduce([edge_left, edge_right, edge_bottom, edge_top])
    source_power_density = source_mask * (float(source_power_W) / source_area)

    channels = [
        original_x[1],
        original_x[2],
        original_x[3],
        original_x[4],
        original_x[5],
        original_x[6],
        original_x[7],
        source_mask,
        source_power_density.astype(np.float32),
        dx.astype(np.float32),
        dy.astype(np.float32),
        radius.astype(np.float32),
        edge_left.astype(np.float32),
        edge_right.astype(np.float32),
        edge_bottom.astype(np.float32),
        edge_top.astype(np.float32),
        min_edge.astype(np.float32),
    ]
    result = np.stack(channels).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("source input contains non-finite values")
    return result


def source_response_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = {"x", "target_rise", "target_unit", "source_power_W", "ambient_K", "full_temperature"}
    result: dict[str, Any] = {}
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in batch])
    result["metadata"] = [item.get("metadata", {}) for item in batch]
    return result


def source_response_package_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    flat_sources: list[dict[str, Any]] = []
    source_to_package: list[int] = []
    package_uids: list[str] = []
    case_ids: list[str] = []
    source_counts: list[int] = []
    total_powers: list[torch.Tensor] = []
    ambients: list[torch.Tensor] = []
    full_temperatures: list[torch.Tensor] = []
    for package_index, package in enumerate(batch):
        package_uids.append(package["original_sample_uid"])
        case_ids.append(package["case_id"])
        source_counts.append(int(package["num_sources"]))
        total_powers.append(package["total_power_W"])
        ambients.append(package["ambient_K"])
        full_temperatures.append(package["full_temperature"])
        for source in package["sources"]:
            flat_sources.append(source)
            source_to_package.append(package_index)
    result = source_response_collate(flat_sources)
    result["source_to_package_index"] = torch.tensor(source_to_package, dtype=torch.long)
    result["package_ambient_K"] = torch.stack(ambients)
    result["package_full_temperature"] = torch.stack(full_temperatures)
    result["package_total_power_W"] = torch.stack(total_powers)
    result["package_source_count"] = torch.tensor(source_counts, dtype=torch.long)
    result["package_original_sample_uid"] = package_uids
    result["package_case_id"] = case_ids
    return result


def compute_source_response_normalization(
    dataset: SourceResponseDataset,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
) -> SourceResponseNormalizationStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=source_response_collate)
    channel_acc = RunningMoments(len(dataset.channel_names))
    target_acc = RunningMoments(1)
    powers: list[float] = []
    target_abs_max = 0.0
    target_unit_abs_max = 0.0
    for batch in loader:
        x = batch["x"].float()
        target_unit = batch["target_unit"].float()
        target_rise = batch["target_rise"].float()
        power = batch["source_power_W"].float()
        channel_acc.update(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))
        target_acc.update(target_unit.reshape(-1, 1))
        powers.extend(float(value) for value in power.tolist())
        target_abs_max = max(target_abs_max, float(target_rise.abs().max().item()))
        target_unit_abs_max = max(target_unit_abs_max, float(target_unit.abs().max().item()))
    power_array = np.asarray(powers, dtype=np.float64)
    normalized_indices = tuple(
        index for index, name in enumerate(dataset.channel_names) if name in NORMALIZED_CHANNELS
    )
    means = channel_acc.mean.tolist()
    stds = channel_acc.std.tolist()
    return SourceResponseNormalizationStats(
        schema_version=1,
        channel_names=tuple(dataset.channel_names),
        channel_means=tuple(float(value) for value in means),
        channel_stds=tuple(float(value) for value in stds),
        normalized_channel_indices=normalized_indices,
        target_unit_mean_K_per_W=float(target_acc.mean[0].item()),
        target_unit_std_K_per_W=float(target_acc.std[0].item()),
        source_power_min_W=float(np.min(power_array)),
        source_power_p01_W=float(np.percentile(power_array, 1)),
        source_power_p05_W=float(np.percentile(power_array, 5)),
        source_power_p50_W=float(np.percentile(power_array, 50)),
        source_power_p95_W=float(np.percentile(power_array, 95)),
        source_power_max_W=float(np.max(power_array)),
        target_rise_abs_max_K=float(target_abs_max),
        target_unit_abs_max_K_per_W=float(target_unit_abs_max),
        power_floor_W=float(dataset.power_floor_W),
        num_sources=len(dataset),
    )


def normalize_source_input(x: torch.Tensor, stats: SourceResponseNormalizationStats) -> torch.Tensor:
    result = x.float().clone()
    for channel in stats.normalized_channel_indices:
        mean = float(stats.channel_means[int(channel)])
        std = max(float(stats.channel_stds[int(channel)]), EPSILON)
        result[:, int(channel)] = (result[:, int(channel)] - mean) / std
    return result


def normalize_source_target_unit(target_unit_K_per_W: torch.Tensor, stats: SourceResponseNormalizationStats) -> torch.Tensor:
    std = max(float(stats.target_unit_std_K_per_W), EPSILON)
    return (target_unit_K_per_W.float() - float(stats.target_unit_mean_K_per_W)) / std


def unnormalize_source_prediction(pred_normalized: torch.Tensor, stats: SourceResponseNormalizationStats) -> torch.Tensor:
    std = max(float(stats.target_unit_std_K_per_W), EPSILON)
    return pred_normalized.float() * std + float(stats.target_unit_mean_K_per_W)


def save_source_response_normalization(stats: SourceResponseNormalizationStats, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_response_normalization(path: str | Path) -> SourceResponseNormalizationStats:
    return SourceResponseNormalizationStats.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.total = torch.zeros(dim, dtype=torch.float64)
        self.total_sq = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        data = values.detach().double().reshape(-1, self.dim).cpu()
        self.count += int(data.shape[0])
        self.total += data.sum(dim=0)
        self.total_sq += (data * data).sum(dim=0)

    @property
    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros(self.dim, dtype=torch.float64)
        return self.total / float(self.count)

    @property
    def std(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones(self.dim, dtype=torch.float64)
        variance = torch.clamp(self.total_sq / float(self.count) - self.mean * self.mean, min=EPSILON)
        return torch.sqrt(variance)
