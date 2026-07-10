from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
EPSILON = 1.0e-8


@dataclass(frozen=True)
class DualPhysicsNormalizationStats:
    schema_version: int
    power_density_mean: float
    power_density_std: float
    physics_v1_mean: float
    physics_v1_std: float
    physics_v2_mean: float
    physics_v2_std: float
    residual_v1_mean: float
    residual_v1_std: float
    num_samples: int
    num_grid_cells: int
    input_channels: int
    context_channel_indices: tuple[int, ...] = ()
    context_channel_means: tuple[float, ...] = ()
    context_channel_stds: tuple[float, ...] = ()
    notes: str = (
        "Computed from train split only. physics_v1 and physics_v2 have separate "
        "normalization statistics. Target is residual_v1 = HotSpot - physics_v1."
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_channel_indices"] = list(self.context_channel_indices)
        data["context_channel_means"] = list(self.context_channel_means)
        data["context_channel_stds"] = list(self.context_channel_stds)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DualPhysicsNormalizationStats":
        for key in ("context_channel_indices", "context_channel_means", "context_channel_stds"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


class DualPhysicsDataset(Dataset):
    """Matched v1/v2 physics dataset with v1 as the residual anchor."""

    def __init__(
        self,
        physics_v1_index: str | Path,
        physics_v2_index: str | Path,
        *,
        verify_residual: bool = True,
        residual_tolerance: float = 1.0e-3,
    ) -> None:
        self.physics_v1_index = Path(physics_v1_index).expanduser().resolve()
        self.physics_v2_index = Path(physics_v2_index).expanduser().resolve()
        self.verify_residual = bool(verify_residual)
        self.residual_tolerance = float(residual_tolerance)
        self.v1_rows = _read_rows(self.physics_v1_index)
        self.v2_rows_by_uid = _rows_by_uid(_read_rows(self.physics_v2_index), self.physics_v2_index)
        self.rows = self._match_rows()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch.is_tensor(index):
            index = int(index.item())
        v1_row, v2_row = self.rows[index]
        x = _load_tensor(v1_row["x_path"], self.physics_v1_index.parent, expected_ndim=3)
        temperature = _load_tensor(v1_row["y_path"], self.physics_v1_index.parent, expected_ndim=2)
        physics_v1 = _load_tensor(v1_row["prediction_path"], self.physics_v1_index.parent, expected_ndim=2)
        physics_v2 = _load_tensor(v2_row["prediction_path"], self.physics_v2_index.parent, expected_ndim=2)
        residual_v1_path = _resolve_path(v1_row["residual_path"], self.physics_v1_index.parent)
        if residual_v1_path.exists():
            residual_v1 = _load_tensor(v1_row["residual_path"], self.physics_v1_index.parent, expected_ndim=2)
        else:
            residual_v1 = temperature - physics_v1
        if self.verify_residual:
            mismatch = torch.max(torch.abs((temperature - physics_v1) - residual_v1)).item()
            if mismatch > self.residual_tolerance:
                raise ValueError(f"{v1_row['sample_uid']} residual_v1 mismatch {mismatch:.6g}")
        return {
            "x": x,
            "temperature": temperature,
            "physics_v1": physics_v1,
            "physics_v2": physics_v2,
            "residual_v1": residual_v1,
            "target": residual_v1,
            "metadata": {
                "sample_uid": v1_row["sample_uid"],
                "original_sample_uid": v1_row.get("original_sample_uid"),
                "case_id": v1_row["case_id"],
                "dataset_source": v1_row.get("dataset_source", ""),
                "split": v1_row.get("split", ""),
                "num_chiplets": _optional_int(v1_row.get("num_chiplets")),
                "total_power_W": _optional_float(v1_row.get("total_power_W")),
                "hotspot_runtime_s": _optional_float(v1_row.get("hotspot_runtime_s")),
                "physics_v1_runtime_s": _optional_float(v1_row.get("physics_runtime_s")),
                "physics_v2_runtime_s": _optional_float(v2_row.get("physics_runtime_s")),
                "x_path": v1_row["x_path"],
                "y_path": v1_row["y_path"],
                "physics_v1_path": v1_row["prediction_path"],
                "physics_v2_path": v2_row["prediction_path"],
                "residual_v1_path": v1_row["residual_path"],
            },
        }

    def _match_rows(self) -> list[tuple[dict[str, str], dict[str, str]]]:
        matched: list[tuple[dict[str, str], dict[str, str]]] = []
        missing: list[str] = []
        errors: list[str] = []
        for row in self.v1_rows:
            uid = row["sample_uid"]
            v2_row = self.v2_rows_by_uid.get(uid)
            if v2_row is None:
                missing.append(uid)
                continue
            if row.get("case_id") != v2_row.get("case_id"):
                errors.append(f"{uid}: case_id differs ({row.get('case_id')} vs {v2_row.get('case_id')})")
            if _resolve_path(row["x_path"], self.physics_v1_index.parent) != _resolve_path(v2_row["x_path"], self.physics_v2_index.parent):
                errors.append(f"{uid}: x_path differs")
            if _resolve_path(row["y_path"], self.physics_v1_index.parent) != _resolve_path(v2_row["y_path"], self.physics_v2_index.parent):
                errors.append(f"{uid}: y_path differs")
            if _resolve_path(row["prediction_path"], self.physics_v1_index.parent) == _resolve_path(v2_row["prediction_path"], self.physics_v2_index.parent):
                errors.append(f"{uid}: v1 and v2 prediction paths are identical")
            matched.append((row, v2_row))
        extra = sorted(set(self.v2_rows_by_uid) - {row["sample_uid"] for row in self.v1_rows})
        if missing or extra or errors:
            details = []
            if missing:
                details.append(f"missing v2 rows: {missing[:5]}")
            if extra:
                details.append(f"extra v2 rows: {extra[:5]}")
            details.extend(errors[:10])
            raise ValueError("; ".join(details))
        return matched


def compute_dual_physics_normalization_stats(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int | None = None,
) -> DualPhysicsNormalizationStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    power_acc = RunningMoments()
    physics_v1_acc = RunningMoments()
    physics_v2_acc = RunningMoments()
    residual_acc = RunningMoments()
    context_accs: list[RunningMoments] | None = None
    input_channels: int | None = None
    num_samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = batch["x"].float()
        physics_v1 = batch["physics_v1"].float()
        physics_v2 = batch["physics_v2"].float()
        residual_v1 = batch["residual_v1"].float()
        if input_channels is None:
            input_channels = int(x.shape[1])
            context_accs = [RunningMoments() for _ in range(max(input_channels - 8, 0))]
        elif input_channels != int(x.shape[1]):
            raise ValueError(f"inconsistent input channel count: expected {input_channels}, got {x.shape[1]}")
        power_acc.update(x[:, 0])
        if context_accs:
            for offset, acc in enumerate(context_accs, start=8):
                acc.update(x[:, offset])
        physics_v1_acc.update(physics_v1)
        physics_v2_acc.update(physics_v2)
        residual_acc.update(residual_v1)
        num_samples += int(x.shape[0])
    if input_channels is None:
        raise ValueError("no batches available for normalization")
    context_accs = context_accs or []
    return DualPhysicsNormalizationStats(
        schema_version=1,
        power_density_mean=power_acc.mean,
        power_density_std=power_acc.std,
        physics_v1_mean=physics_v1_acc.mean,
        physics_v1_std=physics_v1_acc.std,
        physics_v2_mean=physics_v2_acc.mean,
        physics_v2_std=physics_v2_acc.std,
        residual_v1_mean=residual_acc.mean,
        residual_v1_std=residual_acc.std,
        num_samples=num_samples,
        num_grid_cells=int(power_acc.count),
        input_channels=int(input_channels),
        context_channel_indices=tuple(range(8, int(input_channels))),
        context_channel_means=tuple(acc.mean for acc in context_accs),
        context_channel_stds=tuple(acc.std for acc in context_accs),
    )


def build_dual_physics_model_input(
    x: torch.Tensor,
    physics_v1: torch.Tensor,
    physics_v2: torch.Tensor,
    stats: DualPhysicsNormalizationStats,
) -> torch.Tensor:
    x_norm = x.float().clone()
    x_norm[:, 0] = normalize_tensor(x_norm[:, 0], stats.power_density_mean, stats.power_density_std)
    for channel, mean, std in zip(stats.context_channel_indices, stats.context_channel_means, stats.context_channel_stds):
        if int(channel) < x_norm.shape[1]:
            x_norm[:, int(channel)] = normalize_tensor(x_norm[:, int(channel)], float(mean), float(std))
    physics_v1_norm = normalize_tensor(physics_v1.float(), stats.physics_v1_mean, stats.physics_v1_std).unsqueeze(1)
    physics_v2_norm = normalize_tensor(physics_v2.float(), stats.physics_v2_mean, stats.physics_v2_std).unsqueeze(1)
    return torch.cat([x_norm, physics_v1_norm, physics_v2_norm], dim=1)


def normalize_residual_v1(residual: torch.Tensor, stats: DualPhysicsNormalizationStats) -> torch.Tensor:
    return normalize_tensor(residual.float(), stats.residual_v1_mean, stats.residual_v1_std)


def unnormalize_residual_v1(residual_norm: torch.Tensor, stats: DualPhysicsNormalizationStats) -> torch.Tensor:
    return residual_norm.float() * float(stats.residual_v1_std) + float(stats.residual_v1_mean)


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
        return float(self.total / self.count) if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 1.0
        variance = max(self.total_sq / self.count - self.mean * self.mean, EPSILON)
        return float(variance**0.5)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"{path} does not contain any samples")
    return rows


def _rows_by_uid(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    by_uid: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = row.get("sample_uid")
        if not uid:
            raise ValueError(f"{path} contains row without sample_uid")
        if uid in by_uid:
            raise ValueError(f"{path} contains duplicate sample_uid {uid}")
        by_uid[uid] = row
    return by_uid


def _load_tensor(path_value: str, base: Path, *, expected_ndim: int) -> torch.Tensor:
    path = _resolve_path(path_value, base)
    array = np.load(path).astype(np.float32, copy=False)
    if array.ndim != expected_ndim:
        raise ValueError(f"{path} expected {expected_ndim} dimensions, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite values")
    return torch.from_numpy(array)


def _resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        base / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))
