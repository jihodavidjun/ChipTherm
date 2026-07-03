from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


TargetName = Literal["residual", "temperature"]

REPO_ROOT = Path(__file__).resolve().parents[3]


class ChipThermDataset(Dataset):
    """Lazy PyTorch dataset for ChipTherm encoded benchmark samples."""

    def __init__(
        self,
        index_csv: str | Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        target: TargetName = "residual",
        return_metadata: bool = True,
    ) -> None:
        self.index_csv = Path(index_csv).expanduser().resolve()
        self.transform = transform
        self.target = target
        self.return_metadata = return_metadata
        if target not in {"residual", "temperature"}:
            raise ValueError("target must be 'residual' or 'temperature'")
        if not self.index_csv.exists():
            raise FileNotFoundError(self.index_csv)
        self.rows = self._read_rows(self.index_csv)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch.is_tensor(index):
            index = int(index.item())
        row = self.rows[index]

        x = self._load_tensor(row["x_path"], expected_ndim=3)
        temperature = self._load_tensor(row["y_path"], expected_ndim=2)
        physics = self._load_tensor(row["prediction_path"], expected_ndim=2)
        residual_path = self._resolve_path(row["residual_path"])
        if residual_path.exists():
            residual = self._load_tensor(row["residual_path"], expected_ndim=2)
        else:
            residual = temperature - physics

        target_tensor = residual if self.target == "residual" else temperature
        sample: dict[str, Any] = {
            "x": x,
            "target": target_tensor,
            "physics": physics,
            "temperature": temperature,
            "residual": residual,
        }
        if self.return_metadata:
            sample["metadata"] = self._metadata(row)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def cases(self) -> list[str]:
        return sorted({row["case_id"] for row in self.rows})

    def num_cases(self) -> int:
        return len(self.cases())

    def statistics(self) -> dict[str, Any]:
        split_counts = Counter(row.get("split", "") for row in self.rows)
        source_counts = Counter(row["dataset_source"] for row in self.rows)
        case_counts = Counter(row["case_id"] for row in self.rows)
        return {
            "num_samples": len(self.rows),
            "splits": dict(sorted(split_counts.items())),
            "cases": self.cases(),
            "num_cases": self.num_cases(),
            "samples_per_case": dict(sorted(case_counts.items())),
            "dataset_sources": dict(sorted(source_counts.items())),
            "input_shape": self._array_shape("x_path"),
            "target_shape": self._array_shape("residual_path" if self.target == "residual" else "y_path"),
            "target": self.target,
            "mean_hotspot_temperature_K": self._mean("mean_temperature_K"),
            "mean_power_W": self._mean("total_power_W"),
            "mean_chiplet_count": self._mean("num_chiplets"),
        }

    def summary(self) -> str:
        stats = self.statistics()
        splits = ", ".join(f"{name}: {count}" for name, count in stats["splits"].items())
        sources = ", ".join(f"{name}: {count}" for name, count in stats["dataset_sources"].items())
        lines = [
            "-" * 40,
            "ChipThermDataset",
            "-" * 40,
            f"Samples: {stats['num_samples']}",
            f"Split: {splits}",
            f"Cases: {stats['num_cases']} ({', '.join(stats['cases'])})",
            f"Input shape: {stats['input_shape']}",
            f"Target: {stats['target']} {stats['target_shape']}",
            f"Mean hotspot temperature: {stats['mean_hotspot_temperature_K']:.3f} K",
            f"Mean power: {stats['mean_power_W']:.3f} W",
            f"Mean chiplet count: {stats['mean_chiplet_count']:.3f}",
            f"Dataset sources: {sources}",
            "-" * 40,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        if not rows:
            raise ValueError(f"{path} does not contain any samples")
        return rows

    def _load_tensor(self, path_value: str, *, expected_ndim: int) -> torch.Tensor:
        path = self._resolve_path(path_value)
        array = np.load(path).astype(np.float32, copy=False)
        if array.ndim != expected_ndim:
            raise ValueError(f"{path} expected {expected_ndim} dimensions, got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path} contains non-finite values")
        return torch.from_numpy(array)

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        candidates = [
            Path.cwd() / path,
            REPO_ROOT / path,
            self.index_csv.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _metadata(self, row: dict[str, str]) -> dict[str, Any]:
        return {
            "sample_uid": row["sample_uid"],
            "original_sample_uid": row.get("original_sample_uid"),
            "case_id": row["case_id"],
            "dataset_source": row["dataset_source"],
            "split": row.get("split"),
            "num_chiplets": self._optional_int(row.get("num_chiplets")),
            "total_power_W": self._optional_float(row.get("total_power_W")),
            "hotspot_runtime_s": self._optional_float(row.get("hotspot_runtime_s")),
            "physics_runtime_s": self._optional_float(row.get("physics_runtime_s")),
            "x_path": row["x_path"],
            "y_path": row["y_path"],
            "prediction_path": row["prediction_path"],
            "residual_path": row["residual_path"],
        }

    def _array_shape(self, column: str) -> tuple[int, ...]:
        path = self._resolve_path(self.rows[0][column])
        return tuple(int(size) for size in np.load(path, mmap_mode="r").shape)

    def _mean(self, column: str) -> float:
        values = [self._optional_float(row.get(column)) for row in self.rows]
        numeric = [value for value in values if value is not None]
        if not numeric:
            return float("nan")
        return float(sum(numeric) / len(numeric))

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(float(value))
