from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset


TargetName = Literal["residual", "temperature"]
PhysicalRepresentation = Literal["dimensional", "dimensionless_v1", "dimensionless_v2"]

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT_MARKER = ".chiptherm_data_root.json"
V2_TEMPERATURE_PATH_FIELDS = ("y_path", "temp_layer0_path", "original_temp_path")
LEGACY_TEMPERATURE_PATH_FIELDS = ("final_temperature",)


DIMENSIONLESS_V1_TRANSFORMS = {
    "power_density_W_per_mm2": "power_density_W_per_mm2 / (total_power_W / occupied_area_mm2)",
    "total_power_W": "total_power_W / total_power_W = 1",
    "package_width_mm": "package_width_mm / sqrt(package_width_mm * package_height_mm)",
    "package_height_mm": "package_height_mm / sqrt(package_width_mm * package_height_mm)",
    "cell_size_x_mm": "cell_size_x_mm / package_width_mm",
    "cell_size_y_mm": "cell_size_y_mm / package_height_mm",
    "finite_source_L0p5mm": "finite_source_L0p5mm / (total_power_W / L_char_mm)",
    "finite_source_L1mm": "finite_source_L1mm / (total_power_W / L_char_mm)",
    "finite_source_L2mm": "finite_source_L2mm / (total_power_W / L_char_mm)",
    "finite_source_L4mm": "finite_source_L4mm / (total_power_W / L_char_mm)",
    "enclosed_power_R2mm_W": "enclosed_power_R2mm_W / total_power_W",
    "enclosed_power_R4mm_W": "enclosed_power_R4mm_W / total_power_W",
    "enclosed_power_R8mm_W": "enclosed_power_R8mm_W / total_power_W",
    "enclosed_power_R16mm_W": "enclosed_power_R16mm_W / total_power_W",
    "distance_to_left_edge_mm": "distance_to_left_edge_mm / package_width_mm",
    "distance_to_right_edge_mm": "distance_to_right_edge_mm / package_width_mm",
    "distance_to_bottom_edge_mm": "distance_to_bottom_edge_mm / package_height_mm",
    "distance_to_top_edge_mm": "distance_to_top_edge_mm / package_height_mm",
    "minimum_distance_to_package_edge_mm": "minimum_distance_to_package_edge_mm / L_char_mm",
    "chiplet_total_power_W": "chiplet_total_power_W / total_power_W",
    "chiplet_width_mm": "chiplet_width_mm / package_width_mm",
    "chiplet_height_mm": "chiplet_height_mm / package_height_mm",
    "chiplet_area_mm2": "chiplet_area_mm2 / package_area_mm2",
    "chiplet_power_density_W_per_mm2": "chiplet_power_density_W_per_mm2 / (total_power_W / occupied_area_mm2)",
    "thermal_crowding_W_per_mm": "thermal_crowding_W_per_mm / (total_power_W / L_char_mm)",
}


DIMENSIONLESS_V2_TRANSFORMS = {
    "chiplet_width_mm": "chiplet_width_mm / package_width_mm",
    "chiplet_height_mm": "chiplet_height_mm / package_height_mm",
    "chiplet_area_mm2": "chiplet_area_mm2 / (package_width_mm * package_height_mm)",
    "distance_to_left_edge_mm": "distance_to_left_edge_mm / package_width_mm",
    "distance_to_right_edge_mm": "distance_to_right_edge_mm / package_width_mm",
    "distance_to_bottom_edge_mm": "distance_to_bottom_edge_mm / package_height_mm",
    "distance_to_top_edge_mm": "distance_to_top_edge_mm / package_height_mm",
    "minimum_distance_to_package_edge_mm": "minimum_distance_to_package_edge_mm / sqrt(package_width_mm * package_height_mm)",
}


def build_dimensionless_v1_input(
    x: torch.Tensor,
    channel_indices: dict[str, int],
    *,
    sample_uid: str = "",
) -> torch.Tensor:
    """Replace selected dimensional context channels with dimensionless ratios.

    The source-superposition base is not part of ``x`` and remains an absolute
    Kelvin channel when ``build_model_input`` appends it later.
    """
    out = x.float().clone()

    def has(name: str) -> bool:
        return name in channel_indices and int(channel_indices[name]) < int(out.shape[0])

    def channel(name: str) -> torch.Tensor:
        return out[int(channel_indices[name])]

    def scalar(name: str) -> torch.Tensor:
        value = channel(name)
        return value.reshape(-1)[0].clone()

    def require_positive(name: str, value: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(value).all() or torch.any(value <= 0.0):
            raise ValueError(f"{sample_uid or 'sample'} has invalid positive denominator for {name}: {value}")
        return value

    required = ["occupancy_mask", "total_power_W", "package_width_mm", "package_height_mm", "cell_size_x_mm", "cell_size_y_mm"]
    missing = [name for name in required if not has(name)]
    if missing:
        raise ValueError(f"dimensionless_v1 requires channels missing from sample {sample_uid or '<unknown>'}: {missing}")

    package_width = require_positive("package_width_mm", scalar("package_width_mm"))
    package_height = require_positive("package_height_mm", scalar("package_height_mm"))
    cell_size_x = require_positive("cell_size_x_mm", scalar("cell_size_x_mm"))
    cell_size_y = require_positive("cell_size_y_mm", scalar("cell_size_y_mm"))
    total_power = require_positive("total_power_W", scalar("total_power_W"))
    package_area = require_positive("package_area_mm2", package_width * package_height)
    l_char = require_positive("L_char_mm", torch.sqrt(package_area))
    occupancy = channel("occupancy_mask") > 0.5
    occupied_area = require_positive(
        "occupied_area_mm2",
        occupancy.to(out.dtype).sum() * cell_size_x * cell_size_y,
    )
    characteristic_power_density = require_positive(
        "characteristic_power_density_W_per_mm2",
        total_power / occupied_area,
    )
    power_per_length = require_positive("total_power_W / L_char_mm", total_power / l_char)

    if has("power_density_W_per_mm2"):
        out[channel_indices["power_density_W_per_mm2"]] = channel("power_density_W_per_mm2") / characteristic_power_density
    out[channel_indices["total_power_W"]] = channel("total_power_W") / total_power
    out[channel_indices["package_width_mm"]] = channel("package_width_mm") / l_char
    out[channel_indices["package_height_mm"]] = channel("package_height_mm") / l_char
    out[channel_indices["cell_size_x_mm"]] = channel("cell_size_x_mm") / package_width
    out[channel_indices["cell_size_y_mm"]] = channel("cell_size_y_mm") / package_height

    for name in ("finite_source_L0p5mm", "finite_source_L1mm", "finite_source_L2mm", "finite_source_L4mm"):
        if has(name):
            out[channel_indices[name]] = channel(name) / power_per_length
    for name in ("enclosed_power_R2mm_W", "enclosed_power_R4mm_W", "enclosed_power_R8mm_W", "enclosed_power_R16mm_W"):
        if has(name):
            out[channel_indices[name]] = channel(name) / total_power
    for name in ("distance_to_left_edge_mm", "distance_to_right_edge_mm"):
        if has(name):
            out[channel_indices[name]] = channel(name) / package_width
    for name in ("distance_to_bottom_edge_mm", "distance_to_top_edge_mm"):
        if has(name):
            out[channel_indices[name]] = channel(name) / package_height
    if has("minimum_distance_to_package_edge_mm"):
        out[channel_indices["minimum_distance_to_package_edge_mm"]] = channel("minimum_distance_to_package_edge_mm") / l_char
    if has("chiplet_total_power_W"):
        out[channel_indices["chiplet_total_power_W"]] = channel("chiplet_total_power_W") / total_power
    if has("chiplet_width_mm"):
        out[channel_indices["chiplet_width_mm"]] = channel("chiplet_width_mm") / package_width
    if has("chiplet_height_mm"):
        out[channel_indices["chiplet_height_mm"]] = channel("chiplet_height_mm") / package_height
    if has("chiplet_area_mm2"):
        out[channel_indices["chiplet_area_mm2"]] = channel("chiplet_area_mm2") / package_area
    if has("chiplet_power_density_W_per_mm2"):
        out[channel_indices["chiplet_power_density_W_per_mm2"]] = channel("chiplet_power_density_W_per_mm2") / characteristic_power_density
    if has("thermal_crowding_W_per_mm"):
        out[channel_indices["thermal_crowding_W_per_mm"]] = channel("thermal_crowding_W_per_mm") / power_per_length
    if not torch.isfinite(out).all():
        bad = torch.nonzero(~torch.isfinite(out), as_tuple=False)
        first = bad[0].tolist() if bad.numel() else []
        raise ValueError(f"dimensionless_v1 produced non-finite input for {sample_uid or '<unknown>'}: first bad index {first}")
    return out


def build_dimensionless_v2_input(
    x: torch.Tensor,
    channel_indices: dict[str, int],
    *,
    sample_uid: str = "",
) -> torch.Tensor:
    """Apply geometry-only package-relative ratios to the original dimensional tensor."""
    out = x.float().clone()

    def has(name: str) -> bool:
        return name in channel_indices and int(channel_indices[name]) < int(out.shape[0])

    def channel(name: str) -> torch.Tensor:
        return out[int(channel_indices[name])]

    def scalar(name: str) -> torch.Tensor:
        return channel(name).reshape(-1)[0].clone()

    def require_positive(name: str, value: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(value).all() or torch.any(value <= 0.0):
            raise ValueError(f"{sample_uid or 'sample'} has invalid positive denominator for {name}: {value}")
        return value

    required = ["package_width_mm", "package_height_mm"]
    missing = [name for name in required if not has(name)]
    if missing:
        raise ValueError(f"dimensionless_v2 requires channels missing from sample {sample_uid or '<unknown>'}: {missing}")

    package_width = require_positive("package_width_mm", scalar("package_width_mm"))
    package_height = require_positive("package_height_mm", scalar("package_height_mm"))
    package_area = require_positive("package_area_mm2", package_width * package_height)
    l_char = require_positive("L_char_mm", torch.sqrt(package_area))

    if has("chiplet_width_mm"):
        out[channel_indices["chiplet_width_mm"]] = channel("chiplet_width_mm") / package_width
    if has("chiplet_height_mm"):
        out[channel_indices["chiplet_height_mm"]] = channel("chiplet_height_mm") / package_height
    if has("chiplet_area_mm2"):
        out[channel_indices["chiplet_area_mm2"]] = channel("chiplet_area_mm2") / package_area
    for name in ("distance_to_left_edge_mm", "distance_to_right_edge_mm"):
        if has(name):
            out[channel_indices[name]] = channel(name) / package_width
    for name in ("distance_to_bottom_edge_mm", "distance_to_top_edge_mm"):
        if has(name):
            out[channel_indices[name]] = channel(name) / package_height
    if has("minimum_distance_to_package_edge_mm"):
        out[channel_indices["minimum_distance_to_package_edge_mm"]] = channel("minimum_distance_to_package_edge_mm") / l_char
    if not torch.isfinite(out).all():
        bad = torch.nonzero(~torch.isfinite(out), as_tuple=False)
        first = bad[0].tolist() if bad.numel() else []
        raise ValueError(f"dimensionless_v2 produced non-finite input for {sample_uid or '<unknown>'}: first bad index {first}")
    return out


class ChipThermDataset(Dataset):
    """Lazy PyTorch dataset for ChipTherm encoded benchmark samples."""

    def __init__(
        self,
        index_csv: str | Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        target: TargetName = "residual",
        return_metadata: bool = True,
        graph_root: str | Path | None = None,
        return_graph: bool = True,
        physical_representation: PhysicalRepresentation = "dimensional",
    ) -> None:
        self.index_csv = Path(index_csv).expanduser().resolve()
        self.transform = transform
        self.target = target
        self.return_metadata = return_metadata
        self.graph_root = Path(graph_root).expanduser().resolve() if graph_root is not None else None
        self.return_graph = bool(return_graph)
        self.physical_representation = str(physical_representation)
        if target not in {"residual", "temperature"}:
            raise ValueError("target must be 'residual' or 'temperature'")
        if self.physical_representation not in {"dimensional", "dimensionless_v1", "dimensionless_v2"}:
            raise ValueError(f"unsupported physical_representation: {self.physical_representation}")
        if not self.index_csv.exists():
            raise FileNotFoundError(self.index_csv)
        self.declared_data_root = self._discover_declared_data_root()
        self.rows = self._read_rows(self.index_csv)
        self.channel_names = self._load_channel_names()
        self.metadata_feature_names, self.metadata_feature_rows = self._load_metadata_features()
        self.graph_node_feature_names, self.graph_edge_feature_names = self._load_graph_manifest()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch.is_tensor(index):
            index = int(index.item())
        row = self.rows[index]

        x = self._load_tensor(row["x_path"], expected_ndim=3)
        x = self._apply_physical_representation(row, x)
        temperature = self._load_tensor(self._temperature_path_for_row(row), expected_ndim=2)
        physics_path_value = self._prediction_path_for_row(row)
        physics = self._load_tensor(physics_path_value, expected_ndim=2)
        residual_path_value = self._residual_path_for_row(row)
        residual_path = self._resolve_path(residual_path_value) if residual_path_value else None
        if residual_path is not None and residual_path.exists():
            residual = self._load_tensor(residual_path_value, expected_ndim=2)
        else:
            residual = temperature - physics

        target_tensor = residual if self.target == "residual" else temperature
        sample: dict[str, Any] = {
            "x": x,
            "target": target_tensor,
            "physics": physics,
            "temperature": temperature,
            "residual": residual,
            "ambient_K": torch.tensor(self._ambient_for_row(row), dtype=torch.float32),
            "total_power_W": torch.tensor(self._total_power_for_row(row), dtype=torch.float32),
        }
        physics_v1_path_value = self._physics_v1_path_for_row(row)
        if physics_v1_path_value is not None:
            sample["physics_v1"] = self._load_tensor(physics_v1_path_value, expected_ndim=2)
        metadata_vector = self._metadata_vector_for_row(row)
        if metadata_vector is not None:
            sample["metadata_vector"] = metadata_vector
        if self.return_graph:
            graph = self._graph_for_row(row)
            if graph is not None:
                sample["graph"] = graph
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
            "target_shape": self._array_shape("residual_path" if self.target == "residual" else "__temperature__"),
            "target": self.target,
            "physical_representation": self.physical_representation,
            "mean_hotspot_temperature_K": self._mean("mean_temperature_K"),
            "mean_power_W": self._mean("total_power_W"),
            "mean_chiplet_count": self._mean("num_chiplets"),
            "metadata_features": self.metadata_feature_names,
            "graph_node_features": self.graph_node_feature_names,
            "graph_edge_features": self.graph_edge_feature_names,
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

    def _load_metadata_features(self) -> tuple[list[str], dict[str, dict[str, float]]]:
        table_path = self._find_sidecar("metadata_features.csv")
        manifest_path = self._find_sidecar("metadata_manifest.json")
        if table_path is None or manifest_path is None:
            return [], {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        active_features = [str(name) for name in manifest.get("active_features", [])]
        rows: dict[str, dict[str, float]] = {}
        with table_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            missing = [name for name in active_features if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{table_path} missing active metadata columns: {', '.join(missing)}")
            for row in reader:
                rows[row["sample_uid"]] = {name: float(row[name]) for name in active_features}
        return active_features, rows

    def _find_sidecar(self, name: str) -> Path | None:
        candidates = [self.index_csv.parent / name, self.index_csv.parent.parent / name]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_channel_names(self) -> list[str]:
        manifest_path = self._find_sidecar("feature_manifest.json") or self._find_sidecar("context_manifest.json")
        if manifest_path is None:
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        channel_names = manifest.get("channel_names")
        if isinstance(channel_names, list):
            return [str(name) for name in channel_names]
        context_channels = manifest.get("context_channels")
        if isinstance(context_channels, list):
            base = [
                "power_density_W_per_mm2",
                "occupancy_mask",
                "CPU_mask",
                "GPU_or_NPU_mask",
                "memory_mask",
                "IO_or_ANALOG_or_MEMS_mask",
                "normalized_x_coordinate",
                "normalized_y_coordinate",
            ]
            return [*base, *[str(name) for name in context_channels]]
        return []

    def _apply_physical_representation(self, row: dict[str, str], x: torch.Tensor) -> torch.Tensor:
        if self.physical_representation == "dimensional":
            return x
        if self.physical_representation == "dimensionless_v1":
            return build_dimensionless_v1_input(x, self._channel_index_map(x), sample_uid=row.get("sample_uid", ""))
        return build_dimensionless_v2_input(x, self._channel_index_map(x), sample_uid=row.get("sample_uid", ""))

    def _channel_index_map(self, x: torch.Tensor) -> dict[str, int]:
        names = self.channel_names
        if len(names) < int(x.shape[0]):
            names = [
                "power_density_W_per_mm2",
                "occupancy_mask",
                "CPU_mask",
                "GPU_or_NPU_mask",
                "memory_mask",
                "IO_or_ANALOG_or_MEMS_mask",
                "normalized_x_coordinate",
                "normalized_y_coordinate",
                *[f"channel_{index}" for index in range(8, int(x.shape[0]))],
            ]
        return {name: index for index, name in enumerate(names[: int(x.shape[0])])}

    def _load_graph_manifest(self) -> tuple[list[str], list[str]]:
        manifest_path = self._graph_manifest_path()
        if manifest_path is None or not manifest_path.exists():
            return [], []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        node_names = [str(name) for name in manifest.get("node_feature_names", [])]
        edge_names = [str(name) for name in manifest.get("edge_feature_names", [])]
        return node_names, edge_names

    def _graph_manifest_path(self) -> Path | None:
        candidates = [
            self.index_csv.parent / "graph_manifest.json",
            self.index_csv.parent.parent / "graph_manifest.json",
        ]
        if self.graph_root is not None:
            candidates.insert(0, self.graph_root / "graph_manifest.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

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
            *(([self.declared_data_root / path]) if self.declared_data_root is not None else []),
            self.index_csv.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _discover_declared_data_root(self) -> Path | None:
        current = self.index_csv.parent
        for candidate in (current, *current.parents):
            marker = candidate / DATA_ROOT_MARKER
            if not marker.exists():
                continue
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid declared data-root marker {marker}: {exc}") from exc
            if payload.get("path_semantics") != "relative_to_declared_data_root":
                raise ValueError(f"unsupported data-root path semantics in {marker}")
            return candidate
        return None

    def _metadata(self, row: dict[str, str]) -> dict[str, Any]:
        payload = {
            "sample_uid": row["sample_uid"],
            "original_sample_uid": row.get("original_sample_uid", ""),
            "case_id": row["case_id"],
            "dataset_source": row["dataset_source"],
            "split": row.get("split", ""),
            "num_chiplets": self._optional_int(row.get("num_chiplets"), default=-1),
            "total_power_W": self._optional_float(row.get("total_power_W"), default=0.0),
            "hotspot_runtime_s": self._optional_float(row.get("hotspot_runtime_s"), default=0.0),
            "physics_runtime_s": self._optional_float(row.get("physics_runtime_s"), default=0.0),
            "x_path": row["x_path"],
            "y_path": self._temperature_path_for_row(row),
            "prediction_path": row.get("prediction_path", ""),
            "residual_path": row.get("residual_path", ""),
            "effective_prediction_path": self._prediction_path_for_row(row),
            "effective_residual_path": self._residual_path_for_row(row) or "",
            "ambient_K": self._ambient_for_row(row),
        }
        if row.get("source_superposition_base_path"):
            payload["source_superposition_base_path"] = row["source_superposition_base_path"]
        if self.metadata_feature_names:
            payload["metadata_features"] = {
                name: float(self.metadata_feature_rows[row["sample_uid"]][name])
                for name in self.metadata_feature_names
            }
        graph_path = self._graph_path_for_row(row)
        if graph_path is not None:
            payload["graph_path"] = str(graph_path)
        return payload

    def _metadata_vector_for_row(self, row: dict[str, str]) -> torch.Tensor | None:
        if not self.metadata_feature_names:
            return None
        values = self.metadata_feature_rows.get(row["sample_uid"])
        if values is None:
            raise ValueError(f"metadata_features.csv missing sample_uid {row['sample_uid']}")
        vector = torch.tensor([float(values[name]) for name in self.metadata_feature_names], dtype=torch.float32)
        if not torch.isfinite(vector).all():
            bad_indices = torch.nonzero(~torch.isfinite(vector), as_tuple=False).flatten().tolist()
            bad_names = [self.metadata_feature_names[index] for index in bad_indices]
            raise ValueError(f"metadata_features.csv has non-finite values for {row['sample_uid']}: {bad_names}")
        return vector

    def _ambient_for_row(self, row: dict[str, str]) -> float:
        if self.metadata_feature_rows and "ambient_K" in self.metadata_feature_rows.get(row["sample_uid"], {}):
            return float(self.metadata_feature_rows[row["sample_uid"]]["ambient_K"])
        return 318.15

    def _total_power_for_row(self, row: dict[str, str]) -> float:
        value = row.get("total_power_W", "")
        if value not in {"", None}:
            total_power = float(value)
            if not np.isfinite(total_power):
                raise ValueError(f"row {row.get('sample_uid')} has non-finite total_power_W={value!r}")
            return total_power
        values = self.metadata_feature_rows.get(row["sample_uid"], {}) if self.metadata_feature_rows else {}
        if "total_power_W" in values:
            total_power = float(values["total_power_W"])
            if not np.isfinite(total_power):
                raise ValueError(f"metadata_features.csv has non-finite total_power_W for {row.get('sample_uid')}")
            return total_power
        raise ValueError(f"row {row.get('sample_uid')} is missing total_power_W required for package mean diagnostics")

    def _prediction_path_for_row(self, row: dict[str, str]) -> str:
        mode = row.get("source_base_mode") or row.get("base_mode") or row.get("physics_input_mode")
        if mode == "source_superposition_v1":
            value = row.get("source_superposition_base_path") or row.get("source_base_path")
            if not value:
                raise ValueError(
                    f"row {row.get('sample_uid')} declares source_superposition_v1 but has no source base path"
                )
            return value
        return row["prediction_path"]

    def _temperature_path_for_row(self, row: dict[str, str]) -> str:
        for field in (*V2_TEMPERATURE_PATH_FIELDS, *LEGACY_TEMPERATURE_PATH_FIELDS):
            value = str(row.get(field, "")).strip()
            if value:
                return value
        available = sorted(row.keys())
        expected = [*V2_TEMPERATURE_PATH_FIELDS, *LEGACY_TEMPERATURE_PATH_FIELDS]
        raise ValueError(
            f"row {row.get('sample_uid', '<unknown>')} has no temperature target path; "
            f"expected one of {expected}, available columns={available}"
        )

    def _residual_path_for_row(self, row: dict[str, str]) -> str | None:
        mode = row.get("source_base_mode") or row.get("base_mode") or row.get("physics_input_mode")
        if mode == "source_superposition_v1":
            return row.get("source_superposition_residual_path") or row.get("source_base_residual_path")
        return row.get("residual_path")

    def _physics_v1_path_for_row(self, row: dict[str, str]) -> str | None:
        mode = row.get("source_base_mode") or row.get("base_mode") or row.get("physics_input_mode")
        if mode == "source_superposition_v1":
            return row.get("prediction_path") or None
        return None

    def _graph_for_row(self, row: dict[str, str]) -> dict[str, torch.Tensor] | None:
        path = self._graph_path_for_row(row)
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            graph = {
                "node_features": torch.from_numpy(data["node_features"].astype(np.float32, copy=False)),
                "edge_index": torch.from_numpy(data["edge_index"].astype(np.int64, copy=False)),
                "edge_features": torch.from_numpy(data["edge_features"].astype(np.float32, copy=False)),
                "chiplet_rects": torch.from_numpy(data["chiplet_rects"].astype(np.float32, copy=False)),
                "package_size": torch.from_numpy(data["package_size"].astype(np.float32, copy=False)),
            }
        if graph["edge_index"].ndim != 2 or graph["edge_index"].shape[0] != 2:
            raise ValueError(f"{path} edge_index must have shape (2, E)")
        return graph

    def _graph_path_for_row(self, row: dict[str, str]) -> Path | None:
        value = row.get("graph_path")
        if value:
            return self._resolve_path(value)
        root = self.graph_root
        if root is None:
            candidate_root = self.index_csv.parent
            if (candidate_root / "graph_features").exists():
                root = candidate_root
        if root is None:
            return None
        sample_uid = row["sample_uid"]
        case_id = row["case_id"]
        candidates = [
            root / "graph_features" / case_id / f"{sample_uid}_graph.npz",
            root / "graphs" / case_id / f"{sample_uid}_graph.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _array_shape(self, column: str) -> tuple[int, ...]:
        value = self._temperature_path_for_row(self.rows[0]) if column == "__temperature__" else self.rows[0][column]
        path = self._resolve_path(value)
        return tuple(int(size) for size in np.load(path, mmap_mode="r").shape)

    def _mean(self, column: str) -> float:
        values = [self._optional_float(row.get(column)) for row in self.rows]
        numeric = [value for value in values if value is not None]
        if not numeric:
            return float("nan")
        return float(sum(numeric) / len(numeric))

    @staticmethod
    def _optional_float(value: Any, *, default: float | None = None) -> float | None:
        if value is None or value == "":
            return default
        return float(value)

    @staticmethod
    def _optional_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None or value == "":
            return default
        return int(float(value))


def chiptherm_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate ChipTherm samples, including optional variable-sized graph data."""
    graphs = [item.pop("graph", None) for item in batch]
    collated = default_collate(batch)
    if all(graph is None for graph in graphs):
        return collated
    if any(graph is None for graph in graphs):
        raise ValueError("mixed graph/non-graph samples in one batch")
    collated["graph"] = collate_graphs([graph for graph in graphs if graph is not None])
    return collated


def collate_graphs(graphs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    node_features: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    chiplet_rects: list[torch.Tensor] = []
    package_sizes: list[torch.Tensor] = []
    node_batch: list[torch.Tensor] = []
    node_offset = 0
    for graph_index, graph in enumerate(graphs):
        nodes = graph["node_features"]
        edges = graph["edge_features"]
        edge_index = graph["edge_index"].long()
        node_count = int(nodes.shape[0])
        node_features.append(nodes)
        edge_features.append(edges)
        if edge_index.numel() > 0:
            edge_indices.append(edge_index + node_offset)
        chiplet_rects.append(graph["chiplet_rects"])
        package_sizes.append(graph["package_size"].view(1, 2))
        node_batch.append(torch.full((node_count,), graph_index, dtype=torch.long))
        node_offset += node_count
    if edge_indices:
        edge_index_out = torch.cat(edge_indices, dim=1)
    else:
        edge_index_out = torch.empty((2, 0), dtype=torch.long)
    if edge_features and sum(int(edge.shape[0]) for edge in edge_features) > 0:
        edge_features_out = torch.cat(edge_features, dim=0)
    else:
        edge_dim = int(edge_features[0].shape[1]) if edge_features else 0
        edge_features_out = torch.empty((0, edge_dim), dtype=torch.float32)
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": edge_index_out,
        "edge_features": edge_features_out,
        "chiplet_rects": torch.cat(chiplet_rects, dim=0),
        "package_size": torch.cat(package_sizes, dim=0),
        "node_batch": torch.cat(node_batch, dim=0),
        "num_graphs": torch.tensor(len(graphs), dtype=torch.long),
    }
