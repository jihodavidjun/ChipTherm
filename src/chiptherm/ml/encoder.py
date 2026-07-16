from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CHANNEL_NAMES = [
    "power_density_W_per_mm2",
    "occupancy_mask",
    "CPU_mask",
    "GPU_or_NPU_mask",
    "memory_mask",
    "IO_or_ANALOG_or_MEMS_mask",
    "normalized_x_coordinate",
    "normalized_y_coordinate",
    "total_power_W",
    "package_width_mm",
    "package_height_mm",
    "cell_size_x_mm",
    "cell_size_y_mm",
]

CPU_TYPES = {"CPU"}
GPU_NPU_TYPES = {"GPU", "NPU"}
MEMORY_TYPES = {"HBM", "DRAM"}
IO_MISC_TYPES = {"IO", "ANALOG", "MEMS"}


def encode_sample(
    *,
    layout_path: str | Path,
    power_path: str | Path,
    hotspot_path: str | Path,
    temp_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    layout = _load_json(Path(layout_path))
    power = _load_yaml(Path(power_path))
    hotspot = _load_yaml(Path(hotspot_path))
    rows = int(hotspot.get("grid", {}).get("rows", 64))
    cols = int(hotspot.get("grid", {}).get("cols", 64))
    if rows != 64 or cols != 64:
        raise ValueError(f"v1 encoder expects 64x64 grid, got {rows}x{cols}")

    package = layout.get("package", {})
    size = package.get("size", {})
    width_mm = float(size["width"])
    height_mm = float(size["height"])
    powers = active_power_map(power)

    x = np.zeros((len(CHANNEL_NAMES), rows, cols), dtype=np.float32)
    y_coords = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    x_coords = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    grid_x_mm = x_coords * width_mm
    grid_y_mm = y_coords * height_mm
    x[6] = np.broadcast_to(x_coords.reshape(1, cols), (rows, cols))
    x[7] = np.broadcast_to(y_coords.reshape(rows, 1), (rows, cols))
    total_power_W = float(sum(powers.values()))
    cell_size_x_mm = width_mm / float(cols)
    cell_size_y_mm = height_mm / float(rows)
    x[8] = total_power_W
    x[9] = width_mm
    x[10] = height_mm
    x[11] = cell_size_x_mm
    x[12] = cell_size_y_mm

    chiplet_summaries: list[dict[str, Any]] = []
    for chiplet in layout.get("chiplets", []):
        name = str(chiplet["name"])
        chiplet_type = str(chiplet["type"])
        position = chiplet["position"]
        chiplet_size = chiplet["size"]
        left = float(position["x"])
        bottom = float(position["y"])
        chiplet_width = float(chiplet_size["width"])
        chiplet_height = float(chiplet_size["height"])
        area_mm2 = chiplet_width * chiplet_height
        if area_mm2 <= 0.0:
            raise ValueError(f"{name} has non-positive area")
        if name not in powers:
            raise ValueError(f"missing power for chiplet {name}")

        cols_mask = (grid_x_mm >= left) & (grid_x_mm < left + chiplet_width)
        rows_mask = (grid_y_mm >= bottom) & (grid_y_mm < bottom + chiplet_height)
        mask = np.outer(rows_mask, cols_mask)
        power_density = float(powers[name]) / area_mm2

        x[0, mask] = power_density
        x[1, mask] = 1.0
        type_channel = type_channel_index(chiplet_type)
        x[type_channel, mask] = 1.0
        chiplet_summaries.append(
            {
                "name": name,
                "type": chiplet_type,
                "area_mm2": area_mm2,
                "power_W": float(powers[name]),
                "power_density_W_per_mm2": power_density,
                "covered_cells": int(mask.sum()),
            }
        )

    y = np.load(temp_path).astype(np.float32, copy=False)
    metadata = {
        "channel_names": CHANNEL_NAMES,
        "grid_rows": rows,
        "grid_cols": cols,
        "package_width_mm": width_mm,
        "package_height_mm": height_mm,
        "total_power_W": total_power_W,
        "cell_size_x_mm": cell_size_x_mm,
        "cell_size_y_mm": cell_size_y_mm,
        "active_workload": power.get("active_workload"),
        "chiplets": chiplet_summaries,
    }
    validate_encoded_tensors(x, y)
    return x, y, metadata


def active_power_map(power: dict[str, Any]) -> dict[str, float]:
    active = power.get("active_workload")
    workloads = power.get("workloads")
    if active is not None and isinstance(workloads, dict) and active in workloads:
        return {str(name): float(value) for name, value in workloads[active].items()}
    chiplets = power.get("chiplets")
    if isinstance(chiplets, dict):
        return {str(name): float(value) for name, value in chiplets.items()}
    raise ValueError("power.yaml must contain active workload powers or chiplets")


def type_channel_index(chiplet_type: str) -> int:
    if chiplet_type in CPU_TYPES:
        return 2
    if chiplet_type in GPU_NPU_TYPES:
        return 3
    if chiplet_type in MEMORY_TYPES:
        return 4
    if chiplet_type in IO_MISC_TYPES:
        return 5
    raise ValueError(f"unsupported chiplet type for encoding: {chiplet_type}")


def validate_encoded_tensors(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != (13, 64, 64):
        raise ValueError(f"X shape must be (13, 64, 64), got {x.shape}")
    if y.shape != (64, 64):
        raise ValueError(f"Y shape must be (64, 64), got {y.shape}")
    if x.dtype != np.float32:
        raise ValueError(f"X dtype must be float32, got {x.dtype}")
    if y.dtype != np.float32:
        raise ValueError(f"Y dtype must be float32, got {y.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("X contains non-finite values")
    if not np.isfinite(y).all():
        raise ValueError("Y contains non-finite values")

    occupancy = x[1]
    type_masks = x[2:6]
    for channel in range(1, 6):
        values = x[channel]
        if not np.logical_or(values == 0.0, values == 1.0).all():
            raise ValueError(f"mask channel {channel} is not binary")

    type_sum = type_masks.sum(axis=0)
    occupied = occupancy == 1.0
    if not np.all(type_sum[occupied] == 1.0):
        raise ValueError("each occupied cell must belong to exactly one broad type mask")
    if not np.all(type_sum[~occupied] == 0.0):
        raise ValueError("empty cells must not have type masks")
    if np.any((x[0] != 0.0) & ~occupied):
        raise ValueError("power density is nonzero outside occupancy")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data
