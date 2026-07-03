from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except Exception:  # pragma: no cover - exercised only when SciPy is unavailable.
    _scipy_gaussian_filter = None


@dataclass(frozen=True)
class PhysicsBaselineConfig:
    schema_version: int = 1
    ambient_K: float = 318.15
    global_R_eff_K_per_W: float = 0.03
    sigmas_cells: tuple[float, ...] = (1.5, 4.0, 10.0)
    weights_K_per_W_per_mm2: tuple[float, ...] = (20.0, 35.0, 60.0)
    input_power_channel: int = 0
    input_occupancy_channel: int = 1
    use_occupancy_mask: bool = True
    kernel_mode: str = "reflect"
    y_normalized: bool = False
    notes: str = "Fixed global package-heating plus multi-scale Gaussian heat-spreading baseline; no learned parameters."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sigmas_cells"] = list(self.sigmas_cells)
        data["weights_K_per_W_per_mm2"] = list(self.weights_K_per_W_per_mm2)
        return data


def predict_temperature(
    x: np.ndarray,
    config: PhysicsBaselineConfig | None = None,
    *,
    total_power_W: float | None = None,
) -> np.ndarray:
    config = config or PhysicsBaselineConfig()
    if x.ndim != 3:
        raise ValueError(f"expected X with shape (C, H, W), got {x.shape}")
    if len(config.sigmas_cells) != len(config.weights_K_per_W_per_mm2):
        raise ValueError("sigmas_cells and weights_K_per_W_per_mm2 must have the same length")
    if config.global_R_eff_K_per_W != 0.0 and total_power_W is None:
        raise ValueError("total_power_W is required when global_R_eff_K_per_W is nonzero")

    power = x[config.input_power_channel].astype(np.float32, copy=False)
    if config.use_occupancy_mask:
        power = power * x[config.input_occupancy_channel].astype(np.float32, copy=False)

    temp_rise = np.zeros_like(power, dtype=np.float32)
    if total_power_W is not None:
        temp_rise += float(config.global_R_eff_K_per_W) * float(total_power_W)
    for sigma, weight in zip(config.sigmas_cells, config.weights_K_per_W_per_mm2):
        temp_rise += float(weight) * gaussian_blur(power, float(sigma), mode=config.kernel_mode)
    return (float(config.ambient_K) + temp_rise).astype(np.float32, copy=False)


def gaussian_blur(image: np.ndarray, sigma: float, mode: str = "reflect") -> np.ndarray:
    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    if _scipy_gaussian_filter is not None:
        return _scipy_gaussian_filter(image, sigma=sigma, mode=mode).astype(np.float32, copy=False)
    return _numpy_gaussian_blur(image, sigma=sigma, mode=mode)


def sample_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target shape {target.shape}")
    error = pred.astype(np.float64) - target.astype(np.float64)
    abs_error = np.abs(error)
    pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
    target_hotspot = np.unravel_index(int(np.argmax(target)), target.shape)
    row_error = float(pred_hotspot[0] - target_hotspot[0])
    col_error = float(pred_hotspot[1] - target_hotspot[1])
    return {
        "mae_K": float(abs_error.mean()),
        "rmse_K": float(np.sqrt(np.mean(error * error))),
        "max_abs_error_K": float(abs_error.max()),
        "mean_signed_error_K": float(error.mean()),
        "hotspot_temp_error_K": float(pred[pred_hotspot] - target[target_hotspot]),
        "hotspot_location_error_cells": float(np.hypot(row_error, col_error)),
        "target_hotspot_row": float(target_hotspot[0]),
        "target_hotspot_col": float(target_hotspot[1]),
        "pred_hotspot_row": float(pred_hotspot[0]),
        "pred_hotspot_col": float(pred_hotspot[1]),
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = [
        "mae_K",
        "rmse_K",
        "max_abs_error_K",
        "mean_signed_error_K",
        "hotspot_temp_error_K",
        "hotspot_location_error_cells",
    ]
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in metrics]
        if key == "max_abs_error_K":
            aggregated[key] = float(max(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _numpy_gaussian_blur(image: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    if mode != "reflect":
        raise ValueError("NumPy Gaussian fallback only supports reflect mode")
    radius = max(1, int(round(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    blurred = _convolve_axis_reflect(image.astype(np.float32, copy=False), kernel, axis=0)
    return _convolve_axis_reflect(blurred, kernel, axis=1)


def _convolve_axis_reflect(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for idx, weight in enumerate(kernel):
        start = idx
        stop = start + image.shape[axis]
        if axis == 0:
            out += float(weight) * padded[start:stop, :]
        else:
            out += float(weight) * padded[:, start:stop]
    return out
