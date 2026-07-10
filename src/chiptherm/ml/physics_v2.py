from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except Exception:  # pragma: no cover - exercised only when SciPy is unavailable.
    _scipy_gaussian_filter = None


EPSILON = 1.0e-12


@dataclass(frozen=True)
class PhysicsV2Config:
    schema_version: int = 1
    ambient_K: float = 318.15
    sigma_mm: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    ridge_alpha: float = 1.0e-6
    power_density_channel: int = 0
    occupancy_channel: int = 1
    total_power_channel: int = 8
    package_width_channel: int = 9
    package_height_channel: int = 10
    cell_size_x_channel: int = 11
    cell_size_y_channel: int = 12
    kernel_mode: str = "reflect"
    notes: str = (
        "Package-aware physical-mm Gaussian spreading prior. Power density is "
        "converted to cell power before convolution; scalar coefficients are "
        "fit with train-only ridge least squares."
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sigma_mm"] = list(self.sigma_mm)
        return data


@dataclass(frozen=True)
class PhysicsV2Coefficients:
    schema_version: int
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    ambient_K: float
    ridge_alpha: float
    calibration_samples: int
    calibration_cells: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)
        data["coefficients"] = list(self.coefficients)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicsV2Coefficients":
        return cls(
            schema_version=int(data["schema_version"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
            coefficients=tuple(float(value) for value in data["coefficients"]),
            ambient_K=float(data["ambient_K"]),
            ridge_alpha=float(data["ridge_alpha"]),
            calibration_samples=int(data["calibration_samples"]),
            calibration_cells=int(data["calibration_cells"]),
        )


@dataclass(frozen=True)
class PackageGridMetadata:
    total_power_W: float
    package_width_mm: float
    package_height_mm: float
    cell_size_x_mm: float
    cell_size_y_mm: float
    grid_rows: int
    grid_cols: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def feature_names(config: PhysicsV2Config) -> tuple[str, ...]:
    names = ["ambient_offset_K", "global_total_power_W"]
    names.extend(f"gaussian_sigma_{sigma:g}_mm_cell_power_W" for sigma in config.sigma_mm)
    return tuple(names)


def extract_package_grid_metadata(
    x: np.ndarray,
    config: PhysicsV2Config,
    *,
    row_total_power_W: float | None = None,
) -> PackageGridMetadata:
    if x.ndim != 3:
        raise ValueError(f"expected X with shape (C, H, W), got {x.shape}")
    if x.shape[0] <= max(
        config.total_power_channel,
        config.package_width_channel,
        config.package_height_channel,
        config.cell_size_x_channel,
        config.cell_size_y_channel,
    ):
        raise ValueError(
            "physics_v2 requires package_plus_power context channels: "
            "total_power, package width/height, and cell size x/y"
        )

    rows = int(x.shape[1])
    cols = int(x.shape[2])
    total_power = _constant_channel_value(x, config.total_power_channel)
    if row_total_power_W is not None:
        total_power = float(row_total_power_W)
    cell_x = _constant_channel_value(x, config.cell_size_x_channel)
    cell_y = _constant_channel_value(x, config.cell_size_y_channel)
    width = _constant_channel_value(x, config.package_width_channel)
    height = _constant_channel_value(x, config.package_height_channel)
    if width <= 0.0:
        width = cell_x * cols
    if height <= 0.0:
        height = cell_y * rows
    if min(total_power, width, height, cell_x, cell_y) <= 0.0:
        raise ValueError(f"invalid package metadata: total_power={total_power}, width={width}, height={height}, cell=({cell_x}, {cell_y})")
    return PackageGridMetadata(
        total_power_W=float(total_power),
        package_width_mm=float(width),
        package_height_mm=float(height),
        cell_size_x_mm=float(cell_x),
        cell_size_y_mm=float(cell_y),
        grid_rows=rows,
        grid_cols=cols,
    )


def build_feature_stack(
    x: np.ndarray,
    config: PhysicsV2Config,
    *,
    row_total_power_W: float | None = None,
) -> tuple[np.ndarray, PackageGridMetadata]:
    metadata = extract_package_grid_metadata(x, config, row_total_power_W=row_total_power_W)
    power_density = x[config.power_density_channel].astype(np.float32, copy=False)
    occupancy = x[config.occupancy_channel].astype(np.float32, copy=False)
    cell_area_mm2 = metadata.cell_size_x_mm * metadata.cell_size_y_mm
    cell_power = (power_density * occupancy * cell_area_mm2).astype(np.float32, copy=False)

    features = [
        np.ones_like(cell_power, dtype=np.float32),
        np.full_like(cell_power, metadata.total_power_W, dtype=np.float32),
    ]
    for sigma_mm in config.sigma_mm:
        features.append(
            physical_gaussian_blur_cell_power(
                cell_power,
                sigma_mm=float(sigma_mm),
                cell_size_x_mm=metadata.cell_size_x_mm,
                cell_size_y_mm=metadata.cell_size_y_mm,
                mode=config.kernel_mode,
            )
        )
    return np.stack(features, axis=0).astype(np.float32, copy=False), metadata


def predict_temperature_v2(
    x: np.ndarray,
    config: PhysicsV2Config,
    coefficients: PhysicsV2Coefficients,
    *,
    row_total_power_W: float | None = None,
) -> tuple[np.ndarray, PackageGridMetadata]:
    features, metadata = build_feature_stack(x, config, row_total_power_W=row_total_power_W)
    coeff = np.asarray(coefficients.coefficients, dtype=np.float32)
    if coeff.shape[0] != features.shape[0]:
        raise ValueError(f"coefficient count {coeff.shape[0]} does not match feature count {features.shape[0]}")
    temp = float(coefficients.ambient_K) + np.tensordot(coeff, features, axes=(0, 0))
    return temp.astype(np.float32, copy=False), metadata


def fit_coefficients_from_accumulators(
    xtx: np.ndarray,
    xty: np.ndarray,
    *,
    config: PhysicsV2Config,
    calibration_samples: int,
    calibration_cells: int,
) -> PhysicsV2Coefficients:
    if xtx.ndim != 2 or xtx.shape[0] != xtx.shape[1]:
        raise ValueError(f"xtx must be square, got {xtx.shape}")
    if xty.shape != (xtx.shape[0],):
        raise ValueError(f"xty shape {xty.shape} does not match xtx {xtx.shape}")
    ridge = np.eye(xtx.shape[0], dtype=np.float64) * float(config.ridge_alpha)
    ridge[0, 0] = 0.0
    coeff = np.linalg.solve(xtx + ridge, xty)
    return PhysicsV2Coefficients(
        schema_version=1,
        feature_names=feature_names(config),
        coefficients=tuple(float(value) for value in coeff),
        ambient_K=float(config.ambient_K),
        ridge_alpha=float(config.ridge_alpha),
        calibration_samples=int(calibration_samples),
        calibration_cells=int(calibration_cells),
    )


def physical_gaussian_blur_cell_power(
    cell_power: np.ndarray,
    *,
    sigma_mm: float,
    cell_size_x_mm: float,
    cell_size_y_mm: float,
    mode: str,
) -> np.ndarray:
    if sigma_mm <= 0.0:
        return cell_power.astype(np.float32, copy=True)
    sigma_y_cells = max(float(sigma_mm) / max(float(cell_size_y_mm), EPSILON), EPSILON)
    sigma_x_cells = max(float(sigma_mm) / max(float(cell_size_x_mm), EPSILON), EPSILON)
    if _scipy_gaussian_filter is not None:
        return _scipy_gaussian_filter(
            cell_power.astype(np.float32, copy=False),
            sigma=(sigma_y_cells, sigma_x_cells),
            mode=mode,
        ).astype(np.float32, copy=False)
    return _numpy_separable_gaussian_blur(
        cell_power.astype(np.float32, copy=False),
        sigma_y=sigma_y_cells,
        sigma_x=sigma_x_cells,
        mode=mode,
    )


def _numpy_separable_gaussian_blur(image: np.ndarray, *, sigma_y: float, sigma_x: float, mode: str) -> np.ndarray:
    if mode != "reflect":
        raise ValueError("NumPy Gaussian fallback only supports reflect mode")
    y_kernel = _gaussian_kernel_1d(sigma_y)
    x_kernel = _gaussian_kernel_1d(sigma_x)
    blurred = _convolve_axis_reflect(image, y_kernel, axis=0)
    return _convolve_axis_reflect(blurred, x_kernel, axis=1)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= max(float(kernel.sum()), EPSILON)
    return kernel.astype(np.float32, copy=False)


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


def _constant_channel_value(x: np.ndarray, channel: int) -> float:
    values = x[channel]
    value = float(values[0, 0])
    if not np.isfinite(value):
        raise ValueError(f"channel {channel} contains non-finite metadata value")
    return value
