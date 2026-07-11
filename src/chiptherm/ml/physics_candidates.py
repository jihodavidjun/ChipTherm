from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

try:
    from scipy.fft import dctn as _scipy_dctn
    from scipy.fft import idctn as _scipy_idctn
except Exception:  # pragma: no cover - used only when SciPy is unavailable.
    _scipy_dctn = None
    _scipy_idctn = None


CandidateName = Literal["screened_poisson", "hybrid_local_global", "compact_rc"]
EPSILON = 1.0e-12


@dataclass(frozen=True)
class PhysicsCandidateConfig:
    schema_version: int = 1
    name: CandidateName = "screened_poisson"
    ambient_K: float = 318.15
    power_density_channel: int = 0
    occupancy_channel: int = 1
    total_power_channel: int = 8
    package_width_channel: int = 9
    package_height_channel: int = 10
    cell_size_x_channel: int = 11
    cell_size_y_channel: int = 12
    k_spread_W_per_K: float = 0.30
    g_sink_W_per_mm2K: float = 0.004
    global_R_eff_K_per_W: float = 0.0
    local_kernel_length_mm: float = 1.5
    local_kernel_epsilon_mm: float = 0.75
    local_kernel_gain_K_mm_per_W: float = 0.08
    local_quadrature_size: int = 4
    rc_iterations: int = 120
    rc_relaxation: float = 0.90
    notes: str = (
        "Universal label-free compact thermal priors. Parameters are shared "
        "across package families and are not fit to HotSpot labels."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChipletSource:
    name: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    power_W: float

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    @property
    def power_density_W_per_mm2(self) -> float:
        return self.power_W / self.area_mm2


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


def predict_candidate_temperature(
    x: np.ndarray,
    config: PhysicsCandidateConfig,
    *,
    chiplets: list[ChipletSource] | None = None,
    row_total_power_W: float | None = None,
) -> tuple[np.ndarray, PackageGridMetadata]:
    metadata = extract_package_grid_metadata(x, config, row_total_power_W=row_total_power_W)
    q_W_per_mm2 = power_density_source(x, config)

    if config.name == "screened_poisson":
        rise = screened_poisson_rise(q_W_per_mm2, metadata, config)
    elif config.name == "hybrid_local_global":
        rise = screened_poisson_rise(q_W_per_mm2, metadata, config)
        if chiplets is None:
            raise ValueError("hybrid_local_global requires chiplet geometry metadata")
        rise += finite_source_local_rise(chiplets, metadata, config)
    elif config.name == "compact_rc":
        rise = compact_rc_rise(q_W_per_mm2, metadata, config)
    else:  # pragma: no cover - Literal/CLI should prevent this.
        raise ValueError(f"unsupported physics candidate: {config.name}")

    if config.global_R_eff_K_per_W:
        rise = rise + float(config.global_R_eff_K_per_W) * float(metadata.total_power_W)
    temperature = (float(config.ambient_K) + rise).astype(np.float32, copy=False)
    return temperature, metadata


def extract_package_grid_metadata(
    x: np.ndarray,
    config: PhysicsCandidateConfig,
    *,
    row_total_power_W: float | None = None,
) -> PackageGridMetadata:
    if x.ndim != 3:
        raise ValueError(f"expected X with shape (C,H,W), got {x.shape}")
    required = max(
        config.total_power_channel,
        config.package_width_channel,
        config.package_height_channel,
        config.cell_size_x_channel,
        config.cell_size_y_channel,
    )
    if x.shape[0] <= required:
        raise ValueError(f"candidate physics requires package_plus_power channels through index {required}")
    rows = int(x.shape[1])
    cols = int(x.shape[2])
    total_power = constant_channel_value(x, config.total_power_channel)
    if row_total_power_W is not None:
        total_power = float(row_total_power_W)
    width = constant_channel_value(x, config.package_width_channel)
    height = constant_channel_value(x, config.package_height_channel)
    cell_x = constant_channel_value(x, config.cell_size_x_channel)
    cell_y = constant_channel_value(x, config.cell_size_y_channel)
    if width <= 0.0:
        width = cell_x * cols
    if height <= 0.0:
        height = cell_y * rows
    if min(total_power, width, height, cell_x, cell_y) <= 0.0:
        raise ValueError(
            "invalid package metadata: "
            f"total_power={total_power}, width={width}, height={height}, cell=({cell_x}, {cell_y})"
        )
    return PackageGridMetadata(
        total_power_W=float(total_power),
        package_width_mm=float(width),
        package_height_mm=float(height),
        cell_size_x_mm=float(cell_x),
        cell_size_y_mm=float(cell_y),
        grid_rows=rows,
        grid_cols=cols,
    )


def power_density_source(x: np.ndarray, config: PhysicsCandidateConfig) -> np.ndarray:
    power_density = x[config.power_density_channel].astype(np.float64, copy=False)
    occupancy = x[config.occupancy_channel].astype(np.float64, copy=False)
    source = power_density * occupancy
    if not np.isfinite(source).all():
        raise ValueError("power source contains non-finite values")
    return source


def screened_poisson_rise(
    q_W_per_mm2: np.ndarray,
    metadata: PackageGridMetadata,
    config: PhysicsCandidateConfig,
) -> np.ndarray:
    if _scipy_dctn is None or _scipy_idctn is None:
        return screened_poisson_periodic_fft_rise(q_W_per_mm2, metadata, config)
    q_hat = _scipy_dctn(q_W_per_mm2, type=2, norm="ortho")
    rows, cols = q_W_per_mm2.shape
    mode_y = np.arange(rows, dtype=np.float64)
    mode_x = np.arange(cols, dtype=np.float64)
    lambda_y = (np.pi * mode_y / metadata.package_height_mm) ** 2
    lambda_x = (np.pi * mode_x / metadata.package_width_mm) ** 2
    denom = float(config.g_sink_W_per_mm2K) + float(config.k_spread_W_per_K) * (
        lambda_y[:, None] + lambda_x[None, :]
    )
    rise_hat = q_hat / np.maximum(denom, EPSILON)
    rise = _scipy_idctn(rise_hat, type=2, norm="ortho")
    return rise.astype(np.float32, copy=False)


def screened_poisson_periodic_fft_rise(
    q_W_per_mm2: np.ndarray,
    metadata: PackageGridMetadata,
    config: PhysicsCandidateConfig,
) -> np.ndarray:
    rows, cols = q_W_per_mm2.shape
    q_hat = np.fft.rfft2(q_W_per_mm2)
    ky = 2.0 * np.pi * np.fft.fftfreq(rows, d=metadata.cell_size_y_mm)
    kx = 2.0 * np.pi * np.fft.rfftfreq(cols, d=metadata.cell_size_x_mm)
    denom = float(config.g_sink_W_per_mm2K) + float(config.k_spread_W_per_K) * (
        ky[:, None] * ky[:, None] + kx[None, :] * kx[None, :]
    )
    rise_hat = q_hat / np.maximum(denom, EPSILON)
    rise = np.fft.irfft2(rise_hat, s=q_W_per_mm2.shape)
    return rise.astype(np.float32, copy=False)


def compact_rc_rise(
    q_W_per_mm2: np.ndarray,
    metadata: PackageGridMetadata,
    config: PhysicsCandidateConfig,
) -> np.ndarray:
    dx = max(float(metadata.cell_size_x_mm), EPSILON)
    dy = max(float(metadata.cell_size_y_mm), EPSILON)
    kx = float(config.k_spread_W_per_K) / (dx * dx)
    ky = float(config.k_spread_W_per_K) / (dy * dy)
    g = float(config.g_sink_W_per_mm2K)
    denominator = max(g + 2.0 * kx + 2.0 * ky, EPSILON)
    t = np.zeros_like(q_W_per_mm2, dtype=np.float64)
    omega = float(config.rc_relaxation)
    for _ in range(int(config.rc_iterations)):
        padded = np.pad(t, ((1, 1), (1, 1)), mode="edge")
        candidate = (
            q_W_per_mm2
            + kx * (padded[1:-1, :-2] + padded[1:-1, 2:])
            + ky * (padded[:-2, 1:-1] + padded[2:, 1:-1])
        ) / denominator
        t = (1.0 - omega) * t + omega * candidate
    return t.astype(np.float32, copy=False)


def finite_source_local_rise(
    chiplets: list[ChipletSource],
    metadata: PackageGridMetadata,
    config: PhysicsCandidateConfig,
) -> np.ndarray:
    x_centers = (np.arange(metadata.grid_cols, dtype=np.float64) + 0.5) * metadata.cell_size_x_mm
    y_centers = (np.arange(metadata.grid_rows, dtype=np.float64) + 0.5) * metadata.cell_size_y_mm
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    rise = np.zeros((metadata.grid_rows, metadata.grid_cols), dtype=np.float64)
    q_offsets = (np.arange(int(config.local_quadrature_size), dtype=np.float64) + 0.5) / int(config.local_quadrature_size)
    length = float(config.local_kernel_length_mm)
    epsilon = float(config.local_kernel_epsilon_mm)
    gain = float(config.local_kernel_gain_K_mm_per_W)
    for chiplet in chiplets:
        qx = chiplet.x_mm + q_offsets * chiplet.width_mm
        qy = chiplet.y_mm + q_offsets * chiplet.height_mm
        source_x, source_y = np.meshgrid(qx, qy)
        points = np.column_stack([source_x.reshape(-1), source_y.reshape(-1)])
        weight_W = chiplet.power_W / float(points.shape[0])
        for point_x, point_y in points:
            radius = np.sqrt((grid_x - point_x) ** 2 + (grid_y - point_y) ** 2)
            rise += gain * weight_W * np.exp(-radius / max(length, EPSILON)) / np.sqrt(radius * radius + epsilon * epsilon)
    return rise.astype(np.float32, copy=False)


def constant_channel_value(x: np.ndarray, channel: int) -> float:
    value = float(x[channel, 0, 0])
    if not np.isfinite(value):
        raise ValueError(f"channel {channel} contains non-finite metadata value")
    return value
