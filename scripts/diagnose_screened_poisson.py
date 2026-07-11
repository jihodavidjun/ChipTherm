#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.physics_candidates import (  # noqa: E402
    PackageGridMetadata,
    PhysicsCandidateConfig,
    screened_poisson_rise,
)

try:
    from scipy.fft import dctn, idctn
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scipy.fft is required for diagnostics: {exc}")


def main() -> int:
    config = PhysicsCandidateConfig(k_spread_W_per_K=0.30, g_sink_W_per_mm2K=0.004)
    dct_round_trip_test()
    uniform_source_test(config)
    energy_mean_mode_test(config)
    impulse_symmetry_test(config)
    package_scaling_test(config)
    direct_solve_comparison_test(config)
    print("screened-Poisson diagnostics passed")
    return 0


def metadata(rows: int, cols: int, width_mm: float, height_mm: float) -> PackageGridMetadata:
    return PackageGridMetadata(
        total_power_W=1.0,
        package_width_mm=width_mm,
        package_height_mm=height_mm,
        cell_size_x_mm=width_mm / cols,
        cell_size_y_mm=height_mm / rows,
        grid_rows=rows,
        grid_cols=cols,
    )


def dct_round_trip_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(17, 19))
    x_back = idctn(dctn(x, type=2, norm="ortho"), type=2, norm="ortho")
    assert np.allclose(x, x_back, atol=1.0e-11), "DCT-II orthonormal round trip failed"


def uniform_source_test(config: PhysicsCandidateConfig) -> None:
    meta = metadata(32, 40, 45.0, 35.0)
    q_value = 0.25
    q = np.full((meta.grid_rows, meta.grid_cols), q_value, dtype=np.float64)
    rise = screened_poisson_rise(q, meta, config).astype(np.float64)
    expected = q_value / config.g_sink_W_per_mm2K
    assert np.allclose(rise, expected, rtol=1.0e-5, atol=1.0e-4), "uniform source is not spatially constant q/g"


def energy_mean_mode_test(config: PhysicsCandidateConfig) -> None:
    rng = np.random.default_rng(1)
    meta = metadata(31, 37, 50.0, 41.0)
    q = rng.random((meta.grid_rows, meta.grid_cols))
    rise = screened_poisson_rise(q, meta, config).astype(np.float64)
    expected_mean = float(q.mean()) / config.g_sink_W_per_mm2K
    assert abs(float(rise.mean()) - expected_mean) < 1.0e-4, "zero-mode mean relation failed"


def impulse_symmetry_test(config: PhysicsCandidateConfig) -> None:
    meta = metadata(33, 33, 33.0, 33.0)
    q = np.zeros((33, 33), dtype=np.float64)
    q[16, 16] = 1.0
    rise = screened_poisson_rise(q, meta, config).astype(np.float64)
    peak = np.unravel_index(int(np.argmax(rise)), rise.shape)
    assert peak == (16, 16), f"impulse peak is {peak}, expected source location"
    assert np.allclose(rise, np.flipud(rise), atol=1.0e-5), "centered impulse response is not y-symmetric"
    assert np.allclose(rise, np.fliplr(rise), atol=1.0e-5), "centered impulse response is not x-symmetric"
    assert rise[16, 16] > rise[16, 20] > rise[16, 28], "response does not decay away from centered source"


def package_scaling_test(config: PhysicsCandidateConfig) -> None:
    q_value = 0.12
    for rows, cols, width, height in ((32, 32, 32.0, 32.0), (64, 64, 64.0, 64.0)):
        meta = metadata(rows, cols, width, height)
        q = np.full((rows, cols), q_value, dtype=np.float64)
        rise = screened_poisson_rise(q, meta, config).astype(np.float64)
        expected = q_value / config.g_sink_W_per_mm2K
        assert abs(float(rise.mean()) - expected) < 1.0e-4, "constant power-density scaling failed"
        assert float(rise.std()) < 1.0e-5, "constant source produced spatial variation under Neumann BC"


def direct_solve_comparison_test(config: PhysicsCandidateConfig) -> None:
    rng = np.random.default_rng(2)
    rows, cols = 6, 7
    meta = metadata(rows, cols, 12.0, 10.5)
    q = rng.random((rows, cols))
    spectral = screened_poisson_rise(q, meta, config).astype(np.float64)
    direct = direct_fd_solution(q, meta, config)
    assert np.allclose(spectral, direct, rtol=2.0e-6, atol=2.0e-6), "DCT solution does not match direct FD solve"


def direct_fd_solution(q: np.ndarray, meta: PackageGridMetadata, config: PhysicsCandidateConfig) -> np.ndarray:
    rows, cols = q.shape
    dx = meta.cell_size_x_mm
    dy = meta.cell_size_y_mm
    neg_lap = np.zeros((rows * cols, rows * cols), dtype=np.float64)

    def idx(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            i = idx(row, col)
            if col > 0:
                neg_lap[i, i] += 1.0 / (dx * dx)
                neg_lap[i, idx(row, col - 1)] -= 1.0 / (dx * dx)
            if col < cols - 1:
                neg_lap[i, i] += 1.0 / (dx * dx)
                neg_lap[i, idx(row, col + 1)] -= 1.0 / (dx * dx)
            if row > 0:
                neg_lap[i, i] += 1.0 / (dy * dy)
                neg_lap[i, idx(row - 1, col)] -= 1.0 / (dy * dy)
            if row < rows - 1:
                neg_lap[i, i] += 1.0 / (dy * dy)
                neg_lap[i, idx(row + 1, col)] -= 1.0 / (dy * dy)
    matrix = config.g_sink_W_per_mm2K * np.eye(rows * cols) + config.k_spread_W_per_K * neg_lap
    solution = np.linalg.solve(matrix, q.reshape(-1))
    return solution.reshape(rows, cols)


if __name__ == "__main__":
    raise SystemExit(main())
