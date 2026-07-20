from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_spatial_errors import (
    DEFAULT_DISTANCE_BINS,
    AuditResult,
    build_region_masks,
    distance_maps,
    update_distance_bins,
    update_frequency_stats,
    validate_masks,
)


def test_region_masks_cover_grid_and_are_finite() -> None:
    occupancy = np.zeros((8, 8), dtype=bool)
    occupancy[2:6, 3:5] = True
    target = np.arange(64, dtype=np.float64).reshape(8, 8)
    pred = target + 0.5

    masks = build_region_masks(
        occupancy,
        target,
        pred,
        boundary_width=1,
        edge_width=1,
        hotspot_radius=1,
        gradient_thresholds=[1.0, 4.0, 8.0],
    )
    validate_masks(masks, target.shape)

    assert np.all(masks.independent["occupied"] | masks.independent["unoccupied"])
    assert masks.independent["chiplet_boundary_band"].any()
    assert masks.independent["chiplet_interior"].any()
    partition_cover = np.zeros_like(occupancy)
    for mask in masks.partition.values():
        partition_cover |= mask
    assert np.all(partition_cover)

    distances = distance_maps(masks, occupancy)
    for name, dist in distances.items():
        assert dist.shape == target.shape, name
        assert np.isfinite(dist).all(), name


def test_distance_bin_accumulators_count_pixels() -> None:
    occupancy = np.zeros((8, 8), dtype=bool)
    occupancy[3:5, 3:5] = True
    target = np.zeros((8, 8), dtype=np.float64)
    pred = np.ones((8, 8), dtype=np.float64)
    masks = build_region_masks(occupancy, target, pred, 1, 1, 1, [0.0, 0.1, 0.2])
    dist = distance_maps(masks, occupancy)["nearest_occupied_cell_cells"]
    accs = [AuditResult().global_final for _ in range(len(DEFAULT_DISTANCE_BINS) - 1)]

    update_distance_bins(accs, pred - target, dist, DEFAULT_DISTANCE_BINS)

    assert sum(acc.count for acc in accs) == target.size
    assert math.isclose(sum(acc.sum_abs for acc in accs), float(target.size))


def test_frequency_update_records_energy() -> None:
    result = AuditResult()
    result.frequency_radial_energy = np.zeros(64, dtype=np.float64)
    result.frequency_radial_count = np.zeros(64, dtype=np.float64)
    rr, cc = np.indices((8, 8))
    centered_error = np.sin(2.0 * np.pi * rr / 8.0) + 0.25 * np.sin(2.0 * np.pi * cc / 2.0)
    radial_bins = np.linspace(0.0, 0.5 * math.sqrt(2.0), 65)

    update_frequency_stats(result, centered_error, [0.10, 0.25], radial_bins)

    assert result.frequency_energy["low_frequency"] >= 0.0
    assert result.frequency_energy["mid_frequency"] >= 0.0
    assert result.frequency_energy["high_frequency"] >= 0.0
    assert sum(result.frequency_energy.values()) > 0.0
    assert result.frequency_stats["mid_frequency"].count == centered_error.size
