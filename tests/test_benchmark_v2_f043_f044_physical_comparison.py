from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_benchmark_v2_f043_f044_physical_comparison import (  # noqa: E402
    boundary_contrasts,
    compute_workload_metrics,
    directional_gradient_metrics,
    directional_low_frequency_metrics,
    load_saved_prediction,
    match_workloads,
)


def main() -> None:
    test_matched_physical_decomposition()
    print("benchmark v2 f043/f044 physical comparison test passed")


def test_matched_physical_decomposition() -> None:
    left = [
        {"sample_uid": "f043_w002_high", "family_uid": "f043", "workload_uid": "w002_high"},
        {"sample_uid": "f043_w001_low", "family_uid": "f043", "workload_uid": "w001_low"},
    ]
    right = [
        {"sample_uid": "f044_w001_low", "family_uid": "f044", "workload_uid": "w001_low"},
        {"sample_uid": "f044_w002_high", "family_uid": "f044", "workload_uid": "w002_high"},
    ]
    assert match_workloads(left, right, expected_count=2) == ["w001_low", "w002_high"]

    width_mm, height_mm = 64.0, 32.0
    dx_mm, dy_mm = width_mm / 64.0, height_mm / 64.0
    x = (np.arange(64, dtype=np.float64) + 0.5) * dx_mm
    y = (np.arange(64, dtype=np.float64) + 0.5) * dy_mm
    x_map = np.broadcast_to(x[None, :], (64, 64))
    y_map = np.broadcast_to(y[:, None], (64, 64))
    true_residual = 2.0 + 0.5 * x_map + 0.1 * y_map
    predicted_residual = 1.5 + 0.4 * x_map + 0.1 * y_map
    source = np.full((64, 64), 320.0, dtype=np.float64)
    target = source + true_residual
    prediction = source + predicted_residual
    geometry = {
        "package_width_mm": width_mm,
        "package_height_mm": height_mm,
        "package_aspect_ratio": 2.0,
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "chiplet_pairwise_center_distance_mm_min": 5.0,
        "chiplet_pairwise_center_distance_mm_mean": 10.0,
        "chiplet_rectangle_gap_mm_min": 1.0,
        "chiplet_rectangle_gap_mm_mean": 3.0,
        "chiplet_boundary_distance_mm_min": 0.5,
        "chiplet_boundary_distance_mm_mean": 2.0,
    }
    metrics = compute_workload_metrics(
        family="f044",
        workload_uid="w001_low",
        row={
            "sample_uid": "f044_w001_low",
            "workload_uid": "w001_low",
            "power_regime": "low",
        },
        source=source,
        target=target,
        prediction=prediction,
        geometry=geometry,
        boundary_band_mm=4.0,
    )
    assert np.isclose(metrics["true_residual_mean_K"], true_residual.mean())
    assert np.isclose(metrics["predicted_residual_mean_K"], predicted_residual.mean())
    assert metrics["centered_spatial_mae_K"] > 0.0
    assert np.isclose(
        metrics["predicted_residual_error_mae_K"],
        metrics["final_temperature_mae_K"],
    )
    gradient = directional_gradient_metrics(true_residual, dx_mm, dy_mm)
    assert np.isclose(gradient["x_energy"], 0.25, atol=1.0e-12)
    assert np.isclose(gradient["y_energy"], 0.01, atol=1.0e-12)
    assert gradient["x_y_ratio"] > 20.0
    frequency = directional_low_frequency_metrics(true_residual, width_mm, height_mm)
    assert np.isfinite(list(frequency.values())).all()
    contrast = boundary_contrasts(
        true_residual,
        width_mm=width_mm,
        height_mm=height_mm,
        boundary_band_mm=4.0,
    )
    assert all(np.isfinite(value) for value in contrast.values())

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        residual_dir = root / "predicted_residuals/f044"
        residual_dir.mkdir(parents=True)
        residual_path = residual_dir / "f044_w001_low_residual_pred.npy"
        np.save(residual_path, predicted_residual.astype(np.float32))
        loaded, path, kind = load_saved_prediction(
            prediction_root=root,
            family="f044",
            sample_uid="f044_w001_low",
            source=source,
        )
        assert kind == "predicted_residual"
        assert path == residual_path.resolve()
        assert np.allclose(loaded, prediction, atol=1.0e-6)


if __name__ == "__main__":
    main()
