from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_benchmark_v2_oracle_residual_components import (  # noqa: E402
    analyze_sample,
    decompose_centered,
    make_boundary_mask,
    reconstruct_oracles,
)


def main() -> None:
    test_oracle_residual_component_identities()
    print("benchmark v2 oracle residual-component tests passed")


def test_oracle_residual_component_identities() -> None:
    y, x = np.mgrid[0:64, 0:64].astype(np.float64)
    source = 320.0 + 0.02 * x + 0.01 * y
    true_centered_seed = (
        2.0 * np.sin(2.0 * np.pi * x / 64.0)
        + 0.7 * np.cos(2.0 * np.pi * y / 8.0)
    )
    true_centered = true_centered_seed - true_centered_seed.mean()
    predicted_centered_seed = (
        1.6 * np.sin(2.0 * np.pi * x / 64.0)
        + 0.4 * np.cos(2.0 * np.pi * y / 8.0)
    )
    predicted_centered = predicted_centered_seed - predicted_centered_seed.mean()
    true_mean = 3.5
    predicted_mean = 2.25
    target = source + true_mean + true_centered
    final_prediction = source + predicted_mean + predicted_centered
    boundary = make_boundary_mask((64, 64), 4)

    true_low, true_high = decompose_centered(true_centered, 8)
    predicted_low, predicted_high = decompose_centered(predicted_centered, 8)
    assert abs(float(true_low.mean())) < 1.0e-12
    assert abs(float(true_high.mean())) < 1.0e-12
    assert abs(float(predicted_low.mean())) < 1.0e-12
    assert abs(float(predicted_high.mean())) < 1.0e-12
    assert np.allclose(true_low + true_high, true_centered, atol=1.0e-12, rtol=0.0)
    assert np.allclose(
        predicted_low + predicted_high,
        predicted_centered,
        atol=1.0e-12,
        rtol=0.0,
    )

    reconstructions = reconstruct_oracles(
        source=source,
        true_mean=true_mean,
        predicted_mean=predicted_mean,
        true_low=true_low,
        true_high=true_high,
        predicted_low=predicted_low,
        predicted_high=predicted_high,
        optimal_alpha=1.0,
    )
    assert np.allclose(reconstructions["baseline_final"], final_prediction, atol=1.0e-12)
    assert np.allclose(
        reconstructions["oracle_mean"],
        final_prediction + (true_mean - predicted_mean),
        atol=1.0e-12,
    )
    assert np.allclose(reconstructions["full_oracle"], target, atol=1.0e-12)

    first = analyze_sample(
        source=source,
        target=target,
        final_prediction=final_prediction,
        boundary_mask=boundary,
        coarse_size=8,
    )
    second = analyze_sample(
        source=source.copy(),
        target=target.copy(),
        final_prediction=final_prediction.copy(),
        boundary_mask=boundary.copy(),
        coarse_size=8,
    )
    assert np.isclose(first["true_mean_K"], true_mean, atol=1.0e-12)
    assert np.isclose(first["predicted_mean_K"], predicted_mean, atol=1.0e-12)
    assert first["metrics"]["full_oracle"]["temperature_mae_K"] < 1.0e-12
    assert (
        first["metrics"]["oracle_mean"]["temperature_mae_K"]
        < first["metrics"]["baseline_final"]["temperature_mae_K"]
    )
    for variant, metrics in first["metrics"].items():
        for name, value in metrics.items():
            assert np.isfinite(value), (variant, name, value)
            assert np.isclose(value, second["metrics"][variant][name], atol=0.0, rtol=0.0)
    for name, value in first["component_energy"].items():
        assert np.isfinite(value), (name, value)
        assert np.isclose(value, second["component_energy"][name], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    main()
