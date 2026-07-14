#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_source_response_calibration import (  # noqa: E402
    PackagePrediction,
    analyze_test_packages,
    apply_gain,
    apply_gain_offset,
    apply_offset,
    fit_all_global_calibrations,
    fit_gain,
    fit_gain_offset,
    fit_offset,
    fit_offset_sample,
    pred_mean_true_centered,
    regenerate_report,
    true_mean_pred_centered,
    write_outputs,
    write_report,
)


def make_package(delta_pred: np.ndarray, delta_true: np.ndarray, ambient: float = 300.0, case_id: str = "case01", uid: str = "sample0") -> PackagePrediction:
    layout_path = Path("/tmp/nonexistent_layout.json")
    return PackagePrediction(
        sample_uid=uid,
        case_id=case_id,
        ambient_K=ambient,
        pred=ambient + delta_pred.astype(np.float64),
        true=ambient + delta_true.astype(np.float64),
        oracle_source_sum=ambient + delta_true.astype(np.float64),
        layout_path=str(layout_path),
        num_sources=2,
        total_power_W=10.0,
    )


def test_exact_recovery_of_known_scalar_gain() -> None:
    delta_pred = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    package = make_package(delta_pred, 2.5 * delta_pred)
    params = fit_gain([package])
    assert abs(params["a"] - 2.5) < 1.0e-10
    assert np.allclose(apply_gain(package, params), package.true)


def test_exact_recovery_of_known_gain_and_offset() -> None:
    delta_pred = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    package = make_package(delta_pred, 1.7 * delta_pred + 4.2)
    params = fit_gain_offset([package])
    assert abs(params["a"] - 1.7) < 1.0e-10
    assert abs(params["b_K"] - 4.2) < 1.0e-10
    assert np.allclose(apply_gain_offset(package, params), package.true)


def test_mean_bias_removal_and_ambient_not_scaled() -> None:
    delta_pred = np.ones((4, 4)) * 5.0
    package = make_package(delta_pred, delta_pred + 3.0, ambient=321.0)
    params = fit_offset([package])
    assert abs(params["b_K"] - 3.0) < 1.0e-12
    corrected = apply_offset(package, params)
    assert np.allclose(corrected, package.true)
    gain_corrected = apply_gain(package, {"a": 2.0})
    assert np.allclose(gain_corrected, 321.0 + 2.0 * delta_pred)


def test_validation_test_separation_and_per_case_fitting() -> None:
    val = [
        make_package(np.ones((2, 2)) * 2.0, np.ones((2, 2)) * 4.0, case_id="case01", uid="v1"),
        make_package(np.ones((2, 2)) * 3.0, np.ones((2, 2)) * 9.0, case_id="case02", uid="v2"),
    ]
    test = [
        make_package(np.ones((2, 2)) * 5.0, np.ones((2, 2)) * 10.0, case_id="case01", uid="t1"),
        make_package(np.ones((2, 2)) * 7.0, np.ones((2, 2)) * 21.0, case_id="case02", uid="t2"),
    ]
    global_params = fit_all_global_calibrations(val)
    from scripts.analyze_source_response_calibration import fit_case_calibrations  # noqa: E402

    case_params = fit_case_calibrations(val)
    analysis = analyze_test_packages(test, global_params, case_params)
    by_sample = {row["sample_uid"]: row for row in analysis["parameter_records"]}
    assert by_sample["t1"]["per_case_val_gain_a"] == 2.0
    assert by_sample["t2"]["per_case_val_gain_a"] == 3.0
    assert by_sample["t1"]["oracle_gain_a"] == 2.0
    assert by_sample["t2"]["oracle_gain_a"] == 3.0


def test_degenerate_zero_rise_handling() -> None:
    package = make_package(np.zeros((3, 3)), np.ones((3, 3)))
    params = fit_gain([package])
    assert params["a"] == 1.0
    assert params["fallback"] == "zero_denominator"


def test_mean_centered_decomposition_identities() -> None:
    delta_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    delta_true = np.array([[2.0, 4.0], [6.0, 8.0]])
    package = make_package(delta_pred, delta_true)
    true_mean = true_mean_pred_centered(package)
    pred_mean = pred_mean_true_centered(package)
    assert np.isclose(np.mean(true_mean - package.ambient_K), np.mean(package.delta_true))
    assert np.allclose(pred_mean - package.ambient_K - np.mean(package.delta_pred), package.delta_true - np.mean(package.delta_true))


def test_true_mean_and_true_centered_oracles() -> None:
    delta_pred = np.array([[1.0, 1.0], [1.0, 5.0]])
    delta_true = np.array([[2.0, 2.0], [2.0, 6.0]])
    package = make_package(delta_pred, delta_true)
    assert np.allclose(true_mean_pred_centered(package), package.true)
    assert np.allclose(pred_mean_true_centered(package), package.pred)


def test_stable_json_csv_report_output(tmp_path: Path) -> None:
    package = make_package(np.ones((2, 2)), np.ones((2, 2)) * 2.0)
    params = fit_all_global_calibrations([package])
    analysis = analyze_test_packages([package], params, {"case01": params})
    analysis["metadata"] = {"schema_version": 1}
    analysis["validation_fitted_parameters"] = {"global": params, "per_case": {"case01": params}}
    write_outputs(tmp_path, analysis)
    write_report(tmp_path, analysis)
    assert (tmp_path / "calibration_summary.json").exists()
    assert (tmp_path / "calibration_metrics_by_sample.csv").exists()
    assert (tmp_path / "calibration_report.md").exists()
    regenerate_report(tmp_path)
    summary = json.loads((tmp_path / "calibration_summary.json").read_text())
    assert "overall_metrics" in summary


def main() -> int:
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
    print("source response calibration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
