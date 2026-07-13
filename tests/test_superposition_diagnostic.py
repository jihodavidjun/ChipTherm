#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/run_superposition_diagnostic.py"
spec = importlib.util.spec_from_file_location("run_superposition_diagnostic", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["run_superposition_diagnostic"] = mod
spec.loader.exec_module(mod)


def test_ambient_added_once() -> None:
    baseline = np.full((2, 2), 300.0)
    source_a = baseline + np.array([[1.0, 2.0], [3.0, 4.0]])
    source_b = baseline + np.array([[10.0, 20.0], [30.0, 40.0]])
    reconstructed = mod.reconstruct_from_isolated(baseline, [source_a, source_b])
    expected = baseline + (source_a - baseline) + (source_b - baseline)
    assert np.array_equal(reconstructed, expected)
    assert float(reconstructed.mean()) == 327.5


def test_synthetic_source_fields_reconstruct_exactly() -> None:
    baseline = np.full((3, 3), 318.15)
    deltas = [np.eye(3), np.fliplr(np.eye(3)) * 2.0, np.ones((3, 3)) * 0.5]
    isolated = [baseline + delta for delta in deltas]
    full = baseline + sum(deltas)
    reconstructed = mod.reconstruct_from_isolated(baseline, isolated)
    assert np.allclose(reconstructed, full)


def test_sample_selection_is_deterministic() -> None:
    rows = [
        {"case_id": "case01", "sample_uid": f"case01_sample_{idx:03d}"}
        for idx in range(8)
    ] + [
        {"case_id": "case02", "sample_uid": f"case02_sample_{idx:03d}"}
        for idx in range(8)
    ]
    first = mod.select_rows(rows, sample_uids=None, cases=["case01", "case02"], samples_per_case=3, seed=7)
    second = mod.select_rows(rows, sample_uids=None, cases=["case01", "case02"], samples_per_case=3, seed=7)
    assert [row["sample_uid"] for row in first] == [row["sample_uid"] for row in second]


def test_zeroing_non_source_powers_is_correct_and_non_mutating() -> None:
    powers = {"A": 10.0, "B": 20.0, "C": 30.0}
    original = copy.deepcopy(powers)
    isolated = mod.isolated_power_map(powers, "B", scale=1.5)
    assert powers == original
    assert isolated == {"A": 0.0, "B": 30.0, "C": 0.0}
    assert mod.zero_power_map(powers) == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_modified_power_yaml_sets_active_workload_and_chiplets() -> None:
    original = {
        "schema_version": 1,
        "units": {"power": "W"},
        "active_workload": "nominal",
        "workloads": {"nominal": {"A": 1.0, "B": 2.0}, "peak": {"A": 3.0, "B": 4.0}},
        "chiplets": {"A": 1.0, "B": 2.0},
    }
    modified = mod.modified_power_yaml(original, {"A": 0.0, "B": 7.0})
    assert original["workloads"]["nominal"]["B"] == 2.0
    assert modified["active_workload"] == "nominal"
    assert modified["chiplets"] == {"A": 0.0, "B": 7.0}
    assert modified["workloads"]["nominal"] == {"A": 0.0, "B": 7.0}
    assert modified["workloads"]["peak"] == {"A": 3.0, "B": 4.0}


def test_isolated_source_files_preserve_original_and_nominal_power(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    destination = tmp_path / "diagnostic_source"
    source_dir.mkdir()
    for name in ("scenario.yaml", "layout.json", "package.yaml", "hotspot.yaml"):
        (source_dir / name).write_text(f"{name}: original\n", encoding="utf-8")
    original_power = {
        "schema_version": 1,
        "units": {"power": "W"},
        "mode": "fixed",
        "active_workload": "nominal",
        "workloads": {"nominal": {"CPU0": 50.0, "GPU0": 125.0, "IO0": 5.0}},
        "chiplets": {"CPU0": 50.0, "GPU0": 125.0, "IO0": 5.0},
    }
    (source_dir / "power.yaml").write_text(yaml.safe_dump(original_power, sort_keys=False), encoding="utf-8")
    original_text = (source_dir / "power.yaml").read_text(encoding="utf-8")

    isolated = mod.isolated_power_map(original_power["chiplets"], "GPU0")
    modified = mod.modified_power_yaml(original_power, isolated)
    mod.copy_source_with_power(source_dir, destination, modified)

    assert (source_dir / "power.yaml").read_text(encoding="utf-8") == original_text
    written = yaml.safe_load((destination / "power.yaml").read_text(encoding="utf-8"))
    assert written["active_workload"] == "nominal"
    assert written["chiplets"]["GPU0"] == 125.0
    assert written["chiplets"]["CPU0"] == 0.0
    assert written["chiplets"]["IO0"] == 0.0
    assert written["workloads"]["nominal"] == written["chiplets"]


def test_resume_valid_output_detection(tmp_path: Path) -> None:
    temp_path = tmp_path / "parsed/temp_layer0.npy"
    temp_path.parent.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    np.save(temp_path, np.ones((2, 2), dtype=np.float32))
    manifest_path.write_text('{"return_code": 0, "runtime": {"hotspot_s": 1.25}}')
    assert mod.valid_run_output(temp_path, manifest_path, (2, 2))
    assert mod.runtime_from_manifest(manifest_path) == 1.25
    temp_path.unlink()
    np.save(temp_path, np.ones((2, 3), dtype=np.float32))
    assert not mod.valid_run_output(temp_path, manifest_path, (2, 2))


def test_field_metrics_match_manual_values() -> None:
    pred = np.array([[1.0, 3.0], [5.0, 7.0]])
    target = np.array([[0.0, 4.0], [6.0, 10.0]])
    metrics = mod.field_metrics(pred, target)
    assert metrics["mae_K"] == 1.5
    assert metrics["max_abs_error_K"] == 3.0
    assert metrics["mean_signed_error_K"] == -1.0


def test_boundary_mask_marks_both_sides() -> None:
    occupancy = np.array([[False, False, False], [False, True, False], [False, False, False]])
    boundary = mod.boundary_mask(occupancy)
    assert bool(boundary[1, 1])
    assert bool(boundary[0, 1])
    assert bool(boundary[1, 0])
    assert bool(boundary[1, 2])
    assert bool(boundary[2, 1])


def test_chiplet_rectangle_metrics() -> None:
    layout = {
        "package": {"size": {"width": 2.0, "height": 2.0}},
        "chiplets": [
            {"name": "A", "position": {"x": 0.0, "y": 0.0}, "size": {"width": 1.0, "height": 2.0}},
            {"name": "B", "position": {"x": 1.0, "y": 0.0}, "size": {"width": 1.0, "height": 2.0}},
        ],
    }
    target = np.array([[10.0, 20.0], [14.0, 24.0]])
    pred = target + np.array([[1.0, -2.0], [3.0, -4.0]])
    metrics = mod.chiplet_metrics(pred, target, layout, target.shape)
    assert metrics["chiplet_mean_temperature_mae_K"] == 2.5
    assert metrics["chiplet_peak_temperature_mae_K"] == 3.5


def test_power_scaling_comparison_arithmetic() -> None:
    baseline = np.full((2, 2), 300.0)
    reference = baseline + np.array([[2.0, 4.0], [6.0, 8.0]])
    scaled = baseline + 1.5 * (reference - baseline)
    metrics = mod.field_metrics(scaled - baseline, 1.5 * (reference - baseline))
    assert metrics["mae_K"] == 0.0
    assert metrics["max_abs_error_K"] == 0.0


def test_output_safety_rejects_canonical_dataset_root() -> None:
    unsafe = REPO_ROOT / "data/runs/benchmarks/superposition_bad"
    try:
        mod.assert_safe_output_dir(unsafe)
    except ValueError:
        return
    raise AssertionError("canonical benchmark output path should be rejected")


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if name in {"test_resume_valid_output_detection", "test_isolated_source_files_preserve_original_and_nominal_power"}:
                import tempfile

                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
    print("superposition diagnostic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
