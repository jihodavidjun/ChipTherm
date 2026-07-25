from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_benchmark_v2_residual_decomposition import (
    FAMILY_COLUMNS,
    SAMPLE_COLUMNS,
    aggregate_families,
    build_summary,
    cached_prediction_path,
    decompose_sample,
    write_csv,
    write_plots,
    write_report,
)


class BenchmarkV2ResidualDecompositionTests(unittest.TestCase):
    def test_exact_mean_centered_and_final_decomposition(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        base = np.full((64, 64), 320.0, dtype=np.float32)
        true_centered = ((xx - xx.mean()) / 32.0 + (yy - yy.mean()) / 64.0).astype(np.float32)
        predicted_centered = 0.75 * true_centered
        target = base + 2.0 + true_centered
        prediction = base + 1.5 + predicted_centered
        row = {
            "sample_uid": "f044_w001",
            "family_uid": "f044",
            "case_id": "f044",
            "split": "test",
            "workload_uid": "w001",
            "broad_stratum": "high",
            "power_regime": "high",
            "topology_regime": "sparse",
        }
        record = decompose_sample(
            row=row,
            protocol="heldout_test",
            target=target,
            source=base,
            prediction=prediction,
            prediction_source="saved_prediction",
        )
        self.assertAlmostEqual(record["true_residual_mean_K"], 2.0, places=6)
        self.assertAlmostEqual(record["predicted_scalar_mean_correction_K"], 1.5, places=6)
        self.assertAlmostEqual(record["absolute_mean_correction_error_K"], 0.5, places=6)
        expected_centered_mae = float(np.mean(np.abs(predicted_centered - true_centered)))
        self.assertAlmostEqual(record["centered_spatial_mae_K"], expected_centered_mae, places=6)
        self.assertAlmostEqual(
            record["final_cnn_mae_K"],
            float(np.mean(np.abs(prediction.astype(np.float64) - target.astype(np.float64)))),
            places=6,
        )
        self.assertTrue(record["cnn_worse_than_source_baseline"] is False)

    def test_required_outputs_and_f044_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            records = [
                self._record("heldout_validation", "f007", "low", 2.0, 1.0),
                self._record("heldout_test", "f044", "high", 3.0, 1.5),
                self._record("heldout_test", "f044", "stress", 1.0, 1.2),
            ]
            family_rows = aggregate_families(records)
            summary = build_summary(
                records,
                checkpoint_path=checkpoint,
                source_version="final_train40_source_v1",
                indices={"heldout_validation": root / "val.csv", "heldout_test": root / "test.csv"},
                prediction_roots={"heldout_validation": root / "val", "heldout_test": root / "test"},
                cache_counts={"saved_prediction": 3, "checkpoint_inference": 0},
            )
            self.assertEqual(summary["f044"]["num_samples"], 2)
            self.assertAlmostEqual(summary["f044"]["source_superposition_mae_K"], 2.0)
            self.assertEqual(set(summary["by_power_regime"]), {"high", "low", "stress"})

            write_csv(root / "per_sample_decomposition.csv", records, SAMPLE_COLUMNS)
            write_csv(root / "per_family_decomposition.csv", family_rows, FAMILY_COLUMNS)
            write_report(root / "residual_decomposition_report.md", summary, family_rows)
            write_plots(root, family_rows, records)
            for name in (
                "per_sample_decomposition.csv",
                "per_family_decomposition.csv",
                "residual_decomposition_report.md",
                "error_components_by_family.png",
                "cnn_improvement_by_family.png",
                "source_vs_final_by_power_regime.png",
            ):
                self.assertTrue((root / name).is_file(), name)
            with (root / "per_sample_decomposition.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_cached_prediction_path_supports_evaluator_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {"sample_uid": "f044_w001", "family_uid": "f044"}
            expected = root / "f044/f044_w001_tpred.npy"
            expected.parent.mkdir(parents=True)
            np.save(expected, np.zeros((64, 64), dtype=np.float32))
            self.assertEqual(cached_prediction_path(root, row), expected)

    @staticmethod
    def _record(
        protocol: str,
        family: str,
        regime: str,
        source_mae: float,
        final_mae: float,
    ) -> dict[str, object]:
        return {
            "protocol": protocol,
            "split": "val" if protocol == "heldout_validation" else "test",
            "sample_uid": f"{family}_{regime}",
            "family_uid": family,
            "case_id": family,
            "workload_uid": regime,
            "workload_regime": regime,
            "workload_cell": regime,
            "workload_stratum": regime,
            "broad_stratum": regime,
            "power_regime": regime,
            "topology_regime": "fixture",
            "source_superposition_mae_K": source_mae,
            "final_cnn_mae_K": final_mae,
            "cnn_improvement_K": source_mae - final_mae,
            "true_residual_mean_K": 0.5,
            "predicted_scalar_mean_correction_K": 0.4,
            "mean_correction_error_K": -0.1,
            "absolute_mean_correction_error_K": 0.1,
            "true_centered_spatial_residual_abs_mean_K": 1.0,
            "true_centered_spatial_residual_rms_K": 1.2,
            "predicted_centered_spatial_correction_abs_mean_K": 0.9,
            "predicted_centered_spatial_correction_rms_K": 1.1,
            "centered_spatial_mae_K": 0.3,
            "centered_spatial_rmse_K": 0.4,
            "peak_temperature_error_K": -0.2,
            "peak_temperature_abs_error_K": 0.2,
            "cnn_worse_than_source_baseline": final_mae > source_mae,
            "prediction_source": "saved_prediction",
        }


if __name__ == "__main__":
    unittest.main()
