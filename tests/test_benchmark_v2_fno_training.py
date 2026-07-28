#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.fno_models import DirectTemperatureFNO2d
from chiptherm.ml.models import build_model
from chiptherm.ml.normalization import DirectTemperatureTargetStats


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_script("train_residual_cnn_fno_test", REPO_ROOT / "scripts/train_residual_cnn.py")
comparison = load_script(
    "compare_benchmark_v2_fno_models_test",
    REPO_ROOT / "scripts/compare_benchmark_v2_fno_models.py",
)


class BenchmarkV2FNOTrainingTests(unittest.TestCase):
    def test_prediction_mode_validation(self) -> None:
        trainer.validate_prediction_mode(
            "direct_temperature_fno",
            "fno2d_direct_conditioned",
            "none",
        )
        trainer.validate_prediction_mode(
            "residual_decomposed_fno",
            "fno2d_residual_decomposed_conditioned",
            "source_superposition_v1",
        )
        with self.assertRaises(ValueError):
            trainer.validate_prediction_mode(
                "direct_temperature_fno",
                "fno2d_direct_conditioned",
                "source_superposition_v1",
            )
        with self.assertRaises(ValueError):
            trainer.validate_prediction_mode(
                "residual_decomposed_fno",
                "miniunet",
                "source_superposition_v1",
            )
        legacy = build_model(
            {
                "architecture": "miniunet",
                "input_channels": 3,
                "output_channels": 1,
                "base_channels": 4,
                "depth": 2,
            }
        )
        self.assertEqual(tuple(legacy(torch.randn(1, 3, 16, 16)).shape), (1, 1, 16, 16))

    def test_direct_normalization_inverts_exactly(self) -> None:
        stats = DirectTemperatureTargetStats(
            mode="train_standard",
            mean_K=400.0,
            std_K=25.0,
            min_K=300.0,
            max_K=500.0,
            num_samples=2,
            num_grid_cells=8,
        )
        value = torch.tensor([[[350.0, 400.0], [425.0, 500.0]]])
        normalized = (value - stats.mean_K) / stats.std_K
        restored = normalized * stats.std_K + stats.mean_K
        torch.testing.assert_close(restored, value)

    def test_synthetic_cpu_optimization_decreases_loss(self) -> None:
        torch.manual_seed(11)
        model = DirectTemperatureFNO2d(
            input_channels=2,
            metadata_dim=2,
            width=6,
            layers=1,
            modes_x=3,
            modes_y=3,
            projection_channels=8,
        )
        x = torch.randn(4, 2, 8, 8)
        metadata = torch.randn(4, 2)
        target = 0.4 * x[:, :1] - 0.2 * x[:, 1:2]
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
        with torch.no_grad():
            initial = torch.nn.functional.l1_loss(model(x, metadata), target).item()
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x, metadata), target)
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.l1_loss(model(x, metadata), target).item()
        self.assertLess(final, initial * 0.8)

    def test_comparison_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {}
            modes = {
                "direct_cnn": "direct_temperature",
                "direct_fno": "direct_temperature_fno",
                "residual_cnn": "residual_decomposed",
                "residual_fno": "residual_decomposed_fno",
            }
            for model, mode in modes.items():
                model_root = root / model
                roots[model] = model_root
                for protocol in comparison.PROTOCOLS:
                    directory = model_root / protocol
                    directory.mkdir(parents=True)
                    (directory / "metrics.json").write_text(
                        json.dumps(
                            {
                                "num_samples": 2,
                                "model": {
                                    "prediction_mode": mode,
                                    "parameter_count": 100,
                                },
                                "cnn_final_temperature": {"mae_K": 2.0, "rmse_K": 3.0},
                                "inference_runtime_per_sample_s": 0.001,
                            }
                        ),
                        encoding="utf-8",
                    )
                    with (directory / "metrics_by_case.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=["case", "final_temperature_mae_K"],
                        )
                        writer.writeheader()
                        writer.writerow(
                            {"case": "f041", "final_temperature_mae_K": "2.0"}
                        )
            headline, families = comparison.aggregate_comparison(roots)
            self.assertEqual(len(headline), 12)
            self.assertEqual(len(families), 12)


if __name__ == "__main__":
    unittest.main()
