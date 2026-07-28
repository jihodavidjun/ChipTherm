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
evaluator = load_script(
    "evaluate_residual_cnn_fno_test",
    REPO_ROOT / "scripts/evaluate_residual_cnn.py",
)


class BenchmarkV2FNOTrainingTests(unittest.TestCase):
    def test_residual_fno_evaluation_dispatch_uses_raw_total_power(self) -> None:
        torch.manual_seed(3)
        model = build_model(
            {
                "architecture": "fno2d_residual_decomposed_conditioned",
                "input_channels": 4,
                "metadata_dim": 3,
                "metadata_hidden_dim": 8,
                "metadata_embedding_dim": 8,
                "fno_width": 6,
                "fno_layers": 1,
                "fno_modes_x": 3,
                "fno_modes_y": 3,
                "fno_projection_channels": 8,
                "delta_R_eff_target_mean_K_per_W": 0.2,
                "delta_R_eff_target_std_K_per_W": 0.1,
            }
        ).eval()
        model_input = torch.randn(2, 4, 16, 16)
        metadata = torch.randn(2, 3)
        raw_total_power = torch.tensor([11.0, 37.0], dtype=torch.float32)
        with torch.no_grad():
            outputs = evaluator.call_model(
                model,
                model_input,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
                total_power_W=raw_total_power,
            )
        torch.testing.assert_close(
            outputs["mean_rise"],
            raw_total_power * outputs["delta_R_eff"],
        )
        source_base = torch.randn(2, 16, 16) + 350.0
        reconstructed = evaluator.reconstruct_decomposed_temperature(
            outputs,
            torch.tensor([318.15, 318.15]),
            source_base,
            mean_head_mode="residual_resistance",
        )
        expected = (
            source_base
            + raw_total_power[:, None, None] * outputs["delta_R_eff"][:, None, None]
            + outputs["centered_field"]
        )
        torch.testing.assert_close(reconstructed, expected)
        self.assertTrue(torch.isfinite(reconstructed).all())

    def test_residual_fno_missing_or_invalid_total_power_fails_clearly(self) -> None:
        model = build_model(
            {
                "architecture": "fno2d_residual_decomposed_conditioned",
                "input_channels": 2,
                "metadata_dim": 2,
                "metadata_hidden_dim": 4,
                "metadata_embedding_dim": 4,
                "fno_width": 4,
                "fno_layers": 1,
                "fno_modes_x": 2,
                "fno_modes_y": 2,
                "fno_projection_channels": 4,
            }
        )
        x = torch.randn(2, 2, 8, 8)
        metadata = torch.randn(2, 2)
        with self.assertRaisesRegex(ValueError, r"raw batch\['total_power_W'\]"):
            evaluator.call_model(
                model,
                x,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            evaluator.call_model(
                model,
                x,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
                total_power_W=torch.tensor([1.0, 0.0]),
            )

    def test_direct_fno_and_canonical_residual_cnn_dispatch_remain_compatible(self) -> None:
        direct = DirectTemperatureFNO2d(
            input_channels=3,
            metadata_dim=2,
            metadata_hidden_dim=4,
            metadata_embedding_dim=4,
            width=4,
            layers=1,
            modes_x=2,
            modes_y=2,
            projection_channels=4,
        ).eval()
        x = torch.randn(2, 3, 8, 8)
        metadata = torch.randn(2, 2)
        with torch.no_grad():
            direct_output = evaluator.call_model(
                direct,
                x,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
            )
        self.assertEqual(tuple(direct_output.shape), (2, 1, 8, 8))

        canonical_cnn = build_model(
            {
                "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion",
                "input_channels": 6,
                "output_channels": 1,
                "base_channels": 4,
                "depth": 3,
                "refine_channels": 4,
                "refine_blocks": 1,
                "refinement_channel_indices": [0, 1],
                "refinement_channel_names": ["power", "occupancy"],
                "metadata_dim": 3,
                "metadata_hidden_dim": 4,
                "metadata_embedding_dim": 4,
                "physics_input_mode": "source_superposition_v1",
                "global_branch_channel_indices": [0, 1, 5],
                "global_branch_channel_names": ["power", "occupancy", "source_base"],
                "global_hidden_channels": 4,
                "global_pool_size": 8,
                "global_context_blocks": 1,
                "mean_head_mode": "residual_resistance",
                "delta_R_eff_target_mean_K_per_W": 0.1,
                "delta_R_eff_target_std_K_per_W": 0.2,
            }
        ).eval()
        power = torch.tensor([5.0, 9.0])
        with torch.no_grad():
            cnn_output = evaluator.call_model(
                canonical_cnn,
                torch.randn(2, 6, 64, 64),
                torch.randn(2, 3),
                None,
                conditioned=True,
                graph_enabled=False,
                total_power_W=power,
            )
        torch.testing.assert_close(
            cnn_output["mean_rise"],
            power * cnn_output["delta_R_eff"],
        )

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
