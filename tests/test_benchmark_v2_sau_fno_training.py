#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = load_script("trainer_sau_fno", ROOT / "scripts/train_residual_cnn.py")
wrapper = load_script("wrapper_sau_fno", ROOT / "scripts/train_benchmark_v2_fno.py")
evaluator = load_script("evaluator_sau_fno", ROOT / "scripts/evaluate_residual_cnn.py")
analyzer = load_script(
    "analyzer_sau_fno", ROOT / "scripts/analyze_residual_cnn_errors.py"
)

from chiptherm.ml.models import build_model  # noqa: E402


class BenchmarkV2SAUFNOTrainingTests(unittest.TestCase):
    def test_primary_configs_and_dispatch(self) -> None:
        cases = (
            (
                "direct_sau_fno",
                "package_direct_temperature_sau_fno_normalized_seed1.yaml",
                "direct_temperature_sau_fno",
                "sau_fno2d_direct_conditioned",
                "none",
            ),
            (
                "residual_sau_fno",
                "package_residual_sau_fno_decomposed_seed1.yaml",
                "residual_decomposed_sau_fno",
                "sau_fno2d_residual_decomposed_conditioned",
                "source_superposition_v1",
            ),
        )
        for experiment, filename, mode, architecture, physics in cases:
            config = yaml.safe_load(
                (
                    ROOT / "configs/benchmark_v2_50family/training" / filename
                ).read_text(encoding="utf-8")
            )
            wrapper.validate_config(config, wrapper.EXPERIMENTS[experiment])
            trainer.validate_prediction_mode(mode, architecture, physics)
            evaluator.validate_checkpoint_prediction_mode(mode, architecture, physics)
            self.assertEqual(config["fno_width"], 32)
            self.assertEqual(config["fno_modes_x"], 12)
            self.assertEqual(config["fno_modes_y"], 12)
            self.assertEqual(config["fno_layers"], 6)
            self.assertEqual(config["ufno_unet_branch_indices"], [3, 4, 5])
            self.assertEqual(config["sau_attention_dim"], 32)
            self.assertEqual(config["sau_number_of_heads"], 1)

    def test_residual_evaluator_and_analyzer_receive_raw_total_power(self) -> None:
        model = build_model(
            {
                "architecture": "sau_fno2d_residual_decomposed_conditioned",
                "input_channels": 4,
                "metadata_dim": 2,
                "metadata_hidden_dim": 4,
                "metadata_embedding_dim": 4,
                "fno_width": 2,
                "fno_layers": 6,
                "fno_modes_x": 2,
                "fno_modes_y": 2,
                "fno_projection_channels": 3,
                "sau_attention_dim": 2,
            }
        ).eval()
        model_input = torch.randn(2, 4, 8, 8)
        metadata = torch.randn(2, 2)
        power = torch.tensor([5.0, 9.0], dtype=torch.float32)
        with torch.no_grad():
            output = evaluator.call_model(
                model,
                model_input,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
                total_power_W=power,
            )
        torch.testing.assert_close(output["mean_rise"], power * output["delta_R_eff"])
        info = analyzer.architecture_info(model.config())
        self.assertTrue(info["conditioned"])
        self.assertTrue(info["decomposed"])
        self.assertEqual(info["mean_head_mode"], "residual_resistance")
        with self.assertRaisesRegex(ValueError, "raw batch"):
            evaluator.call_model(
                model,
                model_input,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
            )

    def test_direct_does_not_require_total_power_and_residual_signs_are_guarded(self) -> None:
        direct = build_model(
            {
                "architecture": "sau_fno2d_direct_conditioned",
                "input_channels": 3,
                "metadata_dim": 2,
                "metadata_hidden_dim": 4,
                "metadata_embedding_dim": 4,
                "fno_width": 2,
                "fno_layers": 6,
                "fno_modes_x": 2,
                "fno_modes_y": 2,
                "fno_projection_channels": 3,
                "sau_attention_dim": 2,
                "target_normalization_mode": "train_standard",
                "target_std_K": 1.0,
            }
        ).eval()
        with torch.no_grad():
            output = evaluator.call_model(
                direct,
                torch.randn(1, 3, 8, 8),
                torch.randn(1, 2),
                None,
                conditioned=True,
                graph_enabled=False,
            )
        self.assertEqual(tuple(output.shape), (1, 1, 8, 8))

        config = yaml.safe_load(
            (
                ROOT
                / "configs/benchmark_v2_50family/training/"
                "package_residual_sau_fno_decomposed_seed1.yaml"
            ).read_text(encoding="utf-8")
        )
        invalid = dict(config)
        invalid["mean_correction_sign"] = -1
        with self.assertRaisesRegex(ValueError, "mean_correction_sign"):
            wrapper.validate_config(
                invalid, wrapper.EXPERIMENTS["residual_sau_fno"]
            )

    def test_explicit_additive_reconstruction(self) -> None:
        source = torch.full((1, 2, 2), 100.0)
        power = torch.tensor([10.0])
        delta_r = torch.tensor([2.0])
        centered = torch.tensor([[[-1.0, 1.0], [-3.0, 3.0]]])
        outputs = {
            "delta_R_eff": delta_r,
            "mean_rise": power * delta_r,
            "centered_field": centered,
        }
        actual = evaluator.reconstruct_decomposed_temperature(
            outputs,
            torch.tensor([300.0]),
            source,
            mean_head_mode="residual_resistance",
        )
        expected = source + power[:, None, None] * delta_r[:, None, None] + centered
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(actual, source - 20.0 + centered))
        self.assertFalse(torch.equal(actual, source + 20.0 - centered))


if __name__ == "__main__":
    unittest.main()
