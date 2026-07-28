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


trainer = load_script("trainer_ufno", ROOT / "scripts/train_residual_cnn.py")
wrapper = load_script("wrapper_ufno", ROOT / "scripts/train_benchmark_v2_fno.py")
evaluator = load_script("evaluator_ufno", ROOT / "scripts/evaluate_residual_cnn.py")

from chiptherm.ml.models import build_model  # noqa: E402


class BenchmarkV2UFNOTrainingTests(unittest.TestCase):
    def test_configs_and_prediction_mode_validation(self) -> None:
        cases = (
            (
                "direct_ufno",
                "package_direct_temperature_ufno_normalized_seed1.yaml",
                "direct_temperature_ufno",
                "ufno2d_direct_conditioned",
                "none",
            ),
            (
                "residual_ufno",
                "package_residual_ufno_decomposed_seed1.yaml",
                "residual_decomposed_ufno",
                "ufno2d_residual_decomposed_conditioned",
                "source_superposition_v1",
            ),
        )
        for experiment, filename, mode, architecture, physics in cases:
            config = yaml.safe_load(
                (
                    ROOT
                    / "configs/benchmark_v2_50family/training"
                    / filename
                ).read_text(encoding="utf-8")
            )
            wrapper.validate_config(config, wrapper.EXPERIMENTS[experiment])
            trainer.validate_prediction_mode(mode, architecture, physics)
            evaluator.validate_checkpoint_prediction_mode(mode, architecture, physics)
            self.assertEqual(config["ufno_unet_branch_indices"], [3, 4, 5])
            self.assertEqual(config["fno_layers"], 6)
            if experiment == "residual_ufno":
                self.assertEqual(config["mean_correction_sign"], 1)
                self.assertEqual(config["centered_correction_sign"], 1)
                invalid = dict(config)
                invalid["centered_correction_sign"] = -1
                with self.assertRaisesRegex(
                    ValueError,
                    "centered_correction_sign",
                ):
                    wrapper.validate_config(invalid, wrapper.EXPERIMENTS[experiment])

    def test_residual_evaluator_dispatches_raw_total_power(self) -> None:
        model = build_model(
            {
                "architecture": "ufno2d_residual_decomposed_conditioned",
                "input_channels": 4,
                "metadata_dim": 2,
                "metadata_hidden_dim": 4,
                "metadata_embedding_dim": 4,
                "fno_width": 3,
                "fno_layers": 6,
                "fno_modes_x": 2,
                "fno_modes_y": 2,
                "fno_projection_channels": 4,
            }
        ).eval()
        x = torch.randn(2, 4, 8, 8)
        metadata = torch.randn(2, 2)
        power = torch.tensor([7.0, 11.0])
        with torch.no_grad():
            output = evaluator.call_model(
                model,
                x,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
                total_power_W=power,
            )
        torch.testing.assert_close(output["mean_rise"], power * output["delta_R_eff"])
        with self.assertRaisesRegex(ValueError, "raw batch"):
            evaluator.call_model(
                model,
                x,
                metadata,
                None,
                conditioned=True,
                graph_enabled=False,
            )

    def test_residual_reconstruction_adds_both_known_corrections(self) -> None:
        source_base = torch.full((1, 2, 2), 100.0)
        total_power = torch.tensor([10.0])
        delta_r = torch.tensor([2.0])
        centered = torch.tensor([[[-1.0, 1.0], [-3.0, 3.0]]])
        outputs = {
            "delta_R_eff": delta_r,
            "mean_rise": total_power * delta_r,
            "centered_field": centered,
        }
        ambient = torch.tensor([300.0])

        actual = evaluator.reconstruct_decomposed_temperature(
            outputs,
            ambient,
            source_base,
            mean_head_mode="residual_resistance",
        )
        expected = torch.tensor([[[119.0, 121.0], [117.0, 123.0]]])

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertFalse(
            torch.equal(
                actual,
                source_base - total_power[:, None, None] * delta_r[:, None, None] + centered,
            )
        )
        self.assertFalse(
            torch.equal(
                actual,
                source_base + total_power[:, None, None] * delta_r[:, None, None] - centered,
            )
        )

    def test_residual_ufno_reconstruction_matches_residual_fno_helper(self) -> None:
        source_base = torch.tensor([[[315.0, 316.0], [317.0, 318.0]]])
        centered = torch.tensor([[[-2.0, 2.0], [-4.0, 4.0]]])
        outputs = {
            "mean_rise": torch.tensor([6.0]),
            "centered_field": centered,
        }
        ambient = torch.tensor([300.0])

        ufno_training_path = trainer.reconstruct_decomposed_temperature(
            outputs,
            ambient,
            source_base,
            mean_head_mode="residual_resistance",
        )
        established_fno_evaluation_path = evaluator.reconstruct_decomposed_temperature(
            outputs,
            ambient,
            source_base,
            mean_head_mode="residual_resistance",
        )

        torch.testing.assert_close(
            ufno_training_path,
            established_fno_evaluation_path,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            ufno_training_path,
            source_base + 6.0 + centered,
            rtol=0.0,
            atol=0.0,
        )

    def test_direct_ufno_does_not_require_total_power(self) -> None:
        model = build_model(
            {
                "architecture": "ufno2d_direct_conditioned",
                "input_channels": 3,
                "metadata_dim": 2,
                "metadata_hidden_dim": 4,
                "metadata_embedding_dim": 4,
                "fno_width": 3,
                "fno_layers": 6,
                "fno_modes_x": 2,
                "fno_modes_y": 2,
                "fno_projection_channels": 4,
                "target_normalization_mode": "train_standard",
                "target_std_K": 1.0,
            }
        ).eval()
        with torch.no_grad():
            result = evaluator.call_model(
                model,
                torch.randn(2, 3, 8, 8),
                torch.randn(2, 2),
                None,
                conditioned=True,
                graph_enabled=False,
            )
        self.assertEqual(tuple(result.shape), (2, 1, 8, 8))

    def test_architecture_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires architecture"):
            trainer.validate_prediction_mode(
                "direct_temperature_ufno", "fno2d_direct_conditioned", "none"
            )
        with self.assertRaisesRegex(ValueError, "requires its direct"):
            trainer.validate_prediction_mode(
                "residual_decomposed_ufno", "ufno2d_direct_conditioned", "none"
            )


if __name__ == "__main__":
    unittest.main()
