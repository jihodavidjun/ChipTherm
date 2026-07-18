from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.ml.models import DecomposedMiniUNetWithFeatureFusion, build_model  # noqa: E402
from train_residual_cnn import (  # noqa: E402
    compute_delta_R_eff_target_stats,
    decomposed_targets,
    reconstruct_decomposed_temperature,
)


class ResidualResistanceMeanHeadTests(unittest.TestCase):
    def test_reconstruction_uses_source_base_without_double_counting_ambient(self) -> None:
        base = torch.full((2, 4, 4), 320.0)
        ambient = torch.full((2,), 300.0)
        mean = torch.tensor([10.0, -5.0])
        centered = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        outputs = {"mean_rise": mean, "centered_field": centered}

        pred = reconstruct_decomposed_temperature(outputs, ambient, base, mean_head_mode="residual_resistance")

        self.assertTrue(torch.allclose(pred, base + mean[:, None, None] + centered))
        self.assertFalse(torch.allclose(pred, ambient[:, None, None] + base + mean[:, None, None] + centered))
        self.assertTrue(torch.allclose((pred - base).mean(dim=(-2, -1)), mean, atol=1.0e-6))

    def test_targets_are_residual_resistance_not_total_resistance(self) -> None:
        base = torch.full((1, 4, 4), 320.0)
        centered = torch.zeros((1, 4, 4))
        centered[:, 0, 0] = 3.0
        centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
        total_power = torch.tensor([20.0])
        temperature = base + 6.0 + centered
        ambient = torch.tensor([300.0])

        targets = decomposed_targets(
            temperature,
            ambient,
            base,
            total_power,
            mean_head_mode="residual_resistance",
        )

        self.assertAlmostEqual(float(targets["mean_correction_K"].item()), 6.0, places=6)
        self.assertAlmostEqual(float(targets["delta_R_eff_K_per_W"].item()), 0.3, places=6)
        self.assertAlmostEqual(float((temperature - ambient[:, None, None]).mean().item() / 20.0), 1.3, places=6)
        self.assertTrue(torch.allclose(targets["centered_field_K"], centered, atol=1.0e-6))

    def test_signed_delta_R_and_total_power_scaling(self) -> None:
        model = DecomposedMiniUNetWithFeatureFusion(
            input_channels=4,
            base_channels=4,
            refine_channels=4,
            refine_blocks=1,
            refinement_channel_indices=(0, 1),
            refinement_channel_names=("power_density_W_per_mm2", "occupancy_mask"),
            metadata_dim=3,
            metadata_hidden_dim=8,
            metadata_embedding_dim=8,
            global_branch_channel_indices=(0, 1),
            global_branch_channel_names=("power_density_W_per_mm2", "occupancy_mask"),
            global_hidden_channels=4,
            global_pool_size=8,
            global_context_blocks=1,
            mean_head_mode="residual_resistance",
            delta_R_eff_mean_K_per_W=0.0,
            delta_R_eff_std_K_per_W=1.0,
        )
        raw = torch.tensor([-0.25, -0.25])
        total_power = torch.tensor([10.0, 20.0])

        mean, delta_r = model._mean_outputs(raw, total_power)

        self.assertIsNotNone(delta_r)
        self.assertTrue(torch.all(delta_r < 0.0))
        self.assertTrue(torch.allclose(mean, torch.tensor([-2.5, -5.0])))

    def test_zero_total_power_is_rejected(self) -> None:
        model = DecomposedMiniUNetWithFeatureFusion(
            input_channels=4,
            base_channels=4,
            refine_channels=4,
            refine_blocks=1,
            refinement_channel_indices=(0,),
            refinement_channel_names=("power_density_W_per_mm2",),
            metadata_dim=3,
            metadata_hidden_dim=8,
            metadata_embedding_dim=8,
            global_branch_channel_indices=(0,),
            global_branch_channel_names=("power_density_W_per_mm2",),
            global_hidden_channels=4,
            global_pool_size=8,
            global_context_blocks=1,
            mean_head_mode="residual_resistance",
        )
        with self.assertRaises(ValueError):
            model._mean_outputs(torch.tensor([1.0]), torch.tensor([0.0]))

    def test_delta_R_stats_are_train_only(self) -> None:
        train_batches = [
            {
                "temperature": torch.full((1, 2, 2), 330.0),
                "physics": torch.full((1, 2, 2), 320.0),
                "total_power_W": torch.tensor([20.0]),
            },
            {
                "temperature": torch.full((1, 2, 2), 318.0),
                "physics": torch.full((1, 2, 2), 320.0),
                "total_power_W": torch.tensor([10.0]),
            },
        ]
        stats = compute_delta_R_eff_target_stats(train_batches)

        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["mean_K_per_W"], 0.15, places=6)
        self.assertAlmostEqual(stats["std_K_per_W"], 0.35, places=6)

    def test_backward_compatible_direct_k_config_default(self) -> None:
        config = {
            "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion",
            "input_channels": 4,
            "base_channels": 4,
            "refine_channels": 4,
            "refine_blocks": 1,
            "refinement_channel_indices": [0],
            "refinement_channel_names": ["power_density_W_per_mm2"],
            "metadata_dim": 3,
            "metadata_hidden_dim": 8,
            "metadata_embedding_dim": 8,
            "global_branch_channel_indices": [0],
            "global_branch_channel_names": ["power_density_W_per_mm2"],
            "global_hidden_channels": 4,
            "global_pool_size": 8,
            "global_blocks": 1,
        }
        model = build_model(config)

        self.assertEqual(model.mean_head_mode, "direct_k")

    def test_resistance_alias_reconstructs_model(self) -> None:
        config = {
            "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion_resistance_mean",
            "input_channels": 4,
            "base_channels": 4,
            "refine_channels": 4,
            "refine_blocks": 1,
            "refinement_channel_indices": [0],
            "refinement_channel_names": ["power_density_W_per_mm2"],
            "metadata_dim": 3,
            "metadata_hidden_dim": 8,
            "metadata_embedding_dim": 8,
            "global_branch_channel_indices": [0],
            "global_branch_channel_names": ["power_density_W_per_mm2"],
            "global_hidden_channels": 4,
            "global_pool_size": 8,
            "global_blocks": 1,
            "delta_R_eff_target_mean_K_per_W": -0.1,
            "delta_R_eff_target_std_K_per_W": 0.25,
        }
        model = build_model(config)
        x = torch.randn(2, 4, 64, 64)
        metadata = torch.randn(2, 3)
        outputs = model(x, metadata, total_power_W=torch.tensor([10.0, 12.0]))

        self.assertEqual(model.mean_head_mode, "residual_resistance")
        self.assertEqual(outputs["centered_field"].shape, (2, 64, 64))
        self.assertTrue(torch.isfinite(outputs["mean_rise"]).all())
        self.assertTrue(torch.isfinite(outputs["delta_R_eff"]).all())
        self.assertTrue(torch.allclose(outputs["centered_field"].mean(dim=(-2, -1)), torch.zeros(2), atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
