#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.ufno_models import (  # noqa: E402
    PUBLISHED_UFNO_BRANCH_INDICES,
    UFNO_REFERENCE_COMMIT,
    ConditionedDirectUFNO2d,
    ConditionedResidualDecomposedUFNO2d,
    MiniUNet2d,
    UFNO2dBlock,
)


class UFNOModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_correspondence_and_reference_commit(self) -> None:
        report = ROOT / "docs/ufno_architecture_correspondence.md"
        self.assertTrue(report.is_file())
        text = report.read_text(encoding="utf-8")
        self.assertIn(UFNO_REFERENCE_COMMIT, text)
        self.assertIn("task-adapted published", text)

    def test_mini_unet_shape_and_skip_path(self) -> None:
        model = MiniUNet2d(4).eval()
        x = torch.randn(2, 4, 16, 24)
        with torch.no_grad():
            output = model(x)
            zero_output = model(torch.zeros_like(x))
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertFalse(torch.equal(output, zero_output))
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            model(torch.randn(1, 4, 15, 16))

    def test_ufno_block_adds_three_branches(self) -> None:
        block = UFNO2dBlock(
            4, 2, 2, 3, use_unet=True, unet_depth=3, activation="gelu"
        ).eval()
        x = torch.randn(2, 4, 16, 16)
        metadata = torch.randn(2, 3)
        with torch.no_grad():
            branch_sum = block.branch_sum(x)
            expected = block.spectral(x) + block.pointwise(x) + block.unet(x)
            full = block(x, metadata)
            without_unet = block(x, metadata, disable_unet=True)
        torch.testing.assert_close(branch_sum, expected)
        self.assertEqual(tuple(full.shape), tuple(x.shape))
        self.assertGreater(float((full - without_unet).abs().max()), 0.0)

    def test_direct_output_shape_determinism_and_factory(self) -> None:
        config = {
            "architecture": "ufno2d_direct_conditioned",
            "input_channels": 3,
            "output_channels": 1,
            "metadata_dim": 2,
            "metadata_hidden_dim": 4,
            "metadata_embedding_dim": 4,
            "fno_width": 4,
            "fno_layers": 6,
            "fno_modes_x": 2,
            "fno_modes_y": 2,
            "fno_projection_channels": 4,
            "ufno_unet_branch_indices": [3, 4, 5],
            "ufno_unet_depth": 3,
            "ufno_domain_padding": 8,
            "target_normalization_mode": "train_standard",
            "target_mean_K": 400.0,
            "target_std_K": 20.0,
        }
        model = build_model(config).eval()
        self.assertIsInstance(model, ConditionedDirectUFNO2d)
        self.assertEqual(model.backbone.unet_branch_indices, PUBLISHED_UFNO_BRANCH_INDICES)
        x = torch.randn(2, 3, 8, 8)
        metadata = torch.randn(2, 2)
        with torch.no_grad():
            first = model(x, metadata)
            second = model(x, metadata)
        self.assertEqual(tuple(first.shape), (2, 1, 8, 8))
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertGreater(count_parameters(model), 0)

    def test_residual_shapes_zero_mean_and_reconstruction(self) -> None:
        model = ConditionedResidualDecomposedUFNO2d(
            input_channels=4,
            metadata_dim=2,
            metadata_hidden_dim=4,
            metadata_embedding_dim=4,
            width=4,
            layers=6,
            modes_x=2,
            modes_y=2,
            projection_channels=4,
            delta_R_eff_mean_K_per_W=0.1,
            delta_R_eff_std_K_per_W=0.2,
        ).eval()
        x = torch.randn(2, 4, 8, 8)
        metadata = torch.randn(2, 2)
        power = torch.tensor([5.0, 9.0])
        source = torch.randn(2, 8, 8) + 350.0
        with torch.no_grad():
            output = model(x, metadata, total_power_W=power)
        self.assertEqual(tuple(output["mean_rise"].shape), (2,))
        self.assertEqual(tuple(output["centered_field"].shape), (2, 8, 8))
        torch.testing.assert_close(
            output["centered_field"].mean(dim=(-2, -1)),
            torch.zeros(2),
            atol=1.0e-6,
            rtol=0.0,
        )
        reconstructed = (
            source + power[:, None, None] * output["delta_R_eff"][:, None, None]
            + output["centered_field"]
        )
        expected = source + output["mean_rise"][:, None, None] + output["centered_field"]
        torch.testing.assert_close(reconstructed, expected)
        config = model.config()
        self.assertEqual(config["mean_correction_sign"], 1)
        self.assertEqual(config["centered_correction_sign"], 1)
        self.assertEqual(
            config["residual_target"],
            "HotSpot_K - source_superposition_base_K",
        )
        self.assertEqual(
            config["reconstruction"],
            "source_superposition_base_K + total_power_W * "
            "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K",
        )
        with self.assertRaisesRegex(ValueError, "total_power_W"):
            model(x, metadata)

    def test_checkpoint_round_trip(self) -> None:
        model = ConditionedDirectUFNO2d(
            input_channels=2,
            metadata_dim=2,
            metadata_hidden_dim=4,
            metadata_embedding_dim=4,
            width=3,
            layers=6,
            modes_x=2,
            modes_y=2,
            projection_channels=4,
        ).eval()
        x = torch.randn(1, 2, 8, 8)
        metadata = torch.randn(1, 2)
        with torch.no_grad():
            expected = model(x, metadata)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(
                {"model_config": model.config(), "model_state_dict": model.state_dict()},
                path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            restored = build_model(checkpoint["model_config"]).eval()
            restored.load_state_dict(checkpoint["model_state_dict"])
            with torch.no_grad():
                actual = restored(x, metadata)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_tiny_cpu_optimization_decreases_loss(self) -> None:
        model = ConditionedDirectUFNO2d(
            input_channels=1,
            metadata_dim=1,
            metadata_hidden_dim=3,
            metadata_embedding_dim=3,
            width=2,
            layers=6,
            modes_x=2,
            modes_y=2,
            projection_channels=3,
        )
        x = torch.randn(2, 1, 8, 8)
        metadata = torch.randn(2, 1)
        target = 0.3 * x
        optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-3)
        with torch.no_grad():
            initial = torch.nn.functional.l1_loss(model(x, metadata), target).item()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x, metadata), target)
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.l1_loss(model(x, metadata), target).item()
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
