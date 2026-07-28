#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.fno_models import (
    DirectTemperatureFNO2d,
    ResidualDecomposedFNO2d,
    SpectralConv2d,
)
from chiptherm.ml.models import build_model, count_parameters


class FNOModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_spectral_shape_and_round_trip_compatibility(self) -> None:
        x = torch.randn(2, 3, 64, 64)
        layer = SpectralConv2d(3, 5, modes_x=12, modes_y=12)
        self.assertEqual(tuple(layer(x).shape), (2, 5, 64, 64))
        reconstructed = torch.fft.irfft2(
            torch.fft.rfft2(x, norm="ortho"),
            s=x.shape[-2:],
            norm="ortho",
        )
        torch.testing.assert_close(reconstructed, x, atol=2.0e-6, rtol=2.0e-6)

    def test_retained_mode_indexing_filters_unretained_frequency(self) -> None:
        layer = SpectralConv2d(1, 1, modes_x=2, modes_y=2)
        with torch.no_grad():
            layer.weight_positive.zero_()
            layer.weight_negative.zero_()
            layer.weight_positive[..., 0] = 1.0
            layer.weight_negative[..., 0] = 1.0
        coordinate = torch.arange(64, dtype=torch.float32)
        low = torch.cos(2.0 * torch.pi * coordinate / 64.0)[None, None, :, None]
        low = low.expand(1, 1, 64, 64)
        high = torch.cos(8.0 * 2.0 * torch.pi * coordinate / 64.0)[None, None, :, None]
        high = high.expand(1, 1, 64, 64)
        self.assertGreater(float(layer(low).abs().mean().detach()), 0.1)
        self.assertLess(float(layer(high).abs().max().detach()), 1.0e-5)

    def test_direct_forward_is_deterministic_and_conditioned(self) -> None:
        model = DirectTemperatureFNO2d(
            input_channels=33,
            metadata_dim=15,
            width=8,
            layers=2,
            modes_x=4,
            modes_y=4,
            projection_channels=16,
        ).eval()
        x = torch.randn(2, 33, 64, 64)
        metadata = torch.randn(2, 15)
        first = model(x, metadata)
        second = model(x, metadata)
        self.assertEqual(tuple(first.shape), (2, 1, 64, 64))
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
        self.assertTrue(torch.isfinite(first).all())

    def test_residual_outputs_and_reconstruction(self) -> None:
        model = ResidualDecomposedFNO2d(
            input_channels=34,
            metadata_dim=15,
            width=8,
            layers=2,
            modes_x=4,
            modes_y=4,
            projection_channels=16,
            delta_R_eff_mean_K_per_W=0.25,
            delta_R_eff_std_K_per_W=0.5,
        )
        x = torch.randn(3, 34, 64, 64)
        metadata = torch.randn(3, 15)
        power = torch.tensor([10.0, 20.0, 30.0])
        output = model(x, metadata, total_power_W=power)
        self.assertEqual(tuple(output["centered_field"].shape), (3, 64, 64))
        self.assertEqual(tuple(output["mean_rise"].shape), (3,))
        torch.testing.assert_close(
            output["centered_field"].mean(dim=(-2, -1)),
            torch.zeros(3),
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output["mean_rise"],
            power * output["delta_R_eff"],
        )
        source = torch.randn(3, 64, 64) + 350.0
        reconstructed = (
            source
            + power[:, None, None] * output["delta_R_eff"][:, None, None]
            + output["centered_field"]
        )
        canonical = source + output["mean_rise"][:, None, None] + output["centered_field"]
        torch.testing.assert_close(reconstructed, canonical)

    def test_checkpoint_config_round_trip_and_parameter_count(self) -> None:
        model = build_model(
            {
                "architecture": "fno2d_direct_conditioned",
                "input_channels": 33,
                "metadata_dim": 15,
                "target_normalization_mode": "train_standard",
                "target_mean_K": 400.0,
                "target_std_K": 30.0,
            }
        )
        self.assertGreater(count_parameters(model), 2_000_000)
        self.assertLess(count_parameters(model), 3_000_000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save({"model_config": model.config(), "model_state_dict": model.state_dict()}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            restored = build_model(checkpoint["model_config"])
            restored.load_state_dict(checkpoint["model_state_dict"])
        self.assertEqual(model.config(), restored.config())
        standard = build_model(
            {
                "architecture": "fno2d_direct_conditioned",
                "input_channels": 33,
                "metadata_dim": 15,
                "fno_capacity_profile": "fno_standard",
                "target_normalization_mode": "train_standard",
                "target_std_K": 1.0,
            }
        )
        self.assertEqual(standard.width, 48)
        self.assertEqual(standard.modes_x, 16)

    def test_gradients_are_finite(self) -> None:
        model = DirectTemperatureFNO2d(
            input_channels=3,
            metadata_dim=2,
            width=6,
            layers=1,
            modes_x=3,
            modes_y=3,
            projection_channels=8,
        )
        output = model(torch.randn(2, 3, 16, 16), torch.randn(2, 2))
        output.square().mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
