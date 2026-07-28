#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.sau_fno_models import (  # noqa: E402
    ConditionedDirectSAUFNO2d,
    ConditionedResidualDecomposedSAUFNO2d,
    SAUAttention2d,
    attention_memory_estimate,
)
from chiptherm.ml.ufno_models import ConditionedDirectUFNO2d  # noqa: E402


class SAUFNOModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_attention_shape_projections_softmax_and_known_value(self) -> None:
        attention = SAUAttention2d(2).eval()
        self.assertEqual(attention.query.kernel_size, (1, 1))
        self.assertEqual(attention.key.kernel_size, (1, 1))
        self.assertEqual(attention.value.kernel_size, (1, 1))
        with torch.no_grad():
            attention.query.weight.zero_()
            attention.query.bias.zero_()
            attention.key.weight.zero_()
            attention.key.bias.zero_()
            attention.value.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
            attention.value.bias.zero_()
        x = torch.tensor([[[[1.0, 3.0]], [[2.0, 6.0]]]])
        query, key, _ = attention.project_qkv(x)
        weights = attention.attention_weights(query, key)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(1, 2), rtol=0.0, atol=0.0
        )
        with torch.no_grad():
            output = attention(x)
        expected = torch.tensor([[[[2.0, 2.0]], [[4.0, 4.0]]]])
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    def test_finite_forward_backward_and_64_grid_preservation(self) -> None:
        attention = SAUAttention2d(2)
        x = torch.randn(1, 2, 64, 64, requires_grad=True)
        output = attention(x)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in attention.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_attention_is_once_after_final_block_and_after_crop(self) -> None:
        model = tiny_direct_sau().eval()
        attention_modules = [
            module for module in model.modules() if isinstance(module, SAUAttention2d)
        ]
        self.assertEqual(len(attention_modules), 1)
        events: list[tuple[str, tuple[int, int]]] = []
        block_hook = model.backbone.blocks[-1].register_forward_hook(
            lambda _m, _i, output: events.append(("block5", tuple(output.shape[-2:])))
        )
        attention_hook = model.backbone.attention.register_forward_pre_hook(
            lambda _m, inputs: events.append(("attention", tuple(inputs[0].shape[-2:])))
        )
        with torch.no_grad():
            output = model(torch.randn(1, 3, 8, 8), torch.randn(1, 2))
        block_hook.remove()
        attention_hook.remove()
        self.assertEqual(events, [("block5", (16, 16)), ("attention", (8, 8))])
        self.assertEqual(tuple(output.shape), (1, 1, 8, 8))

    def test_attention_disabled_parity_with_existing_ufno(self) -> None:
        ufno = ConditionedDirectUFNO2d(
            input_channels=3,
            metadata_dim=2,
            metadata_hidden_dim=4,
            metadata_embedding_dim=4,
            width=2,
            layers=6,
            modes_x=2,
            modes_y=2,
            projection_channels=3,
        ).eval()
        sau = tiny_direct_sau().eval()
        incompatible = sau.load_state_dict(ufno.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(
            incompatible.missing_keys
            and all("backbone.attention." in key for key in incompatible.missing_keys)
        )
        x = torch.randn(2, 3, 8, 8)
        metadata = torch.randn(2, 2)
        with torch.no_grad():
            expected = ufno(x, metadata)
            actual = sau(x, metadata, disable_attention=True)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_factory_channels_residual_reconstruction_and_zero_mean(self) -> None:
        direct = build_model(
            {
                **tiny_config("sau_fno2d_direct_conditioned", input_channels=33),
                "target_normalization_mode": "train_standard",
                "target_std_K": 2.0,
            }
        )
        residual = build_model(
            tiny_config(
                "sau_fno2d_residual_decomposed_conditioned", input_channels=34
            )
        )
        self.assertEqual(direct.input_channels, 33)
        self.assertEqual(residual.input_channels, 34)
        source = torch.full((2, 8, 8), 300.0)
        power = torch.tensor([4.0, 7.0])
        output = residual(
            torch.randn(2, 34, 8, 8),
            torch.randn(2, 2),
            total_power_W=power,
        )
        torch.testing.assert_close(
            output["centered_field"].mean(dim=(-2, -1)),
            torch.zeros(2),
            rtol=0.0,
            atol=1.0e-6,
        )
        reconstructed = (
            source
            + power[:, None, None] * output["delta_R_eff"][:, None, None]
            + output["centered_field"]
        )
        torch.testing.assert_close(
            reconstructed,
            source + output["mean_rise"][:, None, None] + output["centered_field"],
        )
        config = residual.config()
        self.assertEqual(config["mean_correction_sign"], 1)
        self.assertEqual(config["centered_correction_sign"], 1)
        self.assertEqual(config["sau_number_of_heads"], 1)
        self.assertFalse(config["sau_residual_connection"])
        self.assertEqual(
            config["reconstruction"],
            "source_superposition_base_K + total_power_W * "
            "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K",
        )

    def test_checkpoint_round_trip_parameter_and_memory_report(self) -> None:
        model = tiny_direct_sau().eval()
        x = torch.randn(1, 3, 8, 8)
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
        self.assertEqual(model.config()["parameter_count"], count_parameters(model))
        memory = attention_memory_estimate(
            height=64, width=64, batch_size=64, element_size_bytes=4
        )
        self.assertEqual(memory["tokens"], 4096)
        self.assertEqual(
            memory["attention_matrix_elements_per_sample"], 4096 * 4096
        )
        self.assertEqual(memory["attention_matrix_bytes_per_batch"], 4 * 4096 * 4096 * 64)

    def test_component_profile_exposes_attention_stage(self) -> None:
        model = tiny_direct_sau().eval()
        with torch.no_grad():
            output, timings = model.forward_profile(
                torch.randn(1, 3, 8, 8),
                torch.randn(1, 2),
                synchronize=lambda: None,
            )
        self.assertEqual(tuple(output.shape), (1, 1, 8, 8))
        self.assertEqual(
            set(timings),
            {"ufno_backbone_s", "sau_attention_s", "projection_head_s"},
        )
        self.assertTrue(all(value >= 0.0 for value in timings.values()))


def tiny_config(architecture: str, *, input_channels: int) -> dict[str, object]:
    return {
        "architecture": architecture,
        "input_channels": input_channels,
        "output_channels": 1,
        "metadata_dim": 2,
        "metadata_hidden_dim": 4,
        "metadata_embedding_dim": 4,
        "fno_width": 2,
        "fno_layers": 6,
        "fno_modes_x": 2,
        "fno_modes_y": 2,
        "fno_projection_channels": 3,
        "ufno_unet_branch_indices": [3, 4, 5],
        "ufno_unet_depth": 3,
        "ufno_domain_padding": 8,
        "sau_attention_dim": 2,
    }


def tiny_direct_sau() -> ConditionedDirectSAUFNO2d:
    return ConditionedDirectSAUFNO2d(
        input_channels=3,
        metadata_dim=2,
        metadata_hidden_dim=4,
        metadata_embedding_dim=4,
        width=2,
        layers=6,
        modes_x=2,
        modes_y=2,
        projection_channels=3,
        attention_dim=2,
    )


if __name__ == "__main__":
    unittest.main()
