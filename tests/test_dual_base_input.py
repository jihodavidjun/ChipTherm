#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats, build_model_input  # noqa: E402


SOURCE_BASE_INDEX = REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full/test_index.csv"


class DualBaseInputTest(unittest.TestCase):
    def test_build_model_input_appends_source_base_then_physics_v1(self) -> None:
        x = torch.zeros((2, 33, 4, 4), dtype=torch.float32)
        source_base = torch.full((2, 4, 4), 400.0)
        physics_v1 = torch.full((2, 4, 4), 320.0)
        stats = NormalizationStats(
            schema_version=1,
            power_density_mean=0.0,
            power_density_std=1.0,
            physics_mean=400.0,
            physics_std=10.0,
            residual_mean=0.0,
            residual_std=1.0,
            num_samples=2,
            num_grid_cells=32,
            input_channels=33,
            auxiliary_physics_v1_mean=300.0,
            auxiliary_physics_v1_std=20.0,
        )
        model_input = build_model_input(
            x,
            source_base,
            stats,
            physics_input_mode="source_superposition_plus_physics_v1",
            physics_v1=physics_v1,
        )
        self.assertEqual(tuple(model_input.shape), (2, 35, 4, 4))
        self.assertTrue(torch.allclose(model_input[:, 33], torch.zeros((2, 4, 4))))
        self.assertTrue(torch.allclose(model_input[:, 34], torch.ones((2, 4, 4))))

    def test_dual_base_mode_requires_auxiliary_physics_v1(self) -> None:
        x = torch.zeros((1, 33, 4, 4), dtype=torch.float32)
        base = torch.zeros((1, 4, 4), dtype=torch.float32)
        stats = NormalizationStats(
            schema_version=1,
            power_density_mean=0.0,
            power_density_std=1.0,
            physics_mean=0.0,
            physics_std=1.0,
            residual_mean=0.0,
            residual_std=1.0,
            num_samples=1,
            num_grid_cells=16,
            input_channels=33,
            auxiliary_physics_v1_mean=0.0,
            auxiliary_physics_v1_std=1.0,
        )
        with self.assertRaises(ValueError):
            build_model_input(x, base, stats, physics_input_mode="source_superposition_plus_physics_v1")

    @unittest.skipUnless(SOURCE_BASE_INDEX.exists(), "source-superposition index is not present")
    def test_source_base_index_preserves_physics_v1_tensor(self) -> None:
        dataset = ChipThermDataset(SOURCE_BASE_INDEX, target="residual", return_metadata=True, return_graph=False)
        sample = dataset[0]
        self.assertIn("physics_v1", sample)
        self.assertEqual(tuple(sample["physics"].shape), (64, 64))
        self.assertEqual(tuple(sample["physics_v1"].shape), (64, 64))
        self.assertFalse(torch.equal(sample["physics"], sample["physics_v1"]))
        self.assertEqual(sample["metadata"]["effective_prediction_path"], sample["metadata"]["source_superposition_base_path"])


if __name__ == "__main__":
    unittest.main()
