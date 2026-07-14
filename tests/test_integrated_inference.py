#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.integrated_inference import (  # noqa: E402
    IntegratedChipThermModel,
    reconstruct_decomposed_temperature,
    rows_from_batch_metadata,
)


SOURCE_CHECKPOINT = REPO_ROOT / "outputs/source_response_operator_v1/prototype_seed1/checkpoints/best.pt"
RESIDUAL_CHECKPOINT = (
    REPO_ROOT / "outputs/source_superposition_full/source_superposition_cnn_gnn_seed1/checkpoints/best.pt"
)
CANONICAL_INDEX = (
    REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power/test_index.csv"
)
CACHED_INDEX = REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full/test_index.csv"


def _artifacts_available() -> bool:
    return SOURCE_CHECKPOINT.exists() and RESIDUAL_CHECKPOINT.exists() and CANONICAL_INDEX.exists()


@unittest.skipUnless(_artifacts_available(), "integrated inference checkpoints/data are not present")
class IntegratedInferenceSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.model = IntegratedChipThermModel(
            source_checkpoint=SOURCE_CHECKPOINT,
            residual_checkpoint=RESIDUAL_CHECKPOINT,
            device=torch.device("cpu"),
            deterministic=True,
        )

    def test_rows_from_batch_metadata(self) -> None:
        dataset = ChipThermDataset(CANONICAL_INDEX, target="residual", return_metadata=True, return_graph=True)
        batch = chiptherm_collate([dataset[0], dataset[1]])
        rows = rows_from_batch_metadata(batch["metadata"], 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["sample_uid"])
        self.assertTrue(rows[0]["dataset_source"])
        self.assertTrue(rows[0]["case_id"])

    def test_reconstruct_decomposed_temperature_adds_ambient_once(self) -> None:
        ambient = torch.tensor([300.0, 310.0])
        centered = torch.tensor(
            [
                [[1.0, -1.0], [2.0, -2.0]],
                [[3.0, 3.0], [-1.0, -1.0]],
            ]
        )
        outputs = {"mean_rise": torch.tensor([10.0, 20.0]), "centered_field": centered}
        temperature = reconstruct_decomposed_temperature(outputs, ambient)
        self.assertTrue(torch.allclose(temperature.mean(dim=(-2, -1)), ambient + outputs["mean_rise"]))

    def test_one_package_cpu_forward_shapes_and_finiteness(self) -> None:
        dataset = ChipThermDataset(CANONICAL_INDEX, target="residual", return_metadata=True, return_graph=True)
        sample = dataset[0]
        batch = chiptherm_collate([sample])
        rows = rows_from_batch_metadata(batch["metadata"], 1)
        result = self.model.predict_batch(batch, rows, source_batch_size=8, profile_components=True)

        for key in ("source_superposition_base_K", "final_temperature_K", "cnn_only_temperature_K"):
            value = result[key]
            self.assertEqual(tuple(value.shape), (1, 64, 64), key)
            self.assertTrue(torch.isfinite(value).all(), key)
        self.assertEqual(result["source_counts"][0], int(sample["metadata"]["num_chiplets"]))
        self.assertGreater(result["source_counts"][0], 0)
        self.assertIn("source_response_model_inference_s", result["timings"])
        self.assertIn("residual_total_forward_s", result["timings"])

    def test_residual_stage_accepts_generated_base(self) -> None:
        dataset = ChipThermDataset(CANONICAL_INDEX, target="residual", return_metadata=True, return_graph=True)
        batch = chiptherm_collate([dataset[0]])
        rows = rows_from_batch_metadata(batch["metadata"], 1)
        result = self.model.predict_batch(batch, rows, source_batch_size=8)
        residual = self.model.residual_from_base(batch, result["source_superposition_base_K"])
        self.assertEqual(tuple(residual["final_temperature_K"].shape), (1, 64, 64))
        self.assertTrue(torch.isfinite(residual["final_temperature_K"]).all())

    @unittest.skipUnless(CACHED_INDEX.exists(), "cached source-base index is not present")
    def test_cached_index_matches_canonical_order_for_first_row(self) -> None:
        with CANONICAL_INDEX.open("r", encoding="utf-8", newline="") as fp:
            canonical_row = next(csv.DictReader(fp))
        with CACHED_INDEX.open("r", encoding="utf-8", newline="") as fp:
            cached_row = next(csv.DictReader(fp))
        self.assertEqual(canonical_row["sample_uid"], cached_row["sample_uid"])
        self.assertEqual(cached_row.get("source_base_mode"), "source_superposition_v1")
        self.assertTrue(cached_row.get("source_superposition_base_path"))


if __name__ == "__main__":
    unittest.main()
