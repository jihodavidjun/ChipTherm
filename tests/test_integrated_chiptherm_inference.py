#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.integrated_inference import (  # noqa: E402
    accumulate_source_chunk,
    canonical_source_paths,
    reconstruct_decomposed_temperature,
    validate_total_power,
)


class IntegratedChipThermInferenceTest(unittest.TestCase):
    def test_residual_resistance_reconstruction_uses_source_base(self) -> None:
        source = torch.full((2, 2, 2), 300.0)
        ambient = torch.tensor([270.0, 280.0])
        outputs = {
            "mean_rise": torch.tensor([5.0, 7.0]),
            "centered_field": torch.tensor(
                [[[1.0, -1.0], [3.0, -3.0]], [[2.0, -2.0], [4.0, -4.0]]]
            ),
        }
        result = reconstruct_decomposed_temperature(
            outputs,
            ambient,
            source_base=source,
            mean_head_mode="residual_resistance",
        )
        expected = source + outputs["mean_rise"][:, None, None] + outputs["centered_field"]
        self.assertTrue(torch.equal(result, expected))
        self.assertFalse(torch.equal(result, ambient[:, None, None] + expected - source))

    def test_source_chunking_preserves_float64_accumulation(self) -> None:
        responses = np.arange(5 * 4, dtype=np.float32).reshape(5, 2, 2) / 7.0
        package_ids = [0, 1, 0, 1, 0]
        one = [np.zeros((2, 2), dtype=np.float64) for _ in range(2)]
        accumulate_source_chunk(one, responses, package_ids)
        chunked = [np.zeros((2, 2), dtype=np.float64) for _ in range(2)]
        accumulate_source_chunk(chunked, responses[:2], package_ids[:2])
        accumulate_source_chunk(chunked, responses[2:], package_ids[2:])
        self.assertTrue(np.array_equal(np.stack(one), np.stack(chunked)))

    def test_explicit_data_root_is_relocatable(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = Path(first_tmp)
            logical = Path("canonical/workloads/f001/w001/source")
            source = first / logical
            source.mkdir(parents=True)
            (source / "layout.json").write_text('{"chiplets": [{"name": "c0"}]}', encoding="utf-8")
            for name in ("power.yaml", "package.yaml", "hotspot.yaml"):
                (source / name).write_text("{}\n", encoding="utf-8")
            row = {"sample_uid": "u0", "source_dir": str(logical)}
            paths = canonical_source_paths(row, data_root=first)
            self.assertEqual(paths["layout"].resolve(), (source / "layout.json").resolve())

            second = Path(second_tmp)
            relocated = second / logical
            relocated.parent.mkdir(parents=True)
            source.rename(relocated)
            paths = canonical_source_paths(row, data_root=second)
            self.assertEqual(paths["layout"].resolve(), (relocated / "layout.json").resolve())

    def test_dataset_can_omit_targets_and_cached_workload_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".chiptherm_data_root.json").write_text(
                json.dumps({"path_semantics": "relative_to_declared_data_root"}),
                encoding="utf-8",
            )
            x_path = root / "x.npy"
            np.save(x_path, np.zeros((33, 64, 64), dtype=np.float32))
            index = root / "index.csv"
            row = {
                "sample_uid": "u0",
                "case_id": "f001",
                "dataset_source": "benchmark_v2",
                "x_path": "x.npy",
                "y_path": "missing_target.npy",
                "prediction_path": "missing_cached_base.npy",
                "residual_path": "missing_residual.npy",
                "total_power_W": "10.0",
            }
            with index.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            dataset = ChipThermDataset(
                index,
                return_graph=False,
                load_temperature=False,
                load_physics=False,
                load_residual=False,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["x"].shape), (33, 64, 64))
            self.assertNotIn("temperature", sample)
            self.assertNotIn("physics", sample)
            self.assertNotIn("target", sample)

    def test_total_power_validation(self) -> None:
        value = validate_total_power(
            torch.tensor([[1.0], [2.0]]),
            batch_size=2,
            device=torch.device("cpu"),
            non_blocking=False,
        )
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(value.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_total_power(
                torch.tensor([0.0]),
                batch_size=1,
                device=torch.device("cpu"),
                non_blocking=False,
            )


if __name__ == "__main__":
    unittest.main()
