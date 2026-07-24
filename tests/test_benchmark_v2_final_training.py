from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from chiptherm.benchmark_v2_training import (
    EXPECTED_PRIMARY_SPLIT,
    approve_source_checkpoint,
    assert_source_training_contract,
    deterministic_scaling_subsets,
    gnn_promotion_decision,
    prepare_final_training_indices,
)
from chiptherm.ml.source_response_dataset import (
    SourceResponseDataset,
    source_response_collate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV2FinalTrainingTests(unittest.TestCase):
    def test_exact_primary_family_split(self) -> None:
        self.assertEqual(len(EXPECTED_PRIMARY_SPLIT["train"]), 40)
        self.assertEqual(EXPECTED_PRIMARY_SPLIT["val"], ("f007", "f012", "f023", "f030", "f041"))
        self.assertEqual(EXPECTED_PRIMARY_SPLIT["test"], ("f008", "f016", "f027", "f033", "f044"))
        self.assertFalse(
            set(EXPECTED_PRIMARY_SPLIT["train"])
            & (set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"]))
        )

    def test_source_split_is_deterministic_and_oracle_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_root_marker(root)
            isolation = root / "canonical/stages/full_50x200/source_isolation"
            for split in ("train", "val", "test"):
                rows = []
                for family in EXPECTED_PRIMARY_SPLIT[split]:
                    for source_index in range(2):
                        rows.append(self._source_row(family, split, source_index))
                self._write_csv(isolation / f"{split}_index.csv", rows)
            first = prepare_final_training_indices(root, seed=17)
            second = prepare_final_training_indices(root, seed=17)
            self.assertEqual(first["fit_family_uids"], second["fit_family_uids"])
            self.assertEqual(first["internal_validation_family_uids"], second["internal_validation_family_uids"])
            forbidden = set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"])
            self.assertFalse(set(first["fit_family_uids"]) & forbidden)
            self.assertFalse(set(first["internal_validation_family_uids"]) & forbidden)
            self.assertTrue(
                all(
                    not Path(item["path"]).is_absolute()
                    for item in first["indices"].values()
                )
            )
            contract = assert_source_training_contract(
                root / first["indices"]["train"]["path"],
                root / first["indices"]["internal_val"]["path"],
                root / "derived/indices/full_50x200/source_response/split_manifest.json",
            )
            self.assertEqual(
                contract["normalization_family_uids"],
                first["fit_family_uids"],
            )
            with self.assertRaises(ValueError):
                assert_source_training_contract(
                    root / first["indices"]["oracle_val"]["path"],
                    root / first["indices"]["internal_val"]["path"],
                    root / "derived/indices/full_50x200/source_response/split_manifest.json",
                )

    def test_scaling_subsets_are_nested_and_frozen(self) -> None:
        subsets = deterministic_scaling_subsets(EXPECTED_PRIMARY_SPLIT["train"], 20260721)
        self.assertEqual(sorted(subsets), ["10", "20", "30", "40", "5"])
        self.assertTrue(set(subsets["5"]) <= set(subsets["10"]) <= set(subsets["20"]))
        self.assertTrue(set(subsets["20"]) <= set(subsets["30"]) <= set(subsets["40"]))
        self.assertEqual(
            subsets,
            deterministic_scaling_subsets(EXPECTED_PRIMARY_SPLIT["train"], 20260721),
        )

    def test_explicit_data_root_supports_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_root = Path(tmp) / "root_a"
            second_root = Path(tmp) / "root_b"
            logical = Path("canonical/fixture")
            self._write_source_fixture(first_root, logical)
            self._write_source_fixture(second_root, logical)
            index = second_root / "derived/source_index.csv"
            self._write_csv(index, [self._dataset_row(logical)])
            dataset = SourceResponseDataset(index, data_root=second_root)
            sample = dataset[0]
            self.assertEqual(tuple(sample["x"].shape), (17, 64, 64))
            batch = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_size=1,
                        collate_fn=source_response_collate,
                    )
                )
            )
            self.assertTrue(torch.isfinite(batch["x"]).all())

    def test_optional_gnn_is_disabled_and_gate_requires_all_safeguards(self) -> None:
        config = yaml.safe_load(
            (
                REPO_ROOT
                / "configs/benchmark_v2_50family/training/optional_gnn_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(config["enabled_by_default"])
        cnn = []
        gnn = []
        for family_index, family in enumerate(EXPECTED_PRIMARY_SPLIT["test"]):
            for sample in range(8):
                uid = f"{family}_{sample}"
                cnn.append(
                    {
                        "sample_uid": uid,
                        "family_uid": family,
                        "mae_K": 4.0,
                        "rmse_K": 5.0,
                        "peak_temperature_abs_error_K": 6.0,
                    }
                )
                gnn.append(
                    {
                        "sample_uid": uid,
                        "family_uid": family,
                        "mae_K": 3.75,
                        "rmse_K": 4.9,
                        "peak_temperature_abs_error_K": 5.9,
                    }
                )
        decision = gnn_promotion_decision(
            cnn,
            gnn,
            runtime_overhead_fraction=0.10,
            memory_overhead_fraction=0.10,
            bootstrap_samples=100,
            seed=3,
        )
        self.assertTrue(decision["promote"])
        rejected = gnn_promotion_decision(
            cnn,
            gnn,
            runtime_overhead_fraction=0.50,
            memory_overhead_fraction=0.10,
            bootstrap_samples=100,
            seed=3,
        )
        self.assertFalse(rejected["promote"])
        self.assertEqual(rejected["recommendation"], "OMIT GNN FROM PRIMARY MODEL")

    def test_checkpoint_approval_rejects_oracle_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints/best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "model_state_dict": {"weight": torch.ones(1)},
                    "normalization": {"mean": 0.0},
                },
                checkpoint,
            )
            lineage = root / "training_lineage.json"
            lineage.write_text(
                json.dumps(
                    {
                        "training_family_uids": ["f001", "f007"],
                        "normalization_family_uids": ["f001"],
                        "selection_family_uids": ["f002"],
                        "oracle_metrics_used_for_selection": False,
                    }
                ),
                encoding="utf-8",
            )
            evaluation = root / "evaluation"
            for name in ("train", "internal_val", "oracle_primary_val", "oracle_primary_test"):
                path = evaluation / name
                path.mkdir(parents=True)
                (path / "metrics.json").write_text(
                    json.dumps({"package_reconstruction": {"mae_K": 1.0}}),
                    encoding="utf-8",
                )
            with self.assertRaises(ValueError):
                approve_source_checkpoint(
                    checkpoint,
                    lineage_path=lineage,
                    evaluation_root=evaluation,
                    output_path=root / "approval.json",
                    allow_caveats=True,
                )
            approval = json.loads((root / "approval.json").read_text(encoding="utf-8"))
            self.assertEqual(approval["approval_status"], "REJECTED")
            self.assertFalse(approval["checks"]["lineage_no_oracle_leakage"])

    @staticmethod
    def _write_root_marker(root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / ".chiptherm_data_root.json").write_text(
            json.dumps(
                {
                    "benchmark_id": "benchmark_v2_50family",
                    "path_semantics": "relative_to_declared_data_root",
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _source_row(family: str, split: str, source_index: int) -> dict[str, str]:
        uid = f"{family}_source_{source_index}"
        return {
            "source_response_uid": uid,
            "original_sample_uid": family,
            "sample_uid": uid,
            "family_uid": family,
            "case_id": family,
            "split": split,
            "source_index": str(source_index),
            "target_rise_path": f"canonical/source_targets/{uid}.npy",
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_source_fixture(root: Path, logical: Path) -> None:
        path = root / logical
        path.mkdir(parents=True, exist_ok=True)
        x = np.zeros((13, 64, 64), dtype=np.float32)
        x[1] = 1.0
        np.save(path / "x.npy", x)
        np.save(path / "rise.npy", np.ones((64, 64), dtype=np.float32))
        np.save(path / "target.npy", np.full((64, 64), 320.0, dtype=np.float32))
        layout = {
            "package": {"size": {"width": 16.0, "height": 16.0}},
            "chiplets": [
                {
                    "name": "CPU0",
                    "position": {"x": 4.0, "y": 4.0},
                    "size": {"width": 4.0, "height": 4.0},
                }
            ],
        }
        (path / "layout.json").write_text(json.dumps(layout), encoding="utf-8")

    @staticmethod
    def _dataset_row(logical: Path) -> dict[str, str]:
        return {
            "source_response_uid": "fixture_source_0",
            "original_sample_uid": "fixture",
            "case_id": "f001",
            "source_index": "0",
            "source_name": "CPU0",
            "source_power_W": "10.0",
            "ambient_K": "318.15",
            "num_chiplets": "1",
            "original_x_path": str(logical / "x.npy"),
            "target_rise_path": str(logical / "rise.npy"),
            "full_temperature_path": str(logical / "target.npy"),
            "layout_path": str(logical / "layout.json"),
        }


if __name__ == "__main__":
    unittest.main()
