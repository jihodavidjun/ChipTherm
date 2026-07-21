from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chiptherm.benchmark_v2_pipeline import (  # noqa: E402
    CANONICAL_METADATA_FEATURES,
    PilotBuildOptions,
    audit_index_paths,
    audit_portable_documents,
    build_pilot,
    load_selection,
    relocate_pilot,
    sha256_file,
    validate_artifact_manifest,
    validate_pilot_root,
    validate_source_checkpoint_lineage,
    verify_parent_lock,
)
from chiptherm.benchmark_v2_workloads import (  # noqa: E402
    PILOT_STRATA,
    generate_family_workloads,
    load_family,
    validate_workload,
)
from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.encoder import CHANNEL_NAMES  # noqa: E402
from chiptherm.ml.graph_models import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES  # noqa: E402
from chiptherm.parsers import parse_layer_grid  # noqa: E402


SELECTION_PATH = REPO_ROOT / "configs/benchmark_v2_50family/pilot_5x10.yaml"
FAMILY_DIR = REPO_ROOT / "configs/benchmark_v2_50family/families"
LOCK_PATH = REPO_ROOT / "configs/benchmark_v2_50family/dependency_lock.json"


class BenchmarkV2Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = load_selection(SELECTION_PATH)
        cls.family_uids = [row["family_uid"] for row in cls.selection["selected_families"]]
        cls.families = {uid: load_family(FAMILY_DIR / f"{uid}.yaml") for uid in cls.family_uids}

    def test_workloads_are_deterministic_stratified_and_valid(self) -> None:
        for uid, family in self.families.items():
            first = generate_family_workloads(family, base_seed=20260721)
            second = generate_family_workloads(family, base_seed=20260721)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 10)
            self.assertEqual([row["stratum"] for row in first], [item.key for item in PILOT_STRATA])
            self.assertEqual(len({row["content_hash"] for row in first}), 10)
            self.assertTrue(all(float(row["total_package_power_W"]) > 0 for row in first))
            for row in first:
                self.assertEqual(validate_workload(row, family), [])
                self.assertEqual(set(row["chiplet_power_W"]), {item["name"] for item in family["fixed_structure"]["layout"]["chiplets"]})

    def test_family_geometry_is_identical_across_workloads(self) -> None:
        family = self.families["f002"]
        before = json.dumps(family["fixed_structure"]["layout"], sort_keys=True)
        generate_family_workloads(family, base_seed=1)
        after = json.dumps(family["fixed_structure"]["layout"], sort_keys=True)
        self.assertEqual(before, after)

    def test_parent_lock_and_artifact_schema(self) -> None:
        lock = verify_parent_lock(LOCK_PATH, family_manifest_path=FAMILY_DIR.parent / "family_manifest.yaml")
        self.assertEqual(lock["schema_contracts"]["impedance_channels"], 33)
        fixture = {
            "schema_version": "benchmark_v2_artifact_manifest/0.1",
            "artifact_id": "pilot_fixture",
            "benchmark_id": "benchmark_v2_50family",
            "artifact_class": "evaluation_only",
            "stage": "evaluation",
            "status": "validated",
            "created_at": "2026-07-21T00:00:00+00:00",
            "producer": {},
            "storage": {"root_id": "fixture", "relative_path": "evaluations/fixture", "path_semantics": "relative_to_declared_data_root", "persistent": True},
            "parents": [],
            "content": {},
            "validation": {"passed": True},
            "reproducibility": {},
        }
        validate_artifact_manifest(fixture)
        broken = dict(fixture)
        broken["storage"] = {**fixture["storage"], "relative_path": "/absolute"}
        with self.assertRaises(ValueError):
            validate_artifact_manifest(broken)

    def test_hotspot_grid_parser_shape_and_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.steady"
            path.write_text("Layer 0:\n" + "\n".join(f"n{i} {float(i)}" for i in range(12)) + "\n", encoding="utf-8")
            array = parse_layer_grid(path, layer=0, rows=3, cols=4)
            self.assertEqual(array.shape, (3, 4))
            np.testing.assert_array_equal(array[0], np.asarray([0.0, 1.0, 2.0, 3.0]))
            self.assertEqual(float(array[2, 3]), 11.0)

    def test_schema_dimensions_and_channel_order(self) -> None:
        self.assertEqual(len(CHANNEL_NAMES), 13)
        self.assertEqual(CHANNEL_NAMES[:8], [
            "power_density_W_per_mm2", "occupancy_mask", "CPU_mask", "GPU_or_NPU_mask",
            "memory_mask", "IO_or_ANALOG_or_MEMS_mask", "normalized_x_coordinate", "normalized_y_coordinate",
        ])
        self.assertEqual(len(CANONICAL_METADATA_FEATURES), 15)
        self.assertEqual(len(NODE_FEATURE_NAMES), 24)
        self.assertEqual(len(EDGE_FEATURE_NAMES), 15)

    def test_source_lineage_rejects_forbidden_family(self) -> None:
        checkpoint = REPO_ROOT / "outputs/source_response_operator_v1/prototype_seed1/checkpoints/best.pt"
        if not checkpoint.exists():
            self.skipTest("historical source-response checkpoint is not present")
        lineage_path = REPO_ROOT / "configs/benchmark_v2_50family/source_response_lineage_prototype_seed1.json"
        lineage = validate_source_checkpoint_lineage(checkpoint, lineage_path, self.selection)
        self.assertEqual(lineage["pilot_usage"], "frozen_inference_only")
        with tempfile.TemporaryDirectory() as temporary:
            bad = dict(lineage)
            bad["training_family_uids"] = ["f044"]
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leaks pilot val/test"):
                validate_source_checkpoint_lineage(checkpoint, path, self.selection)

    def test_dry_run_resume_and_no_silent_omission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "pilot"
            first = self._dry_options(root, "first", resume=False)
            report = build_pilot(first)
            self.assertEqual(report["workload_count"], 50)
            self.assertEqual(len(list((root / "staging/runs/first/hotspot_labels").glob("f*/f*/source/scenario.yaml"))), 50)
            self.assertFalse((root / "canonical/hotspot_labels").exists())
            second = self._dry_options(root, "second", resume=True)
            resumed = build_pilot(second)
            self.assertEqual(resumed["workload_count"], 50)
            strict = validate_pilot_root(root, allow_dry_run=True)
            self.assertTrue(strict["passed"])
            missing = next((root / "canonical/workloads/f002").glob("w001_*.yaml"))
            missing.unlink()
            with self.assertRaises(ValueError):
                build_pilot(self._dry_options(root, "third", resume=True))

    def test_root_relative_loader_collates_after_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source_root"
            index = self._make_loader_fixture(source)
            dataset = ChipThermDataset(index, target="residual", return_graph=True)
            batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=chiptherm_collate)))
            self.assertEqual(tuple(batch["x"].shape), (4, 33, 64, 64))
            self.assertEqual(tuple(batch["metadata_vector"].shape), (4, 15))
            self.assertEqual(batch["graph"]["node_features"].shape[1], 24)
            destination = Path(temporary) / "relocated_root"
            shutil.copytree(source, destination)
            relocated_index = destination / index.relative_to(source)
            relocated = ChipThermDataset(relocated_index, target="residual", return_graph=True)
            sample = relocated[0]
            self.assertEqual(tuple(sample["x"].shape), (33, 64, 64))
            self.assertTrue(torch.isfinite(sample["physics"]).all())
            self.assertEqual(audit_index_paths(relocated.rows, destination), [])
            self.assertEqual(audit_portable_documents(destination)["violation_count"], 0)

    def _dry_options(self, root: Path, run_id: str, *, resume: bool) -> PilotBuildOptions:
        return PilotBuildOptions(
            config_path=REPO_ROOT / "configs/benchmark_v2_50family/design_proposal.yaml",
            selection_path=SELECTION_PATH,
            family_dir=FAMILY_DIR,
            parent_lock_path=LOCK_PATH,
            data_root=root,
            scratch_root=root / "staging",
            hotspot_home=None,
            config_template=REPO_ROOT / "configs/hotspot_base.config",
            selected_families=tuple(self.family_uids),
            seed=20260721,
            workers=2,
            resume=resume,
            dry_run=True,
            keep_hotspot_workdirs=True,
            run_id=run_id,
        )

    def _make_loader_fixture(self, root: Path) -> Path:
        (root / "derived/indices/pilot_5x10").mkdir(parents=True)
        (root / "arrays").mkdir()
        (root / "graphs").mkdir()
        marker = {
            "schema_version": "chiptherm_data_root/1",
            "benchmark_id": "benchmark_v2_50family",
            "root_id": "fixture",
            "path_semantics": "relative_to_declared_data_root",
        }
        (root / ".chiptherm_data_root.json").write_text(json.dumps(marker), encoding="utf-8")
        rows: list[dict[str, str]] = []
        metadata_rows: list[dict[str, str]] = []
        for index in range(4):
            uid = f"fixture_{index}"
            np.save(root / f"arrays/{uid}_x.npy", np.zeros((33, 64, 64), dtype=np.float32))
            np.save(root / f"arrays/{uid}_y.npy", np.full((64, 64), 320.0 + index, dtype=np.float32))
            np.save(root / f"arrays/{uid}_base.npy", np.full((64, 64), 319.0, dtype=np.float32))
            np.savez(
                root / f"graphs/{uid}.npz",
                node_features=np.zeros((2, 24), dtype=np.float32),
                edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                edge_features=np.zeros((2, 15), dtype=np.float32),
                chiplet_rects=np.asarray([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.float32),
                package_size=np.asarray([10, 10], dtype=np.float32),
            )
            rows.append({
                "sample_uid": uid, "original_sample_uid": uid, "case_id": "f002", "dataset_source": "fixture", "split": "test",
                "x_path": f"arrays/{uid}_x.npy", "y_path": f"arrays/{uid}_y.npy", "prediction_path": "", "residual_path": "",
                "source_superposition_base_path": f"arrays/{uid}_base.npy", "source_base_mode": "source_superposition_v1",
                "graph_path": f"graphs/{uid}.npz", "num_chiplets": "2", "total_power_W": "10",
            })
            metadata_rows.append({"sample_uid": uid, "case_id": "f002", "split": "test", **{name: "1.0" for name in CANONICAL_METADATA_FEATURES}})
        index_root = root / "derived/indices/pilot_5x10"
        self._write_csv(index_root / "all_index.csv", rows)
        self._write_csv(index_root / "metadata_features.csv", metadata_rows)
        (index_root / "metadata_manifest.json").write_text(json.dumps({"active_features": list(CANONICAL_METADATA_FEATURES)}), encoding="utf-8")
        (index_root / "feature_manifest.json").write_text(json.dumps({"channel_names": [f"channel_{i}" for i in range(33)]}), encoding="utf-8")
        (index_root / "graph_manifest.json").write_text(json.dumps({"node_feature_names": list(NODE_FEATURE_NAMES), "edge_feature_names": list(EDGE_FEATURE_NAMES)}), encoding="utf-8")
        return index_root / "all_index.csv"

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
