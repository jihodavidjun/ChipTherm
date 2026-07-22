from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from chiptherm.benchmark_v2_pipeline import (
    FULL_STAGE,
    PHASE3_STAGE,
    PilotPaths,
    accepted_pilot_immutability_snapshot,
    build_isolation_inputs,
    estimate_full_build_resources,
    generate_hotspot_samples,
    load_selection,
    pilot_hotspot_catalog,
    read_csv,
    reusable_pilot_sample,
    write_canonical_sample_source,
    write_csv,
)
from chiptherm.benchmark_v2_workloads import (
    full_workload_cells,
    generate_full_family_workloads,
    generate_scale_family_workloads,
    load_family,
    validate_workload,
)
from scripts.visualize_benchmark_v2_samples import select_audit_rows


FAMILY_DIR = REPO_ROOT / "configs/benchmark_v2_50family/families"
SELECTION_PATH = REPO_ROOT / "configs/benchmark_v2_50family/full_50x200.yaml"


class BenchmarkV2Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = load_selection(SELECTION_PATH)
        cls.family_uids = [str(row["family_uid"]) for row in cls.selection["selected_families"]]
        cls.families = {uid: load_family(FAMILY_DIR / f"{uid}.yaml") for uid in cls.family_uids}

    def test_frozen_matrix_and_primary_split(self) -> None:
        cells = full_workload_cells()
        self.assertEqual(len(cells), 200)
        self.assertEqual(len({row["workload_cell"] for row in cells}), 200)
        self.assertEqual(len(self.family_uids), 50)
        counts = {
            split: sum(row["primary_split"] == split for row in self.selection["selected_families"])
            for split in ("train", "val", "test")
        }
        self.assertEqual(counts, {"train": 40, "val": 5, "test": 5})

    def test_every_family_has_200_unique_valid_workloads_and_phase3_prefix(self) -> None:
        expected_cells = {row["workload_cell"] for row in full_workload_cells()}
        for uid, family in self.families.items():
            before = json.dumps(family["fixed_structure"]["layout"], sort_keys=True)
            full = generate_full_family_workloads(family)
            phase3 = generate_scale_family_workloads(family)
            self.assertEqual(len(full), 200, uid)
            self.assertEqual({row["workload_cell"] for row in full}, expected_cells, uid)
            self.assertEqual(len({row["content_hash"] for row in full}), 200, uid)
            self.assertEqual(
                [row["content_hash"] for row in full[:50]],
                [row["content_hash"] for row in phase3],
                uid,
            )
            self.assertFalse([problem for row in full for problem in validate_workload(row, family)], uid)
            self.assertEqual(before, json.dumps(family["fixed_structure"]["layout"], sort_keys=True), uid)

    def test_source_policy_and_once_per_family_inputs(self) -> None:
        policy = self.selection["source_response_policy"]
        self.assertEqual(len(policy["train_eligible_families"]), 40)
        self.assertEqual(len(policy["oracle_only_families"]), 10)
        raw_rows = [
            {"family_uid": uid, "sample_uid": f"{uid}_a", "workload_uid": "w001"}
            for uid in self.family_uids
        ] + [
            {"family_uid": uid, "sample_uid": f"{uid}_b", "workload_uid": "w002"}
            for uid in self.family_uids
        ]
        with tempfile.TemporaryDirectory() as temporary:
            outputs = build_isolation_inputs(raw_rows, self.selection, Path(temporary), Path(temporary))
            counts = {split: len(read_csv(path)) for split, path in outputs.items()}
        self.assertEqual(counts, {"train": 40, "val": 5, "test": 5})

    def test_phase3_sample_reuse_requires_exact_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            family = self.families["f002"]
            workload = generate_full_family_workloads(family)[0]
            sample_root = root / f"canonical/stages/{PHASE3_STAGE}/hotspot_labels/f002/{workload['sample_uid']}"
            write_canonical_sample_source(sample_root, family, workload)
            (sample_root / "parsed").mkdir()
            np.save(sample_root / "parsed/temp_layer0.npy", np.full((64, 64), 320.0, dtype=np.float32))
            source_hashes = {
                name: __import__("hashlib").sha256((sample_root / "source" / name).read_bytes()).hexdigest()
                for name in ("layout.json", "power.yaml", "package.yaml", "hotspot.yaml", "family.yaml", "workload.yaml")
            }
            (sample_root / "manifest.json").write_text(
                json.dumps({"status": "validated", "sample_uid": workload["sample_uid"], "source_hashes": source_hashes}),
                encoding="utf-8",
            )
            write_csv(
                root / f"canonical/stages/{PHASE3_STAGE}/hotspot_labels/sample_index.csv",
                [{
                    "sample_uid": workload["sample_uid"], "family_uid": "f002",
                    "workload_content_sha256": workload["content_hash"],
                    "sample_root": str(sample_root.relative_to(root)), "ownership_stage": PHASE3_STAGE,
                }],
            )
            paths = PilotPaths(root, root / "staging", "full", FULL_STAGE)
            reused = reusable_pilot_sample(paths, family, workload, pilot_hotspot_catalog(root))
            self.assertIsNotNone(reused)
            self.assertEqual(reused[1], PHASE3_STAGE)

    def test_full_namespace_and_partial_scheduler_preserve_pilots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = root / f"canonical/stages/{PHASE3_STAGE}/sentinel.txt"
            accepted.parent.mkdir(parents=True)
            accepted.write_text("accepted", encoding="utf-8")
            before = accepted_pilot_immutability_snapshot(root)
            family = self.families["f050"]
            workloads = generate_full_family_workloads(family)[:4]
            report = generate_hotspot_samples(
                PilotPaths(root, root / "staging", "full", FULL_STAGE),
                {"f050": family},
                workloads,
                hotspot_home=None,
                config_template=REPO_ROOT / "configs/hotspot_base.config",
                workers=1,
                resume=True,
                dry_run=True,
                execution_family_uids=("f050",),
                max_new_package_runs=2,
            )
            self.assertEqual(report["dry_run_source_count"], 2)
            self.assertEqual(report["deferred"], 2)
            self.assertFalse(report["complete"])
            self.assertEqual(before["content_sha256"], accepted_pilot_immutability_snapshot(root)["content_sha256"])

    def test_storage_gate_uses_500_package_and_source_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / f"canonical/manifests/{PHASE3_STAGE}_strict_validation.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps({
                    "passed": True,
                    "actual_sample_count": 500,
                    "source_isolation_target_count": 242,
                    "storage": {"bytes_by_artifact_class": {}, "staging_peak_bytes_observed": 0},
                    "runtime": {"wall_clock_s": 450.0},
                }),
                encoding="utf-8",
            )
            report = estimate_full_build_resources(
                root,
                min_free_gb=0,
                min_free_fraction=0,
                max_retained_gb=10_000,
                max_staging_gb=10_000,
            )
            self.assertEqual(report["reusable_package_samples"], 500)
            self.assertEqual(report["new_package_runs"], 9500)
            self.assertEqual(report["reusable_source_rows"], 242)
            self.assertEqual(report["new_source_runs"], 895)
            self.assertEqual(report["recommendation"], "GO")
            completed = root / f"canonical/stages/{FULL_STAGE}/hotspot_labels/f001/f001_w001/parsed"
            completed.mkdir(parents=True)
            np.save(completed / "temp_layer0.npy", np.full((64, 64), 320.0, dtype=np.float32))
            (completed.parent / "manifest.json").write_text("{}", encoding="utf-8")
            resumed = estimate_full_build_resources(
                root,
                min_free_gb=0,
                min_free_fraction=0,
                max_retained_gb=10_000,
                max_staging_gb=10_000,
            )
            self.assertEqual(resumed["completed_full_package_samples"], 1)
            self.assertEqual(resumed["new_package_runs"], 9499)

    def test_full_visual_selection_covers_all_families(self) -> None:
        rows = [
            {
                "family_uid": uid,
                "sample_uid": row["sample_uid"],
                "workload_uid": row["workload_uid"],
                "workload_cell": row["workload_cell"],
            }
            for uid, family in self.families.items()
            for row in generate_full_family_workloads(family)
        ]
        selected = select_audit_rows(rows, None, stage=FULL_STAGE)
        self.assertEqual(len(selected), 50)
        self.assertEqual({row["family_uid"] for row in selected}, set(self.family_uids))


if __name__ == "__main__":
    unittest.main()
