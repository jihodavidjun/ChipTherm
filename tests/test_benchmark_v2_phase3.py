from __future__ import annotations

import json
import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chiptherm.benchmark_v2_pipeline import (  # noqa: E402
    CANONICAL_METADATA_FEATURES,
    PHASE3_STAGE,
    PilotBuildOptions,
    PilotPaths,
    audit_portable_documents,
    build_pilot,
    durable_stage_complete,
    generate_hotspot_samples,
    load_selection,
    make_tree_manifest,
    phase2_immutability_snapshot,
    project_scale_metrics,
    repair_pilot_portability,
    sha256_file,
    stage_spec,
    validate_scale_pilot_root,
    write_artifact_manifest,
    write_canonical_sample_source,
)
from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.graph_models import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES  # noqa: E402
from chiptherm.benchmark_v2_workloads import (  # noqa: E402
    generate_family_workloads,
    generate_scale_family_workloads,
    scale_workload_cells,
    validate_workload,
)
from scripts.visualize_benchmark_v2_samples import draw_layout, select_audit_rows  # noqa: E402


FAMILY_DIR = REPO_ROOT / "configs/benchmark_v2_50family/families"
SELECTION_PATH = REPO_ROOT / "configs/benchmark_v2_50family/pilot_10x50.yaml"
LOCK_PATH = REPO_ROOT / "configs/benchmark_v2_50family/dependency_lock.json"


class BenchmarkV2Phase3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = load_selection(SELECTION_PATH)
        cls.family_uids = [str(row["family_uid"]) for row in cls.selection["selected_families"]]
        cls.families = {
            uid: yaml.safe_load((FAMILY_DIR / f"{uid}.yaml").read_text(encoding="utf-8"))
            for uid in cls.family_uids
        }

    def test_frozen_selection_and_source_policy(self) -> None:
        self.assertEqual(self.family_uids, ["f002", "f007", "f009", "f014", "f023", "f029", "f032", "f039", "f040", "f044"])
        self.assertEqual(stage_spec(PHASE3_STAGE).sample_count, 500)
        policy = self.selection["source_response_policy"]
        self.assertEqual(set(policy["train_eligible_families"]), {uid for uid in self.family_uids if self.families[uid]["primary_split"] == "train"})
        self.assertEqual(set(policy["oracle_only_families"]), {uid for uid in self.family_uids if self.families[uid]["primary_split"] != "train"})

    def test_scale_matrix_is_deterministic_unique_and_complete(self) -> None:
        cells = scale_workload_cells()
        self.assertEqual(len(cells), 50)
        self.assertEqual(len({row["workload_cell"] for row in cells}), 50)
        self.assertEqual({row["power_regime"] for row in cells}, {"phase2_reference", "very_low", "moderate", "high", "stress"})
        self.assertEqual(len({row["topology_regime"] for row in cells}), 10)
        for family in self.families.values():
            before = json.dumps(family["fixed_structure"]["layout"], sort_keys=True)
            first = generate_scale_family_workloads(family)
            second = generate_scale_family_workloads(family)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 50)
            self.assertEqual(len({row["content_hash"] for row in first}), 50)
            self.assertEqual({row["workload_cell"] for row in first}, {row["workload_cell"] for row in cells})
            self.assertTrue(all(validate_workload(row, family) == [] for row in first))
            self.assertEqual(before, json.dumps(family["fixed_structure"]["layout"], sort_keys=True))

    def test_first_ten_reference_cells_match_phase2_hashes(self) -> None:
        for family in self.families.values():
            phase2 = generate_family_workloads(family)
            phase3 = generate_scale_family_workloads(family)[:10]
            self.assertEqual([row["sample_uid"] for row in phase3], [row["sample_uid"] for row in phase2])
            self.assertEqual([row["content_hash"] for row in phase3], [row["content_hash"] for row in phase2])
            self.assertEqual([row["chiplet_power_W"] for row in phase3], [row["chiplet_power_W"] for row in phase2])

    def test_stage_paths_do_not_collide_with_phase2(self) -> None:
        root = Path("/benchmark-root")
        phase3 = PilotPaths(root, root / "staging", "run", PHASE3_STAGE)
        self.assertEqual(phase3.canonical("workloads"), root / "canonical/stages/pilot_10x50/workloads")
        self.assertEqual(phase3.derived("context_33ch"), root / "derived/stages/pilot_10x50/context_33ch")
        self.assertEqual(phase3.derived("indices") / PHASE3_STAGE, root / "derived/indices/pilot_10x50")
        phase2 = PilotPaths(root, root / "staging", "run")
        self.assertEqual(phase2.canonical("workloads"), root / "canonical/workloads")

    def test_full_build_projection_uses_measured_scale_factor(self) -> None:
        projected = project_scale_metrics(retained_bytes=500.0, peak_staging_bytes=1000.0, wall_clock_s=25.0)
        self.assertEqual(projected["projected_retained_bytes"], 10000.0)
        self.assertEqual(projected["projected_peak_staging_bytes"], 20000.0)
        self.assertEqual(projected["projected_wall_clock_s"], 500.0)

    def test_phase2_reference_sample_is_reused_by_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            paths = PilotPaths(root, root / "staging", "reuse", PHASE3_STAGE)
            family = self.families["f002"]
            workload = generate_scale_family_workloads(family)[0]
            phase2_sample = root / "canonical/hotspot_labels/f002" / workload["sample_uid"]
            write_canonical_sample_source(phase2_sample, family, workload)
            (phase2_sample / "parsed").mkdir()
            np.save(phase2_sample / "parsed/temp_layer0.npy", np.full((64, 64), 320.0, dtype=np.float32))
            (phase2_sample / "manifest.json").write_text(
                json.dumps({"status": "validated", "sample_uid": workload["sample_uid"]}),
                encoding="utf-8",
            )
            report = generate_hotspot_samples(
                paths,
                {"f002": family},
                [workload],
                hotspot_home=None,
                config_template=REPO_ROOT / "configs/hotspot_base.config",
                workers=1,
                resume=True,
                dry_run=True,
            )
            self.assertEqual(report["reused_phase2"], 1)
            self.assertEqual(report["dry_run_source_count"], 0)

    def test_500_sample_dry_run_and_phase2_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            legacy = root / "canonical/workloads/phase2_fixture.txt"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("accepted", encoding="utf-8")
            before = phase2_immutability_snapshot(root)
            options = PilotBuildOptions(
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
                resume=True,
                dry_run=True,
                keep_hotspot_workdirs=True,
                run_id="phase3-dry",
                stage=PHASE3_STAGE,
            )
            report = build_pilot(options)
            self.assertEqual(report["workload_count"], 500)
            self.assertEqual(len(list((root / "staging/runs/phase3-dry/hotspot_labels").glob("f*/f*/source/scenario.yaml"))), 500)
            self.assertFalse((root / "canonical/workloads/workload_manifest.yaml").exists())
            self.assertEqual(before["content_sha256"], phase2_immutability_snapshot(root)["content_sha256"])
            resumed_options = PilotBuildOptions(**{**options.__dict__, "run_id": "phase3-dry-resume"})
            resumed = build_pilot(resumed_options)
            self.assertEqual(resumed["workload_count"], 500)
            self.assertEqual(resumed["hotspot"]["dry_run_source_count"], 500)
            strict = validate_scale_pilot_root(root, allow_dry_run=True)
            self.assertTrue(strict["passed"])

    def test_visual_audit_selection_covers_all_families_and_regimes(self) -> None:
        rows = []
        for family_uid in self.family_uids:
            for workload in generate_scale_family_workloads(self.families[family_uid]):
                rows.append({key: str(workload.get(key, "")) for key in ("family_uid", "sample_uid", "workload_uid", "workload_cell")})
        selected = select_audit_rows(rows, None)
        self.assertEqual(len(selected), 10)
        self.assertEqual({row["family_uid"] for row in selected}, set(self.family_uids))
        self.assertTrue(any("single" in row["workload_cell"] for row in selected))
        self.assertTrue(any("cluster" in row["workload_cell"] for row in selected))

    def test_visual_layout_smoke_when_matplotlib_is_available(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib is not installed in the lightweight test environment")
        family = self.families["f002"]
        workload = generate_scale_family_workloads(family)[0]
        figure, axis = plt.subplots(figsize=(3, 3))
        draw_layout(axis, family["fixed_structure"]["layout"], workload["chiplet_power_W"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "layout.png"
            figure.savefig(output)
            self.assertGreater(output.stat().st_size, 100)
        plt.close(figure)

    def test_phase3_index_loads_after_root_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            index_root = source / "derived/indices/pilot_10x50"
            arrays = source / "arrays"
            graphs = source / "graphs"
            for path in (index_root, arrays, graphs):
                path.mkdir(parents=True, exist_ok=True)
            (source / ".chiptherm_data_root.json").write_text(
                json.dumps({"benchmark_id": "benchmark_v2_50family", "path_semantics": "relative_to_declared_data_root"}),
                encoding="utf-8",
            )
            np.save(arrays / "x.npy", np.zeros((33, 64, 64), dtype=np.float32))
            np.save(arrays / "y.npy", np.full((64, 64), 320.0, dtype=np.float32))
            np.save(arrays / "base.npy", np.full((64, 64), 319.0, dtype=np.float32))
            np.savez(
                graphs / "graph.npz",
                node_features=np.zeros((2, 24), dtype=np.float32),
                edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                edge_features=np.zeros((2, 15), dtype=np.float32),
                chiplet_rects=np.ones((2, 4), dtype=np.float32),
                package_size=np.asarray([10.0, 10.0], dtype=np.float32),
            )
            row = {
                "sample_uid": "f002_w001_low_balanced", "original_sample_uid": "f002_w001_low_balanced",
                "family_uid": "f002", "case_id": "f002", "dataset_source": "benchmark_v2_50family", "split": "test",
                "x_path": "arrays/x.npy", "y_path": "arrays/y.npy", "source_superposition_base_path": "arrays/base.npy",
                "source_base_mode": "source_superposition_v1", "prediction_path": "", "residual_path": "",
                "graph_path": "graphs/graph.npz", "num_chiplets": "2", "total_power_W": "10",
            }
            with (index_root / "all_index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            metadata = {"sample_uid": row["sample_uid"], "case_id": "f002", "split": "test", **{name: "1" for name in CANONICAL_METADATA_FEATURES}}
            with (index_root / "metadata_features.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metadata))
                writer.writeheader()
                writer.writerow(metadata)
            (index_root / "metadata_manifest.json").write_text(json.dumps({"active_features": list(CANONICAL_METADATA_FEATURES)}), encoding="utf-8")
            (index_root / "feature_manifest.json").write_text(json.dumps({"channel_names": [f"channel_{i}" for i in range(33)]}), encoding="utf-8")
            (index_root / "graph_manifest.json").write_text(json.dumps({"node_feature_names": list(NODE_FEATURE_NAMES), "edge_feature_names": list(EDGE_FEATURE_NAMES)}), encoding="utf-8")
            destination = Path(temporary) / "relocated"
            shutil.copytree(source, destination)
            dataset = ChipThermDataset(destination / "derived/indices/pilot_10x50/all_index.csv", target="residual", return_graph=True)
            sample = dataset[0]
            self.assertEqual(tuple(sample["x"].shape), (33, 64, 64))
            self.assertEqual(tuple(sample["metadata_vector"].shape), (15,))
            self.assertEqual(sample["graph"]["node_features"].shape[1], 24)

    def test_phase3_portability_repair_is_scoped_idempotent_and_refreshes_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            manifests = root / "canonical/manifests"
            artifact_manifests = manifests / "artifacts"
            isolation = root / "canonical/stages/pilot_10x50/source_isolation"
            phase2_file = root / "canonical/workloads/accepted_phase2.txt"
            artifact_manifests.mkdir(parents=True)
            isolation.mkdir(parents=True)
            phase2_file.parent.mkdir(parents=True)
            phase2_file.write_text("immutable phase2\n", encoding="utf-8")
            (root / ".chiptherm_data_root.json").write_text(
                json.dumps({
                    "schema_version": "chiptherm_data_root/1",
                    "benchmark_id": "benchmark_v2_50family",
                    "root_id": "fixture",
                    "path_semantics": "relative_to_declared_data_root",
                }),
                encoding="utf-8",
            )

            # The real Phase 3 failure had 104 newly generated source-isolation
            # command records after Phase 2 reuse. Model that count exactly.
            for index in range(104):
                command = isolation / f"f002/sample/source_{index:03d}/command.txt"
                command.parent.mkdir(parents=True)
                command.write_text(
                    "/export/hdd/jjun49/hotspot/hotspot -c /export/hdd/jjun49/work/run.config\n",
                    encoding="utf-8",
                )
            completion = make_tree_manifest(isolation, exclude_names={".stage_complete.json"})
            (isolation / ".stage_complete.json").write_text(
                json.dumps({
                    "schema_version": "benchmark_v2_stage_completion/1",
                    "file_count": completion["file_count"],
                    "files": [{"path": row["path"], "sha256": row["sha256"]} for row in completion["files"]],
                }),
                encoding="utf-8",
            )
            manifest_path = artifact_manifests / "pilot_10x50_source_isolation.json"
            write_artifact_manifest(
                isolation,
                manifest_path,
                artifact_id="pilot_10x50_source_isolation",
                artifact_class="canonical_source",
                stage="source_isolation",
                data_root=root,
                parents=[],
                checks=[{"name": "fixture", "passed": True}],
                command=["/nethome/jjun49/chiptherm/scripts/build_source_response_dataset.py"],
            )
            (manifests / "pilot_10x50_runtime_dependency_lock.json").write_text(
                json.dumps({
                    "schema_version": "benchmark_v2_runtime_dependency_lock/1",
                    "hotspot": {
                        "executable_id": "/nethome/jjun49/hotspot/hotspot",
                        "version": "binary /nethome/jjun49/hotspot/hotspot",
                    },
                }),
                encoding="utf-8",
            )
            (manifests / "pilot_10x50_strict_validation.json").write_text(
                json.dumps({
                    "schema_version": "benchmark_v2_scale_pilot_strict_validation/1",
                    "passed": False,
                    "checks": [{
                        "name": "portable_paths",
                        "passed": False,
                        "details": "previous diagnostic: /export/hdd/jjun49/work/source_000/command.txt",
                    }],
                    "portable_path_audit": {
                        "violations": [{"path": "/export/hdd/jjun49/work/source_000/command.txt"}],
                    },
                }),
                encoding="utf-8",
            )

            phase2_before = phase2_immutability_snapshot(root)
            before = audit_portable_documents(root, stage=PHASE3_STAGE)
            self.assertEqual(before["violation_count"], 104)
            classifications = {row.get("classification") for row in before["informational_nonresolving"]}
            self.assertIn("informational_provenance", classifications)
            self.assertIn("validation_diagnostic", classifications)

            dry_run = repair_pilot_portability(root, apply=False, stage=PHASE3_STAGE)
            self.assertEqual(dry_run["changed_file_count"], 104)
            self.assertEqual(phase2_before["content_sha256"], phase2_immutability_snapshot(root)["content_sha256"])

            applied = repair_pilot_portability(root, apply=True, stage=PHASE3_STAGE)
            self.assertEqual(applied["changed_file_count"], 104)
            self.assertEqual(applied["after"]["violation_count"], 0)
            self.assertTrue(durable_stage_complete(isolation))
            artifact = json.loads(manifest_path.read_text(encoding="utf-8"))
            tree_path = root / artifact["content"]["tree_manifest_path"]
            self.assertEqual(artifact["content"]["tree_manifest_sha256"], sha256_file(tree_path))
            self.assertEqual(phase2_before["content_sha256"], phase2_immutability_snapshot(root)["content_sha256"])

            second = repair_pilot_portability(root, apply=True, stage=PHASE3_STAGE)
            self.assertEqual(second["changed_file_count"], 0)
            self.assertEqual(second["lineage_refreshed_file_count"], 0)
            self.assertEqual(second["after"]["violation_count"], 0)
            self.assertTrue(durable_stage_complete(isolation))
            self.assertEqual(phase2_before["content_sha256"], phase2_immutability_snapshot(root)["content_sha256"])


if __name__ == "__main__":
    unittest.main()
