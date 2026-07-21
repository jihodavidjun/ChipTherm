from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chiptherm.benchmark_v2 import (  # noqa: E402
    DEFAULT_PROPOSAL,
    FAMILY_BLUEPRINTS,
    MIN_GAP_MM,
    TYPE_ORDER,
    family_table_rows,
    geometry_for_workload,
    instantiate_all_families,
    load_design_proposal,
    load_family_specs,
    pairwise_distance_rows,
    render_family_preview,
    validate_family_collection,
    validate_family_spec,
)
from chiptherm.layout import SUPPORTED_CHIPLET_TYPES, layout_from_dict  # noqa: E402
from chiptherm.validate import validate_layout  # noqa: E402


class BenchmarkV2DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proposal_path = REPO_ROOT / DEFAULT_PROPOSAL
        cls.proposal = load_design_proposal(cls.proposal_path)
        cls.families = instantiate_all_families(cls.proposal)

    def test_exact_family_taxonomy_and_split_counts(self) -> None:
        self.assertEqual(len(FAMILY_BLUEPRINTS), 50)
        self.assertEqual(len(self.families), 50)
        self.assertEqual(sum(item["primary_split"] == "train" for item in self.families), 40)
        self.assertEqual(sum(item["primary_split"] == "val" for item in self.families), 5)
        self.assertEqual(sum(item["primary_split"] == "test" for item in self.families), 5)
        groups = [int(item["rotational_fold_group"]) for item in self.families]
        self.assertEqual({group: groups.count(group) for group in range(1, 11)}, {group: 5 for group in range(1, 11)})

    def test_deterministic_regeneration_and_workload_independence(self) -> None:
        regenerated = instantiate_all_families(self.proposal)
        first = json.dumps(self.families, sort_keys=True, separators=(",", ":"))
        second = json.dumps(regenerated, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)
        for spec in self.families:
            layout_a = geometry_for_workload(spec, "workload_000", 0)
            layout_b = geometry_for_workload(spec, "workload_199", 999999)
            self.assertEqual(layout_a, layout_b)

    def test_all_geometries_are_valid_and_fingerprints_unique(self) -> None:
        fingerprints: set[str] = set()
        for spec in self.families:
            self.assertEqual(validate_family_spec(spec), [])
            layout = spec["fixed_structure"]["layout"]
            validate_layout(layout_from_dict(layout))
            self.assertGreaterEqual(spec["descriptors"]["minimum_chiplet_gap_mm"] + 1e-7, MIN_GAP_MM)
            self.assertEqual(len(layout["chiplets"]), spec["descriptors"]["chiplet_count"])
            self.assertTrue({item["type"] for item in layout["chiplets"]} <= SUPPORTED_CHIPLET_TYPES)
            self.assertNotIn(spec["structural_fingerprint"], fingerprints)
            fingerprints.add(spec["structural_fingerprint"])

    def test_whitespace_and_composition_recompute(self) -> None:
        for spec in self.families:
            layout = spec["fixed_structure"]["layout"]
            package = layout["package"]["size"]
            package_area = float(package["width"]) * float(package["height"])
            occupied = sum(float(item["size"]["width"]) * float(item["size"]["height"]) for item in layout["chiplets"])
            whitespace = 1.0 - occupied / package_area
            self.assertAlmostEqual(whitespace, float(spec["descriptors"]["whitespace_fraction"]), places=8)
            actual = {name: sum(item["type"] == name for item in layout["chiplets"]) for name in TYPE_ORDER}
            actual = {key: value for key, value in actual.items() if value}
            self.assertEqual(actual, spec["fixed_structure"]["composition"])

    def test_material_cooling_and_hotspot_are_identical(self) -> None:
        stacks = {json.dumps(item["fixed_structure"]["thermal_stack"], sort_keys=True) for item in self.families}
        hotspot = {json.dumps(item["fixed_structure"]["hotspot"], sort_keys=True) for item in self.families}
        self.assertEqual(len(stacks), 1)
        self.assertEqual(len(hotspot), 1)

    def test_no_absolute_paths_and_manifest_parsing(self) -> None:
        serialized = yaml.safe_dump(self.families, sort_keys=False)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/nethome/", serialized)
        with tempfile.TemporaryDirectory() as temporary:
            family_dir = Path(temporary) / "families"
            family_dir.mkdir()
            for spec in self.families:
                (family_dir / f"{spec['family_uid']}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            loaded = load_family_specs(family_dir)
            self.assertEqual(len(loaded), 50)
            report = validate_family_collection(loaded, self.proposal)
            self.assertTrue(report["passed"], report["problems"])

    def test_preview_and_family_table_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preview = Path(temporary) / "f001.png"
            render_family_preview(self.families[0], preview)
            self.assertTrue(preview.exists())
            self.assertGreater(preview.stat().st_size, 1000)
        table = family_table_rows(self.families)
        self.assertEqual(len(table), 50)
        self.assertEqual(table[0]["Family"], "f001")

    def test_intentional_near_duplicate_detection(self) -> None:
        first = deepcopy(self.families[0])
        second = deepcopy(first)
        second["family_uid"] = "near_duplicate_fixture"
        second["primary_split"] = "test"
        rows, _ = pairwise_distance_rows([first, second])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["distance"]), 0.0, places=12)
        self.assertTrue(rows[0]["suspicious"])
        self.assertTrue(rows[0]["cross_split"])


if __name__ == "__main__":
    unittest.main()
