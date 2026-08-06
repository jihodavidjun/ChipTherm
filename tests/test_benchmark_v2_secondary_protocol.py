from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_pipeline import PATH_SEMANTICS, ROOT_MARKER_NAME
from chiptherm.benchmark_v2_secondary_protocol import (
    EXPECTED_COUNTS,
    EXPECTED_FAMILIES,
    PROTOCOL_NAME,
    build_protocol_indices,
    generate_family_split,
    partition_package_rows,
    partition_source_rows,
    validate_normalizer_provenance,
    workload_ordinal,
)
from scripts.build_full_source_superposition_base import portable_relative, resolve_path
from scripts.aggregate_benchmark_v2_secondary_protocol import (
    MODELS,
    PROTOCOLS,
    build_tables,
    comparison,
)


class SecondaryProtocolTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        first = generate_family_split(seed=3510)
        second = generate_family_split(seed=3510)
        self.assertEqual(first, second)
        self.assertEqual({key: len(value) for key, value in first.items()}, {"train": 35, "validation": 5, "test": 10})
        self.assertEqual(set().union(*(set(value) for value in first.values())), set(EXPECTED_FAMILIES))
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))

    def test_exact_frozen_lists(self) -> None:
        split = generate_family_split(seed=3510)
        self.assertEqual(split["validation"], ["f002", "f008", "f018", "f025", "f038"])
        self.assertEqual(split["test"], ["f001", "f015", "f016", "f017", "f019", "f031", "f033", "f035", "f043", "f044"])

    def test_workload_counts_and_no_final_test_training(self) -> None:
        rows = self._package_rows()
        split = generate_family_split(seed=3510)
        partitioned = partition_package_rows(rows, split)
        self.assertEqual({key: len(value) for key, value in partitioned.items()}, EXPECTED_COUNTS)
        forbidden = set(split["validation"]) | set(split["test"])
        self.assertFalse({row["family_uid"] for row in partitioned["familiar_train"]} & forbidden)
        self.assertEqual(
            [row["sample_uid"] for row in partitioned["familiar_train"]],
            sorted(
                [row["sample_uid"] for row in partitioned["familiar_train"]],
                key=lambda uid: (uid.split("_")[0], int(uid.rsplit("w", 1)[1])),
            ),
        )

    def test_train_only_normalizer_provenance(self) -> None:
        split = generate_family_split(seed=3510)
        manifest = {"split": split}
        validate_normalizer_provenance(manifest, split["train"])
        with self.assertRaisesRegex(ValueError, "normalizer provenance"):
            validate_normalizer_provenance(manifest, split["train"][:-1] + [split["test"][0]])

    def test_source_row_ordinal_uses_original_sample_uid(self) -> None:
        self.assertEqual(
            workload_ordinal({"original_sample_uid": "f044_w173_high_clustered"}),
            173,
        )

    def test_source_split_is_family_aware_and_heldout_free(self) -> None:
        split = generate_family_split(seed=3510)
        rows = [
            {
                "case_id": family,
                "original_sample_uid": f"{family}_w099_fixture",
                "source_index": "0",
            }
            for family in EXPECTED_FAMILIES
        ]
        source = partition_source_rows(rows, split, seed=3510)
        fit = {row["case_id"] for row in source["train"]}
        selection = {row["case_id"] for row in source["internal_val"]}
        self.assertEqual(len(fit), 28)
        self.assertEqual(len(selection), 7)
        self.assertFalse(fit & selection)
        self.assertEqual(fit | selection, set(split["train"]))
        self.assertFalse((fit | selection) & (set(split["validation"]) | set(split["test"])))

    def test_builder_isolated_and_relocation_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root_a"
            relocated = Path(tmp) / "root_b"
            self._write_fixture(root)
            config = root / "protocol_config.json"
            split = generate_family_split(seed=3510)
            config.write_text(json.dumps({"split": split}), encoding="utf-8")
            primary = root / "derived/indices/full_50x200/family_split/train_index.csv"
            primary_before = primary.read_bytes()
            out = root / f"derived/protocols/{PROTOCOL_NAME}"
            report = build_protocol_indices(
                root,
                out,
                split=split,
                config_manifest=config,
                validate_files=False,
            )
            self.assertEqual(report["counts"], EXPECTED_COUNTS)
            self.assertEqual(primary.read_bytes(), primary_before)
            train_rows = self._read_csv(out / "package/sample_split/train_index.csv")
            self.assertEqual(len(train_rows), 5600)
            self.assertTrue(all(not Path(row["x_path"]).is_absolute() for row in train_rows))
            shutil.copytree(root, relocated)
            relocated_rows = self._read_csv(
                relocated / f"derived/protocols/{PROTOCOL_NAME}/package/sample_split/train_index.csv"
            )
            self.assertTrue((relocated / relocated_rows[0]["x_path"]).is_file())

    def test_source_map_generator_explicit_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external_benchmark"
            target = root / "derived/maps/example.npy"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"map")
            self.assertEqual(resolve_path("derived/maps/example.npy", data_root=root), target)
            self.assertEqual(portable_relative(target, root), "derived/maps/example.npy")

    def test_secondary_configs_preserve_canonical_hyperparameters(self) -> None:
        pairs = (
            ("source_response.yaml", "source_response_final_train40_v1.yaml"),
            ("chiptherm_residual.yaml", "package_residual_feature_fusion_v1.yaml"),
            ("cnn.yaml", "package_direct_temperature_feature_fusion_normalized_seed1.yaml"),
            ("fno.yaml", "package_residual_fno_decomposed_seed1.yaml"),
        )
        secondary_root = REPO_ROOT / f"configs/{PROTOCOL_NAME}"
        canonical_root = REPO_ROOT / "configs/benchmark_v2_50family/training"
        ignored = {"base_config", "protocol_override_only"}
        for secondary_name, canonical_name in pairs:
            secondary = yaml.safe_load((secondary_root / secondary_name).read_text(encoding="utf-8"))
            canonical = yaml.safe_load((canonical_root / canonical_name).read_text(encoding="utf-8"))
            comparable = {key: value for key, value in secondary.items() if key not in ignored}
            self.assertEqual(comparable, canonical, secondary_name)

    def test_metric_aggregation_is_sample_and_family_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = {model: Path(tmp) / model for model in MODELS}
            for model_index, model in enumerate(MODELS):
                for protocol in PROTOCOLS:
                    rows = [
                        {
                            "sample_uid": f"f001_w{index:03d}",
                            "family_uid": "f001" if index <= 2 else "f015",
                            "mae_K": str(1.0 + model_index + 0.1 * index),
                            "rmse_K": str(2.0 + model_index + 0.1 * index),
                            "mean_signed_error_K": str(0.01 * index),
                        }
                        for index in range(1, 5)
                    ]
                    self._write_csv(roots[model] / protocol / "metrics_by_sample.csv", rows)
            summary, family_rows = build_tables(roots)
            result = comparison(summary, family_rows)
            self.assertEqual(len(summary), len(MODELS) * len(PROTOCOLS))
            self.assertEqual(len(family_rows), len(MODELS) * len(PROTOCOLS) * 2)
            self.assertEqual(result["sample_weighted_ranking"], ["CNN", "FNO", "ChipTherm"])

    def _write_fixture(self, root: Path) -> None:
        root.mkdir(parents=True)
        (root / ROOT_MARKER_NAME).write_text(
            json.dumps({"benchmark_id": "benchmark_v2_50family", "path_semantics": PATH_SEMANTICS}),
            encoding="utf-8",
        )
        rows = self._package_rows()
        split = generate_family_split(seed=3510)
        for role, families in (("train", split["train"]), ("val", split["validation"]), ("test", split["test"])):
            selected = [row for row in rows if row["family_uid"] in set(families)]
            self._write_csv(root / f"derived/indices/full_50x200/family_split/{role}_index.csv", selected)
        source_rows = []
        for row in rows:
            item = dict(row)
            item.update(
                source_response_uid=f"{row['sample_uid']}:source_0000",
                original_sample_uid=row["sample_uid"],
                source_index="0",
                original_x_path=row["x_path"],
                target_rise_path=row["y_path"],
                full_temperature_path=row["y_path"],
            )
            source_rows.append(item)
        for role, families in (("train", split["train"]), ("val", split["validation"]), ("test", split["test"])):
            selected = [row for row in source_rows if row["family_uid"] in set(families)]
            self._write_csv(root / f"canonical/stages/full_50x200/source_isolation/{role}_index.csv", selected)
        for name in ("feature_manifest.json", "metadata_manifest.json", "graph_manifest.json"):
            path = root / f"derived/stages/full_50x200/{'context_33ch' if name == 'feature_manifest.json' else 'metadata' if name.startswith('metadata') else 'graphs'}/{name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        metadata = root / "derived/stages/full_50x200/metadata/metadata_features.csv"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("sample_uid\n", encoding="utf-8")
        first = root / rows[0]["x_path"]
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"fixture")

    @staticmethod
    def _package_rows() -> list[dict[str, str]]:
        rows = []
        for family in EXPECTED_FAMILIES:
            for ordinal in range(1, 201):
                uid = f"{family}_w{ordinal:03d}"
                base = f"canonical/{family}/{uid}"
                rows.append(
                    {
                        "family_uid": family,
                        "case_id": family,
                        "workload_uid": f"w{ordinal:03d}_fixture",
                        "sample_uid": uid,
                        "x_path": "fixture/x.npy",
                        "y_path": "fixture/y.npy",
                        "graph_path": "fixture/g.npz",
                        "layout_path": f"{base}/layout.json",
                        "power_path": f"{base}/power.yaml",
                        "package_path": f"{base}/package.yaml",
                        "source_dir": base,
                    }
                )
        return rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
