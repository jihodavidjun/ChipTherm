from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.benchmark_v2 import BENCHMARK_ID  # noqa: E402
from chiptherm.benchmark_v2_family_scaling import (  # noqa: E402
    CONFIG_KEYS,
    EXPECTED_PRIMARY_SPLIT,
    SOURCE_VERSION,
    aggregate_sample_metrics,
    compare_train40_reuse,
    diversity_first_order,
    nested_subsets,
    read_descriptor_artifacts,
    select_primary_descriptor_names,
    validate_subset_rows,
    write_definition_outputs,
)
from chiptherm.benchmark_v2_pipeline import PATH_SEMANTICS, ROOT_MARKER_NAME  # noqa: E402
from chiptherm.benchmark_v2_training import sha256_file, write_csv  # noqa: E402
from scripts.run_benchmark_v2_family_count_scaling import build_training_command  # noqa: E402


EXPECTED_ORDER = (
    "f002", "f028", "f045", "f032", "f049", "f013", "f024", "f025",
    "f039", "f014", "f047", "f035", "f043", "f018", "f029", "f046",
    "f011", "f036", "f037", "f015", "f022", "f021", "f004", "f048",
    "f031", "f017", "f001", "f042", "f005", "f038", "f040", "f006",
    "f010", "f020", "f009", "f050", "f034", "f003", "f026", "f019",
)


def main() -> None:
    test_exact_pool_and_deterministic_descriptor_order()
    test_nested_subsets_and_no_heldout_leakage()
    test_sample_counts_and_internal_validation_semantics()
    test_resolved_configs_change_only_subset_identity()
    test_train40_reuse_equivalence()
    test_launcher_is_explicit_and_dry_run_compatible()
    test_macro_and_micro_aggregation()
    print("benchmark v2 family-count scaling tests passed")


def descriptor_paths() -> tuple[Path, Path]:
    root = (
        REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1/family_ood_analysis"
    )
    return root / "family_descriptors.csv", root / "summary.json"


def test_exact_pool_and_deterministic_descriptor_order() -> None:
    assert len(EXPECTED_PRIMARY_SPLIT["train"]) == 40
    assert set(EXPECTED_PRIMARY_SPLIT["train"]).isdisjoint(EXPECTED_PRIMARY_SPLIT["val"])
    assert set(EXPECTED_PRIMARY_SPLIT["train"]).isdisjoint(EXPECTED_PRIMARY_SPLIT["test"])
    table, summary = descriptor_paths()
    rows, payload = read_descriptor_artifacts(table, summary)
    names, excluded = select_primary_descriptor_names(rows, payload)
    first = diversity_first_order(rows, names)
    second = diversity_first_order(rows, names)
    assert tuple(first["ordering"]) == EXPECTED_ORDER
    assert first == second
    assert first["normalization"]["fit_family_uids"] == list(EXPECTED_PRIMARY_SPLIT["train"])
    assert excluded["source_response_statistics"]
    assert excluded["workload_aggregated_metadata"]


def test_nested_subsets_and_no_heldout_leakage() -> None:
    subsets = nested_subsets(EXPECTED_ORDER)
    assert [len(subsets[count]) for count in (10, 20, 30, 40)] == [10, 20, 30, 40]
    assert set(subsets[10]) < set(subsets[20]) < set(subsets[30]) < set(subsets[40])
    heldout = set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"])
    assert not set(subsets[40]) & heldout


def make_split_rows(families: tuple[str, ...]) -> dict[str, list[dict[str, str]]]:
    counts = {"train": 160, "internal_val": 20, "known_test": 20}
    output: dict[str, list[dict[str, str]]] = {}
    for role, count in counts.items():
        output[role] = [
            {
                "sample_uid": f"{family}_{role}_{index:03d}",
                "family_uid": family,
            }
            for family in families
            for index in range(count)
        ]
    return output


def test_sample_counts_and_internal_validation_semantics() -> None:
    selected = EXPECTED_ORDER[:10]
    rows = make_split_rows(selected)
    validate_subset_rows(rows, selected)
    assert len(rows["train"]) == 1600
    assert len(rows["internal_val"]) == 200
    assert len(rows["known_test"]) == 200
    rows["internal_val"][0]["sample_uid"] = rows["train"][0]["sample_uid"]
    try:
        validate_subset_rows(rows, selected)
    except ValueError as exc:
        assert "sample overlap" in str(exc)
    else:
        raise AssertionError("sample-level leakage was accepted")


def minimal_manifests() -> dict[int, dict]:
    return {
        count: {
            "manifest_path": f"indices/train{count}/subset_manifest.json",
            "manifest_sha256": f"sha-{count}",
            "selected_family_uids_ordered": list(EXPECTED_ORDER[:count]),
            "counts": {
                "optimization_train": 160 * count,
                "internal_validation": 20 * count,
                "known_family_test": 20 * count,
            },
        }
        for count in (10, 20, 30, 40)
    }


def test_resolved_configs_change_only_subset_identity() -> None:
    table, summary = descriptor_paths()
    rows, payload = read_descriptor_artifacts(table, summary)
    names, excluded = select_primary_descriptor_names(rows, payload)
    ordering = diversity_first_order(rows, names)
    base = yaml.safe_load(
        (
            REPO_ROOT
            / "configs/benchmark_v2_50family/training/"
            "package_residual_feature_fusion_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_definition_outputs(
            output_dir=root,
            ordering_result=ordering,
            excluded_descriptors=excluded,
            manifests=minimal_manifests(),
            equivalence={"canonical_train40_reusable": True},
            base_training_config=base,
        )
        configs = [
            json.loads((root / f"resolved_configs/train{count}.json").read_text())
            for count in (10, 20, 30)
        ]
        assert all(item["training"] == base for item in configs)
        assert [item["family_count"] for item in configs] == [10, 20, 30]
        assert all(item["seed"] == 1 for item in configs)


def rows_for_equivalence(per_family: int, role: str) -> list[dict[str, str]]:
    return [
        {"sample_uid": f"{family}_{role}_{index:03d}", "family_uid": family}
        for family in EXPECTED_PRIMARY_SPLIT["train"]
        for index in range(per_family)
    ]


def test_train40_reuse_equivalence() -> None:
    base_config_path = (
        REPO_ROOT
        / "configs/benchmark_v2_50family/training/"
        "package_residual_feature_fusion_v1.yaml"
    )
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        data_root = temporary_root / "benchmark"
        data_root.mkdir()
        (data_root / ROOT_MARKER_NAME).write_text(
            json.dumps({"benchmark_id": BENCHMARK_ID, "path_semantics": PATH_SEMANTICS})
        )
        canonical = (
            data_root
            / "derived/indices/full_50x200/source_superposition"
            / SOURCE_VERSION
            / "sample_split"
        )
        generated = data_root / "derived/indices/family_count_scaling/diversity_first/train40"
        canonical.mkdir(parents=True)
        generated.mkdir(parents=True)
        train_rows = rows_for_equivalence(160, "train")
        val_rows = rows_for_equivalence(20, "val")
        write_csv(canonical / "train_index.csv", train_rows)
        write_csv(canonical / "val_index.csv", val_rows)
        write_csv(generated / "train_index.csv", train_rows)
        write_csv(generated / "val_index.csv", val_rows)
        manifest_path = generated / "subset_manifest.json"
        manifest_path.write_text("{}\n")
        s40 = {
            "selected_family_uids_sorted": sorted(EXPECTED_PRIMARY_SPLIT["train"]),
            "manifest_path": manifest_path.relative_to(data_root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "indices": {
                "train": {"path": (generated / "train_index.csv").relative_to(data_root).as_posix()},
                "internal_val": {"path": (generated / "val_index.csv").relative_to(data_root).as_posix()},
            },
        }
        run = temporary_root / "canonical_run"
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints/best.pt").write_bytes(b"checkpoint")
        (run / "training_lineage.json").write_text(
            json.dumps(
                {
                    "source_superposition_version": SOURCE_VERSION,
                    "primary_heldout_used_for_selection": False,
                    "reconstruction": (
                        "source_superposition_base_K + total_power_W * "
                        "delta_R_eff_K_per_W + zero_mean_centered_field_K"
                    ),
                }
            )
        )
        (run / "config.json").write_text(
            json.dumps(
                {
                    "seed": 1,
                    "model_input_channels": 34,
                    "dataset_input_channels": 33,
                    "metadata_conditioning": True,
                    "mixed_precision": False,
                    "model": {"mean_correction_sign": 1, "centered_correction_sign": 1},
                }
            )
        )
        (run / "completed_run_manifest.json").write_text(
            json.dumps(
                {
                    "resolved_config": {"training": base_config},
                    "resolved_config_sha256": "resolved",
                }
            )
        )
        result = compare_train40_reuse(
            data_root=data_root,
            source_version=SOURCE_VERSION,
            s40_manifest=s40,
            canonical_run_root=run,
            canonical_config_path=base_config_path,
        )
        assert result["canonical_train40_reusable"] is True
        assert all(item["passed"] for item in result["comparisons"])


def test_launcher_is_explicit_and_dry_run_compatible() -> None:
    command = build_training_command(
        python="/venv/bin/python3",
        data_root=Path("/data"),
        family_count=10,
        index_root=Path("/data/indices"),
        output_root=Path("/outputs"),
        preflight_report=Path("/repo/preflight.json"),
        config=Path("/repo/config.yaml"),
        device="cuda",
        workers=4,
        resume=False,
    )
    assert "--prepared-index-root" in command
    assert command[command.index("--train-family-count") + 1] == "10"
    assert "--resume" not in command
    assert "--execute" not in command


def test_macro_and_micro_aggregation() -> None:
    rows = [
        {"case_id": "f001", "mae_K": "1", "rmse_K": "2"},
        {"case_id": "f002", "mae_K": "3", "rmse_K": "4"},
        {"case_id": "f002", "mae_K": "5", "rmse_K": "6"},
    ]
    result = aggregate_sample_metrics(rows)
    assert abs(result["micro_mae_K"] - 3.0) < 1.0e-12
    assert abs(result["macro_family_mae_K"] - 2.5) < 1.0e-12
    assert abs(result["micro_rmse_K"] - (56.0 / 3.0) ** 0.5) < 1.0e-12


if __name__ == "__main__":
    main()
