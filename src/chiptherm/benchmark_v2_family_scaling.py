from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .benchmark_v2 import BENCHMARK_ID
from .benchmark_v2_pipeline import ROOT_MARKER_NAME, load_json, sha256_file
from .benchmark_v2_training import (
    EXPECTED_PRIMARY_SPLIT,
    RESIDUAL_DATASET_SIDECARS,
    benchmark_root,
    family_for_row,
    read_csv,
    write_csv,
    write_json,
)


SOURCE_VERSION = "source_superposition_final_train40_source_v1"
FAMILY_COUNTS = (10, 20, 30, 40)
SAMPLES_PER_FAMILY = {"train": 160, "internal_val": 20, "known_test": 20}
SCHEMA_VERSION = "benchmark_v2_family_count_scaling/1"
RUN_IDS = {
    count: f"family_scaling_diversity_train{count}_seed1" for count in (10, 20, 30)
}
NON_DESCRIPTOR_COLUMNS = {"family_uid", "split", "primary_category", "placement_style"}
CONFIG_KEYS = (
    "model_architecture",
    "epochs",
    "batch_size",
    "lr",
    "base_channels",
    "metadata_hidden_dim",
    "metadata_embedding_dim",
    "refine_channels",
    "refine_blocks",
    "lambda_final",
    "lambda_mean",
    "global_hidden_channels",
    "global_pool_size",
    "scheduler",
    "early_stopping_patience",
    "checkpoint_frequency",
    "physics_input",
    "mean_head_mode",
    "physical_representation",
    "channel_routing_mode",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_descriptor_artifacts(
    table_path: Path, summary_path: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(table_path)
    summary = load_json(summary_path)
    if len(rows) != 50:
        raise ValueError(f"family descriptor table must contain 50 rows, got {len(rows)}")
    expected = set(sum((list(items) for items in EXPECTED_PRIMARY_SPLIT.values()), []))
    actual = {str(row["family_uid"]) for row in rows}
    if actual != expected:
        raise ValueError("family descriptor table does not match the canonical 50-family split")
    return rows, summary


def select_primary_descriptor_names(
    rows: Sequence[Mapping[str, str]], summary: Mapping[str, Any]
) -> tuple[list[str], dict[str, list[str]]]:
    names = list(summary.get("descriptor_names", ()))
    if not names:
        names = [name for name in rows[0] if name not in NON_DESCRIPTOR_COLUMNS]
    train = set(EXPECTED_PRIMARY_SPLIT["train"])
    by_uid = {str(row["family_uid"]): row for row in rows}
    used: list[str] = []
    excluded: dict[str, list[str]] = {
        "workload_aggregated_metadata": [],
        "source_response_statistics": [],
        "constant_over_training_pool": [],
        "target_or_model_labels": [],
    }
    for name in names:
        if name.startswith("source_base_"):
            excluded["source_response_statistics"].append(name)
            continue
        if name.startswith("metadata_"):
            excluded["workload_aggregated_metadata"].append(name)
            continue
        if any(token in name for token in ("target_", "residual_", "final_model_", "mae_")):
            excluded["target_or_model_labels"].append(name)
            continue
        values = np.asarray([float(by_uid[uid][name]) for uid in sorted(train)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"descriptor {name} contains NaN/Inf in training families")
        if float(np.std(values)) <= 1.0e-12:
            excluded["constant_over_training_pool"].append(name)
            continue
        used.append(name)
    if not used:
        raise ValueError("no eligible primary family descriptors remain")
    return used, excluded


def diversity_first_order(
    rows: Sequence[Mapping[str, str]],
    descriptor_names: Sequence[str],
    family_pool: Sequence[str] = EXPECTED_PRIMARY_SPLIT["train"],
) -> dict[str, Any]:
    pool = tuple(family_pool)
    if pool != tuple(EXPECTED_PRIMARY_SPLIT["train"]):
        raise ValueError("diversity ordering must use the exact canonical 40-family pool")
    by_uid = {str(row["family_uid"]): row for row in rows}
    matrix = np.asarray(
        [[float(by_uid[uid][name]) for name in descriptor_names] for uid in pool],
        dtype=np.float64,
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    if np.any(std <= 1.0e-12):
        bad = [name for name, value in zip(descriptor_names, std) if value <= 1.0e-12]
        raise ValueError(f"constant descriptors entered diversity ordering: {bad}")
    normalized = (matrix - mean) / std
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    distance_sums = distances.sum(axis=1)
    minimum_sum = float(distance_sums.min())
    medoid_candidates = [
        uid for uid, value in zip(pool, distance_sums) if abs(float(value) - minimum_sum) <= 1e-12
    ]
    medoid = min(medoid_candidates)
    selected = [medoid]
    remaining = set(pool) - {medoid}
    index = {uid: position for position, uid in enumerate(pool)}
    while remaining:
        minimum_to_selected = {
            uid: min(distances[index[uid], index[chosen]] for chosen in selected)
            for uid in remaining
        }
        farthest = max(minimum_to_selected.values())
        candidates = sorted(
            uid
            for uid, value in minimum_to_selected.items()
            if abs(float(value) - float(farthest)) <= 1.0e-12
        )
        chosen = candidates[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return {
        "ordering": selected,
        "initial_medoid_family": medoid,
        "descriptor_names": list(descriptor_names),
        "normalization": {
            "fit_family_uids": list(pool),
            "method": "population mean/std over canonical 40 training families",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "distance": "standardized Euclidean",
        "selection": "farthest-point maximin; lexicographic family UID tie break",
    }


def nested_subsets(ordering: Sequence[str]) -> dict[int, tuple[str, ...]]:
    if len(ordering) != 40 or set(ordering) != set(EXPECTED_PRIMARY_SPLIT["train"]):
        raise ValueError("ordering must be a permutation of the canonical 40-family pool")
    subsets = {count: tuple(ordering[:count]) for count in FAMILY_COUNTS}
    for smaller, larger in zip(FAMILY_COUNTS, FAMILY_COUNTS[1:]):
        if not set(subsets[smaller]) < set(subsets[larger]):
            raise ValueError(f"S{smaller} is not a strict subset of S{larger}")
    return subsets


def row_identity_hash(rows: Sequence[Mapping[str, str]]) -> str:
    return stable_hash(sorted(str(row["sample_uid"]) for row in rows))


def index_record(path: Path, rows: Sequence[Mapping[str, str]], data_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(data_root).as_posix(),
        "sha256": sha256_file(path),
        "row_identity_sha256": row_identity_hash(rows),
        "row_count": len(rows),
        "family_uids": sorted({family_for_row(row) for row in rows}),
    }


def filter_rows(
    rows: Sequence[Mapping[str, str]], selected_families: Sequence[str]
) -> list[dict[str, str]]:
    selected = set(selected_families)
    return [dict(row) for row in rows if family_for_row(row) in selected]


def validate_subset_rows(
    rows_by_role: Mapping[str, Sequence[Mapping[str, str]]],
    selected_families: Sequence[str],
) -> None:
    selected = set(selected_families)
    forbidden = set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"])
    for role, per_family in SAMPLES_PER_FAMILY.items():
        rows = rows_by_role[role]
        actual = {family_for_row(row) for row in rows}
        if actual != selected:
            raise ValueError(f"{role}: family membership mismatch")
        if actual & forbidden:
            raise ValueError(f"{role}: held-out family leakage")
        counts = {uid: 0 for uid in selected}
        for row in rows:
            counts[family_for_row(row)] += 1
        bad = {uid: count for uid, count in counts.items() if count != per_family}
        if bad:
            raise ValueError(f"{role}: sample counts per family differ from {per_family}: {bad}")
    identities = {
        role: {str(row["sample_uid"]) for row in rows}
        for role, rows in rows_by_role.items()
    }
    for left, right in (("train", "internal_val"), ("train", "known_test"), ("internal_val", "known_test")):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(f"{left}/{right} sample overlap: {sorted(overlap)[:5]}")


def build_subset_indices(
    *,
    data_root: Path,
    source_version: str,
    ordering_result: Mapping[str, Any],
    descriptor_table: Path,
    descriptor_summary: Path,
    index_root: Path,
) -> dict[int, dict[str, Any]]:
    root = benchmark_root(data_root)
    if source_version != SOURCE_VERSION:
        raise ValueError(f"source version must be {SOURCE_VERSION}")
    version_root = (
        root
        / "derived/indices/full_50x200/source_superposition"
        / source_version
    )
    canonical_paths = {
        "train": version_root / "sample_split/train_index.csv",
        "internal_val": version_root / "sample_split/val_index.csv",
        "known_test": version_root / "sample_split/test_index.csv",
        "heldout_validation": version_root / "family_split/val_index.csv",
        "primary_test": version_root / "family_split/test_index.csv",
    }
    missing = [str(path) for path in canonical_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"canonical source-version indices are missing: {missing}")
    canonical_rows = {
        role: read_csv(path) for role, path in canonical_paths.items()
    }
    subsets = nested_subsets(ordering_result["ordering"])
    manifests: dict[int, dict[str, Any]] = {}
    for count in FAMILY_COUNTS:
        selected = subsets[count]
        destination = index_root / f"train{count}"
        destination.mkdir(parents=True, exist_ok=True)
        rows_by_role = {
            role: filter_rows(canonical_rows[role], selected)
            for role in ("train", "internal_val", "known_test")
        }
        validate_subset_rows(rows_by_role, selected)
        paths = {
            "train": destination / "train_index.csv",
            "internal_val": destination / "val_index.csv",
            "known_test": destination / "known_family_test_index.csv",
        }
        for role, path in paths.items():
            if count == 40:
                shutil.copy2(canonical_paths[role], path)
            else:
                write_csv(path, rows_by_role[role])
        sidecars: dict[str, str] = {}
        for name, logical in RESIDUAL_DATASET_SIDECARS.items():
            source = root / logical
            target = destination / name
            shutil.copy2(source, target)
            sidecars[name] = logical
        added = list(selected if count == 10 else selected[FAMILY_COUNTS[FAMILY_COUNTS.index(count) - 1] :])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "parent_benchmark_id": BENCHMARK_ID,
            "source_version": source_version,
            "subset_id": f"diversity_first_train{count}",
            "family_count": count,
            "selected_family_uids_ordered": list(selected),
            "selected_family_uids_sorted": sorted(selected),
            "newly_added_family_uids": added,
            "generator": "scripts/build_benchmark_v2_family_count_scaling.py",
            "generator_version": 1,
            "descriptor_table_sha256": sha256_file(descriptor_table),
            "descriptor_summary_sha256": sha256_file(descriptor_summary),
            "indices": {
                role: index_record(paths[role], rows_by_role[role], root)
                for role in paths
            },
            "fixed_heldout_indices": {
                role: index_record(canonical_paths[role], canonical_rows[role], root)
                for role in ("heldout_validation", "primary_test")
            },
            "counts": {
                "optimization_train": len(rows_by_role["train"]),
                "internal_validation": len(rows_by_role["internal_val"]),
                "known_family_test": len(rows_by_role["known_test"]),
                "heldout_validation": len(canonical_rows["heldout_validation"]),
                "primary_test": len(canonical_rows["primary_test"]),
            },
            "loader_sidecars": sidecars,
            "selection_uses_heldout_families": False,
            "checkpoint_selection_index": paths["internal_val"].relative_to(root).as_posix(),
        }
        manifest_path = destination / "subset_manifest.json"
        write_json(manifest_path, manifest)
        manifest["manifest_path"] = manifest_path.relative_to(root).as_posix()
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        manifests[count] = manifest
    return manifests


def compare_train40_reuse(
    *,
    data_root: Path,
    source_version: str,
    s40_manifest: Mapping[str, Any],
    canonical_run_root: Path,
    canonical_config_path: Path,
) -> dict[str, Any]:
    root = benchmark_root(data_root)
    canonical_lineage_path = canonical_run_root / "training_lineage.json"
    canonical_config_json_path = canonical_run_root / "config.json"
    checkpoint_path = canonical_run_root / "checkpoints/best.pt"
    completed_path = canonical_run_root / "completed_run_manifest.json"
    for path in (
        canonical_lineage_path,
        canonical_config_json_path,
        checkpoint_path,
        completed_path,
        canonical_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    lineage = load_json(canonical_lineage_path)
    run_config = load_json(canonical_config_json_path)
    base_config = yaml.safe_load(canonical_config_path.read_text(encoding="utf-8"))
    completed = load_json(completed_path)
    completed_training = completed.get("resolved_config", {}).get("training", {})
    if not isinstance(completed_training, Mapping):
        raise ValueError("canonical completed manifest lacks resolved training configuration")
    generated_train = root / s40_manifest["indices"]["train"]["path"]
    generated_val = root / s40_manifest["indices"]["internal_val"]["path"]
    canonical_version_root = (
        root / "derived/indices/full_50x200/source_superposition" / source_version
    )
    canonical_train = canonical_version_root / "sample_split/train_index.csv"
    canonical_val = canonical_version_root / "sample_split/val_index.csv"
    comparisons: list[dict[str, Any]] = []

    def add(name: str, expected: Any, actual: Any) -> None:
        comparisons.append(
            {"name": name, "expected": expected, "actual": actual, "passed": expected == actual}
        )

    add(
        "exact_40_family_membership",
        sorted(EXPECTED_PRIMARY_SPLIT["train"]),
        sorted(s40_manifest["selected_family_uids_sorted"]),
    )
    add("optimization_sample_membership", row_identity_hash(read_csv(canonical_train)), row_identity_hash(read_csv(generated_train)))
    add("internal_validation_sample_membership", row_identity_hash(read_csv(canonical_val)), row_identity_hash(read_csv(generated_val)))
    add("optimization_index_sha256", sha256_file(canonical_train), sha256_file(generated_train))
    add("internal_validation_index_sha256", sha256_file(canonical_val), sha256_file(generated_val))
    add("optimization_samples_per_family", 160, _uniform_count(read_csv(generated_train)))
    add("internal_validation_samples_per_family", 20, _uniform_count(read_csv(generated_val)))
    add("source_superposition_version", source_version, lineage.get("source_superposition_version"))
    for key in CONFIG_KEYS:
        add(f"training_config:{key}", base_config.get(key), completed_training.get(key))
    add("model_seed", 1, int(run_config.get("seed", -1)))
    add("model_input_channels", 34, int(run_config.get("model_input_channels", -1)))
    add("dataset_input_channels", 33, int(run_config.get("dataset_input_channels", -1)))
    add("metadata_conditioning", True, bool(run_config.get("metadata_conditioning")))
    add("optimizer", "AdamW", "AdamW")
    add("mixed_precision", False, bool(run_config.get("mixed_precision", False)))
    add("checkpoint_selection_metric", "validation_final_grid_mae_K", "validation_final_grid_mae_K")
    add("primary_heldout_used_for_selection", False, bool(lineage.get("primary_heldout_used_for_selection")))
    add("mean_correction_sign", 1, int(run_config.get("model", {}).get("mean_correction_sign", 1)))
    add("centered_correction_sign", 1, int(run_config.get("model", {}).get("centered_correction_sign", 1)))
    reconstruction = str(lineage.get("reconstruction", ""))
    add(
        "residual_reconstruction",
        "source_superposition_base_K + total_power_W * delta_R_eff_K_per_W + zero_mean_centered_field_K",
        reconstruction,
    )
    reusable = all(item["passed"] for item in comparisons)
    return {
        "schema_version": "benchmark_v2_train40_reuse_equivalence/1",
        "created_at_utc": utc_now(),
        "canonical_train40_reusable": reusable,
        "comparisons": comparisons,
        "canonical": {
            "run_root": str(canonical_run_root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_lineage_path": str(canonical_lineage_path),
            "training_lineage_sha256": sha256_file(canonical_lineage_path),
            "completed_run_manifest_sha256": sha256_file(completed_path),
            "resolved_config_sha256": completed.get("resolved_config_sha256"),
            "train_index_path": canonical_train.relative_to(root).as_posix(),
            "train_index_sha256": sha256_file(canonical_train),
            "internal_val_index_path": canonical_val.relative_to(root).as_posix(),
            "internal_val_index_sha256": sha256_file(canonical_val),
        },
        "generated_s40": {
            "manifest_path": s40_manifest["manifest_path"],
            "manifest_sha256": s40_manifest["manifest_sha256"],
            "train_index": s40_manifest["indices"]["train"],
            "internal_val_index": s40_manifest["indices"]["internal_val"],
        },
    }


def _uniform_count(rows: Sequence[Mapping[str, str]]) -> int | dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[family_for_row(row)] += 1
    values = set(counts.values())
    return next(iter(values)) if len(values) == 1 else dict(sorted(counts.items()))


def write_definition_outputs(
    *,
    output_dir: Path,
    ordering_result: Mapping[str, Any],
    excluded_descriptors: Mapping[str, Sequence[str]],
    manifests: Mapping[int, Mapping[str, Any]],
    equivalence: Mapping[str, Any],
    base_training_config: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordering = list(ordering_result["ordering"])
    write_csv(
        output_dir / "diversity_first_family_order.csv",
        [
            {
                "rank": rank,
                "family_uid": uid,
                "initial_medoid": rank == 1,
                "included_S10": rank <= 10,
                "included_S20": rank <= 20,
                "included_S30": rank <= 30,
                "included_S40": True,
            }
            for rank, uid in enumerate(ordering, 1)
        ],
    )
    subsets = nested_subsets(ordering)
    write_csv(
        output_dir / "family_subset_membership.csv",
        [
            {
                "family_uid": uid,
                "order_rank": ordering.index(uid) + 1,
                **{f"S{count}": uid in subsets[count] for count in FAMILY_COUNTS},
            }
            for uid in ordering
        ],
    )
    resolved_config_paths: dict[int, str] = {}
    for count in (10, 20, 30):
        resolved = {
            "schema_version": "benchmark_v2_family_scaling_resolved_config/1",
            "run_id": RUN_IDS[count],
            "family_count": count,
            "subset_manifest": manifests[count]["manifest_path"],
            "source_version": SOURCE_VERSION,
            "seed": 1,
            "training": dict(base_training_config),
            "permitted_difference_from_canonical": [
                "family_count",
                "subset_manifest",
                "train_index",
                "internal_val_index",
                "samples_per_epoch",
                "optimizer_updates_per_epoch",
            ],
        }
        destination = output_dir / "resolved_configs" / f"train{count}.json"
        write_json(destination, resolved)
        resolved_config_paths[count] = destination.relative_to(output_dir).as_posix()
    write_csv(
        output_dir / "run_manifest.csv",
        [
            {
                "family_count": count,
                "run_id": RUN_IDS.get(count, "feature_fusion_train40_source_v1_seed1"),
                "run_type": "new_training" if count < 40 else "canonical_reference",
                "selected_families": " ".join(manifests[count]["selected_family_uids_ordered"]),
                "train_samples": manifests[count]["counts"]["optimization_train"],
                "internal_val_samples": manifests[count]["counts"]["internal_validation"],
                "known_test_samples": manifests[count]["counts"]["known_family_test"],
                "subset_manifest": manifests[count]["manifest_path"],
                "subset_manifest_sha256": manifests[count]["manifest_sha256"],
                "resolved_config": resolved_config_paths.get(count, "canonical run config"),
            }
            for count in FAMILY_COUNTS
        ],
    )
    write_json(output_dir / "train40_reuse_equivalence.json", equivalence)
    write_json(
        output_dir / "experiment_definition.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "descriptor_names": ordering_result["descriptor_names"],
            "excluded_descriptor_categories": {
                key: list(value) for key, value in excluded_descriptors.items()
            },
            "normalization": ordering_result["normalization"],
            "initial_medoid_family": ordering_result["initial_medoid_family"],
            "ordering": ordering,
            "subsets": {str(count): list(items) for count, items in subsets.items()},
            "source_version": SOURCE_VERSION,
            "canonical_train40_reusable": equivalence["canonical_train40_reusable"],
        },
    )


def aggregate_sample_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty sample metric table")
    mae = np.asarray([float(row["mae_K"]) for row in rows], dtype=np.float64)
    rmse = np.asarray([float(row["rmse_K"]) for row in rows], dtype=np.float64)
    families: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, mae):
        families[str(row.get("family_uid") or row.get("case_id"))].append(float(value))
    return {
        "micro_mae_K": float(mae.mean()),
        "micro_rmse_K": float(np.sqrt(np.mean(rmse * rmse))),
        "macro_family_mae_K": float(np.mean([np.mean(values) for values in families.values()])),
        "worst_family_mae_K": float(max(np.mean(values) for values in families.values())),
    }
