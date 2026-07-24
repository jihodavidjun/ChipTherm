from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
import numpy as np

from .benchmark_v2 import BENCHMARK_ID
from .benchmark_v2_pipeline import (
    FULL_STAGE,
    PATH_SEMANTICS,
    ROOT_MARKER_NAME,
    load_json,
    resolve_data_path,
    sha256_file,
)


SOURCE_SPLIT_SCHEMA = "benchmark_v2_source_training_split/1"
PREFLIGHT_SCHEMA = "benchmark_v2_final_training_preflight/1"
TRAIN_FAMILY_COUNTS = (5, 10, 20, 30, 40)
EXPECTED_PRIMARY_SPLIT = {
    "train": (
        "f001", "f002", "f003", "f004", "f005", "f006", "f009", "f010", "f011", "f013",
        "f014", "f015", "f017", "f018", "f019", "f020", "f021", "f022", "f024", "f025",
        "f026", "f028", "f029", "f031", "f032", "f034", "f035", "f036", "f037", "f038",
        "f039", "f040", "f042", "f043", "f045", "f046", "f047", "f048", "f049", "f050",
    ),
    "val": ("f007", "f012", "f023", "f030", "f041"),
    "test": ("f008", "f016", "f027", "f033", "f044"),
}

RESIDUAL_DATASET_SIDECARS = {
    "feature_manifest.json": "derived/stages/full_50x200/context_33ch/feature_manifest.json",
    "context_manifest.json": "derived/stages/full_50x200/context_33ch/context_manifest.json",
    "metadata_features.csv": "derived/stages/full_50x200/metadata/metadata_features.csv",
    "metadata_manifest.json": "derived/stages/full_50x200/metadata/metadata_manifest.json",
    "graph_manifest.json": "derived/stages/full_50x200/graphs/graph_manifest.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def benchmark_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    marker = load_json(root / ROOT_MARKER_NAME)
    if marker.get("benchmark_id") != BENCHMARK_ID or marker.get("path_semantics") != PATH_SEMANTICS:
        raise ValueError(f"invalid accepted Benchmark v2 root: {root}")
    return root


def family_for_row(row: Mapping[str, str]) -> str:
    return str(row.get("family_uid") or row.get("case_id") or "")


def deterministic_family_order(families: Sequence[str], seed: int) -> list[str]:
    return sorted(
        {str(value) for value in families},
        key=lambda uid: (hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest(), uid),
    )


def deterministic_scaling_subsets(train_families: Sequence[str], seed: int) -> dict[str, list[str]]:
    ordered = deterministic_family_order(train_families, seed)
    if len(ordered) != 40:
        raise ValueError(f"family scaling requires exactly 40 train families, got {len(ordered)}")
    return {str(count): sorted(ordered[:count]) for count in TRAIN_FAMILY_COUNTS}


def prepare_source_scaling_indices(
    data_root: str | Path,
    *,
    family_count: int,
    seed: int,
) -> dict[str, Any]:
    if family_count not in TRAIN_FAMILY_COUNTS:
        raise ValueError(f"family_count must be one of {TRAIN_FAMILY_COUNTS}")
    if family_count == 40:
        return prepare_final_training_indices(data_root, seed=seed)
    root = benchmark_root(data_root)
    isolation_rows = read_csv(
        root / f"canonical/stages/{FULL_STAGE}/source_isolation/train_index.csv"
    )
    selected = deterministic_scaling_subsets(EXPECTED_PRIMARY_SPLIT["train"], seed)[
        str(family_count)
    ]
    selected_order = deterministic_family_order(selected, seed + 1)
    val_count = max(1, round(0.20 * family_count))
    selection_families = set(selected_order[:val_count])
    training_families = set(selected_order[val_count:])
    output = root / f"derived/indices/{FULL_STAGE}/source_response/scaling/train_{family_count}"
    train_rows = [dict(row) for row in isolation_rows if family_for_row(row) in training_families]
    val_rows = [dict(row) for row in isolation_rows if family_for_row(row) in selection_families]
    train_path = output / "train_index.csv"
    val_path = output / "internal_val_index.csv"
    write_csv(train_path, train_rows)
    write_csv(val_path, val_rows)
    manifest = {
        "schema_version": SOURCE_SPLIT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "stage": FULL_STAGE,
        "seed": seed,
        "family_scaling_count": family_count,
        "fit_family_uids": sorted(training_families),
        "internal_validation_family_uids": sorted(selection_families),
        "oracle_validation_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"]),
        "oracle_test_family_uids": list(EXPECTED_PRIMARY_SPLIT["test"]),
        "normalization_allowed_index": str(train_path.relative_to(root)),
        "checkpoint_selection_allowed_index": str(val_path.relative_to(root)),
        "oracle_policy": "evaluation_only_after_checkpoint_freeze",
        "indices": {
            "train": _index_lineage(train_path, train_rows, root),
            "internal_val": _index_lineage(val_path, val_rows, root),
        },
    }
    write_json(output / "split_manifest.json", manifest)
    return manifest


def prepare_residual_scaling_indices(
    data_root: str | Path,
    *,
    source_version: str,
    family_count: int,
    seed: int,
) -> Path:
    if family_count not in TRAIN_FAMILY_COUNTS:
        raise ValueError(f"family_count must be one of {TRAIN_FAMILY_COUNTS}")
    root = benchmark_root(data_root)
    version_root = root / f"derived/indices/{FULL_STAGE}/source_superposition/{source_version}"
    if family_count == 40:
        return version_root / "sample_split"
    selected = set(
        deterministic_scaling_subsets(EXPECTED_PRIMARY_SPLIT["train"], seed)[
            str(family_count)
        ]
    )
    output = version_root / f"scaling/train_{family_count}"
    install_residual_dataset_sidecars(root, output)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        source = version_root / "sample_split" / f"{split}_index.csv"
        rows = [
            row
            for row in read_csv(source)
            if family_for_row(row) in selected
        ]
        write_csv(output / f"{split}_index.csv", rows)
        counts[split] = len(rows)
    write_json(
        output / "split_manifest.json",
        {
            "schema_version": "benchmark_v2_residual_family_scaling_split/1",
            "source_superposition_version": source_version,
            "family_count": family_count,
            "family_uids": sorted(selected),
            "seed": seed,
            "counts": counts,
            "selection_uses_primary_heldout": False,
        },
    )
    return output


def _source_uid(row: Mapping[str, str]) -> str:
    return str(
        row.get("source_response_uid")
        or f"{row.get('original_sample_uid', '')}:source_{int(float(row.get('source_index', 0))):04d}"
    )


def _index_lineage(path: Path, rows: Sequence[Mapping[str, str]], data_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(data_root)),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "row_identity_sha256": stable_json_hash(
            sorted(
                (
                    _source_uid(row) if "source_index" in row else str(row.get("sample_uid", "")),
                    family_for_row(row),
                )
                for row in rows
            )
        ),
        "family_uids": sorted({family_for_row(row) for row in rows}),
    }


def prepare_final_training_indices(
    data_root: str | Path,
    *,
    seed: int = 20260721,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = benchmark_root(data_root)
    output = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else root / f"derived/indices/{FULL_STAGE}/source_response"
    )
    isolation = root / f"canonical/stages/{FULL_STAGE}/source_isolation"
    source_rows = {
        split: read_csv(isolation / f"{split}_index.csv")
        for split in ("train", "val", "test")
    }
    source_families = {
        split: sorted({family_for_row(row) for row in rows})
        for split, rows in source_rows.items()
    }
    expected = {key: list(value) for key, value in EXPECTED_PRIMARY_SPLIT.items()}
    if source_families != expected:
        raise ValueError(f"source-isolation primary family split mismatch: {source_families}")

    source_uids = [_source_uid(row) for rows in source_rows.values() for row in rows]
    if len(source_uids) != len(set(source_uids)):
        raise ValueError("source-isolation inventory contains duplicate source identities")
    target_values = [
        str(row.get("target_rise_path", ""))
        for rows in source_rows.values()
        for row in rows
    ]
    if any(not value for value in target_values) or len(target_values) != len(set(target_values)):
        raise ValueError("each physical isolated-source target must occur exactly once")

    train_family_order = deterministic_family_order(EXPECTED_PRIMARY_SPLIT["train"], seed)
    internal_val_count = max(1, round(0.20 * len(train_family_order)))
    internal_val_families = set(train_family_order[:internal_val_count])
    fit_families = set(train_family_order[internal_val_count:])
    train_rows = [dict(row) for row in source_rows["train"] if family_for_row(row) in fit_families]
    internal_val_rows = [
        dict(row) for row in source_rows["train"] if family_for_row(row) in internal_val_families
    ]
    oracle_val_rows = [dict(row) for row in source_rows["val"]]
    oracle_test_rows = [dict(row) for row in source_rows["test"]]
    outputs = {
        "train": output / "train_index.csv",
        "internal_val": output / "internal_val_index.csv",
        "oracle_val": output / "oracle_val_family_index.csv",
        "oracle_test": output / "oracle_test_family_index.csv",
    }
    write_csv(outputs["train"], train_rows)
    write_csv(outputs["internal_val"], internal_val_rows)
    write_csv(outputs["oracle_val"], oracle_val_rows)
    write_csv(outputs["oracle_test"], oracle_test_rows)
    inventory = [
        {
            "family_uid": uid,
            "source_training_role": (
                "fit" if uid in fit_families else "internal_validation"
            ),
            "source_count": sum(family_for_row(row) == uid for row in source_rows["train"]),
        }
        for uid in sorted(EXPECTED_PRIMARY_SPLIT["train"])
    ]
    write_csv(output / "train_family_inventory.csv", inventory)
    scaling = deterministic_scaling_subsets(EXPECTED_PRIMARY_SPLIT["train"], seed)
    write_json(output / "family_scaling_subsets.json", scaling)
    lineage = {
        "schema_version": SOURCE_SPLIT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "stage": FULL_STAGE,
        "created_at_utc": utc_now(),
        "seed": seed,
        "split_method": "deterministic_family_aware_80_20_within_primary_train_families",
        "fit_family_uids": sorted(fit_families),
        "internal_validation_family_uids": sorted(internal_val_families),
        "oracle_validation_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"]),
        "oracle_test_family_uids": list(EXPECTED_PRIMARY_SPLIT["test"]),
        "normalization_allowed_index": str(outputs["train"].relative_to(root)),
        "checkpoint_selection_allowed_index": str(outputs["internal_val"].relative_to(root)),
        "oracle_policy": "evaluation_only_after_checkpoint_freeze",
        "indices": {
            name: _index_lineage(path, rows, root)
            for (name, path), rows in zip(
                outputs.items(),
                (train_rows, internal_val_rows, oracle_val_rows, oracle_test_rows),
                strict=True,
            )
        },
        "family_scaling_subsets": scaling,
    }
    write_json(output / "split_manifest.json", lineage)
    return lineage


def assert_source_training_contract(
    train_index: str | Path,
    internal_val_index: str | Path,
    split_manifest: str | Path,
) -> dict[str, Any]:
    manifest = load_json(split_manifest)
    if manifest.get("schema_version") != SOURCE_SPLIT_SCHEMA:
        raise ValueError("unsupported source split manifest")
    split_path = Path(split_manifest).expanduser().resolve()
    data_root = next(
        (parent for parent in (split_path.parent, *split_path.parents) if (parent / ROOT_MARKER_NAME).is_file()),
        None,
    )
    if data_root is None:
        raise ValueError(f"cannot discover Benchmark v2 root from {split_path}")
    train_path = Path(train_index).resolve()
    val_path = Path(internal_val_index).resolve()
    expected_train = resolve_data_path(manifest["indices"]["train"]["path"], data_root).resolve()
    expected_val = resolve_data_path(manifest["indices"]["internal_val"]["path"], data_root).resolve()
    if train_path != expected_train or val_path != expected_val:
        raise ValueError("source training may use only the manifest-declared fit/internal-validation indices")
    train_rows = read_csv(train_path)
    val_rows = read_csv(val_path)
    forbidden = set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"])
    used = {family_for_row(row) for row in train_rows + val_rows}
    overlap = sorted(used & forbidden)
    if overlap:
        raise ValueError(f"oracle families entered source fitting/selection: {overlap}")
    if sha256_file(train_path) != manifest["indices"]["train"]["sha256"]:
        raise ValueError("source train index hash differs from split manifest")
    if sha256_file(val_path) != manifest["indices"]["internal_val"]["sha256"]:
        raise ValueError("source internal-validation index hash differs from split manifest")
    return {
        "training_family_uids": sorted({family_for_row(row) for row in train_rows}),
        "selection_family_uids": sorted({family_for_row(row) for row in val_rows}),
        "normalization_family_uids": sorted({family_for_row(row) for row in train_rows}),
        "train_index_sha256": sha256_file(train_path),
        "internal_val_index_sha256": sha256_file(val_path),
        "split_manifest_sha256": sha256_file(split_manifest),
    }


def _check(name: str, passed: bool, details: Any = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def run_training_preflight(
    data_root: str | Path,
    *,
    output_dir: str | Path,
    seed: int = 20260721,
    source_checkpoint: str | Path | None = None,
    residual_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    root = benchmark_root(data_root)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_manifest = prepare_final_training_indices(root, seed=seed)
    split_root = root / f"derived/indices/{FULL_STAGE}"
    strict_path = root / f"canonical/manifests/{FULL_STAGE}_strict_validation.json"
    strict = load_json(strict_path)
    source_contract = assert_source_training_contract(
        resolve_data_path(split_manifest["indices"]["train"]["path"], root),
        resolve_data_path(split_manifest["indices"]["internal_val"]["path"], root),
        root / f"derived/indices/{FULL_STAGE}/source_response/split_manifest.json",
    )
    all_rows = read_csv(split_root / "all_index.csv")
    source_all_rows = (
        read_csv(root / f"canonical/stages/{FULL_STAGE}/source_isolation/train_index.csv")
        + read_csv(root / f"canonical/stages/{FULL_STAGE}/source_isolation/val_index.csv")
        + read_csv(root / f"canonical/stages/{FULL_STAGE}/source_isolation/test_index.csv")
    )
    source_inventory: dict[str, dict[str, Any]] = {}
    for row in source_all_rows:
        family = family_for_row(row)
        item = source_inventory.setdefault(
            family,
            {"source_count": 0, "source_names": [], "primary_role": row.get("split", "")},
        )
        item["source_count"] += 1
        item["source_names"].append(row.get("source_name", ""))
    source_required = {
        "original_x_path",
        "target_rise_path",
        "full_temperature_path",
        "layout_path",
        "source_index",
        "source_power_W",
        "ambient_K",
    }
    residual_required = {"sample_uid", "x_path", "y_path", "graph_path"}
    source_schema_ok = bool(source_all_rows) and source_required <= set(source_all_rows[0])
    residual_schema_ok = bool(all_rows) and residual_required <= set(all_rows[0])
    portable_source_paths = all(
        not Path(value).is_absolute()
        for row in source_all_rows
        for key, value in row.items()
        if (key.endswith("_path") or key == "source_dir") and value
    )
    representative_shapes: dict[str, Any] = {}
    try:
        representative = all_rows[0]
        x = np.load(resolve_data_path(representative["x_path"], root), mmap_mode="r")
        y = np.load(resolve_data_path(representative["y_path"], root), mmap_mode="r")
        representative_shapes = {"x": list(x.shape), "y": list(y.shape)}
        residual_tensor_schema_ok = tuple(x.shape) == (33, 64, 64) and tuple(y.shape) == (64, 64)
    except Exception as exc:
        residual_tensor_schema_ok = False
        representative_shapes = {"error": str(exc)}
    sample_counts = {
        split: len(read_csv(split_root / "sample_split" / f"{split}_index.csv"))
        for split in ("train", "val", "test")
    }
    family_counts = {
        split: len(read_csv(split_root / "family_split" / f"{split}_index.csv"))
        for split in ("train", "val", "test")
    }
    actual_primary = {
        split: sorted(
            {
                family_for_row(row)
                for row in read_csv(split_root / "family_split" / f"{split}_index.csv")
            }
        )
        for split in ("train", "val", "test")
    }
    provisional = root / f"derived/stages/{FULL_STAGE}/source_superposition/manifest.json"
    immutable_candidates = [
        strict_path,
        split_root / "all_index.csv",
        root / f"derived/stages/{FULL_STAGE}/source_superposition/manifest.json",
        root / f"derived/stages/{FULL_STAGE}/source_superposition/.stage_complete.json",
        root / "canonical/manifests/pilot_5x10_strict_validation.json",
        root / "canonical/manifests/pilot_10x50_strict_validation.json",
    ]
    immutable_snapshot = {
        str(path.relative_to(root)): sha256_file(path)
        for path in immutable_candidates
        if path.is_file()
    }
    usage = shutil.disk_usage(root)
    checks = [
        _check("strict_validation_42_of_42", strict.get("passed") is True and len(strict.get("checks", [])) == 42 and all(item.get("passed") for item in strict.get("checks", []))),
        _check("package_population", len(all_rows) == 10_000, len(all_rows)),
        _check("primary_family_split", actual_primary == {key: list(value) for key, value in EXPECTED_PRIMARY_SPLIT.items()}, actual_primary),
        _check("secondary_sample_split", sample_counts == {"train": 6400, "val": 800, "test": 800}, sample_counts),
        _check("primary_family_counts", family_counts == {"train": 8000, "val": 1000, "test": 1000}, family_counts),
        _check("source_training_leakage", not (set(source_contract["training_family_uids"] + source_contract["selection_family_uids"]) & (set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"]))), source_contract),
        _check("source_input_target_schema", source_schema_ok, sorted(source_all_rows[0]) if source_all_rows else []),
        _check("source_paths_root_relative", portable_source_paths),
        _check("residual_index_schema", residual_schema_ok, sorted(all_rows[0]) if all_rows else []),
        _check("residual_tensor_schema", residual_tensor_schema_ok, representative_shapes),
        _check("provisional_source_lineage", provisional.is_file(), str(provisional)),
        _check("source_checkpoint_path", source_checkpoint is None or Path(source_checkpoint).expanduser().is_file(), str(source_checkpoint or "not supplied")),
        _check("residual_checkpoint_path", residual_checkpoint is None or Path(residual_checkpoint).expanduser().is_file(), str(residual_checkpoint or "not supplied")),
        _check("disk_space", usage.free >= 50 * 1024**3, {"free_GiB": usage.free / 1024**3}),
    ]
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        cuda = {"available": False, "error": str(exc)}
    checks.append(_check("cuda_available", cuda.get("available") is True, cuda))
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "stage": FULL_STAGE,
        "created_at_utc": utc_now(),
        "data_root_id": load_json(root / ROOT_MARKER_NAME),
        "strict_validation_path": str(strict_path.relative_to(root)),
        "strict_validation_sha256": sha256_file(strict_path),
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "source_split_manifest": split_manifest,
        "source_contract": source_contract,
        "isolated_source_inventory": source_inventory,
        "sample_split_counts": sample_counts,
        "family_split_counts": family_counts,
        "cuda": cuda,
        "determinism": {"seed": seed, "family_order": "sha256(seed:family_uid)"},
        "filesystem": {
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "free_GiB": usage.free / 1024**3,
        },
        "environment": environment_report(),
        "accepted_artifact_immutability_snapshot": immutable_snapshot,
        "output_namespaces": {
            "source_response": "outputs/benchmark_v2_50family/source_response/final_train40_v1",
            "package_residual": "outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1",
            "source_superposition_version": f"derived/stages/{FULL_STAGE}/source_superposition_final_train40_source_v1",
        },
    }
    write_json(output / "preflight_report.json", report)
    lines = [
        "# Benchmark v2 Final Training Preflight",
        "",
        f"- Passed: **{report['passed']}**",
        f"- Strict validation: {sum(item.get('passed', False) for item in strict.get('checks', []))}/{len(strict.get('checks', []))}",
        f"- Packages: {len(all_rows)}",
        f"- Source fit families: {len(source_contract['training_family_uids'])}",
        f"- Source internal-validation families: {len(source_contract['selection_family_uids'])}",
        f"- CUDA available: {cuda.get('available')}",
        "",
        "## Checks",
        *[
            f"- {'PASS' if item['passed'] else 'FAIL'} `{item['name']}`: {item['details']}"
            for item in checks
        ],
    ]
    (output / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise ValueError(f"training preflight failed: {[item for item in checks if not item['passed']]}")
    return report


def environment_report() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        commit, dirty = "unavailable", True
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "torch": torch_info,
    }


def write_source_training_lineage(
    output_path: str | Path,
    *,
    contract: Mapping[str, Any],
    preflight_report: str | Path,
    run_id: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "benchmark_v2_source_training_lineage/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": FULL_STAGE,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        **dict(contract),
        "preflight_report_sha256": sha256_file(preflight_report),
        "oracle_metrics_used_for_selection": False,
        "target": "isolated_source_temperature_rise_K_per_W",
        "environment": environment_report(),
    }
    write_json(output_path, payload)
    return payload


def finalize_training_run(
    output_dir: str | Path,
    *,
    lineage_path: str | Path,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    checkpoints = {}
    for name in ("best.pt", "last.pt"):
        path = output / "checkpoints" / name
        if not path.is_file():
            raise FileNotFoundError(f"training completed without required checkpoint: {path}")
        checkpoints[name] = {"path": str(path), "sha256": sha256_file(path)}
    config_payload = json.loads(json.dumps(dict(resolved_config), default=str))
    payload = {
        "schema_version": "benchmark_v2_completed_training_run/1",
        "created_at_utc": utc_now(),
        "checkpoints": checkpoints,
        "lineage_sha256": sha256_file(lineage_path),
        "resolved_config": config_payload,
        "resolved_config_sha256": stable_json_hash(config_payload),
        "environment": environment_report(),
    }
    write_json(output / "completed_run_manifest.json", payload)
    write_learning_curve(output / "train_log.csv", output / "learning_curves.png")
    return payload


def write_learning_curve(csv_path: Path, output_path: Path) -> None:
    if not csv_path.is_file():
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = read_csv(csv_path)
    if not rows:
        return
    epoch = [int(float(row["epoch"])) for row in rows]
    metric_keys = [
        key
        for key in (
            "train_loss",
            "train_total_loss",
            "total_loss",
            "val_final_mae_K",
            "val_package_mae_K",
            "val_source_physical_mae_K",
        )
        if key in rows[0]
    ]
    if not metric_keys:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    for key in metric_keys:
        values = [
            float(row[key]) if row.get(key) not in {None, ""} else float("nan")
            for row in rows
        ]
        axis.plot(epoch, values, label=key)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def approve_source_checkpoint(
    checkpoint_path: str | Path,
    *,
    lineage_path: str | Path,
    evaluation_root: str | Path,
    output_path: str | Path,
    allow_caveats: bool = False,
    prototype_metrics: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"approval records are immutable and already exist: {output}")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    lineage = load_json(lineage_path)
    forbidden = set(EXPECTED_PRIMARY_SPLIT["val"]) | set(EXPECTED_PRIMARY_SPLIT["test"])
    fit_keys = ("training_family_uids", "normalization_family_uids", "selection_family_uids")
    leakage = sorted(
        set().union(*(set(lineage.get(key, [])) for key in fit_keys)) & forbidden
    )
    metric_files = sorted(Path(evaluation_root).expanduser().resolve().glob("*/metrics.json"))
    checkpoint_loads = False
    checkpoint_lineage_matches = False
    if checkpoint.is_file():
        try:
            import torch

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_loads = bool(payload.get("model_state_dict")) and bool(
                payload.get("normalization")
            )
            checkpoint_lineage_matches = payload.get("training_lineage") == lineage
        except Exception:
            checkpoint_loads = False
    finite = True
    catastrophic: list[str] = []
    for path in metric_files:
        payload = load_json(path)
        package = payload.get("package_reconstruction", {})
        value = package.get("mae_K")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            finite = False
        if isinstance(value, (int, float)) and float(value) > 100.0:
            catastrophic.append(f"{path.parent.name}:{value}")
        case_path = path.parent / "metrics_by_case.csv"
        if case_path.is_file():
            for row in read_csv(case_path):
                case_mae = row.get("mae_K")
                if case_mae not in {None, ""} and float(case_mae) > 100.0:
                    catastrophic.append(f"{path.parent.name}/{row.get('case_id')}:{case_mae}")
    qualitative_manifest = Path(evaluation_root) / "qualitative_audit.json"
    hard_checks = {
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_loads": checkpoint_loads,
        "checkpoint_embeds_exact_training_lineage": checkpoint_lineage_matches,
        "lineage_no_oracle_leakage": not leakage,
        "oracle_not_used_for_selection": lineage.get("oracle_metrics_used_for_selection") is False,
        "evaluation_metrics_present": len(metric_files) >= 4,
        "predictions_finite": finite,
        "no_catastrophic_family_failure": not catastrophic,
    }
    caveats = []
    qualitative_reviewed = (
        qualitative_manifest.is_file()
        and load_json(qualitative_manifest).get("reviewed") is True
    )
    if not qualitative_reviewed:
        caveats.append("qualitative source-response audit has not been signed")
    prototype_comparison: dict[str, Any] | None = None
    if prototype_metrics is None:
        caveats.append("prototype competitiveness was not supplied")
    else:
        final_test_path = Path(evaluation_root) / "oracle_primary_test" / "metrics.json"
        final_test = load_json(final_test_path).get("package_reconstruction", {}).get("mae_K")
        prototype_test = load_json(prototype_metrics).get("package_reconstruction", {}).get("mae_K")
        if final_test is None or prototype_test is None:
            hard_checks["prototype_comparison_available"] = False
        else:
            tolerance = max(0.25, 0.05 * float(prototype_test))
            competitive = float(final_test) <= float(prototype_test) + tolerance
            hard_checks["competitive_with_prototype"] = competitive
            prototype_comparison = {
                "final_oracle_test_package_mae_K": float(final_test),
                "prototype_oracle_test_package_mae_K": float(prototype_test),
                "allowed_regression_K": tolerance,
                "competitive": competitive,
            }
    if not all(hard_checks.values()):
        status = "REJECTED"
    elif caveats and allow_caveats:
        status = "APPROVED WITH CAVEATS"
    elif caveats:
        status = "REJECTED"
    else:
        status = "APPROVED"
    approval = {
        "schema_version": "source_response_checkpoint_lineage/1",
        "checkpoint_id": checkpoint.parent.parent.name,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else "",
        "architecture": "source_response_operator_v1",
        "target": "isolated_source_temperature_rise_K_per_W",
        "training_benchmark_ids": [BENCHMARK_ID],
        "training_family_uids": lineage.get("training_family_uids", []),
        "normalization_family_uids": lineage.get("normalization_family_uids", []),
        "selection_family_uids": lineage.get("selection_family_uids", []),
        "oracle_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"]),
        "oracle_metrics_used_for_selection": False,
        "approval_status": status,
        "approved": status in {"APPROVED", "APPROVED WITH CAVEATS"},
        "checks": hard_checks,
        "caveats": caveats,
        "metric_files": [str(path) for path in metric_files],
        "prototype_comparison": prototype_comparison,
        "lineage_sha256": sha256_file(lineage_path),
        "created_at_utc": utc_now(),
    }
    write_json(output, approval)
    if not approval["approved"]:
        raise ValueError(f"source checkpoint approval failed: {approval}")
    return approval


def require_approved_source_checkpoint(
    checkpoint_path: str | Path,
    approval_path: str | Path,
) -> dict[str, Any]:
    approval = load_json(approval_path)
    if approval.get("approved") is not True:
        raise ValueError("source checkpoint has not passed the approval gate")
    if approval.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("approved source checkpoint hash mismatch")
    return approval


def assert_preflight_immutability(
    data_root: str | Path,
    preflight_report: str | Path,
) -> dict[str, str]:
    root = benchmark_root(data_root)
    report = load_json(preflight_report)
    expected = report.get("accepted_artifact_immutability_snapshot", {})
    if not expected:
        raise ValueError("preflight report does not contain an accepted-artifact immutability snapshot")
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for logical, expected_hash in expected.items():
        path = resolve_data_path(logical, root)
        if not path.is_file():
            mismatches.append(f"missing:{logical}")
            continue
        actual[logical] = sha256_file(path)
        if actual[logical] != expected_hash:
            mismatches.append(f"hash_changed:{logical}")
    if mismatches:
        raise ValueError(f"accepted canonical/pilot/provisional artifacts changed: {mismatches}")
    return actual


def install_residual_dataset_sidecars(
    data_root: str | Path,
    destination: str | Path,
) -> dict[str, str]:
    """Install canonical Benchmark v2 loader sidecars beside residual split views."""
    root = benchmark_root(data_root)
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    installed: dict[str, str] = {}
    for name, logical_source in RESIDUAL_DATASET_SIDECARS.items():
        source = root / logical_source
        if not source.is_file():
            raise FileNotFoundError(f"canonical residual loader sidecar is missing: {source}")
        target = output / name
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
        installed[name] = logical_source
    return installed


def prepare_source_version_residual_indices(
    data_root: str | Path,
    *,
    source_version_root: str | Path,
) -> dict[str, Any]:
    root = benchmark_root(data_root)
    source_root = Path(source_version_root).expanduser().resolve()
    combined_path = source_root / "combined_encoded_index.csv"
    source_rows = read_csv(combined_path)
    source_by_uid = {str(row["sample_uid"]): row for row in source_rows}
    if len(source_rows) != 10_000 or len(source_by_uid) != 10_000:
        raise ValueError("approved source-superposition version must contain 10,000 unique package rows")
    manifest = load_json(source_root / "manifest.json")
    validation = load_json(source_root / "validation_report.json")
    if validation.get("ok") is not True:
        raise ValueError("source-superposition version has not passed strict validation")
    version = source_root.name
    output = root / f"derived/indices/{FULL_STAGE}/source_superposition/{version}"
    sidecars = install_residual_dataset_sidecars(root, output)
    canonical = root / f"derived/indices/{FULL_STAGE}"
    output_paths: dict[str, str] = {}
    for protocol in ("sample_split", "family_split"):
        for split in ("train", "val", "test"):
            canonical_path = canonical / protocol / f"{split}_index.csv"
            rows = read_csv(canonical_path)
            merged: list[dict[str, str]] = []
            for row in rows:
                uid = str(row["sample_uid"])
                source = source_by_uid.get(uid)
                if source is None:
                    raise ValueError(f"source-superposition version is missing {uid}")
                item = dict(row)
                for key in (
                    "source_superposition_base_path",
                    "source_superposition_residual_path",
                    "source_checkpoint",
                    "source_checkpoint_sha256",
                    "source_checkpoint_lineage_sha256",
                    "source_normalization_sha256",
                    "source_model_config_sha256",
                    "artifact_status",
                    "source_base_mode",
                ):
                    if source.get(key):
                        item[key] = source[key]
                item["source_superposition_version"] = version
                item["source_superposition_manifest_sha256"] = sha256_file(source_root / "manifest.json")
                merged.append(item)
            destination = output / protocol / f"{split}_index.csv"
            write_csv(destination, merged)
            output_paths[f"{protocol}_{split}"] = str(destination.relative_to(root))
    report = {
        "schema_version": "benchmark_v2_source_version_residual_indices/1",
        "source_superposition_version": version,
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "source_validation_sha256": sha256_file(source_root / "validation_report.json"),
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "loader_sidecars": sidecars,
        "indices": output_paths,
        "counts": {
            name: len(read_csv(root / path)) for name, path in output_paths.items()
        },
    }
    expected = {
        "sample_split_train": 6400,
        "sample_split_val": 800,
        "sample_split_test": 800,
        "family_split_train": 8000,
        "family_split_val": 1000,
        "family_split_test": 1000,
    }
    if report["counts"] != expected:
        raise ValueError(f"source-version residual index counts mismatch: {report['counts']}")
    write_json(output / "index_manifest.json", report)
    return report


def gnn_promotion_decision(
    cnn_by_sample: Sequence[Mapping[str, Any]],
    gnn_by_sample: Sequence[Mapping[str, Any]],
    *,
    runtime_overhead_fraction: float = 0.0,
    memory_overhead_fraction: float = 0.0,
    bootstrap_samples: int = 2000,
    seed: int = 20260721,
) -> dict[str, Any]:
    cnn = {str(row["sample_uid"]): row for row in cnn_by_sample}
    gnn = {str(row["sample_uid"]): row for row in gnn_by_sample}
    common = sorted(set(cnn) & set(gnn))
    if not common:
        raise ValueError("CNN/GNN comparison has no matched samples")
    improvements = [
        float(cnn[uid]["mae_K"]) - float(gnn[uid]["mae_K"]) for uid in common
    ]
    cnn_mae = sum(float(cnn[uid]["mae_K"]) for uid in common) / len(common)
    gnn_mae = sum(float(gnn[uid]["mae_K"]) for uid in common) / len(common)
    by_family: dict[str, list[float]] = {}
    for uid, improvement in zip(common, improvements, strict=True):
        family = str(cnn[uid].get("family_uid") or cnn[uid].get("case_id"))
        by_family.setdefault(family, []).append(improvement)
    improved_families = sum(sum(values) / len(values) > 0.0 for values in by_family.values())
    absolute = cnn_mae - gnn_mae
    relative = absolute / max(cnn_mae, 1e-12)
    cnn_rmse = sum(float(cnn[uid].get("rmse_K", 0.0)) for uid in common) / len(common)
    gnn_rmse = sum(float(gnn[uid].get("rmse_K", 0.0)) for uid in common) / len(common)
    cnn_peak = sum(float(cnn[uid].get("peak_temperature_abs_error_K", 0.0)) for uid in common) / len(common)
    gnn_peak = sum(float(gnn[uid].get("peak_temperature_abs_error_K", 0.0)) for uid in common) / len(common)
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(max(int(bootstrap_samples), 1)):
        bootstrap_means.append(
            sum(improvements[rng.randrange(len(improvements))] for _ in improvements)
            / len(improvements)
        )
    bootstrap_means.sort()
    lower_index = max(0, int(0.025 * len(bootstrap_means)))
    upper_index = min(len(bootstrap_means) - 1, int(0.975 * len(bootstrap_means)))
    ci = [bootstrap_means[lower_index], bootstrap_means[upper_index]]
    checks = {
        "held_out_mae_improvement_at_least_0_10_K": absolute >= 0.10,
        "relative_improvement_at_least_2_percent": relative >= 0.02,
        "at_least_3_of_5_test_families_improve": improved_families >= 3,
        "held_out_rmse_does_not_materially_regress": gnn_rmse <= cnn_rmse + 0.02,
        "peak_temperature_error_does_not_materially_regress": gnn_peak <= cnn_peak + 0.05,
        "runtime_overhead_justified": float(runtime_overhead_fraction) <= 0.25,
        "memory_overhead_justified": float(memory_overhead_fraction) <= 0.25,
        "paired_bootstrap_95pct_ci_excludes_zero": ci[0] > 0.0,
    }
    return {
        "cnn_mae_K": cnn_mae,
        "gnn_mae_K": gnn_mae,
        "absolute_improvement_K": absolute,
        "relative_improvement": relative,
        "improved_family_count": improved_families,
        "cnn_mean_sample_rmse_K": cnn_rmse,
        "gnn_mean_sample_rmse_K": gnn_rmse,
        "cnn_peak_temperature_mae_K": cnn_peak,
        "gnn_peak_temperature_mae_K": gnn_peak,
        "paired_bootstrap_improvement_95pct_CI_K": ci,
        "runtime_overhead_fraction": float(runtime_overhead_fraction),
        "memory_overhead_fraction": float(memory_overhead_fraction),
        "checks": checks,
        "promote": all(checks.values()),
        "recommendation": "PROMOTE GNN" if all(checks.values()) else "OMIT GNN FROM PRIMARY MODEL",
    }
