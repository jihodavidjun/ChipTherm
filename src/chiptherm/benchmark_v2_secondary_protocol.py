from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_v2 import BENCHMARK_ID
from .benchmark_v2_pipeline import PATH_SEMANTICS, ROOT_MARKER_NAME, resolve_data_path
from .benchmark_v2_training import RESIDUAL_DATASET_SIDECARS


PROTOCOL_NAME = "benchmark_v2_family_35_5_10_seed_3510"
PROTOCOL_SCHEMA = "benchmark_v2_secondary_family_protocol/1"
DEFAULT_SEED = 3510
EXPECTED_FAMILIES = tuple(f"f{index:03d}" for index in range(1, 51))
EXPECTED_COUNTS = {
    "familiar_train": 5_600,
    "familiar_internal_val": 700,
    "familiar_test": 700,
    "heldout_validation": 1_000,
    "heldout_test": 2_000,
}
REQUIRED_PACKAGE_PATHS = (
    "x_path",
    "y_path",
    "graph_path",
    "layout_path",
    "power_path",
    "package_path",
)


def generate_family_split(
    families: Sequence[str] = EXPECTED_FAMILIES,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    values = [str(value) for value in families]
    if len(values) != 50 or len(set(values)) != 50:
        raise ValueError(f"secondary protocol requires exactly 50 unique families, got {len(values)}/{len(set(values))}")
    if set(values) != set(EXPECTED_FAMILIES):
        missing = sorted(set(EXPECTED_FAMILIES) - set(values))
        extra = sorted(set(values) - set(EXPECTED_FAMILIES))
        raise ValueError(f"Benchmark v2 family inventory mismatch; missing={missing}, extra={extra}")
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    split = {
        "train": sorted(shuffled[:35]),
        "validation": sorted(shuffled[35:40]),
        "test": sorted(shuffled[40:]),
    }
    validate_family_split(split)
    return split


def validate_family_split(split: Mapping[str, Sequence[str]]) -> None:
    expected = {"train": 35, "validation": 5, "test": 10}
    actual = {name: len(split.get(name, ())) for name in expected}
    if actual != expected:
        raise ValueError(f"family split counts must be {expected}, got {actual}")
    sets = {name: set(split[name]) for name in expected}
    overlaps = {
        f"{left}_{right}": sorted(sets[left] & sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"family partitions overlap: {overlaps}")
    if set().union(*sets.values()) != set(EXPECTED_FAMILIES):
        raise ValueError("family partitions do not cover the complete Benchmark v2 inventory")


def workload_ordinal(row: Mapping[str, str]) -> int:
    raw = str(row.get("workload_ordinal") or "")
    if raw:
        ordinal = int(float(raw))
    else:
        candidates = (
            str(row.get("workload_uid") or ""),
            str(row.get("sample_uid") or ""),
            str(row.get("original_sample_uid") or ""),
        )
        match = next(
            (
                found
                for value in candidates
                if (found := re.search(r"(?:^|_)w(\d{3})(?:_|$)", value)) is not None
            ),
            None,
        )
        if match is None:
            raise ValueError(f"cannot determine workload ordinal for {row.get('sample_uid', '<unknown>')}")
        ordinal = int(match.group(1))
    if not 1 <= ordinal <= 200:
        raise ValueError(f"workload ordinal outside 1..200: {ordinal}")
    return ordinal


def workload_role(ordinal: int) -> str:
    if ordinal <= 160:
        return "familiar_train"
    if ordinal <= 180:
        return "familiar_internal_val"
    return "familiar_test"


def family_for_row(row: Mapping[str, str]) -> str:
    return str(row.get("family_uid") or row.get("case_id") or "")


def sample_uid_for_row(row: Mapping[str, str]) -> str:
    return str(row.get("sample_uid") or row.get("original_sample_uid") or "")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if len(value) == 40 else "0" * 40


def load_canonical_package_rows(data_root: Path) -> list[dict[str, str]]:
    family_root = data_root / "derived/indices/full_50x200/family_split"
    rows = [
        row
        for split in ("train", "val", "test")
        for row in read_csv(family_root / f"{split}_index.csv")
    ]
    return validate_package_inventory(rows)


def load_source_isolation_rows(data_root: Path) -> list[dict[str, str]]:
    source_root = data_root / "canonical/stages/full_50x200/source_isolation"
    rows = [
        row
        for split in ("train", "val", "test")
        for row in read_csv(source_root / f"{split}_index.csv")
    ]
    if not rows:
        raise ValueError("source-isolation inventory is empty")
    source_ids = [
        str(row.get("source_response_uid") or f"{sample_uid_for_row(row)}:{row.get('source_index', '')}")
        for row in rows
    ]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source-isolation inventory contains duplicate source rows")
    families = {family_for_row(row) for row in rows}
    if families != set(EXPECTED_FAMILIES):
        raise ValueError(f"source-isolation inventory does not cover all 50 families: {sorted(families)}")
    return sorted(rows, key=row_sort_key)


def validate_package_inventory(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if len(rows) != 10_000:
        raise ValueError(f"Benchmark v2 package inventory must contain 10,000 rows, got {len(rows)}")
    identities = [sample_uid_for_row(row) for row in rows]
    if not all(identities) or len(identities) != len(set(identities)):
        raise ValueError("package inventory has blank or duplicate sample_uid values")
    counts = {family: 0 for family in EXPECTED_FAMILIES}
    ordinals: dict[str, set[int]] = {family: set() for family in EXPECTED_FAMILIES}
    for row in rows:
        family = family_for_row(row)
        if family not in counts:
            raise ValueError(f"unknown Benchmark v2 family in package inventory: {family}")
        counts[family] += 1
        ordinals[family].add(workload_ordinal(row))
    bad = {
        family: {"count": counts[family], "ordinal_count": len(ordinals[family])}
        for family in EXPECTED_FAMILIES
        if counts[family] != 200 or ordinals[family] != set(range(1, 201))
    }
    if bad:
        raise ValueError(f"families without exactly workloads 1..200: {bad}")
    return sorted((dict(row) for row in rows), key=row_sort_key)


def row_sort_key(row: Mapping[str, str]) -> tuple[str, int, str, int]:
    source_index = int(float(row.get("source_index") or 0))
    return family_for_row(row), workload_ordinal(row), sample_uid_for_row(row), source_index


def partition_package_rows(
    rows: Sequence[Mapping[str, str]], split: Mapping[str, Sequence[str]]
) -> dict[str, list[dict[str, str]]]:
    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])
    result = {name: [] for name in EXPECTED_COUNTS}
    for row in rows:
        family = family_for_row(row)
        item = dict(row)
        item["protocol_name"] = PROTOCOL_NAME
        if family in train:
            role = workload_role(workload_ordinal(row))
            item["protocol_family_role"] = "train"
        elif family in validation:
            role = "heldout_validation"
            item["protocol_family_role"] = "validation"
        elif family in test:
            role = "heldout_test"
            item["protocol_family_role"] = "test"
        else:
            raise ValueError(f"family absent from secondary split: {family}")
        item["protocol_partition"] = role
        item["split"] = {
            "familiar_train": "train",
            "familiar_internal_val": "val",
            "familiar_test": "test",
            "heldout_validation": "val",
            "heldout_test": "test",
        }[role]
        result[role].append(item)
    actual = {name: len(values) for name, values in result.items()}
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"secondary package counts must be {EXPECTED_COUNTS}, got {actual}")
    return result


def partition_source_rows(
    rows: Sequence[Mapping[str, str]],
    split: Mapping[str, Sequence[str]],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, str]]]:
    train_families = list(split["train"])
    ordered = sorted(
        train_families,
        key=lambda uid: (hashlib.sha256(f"{seed}:source:{uid}".encode()).hexdigest(), uid),
    )
    selection = set(ordered[:7])
    fit = set(ordered[7:])
    validation = set(split["validation"])
    test = set(split["test"])
    result = {
        "train": [],
        "internal_val": [],
        "familiar_all": [],
        "heldout_validation": [],
        "heldout_test": [],
    }
    for row in rows:
        family = family_for_row(row)
        item = dict(row)
        item["protocol_name"] = PROTOCOL_NAME
        if family in fit:
            role = "train"
        elif family in selection:
            role = "internal_val"
        elif family in validation:
            role = "heldout_validation"
        elif family in test:
            role = "heldout_test"
        else:
            raise ValueError(f"family absent from secondary split: {family}")
        item["protocol_partition"] = role
        item["split"] = "train" if role == "train" else "val" if role in {"internal_val", "heldout_validation"} else "test"
        result[role].append(item)
        if family in fit or family in selection:
            result["familiar_all"].append(dict(item, protocol_partition="familiar_all"))
    for values in result.values():
        values.sort(key=row_sort_key)
    return result


def partition_package_rows_by_identity(
    rows: Sequence[Mapping[str, str]], split: Mapping[str, Sequence[str]]
) -> dict[str, list[dict[str, str]]]:
    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])
    result = {name: [] for name in EXPECTED_COUNTS}
    for row in rows:
        family = family_for_row(row)
        if family in train:
            role = workload_role(workload_ordinal(row))
            family_role = "train"
        elif family in validation:
            role, family_role = "heldout_validation", "validation"
        elif family in test:
            role, family_role = "heldout_test", "test"
        else:
            raise ValueError(f"family absent from secondary split: {family}")
        item = dict(row)
        item.update(
            protocol_name=PROTOCOL_NAME,
            protocol_family_role=family_role,
            protocol_partition=role,
            split={
                "familiar_train": "train",
                "familiar_internal_val": "val",
                "familiar_test": "test",
                "heldout_validation": "val",
                "heldout_test": "test",
            }[role],
        )
        result[role].append(item)
    return result


def build_protocol_indices(
    data_root: Path,
    output_root: Path,
    *,
    split: Mapping[str, Sequence[str]],
    config_manifest: Path,
    source_version_root: Path | None = None,
    validate_files: bool = True,
) -> dict[str, Any]:
    validate_family_split(split)
    package_rows = load_canonical_package_rows(data_root)
    package = partition_package_rows(package_rows, split)
    source = partition_source_rows(load_source_isolation_rows(data_root), split)
    if validate_files:
        _validate_resolving_paths(package_rows, data_root, REQUIRED_PACKAGE_PATHS)
        _validate_resolving_paths(
            [row for values in source.values() for row in values],
            data_root,
            ("original_x_path", "target_rise_path", "full_temperature_path", "layout_path"),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    package_root = output_root / "package"
    source_root = output_root / "source_response"
    views = {
        "familiar_train": package_root / "sample_split/train_index.csv",
        "familiar_internal_val": package_root / "sample_split/val_index.csv",
        "familiar_test": package_root / "sample_split/test_index.csv",
        "heldout_validation": package_root / "family_split/val_index.csv",
        "heldout_test": package_root / "family_split/test_index.csv",
    }
    for role, path in views.items():
        write_csv(path, package[role])
    write_csv(
        package_root / "family_split/train_index.csv",
        package["familiar_train"] + package["familiar_internal_val"] + package["familiar_test"],
    )
    generation = {
        "train": package["familiar_train"],
        "val": package["familiar_internal_val"] + package["heldout_validation"],
        "test": package["familiar_test"] + package["heldout_test"],
    }
    for name, values in generation.items():
        write_csv(package_root / "source_generation" / f"{name}_index.csv", values)
    write_csv(package_root / "all_index.csv", package_rows)

    source_views = tuple(source)
    for name in source_views:
        write_csv(source_root / f"{name}_index.csv", source[name])
    source_fit_families = sorted({family_for_row(row) for row in source["train"]})
    source_selection_families = sorted({family_for_row(row) for row in source["internal_val"]})

    _install_sidecars(data_root, package_root)
    runtime_manifest: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_name": PROTOCOL_NAME,
        "benchmark_id": BENCHMARK_ID,
        "dataset_version": "benchmark_v2_50family/full_50x200",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "path_semantics": PATH_SEMANTICS,
        "split": {name: list(values) for name, values in split.items()},
        "counts": EXPECTED_COUNTS,
        "source_row_counts": {name: len(source[name]) for name in source_views},
        "source_response_contract": {
            "split_method": "deterministic_80_20_family_split_within_secondary_train_families",
            "fit_family_uids": source_fit_families,
            "internal_validation_family_uids": source_selection_families,
            "heldout_validation_family_uids": list(split["validation"]),
            "heldout_test_family_uids": list(split["test"]),
            "normalization_family_uids": source_fit_families,
        },
        "normalization_contract": {
            "allowed_package_index": _relative(views["familiar_train"], data_root),
            "allowed_source_index": _relative(source_root / "train_index.csv", data_root),
            "family_uids": list(split["train"]),
            "forbidden_family_uids": sorted(set(split["validation"]) | set(split["test"])),
            "fit_scope": "35 training families, workload ordinals 1..160 only",
        },
        "checkpoint_selection_contract": {
            "package_index": _relative(views["familiar_internal_val"], data_root),
            "source_index": _relative(source_root / "internal_val_index.csv", data_root),
            "heldout_validation_is_oracle_for_source_and_training_selection": False,
            "final_test_is_never_used_for_selection": True,
        },
        "indices": {},
        "config_split_manifest": str(config_manifest),
        "config_split_manifest_sha256": sha256_file(config_manifest),
    }
    runtime_manifest["sample_uid_partition_sha256"] = {
        role: stable_hash([sample_uid_for_row(row) for row in package[role]])
        for role in EXPECTED_COUNTS
    }
    for path in sorted(output_root.rglob("*_index.csv")):
        runtime_manifest["indices"][_relative(path, output_root)] = {
            "path": _relative(path, data_root),
            "sha256": sha256_file(path),
            "row_count": len(read_csv(path)),
        }

    if source_version_root is not None:
        residual = attach_source_version(
            data_root,
            output_root,
            package,
            source_version_root=source_version_root,
        )
        runtime_manifest["source_superposition"] = residual

    manifest_path = output_root / "protocol_index_manifest.json"
    manifest_path.write_text(json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runtime_manifest


def attach_source_version(
    data_root: Path,
    output_root: Path,
    package: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    source_version_root: Path,
) -> dict[str, Any]:
    combined = source_version_root / "combined_encoded_index.csv"
    source_rows = read_csv(combined)
    by_uid = {sample_uid_for_row(row): row for row in source_rows}
    if len(by_uid) != 10_000:
        raise ValueError(f"source-superposition version must contain 10,000 unique rows, got {len(by_uid)}")
    version = source_version_root.name
    destination = output_root / "source_superposition" / version
    source_keys = (
        "source_superposition_base_path",
        "source_superposition_residual_path",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_checkpoint_config_sha256",
        "source_normalization_sha256",
        "source_model_config_sha256",
        "source_count",
        "source_base_mode",
        "artifact_status",
    )

    def merged(role: str) -> list[dict[str, str]]:
        result = []
        for canonical in package[role]:
            uid = sample_uid_for_row(canonical)
            source = by_uid.get(uid)
            if source is None:
                raise ValueError(f"source-superposition version is missing {uid}")
            item = dict(canonical)
            for key in source_keys:
                if source.get(key, "") != "":
                    item[key] = source[key]
            item["source_superposition_version"] = version
            result.append(item)
        return result

    views = {
        "sample_split/train_index.csv": merged("familiar_train"),
        "sample_split/val_index.csv": merged("familiar_internal_val"),
        "sample_split/test_index.csv": merged("familiar_test"),
        "family_split/val_index.csv": merged("heldout_validation"),
        "family_split/test_index.csv": merged("heldout_test"),
    }
    views["family_split/train_index.csv"] = (
        views["sample_split/train_index.csv"]
        + views["sample_split/val_index.csv"]
        + views["sample_split/test_index.csv"]
    )
    for logical, rows in views.items():
        write_csv(destination / logical, rows)
    _install_sidecars(data_root, destination)
    return {
        "version": version,
        "combined_index": _relative(combined, data_root),
        "combined_index_sha256": sha256_file(combined),
        "index_root": _relative(destination, data_root),
        "counts": {logical: len(rows) for logical, rows in views.items()},
    }


def _install_sidecars(data_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, logical in RESIDUAL_DATASET_SIDECARS.items():
        source = data_root / logical
        if not source.is_file():
            raise FileNotFoundError(f"canonical loader sidecar is missing: {source}")
        target = destination / name
        if target.exists() and sha256_file(target) == sha256_file(source):
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)


def _validate_resolving_paths(
    rows: Sequence[Mapping[str, str]], data_root: Path, fields: Sequence[str]
) -> None:
    failures: list[str] = []
    for row in rows:
        uid = sample_uid_for_row(row)
        for field in fields:
            value = str(row.get(field) or "")
            if not value:
                failures.append(f"{uid}:{field}:blank")
                continue
            if Path(value).is_absolute():
                failures.append(f"{uid}:{field}:absolute:{value}")
                continue
            if not resolve_data_path(value, data_root).is_file():
                failures.append(f"{uid}:{field}:missing:{value}")
        if len(failures) >= 20:
            break
    if failures:
        raise ValueError(f"protocol rows contain unresolved/nonportable required paths: {failures}")


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def validate_normalizer_provenance(
    manifest: Mapping[str, Any],
    contributing_families: Sequence[str],
    *,
    allowed_families: Sequence[str] | None = None,
    require_all: bool = True,
) -> None:
    allowed = set(allowed_families or manifest["split"]["train"])
    actual = set(str(value) for value in contributing_families)
    invalid = actual - allowed
    missing = allowed - actual
    if invalid or (require_all and missing):
        raise ValueError(
            f"normalizer provenance violates its train-only family contract; missing={sorted(missing)}, forbidden={sorted(invalid)}"
        )


def primary_artifact_hashes(repo_root: Path, data_root: Path | None = None) -> dict[str, str]:
    paths = [
        repo_root / "configs/benchmark_v2_50family/splits/primary_family_split.yaml",
        repo_root / "configs/benchmark_v2_50family/splits/sample_split_proposal.yaml",
    ]
    if data_root is not None:
        paths.extend(
            data_root / f"derived/indices/full_50x200/{protocol}/{split}_index.csv"
            for protocol in ("sample_split", "family_split")
            for split in ("train", "val", "test")
        )
    return {str(path): sha256_file(path) for path in paths if path.is_file()}
