#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATH_COLUMNS = ("x_path", "y_path", "prediction_path", "residual_path")
SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leakage-free per-case ChipTherm split indexes.")
    parser.add_argument(
        "--source-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance/package_plus_power",
        type=Path,
    )
    parser.add_argument(
        "--out-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_per_case/package_plus_power",
        type=Path,
    )
    parser.add_argument("--train-count", default=320, type=int)
    parser.add_argument("--val-count", default=40, type=int)
    parser.add_argument("--test-count", default=41, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    rows, fieldnames = read_source_rows(source_root)
    validate_columns(fieldnames)

    print(f"Hashing {len(rows)} rows by X/Y tensor contents")
    hashed_rows = attach_hashes(rows, source_root)
    groups = group_by_hash(hashed_rows)
    canonical_rows, duplicate_groups = canonicalize_groups(groups)

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical_rows:
        by_case[row["case_id"]].append(row)

    root_manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "out_root": repo_relative(out_root),
        "seed": int(args.seed),
        "requested_counts": {
            "train": int(args.train_count),
            "val": int(args.val_count),
            "test": int(args.test_count),
        },
        "source_rows": len(rows),
        "unique_xy_samples": len(canonical_rows),
        "duplicate_groups": len(duplicate_groups),
        "removed_duplicate_rows": sum(len(group) - 1 for group in duplicate_groups),
        "cases": {},
    }

    out_root.mkdir(parents=True, exist_ok=True)
    for case_id in sorted(by_case):
        case_rows = sorted(by_case[case_id], key=canonical_sort_key)
        case_manifest = write_case_split(
            case_id=case_id,
            rows=case_rows,
            fieldnames=fieldnames,
            source_root=source_root,
            out_root=out_root,
            train_count=int(args.train_count),
            val_count=int(args.val_count),
            test_count=int(args.test_count),
            seed=int(args.seed),
            duplicate_groups=duplicate_groups,
        )
        root_manifest["cases"][case_id] = case_manifest

    write_json(out_root / "split_manifest.json", root_manifest)
    write_root_readme(out_root / "README.md", root_manifest)

    print("Per-case split build complete")
    print(f"Source rows: {len(rows)}")
    print(f"Unique samples: {len(canonical_rows)}")
    print(f"Removed duplicate rows: {root_manifest['removed_duplicate_rows']}")
    for case_id, payload in root_manifest["cases"].items():
        counts = payload["split_counts"]
        print(
            f"{case_id}: unique={payload['total_unique_samples']} "
            f"train/val/test={counts['train']}/{counts['val']}/{counts['test']} "
            f"hash_overlaps={payload['cross_split_content_hash_overlaps']}"
        )
    print(f"Output: {out_root}")
    return 0


def read_source_rows(source_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    source_index = source_root / "combined_encoded_index.csv"
    if source_index.exists():
        return read_csv(source_index)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for split in SPLITS:
        split_rows, split_fields = read_csv(source_root / f"{split}_index.csv")
        rows.extend(split_rows)
        fieldnames = fieldnames or split_fields
    if fieldnames is None:
        raise SystemExit(f"no index files found under {source_root}")
    return rows, fieldnames


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise SystemExit(f"missing index file: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows, fieldnames


def validate_columns(fieldnames: list[str]) -> None:
    required = {"sample_uid", "case_id", "split", *REQUIRED_PATH_COLUMNS}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"source index missing columns: {', '.join(missing)}")


def attach_hashes(rows: list[dict[str, str]], source_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        for column in REQUIRED_PATH_COLUMNS:
            path = resolve_path(row[column], source_root)
            if not path.exists():
                raise SystemExit(f"{row.get('sample_uid', '<unknown>')} missing {column}: {path}")
        x_hash = tensor_hash(resolve_path(row["x_path"], source_root))
        y_hash = tensor_hash(resolve_path(row["y_path"], source_root))
        xy_hash = hashlib.sha256(f"{x_hash}:{y_hash}".encode("ascii")).hexdigest()
        enriched = dict(row)
        enriched["_x_hash"] = x_hash
        enriched["_y_hash"] = y_hash
        enriched["_xy_hash"] = xy_hash
        output.append(enriched)
        if index % 500 == 0:
            print(f"  hashed {index}/{len(rows)}")
    return output


def tensor_hash(path: Path) -> str:
    array = np.load(path, mmap_mode="r")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(size) for size in contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def group_by_hash(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["_xy_hash"]].append(row)
    return groups


def canonicalize_groups(groups: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    canonical_rows: list[dict[str, str]] = []
    duplicate_groups: list[list[dict[str, str]]] = []
    for xy_hash, group in groups.items():
        case_ids = {row["case_id"] for row in group}
        if len(case_ids) != 1:
            raise SystemExit(f"content hash {xy_hash} appears in multiple cases: {sorted(case_ids)}")
        if len(group) > 1:
            verify_duplicate_group(group)
            duplicate_groups.append(group)
        canonical_rows.append(sorted(group, key=canonical_sort_key)[0])
    return canonical_rows, duplicate_groups


def verify_duplicate_group(group: list[dict[str, str]]) -> None:
    first = group[0]
    first_x = np.load(resolve_path(first["x_path"], REPO_ROOT))
    first_y = np.load(resolve_path(first["y_path"], REPO_ROOT))
    for row in group[1:]:
        x = np.load(resolve_path(row["x_path"], REPO_ROOT))
        y = np.load(resolve_path(row["y_path"], REPO_ROOT))
        if not np.array_equal(first_x, x) or not np.array_equal(first_y, y):
            raise SystemExit(f"SHA-256 collision or inconsistent duplicate group at {first['_xy_hash']}")


def canonical_sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("sample_uid", ""), row.get("x_path", ""), row.get("y_path", ""))


def write_case_split(
    *,
    case_id: str,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    source_root: Path,
    out_root: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    duplicate_groups: list[list[dict[str, str]]],
) -> dict[str, Any]:
    if any(row["case_id"] != case_id for row in rows):
        raise SystemExit(f"{case_id} split received rows from other cases")
    rng = np.random.default_rng(seed + stable_case_offset(case_id))
    order = np.arange(len(rows))
    rng.shuffle(order)
    shuffled = [rows[int(index)] for index in order]
    counts = choose_counts(len(shuffled), train_count, val_count, test_count)
    split_rows = {
        "train": shuffled[: counts["train"]],
        "val": shuffled[counts["train"] : counts["train"] + counts["val"]],
        "test": shuffled[counts["train"] + counts["val"] : counts["train"] + counts["val"] + counts["test"]],
    }
    split_rows = {split: [with_split(row, split) for row in split_rows[split]] for split in SPLITS}
    validate_case_split(case_id, split_rows)

    case_dir = out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_csv(case_dir / f"{split}_index.csv", fieldnames, [strip_private_keys(row) for row in split_rows[split]])
    copy_optional_manifest(source_root, case_dir, "feature_manifest.json")
    copy_optional_manifest(source_root, case_dir, "context_manifest.json")

    case_duplicate_groups = [
        group for group in duplicate_groups if group and group[0]["case_id"] == case_id
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "seed": seed,
        "total_unique_samples": len(rows),
        "split_counts": {split: len(split_rows[split]) for split in SPLITS},
        "requested_counts": {"train": train_count, "val": val_count, "test": test_count},
        "used_requested_exact_counts": len(rows) >= train_count + val_count + test_count,
        "case_label_purity": True,
        "duplicate_groups_in_source_case": len(case_duplicate_groups),
        "removed_duplicate_rows_in_source_case": sum(len(group) - 1 for group in case_duplicate_groups),
        "content_hash_overlaps": content_overlap_report(split_rows),
        "cross_split_content_hash_overlaps": sum(content_overlap_report(split_rows).values()),
        "sample_uid_overlaps": sample_uid_overlap_report(split_rows),
        "path_existence": path_existence_report(split_rows),
        "source_note": "Index-only per-case diagnostic split. Tensor paths are reused; no X/Y/physics/residual tensors are copied.",
    }
    write_json(case_dir / "split_manifest.json", manifest)
    write_case_readme(case_dir / "README.md", manifest)
    return manifest


def copy_optional_manifest(source_root: Path, case_dir: Path, filename: str) -> None:
    source = source_root / filename
    if source.exists():
        (case_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def choose_counts(total: int, train_count: int, val_count: int, test_count: int) -> dict[str, int]:
    requested = train_count + val_count + test_count
    if total >= requested:
        return {"train": train_count, "val": val_count, "test": test_count}
    train = int(round(total * 0.80))
    val = int(round(total * 0.10))
    test = total - train - val
    if total >= 3:
        if val <= 0:
            val = 1
            train = max(1, train - 1)
        if test <= 0:
            test = 1
            train = max(1, train - 1)
    return {"train": train, "val": val, "test": test}


def with_split(row: dict[str, str], split: str) -> dict[str, str]:
    rewritten = dict(row)
    rewritten["split"] = split
    return rewritten


def strip_private_keys(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def validate_case_split(case_id: str, split_rows: dict[str, list[dict[str, str]]]) -> None:
    all_rows = [row for rows in split_rows.values() for row in rows]
    if any(row["case_id"] != case_id for row in all_rows):
        raise SystemExit(f"{case_id} has impure case labels")
    sample_sets = {split: {row["sample_uid"] for row in rows} for split, rows in split_rows.items()}
    hash_sets = {split: {row["_xy_hash"] for row in rows} for split, rows in split_rows.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sample_overlap = sample_sets[a] & sample_sets[b]
        hash_overlap = hash_sets[a] & hash_sets[b]
        if sample_overlap:
            raise SystemExit(f"{case_id} sample_uid overlap {a}/{b}: {sorted(sample_overlap)[:5]}")
        if hash_overlap:
            raise SystemExit(f"{case_id} content-hash overlap {a}/{b}: {sorted(hash_overlap)[:3]}")


def content_overlap_report(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    sets = {split: {row["_xy_hash"] for row in rows} for split, rows in split_rows.items()}
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def sample_uid_overlap_report(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    sets = {split: {row["sample_uid"] for row in rows} for split, rows in split_rows.items()}
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def path_existence_report(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, bool]:
    result = {column: True for column in REQUIRED_PATH_COLUMNS}
    for rows in split_rows.values():
        for row in rows:
            for column in REQUIRED_PATH_COLUMNS:
                if not resolve_path(row[column], REPO_ROOT).exists():
                    result[column] = False
    if not all(result.values()):
        raise SystemExit(f"missing tensor paths in split: {result}")
    return result


def stable_case_offset(case_id: str) -> int:
    return int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_case_readme(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["split_counts"]
    text = f"""# ChipTherm Per-Case Split: {manifest['case_id']}

Index-only leakage-free diagnostic split for one package family.

Counts:

- train: {counts['train']}
- val: {counts['val']}
- test: {counts['test']}

Tensor paths point back to the source clean impedance dataset. No large arrays
are copied. This split is intended only for per-case upper-bound diagnostics.
"""
    path.write_text(text, encoding="utf-8")


def write_root_readme(path: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Per-Case Upper-Bound Splits

This directory contains one independent leakage-free train/validation/test
index split per benchmark case. It reuses tensor paths from:

`{manifest['source_root']}`

The experiment is diagnostic: each model is trained and tested within a single
case to estimate an upper bound relative to one universal cross-case model.
"""
    path.write_text(text, encoding="utf-8")


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, base / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
