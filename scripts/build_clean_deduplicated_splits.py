#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_ALGORITHM = "sha256"
HASH_SCHEMA = "chiptherm_xy_tensor_hash_v1"
REQUIRED_PATH_COLUMNS = ("x_path", "y_path", "prediction_path", "residual_path")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean deduplicated ChipTherm train/val/test indexes.")
    parser.add_argument(
        "--source-index",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power/combined_encoded_index.csv",
        type=Path,
    )
    parser.add_argument(
        "--out-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean/package_plus_power",
        type=Path,
    )
    parser.add_argument("--train-frac", default=0.80, type=float)
    parser.add_argument("--val-frac", default=0.10, type=float)
    parser.add_argument("--test-frac", default=0.10, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--residual-atol", default=1.0e-3, type=float)
    parser.add_argument("--verify-only", action="store_true", help="Validate an existing output directory without rebuilding it.")
    args = parser.parse_args()

    if args.verify_only:
        verify_existing_outputs(args.out_root.resolve())
        return 0

    validate_fractions(args.train_frac, args.val_frac, args.test_frac)
    source_index = args.source_index.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_csv_rows(source_index)
    if not rows:
        raise SystemExit(f"{source_index} does not contain any rows")
    if "split" not in fieldnames:
        raise SystemExit(f"{source_index} is missing required split column")

    print(f"Hashing {len(rows)} source rows by X/Y tensor content...")
    hashed_rows = hash_and_validate_rows(rows, source_index=source_index, residual_atol=args.residual_atol)
    groups = group_by_hash(hashed_rows)
    canonical_records, duplicate_group_records, verification = canonicalize_groups(groups)
    assign_case_stratified_splits(
        canonical_records,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    validate_clean_records(canonical_records)

    canonical_records.sort(key=lambda item: (item.row["case_id"], item.row["split"], item.row["sample_uid"]))
    csv_records = [item.row for item in canonical_records]

    write_csv(out_root / "combined_encoded_index.csv", fieldnames, csv_records)
    write_jsonl(out_root / "combined_encoded_index.jsonl", canonical_records)
    for split in ("train", "val", "test"):
        split_rows = [item.row for item in canonical_records if item.row["split"] == split]
        write_csv(out_root / f"{split}_index.csv", fieldnames, split_rows)

    write_duplicate_groups(out_root / "duplicate_groups.csv", duplicate_group_records)
    manifest = build_manifest(
        source_index=source_index,
        out_root=out_root,
        source_row_count=len(rows),
        canonical_records=canonical_records,
        duplicate_group_records=duplicate_group_records,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        residual_atol=args.residual_atol,
        verification=verification,
    )
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root / "README.md", source_index, manifest)

    verify_existing_outputs(out_root)
    print()
    print("Clean deduplicated split build complete")
    print(f"Source rows: {manifest['source_row_count']}")
    print(f"Unique samples: {manifest['unique_sample_count']}")
    print(f"Removed duplicate rows: {manifest['removed_duplicate_rows']}")
    print(
        "Train/val/test: "
        f"{manifest['actual_counts']['train']} / {manifest['actual_counts']['val']} / {manifest['actual_counts']['test']}"
    )
    print(f"Cross-split duplicate hashes: {manifest['verification']['cross_split_duplicate_hashes']}")
    print(f"Output: {out_root}")
    return 0


class HashedRow:
    def __init__(self, row: dict[str, str], x_hash: str, y_hash: str, xy_hash: str) -> None:
        self.row = row
        self.x_hash = x_hash
        self.y_hash = y_hash
        self.xy_hash = xy_hash


def validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1.0e-9:
        raise SystemExit(f"split fractions must sum to 1.0, got {total}")
    if min(train_frac, val_frac, test_frac) < 0.0:
        raise SystemExit("split fractions must be non-negative")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"source index does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = [column for column in ("sample_uid", "case_id", *REQUIRED_PATH_COLUMNS) if column not in fieldnames]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    return fieldnames, rows


def hash_and_validate_rows(rows: list[dict[str, str]], *, source_index: Path, residual_atol: float) -> list[HashedRow]:
    hashed_rows: list[HashedRow] = []
    seen_sample_uids: set[str] = set()
    errors: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        sample_uid = row.get("sample_uid", "")
        if not sample_uid:
            errors.append(f"row {row_index} missing sample_uid")
        if sample_uid in seen_sample_uids:
            errors.append(f"duplicate source sample_uid {sample_uid}")
        seen_sample_uids.add(sample_uid)

        for column in REQUIRED_PATH_COLUMNS:
            path = resolve_index_path(row.get(column, ""), source_index)
            if not path.exists():
                errors.append(f"{sample_uid} missing {column}: {row.get(column, '')}")

        if errors:
            continue

        x_path = resolve_index_path(row["x_path"], source_index)
        y_path = resolve_index_path(row["y_path"], source_index)
        prediction_path = resolve_index_path(row["prediction_path"], source_index)
        residual_path = resolve_index_path(row["residual_path"], source_index)
        try:
            x = np.load(x_path)
            y = np.load(y_path)
            prediction = np.load(prediction_path)
            residual = np.load(residual_path)
        except Exception as exc:  # pragma: no cover - defensive error path
            errors.append(f"{sample_uid} failed to load tensors: {exc}")
            continue

        if not np.isfinite(x).all():
            errors.append(f"{sample_uid} X contains non-finite values")
        if not np.isfinite(y).all():
            errors.append(f"{sample_uid} Y contains non-finite values")
        if prediction.shape != y.shape:
            errors.append(f"{sample_uid} prediction shape {prediction.shape} does not match Y shape {y.shape}")
        if residual.shape != y.shape:
            errors.append(f"{sample_uid} residual shape {residual.shape} does not match Y shape {y.shape}")
        if prediction.shape == y.shape and residual.shape == y.shape:
            residual_error = np.max(np.abs((y.astype(np.float64) - prediction.astype(np.float64)) - residual.astype(np.float64)))
            if float(residual_error) > residual_atol:
                errors.append(f"{sample_uid} residual check failed: max error {residual_error:.6g} > {residual_atol}")

        x_hash = hash_array(x)
        y_hash = hash_array(y)
        xy_hash = hash_arrays(x, y)
        hashed_rows.append(HashedRow(dict(row), x_hash=x_hash, y_hash=y_hash, xy_hash=xy_hash))

        if row_index % 500 == 0:
            print(f"  hashed {row_index}/{len(rows)} rows")

    if errors:
        raise SystemExit("clean split input validation failed:\n" + "\n".join(errors[:50]))
    return hashed_rows


def hash_array(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    update_hash_with_array(hasher, array)
    return hasher.hexdigest()


def hash_arrays(x: np.ndarray, y: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(HASH_SCHEMA.encode("utf-8"))
    hasher.update(b"\0X\0")
    update_hash_with_array(hasher, x)
    hasher.update(b"\0Y\0")
    update_hash_with_array(hasher, y)
    return hasher.hexdigest()


def update_hash_with_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(tuple(int(size) for size in contiguous.shape)).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(contiguous.tobytes(order="C"))


def group_by_hash(rows: list[HashedRow]) -> dict[str, list[HashedRow]]:
    groups: dict[str, list[HashedRow]] = defaultdict(list)
    for row in rows:
        groups[row.xy_hash].append(row)
    return dict(groups)


def canonicalize_groups(groups: dict[str, list[HashedRow]]) -> tuple[list[HashedRow], list[dict[str, Any]], dict[str, Any]]:
    canonical_records: list[HashedRow] = []
    duplicate_group_records: list[dict[str, Any]] = []
    errors: list[str] = []
    exact_duplicate_groups = 0

    for xy_hash, group in sorted(groups.items()):
        case_ids = {item.row["case_id"] for item in group}
        if len(case_ids) != 1:
            errors.append(f"xy_hash {xy_hash} has multiple case_id values: {sorted(case_ids)}")
            continue
        x_hashes = {item.x_hash for item in group}
        y_hashes = {item.y_hash for item in group}
        if len(x_hashes) != 1 or len(y_hashes) != 1:
            errors.append(f"xy_hash {xy_hash} has inconsistent x/y hashes")
            continue
        if len(group) > 1 and not verify_exact_group_arrays(group):
            errors.append(f"xy_hash {xy_hash} failed np.array_equal duplicate verification")
            continue

        canonical = min(
            group,
            key=lambda item: (
                item.row.get("sample_uid", ""),
                item.row.get("x_path", ""),
                item.row.get("y_path", ""),
                item.row.get("prediction_path", ""),
                item.row.get("residual_path", ""),
            ),
        )
        canonical_records.append(canonical)
        if len(group) > 1:
            exact_duplicate_groups += 1
            duplicate_group_records.append(
                {
                    "xy_hash": xy_hash,
                    "x_hash": canonical.x_hash,
                    "y_hash": canonical.y_hash,
                    "canonical_sample_uid": canonical.row["sample_uid"],
                    "duplicate_count": len(group),
                    "removed_count": len(group) - 1,
                    "case_id": canonical.row["case_id"],
                    "all_sample_uids": ";".join(sorted(item.row["sample_uid"] for item in group)),
                    "all_dataset_sources": ";".join(sorted({item.row.get("dataset_source", "") for item in group})),
                    "all_original_sample_uids": ";".join(sorted({item.row.get("original_sample_uid", "") for item in group})),
                    "all_source_splits": ";".join(sorted({item.row.get("split", "") for item in group})),
                }
            )

    if errors:
        raise SystemExit("deduplication failed:\n" + "\n".join(errors[:50]))
    return (
        canonical_records,
        duplicate_group_records,
        {
            "exact_duplicate_groups_verified": exact_duplicate_groups,
            "hash_collision_or_inconsistent_group_errors": 0,
        },
    )


def verify_exact_group_arrays(group: list[HashedRow]) -> bool:
    first = group[0]
    first_x = np.load(resolve_index_path(first.row["x_path"], None))
    first_y = np.load(resolve_index_path(first.row["y_path"], None))
    for item in group[1:]:
        x = np.load(resolve_index_path(item.row["x_path"], None))
        y = np.load(resolve_index_path(item.row["y_path"], None))
        if not np.array_equal(first_x, x) or not np.array_equal(first_y, y):
            return False
    return True


def assign_case_stratified_splits(records: list[HashedRow], *, seed: int, train_frac: float, val_frac: float) -> None:
    rng = random.Random(seed)
    by_case: dict[str, list[HashedRow]] = defaultdict(list)
    for record in records:
        by_case[record.row["case_id"]].append(record)

    for case_id in sorted(by_case):
        case_records = sorted(by_case[case_id], key=lambda item: item.row["sample_uid"])
        rng.shuffle(case_records)
        n = len(case_records)
        train_count = int(n * train_frac)
        val_count = int(n * val_frac)
        for index, record in enumerate(case_records):
            if index < train_count:
                record.row["split"] = "train"
            elif index < train_count + val_count:
                record.row["split"] = "val"
            else:
                record.row["split"] = "test"


def validate_clean_records(records: list[HashedRow]) -> None:
    errors: list[str] = []
    sample_uids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    hashes_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    all_sample_uids: set[str] = set()
    all_hashes: set[str] = set()

    for record in records:
        split = record.row.get("split", "")
        sample_uid = record.row["sample_uid"]
        if split not in sample_uids_by_split:
            errors.append(f"{sample_uid} has invalid split {split!r}")
            continue
        if sample_uid in all_sample_uids:
            errors.append(f"canonical sample appears more than once: {sample_uid}")
        all_sample_uids.add(sample_uid)
        if record.xy_hash in all_hashes:
            errors.append(f"canonical content hash appears more than once: {record.xy_hash}")
        all_hashes.add(record.xy_hash)
        sample_uids_by_split[split].add(sample_uid)
        hashes_by_split[split].add(record.xy_hash)
        for column in REQUIRED_PATH_COLUMNS:
            path = resolve_index_path(record.row[column], None)
            if not path.exists():
                errors.append(f"{sample_uid} missing {column}: {record.row[column]}")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        uid_overlap = sample_uids_by_split[left] & sample_uids_by_split[right]
        hash_overlap = hashes_by_split[left] & hashes_by_split[right]
        if uid_overlap:
            errors.append(f"{left}/{right} sample_uid overlap: {sorted(uid_overlap)[:5]}")
        if hash_overlap:
            errors.append(f"{left}/{right} content hash overlap: {sorted(hash_overlap)[:5]}")

    if sum(len(values) for values in sample_uids_by_split.values()) != len(records):
        errors.append("split totals do not equal unique sample total")
    if errors:
        raise SystemExit("clean split validation failed:\n" + "\n".join(errors[:50]))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_jsonl(path: Path, records: list[HashedRow]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            payload = dict(record.row)
            payload["deduplication"] = {
                "hash_schema": HASH_SCHEMA,
                "hash_algorithm": HASH_ALGORITHM,
                "x_hash": record.x_hash,
                "y_hash": record.y_hash,
                "xy_hash": record.xy_hash,
            }
            fp.write(json.dumps(payload, sort_keys=True) + "\n")


def write_duplicate_groups(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "xy_hash",
        "x_hash",
        "y_hash",
        "canonical_sample_uid",
        "duplicate_count",
        "removed_count",
        "case_id",
        "all_sample_uids",
        "all_dataset_sources",
        "all_original_sample_uids",
        "all_source_splits",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_manifest(
    *,
    source_index: Path,
    out_root: Path,
    source_row_count: int,
    canonical_records: list[HashedRow],
    duplicate_group_records: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    residual_atol: float,
    verification: dict[str, Any],
) -> dict[str, Any]:
    split_counts = Counter(record.row["split"] for record in canonical_records)
    per_case_counts: dict[str, dict[str, int]] = {}
    for record in canonical_records:
        case_id = record.row["case_id"]
        split = record.row["split"]
        per_case_counts.setdefault(case_id, {"train": 0, "val": 0, "test": 0})
        per_case_counts[case_id][split] += 1

    removed_duplicate_rows = source_row_count - len(canonical_records)
    verification = dict(verification)
    verification.update(
        {
            "zero_cross_split_leakage": True,
            "cross_split_duplicate_hashes": 0,
            "missing_required_paths": 0,
            "residual_consistency_checked": True,
            "residual_atol": residual_atol,
        }
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_index": str(source_index),
        "out_root": str(out_root),
        "source_row_count": source_row_count,
        "unique_sample_count": len(canonical_records),
        "duplicate_group_count": len(duplicate_group_records),
        "removed_duplicate_rows": removed_duplicate_rows,
        "split_seed": seed,
        "target_fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "actual_counts": {split: int(split_counts.get(split, 0)) for split in ("train", "val", "test")},
        "per_case_counts": dict(sorted(per_case_counts.items())),
        "hash_algorithm": HASH_ALGORITHM,
        "hash_schema": HASH_SCHEMA,
        "verification": verification,
        "notes": (
            "Rows are canonical representatives of exact X/Y tensor duplicate groups. "
            "Large tensors are not copied; paths point to dataset_v1_context_ablation/package_plus_power artifacts."
        ),
    }


def write_readme(path: Path, source_index: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Dataset v2 Clean: package_plus_power

This directory contains index-only clean splits derived from:

`{source_index}`

No HotSpot runs were regenerated and no large `.npy` tensors were copied.
Rows were deduplicated by SHA-256 hashes over the actual X and Y tensor
contents, including shape, dtype, and raw contiguous bytes.

## Counts

- Source rows: {manifest['source_row_count']}
- Unique samples: {manifest['unique_sample_count']}
- Removed duplicate rows: {manifest['removed_duplicate_rows']}
- Train/val/test: {manifest['actual_counts']['train']} / {manifest['actual_counts']['val']} / {manifest['actual_counts']['test']}

## Files

- `combined_encoded_index.csv`: all canonical samples with clean split labels.
- `train_index.csv`, `val_index.csv`, `test_index.csv`: model-ready clean splits.
- `combined_encoded_index.jsonl`: row metadata plus content hashes.
- `duplicate_groups.csv`: audit table for removed duplicate rows.
- `split_manifest.json`: split counts and leakage verification metadata.

Use these indexes as the first leakage-free baseline for ChipTherm model
comparisons. Clean MAE may be worse than earlier exploratory runs because exact
duplicate X/Y leakage has been removed.
"""
    path.write_text(text, encoding="utf-8")


def verify_existing_outputs(out_root: Path) -> None:
    required = [
        out_root / "combined_encoded_index.csv",
        out_root / "train_index.csv",
        out_root / "val_index.csv",
        out_root / "test_index.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"cannot verify missing output files: {', '.join(str(path) for path in missing)}")

    split_hashes: dict[str, set[str]] = {}
    split_uids: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        _, rows = read_csv_rows(out_root / f"{split}_index.csv")
        hashes: set[str] = set()
        sample_uids: set[str] = set()
        for row in rows:
            for column in REQUIRED_PATH_COLUMNS:
                path = resolve_index_path(row[column], out_root / f"{split}_index.csv")
                if not path.exists():
                    raise SystemExit(f"{row['sample_uid']} missing {column}: {row[column]}")
            x = np.load(resolve_index_path(row["x_path"], out_root / f"{split}_index.csv"))
            y = np.load(resolve_index_path(row["y_path"], out_root / f"{split}_index.csv"))
            xy_hash = hash_arrays(x, y)
            if xy_hash in hashes:
                raise SystemExit(f"{split} contains duplicate content hash {xy_hash}")
            if row["sample_uid"] in sample_uids:
                raise SystemExit(f"{split} contains duplicate sample_uid {row['sample_uid']}")
            hashes.add(xy_hash)
            sample_uids.add(row["sample_uid"])
            if row.get("split") != split:
                raise SystemExit(f"{row['sample_uid']} has split={row.get('split')!r} inside {split}_index.csv")
        split_hashes[split] = hashes
        split_uids[split] = sample_uids
        split_counts[split] = len(rows)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        hash_overlap = split_hashes[left] & split_hashes[right]
        uid_overlap = split_uids[left] & split_uids[right]
        if hash_overlap:
            raise SystemExit(f"{left}/{right} content hash leakage: {len(hash_overlap)} hashes")
        if uid_overlap:
            raise SystemExit(f"{left}/{right} sample_uid leakage: {len(uid_overlap)} sample_uids")

    print("Clean split verification passed")
    print(f"Train/val/test: {split_counts['train']} / {split_counts['val']} / {split_counts['test']}")
    print("Cross-split duplicate hashes: 0")


def resolve_index_path(path_value: str, index_path: Path | None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, Path.cwd() / path]
    if index_path is not None:
        candidates.append(index_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    raise SystemExit(main())
