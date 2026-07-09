#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

RECOMMENDED_SPLITS = [
    ("case02", "case09"),
    ("case04", "case02"),
    ("case09", "case04"),
    ("case10", "case06"),
]

REQUIRED_PATH_COLUMNS = ("x_path", "y_path", "prediction_path", "residual_path")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build held-out benchmark-case train/val/test index splits.")
    parser.add_argument(
        "--base-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power",
        type=Path,
        help="Dataset root containing combined_encoded_index.csv.",
    )
    parser.add_argument(
        "--out-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_heldout/package_plus_power",
        type=Path,
        help="Root where held-out split directories will be written.",
    )
    parser.add_argument("--test-case", default="case02", help="Case ID used only for test_index.csv.")
    parser.add_argument("--val-case", default="case09", help="Case ID used only for val_index.csv.")
    parser.add_argument(
        "--preset",
        choices=["recommended"],
        default=None,
        help="Build a named preset collection of held-out splits.",
    )
    args = parser.parse_args()

    base_root = args.base_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    index_path = find_combined_index(base_root)
    rows, fieldnames = read_index(index_path)
    validate_required_columns(fieldnames)
    validate_paths(rows, base_root)

    requested_splits = RECOMMENDED_SPLITS if args.preset == "recommended" else [(args.test_case, args.val_case)]
    all_cases = sorted({row["case_id"] for row in rows})
    created: list[dict[str, Any]] = []
    for test_case, val_case in requested_splits:
        if test_case == val_case:
            raise SystemExit(f"test-case and val-case must be different, got {test_case}")
        if test_case not in all_cases:
            raise SystemExit(f"test case {test_case!r} not found. Available cases: {', '.join(all_cases)}")
        if val_case not in all_cases:
            raise SystemExit(f"validation case {val_case!r} not found. Available cases: {', '.join(all_cases)}")

        split_dir = out_root / f"holdout_{test_case}_val_{val_case}"
        result = build_split(
            rows,
            fieldnames,
            split_dir=split_dir,
            base_root=base_root,
            index_path=index_path,
            test_case=test_case,
            val_case=val_case,
            all_cases=all_cases,
        )
        created.append(result)
        print_split_summary(result)

    print("Held-out case split generation complete")
    print(f"Base index: {index_path}")
    print(f"Output root: {out_root}")
    print(f"Splits created: {len(created)}")
    return 0


def build_split(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    split_dir: Path,
    base_root: Path,
    index_path: Path,
    test_case: str,
    val_case: str,
    all_cases: list[str],
) -> dict[str, Any]:
    split_rows = {
        "train": [rewrite_split(row, "train") for row in rows if row["case_id"] not in {test_case, val_case}],
        "val": [rewrite_split(row, "val") for row in rows if row["case_id"] == val_case],
        "test": [rewrite_split(row, "test") for row in rows if row["case_id"] == test_case],
    }
    for split_name, selected in split_rows.items():
        if not selected:
            raise SystemExit(f"{split_dir.name}: {split_name} split would be empty")

    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, selected in split_rows.items():
        write_csv(split_dir / f"{split_name}_index.csv", fieldnames, selected)

    counts = {
        split_name: {
            "num_samples": len(selected),
            "cases": dict(sorted(Counter(row["case_id"] for row in selected).items())),
        }
        for split_name, selected in split_rows.items()
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_root": str(base_root),
        "base_index": str(index_path),
        "split_dir": str(split_dir),
        "test_case": test_case,
        "val_case": val_case,
        "train_cases": [case for case in all_cases if case not in {test_case, val_case}],
        "all_cases": all_cases,
        "counts": counts,
        "path_validation": {
            "validated_columns": list(REQUIRED_PATH_COLUMNS),
            "status": "passed",
            "note": "Index paths point to existing tensors/files; tensors were not copied.",
        },
        "notes": [
            "Rows preserve the source index schema.",
            "The split column is rewritten to the held-out split label for correct dataset metadata.",
            "This split is for held-out benchmark-family generalization, not random in-distribution evaluation.",
        ],
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(split_dir, manifest)
    return manifest


def find_combined_index(base_root: Path) -> Path:
    candidates = [
        base_root / "combined_encoded_index.csv",
        base_root / "encoded_index.csv",
        base_root / "dataset_index.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no combined index found under {base_root}")


def read_index(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"{path} has no header")
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows, fieldnames


def validate_required_columns(fieldnames: list[str]) -> None:
    required = {"case_id", "split", *REQUIRED_PATH_COLUMNS}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"combined index is missing required columns: {', '.join(missing)}")


def validate_paths(rows: list[dict[str, str]], base_root: Path) -> None:
    missing: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        for column in REQUIRED_PATH_COLUMNS:
            value = row.get(column, "")
            if not value:
                missing.append(f"row {row_index} column {column}: empty")
                continue
            if not resolve_path(value, base_root).exists():
                missing.append(f"row {row_index} column {column}: {value}")
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
    if missing:
        detail = "\n".join(missing)
        raise SystemExit(f"missing required files in index paths:\n{detail}")


def resolve_path(path_value: str, base_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        base_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def rewrite_split(row: dict[str, str], split_name: str) -> dict[str, str]:
    rewritten = dict(row)
    rewritten["split"] = split_name
    return rewritten


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(split_dir: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    lines = [
        "# ChipTherm Held-Out Case Split",
        "",
        "This directory contains index files only. It does not copy encoded tensors, HotSpot labels, physics predictions, or residual files.",
        "",
        f"- Test case: `{manifest['test_case']}`",
        f"- Validation case: `{manifest['val_case']}`",
        f"- Train cases: `{', '.join(manifest['train_cases'])}`",
        "",
        "## Files",
        "",
        "- `train_index.csv`: all cases except validation and test cases",
        "- `val_index.csv`: validation case only",
        "- `test_index.csv`: held-out test case only",
        "- `split_manifest.json`: reproducibility metadata and sample counts",
        "",
        "## Sample Counts",
        "",
        f"- Train: {counts['train']['num_samples']}",
        f"- Validation: {counts['val']['num_samples']}",
        f"- Test: {counts['test']['num_samples']}",
        "",
        "The CSV schema matches the source combined index. The `split` column is rewritten to the new held-out split label.",
        "",
    ]
    (split_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def print_split_summary(manifest: dict[str, Any]) -> None:
    print("")
    print(f"Created: {manifest['split_dir']}")
    print(f"  test_case: {manifest['test_case']}")
    print(f"  val_case:  {manifest['val_case']}")
    for split_name in ("train", "val", "test"):
        split_counts = manifest["counts"][split_name]
        case_summary = ", ".join(f"{case}:{count}" for case, count in split_counts["cases"].items())
        print(f"  {split_name}: {split_counts['num_samples']} samples ({case_summary})")


if __name__ == "__main__":
    raise SystemExit(main())
