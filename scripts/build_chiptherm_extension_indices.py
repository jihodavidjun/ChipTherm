#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ChipTherm extension split protocols and optional 20-case merged indices.")
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    args = parser.parse_args()

    ext_root = args.extension_root.resolve()
    out_root = (args.out_root or (ext_root / "indices")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ext_rows = read_rows(ext_root / "all_extension_index.csv")
    if not ext_rows:
        ext_rows = read_rows(ext_root / "combined_encoded_index.csv")
    if not ext_rows:
        raise SystemExit(f"missing all_extension_index.csv or combined_encoded_index.csv under {ext_root}")

    sample_split_ext = split_per_case(ext_rows, args.train_frac, args.val_frac)
    write_index_tree(out_root / "sample_split_extension", sample_split_ext)

    family_ext = {
        "train": [dict(row, split="train") for row in ext_rows if row["case_id"] in {f"case{i:02d}" for i in range(11, 17)}],
        "val": [dict(row, split="val") for row in ext_rows if row["case_id"] in {"case17", "case18"}],
        "test": [dict(row, split="test") for row in ext_rows if row["case_id"] in {"case19", "case20"}],
    }
    write_index_tree(out_root / "family_split_extension", family_ext, all_name="all_index.csv")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extension_root": rel(ext_root),
        "original_root": rel(args.original_root.resolve()) if args.original_root else None,
        "protocols": ["sample_split_extension", "family_split_extension"],
    }
    if args.original_root is not None:
        original_root = args.original_root.resolve()
        original = {
            "train": read_rows(original_root / "train_index.csv"),
            "val": read_rows(original_root / "val_index.csv"),
            "test": read_rows(original_root / "test_index.csv"),
        }
        sample_20 = {split: [*original[split], *sample_split_ext[split]] for split in ("train", "val", "test")}
        write_index_tree(out_root / "sample_split_20case", sample_20)
        family_20 = {
            "train": [*original["train"], *family_ext["train"]],
            "val": family_ext["val"],
            "test": family_ext["test"],
        }
        write_index_tree(out_root / "family_split_20case", family_20, all_name="all_index.csv")
        write_rows(out_root / "original_case01_case10_test_index.csv", original["test"])
        manifest["protocols"].extend(["sample_split_20case", "family_split_20case"])
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote split protocols to {out_root}")
    return 0


def split_per_case(rows: list[dict[str, str]], train_frac: float, val_frac: float) -> dict[str, list[dict[str, str]]]:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    out = {"train": [], "val": [], "test": []}
    for case_id in sorted(by_case):
        case_rows = sorted(by_case[case_id], key=lambda row: row["sample_uid"])
        n = len(case_rows)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        splits = {
            "train": case_rows[:n_train],
            "val": case_rows[n_train : n_train + n_val],
            "test": case_rows[n_train + n_val :],
        }
        for split, split_rows in splits.items():
            out[split].extend(dict(row, split=split) for row in split_rows)
    return out


def write_index_tree(root: Path, splits: dict[str, list[dict[str, str]]], *, all_name: str = "combined_index.csv") -> None:
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split in ("train", "val", "test"):
        rows = sorted(splits.get(split, []), key=lambda row: (row.get("case_id", ""), row.get("sample_uid", "")))
        write_rows(root / f"{split}_index.csv", rows)
        all_rows.extend(rows)
    write_rows(root / all_name, sorted(all_rows, key=lambda row: (row.get("case_id", ""), row.get("sample_uid", ""))))
    validate_no_overlap(splits)


def validate_no_overlap(splits: dict[str, list[dict[str, str]]]) -> None:
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            uid = row["sample_uid"]
            if uid in seen:
                raise SystemExit(f"sample_uid {uid} appears in both {seen[uid]} and {split}")
            seen[uid] = split


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["sample_uid", "original_sample_uid", "case_id", "dataset_source", "split", "x_path", "y_path"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
