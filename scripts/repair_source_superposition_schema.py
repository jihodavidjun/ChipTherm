#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402


COMPAT_COLUMNS = ("prediction_path", "residual_path")
PATH_COLUMNS = (
    "x_path",
    "y_path",
    "prediction_path",
    "residual_path",
    "graph_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "benchmark_path",
    "source_dir",
    "source_superposition_base_path",
    "source_superposition_residual_path",
    "source_layout_path",
    "source_power_path",
    "source_package_path",
    "source_hotspot_path",
)
INDEX_PATTERNS = (
    "combined_encoded_index.csv",
    "train_index.csv",
    "val_index.csv",
    "test_index.csv",
    "_input_splits/train_index.csv",
    "_input_splits/val_index.csv",
    "_input_splits/test_index.csv",
    "sample_split_extension/train_index.csv",
    "sample_split_extension/val_index.csv",
    "sample_split_extension/test_index.csv",
    "sample_split_extension/combined_index.csv",
    "family_split_extension/train_index.csv",
    "family_split_extension/val_index.csv",
    "family_split_extension/test_index.csv",
    "family_split_extension/all_index.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair source-superposition CSV schema without regenerating maps.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--canonical-index", required=True, type=Path)
    parser.add_argument("--smoke-cases", nargs="*", default=["case11", "case17", "case19", "case20"])
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    canonical_rows = {row["sample_uid"]: row for row in read_rows(args.canonical_index.expanduser().resolve())}
    if not canonical_rows:
        raise SystemExit(f"canonical index has no rows: {args.canonical_index}")

    repaired_files = []
    total_rows = 0
    for relative in INDEX_PATTERNS:
        path = source_root / relative
        if not path.exists():
            continue
        changed, row_count = repair_csv(path, canonical_rows)
        total_rows += row_count
        repaired_files.append({"path": repo_relative(path), "rows": row_count, "changed": changed})

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "canonical_index": repo_relative(args.canonical_index.expanduser().resolve()),
        "repaired_files": repaired_files,
        "total_rows_seen": total_rows,
    }
    validation = validate(source_root, args.smoke_cases)
    report.update(validation)
    (source_root / "schema_repair_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Source-superposition schema repair complete")
    print(f"Rows checked: {total_rows}")
    for item in repaired_files:
        print(f"  {item['path']}: rows={item['rows']} changed={item['changed']}")
    print(f"Missing prediction_path columns: {validation['missing_prediction_path_column_count']}")
    print(f"Unresolved paths: {validation['unresolved_path_count']}")
    if validation["errors"]:
        for error in validation["errors"][:20]:
            print(f"  - {error}")
        return 2
    return 0


def repair_csv(path: Path, canonical_rows: dict[str, dict[str, str]]) -> tuple[bool, int]:
    rows = read_rows(path)
    if not rows:
        return False, 0
    fieldnames = read_fieldnames(path)
    changed = False
    for column in COMPAT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
            changed = True
    for row in rows:
        canonical = canonical_rows.get(row["sample_uid"], {})
        for column in COMPAT_COLUMNS:
            before = row.get(column)
            if before is None:
                row[column] = canonical.get(column, "")
                changed = True
            elif before:
                row[column] = repo_relative(resolve_path(before))
            elif canonical.get(column):
                row[column] = repo_relative(resolve_path(canonical[column]))
                changed = True
            else:
                row[column] = ""
        if row.get("source_base_mode") != "source_superposition_v1":
            row["source_base_mode"] = "source_superposition_v1"
            if "source_base_mode" not in fieldnames:
                fieldnames.append("source_base_mode")
            changed = True
    if changed:
        write_rows(path, rows, fieldnames)
    return changed, len(rows)


def validate(source_root: Path, smoke_cases: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    unresolved: list[dict[str, str]] = []
    missing_prediction_columns = 0
    index_paths = [source_root / relative for relative in INDEX_PATTERNS if (source_root / relative).exists()]
    for path in index_paths:
        fieldnames = read_fieldnames(path)
        if "prediction_path" not in fieldnames:
            missing_prediction_columns += 1
            errors.append(f"{repo_relative(path)} missing prediction_path column")
        if "residual_path" not in fieldnames:
            errors.append(f"{repo_relative(path)} missing residual_path column")
        for row in read_rows(path):
            if row.get("source_base_mode") != "source_superposition_v1":
                errors.append(f"{repo_relative(path)} {row.get('sample_uid')}: source_base_mode={row.get('source_base_mode')}")
            base_path = row.get("source_superposition_base_path", "")
            if not base_path:
                errors.append(f"{repo_relative(path)} {row.get('sample_uid')}: missing source_superposition_base_path")
            for column in PATH_COLUMNS:
                value = row.get(column, "")
                if not value:
                    continue
                if not resolve_path(value).exists():
                    unresolved.append({"index": repo_relative(path), "sample_uid": row.get("sample_uid", ""), "column": column, "value": value})
    errors.extend(loader_smoke(source_root, smoke_cases))
    return {
        "missing_prediction_path_column_count": missing_prediction_columns,
        "unresolved_path_count": len(unresolved),
        "unresolved_paths": unresolved[:100],
        "errors": errors,
    }


def loader_smoke(source_root: Path, smoke_cases: list[str]) -> list[str]:
    errors: list[str] = []
    candidates = [
        source_root / "sample_split_extension" / "train_index.csv",
        source_root / "family_split_extension" / "val_index.csv",
        source_root / "family_split_extension" / "test_index.csv",
        source_root / "combined_encoded_index.csv",
    ]
    checked: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        rows = read_rows(path)
        for case_id in [case for case in smoke_cases if case not in checked]:
            row = next((item for item in rows if item.get("case_id") == case_id), None)
            if row is None:
                continue
            tmp = path.parent / f".schema_repair_smoke_{case_id}.csv"
            write_rows(tmp, [row], read_fieldnames(path))
            try:
                sample = ChipThermDataset(tmp, target="residual", return_metadata=True, return_graph=True)[0]
                if tuple(sample["x"].shape) != (13, 64, 64):
                    errors.append(f"{case_id}: x shape={tuple(sample['x'].shape)}")
                if tuple(sample["physics"].shape) != (64, 64):
                    errors.append(f"{case_id}: base shape={tuple(sample['physics'].shape)}")
                if "physics_v1" in sample:
                    errors.append(f"{case_id}: physics_v1 unexpectedly loaded from compatibility placeholder")
                metadata = sample.get("metadata", {})
                if metadata.get("effective_prediction_path") != row.get("source_superposition_base_path"):
                    errors.append(f"{case_id}: effective base did not resolve to source_superposition_base_path")
                graph = sample.get("graph")
                if graph is None:
                    errors.append(f"{case_id}: graph missing")
                else:
                    if graph["node_features"].shape[-1] != 24:
                        errors.append(f"{case_id}: node feature dim={graph['node_features'].shape[-1]}")
                    if graph["edge_features"].shape[-1] != 15:
                        errors.append(f"{case_id}: edge feature dim={graph['edge_features'].shape[-1]}")
            except Exception as exc:
                errors.append(f"{case_id}: loader smoke failed: {exc}")
            finally:
                if tmp.exists():
                    tmp.unlink()
            checked.add(case_id)
    missing = set(smoke_cases) - checked
    if missing:
        errors.append(f"loader smoke did not find cases: {sorted(missing)}")
    return errors


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        return next(reader)


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
