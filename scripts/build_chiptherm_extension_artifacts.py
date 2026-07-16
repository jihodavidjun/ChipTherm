#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_COLUMNS = (
    "x_path",
    "y_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "benchmark_path",
    "source_dir",
    "original_temp_path",
    "temp_layer0_path",
    "prediction_path",
    "residual_path",
    "graph_path",
    "source_superposition_base_path",
    "source_superposition_residual_path",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training-ready encoded/metadata/graph artifacts for ChipTherm extension rows.")
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--index-name", default="all_extension_index.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repair-indices-only", action="store_true", help="Rewrite existing artifact CSV path columns to the canonical repo-relative contract.")
    parser.add_argument("--audit-paths-only", action="store_true", help="Audit existing artifact CSV path references without rebuilding tensors.")
    args = parser.parse_args()

    extension_root = args.extension_root.resolve()
    out_root = args.out_root.resolve()
    encoded_root = out_root / "encoded_package_plus_power"
    graph_root = out_root / "package_plus_power_graph"
    index = extension_root / args.index_name
    if args.repair_indices_only:
        changed = repair_artifact_indices(out_root)
        audit = audit_artifact_paths(out_root)
        print(f"Repaired CSV path values: {changed}")
        print(f"Rows checked: {audit['rows_checked']}")
        print(f"Path references checked: {audit['path_references_checked']}")
        print(f"Unresolved paths: {audit['unresolved_count']}")
        if audit["unresolved_count"]:
            for item in audit["unresolved"][:20]:
                print(f"  {item['csv']} {item['sample_uid']} {item['column']}: {item['value']}")
            raise SystemExit(2)
        return 0
    if args.audit_paths_only:
        audit = audit_artifact_paths(out_root)
        print(f"Rows checked: {audit['rows_checked']}")
        print(f"Path references checked: {audit['path_references_checked']}")
        print(f"Unresolved paths: {audit['unresolved_count']}")
        if audit["unresolved_count"]:
            for item in audit["unresolved"][:20]:
                print(f"  {item['csv']} {item['sample_uid']} {item['column']}: {item['value']}")
            raise SystemExit(2)
        return 0
    if not index.exists():
        raise SystemExit(f"missing extension index: {index}")
    out_root.mkdir(parents=True, exist_ok=True)
    adapter_index = out_root / "extension_canonical_adapter_index.csv"

    commands = [
        [
            "python3",
            "scripts/encode_dataset.py",
            "--index",
            str(adapter_index),
            "--out-dir",
            str(encoded_root),
        ],
        [
            "python3",
            "scripts/build_metadata_features.py",
            "--dataset-root",
            str(encoded_root),
        ],
        [
            "python3",
            "scripts/build_graph_features.py",
            "--source-root",
            str(encoded_root),
            "--out-root",
            str(graph_root),
            *(["--overwrite"] if args.overwrite else []),
        ],
    ]
    print(f"Adapter index: {adapter_index}", flush=True)
    if not args.dry_run:
        rows = build_adapter_index(index, adapter_index)
        if not rows:
            raise SystemExit(f"{index} has no rows")
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if command[1] == "scripts/encode_dataset.py":
                finalize_encoded_dataset(encoded_root)
    return 0


def build_adapter_index(source_index: Path, adapter_index: Path) -> list[dict[str, str]]:
    with source_index.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = [adapter_row(row, source_index.parent) for row in reader]
    if not rows:
        return []
    fieldnames = preferred_fieldnames(rows)
    with adapter_index.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def adapter_row(row: dict[str, str], dataset_root: Path) -> dict[str, str]:
    out = dict(row)
    for key in ("layout_path", "power_path", "package_path", "hotspot_path", "benchmark_path", "source_dir", "y_path"):
        value = out.get(key, "")
        if value:
            out[key] = repo_relative(resolve_index_path(value, dataset_root))
    out["temp_layer0_path"] = out.get("y_path", "")
    out["original_temp_path"] = out.get("y_path", "")
    out.setdefault("original_sample_uid", out.get("sample_uid", ""))
    required = ("sample_uid", "case_id", "layout_path", "power_path", "hotspot_path", "y_path")
    missing = [key for key in required if not out.get(key)]
    if missing:
        raise SystemExit(f"{row.get('sample_uid', '<unknown>')} missing required adapter fields: {missing}")
    for key in ("layout_path", "power_path", "hotspot_path", "y_path"):
        path = REPO_ROOT / out[key]
        if not path.exists():
            raise SystemExit(f"{row.get('sample_uid', '<unknown>')} adapter path does not exist: {out[key]}")
    return out


def finalize_encoded_dataset(encoded_root: Path) -> None:
    metadata_path = encoded_root / "encoding_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"encoding did not produce metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    encoded = int(metadata.get("num_encoded", 0))
    failed = int(metadata.get("num_failed", 0))
    if encoded <= 0:
        failures = metadata.get("failures", [])[:20]
        raise SystemExit(f"encoding produced zero samples. First failures: {failures}")
    if failed:
        failures = metadata.get("failures", [])[:20]
        raise SystemExit(f"encoding failed for {failed} samples. First failures: {failures}")
    encoded_index = encoded_root / "encoded_index.csv"
    encoded_jsonl = encoded_root / "encoded_index.jsonl"
    combined_index = encoded_root / "combined_encoded_index.csv"
    combined_jsonl = encoded_root / "combined_encoded_index.jsonl"
    if not encoded_index.exists():
        raise SystemExit(f"encoding did not produce {encoded_index}")
    shutil.copy2(encoded_index, combined_index)
    if encoded_jsonl.exists():
        shutil.copy2(encoded_jsonl, combined_jsonl)
    rows = read_rows(combined_index)
    if not rows:
        raise SystemExit(f"{combined_index} has no rows")
    fieldnames = list(rows[0].keys())
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        write_rows(encoded_root / f"{split}_index.csv", split_rows, fieldnames)
    context_manifest = {
        "schema_version": 1,
        "context_channels": [
            "total_power_W",
            "package_width_mm",
            "package_height_mm",
            "cell_size_x_mm",
            "cell_size_y_mm",
        ],
        "context_channel_indices": [8, 9, 10, 11, 12],
        "source": "scripts/encode_dataset.py package_plus_power adapter",
    }
    (encoded_root / "context_manifest.json").write_text(json.dumps(context_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repair_artifact_indices(artifact_root: Path, *, repo_root: Path = REPO_ROOT) -> int:
    artifact_root = artifact_root.resolve()
    changed = 0
    for csv_path in sorted(artifact_root.rglob("*.csv")):
        rows = read_rows(csv_path)
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        row_changed = False
        for row in rows:
            for column in PATH_COLUMNS:
                value = row.get(column, "")
                if not value:
                    continue
                resolved = resolve_artifact_path(value, csv_path.parent, artifact_root, repo_root=repo_root)
                repaired = repo_relative_to(resolved, repo_root=repo_root)
                if repaired != value:
                    row[column] = repaired
                    changed += 1
                    row_changed = True
        if row_changed:
            write_rows(csv_path, rows, fieldnames)
            jsonl_path = csv_path.with_suffix(".jsonl")
            if jsonl_path.exists():
                with jsonl_path.open("w", encoding="utf-8") as fp:
                    for row in rows:
                        fp.write(json.dumps(row, sort_keys=True) + "\n")
    return changed


def audit_artifact_paths(artifact_root: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    unresolved: list[dict[str, str]] = []
    rows_checked = 0
    refs_checked = 0
    for csv_path in sorted(artifact_root.rglob("*.csv")):
        if csv_path.name in {"metadata_features.csv", "case_statistics.csv", "sample_statistics.csv", "source_validation_failures.csv", "hotspot_failures.csv"}:
            continue
        rows = read_rows(csv_path)
        for row in rows:
            if not any(column in row for column in PATH_COLUMNS):
                continue
            rows_checked += 1
            for column in PATH_COLUMNS:
                value = row.get(column, "")
                if not value:
                    continue
                refs_checked += 1
                resolved = resolve_artifact_path(value, csv_path.parent, artifact_root, repo_root=repo_root)
                if not resolved.exists():
                    unresolved.append(
                        {
                            "csv": repo_relative_to(csv_path, repo_root=repo_root),
                            "sample_uid": row.get("sample_uid", ""),
                            "column": column,
                            "value": value,
                            "resolved": str(resolved),
                        }
                    )
    report = {
        "schema_version": 1,
        "artifact_root": repo_relative_to(artifact_root, repo_root=repo_root),
        "rows_checked": rows_checked,
        "path_references_checked": refs_checked,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
    }
    (artifact_root / "path_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def resolve_index_path(path_value: str, dataset_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, dataset_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_artifact_path(value: str, csv_root: Path, artifact_root: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        repo_root / path,
        csv_root / path,
        artifact_root / "encoded_package_plus_power" / path,
        artifact_root / "package_plus_power_graph" / path,
        artifact_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def repo_relative(path: Path) -> str:
    return repo_relative_to(path, repo_root=REPO_ROOT)


def repo_relative_to(path: Path, *, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def preferred_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "sample_uid",
        "original_sample_uid",
        "case_id",
        "dataset_source",
        "split",
        "source_dir",
        "layout_path",
        "power_path",
        "package_path",
        "hotspot_path",
        "benchmark_path",
        "temp_layer0_path",
        "y_path",
    ]
    keys = sorted({key for row in rows for key in row})
    return [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
