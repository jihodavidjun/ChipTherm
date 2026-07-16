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

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training-ready encoded/metadata/graph artifacts for ChipTherm extension rows.")
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--index-name", default="all_extension_index.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    extension_root = args.extension_root.resolve()
    out_root = args.out_root.resolve()
    encoded_root = out_root / "encoded_package_plus_power"
    graph_root = out_root / "package_plus_power_graph"
    index = extension_root / args.index_name
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


def resolve_index_path(path_value: str, dataset_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, dataset_root / path, Path.cwd() / path]
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
