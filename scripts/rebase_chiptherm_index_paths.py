#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PATH_COLUMNS = [
    "source_dir",
    "scenario_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "benchmark_path",
    "x_path",
    "y_path",
    "prediction_path",
    "residual_path",
    "graph_path",
    "source_superposition_base_path",
    "source_superposition_residual_path",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebase ChipTherm CSV path columns to portable repo-relative paths.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--old-prefix", default=None)
    parser.add_argument("--new-prefix", default="", help="New prefix. Empty means repo-relative if the path contains data/runs/...")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    index = args.index.resolve()
    rows = read_rows(index)
    changed = 0
    for row in rows:
        for column in PATH_COLUMNS:
            value = row.get(column, "")
            if not value:
                continue
            new_value = rebase_value(value, old_prefix=args.old_prefix, new_prefix=args.new_prefix)
            if new_value != value:
                row[column] = new_value
                changed += 1
    print(f"{index}: would update {changed} path value(s)")
    if args.validate:
        missing = []
        root = index.parent
        for row in rows:
            for column in PATH_COLUMNS:
                value = row.get(column, "")
                if not value:
                    continue
                path = resolve_value(value, root)
                if not path.exists():
                    missing.append(f"{row.get('sample_uid','?')} {column}: {value}")
        if missing:
            for item in missing[:20]:
                print(f"missing: {item}")
            raise SystemExit(f"{len(missing)} path(s) do not resolve")
    if args.apply:
        if args.backup:
            backup = index.with_suffix(index.suffix + ".bak")
            shutil.copyfile(index, backup)
            print(f"backup: {backup}")
        write_rows(index, rows)
        print("updated")
    return 0


def rebase_value(value: str, *, old_prefix: str | None, new_prefix: str) -> str:
    text = value
    if old_prefix and text.startswith(old_prefix):
        text = new_prefix + text[len(old_prefix) :]
    if text.startswith("/"):
        marker = "/data/runs/"
        if marker in text:
            text = "data/runs/" + text.split(marker, 1)[1]
        marker = "/outputs/"
        if marker in text:
            text = "outputs/" + text.split(marker, 1)[1]
    return text.lstrip("/") if not new_prefix else text


def resolve_value(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (REPO_ROOT / path, root / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
