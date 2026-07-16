#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402


PROTOCOLS = (
    "sample_split_extension",
    "family_split_extension",
    "sample_split_20case",
    "family_split_20case",
)
PATH_COLUMNS = (
    "x_path",
    "y_path",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge original and extension source-superposition rows into 20-case protocols.")
    parser.add_argument("--original-source-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full", type=Path)
    parser.add_argument("--extension-source-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_extension", type=Path)
    parser.add_argument("--split-root", default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/indices", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_20case", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    original_root = args.original_source_root.expanduser().resolve()
    extension_root = args.extension_source_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    original_rows = read_all_source_rows(original_root)
    extension_rows = read_all_source_rows(extension_root)
    by_uid = {row["sample_uid"]: normalize_row_paths(row) for row in [*original_rows, *extension_rows]}
    if not args.validate_only:
        out_root.mkdir(parents=True, exist_ok=True)
        for protocol in PROTOCOLS:
            source_protocol = split_root / protocol
            if not source_protocol.exists():
                raise SystemExit(f"missing split protocol: {source_protocol}")
            all_name = "all_index.csv" if protocol.startswith("family") else "combined_index.csv"
            write_protocol(source_protocol, out_root / protocol, by_uid, all_name=all_name)
        original_test = original_root / "test_index.csv"
        if original_test.exists():
            write_rows(out_root / "original_case01_case10_test_index.csv", [normalize_row_paths(row) for row in read_rows(original_test)])
    report = validate_outputs(out_root if out_root.exists() else split_root, by_uid)
    report.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_source_root": repo_relative(original_root),
            "extension_source_root": repo_relative(extension_root),
            "split_root": repo_relative(split_root),
            "out_root": repo_relative(out_root),
        }
    )
    if not args.validate_only:
        (out_root / "merge_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_root / "split_manifest.json").write_text(json.dumps(report["protocols"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_root / "compatibility_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_report_md(out_root / "compatibility_report.md", report)
    print("Source-superposition 20-case merge complete")
    print(f"Output: {out_root}")
    for protocol, item in sorted(report["protocols"].items()):
        counts = item.get("split_counts", {})
        print(f"{protocol}: train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}")
    print(f"Unresolved paths: {report['unresolved_path_count']}")
    if report["unresolved_path_count"] or report["errors"]:
        for error in report["errors"][:20]:
            print(f"  - {error}")
        return 2
    return 0


def write_protocol(source_protocol: Path, out_protocol: Path, by_uid: dict[str, dict[str, str]], *, all_name: str) -> None:
    out_protocol.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_rows = []
        for row in read_rows(source_protocol / f"{split}_index.csv"):
            uid = row["sample_uid"]
            if uid not in by_uid:
                raise SystemExit(f"{source_protocol}: sample_uid {uid} missing from source-base rows")
            split_rows.append(dict(by_uid[uid], split=split))
        write_rows(out_protocol / f"{split}_index.csv", split_rows)
        combined.extend(split_rows)
    write_rows(out_protocol / all_name, sorted(combined, key=lambda row: (row["case_id"], row["sample_uid"])))


def validate_outputs(root: Path, by_uid: dict[str, dict[str, str]]) -> dict[str, Any]:
    protocols: dict[str, Any] = {}
    errors: list[str] = []
    unresolved: list[dict[str, str]] = []
    for protocol in PROTOCOLS:
        protocol_root = root / protocol
        if not protocol_root.exists():
            continue
        split_counts: dict[str, int] = {}
        split_cases: dict[str, list[str]] = {}
        seen_by_split: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            path = protocol_root / f"{split}_index.csv"
            rows = read_rows(path)
            split_counts[split] = len(rows)
            split_cases[split] = sorted({row["case_id"] for row in rows})
            seen_by_split[split] = {row["sample_uid"] for row in rows}
            if len(seen_by_split[split]) != len(rows):
                errors.append(f"{protocol}/{split}: duplicate sample_uid")
            for row in rows:
                for column in PATH_COLUMNS:
                    value = row.get(column, "")
                    if not value:
                        continue
                    path_value = resolve_path(value)
                    if not path_value.exists():
                        unresolved.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "value": value})
                if row.get("source_base_mode") != "source_superposition_v1":
                    errors.append(f"{protocol}/{split}/{row['sample_uid']}: source_base_mode is not source_superposition_v1")
                base_value = row.get("source_superposition_base_path", "")
                if base_value and resolve_path(base_value).exists():
                    arr = np.load(resolve_path(base_value), mmap_mode="r")
                    if tuple(arr.shape) != (64, 64):
                        errors.append(f"{protocol}/{split}/{row['sample_uid']}: source base shape {arr.shape}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = seen_by_split[left] & seen_by_split[right]
            if overlap:
                errors.append(f"{protocol}: {left}/{right} overlap {sorted(overlap)[:5]}")
        protocols[protocol] = {
            "split_counts": split_counts,
            "split_cases": split_cases,
            "case_counts": dict(Counter(row["case_id"] for split in ("train", "val", "test") for row in read_rows(protocol_root / f"{split}_index.csv"))),
        }
    smoke_errors = loader_smoke(root)
    errors.extend(smoke_errors)
    return {
        "protocols": protocols,
        "source_row_count": len(by_uid),
        "unresolved_path_count": len(unresolved),
        "unresolved_paths": unresolved[:100],
        "errors": errors,
    }


def loader_smoke(root: Path) -> list[str]:
    errors: list[str] = []
    candidates = [
        root / "sample_split_20case" / "train_index.csv",
        root / "family_split_20case" / "val_index.csv",
        root / "family_split_20case" / "test_index.csv",
    ]
    wanted = {"case01", "case11", "case17", "case19", "case20"}
    checked = set()
    for path in candidates:
        if not path.exists():
            continue
        rows = read_rows(path)
        for case_id in sorted(wanted - checked):
            match = next((row for row in rows if row["case_id"] == case_id), None)
            if match is None:
                continue
            tmp_path = path.parent / f".loader_smoke_{case_id}.csv"
            write_rows(tmp_path, [match])
            try:
                sample = ChipThermDataset(tmp_path, target="residual", return_metadata=True, return_graph=True)[0]
                if tuple(sample["x"].shape[-2:]) != (64, 64):
                    errors.append(f"{case_id}: x shape {tuple(sample['x'].shape)}")
                if tuple(sample["physics"].shape) != (64, 64):
                    errors.append(f"{case_id}: source base shape {tuple(sample['physics'].shape)}")
                if "graph" not in sample:
                    errors.append(f"{case_id}: graph missing")
            except Exception as exc:
                errors.append(f"{case_id}: loader smoke failed: {exc}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            checked.add(case_id)
    missing = wanted - checked
    if missing:
        errors.append(f"loader smoke did not find cases: {sorted(missing)}")
    return errors


def read_all_source_rows(root: Path) -> list[dict[str, str]]:
    path = root / "combined_encoded_index.csv"
    if path.exists():
        return read_rows(path)
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        rows.extend(read_rows(root / f"{split}_index.csv"))
    return rows


def normalize_row_paths(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    for column in PATH_COLUMNS:
        value = out.get(column, "")
        if value:
            out[column] = repo_relative(resolve_path(value))
    return out


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Source-Superposition 20-Case Compatibility", ""]
    lines.append(f"Unresolved paths: {report['unresolved_path_count']}")
    lines.extend(["", "| Protocol | Train | Val | Test | Train cases | Val cases | Test cases |", "|---|---:|---:|---:|---|---|---|"])
    for protocol, item in sorted(report["protocols"].items()):
        counts = item["split_counts"]
        cases = item["split_cases"]
        lines.append(
            f"| {protocol} | {counts.get('train', 0)} | {counts.get('val', 0)} | {counts.get('test', 0)} | "
            f"{','.join(cases.get('train', []))} | {','.join(cases.get('val', []))} | {','.join(cases.get('test', []))} |"
        )
    lines.extend(["", "## Errors", ""])
    lines += [f"- {error}" for error in report["errors"]] or ["- none"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
