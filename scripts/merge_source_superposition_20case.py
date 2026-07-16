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
EXPECTED_PROTOCOL_COUNTS = {
    "sample_split_20case": {"train": 6400, "val": 800, "test": 810},
    "family_split_20case": {"train": 5600, "val": 800, "test": 800},
}
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
    "original_temp_path",
    "temp_layer0_path",
    "source_superposition_base_path",
    "source_superposition_residual_path",
    "source_layout_path",
    "source_power_path",
    "source_package_path",
    "source_hotspot_path",
)
CANONICAL_GENERAL_COLUMNS = (
    "original_sample_uid",
    "dataset_source",
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
    "original_temp_path",
    "temp_layer0_path",
    "hotspot_runtime_s",
    "physics_runtime_s",
    "num_chiplets",
    "total_power_W",
    "mean_temperature_K",
    "max_temperature_K",
    "temp_min_K",
    "temp_mean_K",
    "temp_max_K",
    "C",
    "H",
    "W",
    "channel_names",
)
SOURCE_BASE_COLUMNS = (
    "source_superposition_base_path",
    "source_superposition_residual_path",
    "source_base_mode",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_checkpoint_config_sha256",
    "source_checkpoint_epoch",
    "source_checkpoint_best_metric",
    "source_count",
    "source_model_version",
    "source_base_units",
    "source_base_shape",
    "source_base_dtype",
    "source_generation_runtime_s",
    "source_superposition_runtime_s",
    "generation_status",
    "source_layout_path",
    "source_power_path",
    "source_package_path",
    "source_hotspot_path",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge original and extension source-superposition rows into 20-case protocols.")
    parser.add_argument("--original-source-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full", type=Path)
    parser.add_argument(
        "--original-canonical-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power",
        type=Path,
        help="Authoritative retained case01-case10 graph/context dataset used to repair stale original source-base paths.",
    )
    parser.add_argument("--extension-source-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_extension", type=Path)
    parser.add_argument("--split-root", default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/indices", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_20case", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    original_root = args.original_source_root.expanduser().resolve()
    original_canonical_root = args.original_canonical_root.expanduser().resolve()
    extension_root = args.extension_source_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    original_rows_raw = read_all_source_rows(original_root)
    original_canonical_rows = read_all_source_rows(original_canonical_root)
    original_rows, canonical_report = repair_original_rows_from_canonical(original_rows_raw, original_canonical_rows)
    extension_rows = read_all_source_rows(extension_root)
    by_uid = build_unique_uid_map([*original_rows, *extension_rows])
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
            test_uids = [row["sample_uid"] for row in read_rows(original_test)]
            write_rows(out_root / "original_case01_case10_test_index.csv", [dict(by_uid[uid], split="test") for uid in test_uids if uid in by_uid])
    report = validate_outputs(out_root if out_root.exists() else split_root, by_uid)
    report.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_source_root": repo_relative(original_root),
            "original_canonical_root": repo_relative(original_canonical_root),
            "extension_source_root": repo_relative(extension_root),
            "split_root": repo_relative(split_root),
            "out_root": repo_relative(out_root),
            "canonical_repair": canonical_report,
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


def repair_original_rows_from_canonical(
    source_rows: list[dict[str, str]], canonical_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Overlay case01-case10 source-superposition rows with retained canonical dataset paths.

    The original source-base rows carry source-superposition-specific artifacts,
    but some general dataset paths may point at deleted upstream generation
    trees.  The retained clean graph dataset is the authoritative source for
    the regular x/y/graph/physics/path columns.
    """
    canonical_by_uid = index_rows_by_uid(canonical_rows, label="original canonical dataset")
    missing: list[str] = []
    repaired: list[dict[str, str]] = []
    changed_columns: Counter[str] = Counter()
    for row in source_rows:
        uid = row.get("sample_uid", "")
        canonical = canonical_by_uid.get(uid)
        if canonical is None:
            missing.append(uid)
            continue
        out = dict(row)
        for column in CANONICAL_GENERAL_COLUMNS:
            if column in canonical:
                old_value = out.get(column, "")
                out[column] = canonical[column]
                if old_value != out[column]:
                    changed_columns[column] += 1
        for column in SOURCE_BASE_COLUMNS:
            if column in row:
                out[column] = row[column]
        out["source_base_mode"] = out.get("source_base_mode") or "source_superposition_v1"
        repaired.append(out)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise SystemExit(
            f"{len(missing)} original source-base UID(s) could not be matched to exactly one canonical row: {preview}"
        )
    return repaired, {
        "original_source_rows": len(source_rows),
        "canonical_rows": len(canonical_rows),
        "repaired_rows": len(repaired),
        "changed_columns": dict(sorted(changed_columns.items())),
    }


def index_rows_by_uid(rows: list[dict[str, str]], *, label: str) -> dict[str, dict[str, str]]:
    counts = Counter(row.get("sample_uid", "") for row in rows)
    duplicates = sorted(uid for uid, count in counts.items() if uid and count != 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise SystemExit(f"{label}: duplicate sample_uid entries; expected exactly one row per UID: {preview}")
    missing_uid_count = counts.get("", 0)
    if missing_uid_count:
        raise SystemExit(f"{label}: {missing_uid_count} row(s) are missing sample_uid")
    return {row["sample_uid"]: row for row in rows}


def build_unique_uid_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    counts = Counter(row.get("sample_uid", "") for row in rows)
    duplicates = sorted(uid for uid, count in counts.items() if uid and count != 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise SystemExit(f"merged source-base rows contain duplicate sample_uid entries: {preview}")
    return {row["sample_uid"]: normalize_row_paths(row) for row in rows}


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
        expected_counts = EXPECTED_PROTOCOL_COUNTS.get(protocol)
        if expected_counts and split_counts != expected_counts:
            errors.append(f"{protocol}: expected split counts {expected_counts}, found {split_counts}")
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
    wanted = {"case01", "case10", "case11", "case17", "case19", "case20"}
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
