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


PROTOCOLS = ("sample_split_extension", "family_split_extension")
SPLITS = ("train", "val", "test")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build extension-only split-protocol indices over generated source-superposition rows."
    )
    parser.add_argument(
        "--extension-source-root",
        default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_extension",
        type=Path,
    )
    parser.add_argument(
        "--split-root",
        default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/indices",
        type=Path,
    )
    parser.add_argument(
        "--out-root",
        default=None,
        type=Path,
        help="Directory receiving protocol subdirectories; defaults to --extension-source-root.",
    )
    args = parser.parse_args()

    extension_root = args.extension_source_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    out_root = (args.out_root.expanduser().resolve() if args.out_root else extension_root)

    rows = [normalize_row_paths(row) for row in read_source_rows(extension_root)]
    by_uid = {row["sample_uid"]: row for row in rows}
    if len(by_uid) != len(rows):
        duplicates = [uid for uid, count in Counter(row["sample_uid"] for row in rows).items() if count > 1]
        raise SystemExit(f"duplicate source-base sample_uids: {duplicates[:10]}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extension_source_root": repo_relative(extension_root),
        "split_root": repo_relative(split_root),
        "out_root": repo_relative(out_root),
        "protocols": {},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    for protocol in PROTOCOLS:
        source_protocol = split_root / protocol
        if not source_protocol.exists():
            raise SystemExit(f"missing split protocol: {source_protocol}")
        all_name = "all_index.csv" if protocol.startswith("family") else "combined_index.csv"
        manifest["protocols"][protocol] = write_protocol(source_protocol, out_root / protocol, by_uid, all_name=all_name)

    validation = validate_protocols(out_root)
    manifest.update(validation)
    (out_root / "extension_split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Extension source-superposition split indices written")
    print(f"Output: {out_root}")
    for protocol, item in sorted(manifest["protocols"].items()):
        counts = item["split_counts"]
        print(f"{protocol}: train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}")
    print(f"Unresolved paths: {manifest['unresolved_path_count']}")
    if manifest["errors"] or manifest["unresolved_path_count"]:
        for error in manifest["errors"][:20]:
            print(f"  - {error}")
        return 2
    return 0


def write_protocol(source_protocol: Path, out_protocol: Path, by_uid: dict[str, dict[str, str]], *, all_name: str) -> dict[str, Any]:
    out_protocol.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}
    split_cases: dict[str, list[str]] = {}
    missing: list[str] = []
    for split in SPLITS:
        source_rows = read_rows(source_protocol / f"{split}_index.csv")
        split_rows: list[dict[str, str]] = []
        for row in source_rows:
            uid = row["sample_uid"]
            if uid not in by_uid:
                missing.append(uid)
                continue
            split_rows.append(dict(by_uid[uid], split=split))
        if missing:
            raise SystemExit(f"{source_protocol}: {len(missing)} sample_uids missing from generated source-base rows; first={missing[:5]}")
        write_rows(out_protocol / f"{split}_index.csv", split_rows)
        split_counts[split] = len(split_rows)
        split_cases[split] = sorted({row["case_id"] for row in split_rows})
        combined.extend(split_rows)
    write_rows(out_protocol / all_name, sorted(combined, key=lambda row: (row["case_id"], row["sample_uid"])))
    return {
        "source_protocol": repo_relative(source_protocol),
        "output_protocol": repo_relative(out_protocol),
        "combined_name": all_name,
        "split_counts": split_counts,
        "split_cases": split_cases,
        "case_counts": dict(sorted(Counter(row["case_id"] for row in combined).items())),
    }


def validate_protocols(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    unresolved: list[dict[str, str]] = []
    for protocol in PROTOCOLS:
        protocol_root = root / protocol
        seen_by_split: dict[str, set[str]] = {}
        for split in SPLITS:
            rows = read_rows(protocol_root / f"{split}_index.csv")
            seen_by_split[split] = {row["sample_uid"] for row in rows}
            if len(seen_by_split[split]) != len(rows):
                errors.append(f"{protocol}/{split}: duplicate sample_uid")
            for row in rows:
                if row.get("source_base_mode") != "source_superposition_v1":
                    errors.append(f"{protocol}/{split}/{row['sample_uid']}: source_base_mode={row.get('source_base_mode')}")
                for column in PATH_COLUMNS:
                    value = row.get(column, "")
                    if value and not resolve_path(value).exists():
                        unresolved.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "value": value})
                base_path = resolve_path(row.get("source_superposition_base_path", ""))
                if base_path.exists():
                    arr = np.load(base_path, mmap_mode="r")
                    if tuple(arr.shape) != (64, 64):
                        errors.append(f"{protocol}/{split}/{row['sample_uid']}: base shape={arr.shape}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = seen_by_split[left] & seen_by_split[right]
            if overlap:
                errors.append(f"{protocol}: {left}/{right} overlap {sorted(overlap)[:5]}")
    errors.extend(loader_smoke(root))
    return {
        "unresolved_path_count": len(unresolved),
        "unresolved_paths": unresolved[:100],
        "errors": errors,
    }


def loader_smoke(root: Path) -> list[str]:
    errors: list[str] = []
    wanted = {"case11", "case17", "case19", "case20"}
    checked: set[str] = set()
    candidates = [
        root / "sample_split_extension" / "train_index.csv",
        root / "family_split_extension" / "val_index.csv",
        root / "family_split_extension" / "test_index.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        rows = read_rows(path)
        for case_id in sorted(wanted - checked):
            row = next((item for item in rows if item["case_id"] == case_id), None)
            if row is None:
                continue
            tmp = path.parent / f".loader_smoke_{case_id}.csv"
            write_rows(tmp, [row])
            try:
                sample = ChipThermDataset(tmp, target="residual", return_metadata=True, return_graph=True)[0]
                if tuple(sample["x"].shape[-2:]) != (64, 64):
                    errors.append(f"{case_id}: x shape={tuple(sample['x'].shape)}")
                if tuple(sample["physics"].shape) != (64, 64):
                    errors.append(f"{case_id}: source base shape={tuple(sample['physics'].shape)}")
                if "metadata" not in sample:
                    errors.append(f"{case_id}: metadata missing")
                if "graph" not in sample:
                    errors.append(f"{case_id}: graph missing")
            except Exception as exc:
                errors.append(f"{case_id}: loader smoke failed: {exc}")
            finally:
                if tmp.exists():
                    tmp.unlink()
            checked.add(case_id)
    missing = wanted - checked
    if missing:
        errors.append(f"loader smoke did not find cases: {sorted(missing)}")
    return errors


def read_source_rows(root: Path) -> list[dict[str, str]]:
    combined = root / "combined_encoded_index.csv"
    if combined.exists():
        return read_rows(combined)
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        rows.extend(read_rows(root / f"{split}_index.csv"))
    return rows


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


def normalize_row_paths(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    out.setdefault("prediction_path", "")
    out.setdefault("residual_path", "")
    for column in PATH_COLUMNS:
        value = out.get(column, "")
        if value:
            out[column] = repo_relative(resolve_path(value))
    return out


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
