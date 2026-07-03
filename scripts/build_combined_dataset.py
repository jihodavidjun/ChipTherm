#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


CSV_COLUMNS = [
    "sample_uid",
    "original_sample_uid",
    "case_id",
    "dataset_source",
    "split",
    "x_path",
    "y_path",
    "prediction_path",
    "residual_path",
    "hotspot_runtime_s",
    "physics_runtime_s",
    "num_chiplets",
    "total_power_W",
    "mean_temperature_K",
    "max_temperature_K",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a logical combined ChipTherm dataset index.")
    parser.add_argument("--dataset-a", required=True, type=Path)
    parser.add_argument("--dataset-b", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    args = parser.parse_args()

    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_roots = [args.dataset_a.resolve(), args.dataset_b.resolve()]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for dataset_root in dataset_roots:
        records.extend(_load_dataset_records(dataset_root))

    _assign_case_stratified_splits(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    _validate_combined_records(records)

    records.sort(key=lambda row: (row["case_id"], row["dataset_source"], row["original_sample_uid"]))
    _write_csv(out_dir / "combined_encoded_index.csv", records)
    _write_jsonl(out_dir / "combined_encoded_index.jsonl", records)
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        _write_csv(out_dir / f"{split}_index.csv", split_records)

    manifest = _build_manifest(
        records,
        dataset_roots=dataset_roots,
        out_dir=out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(out_dir / "README.md", dataset_roots)

    print("Combined dataset build complete")
    print(f"Total samples: {manifest['total_samples']}")
    print(f"Train/val/test: {manifest['split_counts']['train']} / {manifest['split_counts']['val']} / {manifest['split_counts']['test']}")
    print(f"Dataset sources: {manifest['dataset_source_counts']}")
    print(f"Cases: {', '.join(sorted(manifest['samples_per_benchmark_case']))}")
    print(f"Output: {out_dir}")
    return 0


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1.0e-9:
        raise SystemExit(f"split ratios must sum to 1.0, got {total}")
    if min(train_ratio, val_ratio, test_ratio) < 0.0:
        raise SystemExit("split ratios must be non-negative")


def _load_dataset_records(dataset_root: Path) -> list[dict[str, Any]]:
    source = dataset_root.name
    encoded_csv = dataset_root / "encoded" / "encoded_index.csv"
    encoded_jsonl = dataset_root / "encoded" / "encoded_index.jsonl"
    baseline_dir = dataset_root / "physics_baseline_global003_residuals"
    metrics_path = baseline_dir / "metrics.json"

    required = [encoded_csv, encoded_jsonl, baseline_dir, metrics_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"{dataset_root} is missing required files: {', '.join(str(path) for path in missing)}")

    jsonl_records = _read_jsonl_by_uid(encoded_jsonl)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    physics_runtime_s = _optional_float(metrics.get("baseline_runtime_per_sample_s"))

    records: list[dict[str, Any]] = []
    with encoded_csv.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            original_uid = row["sample_uid"]
            if original_uid not in jsonl_records:
                raise SystemExit(f"{encoded_jsonl} is missing metadata for {original_uid}")
            rich = jsonl_records[original_uid]
            source_row = rich.get("source_index_row", {})
            sample_uid = f"{source}_{original_uid}"
            case_id = row["case_id"]

            prediction_path = baseline_dir / "predictions" / case_id / f"{original_uid}_tphys.npy"
            residual_path = baseline_dir / "optional_residuals" / case_id / f"{original_uid}_residual.npy"
            x_path = dataset_root / "encoded" / row["x_path"]
            y_path = dataset_root / "encoded" / row["y_path"]

            record = {
                "sample_uid": sample_uid,
                "original_sample_uid": original_uid,
                "case_id": case_id,
                "dataset_source": source,
                "split": "",
                "x_path": _repo_relative(x_path),
                "y_path": _repo_relative(y_path),
                "prediction_path": _repo_relative(prediction_path),
                "residual_path": _repo_relative(residual_path),
                "hotspot_runtime_s": _optional_float(source_row.get("hotspot_runtime_s")),
                "physics_runtime_s": physics_runtime_s,
                "num_chiplets": _optional_int(source_row.get("num_chiplets")),
                "total_power_W": _total_power_from_record(rich),
                "mean_temperature_K": _optional_float(row.get("temp_mean_K")),
                "max_temperature_K": _optional_float(row.get("temp_max_K")),
                "jsonl_record": rich,
            }
            records.append(record)
    return records


def _read_jsonl_by_uid(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_uid = str(record.get("sample_uid", ""))
            if not sample_uid:
                raise SystemExit(f"{path}:{line_number} is missing sample_uid")
            if sample_uid in records:
                raise SystemExit(f"{path} contains duplicate sample_uid {sample_uid}")
            records[sample_uid] = record
    return records


def _assign_case_stratified_splits(records: list[dict[str, Any]], *, seed: int, train_ratio: float, val_ratio: float) -> None:
    rng = random.Random(seed)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)

    for case_id in sorted(by_case):
        case_records = by_case[case_id]
        rng.shuffle(case_records)
        n = len(case_records)
        train_count = int(n * train_ratio)
        val_count = int(n * val_ratio)
        for index, record in enumerate(case_records):
            if index < train_count:
                record["split"] = "train"
            elif index < train_count + val_count:
                record["split"] = "val"
            else:
                record["split"] = "test"


def _validate_combined_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise SystemExit("no samples found")

    seen: set[str] = set()
    errors: list[str] = []
    for record in records:
        sample_uid = record["sample_uid"]
        if sample_uid in seen:
            errors.append(f"duplicate sample_uid {sample_uid}")
        seen.add(sample_uid)

        if record["split"] not in {"train", "val", "test"}:
            errors.append(f"{sample_uid} has invalid split {record['split']!r}")

        for key in ("x_path", "y_path", "prediction_path", "residual_path"):
            path = REPO_ROOT / record[key]
            if not path.exists():
                errors.append(f"{sample_uid} missing {key}: {record[key]}")

        original_uid = record["original_sample_uid"]
        if not record["prediction_path"].endswith(f"{original_uid}_tphys.npy"):
            errors.append(f"{sample_uid} prediction does not correspond to original sample UID")
        if not record["residual_path"].endswith(f"{original_uid}_residual.npy"):
            errors.append(f"{sample_uid} residual does not correspond to original sample UID")

    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... {len(errors) - 20} more errors" if len(errors) > 20 else ""
        raise SystemExit(f"combined dataset consistency checks failed:\n{preview}{suffix}")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in CSV_COLUMNS})


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            payload = {column: record[column] for column in CSV_COLUMNS}
            payload["source_encoded_record"] = record["jsonl_record"]
            fp.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_manifest(
    records: list[dict[str, Any]],
    *,
    dataset_roots: list[Path],
    out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    case_counts = Counter(record["case_id"] for record in records)
    source_counts = Counter(record["dataset_source"] for record in records)
    split_case_counts: dict[str, dict[str, int]] = {}
    split_source_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record["split"] == split]
        split_case_counts[split] = dict(sorted(Counter(record["case_id"] for record in split_records).items()))
        split_source_counts[split] = dict(sorted(Counter(record["dataset_source"] for record in split_records).items()))

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_v1_root": _repo_relative(out_dir),
        "source_datasets": [_repo_relative(path) for path in dataset_roots],
        "total_samples": len(records),
        "split_seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "split_counts": {split: split_counts.get(split, 0) for split in ("train", "val", "test")},
        "samples_per_benchmark_case": dict(sorted(case_counts.items())),
        "samples_per_split_case": split_case_counts,
        "dataset_source_counts": dict(sorted(source_counts.items())),
        "dataset_source_counts_by_split": split_source_counts,
        "mean_total_power_W": _mean(record["total_power_W"] for record in records),
        "mean_hotspot_temperature_K": _mean(record["mean_temperature_K"] for record in records),
        "mean_chiplet_count": _mean(record["num_chiplets"] for record in records),
        "files": {
            "combined_encoded_index_csv": "combined_encoded_index.csv",
            "combined_encoded_index_jsonl": "combined_encoded_index.jsonl",
            "train_index_csv": "train_index.csv",
            "val_index_csv": "val_index.csv",
            "test_index_csv": "test_index.csv",
        },
    }


def _write_readme(path: Path, dataset_roots: list[Path]) -> None:
    source_lines = "\n".join(f"- `{_repo_relative(root)}`" for root in dataset_roots)
    text = f"""# ChipTherm Dataset v1

`dataset_v1` is a logical dataset index for ChipTherm benchmark training.

It does not copy or move tensor files. The original encoded tensors, HotSpot
targets, physics-baseline predictions, and residuals remain in:

{source_lines}

This directory contains:

- `combined_encoded_index.csv`
- `combined_encoded_index.jsonl`
- `train_index.csv`
- `val_index.csv`
- `test_index.csv`
- `split_manifest.json`

Future training code, including a `ChipThermDataset` class, should read these
index files and follow their paths to the original data artifacts.
"""
    path.write_text(text, encoding="utf-8")


def _total_power_from_record(record: dict[str, Any]) -> float:
    chiplets = record.get("encoding", {}).get("metadata", {}).get("chiplets", [])
    if not chiplets:
        raise SystemExit(f"{record.get('sample_uid')} is missing encoding.metadata.chiplets for total_power_W")
    return float(sum(float(chiplet["power_W"]) for chiplet in chiplets))


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


if __name__ == "__main__":
    raise SystemExit(main())
