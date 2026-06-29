#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.encoder import CHANNEL_NAMES, encode_sample


INDEX_COLUMNS = [
    "sample_uid",
    "case_id",
    "x_path",
    "y_path",
    "original_temp_path",
    "H",
    "W",
    "C",
    "channel_names",
    "temp_min_K",
    "temp_max_K",
    "temp_mean_K",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode ChipTherm dataset samples into ML tensors.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    index_path = args.index.resolve()
    dataset_root = index_path.parent
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    power_values: list[float] = []
    y_mins: list[float] = []
    y_maxs: list[float] = []
    y_means: list[float] = []
    example_shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    first_record_stats: dict[str, Any] | None = None

    with index_path.open("r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            try:
                encoded = _encode_row(row, dataset_root, out_dir)
            except Exception as exc:
                failures.append({"sample_uid": row.get("sample_uid", ""), "reason": str(exc)})
                continue

            records.append(encoded["index_record"])
            power_channel = encoded["x"][0]
            y = encoded["y"]
            occupied_power = power_channel[power_channel > 0.0]
            if occupied_power.size:
                power_values.extend(float(value) for value in occupied_power)
            y_mins.append(float(y.min()))
            y_maxs.append(float(y.max()))
            y_means.append(float(y.mean()))
            if example_shape is None:
                example_shape = (encoded["x"].shape, y.shape)
                first_record_stats = {
                    "sample_uid": encoded["index_record"]["sample_uid"],
                    "channel_min": [float(encoded["x"][i].min()) for i in range(encoded["x"].shape[0])],
                    "channel_max": [float(encoded["x"][i].max()) for i in range(encoded["x"].shape[0])],
                    "channel_mean": [float(encoded["x"][i].mean()) for i in range(encoded["x"].shape[0])],
                    "y_min_K": float(y.min()),
                    "y_max_K": float(y.max()),
                    "y_mean_K": float(y.mean()),
                }

    encoded_index_csv = out_dir / "encoded_index.csv"
    encoded_index_jsonl = out_dir / "encoded_index.jsonl"
    metadata_path = out_dir / "encoding_metadata.json"
    _write_csv(encoded_index_csv, records)
    _write_jsonl(encoded_index_jsonl, records)
    metadata = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "source_index": str(index_path),
        "channel_names": CHANNEL_NAMES,
        "num_encoded": len(records),
        "num_failed": len(failures),
        "grid_shape": [64, 64],
        "dtype": "float32",
        "y_normalized": False,
        "output_layout": "case_grouped",
        "notes": "V1 encoder uses cell-center rasterization. Y is raw Layer 0 temperature in Kelvin.",
        "failures": failures,
        "first_sample_channel_statistics": first_record_stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Dataset encoding complete")
    print(f"Encoded samples: {len(records)}")
    print(f"Failed samples: {len(failures)}")
    print(f"Cases encoded: {len({record['case_id'] for record in records})}")
    if example_shape is not None:
        print(f"X shape: {example_shape[0]}")
        print(f"Y shape: {example_shape[1]}")
    if power_values:
        print(f"Power density min/max/mean: {min(power_values):.6g} / {max(power_values):.6g} / {sum(power_values) / len(power_values):.6g} W/mm^2")
    if y_mins and y_maxs and y_means:
        print(f"Y min/max/mean: {min(y_mins):.2f} / {max(y_maxs):.2f} / {sum(y_means) / len(y_means):.2f} K")
    print(f"Output: {out_dir}")
    return 0


def _encode_row(row: dict[str, str], dataset_root: Path, out_dir: Path) -> dict[str, Any]:
    sample_uid = row["sample_uid"]
    case_id = row["case_id"]
    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    x_path = case_dir / f"{sample_uid}_x.npy"
    y_path = case_dir / f"{sample_uid}_y.npy"

    x, y, metadata = encode_sample(
        layout_path=dataset_root / row["layout_path"],
        power_path=dataset_root / row["power_path"],
        hotspot_path=dataset_root / row["hotspot_path"],
        temp_path=dataset_root / row["temp_layer0_path"],
    )
    np.save(x_path, x)
    np.save(y_path, y)

    index_record = {
        "sample_uid": sample_uid,
        "case_id": case_id,
        "x_path": _rel(out_dir, x_path),
        "y_path": _rel(out_dir, y_path),
        "original_temp_path": row["temp_layer0_path"],
        "H": y.shape[0],
        "W": y.shape[1],
        "C": x.shape[0],
        "channel_names": ",".join(CHANNEL_NAMES),
        "temp_min_K": float(y.min()),
        "temp_max_K": float(y.max()),
        "temp_mean_K": float(y.mean()),
        "encoding": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rasterization": "cell_center",
            "channel_names": CHANNEL_NAMES,
            "metadata": metadata,
        },
        "source_index_row": row,
    }
    return {"index_record": index_record, "x": x, "y": y}


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in INDEX_COLUMNS})


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def _rel(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


if __name__ == "__main__":
    raise SystemExit(main())
