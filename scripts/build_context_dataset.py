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


CONTEXT_CHANNEL_NAMES = [
    "total_power_W",
    "package_width_mm",
    "package_height_mm",
    "cell_size_x_mm",
    "cell_size_y_mm",
    "occupied_area_fraction",
    "total_power_per_package_area_W_per_mm2",
    "total_power_per_occupied_area_W_per_mm2",
]


CONTEXT_SETS = {
    "total_power_only": ["total_power_W"],
    "package_geometry": ["package_width_mm", "package_height_mm", "cell_size_x_mm", "cell_size_y_mm"],
    "occupancy_summary": ["occupied_area_fraction"],
    "power_density_summary": [
        "total_power_per_package_area_W_per_mm2",
        "total_power_per_occupied_area_W_per_mm2",
    ],
    "package_plus_power": [
        "total_power_W",
        "package_width_mm",
        "package_height_mm",
        "cell_size_x_mm",
        "cell_size_y_mm",
    ],
    "all_context": CONTEXT_CHANNEL_NAMES,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build context-augmented ChipTherm dataset indices and X tensors.")
    parser.add_argument("--base-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v1", type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--context-set", default="all_context", choices=sorted(CONTEXT_SETS))
    args = parser.parse_args()

    base_root = args.base_root.resolve()
    out_dir = (args.out_dir or default_out_dir(args.context_set)).resolve()
    build_context_dataset(base_root=base_root, out_dir=out_dir, context_set=args.context_set)
    return 0


def build_context_dataset(*, base_root: Path, out_dir: Path, context_set: str) -> dict[str, Any]:
    selected_context_names = CONTEXT_SETS[context_set]
    x_root = out_dir / "encoded_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    x_root.mkdir(parents=True, exist_ok=True)

    metadata_by_uid = read_combined_jsonl(base_root / "combined_encoded_index.jsonl")
    split_records: dict[str, list[dict[str, str]]] = {}
    all_records: list[dict[str, str]] = []
    first_stats: dict[str, Any] | None = None

    for split in ("train", "val", "test"):
        records: list[dict[str, str]] = []
        with (base_root / f"{split}_index.csv").open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                rich = metadata_by_uid[row["sample_uid"]]
                record, stats = build_context_sample(row, rich, x_root, selected_context_names)
                records.append(record)
                all_records.append(record)
                if first_stats is None:
                    first_stats = stats
        split_records[split] = records
        write_csv(out_dir / f"{split}_index.csv", records)

    write_csv(out_dir / "combined_encoded_index.csv", all_records)
    write_jsonl(out_dir / "combined_encoded_index.jsonl", all_records)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_root": repo_relative(base_root),
        "out_dir": repo_relative(out_dir),
        "num_samples": len(all_records),
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "original_channels": 8,
        "context_set": context_set,
        "context_channels": selected_context_names,
        "context_channel_indices": {
            name: 8 + index for index, name in enumerate(selected_context_names)
        },
        "output_channels": 8 + len(selected_context_names),
        "notes": "Only X tensors are regenerated. Y, physics predictions, and residual files are reused from dataset_v1.",
        "first_sample": first_stats,
    }
    (out_dir / "context_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, base_root, context_set, selected_context_names)

    print("Context dataset build complete")
    print(f"Samples: {len(all_records)}")
    print(f"Train/val/test: {len(split_records['train'])} / {len(split_records['val'])} / {len(split_records['test'])}")
    if first_stats:
        print(f"New X shape: {tuple(first_stats['x_shape'])}")
    print(f"Output: {out_dir}")
    return manifest


def read_combined_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["sample_uid"])] = record
    return records


def build_context_sample(
    row: dict[str, str],
    rich: dict[str, Any],
    x_root: Path,
    selected_context_names: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    source_x_path = REPO_ROOT / row["x_path"]
    x = np.load(source_x_path).astype(np.float32, copy=False)
    if x.shape != (8, 64, 64):
        raise ValueError(f"{source_x_path} expected shape (8, 64, 64), got {x.shape}")

    source_meta = rich["source_encoded_record"]["encoding"]["metadata"]
    width_mm = float(source_meta["package_width_mm"])
    height_mm = float(source_meta["package_height_mm"])
    rows = int(source_meta.get("grid_rows", x.shape[1]))
    cols = int(source_meta.get("grid_cols", x.shape[2]))
    total_power_W = float(row["total_power_W"])
    package_area_mm2 = width_mm * height_mm
    occupied_area_fraction = float(np.mean(x[1] > 0.5))
    occupied_area_mm2 = max(occupied_area_fraction * package_area_mm2, 1.0e-8)
    all_context_values = {
        "total_power_W": total_power_W,
        "package_width_mm": width_mm,
        "package_height_mm": height_mm,
        "cell_size_x_mm": width_mm / cols,
        "cell_size_y_mm": height_mm / rows,
        "occupied_area_fraction": occupied_area_fraction,
        "total_power_per_package_area_W_per_mm2": total_power_W / package_area_mm2,
        "total_power_per_occupied_area_W_per_mm2": total_power_W / occupied_area_mm2,
    }
    context_values = [all_context_values[name] for name in selected_context_names]
    context = np.stack(
        [np.full((x.shape[1], x.shape[2]), value, dtype=np.float32) for value in context_values],
        axis=0,
    ) if context_values else np.zeros((0, x.shape[1], x.shape[2]), dtype=np.float32)
    x_context = np.concatenate([x, context], axis=0).astype(np.float32, copy=False)

    case_dir = x_root / row["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    x_path = case_dir / f"{row['sample_uid']}_x_context.npy"
    np.save(x_path, x_context)

    record = dict(row)
    record["x_path"] = repo_relative(x_path)
    stats = {
        "sample_uid": row["sample_uid"],
        "x_shape": list(x_context.shape),
        "context_values": {name: all_context_values[name] for name in selected_context_names},
    }
    return record, stats


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    if not records:
        raise ValueError(f"no records to write for {path}")
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def write_readme(out_dir: Path, base_root: Path, context_set: str, selected_context_names: list[str]) -> None:
    channel_lines = [
        "0. power_density_W_per_mm2",
        "1. occupancy_mask",
        "2. CPU_mask",
        "3. GPU_or_NPU_mask",
        "4. memory_mask",
        "5. IO_or_ANALOG_or_MEMS_mask",
        "6. normalized_x_coordinate",
        "7. normalized_y_coordinate",
    ]
    channel_lines.extend(
        f"{8 + index}. {name}" for index, name in enumerate(selected_context_names)
    )
    text = f"""# ChipTherm Dataset v1 Context

This is a context-augmented logical dataset derived from `{repo_relative(base_root)}`.

Context set: `{context_set}`

Only the input X tensors are regenerated. The HotSpot temperature targets, physics
baseline predictions, residual targets, and metadata paths are reused from the base
dataset.

Input channel layout:

{chr(10).join(channel_lines)}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def default_out_dir(context_set: str) -> Path:
    if context_set == "all_context":
        return REPO_ROOT / "data/runs/benchmarks/dataset_v1_context"
    return REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation" / context_set


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
