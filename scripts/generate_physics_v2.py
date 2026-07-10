#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.physics_v2 import (  # noqa: E402
    PhysicsV2Config,
    build_feature_stack,
    feature_names,
    fit_coefficients_from_accumulators,
    predict_temperature_v2,
)


SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate package-aware physics_v2 predictions and residuals.")
    parser.add_argument(
        "--source-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power",
        type=Path,
        help="Root containing train/val/test index CSV files.",
    )
    parser.add_argument(
        "--out-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1_physics_v2/package_plus_power",
        type=Path,
        help="Output root for physics_v2 predictions, residuals, and rewritten indexes.",
    )
    parser.add_argument("--calibration-index", default=None, type=Path, help="Train-only index used to fit physics_v2 coefficients.")
    parser.add_argument("--ambient-K", default=318.15, type=float)
    parser.add_argument("--sigma-mm", nargs="+", default=[1.0, 2.0, 4.0, 8.0], type=float)
    parser.add_argument("--ridge-alpha", default=1.0e-6, type=float)
    parser.add_argument("--max-calibration-samples", default=None, type=int)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    calibration_index = (args.calibration_index or source_root / "train_index.csv").expanduser().resolve()

    config = PhysicsV2Config(
        ambient_K=args.ambient_K,
        sigma_mm=tuple(float(value) for value in args.sigma_mm),
        ridge_alpha=float(args.ridge_alpha),
    )
    validate_source_root(source_root)
    calibration_rows, _ = read_index(calibration_index)
    if args.max_calibration_samples is not None:
        calibration_rows = calibration_rows[: args.max_calibration_samples]
    validate_rows_have_paths(calibration_rows, calibration_index.parent)

    out_root.mkdir(parents=True, exist_ok=True)
    calibration_start = time.perf_counter()
    coefficients, calibration_summary = calibrate_physics_v2(
        calibration_rows,
        calibration_base=calibration_index.parent,
        config=config,
    )
    calibration_runtime_s = time.perf_counter() - calibration_start
    write_json(out_root / "physics_v2_coefficients.json", coefficients.to_dict())

    split_records: dict[str, list[dict[str, str]]] = {}
    all_records: list[dict[str, str]] = []
    generation_runtimes: list[float] = []
    metadata_preview: dict[str, Any] | None = None

    for split in SPLITS:
        rows, fieldnames = read_index(source_root / f"{split}_index.csv")
        validate_rows_have_paths(rows, source_root)
        output_rows: list[dict[str, str]] = []
        for row in rows:
            new_row, runtime_s, metadata = generate_one_sample(
                row,
                source_base=source_root,
                out_root=out_root,
                config=config,
                coefficients=coefficients,
            )
            new_row["split"] = split
            output_rows.append(new_row)
            all_records.append(new_row)
            generation_runtimes.append(runtime_s)
            if metadata_preview is None:
                metadata_preview = metadata
        split_records[split] = output_rows
        write_csv(out_root / f"{split}_index.csv", fieldnames, output_rows)

    combined_fieldnames = list(split_records["train"][0].keys())
    write_csv(out_root / "combined_encoded_index.csv", combined_fieldnames, all_records)
    write_jsonl(out_root / "combined_encoded_index.jsonl", all_records, config=config, coefficients=coefficients)
    avg_runtime_s = float(sum(generation_runtimes) / len(generation_runtimes)) if generation_runtimes else None
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "out_root": repo_relative(out_root),
        "calibration_index": repo_relative(calibration_index),
        "physics_version": "physics_v2_physical_mm_gaussian_ridge",
        "config": config.to_dict(),
        "feature_names": list(feature_names(config)),
        "coefficients": coefficients.to_dict(),
        "calibration": {
            **calibration_summary,
            "runtime_s": calibration_runtime_s,
            "max_calibration_samples": args.max_calibration_samples,
        },
        "split_counts": {split: len(split_records[split]) for split in SPLITS},
        "case_counts": dict(sorted(Counter(row["case_id"] for row in all_records).items())),
        "generation_runtime_per_sample_s": avg_runtime_s,
        "output_schema": {
            "prediction_path": "physics_v2/predictions/{case_id}/{sample_uid}_tphys_v2.npy",
            "residual_path": "physics_v2/residuals/{case_id}/{sample_uid}_residual_v2.npy",
            "x_path": "reused from source dataset",
            "y_path": "reused from source dataset",
        },
        "first_sample_metadata": metadata_preview,
        "notes": [
            "physics_v1 files are not modified.",
            "Calibration uses only the provided calibration index.",
            "The generated indexes are compatible with ChipThermDataset and train_residual_cnn.py.",
        ],
    }
    write_json(out_root / "manifest.json", manifest)
    write_readme(out_root, manifest)

    print("physics_v2 generation complete")
    print(f"Source root: {source_root}")
    print(f"Output root: {out_root}")
    print(f"Calibration samples: {coefficients.calibration_samples}")
    print(f"Calibration runtime: {calibration_runtime_s:.3f} s")
    print(f"Generated samples: {len(all_records)}")
    print(f"Generation runtime/sample: {avg_runtime_s:.6f} s" if avg_runtime_s else "Generation runtime/sample: n/a")
    print(f"Coefficients: {dict(zip(coefficients.feature_names, coefficients.coefficients))}")
    return 0


def calibrate_physics_v2(
    rows: list[dict[str, str]],
    *,
    calibration_base: Path,
    config: PhysicsV2Config,
) -> tuple[Any, dict[str, Any]]:
    names = feature_names(config)
    xtx = np.zeros((len(names), len(names)), dtype=np.float64)
    xty = np.zeros((len(names),), dtype=np.float64)
    num_cells = 0
    package_counts: Counter[str] = Counter()

    for row in rows:
        x = np.load(resolve_path(row["x_path"], calibration_base)).astype(np.float32, copy=False)
        y = np.load(resolve_path(row["y_path"], calibration_base)).astype(np.float32, copy=False)
        features, metadata = build_feature_stack(x, config, row_total_power_W=optional_float(row.get("total_power_W")))
        f = features.reshape(features.shape[0], -1).astype(np.float64, copy=False)
        target = (y.reshape(-1).astype(np.float64, copy=False) - float(config.ambient_K))
        xtx += f @ f.T
        xty += f @ target
        num_cells += int(target.size)
        package_counts[f"{metadata.package_width_mm:g}x{metadata.package_height_mm:g}mm"] += 1

    coefficients = fit_coefficients_from_accumulators(
        xtx,
        xty,
        config=config,
        calibration_samples=len(rows),
        calibration_cells=num_cells,
    )
    summary = {
        "num_samples": len(rows),
        "num_cells": num_cells,
        "package_counts": dict(sorted(package_counts.items())),
    }
    return coefficients, summary


def generate_one_sample(
    row: dict[str, str],
    *,
    source_base: Path,
    out_root: Path,
    config: PhysicsV2Config,
    coefficients: Any,
) -> tuple[dict[str, str], float, dict[str, Any]]:
    x = np.load(resolve_path(row["x_path"], source_base)).astype(np.float32, copy=False)
    y = np.load(resolve_path(row["y_path"], source_base)).astype(np.float32, copy=False)
    start = time.perf_counter()
    pred, metadata = predict_temperature_v2(
        x,
        config,
        coefficients,
        row_total_power_W=optional_float(row.get("total_power_W")),
    )
    runtime_s = time.perf_counter() - start
    residual = (y - pred).astype(np.float32, copy=False)

    case_dir_pred = out_root / "physics_v2" / "predictions" / row["case_id"]
    case_dir_resid = out_root / "physics_v2" / "residuals" / row["case_id"]
    case_dir_pred.mkdir(parents=True, exist_ok=True)
    case_dir_resid.mkdir(parents=True, exist_ok=True)
    pred_path = case_dir_pred / f"{row['sample_uid']}_tphys_v2.npy"
    residual_path = case_dir_resid / f"{row['sample_uid']}_residual_v2.npy"
    np.save(pred_path, pred.astype(np.float32, copy=False))
    np.save(residual_path, residual)

    new_row = dict(row)
    new_row["prediction_path"] = repo_relative(pred_path)
    new_row["residual_path"] = repo_relative(residual_path)
    new_row["physics_runtime_s"] = f"{runtime_s:.12g}"
    sample_metadata = {
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "physics_runtime_s": runtime_s,
        "package_grid": metadata.to_dict(),
    }
    return new_row, runtime_s, sample_metadata


def validate_source_root(source_root: Path) -> None:
    missing = [source_root / f"{split}_index.csv" for split in SPLITS if not (source_root / f"{split}_index.csv").exists()]
    if missing:
        raise SystemExit(f"source root is missing split indexes: {', '.join(str(path) for path in missing)}")


def read_index(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise SystemExit(f"{path} has no header")
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return rows, fieldnames


def validate_rows_have_paths(rows: list[dict[str, str]], base: Path) -> None:
    required = ("x_path", "y_path", "prediction_path", "residual_path")
    missing: list[str] = []
    for row in rows[:]:
        for key in required:
            if key not in row or not row[key]:
                missing.append(f"{row.get('sample_uid', '<unknown>')} missing {key}")
                continue
            if key in {"x_path", "y_path"} and not resolve_path(row[key], base).exists():
                missing.append(f"{row.get('sample_uid', '<unknown>')} {key} not found: {row[key]}")
        if len(missing) >= 20:
            break
    if missing:
        raise SystemExit("\n".join(missing[:20]))


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path, base / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]], *, config: PhysicsV2Config, coefficients: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            record = dict(row)
            record["physics_v2"] = {
                "config": config.to_dict(),
                "coefficients": coefficients.to_dict(),
            }
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Physics v2 Dataset View

This directory contains generated physics_v2 predictions and residuals while
reusing the source X tensors and HotSpot Y labels.

- Source root: `{manifest['source_root']}`
- Calibration index: `{manifest['calibration_index']}`
- Physics version: `{manifest['physics_version']}`
- Train/val/test samples: {manifest['split_counts']['train']} / {manifest['split_counts']['val']} / {manifest['split_counts']['test']}

The index files are compatible with `ChipThermDataset` and
`scripts/train_residual_cnn.py`. Existing physics_v1 files are not modified.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
