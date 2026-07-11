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
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.physics_candidates import (  # noqa: E402
    ChipletSource,
    PhysicsCandidateConfig,
    predict_candidate_temperature,
)


SPLITS = ("train", "val", "test")
CANDIDATES = ("screened_poisson", "screened_poisson_calibrated", "hybrid_local_global", "compact_rc")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact analytical physics candidate predictions for ChipTherm.")
    parser.add_argument("--source-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean/package_plus_power", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/benchmarks/physics_candidates", type=Path)
    parser.add_argument("--candidate", default="all", choices=[*CANDIDATES, "all"])
    parser.add_argument("--ambient-K", default=318.15, type=float)
    parser.add_argument("--k-spread-W-per-K", default=0.30, type=float)
    parser.add_argument("--g-sink-W-per-mm2K", default=0.004, type=float)
    parser.add_argument("--global-R-eff-K-per-W", default=0.0, type=float)
    parser.add_argument("--source-scale", default=1.0, type=float)
    parser.add_argument("--ambient-offset-K", default=0.0, type=float)
    parser.add_argument("--calibration-file", default=None, type=Path)
    parser.add_argument(
        "--allow-periodic-fft-fallback",
        action="store_true",
        help="Allow FFT periodic fallback if SciPy DCT is unavailable. This changes boundary conditions.",
    )
    parser.add_argument("--local-kernel-length-mm", default=1.5, type=float)
    parser.add_argument("--local-kernel-epsilon-mm", default=0.75, type=float)
    parser.add_argument("--local-kernel-gain-K-mm-per-W", default=0.08, type=float)
    parser.add_argument("--local-quadrature-size", default=4, type=int)
    parser.add_argument("--rc-iterations", default=120, type=int)
    parser.add_argument("--rc-relaxation", default=0.90, type=float)
    parser.add_argument("--max-samples-per-split", default=None, type=int, help="Optional smoke-test limit; omit for full generation.")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    validate_source_root(source_root)
    candidates = list(CANDIDATES) if args.candidate == "all" else [args.candidate]

    for candidate in candidates:
        config = PhysicsCandidateConfig(
            name=candidate,
            ambient_K=args.ambient_K,
            k_spread_W_per_K=args.k_spread_W_per_K,
            g_sink_W_per_mm2K=args.g_sink_W_per_mm2K,
            source_scale=args.source_scale,
            ambient_offset_K=args.ambient_offset_K,
            global_R_eff_K_per_W=args.global_R_eff_K_per_W,
            allow_periodic_fft_fallback=args.allow_periodic_fft_fallback,
            local_kernel_length_mm=args.local_kernel_length_mm,
            local_kernel_epsilon_mm=args.local_kernel_epsilon_mm,
            local_kernel_gain_K_mm_per_W=args.local_kernel_gain_K_mm_per_W,
            local_quadrature_size=args.local_quadrature_size,
            rc_iterations=args.rc_iterations,
            rc_relaxation=args.rc_relaxation,
        )
        if args.calibration_file is not None:
            config = load_calibrated_config(config, args.calibration_file.expanduser().resolve(), candidate)
        generate_candidate(
            source_root=source_root,
            out_dir=out_root / candidate,
            config=config,
            max_samples_per_split=args.max_samples_per_split,
        )
    return 0


def generate_candidate(
    *,
    source_root: Path,
    out_dir: Path,
    config: PhysicsCandidateConfig,
    max_samples_per_split: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, str]] = []
    split_records: dict[str, list[dict[str, str]]] = {}
    runtimes: list[float] = []
    first_metadata: dict[str, Any] | None = None
    fieldnames: list[str] | None = None
    print(f"Generating candidate prior: {config.name}")

    for split in SPLITS:
        rows, split_fieldnames = read_index(source_root / f"{split}_index.csv")
        if max_samples_per_split is not None:
            rows = rows[: max_samples_per_split]
        fieldnames = split_fieldnames
        output_rows: list[dict[str, str]] = []
        for row_index, row in enumerate(rows, start=1):
            new_row, runtime_s, metadata = generate_one_sample(row, source_root=source_root, out_dir=out_dir, config=config)
            new_row["split"] = split
            output_rows.append(new_row)
            all_records.append(new_row)
            runtimes.append(runtime_s)
            if first_metadata is None:
                first_metadata = metadata
            if row_index % 500 == 0:
                print(f"  {split}: generated {row_index}/{len(rows)}")
        split_records[split] = output_rows
        write_csv(out_dir / f"{split}_index.csv", split_fieldnames, output_rows)

    if not fieldnames:
        raise SystemExit("no fieldnames found")
    write_csv(out_dir / "combined_encoded_index.csv", fieldnames, all_records)
    write_jsonl(out_dir / "combined_encoded_index.jsonl", all_records, config)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "out_dir": repo_relative(out_dir),
        "candidate": config.name,
        "config": config.to_dict(),
        "split_counts": {split: len(split_records.get(split, [])) for split in SPLITS},
        "case_counts": dict(sorted(Counter(row["case_id"] for row in all_records).items())),
        "generation_runtime_per_sample_s": float(sum(runtimes) / len(runtimes)) if runtimes else None,
        "generation_runtime_total_s": float(sum(runtimes)),
        "first_sample_metadata": first_metadata,
        "output_schema": {
            "prediction_path": "predictions/{case_id}/{sample_uid}_tphys_{candidate}.npy",
            "residual_path": "residuals/{case_id}/{sample_uid}_residual_{candidate}.npy",
            "x_path": "reused from source dataset",
            "y_path": "reused from source dataset",
        },
        "notes": [
            "No HotSpot labels are used to compute predictions.",
            "No per-case fitting or learned calibration is performed.",
            "The generated indexes are compatible with ChipThermDataset and train_residual_cnn.py.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_readme(out_dir / "README.md", manifest)
    print(f"{config.name} generation complete")
    print(f"Samples: {len(all_records)}")
    print(f"Runtime/sample: {manifest['generation_runtime_per_sample_s']:.6f} s")
    print(f"Output: {out_dir}")


def load_calibrated_config(
    base_config: PhysicsCandidateConfig,
    calibration_file: Path,
    candidate: str,
) -> PhysicsCandidateConfig:
    if not calibration_file.exists():
        raise SystemExit(f"calibration file not found: {calibration_file}")
    payload = json.loads(calibration_file.read_text(encoding="utf-8"))
    params = payload.get("final_parameters") or payload.get("parameters") or {}
    if not params:
        raise SystemExit(f"calibration file has no final_parameters: {calibration_file}")
    name = "screened_poisson_calibrated" if candidate == "screened_poisson_calibrated" else candidate
    return PhysicsCandidateConfig(
        **{
            **base_config.to_dict(),
            "name": name,
            "k_spread_W_per_K": float(params.get("k_eff_W_per_K", params.get("k_spread_W_per_K", base_config.k_spread_W_per_K))),
            "g_sink_W_per_mm2K": float(
                params.get("g_eff_W_per_mm2K", params.get("g_sink_W_per_mm2K", base_config.g_sink_W_per_mm2K))
            ),
            "source_scale": float(params.get("alpha_source", params.get("source_scale", base_config.source_scale))),
            "ambient_offset_K": float(params.get("ambient_offset_K", base_config.ambient_offset_K)),
            "global_R_eff_K_per_W": float(params.get("global_R_eff_K_per_W", base_config.global_R_eff_K_per_W)),
        }
    )


def generate_one_sample(
    row: dict[str, str],
    *,
    source_root: Path,
    out_dir: Path,
    config: PhysicsCandidateConfig,
) -> tuple[dict[str, str], float, dict[str, Any]]:
    x = np.load(resolve_path(row["x_path"], source_root)).astype(np.float32, copy=False)
    y = np.load(resolve_path(row["y_path"], source_root)).astype(np.float32, copy=False)
    chiplets = load_chiplets_for_row(row) if config.name == "hybrid_local_global" else None
    start = time.perf_counter()
    pred, metadata = predict_candidate_temperature(
        x,
        config,
        chiplets=chiplets,
        row_total_power_W=optional_float(row.get("total_power_W")),
    )
    runtime_s = time.perf_counter() - start
    if pred.shape != y.shape:
        raise SystemExit(f"{row['sample_uid']} prediction shape {pred.shape} does not match target {y.shape}")
    if not np.isfinite(pred).all():
        raise SystemExit(f"{row['sample_uid']} candidate prediction contains non-finite values")
    residual = (y - pred).astype(np.float32, copy=False)
    pred_dir = out_dir / "predictions" / row["case_id"]
    residual_dir = out_dir / "residuals" / row["case_id"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{row['sample_uid']}_tphys_{config.name}.npy"
    residual_path = residual_dir / f"{row['sample_uid']}_residual_{config.name}.npy"
    np.save(pred_path, pred.astype(np.float32, copy=False))
    np.save(residual_path, residual)
    new_row = dict(row)
    new_row["prediction_path"] = repo_relative(pred_path)
    new_row["residual_path"] = repo_relative(residual_path)
    new_row["physics_runtime_s"] = f"{runtime_s:.12g}"
    return (
        new_row,
        runtime_s,
        {
            "sample_uid": row["sample_uid"],
            "case_id": row["case_id"],
            "physics_runtime_s": runtime_s,
            "package_grid": metadata.to_dict(),
            "chiplet_count": len(chiplets or []),
        },
    )


def load_chiplets_for_row(row: dict[str, str]) -> list[ChipletSource]:
    layout_path, power_path = source_paths_for_row(row)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    with power_path.open("r", encoding="utf-8") as fp:
        power_data = yaml.safe_load(fp) or {}
    power_by_name = active_power_map(power_data)
    chiplets: list[ChipletSource] = []
    for chiplet in layout.get("chiplets", []):
        name = str(chiplet["name"])
        if name not in power_by_name:
            raise SystemExit(f"{row['sample_uid']} missing power for chiplet {name}")
        position = chiplet["position"]
        size = chiplet["size"]
        chiplets.append(
            ChipletSource(
                name=name,
                x_mm=float(position["x"]),
                y_mm=float(position["y"]),
                width_mm=float(size["width"]),
                height_mm=float(size["height"]),
                power_W=float(power_by_name[name]),
            )
        )
    return chiplets


def source_paths_for_row(row: dict[str, str]) -> tuple[Path, Path]:
    prefix = f"{row['case_id']}_"
    original_uid = row["original_sample_uid"]
    if not original_uid.startswith(prefix):
        raise SystemExit(f"{row['sample_uid']} original_sample_uid does not match case_id")
    sample_dir = original_uid[len(prefix) :]
    source_dir = REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / row["case_id"] / sample_dir / "source"
    layout_path = source_dir / "layout.json"
    power_path = source_dir / "power.yaml"
    if not layout_path.exists() or not power_path.exists():
        raise SystemExit(f"{row['sample_uid']} missing source layout/power metadata")
    return layout_path, power_path


def active_power_map(power_data: dict[str, Any]) -> dict[str, float]:
    active = power_data.get("active_workload")
    workloads = power_data.get("workloads") or {}
    if active and active in workloads:
        return {str(name): float(value) for name, value in workloads[active].items()}
    if "chiplets" in power_data:
        return {str(name): float(value) for name, value in power_data["chiplets"].items()}
    raise SystemExit("power metadata has no active workload or chiplets power map")


def validate_source_root(source_root: Path) -> None:
    missing = [source_root / f"{split}_index.csv" for split in SPLITS if not (source_root / f"{split}_index.csv").exists()]
    if missing:
        raise SystemExit(f"source root is missing split indexes: {', '.join(str(path) for path in missing)}")


def read_index(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"sample_uid", "original_sample_uid", "case_id", "dataset_source", "split", "x_path", "y_path", "prediction_path", "residual_path"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"{path} missing columns: {', '.join(missing)}")
    return rows, fieldnames


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, str]], config: PhysicsCandidateConfig) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            payload = dict(row)
            payload["physics_candidate"] = {
                "candidate": config.name,
                "config": config.to_dict(),
            }
            fp.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Physics Candidate: {manifest['candidate']}

This directory contains prediction and residual tensors for a compact analytical
thermal prior. X tensors and HotSpot Y tensors are reused from:

`{manifest['source_root']}`

No HotSpot labels are used to compute the candidate prediction, and no per-case
fitting is performed.

## Files

- `train_index.csv`, `val_index.csv`, `test_index.csv`
- `combined_encoded_index.csv`
- `combined_encoded_index.jsonl`
- `predictions/`
- `residuals/`
- `manifest.json`

Generation runtime/sample: `{manifest['generation_runtime_per_sample_s']:.6f} s`
"""
    path.write_text(text, encoding="utf-8")


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, base / path, Path.cwd() / path]
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


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
