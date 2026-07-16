#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponseNormalizationStats,
    build_source_input,
    normalize_source_input,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, predict_source_rise  # noqa: E402


SPLITS = ("train", "val", "test")
MAP_SHAPE = (64, 64)
MAP_DTYPE = "float32"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full canonical source-superposition base maps.")
    parser.add_argument("--index", default=None, type=Path, help="Combined index with a split column; split-specific indexes are derived from it.")
    parser.add_argument("--train-index", default=None, type=Path)
    parser.add_argument("--val-index", default=None, type=Path)
    parser.add_argument("--test-index", default=None, type=Path)
    parser.add_argument("--checkpoint", "--source-checkpoint", dest="checkpoint", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--package-batch-size", default=8, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int, help="Reserved for compatibility; source generation is synchronous.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", default=None, type=int)
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--precision", default="fp32", choices=["fp32"])
    parser.add_argument("--device-summation", action="store_true", help="Reserved for compatibility; current implementation uses host float64 accumulation.")
    parser.add_argument("--max-storage-gb", default=None, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    if args.package_batch_size <= 0:
        raise SystemExit("--package-batch-size must be positive")
    if args.source_batch_size <= 0:
        raise SystemExit("--source-batch-size must be positive")

    out_root = args.out_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    split_indices = prepare_split_indices(args, out_root)
    audit = audit_canonical_inputs(split_indices, checkpoint)
    print_audit(audit)
    if args.max_storage_gb is not None:
        estimated_gb = float(audit["estimated_map_storage_bytes"]) / (1024.0**3)
        estimated_gb *= 2.0  # base map plus residual map
        print(f"Estimated durable base+residual storage: {estimated_gb:.4f} GB")
        print(f"Estimated temporary storage: <0.1 GB (microbatches are not persisted)")
        if estimated_gb > args.max_storage_gb:
            raise SystemExit(f"estimated storage {estimated_gb:.4f} GB exceeds --max-storage-gb {args.max_storage_gb:.4f}")
    if args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "alignment_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(checkpoint_payload["normalization"])
    model = build_source_response_model(checkpoint_payload["model_config"]).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    checkpoint_sha = sha256_file(checkpoint)
    checkpoint_identity = {
        "path": str(checkpoint),
        "sha256": checkpoint_sha,
        "config_sha256": stable_json_sha256(checkpoint_payload.get("model_config", {})),
        "model_config": checkpoint_payload.get("model_config", {}),
    }

    start = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_config_sha256": checkpoint_identity["config_sha256"],
        "source_model_config": checkpoint_payload.get("model_config", {}),
        "source_normalization": stats.to_dict(),
        "canonical_indices": {split: str(path.resolve()) for split, path in split_indices.items()},
        "canonical_row_counts": {split: audit["splits"][split]["row_count"] for split in SPLITS},
        "map_units": "absolute_temperature_K",
        "map_shape": list(MAP_SHAPE),
        "map_dtype": MAP_DTYPE,
        "base_definition": "ambient_K + sum_i source_power_i * source_response_operator(source_i)",
        "generation_command": " ".join(sys.argv),
        "precision": args.precision,
        "device_summation_requested": bool(args.device_summation),
        "device_summation_used": False,
        "git_commit": git_commit(),
        "splits": {},
    }

    for split in SPLITS:
        rows = read_rows(split_indices[split])
        generated, split_summary = generate_split(
            split=split,
            rows=rows,
            source_index_path=split_indices[split],
            out_root=out_root,
            model=model,
            stats=stats,
            checkpoint_identity=checkpoint_identity,
            package_batch_size=args.package_batch_size,
            source_batch_size=args.source_batch_size,
            device=device,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        write_index(out_root / f"{split}_index.csv", rows, generated)
        manifest["splits"][split] = split_summary

    copy_support_files(split_indices["train"].parent, out_root)
    write_combined(out_root)
    manifest["generated_row_counts"] = {split: int(manifest["splits"][split]["rows"]) for split in SPLITS}
    manifest["total_packages"] = int(sum(manifest["generated_row_counts"].values()))
    manifest["total_source_count"] = int(sum(manifest["splits"][split]["source_count"] for split in SPLITS))
    manifest["generation_runtime_s"] = time.perf_counter() - start
    manifest["generation_runtime_per_package_s"] = manifest["generation_runtime_s"] / max(manifest["total_packages"], 1)
    manifest["generation_runtime_per_source_s"] = manifest["generation_runtime_s"] / max(manifest["total_source_count"], 1)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "alignment_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root, manifest)
    print(f"Full source-superposition base maps written to {out_root}")
    print(f"Packages: {manifest['total_packages']}")
    print(f"Source predictions: {manifest['total_source_count']}")
    print(f"Runtime/package: {manifest['generation_runtime_per_package_s']:.6f} s")
    print(f"Runtime/source: {manifest['generation_runtime_per_source_s']:.6f} s")
    return 0


def audit_canonical_inputs(split_indices: dict[str, Path], checkpoint: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_exists": checkpoint.exists(),
        "splits": {},
    }
    all_uids: dict[str, set[str]] = {}
    total_sources = 0
    for split in SPLITS:
        path = split_indices[split].expanduser().resolve()
        fieldnames, rows = read_csv_with_fieldnames(path)
        seen: set[str] = set()
        duplicates: list[str] = []
        missing_files: list[dict[str, str]] = []
        case_counts: Counter[str] = Counter()
        source_count = 0
        for index, row in enumerate(rows):
            uid = row["sample_uid"]
            if uid in seen:
                duplicates.append(uid)
            seen.add(uid)
            case_counts[row["case_id"]] += 1
            paths = canonical_source_paths(row)
            for name, item in paths.items():
                if not item.exists():
                    missing_files.append({"row_index": str(index), "sample_uid": uid, "missing": name, "path": str(item)})
            if paths["layout"].exists() and paths["power"].exists():
                layout = load_json(paths["layout"])
                power = load_yaml(paths["power"])
                chiplets = list(layout.get("chiplets", []))
                powers = active_power_map(power)
                missing_power = [str(chiplet.get("name", "")) for chiplet in chiplets if str(chiplet.get("name", "")) not in powers]
                if missing_power:
                    missing_files.append(
                        {
                            "row_index": str(index),
                            "sample_uid": uid,
                            "missing": "chiplet_power",
                            "path": ",".join(missing_power),
                        }
                    )
                source_count += len(chiplets)
        all_uids[split] = seen
        total_sources += source_count
        payload["splits"][split] = {
            "index_path": str(path),
            "fieldnames": fieldnames,
            "row_count": len(rows),
            "first_sample_uid": rows[0]["sample_uid"] if rows else None,
            "duplicate_sample_uids": duplicates,
            "missing_files": missing_files[:50],
            "missing_file_count": len(missing_files),
            "case_counts": dict(sorted(case_counts.items())),
            "source_count": source_count,
            "mean_sources_per_package": source_count / max(len(rows), 1),
        }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = sorted(all_uids[left] & all_uids[right])
    payload["cross_split_uid_overlaps"] = {name: values[:50] for name, values in overlaps.items()}
    payload["cross_split_uid_overlap_count"] = int(sum(len(values) for values in overlaps.values()))
    payload["total_packages"] = int(sum(payload["splits"][split]["row_count"] for split in SPLITS))
    payload["total_source_count"] = int(total_sources)
    payload["estimated_map_storage_bytes"] = int(payload["total_packages"] * MAP_SHAPE[0] * MAP_SHAPE[1] * 4)
    payload["estimated_map_storage_MB"] = payload["estimated_map_storage_bytes"] / 1.0e6
    problems = []
    if not checkpoint.exists():
        problems.append(f"checkpoint missing: {checkpoint}")
    for split in SPLITS:
        item = payload["splits"][split]
        if item["duplicate_sample_uids"]:
            problems.append(f"{split} has duplicate sample_uids")
        if item["missing_file_count"]:
            problems.append(f"{split} has missing required files")
    if payload["cross_split_uid_overlap_count"]:
        problems.append("sample_uid overlap across splits")
    payload["supported"] = not problems
    payload["problems"] = problems
    return payload


def print_audit(audit: dict[str, Any]) -> None:
    print("Full source-superposition dry-run audit:")
    for split in SPLITS:
        item = audit["splits"][split]
        print(
            f"  {split}: rows={item['row_count']} sources={item['source_count']} "
            f"mean_sources={item['mean_sources_per_package']:.2f} missing_files={item['missing_file_count']}"
        )
    print(f"Total packages: {audit['total_packages']}")
    print(f"Total source predictions: {audit['total_source_count']}")
    print(f"Estimated map storage: {audit['estimated_map_storage_MB']:.2f} MB")
    if audit["problems"]:
        print("Problems:")
        for problem in audit["problems"]:
            print(f"  - {problem}")


def prepare_split_indices(args: argparse.Namespace, out_root: Path) -> dict[str, Path]:
    if args.index is not None:
        _, rows = read_csv_with_fieldnames(args.index.expanduser().resolve())
        if args.case_ids:
            allowed = set(args.case_ids)
            rows = [row for row in rows if row.get("case_id") in allowed]
        if args.max_samples is not None:
            rows = rows[: int(args.max_samples)]
        if not rows:
            raise SystemExit("--index selection produced zero rows")
        split_dir = out_root / "_input_splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_indices: dict[str, Path] = {}
        fieldnames = list(rows[0].keys())
        for split in SPLITS:
            split_rows = [dict(row, split=split) for row in rows if row.get("split") == split]
            path = split_dir / f"{split}_index.csv"
            write_csv(path, split_rows, fieldnames)
            split_indices[split] = path
        return split_indices
    missing = [name for name in ("train_index", "val_index", "test_index") if getattr(args, name) is None]
    if missing:
        raise SystemExit("provide either --index or all of --train-index/--val-index/--test-index")
    split_indices = {"train": args.train_index, "val": args.val_index, "test": args.test_index}
    filtered: dict[str, Path] = {}
    if args.case_ids or args.max_samples is not None:
        split_dir = out_root / "_input_splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        allowed = set(args.case_ids or [])
        remaining = args.max_samples
        for split in SPLITS:
            fields, rows = read_csv_with_fieldnames(split_indices[split])
            if allowed:
                rows = [row for row in rows if row.get("case_id") in allowed]
            if remaining is not None:
                rows = rows[: max(0, int(remaining))]
                remaining -= len(rows)
            path = split_dir / f"{split}_index.csv"
            write_csv(path, rows, fields)
            filtered[split] = path
        return filtered
    return {split: path.expanduser().resolve() for split, path in split_indices.items()}


@torch.no_grad()
def generate_split(
    *,
    split: str,
    rows: list[dict[str, str]],
    source_index_path: Path,
    out_root: Path,
    model: torch.nn.Module,
    stats: SourceResponseNormalizationStats,
    checkpoint_identity: dict[str, Any],
    package_batch_size: int,
    source_batch_size: int,
    device: torch.device,
    resume: bool,
    overwrite: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    start = time.perf_counter()
    generated: list[dict[str, str]] = []
    processed_sources = 0
    reused = 0
    regenerated = 0
    for start_index in range(0, len(rows), package_batch_size):
        batch_rows = rows[start_index : start_index + package_batch_size]
        pending_rows: list[dict[str, str]] = []
        pending_packages: list[dict[str, Any]] = []
        for row in batch_rows:
            map_path = map_path_for_row(out_root, split, row)
            residual_path = residual_path_for_row(out_root, split, row)
            status = "reused"
            if resume and not overwrite and valid_existing_map(map_path, sidecar_path(map_path), row, checkpoint_identity):
                ensure_residual(row, map_path, residual_path)
                reused += 1
            else:
                status = "generated"
                pending_rows.append(row)
                pending_packages.append(load_package_inputs(row, source_index_path))
            generated.append(output_row(row, map_path, residual_path, checkpoint_identity, status))
        if pending_rows:
            maps = infer_package_maps(pending_packages, model, stats, source_batch_size, device)
            for row, package, base_map in zip(pending_rows, pending_packages, maps, strict=True):
                map_path = map_path_for_row(out_root, split, row)
                residual_path = residual_path_for_row(out_root, split, row)
                save_map_and_sidecar(
                    row=row,
                    package=package,
                    map_path=map_path,
                    residual_path=residual_path,
                    base_map=base_map,
                    checkpoint_identity=checkpoint_identity,
                )
                processed_sources += int(package["num_sources"])
                regenerated += 1
        completed = min(start_index + package_batch_size, len(rows))
        elapsed = time.perf_counter() - start
        if completed and (completed == len(rows) or completed % max(package_batch_size * 10, 1) == 0):
            per_package = elapsed / completed
            remaining = per_package * (len(rows) - completed)
            print(
                f"{split}: packages {completed}/{len(rows)} regenerated={regenerated} reused={reused} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s"
            )
    return generated, {
        "rows": len(rows),
        "source_count": int(sum(int(row["source_count"]) for row in generated)),
        "regenerated_packages": regenerated,
        "reused_packages": reused,
        "runtime_s": time.perf_counter() - start,
        "runtime_per_package_s": (time.perf_counter() - start) / max(len(rows), 1),
        "processed_source_predictions": processed_sources,
    }


def load_package_inputs(row: dict[str, str], index_path: Path) -> dict[str, Any]:
    paths = canonical_source_paths(row)
    x = np.load(resolve_path(row["x_path"], index_path.parent)).astype(np.float32, copy=False)
    layout = load_json(paths["layout"])
    power = load_yaml(paths["power"])
    package = load_yaml(paths["package"])
    chiplets = list(layout.get("chiplets", []))
    if not chiplets:
        raise ValueError(f"{paths['layout']} has no chiplets")
    powers = active_power_map(power)
    source_inputs: list[np.ndarray] = []
    source_powers: list[float] = []
    source_names: list[str] = []
    for source_index, chiplet in enumerate(chiplets):
        name = str(chiplet["name"])
        if name not in powers:
            raise ValueError(f"{paths['power']} missing power for chiplet {name}")
        source_power = float(powers[name])
        source_inputs.append(build_source_input(x, layout, source_index, source_power))
        source_powers.append(source_power)
        source_names.append(name)
    ambient_K = float(package.get("ambient_K", 318.15))
    return {
        "row": row,
        "x": x,
        "source_inputs": source_inputs,
        "source_powers": np.asarray(source_powers, dtype=np.float32),
        "source_names": source_names,
        "ambient_K": ambient_K,
        "layout_path": paths["layout"],
        "power_path": paths["power"],
        "package_path": paths["package"],
        "hotspot_path": paths["hotspot"],
        "num_sources": len(source_inputs),
    }


@torch.no_grad()
def infer_package_maps(
    packages: list[dict[str, Any]],
    model: torch.nn.Module,
    stats: SourceResponseNormalizationStats,
    source_batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    flat_inputs: list[np.ndarray] = []
    flat_powers: list[float] = []
    package_ids: list[int] = []
    for package_index, package in enumerate(packages):
        for source_input, source_power in zip(package["source_inputs"], package["source_powers"], strict=True):
            flat_inputs.append(source_input)
            flat_powers.append(float(source_power))
            package_ids.append(package_index)
    sums = [np.zeros(MAP_SHAPE, dtype=np.float64) for _ in packages]
    for start in range(0, len(flat_inputs), source_batch_size):
        stop = min(start + source_batch_size, len(flat_inputs))
        x = torch.from_numpy(np.stack(flat_inputs[start:stop]).astype(np.float32, copy=False)).to(device)
        power = torch.tensor(flat_powers[start:stop], dtype=torch.float32, device=device)
        pred_unit = unnormalize_source_prediction(model(normalize_source_input(x, stats)), stats)
        pred_rise = predict_source_rise(pred_unit, power).detach().cpu().numpy()
        for local_index, rise in enumerate(pred_rise):
            sums[package_ids[start + local_index]] += rise.astype(np.float64, copy=False)
    maps: list[np.ndarray] = []
    for package, rise_sum in zip(packages, sums, strict=True):
        base = np.asarray(float(package["ambient_K"]) + rise_sum, dtype=np.float32)
        if base.shape != MAP_SHAPE:
            raise ValueError(f"generated base map has shape {base.shape}, expected {MAP_SHAPE}")
        if not np.isfinite(base).all():
            raise ValueError(f"generated base map for {package['row']['sample_uid']} contains non-finite values")
        maps.append(base)
    return maps


def save_map_and_sidecar(
    *,
    row: dict[str, str],
    package: dict[str, Any],
    map_path: Path,
    residual_path: Path,
    base_map: np.ndarray,
    checkpoint_identity: dict[str, Any],
) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    residual_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(map_path, base_map.astype(np.float32, copy=False))
    target = np.load(resolve_path(row["y_path"])).astype(np.float32, copy=False)
    atomic_save_npy(residual_path, (target - base_map).astype(np.float32, copy=False))
    metadata = {
        "schema_version": 1,
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "split": row.get("split"),
        "canonical_x_path": row["x_path"],
        "canonical_y_path": row["y_path"],
        "source_checkpoint": checkpoint_identity["path"],
        "source_checkpoint_sha256": checkpoint_identity["sha256"],
        "source_count": int(package["num_sources"]),
        "source_names": package["source_names"],
        "ambient_K": float(package["ambient_K"]),
        "map_units": "absolute_temperature_K",
        "map_shape": list(base_map.shape),
        "map_dtype": str(base_map.dtype),
        "layout_path": repo_relative(package["layout_path"]),
        "power_path": repo_relative(package["power_path"]),
        "package_path": repo_relative(package["package_path"]),
        "hotspot_path": repo_relative(package["hotspot_path"]),
    }
    sidecar_path(map_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def valid_existing_map(
    map_path: Path,
    metadata_path: Path,
    row: dict[str, str],
    checkpoint_identity: dict[str, Any],
) -> bool:
    if not map_path.exists() or not metadata_path.exists():
        return False
    try:
        array = np.load(map_path, mmap_mode="r")
        if tuple(array.shape) != MAP_SHAPE or str(array.dtype) != MAP_DTYPE:
            return False
        if not np.isfinite(np.asarray(array)).all():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        metadata.get("sample_uid") == row.get("sample_uid")
        and metadata.get("case_id") == row.get("case_id")
        and metadata.get("source_checkpoint_sha256") == checkpoint_identity.get("sha256")
    )


def ensure_residual(row: dict[str, str], map_path: Path, residual_path: Path) -> None:
    if residual_path.exists():
        try:
            residual = np.load(residual_path, mmap_mode="r")
            if tuple(residual.shape) == MAP_SHAPE and str(residual.dtype) == MAP_DTYPE:
                return
        except Exception:
            pass
    target = np.load(resolve_path(row["y_path"])).astype(np.float32, copy=False)
    base = np.load(map_path).astype(np.float32, copy=False)
    residual_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(residual_path, (target - base).astype(np.float32, copy=False))


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + ".tmp.npy")
    np.save(tmp_path, array)
    tmp_path.replace(path)


def output_row(
    row: dict[str, str],
    map_path: Path,
    residual_path: Path,
    checkpoint_identity: dict[str, Any],
    status: str,
) -> dict[str, str]:
    result = dict(row)
    result.setdefault("prediction_path", "")
    result.setdefault("residual_path", "")
    result["source_superposition_base_path"] = repo_relative(map_path)
    result["source_superposition_residual_path"] = repo_relative(residual_path)
    result["source_checkpoint"] = checkpoint_identity["path"]
    result["source_checkpoint_sha256"] = checkpoint_identity["sha256"]
    result["source_checkpoint_config_sha256"] = str(
        checkpoint_identity.get("config_sha256") or stable_json_sha256(checkpoint_identity.get("model_config", {}))
    )
    result["source_count"] = str(row.get("num_chiplets", ""))
    result["source_model_version"] = str(checkpoint_identity["model_config"].get("architecture", "source_response_operator_v1"))
    result["source_base_units"] = "absolute_temperature_K"
    result["source_base_shape"] = "64x64"
    result["source_base_dtype"] = MAP_DTYPE
    result["source_base_mode"] = "source_superposition_v1"
    result["generation_status"] = status
    paths = canonical_source_paths(row)
    result["source_layout_path"] = repo_relative(paths["layout"])
    result["source_power_path"] = repo_relative(paths["power"])
    result["source_package_path"] = repo_relative(paths["package"])
    result["source_hotspot_path"] = repo_relative(paths["hotspot"])
    return result


def write_index(path: Path, canonical_rows: list[dict[str, str]], generated_rows: list[dict[str, str]]) -> None:
    if len(canonical_rows) != len(generated_rows):
        raise ValueError(f"{path} row count mismatch: canonical={len(canonical_rows)} generated={len(generated_rows)}")
    for index, (left, right) in enumerate(zip(canonical_rows, generated_rows, strict=True)):
        if left["sample_uid"] != right["sample_uid"]:
            raise ValueError(f"{path} row {index} sample_uid reordered: {left['sample_uid']} != {right['sample_uid']}")
    canonical_fields = list(canonical_rows[0].keys()) if canonical_rows else []
    appended = [
        "prediction_path",
        "residual_path",
        "source_superposition_base_path",
        "source_superposition_residual_path",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_checkpoint_config_sha256",
        "source_count",
        "source_model_version",
        "source_base_units",
        "source_base_shape",
        "source_base_dtype",
        "source_base_mode",
        "generation_status",
        "source_layout_path",
        "source_power_path",
        "source_package_path",
        "source_hotspot_path",
    ]
    fieldnames = canonical_fields + [name for name in appended if name not in canonical_fields]
    write_csv(path, generated_rows, fieldnames)


def write_combined(out_root: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for split in SPLITS:
        split_fields, split_rows = read_csv_with_fieldnames(out_root / f"{split}_index.csv")
        rows.extend(split_rows)
        for name in split_fields:
            if name not in fieldnames:
                fieldnames.append(name)
    write_csv(out_root / "combined_encoded_index.csv", rows, fieldnames)
    with (out_root / "combined_encoded_index.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def copy_support_files(source_root: Path, out_root: Path) -> None:
    for name in (
        "feature_manifest.json",
        "context_manifest.json",
        "metadata_manifest.json",
        "metadata_features.csv",
        "graph_manifest.json",
        "README.md",
    ):
        source = source_root / name
        if source.exists():
            shutil.copy2(source, out_root / name)


def canonical_source_paths(row: dict[str, str]) -> dict[str, Path]:
    source_dir = source_dir_for_row(row)
    return {
        "source_dir": source_dir,
        "layout": resolve_path(row.get("layout_path", ""), source_dir) if row.get("layout_path") else source_dir / "layout.json",
        "power": resolve_path(row.get("power_path", ""), source_dir) if row.get("power_path") else source_dir / "power.yaml",
        "package": resolve_path(row.get("package_path", ""), source_dir) if row.get("package_path") else source_dir / "package.yaml",
        "hotspot": resolve_path(row.get("hotspot_path", ""), source_dir) if row.get("hotspot_path") else source_dir / "hotspot.yaml",
        "x": resolve_path(row["x_path"]),
        "y": resolve_path(row["y_path"]),
        "graph": resolve_path(row["graph_path"]) if row.get("graph_path") else source_dir,
    }


def source_dir_for_row(row: dict[str, str]) -> Path:
    if row.get("source_dir"):
        return resolve_path(row["source_dir"])
    case_id = row["case_id"]
    original = row.get("original_sample_uid") or row["sample_uid"]
    sample_name = original
    prefix = f"{case_id}_"
    if sample_name.startswith(prefix):
        sample_name = sample_name[len(prefix) :]
    return REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_name / "source"


def active_power_map(power: dict[str, Any]) -> dict[str, float]:
    workload = power.get("active_workload", "nominal")
    workloads = power.get("workloads") or {}
    if workload in workloads:
        return {str(name): float(value) for name, value in workloads[workload].items()}
    if "chiplets" in power:
        return {str(name): float(value) for name, value in power["chiplets"].items()}
    raise ValueError("power.yaml has no active workload or chiplets map")


def map_path_for_row(out_root: Path, split: str, row: dict[str, str]) -> Path:
    return out_root / "maps" / split / row["case_id"] / f"{row['sample_uid']}_source_superposition_base.npy"


def residual_path_for_row(out_root: Path, split: str, row: dict[str, str]) -> Path:
    return out_root / "residuals" / split / row["case_id"] / f"{row['sample_uid']}_source_superposition_residual.npy"


def sidecar_path(map_path: Path) -> Path:
    return map_path.with_suffix(".json")


def read_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_with_fieldnames(path)[1]


def read_csv_with_fieldnames(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def resolve_path(path_value: str, base: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path]
    if base is not None:
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    return torch.device(choice)


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    text = f"""# Full Source-Superposition Base v1

Generated absolute-temperature base maps from a frozen source-response model.

- Source checkpoint: `{manifest['source_checkpoint']}`
- Train/val/test rows: {manifest['canonical_row_counts']}
- Total packages: {manifest['total_packages']}
- Total source predictions: {manifest['total_source_count']}
- Map definition: `{manifest['base_definition']}`
- Map units: `{manifest['map_units']}`

Canonical columns are preserved in each split index. The appended
`source_superposition_base_path` column points to the generated base map.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
