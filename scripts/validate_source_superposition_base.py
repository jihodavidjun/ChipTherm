#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiptherm.ml.source_response_dataset import SourceResponseNormalizationStats  # noqa: E402
from chiptherm.ml.source_response_models import build_source_response_model  # noqa: E402
from scripts.build_full_source_superposition_base import (  # noqa: E402
    MAP_DTYPE,
    MAP_SHAPE,
    SPLITS,
    infer_package_maps,
    load_package_inputs,
    read_csv_with_fieldnames,
    repo_relative,
    resolve_path,
    select_device,
    sha256_file,
    sidecar_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate full source-superposition base-map alignment.")
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--spot-check-count", default=0, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--atol",
        default=5.0e-2,
        type=float,
        help="Hard absolute tolerance for spot-check max error in K. Default reflects CUDA/CPU and batch-shape float32 variation.",
    )
    parser.add_argument(
        "--warning-atol",
        default=1.0e-2,
        type=float,
        help="Warning threshold for spot-check max error in K. Warnings are reported but do not fail validation.",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    canonical_indices = {"train": args.train_index, "val": args.val_index, "test": args.test_index}
    source_root = args.source_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not source_root.exists():
        raise SystemExit(f"missing source root: {source_root}")
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint)
    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "splits": {},
    }
    if manifest.get("source_checkpoint_sha256") != checkpoint_sha:
        errors.append("checkpoint sha256 does not match manifest")

    all_uids: dict[str, set[str]] = {}
    spot_candidates: list[tuple[str, dict[str, str], Path]] = []
    for split in SPLITS:
        canonical_fields, canonical_rows = read_csv_with_fieldnames(canonical_indices[split])
        generated_path = source_root / f"{split}_index.csv"
        generated_fields, generated_rows = read_csv_with_fieldnames(generated_path)
        split_errors: list[str] = []
        if len(canonical_rows) != len(generated_rows):
            split_errors.append(f"row count mismatch canonical={len(canonical_rows)} generated={len(generated_rows)}")
        if generated_fields[: len(canonical_fields)] != canonical_fields:
            split_errors.append("canonical field order is not preserved as a prefix")
        seen: set[str] = set()
        duplicate_uids: list[str] = []
        checked_maps = 0
        for index, (canonical, generated) in enumerate(zip(canonical_rows, generated_rows, strict=False)):
            uid = generated.get("sample_uid", "")
            if uid in seen:
                duplicate_uids.append(uid)
            seen.add(uid)
            if canonical.get("sample_uid") != uid:
                split_errors.append(f"row {index} sample_uid mismatch {canonical.get('sample_uid')} != {uid}")
                continue
            for field in canonical_fields:
                if canonical.get(field, "") != generated.get(field, ""):
                    split_errors.append(f"row {index} canonical field changed: {field}")
                    break
            map_value = generated.get("source_superposition_base_path")
            if not map_value:
                split_errors.append(f"row {index} missing source_superposition_base_path")
                continue
            map_path = resolve_path(map_value, source_root)
            try:
                validate_map_file(map_path)
                validate_sidecar(map_path, generated, checkpoint_sha)
                checked_maps += 1
            except Exception as exc:
                split_errors.append(f"row {index} map validation failed: {exc}")
        if duplicate_uids:
            split_errors.append(f"duplicate sample_uids: {duplicate_uids[:10]}")
        all_uids[split] = seen
        errors.extend(f"{split}: {item}" for item in split_errors)
        report["splits"][split] = {
            "canonical_rows": len(canonical_rows),
            "generated_rows": len(generated_rows),
            "checked_maps": checked_maps,
            "duplicate_sample_uids": duplicate_uids[:50],
            "errors": split_errors[:50],
        }
        if args.spot_check_count > 0:
            spot_candidates.extend((split, row, generated_path) for row in generated_rows)

    overlap_count = 0
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = all_uids[left] & all_uids[right]
        overlap_count += len(overlap)
        if overlap:
            errors.append(f"{left}/{right} sample_uid overlap: {sorted(overlap)[:10]}")
    report["cross_split_uid_overlap_count"] = overlap_count

    if args.spot_check_count > 0 and spot_candidates:
        report["spot_checks"] = run_spot_checks(
            candidates=spot_candidates,
            checkpoint=checkpoint,
            source_batch_size=args.source_batch_size,
            device=select_device(args.device),
            count=args.spot_check_count,
            seed=args.seed,
            atol=args.atol,
            warning_atol=args.warning_atol,
        )
        report["spot_check_summary"] = summarize_spot_checks(report["spot_checks"])
        for item in report["spot_checks"]:
            if not item["ok"]:
                errors.append(f"spot check failed for {item['sample_uid']}: max_abs_diff={item['max_abs_diff']}")
    else:
        report["spot_check_summary"] = summarize_spot_checks([])

    report["ok"] = not errors
    report["error_count"] = len(errors)
    report["errors"] = errors[:100]
    (source_root / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print(f"Validation failed with {len(errors)} errors")
        for error in errors[:20]:
            print(f"  - {error}")
        return 1
    print("Source-superposition base validation passed")
    for split in SPLITS:
        item = report["splits"][split]
        print(f"{split}: rows={item['generated_rows']} maps={item['checked_maps']}")
    summary = report.get("spot_check_summary", {})
    if summary.get("count"):
        print(
            "Spot checks: "
            f"count={summary['count']} warnings={summary['warnings']} failures={summary['failures']} "
            f"max_abs={summary['max_abs_diff_K']:.6f}K "
            f"mean_abs={summary['mean_abs_diff_K']:.6f}K"
        )
    print(f"Report: {source_root / 'validation_report.json'}")
    return 0


def validate_map_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    if tuple(array.shape) != MAP_SHAPE:
        raise ValueError(f"shape {tuple(array.shape)} != {MAP_SHAPE}")
    if str(array.dtype) != MAP_DTYPE:
        raise ValueError(f"dtype {array.dtype} != {MAP_DTYPE}")
    if not np.isfinite(np.asarray(array)).all():
        raise ValueError("non-finite values")


def validate_sidecar(map_path: Path, row: dict[str, str], checkpoint_sha: str) -> None:
    metadata_path = sidecar_path(map_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sample_uid") != row.get("sample_uid"):
        raise ValueError("sidecar sample_uid mismatch")
    if metadata.get("case_id") != row.get("case_id"):
        raise ValueError("sidecar case_id mismatch")
    if metadata.get("source_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("sidecar checkpoint sha mismatch")


@torch.no_grad()
def run_spot_checks(
    *,
    candidates: list[tuple[str, dict[str, str], Path]],
    checkpoint: Path,
    source_batch_size: int,
    device: torch.device,
    count: int,
    seed: int,
    atol: float,
    warning_atol: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(int(count), len(candidates)))
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(payload["normalization"])
    model = build_source_response_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    results: list[dict[str, Any]] = []
    for split, row, index_path in selected:
        package = load_package_inputs(row, index_path)
        recomputed = infer_package_maps([package], model, stats, source_batch_size, device)[0]
        saved = np.load(resolve_path(row["source_superposition_base_path"], index_path.parent)).astype(np.float32, copy=False)
        abs_diff = np.abs(recomputed.astype(np.float64) - saved.astype(np.float64))
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        rmse_diff = float(np.sqrt(np.mean(abs_diff * abs_diff)))
        results.append(
            {
                "split": split,
                "sample_uid": row["sample_uid"],
                "case_id": row["case_id"],
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "rmse_diff": rmse_diff,
                "hard_atol_K": float(atol),
                "warning_atol_K": float(warning_atol),
                "warning": bool(max_abs_diff > warning_atol),
                "ok": bool(max_abs_diff <= atol),
                "source_superposition_base_path": repo_relative(resolve_path(row["source_superposition_base_path"], index_path.parent)),
            }
        )
    return results


def summarize_spot_checks(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "count": 0,
            "failures": 0,
            "warnings": 0,
            "max_abs_diff_K": None,
            "mean_abs_diff_K": None,
        }
    max_values = [float(item["max_abs_diff"]) for item in items]
    mean_values = [float(item["mean_abs_diff"]) for item in items]
    return {
        "count": len(items),
        "failures": sum(1 for item in items if not item.get("ok", False)),
        "warnings": sum(1 for item in items if item.get("warning", False)),
        "max_abs_diff_K": float(max(max_values)),
        "mean_of_max_abs_diff_K": float(np.mean(max_values)),
        "max_mean_abs_diff_K": float(max(mean_values)),
        "mean_abs_diff_K": float(np.mean(mean_values)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
