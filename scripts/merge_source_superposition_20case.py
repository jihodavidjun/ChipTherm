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
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.models import build_model  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input  # noqa: E402


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
REQUIRED_CACHED_PATH_COLUMNS = (
    "x_path",
    "y_path",
    "graph_path",
    "source_superposition_base_path",
)
LIVE_REQUIRED_COLUMNS = (
    "layout_path",
    "power_path",
    "package_path",
    "source_dir",
)
OPTIONAL_LEGACY_COLUMNS = (
    "prediction_path",
    "residual_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "source_dir",
    "source_layout_path",
    "source_power_path",
    "source_package_path",
    "source_hotspot_path",
)
CANONICAL_GENERAL_COLUMNS = (
    "original_sample_uid",
    "dataset_source",
    "x_path",
    "graph_path",
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
)
MAP_SHAPE = (64, 64)
MAP_DTYPE = np.float32
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_gnn_seed1/checkpoints/best.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained cached-training 20-case source-superposition dataset."
    )
    parser.add_argument("--original-source-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full", type=Path)
    parser.add_argument(
        "--original-canonical-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power",
        type=Path,
        help="Retained case01-case10 graph/context dataset used for x/graph/scalar/metadata sidecars.",
    )
    parser.add_argument("--extension-source-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_extension", type=Path)
    parser.add_argument("--split-root", default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/indices", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/derived/source_superposition_base_v1_20case", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="Audit reconstructability and storage without writing merged indices.")
    parser.add_argument("--reconstruct-targets-only", action="store_true", help="Only reconstruct missing original targets, then exit.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an already-built output tree.")
    parser.add_argument("--overwrite-targets", action="store_true", help="Overwrite reconstructed target maps.")
    parser.add_argument("--skip-checkpoint-smoke", action="store_true")
    parser.add_argument("--full-tensor-check", action="store_true", help="Check finiteness of all required tensors, not just representative rows.")
    args = parser.parse_args()

    original_root = args.original_source_root.expanduser().resolve()
    original_canonical_root = args.original_canonical_root.expanduser().resolve()
    extension_root = args.extension_source_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    original_source_rows = read_all_source_rows(original_root)
    original_canonical_rows = read_all_source_rows(original_canonical_root)
    original_rows, canonical_report = build_original_cached_rows(
        original_source_rows,
        original_canonical_rows,
        out_root,
        overwrite_targets=args.overwrite_targets,
        write_targets=not args.preflight_only and not args.validate_only,
    )
    extension_rows = build_extension_cached_rows(read_all_source_rows(extension_root))
    preflight = preflight_original_rows(original_rows)
    print_preflight(preflight)
    if preflight["unreconstructable_count"]:
        raise SystemExit("original preflight failed: unreconstructable rows remain")
    if args.preflight_only:
        return 0
    if args.reconstruct_targets_only:
        write_json(out_root / "target_reconstruction_report.json", preflight)
        return 0

    by_uid = build_unique_uid_map([*original_rows, *extension_rows])
    if not args.validate_only:
        out_root.mkdir(parents=True, exist_ok=True)
        sidecar_report = write_merged_sidecars(
            out_root=out_root,
            roots=[original_root, original_canonical_root, extension_root],
            sample_uids=sorted(by_uid),
        )
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
    else:
        sidecar_report = sidecar_status(out_root)

    report = validate_outputs(
        out_root,
        by_uid=by_uid,
        checkpoint=checkpoint,
        skip_checkpoint_smoke=args.skip_checkpoint_smoke,
        full_tensor_check=args.full_tensor_check,
    )
    report.update(
        {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_source_root": repo_relative(original_root),
            "original_canonical_root": repo_relative(original_canonical_root),
            "extension_source_root": repo_relative(extension_root),
            "split_root": repo_relative(split_root),
            "out_root": repo_relative(out_root),
            "checkpoint": repo_relative(checkpoint),
            "residual_semantics": "source_superposition_residual_path stores target_y_K - source_superposition_base_K; reconstructed_y = base + residual",
            "canonical_repair": canonical_report,
            "original_preflight": preflight,
            "sidecars": sidecar_report,
            "storage": storage_report(out_root),
        }
    )
    if not args.validate_only:
        write_json(out_root / "merge_manifest.json", report)
        write_json(out_root / "split_manifest.json", report["protocols"])
        write_json(out_root / "compatibility_report.json", report)
        write_report_md(out_root / "compatibility_report.md", report)

    print("Source-superposition 20-case cached-training dataset validation")
    print(f"Output: {out_root}")
    for protocol, item in sorted(report["protocols"].items()):
        counts = item.get("split_counts", {})
        print(f"{protocol}: train={counts.get('train', 0)} val={counts.get('val', 0)} test={counts.get('test', 0)}")
    print(f"Cached-training unresolved required paths: {report['cached_training_unresolved_required_count']}")
    print(f"Shape failures: {report['shape_failure_count']}")
    print(f"Non-finite tensors: {report['nonfinite_tensor_count']}")
    print(f"Live-integrated unavailable rows: {report['live_integrated_unavailable_count']}")
    print(f"Reconstructed target storage: {report['storage']['reconstructed_target_MB']:.3f} MB")
    print(f"Merged-index storage: {report['storage']['index_MB']:.3f} MB")
    if report["errors"]:
        for error in report["errors"][:30]:
            print(f"  - {error}")
        return 2
    print("Cached-training validation passed")
    return 0


def build_original_cached_rows(
    source_rows: list[dict[str, str]],
    canonical_rows: list[dict[str, str]],
    out_root: Path,
    *,
    overwrite_targets: bool,
    write_targets: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    canonical_by_uid = index_rows_by_uid(canonical_rows, label="original canonical dataset")
    missing: list[str] = []
    repaired: list[dict[str, str]] = []
    changed_columns: Counter[str] = Counter()
    reconstruction_stats = ReconstructionStats()
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
        target_path = reconstructed_target_path(out_root, out)
        if write_targets:
            stats = reconstruct_target(row, target_path, overwrite=overwrite_targets)
            reconstruction_stats.add(stats)
        else:
            reconstruction_stats.add(audit_reconstruction(row, target_path))
        out["y_path"] = repo_relative(target_path)
        out["prediction_path"] = ""
        out["residual_path"] = ""
        for column in ("layout_path", "power_path", "package_path", "hotspot_path", "source_dir"):
            out[column] = ""
        for column in ("source_layout_path", "source_power_path", "source_package_path", "source_hotspot_path"):
            out[column] = ""
        out["source_base_mode"] = "source_superposition_v1"
        out["cached_training_ready"] = "1"
        out["live_integrated_inference_ready"] = "0"
        out["reconstructed_target"] = "1"
        repaired.append(out)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise SystemExit(f"{len(missing)} original source-base UID(s) missing canonical match: {preview}")
    report = {
        "original_source_rows": len(source_rows),
        "canonical_rows": len(canonical_rows),
        "repaired_rows": len(repaired),
        "changed_columns": dict(sorted(changed_columns.items())),
        "reconstruction": reconstruction_stats.to_dict(),
    }
    return repaired, report


def build_extension_cached_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        out["source_base_mode"] = out.get("source_base_mode") or "source_superposition_v1"
        out["cached_training_ready"] = "1"
        out["live_integrated_inference_ready"] = "1"
        out["reconstructed_target"] = "0"
        out.setdefault("prediction_path", "")
        out.setdefault("residual_path", "")
        out_rows.append(out)
    return out_rows


def preflight_original_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row.get("sample_uid", "") for row in rows)
    duplicate_count = sum(1 for uid, count in counts.items() if uid and count > 1)
    valid_x = valid_graph = valid_base = valid_residual = reconstructable = 0
    missing: list[dict[str, str]] = []
    for row in rows:
        checks = {
            "x_path": row.get("x_path", ""),
            "graph_path": row.get("graph_path", ""),
            "source_superposition_base_path": row.get("source_superposition_base_path", ""),
            "source_superposition_residual_path": row.get("source_superposition_residual_path", ""),
        }
        resolved = {name: resolve_path(value) for name, value in checks.items() if value}
        if resolved.get("x_path", Path()).exists():
            valid_x += 1
        else:
            missing.append({"sample_uid": row["sample_uid"], "missing": "x_path", "path": checks["x_path"]})
        if resolved.get("graph_path", Path()).exists():
            valid_graph += 1
        else:
            missing.append({"sample_uid": row["sample_uid"], "missing": "graph_path", "path": checks["graph_path"]})
        if resolved.get("source_superposition_base_path", Path()).exists():
            valid_base += 1
        else:
            missing.append({"sample_uid": row["sample_uid"], "missing": "source_superposition_base_path", "path": checks["source_superposition_base_path"]})
        if resolved.get("source_superposition_residual_path", Path()).exists():
            valid_residual += 1
        else:
            missing.append({"sample_uid": row["sample_uid"], "missing": "source_superposition_residual_path", "path": checks["source_superposition_residual_path"]})
        if (
            resolved.get("source_superposition_base_path", Path()).exists()
            and resolved.get("source_superposition_residual_path", Path()).exists()
        ):
            reconstructable += 1
    total = len(rows)
    bytes_needed = reconstructable * MAP_SHAPE[0] * MAP_SHAPE[1] * 4
    return {
        "total_rows": total,
        "valid_x_count": valid_x,
        "valid_graph_count": valid_graph,
        "valid_source_base_count": valid_base,
        "valid_source_residual_count": valid_residual,
        "reconstructable_y_count": reconstructable,
        "unreconstructable_count": total - reconstructable,
        "duplicate_uid_count": duplicate_count,
        "missing_required": missing[:50],
        "estimated_reconstructed_target_bytes": bytes_needed,
        "estimated_reconstructed_target_MB": bytes_needed / 1.0e6,
    }


def print_preflight(report: dict[str, Any]) -> None:
    print("Original case01-case10 cached-training preflight:")
    print(f"  total rows: {report['total_rows']}")
    print(f"  valid X count: {report['valid_x_count']}")
    print(f"  valid graph count: {report['valid_graph_count']}")
    print(f"  valid source-base count: {report['valid_source_base_count']}")
    print(f"  valid source-residual count: {report['valid_source_residual_count']}")
    print(f"  reconstructable Y count: {report['reconstructable_y_count']}")
    print(f"  unreconstructable count: {report['unreconstructable_count']}")
    print(f"  duplicate UID count: {report['duplicate_uid_count']}")
    print(f"  estimated reconstructed-target storage: {report['estimated_reconstructed_target_MB']:.3f} MB")


def reconstruct_target(row: dict[str, str], out_path: Path, *, overwrite: bool) -> dict[str, Any]:
    if out_path.exists() and not overwrite:
        return validate_reconstructed_target(row, out_path, status="reused")
    base = load_required_map(row, "source_superposition_base_path")
    residual = load_required_map(row, "source_superposition_residual_path")
    target = (base.astype(np.float32, copy=False) + residual.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    validate_temperature_array(target, f"reconstructed target for {row['sample_uid']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp.npy")
    np.save(tmp_path, target)
    tmp_path.replace(out_path)
    return validate_reconstructed_target(row, out_path, status="written")


def audit_reconstruction(row: dict[str, str], out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        return validate_reconstructed_target(row, out_path, status="existing")
    base_exists = bool(row.get("source_superposition_base_path")) and resolve_path(row["source_superposition_base_path"]).exists()
    residual_exists = bool(row.get("source_superposition_residual_path")) and resolve_path(row["source_superposition_residual_path"]).exists()
    return {"status": "pending", "target_exists": False, "base_exists": base_exists, "residual_exists": residual_exists}


def validate_reconstructed_target(row: dict[str, str], out_path: Path, *, status: str) -> dict[str, Any]:
    target = load_map(out_path, label=f"reconstructed target {out_path}")
    base = load_required_map(row, "source_superposition_base_path")
    residual = load_required_map(row, "source_superposition_residual_path")
    diff = target.astype(np.float64) - base.astype(np.float64) - residual.astype(np.float64)
    stats = {
        "status": status,
        "target_exists": True,
        "algebraic_max_abs_diff_K": float(np.max(np.abs(diff))),
        "algebraic_mean_abs_diff_K": float(np.mean(np.abs(diff))),
        "algebraic_rmse_diff_K": float(np.sqrt(np.mean(diff * diff))),
    }
    original_y = row.get("y_path", "")
    if original_y and resolve_path(original_y).exists():
        original = load_map(resolve_path(original_y), label=f"original target {original_y}")
        ydiff = target.astype(np.float64) - original.astype(np.float64)
        stats.update(
            {
                "original_y_exists": True,
                "original_y_max_abs_diff_K": float(np.max(np.abs(ydiff))),
                "original_y_mean_abs_diff_K": float(np.mean(np.abs(ydiff))),
                "original_y_rmse_diff_K": float(np.sqrt(np.mean(ydiff * ydiff))),
            }
        )
    else:
        stats["original_y_exists"] = False
    return stats


class ReconstructionStats:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.max_algebraic_diff = 0.0
        self.max_original_diff = 0.0
        self.original_comparisons = 0

    def add(self, stats: dict[str, Any]) -> None:
        self.counts[str(stats.get("status", "unknown"))] += 1
        self.max_algebraic_diff = max(self.max_algebraic_diff, float(stats.get("algebraic_max_abs_diff_K", 0.0) or 0.0))
        if stats.get("original_y_exists"):
            self.original_comparisons += 1
            self.max_original_diff = max(self.max_original_diff, float(stats.get("original_y_max_abs_diff_K", 0.0) or 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_counts": dict(sorted(self.counts.items())),
            "max_algebraic_abs_diff_K": self.max_algebraic_diff,
            "original_y_comparison_count": self.original_comparisons,
            "max_original_y_abs_diff_K": self.max_original_diff,
        }


def reconstructed_target_path(out_root: Path, row: dict[str, str]) -> Path:
    return out_root / "reconstructed_targets" / row.get("split", "") / row["case_id"] / f"{row['sample_uid']}_y.npy"


def validate_outputs(
    root: Path,
    *,
    by_uid: dict[str, dict[str, str]],
    checkpoint: Path,
    skip_checkpoint_smoke: bool,
    full_tensor_check: bool,
) -> dict[str, Any]:
    protocols: dict[str, Any] = {}
    errors: list[str] = []
    unresolved: list[dict[str, str]] = []
    live_unavailable: list[dict[str, str]] = []
    shape_failures: list[dict[str, str]] = []
    nonfinite: list[dict[str, str]] = []
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
                unresolved.extend(validate_cached_required_paths(protocol, split, row))
                live_unavailable.extend(validate_live_required_paths(protocol, split, row))
                if row.get("source_base_mode") != "source_superposition_v1":
                    errors.append(f"{protocol}/{split}/{row['sample_uid']}: source_base_mode is not source_superposition_v1")
                if full_tensor_check:
                    tensor_status = validate_tensors(protocol, split, row)
                    shape_failures.extend(tensor_status["shape_failures"])
                    nonfinite.extend(tensor_status["nonfinite"])
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

    smoke_report = loader_smoke(root, checkpoint=checkpoint, skip_checkpoint_smoke=skip_checkpoint_smoke)
    errors.extend(smoke_report["errors"])
    if unresolved:
        errors.append(f"cached-training required unresolved paths: {len(unresolved)}")
    if shape_failures:
        errors.append(f"shape failures: {len(shape_failures)}")
    if nonfinite:
        errors.append(f"non-finite tensors: {len(nonfinite)}")
    return {
        "protocols": protocols,
        "source_row_count": len(by_uid),
        "cached_training_unresolved_required_count": len(unresolved),
        "cached_training_unresolved_required": unresolved[:100],
        "live_integrated_unavailable_count": len(live_unavailable),
        "live_integrated_unavailable": live_unavailable[:100],
        "shape_failure_count": len(shape_failures),
        "shape_failures": shape_failures[:100],
        "nonfinite_tensor_count": len(nonfinite),
        "nonfinite_tensors": nonfinite[:100],
        "loader_smoke": smoke_report,
        "errors": errors,
    }


def validate_cached_required_paths(protocol: str, split: str, row: dict[str, str]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for column in REQUIRED_CACHED_PATH_COLUMNS:
        value = row.get(column, "")
        if not value:
            failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "value": ""})
            continue
        path = resolve_path(value)
        if not path.exists():
            failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "value": value})
    return failures


def validate_live_required_paths(protocol: str, split: str, row: dict[str, str]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for column in LIVE_REQUIRED_COLUMNS:
        value = row.get(column, "")
        if not value or not resolve_path(value).exists():
            failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "value": value})
    return failures


def validate_tensors(protocol: str, split: str, row: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    shape_failures: list[dict[str, str]] = []
    nonfinite: list[dict[str, str]] = []
    checks = {
        "x_path": ((33, 64, 64), row.get("x_path", "")),
        "y_path": ((64, 64), row.get("y_path", "")),
        "source_superposition_base_path": ((64, 64), row.get("source_superposition_base_path", "")),
    }
    for column, (expected_shape, value) in checks.items():
        if not value or not resolve_path(value).exists():
            continue
        try:
            arr = np.load(resolve_path(value), mmap_mode="r")
            if tuple(arr.shape) != expected_shape:
                shape_failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "shape": str(tuple(arr.shape))})
            if not np.isfinite(np.asarray(arr)).all():
                nonfinite.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column})
        except Exception as exc:
            shape_failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": column, "error": str(exc)})
    graph_value = row.get("graph_path", "")
    if graph_value and resolve_path(graph_value).exists():
        try:
            with np.load(resolve_path(graph_value)) as data:
                node = data["node_features"]
                edge = data["edge_features"]
                if node.ndim != 2 or node.shape[1] != 24:
                    shape_failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": "graph.node_features", "shape": str(tuple(node.shape))})
                if edge.ndim != 2 or edge.shape[1] != 15:
                    shape_failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": "graph.edge_features", "shape": str(tuple(edge.shape))})
                if not np.isfinite(node).all():
                    nonfinite.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": "graph.node_features"})
                if not np.isfinite(edge).all():
                    nonfinite.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": "graph.edge_features"})
        except Exception as exc:
            shape_failures.append({"protocol": protocol, "split": split, "sample_uid": row["sample_uid"], "column": "graph_path", "error": str(exc)})
    return {"shape_failures": shape_failures, "nonfinite": nonfinite}


def loader_smoke(root: Path, *, checkpoint: Path, skip_checkpoint_smoke: bool) -> dict[str, Any]:
    errors: list[str] = []
    successes: list[str] = []
    candidates = [
        root / "sample_split_20case" / "train_index.csv",
        root / "family_split_20case" / "val_index.csv",
        root / "family_split_20case" / "test_index.csv",
        root / "sample_split_20case" / "test_index.csv",
    ]
    wanted = {"case01", "case10", "case11", "case17", "case19", "case20"}
    checked = set()
    checkpoint_payload: dict[str, Any] | None = None
    model: torch.nn.Module | None = None
    stats: NormalizationStats | None = None
    physics_input_mode = "source_superposition_v1"
    graph_enabled = False
    if not skip_checkpoint_smoke:
        if checkpoint.exists():
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(checkpoint_payload["model_config"]).eval()
            stats = NormalizationStats(**checkpoint_payload["normalization"])
            physics_input_mode = str(checkpoint_payload["model_config"].get("physics_input_mode", "source_superposition_v1"))
            graph_enabled = "graph" in str(checkpoint_payload["model_config"].get("architecture", ""))
        else:
            errors.append(f"checkpoint not found for smoke: {checkpoint}")
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
                dataset = ChipThermDataset(tmp_path, target="residual", return_metadata=True, return_graph=True)
                sample = dataset[0]
                if tuple(sample["x"].shape) != (33, 64, 64):
                    errors.append(f"{case_id}: x shape {tuple(sample['x'].shape)}")
                if tuple(sample["temperature"].shape) != (64, 64):
                    errors.append(f"{case_id}: y shape {tuple(sample['temperature'].shape)}")
                if tuple(sample["physics"].shape) != (64, 64):
                    errors.append(f"{case_id}: source base shape {tuple(sample['physics'].shape)}")
                if "metadata_vector" not in sample or tuple(sample["metadata_vector"].shape) != (15,):
                    errors.append(f"{case_id}: metadata dimension {tuple(sample.get('metadata_vector', torch.empty(0)).shape)}")
                if "graph" not in sample:
                    errors.append(f"{case_id}: graph missing")
                else:
                    graph = sample["graph"]
                    if graph["node_features"].shape[-1] != 24 or graph["edge_features"].shape[-1] != 15:
                        errors.append(f"{case_id}: graph feature dims node={tuple(graph['node_features'].shape)} edge={tuple(graph['edge_features'].shape)}")
                batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=chiptherm_collate)))
                if model is not None and stats is not None:
                    model_input = build_model_input(batch["x"], batch["physics"], stats, physics_input_mode=physics_input_mode)
                    if tuple(model_input.shape) != (1, 34, 64, 64):
                        errors.append(f"{case_id}: model input shape {tuple(model_input.shape)}")
                    metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
                    if metadata_input is None or tuple(metadata_input.shape) != (1, 15):
                        errors.append(f"{case_id}: metadata input shape {None if metadata_input is None else tuple(metadata_input.shape)}")
                    if graph_enabled:
                        output = model(model_input, metadata_input, batch.get("graph"))
                    else:
                        output = model(model_input, metadata_input)
                    output_tensor = output_tensor_from_model_output(output)
                    if output_tensor is None or output_tensor.numel() == 0 or not torch.isfinite(output_tensor).all():
                        errors.append(f"{case_id}: checkpoint forward returned non-finite or empty output")
                successes.append(case_id)
            except Exception as exc:
                errors.append(f"{case_id}: loader/checkpoint smoke failed: {exc}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            checked.add(case_id)
    missing = wanted - checked
    if missing:
        errors.append(f"loader smoke did not find cases: {sorted(missing)}")
    return {"checked_cases": sorted(successes), "errors": errors}


def output_tensor_from_model_output(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("final_temperature", "prediction", "pred_norm", "temperature"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        return next((value for value in output.values() if torch.is_tensor(value)), None)
    return None


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


def write_merged_sidecars(out_root: Path, roots: list[Path], sample_uids: list[str]) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    metadata_manifest_path = first_existing(root / "metadata_manifest.json" for root in roots)
    metadata_features_paths = [root / "metadata_features.csv" for root in roots if (root / "metadata_features.csv").exists()]
    report: dict[str, Any] = {
        "metadata_manifest": repo_relative(metadata_manifest_path) if metadata_manifest_path else "",
        "metadata_feature_tables": [repo_relative(path) for path in metadata_features_paths],
        "metadata_rows_written": 0,
        "missing_metadata_rows": [],
    }
    if metadata_manifest_path and metadata_features_paths:
        manifest = json.loads(metadata_manifest_path.read_text(encoding="utf-8"))
        active_features = [str(name) for name in manifest.get("active_features", [])]
        rows_by_uid: dict[str, dict[str, str]] = {}
        for path in metadata_features_paths:
            with path.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    rows_by_uid[row["sample_uid"]] = row
        missing = [uid for uid in sample_uids if uid not in rows_by_uid]
        report["missing_metadata_rows"] = missing[:50]
        if missing:
            raise SystemExit(f"metadata_features missing {len(missing)} merged sample_uid(s); first: {missing[:10]}")
        out_manifest = out_root / "metadata_manifest.json"
        out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_table = out_root / "metadata_features.csv"
        fieldnames = ["sample_uid", *active_features]
        with out_table.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for uid in sample_uids:
                source = rows_by_uid[uid]
                writer.writerow({"sample_uid": uid, **{name: source[name] for name in active_features}})
        report["metadata_rows_written"] = len(sample_uids)
    for name in ("graph_manifest.json", "feature_manifest.json", "context_manifest.json"):
        path = first_existing(root / name for root in roots)
        if path:
            (out_root / name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            report[name] = repo_relative(path)
    return report


def sidecar_status(out_root: Path) -> dict[str, Any]:
    return {name: (out_root / name).exists() for name in ("metadata_manifest.json", "metadata_features.csv", "graph_manifest.json")}


def build_unique_uid_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    counts = Counter(row.get("sample_uid", "") for row in rows)
    duplicates = sorted(uid for uid, count in counts.items() if uid and count != 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise SystemExit(f"merged source-base rows contain duplicate sample_uid entries: {preview}")
    missing_uid_count = counts.get("", 0)
    if missing_uid_count:
        raise SystemExit(f"merged source-base rows contain {missing_uid_count} row(s) without sample_uid")
    return {row["sample_uid"]: normalize_row_paths(row) for row in rows}


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
    path_columns = set(REQUIRED_CACHED_PATH_COLUMNS) | set(LIVE_REQUIRED_COLUMNS) | {
        "source_superposition_residual_path",
        "prediction_path",
        "residual_path",
        "hotspot_path",
        "source_layout_path",
        "source_power_path",
        "source_package_path",
        "source_hotspot_path",
        "original_temp_path",
        "temp_layer0_path",
    }
    for column in path_columns:
        value = out.get(column, "")
        if value:
            out[column] = repo_relative(resolve_path(value))
    return out


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def first_existing(paths: Any) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_required_map(row: dict[str, str], column: str) -> np.ndarray:
    value = row.get(column, "")
    if not value:
        raise ValueError(f"{row.get('sample_uid')} missing {column}")
    return load_map(resolve_path(value), label=f"{column} for {row.get('sample_uid')}")


def load_map(path: Path, *, label: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    validate_temperature_array(arr, label)
    return arr


def validate_temperature_array(arr: np.ndarray, label: str) -> None:
    if tuple(arr.shape) != MAP_SHAPE:
        raise ValueError(f"{label} shape {arr.shape}, expected {MAP_SHAPE}")
    if str(arr.dtype) != "float32":
        arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains non-finite values")
    if float(np.min(arr)) < -1000.0 or float(np.max(arr)) > 2000.0:
        raise ValueError(f"{label} has implausible Kelvin range [{float(np.min(arr))}, {float(np.max(arr))}]")


def storage_report(out_root: Path) -> dict[str, float]:
    reconstructed = sum(path.stat().st_size for path in (out_root / "reconstructed_targets").rglob("*.npy")) if (out_root / "reconstructed_targets").exists() else 0
    indexes = sum(path.stat().st_size for path in out_root.rglob("*.csv")) if out_root.exists() else 0
    return {
        "reconstructed_target_bytes": float(reconstructed),
        "reconstructed_target_MB": float(reconstructed) / 1.0e6,
        "index_bytes": float(indexes),
        "index_MB": float(indexes) / 1.0e6,
        "temporary_MB": 0.0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Source-Superposition 20-Case Cached-Training Dataset", ""]
    lines.append(f"Cached-training unresolved required paths: {report['cached_training_unresolved_required_count']}")
    lines.append(f"Live-integrated unavailable rows: {report['live_integrated_unavailable_count']}")
    lines.append(f"Shape failures: {report['shape_failure_count']}")
    lines.append(f"Non-finite tensors: {report['nonfinite_tensor_count']}")
    lines.extend(["", "| Protocol | Train | Val | Test | Train cases | Val cases | Test cases |", "|---|---:|---:|---:|---|---|---|"])
    for protocol, item in sorted(report["protocols"].items()):
        counts = item["split_counts"]
        cases = item["split_cases"]
        lines.append(
            f"| {protocol} | {counts.get('train', 0)} | {counts.get('val', 0)} | {counts.get('test', 0)} | "
            f"{','.join(cases.get('train', []))} | {','.join(cases.get('val', []))} | {','.join(cases.get('test', []))} |"
        )
    lines.extend(["", "## Residual Semantics", "", report["residual_semantics"], "", "## Errors", ""])
    lines += [f"- {error}" for error in report["errors"]] or ["- none"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
