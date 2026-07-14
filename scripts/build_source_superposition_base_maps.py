#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
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

from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponseDataset,
    SourceResponseNormalizationStats,
    normalize_source_input,
    source_response_collate,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, predict_source_rise  # noqa: E402


SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute learned source-superposition package base maps.")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--canonical-train-index", required=True, type=Path)
    parser.add_argument("--canonical-val-index", required=True, type=Path)
    parser.add_argument("--canonical-test-index", required=True, type=Path)
    parser.add_argument("--source-train-index", required=True, type=Path)
    parser.add_argument("--source-val-index", required=True, type=Path)
    parser.add_argument("--source-test-index", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    canonical_indices = {
        "train": args.canonical_train_index,
        "val": args.canonical_val_index,
        "test": args.canonical_test_index,
    }
    source_indices = {
        "train": args.source_train_index,
        "val": args.source_val_index,
        "test": args.source_test_index,
    }
    coverage = coverage_report(canonical_indices, source_indices)
    print_coverage(coverage)
    if args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "coverage_report.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    missing_total = sum(len(coverage["splits"][split]["missing_sample_uids"]) for split in SPLITS)
    if missing_total:
        raise SystemExit(f"source-response coverage incomplete for {missing_total} canonical samples; run --dry-run for details")
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(checkpoint["normalization"])
    model = build_source_response_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_model_config": checkpoint.get("model_config", {}),
        "normalization": stats.to_dict(),
        "base_definition": "source_superposition_base = ambient_K + sum_i source_power_i * source_model(source_i)",
        "coverage": coverage,
        "splits": {},
    }
    for split in SPLITS:
        split_start = time.perf_counter()
        canonical_rows = read_rows(canonical_indices[split])
        canonical_by_uid = {row["sample_uid"]: row for row in canonical_rows}
        predictions = reconstruct_source_base_maps(model, stats, source_indices[split], args.batch_size, args.num_workers, device)
        matched_uids = sorted(set(canonical_by_uid).intersection(predictions))
        source_rows: list[dict[str, Any]] = []
        physics_rows: list[dict[str, Any]] = []
        for uid in matched_uids:
            row = canonical_by_uid[uid]
            pred = predictions[uid]
            source_base_path = out_root / "maps" / split / row["case_id"] / f"{uid}_source_superposition_base.npy"
            residual_path = out_root / "residuals" / split / row["case_id"] / f"{uid}_source_superposition_residual.npy"
            source_base_path.parent.mkdir(parents=True, exist_ok=True)
            residual_path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite or not (args.resume and source_base_path.exists() and residual_path.exists()):
                target = np.load(resolve_path(row["y_path"])).astype(np.float32, copy=False)
                np.save(source_base_path, pred["temperature"].astype(np.float32, copy=False))
                np.save(residual_path, (target - pred["temperature"]).astype(np.float32, copy=False))
            source_row = dict(row)
            source_row["prediction_path"] = repo_relative(source_base_path)
            source_row["residual_path"] = repo_relative(residual_path)
            source_row["source_base_path"] = repo_relative(source_base_path)
            source_row["source_base_mode"] = "source_superposition_v1"
            source_row["source_checkpoint"] = str(args.source_checkpoint)
            source_row["source_count"] = pred["num_sources"]
            source_row["layout_path"] = pred["layout_path"]
            source_row["full_source_coverage"] = "1"
            source_row["physics_runtime_s"] = ""
            source_rows.append(source_row)
            physics_row = dict(row)
            physics_row["source_base_mode"] = "physics_v1_matched"
            physics_row["layout_path"] = pred["layout_path"]
            physics_rows.append(physics_row)
        copy_static_support_files(canonical_indices[split].parent, out_root / "source_base")
        copy_static_support_files(canonical_indices[split].parent, out_root / "physics_v1_matched")
        write_csv(out_root / "source_base" / f"{split}_index.csv", source_rows)
        write_csv(out_root / "physics_v1_matched" / f"{split}_index.csv", physics_rows)
        manifest["splits"][split] = {
            "matched_packages": len(matched_uids),
            "source_rows": len(source_rows),
            "runtime_s": time.perf_counter() - split_start,
        }
    finalize_tree(out_root / "source_base", canonical_indices["train"].parent)
    finalize_tree(out_root / "physics_v1_matched", canonical_indices["train"].parent)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "coverage_report.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root, manifest)
    print(f"Source-superposition base maps written to {out_root}")
    return 0


def coverage_report(canonical_indices: dict[str, Path], source_indices: dict[str, Path]) -> dict[str, Any]:
    payload: dict[str, Any] = {"splits": {}}
    for split in SPLITS:
        canonical_rows = read_rows(canonical_indices[split])
        source_rows = read_rows(source_indices[split]) if source_indices[split].exists() else []
        groups = group_source_rows(source_rows)
        canonical_uids = [row["sample_uid"] for row in canonical_rows]
        complete = {uid for uid, rows in groups.items() if is_complete_source_group(rows)}
        covered = sorted(set(canonical_uids).intersection(complete))
        missing = sorted(set(canonical_uids).difference(complete))
        payload["splits"][split] = {
            "canonical_samples": len(canonical_uids),
            "source_packages": len(groups),
            "complete_source_packages": len(complete),
            "covered_samples": len(covered),
            "missing_samples": len(missing),
            "covered_sample_uids": covered,
            "missing_sample_uids": missing,
        }
    return payload


def print_coverage(coverage: dict[str, Any]) -> None:
    print("Source-response coverage:")
    for split in SPLITS:
        item = coverage["splits"][split]
        print(
            f"  {split}: canonical={item['canonical_samples']} "
            f"complete_source={item['complete_source_packages']} covered={item['covered_samples']} missing={item['missing_samples']}"
        )


@torch.no_grad()
def reconstruct_source_base_maps(
    model: torch.nn.Module,
    stats: SourceResponseNormalizationStats,
    source_index: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    dataset = SourceResponseDataset(source_index, power_floor_W=stats.power_floor_W)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=source_response_collate,
    )
    groups: dict[str, dict[str, Any]] = {}
    for batch in loader:
        x = normalize_source_input(batch["x"].to(device), stats)
        source_power = batch["source_power_W"].to(device)
        pred_unit = unnormalize_source_prediction(model(x), stats)
        pred_rise = predict_source_rise(pred_unit, source_power).detach().cpu().numpy()
        ambient = batch["ambient_K"].detach().cpu().numpy()
        for i, meta in enumerate(batch["metadata"]):
            uid = str(meta["original_sample_uid"])
            group = groups.setdefault(
                uid,
                {
                    "ambient_K": float(ambient[i]),
                    "sum": np.zeros_like(pred_rise[i], dtype=np.float64),
                    "num_sources": 0,
                    "num_chiplets": int(float(meta["num_chiplets"])),
                    "layout_path": str(meta["layout_path"]),
                },
            )
            group["sum"] += pred_rise[i]
            group["num_sources"] += 1
    result: dict[str, dict[str, Any]] = {}
    for uid, group in groups.items():
        if int(group["num_sources"]) != int(group["num_chiplets"]):
            continue
        result[uid] = {
            "temperature": float(group["ambient_K"]) + group["sum"],
            "num_sources": int(group["num_sources"]),
            "layout_path": str(group["layout_path"]),
        }
    return result


def group_source_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["original_sample_uid"]].append(row)
    return groups


def is_complete_source_group(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    expected = int(float(rows[0]["num_chiplets"]))
    return len(rows) == expected


def finalize_tree(root: Path, source_parent: Path) -> None:
    all_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = root / f"{split}_index.csv"
        if path.exists():
            all_rows.extend(read_rows(path))
    write_csv(root / "combined_encoded_index.csv", all_rows)
    with (root / "combined_encoded_index.jsonl").open("w", encoding="utf-8") as fp:
        for row in all_rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    copy_metadata_features(source_parent, root, {str(row["sample_uid"]) for row in all_rows})


def copy_static_support_files(source_parent: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("feature_manifest.json", "context_manifest.json", "metadata_manifest.json", "graph_manifest.json", "README.md"):
        src = source_parent / name
        if src.exists() and not (dest / name).exists():
            shutil.copyfile(src, dest / name)


def copy_metadata_features(source_parent: Path, dest: Path, sample_uids: set[str]) -> None:
    metadata = source_parent / "metadata_features.csv"
    if metadata.exists():
        with metadata.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        filtered = [row for row in rows if row.get("sample_uid") in sample_uids]
        write_csv(dest / "metadata_features.csv", filtered, fieldnames=fields)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    text = f"""# Source Superposition Base v1

Precomputed learned source-superposition absolute-temperature base maps.

- Source checkpoint: `{manifest['source_checkpoint']}`
- Base definition: `{manifest['base_definition']}`
- Matched source-base indices: `source_base/`
- Matched physics-v1 indices: `physics_v1_matched/`
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
