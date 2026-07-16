#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.models import build_model  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input  # noqa: E402


INDEX_PATTERNS = (
    "combined_encoded_index.csv",
    "train_index.csv",
    "val_index.csv",
    "test_index.csv",
    "_input_splits/train_index.csv",
    "_input_splits/val_index.csv",
    "_input_splits/test_index.csv",
    "sample_split_extension/train_index.csv",
    "sample_split_extension/val_index.csv",
    "sample_split_extension/test_index.csv",
    "sample_split_extension/combined_index.csv",
    "family_split_extension/train_index.csv",
    "family_split_extension/val_index.csv",
    "family_split_extension/test_index.csv",
    "family_split_extension/all_index.csv",
)
SIDECARS = (
    "feature_manifest.json",
    "context_manifest.json",
    "metadata_features.csv",
    "metadata_manifest.json",
    "graph_manifest.json",
    "README.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair source-base indices to reference 33-channel extension context tensors.")
    parser.add_argument("--source-base-root", required=True, type=Path)
    parser.add_argument("--context-graph-root", required=True, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--smoke-cases", nargs="*", default=["case11", "case17", "case19", "case20"])
    args = parser.parse_args()

    source_root = args.source_base_root.expanduser().resolve()
    context_root = args.context_graph_root.expanduser().resolve()
    context_by_uid = {row["sample_uid"]: row for row in read_rows(context_root / "combined_encoded_index.csv")}
    if not context_by_uid:
        raise SystemExit(f"context graph root has no rows: {context_root}")

    changed_files: list[dict[str, Any]] = []
    for relative in INDEX_PATTERNS:
        path = source_root / relative
        if not path.exists():
            continue
        changed, rows = repair_index(path, context_by_uid)
        changed_files.append({"path": repo_relative(path), "rows": rows, "changed": changed})
    copy_sidecars(context_root, source_root)
    report = validate(source_root, args.checkpoint, args.smoke_cases)
    report.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_base_root": repo_relative(source_root),
            "context_graph_root": repo_relative(context_root),
            "changed_files": changed_files,
        }
    )
    (source_root / "context_path_repair_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Extension source-base context path repair complete")
    for item in changed_files:
        print(f"  {item['path']}: rows={item['rows']} changed={item['changed']}")
    print(f"Rows checked: {report['row_count']}")
    print(f"Unresolved paths: {report['unresolved_path_count']}")
    print(f"X shape failures: {report['x_shape_failure_count']}")
    print(f"Model input channels: {report.get('model_input_channels')}")
    if report["errors"]:
        for error in report["errors"][:20]:
            print(f"  - {error}")
        return 2
    return 0


def repair_index(path: Path, context_by_uid: dict[str, dict[str, str]]) -> tuple[bool, int]:
    rows = read_rows(path)
    if not rows:
        return False, 0
    fieldnames = read_fieldnames(path)
    changed = False
    if "graph_path" not in fieldnames:
        fieldnames.append("graph_path")
        changed = True
    for row in rows:
        context = context_by_uid.get(row["sample_uid"])
        if context is None:
            raise SystemExit(f"{path}: {row['sample_uid']} missing from context graph root")
        for column in ("x_path", "graph_path"):
            before = row.get(column, "")
            after = context.get(column, "")
            if after and before != after:
                row[column] = after
                changed = True
        row["source_base_mode"] = "source_superposition_v1"
        if "source_base_mode" not in fieldnames:
            fieldnames.append("source_base_mode")
            changed = True
        for column in ("prediction_path", "residual_path"):
            if column not in fieldnames:
                fieldnames.append(column)
                changed = True
            row.setdefault(column, context.get(column, ""))
    if changed:
        write_rows(path, rows, fieldnames)
    return changed, len(rows)


def copy_sidecars(context_root: Path, source_root: Path) -> None:
    for name in SIDECARS:
        source = context_root / name
        if source.exists():
            shutil.copy2(source, source_root / name)


def validate(source_root: Path, checkpoint: Path | None, smoke_cases: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    unresolved: list[dict[str, str]] = []
    shape_failures = 0
    rows = read_rows(source_root / "combined_encoded_index.csv")
    for row in rows:
        for column in ("x_path", "y_path", "graph_path", "source_superposition_base_path", "source_superposition_residual_path"):
            value = row.get(column, "")
            if value and not resolve_path(value).exists():
                unresolved.append({"sample_uid": row["sample_uid"], "column": column, "value": value})
        if row.get("source_base_mode") != "source_superposition_v1":
            errors.append(f"{row['sample_uid']}: source_base_mode={row.get('source_base_mode')}")
        x_path = resolve_path(row["x_path"])
        if x_path.exists():
            x = np.load(x_path, mmap_mode="r")
            if tuple(x.shape) != (33, 64, 64):
                shape_failures += 1
    smoke = checkpoint_smoke(source_root, checkpoint, smoke_cases)
    errors.extend(smoke["errors"])
    if unresolved:
        errors.append(f"{len(unresolved)} unresolved paths")
    if shape_failures:
        errors.append(f"{shape_failures} X tensors do not have shape (33,64,64)")
    return {
        "row_count": len(rows),
        "unresolved_path_count": len(unresolved),
        "unresolved_paths": unresolved[:100],
        "x_shape_failure_count": shape_failures,
        **smoke,
        "errors": errors,
    }


def checkpoint_smoke(source_root: Path, checkpoint: Path | None, smoke_cases: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    payload: dict[str, Any] = {
        "smoke_cases": [],
        "model_input_channels": None,
        "checkpoint_forward_ok": None,
        "errors": errors,
    }
    rows = read_rows(source_root / "combined_encoded_index.csv")
    sample_rows = []
    for case_id in smoke_cases:
        match = next((row for row in rows if row.get("case_id") == case_id), None)
        if match is not None:
            sample_rows.append(match)
    payload["smoke_cases"] = [row["case_id"] for row in sample_rows]
    smoke_index = source_root / ".context_path_repair_smoke.csv"
    write_rows(smoke_index, sample_rows, list(rows[0].keys()) if rows else [])
    try:
        dataset = ChipThermDataset(smoke_index, target="residual", return_metadata=True, return_graph=True)
        batch = chiptherm_collate([dataset[index] for index in range(len(dataset))])
        stats, model_config, state_dict = checkpoint_payload(checkpoint)
        model_input = build_model_input(
            batch["x"],
            batch["physics"],
            stats,
            physics_input_mode=str((model_config or {}).get("physics_input_mode", "source_superposition_v1")),
            physics_v1=batch.get("physics_v1"),
        )
        payload["model_input_channels"] = int(model_input.shape[1])
        if int(model_input.shape[1]) != 34:
            errors.append(f"model input channels are {model_input.shape[1]}")
        if checkpoint is not None and model_config is not None and state_dict is not None:
            model = build_model(model_config)
            model.load_state_dict(state_dict)
            model.eval()
            metadata = build_metadata_input(batch.get("metadata_vector"), stats)
            with torch.no_grad():
                output = model(model_input, metadata, batch.get("graph"))
            if isinstance(output, dict):
                value = output.get("final_temperature")
                if value is None:
                    value = output.get("prediction")
            else:
                value = output
            if value is None or not torch.isfinite(value).all():
                errors.append("checkpoint forward returned non-finite or empty output")
            payload["checkpoint_forward_ok"] = not errors
    except Exception as exc:
        errors.append(f"checkpoint/data smoke failed: {exc}")
        payload["checkpoint_forward_ok"] = False
    finally:
        if smoke_index.exists():
            smoke_index.unlink()
    return payload


def checkpoint_payload(checkpoint: Path | None) -> tuple[NormalizationStats, dict[str, Any] | None, dict[str, Any] | None]:
    if checkpoint is None:
        return (
            NormalizationStats(
                schema_version=1,
                power_density_mean=0.0,
                power_density_std=1.0,
                physics_mean=0.0,
                physics_std=1.0,
                residual_mean=0.0,
                residual_std=1.0,
                num_samples=1,
                num_grid_cells=4096,
                input_channels=33,
                context_channel_indices=tuple(range(8, 33)),
                context_channel_means=tuple(0.0 for _ in range(25)),
                context_channel_stds=tuple(1.0 for _ in range(25)),
            ),
            None,
            None,
        )
    payload = torch.load(checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    stats_data = dict(payload["normalization"])
    for key in (
        "context_channel_indices",
        "context_channel_names",
        "context_channel_means",
        "context_channel_stds",
        "metadata_feature_names",
        "metadata_means",
        "metadata_stds",
    ):
        if key in stats_data:
            stats_data[key] = tuple(stats_data[key])
    return NormalizationStats(**stats_data), dict(payload["model_config"]), payload["model_state_dict"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return next(csv.reader(fp))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


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
