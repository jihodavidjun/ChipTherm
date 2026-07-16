#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats, build_model_input  # noqa: E402


EXPECTED_ORIGINAL_CONTEXT = REPO_ROOT / "data/runs/derived/source_superposition_base_v1_full"
EXPECTED_CHANNEL_COUNT = 33


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training-compatible 33-channel extension context tensors and graphs.")
    parser.add_argument(
        "--source-root",
        default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/package_plus_power_graph",
        type=Path,
        help="Extension 13-channel canonical graph artifact root.",
    )
    parser.add_argument(
        "--out-root",
        default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/package_plus_power_context_graph",
        type=Path,
        help="Final 33-channel graph artifact root.",
    )
    parser.add_argument(
        "--work-root",
        default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts",
        type=Path,
        help="Parent used for intermediate finite-source and impedance roots.",
    )
    parser.add_argument("--checkpoint", default=None, type=Path, help="Optional package checkpoint for a one-batch forward smoke.")
    parser.add_argument("--max-samples", default=None, type=int, help="Forwarded only to internal audit, not to full feature builders.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Only run compatibility validation against existing outputs.")
    parser.add_argument("--overwrite-graphs", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    finite_root = work_root / "package_plus_power_context_finite"
    impedance_root = work_root / "package_plus_power_context"

    commands = [
        [
            "python3",
            "scripts/build_finite_source_feature_dataset.py",
            "--source-root",
            str(source_root),
            "--out-root",
            str(finite_root),
            "--length-scales-mm",
            "0.5",
            "1.0",
            "2.0",
            "4.0",
            "--quadrature-size",
            "4",
            "--kernel",
            "softened_green",
        ],
        [
            "python3",
            "scripts/build_thermal_impedance_feature_dataset.py",
            "--source-root",
            str(finite_root),
            "--out-root",
            str(impedance_root),
            "--enclosed-power-radii-mm",
            "2",
            "4",
            "8",
            "16",
            "--crowding-epsilon-mm",
            "1.0",
            "--quadrature-size",
            "4",
        ],
        ["python3", "scripts/build_metadata_features.py", "--dataset-root", str(impedance_root)],
        [
            "python3",
            "scripts/build_graph_features.py",
            "--source-root",
            str(impedance_root),
            "--out-root",
            str(out_root),
            *(["--overwrite"] if args.overwrite_graphs else []),
        ],
    ]
    if not args.skip_build:
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
    if args.dry_run:
        return 0

    report = validate_context_root(out_root, checkpoint=args.checkpoint, max_samples=args.max_samples)
    (out_root / "context_compatibility_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report_md(out_root / "context_compatibility_report.md", report)
    print("Extension context compatibility validation complete")
    print(f"Rows: {report['row_count']}")
    print(f"X shape failures: {report['x_shape_failure_count']}")
    print(f"Unresolved paths: {report['unresolved_path_count']}")
    print(f"Model input channels: {report.get('model_input_channels')}")
    if report["errors"]:
        for error in report["errors"][:20]:
            print(f"  - {error}")
        return 2
    return 0


def validate_context_root(root: Path, *, checkpoint: Path | None, max_samples: int | None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    rows = read_rows(root / "combined_encoded_index.csv")
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    original_channels = original_channel_names()
    feature_manifest = json.loads((root / "feature_manifest.json").read_text(encoding="utf-8"))
    channel_names = [str(name) for name in feature_manifest.get("channel_names", [])]
    errors: list[str] = []
    unresolved: list[dict[str, str]] = []
    shape_failures = 0
    channel_stats = running_channel_stats(len(channel_names))
    for row in rows:
        for column in ("x_path", "y_path", "graph_path", "layout_path", "power_path", "package_path", "hotspot_path"):
            value = row.get(column, "")
            if value and not resolve_path(value).exists():
                unresolved.append({"sample_uid": row["sample_uid"], "column": column, "value": value})
        x_path = resolve_path(row["x_path"])
        if x_path.exists():
            x = np.load(x_path, mmap_mode="r")
            if tuple(x.shape) != (EXPECTED_CHANNEL_COUNT, 64, 64):
                shape_failures += 1
            else:
                update_channel_stats(channel_stats, np.asarray(x))
    if channel_names != original_channels:
        errors.append("extension channel_names do not match original source-superposition feature_manifest channel_names")
    if unresolved:
        errors.append(f"{len(unresolved)} unresolved path references")
    if shape_failures:
        errors.append(f"{shape_failures} X tensors do not have shape (33,64,64)")
    smoke = loader_and_checkpoint_smoke(root, checkpoint)
    errors.extend(smoke["errors"])
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": repo_relative(root),
        "row_count": len(rows),
        "channel_names": channel_names,
        "channel_names_match_original": channel_names == original_channels,
        "x_shape_failure_count": shape_failures,
        "unresolved_path_count": len(unresolved),
        "unresolved_paths": unresolved[:100],
        "channel_stats": summarize_channel_stats(channel_stats, channel_names),
        **smoke,
        "errors": errors,
    }


def loader_and_checkpoint_smoke(root: Path, checkpoint: Path | None) -> dict[str, Any]:
    errors: list[str] = []
    rows = read_rows(root / "combined_encoded_index.csv")
    sample_rows: list[dict[str, str]] = []
    for case_id in ("case11", "case14", "case17", "case19", "case20"):
        match = next((row for row in rows if row.get("case_id") == case_id), None)
        if match is not None:
            sample_rows.append(match)
    smoke_index = root / ".context_compatibility_smoke.csv"
    write_rows(smoke_index, sample_rows, list(rows[0].keys()) if rows else [])
    payload: dict[str, Any] = {
        "loader_smoke_cases": [row["case_id"] for row in sample_rows],
        "x_shape": None,
        "metadata_dim": None,
        "graph_node_dim": None,
        "graph_edge_dim": None,
        "model_input_channels": None,
        "checkpoint_forward_ok": None,
        "errors": errors,
    }
    try:
        dataset = ChipThermDataset(smoke_index, target="residual", return_metadata=True, return_graph=True)
        sample = dataset[0]
        payload["x_shape"] = list(sample["x"].shape)
        payload["metadata_dim"] = int(sample["metadata_vector"].shape[0]) if "metadata_vector" in sample else None
        payload["graph_node_dim"] = int(sample["graph"]["node_features"].shape[-1])
        payload["graph_edge_dim"] = int(sample["graph"]["edge_features"].shape[-1])
        if tuple(sample["x"].shape) != (EXPECTED_CHANNEL_COUNT, 64, 64):
            errors.append(f"loader x shape is {tuple(sample['x'].shape)}")
        if payload["metadata_dim"] != 15:
            errors.append(f"metadata dim is {payload['metadata_dim']}")
        if payload["graph_node_dim"] != 24 or payload["graph_edge_dim"] != 15:
            errors.append(f"graph dims are node={payload['graph_node_dim']} edge={payload['graph_edge_dim']}")
        stats = stats_from_checkpoint(checkpoint) if checkpoint else default_stats_for_smoke()
        x = sample["x"].unsqueeze(0)
        physics = sample["physics"].unsqueeze(0)
        model_input = build_model_input(x, physics, stats, physics_input_mode="source_superposition_v1")
        payload["model_input_channels"] = int(model_input.shape[1])
        if int(model_input.shape[1]) != 34:
            errors.append(f"model input channels are {model_input.shape[1]}")
    except Exception as exc:
        errors.append(f"loader/model-input smoke failed: {exc}")
    finally:
        if smoke_index.exists():
            smoke_index.unlink()
    return payload


def stats_from_checkpoint(checkpoint: Path | None) -> NormalizationStats:
    import torch

    if checkpoint is None:
        return default_stats_for_smoke()
    payload = torch.load(checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    data = dict(payload["normalization"])
    for key in (
        "context_channel_indices",
        "context_channel_names",
        "context_channel_means",
        "context_channel_stds",
        "metadata_feature_names",
        "metadata_means",
        "metadata_stds",
    ):
        if key in data:
            data[key] = tuple(data[key])
    return NormalizationStats(**data)


def default_stats_for_smoke() -> NormalizationStats:
    return NormalizationStats(
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
        context_channel_names=tuple(original_channel_names()[8:]),
        context_channel_means=tuple(0.0 for _ in range(25)),
        context_channel_stds=tuple(1.0 for _ in range(25)),
    )


def original_channel_names() -> list[str]:
    manifest = json.loads((EXPECTED_ORIGINAL_CONTEXT / "feature_manifest.json").read_text(encoding="utf-8"))
    return [str(name) for name in manifest["channel_names"]]


def running_channel_stats(count: int) -> list[dict[str, float]]:
    return [{"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")} for _ in range(count)]


def update_channel_stats(stats: list[dict[str, float]], x: np.ndarray) -> None:
    for index, item in enumerate(stats):
        data = np.asarray(x[index], dtype=np.float64)
        item["count"] += float(data.size)
        item["sum"] += float(data.sum())
        item["sum_sq"] += float((data * data).sum())
        item["min"] = min(item["min"], float(data.min()))
        item["max"] = max(item["max"], float(data.max()))


def summarize_channel_stats(stats: list[dict[str, float]], names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, item) in enumerate(zip(names, stats)):
        count = max(float(item["count"]), 1.0)
        mean = item["sum"] / count
        var = max(item["sum_sq"] / count - mean * mean, 0.0)
        rows.append({"index": index, "name": name, "mean": mean, "std": var**0.5, "min": item["min"], "max": item["max"]})
    return rows


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Extension Context Compatibility Report",
        "",
        f"Rows: `{report['row_count']}`",
        f"Channel names match original: `{report['channel_names_match_original']}`",
        f"X shape failures: `{report['x_shape_failure_count']}`",
        f"Unresolved paths: `{report['unresolved_path_count']}`",
        f"Model input channels: `{report.get('model_input_channels')}`",
        "",
        "## Errors",
        "",
    ]
    lines += [f"- {error}" for error in report.get("errors", [])] or ["- none"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
