#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.graph_models import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES


PATH_COLUMNS = (
    "x_path",
    "y_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "benchmark_path",
    "source_dir",
    "original_temp_path",
    "temp_layer0_path",
    "prediction_path",
    "residual_path",
    "source_superposition_base_path",
    "source_superposition_residual_path",
)


TYPE_GROUPS = {
    "cpu": "cpu",
    "gpu": "gpu",
    "npu": "npu",
    "hbm": "memory",
    "dram": "memory",
    "memory": "memory",
    "io": "io",
    "analog": "analog",
    "mems": "mems",
}
TYPE_FEATURES = ("cpu", "gpu", "npu", "memory", "io", "analog", "mems", "other")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact ChipTherm chiplet graph artifacts.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--softening-mm", default=1.0, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if yaml is None:
        raise SystemExit("PyYAML is required to parse power.yaml")
    if args.softening_mm <= 0.0:
        raise SystemExit("--softening-mm must be positive")

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    if not source_root.exists():
        raise SystemExit(f"source root does not exist: {source_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    combined_path = source_root / "combined_encoded_index.csv"
    if not combined_path.exists():
        split_rows = []
        for split in ("train", "val", "test"):
            split_path = source_root / f"{split}_index.csv"
            if split_path.exists():
                split_rows.extend(read_rows(split_path))
        if not split_rows:
            raise SystemExit(f"could not find combined_encoded_index.csv or split indexes under {source_root}")
        rows = split_rows
    else:
        rows = read_rows(combined_path)

    start = time.perf_counter()
    graph_dir = out_root / "graph_features"
    generated_rows: list[dict[str, str]] = []
    bytes_written = 0
    node_counts: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    for row in rows:
        graph = build_graph_for_row(row, softening_mm=float(args.softening_mm))
        case_id = row["case_id"]
        sample_uid = row["sample_uid"]
        out_path = graph_dir / case_id / f"{sample_uid}_graph.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            pass
        else:
            np.savez_compressed(out_path, **graph)
        bytes_written += out_path.stat().st_size
        node_counts[int(graph["node_features"].shape[0])] += 1
        edge_counts[int(graph["edge_features"].shape[0])] += 1
        new_row = normalize_path_columns(row, source_root)
        new_row["graph_path"] = relative_to_repo(out_path)
        generated_rows.append(new_row)

    write_indexes(out_root, generated_rows)
    copy_sidecars(source_root, out_root)
    runtime_s = time.perf_counter() - start
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "out_root": str(out_root),
        "num_samples": len(generated_rows),
        "graph_dir": relative_to_repo(graph_dir),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "type_feature_order": list(TYPE_FEATURES),
        "edge_construction": "fully connected directed graph excluding self-edges",
        "softening_mm": float(args.softening_mm),
        "node_count_histogram": {str(k): v for k, v in sorted(node_counts.items())},
        "edge_count_histogram": {str(k): v for k, v in sorted(edge_counts.items())},
        "storage_bytes": bytes_written,
        "storage_bytes_per_sample": bytes_written / max(len(generated_rows), 1),
        "generation_runtime_s": runtime_s,
        "generation_runtime_per_sample_s": runtime_s / max(len(generated_rows), 1),
        "notes": "Features are derived from source/layout.json and source/power.yaml only; no HotSpot labels are used.",
    }
    (out_root / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root, manifest)

    print("Graph feature build complete")
    print(f"Samples: {len(generated_rows)}")
    print(f"Output: {out_root}")
    print(f"Storage/sample: {manifest['storage_bytes_per_sample']:.1f} bytes")
    print(f"Runtime/sample: {manifest['generation_runtime_per_sample_s']:.6f} s")
    return 0


def build_graph_for_row(row: dict[str, str], *, softening_mm: float) -> dict[str, np.ndarray]:
    source_dir = source_dir_for_row(row)
    layout = json.loads((source_dir / "layout.json").read_text(encoding="utf-8"))
    power = load_yaml(source_dir / "power.yaml")
    package = layout.get("package", {})
    size = package.get("size", {})
    package_width = float(size["width"])
    package_height = float(size["height"])
    chiplets = layout.get("chiplets", [])
    if not chiplets:
        raise ValueError(f"{source_dir / 'layout.json'} has no chiplets")
    power_map = chiplet_power_map(power)
    total_power = sum(float(power_map.get(chiplet["name"], 0.0)) for chiplet in chiplets)
    total_area = 0.0
    parsed = []
    for chiplet in chiplets:
        name = str(chiplet["name"])
        position = chiplet["position"]
        size = chiplet["size"]
        x = float(position["x"])
        y = float(position["y"])
        width = float(size["width"])
        height = float(size["height"])
        area = width * height
        power_w = float(power_map.get(name, 0.0))
        total_area += area
        parsed.append(
            {
                "name": name,
                "type": str(chiplet.get("type", "other")),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "cx": x + 0.5 * width,
                "cy": y + 0.5 * height,
                "power": power_w,
                "pd": power_w / max(area, 1.0e-12),
                "min_edge": min(x, package_width - (x + width), y, package_height - (y + height)),
            }
        )
    nodes = []
    rects = []
    for chiplet in parsed:
        type_group = TYPE_GROUPS.get(chiplet["type"].lower(), "other")
        type_features = [1.0 if name == type_group else 0.0 for name in TYPE_FEATURES]
        nodes.append(
            [
                chiplet["cx"],
                chiplet["cy"],
                chiplet["width"],
                chiplet["height"],
                chiplet["area"],
                chiplet["width"] / max(chiplet["height"], 1.0e-12),
                chiplet["power"],
                chiplet["pd"],
                chiplet["x"],
                package_width - (chiplet["x"] + chiplet["width"]),
                chiplet["y"],
                package_height - (chiplet["y"] + chiplet["height"]),
                chiplet["cx"] / max(package_width, 1.0e-12),
                chiplet["cy"] / max(package_height, 1.0e-12),
                chiplet["power"] / max(total_power, 1.0e-12),
                chiplet["area"] / max(total_area, 1.0e-12),
                *type_features,
            ]
        )
        rects.append([chiplet["x"], chiplet["y"], chiplet["width"], chiplet["height"]])
    edges = []
    edge_features = []
    for src_index, source in enumerate(parsed):
        for dst_index, target in enumerate(parsed):
            if src_index == dst_index:
                continue
            dx = target["cx"] - source["cx"]
            dy = target["cy"] - source["cy"]
            distance = float((dx * dx + dy * dy) ** 0.5)
            softened = float((distance * distance + softening_mm * softening_mm) ** 0.5)
            edges.append([src_index, dst_index])
            edge_features.append(
                [
                    dx,
                    dy,
                    distance,
                    1.0 / softened,
                    float(np.log1p(distance)),
                    dy / max(distance, 1.0e-12),
                    dx / max(distance, 1.0e-12),
                    source["power"],
                    target["power"],
                    source["area"],
                    target["area"],
                    source["pd"],
                    target["pd"],
                    source["min_edge"],
                    target["min_edge"],
                ]
            )
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    edge_values = np.asarray(edge_features, dtype=np.float32) if edge_features else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    return {
        "node_features": np.asarray(nodes, dtype=np.float32),
        "edge_index": edge_index,
        "edge_features": edge_values,
        "chiplet_rects": np.asarray(rects, dtype=np.float32),
        "package_size": np.asarray([package_width, package_height], dtype=np.float32),
    }


def source_dir_for_row(row: dict[str, str]) -> Path:
    value = row.get("source_dir")
    if value:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        for candidate in (REPO_ROOT / path, Path.cwd() / path):
            if candidate.exists():
                return candidate
    case_id = row["case_id"]
    original = row.get("original_sample_uid") or row["sample_uid"]
    sample_name = original
    prefix = f"{case_id}_"
    if sample_name.startswith(prefix):
        sample_name = sample_name[len(prefix) :]
    return REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_name / "source"


def normalize_path_columns(row: dict[str, str], source_root: Path) -> dict[str, str]:
    out = dict(row)
    for column in PATH_COLUMNS:
        value = out.get(column)
        if not value:
            continue
        out[column] = relative_to_repo(resolve_path_value(value, source_root))
    return out


def resolve_path_value(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, base / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def chiplet_power_map(power: dict[str, Any]) -> dict[str, float]:
    workload = power.get("active_workload", "nominal")
    workloads = power.get("workloads", {})
    if workload in workloads:
        values = workloads[workload]
    else:
        values = power.get("chiplets", {})
    return {str(name): float(value) for name, value in values.items()}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_indexes(out_root: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    write_csv(out_root / "combined_encoded_index.csv", rows, fieldnames)
    with (out_root / "combined_encoded_index.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        if split_rows:
            write_csv(out_root / f"{split}_index.csv", split_rows, fieldnames)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_sidecars(source_root: Path, out_root: Path) -> None:
    for name in ("metadata_features.csv", "metadata_manifest.json", "feature_manifest.json", "context_manifest.json", "split_manifest.json"):
        source = source_root / name
        if source.exists() and source.resolve() != (out_root / name).resolve():
            shutil.copy2(source, out_root / name)


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Graph Feature Dataset

This index-only dataset reuses the original X/Y/physics/residual tensors and adds compact chiplet graph artifacts.

- Samples: {manifest['num_samples']}
- Graph construction: {manifest['edge_construction']}
- Storage/sample: {manifest['storage_bytes_per_sample']:.1f} bytes
- Generation runtime/sample: {manifest['generation_runtime_per_sample_s']:.6f} s

No HotSpot labels are used to construct graph features.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
