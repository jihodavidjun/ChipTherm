#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.graph_models import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES  # noqa: E402


TYPE_FEATURES = ("cpu", "gpu", "npu", "memory", "io", "analog", "mems", "other")
GRID_SHAPE = (64, 64)
EPS = 1.0e-12
PRIMARY_DESCRIPTOR_EXCLUDE_PREFIXES = ("target_", "hotspot_", "label_", "error_")


@dataclass(frozen=True)
class CoverageConfig:
    pairwise_hist_bins: int = 6
    edge_threshold_fraction: float = 0.05
    pca_components: int = 3
    distance_metrics: tuple[str, ...] = ("euclidean", "robust_euclidean", "mahalanobis", "cosine")
    seed: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze package-family descriptor coverage for ChipTherm splits.")
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pairwise-hist-bins", default=6, type=int)
    parser.add_argument("--edge-threshold-fraction", default=0.05, type=float)
    parser.add_argument("--pca-components", default=3, type=int)
    parser.add_argument(
        "--distance-metrics",
        nargs="+",
        default=["euclidean", "robust_euclidean", "mahalanobis", "cosine"],
        choices=["euclidean", "robust_euclidean", "mahalanobis", "cosine"],
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    if args.pairwise_hist_bins < 1:
        raise SystemExit("--pairwise-hist-bins must be >= 1")
    if args.edge_threshold_fraction < 0.0:
        raise SystemExit("--edge-threshold-fraction must be nonnegative")
    config = CoverageConfig(
        pairwise_hist_bins=int(args.pairwise_hist_bins),
        edge_threshold_fraction=float(args.edge_threshold_fraction),
        pca_components=int(args.pca_components),
        distance_metrics=tuple(args.distance_metrics),
        seed=int(args.seed),
    )
    result = analyze_coverage(
        train_index=args.train_index,
        val_index=args.val_index,
        test_index=args.test_index,
        out_dir=args.out_dir,
        config=config,
    )
    print("Package-family coverage analysis complete")
    print(f"Output: {args.out_dir}")
    print(f"Families: train={len(result['split_families']['train'])} val={len(result['split_families']['val'])} test={len(result['split_families']['test'])}")
    for item in result["nearest_training_families"]:
        if item["split"] in {"val", "test"}:
            print(
                f"{item['case_id']}: nearest={item['nearest_1_case_id']} "
                f"euclidean={float(item['euclidean_distance']):.3f} "
                f"out_of_range={item['out_of_range_count']}/{item['descriptor_count']}"
            )
    return 0


def analyze_coverage(*, train_index: Path, val_index: Path, test_index: Path, out_dir: Path, config: CoverageConfig) -> dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {
        "train": read_rows(train_index.expanduser().resolve()),
        "val": read_rows(val_index.expanduser().resolve()),
        "test": read_rows(test_index.expanduser().resolve()),
    }
    for split, rows in split_rows.items():
        if not rows:
            raise ValueError(f"{split} index has no rows")
    grouped = group_rows_by_case(split_rows)
    metadata_rows, metadata_feature_names, metadata_sources = load_metadata_sidecars([train_index, val_index, test_index])
    warnings: list[str] = []
    family_records: list[dict[str, Any]] = []
    descriptor_by_case: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        for case_id in sorted(grouped[split]):
            descriptor, family_warnings = compute_family_descriptor(
                grouped[split][case_id],
                split=split,
                config=config,
                metadata_rows=metadata_rows,
                metadata_feature_names=metadata_feature_names,
            )
            warnings.extend(f"{case_id}: {item}" for item in family_warnings)
            descriptor_by_case[case_id] = descriptor
            family_records.append({"case_id": case_id, "split": split, **descriptor})
    common_keys = set(family_records[0])
    for record in family_records[1:]:
        common_keys &= set(record)
    dropped_descriptor_keys = sorted(
        key
        for key in set().union(*(set(record) for record in family_records)) - common_keys
        if key not in {"case_id", "split"} and not key.startswith(PRIMARY_DESCRIPTOR_EXCLUDE_PREFIXES)
    )
    if dropped_descriptor_keys:
        warnings.append(
            "dropped non-rectangular descriptor columns present in only some families: "
            + ", ".join(dropped_descriptor_keys[:20])
        )
    descriptor_names = sorted(
        key
        for key in common_keys
        if key not in {"case_id", "split"} and not key.startswith(PRIMARY_DESCRIPTOR_EXCLUDE_PREFIXES)
    )
    ensure_finite_descriptors(family_records, descriptor_names)
    train_cases = sorted(grouped["train"])
    eval_cases = sorted([*grouped["val"].keys(), *grouped["test"].keys()])
    descriptor_matrix = np.asarray([[float(descriptor_by_case[case][name]) for name in descriptor_names] for case in train_cases], dtype=np.float64)
    scaler = fit_train_scaler(descriptor_matrix, descriptor_names)
    standardized_records = standardize_records(family_records, descriptor_names, scaler)
    nearest_rows, distance_rows, ood_rows = compute_distances_and_ood(
        family_records=family_records,
        standardized_records=standardized_records,
        descriptor_names=descriptor_names,
        train_cases=train_cases,
        eval_cases=eval_cases,
        scaler=scaler,
        metrics=config.distance_metrics,
    )
    pca_rows, pca_summary = compute_pca_projection(standardized_records, descriptor_names, train_cases, config.pca_components)

    write_csv(out_dir / "family_descriptors.csv", family_records)
    write_csv(out_dir / "standardized_family_descriptors.csv", standardized_records)
    write_csv(out_dir / "nearest_training_families.csv", nearest_rows)
    write_csv(out_dir / "out_of_range_features.csv", ood_rows)
    write_csv(out_dir / "distance_matrix.csv", distance_rows)
    write_csv(out_dir / "pca_projection.csv", pca_rows)
    plot_outputs(out_dir, family_records, nearest_rows, pca_rows, descriptor_names)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train_index": repo_relative(train_index.expanduser().resolve()),
            "val_index": repo_relative(val_index.expanduser().resolve()),
            "test_index": repo_relative(test_index.expanduser().resolve()),
        },
        "split_families": {split: sorted(grouped[split]) for split in ("train", "val", "test")},
        "descriptor_names": descriptor_names,
        "descriptor_count": len(descriptor_names),
        "scaler": scaler_to_json(scaler),
        "nearest_neighbors": nearest_rows,
        "out_of_range_features": ood_rows,
        "pca": pca_summary,
        "metadata_feature_names": metadata_feature_names,
        "metadata_sources": [repo_relative(path) for path in metadata_sources],
        "warnings": warnings,
        "leakage_safeguards": [
            "Primary descriptor distances exclude y_path/HotSpot target tensors and model errors.",
            "Standardization, robust scaling, Mahalanobis covariance, and PCA are fit only on training families.",
            "Graph descriptors use preexisting graph_path NPZ fields and graph feature names from chiptherm.ml.graph_models.",
            "source_superposition_base_path statistics are allowed as label-free physics/base descriptors; HotSpot Y is not read.",
        ],
        "config": {
            "pairwise_hist_bins": config.pairwise_hist_bins,
            "edge_threshold_fraction": config.edge_threshold_fraction,
            "pca_components": config.pca_components,
            "distance_metrics": list(config.distance_metrics),
            "seed": config.seed,
        },
    }
    (out_dir / "descriptor_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, summary)
    verify_output_finiteness(out_dir)
    result = dict(summary)
    result["nearest_training_families"] = nearest_rows
    return result


def compute_family_descriptor(
    rows: list[dict[str, str]],
    *,
    split: str,
    config: CoverageConfig,
    metadata_rows: dict[str, dict[str, float]],
    metadata_feature_names: list[str],
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    samples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["sample_uid"]):
        sample, sample_warnings = load_sample_descriptor_inputs(row, metadata_rows, metadata_feature_names)
        samples.append(sample)
        warnings.extend(sample_warnings)
    if not samples:
        raise ValueError(f"no samples for {split}")
    package_widths = np.asarray([sample["package_width_mm"] for sample in samples], dtype=np.float64)
    package_heights = np.asarray([sample["package_height_mm"] for sample in samples], dtype=np.float64)
    package_areas = package_widths * package_heights
    characteristic = np.sqrt(np.maximum(package_areas, EPS))
    all_nodes = np.concatenate([sample["nodes"] for sample in samples], axis=0)
    all_edges = np.concatenate([sample["edges"] for sample in samples if sample["edges"].size], axis=0) if any(sample["edges"].size for sample in samples) else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64)
    node_idx = feature_index_map(NODE_FEATURE_NAMES)
    edge_idx = feature_index_map(EDGE_FEATURE_NAMES)
    widths = all_nodes[:, node_idx["width_mm"]]
    heights = all_nodes[:, node_idx["height_mm"]]
    areas = all_nodes[:, node_idx["area_mm2"]]
    aspects = all_nodes[:, node_idx["aspect_ratio"]]
    powers = all_nodes[:, node_idx["total_power_W"]]
    densities = all_nodes[:, node_idx["power_density_W_per_mm2"]]
    min_edges_per_node = np.minimum.reduce(
        [
            all_nodes[:, node_idx["distance_to_left_edge_mm"]],
            all_nodes[:, node_idx["distance_to_right_edge_mm"]],
            all_nodes[:, node_idx["distance_to_bottom_edge_mm"]],
            all_nodes[:, node_idx["distance_to_top_edge_mm"]],
        ]
    )
    normalized_centers_x = all_nodes[:, node_idx["normalized_center_x"]]
    normalized_centers_y = all_nodes[:, node_idx["normalized_center_y"]]
    chiplet_counts = np.asarray([sample["nodes"].shape[0] for sample in samples], dtype=np.float64)
    occupied_areas = np.asarray([sample["nodes"][:, node_idx["area_mm2"]].sum() for sample in samples], dtype=np.float64)
    occupied_fractions = occupied_areas / np.maximum(package_areas, EPS)
    total_powers = np.asarray([sample["nodes"][:, node_idx["total_power_W"]].sum() for sample in samples], dtype=np.float64)
    max_powers = np.asarray([sample["nodes"][:, node_idx["total_power_W"]].max() for sample in samples], dtype=np.float64)
    mean_pds = np.asarray([sample["nodes"][:, node_idx["power_density_W_per_mm2"]].mean() for sample in samples], dtype=np.float64)
    max_pds = np.asarray([sample["nodes"][:, node_idx["power_density_W_per_mm2"]].max() for sample in samples], dtype=np.float64)
    hottest_power_fraction = max_powers / np.maximum(total_powers, EPS)
    pairwise_distances = all_edges[:, edge_idx["distance_mm"]] if all_edges.size else np.empty(0, dtype=np.float64)
    normalized_pairwise = []
    nearest = []
    normalized_nearest = []
    for sample in samples:
        edges = sample["edges"]
        char = math.sqrt(max(sample["package_width_mm"] * sample["package_height_mm"], EPS))
        distances = edges[:, edge_idx["distance_mm"]] if edges.size else np.empty(0, dtype=np.float64)
        normalized_pairwise.extend((distances / max(char, EPS)).tolist())
        nearest.extend(nearest_neighbor_distances(sample["nodes"][:, [node_idx["center_x_mm"], node_idx["center_y_mm"]]]).tolist())
        normalized_nearest.extend((nearest_neighbor_distances(sample["nodes"][:, [node_idx["center_x_mm"], node_idx["center_y_mm"]]]) / max(char, EPS)).tolist())
    normalized_pairwise_arr = np.asarray(normalized_pairwise, dtype=np.float64)
    nearest_arr = np.asarray(nearest, dtype=np.float64)
    normalized_nearest_arr = np.asarray(normalized_nearest, dtype=np.float64)
    type_counts = {name: float(all_nodes[:, node_idx[f"type_{name}"]].sum()) for name in TYPE_FEATURES}
    type_total = sum(type_counts.values()) or 1.0
    edge_thresholds = config.edge_threshold_fraction * characteristic
    near_edge_count = 0
    total_nodes = 0
    for sample, threshold in zip(samples, edge_thresholds, strict=True):
        node = sample["nodes"]
        min_edge = np.minimum.reduce(
            [
                node[:, node_idx["distance_to_left_edge_mm"]],
                node[:, node_idx["distance_to_right_edge_mm"]],
                node[:, node_idx["distance_to_bottom_edge_mm"]],
                node[:, node_idx["distance_to_top_edge_mm"]],
            ]
        )
        near_edge_count += int(np.sum(min_edge <= threshold))
        total_nodes += int(node.shape[0])
    sample_com_offsets = np.asarray([center_of_mass_offset(sample["nodes"], sample["package_width_mm"], sample["package_height_mm"], node_idx) for sample in samples], dtype=np.float64)
    base_means = np.asarray([sample["source_base_mean_K"] for sample in samples if sample["source_base_mean_K"] is not None], dtype=np.float64)
    base_peaks = np.asarray([sample["source_base_peak_K"] for sample in samples if sample["source_base_peak_K"] is not None], dtype=np.float64)
    descriptor: dict[str, float] = {
        "sample_count": float(len(samples)),
        "package_width_mm": float(np.mean(package_widths)),
        "package_height_mm": float(np.mean(package_heights)),
        "package_area_mm2": float(np.mean(package_areas)),
        "package_aspect_ratio": float(np.mean(package_widths / np.maximum(package_heights, EPS))),
        "characteristic_length_mm": float(np.mean(characteristic)),
        "cell_size_x_mm": float(np.mean(package_widths / GRID_SHAPE[1])),
        "cell_size_y_mm": float(np.mean(package_heights / GRID_SHAPE[0])),
        "chiplet_count": float(np.mean(chiplet_counts)),
        "chiplet_count_std": float(np.std(chiplet_counts)),
        "occupied_area_mm2": float(np.mean(occupied_areas)),
        "occupied_fraction": float(np.mean(occupied_fractions)),
        "whitespace_fraction": float(np.mean(1.0 - occupied_fractions)),
        "fraction_chiplets_near_edge": float(near_edge_count / max(total_nodes, 1)),
        "graph_node_count_mean": float(np.mean(chiplet_counts)),
        "graph_edge_count_mean": float(np.mean([sample["edges"].shape[0] for sample in samples])),
        "average_graph_degree": float(np.mean([sample["edges"].shape[0] / max(sample["nodes"].shape[0], 1) for sample in samples])),
        "center_x_normalized_mean": safe_mean(normalized_centers_x),
        "center_y_normalized_mean": safe_mean(normalized_centers_y),
        "center_x_normalized_std": safe_std(normalized_centers_x),
        "center_y_normalized_std": safe_std(normalized_centers_y),
        "spatial_spread_x_normalized": safe_std(normalized_centers_x),
        "spatial_spread_y_normalized": safe_std(normalized_centers_y),
        "center_of_mass_offset_normalized_mean": safe_mean(sample_com_offsets),
    }
    add_stats(descriptor, "chiplet_area_mm2", areas)
    add_stats(descriptor, "chiplet_width_mm", widths)
    add_stats(descriptor, "chiplet_height_mm", heights)
    add_stats(descriptor, "chiplet_aspect_ratio", aspects)
    add_stats(descriptor, "chiplet_to_edge_distance_mm", min_edges_per_node)
    add_stats(descriptor, "chiplet_to_edge_distance_normalized", min_edges_per_node / np.repeat(characteristic, [sample["nodes"].shape[0] for sample in samples]))
    add_stats(descriptor, "pairwise_center_distance_mm", pairwise_distances)
    add_stats(descriptor, "pairwise_center_distance_normalized", normalized_pairwise_arr)
    add_stats(descriptor, "nearest_neighbor_distance_mm", nearest_arr)
    add_stats(descriptor, "nearest_neighbor_distance_normalized", normalized_nearest_arr)
    add_stats(descriptor, "total_power_W", total_powers)
    add_stats(descriptor, "max_chiplet_power_W", max_powers)
    add_stats(descriptor, "chiplet_power_density_W_per_mm2", densities)
    add_stats(descriptor, "mean_power_density_W_per_mm2", mean_pds)
    add_stats(descriptor, "max_power_density_W_per_mm2", max_pds)
    add_stats(descriptor, "hottest_chiplet_power_fraction", hottest_power_fraction)
    add_stats(descriptor, "source_superposition_base_mean_K", base_means)
    add_stats(descriptor, "source_superposition_base_peak_K", base_peaks)
    hist = normalized_histogram(normalized_pairwise_arr, bins=config.pairwise_hist_bins, range_max=1.5)
    for index, value in enumerate(hist):
        descriptor[f"pairwise_distance_hist_bin_{index:02d}"] = float(value)
    for name, count in type_counts.items():
        descriptor[f"type_{name}_count"] = float(count / max(len(samples), 1))
        descriptor[f"type_{name}_fraction"] = float(count / type_total)
    for metadata_name in metadata_feature_names:
        values = np.asarray([sample["metadata"].get(metadata_name, np.nan) for sample in samples], dtype=np.float64)
        if np.isfinite(values).all():
            descriptor[f"metadata_{metadata_name}_mean"] = float(np.mean(values))
            if metadata_name in {
                "package_width_mm",
                "package_height_mm",
                "total_power_W",
                "chiplet_count",
                "occupied_fraction",
                "whitespace_fraction",
                "mean_power_density_W_per_mm2",
                "max_power_density_W_per_mm2",
            }:
                descriptor[f"metadata_{metadata_name}_std"] = float(np.std(values))
    return descriptor, warnings


def load_sample_descriptor_inputs(
    row: dict[str, str],
    metadata_rows: dict[str, dict[str, float]],
    metadata_feature_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    graph_value = row.get("graph_path", "")
    if not graph_value:
        raise ValueError(f"{row.get('sample_uid')} missing graph_path")
    graph_path = resolve_path(graph_value)
    if not graph_path.exists():
        fallback = find_alternate_graph_path(row)
        if fallback is None:
            raise FileNotFoundError(f"{row.get('sample_uid')} graph_path unresolved: {graph_value}")
        warnings.append(f"{row.get('sample_uid')} graph_path fallback used: {graph_value} -> {repo_relative(fallback)}")
        graph_path = fallback
    with np.load(graph_path) as data:
        nodes = np.asarray(data["node_features"], dtype=np.float64)
        edges = np.asarray(data["edge_features"], dtype=np.float64)
        package_size = np.asarray(data["package_size"], dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] < len(NODE_FEATURE_NAMES):
        raise ValueError(f"{graph_path} node_features shape {nodes.shape} incompatible with known node features")
    if edges.ndim != 2 or edges.shape[1] < len(EDGE_FEATURE_NAMES):
        raise ValueError(f"{graph_path} edge_features shape {edges.shape} incompatible with known edge features")
    if package_size.shape[0] < 2:
        raise ValueError(f"{graph_path} package_size must have two entries")
    if not np.isfinite(nodes).all() or not np.isfinite(edges).all() or not np.isfinite(package_size).all():
        raise ValueError(f"{graph_path} contains non-finite graph descriptors")
    source_base_mean = None
    source_base_peak = None
    base_value = row.get("source_superposition_base_path", "")
    if base_value:
        base_path = resolve_path(base_value)
        if base_path.exists():
            base = np.load(base_path, mmap_mode="r")
            if tuple(base.shape) == GRID_SHAPE and np.isfinite(np.asarray(base)).all():
                source_base_mean = float(np.mean(base))
                source_base_peak = float(np.max(base))
            else:
                warnings.append(f"{row.get('sample_uid')} source base has invalid shape or non-finite values")
        else:
            warnings.append(f"{row.get('sample_uid')} source base path missing: {base_value}")
    metadata = metadata_rows.get(row["sample_uid"], {})
    if metadata_feature_names and not metadata:
        warnings.append(f"{row.get('sample_uid')} missing metadata sidecar row")
    return {
        "sample_uid": row["sample_uid"],
        "case_id": row["case_id"],
        "package_width_mm": float(package_size[0]),
        "package_height_mm": float(package_size[1]),
        "nodes": nodes[:, : len(NODE_FEATURE_NAMES)],
        "edges": edges[:, : len(EDGE_FEATURE_NAMES)],
        "metadata": metadata,
        "source_base_mean_K": source_base_mean,
        "source_base_peak_K": source_base_peak,
    }, warnings


def group_rows_by_case(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {split: defaultdict(list) for split in split_rows}
    uid_seen: dict[str, str] = {}
    for split, rows in split_rows.items():
        for row in rows:
            case_id = row.get("case_id")
            uid = row.get("sample_uid")
            if not case_id:
                raise ValueError(f"{split} row missing case_id")
            if not uid:
                raise ValueError(f"{split} row missing sample_uid")
            if uid in uid_seen and uid_seen[uid] != split:
                raise ValueError(f"sample_uid {uid} appears in both {uid_seen[uid]} and {split}")
            uid_seen[uid] = split
            grouped[split][case_id].append(row)
    return grouped


def load_metadata_sidecars(index_paths: Iterable[Path]) -> tuple[dict[str, dict[str, float]], list[str], list[Path]]:
    metadata_rows: dict[str, dict[str, float]] = {}
    feature_names: list[str] = []
    sources: list[Path] = []
    for index_path in index_paths:
        path = index_path.expanduser().resolve()
        manifest_path = find_sidecar(path, "metadata_manifest.json")
        table_path = find_sidecar(path, "metadata_features.csv")
        if manifest_path is None or table_path is None:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        active = [str(name) for name in manifest.get("active_features", [])]
        if not feature_names:
            feature_names = active
        elif feature_names != active:
            raise ValueError(f"metadata active feature mismatch between sidecars: {manifest_path}")
        with table_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                metadata_rows[row["sample_uid"]] = {name: float(row[name]) for name in feature_names}
        sources.extend([manifest_path, table_path])
    return metadata_rows, feature_names, sorted(set(sources))


def find_sidecar(index_path: Path, name: str) -> Path | None:
    for candidate in (index_path.parent / name, index_path.parent.parent / name):
        if candidate.exists():
            return candidate
    return None


def find_alternate_graph_path(row: dict[str, str]) -> Path | None:
    uid = row.get("sample_uid", "")
    case_id = row.get("case_id", "")
    if not uid or not case_id:
        return None
    expected_name = f"{uid}_graph.npz"
    graph_value = row.get("graph_path", "")
    candidates: list[Path] = []
    if graph_value:
        path = Path(graph_value)
        parts = list(path.parts)
        for index, part in enumerate(parts):
            if part.endswith("_context_graph"):
                alt_parts = parts[:]
                alt_parts[index] = part.replace("_context_graph", "_graph")
                candidates.append(resolve_path(Path(*alt_parts)))
    candidates.extend(
        [
            REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1_artifacts/package_plus_power_graph/graph_features" / case_id / expected_name,
            REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power/graph_features" / case_id / expected_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def fit_train_scaler(train_matrix: np.ndarray, descriptor_names: list[str]) -> dict[str, Any]:
    mean = np.mean(train_matrix, axis=0)
    std = np.std(train_matrix, axis=0)
    std_safe = np.where(std > EPS, std, 1.0)
    median = np.median(train_matrix, axis=0)
    q25 = np.percentile(train_matrix, 25, axis=0)
    q75 = np.percentile(train_matrix, 75, axis=0)
    iqr = np.where((q75 - q25) > EPS, q75 - q25, 1.0)
    min_values = np.min(train_matrix, axis=0)
    max_values = np.max(train_matrix, axis=0)
    cov = np.cov((train_matrix - mean).T) if train_matrix.shape[0] > 1 else np.eye(train_matrix.shape[1])
    cov = np.atleast_2d(cov)
    regularization = 1.0e-3
    cov_reg = cov + np.eye(cov.shape[0]) * regularization
    inv_cov = np.linalg.pinv(cov_reg)
    return {
        "descriptor_names": descriptor_names,
        "mean": mean,
        "std": std_safe,
        "raw_std": std,
        "median": median,
        "iqr": iqr,
        "min": min_values,
        "max": max_values,
        "inv_cov": inv_cov,
        "mahalanobis_regularization": regularization,
    }


def standardize_records(records: list[dict[str, Any]], descriptor_names: list[str], scaler: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    mean = scaler["mean"]
    std = scaler["std"]
    for record in records:
        values = np.asarray([float(record[name]) for name in descriptor_names], dtype=np.float64)
        z = (values - mean) / std
        out.append({"case_id": record["case_id"], "split": record["split"], **{name: float(value) for name, value in zip(descriptor_names, z, strict=True)}})
    return out


def compute_distances_and_ood(
    *,
    family_records: list[dict[str, Any]],
    standardized_records: list[dict[str, Any]],
    descriptor_names: list[str],
    train_cases: list[str],
    eval_cases: list[str],
    scaler: dict[str, Any],
    metrics: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_by_case = {record["case_id"]: record for record in family_records}
    z_by_case = {
        record["case_id"]: np.asarray([float(record[name]) for name in descriptor_names], dtype=np.float64)
        for record in standardized_records
    }
    raw_matrix = {
        record["case_id"]: np.asarray([float(record[name]) for name in descriptor_names], dtype=np.float64)
        for record in family_records
    }
    nearest_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    train_min = scaler["min"]
    train_max = scaler["max"]
    train_median = scaler["median"]
    train_iqr = scaler["iqr"]
    inv_cov = scaler["inv_cov"]
    for case_id in sorted([*train_cases, *eval_cases]):
        split = raw_by_case[case_id]["split"]
        distances = []
        for train_case in train_cases:
            euclidean = float(np.linalg.norm(z_by_case[case_id] - z_by_case[train_case]))
            robust = float(np.linalg.norm((raw_matrix[case_id] - train_median) / train_iqr - (raw_matrix[train_case] - train_median) / train_iqr))
            diff = raw_matrix[case_id] - raw_matrix[train_case]
            mahalanobis = float(math.sqrt(max(float(diff @ inv_cov @ diff.T), 0.0)))
            cosine = cosine_distance(z_by_case[case_id], z_by_case[train_case])
            distance_rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "train_case_id": train_case,
                    "euclidean_distance": euclidean,
                    "robust_euclidean_distance": robust,
                    "mahalanobis_distance": mahalanobis,
                    "cosine_distance": cosine,
                }
            )
            distances.append((train_case, euclidean, robust, mahalanobis, cosine))
        distances_sorted = sorted(distances, key=lambda item: item[1])
        out_of_range = []
        for index, name in enumerate(descriptor_names):
            value = raw_matrix[case_id][index]
            if value < train_min[index] - 1.0e-9 or value > train_max[index] + 1.0e-9:
                side = "below" if value < train_min[index] else "above"
                standardized_gap = 0.0
                if side == "below":
                    standardized_gap = float((train_min[index] - value) / scaler["std"][index])
                else:
                    standardized_gap = float((value - train_max[index]) / scaler["std"][index])
                out_of_range.append((name, value, train_min[index], train_max[index], side, standardized_gap))
                ood_rows.append(
                    {
                        "case_id": case_id,
                        "split": split,
                        "feature": name,
                        "value": value,
                        "train_min": float(train_min[index]),
                        "train_max": float(train_max[index]),
                        "side": side,
                        "standardized_gap": standardized_gap,
                    }
                )
        top_z = top_feature_differences(descriptor_names, z_by_case[case_id], limit=12)
        row = {
            "case_id": case_id,
            "split": split,
            "descriptor_count": len(descriptor_names),
            "out_of_range_count": len(out_of_range),
            "out_of_range_fraction": len(out_of_range) / max(len(descriptor_names), 1),
            "top_standardized_differences": json.dumps(top_z),
        }
        for rank in range(3):
            if rank < len(distances_sorted):
                train_case, euclidean, robust, mahalanobis, cosine = distances_sorted[rank]
                row[f"nearest_{rank + 1}_case_id"] = train_case
                row[f"nearest_{rank + 1}_euclidean_distance"] = euclidean
                if rank == 0:
                    row["euclidean_distance"] = euclidean
                    row["robust_euclidean_distance"] = robust
                    row["mahalanobis_distance"] = mahalanobis
                    row["cosine_distance"] = cosine
            else:
                row[f"nearest_{rank + 1}_case_id"] = ""
                row[f"nearest_{rank + 1}_euclidean_distance"] = ""
        nearest_rows.append(row)
    return nearest_rows, distance_rows, ood_rows


def compute_pca_projection(
    standardized_records: list[dict[str, Any]],
    descriptor_names: list[str],
    train_cases: list[str],
    requested_components: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case = {record["case_id"]: record for record in standardized_records}
    train_matrix = np.asarray([[float(by_case[case][name]) for name in descriptor_names] for case in train_cases], dtype=np.float64)
    centered = train_matrix - np.mean(train_matrix, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    max_components = min(max(1, requested_components), vh.shape[0])
    components = vh[:max_components]
    variance = singular_values**2
    explained = variance / max(float(np.sum(variance)), EPS)
    rows: list[dict[str, Any]] = []
    for record in standardized_records:
        vector = np.asarray([float(record[name]) for name in descriptor_names], dtype=np.float64)
        projected = (vector - np.mean(train_matrix, axis=0)) @ components.T
        row = {"case_id": record["case_id"], "split": record["split"]}
        for index in range(max_components):
            row[f"pc{index + 1}"] = float(projected[index])
        rows.append(row)
    hull = convex_hull_membership_2d(rows, train_cases) if max_components >= 2 and len(train_cases) >= 3 else {}
    for row in rows:
        if row["case_id"] in hull:
            row["inside_train_hull_2d"] = hull[row["case_id"]]
    return rows, {
        "components": max_components,
        "explained_variance_ratio": [float(value) for value in explained[:max_components]],
        "fit_on": "train families only",
        "convex_hull_2d": hull,
        "convex_hull_note": "2D PCA hull membership is diagnostic only and is not proof of high-dimensional in-distribution status.",
    }


def convex_hull_membership_2d(rows: list[dict[str, Any]], train_cases: list[str]) -> dict[str, bool]:
    points = [(float(row["pc1"]), float(row["pc2"]), row["case_id"]) for row in rows if row["case_id"] in train_cases]
    unique = sorted(set((x, y) for x, y, _ in points))
    if len(unique) < 3:
        return {}
    hull = monotonic_chain(unique)
    return {row["case_id"]: point_in_convex_polygon((float(row["pc1"]), float(row["pc2"])), hull) for row in rows}


def monotonic_chain(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_convex_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    signs = []
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        signs.append(cross)
    return all(value >= -1.0e-9 for value in signs) or all(value <= 1.0e-9 for value in signs)


def plot_outputs(
    out_dir: Path,
    family_records: list[dict[str, Any]],
    nearest_rows: list[dict[str, Any]],
    pca_rows: list[dict[str, Any]],
    descriptor_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (out_dir / "plot_warnings.txt").write_text(f"matplotlib unavailable: {exc}; using Pillow fallback plots\n", encoding="utf-8")
        plot_outputs_pillow(out_dir, family_records, nearest_rows, pca_rows, descriptor_names)
        return
    colors = {"train": "#4c78a8", "val": "#f58518", "test": "#e45756"}
    plt.figure(figsize=(7, 5))
    for split in ("train", "val", "test"):
        subset = [row for row in pca_rows if row["split"] == split and "pc1" in row and "pc2" in row]
        if not subset:
            continue
        plt.scatter([row["pc1"] for row in subset], [row["pc2"] for row in subset], label=split, color=colors[split])
        for row in subset:
            plt.text(row["pc1"], row["pc2"], row["case_id"], fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Package-Family Descriptor PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_2d.png", dpi=180)
    plt.close()

    eval_nearest = [row for row in nearest_rows if row["split"] in {"val", "test"}]
    plt.figure(figsize=(8, 4))
    plt.bar([row["case_id"] for row in eval_nearest], [float(row["euclidean_distance"]) for row in eval_nearest], color=[colors[row["split"]] for row in eval_nearest])
    plt.ylabel("Nearest Train Distance")
    plt.title("OOD Distance After Train-Only Standardization")
    plt.tight_layout()
    plt.savefig(out_dir / "nearest_distance_bar.png", dpi=180)
    plt.close()

    selected = [
        "package_area_mm2",
        "chiplet_count",
        "whitespace_fraction",
        "pairwise_center_distance_normalized_mean",
        "total_power_W_mean",
        "max_power_density_W_per_mm2_mean",
    ]
    selected = [name for name in selected if name in descriptor_names]
    if selected:
        records_by_split = {split: [record for record in family_records if record["split"] == split] for split in ("train", "val", "test")}
        fig, axes = plt.subplots(len(selected), 1, figsize=(8, max(2.0 * len(selected), 4)), squeeze=False)
        for ax, name in zip(axes[:, 0], selected, strict=True):
            train_values = [float(record[name]) for record in records_by_split["train"]]
            ax.axhspan(min(train_values), max(train_values), color="#d8e6f3", alpha=0.6)
            for split in ("train", "val", "test"):
                subset = records_by_split[split]
                ax.scatter([record["case_id"] for record in subset], [float(record[name]) for record in subset], color=colors[split], label=split)
            ax.set_ylabel(name)
            ax.tick_params(axis="x", rotation=45)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles[:3], labels[:3], loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "selected_feature_ranges.png", dpi=180)
        plt.close(fig)


def plot_outputs_pillow(
    out_dir: Path,
    family_records: list[dict[str, Any]],
    nearest_rows: list[dict[str, Any]],
    pca_rows: list[dict[str, Any]],
    descriptor_names: list[str],
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        (out_dir / "plot_warnings.txt").write_text(f"matplotlib and Pillow unavailable: {exc}\n", encoding="utf-8")
        return
    colors = {"train": (76, 120, 168), "val": (245, 133, 24), "test": (228, 87, 86)}
    image = Image.new("RGB", (900, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), "Package-Family Descriptor PCA", fill=(20, 20, 20))
    valid = [row for row in pca_rows if "pc1" in row and "pc2" in row]
    if valid:
        xs = [float(row["pc1"]) for row in valid]
        ys = [float(row["pc2"]) for row in valid]
        for row in valid:
            x = scale(float(row["pc1"]), min(xs), max(xs), 70, 830)
            y = scale(float(row["pc2"]), min(ys), max(ys), 580, 70)
            color = colors.get(row["split"], (120, 120, 120))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.text((x + 7, y - 5), row["case_id"], fill=(20, 20, 20))
    draw.rectangle((60, 60, 850, 590), outline=(180, 180, 180))
    image.save(out_dir / "pca_2d.png")

    eval_nearest = [row for row in nearest_rows if row["split"] in {"val", "test"}]
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), "Nearest Train Distance", fill=(20, 20, 20))
    values = [float(row["euclidean_distance"]) for row in eval_nearest] or [0.0]
    max_value = max(values) or 1.0
    bar_width = 110
    for index, row in enumerate(eval_nearest):
        x0 = 80 + index * 180
        value = float(row["euclidean_distance"])
        y0 = scale(value, 0.0, max_value, 450, 80)
        draw.rectangle((x0, y0, x0 + bar_width, 450), fill=colors.get(row["split"], (120, 120, 120)))
        draw.text((x0, 458), row["case_id"], fill=(20, 20, 20))
        draw.text((x0, max(y0 - 18, 60)), f"{value:.2f}", fill=(20, 20, 20))
    draw.line((60, 450, 850, 450), fill=(160, 160, 160))
    image.save(out_dir / "nearest_distance_bar.png")

    selected = [
        "package_area_mm2",
        "chiplet_count",
        "whitespace_fraction",
        "pairwise_center_distance_normalized_mean",
        "total_power_W_mean",
        "max_power_density_W_per_mm2_mean",
    ]
    selected = [name for name in selected if name in descriptor_names]
    image = Image.new("RGB", (1100, max(260, 130 * len(selected) + 60)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), "Selected Feature Ranges", fill=(20, 20, 20))
    for row_index, name in enumerate(selected):
        y = 70 + row_index * 130
        values = [float(record[name]) for record in family_records]
        train_values = [float(record[name]) for record in family_records if record["split"] == "train"]
        min_v = min(values)
        max_v = max(values)
        train_min = min(train_values)
        train_max = max(train_values)
        draw.text((25, y - 18), name, fill=(20, 20, 20))
        x_train_min = scale(train_min, min_v, max_v, 240, 1030)
        x_train_max = scale(train_max, min_v, max_v, 240, 1030)
        draw.rectangle((x_train_min, y - 8, x_train_max, y + 8), fill=(216, 230, 243))
        draw.line((240, y, 1030, y), fill=(180, 180, 180))
        for record in family_records:
            x = scale(float(record[name]), min_v, max_v, 240, 1030)
            color = colors.get(record["split"], (120, 120, 120))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            if record["split"] != "train":
                draw.text((x + 5, y + 8), record["case_id"], fill=(20, 20, 20))
    image.save(out_dir / "selected_feature_ranges.png")


def scale(value: float, min_value: float, max_value: float, out_min: float, out_max: float) -> float:
    if abs(max_value - min_value) <= EPS:
        return 0.5 * (out_min + out_max)
    t = (value - min_value) / (max_value - min_value)
    return out_min + t * (out_max - out_min)


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "ChipTherm package-family coverage analysis",
        "",
        "Primary descriptors are derived from graph_path NPZ files, metadata sidecars, index scalar columns, and optional source_superposition_base_path maps.",
        "HotSpot target Y tensors and model errors are not read for primary descriptor distances.",
        "Scaling, Mahalanobis covariance, and PCA are fitted on training families only.",
        "",
        "Reused project fields:",
        "- ChipThermDataset sidecar convention: metadata_features.csv, metadata_manifest.json, graph_manifest.json.",
        "- chiptherm.ml.graph_models.NODE_FEATURE_NAMES and EDGE_FEATURE_NAMES.",
        "- graph NPZ arrays: node_features, edge_features, chiplet_rects, package_size.",
        "- source_superposition_base_path for label-free base mean/peak descriptors when present.",
        "",
        "Output files:",
        "- family_descriptors.csv: raw family descriptor vectors.",
        "- standardized_family_descriptors.csv: train-standardized descriptor vectors.",
        "- nearest_training_families.csv: nearest train-family distances for all families.",
        "- out_of_range_features.csv: descriptors outside train-family min/max range.",
        "- distance_matrix.csv: eval/train distance table.",
        "- pca_projection.csv and pca_2d.png: PCA fit on training families only.",
        "",
        f"Train families: {', '.join(summary['split_families']['train'])}",
        f"Validation families: {', '.join(summary['split_families']['val'])}",
        f"Test families: {', '.join(summary['split_families']['test'])}",
    ]
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_output_finiteness(out_dir: Path) -> None:
    for path in out_dir.glob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row_index, row in enumerate(reader):
                for key, value in row.items():
                    if value == "" or key in {"case_id", "split", "train_case_id", "feature", "side", "top_standardized_differences", "inside_train_hull_2d"}:
                        continue
                    try:
                        parsed = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(parsed):
                        raise ValueError(f"{path} row {row_index} column {key} is non-finite")


def ensure_finite_descriptors(records: list[dict[str, Any]], descriptor_names: list[str]) -> None:
    for record in records:
        for name in descriptor_names:
            value = float(record[name])
            if not math.isfinite(value):
                raise ValueError(f"{record['case_id']} descriptor {name} is non-finite")


def scaler_to_json(scaler: dict[str, Any]) -> dict[str, Any]:
    names = scaler["descriptor_names"]
    return {
        "fit_on": "train families only",
        "mahalanobis_regularization": float(scaler["mahalanobis_regularization"]),
        "features": {
            name: {
                "mean": float(scaler["mean"][index]),
                "std": float(scaler["std"][index]),
                "raw_std": float(scaler["raw_std"][index]),
                "median": float(scaler["median"][index]),
                "iqr": float(scaler["iqr"][index]),
                "min": float(scaler["min"][index]),
                "max": float(scaler["max"][index]),
            }
            for index, name in enumerate(names)
        },
    }


def add_stats(out: dict[str, float], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    out[f"{prefix}_mean"] = safe_mean(values)
    out[f"{prefix}_std"] = safe_std(values)
    out[f"{prefix}_min"] = safe_min(values)
    out[f"{prefix}_max"] = safe_max(values)


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=np.float64)
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def center_of_mass_offset(nodes: np.ndarray, package_width: float, package_height: float, node_idx: dict[str, int]) -> float:
    powers = nodes[:, node_idx["total_power_W"]]
    weights = powers / max(float(np.sum(powers)), EPS)
    cx = float(np.sum(nodes[:, node_idx["center_x_mm"]] * weights))
    cy = float(np.sum(nodes[:, node_idx["center_y_mm"]] * weights))
    normalized_dx = cx / max(package_width, EPS) - 0.5
    normalized_dy = cy / max(package_height, EPS) - 0.5
    return float(math.sqrt(normalized_dx * normalized_dx + normalized_dy * normalized_dy))


def normalized_histogram(values: np.ndarray, *, bins: int, range_max: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(bins, dtype=np.float64)
    clipped = np.clip(values, 0.0, range_max)
    counts, _ = np.histogram(clipped, bins=bins, range=(0.0, range_max))
    return counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)


def top_feature_differences(names: list[str], z: np.ndarray, *, limit: int) -> list[dict[str, float | str]]:
    order = np.argsort(-np.abs(z))[:limit]
    return [{"feature": names[index], "z": float(z[index])} for index in order]


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= EPS:
        return 0.0
    return float(1.0 - np.dot(left, right) / denom)


def feature_index_map(names: Iterable[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if values.size else 0.0


def safe_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else 0.0


def safe_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    candidates = [Path.cwd() / path, REPO_ROOT / path]
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


if __name__ == "__main__":
    raise SystemExit(main())
