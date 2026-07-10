#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ChipletSource:
    name: str
    chiplet_type: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    power_W: float

    @property
    def center_x_mm(self) -> float:
        return self.x_mm + 0.5 * self.width_mm

    @property
    def center_y_mm(self) -> float:
        return self.y_mm + 0.5 * self.height_mm

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    @property
    def aspect_ratio(self) -> float:
        return self.width_mm / self.height_mm

    @property
    def power_density_W_per_mm2(self) -> float:
        return self.power_W / self.area_mm2


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, array: np.ndarray) -> None:
        data = array.astype(np.float64, copy=False)
        self.count += int(data.size)
        self.total += float(data.sum())
        self.total_sq += float((data * data).sum())
        self.minimum = min(self.minimum, float(data.min()))
        self.maximum = max(self.maximum, float(data.max()))

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 1.0
        variance = max(self.total_sq / self.count - self.mean * self.mean, 1.0e-12)
        return float(variance**0.5)

    def to_dict(self) -> dict[str, float]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append universal thermal-impedance descriptor channels to ChipTherm X tensors.")
    parser.add_argument("--source-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_finite_source/package_plus_power", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance/package_plus_power", type=Path)
    parser.add_argument("--enclosed-power-radii-mm", nargs="+", default=[2.0, 4.0, 8.0, 16.0], type=float)
    parser.add_argument("--crowding-epsilon-mm", default=1.0, type=float)
    parser.add_argument("--quadrature-size", default=4, type=int)
    parser.add_argument("--power-tolerance-W", default=1.0e-3, type=float)
    parser.add_argument("--save-diagnostic-plot", action="store_true")
    args = parser.parse_args()

    validate_args(args.enclosed_power_radii_mm, args.crowding_epsilon_mm, args.quadrature_size)
    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    combined_index = source_root / "combined_encoded_index.csv"
    if not combined_index.exists():
        raise SystemExit(f"missing source combined index: {combined_index}")
    fieldnames, rows = read_csv_rows(combined_index)
    source_manifest = load_feature_manifest(source_root)
    source_channel_names = list(source_manifest["channel_names"])
    new_channel_specs = channel_specs(args.enclosed_power_radii_mm)
    new_channel_names = [spec["name"] for spec in new_channel_specs]
    output_channel_names = source_channel_names + new_channel_names

    out_root.mkdir(parents=True, exist_ok=True)
    encoded_root = out_root / "encoded_impedance"
    encoded_root.mkdir(parents=True, exist_ok=True)

    stats = [RunningMoments() for _ in new_channel_names]
    case_stats: dict[str, list[RunningMoments]] = {}
    records: list[dict[str, str]] = []
    jsonl_records: list[dict[str, Any]] = []
    runtimes: list[float] = []
    power_errors: list[float] = []
    first_diagnostic: dict[str, Any] | None = None
    split_membership = {row["sample_uid"]: row["split"] for row in rows}

    print(f"Generating thermal-impedance descriptor features for {len(rows)} samples...")
    for row_index, row in enumerate(rows, start=1):
        start = time.perf_counter()
        sample_uid = row["sample_uid"]
        layout_path, power_path = source_paths_for_row(row)
        layout = load_json(layout_path)
        power_data = load_yaml(power_path)
        chiplets = load_chiplets(layout, power_data)
        package_width_mm, package_height_mm = package_size_mm(layout)

        x_path = resolve_repo_path(row["x_path"])
        x = np.load(x_path).astype(np.float32, copy=False)
        if x.ndim != 3:
            raise SystemExit(f"{sample_uid} expected X tensor with shape (C,H,W), got {x.shape}")
        _, grid_rows, grid_cols = x.shape
        grid_x, grid_y = grid_cell_centers(package_width_mm, package_height_mm, grid_rows, grid_cols)

        features = compute_impedance_features(
            chiplets,
            grid_x=grid_x,
            grid_y=grid_y,
            package_width_mm=package_width_mm,
            package_height_mm=package_height_mm,
            radii_mm=args.enclosed_power_radii_mm,
            crowding_epsilon_mm=args.crowding_epsilon_mm,
            quadrature_size=args.quadrature_size,
        )
        validate_feature_maps(
            sample_uid,
            features,
            radii_count=len(args.enclosed_power_radii_mm),
            total_power_W=sum(chiplet.power_W for chiplet in chiplets),
        )

        represented_power_W = sum(chiplet.power_W for chiplet in chiplets)
        metadata_power_W = optional_float(row.get("total_power_W"))
        if metadata_power_W is not None:
            power_error = abs(represented_power_W - metadata_power_W)
            power_errors.append(power_error)
            if power_error > args.power_tolerance_W:
                raise SystemExit(
                    f"{sample_uid} source power {represented_power_W:.6f} W does not match metadata "
                    f"{metadata_power_W:.6f} W within {args.power_tolerance_W}"
                )

        x_aug = np.concatenate([x, features], axis=0).astype(np.float32, copy=False)
        out_x_path = encoded_root / row["case_id"] / f"{sample_uid}_x_impedance.npy"
        out_x_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_x_path, x_aug)

        for index, channel_stats in enumerate(stats):
            channel_stats.update(features[index])
        case_stats.setdefault(row["case_id"], [RunningMoments() for _ in new_channel_names])
        for index, channel_stats in enumerate(case_stats[row["case_id"]]):
            channel_stats.update(features[index])

        new_row = dict(row)
        new_row["x_path"] = repo_relative(out_x_path)
        records.append(new_row)
        runtime_s = time.perf_counter() - start
        runtimes.append(runtime_s)
        jsonl_records.append(
            {
                **new_row,
                "thermal_impedance": {
                    "layout_path": repo_relative(layout_path),
                    "power_path": repo_relative(power_path),
                    "enclosed_power_radii_mm": args.enclosed_power_radii_mm,
                    "crowding_epsilon_mm": args.crowding_epsilon_mm,
                    "quadrature_size": args.quadrature_size,
                    "chiplet_count": len(chiplets),
                    "represented_power_W": represented_power_W,
                    "metadata_total_power_W": metadata_power_W,
                    "power_error_W": None if metadata_power_W is None else abs(represented_power_W - metadata_power_W),
                    "output_x_shape": list(x_aug.shape),
                },
            }
        )
        if first_diagnostic is None:
            first_diagnostic = {
                "sample_uid": sample_uid,
                "case_id": row["case_id"],
                "power_density": x[0],
                "features": features,
                "channel_names": new_channel_names,
            }
        if row_index % 250 == 0:
            print(f"  generated {row_index}/{len(rows)} samples")

    validate_split_membership(split_membership, records)
    leakage = check_content_hash_leakage(records)
    write_dataset_outputs(
        out_root=out_root,
        fieldnames=fieldnames,
        records=records,
        jsonl_records=jsonl_records,
        source_root=source_root,
        source_manifest=source_manifest,
        source_channel_names=source_channel_names,
        new_channel_specs=new_channel_specs,
        output_channel_names=output_channel_names,
        stats=stats,
        case_stats=case_stats,
        runtimes=runtimes,
        power_errors=power_errors,
        radii_mm=args.enclosed_power_radii_mm,
        crowding_epsilon_mm=args.crowding_epsilon_mm,
        quadrature_size=args.quadrature_size,
        leakage=leakage,
    )
    if args.save_diagnostic_plot and first_diagnostic is not None:
        save_diagnostic_plot(out_root / "thermal_impedance_diagnostic.png", first_diagnostic, args.enclosed_power_radii_mm)

    print("Thermal-impedance feature dataset build complete")
    print(f"Samples: {len(records)}")
    print(f"Input channels: {len(source_channel_names)} -> {len(output_channel_names)}")
    print(f"New channels: {', '.join(new_channel_names)}")
    print(f"Feature generation runtime/sample: {float(np.mean(runtimes)):.6f} s")
    print(f"Cross-split content-hash overlaps: {leakage['cross_split_content_hash_overlaps']}")
    print(f"Output: {out_root}")
    return 0


def validate_args(radii_mm: list[float], crowding_epsilon_mm: float, quadrature_size: int) -> None:
    if not radii_mm:
        raise SystemExit("--enclosed-power-radii-mm must contain at least one radius")
    if any(radius <= 0.0 for radius in radii_mm):
        raise SystemExit("enclosed-power radii must be positive")
    if sorted(radii_mm) != list(radii_mm):
        raise SystemExit("enclosed-power radii must be sorted ascending")
    if crowding_epsilon_mm <= 0.0:
        raise SystemExit("--crowding-epsilon-mm must be positive")
    if quadrature_size <= 0:
        raise SystemExit("--quadrature-size must be positive")


def channel_specs(radii_mm: list[float]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for radius in radii_mm:
        specs.append(
            {
                "name": f"enclosed_power_R{format_mm(radius)}mm_W",
                "units": "W",
                "group": "multi_radius_enclosed_power",
                "equation": f"sum quadrature source weights whose distance to the grid-cell center is <= {radius:g} mm",
            }
        )
    specs.extend(
        [
            {
                "name": "distance_to_left_edge_mm",
                "units": "mm",
                "group": "package_edge_distance",
                "equation": "x_cell_center_mm",
            },
            {
                "name": "distance_to_right_edge_mm",
                "units": "mm",
                "group": "package_edge_distance",
                "equation": "package_width_mm - x_cell_center_mm",
            },
            {
                "name": "distance_to_bottom_edge_mm",
                "units": "mm",
                "group": "package_edge_distance",
                "equation": "y_cell_center_mm",
            },
            {
                "name": "distance_to_top_edge_mm",
                "units": "mm",
                "group": "package_edge_distance",
                "equation": "package_height_mm - y_cell_center_mm",
            },
            {
                "name": "minimum_distance_to_package_edge_mm",
                "units": "mm",
                "group": "package_edge_distance",
                "equation": "min(left, right, bottom, top)",
            },
            {
                "name": "chiplet_total_power_W",
                "units": "W",
                "group": "chiplet_instance_descriptor",
                "equation": "chiplet total power rasterized inside its exact rectangle using cell-center inclusion",
            },
            {
                "name": "chiplet_width_mm",
                "units": "mm",
                "group": "chiplet_instance_descriptor",
                "equation": "chiplet width rasterized inside its exact rectangle using cell-center inclusion",
            },
            {
                "name": "chiplet_height_mm",
                "units": "mm",
                "group": "chiplet_instance_descriptor",
                "equation": "chiplet height rasterized inside its exact rectangle using cell-center inclusion",
            },
            {
                "name": "chiplet_area_mm2",
                "units": "mm^2",
                "group": "chiplet_instance_descriptor",
                "equation": "chiplet width_mm * chiplet height_mm rasterized inside its exact rectangle",
            },
            {
                "name": "chiplet_aspect_ratio",
                "units": "dimensionless",
                "group": "chiplet_instance_descriptor",
                "equation": "chiplet width_mm / chiplet height_mm rasterized inside its exact rectangle",
            },
            {
                "name": "chiplet_power_density_W_per_mm2",
                "units": "W/mm^2",
                "group": "chiplet_instance_descriptor",
                "equation": "chiplet_total_power_W / chiplet_area_mm2 rasterized inside its exact rectangle",
            },
            {
                "name": "thermal_crowding_W_per_mm",
                "units": "W/mm",
                "group": "thermal_crowding",
                "equation": "sum_i P_i / sqrt(||r - c_i||^2 + epsilon_mm^2)",
            },
        ]
    )
    return specs


def format_mm(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {"sample_uid", "original_sample_uid", "case_id", "dataset_source", "split", "x_path", "y_path", "prediction_path", "residual_path"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"{path} missing required columns: {', '.join(missing)}")
    return fieldnames, rows


def load_feature_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "feature_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "channel_names" not in manifest:
        raise SystemExit(f"{manifest_path} is missing channel_names")
    return manifest


def source_paths_for_row(row: dict[str, str]) -> tuple[Path, Path]:
    dataset_source = row["dataset_source"]
    case_id = row["case_id"]
    original_uid = row["original_sample_uid"]
    prefix = f"{case_id}_"
    if not original_uid.startswith(prefix):
        raise SystemExit(f"{row['sample_uid']} original_sample_uid {original_uid!r} does not match case_id {case_id!r}")
    sample_dir = original_uid[len(prefix) :]
    source_dir = REPO_ROOT / "data/runs/benchmarks" / dataset_source / case_id / sample_dir / "source"
    layout_path = source_dir / "layout.json"
    power_path = source_dir / "power.yaml"
    missing = [path for path in (layout_path, power_path) if not path.exists()]
    if missing:
        raise SystemExit(f"{row['sample_uid']} missing source metadata: {', '.join(str(path) for path in missing)}")
    return layout_path, power_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def package_size_mm(layout: dict[str, Any]) -> tuple[float, float]:
    size = layout.get("package", {}).get("size", {})
    width = float(size["width"])
    height = float(size["height"])
    if width <= 0.0 or height <= 0.0:
        raise SystemExit(f"invalid package size {width} x {height} mm")
    return width, height


def load_chiplets(layout: dict[str, Any], power_data: dict[str, Any]) -> list[ChipletSource]:
    power_by_name = active_power_map(power_data)
    chiplets: list[ChipletSource] = []
    for chiplet in layout.get("chiplets", []):
        name = str(chiplet["name"])
        if name not in power_by_name:
            raise SystemExit(f"power file missing chiplet {name}")
        size = chiplet["size"]
        position = chiplet["position"]
        source = ChipletSource(
            name=name,
            chiplet_type=str(chiplet.get("type", "")),
            x_mm=float(position["x"]),
            y_mm=float(position["y"]),
            width_mm=float(size["width"]),
            height_mm=float(size["height"]),
            power_W=float(power_by_name[name]),
        )
        if source.width_mm <= 0.0 or source.height_mm <= 0.0 or source.power_W < 0.0:
            raise SystemExit(f"invalid chiplet source: {source}")
        density_error = abs(source.power_density_W_per_mm2 - source.power_W / source.area_mm2)
        if density_error > 1.0e-12:
            raise SystemExit(f"internal chiplet density error for {source.name}")
        chiplets.append(source)
    if not chiplets:
        raise SystemExit("layout contains no chiplets")
    extra = set(power_by_name) - {chiplet.name for chiplet in chiplets}
    if extra:
        raise SystemExit(f"power file has chiplets absent from layout: {sorted(extra)}")
    return chiplets


def active_power_map(power_data: dict[str, Any]) -> dict[str, float]:
    active = power_data.get("active_workload")
    workloads = power_data.get("workloads") or {}
    if active and active in workloads:
        return {str(name): float(value) for name, value in workloads[active].items()}
    if "chiplets" in power_data:
        return {str(name): float(value) for name, value in power_data["chiplets"].items()}
    raise SystemExit("power data has no active workload or chiplets power map")


def grid_cell_centers(package_width_mm: float, package_height_mm: float, grid_rows: int, grid_cols: int) -> tuple[np.ndarray, np.ndarray]:
    x_centers = (np.arange(grid_cols, dtype=np.float64) + 0.5) * package_width_mm / grid_cols
    y_centers = (np.arange(grid_rows, dtype=np.float64) + 0.5) * package_height_mm / grid_rows
    return np.meshgrid(x_centers, y_centers)


def compute_impedance_features(
    chiplets: list[ChipletSource],
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    package_width_mm: float,
    package_height_mm: float,
    radii_mm: list[float],
    crowding_epsilon_mm: float,
    quadrature_size: int,
) -> np.ndarray:
    grid_rows, grid_cols = grid_x.shape
    channels: list[np.ndarray] = []
    channels.extend(enclosed_power_maps(chiplets, grid_x, grid_y, radii_mm, quadrature_size))
    left = grid_x
    right = package_width_mm - grid_x
    bottom = grid_y
    top = package_height_mm - grid_y
    channels.extend([left, right, bottom, top, np.minimum.reduce([left, right, bottom, top])])
    channels.extend(chiplet_descriptor_maps(chiplets, grid_x, grid_y, grid_rows, grid_cols))
    channels.append(thermal_crowding_map(chiplets, grid_x, grid_y, crowding_epsilon_mm))
    return np.stack(channels, axis=0).astype(np.float32)


def enclosed_power_maps(
    chiplets: list[ChipletSource],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    radii_mm: list[float],
    quadrature_size: int,
) -> list[np.ndarray]:
    maps = [np.zeros_like(grid_x, dtype=np.float64) for _ in radii_mm]
    offsets = (np.arange(quadrature_size, dtype=np.float64) + 0.5) / quadrature_size
    radii_sq = [radius * radius for radius in radii_mm]
    for chiplet in chiplets:
        qx = chiplet.x_mm + offsets * chiplet.width_mm
        qy = chiplet.y_mm + offsets * chiplet.height_mm
        source_x, source_y = np.meshgrid(qx, qy)
        source_points = np.column_stack([source_x.reshape(-1), source_y.reshape(-1)])
        weight_W = chiplet.power_W / float(source_points.shape[0])
        for point_x, point_y in source_points:
            distance_sq = (grid_x - point_x) ** 2 + (grid_y - point_y) ** 2
            for index, radius_sq in enumerate(radii_sq):
                maps[index] += weight_W * (distance_sq <= radius_sq)
    return maps


def chiplet_descriptor_maps(
    chiplets: list[ChipletSource],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> list[np.ndarray]:
    descriptor_maps = [np.zeros((grid_rows, grid_cols), dtype=np.float64) for _ in range(6)]
    assigned = np.zeros((grid_rows, grid_cols), dtype=bool)
    for chiplet in chiplets:
        inside = (
            (grid_x >= chiplet.x_mm)
            & (grid_x <= chiplet.x_mm + chiplet.width_mm)
            & (grid_y >= chiplet.y_mm)
            & (grid_y <= chiplet.y_mm + chiplet.height_mm)
        )
        if np.any(assigned & inside):
            raise SystemExit(f"raster cell-center overlap detected for chiplet {chiplet.name}")
        assigned |= inside
        values = [
            chiplet.power_W,
            chiplet.width_mm,
            chiplet.height_mm,
            chiplet.area_mm2,
            chiplet.aspect_ratio,
            chiplet.power_density_W_per_mm2,
        ]
        for descriptor_map, value in zip(descriptor_maps, values):
            descriptor_map[inside] = value
    return descriptor_maps


def thermal_crowding_map(
    chiplets: list[ChipletSource],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    epsilon_mm: float,
) -> np.ndarray:
    crowding = np.zeros_like(grid_x, dtype=np.float64)
    for chiplet in chiplets:
        distance_sq = (grid_x - chiplet.center_x_mm) ** 2 + (grid_y - chiplet.center_y_mm) ** 2
        crowding += chiplet.power_W / np.sqrt(distance_sq + epsilon_mm * epsilon_mm)
    return crowding


def validate_feature_maps(sample_uid: str, features: np.ndarray, *, radii_count: int, total_power_W: float) -> None:
    if not np.isfinite(features).all():
        raise SystemExit(f"{sample_uid} generated NaN/Inf thermal-impedance features")
    enclosed = features[:radii_count]
    if np.any(np.diff(enclosed, axis=0) < -1.0e-4):
        raise SystemExit(f"{sample_uid} enclosed-power maps are not monotonic by radius")
    if float(enclosed.max()) > total_power_W + 1.0e-3:
        raise SystemExit(f"{sample_uid} enclosed power exceeds total power")
    if float(features[-1].min()) < -1.0e-8:
        raise SystemExit(f"{sample_uid} thermal crowding map contains negative values")


def validate_split_membership(source_membership: dict[str, str], records: list[dict[str, str]]) -> None:
    output_membership = {row["sample_uid"]: row["split"] for row in records}
    if source_membership != output_membership:
        raise SystemExit("output split membership does not match source clean split")
    split_sets: dict[str, set[str]] = defaultdict(set)
    for sample_uid, split in output_membership.items():
        split_sets[split].add(sample_uid)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise SystemExit(f"sample_uid overlap between {left} and {right}: {sorted(overlap)[:5]}")


def check_content_hash_leakage(records: list[dict[str, str]]) -> dict[str, Any]:
    split_hashes: dict[str, set[str]] = defaultdict(set)
    duplicate_within_split: dict[str, int] = {}
    for row in records:
        x = np.load(resolve_repo_path(row["x_path"]), mmap_mode="r")
        y = np.load(resolve_repo_path(row["y_path"]), mmap_mode="r")
        xy_hash = hash_arrays(x, y)
        split = row["split"]
        if xy_hash in split_hashes[split]:
            duplicate_within_split[split] = duplicate_within_split.get(split, 0) + 1
        split_hashes[split].add(xy_hash)
    cross_overlaps = 0
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        cross_overlaps += len(split_hashes[left] & split_hashes[right])
    if cross_overlaps:
        raise SystemExit(f"new impedance dataset has {cross_overlaps} cross-split content hash overlaps")
    return {
        "cross_split_content_hash_overlaps": cross_overlaps,
        "within_split_duplicate_hashes": duplicate_within_split,
    }


def hash_arrays(x: np.ndarray, y: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"chiptherm_impedance_xy_hash_v1")
    for name, array in (("x", x), ("y", y)):
        contiguous = np.ascontiguousarray(array)
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tuple(int(size) for size in contiguous.shape)).encode("utf-8"))
        hasher.update(str(contiguous.dtype).encode("utf-8"))
        hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def write_dataset_outputs(
    *,
    out_root: Path,
    fieldnames: list[str],
    records: list[dict[str, str]],
    jsonl_records: list[dict[str, Any]],
    source_root: Path,
    source_manifest: dict[str, Any],
    source_channel_names: list[str],
    new_channel_specs: list[dict[str, str]],
    output_channel_names: list[str],
    stats: list[RunningMoments],
    case_stats: dict[str, list[RunningMoments]],
    runtimes: list[float],
    power_errors: list[float],
    radii_mm: list[float],
    crowding_epsilon_mm: float,
    quadrature_size: int,
    leakage: dict[str, Any],
) -> None:
    write_csv(out_root / "combined_encoded_index.csv", fieldnames, records)
    for split in ("train", "val", "test"):
        write_csv(out_root / f"{split}_index.csv", fieldnames, [row for row in records if row["split"] == split])
    write_jsonl(out_root / "combined_encoded_index.jsonl", jsonl_records)

    split_counts = Counter(row["split"] for row in records)
    new_channel_names = [spec["name"] for spec in new_channel_specs]
    new_channel_indices = {name: len(source_channel_names) + index for index, name in enumerate(new_channel_names)}
    channel_stats = {
        name: {
            **stats[index].to_dict(),
            "units": new_channel_specs[index]["units"],
            "group": new_channel_specs[index]["group"],
        }
        for index, name in enumerate(new_channel_names)
    }
    per_case_stats = {
        case_id: {
            name: case_stats_list[index].to_dict()
            for index, name in enumerate(new_channel_names)
        }
        for case_id, case_stats_list in sorted(case_stats.items())
    }
    feature_generation_mean = float(np.mean(runtimes)) if runtimes else 0.0
    physics_runtime = optional_float(source_manifest.get("runtime", {}).get("feature_generation_mean_s_per_sample"))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "out_root": repo_relative(out_root),
        "num_samples": len(records),
        "split_counts": {split: int(split_counts.get(split, 0)) for split in ("train", "val", "test")},
        "source_channels": len(source_channel_names),
        "thermal_impedance_channels": len(new_channel_names),
        "output_channels": len(output_channel_names),
        "channel_names": output_channel_names,
        "thermal_impedance_channel_indices": new_channel_indices,
        "thermal_impedance_channel_specs": new_channel_specs,
        "thermal_impedance_channel_stats": channel_stats,
        "per_case_descriptor_stats": per_case_stats,
        "parameters": {
            "enclosed_power_radii_mm": radii_mm,
            "crowding_epsilon_mm": crowding_epsilon_mm,
            "quadrature_size": quadrature_size,
        },
        "equations": {
            "enclosed_power": "For each destination cell center r, sum quadrature source weights P_i/N_i whose physical distance to r is <= R.",
            "edge_distance": "Physical distance in mm from each grid-cell center to package boundaries.",
            "chiplet_descriptor": "Exact source/layout.json chiplet constants rasterized inside chiplet rectangles by cell-center inclusion.",
            "thermal_crowding": "crowding(r)=sum_i P_i/sqrt(||r-c_i||^2+epsilon_mm^2).",
        },
        "sources": {
            "geometry": "source/layout.json",
            "power": "source/power.yaml active_workload when present, otherwise chiplets map",
            "labels_used_for_feature_generation": False,
        },
        "normalization_note": (
            "Training normalizes channels >=8 using train-only full-map mean/std. "
            "Chiplet descriptor maps intentionally keep zero-background samples in the statistics, "
            "matching existing context-channel normalization and preserving backward compatibility."
        ),
        "runtime": {
            "feature_generation_total_s": float(sum(runtimes)),
            "feature_generation_mean_s_per_sample": feature_generation_mean,
            "feature_generation_median_s_per_sample": float(np.median(runtimes)) if runtimes else 0.0,
            "source_finite_feature_generation_mean_s_per_sample": physics_runtime,
            "online_feature_generation_estimate_s_per_sample": feature_generation_mean + float(physics_runtime or 0.0),
            "online_feature_generation_note": "Estimate sums impedance generation plus source finite-feature generation when available; excludes physics-v1 and CNN.",
        },
        "verification": {
            "split_membership_preserved_from_source": True,
            "cross_split_content_hash_overlaps": leakage["cross_split_content_hash_overlaps"],
            "within_split_duplicate_hashes": leakage["within_split_duplicate_hashes"],
            "represented_power_matches_metadata": True,
            "max_total_power_error_W": float(max(power_errors)) if power_errors else 0.0,
            "enclosed_power_monotonic_by_radius": True,
            "crowding_nonnegative": True,
            "nonfinite_feature_values": 0,
            "case_specific_fitting_or_calibration": False,
        },
    }
    (out_root / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root / "README.md", manifest)


def write_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Dataset v2 Clean Impedance: package_plus_power

This dataset appends compact universal thermal-impedance descriptor channels to:

`{manifest['source_root']}`

Only augmented X tensors are written. Y, physics-v1 predictions, and residual
tensors are reused by path. Train/validation/test membership is unchanged.

## New Feature Groups

- Multi-radius enclosed power maps in watts
- Package-edge distance maps in millimeters
- Chiplet-instance descriptor maps from exact layout/power metadata
- Thermal crowding map: `sum_i P_i / sqrt(||r-c_i||^2 + epsilon_mm^2)`

## Counts

- Source channels: {manifest['source_channels']}
- Added impedance channels: {manifest['thermal_impedance_channels']}
- Output channels: {manifest['output_channels']}
- Samples: {manifest['num_samples']}
- Train/val/test: {manifest['split_counts']['train']} / {manifest['split_counts']['val']} / {manifest['split_counts']['test']}

Feature generation runtime/sample:
`{manifest['runtime']['feature_generation_mean_s_per_sample']:.6f} s`
"""
    path.write_text(text, encoding="utf-8")


def save_diagnostic_plot(path: Path, diagnostic: dict[str, Any], radii_mm: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping diagnostic plot because matplotlib is unavailable: {exc}")
        return
    names = diagnostic["channel_names"]
    features = diagnostic["features"]
    wanted = [0, 1, 2, 3, 8, 10, 14, 15]
    wanted = [index for index in wanted if index < len(names)]
    fig, axes = plt.subplots(1, len(wanted) + 1, figsize=(3.2 * (len(wanted) + 1), 3.2), constrained_layout=True)
    image = axes[0].imshow(diagnostic["power_density"], origin="lower", cmap="magma")
    axes[0].set_title("power density")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    for ax, index in zip(axes[1:], wanted):
        image = ax.imshow(features[index], origin="lower", cmap="viridis")
        ax.set_title(names[index])
        fig.colorbar(image, ax=ax, fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{diagnostic['sample_uid']} ({diagnostic['case_id']})")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, Path.cwd() / path]
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
