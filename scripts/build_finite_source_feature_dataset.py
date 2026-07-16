#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CHANNEL_NAMES = [
    "power_density_W_per_mm2",
    "occupancy_mask",
    "CPU_mask",
    "GPU_or_NPU_mask",
    "memory_mask",
    "IO_or_ANALOG_or_MEMS_mask",
    "normalized_x_coordinate",
    "normalized_y_coordinate",
]


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
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm

    @property
    def power_density_W_per_mm2(self) -> float:
        return self.power_W / self.area_mm2


def main() -> int:
    parser = argparse.ArgumentParser(description="Append finite rectangular-source response channels to a clean ChipTherm index.")
    parser.add_argument("--source-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean/package_plus_power", type=Path)
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_finite_source/package_plus_power", type=Path)
    parser.add_argument("--length-scales-mm", nargs="+", default=[0.5, 1.0, 2.0, 4.0], type=float)
    parser.add_argument("--quadrature-size", default=4, type=int)
    parser.add_argument("--kernel", default="softened_green", choices=["softened_green", "screened_softened_green"])
    parser.add_argument("--power-tolerance-W", default=1.0e-3, type=float)
    parser.add_argument("--save-diagnostic-plot", action="store_true")
    args = parser.parse_args()

    validate_args(args.length_scales_mm, args.quadrature_size)
    source_root = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    combined_index = source_root / "combined_encoded_index.csv"
    if not combined_index.exists():
        raise SystemExit(f"missing source combined index: {combined_index}")

    fieldnames, rows = read_csv_rows(combined_index)
    out_root.mkdir(parents=True, exist_ok=True)
    encoded_root = out_root / "encoded_finite_source"
    encoded_root.mkdir(parents=True, exist_ok=True)

    source_channel_names = read_source_channel_names(source_root)
    finite_channel_names = [finite_channel_name(length) for length in args.length_scales_mm]
    output_channel_names = source_channel_names + finite_channel_names

    feature_stats = [RunningMoments() for _ in finite_channel_names]
    split_records: list[dict[str, str]] = []
    jsonl_records: list[dict[str, Any]] = []
    power_errors: list[float] = []
    runtimes: list[float] = []
    first_diagnostic: dict[str, Any] | None = None

    print(f"Generating finite-source features for {len(rows)} clean samples...")
    for index, row in enumerate(rows, start=1):
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
            raise SystemExit(f"{sample_uid} expected X with 3 dimensions, got {x.shape}")
        _, grid_rows, grid_cols = x.shape
        finite_maps = compute_finite_source_maps(
            chiplets,
            package_width_mm=package_width_mm,
            package_height_mm=package_height_mm,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            length_scales_mm=args.length_scales_mm,
            quadrature_size=args.quadrature_size,
            kernel=args.kernel,
        )
        if not np.isfinite(finite_maps).all():
            raise SystemExit(f"{sample_uid} generated non-finite finite-source features")

        metadata_power_W = optional_float(row.get("total_power_W"))
        represented_power_W = sum(chiplet.power_W for chiplet in chiplets)
        if metadata_power_W is not None:
            power_error = abs(represented_power_W - metadata_power_W)
            power_errors.append(power_error)
            if power_error > args.power_tolerance_W:
                raise SystemExit(
                    f"{sample_uid} source power {represented_power_W:.6f} W does not match metadata "
                    f"{metadata_power_W:.6f} W within {args.power_tolerance_W}"
                )

        x_aug = np.concatenate([x, finite_maps], axis=0).astype(np.float32, copy=False)
        out_x_path = encoded_root / row["case_id"] / f"{sample_uid}_x_finite_source.npy"
        out_x_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_x_path, x_aug)

        for channel_index, stats in enumerate(feature_stats):
            stats.update(finite_maps[channel_index])

        new_row = dict(row)
        new_row["x_path"] = repo_relative(out_x_path)
        split_records.append(new_row)

        runtime_s = time.perf_counter() - start
        runtimes.append(runtime_s)
        jsonl_records.append(
            {
                **new_row,
                "finite_source": {
                    "layout_path": repo_relative(layout_path),
                    "power_path": repo_relative(power_path),
                    "kernel": args.kernel,
                    "length_scales_mm": args.length_scales_mm,
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
                "x_aug_path": out_x_path,
                "layout": layout,
                "chiplets": chiplets,
                "finite_maps": finite_maps,
                "power_density": x[0],
            }

        if index % 250 == 0:
            print(f"  generated {index}/{len(rows)} samples")

    write_outputs(
        out_root=out_root,
        fieldnames=fieldnames,
        records=split_records,
        jsonl_records=jsonl_records,
        source_root=source_root,
        source_channel_names=source_channel_names,
        finite_channel_names=finite_channel_names,
        output_channel_names=output_channel_names,
        length_scales_mm=args.length_scales_mm,
        quadrature_size=args.quadrature_size,
        kernel=args.kernel,
        feature_stats=feature_stats,
        runtimes=runtimes,
        power_errors=power_errors,
    )
    if args.save_diagnostic_plot and first_diagnostic is not None:
        save_diagnostic_plot(out_root / "finite_source_diagnostic.png", first_diagnostic, args.length_scales_mm)

    print("Finite-source feature dataset build complete")
    print(f"Samples: {len(split_records)}")
    print(f"Input channels: {len(source_channel_names)} -> {len(output_channel_names)}")
    print(f"Finite-source channels: {', '.join(finite_channel_names)}")
    print(f"Feature generation runtime/sample: {float(np.mean(runtimes)):.6f} s")
    print(f"Output: {out_root}")
    return 0


def validate_args(length_scales_mm: list[float], quadrature_size: int) -> None:
    if not length_scales_mm:
        raise SystemExit("--length-scales-mm must contain at least one value")
    if any(length <= 0.0 for length in length_scales_mm):
        raise SystemExit("all length scales must be positive")
    if quadrature_size <= 0:
        raise SystemExit("--quadrature-size must be positive")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for optional in ("prediction_path", "residual_path"):
        if optional not in fieldnames:
            fieldnames.append(optional)
        for row in rows:
            row.setdefault(optional, "")
    required = {"sample_uid", "original_sample_uid", "case_id", "dataset_source", "split", "x_path", "y_path"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"{path} missing required columns: {', '.join(missing)}")
    return fieldnames, rows


def read_source_channel_names(source_root: Path) -> list[str]:
    context_manifest = find_context_manifest(source_root)
    if context_manifest is None:
        first_x = None
        combined_index = source_root / "combined_encoded_index.csv"
        _, rows = read_csv_rows(combined_index)
        if rows:
            first_x = np.load(resolve_repo_path(rows[0]["x_path"]), mmap_mode="r")
        channel_count = int(first_x.shape[0]) if first_x is not None else 8
        return BASE_CHANNEL_NAMES + [f"context_channel_{idx}" for idx in range(8, channel_count)]

    manifest = json.loads(context_manifest.read_text(encoding="utf-8"))
    context_channels = list(manifest.get("context_channels", []))
    original_channels = int(manifest.get("original_channels", len(BASE_CHANNEL_NAMES)))
    names = BASE_CHANNEL_NAMES[:original_channels] + [str(name) for name in context_channels]
    output_channels = manifest.get("output_channels")
    if output_channels is not None and len(names) != int(output_channels):
        names.extend(f"context_channel_{idx}" for idx in range(len(names), int(output_channels)))
    return names


def find_context_manifest(source_root: Path) -> Path | None:
    current = source_root
    candidates = [
        source_root / "context_manifest.json",
        source_root.parent / "context_manifest.json",
        REPO_ROOT / "data/runs/benchmarks/dataset_v1_context_ablation/package_plus_power/context_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    while current != current.parent:
        candidate = current / "context_manifest.json"
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def source_paths_for_row(row: dict[str, str]) -> tuple[Path, Path]:
    explicit = (row.get("layout_path"), row.get("power_path"))
    if all(explicit):
        paths = tuple(resolve_repo_path(value) for value in explicit)
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return paths  # type: ignore[return-value]
    if row.get("source_dir"):
        source_dir = resolve_repo_path(row["source_dir"])
        layout_path = source_dir / "layout.json"
        power_path = source_dir / "power.yaml"
        missing = [path for path in (layout_path, power_path) if not path.exists()]
        if not missing:
            return layout_path, power_path
    dataset_source = row["dataset_source"]
    case_id = row["case_id"]
    original_uid = row["original_sample_uid"]
    sample_suffix = sample_dir_from_original_uid(original_uid, case_id)
    source_dir = REPO_ROOT / "data/runs/benchmarks" / dataset_source / case_id / sample_suffix / "source"
    layout_path = source_dir / "layout.json"
    power_path = source_dir / "power.yaml"
    missing = [path for path in (layout_path, power_path) if not path.exists()]
    if missing:
        raise SystemExit(f"{row['sample_uid']} missing source metadata: {', '.join(str(path) for path in missing)}")
    return layout_path, power_path


def sample_dir_from_original_uid(original_uid: str, case_id: str) -> str:
    prefix = f"{case_id}_"
    if not original_uid.startswith(prefix):
        raise SystemExit(f"original_sample_uid {original_uid!r} does not match case_id {case_id!r}")
    return original_uid[len(prefix) :]


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
            raise SystemExit(f"power file missing chiplet power for {name}")
        position = chiplet["position"]
        size = chiplet["size"]
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
        chiplets.append(source)
    if not chiplets:
        raise SystemExit("layout contains no chiplets")
    missing_layout = set(power_by_name) - {chiplet.name for chiplet in chiplets}
    if missing_layout:
        raise SystemExit(f"power file has chiplets not present in layout: {sorted(missing_layout)}")
    return chiplets


def active_power_map(power_data: dict[str, Any]) -> dict[str, float]:
    active = power_data.get("active_workload")
    workloads = power_data.get("workloads") or {}
    if active and active in workloads:
        return {str(name): float(value) for name, value in workloads[active].items()}
    if "chiplets" in power_data:
        return {str(name): float(value) for name, value in power_data["chiplets"].items()}
    raise SystemExit("power file has neither active workload powers nor chiplets powers")


def compute_finite_source_maps(
    chiplets: list[ChipletSource],
    *,
    package_width_mm: float,
    package_height_mm: float,
    grid_rows: int,
    grid_cols: int,
    length_scales_mm: list[float],
    quadrature_size: int,
    kernel: str,
) -> np.ndarray:
    x_centers = (np.arange(grid_cols, dtype=np.float64) + 0.5) * package_width_mm / grid_cols
    y_centers = (np.arange(grid_rows, dtype=np.float64) + 0.5) * package_height_mm / grid_rows
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    maps = np.zeros((len(length_scales_mm), grid_rows, grid_cols), dtype=np.float64)

    q_offsets = (np.arange(quadrature_size, dtype=np.float64) + 0.5) / quadrature_size
    for chiplet in chiplets:
        qx = chiplet.x_mm + q_offsets * chiplet.width_mm
        qy = chiplet.y_mm + q_offsets * chiplet.height_mm
        source_x, source_y = np.meshgrid(qx, qy)
        source_points = np.column_stack([source_x.reshape(-1), source_y.reshape(-1)])
        weight_W = chiplet.power_density_W_per_mm2 * chiplet.area_mm2 / float(source_points.shape[0])
        represented_power = weight_W * source_points.shape[0]
        if abs(represented_power - chiplet.power_W) > 1.0e-9:
            raise SystemExit(f"internal power quadrature error for {chiplet.name}")
        for point_x, point_y in source_points:
            dx = grid_x - point_x
            dy = grid_y - point_y
            r2 = dx * dx + dy * dy
            for channel, length in enumerate(length_scales_mm):
                if kernel == "softened_green":
                    response = 1.0 / np.sqrt(r2 + length * length)
                elif kernel == "screened_softened_green":
                    radius = np.sqrt(r2)
                    response = np.exp(-radius / length) / np.sqrt(r2 + length * length)
                else:  # pragma: no cover - argparse guards this
                    raise ValueError(kernel)
                maps[channel] += weight_W * response
    return maps.astype(np.float32)


def finite_channel_name(length_mm: float) -> str:
    text = f"{float(length_mm):g}".replace(".", "p")
    return f"finite_source_L{text}mm"


def write_outputs(
    *,
    out_root: Path,
    fieldnames: list[str],
    records: list[dict[str, str]],
    jsonl_records: list[dict[str, Any]],
    source_root: Path,
    source_channel_names: list[str],
    finite_channel_names: list[str],
    output_channel_names: list[str],
    length_scales_mm: list[float],
    quadrature_size: int,
    kernel: str,
    feature_stats: list["RunningMoments"],
    runtimes: list[float],
    power_errors: list[float],
) -> None:
    write_csv(out_root / "combined_encoded_index.csv", fieldnames, records)
    for split in ("train", "val", "test"):
        split_rows = [row for row in records if row["split"] == split]
        write_csv(out_root / f"{split}_index.csv", fieldnames, split_rows)
    write_jsonl(out_root / "combined_encoded_index.jsonl", jsonl_records)

    split_counts = Counter(row["split"] for row in records)
    channel_stats = {
        name: {
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.minimum,
            "max": stats.maximum,
        }
        for name, stats in zip(finite_channel_names, feature_stats)
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "out_root": repo_relative(out_root),
        "num_samples": len(records),
        "split_counts": {split: int(split_counts.get(split, 0)) for split in ("train", "val", "test")},
        "source_channels": len(source_channel_names),
        "finite_source_channels": len(finite_channel_names),
        "output_channels": len(output_channel_names),
        "channel_names": output_channel_names,
        "finite_source_channel_indices": {
            name: len(source_channel_names) + index for index, name in enumerate(finite_channel_names)
        },
        "finite_source_channel_stats": channel_stats,
        "length_scales_mm": length_scales_mm,
        "quadrature_size": quadrature_size,
        "kernel": {
            "name": kernel,
            "equation": "softened_green: K_L(r)=1/sqrt(r^2+L^2); screened_softened_green: exp(-r/L)/sqrt(r^2+L^2)",
            "units": "r and L are in millimeters; each quadrature weight is in watts.",
        },
        "geometry_source": "source/layout.json from each canonical clean sample directory",
        "power_source": "source/power.yaml active_workload if present, otherwise chiplets nominal map",
        "label_usage": "No HotSpot temperature labels are used to generate finite-source channels.",
        "runtime": {
            "feature_generation_total_s": float(sum(runtimes)),
            "feature_generation_mean_s_per_sample": float(np.mean(runtimes)) if runtimes else 0.0,
            "feature_generation_median_s_per_sample": float(np.median(runtimes)) if runtimes else 0.0,
        },
        "verification": {
            "represented_power_matches_metadata": True,
            "max_total_power_error_W": float(max(power_errors)) if power_errors else 0.0,
            "nonfinite_feature_values": 0,
            "split_membership_preserved_from_source": True,
            "large_tensor_copy_policy": "Only augmented X tensors are written. Y, physics predictions, and residual paths are reused.",
        },
    }
    (out_root / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root / "README.md", manifest)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Dataset v2 Clean Finite Source: package_plus_power

This index reuses the clean leakage-free split from:

`{manifest['source_root']}`

Only augmented X tensors are written. HotSpot targets, physics-v1 predictions,
and residual tensors are reused by path.

## Finite Rectangular-Source Features

Kernel:

`K_L(r) = 1 / sqrt(r^2 + L^2)`

for `--kernel softened_green`, with `r` and `L` in millimeters.

Each chiplet rectangle is integrated by deterministic quadrature over its exact
source rectangle from `source/layout.json`. Chiplet power comes from
`source/power.yaml`, using the active workload when present. The quadrature
weighting preserves total chiplet power in watts.

## Channels

- Source channels: {manifest['source_channels']}
- Added finite-source channels: {manifest['finite_source_channels']}
- Output channels: {manifest['output_channels']}
- Added names: {', '.join(manifest['finite_source_channel_indices'])}

Feature generation runtime/sample:
`{manifest['runtime']['feature_generation_mean_s_per_sample']:.6f} s`
"""
    path.write_text(text, encoding="utf-8")


def save_diagnostic_plot(path: Path, diagnostic: dict[str, Any], length_scales_mm: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"Skipping diagnostic plot because matplotlib is unavailable: {exc}")
        return

    finite_maps = diagnostic["finite_maps"]
    num_cols = 2 + finite_maps.shape[0]
    fig, axes = plt.subplots(1, num_cols, figsize=(3.2 * num_cols, 3.5), constrained_layout=True)
    axes[0].imshow(diagnostic["power_density"], origin="lower", cmap="magma")
    axes[0].set_title("power density")
    axes[1].imshow(diagnostic["power_density"], origin="lower", cmap="gray")
    axes[1].set_title("chiplet rectangles")
    layout = diagnostic["layout"]
    package_width, package_height = package_size_mm(layout)
    for chiplet in diagnostic["chiplets"]:
        rect = Rectangle(
            (chiplet.x_mm / package_width * 64, chiplet.y_mm / package_height * 64),
            chiplet.width_mm / package_width * 64,
            chiplet.height_mm / package_height * 64,
            fill=False,
            edgecolor="cyan",
            linewidth=1.0,
        )
        axes[1].add_patch(rect)
    for idx, length in enumerate(length_scales_mm):
        ax = axes[2 + idx]
        image = ax.imshow(finite_maps[idx], origin="lower", cmap="viridis")
        ax.set_title(f"L={length:g} mm")
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
        return float(self.total / self.count) if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 1.0
        variance = max(self.total_sq / self.count - self.mean * self.mean, 1.0e-12)
        return float(variance**0.5)


if __name__ == "__main__":
    raise SystemExit(main())
