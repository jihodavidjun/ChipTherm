#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponseNormalizationStats,
    normalize_source_input,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, predict_source_rise  # noqa: E402
from scripts.build_full_source_superposition_base import (  # noqa: E402
    load_package_inputs,
    read_rows,
    resolve_path,
    select_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit numerical recomputation differences for one source-superposition map.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--sample-uid", required=True)
    parser.add_argument("--package-batch-size", default=8, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--out-json", default=None, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    index_path = source_root / f"{args.split}_index.csv"
    rows = read_rows(index_path)
    row_index = next((index for index, row in enumerate(rows) if row["sample_uid"] == args.sample_uid), None)
    if row_index is None:
        raise SystemExit(f"sample_uid {args.sample_uid} not found in {index_path}")
    batch_start = (row_index // args.package_batch_size) * args.package_batch_size
    generation_rows = rows[batch_start : batch_start + args.package_batch_size]
    position = row_index - batch_start
    saved = np.load(resolve_path(rows[row_index]["source_superposition_base_path"], index_path.parent)).astype(np.float32, copy=False)

    devices = [select_device(args.device)]
    if args.include_cpu and devices[0].type != "cpu":
        devices.append(torch.device("cpu"))
    report: dict[str, Any] = {
        "source_root": str(source_root),
        "split": args.split,
        "sample_uid": args.sample_uid,
        "row_index": row_index,
        "generation_batch_start": batch_start,
        "package_batch_size": args.package_batch_size,
        "source_batch_size": args.source_batch_size,
        "source_names": None,
        "devices": {},
    }

    for device in devices:
        model, stats = load_model(args.checkpoint, device)
        generation_packages = [load_package_inputs(row, index_path) for row in generation_rows]
        validator_package = [load_package_inputs(rows[row_index], index_path)]
        report["source_names"] = validator_package[0]["source_names"]

        gen_maps, gen_sources = compute_maps_and_sources(generation_packages, model, stats, args.source_batch_size, device, np.float64)
        val_maps, val_sources = compute_maps_and_sources(validator_package, model, stats, args.source_batch_size, device, np.float64)
        val_maps_f32, _ = compute_maps_and_sources(validator_package, model, stats, args.source_batch_size, device, np.float32)

        device_report: dict[str, Any] = {
            "generation_context_vs_saved": diff_summary(gen_maps[position], saved),
            "validator_context_vs_saved": diff_summary(val_maps[0], saved),
            "generation_context_vs_validator_context": diff_summary(gen_maps[position], val_maps[0]),
            "float32_accum_vs_float64_accum_validator": diff_summary(val_maps_f32[0], val_maps[0]),
            "source_level_generation_vs_validator": source_diff_summary(gen_sources[position], val_sources[0]),
            "validator_batch_size_sensitivity": {},
        }
        for batch_size in (1, 2, 4, 8, 16, 32, 64, 128):
            maps, _ = compute_maps_and_sources(validator_package, model, stats, batch_size, device, np.float64)
            device_report["validator_batch_size_sensitivity"][str(batch_size)] = diff_summary(maps[0], val_maps[0])
        report["devices"][device.type] = device_report

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)
    return 0


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, SourceResponseNormalizationStats]:
    payload = torch.load(checkpoint.expanduser().resolve(), map_location=device, weights_only=False)
    stats = SourceResponseNormalizationStats.from_dict(payload["normalization"])
    model = build_source_response_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, stats


@torch.no_grad()
def compute_maps_and_sources(
    packages: list[dict[str, Any]],
    model: torch.nn.Module,
    stats: SourceResponseNormalizationStats,
    source_batch_size: int,
    device: torch.device,
    accumulation_dtype: type[np.float32] | type[np.float64],
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    flat_inputs: list[np.ndarray] = []
    flat_powers: list[float] = []
    flat_package_ids: list[int] = []
    flat_source_ids: list[int] = []
    for package_index, package in enumerate(packages):
        for source_index, (source_input, source_power) in enumerate(zip(package["source_inputs"], package["source_powers"], strict=True)):
            flat_inputs.append(source_input)
            flat_powers.append(float(source_power))
            flat_package_ids.append(package_index)
            flat_source_ids.append(source_index)
    sums = [np.zeros((64, 64), dtype=accumulation_dtype) for _ in packages]
    sources: list[list[tuple[int, np.ndarray]]] = [[] for _ in packages]
    for start in range(0, len(flat_inputs), source_batch_size):
        stop = min(start + source_batch_size, len(flat_inputs))
        x = torch.from_numpy(np.stack(flat_inputs[start:stop]).astype(np.float32, copy=False)).to(device)
        power = torch.tensor(flat_powers[start:stop], dtype=torch.float32, device=device)
        pred_unit = unnormalize_source_prediction(model(normalize_source_input(x, stats)), stats)
        pred_rise = predict_source_rise(pred_unit, power).detach().cpu().numpy()
        for local_index, rise in enumerate(pred_rise):
            global_index = start + local_index
            package_id = flat_package_ids[global_index]
            source_id = flat_source_ids[global_index]
            sources[package_id].append((source_id, rise.astype(np.float32, copy=False)))
            sums[package_id] += rise.astype(accumulation_dtype, copy=False)
    maps = [np.asarray(float(package["ambient_K"]) + rise_sum, dtype=np.float32) for package, rise_sum in zip(packages, sums, strict=True)]
    ordered_sources: list[list[np.ndarray]] = []
    for entries in sources:
        ordered_sources.append([entry[1] for entry in sorted(entries, key=lambda item: item[0])])
    return maps, ordered_sources


def diff_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = left.astype(np.float64) - right.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        "max_abs_K": float(np.max(abs_diff)),
        "mean_abs_K": float(np.mean(abs_diff)),
        "rmse_K": float(np.sqrt(np.mean(diff * diff))),
    }


def source_diff_summary(left: list[np.ndarray], right: list[np.ndarray]) -> dict[str, Any]:
    per_source = [diff_summary(a, b) for a, b in zip(left, right, strict=True)]
    return {
        "num_sources": len(per_source),
        "max_source_max_abs_K": float(max(item["max_abs_K"] for item in per_source)) if per_source else 0.0,
        "mean_source_mean_abs_K": float(np.mean([item["mean_abs_K"] for item in per_source])) if per_source else 0.0,
        "nonzero_source_count": sum(1 for item in per_source if item["max_abs_K"] > 0.0),
        "per_source": per_source,
    }


if __name__ == "__main__":
    raise SystemExit(main())
