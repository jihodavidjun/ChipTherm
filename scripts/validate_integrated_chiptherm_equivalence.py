#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from chiptherm.ml.integrated_inference import (  # noqa: E402
    IntegratedChipThermModel,
    resolve_declared_path,
    sha256_file,
)
from evaluate_integrated_chiptherm import (  # noqa: E402
    cached_rows_for_selected,
    load_target_batch,
    read_rows,
    select_device,
    select_rows,
)


MAX_OUTPUT_DIFF_K = 1e-5
MAX_AGGREGATE_MAE_DIFF_K = 1e-6
CACHED_HARD_TOLERANCE_K = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict reference/optimized integrated ChipTherm equivalence gate.")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--residual-checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata-root", default=None, type=Path)
    parser.add_argument("--graph-root", default=None, type=Path)
    parser.add_argument("--compare-cached-index", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-samples", default=4, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    rows = select_rows(read_rows(args.index), max_samples=args.max_samples, mode="stratified")
    checkpoint_hashes_before = {
        "source": sha256_file(args.source_checkpoint),
        "residual": sha256_file(args.residual_checkpoint),
    }
    reference = IntegratedChipThermModel(
        source_checkpoint=args.source_checkpoint,
        residual_checkpoint=args.residual_checkpoint,
        data_root=args.data_root,
        device=device,
        execution_mode="reference",
        deterministic=True,
        precision="fp32",
    )
    optimized = IntegratedChipThermModel(
        source_checkpoint=args.source_checkpoint,
        residual_checkpoint=args.residual_checkpoint,
        data_root=args.data_root,
        device=device,
        execution_mode="optimized",
        deterministic=True,
        precision="fp32",
    )
    dataset = ChipThermDataset(
        args.index,
        target="residual",
        return_metadata=True,
        metadata_root=args.metadata_root,
        graph_root=args.graph_root,
        return_graph=reference.graph_enabled,
        physical_representation=str(
            reference.residual_config.get("physical_representation", "dimensional")
        ),
        load_temperature=False,
        load_physics=False,
        load_residual=False,
    )
    dataset.rows = rows
    samples = [dataset[index] for index in range(len(dataset))]
    batch = chiptherm_collate(samples) if reference.graph_enabled else torch.utils.data.default_collate(samples)
    with torch.no_grad():
        reference_result = reference.predict_batch(
            batch,
            rows,
            source_batch_size=args.source_batch_size,
            return_source_responses=True,
        )
    with torch.inference_mode():
        optimized_result = optimized.predict_batch(
            batch,
            rows,
            source_batch_size=args.source_batch_size,
            return_source_responses=True,
        )
    targets = load_target_batch(rows, data_root=args.data_root).to(device)
    comparisons = compare_results(reference_result, optimized_result)
    reference_sample_mae = (
        reference_result["final_temperature_K"] - targets
    ).abs().mean(dim=(-2, -1))
    optimized_sample_mae = (
        optimized_result["final_temperature_K"] - targets
    ).abs().mean(dim=(-2, -1))
    comparisons["per_sample_mae_against_target_K"] = tensor_difference(
        reference_sample_mae,
        optimized_sample_mae,
    )
    reference_mae = float((reference_result["final_temperature_K"] - targets).abs().mean().item())
    optimized_mae = float((optimized_result["final_temperature_K"] - targets).abs().mean().item())
    aggregate_mae_diff = abs(reference_mae - optimized_mae)
    hotspot_location_equal = hotspot_locations(reference_result) == hotspot_locations(optimized_result)
    checkpoint_hashes_after = {
        "source": sha256_file(args.source_checkpoint),
        "residual": sha256_file(args.residual_checkpoint),
    }
    strict_ok = (
        all(item["max_abs_diff_K"] <= MAX_OUTPUT_DIFF_K for item in comparisons.values())
        and aggregate_mae_diff <= MAX_AGGREGATE_MAE_DIFF_K
        and hotspot_location_equal
        and checkpoint_hashes_before == checkpoint_hashes_after
    )

    cached_audit: dict[str, Any] | None = None
    if args.compare_cached_index is not None:
        cached_rows = cached_rows_for_selected(read_rows(args.compare_cached_index), rows)
        cached_maps = torch.from_numpy(
            np.stack(
                [
                    np.load(
                        resolve_declared_path(row["source_superposition_base_path"], data_root=args.data_root)
                    ).astype(np.float32, copy=False)
                    for row in cached_rows
                ]
            )
        )
        cached_result = reference.residual_from_base(batch, cached_maps)
        cached_comparisons = {
            "source_superposition_base_K": tensor_difference(
                reference_result["source_superposition_base_K"], cached_maps.to(device)
            ),
            "final_temperature_K": tensor_difference(
                reference_result["final_temperature_K"], cached_result["final_temperature_K"]
            ),
        }
        cached_audit = {
            "purpose": "Legacy cached CUDA accumulation audit; not the optimization acceptance gate.",
            "hard_tolerance_K": CACHED_HARD_TOLERANCE_K,
            "comparisons": cached_comparisons,
            "ok": all(
                item["max_abs_diff_K"] <= CACHED_HARD_TOLERANCE_K
                for item in cached_comparisons.values()
            ),
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_uids": [row["sample_uid"] for row in rows],
        "thresholds": {
            "reference_optimized_max_abs_K": MAX_OUTPUT_DIFF_K,
            "aggregate_mae_difference_K": MAX_AGGREGATE_MAE_DIFF_K,
        },
        "reference_final_mae_K": reference_mae,
        "optimized_final_mae_K": optimized_mae,
        "aggregate_mae_difference_K": aggregate_mae_diff,
        "comparisons": comparisons,
        "hotspot_location_equal": hotspot_location_equal,
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "checkpoints_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "cached_path_audit": cached_audit,
        "passed": strict_ok,
    }
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "integrated_equivalence_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inventory = {
        "accepted": ["torch.inference_mode"] if strict_ok else [],
        "rejected": [],
        "pending_gt_measurement": ["pinned_memory", "non_blocking_transfer"],
        "not_authoritative": ["fp16", "bf16", "tf32", "channels_last", "device_summation"],
        "acceptance_passed": strict_ok,
    }
    (out_dir / "optimization_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not strict_ok:
        raise SystemExit("strict reference/optimized equivalence gate failed")
    return 0


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, float]]:
    pairs: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
        "source_responses_K": (left.get("source_responses_K"), right.get("source_responses_K")),
        "source_superposition_base_K": (
            left["source_superposition_base_K"],
            right["source_superposition_base_K"],
        ),
        "scalar_mean_correction_K": (
            left["outputs"]["mean_rise"],
            right["outputs"]["mean_rise"],
        ),
        "centered_spatial_residual_K": (
            left["outputs"]["centered_field"],
            right["outputs"]["centered_field"],
        ),
        "final_temperature_K": (left["final_temperature_K"], right["final_temperature_K"]),
        "final_temperature_host_K": (
            left["final_temperature_host_K"],
            right["final_temperature_host_K"],
        ),
    }
    result: dict[str, dict[str, float]] = {}
    for name, (left_value, right_value) in pairs.items():
        if left_value is None or right_value is None:
            raise ValueError(f"equivalence component {name} was not returned")
        result[name] = tensor_difference(left_value, right_value)
    left_peak = left["final_temperature_K"].amax(dim=(-2, -1))
    right_peak = right["final_temperature_K"].amax(dim=(-2, -1))
    result["hotspot_temperature_K"] = tensor_difference(left_peak, right_peak)
    return result


def tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_cpu = left.detach().cpu().float()
    right_cpu = right.detach().cpu().float()
    if left_cpu.shape != right_cpu.shape:
        raise ValueError(f"shape mismatch: {tuple(left_cpu.shape)} != {tuple(right_cpu.shape)}")
    if not torch.isfinite(left_cpu).all() or not torch.isfinite(right_cpu).all():
        raise ValueError("non-finite value in equivalence comparison")
    diff = (left_cpu - right_cpu).abs()
    return {
        "max_abs_diff_K": float(diff.max().item()),
        "mean_abs_diff_K": float(diff.mean().item()),
        "rmse_diff_K": float(torch.sqrt(torch.mean(diff.square())).item()),
    }


def hotspot_locations(result: dict[str, Any]) -> list[int]:
    maps = result["final_temperature_K"].detach().cpu()
    return [int(value) for value in maps.reshape(maps.shape[0], -1).argmax(dim=1).tolist()]


if __name__ == "__main__":
    raise SystemExit(main())
