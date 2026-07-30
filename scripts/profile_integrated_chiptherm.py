#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate  # noqa: E402
from chiptherm.ml.integrated_inference import IntegratedChipThermModel  # noqa: E402
from evaluate_integrated_chiptherm import read_rows, select_device, select_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Operator-level profile of uncached integrated ChipTherm inference.")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--residual-checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata-root", default=None, type=Path)
    parser.add_argument("--graph-root", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", default="reference", choices=["reference", "optimized"])
    parser.add_argument("--package-batch-size", default=8, type=int)
    parser.add_argument("--source-batch-size", default=64, type=int)
    parser.add_argument("--warmup-batches", default=2, type=int)
    parser.add_argument("--profile-batches", default=5, type=int)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps", "auto"])
    args = parser.parse_args()

    if args.package_batch_size <= 0 or args.source_batch_size <= 0:
        raise SystemExit("batch sizes must be positive")
    device = select_device(args.device)
    model = IntegratedChipThermModel(
        source_checkpoint=args.source_checkpoint,
        residual_checkpoint=args.residual_checkpoint,
        data_root=args.data_root,
        device=device,
        execution_mode=args.mode,
        precision="fp32",
        non_blocking_transfer=False,
    )
    max_samples = args.package_batch_size * (args.warmup_batches + args.profile_batches)
    rows = select_rows(read_rows(args.index), max_samples=max_samples, mode="stratified")
    dataset = ChipThermDataset(
        args.index,
        target="residual",
        return_metadata=True,
        metadata_root=args.metadata_root,
        graph_root=args.graph_root,
        return_graph=model.graph_enabled,
        physical_representation=str(model.residual_config.get("physical_representation", "dimensional")),
        load_temperature=False,
        load_physics=False,
        load_residual=False,
    )
    dataset.rows = rows
    loader = DataLoader(
        dataset,
        batch_size=args.package_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda" and args.mode == "optimized",
        collate_fn=chiptherm_collate if model.graph_enabled else None,
    )
    batches = list(loader)
    if len(batches) <= args.warmup_batches:
        raise SystemExit("not enough batches for requested warmup/profile schedule")
    offset = 0
    with torch.inference_mode():
        for batch in batches[: args.warmup_batches]:
            count = int(batch["x"].shape[0])
            model.predict_batch(
                batch,
                rows[offset : offset + count],
                source_batch_size=args.source_batch_size,
            )
            offset += count
    model.device_synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    profile_batches = batches[args.warmup_batches : args.warmup_batches + args.profile_batches]
    stage_totals: dict[str, float] = {}
    source_counts = 0
    package_count = 0
    start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            for batch in profile_batches:
                count = int(batch["x"].shape[0])
                with record_function("chiptherm_integrated_inference"):
                    result = model.predict_batch(
                        batch,
                        rows[offset : offset + count],
                        source_batch_size=args.source_batch_size,
                        profile_components=True,
                    )
                for name, value in result["timings"].items():
                    stage_totals[name] = stage_totals.get(name, 0.0) + float(value)
                source_counts += sum(result["source_counts"])
                package_count += count
                offset += count
                prof.step()
    model.device_synchronize()
    host_elapsed = time.perf_counter() - start

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=100)
    (out_dir / "profiler_summary.txt").write_text(table + "\n", encoding="utf-8")
    prof.export_chrome_trace(str(out_dir / "profiler_trace.json"))
    total_stage = sum(stage_totals.values())
    summary = {
        "mode": args.mode,
        "device": str(device),
        "package_count": package_count,
        "source_count": source_counts,
        "mean_sources_per_package": source_counts / max(package_count, 1),
        "host_to_host_profiled_s": host_elapsed,
        "host_to_host_s_per_package": host_elapsed / max(package_count, 1),
        "stage_totals_s": stage_totals,
        "stage_percentages": {
            name: 100.0 * value / max(total_stage, 1e-12)
            for name, value in sorted(stage_totals.items())
        },
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "trace_note": "The Chrome trace may be large; preserve it as a selected profiler artifact.",
    }
    (out_dir / "profiler_runtime_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(table)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
