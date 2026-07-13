#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.graph_models import build_geometry_raster_cache, rasterize_node_values


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ChipTherm graph rasterizer implementations.")
    parser.add_argument("--index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power/test_index.csv", type=Path)
    parser.add_argument("--out-json", default=REPO_ROOT / "outputs/phase2_cnn_gnn/rasterizer_benchmark.json", type=Path)
    parser.add_argument("--batch-sizes", nargs="+", default=[1, 8, 32, 64], type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--height", default=64, type=int)
    parser.add_argument("--width", default=64, type=int)
    parser.add_argument("--halo-decay-mm", default=4.0, type=float)
    parser.add_argument("--iterations", default=25, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    device = select_device(args.device)
    dataset = ChipThermDataset(args.index, return_graph=True, return_metadata=True)
    results: dict[str, Any] = {
        "index": str(args.index.resolve()),
        "device": str(device),
        "channels": args.channels,
        "height": args.height,
        "width": args.width,
        "halo_decay_mm": args.halo_decay_mm,
        "batch_sizes": {},
    }
    for batch_size in args.batch_sizes:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=chiptherm_collate)
        batch = next(iter(loader))
        graph = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch["graph"].items()}
        node_values = torch.randn(int(graph["node_batch"].shape[0]), args.channels, dtype=torch.float32, device=device)
        cache = build_geometry_raster_cache(
            graph,
            height=args.height,
            width=args.width,
            halo_decay_mm=args.halo_decay_mm,
            dtype=node_values.dtype,
            cache_key=f"batch{batch_size}",
        )
        batch_result = {
            "num_graphs": int(graph["num_graphs"].item()),
            "num_nodes": int(graph["node_batch"].shape[0]),
            "num_edges": int(graph["edge_features"].shape[0]),
            "legacy": time_rasterizer(node_values, graph, "legacy", args, device, cache=None),
            "vectorized": time_rasterizer(node_values, graph, "vectorized", args, device, cache=None),
            "vectorized_cached": time_rasterizer(node_values, graph, "vectorized", args, device, cache=cache),
        }
        legacy_mean = batch_result["legacy"]["mean_ms_per_batch"]
        vector_mean = batch_result["vectorized"]["mean_ms_per_batch"]
        cached_mean = batch_result["vectorized_cached"]["mean_ms_per_batch"]
        batch_result["speedup_vectorized_vs_legacy"] = legacy_mean / max(vector_mean, 1.0e-12)
        batch_result["speedup_cached_vs_legacy"] = legacy_mean / max(cached_mean, 1.0e-12)
        results["batch_sizes"][str(batch_size)] = batch_result
        print(
            f"batch={batch_size} legacy={legacy_mean:.3f}ms "
            f"vectorized={vector_mean:.3f}ms cached={cached_mean:.3f}ms "
            f"speedup={batch_result['speedup_vectorized_vs_legacy']:.2f}x"
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Output: {args.out_json}")
    return 0


def time_rasterizer(
    node_values: torch.Tensor,
    graph: dict[str, torch.Tensor],
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    cache: Any | None,
) -> dict[str, float]:
    samples: list[float] = []
    total_iterations = int(args.warmup) + int(args.iterations)
    for index in range(total_iterations):
        synchronize(device)
        start = time.perf_counter()
        output = rasterize_node_values(
            node_values,
            graph,
            height=args.height,
            width=args.width,
            halo_decay_mm=args.halo_decay_mm,
            mode=mode,
            cache=cache,
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        if index >= int(args.warmup):
            samples.append(elapsed)
        if output.numel() == 0:
            raise RuntimeError("empty rasterizer output")
    array = np.asarray(samples, dtype=np.float64)
    batch_size = int(graph["num_graphs"].item())
    return {
        "mean_ms_per_batch": float(array.mean() * 1000.0),
        "median_ms_per_batch": float(np.median(array) * 1000.0),
        "p95_ms_per_batch": float(np.percentile(array, 95) * 1000.0),
        "mean_ms_per_sample": float(array.mean() * 1000.0 / max(batch_size, 1)),
    }


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
