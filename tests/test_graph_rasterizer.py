#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.graph_models import (
    build_geometry_raster_cache,
    rasterize_node_values,
)
from chiptherm.ml.models import build_model
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input
from scripts.evaluate_residual_cnn import prepare_graph_batch, reconstruct_decomposed_temperature


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate legacy and vectorized graph rasterizers.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", default=REPO_ROOT / "outputs/phase2_frozen_graph/context_only_seed1/checkpoints/best.pt", type=Path)
    parser.add_argument("--index", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_graph/package_plus_power/test_index.csv", type=Path)
    parser.add_argument("--model-batch-size", default=4, type=int)
    parser.add_argument("--tolerance", default=1.0e-5, type=float)
    args = parser.parse_args()
    device = select_device(args.device)
    max_diffs: list[float] = []
    grad_diffs: list[float] = []
    for graph in synthetic_graphs(device):
        output_diff, grad_diff = compare_graph(graph, device=device)
        max_diffs.append(output_diff)
        grad_diffs.append(grad_diff)
    print(f"synthetic output max abs diff: {max(max_diffs):.8g}")
    print(f"synthetic grad max abs diff: {max(grad_diffs):.8g}")
    limit = args.tolerance * (10.0 if device.type == "cuda" else 1.0)
    if max(max_diffs) > limit or max(grad_diffs) > limit:
        raise SystemExit(f"rasterizer equivalence failed tolerance {limit}")
    if args.checkpoint.exists() and args.index.exists():
        model_diff = compare_checkpoint_batch(args.checkpoint, args.index, args.model_batch_size, device)
        print("checkpoint batch max abs diffs:", model_diff)
        if model_diff and max(model_diff.values()) > max(limit, 1.0e-4):
            raise SystemExit("checkpoint rasterizer comparison failed")
    else:
        print("checkpoint comparison skipped; checkpoint or index not found")
    print("graph rasterizer equivalence checks passed")
    return 0


def synthetic_graphs(device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [
        make_graph([[(10.0, 10.0, 6.0, 6.0)]], [(64.0, 64.0)], device),
        make_graph([[(0.0, 0.0, 4.0, 4.0), (20.0, 15.0, 8.0, 3.0), (50.0, 50.0, 10.0, 10.0)]], [(64.0, 64.0)], device),
        make_graph(
            [
                [(0.0, 0.0, 2.0, 2.0), (30.0, 30.0, 20.0, 20.0)],
                [(5.0, 5.0, 0.5, 0.5), (70.0, 40.0, 25.0, 12.0), (90.0, 80.0, 5.0, 5.0)],
            ],
            [(64.0, 64.0), (128.0, 96.0)],
            device,
        ),
    ]


def make_graph(
    rects_by_graph: list[list[tuple[float, float, float, float]]],
    package_sizes: list[tuple[float, float]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rects: list[list[float]] = []
    node_batch: list[int] = []
    for graph_index, rects_for_graph in enumerate(rects_by_graph):
        for rect in rects_for_graph:
            rects.append([float(value) for value in rect])
            node_batch.append(graph_index)
    return {
        "chiplet_rects": torch.tensor(rects, dtype=torch.float32, device=device),
        "node_batch": torch.tensor(node_batch, dtype=torch.long, device=device),
        "package_size": torch.tensor(package_sizes, dtype=torch.float32, device=device),
        "num_graphs": torch.tensor(len(package_sizes), dtype=torch.long, device=device),
    }


def compare_graph(graph: dict[str, torch.Tensor], *, device: torch.device) -> tuple[float, float]:
    node_count = int(graph["chiplet_rects"].shape[0])
    channels = 5
    torch.manual_seed(123)
    values_legacy = torch.randn(node_count, channels, dtype=torch.float32, device=device, requires_grad=True)
    values_vector = values_legacy.detach().clone().requires_grad_(True)
    legacy = rasterize_node_values(values_legacy, graph, mode="legacy", height=64, width=64, halo_decay_mm=4.0)
    vector = rasterize_node_values(values_vector, graph, mode="vectorized", height=64, width=64, halo_decay_mm=4.0)
    cache = build_geometry_raster_cache(graph, height=64, width=64, halo_decay_mm=4.0)
    cached = rasterize_node_values(values_vector, graph, mode="vectorized", height=64, width=64, halo_decay_mm=4.0, cache=cache)
    output_diff = max(float((legacy - vector).abs().max().item()), float((vector - cached).abs().max().item()))
    legacy_loss = (legacy * legacy).mean()
    vector_loss = (vector * vector).mean()
    legacy_loss.backward()
    vector_loss.backward()
    grad_diff = float((values_legacy.grad - values_vector.grad).abs().max().item())
    return output_diff, grad_diff


@torch.no_grad()
def compare_checkpoint_batch(checkpoint_path: Path, index_path: Path, batch_size: int, device: torch.device) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = NormalizationStats(**checkpoint["normalization"])
    dataset = ChipThermDataset(index_path, return_graph=True, return_metadata=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=chiptherm_collate)
    batch = next(iter(loader))
    physics_mode = str(checkpoint["model_config"].get("physics_input_mode", "v1"))
    actual_channels = int(batch["x"].shape[1]) + (1 if physics_mode in {"v1", "gated_v1"} else 0)
    expected_channels = int(checkpoint["model_config"].get("input_channels", actual_channels))
    if actual_channels != expected_channels:
        print(f"checkpoint comparison skipped; index gives {actual_channels} channels but checkpoint expects {expected_channels}")
        return {}
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x = batch["x"].to(device)
    physics = batch["physics"].to(device)
    ambient = batch["ambient_K"].to(device).float()
    metadata = build_metadata_input(batch.get("metadata_vector"), stats)
    if metadata is not None:
        metadata = metadata.to(device)
    graph = prepare_graph_batch(batch, True, checkpoint["model_config"].get("graph_normalization"), device)
    model_input = build_model_input(x, physics, stats, physics_input_mode=physics_mode)
    model.graph_rasterizer_mode = "legacy"
    legacy = model(model_input, metadata, graph, return_diagnostics=True, ambient=ambient)
    legacy_temp = reconstruct_decomposed_temperature(legacy, ambient)
    model.graph_rasterizer_mode = "vectorized"
    vector = model(model_input, metadata, graph, return_diagnostics=True, ambient=ambient)
    vector_temp = reconstruct_decomposed_temperature(vector, ambient)
    return {
        "final_temperature": float((legacy_temp - vector_temp).abs().max().item()),
        "graph_correction": float((legacy["graph_correction_field"] - vector["graph_correction_field"]).abs().max().item()),
        "centered_field": float((legacy["centered_field"] - vector["centered_field"]).abs().max().item()),
        "mean_rise": float((legacy["mean_rise"] - vector["mean_rise"]).abs().max().item()),
    }


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
