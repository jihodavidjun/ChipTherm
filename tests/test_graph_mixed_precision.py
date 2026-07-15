#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.graph_models import ChipletMessagePassingGNN, rasterize_node_values  # noqa: E402


def make_graph(device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.randn(5, 6, dtype=torch.float32, device=device),
        "edge_features": torch.randn(8, 4, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(
            [
                [0, 1, 2, 0, 3, 4, 3, 4],
                [1, 2, 0, 2, 4, 3, 3, 4],
            ],
            dtype=torch.long,
            device=device,
        ),
        "node_batch": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long, device=device),
        "num_graphs": torch.tensor(2, dtype=torch.long, device=device),
        "chiplet_rects": torch.tensor(
            [
                [1.0, 1.0, 1.5, 1.0],
                [4.0, 2.0, 1.2, 1.4],
                [7.0, 5.0, 1.0, 1.0],
                [2.0, 2.0, 1.6, 1.2],
                [6.0, 4.5, 1.1, 1.3],
            ],
            dtype=torch.float32,
            device=device,
        ),
        "package_size": torch.tensor([[10.0, 8.0], [9.0, 7.0]], dtype=torch.float32, device=device),
    }


def run_graph_under_autocast(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    graph = make_graph(device)
    model = ChipletMessagePassingGNN(
        node_feature_dim=6,
        edge_feature_dim=4,
        hidden_dim=16,
        edge_hidden_dim=8,
        layers=2,
        raster_channels=4,
    ).to(device)
    context = torch.autocast(device_type=device.type, dtype=dtype) if device.type in {"cpu", "cuda"} else torch.no_grad()
    with context:
        outputs = model(graph, return_diagnostics=True)
        maps = rasterize_node_values(outputs["node_raster_values"], graph, height=16, width=16)
    assert torch.isfinite(outputs["graph_embedding"]).all()
    assert torch.isfinite(outputs["node_raster_values"]).all()
    assert torch.isfinite(maps).all()
    assert tuple(maps.shape) == (2, 4, 16, 16)


def test_cpu_bfloat16_autocast_graph_reductions_are_finite() -> None:
    run_graph_under_autocast(torch.device("cpu"), torch.bfloat16)


def test_manual_half_rasterizer_reductions_are_finite() -> None:
    graph = make_graph()
    node_values = torch.randn(5, 4, dtype=torch.float16)
    maps = rasterize_node_values(node_values, graph, height=16, width=16)
    assert maps.dtype == torch.float16
    assert torch.isfinite(maps).all()


def test_cuda_fp16_and_bfloat16_autocast_if_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    run_graph_under_autocast(device, torch.float16)
    if torch.cuda.is_bf16_supported():
        run_graph_under_autocast(device, torch.bfloat16)


if __name__ == "__main__":
    test_cpu_bfloat16_autocast_graph_reductions_are_finite()
    test_manual_half_rasterizer_reductions_are_finite()
    test_cuda_fp16_and_bfloat16_autocast_if_available()
    print("graph mixed-precision tests passed")
