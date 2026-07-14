#!/usr/bin/env python3
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.models import build_model


def base_config(architecture: str = "miniunet_refine_conditioned_decomposed_global") -> dict[str, object]:
    return {
        "architecture": architecture,
        "input_channels": 12,
        "output_channels": 1,
        "base_channels": 8,
        "depth": 3,
        "refine_channels": 8,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1, 2, 3],
        "refinement_channel_names": ["power", "occ", "x", "y"],
        "metadata_dim": 5,
        "metadata_hidden_dim": 8,
        "metadata_embedding_dim": 8,
        "physics_input_mode": "source_superposition_v1",
        "global_branch_channel_indices": [0, 1, 2, 3, 11],
        "global_branch_channel_names": ["power", "occ", "x", "y", "source_base"],
        "global_hidden_channels": 8,
        "global_blocks": 1,
        "global_pool_size": 8,
    }


def test_global_branch_starts_as_exact_noop() -> None:
    torch.manual_seed(0)
    model = build_model(base_config())
    model.eval()
    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    with torch.no_grad():
        outputs = model(x, metadata)
    assert outputs["centered_field"].shape == (2, 64, 64)
    assert outputs["global_correction_field"].shape == (2, 64, 64)
    assert torch.allclose(outputs["global_correction_field"], torch.zeros_like(outputs["global_correction_field"]))
    assert torch.allclose(outputs["centered_field"], outputs["local_centered_field"], atol=1.0e-6)
    assert torch.allclose(outputs["centered_field"].mean(dim=(-2, -1)), torch.zeros(2), atol=1.0e-6)
    assert model.config()["global_branch_parameter_count"] > 0


def test_global_branch_output_layer_receives_gradients() -> None:
    torch.manual_seed(1)
    model = build_model(base_config())
    model.train()
    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    outputs = model(x, metadata)
    target = torch.randn_like(outputs["centered_field"])
    loss = torch.nn.functional.smooth_l1_loss(outputs["centered_field"], target)
    loss.backward()
    grad = model.global_branch.output_projection.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0


def test_global_graph_freezes_entire_cnn_branch() -> None:
    config = base_config("miniunet_refine_conditioned_decomposed_global_graph")
    config.update(
        {
            "graph_node_feature_dim": 6,
            "graph_edge_feature_dim": 4,
            "graph_hidden_dim": 8,
            "graph_edge_hidden_dim": 8,
            "graph_layers": 1,
            "graph_raster_channels": 4,
            "freeze_cnn": True,
        }
    )
    model = build_model(config)
    assert all(not parameter.requires_grad for parameter in model.cnn_model.parameters())
    assert any(parameter.requires_grad for parameter in model.graph_model.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion_head.parameters())
    cfg = model.config()
    assert cfg["architecture"] == "miniunet_refine_conditioned_decomposed_global_graph"
    assert cfg["global_branch_enabled"] is True
    assert hasattr(model.cnn_model, "global_branch")


def test_global_branch_checkpoint_round_trip() -> None:
    torch.manual_seed(2)
    config = base_config()
    model = build_model(config)
    x = torch.randn(1, 12, 64, 64)
    metadata = torch.randn(1, 5)
    with torch.no_grad():
        before = model(x, metadata)["centered_field"]
    buffer = BytesIO()
    torch.save({"model_config": model.config(), "model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = build_model(payload["model_config"])
    restored.load_state_dict(payload["model_state_dict"])
    with torch.no_grad():
        after = restored(x, metadata)["centered_field"]
    assert torch.allclose(before, after, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    test_global_branch_starts_as_exact_noop()
    test_global_branch_output_layer_receives_gradients()
    test_global_graph_freezes_entire_cnn_branch()
    test_global_branch_checkpoint_round_trip()
    print("global branch model tests passed")
