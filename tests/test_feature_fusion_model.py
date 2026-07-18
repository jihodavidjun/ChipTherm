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

from chiptherm.ml.models import build_model  # noqa: E402


def base_config(
    architecture: str = "miniunet_refine_conditioned_decomposed_feature_fusion",
) -> dict[str, object]:
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


def small_graph(batch_size: int = 2) -> dict[str, torch.Tensor]:
    node_features = []
    edge_features = []
    edge_index = []
    node_batch = []
    chiplet_rects = []
    package_size = []
    node_offset = 0
    for graph_index in range(batch_size):
        package_size.append([10.0 + graph_index, 8.0 + graph_index])
        graph_nodes = [
            [2.0, 2.0, 1.5, 1.0, 1.5, 1.5],
            [7.0, 5.0, 1.0, 1.2, 1.2, 0.833333],
        ]
        node_features.extend(graph_nodes)
        node_batch.extend([graph_index, graph_index])
        chiplet_rects.extend([[1.25, 1.5, 1.5, 1.0], [6.5, 4.4, 1.0, 1.2]])
        edge_index.extend([[node_offset, node_offset + 1], [node_offset + 1, node_offset]])
        edge_features.extend([[5.0, 3.0, 5.83, 0.17], [-5.0, -3.0, 5.83, 0.17]])
        node_offset += 2
    return {
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "edge_features": torch.tensor(edge_features, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        "node_batch": torch.tensor(node_batch, dtype=torch.long),
        "num_graphs": torch.tensor(batch_size, dtype=torch.long),
        "chiplet_rects": torch.tensor(chiplet_rects, dtype=torch.float32),
        "package_size": torch.tensor(package_size, dtype=torch.float32),
    }


def test_feature_fusion_forward_shapes_and_zero_mean() -> None:
    torch.manual_seed(0)
    model = build_model(base_config())
    model.eval()
    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    with torch.no_grad():
        outputs = model(x, metadata, return_diagnostics=True)
        disabled = model(x, metadata, return_diagnostics=True, disabled_fusion_scales=("all",))
    assert outputs["centered_field"].shape == (2, 64, 64)
    assert outputs["coarse_centered_field"].shape == (2, 64, 64)
    assert torch.allclose(outputs["centered_field"].mean(dim=(-2, -1)), torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(disabled["centered_field"].mean(dim=(-2, -1)), torch.zeros(2), atol=1.0e-6)
    assert float(outputs["global_fusion_enabled_16"].mean().item()) == 1.0
    assert float(disabled["global_fusion_enabled_16"].mean().item()) == 0.0
    assert outputs["global_feature_16_abs_mean"].shape == (2,)
    assert outputs["global_feature_32_abs_mean"].shape == (2,)
    assert outputs["global_feature_64_abs_mean"].shape == (2,)
    assert model.config()["feature_fusion_parameter_count"] > 0


def test_feature_fusion_residual_path_receives_gradients() -> None:
    torch.manual_seed(1)
    model = build_model(base_config())
    model.train()
    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    outputs = model(x, metadata)
    target = torch.randn_like(outputs["centered_field"])
    loss = torch.nn.functional.smooth_l1_loss(outputs["centered_field"], target)
    loss.backward()
    grad = model.coarse_model.fuse16.delta.conv2.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum().item()) > 0.0


def test_feature_fusion_checkpoint_round_trip() -> None:
    torch.manual_seed(2)
    model = build_model(base_config())
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


def test_feature_fusion_graph_freezes_entire_cnn_branch_and_forwards() -> None:
    torch.manual_seed(3)
    config = base_config("miniunet_refine_conditioned_decomposed_feature_fusion_graph")
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
    assert cfg["architecture"] == "miniunet_refine_conditioned_decomposed_feature_fusion_graph"
    assert cfg["feature_fusion_enabled"] is True

    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    outputs = model(x, metadata, small_graph(2), return_diagnostics=True)
    assert outputs["centered_field"].shape == (2, 64, 64)
    assert outputs["graph_correction_field"].shape == (2, 64, 64)
    assert torch.allclose(outputs["centered_field"].mean(dim=(-2, -1)), torch.zeros(2), atol=1.0e-6)


def test_feature_fusion_graph_supports_frozen_residual_resistance_cnn() -> None:
    torch.manual_seed(4)
    config = base_config("miniunet_refine_conditioned_decomposed_feature_fusion_graph")
    config.update(
        {
            "graph_node_feature_dim": 6,
            "graph_edge_feature_dim": 4,
            "graph_hidden_dim": 8,
            "graph_edge_hidden_dim": 8,
            "graph_layers": 1,
            "graph_raster_channels": 4,
            "freeze_cnn": True,
            "mean_head_mode": "residual_resistance",
            "delta_R_eff_target_mean_K_per_W": 0.25,
            "delta_R_eff_target_std_K_per_W": 0.5,
            "graph_mean_correction_enabled": False,
        }
    )
    model = build_model(config)
    assert model.mean_head_mode == "residual_resistance"
    assert model.cnn_model.mean_head_mode == "residual_resistance"
    assert model.graph_mean_correction_enabled is False
    assert all(not parameter.requires_grad for parameter in model.cnn_model.parameters())

    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    total_power = torch.tensor([10.0, 20.0])
    graph = small_graph(2)
    with torch.no_grad():
        outputs = model(x, metadata, graph, return_diagnostics=True, total_power_W=total_power)

    assert torch.allclose(outputs["graph_mean_delta"], torch.zeros(2), atol=0.0, rtol=0.0)
    assert torch.allclose(outputs["mean_rise"], outputs["cnn_mean_rise"], atol=0.0, rtol=0.0)
    assert outputs["delta_R_eff"].shape == (2,)
    assert torch.allclose(outputs["mean_rise"], total_power * outputs["delta_R_eff"], atol=1.0e-5)

    source_base = torch.randn(2, 64, 64)
    t_cnn = source_base + outputs["cnn_mean_rise"][:, None, None] + outputs["cnn_centered_field"]
    t_cnn_gnn = source_base + outputs["mean_rise"][:, None, None] + outputs["centered_field"]
    expected = t_cnn + outputs["scaled_graph_correction_field"]
    expected = expected - expected.mean(dim=(-2, -1), keepdim=True) + t_cnn.mean(dim=(-2, -1), keepdim=True)
    assert t_cnn.shape == t_cnn_gnn.shape == (2, 64, 64)
    assert torch.isfinite(t_cnn_gnn).all()

    with torch.no_grad():
        try:
            model(x, metadata, graph)
        except ValueError as exc:
            assert "total_power_W" in str(exc)
        else:
            raise AssertionError("residual-resistance graph model should require total_power_W")


def test_frozen_feature_fusion_cnn_weights_stay_bitwise_unchanged_after_graph_step() -> None:
    torch.manual_seed(5)
    config = base_config("miniunet_refine_conditioned_decomposed_feature_fusion_graph")
    config.update(
        {
            "graph_node_feature_dim": 6,
            "graph_edge_feature_dim": 4,
            "graph_hidden_dim": 8,
            "graph_edge_hidden_dim": 8,
            "graph_layers": 1,
            "graph_raster_channels": 4,
            "freeze_cnn": True,
            "mean_head_mode": "residual_resistance",
            "delta_R_eff_target_mean_K_per_W": 0.1,
            "delta_R_eff_target_std_K_per_W": 0.2,
            "graph_mean_correction_enabled": False,
        }
    )
    model = build_model(config)
    before = {name: tensor.detach().clone() for name, tensor in model.cnn_model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    x = torch.randn(2, 12, 64, 64)
    metadata = torch.randn(2, 5)
    total_power = torch.tensor([10.0, 20.0])
    outputs = model(x, metadata, small_graph(2), total_power_W=total_power)
    target = torch.randn_like(outputs["centered_field"])
    loss = torch.nn.functional.smooth_l1_loss(outputs["centered_field"], target)
    loss.backward()
    optimizer.step()

    after = model.cnn_model.state_dict()
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name
    assert any(parameter.grad is not None for parameter in model.fusion_head.parameters())


if __name__ == "__main__":
    test_feature_fusion_forward_shapes_and_zero_mean()
    test_feature_fusion_residual_path_receives_gradients()
    test_feature_fusion_checkpoint_round_trip()
    test_feature_fusion_graph_freezes_entire_cnn_branch_and_forwards()
    test_feature_fusion_graph_supports_frozen_residual_resistance_cnn()
    test_frozen_feature_fusion_cnn_weights_stay_bitwise_unchanged_after_graph_step()
    print("feature fusion model tests passed")
