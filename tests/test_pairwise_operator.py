from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.graph_models import (  # noqa: E402
    PairwiseBasisThermalOperator,
    PairwiseThermalImpedanceOperator,
    chiplet_mean_temperatures,
    chiplet_metric_values,
    chiplet_peak_temperatures,
    pairwise_basis_chunk,
    pairwise_basis_superpose,
)
from chiptherm.ml.models import build_model  # noqa: E402


def synthetic_graph() -> dict[str, torch.Tensor]:
    node_features = torch.zeros(4, 8)
    node_features[:, 6] = torch.tensor([2.0, 3.0, 5.0, 7.0])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_features = torch.zeros(4, 3)
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "chiplet_rects": torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "package_size": torch.tensor([[2.0, 1.0], [2.0, 1.0]], dtype=torch.float32),
        "node_batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "num_graphs": torch.tensor(2, dtype=torch.long),
    }


def set_final_bias(module: torch.nn.Module, value: float) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
    assert linear_layers
    final = linear_layers[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(value)


def test_pairwise_aggregation_matches_manual_reference() -> None:
    graph = synthetic_graph()
    operator = PairwiseThermalImpedanceOperator(8, 3, metadata_dim=0, hidden_dim=8, layers=2)
    set_final_bias(operator.pairwise_mlp, 2.0)
    set_final_bias(operator.self_mlp, 0.5)
    out = operator(graph)
    expected = torch.tensor([6.5, 4.5, 14.5, 10.5])
    assert torch.allclose(out["node_corrections"], expected)


def test_pairwise_zero_initialization_outputs_zero() -> None:
    graph = synthetic_graph()
    operator = PairwiseThermalImpedanceOperator(8, 3, metadata_dim=2, hidden_dim=8, layers=3)
    metadata = torch.randn(2, 2)
    out = operator(graph, metadata)
    assert torch.allclose(out["k_values"], torch.zeros_like(out["k_values"]))
    assert torch.allclose(out["node_corrections"], torch.zeros_like(out["node_corrections"]))


def test_pairwise_gradients_reach_k_parameters() -> None:
    graph = synthetic_graph()
    operator = PairwiseThermalImpedanceOperator(8, 3, metadata_dim=0, hidden_dim=8, layers=2)
    set_final_bias(operator.pairwise_mlp, 1.0)
    out = operator(graph)
    loss = (out["node_corrections"] ** 2).mean()
    loss.backward()
    grads = [p.grad for p in operator.pairwise_mlp.parameters() if p.grad is not None]
    assert grads
    assert sum(float(grad.abs().sum().item()) for grad in grads) > 0.0


def test_chiplet_mean_peak_and_delta_metrics() -> None:
    graph = {
        "chiplet_rects": torch.tensor([[0.0, 0.0, 2.0, 4.0], [2.0, 0.0, 2.0, 4.0]], dtype=torch.float32),
        "package_size": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
        "node_batch": torch.tensor([0, 0], dtype=torch.long),
        "num_graphs": torch.tensor(1, dtype=torch.long),
    }
    target = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    pred = target + 1.0
    means = chiplet_mean_temperatures(target, graph)
    peaks = chiplet_peak_temperatures(target, graph)
    metrics = chiplet_metric_values(pred, target, graph)
    assert torch.allclose(means, torch.tensor([6.5, 8.5]))
    assert torch.allclose(peaks, torch.tensor([13.0, 15.0]))
    assert torch.allclose(metrics["mean_abs_error"], torch.ones(2))
    assert torch.allclose(metrics["peak_abs_error"], torch.ones(2))
    assert torch.allclose(metrics["delta_mae"], torch.tensor(0.0))


def test_pairwise_model_zero_correction_matches_cnn_centered_field() -> None:
    config = {
        "architecture": "miniunet_refine_conditioned_decomposed_pairwise",
        "input_channels": 3,
        "base_channels": 4,
        "depth": 2,
        "refine_channels": 4,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1],
        "metadata_dim": 2,
        "metadata_hidden_dim": 4,
        "metadata_embedding_dim": 4,
        "graph_node_feature_dim": 8,
        "graph_edge_feature_dim": 3,
        "pairwise_hidden_dim": 8,
        "pairwise_layers": 2,
        "freeze_cnn": True,
    }
    model = build_model(config)
    x = torch.randn(2, 3, 16, 16)
    metadata = torch.randn(2, 2)
    graph = synthetic_graph()
    out = model(x, metadata, graph)
    assert torch.allclose(out["graph_correction_field"], torch.zeros_like(out["graph_correction_field"]), atol=1.0e-6)
    assert torch.allclose(out["centered_field"], out["cnn_centered_field"], atol=1.0e-6)


def test_pairwise_basis_coefficient_shape_and_zero_init() -> None:
    graph = synthetic_graph()
    operator = PairwiseBasisThermalOperator(8, 3, metadata_dim=2, basis_rank=8, hidden_dim=8, layers=2)
    out = operator(graph, torch.randn(2, 2), height=8, width=8)
    assert out["coefficients"].shape == (4, 8)
    assert out["field"].shape == (2, 8, 8)
    assert torch.allclose(out["coefficients"], torch.zeros_like(out["coefficients"]))
    assert torch.allclose(out["field"], torch.zeros_like(out["field"]))


def source_left_target_right_graph() -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.zeros(2, 8),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_features": torch.zeros(1, 3),
        "chiplet_rects": torch.tensor([[0.0, 1.0, 1.0, 2.0], [3.0, 1.0, 2.0, 2.0]], dtype=torch.float32),
        "package_size": torch.tensor([[6.0, 4.0]], dtype=torch.float32),
        "node_batch": torch.tensor([0, 0], dtype=torch.long),
        "num_graphs": torch.tensor(1, dtype=torch.long),
    }


def test_source_facing_basis_orients_horizontally() -> None:
    graph = source_left_target_right_graph()
    rows = torch.arange(4, dtype=torch.float32) + 0.5
    cols = torch.arange(6, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(rows / 4.0, cols / 6.0, indexing="ij")
    basis = pairwise_basis_chunk(
        graph,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
        xx.reshape(1, -1),
        yy.reshape(1, -1),
        basis_rank=8,
        halo_decay_mm=4.0,
    ).view(1, 8, 4, 6)
    facing = basis[0, 4]
    opposing = basis[0, 5]
    assert facing[:, 3].mean() > facing[:, 4].mean()
    assert opposing[:, 4].mean() > opposing[:, 3].mean()


def test_pairwise_basis_superpose_has_no_cross_graph_leakage() -> None:
    graph = synthetic_graph()
    coeff = torch.zeros(4, 1)
    coeff[0, 0] = 1.0
    field = pairwise_basis_superpose(coeff, graph, graph["edge_index"], basis_rank=1, height=8, width=8)
    assert field.shape == (2, 64)
    assert field[0].abs().sum() > 0.0
    assert field[1].abs().sum() == 0.0


def test_pairwise_basis_model_zero_mean_and_gradients() -> None:
    config = {
        "architecture": "miniunet_refine_conditioned_decomposed_pairwise_basis",
        "input_channels": 3,
        "base_channels": 4,
        "depth": 2,
        "refine_channels": 4,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1],
        "metadata_dim": 2,
        "metadata_hidden_dim": 4,
        "metadata_embedding_dim": 4,
        "graph_node_feature_dim": 8,
        "graph_edge_feature_dim": 3,
        "pairwise_basis_rank": 4,
        "pairwise_basis_hidden_dim": 8,
        "pairwise_basis_layers": 2,
        "pairwise_basis_edge_chunk_size": 2,
        "freeze_cnn": True,
    }
    model = build_model(config)
    graph = synthetic_graph()
    set_final_bias(model.basis_operator.coefficient_mlp, 0.1)
    x = torch.randn(2, 3, 16, 16)
    metadata = torch.randn(2, 2)
    out = model(x, metadata, graph)
    assert torch.allclose(out["graph_correction_field"].mean(dim=(-2, -1)), torch.zeros(2), atol=1.0e-6)
    loss = (out["centered_field"] ** 2).mean()
    loss.backward()
    grad_sum = sum(
        float(parameter.grad.abs().sum().item())
        for name, parameter in model.named_parameters()
        if name.startswith("basis_operator") and parameter.grad is not None
    )
    assert grad_sum > 0.0


def test_pairwise_basis_checkpoint_save_load() -> None:
    config = {
        "architecture": "miniunet_refine_conditioned_decomposed_pairwise_basis",
        "input_channels": 3,
        "base_channels": 4,
        "depth": 2,
        "refine_channels": 4,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1],
        "metadata_dim": 2,
        "metadata_hidden_dim": 4,
        "metadata_embedding_dim": 4,
        "graph_node_feature_dim": 8,
        "graph_edge_feature_dim": 3,
        "pairwise_basis_rank": 4,
        "pairwise_basis_hidden_dim": 8,
        "pairwise_basis_layers": 2,
        "freeze_cnn": True,
    }
    model = build_model(config)
    reloaded = build_model(model.config())
    reloaded.load_state_dict(model.state_dict())
    assert reloaded.config()["architecture"] == "miniunet_refine_conditioned_decomposed_pairwise_basis"
