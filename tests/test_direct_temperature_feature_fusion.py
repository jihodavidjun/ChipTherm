#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.normalization import (  # noqa: E402
    DirectTemperatureTargetStats,
    NormalizationStats,
    build_model_input,
    compute_direct_temperature_target_stats,
    normalize_direct_temperature,
    unnormalize_direct_temperature,
)
from scripts.evaluate_residual_cnn import (  # noqa: E402
    checkpoint_prediction_mode,
    direct_target_stats_from_checkpoint,
    evaluate,
    validate_checkpoint_prediction_mode,
)
from scripts.train_benchmark_v2_package_direct import validate_direct_config  # noqa: E402
from scripts.train_residual_cnn import train_one_epoch  # noqa: E402


DIRECT_ARCH = "miniunet_refine_conditioned_direct_temperature_feature_fusion"


def direct_config(*, input_channels: int = 8, base_channels: int = 4) -> dict[str, object]:
    return {
        "architecture": DIRECT_ARCH,
        "input_channels": input_channels,
        "output_channels": 1,
        "base_channels": base_channels,
        "depth": 3,
        "refine_channels": base_channels,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1, 2, 3],
        "refinement_channel_names": ["power", "occupancy", "x", "y"],
        "metadata_dim": 3,
        "metadata_hidden_dim": 8,
        "metadata_embedding_dim": 8,
        "physics_input_mode": "none",
        "prediction_mode": "direct_temperature",
        "target_normalization_mode": "none",
        "target_mean_K": 0.0,
        "target_std_K": 1.0,
        "global_branch_channel_indices": [0, 1, 2, 3],
        "global_branch_channel_names": ["power", "occupancy", "x", "y"],
        "global_hidden_channels": 4,
        "global_blocks": 1,
        "global_pool_size": 8,
    }


def residual_config() -> dict[str, object]:
    config = direct_config(input_channels=9)
    config.update(
        {
            "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion",
            "physics_input_mode": "source_superposition_v1",
            "mean_head_mode": "residual_resistance",
            "global_branch_channel_indices": [0, 1, 2, 3, 8],
            "global_branch_channel_names": ["power", "occupancy", "x", "y", "source_base"],
        }
    )
    config.pop("prediction_mode")
    config.pop("target_normalization_mode")
    config.pop("target_mean_K")
    config.pop("target_std_K")
    return config


def normalization_stats(input_channels: int = 8) -> NormalizationStats:
    return NormalizationStats(
        schema_version=1,
        power_density_mean=0.0,
        power_density_std=1.0,
        physics_mean=300.0,
        physics_std=20.0,
        residual_mean=0.0,
        residual_std=1.0,
        num_samples=2,
        num_grid_cells=8192,
        input_channels=input_channels,
        metadata_feature_names=("a", "b", "c"),
        metadata_means=(0.0, 0.0, 0.0),
        metadata_stds=(1.0, 1.0, 1.0),
    )


class TemperatureDataset(Dataset):
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"temperature": torch.full((4, 4), self.values[index], dtype=torch.float32)}


def test_direct_mode_excludes_source_and_outputs_absolute_map() -> None:
    torch.manual_seed(0)
    model = build_model(direct_config())
    x = torch.randn(2, 8, 64, 64)
    metadata = torch.randn(2, 3)
    physics_a = torch.full((2, 64, 64), 300.0)
    physics_b = torch.full((2, 64, 64), 900.0)
    model_input_a = build_model_input(
        x, physics_a, normalization_stats(), physics_input_mode="none"
    )
    model_input_b = build_model_input(
        x, physics_b, normalization_stats(), physics_input_mode="none"
    )
    assert model_input_a.shape == (2, 8, 64, 64)
    assert torch.equal(model_input_a, model_input_b)
    with torch.no_grad():
        prediction_a = model(model_input_a, metadata)
        prediction_b = model(model_input_b, metadata)
    assert prediction_a.shape == (2, 1, 64, 64)
    assert torch.equal(prediction_a, prediction_b)
    assert model.config()["prediction_mode"] == "direct_temperature"
    assert model.config()["physics_input_mode"] == "none"


def test_train_only_target_normalization_and_exact_inverse() -> None:
    train = TemperatureDataset([300.0, 340.0])
    heldout = TemperatureDataset([1000.0])
    stats = compute_direct_temperature_target_stats(
        train, mode="train_standard", batch_size=2
    )
    assert stats.mean_K == 320.0
    assert stats.max_K == 340.0
    assert stats.max_K != heldout[0]["temperature"].max().item()
    value = torch.tensor([[300.0, 320.0, 340.0]])
    restored = unnormalize_direct_temperature(
        normalize_direct_temperature(value, stats),
        stats,
    )
    assert torch.allclose(restored, value, atol=1.0e-6, rtol=0.0)


def test_checkpoint_modes_are_distinguishable_and_backward_compatible() -> None:
    direct_stats = DirectTemperatureTargetStats(
        mode="none",
        mean_K=320.0,
        std_K=20.0,
        min_K=280.0,
        max_K=400.0,
        num_samples=2,
        num_grid_cells=8192,
    )
    checkpoint = {
        "model_config": {
            **direct_config(),
            "target_normalization_mode": "none",
        },
        "training_config": {
            "prediction_mode": "direct_temperature",
            "direct_temperature_target_normalization": direct_stats.to_dict(),
        },
    }
    assert checkpoint_prediction_mode(checkpoint, DIRECT_ARCH) == "direct_temperature"
    assert direct_target_stats_from_checkpoint(checkpoint, "direct_temperature") == direct_stats
    validate_checkpoint_prediction_mode("direct_temperature", DIRECT_ARCH, "none")
    old = {"model_config": residual_config(), "training_config": {}}
    assert (
        checkpoint_prediction_mode(
            old, "miniunet_refine_conditioned_decomposed_feature_fusion"
        )
        == "residual_decomposed"
    )
    build_model(old["model_config"])
    try:
        validate_checkpoint_prediction_mode(
            "direct_temperature",
            "miniunet_refine_conditioned_decomposed_feature_fusion",
            "none",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("direct checkpoint mode must not be accepted as residual architecture")


def test_parameter_count_is_comparable_and_config_guard_rejects_source_input() -> None:
    direct = build_model(direct_config())
    residual = build_model(residual_config())
    difference = abs(count_parameters(direct) - count_parameters(residual))
    assert difference / count_parameters(residual) < 0.10
    validate_direct_config(
        {
            "model_architecture": DIRECT_ARCH,
            "prediction_mode": "direct_temperature",
            "physics_input": "none",
            "coarse_spatial_loss_enabled": False,
            "graph_enabled": False,
        }
    )
    try:
        validate_direct_config(
            {
                "model_architecture": DIRECT_ARCH,
                "prediction_mode": "direct_temperature",
                "physics_input": "source_superposition_v1",
                "coarse_spatial_loss_enabled": False,
                "graph_enabled": False,
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("canonical direct config must reject source-superposition input")


def test_direct_evaluator_reports_kelvin_after_inverse_normalization() -> None:
    class ConstantDirectModel(nn.Module):
        def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
            return x.new_ones((x.shape[0], 1, x.shape[-2], x.shape[-1]))

    stats = DirectTemperatureTargetStats(
        mode="train_standard",
        mean_K=300.0,
        std_K=20.0,
        min_K=280.0,
        max_K=360.0,
        num_samples=2,
        num_grid_cells=8192,
    )
    temperature = torch.full((1, 64, 64), 320.0)
    physics = torch.full((1, 64, 64), 315.0)
    batch = {
        "x": torch.zeros((1, 8, 64, 64)),
        "physics": physics,
        "residual": temperature - physics,
        "temperature": temperature,
        "ambient_K": torch.tensor([300.0]),
        "total_power_W": torch.tensor([10.0]),
        "metadata_vector": torch.zeros((1, 3)),
        "metadata": {
            "case_id": ["f001"],
            "sample_uid": ["f001_w001"],
            "hotspot_runtime_s": [1.0],
            "physics_runtime_s": [0.0],
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        metrics, _, _, _, _, _ = evaluate(
            ConstantDirectModel(),
            [batch],
            normalization_stats(),
            torch.device("cpu"),
            measure_end_to_end=False,
            save_predictions=False,
            out_dir=Path(tmp),
            conditioned=True,
            physics_input_mode="none",
            prediction_mode="direct_temperature",
            direct_target_stats=stats,
        )
    assert metrics["cnn_final_temperature"]["mae_K"] == 0.0
    assert metrics["prediction_mode"] == "direct_temperature"
    assert metrics["physics_baseline"]["mae_K"] == 5.0


def test_direct_training_target_is_hotspot_temperature_not_residual() -> None:
    class ScalarDirectModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
            return self.value.expand(x.shape[0], 1, x.shape[-2], x.shape[-1])

    temperature = torch.full((1, 64, 64), 2.0)
    physics = torch.full((1, 64, 64), 100.0)
    batch = {
        "x": torch.zeros((1, 8, 64, 64)),
        "physics": physics,
        "residual": temperature - physics,
        "temperature": temperature,
        "ambient_K": torch.tensor([300.0]),
        "total_power_W": torch.tensor([10.0]),
        "metadata_vector": torch.zeros((1, 3)),
        "metadata": {"case_id": ["f001"], "sample_uid": ["f001_w001"]},
    }
    model = ScalarDirectModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = train_one_epoch(
        model,
        [batch],
        optimizer,
        nn.SmoothL1Loss(),
        nn.L1Loss(),
        normalization_stats(),
        torch.device("cpu"),
        temp_loss_weight=0.0,
        hotspot_loss_weight=0.0,
        hotspot_top_frac=0.01,
        decomposed=False,
        conditioned=True,
        lambda_final=1.0,
        lambda_mean=0.0,
        coarse_spatial_loss_enabled=False,
        coarse_spatial_loss_weight=0.0,
        coarse_spatial_loss_size=8,
        coarse_spatial_loss_type="l1",
        physics_input_mode="none",
        physics_gate_regularization=0.0,
        physics_gate_init=0.9,
        prediction_mode="direct_temperature",
        direct_target_stats=DirectTemperatureTargetStats(
            mode="none",
            mean_K=2.0,
            std_K=1.0,
            min_K=2.0,
            max_K=2.0,
            num_samples=1,
            num_grid_cells=4096,
        ),
    )
    assert abs(losses["direct_map_loss"] - 2.0) < 1.0e-6
    assert abs(losses["final_map_loss_K"] - 2.0) < 1.0e-6


def test_deterministic_cpu_smoke_optimization_decreases_direct_loss() -> None:
    torch.manual_seed(7)
    model = build_model(direct_config(base_channels=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    x = torch.randn(1, 8, 64, 64)
    metadata = torch.randn(1, 3)
    target = torch.full((1, 1, 64, 64), 2.0)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, metadata)
        loss = torch.nn.functional.l1_loss(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0] * 0.98, losses


if __name__ == "__main__":
    test_direct_mode_excludes_source_and_outputs_absolute_map()
    test_train_only_target_normalization_and_exact_inverse()
    test_checkpoint_modes_are_distinguishable_and_backward_compatible()
    test_parameter_count_is_comparable_and_config_guard_rejects_source_input()
    test_direct_evaluator_reports_kelvin_after_inverse_normalization()
    test_direct_training_target_is_hotspot_temperature_not_residual()
    test_deterministic_cpu_smoke_optimization_decreases_direct_loss()
    print("direct-temperature feature-fusion tests passed")
