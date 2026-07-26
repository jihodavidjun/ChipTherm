#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.ml.models import build_model  # noqa: E402
from scripts.train_benchmark_v2_package_residual import resolve_coarse_spatial_loss_config  # noqa: E402
from scripts.train_residual_cnn import (  # noqa: E402
    build_resume_signature,
    coarse_spatial_components,
    compute_decomposed_training_losses,
    ensure_train_log_schema,
    normalize_resume_signature,
)


def synthetic_losses(
    pred_centered: torch.Tensor,
    true_centered: torch.Tensor,
    *,
    enabled: bool = True,
    weight: float = 0.1,
    size: int = 8,
) -> dict[str, torch.Tensor]:
    batch = int(pred_centered.shape[0])
    zeros = torch.zeros(batch)
    return compute_decomposed_training_losses(
        pred_temperature=pred_centered,
        true_temperature=true_centered,
        pred_mean=zeros,
        true_mean=zeros,
        pred_centered=pred_centered,
        true_centered=true_centered,
        lambda_final=1.0,
        lambda_mean=0.1,
        coarse_spatial_loss_enabled=enabled,
        coarse_spatial_loss_weight=weight,
        coarse_spatial_loss_size=size,
        coarse_spatial_loss_type="l1",
    )


def test_coarse_loss_zero_for_equal_fields() -> None:
    target = torch.randn(2, 1, 64, 64)
    losses = synthetic_losses(target, target)
    assert float(losses["coarse_spatial_loss_K"].item()) == 0.0


def test_coarse_loss_detects_broad_spatial_mismatch() -> None:
    target = torch.zeros(1, 1, 64, 64)
    prediction = torch.zeros_like(target)
    prediction[:, :, :32] = 2.0
    prediction[:, :, 32:] = -2.0
    losses = synthetic_losses(prediction, target)
    assert float(losses["coarse_spatial_loss_K"].item()) > 1.9


def test_area_downsampling_attenuates_high_frequency_perturbations() -> None:
    rows = torch.arange(64).view(64, 1)
    columns = torch.arange(64).view(1, 64)
    checkerboard = ((rows + columns) % 2).float().mul(2.0).sub(1.0)[None, None]
    target = torch.zeros_like(checkerboard)
    losses = synthetic_losses(checkerboard, target)
    full_l1 = torch.nn.functional.l1_loss(checkerboard, target)
    assert float(full_l1.item()) == 1.0
    assert float(losses["coarse_spatial_loss_K"].item()) < 1.0e-7


def test_coarse_components_are_spatially_zero_mean() -> None:
    prediction = torch.randn(3, 64, 64) + 7.0
    target = torch.randn(3, 64, 64) - 4.0
    pred_coarse, true_coarse = coarse_spatial_components(prediction, target, size=8)
    assert torch.allclose(pred_coarse.mean(dim=(-2, -1)), torch.zeros(3, 1), atol=1.0e-6)
    assert torch.allclose(true_coarse.mean(dim=(-2, -1)), torch.zeros(3, 1), atol=1.0e-6)


def test_coarse_loss_gradients_reach_original_prediction() -> None:
    prediction = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros_like(prediction)
    losses = synthetic_losses(prediction, target, weight=0.3)
    losses["weighted_coarse_spatial_loss"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum().item()) > 0.0


def test_disabled_coarse_loss_reproduces_old_total_exactly() -> None:
    prediction = torch.randn(2, 64, 64)
    target = torch.randn_like(prediction)
    pred_mean = torch.randn(2)
    true_mean = torch.randn(2)
    final_loss = torch.nn.functional.l1_loss(prediction, target)
    mean_loss = torch.nn.functional.l1_loss(pred_mean, true_mean)
    expected = final_loss + 0.1 * mean_loss
    losses = compute_decomposed_training_losses(
        pred_temperature=prediction,
        true_temperature=target,
        pred_mean=pred_mean,
        true_mean=true_mean,
        pred_centered=prediction,
        true_centered=target,
        lambda_final=1.0,
        lambda_mean=0.1,
        coarse_spatial_loss_enabled=False,
        coarse_spatial_loss_weight=0.3,
    )
    assert torch.equal(losses["total_loss"], expected)
    assert float(losses["coarse_spatial_loss_K"].item()) == 0.0


def test_invalid_coarse_size_fails_clearly() -> None:
    field = torch.zeros(1, 64, 64)
    try:
        coarse_spatial_components(field, field, size=7)
    except ValueError as exc:
        assert "must be one of" in str(exc)
    else:
        raise AssertionError("invalid coarse size should fail")


def test_old_config_uses_backward_compatible_defaults() -> None:
    resolved = resolve_coarse_spatial_loss_config({"lambda_final": 1.0, "lambda_mean": 0.1})
    assert resolved == {"enabled": False, "weight": 0.0, "size": 8, "type": "l1"}


def test_resume_signature_captures_coarse_configuration_and_normalizes_old_signature() -> None:
    config = {
        "train_index": "train.csv",
        "val_index": "val.csv",
        "batch_size": 64,
        "lr": 1.0e-3,
        "physics_input_mode": "source_superposition_v1",
        "physical_representation": "dimensional",
        "mean_head_mode": "residual_resistance",
        "scheduler": "none",
        "temp_loss_weight": 0.0,
        "hotspot_loss_weight": 0.0,
        "hotspot_top_frac": 0.05,
        "lambda_final": 1.0,
        "lambda_mean": 0.1,
        "coarse_spatial_loss_enabled": True,
        "coarse_spatial_loss_weight": 0.3,
        "coarse_spatial_loss_size": 8,
        "coarse_spatial_loss_type": "l1",
        "lambda_graph": 0.0,
        "lambda_chiplet_mean": 0.0,
        "seed": 1,
        "model": {"architecture": "test"},
    }
    signature = build_resume_signature(config)
    assert signature["coarse_spatial_loss_enabled"] is True
    assert signature["coarse_spatial_loss_weight"] == 0.3
    old_signature = {
        key: value
        for key, value in signature.items()
        if not key.startswith("coarse_spatial_loss_")
    }
    normalized = normalize_resume_signature(old_signature)
    assert normalized["coarse_spatial_loss_enabled"] is False
    assert normalized["coarse_spatial_loss_weight"] == 0.0
    assert normalized["coarse_spatial_loss_size"] == 8
    assert normalized["coarse_spatial_loss_type"] == "l1"


def test_model_inference_outputs_and_reconstruction_are_unchanged() -> None:
    torch.manual_seed(7)
    config = {
        "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion",
        "input_channels": 12,
        "output_channels": 1,
        "base_channels": 8,
        "depth": 3,
        "refine_channels": 8,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1, 2, 3],
        "metadata_dim": 5,
        "metadata_hidden_dim": 8,
        "metadata_embedding_dim": 8,
        "physics_input_mode": "source_superposition_v1",
        "global_branch_channel_indices": [0, 1, 2, 3, 11],
        "global_hidden_channels": 8,
        "global_blocks": 1,
        "global_pool_size": 8,
    }
    model = build_model(config).eval()
    x = torch.randn(1, 12, 64, 64)
    metadata = torch.randn(1, 5)
    source_base = torch.randn(1, 64, 64)
    with torch.no_grad():
        before = model(x, metadata)
        _ = synthetic_losses(
            before["centered_field"],
            torch.zeros_like(before["centered_field"]),
        )
        after = model(x, metadata)
    assert before["mean_rise"].shape == (1,)
    assert before["centered_field"].shape == (1, 64, 64)
    assert torch.equal(before["mean_rise"], after["mean_rise"])
    assert torch.equal(before["centered_field"], after["centered_field"])
    reconstructed = source_base + before["mean_rise"][:, None, None] + before["centered_field"]
    assert reconstructed.shape == (1, 64, 64)


def test_old_training_log_schema_is_migrated_for_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_log.csv"
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "is_best"])
            writer.writeheader()
            writer.writerow({"epoch": 1, "train_loss": 2.5, "is_best": 1})
        ensure_train_log_schema(path)
        with path.open("r", encoding="utf-8", newline="") as fp:
            row = next(csv.DictReader(fp))
        assert row["epoch"] == "1"
        assert row["train_loss"] == "2.5"
        assert row["train_coarse_spatial_loss_K"] == ""


if __name__ == "__main__":
    test_coarse_loss_zero_for_equal_fields()
    test_coarse_loss_detects_broad_spatial_mismatch()
    test_area_downsampling_attenuates_high_frequency_perturbations()
    test_coarse_components_are_spatially_zero_mean()
    test_coarse_loss_gradients_reach_original_prediction()
    test_disabled_coarse_loss_reproduces_old_total_exactly()
    test_invalid_coarse_size_fails_clearly()
    test_old_config_uses_backward_compatible_defaults()
    test_resume_signature_captures_coarse_configuration_and_normalizes_old_signature()
    test_model_inference_outputs_and_reconstruction_are_unchanged()
    test_old_training_log_schema_is_migrated_for_resume()
    print("coarse-spatial auxiliary-loss tests passed")
