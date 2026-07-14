#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiptherm.ml.source_response_models import (  # noqa: E402
    SourceResponseOperatorV1,
    build_source_response_model,
    predict_source_rise,
    segment_sum_by_sample,
)
from scripts.train_source_response_model import package_loss_weight, segment_sum_fields  # noqa: E402


def test_linear_output_is_not_softplus_pinned_near_zero() -> None:
    torch.manual_seed(0)
    model = SourceResponseOperatorV1(input_channels=3, base_channels=4, depth=2)
    x = torch.zeros(2, 3, 8, 8)
    y = model(x)
    assert y.shape == (2, 8, 8)
    assert torch.isfinite(y).all()
    assert model.output_mode == "linear_normalized"


def test_physical_prediction_multiplies_source_power() -> None:
    unit = torch.ones(2, 4, 4) * 0.25
    power = torch.tensor([4.0, 8.0])
    physical = predict_source_rise(unit, power)
    assert torch.allclose(physical[0], torch.ones(4, 4))
    assert torch.allclose(physical[1], torch.ones(4, 4) * 2.0)


def test_segment_sum_by_package() -> None:
    values = torch.stack([torch.ones(2, 2), torch.ones(2, 2) * 2.0, torch.ones(2, 2) * 3.0])
    groups = torch.tensor([0, 0, 1])
    summed = segment_sum_by_sample(values, groups, 2)
    assert torch.allclose(summed[0], torch.ones(2, 2) * 3.0)
    assert torch.allclose(summed[1], torch.ones(2, 2) * 3.0)


def test_segment_sum_fields_matches_loop_reference() -> None:
    values = torch.randn(5, 4, 4)
    groups = torch.tensor([0, 0, 1, 1, 1])
    summed = segment_sum_fields(values, groups, 2)
    reference = torch.stack([values[:2].sum(dim=0), values[2:].sum(dim=0)])
    assert torch.allclose(summed, reference)


def test_package_loss_zero_for_exact_synthetic_source_predictions() -> None:
    source_rise = torch.stack([torch.ones(3, 3), torch.ones(3, 3) * 2.0, torch.ones(3, 3) * 4.0])
    groups = torch.tensor([0, 0, 1])
    ambient = torch.tensor([318.0, 319.0])
    target = ambient[:, None, None] + segment_sum_fields(source_rise, groups, 2)
    pred = ambient[:, None, None] + segment_sum_fields(source_rise, groups, 2)
    assert torch.nn.functional.smooth_l1_loss(pred, target).item() == 0.0


def test_package_loss_detects_correlated_source_bias() -> None:
    source_rise = torch.stack([torch.ones(3, 3), torch.ones(3, 3) * 2.0])
    biased = source_rise + 0.5
    groups = torch.tensor([0, 0])
    ambient = torch.tensor([318.0])
    target = ambient[:, None, None] + segment_sum_fields(source_rise, groups, 1)
    pred = ambient[:, None, None] + segment_sum_fields(biased, groups, 1)
    assert torch.nn.functional.smooth_l1_loss(pred, target).item() > 0.0


def test_package_loss_gradients_reach_all_source_predictions() -> None:
    pred_rise = torch.zeros(3, 4, 4, requires_grad=True)
    groups = torch.tensor([0, 0, 0])
    ambient = torch.tensor([318.0])
    target = torch.ones(1, 4, 4) * 321.0
    pred_temp = ambient[:, None, None] + segment_sum_fields(pred_rise, groups, 1)
    loss = torch.nn.functional.smooth_l1_loss(pred_temp, target)
    loss.backward()
    assert pred_rise.grad is not None
    assert torch.isfinite(pred_rise.grad).all()
    assert torch.all(pred_rise.grad.abs().sum(dim=(1, 2)) > 0)


def test_package_loss_warmup_schedule() -> None:
    assert package_loss_weight(1.0, 5, 1) == 0.0
    assert package_loss_weight(1.0, 5, 3) == 0.4
    assert package_loss_weight(1.0, 5, 6) == 1.0
    assert package_loss_weight(0.0, 5, 6) == 0.0


def test_forward_backward_gradients_are_finite() -> None:
    model = SourceResponseOperatorV1(input_channels=5, base_channels=4, depth=2)
    x = torch.randn(2, 5, 8, 8)
    target = torch.ones(2, 8, 8) * 0.01
    pred = model(x)
    loss = torch.nn.functional.smooth_l1_loss(pred, target)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_final_output_layer_receives_nontrivial_gradient_at_initialization() -> None:
    torch.manual_seed(0)
    model = SourceResponseOperatorV1(input_channels=5, base_channels=4, depth=2)
    x = torch.randn(2, 5, 8, 8)
    target = torch.randn(2, 8, 8)
    loss = torch.nn.functional.smooth_l1_loss(model(x), target)
    loss.backward()
    grad_norm = float(model.head.weight.grad.norm().item())
    assert grad_norm > 1.0e-5


def test_tiny_cpu_overfit_drops_loss_and_moves_output() -> None:
    torch.manual_seed(1)
    model = SourceResponseOperatorV1(input_channels=3, base_channels=4, depth=2)
    x = torch.randn(4, 3, 8, 8)
    target = 0.7 * x[:, 0] - 0.2 * x[:, 1] + 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-2)
    with torch.no_grad():
        initial_pred = model(x)
        initial_loss = torch.nn.functional.smooth_l1_loss(initial_pred, target).item()
        source_power = torch.tensor([2.0, 4.0, 6.0, 8.0])
        initial_physical_mae = (predict_source_rise(initial_pred - target, source_power)).abs().mean().item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(model(x), target)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_pred = model(x)
        final_loss = torch.nn.functional.smooth_l1_loss(final_pred, target).item()
        final_physical_mae = (predict_source_rise(final_pred - target, source_power)).abs().mean().item()
    assert final_loss < initial_loss * 0.5
    assert final_physical_mae < initial_physical_mae * 0.5
    assert float((final_pred - initial_pred).abs().mean().item()) > 1.0e-2


def test_tiny_grouped_cpu_overfit_reduces_source_and_package_loss() -> None:
    torch.manual_seed(2)
    model = SourceResponseOperatorV1(input_channels=3, base_channels=4, depth=2)
    x = torch.randn(5, 3, 8, 8)
    groups = torch.tensor([0, 0, 1, 1, 1])
    source_power = torch.tensor([2.0, 4.0, 3.0, 5.0, 7.0])
    target_norm = torch.ones(5, 8, 8) * 0.2
    target_rise = predict_source_rise(target_norm, source_power)
    ambient = torch.tensor([318.0, 319.0])
    package_target = ambient[:, None, None] + segment_sum_fields(target_rise, groups, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)

    def losses() -> tuple[float, float]:
        pred_norm = model(x)
        source_loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm)
        pred_temp = ambient[:, None, None] + segment_sum_fields(predict_source_rise(pred_norm, source_power), groups, 2)
        package_loss = torch.nn.functional.smooth_l1_loss(pred_temp, package_target)
        return float(source_loss.item()), float(package_loss.item())

    initial_source, initial_package = losses()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        pred_norm = model(x)
        source_loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm)
        pred_temp = ambient[:, None, None] + segment_sum_fields(predict_source_rise(pred_norm, source_power), groups, 2)
        package_loss = torch.nn.functional.smooth_l1_loss(pred_temp, package_target)
        (source_loss + package_loss).backward()
        optimizer.step()
    final_source, final_package = losses()
    assert final_source < initial_source * 0.5
    assert final_package < initial_package * 0.5


def test_checkpoint_save_load_preserves_output(tmp_path: Path) -> None:
    config = {"architecture": "source_response_operator_v1", "input_channels": 3, "base_channels": 4, "depth": 2, "output_mode": "linear_normalized"}
    model = build_source_response_model(config)
    x = torch.randn(1, 3, 8, 8)
    y = model(x)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    loaded = build_source_response_model(ckpt["model_config"])
    loaded.load_state_dict(ckpt["model_state_dict"])
    assert torch.allclose(y, loaded(x))


def main() -> int:
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
    print("source response model tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
