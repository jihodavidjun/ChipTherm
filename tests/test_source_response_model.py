#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.source_response_models import (  # noqa: E402
    SourceResponseOperatorV1,
    build_source_response_model,
    predict_source_rise,
    segment_sum_by_sample,
)


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
