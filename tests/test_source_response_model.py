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


def test_nonnegative_output_initializes_near_zero() -> None:
    model = SourceResponseOperatorV1(input_channels=3, base_channels=4, depth=2, output_init_K_per_W=1.0e-4)
    x = torch.zeros(2, 3, 8, 8)
    y = model(x)
    assert torch.all(y >= 0.0)
    assert torch.allclose(y, torch.full_like(y, 1.0e-4), atol=1.0e-7)


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
    model = SourceResponseOperatorV1(input_channels=5, base_channels=4, depth=2, output_init_K_per_W=1.0e-4)
    x = torch.randn(2, 5, 8, 8)
    target = torch.ones(2, 8, 8) * 0.01
    pred = model(x)
    loss = torch.nn.functional.smooth_l1_loss(pred, target)
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_checkpoint_save_load_preserves_output(tmp_path: Path) -> None:
    config = {"architecture": "source_response_operator_v1", "input_channels": 3, "base_channels": 4, "depth": 2, "output_init_K_per_W": 1.0e-4}
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
