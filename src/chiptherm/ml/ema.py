from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn


class ExponentialMovingAverage:
    """EMA over a module state with exact restoration of raw parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be strictly between 0 and 1")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA model state keys changed after initialization")
        for name, value in current.items():
            target = self.shadow[name]
            if target.shape != value.shape:
                raise ValueError(f"EMA state shape changed for {name}")
            value = value.detach()
            if target.is_floating_point() or target.is_complex():
                target.mul_(self.decay).add_(
                    value.to(device=target.device, dtype=target.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                target.copy_(value.to(device=target.device, dtype=target.dtype))
        self.num_updates += 1

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.model_state_dict(),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        model: nn.Module | None = None,
    ) -> None:
        decay = float(state["decay"])
        if abs(decay - self.decay) > 1.0e-15:
            raise ValueError(
                f"EMA decay mismatch: checkpoint={decay}, requested={self.decay}"
            )
        source = state.get("shadow")
        if not isinstance(source, Mapping):
            raise ValueError("EMA checkpoint is missing shadow state")
        expected = model.state_dict() if model is not None else self.shadow
        if source.keys() != expected.keys():
            raise ValueError("EMA checkpoint model state keys differ")
        restored: dict[str, torch.Tensor] = {}
        for name, reference in expected.items():
            value = source[name]
            if not torch.is_tensor(value) or value.shape != reference.shape:
                raise ValueError(f"invalid EMA tensor for {name}")
            restored[name] = value.detach().to(
                device=reference.device,
                dtype=reference.dtype,
            ).clone()
        self.shadow = restored
        self.num_updates = int(state.get("num_updates", 0))

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        raw_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(raw_state, strict=True)
