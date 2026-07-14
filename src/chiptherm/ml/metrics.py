from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ErrorMetricSummary:
    mae_K: float
    global_pixel_rmse_K: float
    mean_sample_rmse_K: float
    max_abs_error_K: float
    mean_signed_error_K: float
    num_samples: int
    num_cells: int

    def to_dict(self) -> dict[str, float]:
        return {
            "mae_K": float(self.mae_K),
            "global_pixel_rmse_K": float(self.global_pixel_rmse_K),
            "mean_sample_rmse_K": float(self.mean_sample_rmse_K),
            "rmse_K": float(self.global_pixel_rmse_K),
            "max_abs_error_K": float(self.max_abs_error_K),
            "mean_signed_error_K": float(self.mean_signed_error_K),
            "num_samples": float(self.num_samples),
            "num_cells": float(self.num_cells),
        }


def error_metric_summary(pred: np.ndarray, target: np.ndarray) -> ErrorMetricSummary:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if error.ndim < 2:
        sample_errors = error.reshape(1, -1)
    elif error.ndim == 2:
        sample_errors = error.reshape(1, -1)
    else:
        sample_errors = error.reshape(error.shape[0], -1)
    abs_error = np.abs(error)
    sample_rmse = np.sqrt(np.mean(sample_errors * sample_errors, axis=1))
    return ErrorMetricSummary(
        mae_K=float(abs_error.mean()),
        global_pixel_rmse_K=float(np.sqrt(np.mean(error * error))),
        mean_sample_rmse_K=float(np.mean(sample_rmse)),
        max_abs_error_K=float(abs_error.max()),
        mean_signed_error_K=float(error.mean()),
        num_samples=int(sample_errors.shape[0]),
        num_cells=int(error.size),
    )


def torch_sample_rmse(error: torch.Tensor) -> torch.Tensor:
    if error.ndim < 2:
        flat = error.reshape(1, -1)
    else:
        flat = error.reshape(error.shape[0], -1)
    return torch.sqrt(torch.mean(flat * flat, dim=1))


class ErrorMetricAccumulator:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_cells = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
        self.sum_sample_rmse = 0.0
        self.max_abs = 0.0

    def update(self, pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        if torch.is_tensor(pred):
            pred_cpu = pred.detach().float().cpu()
        else:
            pred_cpu = torch.from_numpy(np.asarray(pred, dtype=np.float32))
        if torch.is_tensor(target):
            target_cpu = target.detach().float().cpu()
        else:
            target_cpu = torch.from_numpy(np.asarray(target, dtype=np.float32))
        error = pred_cpu - target_cpu
        abs_error = error.abs()
        self.num_samples += int(error.shape[0]) if error.ndim >= 3 else 1
        self.num_cells += int(error.numel())
        self.sum_abs += float(abs_error.sum().item())
        self.sum_sq += float((error * error).sum().item())
        self.sum_signed += float(error.sum().item())
        self.sum_sample_rmse += float(torch_sample_rmse(error).sum().item())
        self.max_abs = max(self.max_abs, float(abs_error.max().item()))

    def compute(self) -> dict[str, float]:
        if self.num_cells == 0:
            return {}
        global_rmse = (self.sum_sq / self.num_cells) ** 0.5
        return {
            "num_samples": float(self.num_samples),
            "num_cells": float(self.num_cells),
            "mae_K": self.sum_abs / self.num_cells,
            "global_pixel_rmse_K": global_rmse,
            "mean_sample_rmse_K": self.sum_sample_rmse / max(self.num_samples, 1),
            "rmse_K": global_rmse,
            "max_abs_error_K": self.max_abs,
            "mean_signed_error_K": self.sum_signed / self.num_cells,
        }


def add_rmse_aliases(payload: dict[str, Any], *, global_pixel_rmse: float, mean_sample_rmse: float) -> None:
    payload["global_pixel_rmse_K"] = float(global_pixel_rmse)
    payload["mean_sample_rmse_K"] = float(mean_sample_rmse)
    payload["rmse_K"] = float(global_pixel_rmse)
