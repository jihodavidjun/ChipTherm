#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.metrics import ErrorMetricAccumulator, error_metric_summary  # noqa: E402


class MetricDefinitionTest(unittest.TestCase):
    def test_global_pixel_rmse_differs_from_mean_sample_rmse(self) -> None:
        pred = np.asarray(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ],
            dtype=np.float32,
        )
        target = np.zeros_like(pred)
        summary = error_metric_summary(pred, target).to_dict()
        self.assertAlmostEqual(summary["global_pixel_rmse_K"], 2.8284271247461903)
        self.assertAlmostEqual(summary["mean_sample_rmse_K"], 2.0)
        self.assertAlmostEqual(summary["rmse_K"], summary["global_pixel_rmse_K"])

    def test_accumulator_uses_global_pixel_rmse_alias(self) -> None:
        pred = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ]
        )
        target = torch.zeros_like(pred)
        acc = ErrorMetricAccumulator()
        acc.update(pred, target)
        result = acc.compute()
        self.assertAlmostEqual(result["global_pixel_rmse_K"], 2.8284271247461903, places=6)
        self.assertAlmostEqual(result["mean_sample_rmse_K"], 2.0, places=6)
        self.assertAlmostEqual(result["rmse_K"], result["global_pixel_rmse_K"])


if __name__ == "__main__":
    unittest.main()
