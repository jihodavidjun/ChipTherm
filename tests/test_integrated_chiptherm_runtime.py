#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.ml.integrated_inference import IntegratedChipThermModel, StageTimer, sha256_file  # noqa: E402
from evaluate_integrated_chiptherm import RuntimeAccumulator, summarize_runtime_values  # noqa: E402


SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/source_response/final_train40_v1/checkpoints/best.pt"
)
RESIDUAL_CHECKPOINT = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/package_residual/"
    "feature_fusion_train40_source_v1_seed1/checkpoints/best.pt"
)


class IntegratedChipThermRuntimeTest(unittest.TestCase):
    def test_runtime_summary_has_required_quantiles(self) -> None:
        summary = summarize_runtime_values([0.1, 0.2, 0.3, 0.4])
        for key in ("mean_s", "median_s", "std_s", "p50_s", "p90_s", "p99_s"):
            self.assertIn(key, summary)
        self.assertAlmostEqual(summary["p50_s"], 0.25)

    def test_stage_timer_synchronizes_measurement_boundaries(self) -> None:
        timer = StageTimer(torch.device("cpu"))
        calls: list[int] = []
        timer.synchronize = lambda: calls.append(1)  # type: ignore[method-assign]
        with timer.time("stage"):
            time.sleep(0.001)
        self.assertEqual(len(calls), 2)
        self.assertGreater(timer.values["stage"], 0.0)
        self.assertEqual(timer.methods["stage"], "synchronized_wall_clock")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_stage_timer_uses_events(self) -> None:
        device = torch.device("cuda")
        timer = StageTimer(device)
        with timer.time("cuda_stage", gpu=True):
            torch.ones((64, 64), device=device).square_()
        self.assertEqual(timer.methods["cuda_stage"], "cuda_event")
        self.assertGreaterEqual(timer.values["cuda_stage"], 0.0)

    def test_runtime_separates_batch_samples_and_source_counts(self) -> None:
        accumulator = RuntimeAccumulator()
        accumulator.update(
            {"raw_input_to_output_latency_s": 0.4, "source_response_model_inference_s": 0.1},
            packages=2,
            source_counts=[3, 5],
        )
        result = accumulator.compute()
        self.assertEqual(result["num_packages"], 2)
        self.assertEqual(result["num_sources"], 8)
        self.assertAlmostEqual(result["runtime_per_package_s"], 0.2)
        self.assertEqual(result["runtime_by_source_count"][0]["mean_sources_per_package"], 4.0)

    @unittest.skipUnless(
        SOURCE_CHECKPOINT.exists() and RESIDUAL_CHECKPOINT.exists(),
        "Benchmark v2 checkpoints are not available",
    )
    def test_checkpoint_metadata_and_parameter_count_are_preserved(self) -> None:
        source_before = sha256_file(SOURCE_CHECKPOINT)
        residual_before = sha256_file(RESIDUAL_CHECKPOINT)
        model = IntegratedChipThermModel(
            source_checkpoint=SOURCE_CHECKPOINT,
            residual_checkpoint=RESIDUAL_CHECKPOINT,
            device=torch.device("cpu"),
            execution_mode="reference",
        )
        manifest = model.manifest()
        self.assertEqual(manifest["source_checkpoint_sha256"], source_before)
        self.assertEqual(manifest["residual_checkpoint_sha256"], residual_before)
        self.assertEqual(manifest["residual_parameter_count"], 2_188_803)
        self.assertEqual(model.mean_head_mode, "residual_resistance")
        batch = {
            "x": torch.zeros((1, 33, 64, 64), dtype=torch.float32),
            "ambient_K": torch.tensor([318.15], dtype=torch.float32),
            "total_power_W": torch.tensor([100.0], dtype=torch.float32),
            "metadata_vector": torch.zeros((1, 15), dtype=torch.float32),
        }
        with torch.inference_mode():
            output = model.residual_from_base(
                batch,
                torch.full((1, 64, 64), 330.0, dtype=torch.float32),
            )
        with torch.no_grad():
            reference_output = model.residual_from_base(
                batch,
                torch.full((1, 64, 64), 330.0, dtype=torch.float32),
            )
        self.assertEqual(tuple(output["final_temperature_K"].shape), (1, 64, 64))
        self.assertTrue(torch.isfinite(output["final_temperature_K"]).all())
        self.assertTrue(
            torch.equal(
                output["final_temperature_K"],
                reference_output["final_temperature_K"],
            )
        )
        self.assertEqual(sha256_file(SOURCE_CHECKPOINT), source_before)
        self.assertEqual(sha256_file(RESIDUAL_CHECKPOINT), residual_before)


if __name__ == "__main__":
    unittest.main()
