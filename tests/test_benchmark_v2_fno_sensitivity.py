#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = load_script(
    "select_benchmark_v2_fno_sensitivity_test",
    REPO_ROOT / "scripts/select_benchmark_v2_fno_sensitivity.py",
)


class FNOSensitivityTests(unittest.TestCase):
    def test_configs_change_only_width_and_modes(self) -> None:
        config_root = REPO_ROOT / "configs/benchmark_v2_50family/training"
        baseline = selector.load_yaml(
            config_root / "package_residual_fno_decomposed_seed1.yaml"
        )
        variants = {
            "w32_m12": config_root / "package_residual_fno_decomposed_seed1.yaml",
            "w32_m16": config_root
            / "package_residual_fno_decomposed_w32_m16_seed1.yaml",
            "w40_m12": config_root
            / "package_residual_fno_decomposed_w40_m12_seed1.yaml",
            "w40_m16": config_root
            / "package_residual_fno_decomposed_w40_m16_seed1.yaml",
        }
        rows = selector.build_capacity_rows(variants, baseline=baseline, batch_size=64)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {
                (row["fno_width"], row["fno_modes_x"], row["fno_modes_y"])
                for row in rows
            },
            {(32, 12, 12), (32, 16, 16), (40, 12, 12), (40, 16, 16)},
        )
        self.assertTrue(all(row["parameter_count"] > 2_000_000 for row in rows))
        self.assertTrue(
            all(row["estimated_fp32_adam_training_lower_bound_bytes"] > 0 for row in rows)
        )

    def test_selection_uses_validation_not_primary_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluations = {}
            capacities = []
            values = {
                "a": (2.0, 2.5, 9.0),
                "b": (1.8, 2.2, 99.0),
                "c": (1.8, 2.1, 0.1),
            }
            for index, (name, (val_mae, val_rmse, test_mae)) in enumerate(values.items()):
                evaluation = root / name
                evaluations[name] = evaluation
                capacities.append({"variant": name, "parameter_count": 100 + index})
                self.write_metrics(
                    evaluation / selector.REFERENCE_PROTOCOL / "metrics.json",
                    mae=1.0 + index,
                    rmse=1.5 + index,
                    runtime=0.003,
                )
                self.write_metrics(
                    evaluation / selector.SELECTION_PROTOCOL / "metrics.json",
                    mae=val_mae,
                    rmse=val_rmse,
                    runtime=0.002 + index * 0.001,
                )
                self.write_metrics(
                    evaluation / "primary_test_families" / "metrics.json",
                    mae=test_mae,
                    rmse=test_mae,
                    runtime=0.001,
                )
            rows = selector.load_validation_rows(evaluations, capacities)
            selection = selector.select_variant(rows)
            self.assertEqual(selection["selected_variant"], "c")
            self.assertFalse(selection["primary_test_family_metrics_used"])

    def test_evaluation_wrapper_protocol_filter_excludes_primary_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_benchmark_v2_models.py",
                    "--data-root",
                    temporary,
                    "--source-version",
                    "synthetic",
                    "--checkpoint",
                    "checkpoint.pt",
                    "--out-dir",
                    str(Path(temporary) / "evaluation"),
                    "--protocols",
                    "known_family_sample_test",
                    "primary_validation_families",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("known_family_sample_test", result.stdout)
        self.assertIn("primary_validation_families", result.stdout)
        self.assertNotIn("primary_test_families", result.stdout)

    @staticmethod
    def write_metrics(path: Path, *, mae: float, rmse: float, runtime: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": {
                        "prediction_mode": "residual_decomposed_fno",
                        "parameter_count": None,
                        "config": {
                            "architecture": "fno2d_residual_decomposed_conditioned"
                        },
                    },
                    "cnn_final_temperature": {"mae_K": mae, "rmse_K": rmse},
                    "inference_runtime_per_sample_s": runtime,
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
