#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


comparison = load_script(
    "comparison_fno", ROOT / "scripts/compare_benchmark_v2_fno_models.py"
)
operator = load_script(
    "comparison_operator", ROOT / "scripts/compare_benchmark_v2_operator_models.py"
)


class OperatorComparisonTests(unittest.TestCase):
    def test_ufno_per_family_parser_and_gains(self) -> None:
        modes = {
            "direct_fno": "direct_temperature_fno",
            "residual_fno": "residual_decomposed_fno",
            "direct_ufno": "direct_temperature_ufno",
            "residual_ufno": "residual_decomposed_ufno",
        }
        maes = {
            "direct_fno": 2.0,
            "residual_fno": 1.5,
            "direct_ufno": 1.8,
            "residual_ufno": 1.2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {}
            for model, mode in modes.items():
                model_root = root / model
                roots[model] = model_root
                for protocol in comparison.PROTOCOLS:
                    folder = model_root / protocol
                    folder.mkdir(parents=True)
                    (folder / "metrics.json").write_text(
                        json.dumps(
                            {
                                "num_samples": 2,
                                "model": {
                                    "prediction_mode": mode,
                                    "parameter_count": 100,
                                },
                                "cnn_final_temperature": {
                                    "mae_K": maes[model],
                                    "rmse_K": maes[model] + 1.0,
                                },
                                "inference_runtime_per_sample_s": 0.001,
                            }
                        ),
                        encoding="utf-8",
                    )
                    with (folder / "metrics_by_case.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=["case", "final_temperature_mae_K"],
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "case": "f044",
                                "final_temperature_mae_K": str(maes[model]),
                            }
                        )
            headline, families = comparison.aggregate_comparison(roots)
            self.assertEqual(len(headline), 12)
            self.assertEqual(len(families), 12)
            self.assertTrue(all(row["mae_K"] is not None for row in families))
            operator.enrich_effects(headline)
            lookup = {
                (row["model"], row["protocol"]): row for row in headline
            }
            test = "primary_test_families"
            self.assertAlmostEqual(
                lookup[("residual_ufno", test)]["decomposition_gain_K"], 0.6
            )
            self.assertAlmostEqual(
                lookup[("direct_ufno", test)]["local_multiscale_gain_K"], 0.2
            )
            self.assertAlmostEqual(
                lookup[("residual_ufno", test)]["local_multiscale_gain_K"], 0.3
            )

    def test_missing_learned_family_metric_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "direct_ufno"
            for protocol in comparison.PROTOCOLS:
                folder = root / protocol
                folder.mkdir(parents=True)
                (folder / "metrics.json").write_text(
                    json.dumps(
                        {
                            "model": {
                                "prediction_mode": "direct_temperature_ufno"
                            },
                            "final_temperature": {"mae_K": 1.0, "rmse_K": 2.0},
                        }
                    ),
                    encoding="utf-8",
                )
                with (folder / "metrics_by_case.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=["case", "wrong_metric"])
                    writer.writeheader()
                    writer.writerow({"case": "f044", "wrong_metric": "1.0"})
            with self.assertRaisesRegex(ValueError, "per-family MAE is missing"):
                comparison.aggregate_comparison({"direct_ufno": root})


if __name__ == "__main__":
    unittest.main()
