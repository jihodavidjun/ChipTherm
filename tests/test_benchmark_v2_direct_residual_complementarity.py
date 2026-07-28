#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_benchmark_v2_direct_residual_complementarity import (  # noqa: E402
    PROTOCOLS,
    aggregate_families,
    aggregate_protocols,
    analyze_descriptors,
    apply_standardizer,
    apply_threshold_route,
    classify_winner,
    compute_sample_record,
    fit_logistic,
    fit_standardizer,
    load_protocol_records,
    predict_logistic,
    reliability_bins,
    roc_auc,
    run_analysis,
    select_disagreement_threshold,
    strict_uid_map,
)


class ComplementarityTests(unittest.TestCase):
    def test_per_sample_metrics_tie_oracle_ensemble_and_disagreement(self) -> None:
        target = field(330.0)
        direct = target + 2.0
        residual = target - 1.0
        record = compute_sample_record(
            uid="f044_w001",
            family="f044",
            protocol="primary_test_families",
            row={
                "workload_uid": "f044_w001_sparse_balanced",
                "topology_regime": "sparse_asymmetric",
            },
            target=target,
            direct_prediction=direct,
            residual_prediction=residual,
            occupancy=np.ones((64, 64), dtype=np.float32),
            source=None,
            tie_tolerance_K=0.01,
        )
        self.assertAlmostEqual(record["direct_mae_K"], 2.0)
        self.assertAlmostEqual(record["residual_mae_K"], 1.0)
        self.assertAlmostEqual(record["absolute_model_disagreement_mae_K"], 3.0)
        self.assertAlmostEqual(record["average_ensemble_mae_K"], 0.5)
        self.assertAlmostEqual(record["sample_oracle_mae_K"], 1.0)
        self.assertEqual(record["winner"], "residual")
        self.assertEqual(classify_winner(1.0, 1.009, 0.01), "tie")

    def test_sample_and_family_oracles(self) -> None:
        rows = [
            metric_row("known_family_sample_test", "f1", "a", direct=1.0, residual=2.0, disagreement=1.0),
            metric_row("known_family_sample_test", "f1", "b", direct=3.0, residual=2.0, disagreement=1.0),
            metric_row("known_family_sample_test", "f2", "c", direct=4.0, residual=1.0, disagreement=2.0),
        ]
        family = aggregate_families(rows)
        protocols, oracle, ensemble = aggregate_protocols(rows)
        f1 = next(row for row in family if row["family_uid"] == "f1")
        self.assertEqual(f1["family_oracle_choice"], "residual")
        self.assertAlmostEqual(f1["family_oracle_mae_K"], 2.0)
        self.assertAlmostEqual(protocols[0]["sample_oracle_mae_K"], 4.0 / 3.0)
        self.assertAlmostEqual(protocols[0]["family_oracle_mae_K"], 5.0 / 3.0)
        self.assertEqual(len(oracle), 3)
        self.assertEqual({row["method"] for row in ensemble}, {
            "direct_only", "residual_only", "simple_average", "sample_oracle"
        })

    def test_uid_duplicate_and_mismatch_detection(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            strict_uid_map(
                [{"sample_uid": "x"}, {"sample_uid": "x"}],
                "synthetic",
            )
        with tempfile.TemporaryDirectory() as tmp:
            fixture = create_fixture(Path(tmp))
            prediction = (
                fixture["direct_root"]
                / "known_family_sample_test/predictions/f001/f001_w001_tpred.npy"
            )
            prediction.unlink()
            with self.assertRaisesRegex(ValueError, "UID mismatch"):
                load_protocol_records(
                    protocol="known_family_sample_test",
                    direct_dir=fixture["direct_root"] / "known_family_sample_test",
                    residual_dir=fixture["residual_root"] / "known_family_sample_test",
                    source_metrics={},
                    tie_tolerance_K=0.01,
                )

    def test_target_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = create_fixture(Path(tmp))
            residual_metrics = read_json(
                fixture["residual_root"] / "primary_test_families/metrics.json"
            )
            residual_index = Path(residual_metrics["index"])
            rows = read_csv(residual_index)
            mismatched = fixture["data_root"] / "arrays/mismatched_y.npy"
            np.save(mismatched, field(331.0))
            rows[0]["y_path"] = relative(mismatched, fixture["data_root"])
            alternate = fixture["data_root"] / "indices/primary_test_residual.csv"
            write_csv(alternate, rows)
            residual_metrics["index"] = str(alternate)
            write_json(
                fixture["residual_root"] / "primary_test_families/metrics.json",
                residual_metrics,
            )
            with self.assertRaisesRegex(ValueError, "target arrays differ"):
                load_protocol_records(
                    protocol="primary_test_families",
                    direct_dir=fixture["direct_root"] / "primary_test_families",
                    residual_dir=fixture["residual_root"] / "primary_test_families",
                    source_metrics={},
                    tie_tolerance_K=0.01,
                )

    def test_auc_reliability_and_undefined_behavior(self) -> None:
        auc, reason = roc_auc(np.asarray([0.1, 0.4, 0.2, 0.9]), np.asarray([0, 1, 0, 1]))
        self.assertAlmostEqual(auc, 1.0)
        self.assertIsNone(reason)
        auc, reason = roc_auc(np.asarray([0.1, 0.2]), np.asarray([1, 1]))
        self.assertIsNone(auc)
        self.assertIn("negative", reason)
        rows = [
            metric_row("p", "f1", f"s{i}", direct=float(i + 1), residual=2.0, disagreement=float(i))
            for i in range(6)
        ]
        bins = reliability_bins(rows, 3)
        self.assertEqual([row["sample_count"] for row in bins], [2, 2, 2])

    def test_validation_only_threshold_selection(self) -> None:
        validation = [
            metric_row("primary_validation_families", "f1", "a", 1.0, 3.0, 4.0),
            metric_row("primary_validation_families", "f1", "b", 3.0, 1.0, 0.2),
            metric_row("primary_validation_families", "f2", "c", 1.0, 2.0, 3.0),
        ]
        threshold = select_disagreement_threshold(validation)
        changed_test = [
            metric_row("primary_test_families", "f9", "x", 100.0, 0.1, 10.0),
        ]
        threshold_again = select_disagreement_threshold(validation)
        self.assertEqual(threshold, threshold_again)
        mae, _fraction = apply_threshold_route(
            changed_test,
            threshold=threshold["threshold_K"],
            high_disagreement_model=threshold["high_disagreement_model"],
        )
        self.assertTrue(np.isfinite(mae))
        self.assertFalse(threshold["test_labels_used"])

    def test_train_only_descriptor_standardization_and_deterministic_models(self) -> None:
        train = np.asarray([[0.0], [2.0], [4.0]])
        test = np.asarray([[1000.0]])
        mean, std = fit_standardizer(train)
        self.assertAlmostEqual(float(mean[0]), 2.0)
        self.assertNotAlmostEqual(float(mean[0]), float(np.mean(np.vstack((train, test)))))
        x = apply_standardizer(train, mean, std)
        y = np.asarray([0, 0, 1])
        first = predict_logistic(fit_logistic(x, y), x)
        second = predict_logistic(fit_logistic(x, y), x)
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)

    def test_optional_descriptor_analysis_and_source_absence(self) -> None:
        families = [
            family_metric("known_family_sample_test", "f001", -0.2),
            family_metric("known_family_sample_test", "f002", 0.1),
            family_metric("primary_validation_families", "f041", -0.1),
            family_metric("primary_test_families", "f044", -0.3),
        ]
        skipped = analyze_descriptors(
            families,
            family_descriptor_csv=None,
            family_cluster_csv=None,
        )
        self.assertEqual(skipped["summary"]["status"], "skipped")
        with tempfile.TemporaryDirectory() as tmp:
            descriptor = Path(tmp) / "descriptors.csv"
            write_csv(
                descriptor,
                [
                    {"family_uid": row["family_uid"], "feature_a": index, "feature_b": index ** 2}
                    for index, row in enumerate(families)
                ],
            )
            completed = analyze_descriptors(
                families,
                family_descriptor_csv=descriptor,
                family_cluster_csv=None,
            )
            self.assertEqual(completed["summary"]["status"], "completed")
            self.assertEqual(len(completed["test_predictions"]), 1)

    def test_end_to_end_outputs_are_deterministic_without_optional_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = create_fixture(Path(tmp))
            first_out = Path(tmp) / "out_first"
            second_out = Path(tmp) / "out_second"
            first = run_analysis(
                direct_eval_root=fixture["direct_root"],
                residual_eval_root=fixture["residual_root"],
                out_dir=first_out,
            )
            second = run_analysis(
                direct_eval_root=fixture["direct_root"],
                residual_eval_root=fixture["residual_root"],
                out_dir=second_out,
            )
            self.assertEqual(first["sample_count"], 6)
            self.assertEqual(first["summary"]["protocol_summary"], second["summary"]["protocol_summary"])
            required = {
                "per_sample_complementarity.csv",
                "per_family_complementarity.csv",
                "protocol_summary.csv",
                "oracle_selector_summary.csv",
                "ensemble_summary.csv",
                "disagreement_correlation_summary.csv",
                "disagreement_reliability_bins.csv",
                "ood_descriptor_correlations.csv",
                "routing_proxy_summary.csv",
                "test_family_routing_predictions.csv",
                "complementarity_analysis.json",
                "direct_residual_complementarity_report.md",
                "test_family_model_comparison.png",
            }
            self.assertTrue(required.issubset({path.name for path in first_out.iterdir()}))
            summary = read_json(first_out / "complementarity_analysis.json")
            self.assertEqual(summary["input_audit"]["primary_test_families"]["target_exact_match_count"], 2)
            self.assertEqual(summary["descriptor_analysis"]["status"], "skipped")


def create_fixture(root: Path) -> dict[str, Path]:
    data_root = root / "data"
    direct_root = root / "direct"
    residual_root = root / "residual"
    (data_root / "arrays").mkdir(parents=True)
    (data_root / ".chiptherm_data_root.json").write_text(
        json.dumps({"path_semantics": "relative_to_declared_data_root"}),
        encoding="utf-8",
    )
    protocol_families = {
        "known_family_sample_test": ("f001", "f002"),
        "primary_validation_families": ("f041", "f042"),
        "primary_test_families": ("f044", "f050"),
    }
    for protocol, families in protocol_families.items():
        index_rows = []
        direct_sample_rows = []
        residual_sample_rows = []
        for offset, family in enumerate(families):
            uid = f"{family}_w001"
            target = field(330.0 + offset)
            direct_prediction = target + (0.5 if offset == 0 else 1.5)
            residual_prediction = target + (1.0 if offset == 0 else 0.5)
            y_path = data_root / f"arrays/{uid}_y.npy"
            x_path = data_root / f"arrays/{uid}_x.npy"
            source_path = data_root / f"arrays/{uid}_source.npy"
            np.save(y_path, target)
            x = np.zeros((33, 64, 64), dtype=np.float32)
            x[1, 8:56, 8:56] = 1.0
            np.save(x_path, x)
            np.save(source_path, target + 2.0)
            index_rows.append(
                {
                    "sample_uid": uid,
                    "family_uid": family,
                    "case_id": family,
                    "workload_uid": f"{uid}_sparse_balanced",
                    "power_regime": "low",
                    "topology_regime": "sparse_asymmetric",
                    "y_path": relative(y_path, data_root),
                    "x_path": relative(x_path, data_root),
                    "source_superposition_base_path": relative(source_path, data_root),
                }
            )
            for eval_root, prediction, rows in (
                (direct_root, direct_prediction, direct_sample_rows),
                (residual_root, residual_prediction, residual_sample_rows),
            ):
                prediction_dir = eval_root / protocol / "predictions" / family
                prediction_dir.mkdir(parents=True, exist_ok=True)
                np.save(prediction_dir / f"{uid}_tpred.npy", prediction)
                error = prediction.astype(np.float64) - target.astype(np.float64)
                rows.append(
                    {
                        "sample_uid": uid,
                        "family_uid": family,
                        "case_id": family,
                        "mae_K": float(np.mean(np.abs(error))),
                        "rmse_K": float(np.sqrt(np.mean(error * error))),
                    }
                )
        index_path = data_root / f"indices/{protocol}.csv"
        write_csv(index_path, index_rows)
        for eval_root, mode, rows in (
            (direct_root, "direct_temperature", direct_sample_rows),
            (residual_root, "residual_decomposed", residual_sample_rows),
        ):
            protocol_root = eval_root / protocol
            protocol_root.mkdir(parents=True, exist_ok=True)
            model = {
                "prediction_mode": mode,
                "physics_input_mode": "none"
                if mode == "direct_temperature"
                else "source_superposition_v1",
                "mean_head_mode": "direct_k"
                if mode == "direct_temperature"
                else "residual_resistance",
                "direct_temperature_target_normalization": (
                    {"mode": "train_standard", "target_name": "absolute_temperature_K"}
                    if mode == "direct_temperature"
                    else None
                ),
                "config": {
                    "prediction_mode": mode,
                    "architecture": (
                        "miniunet_refine_conditioned_direct_temperature_feature_fusion"
                        if mode == "direct_temperature"
                        else "miniunet_refine_conditioned_decomposed_feature_fusion"
                    ),
                    "physics_input_mode": "none"
                    if mode == "direct_temperature"
                    else "source_superposition_v1",
                    "mean_head_mode": "direct_k"
                    if mode == "direct_temperature"
                    else "residual_resistance",
                    "target_normalization_mode": "train_standard"
                    if mode == "direct_temperature"
                    else "not_applicable",
                    "target_name": "absolute_temperature_K"
                    if mode == "direct_temperature"
                    else "residual",
                },
            }
            write_json(
                protocol_root / "metrics.json",
                {
                    "index": str(index_path),
                    "model": model,
                    "cnn_final_temperature": {"mae_K": 1.0, "rmse_K": 1.0},
                },
            )
            write_csv(protocol_root / "metrics_by_sample.csv", rows)
    return {
        "data_root": data_root,
        "direct_root": direct_root,
        "residual_root": residual_root,
    }


def field(value: float) -> np.ndarray:
    base = np.full((64, 64), value, dtype=np.float32)
    base[32, 32] += 2.0
    return base


def metric_row(
    protocol: str,
    family: str,
    uid: str,
    direct: float,
    residual: float,
    disagreement: float,
) -> dict[str, object]:
    winner = "direct" if direct < residual else "residual" if residual < direct else "tie"
    return {
        "protocol": protocol,
        "family_uid": family,
        "sample_uid": uid,
        "direct_mae_K": direct,
        "residual_mae_K": residual,
        "source_mae_K": None,
        "average_ensemble_mae_K": 0.5 * (direct + residual),
        "sample_oracle_mae_K": min(direct, residual),
        "absolute_model_disagreement_mae_K": disagreement,
        "direct_minus_residual_mae_K": direct - residual,
        "direct_wins": int(winner == "direct"),
        "residual_wins": int(winner == "residual"),
        "tied": int(winner == "tie"),
        "winner": winner,
    }


def family_metric(protocol: str, family: str, delta: float) -> dict[str, object]:
    residual = 1.0
    direct = residual + delta
    return {
        "protocol": protocol,
        "family_uid": family,
        "mean_direct_minus_residual_mae_K": delta,
        "direct_win_fraction": float(delta < 0),
        "mean_disagreement_mae_K": abs(delta) + 0.1,
        "residual_mae_K": residual,
        "direct_mae_K": direct,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


if __name__ == "__main__":
    unittest.main()
