from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_benchmark_v2_zero_shot import (  # noqa: E402
    EXPECTED_PRIMARY_SPLIT,
    EXPECTED_PROTOCOLS,
    MODEL_LABELS,
    _locate_protocol_dir_from_tiers,
    aggregate_metric_rows,
    assign_ood_tiers,
    build_aggregate_model_comparison,
    build_per_family_metrics,
    build_source_improvement_rows,
    fit_descriptor_space,
    load_training_lineage,
    locate_protocol_dir,
    reconcile_aggregate_metrics,
    validate_descriptor_table,
)


def main() -> None:
    test_canonical_selection_preferred_over_legacy()
    test_canonical_primary_test_preferred()
    test_legacy_evaluation_fallback()
    test_missing_protocol_is_clear()
    test_same_priority_tier_ambiguity_is_clear()
    test_real_run_layout_needs_no_symlink_staging()
    test_training_lineage_precedes_checkpoint_fallback()
    test_family_aggregation_and_macro_micro()
    test_train_only_descriptor_normalization()
    test_out_of_range_and_deterministic_tiers()
    test_missing_family_detection()
    test_aggregate_metric_reconciliation()
    test_source_improvement_calculation()
    print("benchmark v2 zero-shot diagnostic tests passed")


def make_directory(root: Path, relative: str) -> Path:
    path = root / relative
    path.mkdir(parents=True)
    return path


def test_canonical_selection_preferred_over_legacy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        preferred = make_directory(root, "evaluation_selection/known_family_sample_test")
        make_directory(root, "evaluation/known_family_sample_test")
        assert locate_protocol_dir(root, "known_family_sample_test") == preferred

        preferred_val = make_directory(
            root, "evaluation_selection/primary_validation_families"
        )
        make_directory(root, "evaluation/primary_validation_families")
        assert locate_protocol_dir(root, "primary_validation_families") == preferred_val


def test_canonical_primary_test_preferred() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        preferred = make_directory(
            root, "evaluation_primary_test/primary_test_families"
        )
        make_directory(root, "evaluation/primary_test_families")
        make_directory(root, "evaluation_selection/primary_test_families")
        assert locate_protocol_dir(root, "primary_test_families") == preferred


def test_legacy_evaluation_fallback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        legacy_known = make_directory(root, "evaluation/known_family_sample_test")
        legacy_val = make_directory(root, "evaluation/primary_validation_families")
        legacy_test = make_directory(root, "evaluation/primary_test_families")
        assert locate_protocol_dir(root, "known_family_sample_test") == legacy_known
        assert locate_protocol_dir(root, "primary_validation_families") == legacy_val
        assert locate_protocol_dir(root, "primary_test_families") == legacy_test


def test_missing_protocol_is_clear() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            locate_protocol_dir(root, "primary_test_families")
        except FileNotFoundError as exc:
            message = str(exc)
            assert "no valid primary_test_families directory" in message
            assert "evaluation_primary_test/primary_test_families" in message
        else:
            raise AssertionError("missing protocol directory was accepted")


def test_same_priority_tier_ambiguity_is_clear() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        make_directory(root, "canonical_a/known_family_sample_test")
        make_directory(root, "canonical_b/known_family_sample_test")
        tiers = (
            (
                "canonical_a/known_family_sample_test",
                "canonical_b/known_family_sample_test",
            ),
        )
        try:
            _locate_protocol_dir_from_tiers(
                root, "known_family_sample_test", tiers
            )
        except ValueError as exc:
            message = str(exc)
            assert "ambiguous known_family_sample_test" in message
            assert "priority tier 1" in message
        else:
            raise AssertionError("same-tier ambiguity was accepted")


def test_real_run_layout_needs_no_symlink_staging() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "residual_operator_run"
        expected = {
            "known_family_sample_test": make_directory(
                root, "evaluation_selection/known_family_sample_test"
            ),
            "primary_validation_families": make_directory(
                root, "evaluation_selection/primary_validation_families"
            ),
            "primary_test_families": make_directory(
                root, "evaluation_primary_test/primary_test_families"
            ),
        }
        # Legacy duplicates are intentionally present, as in the synced run roots.
        for protocol in expected:
            make_directory(root, f"evaluation/{protocol}")
        assert {
            protocol: locate_protocol_dir(root, protocol)
            for protocol in expected
        } == expected


def test_training_lineage_precedes_checkpoint_fallback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        expected = {"schema_version": 1, "source_superposition_version": "test"}
        (root / "training_lineage.json").write_text(
            json.dumps(expected), encoding="utf-8"
        )
        checkpoint = root / "checkpoints/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"not a checkpoint")
        assert load_training_lineage(root) == expected


def sample(uid: str, family: str, mae: float, source: float = 3.0) -> dict[str, str]:
    return {
        "sample_uid": uid,
        "family_uid": family,
        "case_id": family,
        "mae_K": str(mae),
        "rmse_K": str(mae * 2.0),
        "mean_signed_error_K": str(mae / 10.0),
        "max_abs_error_K": str(mae * 3.0),
        "peak_temperature_abs_error_K": str(mae * 1.5),
        "hotspot_location_error_cells": "2.0",
        "physics_baseline_mae_K": str(source),
        "occupied_region_mae_K": str(mae),
        "unoccupied_region_mae_K": str(mae),
        "boundary_region_mae_K": str(mae),
        "non_boundary_region_mae_K": str(mae),
        "hotspot_top1pct_mae_K": str(mae * 1.2),
        "centered_field_mae_K": str(mae * 0.8),
        "centered_field_rmse_K": str(mae * 1.6),
        "mean_head_abs_error_K": str(mae * 0.2),
    }


def synthetic_model_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    protocol_rows = [
        sample("a_1", "a", 1.0),
        sample("b_1", "b", 3.0),
        sample("b_2", "b", 5.0),
    ]
    return {
        model: {protocol: copy.deepcopy(protocol_rows) for protocol in EXPECTED_PROTOCOLS}
        for model in MODEL_LABELS
    }


def synthetic_metrics_payloads() -> dict[str, dict[str, dict]]:
    return {
        model: {
            protocol: {
                "inference_runtime_per_sample_s": 0.001,
                "model": {"parameter_count": 100},
            }
            for protocol in EXPECTED_PROTOCOLS
        }
        for model in MODEL_LABELS
    }


def test_family_aggregation_and_macro_micro() -> None:
    rows = synthetic_model_rows()
    family = build_per_family_metrics(rows)
    cnn_known = [
        row
        for row in family
        if row["model"] == "cnn" and row["protocol"] == "known_family_sample_test"
    ]
    assert len(cnn_known) == 2
    assert next(row for row in cnn_known if row["family_uid"] == "a")["mae_K"] == 1.0
    assert next(row for row in cnn_known if row["family_uid"] == "b")["mae_K"] == 4.0

    aggregate = build_aggregate_model_comparison(rows, synthetic_metrics_payloads())
    cnn = next(
        row
        for row in aggregate
        if row["model"] == "cnn" and row["protocol"] == "known_family_sample_test"
    )
    assert np.isclose(cnn["micro_mae_K"], 3.0)
    assert np.isclose(cnn["macro_family_mae_K"], 2.5)
    assert not np.isclose(cnn["micro_mae_K"], cnn["macro_family_mae_K"])


def descriptor_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_by_uid = {
        uid: split for split, uids in EXPECTED_PRIMARY_SPLIT.items() for uid in uids
    }
    for index in range(1, 51):
        uid = f"f{index:03d}"
        value = float(index)
        if uid == "f044":
            value = 100.0
        rows.append(
            {
                "family_uid": uid,
                "split": split_by_uid[uid],
                "primary_category": "synthetic",
                "placement_style": "synthetic",
                "feature_a": str(value),
                "feature_b": str(value * 0.5),
            }
        )
    return rows


def family_errors() -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for index in range(1, 51):
        uid = f"f{index:03d}"
        source = float(index) / 100.0
        if uid == "f044":
            source = 20.0
        output[uid] = {
            model: {
                "source_mae_K": source,
                "final_mae_K": source / 2.0,
                "mean_mae_K": source / 4.0,
                "centered_mae_K": source / 3.0,
                "hotspot_mae_K": source,
            }
            for model in MODEL_LABELS
        }
    return output


def test_train_only_descriptor_normalization() -> None:
    rows = descriptor_rows()
    space = fit_descriptor_space(
        rows,
        ("feature_a", "feature_b"),
        train_family_uids=EXPECTED_PRIMARY_SPLIT["train"],
    )
    train_values = np.asarray(
        [
            [float(row["feature_a"]), float(row["feature_b"])]
            for row in rows
            if row["family_uid"] in EXPECTED_PRIMARY_SPLIT["train"]
        ]
    )
    assert np.allclose(space["mean"], train_values.mean(axis=0))
    changed = copy.deepcopy(rows)
    next(row for row in changed if row["family_uid"] == "f044")["feature_a"] = "10000"
    changed_space = fit_descriptor_space(
        changed,
        ("feature_a", "feature_b"),
        train_family_uids=EXPECTED_PRIMARY_SPLIT["train"],
    )
    assert np.array_equal(space["mean"], changed_space["mean"])
    assert np.array_equal(space["scale"], changed_space["scale"])


def test_out_of_range_and_deterministic_tiers() -> None:
    rows = descriptor_rows()
    names = ("feature_a", "feature_b")
    space = fit_descriptor_space(
        rows, names, train_family_uids=EXPECTED_PRIMARY_SPLIT["train"]
    )
    first, thresholds_a = assign_ood_tiers(
        descriptor_rows=rows,
        descriptor_names=names,
        descriptor_space=space,
        family_errors=family_errors(),
    )
    second, thresholds_b = assign_ood_tiers(
        descriptor_rows=rows,
        descriptor_names=names,
        descriptor_space=space,
        family_errors=family_errors(),
    )
    assert first == second
    assert thresholds_a == thresholds_b
    f044 = next(row for row in first if row["family_uid"] == "f044")
    assert f044["descriptors_outside_train_range_count"] > 0
    assert "marginal_extrapolation" in f044["secondary_flags"]


def test_missing_family_detection() -> None:
    rows = descriptor_rows()[:-1]
    try:
        validate_descriptor_table(rows, {"descriptor_names": ["feature_a", "feature_b"]})
    except ValueError as exc:
        assert "canonical 50-family" in str(exc)
    else:
        raise AssertionError("missing family was accepted")


def test_aggregate_metric_reconciliation() -> None:
    rows = [sample("a", "f001", 1.0), sample("b", "f001", 3.0)]
    metrics = {
        "cnn_final_temperature": {
            "mae_K": 2.0,
            "rmse_K": np.sqrt((2.0**2 + 6.0**2) / 2.0),
        }
    }
    reconcile_aggregate_metrics(
        model="cnn",
        protocol="synthetic",
        rows=rows,
        metrics=metrics,
        tolerance_K=1.0e-12,
    )
    bad = copy.deepcopy(metrics)
    bad["cnn_final_temperature"]["mae_K"] = 2.1
    try:
        reconcile_aggregate_metrics(
            model="cnn",
            protocol="synthetic",
            rows=rows,
            metrics=bad,
            tolerance_K=1.0e-6,
        )
    except ValueError as exc:
        assert "recomputed MAE" in str(exc)
    else:
        raise AssertionError("aggregate mismatch was accepted")


def test_source_improvement_calculation() -> None:
    aggregate = aggregate_metric_rows([sample("a", "f001", 1.0, source=4.0)])
    rows = build_source_improvement_rows(
        [
            {
                "model": "cnn",
                "protocol": "test",
                "family_uid": "f001",
                **aggregate,
            }
        ]
    )
    assert len(rows) == 1
    assert np.isclose(rows[0]["absolute_improvement_K"], 3.0)
    assert np.isclose(rows[0]["percentage_improvement"], 75.0)


if __name__ == "__main__":
    main()
