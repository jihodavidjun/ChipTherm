from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.compact_weight_interpolation import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    FROZEN_ALPHAS,
    alpha_run_id,
    build_mixed_checkpoint,
    compatibility_report,
    interpolate_state_dict,
    select_endpoint_states,
    validate_alpha_grid,
)
from scripts.analyze_compact_weight_interpolation import (  # noqa: E402
    analyze,
    select_candidate,
    validation_fingerprint,
)
from scripts.build_compact_weight_interpolation import (  # noqa: E402
    build_artifacts,
    verify_existing,
)


CANONICAL = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/package_residual/"
    "feature_fusion_train40_source_v1_seed1/checkpoints/best.pt"
)
COSINE = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/interpolation_capacity_runs/"
    "feature_fusion_train40_cosine_ema_seed1/checkpoints/epoch_0100.pt"
)


def main() -> None:
    test_real_or_synthetic_endpoint_compatibility()
    test_endpoint_and_midpoint_math()
    test_nonfloating_state_contract()
    test_incompatible_checkpoint_rejection()
    test_mixed_checkpoint_metadata_and_lineage()
    test_alpha_names_and_frozen_grid()
    test_builder_dry_run_writes_nothing()
    test_existing_checkpoint_hash_and_metadata_verification()
    test_selection_threshold_tie_and_family_safeguard()
    test_missing_results_freeze_and_primary_gate()
    print("compact weight interpolation tests passed")


def endpoint_pair() -> tuple[dict, dict]:
    canonical = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    if COSINE.is_file():
        cosine = torch.load(COSINE, map_location="cpu", weights_only=False)
        return canonical, cosine
    cosine = copy.deepcopy(canonical)
    cosine["ema_model_state_dict"] = {
        name: tensor.detach().clone()
        for name, tensor in canonical["model_state_dict"].items()
    }
    cosine["parameter_count"] = EXPECTED_PARAMETER_COUNT
    cosine["source_version"] = canonical["training_lineage"][
        "source_superposition_version"
    ]
    cosine["model_config"]["prediction_mode"] = "residual_decomposed"
    cosine["training_config"]["prediction_mode"] = "residual_decomposed"
    return canonical, cosine


def test_real_or_synthetic_endpoint_compatibility() -> None:
    canonical, cosine = endpoint_pair()
    report = compatibility_report(canonical, cosine)
    assert report["compatible"]
    assert report["state_tensor_count"] == 102
    assert report["floating_tensor_count"] == 102
    assert report["nonfloating_tensor_count"] == 0
    assert not report["batchnorm_or_running_state_present"]
    assert report["invariants"]["parameter_count"] == EXPECTED_PARAMETER_COUNT


def test_endpoint_and_midpoint_math() -> None:
    canonical = {
        "weight": torch.tensor([0.0, 2.0], dtype=torch.float32),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    cosine = {
        "weight": torch.tensor([2.0, 6.0], dtype=torch.float32),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    alpha0 = interpolate_state_dict(canonical, cosine, 0.0)
    alpha1 = interpolate_state_dict(canonical, cosine, 1.0)
    midpoint = interpolate_state_dict(canonical, cosine, 0.5)
    assert torch.equal(alpha0["weight"], canonical["weight"])
    assert torch.equal(alpha1["weight"], cosine["weight"])
    assert torch.equal(midpoint["weight"], torch.tensor([1.0, 4.0]))
    assert torch.equal(midpoint["counter"], canonical["counter"])


def test_nonfloating_state_contract() -> None:
    canonical = {"counter": torch.tensor([1], dtype=torch.int64)}
    cosine = {"counter": torch.tensor([2], dtype=torch.int64)}
    try:
        interpolate_state_dict(canonical, cosine, 0.5)
    except ValueError as exc:
        assert "non-floating state differs" in str(exc)
    else:
        raise AssertionError("different integer buffers were silently mixed")


def test_incompatible_checkpoint_rejection() -> None:
    canonical, cosine = endpoint_pair()
    bad = copy.deepcopy(cosine)
    bad["normalization"] = copy.deepcopy(cosine["normalization"])
    bad["normalization"]["physics_mean"] += 1.0
    try:
        compatibility_report(canonical, bad)
    except ValueError as exc:
        assert "normalization" in str(exc)
    else:
        raise AssertionError("incompatible normalization was accepted")


def test_mixed_checkpoint_metadata_and_lineage() -> None:
    canonical, cosine = endpoint_pair()
    canonical_state, cosine_state = select_endpoint_states(canonical, cosine)
    mixed = interpolate_state_dict(canonical_state, cosine_state, 0.25)
    checkpoint = build_mixed_checkpoint(
        canonical=canonical,
        cosine=cosine,
        mixed_state=mixed,
        alpha=0.25,
        canonical_path=Path("/canonical.pt"),
        cosine_path=Path("/cosine.pt"),
        canonical_sha256="a" * 64,
        cosine_sha256="b" * 64,
    )
    assert checkpoint["post_training_interpolated_checkpoint"]
    assert not checkpoint["resumable_training_checkpoint"]
    assert checkpoint["evaluation_default_weights"] == "raw"
    assert checkpoint["ema_model_state_dict"] is None
    assert checkpoint["optimizer_state_dict"] is None
    assert checkpoint["scheduler_state_dict"] is None
    assert checkpoint["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert checkpoint["training_lineage"]["alpha"] == 0.25
    assert checkpoint["training_lineage"]["primary_heldout_used_for_selection"] is False


def test_alpha_names_and_frozen_grid() -> None:
    assert [alpha_run_id(alpha) for alpha in FROZEN_ALPHAS] == [
        "compact_soup_alpha000",
        "compact_soup_alpha025",
        "compact_soup_alpha050",
        "compact_soup_alpha075",
        "compact_soup_alpha100",
    ]
    assert validate_alpha_grid(FROZEN_ALPHAS) == FROZEN_ALPHAS
    try:
        validate_alpha_grid((0.0, 0.5, 1.0))
    except ValueError as exc:
        assert "requires exactly" in str(exc)
    else:
        raise AssertionError("an adaptive alpha grid was accepted")


def test_builder_dry_run_writes_nothing() -> None:
    if not CANONICAL.is_file() or not COSINE.is_file():
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "soups"
        result = build_artifacts(
            canonical_path=CANONICAL,
            cosine_path=COSINE,
            out_root=root,
            alphas=FROZEN_ALPHAS,
            execute=False,
        )
        assert result["manifest"]["status"] == "dry_run_validated"
        assert not root.exists()


def test_existing_checkpoint_hash_and_metadata_verification() -> None:
    from chiptherm.compact_weight_interpolation import sha256_file

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint_path = (
            root / "compact_soup_alpha000/checkpoints/interpolated.pt"
        )
        checkpoint_path.parent.mkdir(parents=True)
        torch.save(
            {
                "post_training_interpolated_checkpoint": True,
                "resumable_training_checkpoint": False,
                "evaluation_default_weights": "raw",
                "ema_model_state_dict": None,
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
                "alpha": 0.0,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "model_state_dict": {"weight": torch.tensor([1.0])},
                "training_lineage": {
                    "canonical_parent": {"sha256": "a" * 64},
                    "cosine_parent": {"sha256": "b" * 64},
                },
            },
            checkpoint_path,
        )
        manifest = {
            "canonical_parent": {"sha256": "a" * 64},
            "cosine_parent": {"sha256": "b" * 64},
            "runs": [
                {
                    "run_id": "compact_soup_alpha000",
                    "alpha": 0.0,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            ]
        }
        (root / "interpolation_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        report = verify_existing(
            root,
            expected_parent_hashes={
                "canonical": "a" * 64,
                "cosine": "b" * 64,
            },
        )
        assert report["passed"]
        assert len(report["verified_checkpoints"]) == 1


def metric_row(
    model: str,
    protocol: str,
    mae: float,
    *,
    hotspot: float = 0.10,
    fraction: float = 0.0025,
) -> dict[str, object]:
    return {
        "model": model,
        "protocol": protocol,
        "micro_mae_K": mae,
        "fraction_worse_than_source": fraction,
        "hotspot_temperature_abs_error_K": hotspot,
        "outputs_finite": True,
        "metrics_sha256": f"{model}-{protocol}-metrics",
        "sample_metrics_sha256": f"{model}-{protocol}-samples",
    }


def family_row(
    model: str,
    family: str,
    mae: float,
) -> dict[str, object]:
    return {
        "model": model,
        "protocol": "primary_validation_families",
        "family_uid": family,
        "mae_K": mae,
    }


def selection_fixture() -> tuple[list[dict], list[dict]]:
    rows = [
        metric_row("canonical_reference", "primary_validation_families", 0.913),
    ]
    families = [
        family_row("canonical_reference", "f007", 0.90),
        family_row("canonical_reference", "f012", 0.92),
    ]
    values = {
        0.25: (0.140, 0.920),
        0.50: (0.133, 0.935),
        0.75: (0.132, 0.936),
    }
    for alpha, (known, validation) in values.items():
        model = alpha_run_id(alpha)
        rows.extend(
            [
                metric_row(model, "known_family_sample_test", known),
                metric_row(model, "primary_validation_families", validation),
            ]
        )
        families.extend(
            [
                family_row(model, "f007", validation - 0.01),
                family_row(model, "f012", validation + 0.01),
            ]
        )
    return rows, families


def test_selection_threshold_tie_and_family_safeguard() -> None:
    rows, families = selection_fixture()
    result = select_candidate(rows, families, endpoint_checks_passed=True)
    assert result["status"] == "selected"
    assert result["selected_alpha"] == 0.50
    degraded = copy.deepcopy(families)
    for row in degraded:
        if row["model"] == alpha_run_id(0.50) and row["family_uid"] == "f007":
            row["mae_K"] = 1.01
        if row["model"] == alpha_run_id(0.75) and row["family_uid"] == "f007":
            row["mae_K"] = 1.01
    rejected = select_candidate(rows, degraded, endpoint_checks_passed=True)
    assert rejected["status"] == "no_candidate"
    assert all(
        item["status"] == "rejected"
        for item in rejected["candidates"]
        if item["alpha"] in {0.50, 0.75}
    )
    no_endpoint = select_candidate(rows, families, endpoint_checks_passed=False)
    assert no_endpoint["status"] == "no_candidate"


def write_protocol(
    root: Path,
    *,
    mae: float,
    family: str = "f007",
    predictions: bool,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    final = {
        "mae_K": mae,
        "rmse_K": mae * 1.5,
        "hotspot_temp_error_K": 0.1,
        "hotspot_location_error_cells": 1.0,
        "mean_signed_error_K": 0.01,
    }
    metrics = {
        "cnn_final_temperature": final,
        "worse_than_physics_baseline_fraction": 0.0025,
        "inference_runtime_per_sample_s": 0.001,
        "model": {"parameter_count": EXPECTED_PARAMETER_COUNT},
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "sample_uid",
        "family_uid",
        "case_id",
        "mae_K",
        "rmse_K",
        "mean_signed_error_K",
        "physics_baseline_mae_K",
        "centered_field_mae_K",
        "mean_head_abs_error_K",
        "hotspot_top1pct_mae_K",
        "boundary_region_mae_K",
        "hotspot_location_error_cells",
    ]
    values = [
        "sample",
        family,
        family,
        mae,
        mae * 1.5,
        0.01,
        2.0,
        mae * 0.8,
        mae * 0.2,
        mae * 1.1,
        mae,
        1.0,
    ]
    with (root / "metrics_by_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerow(values)
    if predictions:
        prediction = root / "predictions" / family
        prediction.mkdir(parents=True)
        np.save(
            prediction / "sample_tpred.npy",
            np.full((64, 64), 350.0, dtype=np.float32),
        )


def test_missing_results_freeze_and_primary_gate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        experiment = root / "experiment"
        out = root / "summary"
        canonical = root / "canonical"
        cosine = root / "cosine"
        experiment.mkdir()
        state = {
            "alpha000": {"state_exact": True},
            "alpha100": {"state_exact": True},
        }
        (experiment / "endpoint_reproduction_report.json").write_text(
            json.dumps(state),
            encoding="utf-8",
        )
        pending = analyze(
            experiment_root=experiment,
            canonical_eval_root=canonical,
            cosine_eval_root=cosine,
            out_dir=out,
            freeze_validation=False,
            include_primary_test=False,
        )
        assert pending["status"] == "pending"
        for protocol, canonical_mae, cosine_mae in (
            ("known_family_sample_test", 0.150, 0.1268),
            ("primary_validation_families", 0.913, 0.9817),
        ):
            write_protocol(
                canonical / protocol,
                mae=canonical_mae,
                predictions=False,
            )
            write_protocol(
                cosine / protocol,
                mae=cosine_mae,
                predictions=False,
            )
            soup_values = {
                0.0: canonical_mae,
                0.25: 0.140 if protocol.startswith("known") else 0.920,
                0.50: 0.133 if protocol.startswith("known") else 0.935,
                0.75: 0.132 if protocol.startswith("known") else 0.936,
                1.0: cosine_mae,
            }
            for alpha, mae in soup_values.items():
                write_protocol(
                    experiment
                    / alpha_run_id(alpha)
                    / "evaluation_validation"
                    / protocol,
                    mae=mae,
                    predictions=True,
                )
        ready = analyze(
            experiment_root=experiment,
            canonical_eval_root=canonical,
            cosine_eval_root=cosine,
            out_dir=out,
            freeze_validation=False,
            include_primary_test=False,
        )
        assert ready["status"] == "ready_to_freeze"
        frozen = analyze(
            experiment_root=experiment,
            canonical_eval_root=canonical,
            cosine_eval_root=cosine,
            out_dir=out,
            freeze_validation=True,
            include_primary_test=False,
        )
        assert frozen["status"] == "frozen"
        assert frozen["selection"]["selected_alpha"] == 0.50
        frozen_rows = [
            metric_row("a", "known_family_sample_test", 1.0),
            metric_row("b", "primary_validation_families", 2.0),
        ]
        first = validation_fingerprint(frozen_rows)
        frozen_rows.append(metric_row("a", "primary_test_families", 99.0))
        assert validation_fingerprint(frozen_rows) == first
        selected_root = (
            experiment
            / alpha_run_id(0.50)
            / "evaluation_primary_test"
            / "primary_test_families"
        )
        write_protocol(selected_root, mae=1.30, predictions=True)
        final = analyze(
            experiment_root=experiment,
            canonical_eval_root=canonical,
            cosine_eval_root=cosine,
            out_dir=out,
            freeze_validation=False,
            include_primary_test=True,
        )
        assert final["primary_test_included"]
        metrics_path = (
            experiment
            / alpha_run_id(0.50)
            / "evaluation_validation"
            / "known_family_sample_test"
            / "metrics.json"
        )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["cnn_final_temperature"]["mae_K"] += 0.001
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            analyze(
                experiment_root=experiment,
                canonical_eval_root=canonical,
                cosine_eval_root=cosine,
                out_dir=out,
                freeze_validation=False,
                include_primary_test=True,
            )
        except ValueError as exc:
            assert "changed after freezing" in str(exc)
        else:
            raise AssertionError("changed validation artifacts passed the gate")


if __name__ == "__main__":
    main()
