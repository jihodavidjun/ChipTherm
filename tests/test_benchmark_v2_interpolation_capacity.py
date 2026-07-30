from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.benchmark_v2_interpolation_capacity import (  # noqa: E402
    PARAMETER_TARGET_RANGE,
    aggregate_sample_rows,
    deterministic_width_search,
    read_yaml,
    validate_two_factor_configs,
)
from chiptherm.ml.ema import ExponentialMovingAverage  # noqa: E402
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats  # noqa: E402
from scripts.analyze_benchmark_v2_interpolation_capacity import (  # noqa: E402
    CNN_MATRIX_NAMES,
    build_final_comparison,
    build_primary_gate,
    build_validation_gate,
    compute_two_factor_effects,
)
from scripts.evaluate_residual_cnn import select_checkpoint_state_dict  # noqa: E402
from scripts.inspect_benchmark_v2_interpolation_checkpoints import (  # noqa: E402
    inspect_run,
)
from scripts.run_benchmark_v2_interpolation_capacity import (  # noqa: E402
    build_training_command,
)
from scripts.train_residual_cnn import make_scheduler, save_checkpoint  # noqa: E402


def main() -> None:
    test_cosine_scheduler_reaches_configured_minimum()
    test_ema_initialization_update_context_and_resume()
    test_ema_training_checkpoint_contract()
    test_ema_checkpoint_selection_and_legacy_compatibility()
    test_canonical_config_invariance()
    test_deterministic_parameter_match()
    test_dry_run_command_is_one_variant_only()
    test_checkpoint_inspection_distinguishes_best_and_stale_last()
    test_validation_freeze_inputs_and_primary_test_exclusion()
    test_two_factor_aggregation()
    test_final_comparison_aggregation()
    test_analysis_macro_micro_aggregation()
    print("benchmark v2 interpolation-capacity tests passed")


def test_cosine_scheduler_reaches_configured_minimum() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3)
    scheduler = make_scheduler("cosine", optimizer, 10, eta_min=1.0e-5)
    assert scheduler is not None
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 1.0e-5) < 1.0e-12


def test_ema_initialization_update_context_and_resume() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    assert torch.equal(ema.shadow["weight"], model.weight)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight, 2.0))
    with ema.average_parameters(model):
        assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))

    buffer = io.BytesIO()
    torch.save(ema.state_dict(), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = ExponentialMovingAverage(model, decay=0.5)
    restored.load_state_dict(payload, model=model)
    assert restored.num_updates == 1
    assert torch.equal(restored.shadow["weight"], ema.shadow["weight"])


def test_ema_checkpoint_selection_and_legacy_compatibility() -> None:
    raw = {"weight": torch.tensor([1.0])}
    averaged = {"weight": torch.tensor([2.0])}
    checkpoint = {
        "model_state_dict": raw,
        "ema_model_state_dict": averaged,
        "evaluation_default_weights": "ema",
    }
    selected, name = select_checkpoint_state_dict(checkpoint, "auto")
    assert name == "ema" and selected is averaged
    selected, name = select_checkpoint_state_dict(checkpoint, "raw")
    assert name == "raw" and selected is raw
    legacy, name = select_checkpoint_state_dict({"model_state_dict": raw}, "auto")
    assert name == "raw" and legacy is raw
    try:
        select_checkpoint_state_dict({"model_state_dict": raw}, "ema")
    except ValueError as exc:
        assert "no ema_model_state_dict" in str(exc)
    else:
        raise AssertionError("legacy checkpoint incorrectly supplied EMA weights")


def test_ema_training_checkpoint_contract() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = make_scheduler("cosine", optimizer, 10, eta_min=1.0e-5)
    ema = ExponentialMovingAverage(model, decay=0.999)
    ema.update(model)
    stats = NormalizationStats(
        schema_version=1,
        power_density_mean=0.0,
        power_density_std=1.0,
        physics_mean=0.0,
        physics_std=1.0,
        residual_mean=0.0,
        residual_std=1.0,
        num_samples=1,
        num_grid_cells=1,
    )
    config = {
        "model": {"architecture": "synthetic"},
        "seed": 1,
        "resume_signature": {"scheduler": "cosine", "ema_enabled": True},
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            3,
            config,
            stats,
            {"final_temperature": {"mae_K": 1.0}},
            best=True,
            best_val_mae=1.0,
            epochs_without_improvement=0,
            training_lineage={"source_superposition_version": "source_v1"},
            ema=ema,
            global_optimizer_step=17,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["evaluation_default_weights"] == "ema"
    assert payload["global_optimizer_step"] == 17
    assert payload["ema_decay"] == 0.999
    assert payload["ema_state_dict"]["num_updates"] == 1
    assert payload["scheduler_state_dict"] is not None
    assert payload["config_sha256"]
    assert payload["parameter_count"] == 3
    assert payload["model_seed"] == 1
    assert payload["source_version"] == "source_v1"
    assert payload["training_lineage"]["source_superposition_version"] == "source_v1"
    restored = ExponentialMovingAverage(model, decay=0.999)
    restored.load_state_dict(
        {
            **payload["ema_state_dict"],
            "shadow": payload["ema_model_state_dict"],
        },
        model=model,
    )
    assert restored.num_updates == 1


def config_paths() -> tuple[Path, Path, Path, Path]:
    canonical = (
        REPO_ROOT
        / "configs/benchmark_v2_50family/training/"
        "package_residual_feature_fusion_v1.yaml"
    )
    variants = (
        REPO_ROOT
        / "configs/benchmark_v2_50family/interpolation_capacity"
    )
    return (
        canonical,
        variants / "cnn_cosine_ema.yaml",
        variants / "cnn_param_matched_constant.yaml",
        variants / "cnn_param_matched_cosine_ema.yaml",
    )


def test_canonical_config_invariance() -> None:
    canonical_path, cosine_path, constant_path, wide_cosine_path = config_paths()
    canonical = read_yaml(canonical_path)
    cosine = read_yaml(cosine_path)
    wide_constant = read_yaml(constant_path)
    wide_cosine = read_yaml(wide_cosine_path)
    result = validate_two_factor_configs(
        canonical,
        cosine,
        wide_constant,
        wide_cosine,
    )
    assert result["two_factor_contract_preserved"]
    assert result["wide_constant"]["changed_keys"] == [
        "base_channels",
        "global_hidden_channels",
        "refine_channels",
    ]
    assert result["wide_cosine_vs_wide_constant_changed_keys"] == [
        "cosine_eta_min",
        "early_stopping_patience",
        "ema_decay",
        "ema_enabled",
        "epochs",
        "scheduler",
    ]


def test_deterministic_parameter_match() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "outputs/benchmark_v2_50family/package_residual/"
            "feature_fusion_train40_source_v1_seed1/config.json"
        ).read_text(encoding="utf-8")
    )
    first = deterministic_width_search(payload["model"])
    second = deterministic_width_search(payload["model"])
    assert first == second
    assert first["selected_width"] == 43
    assert first["selected_parameter_count"] == 3_919_642
    assert PARAMETER_TARGET_RANGE[0] <= first["selected_parameter_count"] <= PARAMETER_TARGET_RANGE[1]
    model_config = dict(payload["model"])
    model_config.update(
        base_channels=43,
        refine_channels=43,
        global_hidden_channels=43,
    )
    assert count_parameters(build_model(model_config)) == 3_919_642


def test_dry_run_command_is_one_variant_only() -> None:
    for variant, expected, forbidden in (
        (
            "param_matched_constant",
            "cnn_param_matched_constant.yaml",
            "cnn_param_matched_cosine_ema.yaml",
        ),
        (
            "param_matched_cosine_ema",
            "cnn_param_matched_cosine_ema.yaml",
            "cnn_param_matched_constant.yaml",
        ),
    ):
        command = build_training_command(
            python="/venv/bin/python3",
            variant=variant,
            data_root=Path("/data"),
            output_root=Path("/outputs"),
            config_dir=Path("/configs"),
            preflight_report=Path("/preflight.json"),
            device="cuda",
            workers=4,
            resume=False,
        )
        joined = " ".join(command)
        assert expected in joined
        assert forbidden not in joined
        assert "--resume" not in command
        assert "--execute" not in command
        assert "evaluate" not in joined
        assert "primary_test" not in joined


def metric_row(model: str, protocol: str, mae: float) -> dict[str, object]:
    return {
        "model": model,
        "protocol": protocol,
        "micro_mae_K": mae,
        "parameter_count": (
            3_919_642 if model.startswith("wide_") else 2_188_803
        ),
        "runtime_per_sample_s": 0.001,
        "fraction_worse_than_source": 0.1,
        "centered_field_mae_K": mae * 0.8,
        "mean_correction_mae_K": mae * 0.2,
        "hotspot_top1pct_mae_K": mae * 1.2,
        "metrics_sha256": f"{model}-{protocol}-metrics",
        "sample_metrics_sha256": f"{model}-{protocol}-samples",
    }


def test_checkpoint_inspection_distinguishes_best_and_stale_last() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        base = {
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "ema_model_state_dict": {"weight": torch.tensor([1.1])},
            "ema_state_dict": {"decay": 0.999, "num_updates": 5},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "scheduler_state_dict": {"T_max": 150},
            "global_optimizer_step": 40,
            "best_val_mae_K": 0.9,
            "epochs_without_improvement": 0,
            "evaluation_default_weights": "ema",
        }
        torch.save({**base, "epoch": 4, "best": True}, checkpoints / "best.pt")
        torch.save({**base, "epoch": 4, "best": False}, checkpoints / "last.pt")
        torch.save(
            {**base, "epoch": 100, "best": False},
            checkpoints / "epoch_0100.pt",
        )
        torch.save(
            {**base, "epoch": 150, "best": False},
            checkpoints / "epoch_0150.pt",
        )
        (root / "train_log.csv").write_text(
            "epoch,val_final_mae_K,epoch_runtime_s\n"
            "4,0.8,1.0\n"
            "100,0.9,1.0\n"
            "150,0.95,1.0\n",
            encoding="utf-8",
        )
        report = inspect_run(root)
    findings = {
        row["artifact"]: row["classification"] for row in report["findings"]
    }
    assert findings["best.pt"] == "consistent_best_internal_validation_epoch"
    assert findings["last.pt"] == "stale_or_overwritten_after_training"
    assert findings["periodic_checkpoints"] == "consistent"
    assert report["checkpoints"]["epoch_0100.pt"]["epoch"] == 100
    assert report["checkpoints"]["epoch_0150.pt"]["epoch"] == 150
    assert report["checkpoints"]["epoch_0150.pt"]["raw_state_present"]
    assert report["checkpoints"]["epoch_0150.pt"]["ema_state_present"]
    assert report["trainer_patch_required"] is False


def test_validation_freeze_inputs_and_primary_test_exclusion() -> None:
    rows = []
    for index, model in enumerate(CNN_MATRIX_NAMES):
        rows.append(
            metric_row(model, "known_family_sample_test", 0.1 + index * 0.01)
        )
        rows.append(
            metric_row(
                model,
                "primary_validation_families",
                0.9 + index * 0.01,
            )
        )
    gate = build_validation_gate(rows)
    assert gate["status"] == "ready_to_freeze"
    assert gate["primary_test_used"] is False
    original_fingerprint = gate["input_fingerprint"]
    rows.extend(
        metric_row(model, "primary_test_families", 100.0)
        for model in CNN_MATRIX_NAMES
    )
    assert build_validation_gate(rows)["input_fingerprint"] == original_fingerprint
    gate["status"] = "frozen"
    primary = build_primary_gate(
        include_primary_test=True,
        validation_gate=gate,
        metric_rows=rows,
    )
    assert primary["status"] == "included"
    assert primary["primary_test_used_for_selection"] is False
    pending = build_validation_gate(rows[:2])
    assert pending["status"] == "pending"
    assert pending["missing_entries"]


def test_two_factor_aggregation() -> None:
    values = {
        "canonical_small_constant": 0.15,
        "small_cosine_ema_epoch100": 0.12,
        "wide_constant_epoch100": 0.13,
        "wide_cosine_ema_epoch100": 0.10,
        "small_cosine_ema_epoch150": 0.11,
        "wide_cosine_ema_epoch150": 0.09,
    }
    rows = [
        metric_row(model, "known_family_sample_test", mae)
        for model, mae in values.items()
    ]
    effects = compute_two_factor_effects(rows)
    epoch100 = effects["epoch100"]["known_family_sample_test"]
    assert abs(epoch100["width_effect_constant_K"] + 0.02) < 1.0e-12
    assert abs(epoch100["recipe_effect_small_K"] + 0.03) < 1.0e-12
    assert abs(epoch100["width_recipe_interaction_K"]) < 1.0e-12
    assert (
        abs(
            effects["bounded_epoch150"]["known_family_sample_test"][
                "width_effect_cosine_ema_K"
            ]
            + 0.02
        )
        < 1.0e-12
    )


def test_final_comparison_aggregation() -> None:
    rows = [
        metric_row("canonical_small_constant", protocol, mae)
        for protocol, mae in (
            ("known_family_sample_test", 0.15),
            ("primary_validation_families", 0.91),
            ("primary_test_families", 1.33),
        )
    ]
    specs = [
        {
            "model": "canonical_small_constant",
            "root": Path("/does/not/exist"),
            "epoch": 100,
            "checkpoint": "best.pt",
            "training_recipe": "constant",
            "scheduler": "none",
            "weights": "raw",
            "architecture_family": "cnn",
        }
    ]
    comparison = build_final_comparison(rows, specs)
    assert len(comparison) == 1
    assert comparison[0]["known_family_mae_K"] == 0.15
    assert comparison[0]["heldout_validation_mae_K"] == 0.91
    assert comparison[0]["heldout_primary_test_mae_K"] == 1.33


def sample(family: str, uid: str, mae: float) -> dict[str, str]:
    return {
        "family_uid": family,
        "sample_uid": uid,
        "mae_K": str(mae),
        "rmse_K": str(mae * 2.0),
        "physics_baseline_mae_K": "3.0",
        "centered_field_mae_K": str(mae * 0.8),
        "mean_head_abs_error_K": str(mae * 0.2),
        "hotspot_top1pct_mae_K": str(mae * 1.2),
        "boundary_region_mae_K": str(mae),
    }


def test_analysis_macro_micro_aggregation() -> None:
    rows = [
        sample("f001", "a", 1.0),
        sample("f002", "b", 3.0),
        sample("f002", "c", 5.0),
    ]
    result = aggregate_sample_rows(rows)
    assert abs(result["micro_mae_K"] - 3.0) < 1.0e-12
    assert abs(result["macro_family_mae_K"] - 2.5) < 1.0e-12
    assert result["fraction_worse_than_source"] == 1.0 / 3.0


if __name__ == "__main__":
    main()
