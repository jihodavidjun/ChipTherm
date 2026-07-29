from __future__ import annotations

import copy
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
    interpolation_decision_gate,
    read_yaml,
    validate_variant_config,
)
from chiptherm.ml.ema import ExponentialMovingAverage  # noqa: E402
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats  # noqa: E402
from scripts.analyze_benchmark_v2_interpolation_capacity import build_gate  # noqa: E402
from scripts.evaluate_residual_cnn import select_checkpoint_state_dict  # noqa: E402
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
    test_decision_gate_and_primary_test_exclusion()
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


def config_paths() -> tuple[Path, Path, Path]:
    canonical = (
        REPO_ROOT
        / "configs/benchmark_v2_50family/training/"
        "package_residual_feature_fusion_v1.yaml"
    )
    variants = (
        REPO_ROOT
        / "configs/benchmark_v2_50family/interpolation_capacity"
    )
    return canonical, variants / "cnn_cosine_ema.yaml", variants / "cnn_param_matched.yaml"


def test_canonical_config_invariance() -> None:
    canonical_path, cosine_path, parameter_path = config_paths()
    canonical = read_yaml(canonical_path)
    cosine = read_yaml(cosine_path)
    parameter = read_yaml(parameter_path)
    cosine_result = validate_variant_config(canonical, cosine, "cosine_ema")
    parameter_result = validate_variant_config(
        canonical, parameter, "param_matched"
    )
    assert cosine_result["canonical_invariants_preserved"]
    assert parameter_result["canonical_invariants_preserved"]
    assert {
        "base_channels",
        "refine_channels",
        "global_hidden_channels",
    }.isdisjoint(cosine_result["changed_keys"])


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
    command = build_training_command(
        python="/venv/bin/python3",
        variant="cosine_ema",
        data_root=Path("/data"),
        output_root=Path("/outputs"),
        config_dir=Path("/configs"),
        preflight_report=Path("/preflight.json"),
        device="cuda",
        workers=4,
        resume=False,
    )
    joined = " ".join(command)
    assert "cnn_cosine_ema.yaml" in joined
    assert "cnn_param_matched.yaml" not in joined
    assert "--resume" not in command
    assert "--execute" not in command


def metric_row(model: str, protocol: str, mae: float) -> dict[str, object]:
    return {"model": model, "protocol": protocol, "micro_mae_K": mae}


def test_decision_gate_and_primary_test_exclusion() -> None:
    rows = [
        metric_row("canonical_cnn", "known_family_sample_test", 0.150),
        metric_row("canonical_cnn", "primary_validation_families", 0.913),
        metric_row("cnn_cosine_ema", "known_family_sample_test", 0.120),
        metric_row("cnn_cosine_ema", "primary_validation_families", 0.930),
    ]
    gate = build_gate(rows)
    assert gate["strong_success"]
    assert not gate["recommend_param_matched_training"]
    altered = copy.deepcopy(rows)
    altered.extend(
        [
            metric_row("canonical_cnn", "primary_test_families", 1.0),
            metric_row("cnn_cosine_ema", "primary_test_families", 100.0),
        ]
    )
    assert build_gate(altered) == gate
    failed = interpolation_decision_gate(
        canonical_known_mae_K=0.150,
        canonical_validation_mae_K=0.913,
        candidate_known_mae_K=0.140,
        candidate_validation_mae_K=0.920,
    )
    assert failed["recommend_param_matched_training"]


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
