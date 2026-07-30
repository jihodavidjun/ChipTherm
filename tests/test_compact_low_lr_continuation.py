from __future__ import annotations

import copy
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.compact_low_lr_continuation import (  # noqa: E402
    CONTINUATION_EPOCHS,
    EXPECTED_FINAL_LR,
    EXPECTED_INITIAL_LR,
    EXPECTED_PARAMETER_COUNT,
    authorize_primary_test,
    continuation_lr,
    load_checkpoint,
    load_yaml,
    select_checkpoint,
    validate_continuation_config,
    validate_parent_checkpoint,
    validation_fingerprint,
)
from scripts.train_residual_cnn import load_initial_checkpoint  # noqa: E402


CANONICAL_CHECKPOINT = (
    REPO_ROOT
    / "outputs/benchmark_v2_50family/package_residual/"
    "feature_fusion_train40_source_v1_seed1/checkpoints/best.pt"
)
CANONICAL_CONFIG = (
    REPO_ROOT
    / "configs/benchmark_v2_50family/training/"
    "package_residual_feature_fusion_v1.yaml"
)
CONTINUATION_CONFIG = (
    REPO_ROOT
    / "configs/benchmark_v2_50family/training/"
    "package_residual_feature_fusion_continuation_v1.yaml"
)


def main() -> None:
    test_weights_only_initialization_and_fresh_training_state()
    test_strict_initialization_rejects_partial_state()
    test_exact_continuation_config_and_schedule()
    test_real_parent_architecture_and_metadata()
    test_selection_thresholds_and_tie_break()
    test_no_candidate_behavior()
    test_frozen_fingerprint_and_primary_gate()
    print("compact low-LR continuation tests passed")


def test_weights_only_initialization_and_fresh_training_state() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / "parent.pt"
        parent_model = nn.Linear(3, 2)
        with torch.no_grad():
            parent_model.weight.fill_(2.0)
            parent_model.bias.fill_(3.0)
        parent_optimizer = torch.optim.AdamW(parent_model.parameters(), lr=0.7)
        parent_optimizer.zero_grad()
        parent_model(torch.ones(1, 3)).sum().backward()
        parent_optimizer.step()
        torch.save(
            {
                "model_state_dict": copy.deepcopy(parent_model.state_dict()),
                "optimizer_state_dict": parent_optimizer.state_dict(),
                "scheduler_state_dict": {"last_epoch": 94},
                "epoch": 94,
                "ema_model_state_dict": {"not": "loaded"},
            },
            checkpoint_path,
        )
        child = nn.Linear(3, 2)
        summary = load_initial_checkpoint(
            child,
            checkpoint_path,
            torch.device("cpu"),
            require_full=True,
        )
        assert summary["full_match"]
        assert all(
            torch.equal(child.state_dict()[name], value)
            for name, value in parent_model.state_dict().items()
        )
        fresh_optimizer = torch.optim.AdamW(
            child.parameters(),
            lr=EXPECTED_INITIAL_LR,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            fresh_optimizer,
            T_max=20,
            eta_min=EXPECTED_FINAL_LR,
        )
        assert fresh_optimizer.state_dict()["state"] == {}
        assert scheduler.last_epoch == 0
        assert fresh_optimizer.param_groups[0]["lr"] == EXPECTED_INITIAL_LR


def test_strict_initialization_rejects_partial_state() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / "partial.pt"
        torch.save(
            {"model_state_dict": {"weight": torch.ones(2, 3)}},
            checkpoint_path,
        )
        try:
            load_initial_checkpoint(
                nn.Linear(3, 2),
                checkpoint_path,
                torch.device("cpu"),
                require_full=True,
            )
        except ValueError as exc:
            assert "does not exactly match" in str(exc)
        else:
            raise AssertionError("partial initialization was accepted")


def test_exact_continuation_config_and_schedule() -> None:
    canonical = load_yaml(CANONICAL_CONFIG)
    continuation = load_yaml(CONTINUATION_CONFIG)
    report = validate_continuation_config(canonical, continuation)
    assert continuation["epochs"] == 20
    assert continuation["checkpoint_frequency"] == 5
    assert tuple(range(5, 21, 5)) == CONTINUATION_EPOCHS
    assert continuation["ema_enabled"] is False
    assert continuation["swa_enabled"] is False
    assert continuation["warmup_epochs"] == 0
    assert report["optimizer"] == "AdamW"
    assert report["canonical_weight_decay"] == 0.01
    assert math.isclose(continuation_lr(0), EXPECTED_INITIAL_LR)
    assert math.isclose(continuation_lr(20), EXPECTED_FINAL_LR)
    optimizer = torch.optim.AdamW(
        [torch.zeros(1, requires_grad=True)],
        lr=EXPECTED_INITIAL_LR,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=EXPECTED_FINAL_LR,
    )
    observed = {0: optimizer.param_groups[0]["lr"]}
    for epoch in range(1, 21):
        optimizer.step()
        scheduler.step()
        if epoch in CONTINUATION_EPOCHS:
            observed[epoch] = optimizer.param_groups[0]["lr"]
    for epoch, value in observed.items():
        assert math.isclose(
            value,
            continuation_lr(epoch),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )


def test_real_parent_architecture_and_metadata() -> None:
    if not CANONICAL_CHECKPOINT.is_file():
        return
    checkpoint = load_checkpoint(CANONICAL_CHECKPOINT)
    report = validate_parent_checkpoint(CANONICAL_CHECKPOINT, checkpoint)
    assert report["compatible"]
    assert report["invariants"]["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert report["parent_checkpoint"]["epoch"] == 94
    assert report["strict_state_match"]
    assert report["training_state"]["optimizer_restored"] is False
    assert report["training_state"]["scheduler_restored"] is False


def synthetic_metrics(
    *,
    known: float,
    validation: float,
    fraction: float = 0.1,
    hotspot: float = 0.2,
) -> dict[str, float | bool]:
    return {
        "micro_mae_K": known,
        "fraction_worse_than_source": fraction,
        "hotspot_temperature_abs_error_K": hotspot,
        "outputs_finite": True,
        "micro_rmse_K": known * 1.2,
    }


def test_selection_thresholds_and_tie_break() -> None:
    metrics = [
        {
            "model": "canonical_reference",
            "protocol": "primary_validation_families",
            **synthetic_metrics(known=0.92, validation=0.92),
        },
    ]
    families = [
        {
            "model": "canonical_reference",
            "protocol": "primary_validation_families",
            "family_uid": family,
            "mae_K": 0.9,
        }
        for family in ("f007", "f012")
    ]
    values = {
        5: (0.130, 0.920),
        10: (0.131, 0.900),
        15: (0.134, 0.910),
        20: (0.140, 0.890),
    }
    for epoch, (known, validation) in values.items():
        model = f"continuation_epoch{epoch:03d}"
        metrics.extend(
            [
                {
                    "model": model,
                    "protocol": "known_family_sample_test",
                    **synthetic_metrics(known=known, validation=validation),
                },
                {
                    "model": model,
                    "protocol": "primary_validation_families",
                    **synthetic_metrics(
                        known=validation,
                        validation=validation,
                    ),
                },
            ]
        )
        families.extend(
            {
                "model": model,
                "protocol": "primary_validation_families",
                "family_uid": family,
                "mae_K": validation,
            }
            for family in ("f007", "f012")
        )
    selection = select_checkpoint(metrics, families)
    assert selection["status"] == "selected"
    assert selection["selected_epoch"] == 10
    assert sum(
        row["status"] == "admissible" for row in selection["candidates"]
    ) == 3


def test_no_candidate_behavior() -> None:
    metrics = [
        {
            "model": "canonical_reference",
            "protocol": "primary_validation_families",
            **synthetic_metrics(known=0.92, validation=0.92),
        },
    ]
    families = [
        {
            "model": "canonical_reference",
            "protocol": "primary_validation_families",
            "family_uid": "f007",
            "mae_K": 0.9,
        }
    ]
    for epoch in CONTINUATION_EPOCHS:
        model = f"continuation_epoch{epoch:03d}"
        metrics.extend(
            [
                {
                    "model": model,
                    "protocol": "known_family_sample_test",
                    **synthetic_metrics(known=0.2, validation=1.1),
                },
                {
                    "model": model,
                    "protocol": "primary_validation_families",
                    **synthetic_metrics(known=1.1, validation=1.1),
                },
            ]
        )
        families.append(
            {
                "model": model,
                "protocol": "primary_validation_families",
                "family_uid": "f007",
                "mae_K": 1.1,
            }
        )
    selection = select_checkpoint(metrics, families)
    assert selection["status"] == "no_candidate"


def test_frozen_fingerprint_and_primary_gate() -> None:
    rows = [
        {
            "model": "continuation_epoch005",
            "protocol": protocol,
            "metrics_sha256": protocol + "-metrics",
            "sample_metrics_sha256": protocol + "-samples",
        }
        for protocol in (
            "known_family_sample_test",
            "primary_validation_families",
        )
    ]
    first = validation_fingerprint(
        rows,
        checkpoint_inventory_sha256="inventory",
    )
    second = validation_fingerprint(
        list(reversed(rows)),
        checkpoint_inventory_sha256="inventory",
    )
    assert first == second
    gate = {
        "status": "frozen",
        "validation_fingerprint": first,
        "selection": {
            "status": "selected",
            "selected_epoch": 5,
        },
    }
    assert authorize_primary_test(
        gate,
        validation_fingerprint_value=first,
    ) == 5
    no_candidate = {
        **gate,
        "selection": {"status": "no_candidate"},
    }
    try:
        authorize_primary_test(no_candidate)
    except ValueError as exc:
        assert "exactly one selected" in str(exc)
    else:
        raise AssertionError("no-candidate gate opened primary test")
    try:
        authorize_primary_test(
            gate,
            validation_fingerprint_value="changed",
        )
    except ValueError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("changed validation fingerprint was accepted")


if __name__ == "__main__":
    main()
