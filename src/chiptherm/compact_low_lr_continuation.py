from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from chiptherm.compact_weight_interpolation import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PREDICTION_MODE,
    EXPECTED_RECONSTRUCTION,
    KNOWN_MAE_LIMIT_K,
    KNOWN_MAE_TIE_K,
    MAX_FAMILY_REGRESSION_K,
    MAX_FRACTION_WORSE_ABSOLUTE_INCREASE,
    MAX_HOTSPOT_ABS_ERROR_INCREASE_K,
    VALIDATION_MAE_LIMIT_K,
    model_invariants,
    sha256_file,
    stable_hash,
)
from chiptherm.ml.models import build_model, count_parameters


CONTINUATION_EPOCHS = (5, 10, 15, 20)
EXPECTED_PARENT_EPOCH = 94
EXPECTED_EPOCHS = 20
EXPECTED_INITIAL_LR = 1.0e-4
EXPECTED_FINAL_LR = 1.0e-5
EXPECTED_WEIGHT_DECAY = 1.0e-2
EXPECTED_BATCH_SIZE = 64
EXPECTED_SEED = 1
VALIDATION_PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
)
PRIMARY_PROTOCOL = "primary_test_families"
INTENTIONAL_OVERRIDES = {
    "epochs",
    "lr",
    "scheduler",
    "cosine_eta_min",
    "checkpoint_frequency",
    "weight_decay",
    "ema_enabled",
    "swa_enabled",
    "warmup_epochs",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def continuation_lr(epoch: int) -> float:
    if not 0 <= epoch <= EXPECTED_EPOCHS:
        raise ValueError(f"continuation epoch must be in [0, 20], got {epoch}")
    cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / EXPECTED_EPOCHS))
    return EXPECTED_FINAL_LR + (
        EXPECTED_INITIAL_LR - EXPECTED_FINAL_LR
    ) * cosine


def validate_continuation_config(
    canonical: Mapping[str, Any],
    continuation: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "lr": EXPECTED_INITIAL_LR,
        "weight_decay": EXPECTED_WEIGHT_DECAY,
        "scheduler": "cosine",
        "cosine_eta_min": EXPECTED_FINAL_LR,
        "checkpoint_frequency": 5,
        "ema_enabled": False,
        "swa_enabled": False,
        "warmup_epochs": 0,
        "model_architecture": EXPECTED_ARCHITECTURE,
        "physics_input": "source_superposition_v1",
        "mean_head_mode": "residual_resistance",
        "physical_representation": "dimensional",
        "channel_routing_mode": "dimensional_baseline",
    }
    violations = {
        key: {"expected": expected, "actual": continuation.get(key)}
        for key, expected in required.items()
        if continuation.get(key) != expected
    }
    canonical_weight_decay = float(
        canonical.get("weight_decay", EXPECTED_WEIGHT_DECAY)
    )
    if canonical_weight_decay != EXPECTED_WEIGHT_DECAY:
        violations["canonical_weight_decay"] = {
            "expected": EXPECTED_WEIGHT_DECAY,
            "actual": canonical_weight_decay,
        }
    changed: dict[str, dict[str, Any]] = {}
    preserved: dict[str, Any] = {}
    for key, canonical_value in canonical.items():
        if key == "schema_version":
            continue
        continuation_value = continuation.get(key, canonical_value)
        if continuation_value != canonical_value:
            changed[key] = {
                "canonical": canonical_value,
                "continuation": continuation_value,
            }
        else:
            preserved[key] = continuation_value
    unexpected = sorted(set(changed) - INTENTIONAL_OVERRIDES)
    missing_preserved = sorted(
        key
        for key in canonical
        if key not in {"schema_version"}
        and key not in INTENTIONAL_OVERRIDES
        and key not in continuation
    )
    if unexpected:
        violations["unexpected_overrides"] = unexpected
    if missing_preserved:
        violations["missing_preserved_keys"] = missing_preserved
    if violations:
        raise ValueError(f"invalid continuation configuration: {violations}")
    return {
        "schema_version": "compact_low_lr_training_config_diff/1",
        "canonical": dict(canonical),
        "continuation": dict(continuation),
        "intentional_overrides": changed,
        "added_explicit_controls": {
            key: continuation[key]
            for key in sorted(INTENTIONAL_OVERRIDES)
            if key not in canonical
        },
        "preserved": preserved,
        "optimizer": "AdamW",
        "canonical_weight_decay": canonical_weight_decay,
        "warmup": False,
        "ema": False,
        "swa": False,
        "lr_schedule": {
            "type": "CosineAnnealingLR",
            "initial_lr": EXPECTED_INITIAL_LR,
            "final_lr": EXPECTED_FINAL_LR,
            "t_max": EXPECTED_EPOCHS,
            "values_after_epoch": {
                str(epoch): continuation_lr(epoch)
                for epoch in (0, *CONTINUATION_EPOCHS)
            },
        },
    }


def validate_parent_checkpoint(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    invariants = model_invariants(checkpoint)
    expected = {
        "architecture": EXPECTED_ARCHITECTURE,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "prediction_mode": EXPECTED_PREDICTION_MODE,
        "reconstruction": EXPECTED_RECONSTRUCTION,
        "physics_input_mode": "source_superposition_v1",
        "mean_head_mode": "residual_resistance",
        "physical_representation": "dimensional",
        "channel_routing_mode": "dimensional_baseline",
    }
    mismatches = {
        key: {"expected": value, "actual": invariants.get(key)}
        for key, value in expected.items()
        if invariants.get(key) != value
    }
    if int(checkpoint.get("epoch", -1)) != EXPECTED_PARENT_EPOCH:
        mismatches["epoch"] = {
            "expected": EXPECTED_PARENT_EPOCH,
            "actual": checkpoint.get("epoch"),
        }
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        mismatches["model_state_dict"] = "missing or empty"
    elif not all(
        isinstance(value, torch.Tensor) and torch.isfinite(value).all()
        for value in state.values()
    ):
        mismatches["model_state_dict"] = "contains non-tensor or non-finite values"
    model = build_model(
        {
            **checkpoint["model_config"],
            "prediction_mode": EXPECTED_PREDICTION_MODE,
        }
    )
    model_state = model.state_dict()
    state_compatible = (
        isinstance(state, Mapping)
        and tuple(state.keys()) == tuple(model_state.keys())
        and all(
            tuple(state[name].shape) == tuple(model_state[name].shape)
            and state[name].dtype == model_state[name].dtype
            for name in model_state
        )
    )
    if not state_compatible:
        mismatches["state_compatibility"] = "state keys, shapes, or dtypes differ"
    if count_parameters(model) != EXPECTED_PARAMETER_COUNT:
        mismatches["rebuilt_parameter_count"] = count_parameters(model)
    if mismatches:
        raise ValueError(f"canonical parent checkpoint is incompatible: {mismatches}")
    return {
        "schema_version": "compact_low_lr_initialization_compatibility/1",
        "compatible": True,
        "parent_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": EXPECTED_PARENT_EPOCH,
            "weights": "model_state_dict",
        },
        "invariants": invariants,
        "state_tensor_count": len(model_state),
        "strict_state_match": True,
        "initialization_mode": "weights_only",
        "training_state": {
            "optimizer_restored": False,
            "scheduler_restored": False,
            "epoch_restored": False,
            "ema_restored": False,
            "swa_restored": False,
            "start_epoch": 1,
            "new_lineage": True,
        },
        "normalization_sha256": stable_hash(checkpoint["normalization"]),
    }


def checkpoint_id(epoch: int) -> str:
    if epoch not in CONTINUATION_EPOCHS:
        raise ValueError(f"unsupported continuation checkpoint epoch: {epoch}")
    return f"continuation_epoch{epoch:03d}"


def checkpoint_path(experiment_root: Path, epoch: int) -> Path:
    return experiment_root / "checkpoints" / f"epoch_{epoch:04d}.pt"


def evaluation_path(
    experiment_root: Path,
    epoch: int,
    protocol: str,
) -> Path:
    stage = (
        "evaluation_primary_test"
        if protocol == PRIMARY_PROTOCOL
        else "evaluation_selection"
    )
    return experiment_root / stage / checkpoint_id(epoch) / protocol


def selection_thresholds() -> dict[str, float]:
    return {
        "known_family_mae_limit_K": KNOWN_MAE_LIMIT_K,
        "heldout_validation_mae_limit_K": VALIDATION_MAE_LIMIT_K,
        "max_per_family_regression_K": MAX_FAMILY_REGRESSION_K,
        "max_fraction_worse_absolute_increase": (
            MAX_FRACTION_WORSE_ABSOLUTE_INCREASE
        ),
        "max_hotspot_abs_error_increase_K": (
            MAX_HOTSPOT_ABS_ERROR_INCREASE_K
        ),
        "known_mae_tie_K": KNOWN_MAE_TIE_K,
    }


def select_checkpoint(
    metrics: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {(row["model"], row["protocol"]): row for row in metrics}
    family_lookup = {
        (row["model"], row["protocol"], row["family_uid"]): row
        for row in families
    }
    canonical_validation = lookup.get(
        ("canonical_reference", "primary_validation_families")
    )
    if canonical_validation is None:
        return {"status": "pending", "reason": "canonical validation missing"}
    canonical_families = {
        row["family_uid"]: row
        for row in families
        if row["model"] == "canonical_reference"
        and row["protocol"] == "primary_validation_families"
    }
    candidates: list[dict[str, Any]] = []
    for epoch in CONTINUATION_EPOCHS:
        model = checkpoint_id(epoch)
        known = lookup.get((model, "known_family_sample_test"))
        validation = lookup.get((model, "primary_validation_families"))
        if known is None or validation is None:
            candidates.append(
                {"epoch": epoch, "run_id": model, "status": "pending"}
            )
            continue
        deltas = {
            family: float(
                family_lookup[(model, "primary_validation_families", family)][
                    "mae_K"
                ]
            )
            - float(reference["mae_K"])
            for family, reference in canonical_families.items()
            if (model, "primary_validation_families", family) in family_lookup
        }
        complete = set(deltas) == set(canonical_families)
        max_regression = max(deltas.values()) if complete else float("inf")
        checks = {
            "known_mae": float(known["micro_mae_K"]) <= KNOWN_MAE_LIMIT_K,
            "validation_mae": (
                float(validation["micro_mae_K"]) <= VALIDATION_MAE_LIMIT_K
            ),
            "per_family_regression": (
                complete and max_regression <= MAX_FAMILY_REGRESSION_K
            ),
            "fraction_worse": (
                float(validation["fraction_worse_than_source"])
                <= float(canonical_validation["fraction_worse_than_source"])
                + MAX_FRACTION_WORSE_ABSOLUTE_INCREASE
            ),
            "hotspot_error": (
                float(validation["hotspot_temperature_abs_error_K"])
                <= float(canonical_validation["hotspot_temperature_abs_error_K"])
                + MAX_HOTSPOT_ABS_ERROR_INCREASE_K
            ),
            "outputs_finite": bool(known["outputs_finite"])
            and bool(validation["outputs_finite"]),
        }
        candidates.append(
            {
                "epoch": epoch,
                "run_id": model,
                "status": "admissible" if all(checks.values()) else "rejected",
                "known_family_mae_K": float(known["micro_mae_K"]),
                "heldout_validation_mae_K": float(
                    validation["micro_mae_K"]
                ),
                "max_validation_family_regression_K": max_regression,
                "fraction_worse_than_source": float(
                    validation["fraction_worse_than_source"]
                ),
                "hotspot_temperature_abs_error_K": float(
                    validation["hotspot_temperature_abs_error_K"]
                ),
                "checks": checks,
                "family_deltas_K": deltas,
            }
        )
    if any(row["status"] == "pending" for row in candidates):
        return {
            "status": "pending",
            "candidates": candidates,
            "thresholds": selection_thresholds(),
        }
    admissible = [
        row for row in candidates if row["status"] == "admissible"
    ]
    if not admissible:
        return {
            "status": "no_candidate",
            "reason": "no continuation checkpoint satisfies every frozen safeguard",
            "candidates": candidates,
            "thresholds": selection_thresholds(),
        }
    best_known = min(row["known_family_mae_K"] for row in admissible)
    tie_group = [
        row
        for row in admissible
        if row["known_family_mae_K"] <= best_known + KNOWN_MAE_TIE_K
    ]
    selected = min(
        tie_group,
        key=lambda row: (
            row["heldout_validation_mae_K"],
            row["known_family_mae_K"],
            row["epoch"],
        ),
    )
    return {
        "status": "selected",
        "selected_epoch": selected["epoch"],
        "selected_run_id": selected["run_id"],
        "selection_order": (
            "lowest known-family MAE; candidates within 0.002 K form a tie "
            "group resolved by lower held-out-validation MAE"
        ),
        "selected": selected,
        "candidates": candidates,
        "thresholds": selection_thresholds(),
    }


def validation_fingerprint(
    metrics: Sequence[Mapping[str, Any]],
    *,
    checkpoint_inventory_sha256: str,
) -> str:
    records = sorted(
        (
            {
                "model": row["model"],
                "protocol": row["protocol"],
                "metrics_sha256": row["metrics_sha256"],
                "sample_metrics_sha256": row["sample_metrics_sha256"],
            }
            for row in metrics
            if row["protocol"] in VALIDATION_PROTOCOLS
        ),
        key=lambda row: (row["model"], row["protocol"]),
    )
    return stable_hash(
        {
            "validation_artifacts": records,
            "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
            "thresholds": selection_thresholds(),
        }
    )


def authorize_primary_test(
    gate: Mapping[str, Any],
    *,
    validation_fingerprint_value: str | None = None,
) -> int:
    if gate.get("status") != "frozen":
        raise ValueError("primary-test evaluation requires a frozen validation gate")
    selection = gate.get("selection") or {}
    if selection.get("status") != "selected":
        raise ValueError(
            "primary-test evaluation requires exactly one selected checkpoint"
        )
    if validation_fingerprint_value is not None and gate.get(
        "validation_fingerprint"
    ) != validation_fingerprint_value:
        raise ValueError("validation fingerprint differs from the frozen gate")
    epoch = int(selection["selected_epoch"])
    if epoch not in CONTINUATION_EPOCHS:
        raise ValueError(f"frozen gate selected an invalid epoch: {epoch}")
    return epoch


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
