from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .ml.models import build_model, count_parameters


VARIANTS = ("param_matched_constant", "param_matched_cosine_ema")
RUN_IDS = {
    "small_cosine_ema": "feature_fusion_train40_cosine_ema_seed1",
    "param_matched_constant": "feature_fusion_train40_param_matched_constant_seed1",
    "param_matched_cosine_ema": (
        "feature_fusion_train40_param_matched_cosine_ema_seed1"
    ),
}
CANONICAL_RUN_ID = "feature_fusion_train40_source_v1_seed1"
SOURCE_VERSION = "source_superposition_final_train40_source_v1"
PARAMETER_TARGET_RANGE = (3_800_000, 4_200_000)
U_FNO_PARAMETER_COUNT = 4_025_634
SAU_FNO_PARAMETER_COUNT = 4_028_802
COSINE_RECIPE_CHANGE_KEYS = {
    "epochs",
    "scheduler",
    "early_stopping_patience",
    "cosine_eta_min",
    "ema_enabled",
    "ema_decay",
}
WIDTH_CHANGE_KEYS = {"base_channels", "refine_channels", "global_hidden_channels"}


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_variant_config(
    canonical: Mapping[str, Any],
    candidate: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    changed = {
        key
        for key in set(canonical) | set(candidate)
        if canonical.get(key) != candidate.get(key)
    }
    allowed = set(WIDTH_CHANGE_KEYS)
    if variant == "param_matched_cosine_ema":
        allowed |= COSINE_RECIPE_CHANGE_KEYS
    unexpected = sorted(changed - allowed)
    if unexpected:
        raise ValueError(f"{variant} changes forbidden canonical fields: {unexpected}")
    required = (
        {
            "epochs": 150,
            "scheduler": "cosine",
            "early_stopping_patience": 30,
            "cosine_eta_min": 1.0e-5,
            "ema_enabled": True,
            "ema_decay": 0.999,
        }
        if variant == "param_matched_cosine_ema"
        else {
            "epochs": canonical["epochs"],
            "scheduler": canonical["scheduler"],
            "early_stopping_patience": canonical["early_stopping_patience"],
        }
    )
    mismatches = {
        key: {"expected": value, "actual": candidate.get(key)}
        for key, value in required.items()
        if candidate.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{variant} bounded-recipe mismatch: {mismatches}")
    return {
        "variant": variant,
        "changed_keys": sorted(changed),
        "allowed_changed_keys": sorted(allowed),
        "canonical_invariants_preserved": not unexpected and not mismatches,
        "config_sha256": stable_hash(candidate),
    }


def validate_two_factor_configs(
    canonical: Mapping[str, Any],
    small_cosine: Mapping[str, Any],
    wide_constant: Mapping[str, Any],
    wide_cosine: Mapping[str, Any],
) -> dict[str, Any]:
    small_changed = {
        key
        for key in set(canonical) | set(small_cosine)
        if canonical.get(key) != small_cosine.get(key)
    }
    if small_changed != COSINE_RECIPE_CHANGE_KEYS:
        raise ValueError(
            "small cosine config differs from canonical outside the exact recipe "
            f"factor: {sorted(small_changed)}"
        )
    constant_result = validate_variant_config(
        canonical,
        wide_constant,
        "param_matched_constant",
    )
    cosine_result = validate_variant_config(
        canonical,
        wide_cosine,
        "param_matched_cosine_ema",
    )
    wide_recipe_changed = {
        key
        for key in set(wide_constant) | set(wide_cosine)
        if wide_constant.get(key) != wide_cosine.get(key)
    }
    if wide_recipe_changed != COSINE_RECIPE_CHANGE_KEYS:
        raise ValueError(
            "wide cosine config differs from wide constant outside the exact "
            f"recipe factor: {sorted(wide_recipe_changed)}"
        )
    return {
        "small_cosine_vs_canonical_changed_keys": sorted(small_changed),
        "wide_constant": constant_result,
        "wide_cosine": cosine_result,
        "wide_cosine_vs_wide_constant_changed_keys": sorted(
            wide_recipe_changed
        ),
        "two_factor_contract_preserved": True,
    }


def model_config_with_width(
    canonical_model_config: Mapping[str, Any],
    width: int,
) -> dict[str, Any]:
    config = dict(canonical_model_config)
    config.update(
        {
            "base_channels": int(width),
            "refine_channels": int(width),
            "global_hidden_channels": int(width),
        }
    )
    return config


def deterministic_width_search(
    canonical_model_config: Mapping[str, Any],
    *,
    minimum_parameters: int = PARAMETER_TARGET_RANGE[0],
    maximum_parameters: int = PARAMETER_TARGET_RANGE[1],
    maximum_width: int = 96,
) -> dict[str, Any]:
    canonical_width = int(canonical_model_config["base_channels"])
    candidates: list[dict[str, int]] = []
    selected: dict[str, int] | None = None
    for width in range(canonical_width + 1, maximum_width + 1):
        model = build_model(model_config_with_width(canonical_model_config, width))
        parameters = count_parameters(model)
        record = {"width": width, "parameter_count": parameters}
        candidates.append(record)
        if minimum_parameters <= parameters <= maximum_parameters:
            selected = record
            break
    if selected is None:
        raise ValueError(
            f"no equal-width CNN found in parameter range "
            f"[{minimum_parameters}, {maximum_parameters}]"
        )
    return {
        "search": (
            "ascending equal base/refine/global width; first configuration "
            "inside the target range"
        ),
        "target_parameter_range": [minimum_parameters, maximum_parameters],
        "selected_width": selected["width"],
        "selected_parameter_count": selected["parameter_count"],
        "candidates_evaluated": candidates,
    }


def interpolation_decision_gate(
    *,
    canonical_known_mae_K: float,
    canonical_validation_mae_K: float,
    candidate_known_mae_K: float,
    candidate_validation_mae_K: float,
) -> dict[str, Any]:
    relative_improvement = (
        (canonical_known_mae_K - candidate_known_mae_K)
        / max(canonical_known_mae_K, 1.0e-12)
    )
    validation_delta = candidate_validation_mae_K - canonical_validation_mae_K
    primary_success = relative_improvement >= 0.15 and validation_delta <= 0.05
    strong_success = (
        candidate_known_mae_K <= 0.12 and validation_delta <= 0.03
    )
    near_complete = (
        candidate_known_mae_K <= 0.10 and validation_delta <= 0.05
    )
    return {
        "canonical_known_mae_K": canonical_known_mae_K,
        "canonical_validation_mae_K": canonical_validation_mae_K,
        "candidate_known_mae_K": candidate_known_mae_K,
        "candidate_validation_mae_K": candidate_validation_mae_K,
        "known_relative_improvement_fraction": relative_improvement,
        "validation_delta_K": validation_delta,
        "primary_success": primary_success,
        "strong_success": strong_success,
        "near_complete_closure": near_complete,
        "recommend_param_matched_training": not strong_success,
        "decision_basis": "known-family and held-out validation only",
        "primary_test_used": False,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_sample_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty metrics")

    def values(name: str) -> np.ndarray:
        output = [
            float(row[name])
            for row in rows
            if str(row.get(name, "")).strip()
        ]
        return np.asarray(output, dtype=np.float64)

    mae = values("mae_K")
    rmse = values("rmse_K")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family_uid") or row.get("case_id"))].append(
            float(row["mae_K"])
        )
    source = values("physics_baseline_mae_K")
    return {
        "micro_mae_K": float(mae.mean()),
        "micro_rmse_K": float(np.sqrt(np.mean(rmse * rmse))),
        "macro_family_mae_K": float(
            np.mean([np.mean(items) for items in grouped.values()])
        ),
        "worst_family_mae_K": float(
            max(np.mean(items) for items in grouped.values())
        ),
        "centered_field_mae_K": float(values("centered_field_mae_K").mean()),
        "mean_correction_mae_K": float(values("mean_head_abs_error_K").mean()),
        "hotspot_top1pct_mae_K": float(values("hotspot_top1pct_mae_K").mean()),
        "boundary_mae_K": float(values("boundary_region_mae_K").mean()),
        "fraction_worse_than_source": float(np.mean(mae > source)),
    }
