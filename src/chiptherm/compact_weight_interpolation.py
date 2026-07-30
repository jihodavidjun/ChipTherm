from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


FROZEN_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_ARCHITECTURE = (
    "miniunet_refine_conditioned_decomposed_feature_fusion"
)
EXPECTED_PARAMETER_COUNT = 2_188_803
EXPECTED_PREDICTION_MODE = "residual_decomposed"
EXPECTED_RECONSTRUCTION = (
    "source_superposition_base_K + total_power_W * "
    "delta_R_eff_K_per_W + zero_mean_centered_field_K"
)
KNOWN_MAE_LIMIT_K = 0.135
VALIDATION_MAE_LIMIT_K = 0.940
MAX_FAMILY_REGRESSION_K = 0.10
MAX_FRACTION_WORSE_ABSOLUTE_INCREASE = 0.01
MAX_HOTSPOT_ABS_ERROR_INCREASE_K = 0.05
KNOWN_MAE_TIE_K = 0.002
ENDPOINT_METRIC_TOLERANCE_K = 1.0e-4
ENDPOINT_FRACTION_TOLERANCE = 1.0e-6


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def alpha_run_id(alpha: float) -> str:
    value = float(alpha)
    if value not in FROZEN_ALPHAS:
        raise ValueError(
            f"alpha {value} is outside the frozen grid {list(FROZEN_ALPHAS)}"
        )
    return f"compact_soup_alpha{int(round(value * 100)):03d}"


def validate_alpha_grid(alphas: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in alphas)
    if normalized != FROZEN_ALPHAS:
        raise ValueError(
            "the bounded experiment requires exactly "
            f"{list(FROZEN_ALPHAS)}, got {list(normalized)}"
        )
    return normalized


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def prediction_mode(checkpoint: Mapping[str, Any]) -> str:
    model = checkpoint.get("model_config", {})
    training = checkpoint.get("training_config", {})
    explicit = model.get("prediction_mode") or training.get("prediction_mode")
    if explicit:
        return str(explicit)
    architecture = str(model.get("architecture", ""))
    if "decomposed" in architecture:
        return "residual_decomposed"
    raise ValueError("checkpoint has no unambiguous prediction mode")


def source_version(checkpoint: Mapping[str, Any]) -> str:
    lineage = checkpoint.get("training_lineage") or {}
    value = (
        checkpoint.get("source_version")
        or lineage.get("source_superposition_version")
    )
    if not value:
        raise ValueError("checkpoint is missing source-superposition version")
    return str(value)


def lineage_hash(checkpoint: Mapping[str, Any], name: str) -> str:
    lineage = checkpoint.get("training_lineage") or {}
    value = lineage.get(name)
    if not value:
        raise ValueError(f"checkpoint lineage is missing {name}")
    return str(value)


def model_invariants(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    model = checkpoint.get("model_config") or {}
    training = checkpoint.get("training_config") or {}
    lineage = checkpoint.get("training_lineage") or {}
    reconstruction = lineage.get("reconstruction")
    return {
        "architecture": model.get("architecture"),
        "parameter_count": int(
            checkpoint.get(
                "parameter_count",
                model.get("total_parameters", -1),
            )
        ),
        "input_channels": int(
            model.get("model_input_channels", model.get("input_channels", -1))
        ),
        "output_channels": int(model.get("output_channels", -1)),
        "prediction_mode": prediction_mode(checkpoint),
        "target": training.get("target"),
        "target_decomposition": bool(training.get("target_decomposition", False)),
        "physical_representation": model.get(
            "physical_representation",
            training.get("physical_representation"),
        ),
        "channel_routing_mode": model.get(
            "channel_routing_mode",
            training.get("channel_routing_mode"),
        ),
        "mean_head_mode": model.get(
            "mean_head_mode",
            training.get("mean_head_mode"),
        ),
        "physics_input_mode": model.get(
            "physics_input_mode",
            training.get("physics_input_mode"),
        ),
        "source_version": source_version(checkpoint),
        "train_index_sha256": lineage_hash(
            checkpoint, "train_index_sha256"
        ),
        "internal_val_index_sha256": lineage_hash(
            checkpoint, "internal_val_index_sha256"
        ),
        "metadata_dim": int(model.get("metadata_dim", -1)),
        "metadata_feature_names": tuple(
            str(value) for value in model.get("metadata_feature_names", ())
        ),
        "delta_R_eff_mean_K_per_W": float(
            model.get("delta_R_eff_target_mean_K_per_W", math.nan)
        ),
        "delta_R_eff_std_K_per_W": float(
            model.get("delta_R_eff_target_std_K_per_W", math.nan)
        ),
        "reconstruction": reconstruction,
        "mean_correction_sign": 1,
        "centered_correction_sign": 1,
    }


def select_endpoint_states(
    canonical: Mapping[str, Any],
    cosine: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    canonical_state = canonical.get("model_state_dict")
    cosine_state = cosine.get("ema_model_state_dict")
    if not isinstance(canonical_state, Mapping):
        raise ValueError("canonical checkpoint has no model_state_dict")
    if not isinstance(cosine_state, Mapping):
        raise ValueError(
            "cosine checkpoint has no ema_model_state_dict; raw weights are "
            "not an allowed endpoint"
        )
    return canonical_state, cosine_state


def compatibility_report(
    canonical: Mapping[str, Any],
    cosine: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_invariants = model_invariants(canonical)
    cosine_invariants = model_invariants(cosine)
    if canonical_invariants != cosine_invariants:
        differences = {
            key: {
                "canonical": canonical_invariants.get(key),
                "cosine": cosine_invariants.get(key),
            }
            for key in canonical_invariants
            if canonical_invariants.get(key) != cosine_invariants.get(key)
        }
        raise ValueError(f"checkpoint semantic incompatibility: {differences}")
    if canonical_invariants["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"unexpected architecture: {canonical_invariants['architecture']}"
        )
    if canonical_invariants["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            "checkpoint parameter count does not match frozen compact model: "
            f"{canonical_invariants['parameter_count']}"
        )
    if canonical_invariants["prediction_mode"] != EXPECTED_PREDICTION_MODE:
        raise ValueError("checkpoint prediction mode is not residual_decomposed")
    if canonical_invariants["reconstruction"] != EXPECTED_RECONSTRUCTION:
        raise ValueError(
            "checkpoint reconstruction semantics differ from the frozen "
            "source-plus-residual formulation"
        )
    if canonical.get("normalization") != cosine.get("normalization"):
        raise ValueError("checkpoint normalization statistics differ")
    canonical_state, cosine_state = select_endpoint_states(canonical, cosine)
    if tuple(canonical_state.keys()) != tuple(cosine_state.keys()):
        missing = sorted(set(canonical_state) - set(cosine_state))
        extra = sorted(set(cosine_state) - set(canonical_state))
        raise ValueError(
            f"state-dict keys differ: missing={missing}, extra={extra}"
        )
    tensor_rows: list[dict[str, Any]] = []
    floating_count = 0
    nonfloating_count = 0
    for name, canonical_tensor in canonical_state.items():
        cosine_tensor = cosine_state[name]
        if not isinstance(canonical_tensor, torch.Tensor) or not isinstance(
            cosine_tensor, torch.Tensor
        ):
            raise ValueError(f"state entry is not a tensor: {name}")
        if canonical_tensor.shape != cosine_tensor.shape:
            raise ValueError(
                f"state tensor shape differs for {name}: "
                f"{tuple(canonical_tensor.shape)} vs {tuple(cosine_tensor.shape)}"
            )
        if canonical_tensor.dtype != cosine_tensor.dtype:
            raise ValueError(
                f"state tensor dtype differs for {name}: "
                f"{canonical_tensor.dtype} vs {cosine_tensor.dtype}"
            )
        if not torch.isfinite(canonical_tensor).all() or not torch.isfinite(
            cosine_tensor
        ).all():
            raise ValueError(f"state tensor contains NaN/Inf: {name}")
        floating = canonical_tensor.is_floating_point() or canonical_tensor.is_complex()
        if floating:
            floating_count += 1
        else:
            nonfloating_count += 1
            if not torch.equal(canonical_tensor, cosine_tensor):
                raise ValueError(
                    f"non-floating state differs and cannot be interpolated: {name}"
                )
        tensor_rows.append(
            {
                "name": name,
                "shape": list(canonical_tensor.shape),
                "dtype": str(canonical_tensor.dtype),
                "floating": floating,
            }
        )
    return {
        "schema_version": "compact_weight_interpolation_compatibility/1",
        "compatible": True,
        "invariants": json_safe(canonical_invariants),
        "normalization_sha256": stable_hash(canonical["normalization"]),
        "state_tensor_count": len(tensor_rows),
        "floating_tensor_count": floating_count,
        "nonfloating_tensor_count": nonfloating_count,
        "batchnorm_or_running_state_present": any(
            "running_mean" in row["name"]
            or "running_var" in row["name"]
            or "num_batches_tracked" in row["name"]
            for row in tensor_rows
        ),
        "state_tensors": tensor_rows,
    }


def interpolate_state_dict(
    canonical_state: Mapping[str, torch.Tensor],
    cosine_state: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    output: dict[str, torch.Tensor] = {}
    for name, canonical_tensor in canonical_state.items():
        if name not in cosine_state:
            raise ValueError(f"cosine state is missing {name}")
        cosine_tensor = cosine_state[name]
        if canonical_tensor.shape != cosine_tensor.shape:
            raise ValueError(f"shape mismatch for {name}")
        if canonical_tensor.dtype != cosine_tensor.dtype:
            raise ValueError(f"dtype mismatch for {name}")
        floating = canonical_tensor.is_floating_point() or canonical_tensor.is_complex()
        if not floating:
            if not torch.equal(canonical_tensor, cosine_tensor):
                raise ValueError(f"non-floating state differs for {name}")
            output[name] = canonical_tensor.detach().clone()
        elif value == 0.0:
            output[name] = canonical_tensor.detach().clone()
        elif value == 1.0:
            output[name] = cosine_tensor.detach().clone()
        else:
            output[name] = torch.lerp(
                canonical_tensor,
                cosine_tensor,
                value,
            )
    if set(output) != set(cosine_state):
        raise ValueError("state dictionaries contain different keys")
    return output


def build_mixed_checkpoint(
    *,
    canonical: Mapping[str, Any],
    cosine: Mapping[str, Any],
    mixed_state: Mapping[str, torch.Tensor],
    alpha: float,
    canonical_path: Path,
    cosine_path: Path,
    canonical_sha256: str,
    cosine_sha256: str,
) -> dict[str, Any]:
    model_config = copy.deepcopy(canonical["model_config"])
    model_config["prediction_mode"] = EXPECTED_PREDICTION_MODE
    training_config = {
        "schema_version": "compact_weight_interpolation_training_config/1",
        "prediction_mode": EXPECTED_PREDICTION_MODE,
        "post_training_interpolation": True,
        "alpha": float(alpha),
        "formula": (
            "W_alpha = (1 - alpha) * W_canonical_raw + "
            "alpha * W_cosine_epoch100_ema"
        ),
        "resumable_training_checkpoint": False,
    }
    canonical_lineage = canonical.get("training_lineage") or {}
    lineage = {
        "schema_version": "compact_weight_interpolation_lineage/1",
        "post_training_interpolated_checkpoint": True,
        "alpha": float(alpha),
        "canonical_parent": {
            "path": str(canonical_path),
            "sha256": canonical_sha256,
            "weights": "model_state_dict",
        },
        "cosine_parent": {
            "path": str(cosine_path),
            "sha256": cosine_sha256,
            "weights": "ema_model_state_dict",
        },
        "source_superposition_version": source_version(canonical),
        "train_index_sha256": canonical_lineage["train_index_sha256"],
        "internal_val_index_sha256": canonical_lineage[
            "internal_val_index_sha256"
        ],
        "reconstruction": EXPECTED_RECONSTRUCTION,
        "primary_heldout_used_for_selection": False,
    }
    return {
        "schema_version": "compact_weight_interpolation_checkpoint/1",
        "created_at_utc": now_utc(),
        "post_training_interpolated_checkpoint": True,
        "resumable_training_checkpoint": False,
        "alpha": float(alpha),
        "interpolation_formula": training_config["formula"],
        "model_state_dict": dict(mixed_state),
        "ema_model_state_dict": None,
        "evaluation_default_weights": "raw",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_config": model_config,
        "training_config": training_config,
        "normalization": copy.deepcopy(canonical["normalization"]),
        "training_lineage": lineage,
        "source_version": source_version(canonical),
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value

