#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.compact_weight_interpolation import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    FROZEN_ALPHAS,
    alpha_run_id,
    build_mixed_checkpoint,
    compatibility_report,
    interpolate_state_dict,
    load_checkpoint,
    now_utc,
    select_endpoint_states,
    sha256_file,
    validate_alpha_grid,
)
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def endpoint_state_check(
    mixed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    exact = True
    maximum = 0.0
    for name, tensor in mixed.items():
        reference = expected[name]
        exact = exact and torch.equal(tensor, reference)
        if tensor.is_floating_point() or tensor.is_complex():
            maximum = max(
                maximum,
                float((tensor - reference).abs().max().item()),
            )
        elif not torch.equal(tensor, reference):
            maximum = float("inf")
    return {
        "state_exact": exact,
        "state_max_abs_difference": maximum,
        "tensor_count": len(mixed),
    }


def build_artifacts(
    *,
    canonical_path: Path,
    cosine_path: Path,
    out_root: Path,
    alphas: tuple[float, ...],
    execute: bool,
) -> dict[str, Any]:
    canonical = load_checkpoint(canonical_path)
    cosine = load_checkpoint(cosine_path)
    compatibility = compatibility_report(canonical, cosine)
    canonical_state, cosine_state = select_endpoint_states(canonical, cosine)
    canonical_sha256 = sha256_file(canonical_path)
    cosine_sha256 = sha256_file(cosine_path)
    model = build_model(
        {
            **canonical["model_config"],
            "prediction_mode": "residual_decomposed",
        }
    )
    if count_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            "real model builder does not reproduce the frozen compact "
            f"parameter count {EXPECTED_PARAMETER_COUNT}"
        )
    runs: list[dict[str, Any]] = []
    endpoint_checks: dict[str, Any] = {
        "schema_version": "compact_weight_interpolation_endpoints/1",
        "state_tolerance": 0.0,
        "metric_status": "pending_manual_evaluation",
    }
    for alpha in alphas:
        run_id = alpha_run_id(alpha)
        mixed_state = interpolate_state_dict(
            canonical_state,
            cosine_state,
            alpha,
        )
        model.load_state_dict(mixed_state, strict=True)
        if not all(torch.isfinite(value).all() for value in mixed_state.values()):
            raise ValueError(f"mixed state contains NaN/Inf at alpha={alpha}")
        checkpoint = build_mixed_checkpoint(
            canonical=canonical,
            cosine=cosine,
            mixed_state=mixed_state,
            alpha=alpha,
            canonical_path=canonical_path,
            cosine_path=cosine_path,
            canonical_sha256=canonical_sha256,
            cosine_sha256=cosine_sha256,
        )
        checkpoint_path = out_root / run_id / "checkpoints/interpolated.pt"
        record = {
            "run_id": run_id,
            "alpha": alpha,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_validation_root": str(
                out_root / run_id / "evaluation_validation"
            ),
            "evaluation_primary_test_root": str(
                out_root / run_id / "evaluation_primary_test"
            ),
            "status": "planned" if not execute else "built",
        }
        if alpha == 0.0:
            endpoint_checks["alpha000"] = endpoint_state_check(
                mixed_state, canonical_state
            )
        if alpha == 1.0:
            endpoint_checks["alpha100"] = endpoint_state_check(
                mixed_state, cosine_state
            )
        if execute:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite interpolated checkpoint: {checkpoint_path}"
                )
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
            loaded = load_checkpoint(checkpoint_path)
            model.load_state_dict(loaded["model_state_dict"], strict=True)
            if loaded.get("evaluation_default_weights") != "raw":
                raise ValueError("mixed checkpoint is not marked for raw evaluation")
            record["checkpoint_sha256"] = sha256_file(checkpoint_path)
        runs.append(record)
    manifest = {
        "schema_version": "compact_weight_interpolation_manifest/1",
        "created_at_utc": now_utc(),
        "status": "built" if execute else "dry_run_validated",
        "post_training_diagnostic": True,
        "training_launched": False,
        "evaluation_launched": False,
        "canonical_parent": {
            "path": str(canonical_path),
            "sha256": canonical_sha256,
            "weights": "model_state_dict",
            "epoch": int(canonical.get("epoch", -1)),
        },
        "cosine_parent": {
            "path": str(cosine_path),
            "sha256": cosine_sha256,
            "weights": "ema_model_state_dict",
            "epoch": int(cosine.get("epoch", -1)),
        },
        "alphas": list(alphas),
        "formula": (
            "W_alpha = (1 - alpha) * W_canonical_raw + "
            "alpha * W_cosine_epoch100_ema"
        ),
        "runs": runs,
    }
    if execute:
        out_root.mkdir(parents=True, exist_ok=True)
        write_json(
            out_root / "checkpoint_compatibility_report.json",
            compatibility,
        )
        write_json(
            out_root / "endpoint_reproduction_report.json",
            endpoint_checks,
        )
        write_json(out_root / "interpolation_manifest.json", manifest)
    return {
        "manifest": manifest,
        "compatibility": compatibility,
        "endpoint_checks": endpoint_checks,
    }


def verify_existing(
    out_root: Path,
    *,
    expected_parent_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest_path = out_root / "interpolation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"interpolation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_parent_hashes = {
        "canonical": str(manifest.get("canonical_parent", {}).get("sha256", "")),
        "cosine": str(manifest.get("cosine_parent", {}).get("sha256", "")),
    }
    if expected_parent_hashes is not None:
        for name, expected in expected_parent_hashes.items():
            if manifest_parent_hashes.get(name) != expected:
                raise ValueError(
                    f"{name} parent hash changed since interpolation build"
                )
    verified = []
    for run in manifest.get("runs", []):
        checkpoint_path = Path(run["checkpoint_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"interpolated checkpoint is missing: {checkpoint_path}"
            )
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != run.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        expected_alpha = float(run["alpha"])
        lineage = checkpoint.get("training_lineage") or {}
        checks = {
            "post_training_interpolated_checkpoint": (
                checkpoint.get("post_training_interpolated_checkpoint") is True
            ),
            "not_resumable": (
                checkpoint.get("resumable_training_checkpoint") is False
            ),
            "raw_evaluation": (
                checkpoint.get("evaluation_default_weights") == "raw"
            ),
            "no_ema_state": checkpoint.get("ema_model_state_dict") is None,
            "no_optimizer_state": checkpoint.get("optimizer_state_dict") is None,
            "no_scheduler_state": checkpoint.get("scheduler_state_dict") is None,
            "alpha": float(checkpoint.get("alpha", -1.0)) == expected_alpha,
            "parameter_count": (
                int(checkpoint.get("parameter_count", -1))
                == EXPECTED_PARAMETER_COUNT
            ),
            "finite_state": all(
                torch.isfinite(value).all()
                for value in checkpoint["model_state_dict"].values()
            ),
            "canonical_parent_hash": (
                lineage.get("canonical_parent", {}).get("sha256")
                == manifest_parent_hashes["canonical"]
            ),
            "cosine_parent_hash": (
                lineage.get("cosine_parent", {}).get("sha256")
                == manifest_parent_hashes["cosine"]
            ),
        }
        if not all(checks.values()):
            raise ValueError(
                f"interpolated checkpoint metadata verification failed: "
                f"{checkpoint_path}: {checks}"
            )
        verified.append(
            {
                "run_id": run["run_id"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": actual_sha256,
                "checks": checks,
            }
        )
    return {
        "schema_version": "compact_weight_interpolation_verification/1",
        "passed": True,
        "verified_checkpoints": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and optionally build the five frozen compact-CNN "
            "weight-interpolation checkpoints."
        )
    )
    parser.add_argument("--canonical-checkpoint", required=True, type=Path)
    parser.add_argument("--cosine-checkpoint", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(FROZEN_ALPHAS),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    alphas = validate_alpha_grid(args.alphas)
    out_root = args.out_root.expanduser().resolve()
    report = build_artifacts(
        canonical_path=args.canonical_checkpoint.expanduser().resolve(),
        cosine_path=args.cosine_checkpoint.expanduser().resolve(),
        out_root=out_root,
        alphas=alphas,
        execute=bool(args.execute),
    )
    if args.verify_existing:
        verification = verify_existing(
            out_root,
            expected_parent_hashes={
                "canonical": report["manifest"]["canonical_parent"]["sha256"],
                "cosine": report["manifest"]["cosine_parent"]["sha256"],
            },
        )
        write_json(
            out_root / "checkpoint_verification_report.json",
            verification,
        )
        print(
            f"Verified checkpoints: {len(verification['verified_checkpoints'])}"
        )
    print(
        "Compatibility:",
        "PASS" if report["compatibility"]["compatible"] else "FAIL",
    )
    print("State tensors:", report["compatibility"]["state_tensor_count"])
    print(
        "Mode:",
        "EXECUTE"
        if args.execute
        else "VERIFY"
        if args.verify_existing
        else "DRY RUN",
    )
    for run in report["manifest"]["runs"]:
        print(f"{run['run_id']}: alpha={run['alpha']:.2f} {run['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
