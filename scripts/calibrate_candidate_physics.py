#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.physics_candidates import (  # noqa: E402
    PackageGridMetadata,
    PhysicsCandidateConfig,
    extract_package_grid_metadata,
    power_density_source,
    screened_poisson_rise,
)

from generate_candidate_physics import generate_candidate, optional_float, read_index, resolve_path  # noqa: E402

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


@dataclass(frozen=True)
class CalibrationSample:
    sample_uid: str
    case_id: str
    q_W_per_mm2: np.ndarray
    target_K: np.ndarray
    metadata: PackageGridMetadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit globally shared constrained parameters for candidate physics priors.")
    parser.add_argument("--source-root", default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean/package_plus_power", type=Path)
    parser.add_argument("--candidate", default="screened_poisson", choices=["screened_poisson"])
    parser.add_argument("--train-index", default=None, type=Path)
    parser.add_argument("--val-index", default=None, type=Path)
    parser.add_argument("--test-index", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ambient-K", default=318.15, type=float)
    parser.add_argument("--k-spread-W-per-K", default=0.30, type=float)
    parser.add_argument("--g-sink-W-per-mm2K", default=0.004, type=float)
    parser.add_argument("--source-scale", default=1.0, type=float)
    parser.add_argument("--ambient-offset-K", default=0.0, type=float)
    parser.add_argument("--global-R-eff-K-per-W", default=0.0, type=float)
    parser.add_argument("--fit-k", action="store_true")
    parser.add_argument("--fit-g", action="store_true")
    parser.add_argument("--fit-source-scale", action="store_true")
    parser.add_argument("--fit-ambient-offset", action="store_true")
    parser.add_argument("--fit-global-R", action="store_true")
    parser.add_argument("--lambda-case-bias", default=0.0, type=float)
    parser.add_argument("--lambda-hotspot", default=0.0, type=float)
    parser.add_argument("--hotspot-top-frac", default=0.05, type=float)
    parser.add_argument("--maxiter", default=80, type=int)
    parser.add_argument("--max-calibration-samples", default=None, type=int)
    parser.add_argument("--max-generate-samples-per-split", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    if minimize is None:
        raise SystemExit("scipy.optimize is required for constrained calibration")

    source_root = args.source_root.expanduser().resolve()
    train_index = (args.train_index or source_root / "train_index.csv").expanduser().resolve()
    val_index = (args.val_index or source_root / "val_index.csv").expanduser().resolve()
    test_index = (args.test_index or source_root / "test_index.csv").expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = PhysicsCandidateConfig(
        name="screened_poisson",
        ambient_K=args.ambient_K,
        k_spread_W_per_K=args.k_spread_W_per_K,
        g_sink_W_per_mm2K=args.g_sink_W_per_mm2K,
        source_scale=args.source_scale,
        ambient_offset_K=args.ambient_offset_K,
        global_R_eff_K_per_W=args.global_R_eff_K_per_W,
    )

    rng = np.random.default_rng(int(args.seed))
    train_rows, _ = read_index(train_index)
    if args.max_calibration_samples is not None and len(train_rows) > args.max_calibration_samples:
        selected = rng.choice(len(train_rows), size=int(args.max_calibration_samples), replace=False)
        train_rows = [train_rows[int(index)] for index in sorted(selected)]
    val_rows, _ = read_index(val_index)
    test_rows, _ = read_index(test_index)

    print("Loading calibration samples")
    train_samples = load_samples(train_rows, train_index.parent, base_config)
    val_samples = load_samples(val_rows, val_index.parent, base_config)
    test_samples = load_samples(test_rows, test_index.parent, base_config)

    fitter = ConstrainedFitter(
        base_config=base_config,
        fit_k=args.fit_k,
        fit_g=args.fit_g,
        fit_source_scale=args.fit_source_scale,
        fit_ambient_offset=args.fit_ambient_offset,
        fit_global_R=args.fit_global_R,
        lambda_case_bias=float(args.lambda_case_bias),
        lambda_hotspot=float(args.lambda_hotspot),
        hotspot_top_frac=float(args.hotspot_top_frac),
    )

    print("Fitting globally shared screened-Poisson parameters")
    start = time.perf_counter()
    result = fitter.fit(train_samples, maxiter=int(args.maxiter))
    calibration_runtime_s = time.perf_counter() - start
    calibrated_config = fitter.config_from_vector(result.x)
    final_parameters = parameters_payload(calibrated_config)

    train_metrics = metrics_for_samples(train_samples, calibrated_config)
    val_metrics = metrics_for_samples(val_samples, calibrated_config)
    test_metrics = metrics_for_samples(test_samples, calibrated_config)

    calibration_payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": repo_relative(source_root),
        "candidate": "screened_poisson_calibrated",
        "base_candidate": args.candidate,
        "base_config": base_config.to_dict(),
        "final_config": calibrated_config.to_dict(),
        "final_parameters": final_parameters,
        "constraints": {
            "k_eff_W_per_K": [0.02, 5.0],
            "g_eff_W_per_mm2K": [5.0e-4, 5.0e-2],
            "alpha_source": [0.05, 20.0],
            "ambient_offset_K": [-100.0, 100.0],
            "global_R_eff_K_per_W": [0.0, 0.2],
        },
        "fit_flags": {
            "fit_k": bool(args.fit_k),
            "fit_g": bool(args.fit_g),
            "fit_source_scale": bool(args.fit_source_scale),
            "fit_ambient_offset": bool(args.fit_ambient_offset),
            "fit_global_R": bool(args.fit_global_R),
        },
        "objective": {
            "primary": "train field MSE in Kelvin squared",
            "lambda_case_bias": float(args.lambda_case_bias),
            "lambda_hotspot": float(args.lambda_hotspot),
            "hotspot_top_frac": float(args.hotspot_top_frac),
            "case_bias_term": "variance of per-case mean residual, residual = HotSpot - physics",
            "hotspot_term": "mean absolute error on hottest ground-truth cells if lambda_hotspot > 0",
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(result.success),
            "message": str(result.message),
            "objective_value": float(result.fun),
            "num_iterations": int(result.nit),
            "num_function_evaluations": int(result.nfev),
            "maxiter": int(args.maxiter),
        },
        "calibration_runtime_s": float(calibration_runtime_s),
        "calibration_sample_count": len(train_samples),
        "report_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "notes": [
            "Parameters are fitted once globally on the clean train split.",
            "Validation and test labels are not used by the optimizer.",
            "No case IDs, per-package parameters, or HotSpot-derived features are used at inference.",
        ],
    }
    write_json(out_dir / "calibration.json", calibration_payload)

    print("Generating calibrated candidate artifacts")
    generate_candidate(
        source_root=source_root,
        out_dir=out_dir,
        config=calibrated_config,
        max_samples_per_split=args.max_generate_samples_per_split,
    )
    merge_manifest(out_dir / "manifest.json", calibration_payload)
    write_json(out_dir / "calibration.json", calibration_payload)
    write_calibrated_readme(out_dir / "README.md", calibration_payload)

    print("Calibration complete")
    print(f"Final parameters: {json.dumps(final_parameters, sort_keys=True)}")
    print(f"Train MAE/RMSE: {train_metrics['mae_K']:.3f} / {train_metrics['rmse_K']:.3f} K")
    print(f"Val MAE/RMSE: {val_metrics['mae_K']:.3f} / {val_metrics['rmse_K']:.3f} K")
    print(f"Test MAE/RMSE: {test_metrics['mae_K']:.3f} / {test_metrics['rmse_K']:.3f} K")
    print(f"Output: {out_dir}")
    return 0


class ConstrainedFitter:
    def __init__(
        self,
        *,
        base_config: PhysicsCandidateConfig,
        fit_k: bool,
        fit_g: bool,
        fit_source_scale: bool,
        fit_ambient_offset: bool,
        fit_global_R: bool,
        lambda_case_bias: float,
        lambda_hotspot: float,
        hotspot_top_frac: float,
    ) -> None:
        self.base_config = base_config
        self.fit_k = fit_k
        self.fit_g = fit_g
        self.fit_source_scale = fit_source_scale
        self.fit_ambient_offset = fit_ambient_offset
        self.fit_global_R = fit_global_R
        self.lambda_case_bias = lambda_case_bias
        self.lambda_hotspot = lambda_hotspot
        self.hotspot_top_frac = hotspot_top_frac

    def fit(self, samples: list[CalibrationSample], *, maxiter: int):
        x0: list[float] = []
        bounds: list[tuple[float, float]] = []
        if self.fit_k:
            x0.append(np.log(float(self.base_config.k_spread_W_per_K)))
            bounds.append((np.log(0.02), np.log(5.0)))
        if self.fit_g:
            x0.append(np.log(float(self.base_config.g_sink_W_per_mm2K)))
            bounds.append((np.log(5.0e-4), np.log(5.0e-2)))
        if self.fit_source_scale:
            x0.append(np.log(float(self.base_config.source_scale)))
            bounds.append((np.log(0.05), np.log(20.0)))
        if self.fit_ambient_offset:
            x0.append(float(self.base_config.ambient_offset_K))
            bounds.append((-100.0, 100.0))
        if self.fit_global_R:
            x0.append(np.log(max(float(self.base_config.global_R_eff_K_per_W), 1.0e-6)))
            bounds.append((np.log(1.0e-6), np.log(0.2)))
        if not x0:
            raise SystemExit("no parameters selected; pass at least one --fit-* option")
        return minimize(
            lambda vector: self.objective(vector, samples),
            np.asarray(x0, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1.0e-8, "maxls": 20},
        )

    def config_from_vector(self, vector: np.ndarray) -> PhysicsCandidateConfig:
        cursor = 0
        k = float(self.base_config.k_spread_W_per_K)
        g = float(self.base_config.g_sink_W_per_mm2K)
        source_scale = float(self.base_config.source_scale)
        ambient_offset = float(self.base_config.ambient_offset_K)
        global_R = float(self.base_config.global_R_eff_K_per_W)
        if self.fit_k:
            k = float(np.exp(vector[cursor]))
            cursor += 1
        if self.fit_g:
            g = float(np.exp(vector[cursor]))
            cursor += 1
        if self.fit_source_scale:
            source_scale = float(np.exp(vector[cursor]))
            cursor += 1
        if self.fit_ambient_offset:
            ambient_offset = float(vector[cursor])
            cursor += 1
        if self.fit_global_R:
            global_R = float(np.exp(vector[cursor]))
            cursor += 1
        return PhysicsCandidateConfig(
            **{
                **self.base_config.to_dict(),
                "name": "screened_poisson_calibrated",
                "k_spread_W_per_K": k,
                "g_sink_W_per_mm2K": g,
                "source_scale": source_scale,
                "ambient_offset_K": ambient_offset,
                "global_R_eff_K_per_W": global_R,
            }
        )

    def objective(self, vector: np.ndarray, samples: list[CalibrationSample]) -> float:
        config = self.config_from_vector(vector)
        total_sq = 0.0
        total_count = 0
        residual_means_by_case: dict[str, list[float]] = defaultdict(list)
        hotspot_abs_sum = 0.0
        hotspot_count = 0
        for sample in samples:
            pred = predict_from_sample(sample, config)
            error = pred - sample.target_K
            total_sq += float(np.sum(error * error))
            total_count += int(error.size)
            residual_means_by_case[sample.case_id].append(float((sample.target_K - pred).mean()))
            if self.lambda_hotspot > 0.0:
                k = max(1, int(np.ceil(sample.target_K.size * self.hotspot_top_frac)))
                indices = np.argpartition(sample.target_K.reshape(-1), -k)[-k:]
                hotspot_abs_sum += float(np.abs(error.reshape(-1)[indices]).sum())
                hotspot_count += int(k)
        objective = total_sq / max(total_count, 1)
        if self.lambda_case_bias > 0.0:
            case_means = [float(np.mean(values)) for values in residual_means_by_case.values()]
            objective += self.lambda_case_bias * float(np.var(case_means))
        if self.lambda_hotspot > 0.0 and hotspot_count:
            objective += self.lambda_hotspot * hotspot_abs_sum / hotspot_count
        return float(objective)


def load_samples(rows: list[dict[str, str]], index_base: Path, config: PhysicsCandidateConfig) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    for row in rows:
        x = np.load(resolve_path(row["x_path"], index_base)).astype(np.float32, copy=False)
        y = np.load(resolve_path(row["y_path"], index_base)).astype(np.float64, copy=False)
        metadata = extract_package_grid_metadata(x, config, row_total_power_W=optional_float(row.get("total_power_W")))
        q = power_density_source(x, config).astype(np.float64, copy=False)
        samples.append(
            CalibrationSample(
                sample_uid=row["sample_uid"],
                case_id=row["case_id"],
                q_W_per_mm2=q,
                target_K=y,
                metadata=metadata,
            )
        )
    return samples


def predict_from_sample(sample: CalibrationSample, config: PhysicsCandidateConfig) -> np.ndarray:
    q = sample.q_W_per_mm2 * float(config.source_scale)
    rise = screened_poisson_rise(q, sample.metadata, config).astype(np.float64, copy=False)
    if config.global_R_eff_K_per_W:
        rise = rise + float(config.global_R_eff_K_per_W) * float(sample.metadata.total_power_W)
    return float(config.ambient_K) + float(config.ambient_offset_K) + rise


def metrics_for_samples(samples: list[CalibrationSample], config: PhysicsCandidateConfig) -> dict[str, Any]:
    abs_sum = 0.0
    sq_sum = 0.0
    signed_sum = 0.0
    count = 0
    max_abs = 0.0
    hotspot_temp_errors: list[float] = []
    hotspot_location_errors: list[float] = []
    hotspot_region_errors: dict[str, list[float]] = {"top1": [], "top5": [], "top10": []}
    per_case_errors: dict[str, list[float]] = defaultdict(list)
    per_case_signed: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        pred = predict_from_sample(sample, config)
        error = pred - sample.target_K
        abs_error = np.abs(error)
        abs_sum += float(abs_error.sum())
        sq_sum += float(np.sum(error * error))
        signed_sum += float(error.sum())
        count += int(error.size)
        max_abs = max(max_abs, float(abs_error.max()))
        pred_hotspot = np.unravel_index(int(np.argmax(pred)), pred.shape)
        target_hotspot = np.unravel_index(int(np.argmax(sample.target_K)), sample.target_K.shape)
        hotspot_temp_errors.append(float(pred[pred_hotspot] - sample.target_K[target_hotspot]))
        hotspot_location_errors.append(float(np.hypot(pred_hotspot[0] - target_hotspot[0], pred_hotspot[1] - target_hotspot[1])))
        for frac, name in ((0.01, "top1"), (0.05, "top5"), (0.10, "top10")):
            k = max(1, int(np.ceil(sample.target_K.size * frac)))
            indices = np.argpartition(sample.target_K.reshape(-1), -k)[-k:]
            hotspot_region_errors[name].append(float(abs_error.reshape(-1)[indices].mean()))
        per_case_errors[sample.case_id].append(float(abs_error.mean()))
        per_case_signed[sample.case_id].append(float(error.mean()))
    return {
        "num_samples": len(samples),
        "mae_K": abs_sum / max(count, 1),
        "rmse_K": float(np.sqrt(sq_sum / max(count, 1))),
        "mean_signed_error_K": signed_sum / max(count, 1),
        "max_abs_error_K": max_abs,
        "hotspot_temp_error_K": float(np.mean(hotspot_temp_errors)) if hotspot_temp_errors else None,
        "hotspot_location_error_cells": float(np.mean(hotspot_location_errors)) if hotspot_location_errors else None,
        "hotspot_top_1pct_mae_K": float(np.mean(hotspot_region_errors["top1"])) if hotspot_region_errors["top1"] else None,
        "hotspot_top_5pct_mae_K": float(np.mean(hotspot_region_errors["top5"])) if hotspot_region_errors["top5"] else None,
        "hotspot_top_10pct_mae_K": float(np.mean(hotspot_region_errors["top10"])) if hotspot_region_errors["top10"] else None,
        "per_case_mae_K": {case: float(np.mean(values)) for case, values in sorted(per_case_errors.items())},
        "per_case_signed_error_K": {case: float(np.mean(values)) for case, values in sorted(per_case_signed.items())},
    }


def parameters_payload(config: PhysicsCandidateConfig) -> dict[str, float]:
    return {
        "k_eff_W_per_K": float(config.k_spread_W_per_K),
        "g_eff_W_per_mm2K": float(config.g_sink_W_per_mm2K),
        "alpha_source": float(config.source_scale),
        "ambient_offset_K": float(config.ambient_offset_K),
        "global_R_eff_K_per_W": float(config.global_R_eff_K_per_W),
    }


def merge_manifest(manifest_path: Path, calibration_payload: dict[str, Any]) -> None:
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_file"] = "calibration.json"
    manifest["calibration"] = {
        "final_parameters": calibration_payload["final_parameters"],
        "optimizer": calibration_payload["optimizer"],
        "report_metrics": calibration_payload["report_metrics"],
    }
    write_json(manifest_path, manifest)


def write_calibrated_readme(path: Path, payload: dict[str, Any]) -> None:
    params = payload["final_parameters"]
    text = f"""# ChipTherm Physics Candidate: screened_poisson_calibrated

This directory contains globally calibrated screened-Poisson prediction and
residual tensors. X tensors and HotSpot Y tensors are reused from the clean
source dataset; only analytical prediction and residual tensors are generated.

The calibrated model is:

`T = ambient + ambient_offset + delta_T + R_eff * total_power`

where:

`(-k_eff * Laplacian + g_eff) delta_T = alpha_source * q`

and `q` is the raster source power density in W/mm^2. Parameters are fitted
once on the clean train split only with bounded global constraints. No case IDs
or per-package fitting are used.

## Final Parameters

- `k_eff_W_per_K`: `{params['k_eff_W_per_K']:.12g}`
- `g_eff_W_per_mm2K`: `{params['g_eff_W_per_mm2K']:.12g}`
- `alpha_source`: `{params['alpha_source']:.12g}`
- `ambient_offset_K`: `{params['ambient_offset_K']:.12g}`
- `global_R_eff_K_per_W`: `{params['global_R_eff_K_per_W']:.12g}`

## Files

- `train_index.csv`, `val_index.csv`, `test_index.csv`
- `combined_encoded_index.csv`, `combined_encoded_index.jsonl`
- `predictions/`, `residuals/`
- `manifest.json`
- `calibration.json`
"""
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
