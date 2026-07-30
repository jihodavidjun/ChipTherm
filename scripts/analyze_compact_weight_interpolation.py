#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.compact_weight_interpolation import (  # noqa: E402
    ENDPOINT_FRACTION_TOLERANCE,
    ENDPOINT_METRIC_TOLERANCE_K,
    FROZEN_ALPHAS,
    KNOWN_MAE_LIMIT_K,
    KNOWN_MAE_TIE_K,
    MAX_FAMILY_REGRESSION_K,
    MAX_FRACTION_WORSE_ABSOLUTE_INCREASE,
    MAX_HOTSPOT_ABS_ERROR_INCREASE_K,
    VALIDATION_MAE_LIMIT_K,
    alpha_run_id,
    now_utc,
    sha256_file,
    stable_hash,
)


VALIDATION_PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
)
PRIMARY_PROTOCOL = "primary_test_families"
INTERIOR_ALPHAS = (0.25, 0.50, 0.75)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def final_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metrics.get("cnn_final_temperature") or metrics.get(
        "final_temperature"
    )
    if not isinstance(value, Mapping):
        raise ValueError("metrics.json is missing final-temperature metrics")
    return value


def aggregate_protocol(
    protocol_dir: Path,
    *,
    model: str,
    alpha: float | None,
    protocol: str,
    require_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path = protocol_dir / "metrics.json"
    samples_path = protocol_dir / "metrics_by_sample.csv"
    if not metrics_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(
            f"missing evaluation artifacts under {protocol_dir}"
        )
    metrics = read_json(metrics_path)
    samples = read_csv(samples_path)
    if not samples:
        raise ValueError(f"empty sample metrics: {samples_path}")
    numeric_fields = (
        "mae_K",
        "rmse_K",
        "mean_signed_error_K",
        "physics_baseline_mae_K",
        "centered_field_mae_K",
        "mean_head_abs_error_K",
        "hotspot_top1pct_mae_K",
        "boundary_region_mae_K",
        "hotspot_location_error_cells",
    )
    for row in samples:
        for field in numeric_fields:
            if field in row and row[field] != "":
                if not math.isfinite(float(row[field])):
                    raise ValueError(
                        f"non-finite {field} for sample {row.get('sample_uid')}"
                    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        grouped[str(row.get("family_uid") or row.get("case_id"))].append(row)
    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        family_rows.append(
            {
                "model": model,
                "alpha": "" if alpha is None else alpha,
                "protocol": protocol,
                "family_uid": family,
                "sample_count": len(rows),
                "mae_K": mean(rows, "mae_K"),
                "rmse_K": root_mean_square(rows, "rmse_K"),
                "centered_field_mae_K": mean(rows, "centered_field_mae_K"),
                "mean_correction_mae_K": mean(rows, "mean_head_abs_error_K"),
                "hotspot_top1pct_mae_K": mean(rows, "hotspot_top1pct_mae_K"),
                "boundary_mae_K": mean(rows, "boundary_region_mae_K"),
            }
        )
    final = final_metrics(metrics)
    prediction_audit = audit_predictions(protocol_dir, len(samples))
    if require_predictions and prediction_audit["status"] != "finite":
        raise ValueError(
            f"prediction finiteness audit failed for {protocol_dir}: "
            f"{prediction_audit}"
        )
    parameter_count = int(metrics.get("model", {}).get("parameter_count", 0))
    if alpha is not None and parameter_count != 2_188_803:
        raise ValueError(
            f"{model} parameter count {parameter_count} does not match 2,188,803"
        )
    row = {
        "model": model,
        "alpha": "" if alpha is None else alpha,
        "protocol": protocol,
        "sample_count": len(samples),
        "micro_mae_K": float(final["mae_K"]),
        "micro_rmse_K": float(final["rmse_K"]),
        "macro_family_mae_K": float(
            np.mean([item["mae_K"] for item in family_rows])
        ),
        "centered_field_mae_K": mean(samples, "centered_field_mae_K"),
        "mean_correction_mae_K": mean(samples, "mean_head_abs_error_K"),
        "hotspot_temperature_abs_error_K": abs(
            float(final["hotspot_temp_error_K"])
        ),
        "hotspot_top1pct_mae_K": mean(samples, "hotspot_top1pct_mae_K"),
        "hotspot_location_error_cells": float(
            final["hotspot_location_error_cells"]
        ),
        "boundary_mae_K": mean(samples, "boundary_region_mae_K"),
        "fraction_worse_than_source": float(
            metrics["worse_than_physics_baseline_fraction"]
        ),
        "mean_signed_error_K": float(final["mean_signed_error_K"]),
        "runtime_per_sample_s": float(
            metrics["inference_runtime_per_sample_s"]
        ),
        "parameter_count": parameter_count,
        "outputs_finite": prediction_audit["status"] == "finite",
        "prediction_file_count": prediction_audit["file_count"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "sample_metrics_sha256": sha256_file(samples_path),
    }
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite aggregate metric {key}: {protocol_dir}")
    return row, family_rows


def audit_predictions(protocol_dir: Path, expected_samples: int) -> dict[str, Any]:
    prediction_root = protocol_dir / "predictions"
    if not prediction_root.is_dir():
        return {
            "status": "missing",
            "file_count": 0,
            "expected_samples": expected_samples,
        }
    files = sorted(prediction_root.rglob("*_tpred.npy"))
    if len(files) != expected_samples:
        return {
            "status": "count_mismatch",
            "file_count": len(files),
            "expected_samples": expected_samples,
        }
    for path in files:
        array = np.load(path, mmap_mode="r")
        if array.shape != (64, 64) or not np.isfinite(array).all():
            return {
                "status": "nonfinite_or_shape_error",
                "file_count": len(files),
                "expected_samples": expected_samples,
                "offending_path": str(path),
                "shape": list(array.shape),
            }
    return {
        "status": "finite",
        "file_count": len(files),
        "expected_samples": expected_samples,
    }


def mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    if not values:
        raise ValueError(f"sample metrics are missing {field}")
    return float(np.mean(values))


def root_mean_square(
    rows: Sequence[Mapping[str, str]], field: str
) -> float:
    values = np.asarray(
        [float(row[field]) for row in rows if row.get(field, "") != ""],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError(f"sample metrics are missing {field}")
    return float(np.sqrt(np.mean(values * values)))


def protocol_path(
    experiment_root: Path,
    alpha: float,
    protocol: str,
) -> Path:
    stage = (
        "evaluation_primary_test"
        if protocol == PRIMARY_PROTOCOL
        else "evaluation_validation"
    )
    return experiment_root / alpha_run_id(alpha) / stage / protocol


def reference_path(root: Path, protocol: str) -> Path:
    return root / protocol


def collect_results(
    *,
    experiment_root: Path,
    canonical_eval_root: Path,
    cosine_eval_root: Path,
    include_primary_test: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocols = list(VALIDATION_PROTOCOLS)
    if include_primary_test:
        protocols.append(PRIMARY_PROTOCOL)
    metric_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for alpha in FROZEN_ALPHAS:
        for protocol in protocols:
            path = protocol_path(experiment_root, alpha, protocol)
            if not (path / "metrics.json").is_file():
                missing.append(
                    {
                        "model": alpha_run_id(alpha),
                        "alpha": alpha,
                        "protocol": protocol,
                        "path": str(path),
                    }
                )
                continue
            row, families = aggregate_protocol(
                path,
                model=alpha_run_id(alpha),
                alpha=alpha,
                protocol=protocol,
                require_predictions=True,
            )
            metric_rows.append(row)
            family_rows.extend(families)
    for model, root in (
        ("canonical_reference", canonical_eval_root),
        ("cosine_ema_reference", cosine_eval_root),
    ):
        for protocol in VALIDATION_PROTOCOLS:
            path = reference_path(root, protocol)
            if not (path / "metrics.json").is_file():
                missing.append(
                    {
                        "model": model,
                        "alpha": "",
                        "protocol": protocol,
                        "path": str(path),
                    }
                )
                continue
            row, families = aggregate_protocol(
                path,
                model=model,
                alpha=None,
                protocol=protocol,
                require_predictions=False,
            )
            metric_rows.append(row)
            family_rows.extend(families)
    return metric_rows, family_rows, missing


def endpoint_reproduction(
    rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    state_report: Mapping[str, Any],
) -> dict[str, Any]:
    lookup = {(row["model"], row["protocol"]): row for row in rows}
    family_lookup = {
        (row["model"], row["protocol"], row["family_uid"]): row
        for row in family_rows
    }
    pairs = (
        ("alpha000", alpha_run_id(0.0), "canonical_reference"),
        ("alpha100", alpha_run_id(1.0), "cosine_ema_reference"),
    )
    metric_names = (
        "micro_mae_K",
        "micro_rmse_K",
        "centered_field_mae_K",
        "mean_correction_mae_K",
        "hotspot_temperature_abs_error_K",
        "hotspot_top1pct_mae_K",
        "hotspot_location_error_cells",
        "boundary_mae_K",
        "mean_signed_error_K",
    )
    result: dict[str, Any] = {
        "schema_version": "compact_weight_interpolation_endpoints/2",
        "state_checks": {
            name: state_report.get(name, {}) for name, _, _ in pairs
        },
        "metric_tolerance_K": ENDPOINT_METRIC_TOLERANCE_K,
        "fraction_tolerance": ENDPOINT_FRACTION_TOLERANCE,
        "metric_checks": {},
        "all_passed": True,
    }
    for label, candidate, reference in pairs:
        checks = []
        for protocol in VALIDATION_PROTOCOLS:
            candidate_row = lookup.get((candidate, protocol))
            reference_row = lookup.get((reference, protocol))
            if candidate_row is None or reference_row is None:
                result["all_passed"] = False
                checks.append(
                    {"protocol": protocol, "status": "pending"}
                )
                continue
            differences = {
                name: abs(float(candidate_row[name]) - float(reference_row[name]))
                for name in metric_names
            }
            fraction_difference = abs(
                float(candidate_row["fraction_worse_than_source"])
                - float(reference_row["fraction_worse_than_source"])
            )
            families = sorted(
                {
                    key[2]
                    for key in family_lookup
                    if key[0] == reference and key[1] == protocol
                }
            )
            family_difference = max(
                (
                    abs(
                        float(
                            family_lookup[(candidate, protocol, family)]["mae_K"]
                        )
                        - float(
                            family_lookup[(reference, protocol, family)]["mae_K"]
                        )
                    )
                    for family in families
                    if (candidate, protocol, family) in family_lookup
                ),
                default=float("inf"),
            )
            passed = (
                max(differences.values(), default=0.0)
                <= ENDPOINT_METRIC_TOLERANCE_K
                and family_difference <= ENDPOINT_METRIC_TOLERANCE_K
                and fraction_difference <= ENDPOINT_FRACTION_TOLERANCE
                and bool(candidate_row["outputs_finite"])
            )
            result["all_passed"] = result["all_passed"] and passed
            checks.append(
                {
                    "protocol": protocol,
                    "status": "passed" if passed else "failed",
                    "metric_abs_differences": differences,
                    "max_family_mae_abs_difference_K": family_difference,
                    "fraction_worse_abs_difference": fraction_difference,
                }
            )
        state = state_report.get(label, {})
        if not state.get("state_exact", False):
            result["all_passed"] = False
        result["metric_checks"][label] = checks
    return result


def select_candidate(
    rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    *,
    endpoint_checks_passed: bool,
) -> dict[str, Any]:
    lookup = {(row["model"], row["protocol"]): row for row in rows}
    family_lookup = {
        (row["model"], row["protocol"], row["family_uid"]): row
        for row in family_rows
    }
    canonical_validation = lookup.get(
        ("canonical_reference", "primary_validation_families")
    )
    if canonical_validation is None:
        return {"status": "pending", "reason": "canonical validation missing"}
    canonical_families = {
        row["family_uid"]: row
        for row in family_rows
        if row["model"] == "canonical_reference"
        and row["protocol"] == "primary_validation_families"
    }
    candidates: list[dict[str, Any]] = []
    for alpha in INTERIOR_ALPHAS:
        model = alpha_run_id(alpha)
        known = lookup.get((model, "known_family_sample_test"))
        validation = lookup.get((model, "primary_validation_families"))
        if known is None or validation is None:
            candidates.append(
                {
                    "alpha": alpha,
                    "run_id": model,
                    "status": "pending",
                }
            )
            continue
        family_deltas = {
            family: float(
                family_lookup[
                    (model, "primary_validation_families", family)
                ]["mae_K"]
            )
            - float(reference["mae_K"])
            for family, reference in canonical_families.items()
            if (model, "primary_validation_families", family) in family_lookup
        }
        family_coverage_complete = set(family_deltas) == set(canonical_families)
        max_family_regression = (
            max(family_deltas.values())
            if family_coverage_complete
            else float("inf")
        )
        checks = {
            "endpoint_checks_passed": endpoint_checks_passed,
            "known_mae": float(known["micro_mae_K"]) <= KNOWN_MAE_LIMIT_K,
            "validation_mae": (
                float(validation["micro_mae_K"]) <= VALIDATION_MAE_LIMIT_K
            ),
            "per_family_regression": (
                family_coverage_complete
                and max_family_regression <= MAX_FAMILY_REGRESSION_K
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
                "alpha": alpha,
                "run_id": model,
                "status": "admissible" if all(checks.values()) else "rejected",
                "known_family_mae_K": float(known["micro_mae_K"]),
                "heldout_validation_mae_K": float(
                    validation["micro_mae_K"]
                ),
                "max_family_regression_K": max_family_regression,
                "fraction_worse_than_source": float(
                    validation["fraction_worse_than_source"]
                ),
                "hotspot_temperature_abs_error_K": float(
                    validation["hotspot_temperature_abs_error_K"]
                ),
                "checks": checks,
                "family_deltas_K": family_deltas,
            }
        )
    pending = [item for item in candidates if item["status"] == "pending"]
    if pending:
        return {
            "status": "pending",
            "candidates": candidates,
            "thresholds": selection_thresholds(),
        }
    admissible = [
        item for item in candidates if item["status"] == "admissible"
    ]
    if not admissible:
        return {
            "status": "no_candidate",
            "reason": "no interior alpha satisfies every frozen safeguard",
            "candidates": candidates,
            "thresholds": selection_thresholds(),
        }
    best_known = min(item["known_family_mae_K"] for item in admissible)
    tie_group = [
        item
        for item in admissible
        if item["known_family_mae_K"] <= best_known + KNOWN_MAE_TIE_K
    ]
    selected = min(
        tie_group,
        key=lambda item: (
            item["heldout_validation_mae_K"],
            item["known_family_mae_K"],
            item["alpha"],
        ),
    )
    return {
        "status": "selected",
        "selected_alpha": selected["alpha"],
        "selected_run_id": selected["run_id"],
        "selection_order": (
            "lowest known-family MAE; candidates within 0.002 K form a tie "
            "group resolved by lower held-out-validation MAE"
        ),
        "selected": selected,
        "candidates": candidates,
        "thresholds": selection_thresholds(),
    }


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


def validation_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash(
        sorted(
            (
                {
                    "model": row["model"],
                    "protocol": row["protocol"],
                    "metrics_sha256": row["metrics_sha256"],
                    "sample_metrics_sha256": row["sample_metrics_sha256"],
                }
                for row in rows
                if row["protocol"] in VALIDATION_PROTOCOLS
            ),
            key=lambda item: (item["model"], item["protocol"]),
        )
    )


def analyze(
    *,
    experiment_root: Path,
    canonical_eval_root: Path,
    cosine_eval_root: Path,
    out_dir: Path,
    freeze_validation: bool,
    include_primary_test: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_endpoint_path = experiment_root / "endpoint_reproduction_report.json"
    if not state_endpoint_path.is_file():
        raise FileNotFoundError(
            "builder endpoint report is missing; build checkpoints first"
        )
    state_endpoint = read_json(state_endpoint_path)
    state_checks = state_endpoint.get("state_checks", state_endpoint)
    frozen_path = out_dir / "validation_decision_gate.json"
    frozen = read_json(frozen_path) if frozen_path.is_file() else None
    if include_primary_test and (
        frozen is None
        or frozen.get("status") != "frozen"
        or frozen.get("selection", {}).get("status") != "selected"
    ):
        raise ValueError(
            "--include-primary-test requires a frozen, selected interior candidate"
        )
    rows, families, missing = collect_results(
        experiment_root=experiment_root,
        canonical_eval_root=canonical_eval_root,
        cosine_eval_root=cosine_eval_root,
        include_primary_test=include_primary_test,
    )
    fingerprint = validation_fingerprint(rows)
    if frozen is not None and frozen.get("status") == "frozen":
        if frozen.get("validation_fingerprint") != fingerprint:
            raise ValueError(
                "validation artifacts changed after freezing; primary-test "
                "inclusion is forbidden"
            )
    endpoint = endpoint_reproduction(rows, families, state_checks)
    write_json(out_dir / "endpoint_reproduction_report.json", endpoint)
    selection = select_candidate(
        rows,
        families,
        endpoint_checks_passed=bool(endpoint["all_passed"]),
    )
    validation_complete = not any(
        item["protocol"] in VALIDATION_PROTOCOLS for item in missing
    )
    gate: dict[str, Any] = {
        "schema_version": "compact_weight_interpolation_validation_gate/1",
        "status": "ready_to_freeze" if validation_complete else "pending",
        "validation_fingerprint": fingerprint,
        "primary_test_used_for_selection": False,
        "selection": selection,
        "missing_results": [
            item for item in missing if item["protocol"] in VALIDATION_PROTOCOLS
        ],
        "thresholds": selection_thresholds(),
    }
    if frozen is not None and frozen.get("status") == "frozen":
        gate = frozen
    elif freeze_validation:
        if not validation_complete:
            raise ValueError("cannot freeze while validation results are missing")
        if not endpoint["all_passed"]:
            raise ValueError("cannot freeze because endpoint reproduction failed")
        gate["status"] = "frozen"
        gate["frozen_at_utc"] = now_utc()
    write_json(frozen_path, gate)
    if gate["status"] == "frozen":
        write_json(out_dir / "selected_candidate.json", gate["selection"])
    else:
        write_json(out_dir / "selected_candidate.json", selection)
    primary_gate = {
        "schema_version": "compact_weight_interpolation_primary_gate/1",
        "status": "closed",
        "validation_gate_status": gate["status"],
        "validation_fingerprint": gate.get("validation_fingerprint"),
        "selected_run_id": gate.get("selection", {}).get("selected_run_id"),
        "primary_test_used_for_selection": False,
    }
    if include_primary_test:
        selected_run = gate["selection"]["selected_run_id"]
        primary = next(
            (
                row
                for row in rows
                if row["model"] == selected_run
                and row["protocol"] == PRIMARY_PROTOCOL
            ),
            None,
        )
        if primary is None:
            raise ValueError("selected candidate primary-test metrics are missing")
        primary_gate.update(
            status="complete",
            primary_metrics_path=primary["metrics_path"],
            primary_metrics_sha256=primary["metrics_sha256"],
        )
    write_json(out_dir / "primary_test_gate.json", primary_gate)
    write_csv(out_dir / "compact_weight_interpolation_metrics.csv", rows)
    write_csv(out_dir / "compact_weight_interpolation_per_family.csv", families)
    summary = {
        "schema_version": "compact_weight_interpolation_summary/1",
        "created_at_utc": now_utc(),
        "status": gate["status"],
        "alphas": list(FROZEN_ALPHAS),
        "missing_results": missing,
        "endpoint_checks_passed": endpoint["all_passed"],
        "selection": gate.get("selection", selection),
        "primary_test_included": include_primary_test,
        "primary_test_used_for_selection": False,
    }
    write_json(
        out_dir / "compact_weight_interpolation_summary.json",
        summary,
    )
    write_report(
        out_dir / "compact_weight_interpolation_report.md",
        summary,
        rows,
    )
    write_plots(out_dir, rows, families)
    return summary


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Compact CNN Weight Interpolation",
        "",
        "This is a post-training linear weight-space diagnostic. Primary-test "
        "results are excluded from alpha selection.",
        "",
        "| Alpha | Protocol | MAE (K) | RMSE (K) | Macro-family MAE (K) | Centered MAE (K) | Mean MAE (K) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["alpha"] == "":
            continue
        lines.append(
            f"| {float(row['alpha']):.2f} | {row['protocol']} | "
            f"{float(row['micro_mae_K']):.5f} | "
            f"{float(row['micro_rmse_K']):.5f} | "
            f"{float(row['macro_family_mae_K']):.5f} | "
            f"{float(row['centered_field_mae_K']):.5f} | "
            f"{float(row['mean_correction_mae_K']):.5f} |"
        )
    lines.extend(
        [
            "",
            f"Validation status: **{summary['status']}**.",
            "",
            "```json",
            json.dumps(summary["selection"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    soup = [row for row in rows if row["alpha"] != ""]
    line_plot(
        soup,
        "known_family_sample_test",
        "micro_mae_K",
        out_dir / "alpha_vs_known_family_mae.png",
        "Known-family MAE (K)",
    )
    line_plot(
        soup,
        "primary_validation_families",
        "micro_mae_K",
        out_dir / "alpha_vs_validation_mae.png",
        "Held-out validation MAE (K)",
    )
    lookup = {(row["model"], row["protocol"]): row for row in soup}
    names = sorted(
        {
            row["model"]
            for row in soup
            if (row["model"], "known_family_sample_test") in lookup
            and (row["model"], "primary_validation_families") in lookup
        }
    )
    if names:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for name in names:
            known = lookup[(name, "known_family_sample_test")]
            validation = lookup[(name, "primary_validation_families")]
            ax.scatter(
                known["micro_mae_K"],
                validation["micro_mae_K"],
                label=f"alpha={float(known['alpha']):.2f}",
            )
        ax.set_xlabel("Known-family MAE (K)")
        ax.set_ylabel("Held-out validation MAE (K)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "known_vs_validation_tradeoff.png", dpi=180)
        plt.close(fig)
    component_plot(soup, out_dir / "component_error_vs_alpha.png")
    family_delta_plot(
        family_rows,
        out_dir / "per_family_validation_delta.png",
    )


def line_plot(
    rows: Sequence[Mapping[str, Any]],
    protocol: str,
    metric: str,
    path: Path,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    selected = sorted(
        (row for row in rows if row["protocol"] == protocol),
        key=lambda row: float(row["alpha"]),
    )
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        [float(row["alpha"]) for row in selected],
        [float(row[metric]) for row in selected],
        marker="o",
    )
    ax.set_xlabel("Interpolation alpha")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def component_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = sorted(
        (
            row
            for row in rows
            if row["protocol"] == "primary_validation_families"
        ),
        key=lambda row: float(row["alpha"]),
    )
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    alpha = [float(row["alpha"]) for row in selected]
    ax.plot(
        alpha,
        [row["centered_field_mae_K"] for row in selected],
        marker="o",
        label="centered",
    )
    ax.plot(
        alpha,
        [row["mean_correction_mae_K"] for row in selected],
        marker="o",
        label="mean",
    )
    ax.plot(
        alpha,
        [row["hotspot_top1pct_mae_K"] for row in selected],
        marker="o",
        label="hotspot top 1%",
    )
    ax.set_xlabel("Interpolation alpha")
    ax.set_ylabel("Error (K)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def family_delta_plot(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    canonical = {
        row["family_uid"]: float(row["mae_K"])
        for row in rows
        if row["model"] == "canonical_reference"
        and row["protocol"] == "primary_validation_families"
    }
    soup = [
        row
        for row in rows
        if row["alpha"] != ""
        and row["protocol"] == "primary_validation_families"
    ]
    if not canonical or not soup:
        return
    families = sorted(canonical)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for alpha in INTERIOR_ALPHAS:
        by_family = {
            row["family_uid"]: float(row["mae_K"])
            for row in soup
            if float(row["alpha"]) == alpha
        }
        if by_family:
            ax.plot(
                families,
                [by_family[family] - canonical[family] for family in families],
                marker="o",
                label=f"alpha={alpha:.2f}",
            )
    ax.axhline(MAX_FAMILY_REGRESSION_K, color="red", linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("MAE delta vs canonical (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze and freeze compact-CNN weight interpolation."
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--canonical-eval-root", required=True, type=Path)
    parser.add_argument("--cosine-eval-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--freeze-validation", action="store_true")
    parser.add_argument("--include-primary-test", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        experiment_root=args.experiment_root.expanduser().resolve(),
        canonical_eval_root=args.canonical_eval_root.expanduser().resolve(),
        cosine_eval_root=args.cosine_eval_root.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        freeze_validation=args.freeze_validation,
        include_primary_test=args.include_primary_test,
    )
    print(f"Status: {summary['status']}")
    print(f"Endpoint checks: {summary['endpoint_checks_passed']}")
    print(f"Selection: {summary['selection']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
