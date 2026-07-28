#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from chiptherm.benchmark_v2_training import EXPECTED_PRIMARY_SPLIT  # noqa: E402


SOURCE_VERSION = "source_superposition_final_train40_source_v1"
EXPECTED_PROTOCOLS = {
    "known_family_sample_test": {
        "families": tuple(EXPECTED_PRIMARY_SPLIT["train"]),
        "samples_per_family": 20,
    },
    "primary_validation_families": {
        "families": tuple(EXPECTED_PRIMARY_SPLIT["val"]),
        "samples_per_family": 200,
    },
    "primary_test_families": {
        "families": tuple(EXPECTED_PRIMARY_SPLIT["test"]),
        "samples_per_family": 200,
    },
}
MODEL_LABELS = ("cnn", "fno", "ufno", "sau_fno")
PROTOCOL_DIRECTORY_PRIORITY = {
    "known_family_sample_test": (
        ("evaluation_selection/known_family_sample_test",),
        ("evaluation/known_family_sample_test",),
    ),
    "primary_validation_families": (
        ("evaluation_selection/primary_validation_families",),
        ("evaluation/primary_validation_families",),
    ),
    "primary_test_families": (
        ("evaluation_primary_test/primary_test_families",),
        ("evaluation/primary_test_families",),
        ("evaluation_selection/primary_test_families",),
    ),
}
NON_DESCRIPTOR_COLUMNS = {"family_uid", "split", "primary_category", "placement_style"}
METRIC_COLUMNS = {
    "mae_K": "mae_K",
    "rmse_K": "rmse_K",
    "mean_signed_error_K": "mean_signed_error_K",
    "mean_rise_mae_K": "mean_head_abs_error_K",
    "centered_field_mae_K": "centered_field_mae_K",
    "centered_field_rmse_K": "centered_field_rmse_K",
    "boundary_region_mae_K": "boundary_region_mae_K",
    "non_boundary_region_mae_K": "non_boundary_region_mae_K",
    "occupied_region_mae_K": "occupied_region_mae_K",
    "unoccupied_region_mae_K": "unoccupied_region_mae_K",
    "hotspot_top1pct_mae_K": "hotspot_top1pct_mae_K",
    "hotspot_temperature_error_K": "peak_temperature_abs_error_K",
    "hotspot_location_error_cells": "hotspot_location_error_cells",
    "max_abs_error_K": "max_abs_error_K",
    "source_baseline_mae_K": "physics_baseline_mae_K",
}
REQUIRED_SAMPLE_METRICS = {
    "mae_K",
    "rmse_K",
    "mean_signed_error_K",
    "mean_head_abs_error_K",
    "centered_field_mae_K",
    "centered_field_rmse_K",
    "boundary_region_mae_K",
    "occupied_region_mae_K",
    "hotspot_top1pct_mae_K",
    "peak_temperature_abs_error_K",
    "hotspot_location_error_cells",
    "max_abs_error_K",
    "physics_baseline_mae_K",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close the Benchmark v2 four-model zero-shot study from saved artifacts only."
    )
    parser.add_argument(
        "--cnn-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1",
    )
    parser.add_argument(
        "--fno-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1",
    )
    parser.add_argument(
        "--ufno-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1",
    )
    parser.add_argument(
        "--sau-fno-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/sau_fno/residual_sau_fno_decomposed_train40_seed1",
    )
    parser.add_argument(
        "--family-ood-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1/family_ood_analysis",
    )
    parser.add_argument(
        "--residual-decomposition-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1/residual_decomposition/"
        "per_sample_decomposition.csv",
    )
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument("--source-version", default=SOURCE_VERSION)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/benchmark_v2_50family/zero_shot_diagnostics",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--aggregate-tolerance-K", type=float, default=1.0e-5)
    args = parser.parse_args()

    roots = {
        "cnn": args.cnn_root.expanduser().resolve(),
        "fno": args.fno_root.expanduser().resolve(),
        "ufno": args.ufno_root.expanduser().resolve(),
        "sau_fno": args.sau_fno_root.expanduser().resolve(),
    }
    result = analyze_zero_shot(
        model_roots=roots,
        family_ood_dir=args.family_ood_dir.expanduser().resolve(),
        residual_decomposition_csv=args.residual_decomposition_csv.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve() if args.data_root else None,
        source_version=args.source_version,
        out_dir=args.out_dir.expanduser().resolve(),
        top_k=args.top_k,
        aggregate_tolerance_K=args.aggregate_tolerance_K,
    )
    print("Benchmark v2 zero-shot diagnostic complete")
    print(f"Models: {', '.join(result['models'])}")
    print(f"Held-out families: {len(result['heldout_families'])}")
    print(f"Output: {args.out_dir}")
    if result["blocked_outputs"]:
        print("Blocked outputs:")
        for item in result["blocked_outputs"]:
            print(f"  - {item}")
    return 0


def analyze_zero_shot(
    *,
    model_roots: Mapping[str, Path],
    family_ood_dir: Path,
    residual_decomposition_csv: Path,
    data_root: Path | None,
    source_version: str,
    out_dir: Path,
    top_k: int,
    aggregate_tolerance_K: float,
) -> dict[str, Any]:
    if tuple(model_roots) != MODEL_LABELS:
        raise ValueError(f"model roots must be ordered as {MODEL_LABELS}; got {tuple(model_roots)}")
    if source_version != SOURCE_VERSION:
        raise ValueError(f"expected source version {SOURCE_VERSION}, got {source_version}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts, sample_rows, metrics_payloads = load_and_validate_model_artifacts(
        model_roots=model_roots,
        source_version=source_version,
        aggregate_tolerance_K=aggregate_tolerance_K,
    )
    aligned_samples = validate_cross_model_alignment(sample_rows)
    family_rows = build_per_family_metrics(sample_rows)
    aggregate_rows = build_aggregate_model_comparison(sample_rows, metrics_payloads)
    ranking_rows = rank_models_by_family(family_rows)
    source_improvement_rows = build_source_improvement_rows(family_rows)

    descriptor_path = family_ood_dir / "family_descriptors.csv"
    prior_ood_summary_path = family_ood_dir / "summary.json"
    descriptor_rows = read_csv_required(descriptor_path)
    prior_ood_summary = read_json_required(prior_ood_summary_path)
    descriptor_names = validate_descriptor_table(descriptor_rows, prior_ood_summary)
    descriptor_space = fit_descriptor_space(
        descriptor_rows,
        descriptor_names,
        train_family_uids=EXPECTED_PRIMARY_SPLIT["train"],
    )
    distance_rows, nearest_rows = build_distance_outputs(
        descriptor_rows,
        descriptor_names,
        descriptor_space,
        top_k=top_k,
    )
    family_error_lookup = family_error_labels(family_rows)
    tier_rows, tier_thresholds = assign_ood_tiers(
        descriptor_rows=descriptor_rows,
        descriptor_names=descriptor_names,
        descriptor_space=descriptor_space,
        family_errors=family_error_lookup,
    )
    tier_lookup = {row["family_uid"]: row for row in tier_rows}
    for row in family_rows:
        tier = tier_lookup.get(str(row["family_uid"]))
        row["ood_primary_tier"] = tier["primary_tier"] if tier else ""
        row["ood_secondary_flags"] = tier["secondary_flags"] if tier else ""
    descriptor_output_rows = attach_ood_columns(descriptor_rows, tier_rows)
    correlation_rows = build_error_descriptor_correlations(
        tier_rows=tier_rows,
        family_errors=family_error_lookup,
    )

    write_csv(out_dir / "aggregate_model_comparison.csv", aggregate_rows)
    write_csv(out_dir / "per_family_metrics.csv", family_rows)
    write_csv(out_dir / "per_family_model_ranking.csv", ranking_rows)
    write_csv(out_dir / "family_descriptor_table.csv", descriptor_output_rows)
    write_csv(out_dir / "family_distance_matrix.csv", distance_rows)
    write_csv(out_dir / "heldout_family_nearest_train.csv", nearest_rows)
    write_csv(out_dir / "ood_tier_assignments.csv", tier_rows)
    write_csv(out_dir / "error_descriptor_correlations.csv", correlation_rows)
    write_csv(out_dir / "source_improvement_by_family.csv", source_improvement_rows)

    blocked_outputs: list[str] = []
    heatmap_status = maybe_plot_representative_heatmaps(
        out_dir=out_dir,
        data_root=data_root,
        source_version=source_version,
        model_roots=model_roots,
        sample_rows=sample_rows,
    )
    if heatmap_status:
        blocked_outputs.append(heatmap_status)
    plot_status = write_summary_plots(
        out_dir=out_dir,
        aggregate_rows=aggregate_rows,
        family_rows=family_rows,
        source_improvement_rows=source_improvement_rows,
        tier_rows=tier_rows,
        descriptor_rows=descriptor_rows,
        descriptor_names=descriptor_names,
        descriptor_space=descriptor_space,
    )
    blocked_outputs.extend(plot_status)

    decomposition_status = audit_residual_decomposition(residual_decomposition_csv)
    findings = derive_findings(
        aggregate_rows=aggregate_rows,
        family_rows=family_rows,
        ranking_rows=ranking_rows,
        tier_rows=tier_rows,
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": source_version,
        "reconstruction_contract": (
            "source_superposition_base_K + total_power_W * "
            "delta_R_eff_pred_K_per_W + zero_mean_centered_field_K"
        ),
        "correction_signs": {"mean": 1, "centered": 1},
        "models": list(model_roots),
        "protocols": EXPECTED_PROTOCOLS,
        "heldout_families": list(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"]),
        "artifacts": artifacts,
        "alignment": aligned_samples,
        "descriptor_methodology": {
            "feature_count": len(descriptor_names),
            "feature_names": descriptor_names,
            "fit_families": list(EXPECTED_PRIMARY_SPLIT["train"]),
            "standardization": "per-feature mean/std fit on the 40 training families only",
            "constant_feature_scale": 1.0,
            "euclidean": "L2 distance in train-standardized descriptor space",
            "mean_knn_k": min(top_k, len(EXPECTED_PRIMARY_SPLIT["train"])),
            "mahalanobis": "regularized covariance: 0.9*covariance + 0.1*diagonal(covariance)",
            "target_leakage": "No HotSpot or residual-error label enters descriptors or distances.",
        },
        "ood_tier_thresholds": tier_thresholds,
        "residual_decomposition": decomposition_status,
        "findings": findings,
        "blocked_outputs": blocked_outputs,
        "data_root_available": bool(data_root and data_root.is_dir()),
        "aggregate_tolerance_K": aggregate_tolerance_K,
    }
    write_json(out_dir / "zero_shot_diagnostic_summary.json", summary)
    write_report(out_dir / "zero_shot_diagnostic_report.md", summary, aggregate_rows, family_rows)
    return summary


def load_and_validate_model_artifacts(
    *,
    model_roots: Mapping[str, Path],
    source_version: str,
    aggregate_tolerance_K: float,
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, str]]]], dict[str, Any]]:
    artifacts: dict[str, Any] = {}
    sample_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    payloads: dict[str, Any] = {}
    for model, root in model_roots.items():
        sample_rows[model] = {}
        payloads[model] = {}
        model_artifacts: dict[str, Any] = {"root": str(root), "protocols": {}}
        lineage = load_training_lineage(root)
        validate_training_lineage(lineage, source_version, model)
        model_artifacts["training_lineage"] = lineage
        for protocol in EXPECTED_PROTOCOLS:
            protocol_dir = locate_protocol_dir(root, protocol)
            metrics_path = protocol_dir / "metrics.json"
            samples_path = protocol_dir / "metrics_by_sample.csv"
            metrics = read_json_required(metrics_path)
            rows = read_csv_required(samples_path)
            validate_protocol_rows(model, protocol, rows)
            validate_metrics_contract(model, metrics, source_version)
            reconcile_aggregate_metrics(
                model=model,
                protocol=protocol,
                rows=rows,
                metrics=metrics,
                tolerance_K=aggregate_tolerance_K,
            )
            sample_rows[model][protocol] = rows
            payloads[model][protocol] = metrics
            model_artifacts["protocols"][protocol] = {
                "directory": str(protocol_dir),
                "metrics": str(metrics_path),
                "per_sample": str(samples_path),
                "prediction_count": len(list((protocol_dir / "predictions").rglob("*_tpred.npy"))),
            }
        artifacts[model] = model_artifacts
    return artifacts, sample_rows, payloads


def locate_protocol_dir(root: Path, protocol: str) -> Path:
    try:
        priority_tiers = PROTOCOL_DIRECTORY_PRIORITY[protocol]
    except KeyError as exc:
        raise ValueError(
            f"unsupported protocol {protocol!r}; expected one of "
            f"{tuple(PROTOCOL_DIRECTORY_PRIORITY)}"
        ) from exc
    return _locate_protocol_dir_from_tiers(root, protocol, priority_tiers)


def _locate_protocol_dir_from_tiers(
    root: Path,
    protocol: str,
    priority_tiers: Sequence[Sequence[str]],
) -> Path:
    searched: list[str] = []
    for tier_index, relative_paths in enumerate(priority_tiers, 1):
        candidates = [root / relative_path for relative_path in relative_paths]
        searched.extend(str(path) for path in candidates)
        found = [path for path in candidates if path.is_dir()]
        if len(found) > 1:
            raise ValueError(
                f"ambiguous {protocol} directories in priority tier {tier_index} "
                f"under {root}: {[str(path) for path in found]}"
            )
        if found:
            return found[0]
    raise FileNotFoundError(
        f"no valid {protocol} directory exists under {root}; searched={searched}"
    )


def load_training_lineage(root: Path) -> dict[str, Any]:
    lineage_path = root / "training_lineage.json"
    if lineage_path.is_file():
        return read_json_required(lineage_path)
    checkpoint_path = root / "checkpoints/best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing training_lineage.json and local checkpoint metadata under {root}"
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"torch is required to inspect {checkpoint_path}") from exc
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lineage = payload.get("training_lineage")
    if not isinstance(lineage, dict):
        raise ValueError(f"{checkpoint_path} does not contain training_lineage")
    return lineage


def validate_training_lineage(lineage: Mapping[str, Any], source_version: str, model: str) -> None:
    if str(lineage.get("source_superposition_version")) != source_version:
        raise ValueError(f"{model}: source version mismatch in training lineage")
    if bool(lineage.get("primary_heldout_used_for_selection")):
        raise ValueError(f"{model}: primary held-out families were used for checkpoint selection")
    expected_train = set(EXPECTED_PRIMARY_SPLIT["train"])
    optimized = set(lineage.get("optimization_family_uids", ()))
    if optimized != expected_train:
        raise ValueError(f"{model}: optimization families differ from canonical train families")
    excluded = set(lineage.get("excluded_primary_test_family_uids", ()))
    if excluded != set(EXPECTED_PRIMARY_SPLIT["test"]):
        raise ValueError(f"{model}: primary test exclusion lineage is incomplete")
    reconstruction = str(lineage.get("reconstruction", ""))
    if " + " not in reconstruction or " - " in reconstruction:
        raise ValueError(f"{model}: residual reconstruction is not explicitly additive: {reconstruction}")


def validate_metrics_contract(model: str, metrics: Mapping[str, Any], source_version: str) -> None:
    config = metrics.get("model", {}).get("config", {})
    if config.get("physics_input_mode") != "source_superposition_v1":
        raise ValueError(f"{model}: wrong physics_input_mode")
    reconstruction = str(config.get("reconstruction", ""))
    if reconstruction and (" - " in reconstruction or reconstruction.count(" + ") < 2):
        raise ValueError(f"{model}: non-additive reconstruction in metrics")
    for key in ("mean_correction_sign", "centered_correction_sign"):
        if key in config and int(config[key]) != 1:
            raise ValueError(f"{model}: {key} must be +1")
    index = str(metrics.get("index", ""))
    if source_version not in index:
        raise ValueError(f"{model}: metrics index does not identify source version {source_version}")


def validate_protocol_rows(model: str, protocol: str, rows: Sequence[Mapping[str, str]]) -> None:
    expected = EXPECTED_PROTOCOLS[protocol]
    expected_families = set(expected["families"])
    expected_count = expected["samples_per_family"]
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        uid = require_text(row, "sample_uid")
        family = str(row.get("family_uid") or row.get("case_id") or "")
        if not family:
            raise ValueError(f"{model}/{protocol}/{uid}: missing family_uid")
        grouped[family].append(uid)
    if set(grouped) != expected_families:
        raise ValueError(
            f"{model}/{protocol}: families={sorted(grouped)}, expected={sorted(expected_families)}"
        )
    all_uids = [uid for uids in grouped.values() for uid in uids]
    duplicates = sorted(uid for uid in set(all_uids) if all_uids.count(uid) > 1)
    if duplicates:
        raise ValueError(f"{model}/{protocol}: duplicate sample IDs: {duplicates[:10]}")
    bad_counts = {family: len(uids) for family, uids in grouped.items() if len(uids) != expected_count}
    if bad_counts:
        raise ValueError(f"{model}/{protocol}: unexpected family sample counts: {bad_counts}")


def validate_cross_model_alignment(
    sample_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]]
) -> dict[str, Any]:
    reference = sample_rows["cnn"]
    result: dict[str, Any] = {}
    for protocol in EXPECTED_PROTOCOLS:
        expected = tuple(sorted(require_text(row, "sample_uid") for row in reference[protocol]))
        for model in MODEL_LABELS[1:]:
            actual = tuple(sorted(require_text(row, "sample_uid") for row in sample_rows[model][protocol]))
            if actual != expected:
                raise ValueError(f"{model}/{protocol}: sample IDs do not align with CNN")
        result[protocol] = {"sample_count": len(expected), "sample_uid_alignment": True}
    return result


def reconcile_aggregate_metrics(
    *,
    model: str,
    protocol: str,
    rows: Sequence[Mapping[str, str]],
    metrics: Mapping[str, Any],
    tolerance_K: float,
) -> None:
    mae = float(np.mean([float(row["mae_K"]) for row in rows]))
    rmse = math.sqrt(float(np.mean([float(row["rmse_K"]) ** 2 for row in rows])))
    reported = metrics["cnn_final_temperature"]
    for name, computed, saved in (
        ("MAE", mae, float(reported["mae_K"])),
        ("RMSE", rmse, float(reported["rmse_K"])),
    ):
        if abs(computed - saved) > tolerance_K:
            raise ValueError(
                f"{model}/{protocol}: recomputed {name}={computed:.9f} differs from "
                f"metrics.json={saved:.9f} by more than {tolerance_K}"
            )


def aggregate_metric_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for output_name, input_name in METRIC_COLUMNS.items():
        if input_name not in rows[0]:
            if input_name in REQUIRED_SAMPLE_METRICS:
                raise ValueError(f"required sample metric is missing: {input_name}")
            result[output_name] = ""
            continue
        values = np.asarray([float(row[input_name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {input_name}")
        result[output_name] = float(np.mean(values))
    result["rmse_K"] = math.sqrt(float(np.mean([float(row["rmse_K"]) ** 2 for row in rows])))
    result["centered_field_rmse_K"] = math.sqrt(
        float(np.mean([float(row["centered_field_rmse_K"]) ** 2 for row in rows]))
    )
    source = np.asarray([float(row["physics_baseline_mae_K"]) for row in rows])
    final = np.asarray([float(row["mae_K"]) for row in rows])
    result["fraction_worse_than_source"] = float(np.mean(final > source))
    result["absolute_improvement_over_source_K"] = float(np.mean(source - final))
    result["percentage_improvement_over_source"] = float(
        100.0 * result["absolute_improvement_over_source_K"] / max(float(np.mean(source)), 1.0e-12)
    )
    return result


def build_per_family_metrics(
    sample_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_LABELS:
        for protocol in EXPECTED_PROTOCOLS:
            grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
            for row in sample_rows[model][protocol]:
                grouped[str(row.get("family_uid") or row.get("case_id"))].append(row)
            for family in sorted(grouped):
                output.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "family_uid": family,
                        "sample_count": len(grouped[family]),
                        **aggregate_metric_rows(grouped[family]),
                    }
                )
    return output


def build_aggregate_model_comparison(
    sample_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    metrics_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_LABELS:
        for protocol in EXPECTED_PROTOCOLS:
            rows = sample_rows[model][protocol]
            by_family: dict[str, list[Mapping[str, str]]] = defaultdict(list)
            for row in rows:
                by_family[str(row.get("family_uid") or row.get("case_id"))].append(row)
            family_metrics = [aggregate_metric_rows(items) for items in by_family.values()]
            micro = aggregate_metric_rows(rows)
            metrics = metrics_payloads[model][protocol]
            output.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "sample_count": len(rows),
                    "family_count": len(by_family),
                    "micro_mae_K": micro["mae_K"],
                    "micro_rmse_K": micro["rmse_K"],
                    "macro_family_mae_K": float(np.mean([item["mae_K"] for item in family_metrics])),
                    "macro_family_rmse_K": float(np.mean([item["rmse_K"] for item in family_metrics])),
                    "source_baseline_micro_mae_K": micro["source_baseline_mae_K"],
                    "mean_rise_micro_mae_K": micro["mean_rise_mae_K"],
                    "centered_field_micro_mae_K": micro["centered_field_mae_K"],
                    "fraction_worse_than_source": micro["fraction_worse_than_source"],
                    "runtime_per_sample_s": float(metrics["inference_runtime_per_sample_s"]),
                    "parameter_count": int(metrics["model"]["parameter_count"]),
                }
            )
    return output


def rank_models_by_family(family_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in family_rows:
        if row["protocol"] != "known_family_sample_test":
            grouped[(str(row["protocol"]), str(row["family_uid"]))].append(row)
    output: list[dict[str, Any]] = []
    for (protocol, family), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (float(row["mae_K"]), str(row["model"])))
        best = float(ordered[0]["mae_K"])
        lookup = {str(row["model"]): float(row["mae_K"]) for row in rows}
        for rank, row in enumerate(ordered, 1):
            record = {
                "protocol": protocol,
                "family_uid": family,
                "rank": rank,
                "model": row["model"],
                "mae_K": row["mae_K"],
                "difference_from_best_K": float(row["mae_K"]) - best,
            }
            for left in MODEL_LABELS:
                for right in MODEL_LABELS:
                    if left < right:
                        record[f"{left}_minus_{right}_mae_K"] = lookup[left] - lookup[right]
            output.append(record)
    return output


def build_source_improvement_rows(
    family_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "protocol": row["protocol"],
            "family_uid": row["family_uid"],
            "source_baseline_mae_K": row["source_baseline_mae_K"],
            "final_mae_K": row["mae_K"],
            "absolute_improvement_K": row["absolute_improvement_over_source_K"],
            "percentage_improvement": row["percentage_improvement_over_source"],
            "fraction_worse_than_source": row["fraction_worse_than_source"],
        }
        for row in family_rows
    ]


def validate_descriptor_table(
    rows: Sequence[Mapping[str, str]], prior_summary: Mapping[str, Any]
) -> list[str]:
    expected = set(sum((list(values) for values in EXPECTED_PRIMARY_SPLIT.values()), []))
    actual = {require_text(row, "family_uid") for row in rows}
    if actual != expected or len(rows) != 50:
        raise ValueError(f"descriptor table families do not match canonical 50-family benchmark")
    names = list(prior_summary.get("descriptor_names", ()))
    if not names:
        names = [name for name in rows[0] if name not in NON_DESCRIPTOR_COLUMNS]
    for row in rows:
        for name in names:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"{row['family_uid']}: non-finite descriptor {name}")
    return names


def fit_descriptor_space(
    rows: Sequence[Mapping[str, str]],
    descriptor_names: Sequence[str],
    *,
    train_family_uids: Sequence[str],
) -> dict[str, Any]:
    by_uid = {row["family_uid"]: row for row in rows}
    train = np.asarray(
        [[float(by_uid[uid][name]) for name in descriptor_names] for uid in train_family_uids],
        dtype=np.float64,
    )
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    scale = np.where(std > 1.0e-12, std, 1.0)
    standardized = {
        uid: (np.asarray([float(row[name]) for name in descriptor_names]) - mean) / scale
        for uid, row in by_uid.items()
    }
    covariance = np.cov(np.vstack([standardized[uid] for uid in train_family_uids]), rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    regularized = 0.9 * covariance + 0.1 * diagonal + np.eye(len(descriptor_names)) * 1.0e-8
    inverse_covariance = np.linalg.pinv(regularized, hermitian=True)
    return {
        "mean": mean,
        "std": std,
        "scale": scale,
        "minimum": train.min(axis=0),
        "maximum": train.max(axis=0),
        "standardized": standardized,
        "inverse_covariance": inverse_covariance,
    }


def build_distance_outputs(
    rows: Sequence[Mapping[str, str]],
    descriptor_names: Sequence[str],
    space: Mapping[str, Any],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del descriptor_names
    split = {row["family_uid"]: row["split"] for row in rows}
    standardized = space["standardized"]
    inverse = space["inverse_covariance"]
    uids = sorted(standardized)
    distance_rows: list[dict[str, Any]] = []
    for left in uids:
        record: dict[str, Any] = {"family_uid": left, "split": split[left]}
        for right in uids:
            delta = standardized[left] - standardized[right]
            record[right] = float(np.linalg.norm(delta))
        distance_rows.append(record)
    nearest_rows: list[dict[str, Any]] = []
    train = tuple(EXPECTED_PRIMARY_SPLIT["train"])
    for heldout in EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"]:
        candidates = []
        for train_uid in train:
            delta = standardized[heldout] - standardized[train_uid]
            candidates.append(
                (
                    float(np.linalg.norm(delta)),
                    float(math.sqrt(max(float(delta @ inverse @ delta), 0.0))),
                    train_uid,
                )
            )
        for rank, (euclidean, mahalanobis, train_uid) in enumerate(sorted(candidates)[:top_k], 1):
            nearest_rows.append(
                {
                    "heldout_family_uid": heldout,
                    "heldout_split": split[heldout],
                    "rank": rank,
                    "train_family_uid": train_uid,
                    "euclidean_distance": euclidean,
                    "mahalanobis_distance": mahalanobis,
                }
            )
    return distance_rows, nearest_rows


def family_error_labels(
    family_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in family_rows:
        result[str(row["family_uid"])][str(row["model"])] = {
            "final_mae_K": float(row["mae_K"]),
            "source_mae_K": float(row["source_baseline_mae_K"]),
            "mean_mae_K": float(row["mean_rise_mae_K"]),
            "centered_mae_K": float(row["centered_field_mae_K"]),
            "hotspot_mae_K": float(row["hotspot_top1pct_mae_K"]),
        }
    return dict(result)


def assign_ood_tiers(
    *,
    descriptor_rows: Sequence[Mapping[str, str]],
    descriptor_names: Sequence[str],
    descriptor_space: Mapping[str, Any],
    family_errors: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    standardized = descriptor_space["standardized"]
    train = tuple(EXPECTED_PRIMARY_SPLIT["train"])
    heldout = tuple(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"])
    by_uid = {row["family_uid"]: row for row in descriptor_rows}

    train_nearest: dict[str, tuple[float, str]] = {}
    for uid in train:
        candidates = [
            (float(np.linalg.norm(standardized[uid] - standardized[other])), other)
            for other in train
            if other != uid
        ]
        train_nearest[uid] = min(candidates)
    train_distances = np.asarray([value[0] for value in train_nearest.values()])
    close_threshold = float(np.quantile(train_distances, 0.75))
    distant_threshold = float(np.quantile(train_distances, 0.95))
    train_source_gaps = np.asarray(
        [
            abs(
                family_errors[uid]["cnn"]["source_mae_K"]
                - family_errors[neighbor]["cnn"]["source_mae_K"]
            )
            for uid, (_, neighbor) in train_nearest.items()
        ]
    )
    response_gap_threshold = float(np.quantile(train_source_gaps, 0.95))
    train_final_gaps = np.asarray(
        [
            abs(
                family_errors[uid]["cnn"]["final_mae_K"]
                - family_errors[neighbor]["cnn"]["final_mae_K"]
            )
            for uid, (_, neighbor) in train_nearest.items()
        ]
    )
    final_response_gap_threshold = float(np.quantile(train_final_gaps, 0.95))
    records: list[dict[str, Any]] = []
    for uid in heldout:
        candidates = sorted(
            (
                float(np.linalg.norm(standardized[uid] - standardized[train_uid])),
                train_uid,
            )
            for train_uid in train
        )
        nearest_distance, nearest_uid = candidates[0]
        mean_knn = float(np.mean([item[0] for item in candidates[:5]]))
        values = np.asarray([float(by_uid[uid][name]) for name in descriptor_names])
        outside = (values < descriptor_space["minimum"] - 1.0e-12) | (
            values > descriptor_space["maximum"] + 1.0e-12
        )
        response_gap = abs(
            family_errors[uid]["cnn"]["source_mae_K"]
            - family_errors[nearest_uid]["cnn"]["source_mae_K"]
        )
        final_response_gap = abs(
            family_errors[uid]["cnn"]["final_mae_K"]
            - family_errors[nearest_uid]["cnn"]["final_mae_K"]
        )
        close = nearest_distance <= close_threshold
        response_anomalous = (
            nearest_distance <= distant_threshold
            and (
                response_gap > response_gap_threshold
                or final_response_gap > final_response_gap_threshold
            )
        )
        flags: list[str] = []
        if outside.any():
            flags.append("marginal_extrapolation")
        if nearest_distance > distant_threshold:
            flags.append("descriptor_distant_holdout")
        if response_anomalous:
            flags.append("response_anomalous_holdout")
        if response_anomalous:
            primary = "response_anomalous_holdout"
        elif nearest_distance > distant_threshold:
            primary = "descriptor_distant_holdout"
        elif outside.any():
            primary = "marginal_extrapolation"
        else:
            primary = "close_combinational_holdout"
        records.append(
            {
                "family_uid": uid,
                "split": by_uid[uid]["split"],
                "primary_tier": primary,
                "secondary_flags": ";".join(flags),
                "nearest_train_family": nearest_uid,
                "nearest_train_distance": nearest_distance,
                "mean_5nn_train_distance": mean_knn,
                "source_response_gap_to_nearest_K": response_gap,
                "final_response_gap_to_nearest_K": final_response_gap,
                "descriptors_outside_train_range_count": int(outside.sum()),
                "descriptors_outside_train_range_fraction": float(outside.mean()),
            }
        )
    return records, {
        "close_distance_train_leave_one_out_q75": close_threshold,
        "distant_distance_train_leave_one_out_q95": distant_threshold,
        "response_gap_train_nearest_neighbor_q95_K": response_gap_threshold,
        "final_error_gap_train_nearest_neighbor_q95_K": final_response_gap_threshold,
        "marginal_extrapolation_rule": "at least one descriptor outside the training-family min/max",
        "response_anomaly_rule": (
            "distance <= train q95 and either source-MAE gap or final-error gap exceeds "
            "the corresponding train-neighbor gap q95"
        ),
    }


def attach_ood_columns(
    descriptor_rows: Sequence[Mapping[str, str]], tier_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    tiers = {row["family_uid"]: row for row in tier_rows}
    return [{**row, **tiers.get(row["family_uid"], {})} for row in descriptor_rows]


def build_error_descriptor_correlations(
    *,
    tier_rows: Sequence[Mapping[str, Any]],
    family_errors: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for distance_name in ("nearest_train_distance", "mean_5nn_train_distance"):
        x = np.asarray([float(row[distance_name]) for row in tier_rows])
        for model in MODEL_LABELS:
            for error_name in ("source_mae_K", "final_mae_K", "mean_mae_K", "centered_mae_K"):
                y = np.asarray(
                    [family_errors[str(row["family_uid"])][model][error_name] for row in tier_rows]
                )
                outputs.append(
                    {
                        "distance_metric": distance_name,
                        "model": model,
                        "error_metric": error_name,
                        "family_count": len(x),
                        "pearson_r": pearson(x, y),
                        "spearman_rho": spearman(x, y),
                        "interpretation": "exploratory; only 10 held-out families",
                    }
                )
    return outputs


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def audit_residual_decomposition(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "file missing"}
    rows = read_csv_required(path)
    families = sorted({str(row.get("family_uid") or row.get("case_id")) for row in rows})
    return {
        "available": True,
        "path": str(path),
        "sample_count": len(rows),
        "families": families,
        "required_columns_present": all(
            name in rows[0]
            for name in (
                "true_residual_mean_K",
                "absolute_mean_correction_error_K",
                "centered_spatial_mae_K",
            )
        ),
    }


def derive_findings(
    *,
    aggregate_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    tier_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    test_aggregate = {
        row["model"]: row
        for row in aggregate_rows
        if row["protocol"] == "primary_test_families"
    }
    test_family = [
        row for row in family_rows if row["protocol"] == "primary_test_families"
    ]
    winners = [
        row for row in ranking_rows if row["protocol"] == "primary_test_families" and row["rank"] == 1
    ]
    cnn_test = sorted(
        (row for row in test_family if row["model"] == "cnn"),
        key=lambda row: float(row["mae_K"]),
        reverse=True,
    )
    total_excess = sum(float(row["mae_K"]) for row in cnn_test)
    worst_share = float(cnn_test[0]["mae_K"]) / max(total_excess, 1.0e-12)
    known_aggregate = {
        row["model"]: row
        for row in aggregate_rows
        if row["protocol"] == "known_family_sample_test"
    }
    heldout_families = tuple(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"])
    tier_lookup = {str(row["family_uid"]): row for row in tier_rows}
    tier_model_mae: dict[str, dict[str, float]] = {}
    for tier in sorted({str(row["primary_tier"]) for row in tier_rows}):
        families = [uid for uid in heldout_families if tier_lookup[uid]["primary_tier"] == tier]
        tier_model_mae[tier] = {
            model: float(
                np.mean(
                    [
                        float(row["mae_K"])
                        for row in family_rows
                        if row["family_uid"] in families and row["model"] == model
                    ]
                )
            )
            for model in MODEL_LABELS
        }
        tier_model_mae[tier]["winner"] = min(
            MODEL_LABELS, key=lambda model: tier_model_mae[tier][model]
        )
    source_final_association: dict[str, dict[str, float]] = {}
    for model in MODEL_LABELS:
        model_rows = [
            row
            for row in family_rows
            if row["family_uid"] in heldout_families and row["model"] == model
        ]
        source = np.asarray([float(row["source_baseline_mae_K"]) for row in model_rows])
        final = np.asarray([float(row["mae_K"]) for row in model_rows])
        source_final_association[model] = {
            "pearson_r": pearson(source, final),
            "spearman_rho": spearman(source, final),
            "family_count": len(model_rows),
        }
    return {
        "primary_test_best_aggregate_model": min(
            test_aggregate, key=lambda model: float(test_aggregate[model]["micro_mae_K"])
        ),
        "primary_test_model_wins_by_family": {
            model: sum(row["model"] == model for row in winners) for model in MODEL_LABELS
        },
        "known_family_best_model": min(
            known_aggregate, key=lambda model: float(known_aggregate[model]["micro_mae_K"])
        ),
        "known_family_model_order": sorted(
            MODEL_LABELS, key=lambda model: float(known_aggregate[model]["micro_mae_K"])
        ),
        "heldout_test_model_order": sorted(
            MODEL_LABELS, key=lambda model: float(test_aggregate[model]["micro_mae_K"])
        ),
        "model_mae_by_ood_tier": tier_model_mae,
        "source_baseline_vs_final_error_association": source_final_association,
        "cnn_worst_test_family": cnn_test[0]["family_uid"],
        "cnn_worst_family_mae_K": cnn_test[0]["mae_K"],
        "cnn_worst_family_share_of_sum_family_mae": worst_share,
        "cnn_test_family_mae_range_K": [
            min(float(row["mae_K"]) for row in cnn_test),
            max(float(row["mae_K"]) for row in cnn_test),
        ],
        "cnn_centered_vs_mean_test": {
            "centered_field_mae_K": test_aggregate["cnn"]["centered_field_micro_mae_K"],
            "mean_rise_mae_K": test_aggregate["cnn"]["mean_rise_micro_mae_K"],
            "centered_to_mean_ratio": (
                float(test_aggregate["cnn"]["centered_field_micro_mae_K"])
                / max(float(test_aggregate["cnn"]["mean_rise_micro_mae_K"]), 1.0e-12)
            ),
            "hotspot_top1pct_mae_K": float(
                np.mean(
                    [
                        float(row["hotspot_top1pct_mae_K"])
                        for row in cnn_test
                    ]
                )
            ),
        },
        "f044_ood_tier": next(row for row in tier_rows if row["family_uid"] == "f044"),
    }


def write_summary_plots(
    *,
    out_dir: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    source_improvement_rows: Sequence[Mapping[str, Any]],
    tier_rows: Sequence[Mapping[str, Any]],
    descriptor_rows: Sequence[Mapping[str, str]],
    descriptor_names: Sequence[str],
    descriptor_space: Mapping[str, Any],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["publication plots: matplotlib is unavailable"]
    colors = plt.get_cmap("tab10").colors
    heldout = list(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"])
    by_key = {(row["model"], row["family_uid"]): row for row in family_rows}

    def grouped_plot(path: Path, metric: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        x = np.arange(len(heldout))
        width = 0.19
        for index, model in enumerate(MODEL_LABELS):
            values = [float(by_key[(model, uid)][metric]) for uid in heldout]
            ax.bar(x + (index - 1.5) * width, values, width, label=model.upper(), color=colors[index])
        ax.set_xticks(x, heldout)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    grouped_plot(out_dir / "per_family_mae.png", "mae_K", "MAE (K)")
    grouped_plot(out_dir / "per_family_rmse.png", "rmse_K", "RMSE (K)")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(heldout))
    width = 0.19
    improvement = {
        (row["model"], row["family_uid"]): row
        for row in source_improvement_rows
        if row["protocol"] != "known_family_sample_test"
    }
    for index, model in enumerate(MODEL_LABELS):
        ax.bar(
            x + (index - 1.5) * width,
            [float(improvement[(model, uid)]["absolute_improvement_K"]) for uid in heldout],
            width,
            label=model.upper(),
            color=colors[index],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, heldout)
    ax.set_ylabel("Source MAE - final MAE (K)")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "source_improvement_by_family.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    tiers = {row["family_uid"]: row for row in tier_rows}
    for index, model in enumerate(MODEL_LABELS):
        ax.scatter(
            [float(tiers[uid]["nearest_train_distance"]) for uid in heldout],
            [float(by_key[(model, uid)]["mae_K"]) for uid in heldout],
            label=model.upper(),
            color=colors[index],
            s=42,
        )
    for uid in heldout:
        ax.annotate(
            uid,
            (
                float(tiers[uid]["nearest_train_distance"]),
                float(by_key[("cnn", uid)]["mae_K"]),
            ),
            fontsize=7,
        )
    ax.set_xlabel("Nearest training-family distance")
    ax.set_ylabel("Final MAE (K)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_descriptor_distance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for index, model in enumerate(MODEL_LABELS):
        ax.scatter(
            [float(by_key[(model, uid)]["mean_rise_mae_K"]) for uid in heldout],
            [float(by_key[(model, uid)]["centered_field_mae_K"]) for uid in heldout],
            label=model.upper(),
            color=colors[index],
            s=42,
        )
    ax.set_xlabel("Mean-correction MAE (K)")
    ax.set_ylabel("Centered-field MAE (K)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "centered_vs_mean_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, title in (
        (axes[0], "boundary_region_mae_K", "Boundary MAE"),
        (axes[1], "hotspot_top1pct_mae_K", "Hotspot top-1% MAE"),
    ):
        for index, model in enumerate(MODEL_LABELS):
            axis.plot(
                heldout,
                [float(by_key[(model, uid)][metric]) for uid in heldout],
                marker="o",
                label=model.upper(),
                color=colors[index],
            )
        axis.set_title(title)
        axis.set_ylabel("K")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_and_hotspot_error.png", dpi=180)
    plt.close(fig)

    test_aggregate = [
        row for row in aggregate_rows if row["protocol"] == "primary_test_families"
    ]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for index, row in enumerate(test_aggregate):
        ax.scatter(
            float(row["runtime_per_sample_s"]) * 1.0e3,
            float(row["micro_mae_K"]),
            s=70,
            label=str(row["model"]).upper(),
            color=colors[index],
        )
    ax.set_xlabel("Cached model runtime (ms/sample)")
    ax.set_ylabel("Held-out test MAE (K)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "model_accuracy_runtime_pareto.png", dpi=180)
    plt.close(fig)

    by_uid = {row["family_uid"]: row for row in descriptor_rows}
    train = tuple(EXPECTED_PRIMARY_SPLIT["train"])
    matrix = np.vstack([descriptor_space["standardized"][uid] for uid in train])
    _, _, vt = np.linalg.svd(matrix - matrix.mean(axis=0), full_matrices=False)
    projection = vt[:2].T
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for split_index, (split, uids) in enumerate(EXPECTED_PRIMARY_SPLIT.items()):
        points = np.vstack([descriptor_space["standardized"][uid] @ projection for uid in uids])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            label=split,
            color=colors[split_index],
            s=35 if split == "train" else 55,
        )
        if split != "train":
            for uid, point in zip(uids, points):
                ax.annotate(uid, point, fontsize=7)
    ax.set_xlabel("Train-fit PCA component 1")
    ax.set_ylabel("Train-fit PCA component 2")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_family_descriptor_embedding.png", dpi=180)
    plt.close(fig)
    return []


def maybe_plot_representative_heatmaps(
    *,
    out_dir: Path,
    data_root: Path | None,
    source_version: str,
    model_roots: Mapping[str, Path],
    sample_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
) -> str | None:
    if data_root is None or not data_root.is_dir():
        return (
            "representative_best_worst_heatmaps.png: raw Benchmark v2 data root is not "
            "available locally; predictions exist, but source and HotSpot arrays are required "
            "for honest common-scale panels"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "representative_best_worst_heatmaps.png: matplotlib is unavailable"

    index_path = (
        data_root
        / "derived/indices/full_50x200/source_superposition"
        / source_version
        / "family_split/test_index.csv"
    )
    if not index_path.is_file():
        return f"representative_best_worst_heatmaps.png: missing portable test index {index_path}"
    index_rows = {row["sample_uid"]: row for row in read_csv_required(index_path)}
    cnn_rows = {row["sample_uid"]: row for row in sample_rows["cnn"]["primary_test_families"]}
    fno_rows = {row["sample_uid"]: row for row in sample_rows["fno"]["primary_test_families"]}
    if set(index_rows) != set(cnn_rows):
        raise ValueError("portable test index and saved CNN sample metrics do not align")

    ordered = sorted(cnn_rows, key=lambda uid: float(cnn_rows[uid]["mae_K"]))
    f044 = sorted(uid for uid in ordered if str(cnn_rows[uid]["family_uid"]) == "f044")
    selections = [
        ("best CNN", ordered[0]),
        ("median CNN", ordered[len(ordered) // 2]),
        ("worst CNN", ordered[-1]),
        (
            "FNO beats CNN most",
            max(ordered, key=lambda uid: float(cnn_rows[uid]["mae_K"]) - float(fno_rows[uid]["mae_K"])),
        ),
        (
            "CNN beats FNO most",
            max(ordered, key=lambda uid: float(fno_rows[uid]["mae_K"]) - float(cnn_rows[uid]["mae_K"])),
        ),
        ("representative f044", f044[len(f044) // 2]),
    ]
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, uid in selections:
        if uid not in seen:
            unique.append((label, uid))
            seen.add(uid)

    model_prediction_roots = {
        model: locate_protocol_dir(root, "primary_test_families") / "predictions"
        for model, root in model_roots.items()
    }
    column_titles = ("HotSpot target", "Source base", "CNN", "FNO", "U-FNO", "SAU-FNO")
    fig, axes = plt.subplots(
        2 * len(unique),
        len(column_titles),
        figsize=(15, 4.2 * len(unique)),
        squeeze=False,
    )
    for selection_index, (label, uid) in enumerate(unique):
        row = index_rows[uid]
        family = str(row.get("family_uid") or row.get("case_id"))
        target = load_thermal_map(
            resolve_data_path(require_any_text(row, ("y_path", "target_path", "final_temperature")), data_root),
            f"{uid} target",
        )
        source = load_thermal_map(
            resolve_data_path(require_text(row, "source_superposition_base_path"), data_root),
            f"{uid} source base",
        )
        predictions = {
            model: load_thermal_map(
                model_prediction_roots[model] / family / f"{uid}_tpred.npy",
                f"{uid} {model} prediction",
            )
            for model in MODEL_LABELS
        }
        temperatures = [target, source, *(predictions[model] for model in MODEL_LABELS)]
        temp_min = min(float(array.min()) for array in temperatures)
        temp_max = max(float(array.max()) for array in temperatures)
        errors = [np.abs(source - target), *(np.abs(predictions[model] - target) for model in MODEL_LABELS)]
        error_max = max(float(array.max()) for array in errors)
        top_axes = axes[2 * selection_index]
        error_axes = axes[2 * selection_index + 1]
        for column, (axis, array, title) in enumerate(zip(top_axes, temperatures, column_titles)):
            image = axis.imshow(array, cmap="inferno", vmin=temp_min, vmax=temp_max)
            axis.set_title(title if selection_index == 0 else "")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f"{label}\n{uid}", fontsize=8)
        fig.colorbar(image, ax=top_axes.tolist(), fraction=0.012, pad=0.01, label="Temperature (K)")
        error_axes[0].axis("off")
        error_titles = ("Source error", "CNN error", "FNO error", "U-FNO error", "SAU-FNO error")
        for axis, array, title in zip(error_axes[1:], errors, error_titles):
            error_image = axis.imshow(array, cmap="magma", vmin=0.0, vmax=error_max)
            axis.set_title(title if selection_index == 0 else "")
            axis.set_xticks([])
            axis.set_yticks([])
        fig.colorbar(
            error_image,
            ax=error_axes[1:].tolist(),
            fraction=0.012,
            pad=0.01,
            label="Absolute error (K)",
        )
    fig.subplots_adjust(left=0.08, right=0.94, top=0.98, bottom=0.02, hspace=0.18, wspace=0.08)
    fig.savefig(out_dir / "representative_best_worst_heatmaps.png", dpi=170)
    plt.close(fig)
    return None


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
) -> None:
    findings = summary["findings"]
    lines = [
        "# Benchmark v2 Zero-Shot Diagnostic Report",
        "",
        "## Protocol",
        "",
        "The comparison uses the immutable 40/5/5 family split and the frozen "
        "`source_superposition_final_train40_source_v1` source baseline. All results are "
        "read from saved evaluation artifacts; no checkpoint inference or training is run.",
        "",
        "Residual reconstruction is:",
        "",
        "`T = source_base + total_power * delta_R_eff + zero_mean_centered_field`.",
        "",
        "Both correction signs are +1.",
        "",
        "## Canonical Results",
        "",
        "| Model | Protocol | Micro MAE (K) | Micro RMSE (K) | Macro family MAE (K) | Runtime (ms) | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {str(row['model']).upper()} | {row['protocol']} | "
            f"{float(row['micro_mae_K']):.4f} | {float(row['micro_rmse_K']):.4f} | "
            f"{float(row['macro_family_mae_K']):.4f} | "
            f"{1000.0 * float(row['runtime_per_sample_s']):.3f} | "
            f"{int(row['parameter_count']):,} |"
        )
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- Aggregate held-out test winner: **{findings['primary_test_best_aggregate_model'].upper()}**.",
            f"- Worst CNN test family: **{findings['cnn_worst_test_family']}** at "
            f"{float(findings['cnn_worst_family_mae_K']):.3f} K MAE.",
            f"- CNN test-family MAE range: {findings['cnn_test_family_mae_range_K'][0]:.3f} to "
            f"{findings['cnn_test_family_mae_range_K'][1]:.3f} K.",
            f"- CNN centered-field versus mean-correction MAE: "
            f"{findings['cnn_centered_vs_mean_test']['centered_field_mae_K']:.3f} versus "
            f"{findings['cnn_centered_vs_mean_test']['mean_rise_mae_K']:.3f} K.",
            f"- Known-family winner: **{findings['known_family_best_model'].upper()}**; "
            f"held-out test order: "
            f"{', '.join(model.upper() for model in findings['heldout_test_model_order'])}.",
            f"- Source-baseline versus CNN final-error Spearman association over held-out "
            f"families: "
            f"{findings['source_baseline_vs_final_error_association']['cnn']['spearman_rho']:.3f}.",
            f"- f044 OOD tier: **{findings['f044_ood_tier']['primary_tier']}**; nearest "
            f"training family is {findings['f044_ood_tier']['nearest_train_family']}.",
            "",
            "## Distance And OOD Method",
            "",
            "Descriptors are inference-time geometry, material, context, and source-response "
            "statistics from the existing family descriptor artifact. Standardization and PCA "
            "are fit on the 40 training families only. Euclidean distance is measured in that "
            "standardized space; Mahalanobis distance uses a 10% diagonal regularization. "
            "No target or residual-error label enters a descriptor or distance.",
            "",
            "Tier thresholds are train-derived and are recorded in "
            "`zero_shot_diagnostic_summary.json`. Correlations over ten held-out families are "
            "exploratory rather than inferential.",
            "",
            "## Limitations",
            "",
        ]
    )
    if summary["blocked_outputs"]:
        lines.extend(f"- {item}" for item in summary["blocked_outputs"])
    else:
        lines.append("- No requested local diagnostic was blocked.")
    lines.extend(
        [
            "",
            "## Closure Recommendation",
            "",
            "The zero-shot architecture comparison is closed for this benchmark protocol: "
            "the four canonical models have aligned family-wise evaluation, and increased "
            "operator complexity does not produce a held-out-family gain over the residual "
            "CNN. The next evidence-bearing step is family-count scaling or carefully scoped "
            "few-shot adaptation, not another zero-shot backbone. Benchmark extensions should "
            "retain response-anomalous, descriptor-close families such as f044 because they "
            "stress behavior that marginal-range coverage alone does not expose.",
            "",
            "This conclusion is specific to ChipTherm Benchmark v2 and does not claim universal "
            "superiority over published thermal-surrogate methods.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_required(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def read_json_required(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def require_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required field {key}")
    return value


def require_any_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"none of the required fields are populated: {tuple(keys)}")


def resolve_data_path(logical_path: str, data_root: Path) -> Path:
    path = Path(logical_path).expanduser()
    candidates = [path] if path.is_absolute() else [data_root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve logical path {logical_path!r} against data_root={data_root}"
    )


def load_thermal_map(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    array = np.asarray(np.load(path), dtype=np.float64).squeeze()
    if array.shape != (64, 64):
        raise ValueError(f"{label} must be 64x64, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN/Inf")
    return array


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
