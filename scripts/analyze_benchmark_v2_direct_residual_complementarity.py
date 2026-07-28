#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOLS = (
    "known_family_sample_test",
    "primary_validation_families",
    "primary_test_families",
)
PATH_FIELDS = ("y_path", "temp_layer0_path", "original_temp_path")
SOURCE_FIELDS = ("source_superposition_base_path", "prediction_path")
NUMERIC_EPS = 1.0e-12


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline oracle and complementarity analysis for Benchmark v2 direct-temperature "
            "and source-plus-residual predictions."
        )
    )
    parser.add_argument("--direct-eval-root", required=True, type=Path)
    parser.add_argument("--residual-eval-root", required=True, type=Path)
    parser.add_argument("--source-eval-root", default=None, type=Path)
    parser.add_argument("--family-descriptor-csv", default=None, type=Path)
    parser.add_argument("--family-cluster-csv", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tie-tolerance-K", default=0.01, type=float)
    parser.add_argument("--high-error-threshold-K", default=2.0, type=float)
    parser.add_argument("--catastrophic-error-threshold-K", default=3.0, type=float)
    parser.add_argument("--reliability-bins", default=10, type=int)
    args = parser.parse_args()
    if args.tie_tolerance_K < 0.0:
        raise SystemExit("--tie-tolerance-K must be nonnegative")
    if args.high_error_threshold_K <= 0.0 or args.catastrophic_error_threshold_K <= 0.0:
        raise SystemExit("error thresholds must be positive")
    if args.reliability_bins <= 0:
        raise SystemExit("--reliability-bins must be positive")

    result = run_analysis(
        direct_eval_root=args.direct_eval_root,
        residual_eval_root=args.residual_eval_root,
        source_eval_root=args.source_eval_root,
        family_descriptor_csv=args.family_descriptor_csv,
        family_cluster_csv=args.family_cluster_csv,
        out_dir=args.out_dir,
        tie_tolerance_K=args.tie_tolerance_K,
        high_error_threshold_K=args.high_error_threshold_K,
        catastrophic_error_threshold_K=args.catastrophic_error_threshold_K,
        reliability_bin_count=args.reliability_bins,
    )
    print("Benchmark v2 direct/residual complementarity analysis complete")
    print(f"Matched samples: {result['sample_count']}")
    print(f"Recommendation: {result['recommendation']['code']} - {result['recommendation']['label']}")
    print(f"Output: {Path(args.out_dir).expanduser().resolve()}")
    return 0


def run_analysis(
    *,
    direct_eval_root: str | Path,
    residual_eval_root: str | Path,
    out_dir: str | Path,
    source_eval_root: str | Path | None = None,
    family_descriptor_csv: str | Path | None = None,
    family_cluster_csv: str | Path | None = None,
    tie_tolerance_K: float = 0.01,
    high_error_threshold_K: float = 2.0,
    catastrophic_error_threshold_K: float = 3.0,
    reliability_bin_count: int = 10,
) -> dict[str, Any]:
    direct_root = Path(direct_eval_root).expanduser().resolve()
    residual_root = Path(residual_eval_root).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_metrics = load_optional_source_metrics(source_eval_root)
    records: list[dict[str, Any]] = []
    input_audit: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        protocol_records, _arrays, audit = load_protocol_records(
            protocol=protocol,
            direct_dir=direct_root / protocol,
            residual_dir=residual_root / protocol,
            source_metrics=source_metrics,
            tie_tolerance_K=tie_tolerance_K,
        )
        records.extend(protocol_records)
        input_audit[protocol] = audit

    family_rows = aggregate_families(records)
    protocol_rows, oracle_rows, ensemble_rows = aggregate_protocols(records)
    correlation_rows = disagreement_correlations(
        records,
        high_error_threshold_K=high_error_threshold_K,
        catastrophic_error_threshold_K=catastrophic_error_threshold_K,
    )
    reliability_rows = reliability_bins(records, reliability_bin_count)
    regime_rows = aggregate_regimes(records)

    descriptor_result = analyze_descriptors(
        family_rows,
        family_descriptor_csv=family_descriptor_csv,
        family_cluster_csv=family_cluster_csv,
    )
    routing_rows, test_routing_rows, threshold_result = analyze_routing_proxies(
        records,
        family_rows,
        descriptor_result=descriptor_result,
    )
    recommendation = make_recommendation(
        protocol_rows,
        family_rows,
        regime_rows,
        routing_rows,
    )

    write_csv(output_root / "per_sample_complementarity.csv", records)
    write_csv(output_root / "per_family_complementarity.csv", family_rows)
    write_csv(output_root / "protocol_summary.csv", protocol_rows)
    write_csv(output_root / "oracle_selector_summary.csv", oracle_rows)
    write_csv(output_root / "ensemble_summary.csv", ensemble_rows)
    write_csv(output_root / "disagreement_correlation_summary.csv", correlation_rows)
    write_csv(output_root / "disagreement_reliability_bins.csv", reliability_rows)
    write_csv(
        output_root / "ood_descriptor_correlations.csv",
        descriptor_result["correlations"],
        allow_empty=True,
    )
    write_csv(output_root / "routing_proxy_summary.csv", routing_rows)
    write_csv(
        output_root / "test_family_routing_predictions.csv",
        test_routing_rows,
        allow_empty=True,
    )

    summary = {
        "schema_version": "benchmark_v2_direct_residual_complementarity/1",
        "analysis_type": "offline_oracle_and_complementarity_non_deployable",
        "sample_count": len(records),
        "protocols": list(PROTOCOLS),
        "tie_tolerance_K": float(tie_tolerance_K),
        "high_error_threshold_K": float(high_error_threshold_K),
        "catastrophic_error_threshold_K": float(catastrophic_error_threshold_K),
        "inputs": {
            "direct_eval_root": str(direct_root),
            "residual_eval_root": str(residual_root),
            "source_eval_root": str(Path(source_eval_root).expanduser().resolve())
            if source_eval_root is not None
            else None,
            "family_descriptor_csv": str(Path(family_descriptor_csv).expanduser().resolve())
            if family_descriptor_csv is not None
            else None,
            "family_cluster_csv": str(Path(family_cluster_csv).expanduser().resolve())
            if family_cluster_csv is not None
            else None,
        },
        "input_audit": input_audit,
        "protocol_summary": protocol_rows,
        "oracle_selector_summary": oracle_rows,
        "ensemble_summary": ensemble_rows,
        "workload_regime_summary": regime_rows,
        "disagreement_correlations": correlation_rows,
        "routing_proxies": routing_rows,
        "validation_tuned_threshold": threshold_result,
        "descriptor_analysis": descriptor_result["summary"],
        "recommendation": recommendation,
    }
    write_json(output_root / "complementarity_analysis.json", summary)
    write_report(output_root / "direct_residual_complementarity_report.md", summary, family_rows)
    write_figures(output_root, records, family_rows, protocol_rows, routing_rows, reliability_rows)
    return {
        "sample_count": len(records),
        "recommendation": recommendation,
        "summary": summary,
    }


def load_protocol_records(
    *,
    protocol: str,
    direct_dir: Path,
    residual_dir: Path,
    source_metrics: Mapping[str, Mapping[str, str]],
    tie_tolerance_K: float,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    direct_metrics = load_json(direct_dir / "metrics.json")
    residual_metrics = load_json(residual_dir / "metrics.json")
    validate_prediction_contract(direct_metrics, expected="direct")
    validate_prediction_contract(residual_metrics, expected="residual")

    direct_index = resolve_evaluation_index(direct_metrics)
    residual_index = resolve_evaluation_index(residual_metrics)
    direct_rows = strict_uid_map(read_csv(direct_index), f"{protocol} direct index")
    residual_rows = strict_uid_map(read_csv(residual_index), f"{protocol} residual index")
    direct_sample_metrics = strict_uid_map(
        read_csv(direct_dir / "metrics_by_sample.csv"),
        f"{protocol} direct sample metrics",
    )
    residual_sample_metrics = strict_uid_map(
        read_csv(residual_dir / "metrics_by_sample.csv"),
        f"{protocol} residual sample metrics",
    )
    expected = set(direct_rows)
    require_same_uids(expected, set(residual_rows), protocol, "index")
    require_same_uids(expected, set(direct_sample_metrics), protocol, "direct metrics")
    require_same_uids(expected, set(residual_sample_metrics), protocol, "residual metrics")

    direct_predictions = prediction_map(direct_dir / "predictions")
    residual_predictions = prediction_map(residual_dir / "predictions")
    require_same_uids(expected, set(direct_predictions), protocol, "direct predictions")
    require_same_uids(expected, set(residual_predictions), protocol, "residual predictions")

    records: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    target_exact_matches = 0
    for uid in sorted(expected):
        direct_row = direct_rows[uid]
        residual_row = residual_rows[uid]
        family = family_for_row(direct_row)
        if family != family_for_row(residual_row):
            raise ValueError(
                f"{protocol} family mismatch for {uid}: "
                f"{family!r} != {family_for_row(residual_row)!r}"
            )
        direct_target = load_index_map(direct_index, direct_row, PATH_FIELDS, "direct target")
        residual_target = load_index_map(residual_index, residual_row, PATH_FIELDS, "residual target")
        if not np.array_equal(direct_target, residual_target):
            delta = np.abs(direct_target.astype(np.float64) - residual_target.astype(np.float64))
            raise ValueError(
                f"{protocol} target arrays differ for {uid}: "
                f"max_abs={float(delta.max()):.9f} mean_abs={float(delta.mean()):.9f}"
            )
        target_exact_matches += 1
        direct_prediction = load_temperature_map(direct_predictions[uid], "direct prediction")
        residual_prediction = load_temperature_map(residual_predictions[uid], "residual prediction")
        occupancy = load_occupancy(direct_index, direct_row)
        source = None
        source_row = source_metrics.get(uid)
        if source_row is not None:
            source = load_index_map(
                residual_index,
                residual_row,
                SOURCE_FIELDS,
                "source-superposition map",
            )
        record = compute_sample_record(
            uid=uid,
            family=family,
            protocol=protocol,
            row=direct_row,
            target=direct_target,
            direct_prediction=direct_prediction,
            residual_prediction=residual_prediction,
            occupancy=occupancy,
            source=source,
            tie_tolerance_K=tie_tolerance_K,
        )
        if source_row is not None and source_row.get("mae_K") not in {None, ""}:
            expected_source_mae = float(source_row["mae_K"])
            if not math.isclose(
                float(record["source_mae_K"]),
                expected_source_mae,
                rel_tol=0.0,
                abs_tol=1.0e-4,
            ):
                raise ValueError(
                    f"{protocol} source metric mismatch for {uid}: "
                    f"recomputed={record['source_mae_K']:.8f} saved={expected_source_mae:.8f}"
                )
        validate_saved_sample_metrics(record, direct_sample_metrics[uid], residual_sample_metrics[uid])
        records.append(record)
    return records, arrays, {
        "sample_count": len(records),
        "direct_index": str(direct_index),
        "residual_index": str(residual_index),
        "target_exact_match_count": target_exact_matches,
        "prediction_units": "K",
        "source_available_count": sum(row["source_mae_K"] is not None for row in records),
    }


def compute_sample_record(
    *,
    uid: str,
    family: str,
    protocol: str,
    row: Mapping[str, str],
    target: np.ndarray,
    direct_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    occupancy: np.ndarray,
    source: np.ndarray | None,
    tie_tolerance_K: float,
) -> dict[str, Any]:
    for name, array in (
        ("target", target),
        ("direct prediction", direct_prediction),
        ("residual prediction", residual_prediction),
    ):
        validate_temperature_array(array, name)
    if target.shape != direct_prediction.shape or target.shape != residual_prediction.shape:
        raise ValueError(
            f"shape mismatch for {uid}: target={target.shape}, "
            f"direct={direct_prediction.shape}, residual={residual_prediction.shape}"
        )
    if occupancy.shape != target.shape:
        raise ValueError(f"occupancy shape mismatch for {uid}: {occupancy.shape} != {target.shape}")
    direct_error = direct_prediction.astype(np.float64) - target.astype(np.float64)
    residual_error = residual_prediction.astype(np.float64) - target.astype(np.float64)
    disagreement = direct_prediction.astype(np.float64) - residual_prediction.astype(np.float64)
    ensemble = 0.5 * (
        direct_prediction.astype(np.float64) + residual_prediction.astype(np.float64)
    )
    direct_mae = float(np.mean(np.abs(direct_error)))
    residual_mae = float(np.mean(np.abs(residual_error)))
    winner = classify_winner(direct_mae, residual_mae, tie_tolerance_K)
    boundary = np.zeros(target.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    occupied = occupancy > 0.5
    direct_peak = np.unravel_index(int(np.argmax(direct_prediction)), target.shape)
    residual_peak = np.unravel_index(int(np.argmax(residual_prediction)), target.shape)
    true_peak = np.unravel_index(int(np.argmax(target)), target.shape)
    source_mae = None
    if source is not None:
        validate_temperature_array(source, "source-superposition map")
        if source.shape != target.shape:
            raise ValueError(f"source shape mismatch for {uid}: {source.shape} != {target.shape}")
        source_mae = float(np.mean(np.abs(source.astype(np.float64) - target.astype(np.float64))))
    activity, balance, interaction = workload_regimes(row)
    return {
        "protocol": protocol,
        "split": protocol_split(protocol),
        "sample_uid": uid,
        "family_uid": family,
        "case_id": family,
        "workload_uid": str(row.get("workload_uid") or uid),
        "workload_id": str(row.get("workload_cell") or row.get("workload_stratum") or uid),
        "power_regime": str(row.get("power_regime") or ""),
        "topology_regime": str(row.get("topology_regime") or ""),
        "activity_regime": activity,
        "balance_regime": balance,
        "interaction_regime": interaction,
        "direct_mae_K": direct_mae,
        "residual_mae_K": residual_mae,
        "source_mae_K": source_mae,
        "direct_rmse_K": rmse(direct_error),
        "residual_rmse_K": rmse(residual_error),
        "direct_peak_temperature_abs_error_K": float(
            abs(float(np.max(direct_prediction)) - float(np.max(target)))
        ),
        "residual_peak_temperature_abs_error_K": float(
            abs(float(np.max(residual_prediction)) - float(np.max(target)))
        ),
        "direct_hotspot_location_error_cells": euclidean_index_distance(direct_peak, true_peak),
        "residual_hotspot_location_error_cells": euclidean_index_distance(residual_peak, true_peak),
        "direct_boundary_mae_K": masked_mae(direct_error, boundary),
        "residual_boundary_mae_K": masked_mae(residual_error, boundary),
        "direct_occupied_mae_K": masked_mae(direct_error, occupied),
        "residual_occupied_mae_K": masked_mae(residual_error, occupied),
        "direct_minus_residual_mae_K": direct_mae - residual_mae,
        "absolute_model_disagreement_mae_K": float(np.mean(np.abs(disagreement))),
        "disagreement_rmse_K": rmse(disagreement),
        "mean_signed_disagreement_K": float(np.mean(disagreement)),
        "maximum_absolute_disagreement_K": float(np.max(np.abs(disagreement))),
        "direct_wins": int(winner == "direct"),
        "residual_wins": int(winner == "residual"),
        "tied": int(winner == "tie"),
        "winner": winner,
        "average_ensemble_mae_K": float(
            np.mean(np.abs(ensemble - target.astype(np.float64)))
        ),
        "sample_oracle_mae_K": min(direct_mae, residual_mae),
    }


def aggregate_families(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = group_rows(records, ("protocol", "family_uid"))
    output: list[dict[str, Any]] = []
    for (protocol, family), rows in sorted(groups.items()):
        direct = mean_value(rows, "direct_mae_K")
        residual = mean_value(rows, "residual_mae_K")
        family_choice = "direct" if direct < residual else "residual"
        family_oracle = direct if family_choice == "direct" else residual
        output.append(
            {
                "protocol": protocol,
                "split": protocol_split(protocol),
                "family_uid": family,
                "sample_count": len(rows),
                "direct_mae_K": direct,
                "residual_mae_K": residual,
                "source_mae_K": optional_mean(rows, "source_mae_K"),
                "average_ensemble_mae_K": mean_value(rows, "average_ensemble_mae_K"),
                "sample_oracle_mae_K": mean_value(rows, "sample_oracle_mae_K"),
                "family_oracle_mae_K": family_oracle,
                "family_oracle_choice": family_choice,
                "direct_win_fraction": mean_value(rows, "direct_wins"),
                "residual_win_fraction": mean_value(rows, "residual_wins"),
                "tie_fraction": mean_value(rows, "tied"),
                "mean_direct_minus_residual_mae_K": mean_value(
                    rows, "direct_minus_residual_mae_K"
                ),
                "median_direct_minus_residual_mae_K": median_value(
                    rows, "direct_minus_residual_mae_K"
                ),
                "mean_disagreement_mae_K": mean_value(
                    rows, "absolute_model_disagreement_mae_K"
                ),
                "direct_win_margin_mean_K": conditional_margin(rows, winner="direct"),
                "direct_win_margin_median_K": conditional_margin(
                    rows, winner="direct", statistic="median"
                ),
                "direct_win_margin_p90_K": conditional_margin(
                    rows, winner="direct", statistic="p90"
                ),
                "residual_win_margin_mean_K": conditional_margin(rows, winner="residual"),
                "residual_win_margin_median_K": conditional_margin(
                    rows, winner="residual", statistic="median"
                ),
                "residual_win_margin_p90_K": conditional_margin(
                    rows, winner="residual", statistic="p90"
                ),
                "ensemble_improvement_vs_residual_K": residual
                - mean_value(rows, "average_ensemble_mae_K"),
                "sample_oracle_improvement_vs_residual_K": residual
                - mean_value(rows, "sample_oracle_mae_K"),
            }
        )
    return output


def aggregate_protocols(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = group_rows(records, ("protocol",))
    protocol_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    for (protocol,), rows in sorted(groups.items()):
        direct = mean_value(rows, "direct_mae_K")
        residual = mean_value(rows, "residual_mae_K")
        average = mean_value(rows, "average_ensemble_mae_K")
        sample_oracle = mean_value(rows, "sample_oracle_mae_K")
        family_groups = group_rows(rows, ("family_uid",))
        family_choices = {
            family: (
                "direct"
                if mean_value(family_rows, "direct_mae_K")
                < mean_value(family_rows, "residual_mae_K")
                else "residual"
            )
            for (family,), family_rows in family_groups.items()
        }
        family_oracle = float(
            np.mean(
                [
                    float(row[f"{family_choices[str(row['family_uid'])]}_mae_K"])
                    for row in rows
                ]
            )
        )
        protocol_choice = "direct" if direct < residual else "residual"
        protocol_oracle = min(direct, residual)
        base = {
            "protocol": protocol,
            "split": protocol_split(protocol),
            "sample_count": len(rows),
            "family_count": len(family_groups),
            "direct_mae_K": direct,
            "residual_mae_K": residual,
            "source_mae_K": optional_mean(rows, "source_mae_K"),
            "average_ensemble_mae_K": average,
            "sample_oracle_mae_K": sample_oracle,
            "family_oracle_mae_K": family_oracle,
            "protocol_oracle_mae_K": protocol_oracle,
            "protocol_oracle_choice": protocol_choice,
            "direct_win_fraction": mean_value(rows, "direct_wins"),
            "residual_win_fraction": mean_value(rows, "residual_wins"),
            "tie_fraction": mean_value(rows, "tied"),
            "mean_direct_minus_residual_mae_K": mean_value(
                rows, "direct_minus_residual_mae_K"
            ),
            "median_direct_minus_residual_mae_K": median_value(
                rows, "direct_minus_residual_mae_K"
            ),
            "mean_disagreement_mae_K": mean_value(
                rows, "absolute_model_disagreement_mae_K"
            ),
            "direct_win_margin_mean_K": conditional_margin(rows, winner="direct"),
            "direct_win_margin_median_K": conditional_margin(
                rows, winner="direct", statistic="median"
            ),
            "direct_win_margin_p90_K": conditional_margin(
                rows, winner="direct", statistic="p90"
            ),
            "residual_win_margin_mean_K": conditional_margin(rows, winner="residual"),
            "residual_win_margin_median_K": conditional_margin(
                rows, winner="residual", statistic="median"
            ),
            "residual_win_margin_p90_K": conditional_margin(
                rows, winner="residual", statistic="p90"
            ),
            "average_ensemble_improvement_vs_residual_K": residual - average,
            "sample_oracle_improvement_vs_residual_K": residual - sample_oracle,
            "family_oracle_improvement_vs_residual_K": residual - family_oracle,
        }
        protocol_rows.append(base)
        for selector, value, fraction_direct in (
            ("sample_level_oracle", sample_oracle, oracle_direct_fraction(rows)),
            (
                "family_level_oracle",
                family_oracle,
                float(np.mean([family_choices[str(row["family_uid"])] == "direct" for row in rows])),
            ),
            ("protocol_level_oracle", protocol_oracle, float(protocol_choice == "direct")),
        ):
            oracle_rows.append(
                {
                    "protocol": protocol,
                    "selector": selector,
                    "upper_bound_not_deployable": True,
                    "mae_K": value,
                    "improvement_over_residual_K": residual - value,
                    "improvement_over_direct_K": direct - value,
                    "fraction_routed_direct": fraction_direct,
                    "fraction_routed_residual": 1.0 - fraction_direct,
                }
            )
        for model, value in (
            ("direct_only", direct),
            ("residual_only", residual),
            ("simple_average", average),
            ("sample_oracle", sample_oracle),
        ):
            ensemble_rows.append(
                {
                    "protocol": protocol,
                    "method": model,
                    "mae_K": value,
                    "improvement_over_residual_K": residual - value,
                }
            )
    return protocol_rows, oracle_rows, ensemble_rows


def disagreement_correlations(
    records: Sequence[Mapping[str, Any]],
    *,
    high_error_threshold_K: float,
    catastrophic_error_threshold_K: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for protocol, rows in sorted(group_rows(records, ("protocol",)).items()):
        protocol_name = protocol[0]
        disagreement = array_values(rows, "absolute_model_disagreement_mae_K")
        direct = array_values(rows, "direct_mae_K")
        residual = array_values(rows, "residual_mae_K")
        advantage_abs = np.abs(direct - residual)
        for target_name, target in (
            ("direct_mae_K", direct),
            ("residual_mae_K", residual),
            ("absolute_direct_minus_residual_mae_K", advantage_abs),
        ):
            output.append(
                {
                    "protocol": protocol_name,
                    "analysis": "correlation",
                    "target": target_name,
                    "sample_count": len(rows),
                    "pearson": pearson(disagreement, target),
                    "spearman": spearman(disagreement, target),
                    "auroc": None,
                    "undefined_reason": "",
                }
            )
        labels = {
            "direct_wins": direct < residual,
            "residual_error_above_threshold": residual > high_error_threshold_K,
            "either_model_catastrophic": np.maximum(direct, residual)
            > catastrophic_error_threshold_K,
        }
        for target_name, label in labels.items():
            auc, reason = roc_auc(disagreement, label.astype(np.int64))
            output.append(
                {
                    "protocol": protocol_name,
                    "analysis": "auroc",
                    "target": target_name,
                    "sample_count": len(rows),
                    "pearson": None,
                    "spearman": None,
                    "auroc": auc,
                    "undefined_reason": reason or "",
                }
            )
    return output


def reliability_bins(
    records: Sequence[Mapping[str, Any]],
    bin_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (protocol,), rows in sorted(group_rows(records, ("protocol",)).items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["absolute_model_disagreement_mae_K"]),
                str(row["sample_uid"]),
            ),
        )
        for bin_index, indices in enumerate(np.array_split(np.arange(len(ordered)), bin_count)):
            if len(indices) == 0:
                continue
            current = [ordered[int(index)] for index in indices]
            residual = mean_value(current, "residual_mae_K")
            oracle = mean_value(current, "sample_oracle_mae_K")
            output.append(
                {
                    "protocol": protocol,
                    "quantile_bin": bin_index + 1,
                    "quantile_bin_count": bin_count,
                    "sample_count": len(current),
                    "disagreement_min_K": min(
                        float(row["absolute_model_disagreement_mae_K"]) for row in current
                    ),
                    "disagreement_max_K": max(
                        float(row["absolute_model_disagreement_mae_K"]) for row in current
                    ),
                    "disagreement_mean_K": mean_value(
                        current, "absolute_model_disagreement_mae_K"
                    ),
                    "direct_mae_K": mean_value(current, "direct_mae_K"),
                    "residual_mae_K": residual,
                    "direct_win_fraction": mean_value(current, "direct_wins"),
                    "oracle_gain_over_residual_K": residual - oracle,
                }
            )
    return output


def aggregate_regimes(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in (
        "power_regime",
        "topology_regime",
        "activity_regime",
        "balance_regime",
        "interaction_regime",
    ):
        groups = group_rows(
            [row for row in records if str(row.get(field) or "")],
            ("protocol", field),
        )
        for (protocol, value), rows in sorted(groups.items()):
            output.append(
                {
                    "protocol": protocol,
                    "regime_field": field,
                    "regime": value,
                    "sample_count": len(rows),
                    "direct_win_fraction": mean_value(rows, "direct_wins"),
                    "residual_win_fraction": mean_value(rows, "residual_wins"),
                    "tie_fraction": mean_value(rows, "tied"),
                    "direct_mae_K": mean_value(rows, "direct_mae_K"),
                    "residual_mae_K": mean_value(rows, "residual_mae_K"),
                    "mean_direct_minus_residual_mae_K": mean_value(
                        rows, "direct_minus_residual_mae_K"
                    ),
                }
            )
    return output


def analyze_descriptors(
    family_rows: Sequence[Mapping[str, Any]],
    *,
    family_descriptor_csv: str | Path | None,
    family_cluster_csv: str | Path | None,
) -> dict[str, Any]:
    if family_descriptor_csv is None:
        return {
            "correlations": [],
            "test_predictions": [],
            "summary": {
                "status": "skipped",
                "reason": "--family-descriptor-csv was not supplied",
            },
            "predictor": None,
        }
    descriptor_path = Path(family_descriptor_csv).expanduser().resolve()
    descriptor_rows = strict_family_map(read_csv(descriptor_path), "family descriptors")
    cluster_rows = (
        strict_family_map(
            read_csv(Path(family_cluster_csv).expanduser().resolve()),
            "family clusters",
        )
        if family_cluster_csv is not None
        else {}
    )
    family_metrics = {str(row["family_uid"]): row for row in family_rows}
    missing = sorted(set(family_metrics) - set(descriptor_rows))
    if missing:
        raise ValueError(f"family descriptor table is missing evaluated families: {missing}")
    features = numeric_descriptor_names(descriptor_rows)
    if not features:
        raise ValueError("family descriptor table has no complete finite numeric descriptors")
    cluster_values = sorted(
        {
            str(row.get("cluster_id"))
            for row in cluster_rows.values()
            if row.get("cluster_id") not in {None, ""}
        }
    )
    feature_names = list(features) + [f"cluster_{value}" for value in cluster_values]
    family_order = sorted(family_metrics)
    matrix = np.asarray(
        [
            descriptor_vector(
                descriptor_rows[family],
                features,
                cluster_rows.get(family),
                cluster_values,
            )
            for family in family_order
        ],
        dtype=np.float64,
    )
    targets = {
        "direct_minus_residual_mae_K": np.asarray(
            [
                float(family_metrics[family]["mean_direct_minus_residual_mae_K"])
                for family in family_order
            ]
        ),
        "direct_win_fraction": np.asarray(
            [float(family_metrics[family]["direct_win_fraction"]) for family in family_order]
        ),
        "mean_disagreement_mae_K": np.asarray(
            [float(family_metrics[family]["mean_disagreement_mae_K"]) for family in family_order]
        ),
        "residual_mae_K": np.asarray(
            [float(family_metrics[family]["residual_mae_K"]) for family in family_order]
        ),
        "direct_mae_K": np.asarray(
            [float(family_metrics[family]["direct_mae_K"]) for family in family_order]
        ),
    }
    correlations: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(feature_names):
        for target_name, target in targets.items():
            correlations.append(
                {
                    "feature": feature,
                    "target": target_name,
                    "family_count": len(family_order),
                    "pearson": pearson(matrix[:, feature_index], target),
                    "spearman": spearman(matrix[:, feature_index], target),
                    "exploratory_small_family_sample": True,
                }
            )

    fit_families = [
        family
        for family in family_order
        if str(family_metrics[family]["protocol"]) != "primary_test_families"
    ]
    test_families = [
        family
        for family in family_order
        if str(family_metrics[family]["protocol"]) == "primary_test_families"
    ]
    fit_indices = np.asarray([family_order.index(family) for family in fit_families], dtype=int)
    test_indices = np.asarray([family_order.index(family) for family in test_families], dtype=int)
    fit_x = matrix[fit_indices]
    mean, std = fit_standardizer(fit_x)
    standardized_fit = apply_standardizer(fit_x, mean, std)
    fit_delta = targets["direct_minus_residual_mae_K"][fit_indices]
    fit_labels = (fit_delta < 0.0).astype(np.int64)
    logistic = fit_logistic(standardized_fit, fit_labels)
    ridge = fit_ridge(standardized_fit, fit_delta)
    loo_probabilities: list[float] = []
    loo_labels: list[int] = []
    loo_ridge: list[float] = []
    for heldout in range(len(fit_families)):
        keep = np.asarray([index != heldout for index in range(len(fit_families))])
        loo_mean, loo_std = fit_standardizer(fit_x[keep])
        x_train = apply_standardizer(fit_x[keep], loo_mean, loo_std)
        x_holdout = apply_standardizer(fit_x[heldout : heldout + 1], loo_mean, loo_std)
        loo_probabilities.append(float(predict_logistic(fit_logistic(x_train, fit_labels[keep]), x_holdout)[0]))
        loo_labels.append(int(fit_labels[heldout]))
        loo_ridge.append(float(predict_ridge(fit_ridge(x_train, fit_delta[keep]), x_holdout)[0]))

    test_predictions: list[dict[str, Any]] = []
    if len(test_indices):
        standardized_test = apply_standardizer(matrix[test_indices], mean, std)
        probabilities = predict_logistic(logistic, standardized_test)
        ridge_predictions = predict_ridge(ridge, standardized_test)
        for offset, family in enumerate(test_families):
            metric = family_metrics[family]
            test_predictions.append(
                {
                    "family_uid": family,
                    "cluster_id": cluster_rows.get(family, {}).get("cluster_id", ""),
                    "predicted_direct_preference_probability": float(probabilities[offset]),
                    "predicted_direct_minus_residual_mae_K": float(ridge_predictions[offset]),
                    "selected_model": "direct" if probabilities[offset] >= 0.5 else "residual",
                    "actual_direct_minus_residual_mae_K_analysis_only": float(
                        metric["mean_direct_minus_residual_mae_K"]
                    ),
                    "actual_preferred_model_analysis_only": (
                        "direct"
                        if float(metric["mean_direct_minus_residual_mae_K"]) < 0.0
                        else "residual"
                    ),
                }
            )
    loo_accuracy = float(
        np.mean((np.asarray(loo_probabilities) >= 0.5) == np.asarray(loo_labels))
    )
    loo_ridge_rmse = rmse(np.asarray(loo_ridge) - fit_delta)
    top_positive = top_correlations(
        correlations, target="direct_minus_residual_mae_K", positive=True
    )
    top_negative = top_correlations(
        correlations, target="direct_minus_residual_mae_K", positive=False
    )
    f044 = next((row for row in test_predictions if row["family_uid"] == "f044"), None)
    return {
        "correlations": correlations,
        "test_predictions": test_predictions,
        "predictor": {
            "family_selection": {
                row["family_uid"]: row["selected_model"] for row in test_predictions
            }
        },
        "summary": {
            "status": "completed",
            "descriptor_csv": str(descriptor_path),
            "cluster_csv": str(Path(family_cluster_csv).expanduser().resolve())
            if family_cluster_csv is not None
            else None,
            "numeric_descriptor_count": len(features),
            "model_feature_count": len(feature_names),
            "feature_names": feature_names,
            "fit_family_count": len(fit_families),
            "test_family_count": len(test_families),
            "fit_families": fit_families,
            "test_families": test_families,
            "standardization_fit_scope": "known-family sample-test and primary validation families only",
            "logistic_leave_one_family_out_accuracy": loo_accuracy,
            "ridge_leave_one_family_out_rmse_K": loo_ridge_rmse,
            "top_positive_direct_minus_residual_correlations": top_positive,
            "top_negative_direct_minus_residual_correlations": top_negative,
            "f044_prediction": f044,
            "small_sample_warning": (
                "Family-level descriptor associations and leave-one-family-out results are "
                "exploratory because only 45 non-test families are available for fitting."
            ),
        },
    }


def analyze_routing_proxies(
    records: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    *,
    descriptor_result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_protocol = group_rows(records, ("protocol",))
    validation = by_protocol.get(("primary_validation_families",), [])
    test = by_protocol.get(("primary_test_families",), [])
    if not validation or not test:
        raise ValueError("routing proxy analysis requires primary validation and test protocols")
    threshold = select_disagreement_threshold(validation)
    validation_direct = mean_value(validation, "direct_mae_K")
    validation_residual = mean_value(validation, "residual_mae_K")
    validation_best = "direct" if validation_direct < validation_residual else "residual"

    routing_rows: list[dict[str, Any]] = []
    for protocol, rows in (
        ("primary_validation_families", validation),
        ("primary_test_families", test),
    ):
        residual = mean_value(rows, "residual_mae_K")
        sample_oracle = mean_value(rows, "sample_oracle_mae_K")
        methods = {
            "always_residual": (
                residual,
                0.0,
                "fixed baseline",
            ),
            "always_direct": (
                mean_value(rows, "direct_mae_K"),
                1.0,
                "fixed baseline",
            ),
            "simple_average": (
                mean_value(rows, "average_ensemble_mae_K"),
                None,
                "deployable ensemble requiring both model forwards",
            ),
            "validation_best_single_model": (
                mean_value(rows, f"{validation_best}_mae_K"),
                float(validation_best == "direct"),
                f"selected on validation only: {validation_best}",
            ),
        }
        threshold_mae, threshold_direct_fraction = apply_threshold_route(
            rows,
            threshold=float(threshold["threshold_K"]),
            high_disagreement_model=str(threshold["high_disagreement_model"]),
        )
        methods["validation_tuned_disagreement_threshold"] = (
            threshold_mae,
            threshold_direct_fraction,
            (
                f"threshold={threshold['threshold_K']:.9f} K; "
                f"high disagreement -> {threshold['high_disagreement_model']}"
            ),
        )
        family_selection = (
            descriptor_result.get("predictor", {}) or {}
        ).get("family_selection", {})
        if family_selection and protocol == "primary_test_families":
            descriptor_mae, descriptor_fraction = apply_family_route(rows, family_selection)
            methods["descriptor_logistic_routing"] = (
                descriptor_mae,
                descriptor_fraction,
                "fit on known-family sample-test plus validation families; no test labels",
            )
        for method, (mae, fraction_direct, note) in methods.items():
            headroom = residual - sample_oracle
            routing_rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "mae_K": mae,
                    "improvement_over_always_residual_K": residual - mae,
                    "fraction_routed_direct": fraction_direct,
                    "fraction_oracle_headroom_recovered": (
                        (residual - mae) / headroom if headroom > NUMERIC_EPS else None
                    ),
                    "oracle_reference_mae_K": sample_oracle,
                    "selection_note": note,
                    "test_labels_used_for_tuning": False,
                }
            )
    test_predictions = list(descriptor_result.get("test_predictions", []))
    test_family_metrics = {
        str(row["family_uid"]): row
        for row in family_rows
        if str(row["protocol"]) == "primary_test_families"
    }
    for row in test_predictions:
        metric = test_family_metrics[str(row["family_uid"])]
        row.update(
            {
                "direct_mae_K_analysis_only": metric["direct_mae_K"],
                "residual_mae_K_analysis_only": metric["residual_mae_K"],
                "direct_win_fraction_analysis_only": metric["direct_win_fraction"],
            }
        )
    return routing_rows, test_predictions, threshold


def select_disagreement_threshold(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    disagreement = np.asarray(
        [float(row["absolute_model_disagreement_mae_K"]) for row in rows],
        dtype=np.float64,
    )
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([-np.inf, np.inf]),
                np.quantile(disagreement, np.linspace(0.0, 1.0, 101)),
            )
        )
    )
    best: tuple[float, int, float, str] | None = None
    for direction_index, high_model in enumerate(("direct", "residual")):
        for threshold in candidates:
            mae, fraction_direct = apply_threshold_route(
                rows,
                threshold=float(threshold),
                high_disagreement_model=high_model,
            )
            candidate = (mae, direction_index, float(threshold), high_model)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    mae, _direction_index, threshold, high_model = best
    return {
        "fit_protocol": "primary_validation_families",
        "threshold_K": threshold,
        "high_disagreement_model": high_model,
        "validation_mae_K": mae,
        "candidate_count": int(len(candidates) * 2),
        "test_labels_used": False,
    }


def apply_threshold_route(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    high_disagreement_model: str,
) -> tuple[float, float]:
    if high_disagreement_model not in {"direct", "residual"}:
        raise ValueError(f"unsupported high-disagreement route: {high_disagreement_model}")
    low_model = "residual" if high_disagreement_model == "direct" else "direct"
    selected_errors = []
    direct_count = 0
    for row in rows:
        high = float(row["absolute_model_disagreement_mae_K"]) >= threshold
        model = high_disagreement_model if high else low_model
        direct_count += int(model == "direct")
        selected_errors.append(float(row[f"{model}_mae_K"]))
    return float(np.mean(selected_errors)), direct_count / max(len(rows), 1)


def apply_family_route(
    rows: Sequence[Mapping[str, Any]],
    family_selection: Mapping[str, str],
) -> tuple[float, float]:
    selected = []
    direct_count = 0
    for row in rows:
        family = str(row["family_uid"])
        if family not in family_selection:
            raise ValueError(f"descriptor routing has no prediction for test family {family}")
        model = str(family_selection[family])
        if model not in {"direct", "residual"}:
            raise ValueError(f"invalid descriptor route for {family}: {model}")
        selected.append(float(row[f"{model}_mae_K"]))
        direct_count += int(model == "direct")
    return float(np.mean(selected)), direct_count / max(len(rows), 1)


def make_recommendation(
    protocol_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    regime_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    test = next(row for row in protocol_rows if row["protocol"] == "primary_test_families")
    oracle_gain = float(test["sample_oracle_improvement_vs_residual_K"])
    test_routes = {
        str(row["method"]): row
        for row in routing_rows
        if row["protocol"] == "primary_test_families"
    }
    deployable_gains = [
        float(row["improvement_over_always_residual_K"])
        for name, row in test_routes.items()
        if name
        in {
            "validation_tuned_disagreement_threshold",
            "descriptor_logistic_routing",
            "simple_average",
        }
    ]
    best_proxy_gain = max(deployable_gains, default=-float("inf"))
    recovered = best_proxy_gain / oracle_gain if oracle_gain > NUMERIC_EPS else 0.0
    test_family_rows = [
        row
        for row in family_rows
        if row["protocol"] == "primary_test_families"
    ]
    test_direct_family_wins = sum(
        str(row["family_oracle_choice"]) == "direct" for row in test_family_rows
    )
    test_direct_regime_wins = sum(
        float(row["mean_direct_minus_residual_mae_K"]) < 0.0
        for row in regime_rows
        if row["protocol"] == "primary_test_families"
    )
    if oracle_gain < 0.05:
        code, label = "A", "No routing work justified"
        reason = "Sample-level oracle improvement over residual is below 0.05 K."
    elif best_proxy_gain >= 0.05:
        descriptor_gain = float(
            test_routes.get("descriptor_logistic_routing", {}).get(
                "improvement_over_always_residual_K", -float("inf")
            )
        )
        disagreement_gain = float(
            test_routes.get("validation_tuned_disagreement_threshold", {}).get(
                "improvement_over_always_residual_K", -float("inf")
            )
        )
        if (
            descriptor_gain >= 0.05
            and disagreement_gain >= 0.05
            and test_direct_family_wins >= 2
            and test_direct_regime_wins >= 2
        ):
            code, label = "D", "Implement a learned uncertainty-aware mixture-of-experts gate"
            reason = (
                "Multiple leakage-safe proxies improve primary test MAE by at least 0.05 K."
            )
        else:
            code, label = "C", "Implement a simple validation-tuned selector"
            reason = (
                "At least one validation-derived proxy improves primary test MAE by at least 0.05 K."
            )
    elif oracle_gain <= 0.15 and recovered < 0.25:
        code, label = "B", "Keep as analysis only"
        reason = (
            "Oracle headroom is modest and deployable proxies recover less than 25% of it."
        )
    else:
        code, label = "B", "Keep as analysis only"
        reason = "Oracle complementarity exists, but leakage-safe routing evidence is not yet strong."
    return {
        "code": code,
        "label": label,
        "reason": reason,
        "sample_oracle_test_improvement_over_residual_K": oracle_gain,
        "best_leakage_safe_proxy_test_improvement_K": best_proxy_gain,
        "fraction_oracle_headroom_recovered": recovered,
        "primary_test_direct_family_win_count": test_direct_family_wins,
        "primary_test_direct_regime_win_count": test_direct_regime_wins,
        "thresholds": {
            "A": "oracle test improvement < 0.05 K",
            "B": "oracle gain 0.05-0.15 K and proxies recover <25%, or proxy evidence is weak",
            "C": "validation-tuned proxy improves test by >=0.05 K",
            "D": "multiple leakage-safe proxies improve test by >=0.05 K",
        },
    }


def validate_prediction_contract(metrics: Mapping[str, Any], *, expected: str) -> None:
    model = metrics.get("model") or {}
    mode = str(model.get("prediction_mode") or model.get("config", {}).get("prediction_mode") or "")
    if expected == "direct":
        if mode not in {"direct_temperature", "direct_temperature_source_conditioned"}:
            raise ValueError(f"expected direct-temperature evaluation, found mode={mode!r}")
        target_name = str(
            model.get("config", {}).get("target_name")
            or (model.get("direct_temperature_target_normalization") or {}).get("target_name")
            or ""
        )
        if target_name != "absolute_temperature_K":
            raise ValueError(
                "direct prediction units are not documented as absolute Kelvin: "
                f"target_name={target_name!r}"
            )
        normalization_mode = str(
            (model.get("direct_temperature_target_normalization") or {}).get("mode")
            or model.get("config", {}).get("target_normalization_mode")
            or ""
        )
        if normalization_mode != "train_standard":
            raise ValueError(
                "this analysis requires the normalized direct-temperature checkpoint: "
                f"target_normalization_mode={normalization_mode!r}"
            )
    else:
        if mode not in {"", "residual", "residual_decomposed"}:
            raise ValueError(f"expected residual evaluation, found mode={mode!r}")
        if "cnn_final_temperature" not in metrics:
            raise ValueError("residual evaluation lacks reconstructed Kelvin temperature metrics")
        config = model.get("config") or {}
        architecture = str(config.get("architecture") or "")
        physics_input = str(model.get("physics_input_mode") or config.get("physics_input_mode") or "")
        mean_head = str(model.get("mean_head_mode") or config.get("mean_head_mode") or "")
        if "decomposed_feature_fusion" not in architecture:
            raise ValueError(f"expected canonical decomposed feature-fusion residual model, got {architecture!r}")
        if physics_input != "source_superposition_v1":
            raise ValueError(f"expected source_superposition_v1 residual input, got {physics_input!r}")
        if mean_head != "residual_resistance":
            raise ValueError(f"expected residual_resistance mean head, got {mean_head!r}")
    final = metrics.get("cnn_final_temperature") or {}
    if not {"mae_K", "rmse_K"}.issubset(final):
        raise ValueError("evaluation metrics do not document final prediction errors in Kelvin")


def resolve_evaluation_index(metrics: Mapping[str, Any]) -> Path:
    value = metrics.get("index")
    if not value:
        raise ValueError("evaluation metrics are missing the immutable index path")
    path = Path(str(value)).expanduser()
    if path.is_file():
        return path.resolve()
    root_value = os.environ.get("CHIPTHERM_V2_DATA_ROOT")
    if root_value:
        root = Path(root_value).expanduser().resolve()
        parts = path.parts
        for anchor in ("derived", "canonical"):
            if anchor in parts:
                candidate = root.joinpath(*parts[parts.index(anchor) :])
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(
        f"evaluation index is unavailable: logical path={value!r}; "
        f"CHIPTHERM_V2_DATA_ROOT={root_value!r}"
    )


def prediction_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"cached prediction directory is missing: {root}")
    output: dict[str, Path] = {}
    for path in sorted(root.rglob("*_tpred.npy")):
        uid = path.name[: -len("_tpred.npy")]
        if uid in output:
            raise ValueError(f"duplicate cached prediction UID {uid}: {output[uid]} and {path}")
        output[uid] = path.resolve()
    if not output:
        raise ValueError(f"no cached predictions found under {root}")
    return output


def strict_uid_map(
    rows: Sequence[Mapping[str, str]],
    label: str,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = str(row.get("sample_uid") or "")
        if not uid:
            raise ValueError(f"{label} contains a row without sample_uid")
        if uid in output:
            raise ValueError(f"{label} contains duplicate sample_uid={uid}")
        output[uid] = dict(row)
    return output


def strict_family_map(
    rows: Sequence[Mapping[str, str]],
    label: str,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        family = str(row.get("family_uid") or "")
        if not family:
            raise ValueError(f"{label} contains a row without family_uid")
        if family in output:
            raise ValueError(f"{label} contains duplicate family_uid={family}")
        output[family] = dict(row)
    return output


def require_same_uids(
    expected: set[str],
    actual: set[str],
    protocol: str,
    label: str,
) -> None:
    if expected != actual:
        raise ValueError(
            f"{protocol} {label} UID mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )


def load_index_map(
    index_path: Path,
    row: Mapping[str, str],
    fields: Sequence[str],
    label: str,
) -> np.ndarray:
    value = next((str(row[field]) for field in fields if row.get(field)), "")
    if not value:
        raise ValueError(
            f"{label} path is absent for {row.get('sample_uid')}; available={sorted(row)}"
        )
    path = resolve_data_path(index_path, value)
    return load_temperature_map(path, label)


def load_occupancy(index_path: Path, row: Mapping[str, str]) -> np.ndarray:
    value = row.get("x_path")
    if not value:
        raise ValueError(f"x_path is absent for {row.get('sample_uid')}")
    path = resolve_data_path(index_path, str(value))
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError(f"expected X[C,H,W] with occupancy channel at {path}, got {array.shape}")
    occupancy = np.asarray(array[1], dtype=np.float32)
    if not np.isfinite(occupancy).all():
        raise ValueError(f"occupancy contains non-finite values: {path}")
    return occupancy


def resolve_data_path(index_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        root_value = os.environ.get("CHIPTHERM_V2_DATA_ROOT")
        if root_value:
            parts = path.parts
            for anchor in ("derived", "canonical"):
                if anchor in parts:
                    candidate = Path(root_value).expanduser().resolve().joinpath(
                        *parts[parts.index(anchor) :]
                    )
                    if candidate.is_file():
                        return candidate
        raise FileNotFoundError(path)
    root = discover_data_root(index_path)
    candidates = [root / path] if root is not None else []
    candidates.extend((index_path.parent / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve logical path={value!r} from index={index_path}; "
        f"declared_root={root}"
    )


def discover_data_root(index_path: Path) -> Path | None:
    for candidate in (index_path.parent, *index_path.parents):
        if (candidate / ".chiptherm_data_root.json").is_file():
            return candidate.resolve()
    return None


def load_temperature_map(path: Path, label: str) -> np.ndarray:
    array = np.asarray(np.load(path, allow_pickle=False))
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    validate_temperature_array(array, f"{label} at {path}")
    return np.asarray(array, dtype=np.float32)


def validate_temperature_array(array: np.ndarray, label: str) -> None:
    if array.ndim != 2 or array.shape != (64, 64):
        raise ValueError(f"{label} must have shape (64,64), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN/Inf")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < 150.0 or maximum > 2000.0:
        raise ValueError(
            f"{label} is not a plausible absolute-Kelvin map: min={minimum}, max={maximum}"
        )


def validate_saved_sample_metrics(
    record: Mapping[str, Any],
    direct: Mapping[str, str],
    residual: Mapping[str, str],
) -> None:
    for prefix, saved in (("direct", direct), ("residual", residual)):
        for key in ("mae_K", "rmse_K"):
            saved_value = float(saved[key])
            computed = float(record[f"{prefix}_{key}"])
            if not math.isclose(saved_value, computed, rel_tol=0.0, abs_tol=2.0e-4):
                raise ValueError(
                    f"saved {prefix} metric mismatch for {record['sample_uid']} {key}: "
                    f"saved={saved_value:.8f} recomputed={computed:.8f}"
                )


def load_optional_source_metrics(
    root_value: str | Path | None,
) -> dict[str, dict[str, str]]:
    if root_value is None:
        return {}
    root = Path(root_value).expanduser().resolve()
    path = root / "base_quality_by_sample.csv" if root.is_dir() else root
    return strict_uid_map(read_csv(path), "source-superposition sample metrics")


def classify_winner(direct_mae: float, residual_mae: float, tolerance: float) -> str:
    delta = direct_mae - residual_mae
    if abs(delta) <= tolerance:
        return "tie"
    return "direct" if delta < 0.0 else "residual"


def workload_regimes(row: Mapping[str, str]) -> tuple[str, str, str]:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("workload_uid", "workload_cell", "workload_stratum", "topology_regime")
    )
    activity = "sparse_activity" if "sparse" in text else "dense_activity" if "dense" in text else ""
    if "single" in text and "dominant" in text:
        balance = "single_dominant"
    elif "balanced" in text:
        balance = "balanced"
    elif "skewed" in text or "dominant" in text:
        balance = "skewed_or_dominant"
    else:
        balance = ""
    interaction = (
        "interacting_source"
        if any(token in text for token in ("interacting", "cluster", "two_source", "three_source"))
        else ""
    )
    return activity, balance, interaction


def numeric_descriptor_names(
    rows: Mapping[str, Mapping[str, str]],
) -> list[str]:
    excluded = {
        "family_uid",
        "split",
        "primary_category",
        "placement_style",
        "secondary_tags",
        "family_config_path",
        "substrate",
        "material_and_cooling_variant",
    }
    names = sorted({key for row in rows.values() for key in row} - excluded)
    output = []
    for name in names:
        values = []
        valid = True
        for row in rows.values():
            value = row.get(name)
            if value in {None, ""}:
                valid = False
                break
            try:
                number = float(value)
            except (TypeError, ValueError):
                valid = False
                break
            if not math.isfinite(number):
                valid = False
                break
            values.append(number)
        if valid and values:
            output.append(name)
    return output


def descriptor_vector(
    row: Mapping[str, str],
    names: Sequence[str],
    cluster_row: Mapping[str, str] | None,
    cluster_values: Sequence[str],
) -> list[float]:
    values = [float(row[name]) for name in names]
    cluster = str((cluster_row or {}).get("cluster_id") or "")
    values.extend(float(cluster == value) for value in cluster_values)
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError(f"non-finite family descriptor vector for {row.get('family_uid')}")
    return values


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"cannot fit standardizer on shape {x.shape}")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1.0e-12, std, 1.0)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) / std


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    regularization: float = 1.0,
    iterations: int = 600,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"constant_probability": float(np.mean(y)), "weights": None}
    design = np.column_stack((np.ones(x.shape[0]), x))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - y) / max(len(y), 1)
        gradient[1:] += float(regularization) * weights[1:] / max(len(y), 1)
        step = float(learning_rate) * gradient
        weights -= step
        if float(np.linalg.norm(step)) < 1.0e-9:
            break
    return {"constant_probability": None, "weights": weights}


def predict_logistic(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    constant = model.get("constant_probability")
    if constant is not None:
        return np.full(x.shape[0], float(constant), dtype=np.float64)
    design = np.column_stack((np.ones(x.shape[0]), np.asarray(x, dtype=np.float64)))
    logits = np.clip(design @ np.asarray(model["weights"]), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    regularization: float = 10.0,
) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    feature_mean = np.mean(features, axis=0)
    target_mean = float(np.mean(target))
    centered_x = features - feature_mean
    centered_y = target - target_mean
    dual = np.linalg.solve(
        centered_x @ centered_x.T
        + float(regularization) * np.eye(centered_x.shape[0], dtype=np.float64),
        centered_y,
    )
    coefficients = centered_x.T @ dual
    intercept = target_mean - float(feature_mean @ coefficients)
    return np.concatenate(([intercept], coefficients))


def predict_ridge(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(x.shape[0]), np.asarray(x, dtype=np.float64)))
    return design @ weights


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, str | None]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == 1
    negative = labels == 0
    if not np.any(positive):
        return None, "undefined: no positive samples"
    if not np.any(negative):
        return None, "undefined: no negative samples"
    ranks = average_ranks(scores)
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(negative))
    auc = (
        float(np.sum(ranks[positive]))
        - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    return float(auc), None


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= NUMERIC_EPS or np.std(y) <= NUMERIC_EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(average_ranks(np.asarray(x)), average_ranks(np.asarray(y)))


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        rank = 0.5 * ((position + 1) + end)
        ranks[order[position:end]] = rank
        position = end
    return ranks


def top_correlations(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    positive: bool,
    count: int = 5,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["target"] == target and row.get("spearman") is not None
    ]
    selected.sort(key=lambda row: float(row["spearman"]), reverse=positive)
    return [
        {"feature": row["feature"], "spearman": row["spearman"], "pearson": row["pearson"]}
        for row in selected[:count]
    ]


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
) -> None:
    protocols = {row["protocol"]: row for row in summary["protocol_summary"]}
    test = protocols["primary_test_families"]
    validation = protocols["primary_validation_families"]
    families = {
        str(row["family_uid"]): row
        for row in family_rows
        if row["protocol"] == "primary_test_families"
    }
    direct_family_wins = [
        family for family, row in families.items() if row["family_oracle_choice"] == "direct"
    ]
    f044 = families.get("f044")
    threshold_route = next(
        (
            row
            for row in summary["routing_proxies"]
            if row["protocol"] == "primary_test_families"
            and row["method"] == "validation_tuned_disagreement_threshold"
        ),
        None,
    )
    descriptor = summary["descriptor_analysis"]
    recommendation = summary["recommendation"]
    test_regimes = sorted(
        (
            row
            for row in summary["workload_regime_summary"]
            if row["protocol"] == "primary_test_families"
        ),
        key=lambda row: float(row["mean_direct_minus_residual_mae_K"]),
    )
    strongest_direct_regimes = [
        f"{row['regime_field']}={row['regime']} ({float(row['mean_direct_minus_residual_mae_K']):.4f} K)"
        for row in test_regimes
        if float(row["mean_direct_minus_residual_mae_K"]) < 0.0
    ][:5]
    lines = [
        "# Direct vs Decomposed Complementarity",
        "",
        "This is an offline oracle analysis. Oracle selectors use target labels and are not deployable.",
        "All cached predictions were matched by `sample_uid`; no checkpoint inference or fitting on primary test labels was performed.",
        "",
        "## Main Results",
        "",
        "| Protocol | Direct MAE K | Residual MAE K | Average K | Sample oracle K | Family oracle K | Direct win fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        row = protocols[protocol]
        lines.append(
            f"| {protocol} | {fmt(row['direct_mae_K'])} | {fmt(row['residual_mae_K'])} | "
            f"{fmt(row['average_ensemble_mae_K'])} | {fmt(row['sample_oracle_mae_K'])} | "
            f"{fmt(row['family_oracle_mae_K'])} | {fmt(row['direct_win_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Required Interpretation",
            "",
            f"1. **Overall direct wins:** {fmt(test['direct_win_fraction'])} of primary-test samples "
            f"({fmt(validation['direct_win_fraction'])} on validation).",
            f"2. **Family wins:** direct is the lower-MAE family model for "
            f"{len(direct_family_wins)}/{len(families)} test families: {', '.join(direct_family_wins) or 'none'}.",
            f"3. **Is f044 the only family win?** "
            f"{'Yes.' if direct_family_wins == ['f044'] else 'No.'}",
            f"4. **f044 consistency:** "
            + (
                f"direct wins {fmt(f044['direct_win_fraction'])} of f044 samples; "
                f"mean direct-minus-residual MAE is {fmt(f044['mean_direct_minus_residual_mae_K'])} K."
                if f044
                else "f044 is absent from the primary-test evaluation."
            ),
            "   Strongest direct-favoring workload regimes: "
            + (", ".join(strongest_direct_regimes) if strongest_direct_regimes else "none."),
            f"5. **Sample-oracle headroom:** "
            f"{fmt(test['sample_oracle_improvement_vs_residual_K'])} K over always residual.",
            f"6. **Family-oracle headroom:** "
            f"{fmt(test['family_oracle_improvement_vs_residual_K'])} K over always residual.",
            f"7. **Simple averaging:** "
            f"{'helps' if test['average_ensemble_improvement_vs_residual_K'] > 0 else 'regresses'} "
            f"by {fmt(abs(test['average_ensemble_improvement_vs_residual_K']))} K versus residual.",
            "8. **Disagreement as error signal:** see `disagreement_correlation_summary.csv`; "
            "correlations and AUROCs are reported independently for each protocol.",
            f"9. **Validation-tuned disagreement routing:** "
            + (
                f"test improvement is {fmt(threshold_route['improvement_over_always_residual_K'])} K."
                if threshold_route
                else "not available."
            ),
            f"10. **Static OOD descriptors:** {descriptor.get('status')}. "
            f"LOFO logistic accuracy={fmt(descriptor.get('logistic_leave_one_family_out_accuracy'))}; "
            f"f044 predicted preference={descriptor.get('f044_prediction')}.",
            "11. **Learned-selector evidence:** determined from oracle headroom and leakage-safe proxy recovery, "
            "not from f044 alone.",
            f"12. **Paper tradeoff:** {recommendation['label']}. {recommendation['reason']}",
            "",
            "## Recommendation",
            "",
            f"**{recommendation['code']}. {recommendation['label']}**",
            "",
            recommendation["reason"],
            "",
            "Threshold policy:",
        ]
    )
    for code, rule in recommendation["thresholds"].items():
        lines.append(f"- {code}: {rule}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    protocol_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
    reliability_rows: Sequence[Mapping[str, Any]],
) -> None:
    names = (
        "per_family_direct_win_fraction.png",
        "per_family_direct_minus_residual_mae.png",
        "disagreement_vs_direct_error.png",
        "disagreement_vs_residual_error.png",
        "disagreement_vs_model_advantage.png",
        "oracle_gain_by_family.png",
        "ensemble_vs_oracle_mae.png",
        "disagreement_reliability_curve.png",
        "routing_proxy_comparison.png",
        "test_family_model_comparison.png",
    )
    try:
        import matplotlib.pyplot as plt
    except Exception:
        for name in names:
            write_placeholder_png(out_dir / name)
        return

    ordered_families = sorted(family_rows, key=lambda row: (row["protocol"], row["family_uid"]))
    labels = [f"{row['family_uid']}\n{short_protocol(str(row['protocol']))}" for row in ordered_families]
    bar_plot(
        plt,
        out_dir / names[0],
        labels,
        [float(row["direct_win_fraction"]) for row in ordered_families],
        "Direct win fraction",
    )
    bar_plot(
        plt,
        out_dir / names[1],
        labels,
        [float(row["mean_direct_minus_residual_mae_K"]) for row in ordered_families],
        "Direct minus residual MAE (K)",
    )
    disagreement = array_values(records, "absolute_model_disagreement_mae_K")
    scatter_plot(
        plt,
        out_dir / names[2],
        disagreement,
        array_values(records, "direct_mae_K"),
        "Model disagreement MAE (K)",
        "Direct MAE (K)",
    )
    scatter_plot(
        plt,
        out_dir / names[3],
        disagreement,
        array_values(records, "residual_mae_K"),
        "Model disagreement MAE (K)",
        "Residual MAE (K)",
    )
    scatter_plot(
        plt,
        out_dir / names[4],
        disagreement,
        array_values(records, "direct_minus_residual_mae_K"),
        "Model disagreement MAE (K)",
        "Direct minus residual MAE (K)",
    )
    bar_plot(
        plt,
        out_dir / names[5],
        labels,
        [float(row["sample_oracle_improvement_vs_residual_K"]) for row in ordered_families],
        "Sample-oracle gain over residual (K)",
    )
    protocol_labels = [short_protocol(str(row["protocol"])) for row in protocol_rows]
    grouped_bar_plot(
        plt,
        out_dir / names[6],
        protocol_labels,
        {
            "direct": [float(row["direct_mae_K"]) for row in protocol_rows],
            "residual": [float(row["residual_mae_K"]) for row in protocol_rows],
            "average": [float(row["average_ensemble_mae_K"]) for row in protocol_rows],
            "oracle": [float(row["sample_oracle_mae_K"]) for row in protocol_rows],
        },
        "MAE (K)",
    )
    reliability_test = [
        row for row in reliability_rows if row["protocol"] == "primary_test_families"
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        [row["disagreement_mean_K"] for row in reliability_test],
        [row["residual_mae_K"] for row in reliability_test],
        marker="o",
        label="residual",
    )
    ax.plot(
        [row["disagreement_mean_K"] for row in reliability_test],
        [row["direct_mae_K"] for row in reliability_test],
        marker="o",
        label="direct",
    )
    ax.set_xlabel("Mean disagreement in quantile bin (K)")
    ax.set_ylabel("MAE (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / names[7], dpi=160)
    plt.close(fig)

    test_routes = [row for row in routing_rows if row["protocol"] == "primary_test_families"]
    bar_plot(
        plt,
        out_dir / names[8],
        [str(row["method"]) for row in test_routes],
        [float(row["mae_K"]) for row in test_routes],
        "Primary-test MAE (K)",
    )
    test_families = [
        row for row in family_rows if row["protocol"] == "primary_test_families"
    ]
    grouped = {
        "direct": [float(row["direct_mae_K"]) for row in test_families],
        "residual": [float(row["residual_mae_K"]) for row in test_families],
        "oracle": [float(row["sample_oracle_mae_K"]) for row in test_families],
    }
    if all(row.get("source_mae_K") is not None for row in test_families):
        grouped["source"] = [float(row["source_mae_K"]) for row in test_families]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    test_labels = [str(row["family_uid"]) for row in test_families]
    x = np.arange(len(test_labels))
    width = 0.8 / max(len(grouped), 1)
    for index, (name, values) in enumerate(grouped.items()):
        ax.bar(x - 0.4 + width / 2 + index * width, values, width=width, label=name)
    ax.set_xticks(x, test_labels)
    ax.set_ylabel("MAE (K)")
    fraction_axis = ax.twinx()
    fraction_axis.plot(
        x,
        [float(row["direct_win_fraction"]) for row in test_families],
        color="black",
        marker="o",
        label="direct win fraction",
    )
    fraction_axis.set_ylim(0.0, 1.0)
    fraction_axis.set_ylabel("Direct win fraction")
    handles, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = fraction_axis.get_legend_handles_labels()
    ax.legend(handles + handles_right, labels_left + labels_right, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / names[9], dpi=160)
    plt.close(fig)


def bar_plot(
    plt: Any,
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.3), 5.0))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=75, ha="right")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def grouped_bar_plot(
    plt: Any,
    path: Path,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.2), 5.0))
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series.items()):
        ax.bar(x - 0.4 + width / 2 + index * width, values, width=width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scatter_plot(
    plt: Any,
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_placeholder_png(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("matplotlib or Pillow is required to write requested figures") from exc
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 30), path.stem.replace("_", " "), fill="black")
    draw.text((30, 70), "Plotting backend unavailable; numeric CSV outputs remain authoritative.", fill="black")
    image.save(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows:
        if not allow_empty:
            raise ValueError(f"refusing to write empty required table: {path}")
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def group_rows(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    output: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(str(row.get(key) or "") for key in keys)].append(row)
    return output


def family_for_row(row: Mapping[str, str]) -> str:
    family = str(row.get("family_uid") or row.get("case_id") or "")
    if not family:
        raise ValueError(f"row has no family identifier: available={sorted(row)}")
    return family


def protocol_split(protocol: str) -> str:
    return {
        "known_family_sample_test": "known_family_sample_test",
        "primary_validation_families": "heldout_validation",
        "primary_test_families": "heldout_test",
    }[protocol]


def short_protocol(protocol: str) -> str:
    return {
        "known_family_sample_test": "known",
        "primary_validation_families": "val",
        "primary_test_families": "test",
    }[protocol]


def array_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def mean_value(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean(array_values(rows, key)))


def median_value(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.median(array_values(rows, key)))


def optional_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def conditional_margin(
    rows: Sequence[Mapping[str, Any]],
    *,
    winner: str,
    statistic: str = "mean",
) -> float | None:
    if winner == "direct":
        values = [
            -float(row["direct_minus_residual_mae_K"])
            for row in rows
            if row["winner"] == "direct"
        ]
    else:
        values = [
            float(row["direct_minus_residual_mae_K"])
            for row in rows
            if row["winner"] == "residual"
        ]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if statistic == "mean":
        return float(np.mean(array))
    if statistic == "median":
        return float(np.median(array))
    if statistic == "p90":
        return float(np.quantile(array, 0.90))
    raise ValueError(f"unsupported margin statistic: {statistic}")


def oracle_direct_fraction(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(
        np.mean(
            [
                float(row["direct_mae_K"]) < float(row["residual_mae_K"])
                for row in rows
            ]
        )
    )


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(error, dtype=np.float64) ** 2)))


def masked_mae(error: np.ndarray, mask: np.ndarray) -> float | None:
    return float(np.mean(np.abs(error[mask]))) if np.any(mask) else None


def euclidean_index_distance(first: Sequence[int], second: Sequence[int]) -> float:
    return float(math.sqrt(sum((int(a) - int(b)) ** 2 for a, b in zip(first, second))))


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
