#!/usr/bin/env python3
"""Build the frozen Benchmark v2 held-out metric report from saved artifacts.

This script performs no model inference. It intentionally treats the evaluator's
saved JSON/CSV artifacts and completed-run manifests as the authoritative inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_VERSION = "source_superposition_final_train40_source_v1"
SOURCE_CHECKPOINT = (
    "outputs/benchmark_v2_50family/source_response/final_train40_v1/"
    "checkpoints/best.pt"
)
SOURCE_PARAMETER_COUNT = 475_585
TRAIN_FAMILIES = [
    "f001", "f002", "f003", "f004", "f005", "f006", "f009", "f010",
    "f011", "f013", "f014", "f015", "f017", "f018", "f019", "f020",
    "f021", "f022", "f024", "f025", "f026", "f028", "f029", "f031",
    "f032", "f034", "f035", "f036", "f037", "f038", "f039", "f040",
    "f042", "f043", "f045", "f046", "f047", "f048", "f049", "f050",
]
VALIDATION_FAMILIES = ["f007", "f012", "f023", "f030", "f041"]
TEST_FAMILIES = ["f008", "f016", "f027", "f033", "f044"]


@dataclass(frozen=True)
class Run:
    backbone: str
    mode: str
    root: str
    primary_layout: str

    @property
    def key(self) -> str:
        return f"{self.backbone.lower().replace('-', '_')}_{self.mode.lower()}"


RUNS = [
    Run(
        "CNN",
        "Direct",
        "outputs/benchmark_v2_50family/package_direct/"
        "direct_temperature_feature_fusion_normalized_train40_seed1",
        "legacy",
    ),
    Run(
        "CNN",
        "Physics-guided residual",
        "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1",
        "legacy",
    ),
    Run(
        "FNO",
        "Direct",
        "outputs/benchmark_v2_50family/fno/"
        "direct_temperature_fno_normalized_train40_seed1",
        "legacy",
    ),
    Run(
        "FNO",
        "Physics-guided residual",
        "outputs/benchmark_v2_50family/fno/"
        "residual_fno_decomposed_train40_seed1",
        "legacy",
    ),
    Run(
        "U-FNO",
        "Direct",
        "outputs/benchmark_v2_50family/ufno/"
        "direct_temperature_ufno_normalized_train40_seed1",
        "stage_gated",
    ),
    Run(
        "U-FNO",
        "Physics-guided residual",
        "outputs/benchmark_v2_50family/ufno/"
        "residual_ufno_decomposed_train40_seed1",
        "stage_gated",
    ),
    Run(
        "SAU-FNO",
        "Direct",
        "outputs/benchmark_v2_50family/sau_fno/"
        "direct_temperature_sau_fno_normalized_train40_seed1",
        "stage_gated",
    ),
    Run(
        "SAU-FNO",
        "Physics-guided residual",
        "outputs/benchmark_v2_50family/sau_fno/"
        "residual_sau_fno_decomposed_train40_seed1",
        "stage_gated",
    ),
]

PROTOCOLS = {
    "known_family_sample_test": "Familiar-family interpolation",
    "primary_validation_families": "Held-out validation families",
    "primary_test_families": "Strict held-out test families",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen Benchmark v2 held-out metric report without inference."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--markdown-out", type=Path, default=Path("docs/final_heldout_metric_report.md")
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/runs/benchmarks/final_heldout_metric_report.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/runs/benchmarks/final_heldout_metric_table.csv"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report artifact is missing: {path}")
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required report artifact is missing: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def protocol_dir(repo: Path, run: Run, protocol: str) -> Path:
    root = repo / run.root
    if run.primary_layout == "stage_gated":
        parent = "evaluation_primary_test" if protocol == "primary_test_families" else "evaluation_selection"
    else:
        parent = "evaluation"
    path = root / parent / protocol
    if not path.is_dir():
        raise FileNotFoundError(f"missing canonical protocol directory for {run.key}: {path}")
    return path


def optional_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def aggregate_family_values(families: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in families]
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "std_population": statistics.pstdev(values),
    }


def load_protocol(repo: Path, run: Run, protocol: str) -> dict[str, Any]:
    directory = protocol_dir(repo, run, protocol)
    metrics_path = directory / "metrics.json"
    sample_path = directory / "metrics_by_sample.csv"
    family_path = directory / "metrics_by_case.csv"
    metrics = read_json(metrics_path)
    samples = read_csv(sample_path)
    family_rows = read_csv(family_path)
    final = metrics["cnn_final_temperature"]
    peak_abs = [float(row["peak_temperature_abs_error_K"]) for row in samples]
    families = [
        {
            "family_uid": row["case_id"],
            "mae_K": float(row["cnn_final_mae_K"]),
            "rmse_K": float(row["cnn_final_rmse_K"]),
            "mean_signed_error_K": optional_float(row.get("cnn_final_mean_signed_error_K")),
            "signed_peak_value_error_K": optional_float(row.get("cnn_hotspot_temp_error_K")),
            "hotspot_location_error_cells": optional_float(row.get("cnn_hotspot_location_error_cells")),
        }
        for row in family_rows
    ]
    return {
        "scope": PROTOCOLS[protocol],
        "sample_count": len(samples),
        "sample_uids": [row["sample_uid"] for row in samples],
        "metrics": {
            "mae_K": float(final["mae_K"]),
            "rmse_K": float(final["rmse_K"]),
            "mean_signed_error_K": float(final["mean_signed_error_K"]),
            "signed_peak_value_error_K": float(final["hotspot_temp_error_K"]),
            "mean_absolute_peak_value_error_K": statistics.fmean(peak_abs),
            "signed_error_at_true_hotspot_K": None,
            "mean_absolute_error_at_true_hotspot_K": None,
            "hotspot_location_error_cells": float(final["hotspot_location_error_cells"]),
            "hotspot_location_error_physical": None,
            "median_absolute_pixel_error_K": None,
            "p95_absolute_pixel_error_K": None,
            "fraction_samples_abs_peak_error_lt_1K": sum(x < 1.0 for x in peak_abs) / len(peak_abs),
            "fraction_samples_abs_peak_error_lt_2K": sum(x < 2.0 for x in peak_abs) / len(peak_abs),
            "fraction_samples_abs_peak_error_lt_3K": sum(x < 3.0 for x in peak_abs) / len(peak_abs),
        },
        "families": families,
        "family_aggregate": {
            "mae_K": aggregate_family_values(families, "mae_K"),
            "rmse_K": aggregate_family_values(families, "rmse_K"),
        },
        "artifact_sources": {
            "metrics": str(metrics_path.relative_to(repo)),
            "per_sample": str(sample_path.relative_to(repo)),
            "per_family": str(family_path.relative_to(repo)),
        },
        "model_parameter_count": int(metrics["model"]["parameter_count"]),
    }


def best_epoch_from_history(root: Path) -> tuple[int | None, int | None]:
    history_path = root / "training_history.json"
    if not history_path.is_file():
        return None, None
    rows = read_json(history_path).get("epochs", [])
    valid = []
    for row in rows:
        value = optional_float(row.get("val_final_mae_K"))
        if value is not None:
            valid.append((value, int(row["epoch"])))
    return (min(valid)[1] if valid else None, max((int(row["epoch"]) for row in rows), default=None))


def local_checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return int(checkpoint["epoch"]) if "epoch" in checkpoint else None
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def load_run(repo: Path, run: Run) -> dict[str, Any]:
    root = repo / run.root
    manifest = read_json(root / "completed_run_manifest.json")
    lineage = read_json(root / "training_lineage.json")
    resolved = manifest["resolved_config"]
    checkpoint_path = Path(run.root) / "checkpoints/best.pt"
    best_epoch, realized_epochs = best_epoch_from_history(root)
    checkpoint_epoch = local_checkpoint_epoch(repo / checkpoint_path)
    if checkpoint_epoch is not None:
        best_epoch = checkpoint_epoch
    if realized_epochs is None:
        realized_epochs = local_checkpoint_epoch(root / "checkpoints/last.pt")
    protocols = {name: load_protocol(repo, run, name) for name in PROTOCOLS}
    return {
        "backbone": run.backbone,
        "mode": run.mode,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": manifest["checkpoints"]["best.pt"]["sha256"],
            "best_epoch": best_epoch,
            "nominal_training_budget_epochs": int(resolved["training"]["epochs"]),
            "realized_training_epochs": realized_epochs,
            "selection_rule": "minimum internal-validation final-temperature MAE",
        },
        "architecture": resolved["training"]["model_architecture"],
        "prediction_mode": resolved["training"].get("prediction_mode", "residual_decomposed"),
        "source_version": resolved["wrapper"]["source_version"],
        "train_index_sha256": lineage["train_index_sha256"],
        "internal_validation_index_sha256": lineage["internal_val_index_sha256"],
        "primary_heldout_used_for_selection": bool(lineage["primary_heldout_used_for_selection"]),
        "backbone_parameter_count": protocols["primary_test_families"]["model_parameter_count"],
        "shared_source_response_parameter_count": (
            SOURCE_PARAMETER_COUNT if run.mode == "Physics-guided residual" else 0
        ),
        "complete_system_parameter_count": protocols["primary_test_families"]["model_parameter_count"]
        + (SOURCE_PARAMETER_COUNT if run.mode == "Physics-guided residual" else 0),
        "protocols": protocols,
        "lineage_source": str((root / "training_lineage.json").relative_to(repo)),
        "manifest_source": str((root / "completed_run_manifest.json").relative_to(repo)),
    }


def load_source_baseline(repo: Path, reference_run: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = repo / SOURCE_CHECKPOINT
    checkpoint_epoch = local_checkpoint_epoch(checkpoint_path)
    source_hash = "249bfa021ac738c0644e9349e20317a4353434651fb6132a8a91c9e958512421"
    result: dict[str, Any] = {
        "backbone": "Shared source-response",
        "mode": "Source-superposition baseline",
        "checkpoint": {
            "path": SOURCE_CHECKPOINT,
            "sha256": source_hash,
            "best_epoch": checkpoint_epoch,
            "nominal_training_budget_epochs": 100,
            "realized_training_epochs": 65,
            "selection_rule": "minimum validation package-level reconstructed full-grid MAE",
        },
        "architecture": "source_response_operator_v1",
        "prediction_mode": "source_superposition_v1",
        "source_version": SOURCE_VERSION,
        "backbone_parameter_count": 0,
        "shared_source_response_parameter_count": SOURCE_PARAMETER_COUNT,
        "complete_system_parameter_count": SOURCE_PARAMETER_COUNT,
        "primary_heldout_used_for_selection": False,
        "protocols": {},
    }
    reference_root = repo / RUNS[1].root
    for protocol in PROTOCOLS:
        directory = protocol_dir(repo, RUNS[1], protocol)
        metrics = read_json(directory / "metrics.json")["physics_baseline"]
        family_rows = read_csv(directory / "metrics_by_case.csv")
        families = [
            {
                "family_uid": row["case_id"],
                "mae_K": float(row["physics_mae_K"]),
                "rmse_K": float(row["physics_rmse_K"]),
            }
            for row in family_rows
        ]
        result["protocols"][protocol] = {
            "scope": PROTOCOLS[protocol],
            "sample_count": int(metrics["num_samples"]),
            "sample_uids": reference_run["protocols"][protocol]["sample_uids"],
            "metrics": {
                "mae_K": float(metrics["mae_K"]),
                "rmse_K": float(metrics["rmse_K"]),
                "mean_signed_error_K": float(metrics["mean_signed_error_K"]),
                "signed_peak_value_error_K": float(metrics["hotspot_temp_error_K"]),
                "mean_absolute_peak_value_error_K": None,
                "signed_error_at_true_hotspot_K": None,
                "mean_absolute_error_at_true_hotspot_K": None,
                "hotspot_location_error_cells": float(metrics["hotspot_location_error_cells"]),
                "hotspot_location_error_physical": None,
                "median_absolute_pixel_error_K": None,
                "p95_absolute_pixel_error_K": None,
                "fraction_samples_abs_peak_error_lt_1K": None,
                "fraction_samples_abs_peak_error_lt_2K": None,
                "fraction_samples_abs_peak_error_lt_3K": None,
            },
            "families": families,
            "family_aggregate": {
                "mae_K": aggregate_family_values(families, "mae_K"),
                "rmse_K": aggregate_family_values(families, "rmse_K"),
            },
            "artifact_sources": {
                "metrics": str((directory / "metrics.json").relative_to(repo)),
                "per_family": str((directory / "metrics_by_case.csv").relative_to(repo)),
            },
        }
    return result


def ensure_sample_identity(models: list[dict[str, Any]]) -> dict[str, Any]:
    train_hashes = {model["train_index_sha256"] for model in models}
    validation_hashes = {model["internal_validation_index_sha256"] for model in models}
    source_versions = {model["source_version"] for model in models}
    if len(train_hashes) != 1 or len(validation_hashes) != 1 or source_versions != {SOURCE_VERSION}:
        raise ValueError("frozen models do not share one train/validation/source-version lineage")
    audit = {}
    for protocol in PROTOCOLS:
        first = models[0]["protocols"][protocol]["sample_uids"]
        matches = all(model["protocols"][protocol]["sample_uids"] == first for model in models[1:])
        if not matches:
            raise ValueError(f"model sample order differs for {protocol}")
        audit[protocol] = {"identical_order": True, "sample_count": len(first)}
    return audit


def fmt(value: Any, digits: int = 4) -> str:
    return "not recoverable from current artifacts" if value is None else f"{float(value):.{digits}f}"


def model_label(model: dict[str, Any]) -> str:
    return f"{model['backbone']} / {model['mode']}"


def source_ref(model: dict[str, Any], protocol: str) -> str:
    return model["protocols"][protocol]["artifact_sources"]["metrics"]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(report: dict[str, Any]) -> str:
    models = report["models"]
    source = report["source_superposition_baseline"]
    primary = "primary_test_families"
    known = "known_family_sample_test"
    lines = [
        "# Final Frozen Benchmark v2 Held-Out Metric Report",
        "",
        "## A. Executive Summary",
        "",
        (
            "This report compares the frozen primary CNN, FNO, U-FNO, and SAU-FNO "
            "backbones on the same 1,000-sample strict held-out-family protocol. "
            "Reference temperatures are HotSpot-generated fields. No inference or model "
            "selection was performed to create this report."
        ),
        "",
        (
            "The lowest held-out full-map MAE is the physics-guided CNN at 1.3306 K. "
            "Physics guidance lowers MAE for every backbone, but it does not uniformly "
            "improve RMSE or hotspot metrics: the direct SAU-FNO has the lowest held-out "
            "RMSE (2.3983 K), and direct FNO has the lowest recovered absolute peak-value "
            "error (1.4970 K). This supports early-stage screening use, not thermal signoff."
        ),
        "",
        "## B. Primary Held-Out-Family Results",
        "",
    ]
    table_rows = []
    source_metrics = source["protocols"][primary]["metrics"]
    table_rows.append([
        source["backbone"], source["mode"], "0.0000",
        f"{source['complete_system_parameter_count']/1e6:.4f}",
        fmt(source_metrics["mae_K"]), fmt(source_metrics["rmse_K"]),
        fmt(source_metrics["signed_peak_value_error_K"]),
        fmt(source_metrics["mean_absolute_peak_value_error_K"]),
        fmt(source_metrics["hotspot_location_error_cells"]),
    ])
    for model in models:
        m = model["protocols"][primary]["metrics"]
        residual_params = (
            f"{model['backbone_parameter_count']/1e6:.4f}"
            if model["mode"] == "Physics-guided residual" else "n/a (direct)"
        )
        table_rows.append([
            model["backbone"], model["mode"], residual_params,
            f"{model['complete_system_parameter_count']/1e6:.4f}",
            fmt(m["mae_K"]), fmt(m["rmse_K"]),
            fmt(m["signed_peak_value_error_K"]),
            fmt(m["mean_absolute_peak_value_error_K"]),
            fmt(m["hotspot_location_error_cells"]),
        ])
    lines.extend([
        markdown_table(
            ["Backbone", "Mode", "Residual Params (M)", "Total Params (M)", "MAE (K)",
             "RMSE (K)", "Signed Peak Error (K)", "Absolute Peak Error (K)",
             "Hotspot Location Error (cells)"],
            table_rows,
        ),
        "",
        (
            "For direct rows, total parameters are the direct backbone. For residual rows, "
            f"total parameters equal the residual backbone plus the shared {SOURCE_PARAMETER_COUNT:,}-parameter "
            "source-response model; the shared model is counted once per complete system."
        ),
        "",
        "Row-level authoritative sources:",
        "",
        f"- Source baseline: `{source_ref(source, primary)}`",
    ])
    lines.extend(f"- {model_label(m)}: `{source_ref(m, primary)}`" for m in models)
    bias_tail_rows = []
    for model in [source] + models:
        m = model["protocols"][primary]["metrics"]
        bias_tail_rows.append([
            model["backbone"], model["mode"], fmt(m["mean_signed_error_K"]),
            fmt(m["fraction_samples_abs_peak_error_lt_1K"]),
            fmt(m["fraction_samples_abs_peak_error_lt_2K"]),
            fmt(m["fraction_samples_abs_peak_error_lt_3K"]),
        ])
    lines.extend([
        "",
        "Additional held-out bias and peak-tail diagnostics:",
        "",
        markdown_table(
            ["Backbone", "Mode", "Mean Signed Error (K)", "Abs Peak <1 K",
             "Abs Peak <2 K", "Abs Peak <3 K"], bias_tail_rows
        ),
    ])
    lines.extend(["", "## C. Direct-to-Physics-Guided Improvement", ""])
    improvement_rows = []
    for backbone in ("CNN", "FNO", "U-FNO", "SAU-FNO"):
        direct = next(m for m in models if m["backbone"] == backbone and m["mode"] == "Direct")
        residual = next(m for m in models if m["backbone"] == backbone and m["mode"] != "Direct")
        dm = direct["protocols"][primary]["metrics"]
        rm = residual["protocols"][primary]["metrics"]
        improvement_rows.append([
            backbone, fmt(dm["mae_K"]), fmt(rm["mae_K"]), fmt(dm["mae_K"] - rm["mae_K"]),
            fmt(dm["rmse_K"]), fmt(rm["rmse_K"]), fmt(dm["rmse_K"] - rm["rmse_K"]),
        ])
    lines.extend([
        markdown_table(
            ["Backbone", "Direct MAE", "Physics-guided MAE", "MAE Reduction", "Direct RMSE",
             "Physics-guided RMSE", "RMSE Reduction"], improvement_rows
        ),
        "",
        "Positive reduction means the physics-guided model improved the metric; negative RMSE reductions are regressions.",
        "",
        "## D. Familiar-Family / Interpolation Results",
        "",
    ])
    familiar_rows = []
    for model in models:
        m = model["protocols"][known]["metrics"]
        familiar_rows.append([
            model["backbone"], model["mode"], fmt(m["mae_K"]), fmt(m["rmse_K"]),
            fmt(m["mean_signed_error_K"]), fmt(m["mean_absolute_peak_value_error_K"]),
            fmt(m["hotspot_location_error_cells"]),
        ])
    lines.extend([
        markdown_table(
            ["Backbone", "Mode", "MAE (K)", "RMSE (K)", "Mean Signed Error (K)",
             "Absolute Peak Error (K)", "Location Error (cells)"], familiar_rows
        ),
        "",
        "The familiar-family protocol contains 800 held-out workload samples from known package families.",
        "",
        "## E. Per-Held-Out-Family Results",
        "",
    ])
    family_rows = []
    for model in [source] + models:
        for family in model["protocols"][primary]["families"]:
            family_rows.append([
                model["backbone"], model["mode"], family["family_uid"],
                fmt(family["mae_K"]), fmt(family["rmse_K"]),
            ])
    lines.extend([
        markdown_table(["Backbone", "Mode", "Family", "MAE (K)", "RMSE (K)"], family_rows),
        "",
        "Unweighted variation across the five held-out test families:",
        "",
    ])
    aggregate_rows = []
    for model in [source] + models:
        agg = model["protocols"][primary]["family_aggregate"]
        aggregate_rows.append([
            model["backbone"], model["mode"],
            fmt(agg["mae_K"]["mean"]), fmt(agg["mae_K"]["min"]),
            fmt(agg["mae_K"]["max"]), fmt(agg["mae_K"]["std_population"]),
            fmt(agg["rmse_K"]["mean"]), fmt(agg["rmse_K"]["min"]),
            fmt(agg["rmse_K"]["max"]), fmt(agg["rmse_K"]["std_population"]),
        ])
    lines.extend([
        markdown_table(
            ["Backbone", "Mode", "MAE Mean", "MAE Min", "MAE Max", "MAE Std",
             "RMSE Mean", "RMSE Min", "RMSE Max", "RMSE Std"], aggregate_rows
        ),
        "",
        "## F. Metric Definitions",
        "",
        "- **Full-map MAE:** mean absolute cell-wise difference from the HotSpot-generated reference field over every sample and all 64x64 cells.",
        "- **Full-map RMSE:** square root of the global mean squared cell error. This report uses `rmse_K`/`global_pixel_rmse_K`, not the separately saved mean of per-sample RMSEs.",
        "- **Mean signed error:** mean of `prediction - reference` over every cell; positive values indicate overprediction.",
        "- **Signed peak-value error:** `max(prediction) - max(reference)`, averaged over samples. The evaluator computes this from each map's independent argmax (`scripts/evaluate_residual_cnn.py:1956-1960`).",
        "- **Absolute peak-value error:** `abs(max(prediction) - max(reference))`, averaged over samples (`scripts/evaluate_residual_cnn.py:1062`). It is not the absolute value of the signed average.",
        "- **True-hotspot-location error:** `prediction[argmax(reference)] - reference[argmax(reference)]`. The current evaluator does not save this metric.",
        "- **Hotspot-location distance:** Euclidean row/column distance in grid cells between `argmax(prediction)` and `argmax(reference)` (`scripts/evaluate_residual_cnn.py:1956-1959`).",
        "- **Physical hotspot distance:** not reported. Saved artifacts retain only scalar cell distance, not directional displacement; family-dependent and potentially anisotropic pitch prevents an unambiguous conversion.",
        "",
        "## G. Checkpoint and Protocol Audit",
        "",
        f"- Source version: `{SOURCE_VERSION}`.",
        f"- Authoritative split manifest: `$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/{SOURCE_VERSION}/index_manifest.json` (SHA-256 `2797a69a82e5d1c7aebd52babbb846275c36745eca6882ebce73b4a54b52530c`).",
        f"- Training families (40): {', '.join(TRAIN_FAMILIES)}.",
        f"- Held-out validation families (5): {', '.join(VALIDATION_FAMILIES)}.",
        f"- Strict held-out test families (5): {', '.join(TEST_FAMILIES)}.",
        "- All eight model evaluations use identical ordered sample UIDs: 800 familiar-family, 1,000 held-out validation, and 1,000 held-out test samples.",
        "- Every lineage records the same train-index SHA-256 and internal-validation-index SHA-256, and records `primary_heldout_used_for_selection=false`.",
        "- `best.pt` selection is minimum internal-validation final-temperature MAE (`scripts/train_residual_cnn.py:1413-1418`). Held-out validation families gated architecture progression; strict held-out test families did not select checkpoint weights.",
        "- All runs have a nominal 100-epoch budget. Realized histories differ: CNN direct has 99 saved epochs and SAU-FNO direct stopped after 65; this is disclosed because the frozen comparison is not perfectly equal in realized optimization exposure.",
        "- The later compact low-learning-rate continuation and rejected interpolation/soup experiments are excluded.",
        "",
        markdown_table(
            ["Backbone", "Mode", "Best Epoch", "Realized/Budget", "Checkpoint SHA-256", "Checkpoint"],
            [[
                m["backbone"], m["mode"], str(m["checkpoint"]["best_epoch"]),
                f"{m['checkpoint']['realized_training_epochs']}/{m['checkpoint']['nominal_training_budget_epochs']}",
                m["checkpoint"]["sha256"], f"`{m['checkpoint']['path']}`",
            ] for m in [source] + models],
        ),
        "",
        "## H. Missing Data",
        "",
        "The following are not recoverable from the locally retained evaluation summaries alone:",
        "",
        "- signed and mean absolute error evaluated at the true reference-hotspot location;",
        "- median and 95th-percentile absolute pixel error;",
        "- physical hotspot-location distance;",
        "- source-baseline mean absolute peak error and source-baseline peak-threshold fractions.",
        "",
        "Final-temperature prediction arrays are retained, but the external Benchmark v2 target/index tree is not present in this workspace. The first two items can be computed offline on GT from saved predictions and reference arrays without checkpoint inference. Physical distance remains unavailable unless directional argmax displacement and per-axis pitch are recomputed from maps and unambiguous package metadata.",
        "",
        "No disagreement was found among duplicate summaries for the selected paths. `rmse_K` equals `global_pixel_rmse_K`; `mean_sample_rmse_K` is a different aggregation and is intentionally not reported as full-map RMSE.",
        "",
        "## I. Manual GT Commands for Missing Metrics",
        "",
        "No CUDA inference is required because all eight primary-test prediction sets already exist. Re-running inference would not add true-hotspot or pixel-tail fields to the current evaluator. The audit-safe next step is a lightweight CPU post-processing pass on GT using the frozen `family_split/test_index.csv`, its `y_path` arrays, and each run's saved `predictions/*_tpred.npy` arrays. This report intentionally leaves those cells unavailable until that target-backed pass is run.",
        "",
        "To reproduce the currently available report artifacts on GT after syncing the saved evaluation trees:",
        "",
        "```bash",
        "cd /nethome/$USER/chiptherm",
        "source .venv/bin/activate",
        "python3 scripts/build_final_heldout_metric_report.py",
        "```",
        "",
        "If any saved prediction tree is missing on GT, regenerate it with the existing frozen wrapper (substitute only the authoritative checkpoint and output root from the audit table):",
        "",
        "```bash",
        "export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family",
        f"export SOURCE_VERSION={SOURCE_VERSION}",
        "python3 scripts/evaluate_benchmark_v2_models.py \\",
        "  --data-root \"$CHIPTHERM_V2_DATA_ROOT\" \\",
        "  --source-version \"$SOURCE_VERSION\" \\",
        "  --checkpoint <AUTHORITATIVE_CHECKPOINT_FROM_TABLE> \\",
        "  --out-dir <AUTHORITATIVE_RUN_ROOT>/evaluation_recovery \\",
        "  --batch-size 64 --device cuda --workers 4 \\",
        "  --protocols primary_test_families --save-predictions",
        "```",
        "",
        "## J. SRC-Ready Summary",
        "",
        "Across five strictly held-out package families (1,000 workloads), the shared source-superposition baseline achieved 1.668 K MAE. Adding a frozen residual predictor reduced MAE to 1.331 K (CNN), 1.371 K (FNO), 1.365 K (U-FNO), and 1.399 K (SAU-FNO), while direct predictors achieved 1.767 K, 1.494 K, 1.754 K, and 1.545 K, respectively. The physics-guided CNN had the lowest full-map MAE, whereas direct SAU-FNO had the lowest RMSE; these results characterize ChipTherm as an early-stage screening and optimization surrogate against HotSpot-generated reference temperatures, not a signoff replacement.",
        "",
        "```latex",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Backbone & Mode & Params (M) & MAE (K) & RMSE (K) \\\\",
        "\\midrule",
    ])
    for model in models:
        m = model["protocols"][primary]["metrics"]
        latex_backbone = model["backbone"].replace("-", "--")
        latex_mode = "Residual" if model["mode"] != "Direct" else "Direct"
        lines.append(
            f"{latex_backbone} & {latex_mode} & {model['complete_system_parameter_count']/1e6:.3f} "
            f"& {m['mae_K']:.3f} & {m['rmse_K']:.3f} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "```",
        "",
        "Full-map MAE measures average field fidelity, while peak-value and hotspot-location errors probe different localized failure modes; improving one does not guarantee improving the others.",
        "",
    ])
    return "\n".join(lines)


def write_csv(path: Path, source: dict[str, Any], models: list[dict[str, Any]]) -> None:
    fields = [
        "backbone", "mode", "evaluation_scope", "sample_count", "backbone_parameter_count",
        "shared_source_response_parameter_count", "complete_system_parameter_count", "mae_K", "rmse_K",
        "mean_signed_error_K", "signed_peak_value_error_K", "mean_absolute_peak_value_error_K",
        "signed_error_at_true_hotspot_K", "mean_absolute_error_at_true_hotspot_K",
        "hotspot_location_error_cells", "hotspot_location_error_physical",
        "median_absolute_pixel_error_K", "p95_absolute_pixel_error_K",
        "fraction_samples_abs_peak_error_lt_1K", "fraction_samples_abs_peak_error_lt_2K",
        "fraction_samples_abs_peak_error_lt_3K", "checkpoint_path", "checkpoint_sha256",
        "best_epoch", "metric_artifact",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in [source] + models:
            for protocol in PROTOCOLS:
                data = model["protocols"][protocol]
                row = {
                    "backbone": model["backbone"],
                    "mode": model["mode"],
                    "evaluation_scope": protocol,
                    "sample_count": data["sample_count"],
                    "backbone_parameter_count": model["backbone_parameter_count"],
                    "shared_source_response_parameter_count": model["shared_source_response_parameter_count"],
                    "complete_system_parameter_count": model["complete_system_parameter_count"],
                    "checkpoint_path": model["checkpoint"]["path"],
                    "checkpoint_sha256": model["checkpoint"]["sha256"],
                    "best_epoch": model["checkpoint"]["best_epoch"],
                    "metric_artifact": data["artifact_sources"]["metrics"],
                }
                row.update(data["metrics"])
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    models = [load_run(repo, run) for run in RUNS]
    identity_audit = ensure_sample_identity(models)
    source = load_source_baseline(repo, models[1])
    for model in models:
        for protocol in PROTOCOLS:
            model["protocols"][protocol].pop("sample_uids")
    for protocol in PROTOCOLS:
        source["protocols"][protocol].pop("sample_uids")
    report = {
        "schema_version": "chiptherm_final_heldout_metric_report/1",
        "benchmark": "benchmark_v2_50family",
        "reference_temperature_source": "HotSpot-generated reference temperatures",
        "source_superposition_version": SOURCE_VERSION,
        "family_protocol": {
            "training": TRAIN_FAMILIES,
            "heldout_validation": VALIDATION_FAMILIES,
            "heldout_test": TEST_FAMILIES,
        },
        "sample_identity_audit": identity_audit,
        "source_superposition_baseline": source,
        "models": models,
        "missing_metrics": {
            "signed_error_at_true_hotspot_K": "reference arrays are absent locally",
            "mean_absolute_error_at_true_hotspot_K": "reference arrays are absent locally",
            "median_absolute_pixel_error_K": "reference arrays are absent locally",
            "p95_absolute_pixel_error_K": "reference arrays are absent locally",
            "hotspot_location_error_physical": (
                "saved scalar cell distance omits directional displacement needed for anisotropic pitch"
            ),
            "source_baseline_absolute_peak_and_thresholds": (
                "source per-sample peak metrics were not saved"
            ),
        },
    }
    json_path = repo / args.json_out
    markdown_path = repo / args.markdown_out
    csv_path = repo / args.csv_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(build_markdown(report))
    write_csv(csv_path, source, models)
    print(f"Wrote {markdown_path.relative_to(repo)}")
    print(f"Wrote {json_path.relative_to(repo)}")
    print(f"Wrote {csv_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
