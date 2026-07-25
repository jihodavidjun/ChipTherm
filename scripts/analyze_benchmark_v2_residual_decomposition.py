#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.models import build_model
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input
from scripts.analyze_residual_cnn_errors import architecture_info, predict_temperature, prepare_graph_batch, select_device


PROTOCOLS = {
    "heldout_validation": ("family_split/val_index.csv", "primary_validation_families"),
    "heldout_test": ("family_split/test_index.csv", "primary_test_families"),
}
SAMPLE_COLUMNS = (
    "protocol",
    "split",
    "sample_uid",
    "family_uid",
    "case_id",
    "workload_uid",
    "workload_regime",
    "workload_cell",
    "workload_stratum",
    "broad_stratum",
    "power_regime",
    "topology_regime",
    "source_superposition_mae_K",
    "final_cnn_mae_K",
    "cnn_improvement_K",
    "true_residual_mean_K",
    "predicted_scalar_mean_correction_K",
    "mean_correction_error_K",
    "absolute_mean_correction_error_K",
    "true_centered_spatial_residual_abs_mean_K",
    "true_centered_spatial_residual_rms_K",
    "predicted_centered_spatial_correction_abs_mean_K",
    "predicted_centered_spatial_correction_rms_K",
    "centered_spatial_mae_K",
    "centered_spatial_rmse_K",
    "peak_temperature_error_K",
    "peak_temperature_abs_error_K",
    "cnn_worse_than_source_baseline",
    "prediction_source",
)
FAMILY_COLUMNS = (
    "protocol",
    "family_uid",
    "num_samples",
    "source_superposition_mae_K",
    "final_cnn_mae_K",
    "cnn_improvement_K",
    "true_residual_mean_abs_K",
    "predicted_scalar_mean_correction_abs_K",
    "mean_correction_mae_K",
    "centered_spatial_mae_K",
    "peak_temperature_mae_K",
    "cnn_worse_than_source_count",
    "cnn_worse_than_source_fraction",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline source/mean/centered/final residual-error decomposition for Benchmark v2."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=None,
        help="Existing residual evaluation root containing primary_*_families/predictions.",
    )
    parser.add_argument("--validation-index", type=Path, default=None)
    parser.add_argument("--test-index", type=Path, default=None)
    parser.add_argument("--validation-predictions", type=Path, default=None)
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--require-saved-predictions",
        action="store_true",
        help="Fail instead of running checkpoint inference when any saved final prediction is missing.",
    )
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")

    data_root = args.data_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_root = data_root / f"derived/indices/full_50x200/source_superposition/{args.source_version}"
    evaluation_root = (
        args.evaluation_root.expanduser().resolve()
        if args.evaluation_root is not None
        else checkpoint_path.parent.parent / "evaluation"
    )
    indices = {
        "heldout_validation": (
            args.validation_index.expanduser().resolve()
            if args.validation_index is not None
            else index_root / PROTOCOLS["heldout_validation"][0]
        ),
        "heldout_test": (
            args.test_index.expanduser().resolve()
            if args.test_index is not None
            else index_root / PROTOCOLS["heldout_test"][0]
        ),
    }
    prediction_roots = {
        "heldout_validation": _explicit_or_evaluation_prediction_root(
            args.validation_predictions,
            evaluation_root / PROTOCOLS["heldout_validation"][1],
        ),
        "heldout_test": _explicit_or_evaluation_prediction_root(
            args.test_predictions,
            evaluation_root / PROTOCOLS["heldout_test"][1],
        ),
    }

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_info = architecture_info(payload["model_config"])
    _validate_checkpoint_contract(model_info)
    stats = NormalizationStats(**payload["normalization"])
    device = select_device(args.device)

    cached_availability = {
        protocol: _cached_availability(index, prediction_roots[protocol])
        for protocol, index in indices.items()
    }
    missing_saved = sum(
        sum(not available for available in availability)
        for availability in cached_availability.values()
    )
    if args.require_saved_predictions and missing_saved:
        raise ValueError(f"{missing_saved} held-out samples do not have saved final predictions")

    model = None
    if missing_saved:
        model = build_model(payload["model_config"]).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

    records: list[dict[str, Any]] = []
    cache_counts = {"saved_prediction": 0, "checkpoint_inference": 0}
    for protocol, index_path in indices.items():
        protocol_records, protocol_counts = analyze_protocol(
            protocol=protocol,
            index_path=index_path,
            prediction_root=prediction_roots[protocol],
            model=model,
            model_info=model_info,
            stats=stats,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        records.extend(protocol_records)
        for key, value in protocol_counts.items():
            cache_counts[key] += value

    family_rows = aggregate_families(records)
    summary = build_summary(
        records,
        checkpoint_path=checkpoint_path,
        source_version=args.source_version,
        indices=indices,
        prediction_roots=prediction_roots,
        cache_counts=cache_counts,
    )
    write_csv(out_dir / "per_sample_decomposition.csv", records, SAMPLE_COLUMNS)
    write_csv(out_dir / "per_family_decomposition.csv", family_rows, FAMILY_COLUMNS)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "residual_decomposition_report.md", summary, family_rows)
    write_plots(out_dir, family_rows, records)

    print("Benchmark v2 residual decomposition complete")
    print(f"Samples: {len(records)}")
    print(f"Saved predictions reused: {cache_counts['saved_prediction']}")
    print(f"Checkpoint predictions generated: {cache_counts['checkpoint_inference']}")
    print(f"Overall source/final MAE: {summary['overall']['source_superposition_mae_K']:.4f} / "
          f"{summary['overall']['final_cnn_mae_K']:.4f} K")
    print(f"f044 final MAE: {summary['f044']['final_cnn_mae_K']:.4f} K")
    print(f"Output: {out_dir}")
    return 0


def _validate_checkpoint_contract(model_info: Mapping[str, Any]) -> None:
    if not bool(model_info.get("decomposed")):
        raise ValueError("residual decomposition requires a decomposed residual checkpoint")
    if str(model_info.get("mean_head_mode")) != "residual_resistance":
        raise ValueError("Benchmark v2 decomposition expects mean_head_mode=residual_resistance")
    if str(model_info.get("physics_input_mode")) != "source_superposition_v1":
        raise ValueError("Benchmark v2 decomposition expects physics_input_mode=source_superposition_v1")


def _explicit_or_evaluation_prediction_root(explicit: Path | None, evaluation_dir: Path) -> Path:
    root = explicit.expanduser().resolve() if explicit is not None else evaluation_dir
    return root / "predictions" if (root / "predictions").is_dir() else root


def _cached_availability(index_path: Path, prediction_root: Path) -> list[bool]:
    rows = read_csv(index_path)
    return [cached_prediction_path(prediction_root, row).is_file() for row in rows]


def cached_prediction_path(prediction_root: Path, row: Mapping[str, str]) -> Path:
    uid = str(row["sample_uid"])
    family = str(row.get("family_uid") or row.get("case_id"))
    family_path = prediction_root / family / f"{uid}_tpred.npy"
    if family_path.is_file():
        return family_path
    return prediction_root / f"{uid}_tpred.npy"


@torch.inference_mode()
def analyze_protocol(
    *,
    protocol: str,
    index_path: Path,
    prediction_root: Path,
    model: torch.nn.Module | None,
    model_info: Mapping[str, Any],
    stats: NormalizationStats,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset = ChipThermDataset(
        index_path,
        target="residual",
        return_metadata=True,
        return_graph=bool(model_info.get("graph_enabled")),
        physical_representation=str(model_info.get("physical_representation", "dimensional")),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=chiptherm_collate if bool(model_info.get("graph_enabled")) else None,
    )
    records: list[dict[str, Any]] = []
    counts = {"saved_prediction": 0, "checkpoint_inference": 0}
    offset = 0
    for batch in loader:
        current_rows = dataset.rows[offset : offset + int(batch["temperature"].shape[0])]
        cached_paths = [cached_prediction_path(prediction_root, row) for row in current_rows]
        missing = [not path.is_file() for path in cached_paths]
        inferred: np.ndarray | None = None
        if any(missing):
            if model is None:
                raise ValueError("saved prediction is missing and checkpoint inference was not initialized")
            inferred = infer_batch(model, batch, stats, device, model_info)

        target = batch["temperature"].detach().cpu().numpy()
        source = batch["physics"].detach().cpu().numpy()
        for index, row in enumerate(current_rows):
            path = cached_paths[index]
            if path.is_file():
                prediction = load_map(path, "saved final prediction")
                prediction_source = "saved_prediction"
            else:
                assert inferred is not None
                prediction = inferred[index]
                prediction_source = "checkpoint_inference"
            counts[prediction_source] += 1
            records.append(
                decompose_sample(
                    row=row,
                    protocol=protocol,
                    target=target[index],
                    source=source[index],
                    prediction=prediction,
                    prediction_source=prediction_source,
                )
            )
        offset += len(current_rows)
    return records, counts


def infer_batch(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    stats: NormalizationStats,
    device: torch.device,
    model_info: Mapping[str, Any],
) -> np.ndarray:
    x = batch["x"].to(device, non_blocking=True)
    physics = batch["physics"].to(device, non_blocking=True)
    physics_v1 = batch.get("physics_v1")
    if physics_v1 is not None:
        physics_v1 = physics_v1.to(device, non_blocking=True)
    ambient = batch["ambient_K"].to(device, non_blocking=True).float()
    total_power = batch["total_power_W"].to(device, non_blocking=True).float()
    metadata_input = build_metadata_input(batch.get("metadata_vector"), stats)
    if metadata_input is not None:
        metadata_input = metadata_input.to(device, non_blocking=True)
    graph_batch = prepare_graph_batch(
        batch,
        bool(model_info.get("graph_enabled")),
        model_info.get("graph_normalization"),
        device,
    )
    model_input = build_model_input(
        x,
        physics,
        stats,
        physics_input_mode=str(model_info["physics_input_mode"]),
        physics_v1=physics_v1,
    )
    outputs = predict_temperature(
        model,
        model_input,
        physics,
        ambient,
        total_power,
        metadata_input,
        graph_batch,
        stats,
        dict(model_info),
    )
    prediction = outputs["temperature"].detach().float().cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("checkpoint inference produced NaN or Inf")
    return prediction


def decompose_sample(
    *,
    row: Mapping[str, str],
    protocol: str,
    target: np.ndarray,
    source: np.ndarray,
    prediction: np.ndarray,
    prediction_source: str,
) -> dict[str, Any]:
    target64 = _validated_map(target, "HotSpot target")
    source64 = _validated_map(source, "source-superposition base")
    prediction64 = _validated_map(prediction, "final CNN prediction")
    true_residual = target64 - source64
    predicted_residual = prediction64 - source64
    true_mean = float(true_residual.mean())
    predicted_mean = float(predicted_residual.mean())
    true_centered = true_residual - true_mean
    predicted_centered = predicted_residual - predicted_mean
    source_mae = float(np.mean(np.abs(source64 - target64)))
    final_mae = float(np.mean(np.abs(prediction64 - target64)))
    centered_error = predicted_centered - true_centered
    peak_error = float(prediction64.max() - target64.max())
    family = str(row.get("family_uid") or row.get("case_id"))
    power_regime = str(row.get("power_regime", ""))
    workload_regime = str(
        row.get("broad_stratum")
        or power_regime
        or row.get("workload_stratum")
        or row.get("workload_cell")
        or row.get("workload_uid")
        or "unknown"
    )
    return {
        "protocol": protocol,
        "split": str(row.get("split", "")),
        "sample_uid": str(row["sample_uid"]),
        "family_uid": family,
        "case_id": str(row.get("case_id") or family),
        "workload_uid": str(row.get("workload_uid", "")),
        "workload_regime": workload_regime,
        "workload_cell": str(row.get("workload_cell", "")),
        "workload_stratum": str(row.get("workload_stratum", "")),
        "broad_stratum": str(row.get("broad_stratum", "")),
        "power_regime": power_regime,
        "topology_regime": str(row.get("topology_regime", "")),
        "source_superposition_mae_K": source_mae,
        "final_cnn_mae_K": final_mae,
        "cnn_improvement_K": source_mae - final_mae,
        "true_residual_mean_K": true_mean,
        "predicted_scalar_mean_correction_K": predicted_mean,
        "mean_correction_error_K": predicted_mean - true_mean,
        "absolute_mean_correction_error_K": abs(predicted_mean - true_mean),
        "true_centered_spatial_residual_abs_mean_K": float(np.mean(np.abs(true_centered))),
        "true_centered_spatial_residual_rms_K": float(np.sqrt(np.mean(true_centered * true_centered))),
        "predicted_centered_spatial_correction_abs_mean_K": float(np.mean(np.abs(predicted_centered))),
        "predicted_centered_spatial_correction_rms_K": float(
            np.sqrt(np.mean(predicted_centered * predicted_centered))
        ),
        "centered_spatial_mae_K": float(np.mean(np.abs(centered_error))),
        "centered_spatial_rmse_K": float(np.sqrt(np.mean(centered_error * centered_error))),
        "peak_temperature_error_K": peak_error,
        "peak_temperature_abs_error_K": abs(peak_error),
        "cnn_worse_than_source_baseline": bool(final_mae > source_mae),
        "prediction_source": prediction_source,
    }


def _validated_map(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (64, 64):
        raise ValueError(f"{name} must have shape (64, 64), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def load_map(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _validated_map(np.load(path), name)


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "source_superposition_mae_K": None,
            "final_cnn_mae_K": None,
            "cnn_improvement_K": None,
            "true_residual_mean_abs_K": None,
            "predicted_scalar_mean_correction_abs_K": None,
            "mean_correction_mae_K": None,
            "centered_spatial_mae_K": None,
            "peak_temperature_mae_K": None,
            "cnn_worse_than_source_count": 0,
            "cnn_worse_than_source_fraction": None,
        }

    def mean(key: str) -> float:
        return float(np.mean([float(record[key]) for record in records]))

    worse = sum(bool(record["cnn_worse_than_source_baseline"]) for record in records)
    return {
        "num_samples": len(records),
        "source_superposition_mae_K": mean("source_superposition_mae_K"),
        "final_cnn_mae_K": mean("final_cnn_mae_K"),
        "cnn_improvement_K": mean("cnn_improvement_K"),
        "true_residual_mean_abs_K": float(
            np.mean([abs(float(record["true_residual_mean_K"])) for record in records])
        ),
        "predicted_scalar_mean_correction_abs_K": float(
            np.mean([abs(float(record["predicted_scalar_mean_correction_K"])) for record in records])
        ),
        "mean_correction_mae_K": mean("absolute_mean_correction_error_K"),
        "centered_spatial_mae_K": mean("centered_spatial_mae_K"),
        "peak_temperature_mae_K": mean("peak_temperature_abs_error_K"),
        "cnn_worse_than_source_count": worse,
        "cnn_worse_than_source_fraction": worse / len(records),
    }


def aggregate_families(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["protocol"]), str(record["family_uid"]))].append(record)
    output: list[dict[str, Any]] = []
    for (protocol, family), items in sorted(grouped.items()):
        output.append({"protocol": protocol, "family_uid": family, **aggregate_records(items)})
    return output


def aggregate_by_field(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        value = str(record.get(field, "") or "unknown")
        grouped[value].append(record)
    return {value: aggregate_records(items) for value, items in sorted(grouped.items())}


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    source_version: str,
    indices: Mapping[str, Path],
    prediction_roots: Mapping[str, Path],
    cache_counts: Mapping[str, int],
) -> dict[str, Any]:
    by_protocol = {
        protocol: aggregate_records([record for record in records if record["protocol"] == protocol])
        for protocol in PROTOCOLS
    }
    by_family = {
        family: aggregate_records([record for record in records if record["family_uid"] == family])
        for family in sorted({str(record["family_uid"]) for record in records})
    }
    f044 = by_family.get("f044")
    if f044 is None:
        raise ValueError("held-out decomposition does not contain required family f044")
    return {
        "schema_version": "benchmark_v2_residual_decomposition/1",
        "definition": {
            "true_residual": "HotSpot_temperature_K - source_superposition_base_K",
            "true_mean": "spatial_mean(true_residual)",
            "true_centered": "true_residual - true_mean",
            "predicted_residual": "final_CNN_temperature_K - source_superposition_base_K",
            "predicted_scalar_mean": "spatial_mean(predicted_residual)",
            "predicted_centered": "predicted_residual - predicted_scalar_mean",
            "note": "The centered model output is zero-mean, so saved final predictions identify both components without a second model pass.",
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_superposition_version": source_version,
        "indices": {key: str(path) for key, path in indices.items()},
        "prediction_roots": {key: str(path) for key, path in prediction_roots.items()},
        "prediction_usage": dict(cache_counts),
        "overall": aggregate_records(records),
        "heldout_validation": by_protocol["heldout_validation"],
        "heldout_test": by_protocol["heldout_test"],
        "f044": f044,
        "by_family": by_family,
        "by_workload_regime": aggregate_by_field(records, "workload_regime"),
        "by_power_regime": aggregate_by_field(records, "power_regime"),
        "by_topology_regime": aggregate_by_field(records, "topology_regime"),
        "by_workload_stratum": aggregate_by_field(records, "workload_stratum"),
    }


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
) -> None:
    overall = summary["overall"]
    lines = [
        "# Benchmark v2 Residual-Error Decomposition",
        "",
        "The decomposition uses the frozen source-superposition base and final residual CNN. "
        "Saved final predictions are reused where available.",
        "",
        "## Headline",
        "",
        "| Scope | N | Source MAE (K) | Mean MAE (K) | Centered MAE (K) | Final MAE (K) | Worse fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("overall", "heldout_validation", "heldout_test", "f044"):
        item = summary[name]
        lines.append(
            f"| {name} | {item['num_samples']} | {item['source_superposition_mae_K']:.4f} | "
            f"{item['mean_correction_mae_K']:.4f} | {item['centered_spatial_mae_K']:.4f} | "
            f"{item['final_cnn_mae_K']:.4f} | {item['cnn_worse_than_source_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Held-Out Families",
            "",
            "| Protocol | Family | N | Source MAE | Mean MAE | Centered MAE | Final MAE | Improvement |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| {row['protocol']} | {row['family_uid']} | {row['num_samples']} | "
            f"{row['source_superposition_mae_K']:.4f} | {row['mean_correction_mae_K']:.4f} | "
            f"{row['centered_spatial_mae_K']:.4f} | {row['final_cnn_mae_K']:.4f} | "
            f"{row['cnn_improvement_K']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Overall source-to-final MAE change: {overall['cnn_improvement_K']:.4f} K.",
            f"- Mean-correction MAE: {overall['mean_correction_mae_K']:.4f} K.",
            f"- Centered-spatial MAE: {overall['centered_spatial_mae_K']:.4f} K.",
            f"- CNN is worse than the source baseline for {overall['cnn_worse_than_source_fraction']:.1%} of samples.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(
    out_dir: Path,
    family_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        write_plots_with_pillow(out_dir, family_rows, records)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["family_uid"]) for row in family_rows]
    x = np.arange(len(labels))
    width = 0.20
    figure, axis = plt.subplots(figsize=(12, 5))
    series = (
        ("source_superposition_mae_K", "Source"),
        ("mean_correction_mae_K", "Mean"),
        ("centered_spatial_mae_K", "Centered"),
        ("final_cnn_mae_K", "Final"),
    )
    for index, (key, label) in enumerate(series):
        axis.bar(x + (index - 1.5) * width, [float(row[key]) for row in family_rows], width, label=label)
    axis.set_xticks(x, labels, rotation=45)
    axis.set_ylabel("Mean absolute error (K)")
    axis.set_title("Residual-error decomposition by held-out family")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "error_components_by_family.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4))
    improvements = [float(row["cnn_improvement_K"]) for row in family_rows]
    colors = ["#b23a48" if label == "f044" else "#2a6f97" for label in labels]
    axis.bar(labels, improvements, color=colors)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Source MAE - final MAE (K)")
    axis.set_title("CNN improvement by held-out family")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "cnn_improvement_by_family.png", dpi=160)
    plt.close(figure)

    regime_rows = aggregate_by_field(records, "power_regime")
    regime_labels = list(regime_rows)
    if regime_labels:
        figure, axis = plt.subplots(figsize=(10, 4))
        rx = np.arange(len(regime_labels))
        axis.bar(
            rx - 0.2,
            [float(regime_rows[label]["source_superposition_mae_K"]) for label in regime_labels],
            0.4,
            label="Source",
        )
        axis.bar(
            rx + 0.2,
            [float(regime_rows[label]["final_cnn_mae_K"]) for label in regime_labels],
            0.4,
            label="Final",
        )
        axis.set_xticks(rx, regime_labels, rotation=30)
        axis.set_ylabel("Mean absolute error (K)")
        axis.set_title("Source and final error by workload power regime")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(out_dir / "source_vs_final_by_power_regime.png", dpi=160)
        plt.close(figure)


def write_plots_with_pillow(
    out_dir: Path,
    family_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    labels = [str(row["family_uid"]) for row in family_rows]
    series = [
        ("Source", [float(row["source_superposition_mae_K"]) for row in family_rows], "#577590"),
        ("Mean", [float(row["mean_correction_mae_K"]) for row in family_rows], "#43aa8b"),
        ("Centered", [float(row["centered_spatial_mae_K"]) for row in family_rows], "#f9c74f"),
        ("Final", [float(row["final_cnn_mae_K"]) for row in family_rows], "#f94144"),
    ]
    _write_pillow_bar_chart(
        out_dir / "error_components_by_family.png",
        "Residual-error decomposition by held-out family",
        labels,
        series,
        Image,
        ImageDraw,
        ImageFont,
    )
    _write_pillow_bar_chart(
        out_dir / "cnn_improvement_by_family.png",
        "CNN improvement by held-out family",
        labels,
        [("Source MAE - final MAE", [float(row["cnn_improvement_K"]) for row in family_rows], "#2a6f97")],
        Image,
        ImageDraw,
        ImageFont,
        allow_negative=True,
    )
    regime_rows = aggregate_by_field(records, "power_regime")
    regime_labels = list(regime_rows)
    if regime_labels:
        _write_pillow_bar_chart(
            out_dir / "source_vs_final_by_power_regime.png",
            "Source and final error by workload power regime",
            regime_labels,
            [
                (
                    "Source",
                    [float(regime_rows[label]["source_superposition_mae_K"]) for label in regime_labels],
                    "#577590",
                ),
                (
                    "Final",
                    [float(regime_rows[label]["final_cnn_mae_K"]) for label in regime_labels],
                    "#f94144",
                ),
            ],
            Image,
            ImageDraw,
            ImageFont,
        )


def _write_pillow_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    image_module: Any,
    draw_module: Any,
    font_module: Any,
    *,
    allow_negative: bool = False,
) -> None:
    width, height = 1400, 620
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 75, 100
    image = image_module.new("RGB", (width, height), "white")
    draw = draw_module.Draw(image)
    font = font_module.load_default()
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom
    values = [float(value) for _, items, _ in series for value in items]
    minimum = min(0.0, min(values, default=0.0)) if allow_negative else 0.0
    maximum = max(0.0, max(values, default=1.0))
    if maximum - minimum < 1.0e-12:
        maximum = minimum + 1.0

    def y_position(value: float) -> float:
        fraction = (value - minimum) / (maximum - minimum)
        return plot_bottom - fraction * (plot_bottom - plot_top)

    zero_y = y_position(0.0)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill="#333333", width=2)
    draw.text((margin_left, 20), title, fill="#111111", font=font)
    group_width = (plot_right - plot_left) / max(len(labels), 1)
    bar_width = max(2.0, 0.75 * group_width / max(len(series), 1))
    for label_index, label in enumerate(labels):
        center = plot_left + (label_index + 0.5) * group_width
        for series_index, (_, items, color) in enumerate(series):
            value = float(items[label_index])
            left = center + (series_index - (len(series) - 1) / 2.0) * bar_width - bar_width * 0.45
            right = left + bar_width * 0.9
            value_y = y_position(value)
            draw.rectangle((left, min(zero_y, value_y), right, max(zero_y, value_y)), fill=color)
        draw.text((center - 14, plot_bottom + 12), label, fill="#222222", font=font)
    legend_x = plot_left
    for name, _, color in series:
        draw.rectangle((legend_x, height - 35, legend_x + 16, height - 19), fill=color)
        draw.text((legend_x + 22, height - 36), name, fill="#222222", font=font)
        legend_x += 22 + max(70, len(name) * 8)
    draw.text((8, plot_top), f"{maximum:.3f} K", fill="#444444", font=font)
    draw.text((8, plot_bottom - 10), f"{minimum:.3f} K", fill="#444444", font=font)
    image.save(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
