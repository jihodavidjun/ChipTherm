#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import (
    EXPECTED_PRIMARY_SPLIT,
    finalize_training_run,
    read_csv,
    sha256_file,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional frozen-CNN Benchmark v2 graph correction experiment.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--cnn-checkpoint", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--config", default=REPO_ROOT / "configs/benchmark_v2_50family/training/optional_gnn_v1.yaml", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise SystemExit("training preflight has not passed")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.data_root.expanduser().resolve()
    cnn_payload = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=False)
    cnn_config = cnn_payload.get("model_config", {})
    expected_cnn = {
        "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion",
        "physics_input_mode": "source_superposition_v1",
        "mean_head_mode": "residual_resistance",
        "physical_representation": "dimensional",
    }
    mismatches = {
        key: {"expected": value, "actual": cnn_config.get(key)}
        for key, value in expected_cnn.items()
        if cnn_config.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"frozen CNN is incompatible with the controlled GNN experiment: {mismatches}")
    cnn_lineage = cnn_payload.get("training_lineage") or {}
    if cnn_lineage.get("source_superposition_version") != args.source_version:
        raise SystemExit(
            "frozen CNN source-superposition lineage differs from --source-version"
        )
    indices = root / f"derived/indices/full_50x200/source_superposition/{args.source_version}/sample_split"
    index_manifest = indices.parent / "index_manifest.json"
    if not all(path.is_file() for path in (indices / "train_index.csv", indices / "val_index.csv", index_manifest)):
        raise SystemExit("validated source-version residual indices are missing")
    counts = {
        "train": len(read_csv(indices / "train_index.csv")),
        "val": len(read_csv(indices / "val_index.csv")),
    }
    if counts != {"train": 6400, "val": 800}:
        raise SystemExit(f"optional GNN must use the same 6400/800 split as its frozen CNN, got {counts}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = args.out_dir / "training_lineage.json"
    write_json(
        lineage_path,
        {
            "schema_version": "benchmark_v2_optional_gnn_training_lineage/1",
            "benchmark_id": "benchmark_v2_50family",
            "stage": "full_50x200",
            "preflight_report_sha256": sha256_file(args.preflight_report),
            "source_superposition_version": args.source_version,
            "source_version_index_manifest_sha256": sha256_file(index_manifest),
            "frozen_cnn_checkpoint_sha256": sha256_file(args.cnn_checkpoint),
            "optimization_family_uids": list(EXPECTED_PRIMARY_SPLIT["train"]),
            "checkpoint_selection_family_uids": list(EXPECTED_PRIMARY_SPLIT["train"]),
            "excluded_primary_val_family_uids": list(EXPECTED_PRIMARY_SPLIT["val"]),
            "excluded_primary_test_family_uids": list(EXPECTED_PRIMARY_SPLIT["test"]),
            "frozen_cnn_required": True,
            "graph_mean_correction": False,
            "reconstruction": "T_final = frozen_T_cnn + zero_mean_graph_correction_K",
        },
    )
    epochs = 2 if args.smoke_test else int(config["epochs"])
    command = [
        sys.executable,
        "scripts/train_residual_cnn.py",
        "--train-index", str(indices / "train_index.csv"),
        "--val-index", str(indices / "val_index.csv"),
        "--out-dir", str(args.out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--base-channels", "32",
        "--model-architecture", "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
        "--metadata-conditioning",
        "--metadata-hidden-dim", "64",
        "--metadata-embedding-dim", "64",
        "--refine-channels", "32",
        "--refine-blocks", "4",
        "--physics-input", "source_superposition_v1",
        "--mean-head-mode", "residual_resistance",
        "--physical-representation", "dimensional",
        "--channel-routing-mode", "dimensional_baseline",
        "--graph-hidden-dim", str(config["graph_hidden_dim"]),
        "--graph-edge-hidden-dim", str(config["graph_edge_hidden_dim"]),
        "--graph-layers", str(config["graph_layers"]),
        "--graph-message-aggregation", str(config["graph_message_aggregation"]),
        "--graph-raster-channels", str(config["graph_raster_channels"]),
        "--graph-halo-decay-mm", str(config["graph_halo_decay_mm"]),
        "--lambda-final", str(config["lambda_final"]),
        "--lambda-mean", str(config["lambda_mean"]),
        "--scheduler", str(config["scheduler"]),
        "--early-stopping-patience", str(config["early_stopping_patience"]),
        "--checkpoint-frequency", str(config["checkpoint_frequency"]),
        "--lineage-manifest", str(lineage_path),
        "--no-graph-mean-correction",
        "--init-checkpoint", str(args.cnn_checkpoint),
        "--freeze-cnn",
        "--device", args.device,
        "--num-workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    if args.resume:
        command.append("--resume")
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        verify_frozen_cnn_unchanged(
            args.cnn_checkpoint,
            args.out_dir / "checkpoints" / "best.pt",
            args.out_dir / "frozen_cnn_audit.json",
        )
        finalize_training_run(
            args.out_dir,
            lineage_path=lineage_path,
            resolved_config={
                "wrapper": vars(args),
                "training": config,
                "command": command,
            },
        )
    return 0


def verify_frozen_cnn_unchanged(
    cnn_checkpoint: Path,
    graph_checkpoint: Path,
    output_path: Path,
) -> None:
    cnn_payload = torch.load(cnn_checkpoint, map_location="cpu", weights_only=False)
    graph_payload = torch.load(graph_checkpoint, map_location="cpu", weights_only=False)
    cnn_state = cnn_payload["model_state_dict"]
    graph_state = graph_payload["model_state_dict"]
    mismatches = []
    for key, value in cnn_state.items():
        graph_key = f"cnn.{key}"
        if graph_key not in graph_state or not torch.equal(value.cpu(), graph_state[graph_key].cpu()):
            mismatches.append(key)
    payload = {
        "schema_version": "benchmark_v2_frozen_cnn_bitwise_audit/1",
        "source_parameter_count": len(cnn_state),
        "matched_parameter_count": len(cnn_state) - len(mismatches),
        "mismatches": mismatches,
        "bitwise_unchanged": not mismatches,
    }
    write_json(output_path, payload)
    if mismatches:
        raise RuntimeError(f"frozen CNN changed during GNN training: {mismatches[:10]}")


if __name__ == "__main__":
    raise SystemExit(main())
