#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_pipeline import FULL_STAGE, PHASE3_STAGE, STAGE_SPECS, read_csv, resolve_data_path, write_csv
from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.graph_models import move_graph_to_device, normalize_graph_batch
from chiptherm.ml.models import build_model
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input


AUDIT_ORDINALS = (1, 6, 9, 10, 7, 8, 2, 3, 4, 5)
FULL_AUDIT_ORDINALS = (11, 121, 151, 31, 41, 102, 132, 162, 192, 200)


def select_audit_rows(
    rows: list[dict[str, str]],
    sample_uids: list[str] | None,
    *,
    stage: str = PHASE3_STAGE,
) -> list[dict[str, str]]:
    if sample_uids:
        by_uid = {row["sample_uid"]: row for row in rows}
        missing = sorted(set(sample_uids) - set(by_uid))
        if missing:
            raise ValueError(f"requested samples are absent from index: {missing}")
        return [by_uid[uid] for uid in sample_uids]
    selected: list[dict[str, str]] = []
    ordinals = FULL_AUDIT_ORDINALS if stage == FULL_STAGE else AUDIT_ORDINALS
    for family_index, family_uid in enumerate(sorted({row["family_uid"] for row in rows})):
        ordinal = ordinals[family_index % len(ordinals)]
        candidates = [row for row in rows if row["family_uid"] == family_uid and int(row["workload_uid"][1:4]) == ordinal]
        if len(candidates) != 1:
            raise ValueError(f"expected one audit row for {family_uid} workload ordinal {ordinal}, got {len(candidates)}")
        selected.append(candidates[0])
    return selected


def checkpoint_prediction(index_path: Path, sample_uid: str, checkpoint_path: Path, device: torch.device) -> np.ndarray:
    dataset = ChipThermDataset(index_path, target="residual", return_metadata=True, return_graph=True)
    sample_index = next(index for index, row in enumerate(dataset.rows) if row["sample_uid"] == sample_uid)
    batch = chiptherm_collate([dataset[sample_index]])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = NormalizationStats(**checkpoint["normalization"])
    model_input = build_model_input(batch["x"].to(device), batch["physics"].to(device), stats, physics_input_mode=str(config["physics_input_mode"]))
    metadata = build_metadata_input(batch["metadata_vector"].to(device), stats)
    graph = normalize_graph_batch(move_graph_to_device(batch["graph"], device), config.get("graph_normalization"))
    kwargs: dict[str, Any] = {"return_diagnostics": True}
    if str(config.get("mean_head_mode", "direct_k")) == "residual_resistance":
        kwargs["total_power_W"] = batch["total_power_W"].to(device)
    with torch.inference_mode():
        output = model(model_input, metadata, graph, **kwargs)
    if isinstance(output, dict) and "final_temperature" in output:
        final = output["final_temperature"]
    elif isinstance(output, dict):
        centered = output["centered_field"] - output["centered_field"].mean(dim=(-2, -1), keepdim=True)
        base = batch["physics"].to(device) if str(config.get("mean_head_mode")) == "residual_resistance" else batch["ambient_K"].to(device)[:, None, None]
        final = base + output["mean_rise"][:, None, None] + centered
    else:
        final = output
    return final[0].detach().cpu().numpy()


def add_full_extreme_rows(
    selected: list[dict[str, str]],
    rows: list[dict[str, str]],
    data_root: Path,
) -> list[dict[str, str]]:
    peak_rows: list[tuple[float, dict[str, str]]] = []
    residual_rows: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        target = np.load(resolve_data_path(row["y_path"], data_root), mmap_mode="r")
        peak_rows.append((float(np.max(target)), row))
        base_value = row.get("source_superposition_base_path", "")
        if base_value:
            base = np.load(resolve_data_path(base_value, data_root), mmap_mode="r")
            residual_rows.append((float(np.mean(np.abs(np.asarray(target) - np.asarray(base)))), row))
    extras = [min(peak_rows, key=lambda item: item[0])[1], max(peak_rows, key=lambda item: item[0])[1]] if peak_rows else []
    extras.extend(row for _, row in sorted(residual_rows, key=lambda item: item[0], reverse=True)[:4])
    seen = {row["sample_uid"] for row in selected}
    output = list(selected)
    for row in extras:
        if row["sample_uid"] not in seen:
            output.append(row)
            seen.add(row["sample_uid"])
    return output


def draw_layout(axis: Any, layout: dict[str, Any], powers: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    package = layout["package"]["size"]
    axis.set_xlim(0, float(package["width"]))
    axis.set_ylim(0, float(package["height"]))
    axis.set_aspect("equal")
    maximum = max(powers.values()) if powers else 1.0
    for chiplet in layout["chiplets"]:
        x = float(chiplet["position"]["x"])
        y = float(chiplet["position"]["y"])
        width = float(chiplet["size"]["width"])
        height = float(chiplet["size"]["height"])
        fraction = powers.get(str(chiplet["name"]), 0.0) / maximum
        axis.add_patch(plt.Rectangle((x, y), width, height, facecolor=plt.cm.inferno(fraction), edgecolor="black", linewidth=0.4))
    axis.set_title("Layout / source power")


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for Benchmark v2 visualization") from exc

    parser = argparse.ArgumentParser(description="Create deterministic Benchmark v2 staged visual audit panels.")
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--index", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-uids", nargs="*", default=None)
    parser.add_argument("--residual-checkpoint", default=None, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", default=PHASE3_STAGE, choices=sorted(STAGE_SPECS))
    parser.add_argument("--rows-per-sheet", default=10, type=int)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    data_root = args.data_root.expanduser().resolve()
    index_path = (args.index or data_root / f"derived/indices/{args.stage}/all_index.csv").resolve()
    all_rows = read_csv(index_path)
    selected = select_audit_rows(all_rows, args.sample_uids, stage=args.stage)
    if args.stage == FULL_STAGE and not args.sample_uids:
        selected = add_full_extreme_rows(selected, all_rows, data_root)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    column_count = 7 if args.residual_checkpoint else 5
    summary_rows: list[dict[str, str]] = []
    if args.rows_per_sheet <= 0:
        raise ValueError("--rows-per-sheet must be positive")
    for page, start in enumerate(range(0, len(selected), args.rows_per_sheet), start=1):
        page_rows = selected[start : start + args.rows_per_sheet]
        contact_fig, contact_axes = plt.subplots(len(page_rows), column_count, figsize=(3.2 * column_count, 3.2 * len(page_rows)), squeeze=False)
        for row_index, row in enumerate(page_rows):
            x = np.load(resolve_data_path(row["x_path"], data_root))
            target = np.load(resolve_data_path(row["y_path"], data_root))
            base = np.load(resolve_data_path(row["source_superposition_base_path"], data_root))
            layout = json.loads(resolve_data_path(row["layout_path"], data_root).read_text(encoding="utf-8"))
            import yaml

            power_doc = yaml.safe_load(resolve_data_path(row["power_path"], data_root).read_text(encoding="utf-8"))
            powers = {str(key): float(value) for key, value in power_doc["workloads"][power_doc.get("active_workload", "nominal")].items()}
            axes = contact_axes[row_index]
            draw_layout(axes[0], layout, powers)
            for axis, array, title, cmap in (
                (axes[1], x[0], "Power density (W/mm2)", "magma"),
                (axes[2], target, "HotSpot target (K)", "inferno"),
                (axes[3], base, "Source-superposition base (K)", "inferno"),
                (axes[4], target - base, "Target - source base (K)", "coolwarm"),
            ):
                image = axis.imshow(array, origin="lower", cmap=cmap)
                axis.set_title(title)
                axis.set_xticks([])
                axis.set_yticks([])
                contact_fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            axes[0].set_ylabel(f"{row['family_uid']}\n{row.get('workload_cell', row['workload_uid'])}")
            if args.residual_checkpoint:
                prediction = checkpoint_prediction(index_path, row["sample_uid"], args.residual_checkpoint, device)
                np.save(out_dir / f"{row['sample_uid']}_prediction.npy", prediction.astype(np.float32))
                for axis, array, title, cmap in (
                    (axes[5], prediction, "Residual-model result (K)", "inferno"),
                    (axes[6], prediction - target, "Prediction error (K)", "coolwarm"),
                ):
                    image = axis.imshow(array, origin="lower", cmap=cmap)
                    axis.set_title(title)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    contact_fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            summary_rows.append({key: row.get(key, "") for key in ("sample_uid", "family_uid", "workload_uid", "workload_cell", "power_regime", "topology_regime")})
        contact_fig.suptitle(f"ChipTherm Benchmark v2 {args.stage} audit set, sheet {page}")
        contact_fig.tight_layout(rect=(0, 0, 1, 0.995))
        contact_fig.savefig(out_dir / f"{args.stage}_contact_sheet_{page:02d}.png", dpi=160)
        plt.close(contact_fig)
    write_csv(out_dir / "audit_samples.csv", summary_rows)
    (out_dir / "visual_audit_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "benchmark_v2_visual_audit/1",
                "stage": args.stage,
                "sample_count": len(summary_rows),
                "sample_uids": [row["sample_uid"] for row in summary_rows],
                "checkpoint_supplied": bool(args.residual_checkpoint),
                "status": "generated_pending_manual_review",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Visualized {len(selected)} samples")
    print(f"Contact sheets: {len(list(out_dir.glob(f'{args.stage}_contact_sheet_*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
