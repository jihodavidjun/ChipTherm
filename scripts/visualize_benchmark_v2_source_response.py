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


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_training import EXPECTED_PRIMARY_SPLIT, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the signed qualitative audit used by the Benchmark v2 source-checkpoint gate."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="Mark the generated contact sheets as manually reviewed and physically plausible.",
    )
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise SystemExit("matplotlib is required for the qualitative source-response audit") from exc

    data_root = args.data_root.expanduser().resolve()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, str]] = []
    for split_dir in sorted(path for path in evaluation_root.iterdir() if path.is_dir()):
        metrics_path = split_dir / "source_metrics.csv"
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("prediction_rise_path") and row.get("target_rise_saved_path"):
                    row = dict(row)
                    row["evaluation_split"] = split_dir.name
                    candidates.append(row)
    if not candidates:
        raise ValueError(
            "no saved source predictions found; rerun source evaluation with --save-predictions"
        )
    ordered = sorted(candidates, key=lambda row: float(row["physical_mae_K"]))
    heldout = [
        row
        for row in ordered
        if row.get("case_id") in set(EXPECTED_PRIMARY_SPLIT["val"] + EXPECTED_PRIMARY_SPLIT["test"])
    ]
    selected = [
        ("easy", ordered[0]),
        ("median", ordered[len(ordered) // 2]),
        ("difficult", ordered[-1]),
        ("heldout", heldout[len(heldout) // 2] if heldout else ordered[-1]),
    ]
    artifacts: list[dict[str, Any]] = []
    for label, row in selected:
        pred = np.load(row["prediction_rise_path"]).astype(np.float64)
        target = np.load(row["target_rise_saved_path"]).astype(np.float64)
        signed = pred - target
        layout_path = resolve_portable(row["layout_path"], data_root)
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        figure, axes = plt.subplots(1, 5, figsize=(18, 3.6), constrained_layout=True)
        arrays = (target, pred, signed, np.abs(signed))
        titles = ("Target rise (K)", "Prediction rise (K)", "Signed residual (K)", "Absolute residual (K)")
        for axis, array, title in zip(axes[:4], arrays, titles, strict=True):
            image = axis.imshow(array, origin="lower", cmap="coolwarm" if "Signed" in title else "inferno")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, fraction=0.046)
        axes[4].set_title("Package/source geometry")
        chiplets = layout.get("chiplets", [])
        for index, chiplet in enumerate(chiplets):
            position = chiplet.get("position", {})
            size = chiplet.get("size", {})
            x = float(position.get("x", 0.0))
            y = float(position.get("y", 0.0))
            width = float(size.get("width", 0.0))
            height = float(size.get("height", 0.0))
            active = index == int(float(row["source_index"]))
            axes[4].add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor="tab:red" if active else "none",
                    edgecolor="black",
                    linewidth=1.2,
                    alpha=0.65,
                )
            )
            axes[4].text(x + width / 2, y + height / 2, str(index), ha="center", va="center", fontsize=7)
        package = layout.get("package", {})
        package_size = package.get("size", package)
        axes[4].set_xlim(0.0, float(package_size.get("width", 1.0)))
        axes[4].set_ylim(0.0, float(package_size.get("height", 1.0)))
        axes[4].set_aspect("equal")
        figure.suptitle(
            f"{label}: {row['source_response_uid']} | MAE={float(row['physical_mae_K']):.3f} K"
        )
        output = out_dir / f"{label}_{row['source_response_uid']}.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        artifacts.append(
            {
                "category": label,
                "source_response_uid": row["source_response_uid"],
                "family_uid": row.get("case_id", ""),
                "physical_mae_K": float(row["physical_mae_K"]),
                "plot": output.name,
            }
        )
    manifest = {
        "schema_version": "benchmark_v2_source_response_qualitative_audit/1",
        "reviewed": bool(args.reviewed),
        "review_statement": (
            "Representative target/prediction/residual/geometry panels manually accepted."
            if args.reviewed
            else "PENDING MANUAL REVIEW"
        ),
        "artifacts": artifacts,
    }
    write_json(evaluation_root / "qualitative_audit.json", manifest)
    write_json(out_dir / "qualitative_audit.json", manifest)
    print(f"Qualitative audit: {'reviewed' if args.reviewed else 'pending manual review'}")
    return 0


def resolve_portable(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else data_root / path


if __name__ == "__main__":
    raise SystemExit(main())
