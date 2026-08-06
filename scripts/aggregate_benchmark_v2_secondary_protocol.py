#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


MODELS = ("CNN", "FNO", "ChipTherm")
PROTOCOLS = (
    "familiar_family_sample_test",
    "heldout_validation_families",
    "heldout_final_test_families",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric set")
    maes = [float(row["mae_K"]) for row in rows]
    rmses = [float(row["rmse_K"]) for row in rows]
    signed = [float(row.get("mean_signed_error_K") or 0.0) for row in rows]
    if not all(math.isfinite(value) for value in maes + rmses + signed):
        raise ValueError("sample metrics contain NaN or Inf")
    return {
        "mae_K": statistics.fmean(maes),
        "rmse_K": math.sqrt(statistics.fmean(value * value for value in rmses)),
        "mean_signed_error_K": statistics.fmean(signed),
    }


def family(row: Mapping[str, str]) -> str:
    value = str(row.get("family_uid") or row.get("case_id") or "")
    if not value:
        raise ValueError("sample metric row has no family_uid/case_id")
    return value


def locate_metrics(root: Path, protocol: str) -> Path:
    candidates = (
        root / protocol / "metrics_by_sample.csv",
        root / protocol / "per_sample_metrics.csv",
    )
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected exactly one per-sample metric file for {protocol} under {root}, found {found}"
        )
    return found[0]


def build_tables(model_roots: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    per_family: list[dict[str, Any]] = []
    for model in MODELS:
        root = model_roots[model]
        for protocol in PROTOCOLS:
            rows = read_csv(locate_metrics(root, protocol))
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[family(row)].append(row)
            family_metrics = []
            for family_uid in sorted(grouped):
                metrics = aggregate(grouped[family_uid])
                family_metrics.append(metrics)
                per_family.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "family_uid": family_uid,
                        "sample_count": len(grouped[family_uid]),
                        **metrics,
                    }
                )
            micro = aggregate(rows)
            family_maes = [item["mae_K"] for item in family_metrics]
            summary.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "sample_count": len(rows),
                    "family_count": len(grouped),
                    "sample_weighted_mae_K": micro["mae_K"],
                    "sample_weighted_rmse_K": micro["rmse_K"],
                    "family_weighted_mae_K": statistics.fmean(family_maes),
                    "family_mae_std_K": statistics.pstdev(family_maes),
                    "family_mae_median_K": statistics.median(family_maes),
                    "family_mae_min_K": min(family_maes),
                    "family_mae_max_K": max(family_maes),
                    "mean_signed_error_K": micro["mean_signed_error_K"],
                }
            )
    return summary, per_family


def comparison(summary: Sequence[Mapping[str, Any]], per_family: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    heldout = {
        str(row["model"]): row
        for row in summary
        if row["protocol"] == "heldout_final_test_families"
    }
    chip = float(heldout["ChipTherm"]["sample_weighted_mae_K"])
    wins = {model: 0 for model in MODELS}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_family:
        if row["protocol"] == "heldout_final_test_families":
            grouped[str(row["family_uid"])].append(row)
    for rows in grouped.values():
        best = min(float(row["mae_K"]) for row in rows)
        for row in rows:
            if abs(float(row["mae_K"]) - best) <= 1.0e-12:
                wins[str(row["model"])] += 1
    return {
        "heldout_final_test_family_wins": wins,
        "chiptherm_improvement_vs_fno_percent": 100.0 * (float(heldout["FNO"]["sample_weighted_mae_K"]) - chip) / max(float(heldout["FNO"]["sample_weighted_mae_K"]), 1.0e-12),
        "chiptherm_improvement_vs_cnn_percent": 100.0 * (float(heldout["CNN"]["sample_weighted_mae_K"]) - chip) / max(float(heldout["CNN"]["sample_weighted_mae_K"]), 1.0e-12),
        "sample_weighted_ranking": sorted(MODELS, key=lambda model: float(heldout[model]["sample_weighted_mae_K"])),
        "family_weighted_ranking": sorted(MODELS, key=lambda model: float(heldout[model]["family_weighted_mae_K"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the frozen Benchmark v2 35/5/10 robustness protocol.")
    parser.add_argument("--cnn-eval-root", required=True, type=Path)
    parser.add_argument("--fno-eval-root", required=True, type=Path)
    parser.add_argument("--chiptherm-eval-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = {
        "CNN": args.cnn_eval_root.expanduser().resolve(),
        "FNO": args.fno_eval_root.expanduser().resolve(),
        "ChipTherm": args.chiptherm_eval_root.expanduser().resolve(),
    }
    summary, per_family = build_tables(roots)
    result = comparison(summary, per_family)
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics_summary.csv", summary)
    write_csv(out / "per_family_metrics.csv", per_family)
    (out / "reproducibility_manifest.json").write_text(
        json.dumps({"model_evaluation_roots": {key: str(value) for key, value in roots.items()}, "comparison": result}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Benchmark v2 Secondary 35/5/10 Comparison",
        "",
        "This is a secondary robustness experiment; it does not replace the frozen 40/5/5 result.",
        "",
        "| Model | Protocol | Sample MAE (K) | Family MAE (K) | RMSE (K) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['protocol']} | {float(row['sample_weighted_mae_K']):.6f} | {float(row['family_weighted_mae_K']):.6f} | {float(row['sample_weighted_rmse_K']):.6f} |"
        )
    lines.extend(["", f"Final-test family wins: `{result['heldout_final_test_family_wins']}`"])
    (out / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote secondary-protocol comparison to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
