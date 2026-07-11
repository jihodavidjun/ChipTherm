#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-case ChipTherm upper-bound experiment metrics.")
    parser.add_argument("--results-root", default=REPO_ROOT / "outputs/per_case_upper_bound", type=Path)
    parser.add_argument(
        "--splits-root",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v2_clean_impedance_per_case/package_plus_power",
        type=Path,
    )
    parser.add_argument("--out-csv", default=None, type=Path)
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    splits_root = args.splits_root.expanduser().resolve()
    out_csv = (args.out_csv.expanduser().resolve() if args.out_csv else results_root / "summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in splits_root.iterdir() if path.is_dir() and path.name.startswith("case")):
        case_id = case_dir.name
        train_count = count_index_rows(case_dir / "train_index.csv")
        val_count = count_index_rows(case_dir / "val_index.csv")
        test_count = count_index_rows(case_dir / "test_index.csv")
        result_dir = results_root / case_id
        val_metrics = load_optional_json(result_dir / "val_metrics.json")
        test_metrics = load_optional_json(result_dir / "test_eval_e2e" / "metrics.json")
        record = {
            "case": case_id,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "best_val_mae_K": nested_float(val_metrics, ["metrics", "final_temperature", "mae_K"]),
            "test_mae_K": nested_float(test_metrics, ["cnn_final_temperature", "mae_K"]),
            "test_rmse_K": nested_float(test_metrics, ["cnn_final_temperature", "rmse_K"]),
            "hotspot_mae_K": abs_or_blank(nested_float(test_metrics, ["cnn_final_temperature", "hotspot_temp_error_K"])),
            "runtime_per_sample_s": nested_float(test_metrics, ["runtime", "end_to_end_runtime_per_sample_s"])
            or nested_float(test_metrics, ["inference_runtime_per_sample_s"]),
            "parameter_count": nested_float(test_metrics, ["model", "parameter_count"]),
        }
        records.append(record)

    write_csv(out_csv, records)
    valid_mae = [float(record["test_mae_K"]) for record in records if record["test_mae_K"] != ""]
    weighted_terms = [
        (float(record["test_mae_K"]), int(record["test_count"]))
        for record in records
        if record["test_mae_K"] != ""
    ]
    unweighted = sum(valid_mae) / len(valid_mae) if valid_mae else None
    weighted = sum(mae * count for mae, count in weighted_terms) / sum(count for _, count in weighted_terms) if weighted_terms else None
    best = min((record for record in records if record["test_mae_K"] != ""), key=lambda record: float(record["test_mae_K"]), default=None)
    worst = max((record for record in records if record["test_mae_K"] != ""), key=lambda record: float(record["test_mae_K"]), default=None)

    summary = {
        "num_cases": len(records),
        "num_completed_cases": len(valid_mae),
        "unweighted_mean_per_case_mae_K": unweighted,
        "sample_weighted_global_mae_K": weighted,
        "best_case": best,
        "worst_case": worst,
        "summary_csv": repo_relative(out_csv),
    }
    (out_csv.parent / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Per-case result aggregation complete")
    print(f"Cases found: {len(records)}")
    print(f"Completed cases: {len(valid_mae)}")
    print(f"Unweighted mean per-case MAE: {unweighted:.3f} K" if unweighted is not None else "Unweighted mean per-case MAE: n/a")
    print(f"Sample-weighted global MAE: {weighted:.3f} K" if weighted is not None else "Sample-weighted global MAE: n/a")
    if best:
        print(f"Best case: {best['case']} ({float(best['test_mae_K']):.3f} K)")
    if worst:
        print(f"Worst case: {worst['case']} ({float(worst['test_mae_K']):.3f} K)")
    print(f"Output: {out_csv}")
    return 0


def count_index_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fp:
        return max(0, sum(1 for _ in fp) - 1)


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_float(payload: dict[str, Any], keys: list[str]) -> float | str:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return ""
        value = value[key]
    if value is None or value == "":
        return ""
    return float(value)


def abs_or_blank(value: float | str) -> float | str:
    if value == "":
        return ""
    return abs(float(value))


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "case",
        "train_count",
        "val_count",
        "test_count",
        "best_val_mae_K",
        "test_mae_K",
        "test_rmse_K",
        "hotspot_mae_K",
        "runtime_per_sample_s",
        "parameter_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
