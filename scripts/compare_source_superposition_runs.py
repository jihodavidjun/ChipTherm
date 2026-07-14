#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a compact matched-subset report comparing physics-v1 and source-superposition residual runs."
    )
    parser.add_argument("--physics-metrics", required=True, type=Path)
    parser.add_argument("--source-metrics", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--full-data-reference-mae-K", default=2.638, type=float)
    parser.add_argument("--full-data-reference-label", default="full-data physics-v1 CNN+frozen-GNN reference")
    args = parser.parse_args()

    physics = load_json(args.physics_metrics)
    source = load_json(args.source_metrics)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    physics_summary = summarize_run(physics)
    source_summary = summarize_run(source)
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics_metrics": str(args.physics_metrics.resolve()),
        "source_metrics": str(args.source_metrics.resolve()),
        "full_data_reference": {
            "label": args.full_data_reference_label,
            "mae_K": args.full_data_reference_mae_K,
            "note": "Reference is not matched-subset comparable unless separately evaluated on the same source-coverage subset.",
        },
        "physics_v1_matched": physics_summary,
        "source_superposition_matched": source_summary,
        "matched_delta_source_minus_physics": diff_summary(source_summary, physics_summary),
    }
    (out_dir / "matched_source_superposition_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "matched_source_superposition_comparison.md").write_text(
        render_markdown(comparison),
        encoding="utf-8",
    )
    print("Matched comparison report written")
    print(f"Physics-v1 matched MAE: {physics_summary.get('final_mae_K'):.3f} K")
    print(f"Source-base matched MAE: {source_summary.get('final_mae_K'):.3f} K")
    print(f"Delta source - physics: {comparison['matched_delta_source_minus_physics'].get('final_mae_K'):.3f} K")
    print(f"Output: {out_dir}")
    return 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested_float(payload: dict[str, Any], path: list[str]) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current or current[key] is None:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model", {})
    runtime = payload.get("runtime", {})
    return {
        "checkpoint": payload.get("checkpoint"),
        "index": payload.get("index"),
        "num_samples": payload.get("num_samples"),
        "physics_input_mode": model.get("physics_input_mode"),
        "parameter_count": model.get("parameter_count"),
        "final_mae_K": nested_float(payload, ["cnn_final_temperature", "mae_K"]),
        "final_rmse_K": nested_float(payload, ["cnn_final_temperature", "rmse_K"]),
        "centered_field_mae_K": nested_float(payload, ["centered_field", "mae_K"]),
        "mean_rise_mae_K": nested_float(payload, ["mean_rise", "mae_K"]),
        "cnn_only_final_mae_K": nested_float(payload, ["cnn_only_final_temperature", "mae_K"]),
        "graph_improvement_mae_K": nested_float(payload, ["graph_improvement", "mae_K"]),
        "base_mae_K": nested_float(payload, ["physics_baseline", "mae_K"]),
        "chiplet_mean_mae_K": nested_float(payload, ["chiplet_mean_temperature", "mae_K"]),
        "chiplet_peak_mae_K": nested_float(payload, ["chiplet_peak_temperature", "mae_K"]),
        "inter_chiplet_delta_mae_K": nested_float(payload, ["inter_chiplet_delta_T", "mean"]),
        "cnn_runtime_per_sample_s": runtime.get("cnn_runtime_per_sample_s"),
        "end_to_end_runtime_per_sample_s": runtime.get("end_to_end_runtime_per_sample_s"),
        "timing_note": runtime.get("timing_note"),
    }


def diff_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    keys = [
        "final_mae_K",
        "final_rmse_K",
        "centered_field_mae_K",
        "mean_rise_mae_K",
        "cnn_only_final_mae_K",
        "graph_improvement_mae_K",
        "base_mae_K",
        "chiplet_mean_mae_K",
        "chiplet_peak_mae_K",
        "inter_chiplet_delta_mae_K",
        "cnn_runtime_per_sample_s",
        "end_to_end_runtime_per_sample_s",
    ]
    delta: dict[str, float | None] = {}
    for key in keys:
        lhs = left.get(key)
        rhs = right.get(key)
        delta[key] = float(lhs) - float(rhs) if lhs is not None and rhs is not None else None
    return delta


def fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def render_markdown(comparison: dict[str, Any]) -> str:
    physics = comparison["physics_v1_matched"]
    source = comparison["source_superposition_matched"]
    delta = comparison["matched_delta_source_minus_physics"]
    ref = comparison["full_data_reference"]
    rows = [
        ("Final MAE K", "final_mae_K", 3),
        ("Final RMSE K", "final_rmse_K", 3),
        ("Centered-field MAE K", "centered_field_mae_K", 3),
        ("Mean-rise MAE K", "mean_rise_mae_K", 3),
        ("Base-map MAE K", "base_mae_K", 3),
        ("Graph improvement K", "graph_improvement_mae_K", 3),
        ("CNN runtime/sample s", "cnn_runtime_per_sample_s", 6),
        ("Cached E2E runtime/sample s", "end_to_end_runtime_per_sample_s", 6),
    ]
    lines = [
        "# Matched Source-Superposition Comparison",
        "",
        f"Created: {comparison['created_at_utc']}",
        "",
        f"Full-data reference: {ref['label']} = {ref['mae_K']:.3f} K MAE.",
        "",
        "This report compares only the matched source-coverage subset. The full-data reference is contextual, not a direct denominator.",
        "",
        "| Metric | Physics-v1 matched | Source-superposition matched | Source - physics |",
        "|---|---:|---:|---:|",
    ]
    for label, key, precision in rows:
        lines.append(
            f"| {label} | {fmt(physics.get(key), precision)} | {fmt(source.get(key), precision)} | {fmt(delta.get(key), precision)} |"
        )
    lines.extend(
        [
            "",
            f"Physics metrics: `{comparison['physics_metrics']}`",
            "",
            f"Source metrics: `{comparison['source_metrics']}`",
            "",
            "Runtime note: source-superposition cached E2E excludes frozen source-response package inference unless that cost is added separately.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
