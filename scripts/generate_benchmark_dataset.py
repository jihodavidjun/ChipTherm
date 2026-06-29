#!/usr/bin/env python3
"""For benchmark dataset generation"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chiptherm.paths import hotspot_home
from chiptherm.scenario import load_simulation_input
from chiptherm.validate import validate_simulation_input
from chiptherm.writers import write_manifest
from generate_dataset import SampleResult, _build_dataset_manifest, _load_base_bundle, _mean_or_none, _run_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a grouped benchmark ChipTherm dataset.")
    parser.add_argument("--benchmarks-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--samples-per-case", required=True, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", default=100, type=int)
    parser.add_argument("--perturb-radius-mm", default=3.0, type=float)
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    args = parser.parse_args()

    if args.samples_per_case <= 0:
        raise SystemExit("--samples-per-case must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    benchmarks_root = args.benchmarks_root.resolve()
    out_dir = args.out_dir.resolve()
    case_ids = _select_cases(benchmarks_root, args.cases)
    planned = _plan_cases(
        case_ids=case_ids,
        benchmarks_root=benchmarks_root,
        out_dir=out_dir,
        samples_per_case=args.samples_per_case,
        skip_existing=args.skip_existing,
    )

    if args.dry_run:
        _print_dry_run(planned, args.samples_per_case, args.workers, args.seed, out_dir)
        return 0

    total_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    home = (args.hotspot_home or hotspot_home()).resolve()
    config_template = args.config_template.resolve()

    all_results_by_case: dict[str, list[SampleResult]] = {
        case_id: list(plan["existing_results"]) for case_id, plan in planned.items()
    }
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for case_ordinal, (case_id, plan) in enumerate(planned.items(), start=1):
            base_bundle = _load_base_bundle(plan["scenario_path"])
            case_seed = args.seed + case_ordinal * 100000
            for sample_index in plan["missing_indices"]:
                future = executor.submit(
                    _run_sample,
                    sample_index,
                    case_seed,
                    str(plan["case_out_dir"]),
                    str(plan["scenario_path"]),
                    base_bundle,
                    str(home),
                    str(config_template),
                    args.max_retries,
                    args.perturb_radius_mm,
                )
                futures[future] = case_id

        for future in as_completed(futures):
            case_id = futures[future]
            all_results_by_case[case_id].append(future.result())

    case_manifests = {}
    case_summaries = {}
    for case_ordinal, (case_id, plan) in enumerate(planned.items(), start=1):
        results = sorted(all_results_by_case[case_id], key=lambda item: item.sample_id)
        case_manifest = _build_dataset_manifest(
            results=results,
            base_scenario=plan["scenario_path"],
            requested=args.samples_per_case,
            seed=args.seed + case_ordinal * 100000,
            workers=args.workers,
            total_runtime_s=sum(result.runtime.get("total_s", 0.0) for result in results if result.success),
        )
        case_manifest_path = plan["case_out_dir"] / "dataset_manifest.json"
        write_manifest(case_manifest_path, case_manifest)
        case_manifests[case_id] = case_manifest_path
        case_summaries[case_id] = {
            "requested": args.samples_per_case,
            "successful": case_manifest["number_successful"],
            "failed": case_manifest["number_failed"],
            "manifest": str(case_manifest_path),
        }

    top_manifest = _build_top_manifest(
        planned=planned,
        case_manifests=case_manifests,
        case_summaries=case_summaries,
        samples_per_case=args.samples_per_case,
        seed=args.seed,
        workers=args.workers,
        total_runtime_s=time.perf_counter() - total_start,
        skip_existing=args.skip_existing,
    )
    write_manifest(out_dir / "dataset_manifest.json", top_manifest)

    print("Benchmark dataset generation complete")
    print(f"Cases: {len(planned)}")
    print(f"Requested: {top_manifest['total_requested']}")
    print(f"Successful: {top_manifest['total_successful']}")
    print(f"Failed: {top_manifest['total_failed']}")
    print(f"Total runtime: {top_manifest['total_runtime_s']:.3f} s")
    print(f"Avg HotSpot runtime: {_format_optional_float(top_manifest['average_hotspot_runtime_s'])} s")
    print(f"Avg sample runtime: {_format_optional_float(top_manifest['average_total_sample_runtime_s'])} s")
    print(f"Output: {out_dir}")
    return 0


def _select_cases(benchmarks_root: Path, requested_cases: list[str] | None) -> list[str]:
    if requested_cases is not None:
        case_ids = requested_cases
    else:
        case_ids = sorted(path.name for path in benchmarks_root.glob("case*") if path.is_dir())
    if not case_ids:
        raise SystemExit(f"no benchmark cases found under {benchmarks_root}")
    for case_id in case_ids:
        scenario_path = benchmarks_root / case_id / "scenario.yaml"
        if not scenario_path.exists():
            raise SystemExit(f"missing scenario for {case_id}: {scenario_path}")
        validate_simulation_input(load_simulation_input(scenario_path))
    return case_ids


def _plan_cases(
    *,
    case_ids: list[str],
    benchmarks_root: Path,
    out_dir: Path,
    samples_per_case: int,
    skip_existing: bool,
) -> dict[str, dict[str, Any]]:
    planned: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        scenario_path = (benchmarks_root / case_id / "scenario.yaml").resolve()
        case_out_dir = out_dir / case_id
        existing_results: list[SampleResult] = []
        missing_indices: list[int] = []
        for sample_index in range(1, samples_per_case + 1):
            sample_dir = case_out_dir / f"sample_{sample_index:06d}"
            existing = _load_existing_result(sample_dir)
            if skip_existing and existing is not None and existing.success:
                existing_results.append(existing)
            else:
                missing_indices.append(sample_index)
        planned[case_id] = {
            "scenario_path": scenario_path,
            "case_out_dir": case_out_dir,
            "existing_results": existing_results,
            "missing_indices": missing_indices,
        }
    return planned


def _load_existing_result(sample_dir: Path) -> SampleResult | None:
    manifest_path = sample_dir / "manifest.json"
    temp_path = sample_dir / "parsed" / "temp_layer0.npy"
    if not manifest_path.exists() or not temp_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    output_summary = manifest.get("output_summary")
    if not isinstance(output_summary, dict):
        return None
    return SampleResult(
        sample_id=sample_dir.name,
        success=True,
        run_dir=str(sample_dir),
        reason=None,
        runtime=manifest.get("runtime", {}),
        output_summary=output_summary,
        sampled_power=manifest.get("sampled_power", {}),
        sampled_position=manifest.get("sampled_position", {}),
    )


def _build_top_manifest(
    *,
    planned: dict[str, dict[str, Any]],
    case_manifests: dict[str, Path],
    case_summaries: dict[str, dict[str, Any]],
    samples_per_case: int,
    seed: int,
    workers: int,
    total_runtime_s: float,
    skip_existing: bool,
) -> dict[str, Any]:
    manifests = {case_id: json.loads(path.read_text(encoding="utf-8")) for case_id, path in case_manifests.items()}
    hotspot_times: list[float] = []
    sample_times: list[float] = []
    temp_mins: list[float] = []
    temp_maxs: list[float] = []
    temp_means: list[float] = []
    total_success = 0
    total_failed = 0

    for manifest in manifests.values():
        total_success += int(manifest["number_successful"])
        total_failed += int(manifest["number_failed"])
        for sample in manifest["samples"]:
            runtime = sample.get("runtime", {})
            if sample.get("success") and "hotspot_s" in runtime:
                hotspot_times.append(float(runtime["hotspot_s"]))
            if sample.get("success") and "total_s" in runtime:
                sample_times.append(float(runtime["total_s"]))
        temperature = manifest.get("temperature_K", {})
        if temperature.get("min") is not None:
            temp_mins.append(float(temperature["min"]))
        if temperature.get("max") is not None:
            temp_maxs.append(float(temperature["max"]))
        if temperature.get("mean_of_sample_means") is not None:
            temp_means.append(float(temperature["mean_of_sample_means"]))

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": list(planned),
        "samples_per_case": samples_per_case,
        "seed": seed,
        "workers": workers,
        "skip_existing": skip_existing,
        "total_requested": len(planned) * samples_per_case,
        "total_successful": total_success,
        "total_failed": total_failed,
        "total_runtime_s": total_runtime_s,
        "average_hotspot_runtime_s": _mean_or_none(hotspot_times),
        "average_total_sample_runtime_s": _mean_or_none(sample_times),
        "temperature_K": {
            "min": min(temp_mins) if temp_mins else None,
            "max": max(temp_maxs) if temp_maxs else None,
            "mean_of_case_means": _mean_or_none(temp_means),
        },
        "per_case": case_summaries,
    }


def _print_dry_run(
    planned: dict[str, dict[str, Any]],
    samples_per_case: int,
    workers: int,
    seed: int,
    out_dir: Path,
) -> None:
    print("Benchmark dataset dry run")
    print(f"Output: {out_dir}")
    print(f"Cases: {len(planned)}")
    print(f"Samples per case: {samples_per_case}")
    print(f"Total planned samples: {len(planned) * samples_per_case}")
    print(f"Workers: {workers}")
    print(f"Seed: {seed}")
    for case_id, plan in planned.items():
        existing = len(plan["existing_results"])
        missing = len(plan["missing_indices"])
        print(f"{case_id}: scenario={plan['scenario_path']} out={plan['case_out_dir']} existing={existing} planned={missing}")


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
