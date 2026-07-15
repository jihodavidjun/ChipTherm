#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_extension import (
    DEFAULT_CONFIG_PATH,
    estimate_storage,
    file_sha256,
    generate_valid_sample,
    layout_statistics,
    load_extension_config,
    row_for_sample,
    select_cases,
    validate_sample_sources,
    verify_approval,
    write_audit_reports,
    write_indexes,
    write_sample_sources,
)
from chiptherm.parsers import parse_layer_grid
from chiptherm.paths import hotspot_home
from chiptherm.scenario import load_simulation_input
from chiptherm.writers import read_grid_shape, write_flp, write_hotspot_config, write_ptrace
from chiptherm.writers import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build controlled ChipTherm benchmark-extension source samples.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Plan generation without writing samples.")
    mode.add_argument("--pilot", action="store_true", help="Generate a pilot/smoke extension set.")
    mode.add_argument("--full", action="store_true", help="Generate the full approved extension set.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1")
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-case", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-missing-labels", action="store_true")
    parser.add_argument("--repair-invalid-sources", action="store_true", help="Regenerate durable source files that fail hard source validation.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-layout-attempts", type=int, default=100)
    parser.add_argument("--sample-uids", nargs="+", default=None)
    parser.add_argument("--keep-hotspot-workdirs", action="store_true")
    parser.add_argument("--cleanup-hotspot-workdirs", action="store_true")
    parser.add_argument("--max-storage-gb", type=float, default=None)
    parser.add_argument("--approval-file", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--run-hotspot", action="store_true", help="Run full-package HotSpot labels for generated samples.")
    parser.add_argument("--hotspot-executable", type=Path, default=None)
    parser.add_argument("--hotspot-home", type=Path, default=None)
    parser.add_argument("--hotspot-workers", type=int, default=1)
    parser.add_argument("--hotspot-timeout-s", type=float, default=None)
    parser.add_argument("--worker-benchmark", action="store_true", help="Print worker benchmark commands; does not run repeated benchmarks.")
    parser.add_argument("--config-template", type=Path, default=REPO_ROOT / "configs/hotspot_base.config")
    args = parser.parse_args()

    config = load_extension_config(args.config)
    cases = select_cases(config, args.case_ids)
    samples_per_case = _samples_per_case(args)
    stage = _stage(args, samples_per_case)
    out_dir = (args.out_root / stage).resolve() if args.out_root.name != stage else args.out_root.resolve()
    total_samples = samples_per_case * len(cases)
    storage = estimate_storage(total_samples, include_hotspot_labels=args.run_hotspot)

    if args.max_storage_gb is not None and storage["total_GB_for_requested_mode"] > args.max_storage_gb:
        raise SystemExit(
            f"estimated storage {storage['total_GB_for_requested_mode']:.3f} GB exceeds --max-storage-gb {args.max_storage_gb:.3f}"
        )
    if args.full:
        pilot_root = args.pilot_root or (args.out_root / "pilot")
        verify_approval(pilot_root.resolve(), args.approval_file.resolve() if args.approval_file else None)
        _verify_smoke_gate(args.out_root / "smoke")
    if args.run_hotspot and args.hotspot_workers <= 0:
        raise SystemExit("--hotspot-workers must be positive")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be nonnegative")
    if args.max_layout_attempts <= 0:
        raise SystemExit("--max-layout-attempts must be positive")

    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": stage,
        "config": str(args.config.resolve()),
        "config_hash_sha256": file_sha256(args.config.resolve()),
        "out_dir": str(out_dir),
        "case_ids": [case["case_id"] for case in cases],
        "samples_per_case": samples_per_case,
        "total_samples": total_samples,
        "seed": args.seed,
        "storage_estimate": storage,
        "hotspot_labels": "full_package" if args.run_hotspot else "not_generated_by_this_stage",
        "hotspot_workers": args.hotspot_workers if args.run_hotspot else 0,
    }

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(out_dir / "dry_run_manifest.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.worker_benchmark:
        print(_worker_benchmark_commands(args, out_dir))
        return 0

    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rebased_values = _rebase_stage_indexes(out_dir)
    if args.run_hotspot and rebased_values:
        print(f"Auto-rebased portable path values across active CSVs: {rebased_values}")
    active_index_path, active_index_rows = _load_active_index(out_dir)
    active_by_uid = {row.get("sample_uid", ""): row for row in active_index_rows}
    missing_label_uids = _missing_label_uids_from_index(active_index_rows, out_dir) if args.retry_missing_labels else set()
    requested_uids = set(args.sample_uids or []) | missing_label_uids
    selected_uids = set(requested_uids)
    matched_uids: set[str] = set()
    scheduled_uids: set[str] = set()
    skipped_valid_uids: set[str] = set()
    unscheduled_reasons: dict[str, str] = {}
    if args.run_hotspot:
        print(f"Active index: {active_index_path if active_index_path else 'generated sample grid'}")
        print(f"Requested UIDs: {sorted(requested_uids)}")
        if args.retry_missing_labels:
            print(f"Discovered missing-label UIDs: {sorted(missing_label_uids)}")
    rows = []
    sample_stats = []
    validations = []
    hotspot_jobs: list[dict[str, Any]] = []
    hotspot_results: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_dir = out_dir / case["case_id"]
        for sample_index in range(1, samples_per_case + 1):
            sample_uid = f"benchmark_extension_v1_{case['case_id']}_sample_{sample_index:06d}"
            is_requested = not selected_uids or sample_uid in selected_uids
            if sample_uid in requested_uids:
                matched_uids.add(sample_uid)
            sample_dir = case_dir / f"sample_{sample_index:06d}"
            if args.resume and (sample_dir / "source/scenario.yaml").exists():
                layout, power, benchmark, paths = _load_sample_sources(sample_dir)
            else:
                layout, power, benchmark, attempt_info = generate_valid_sample(
                    case,
                    config["defaults"],
                    sample_index,
                    args.seed,
                    max_attempts=args.max_layout_attempts,
                )
                if int(attempt_info["attempt"]) > 0:
                    print(
                        f"UID {sample_uid}: generated valid source on attempt {attempt_info['attempt']} "
                        f"(candidate_seed={attempt_info['candidate_seed']})"
                    )
                paths = write_sample_sources(
                    sample_dir,
                    sample_uid,
                    layout,
                    power,
                    benchmark,
                    cleanup_hotspot_workdirs=args.cleanup_hotspot_workdirs and not args.keep_hotspot_workdirs,
                )
            stats = layout_statistics(layout, power)
            stats["sample_uid"] = sample_uid
            stats["case_id"] = case["case_id"]
            stats["split"] = case["split_role"]
            validation = validate_sample_sources(paths["scenario_path"], case)
            if not validation["passed"] and args.repair_invalid_sources and is_requested:
                old_stats = stats
                print(f"UID {sample_uid}: invalid source detected")
                print(f"  old achieved metrics: {_format_source_metrics(old_stats)}")
                print(f"  repairing invalid source")
                layout, power, benchmark, paths, repair_info = _repair_invalid_source(
                    sample_dir=sample_dir,
                    sample_uid=sample_uid,
                    case=case,
                    defaults=config["defaults"],
                    sample_index=sample_index,
                    seed=args.seed,
                    max_layout_attempts=args.max_layout_attempts,
                    cleanup_hotspot_workdirs=args.cleanup_hotspot_workdirs and not args.keep_hotspot_workdirs,
                )
                stats = layout_statistics(layout, power)
                stats["sample_uid"] = sample_uid
                stats["case_id"] = case["case_id"]
                stats["split"] = case["split_role"]
                validation = validate_sample_sources(paths["scenario_path"], case)
                print(
                    f"UID {sample_uid}: regenerated successfully on attempt {repair_info['attempt']} "
                    f"(candidate_seed={repair_info['candidate_seed']})"
                )
                print(f"  new achieved metrics: {_format_source_metrics(stats)}")
                if not validation["passed"]:
                    raise SystemExit(f"repaired source for {sample_uid} still fails validation: {validation['problems']}")
            validations.append({"sample_uid": sample_uid, "passed": validation["passed"], "problems": validation["problems"]})
            hotspot_result = _existing_hotspot_status(sample_dir)
            if args.run_hotspot and is_requested:
                active_row = active_by_uid.get(sample_uid, {})
                decision = _schedule_decision(
                    uid=sample_uid,
                    sample_dir=sample_dir,
                    active_row=active_row,
                    out_dir=out_dir,
                    resume=args.resume,
                    max_retries=args.max_retries,
                    source_validation_passed=validation["passed"],
                    source_validation_problems=validation["problems"],
                )
                _print_schedule_decision(decision)
                if not validation["passed"]:
                    unscheduled_reasons[sample_uid] = decision["exact_exclusion_reason"]
                elif decision["final_should_schedule"]:
                    hotspot_jobs.append(
                        {
                            "sample_uid": sample_uid,
                            "case_id": case["case_id"],
                            "scenario_path": str(paths["scenario_path"].resolve()),
                            "sample_dir": str(sample_dir.resolve()),
                        }
                    )
                    scheduled_uids.add(sample_uid)
                    hotspot_result = {"status": "queued", "runtime_s": ""}
                else:
                    unscheduled_reasons[sample_uid] = decision["exact_exclusion_reason"]
                    if decision["resume_skip_reason"]:
                        skipped_valid_uids.add(sample_uid)
            sample_stats.append(stats)
            rows.append(
                row_for_sample(
                    sample_uid=sample_uid,
                    case=case,
                    paths=paths,
                    statistics=stats,
                    stage=stage,
                    hotspot_status=hotspot_result["status"],
                )
            )
            rows[-1]["hotspot_runtime_s"] = hotspot_result["runtime_s"]

    if args.run_hotspot and requested_uids and not matched_uids:
        _write_hotspot_reports(out_dir, [], workers=args.hotspot_workers, executable=Path(args.hotspot_executable or "unresolved"), requested_uids=sorted(requested_uids), matched_uids=[], scheduled_uids=[], skipped_valid_uids=[], unresolved_reasons={uid: "requested UID did not match generated sample grid" for uid in sorted(requested_uids)})
        raise SystemExit(f"requested UIDs matched zero rows: {sorted(requested_uids)}")

    source_report = _write_source_validation_report(out_dir, validations)
    if source_report["failed_count"]:
        failed = [item["sample_uid"] for item in source_report["failed_samples"]]
        print(f"Source validation failures: {source_report['failed_count']} ({failed})")

    if args.run_hotspot:
        print(f"Matched UIDs: {sorted(matched_uids)}")
        print(f"Scheduled UIDs: {sorted(scheduled_uids)}")
        print(f"Skipped-valid UIDs: {sorted(skipped_valid_uids)}")

    executable: Path | None = None
    if args.run_hotspot:
        try:
            executable = resolve_hotspot_executable(args.hotspot_executable, args.hotspot_home)
        except SystemExit as exc:
            message = str(exc)
            for uid in sorted(scheduled_uids or requested_uids):
                unscheduled_reasons.setdefault(uid, f"executable resolution failed: {message}")
            _write_hotspot_reports(
                out_dir,
                [],
                workers=args.hotspot_workers,
                executable=Path(args.hotspot_executable or "unresolved"),
                requested_uids=sorted(requested_uids),
                matched_uids=sorted(matched_uids),
                scheduled_uids=sorted(scheduled_uids),
                skipped_valid_uids=sorted(skipped_valid_uids),
                unresolved_reasons=unscheduled_reasons,
            )
            raise

    if args.run_hotspot and hotspot_jobs:
        assert executable is not None
        print(f"Running HotSpot for {len(hotspot_jobs)} sample(s) with {args.hotspot_workers} worker(s)")
        hotspot_results = _run_hotspot_jobs(
            hotspot_jobs,
            executable=executable,
            config_template=args.config_template.resolve(),
            keep_hotspot_workdirs=args.keep_hotspot_workdirs,
            cleanup_hotspot_workdirs=args.cleanup_hotspot_workdirs,
            timeout_s=args.hotspot_timeout_s,
            workers=args.hotspot_workers,
            max_retries=args.max_retries,
            out_dir=out_dir,
            requested_uids=sorted(requested_uids),
            matched_uids=sorted(matched_uids),
            scheduled_uids=sorted(scheduled_uids),
            skipped_valid_uids=sorted(skipped_valid_uids),
            unresolved_reasons=unscheduled_reasons,
        )
        for row in rows:
            result = hotspot_results.get(row["sample_uid"])
            if result is None:
                result = _existing_hotspot_status(_sample_dir_for_uid(out_dir, row["sample_uid"]))
            if result and result.get("status"):
                row["hotspot_status"] = result["status"]
                row["hotspot_runtime_s"] = result.get("runtime_s", "")
                sample_dir = _sample_dir_for_uid(out_dir, row["sample_uid"])
                y_path = sample_dir / "parsed/temp_layer0.npy"
                row["y_path"] = str(y_path.resolve().relative_to(REPO_ROOT)) if y_path.exists() else ""
    elif args.run_hotspot:
        assert executable is not None
        _write_hotspot_reports(
            out_dir,
            [],
            workers=args.hotspot_workers,
            executable=executable,
            requested_uids=sorted(requested_uids),
            matched_uids=sorted(matched_uids),
            scheduled_uids=[],
            skipped_valid_uids=sorted(skipped_valid_uids),
            unresolved_reasons=unscheduled_reasons,
        )
        for row in rows:
            sample_dir = _sample_dir_for_uid(out_dir, row["sample_uid"])
            status = _existing_hotspot_status(sample_dir)
            row["hotspot_status"] = status["status"]
            row["hotspot_runtime_s"] = status.get("runtime_s", "")
            y_path = sample_dir / "parsed/temp_layer0.npy"
            row["y_path"] = str(y_path.resolve().relative_to(REPO_ROOT)) if y_path.exists() else ""

    write_indexes(out_dir, rows)
    manifest = write_audit_reports(
        out_dir,
        rows,
        sample_stats,
        stage=stage,
        validation=validations,
        config_hash=file_sha256(args.config.resolve()),
    )
    manifest["runtime_s"] = time.perf_counter() - start
    if args.run_hotspot:
        manifest["hotspot_generation"] = _summarize_hotspot_generation(out_dir, rows)
        manifest["hotspot_executable_provenance"] = str(executable) if executable is not None else ""
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"Generated ChipTherm extension {stage}: {len(rows)} samples")
    print(f"Output: {out_dir}")
    print(f"Validation passed: {manifest['validation']['passed']}")
    if args.run_hotspot:
        unresolved = [row["sample_uid"] for row in rows if row.get("hotspot_status") != "full_package_done"]
        print(f"Unresolved UIDs: {sorted(unresolved)}")
        if unresolved:
            print(f"Unresolved HotSpot labels: {len(unresolved)}")
            return 3
    return 0 if manifest["validation"]["passed"] else 2


def _hotspot_worker(
    job: dict[str, Any],
    *,
    executable: str,
    config_template: str,
    keep_hotspot_workdirs: bool,
    cleanup_hotspot_workdirs: bool,
    timeout_s: float | None,
) -> dict[str, Any]:
    scenario_path = Path(job["scenario_path"])
    sample_dir = Path(job["sample_dir"])
    sample_uid = str(job["sample_uid"])
    hotspot_dir = sample_dir / "hotspot"
    outputs_dir = sample_dir / "outputs"
    parsed_dir = sample_dir / "parsed"
    for path in (hotspot_dir, outputs_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)
    result_payload: dict[str, Any] = {
        "sample_uid": sample_uid,
        "case_id": job["case_id"],
        "status": "failed",
        "failure_category": "",
        "runtime_s": "",
        "return_code": "",
        "stdout_tail": "",
        "stderr_tail": "",
        "output_files": {},
        "parse_status": "",
        "shape": "",
        "finite": False,
        "temperature_min_K": "",
        "temperature_max_K": "",
        "cleanup_status": "",
        "command": "",
    }
    start = time.perf_counter()
    try:
        if not Path(executable).exists():
            raise FileNotFoundError(executable)
        sim = load_simulation_input(scenario_path)
        flp_path = write_flp(sim.layout, hotspot_dir / "chiplet.flp")
        ptrace_path = write_ptrace(sim.layout, sim.power, hotspot_dir / "power.ptrace")
        config_path = write_hotspot_config(config_template, hotspot_dir / "hotspot.config", sim.package, sim.hotspot)
        rows, cols = read_grid_shape(config_path)
        block_steady_path = outputs_dir / "block.steady"
        grid_steady_path = outputs_dir / "grid.steady"
        command = [
            executable,
            "-c",
            str(config_path.resolve()),
            "-f",
            str(flp_path.resolve()),
            "-p",
            str(ptrace_path.resolve()),
            "-model_type",
            "grid",
            "-steady_file",
            str(block_steady_path.resolve()),
            "-grid_steady_file",
            str(grid_steady_path.resolve()),
        ]
        result_payload["command"] = shlex.join(command)
        (sample_dir / "command.txt").write_text(result_payload["command"] + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(command, cwd=hotspot_dir, check=False, text=True, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            result_payload["failure_category"] = "timeout"
            result_payload["stdout_tail"] = _tail(exc.stdout or "")
            result_payload["stderr_tail"] = _tail(exc.stderr or "")
            raise
        (outputs_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (outputs_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        result_payload["return_code"] = completed.returncode
        result_payload["stdout_tail"] = _tail(completed.stdout)
        result_payload["stderr_tail"] = _tail(completed.stderr)
        result_payload["output_files"] = {
            "block_steady": block_steady_path.exists(),
            "grid_steady": grid_steady_path.exists(),
        }
        if completed.returncode != 0:
            result_payload["failure_category"] = "nonzero_return_code"
            return _finish_hotspot_worker(sample_dir, result_payload, start)
        if not grid_steady_path.exists():
            result_payload["failure_category"] = "missing_output"
            return _finish_hotspot_worker(sample_dir, result_payload, start)
        try:
            layer0 = parse_layer_grid(grid_steady_path, layer=0, rows=rows, cols=cols)
            result_payload["parse_status"] = "ok"
        except Exception as exc:
            result_payload["failure_category"] = "parse_failure"
            result_payload["parse_status"] = str(exc)
            return _finish_hotspot_worker(sample_dir, result_payload, start)
        result_payload["shape"] = list(layer0.shape)
        if layer0.shape != (rows, cols):
            result_payload["failure_category"] = "shape_failure"
            return _finish_hotspot_worker(sample_dir, result_payload, start)
        if not bool(np.isfinite(layer0).all()):
            result_payload["failure_category"] = "nonfinite_result"
            return _finish_hotspot_worker(sample_dir, result_payload, start)
        tmp_path = parsed_dir / "temp_layer0.tmp.npy"
        final_path = parsed_dir / "temp_layer0.npy"
        np.save(tmp_path, layer0)
        try:
            tmp_path.replace(final_path)
            reopened = np.load(final_path)
            if reopened.shape != layer0.shape or not np.isfinite(reopened).all():
                result_payload["failure_category"] = "atomic_write_failure"
                return _finish_hotspot_worker(sample_dir, result_payload, start)
        except Exception:
            result_payload["failure_category"] = "atomic_write_failure"
            return _finish_hotspot_worker(sample_dir, result_payload, start)
        result_payload.update(
            {
                "status": "full_package_done",
                "failure_category": "",
                "finite": True,
                "temperature_min_K": float(layer0.min()),
                "temperature_max_K": float(layer0.max()),
            }
        )
        if cleanup_hotspot_workdirs and not keep_hotspot_workdirs:
            try:
                for generated in (flp_path, ptrace_path, config_path):
                    if generated.exists():
                        generated.unlink()
                result_payload["cleanup_status"] = "generated_inputs_removed"
            except Exception as exc:
                result_payload["cleanup_status"] = f"cleanup_failure: {exc}"
                result_payload["status"] = "failed"
                result_payload["failure_category"] = "cleanup_failure"
        else:
            result_payload["cleanup_status"] = "preserved"
        return _finish_hotspot_worker(sample_dir, result_payload, start)
    except FileNotFoundError as exc:
        result_payload["failure_category"] = "executable_error"
        result_payload["stderr_tail"] = str(exc)
        return _finish_hotspot_worker(sample_dir, result_payload, start)
    except subprocess.TimeoutExpired:
        return _finish_hotspot_worker(sample_dir, result_payload, start)
    except Exception as exc:
        result_payload["failure_category"] = result_payload["failure_category"] or "parse_failure"
        result_payload["stderr_tail"] = str(exc)
        return _finish_hotspot_worker(sample_dir, result_payload, start)


def _samples_per_case(args: argparse.Namespace) -> int:
    if args.samples_per_case is not None:
        return args.samples_per_case
    if args.full:
        return 400
    return 50


def _load_sample_sources(sample_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    import yaml

    paths = _source_paths(sample_dir)
    layout = json.loads(paths["layout_path"].read_text(encoding="utf-8"))
    power = yaml.safe_load(paths["power_path"].read_text(encoding="utf-8")) or {}
    benchmark = {}
    if paths["benchmark_path"].exists():
        benchmark = yaml.safe_load(paths["benchmark_path"].read_text(encoding="utf-8")) or {}
    return layout, power, benchmark, paths


def _source_paths(sample_dir: Path) -> dict[str, Path]:
    source_dir = sample_dir / "source"
    return {
        "source_dir": source_dir,
        "scenario_path": source_dir / "scenario.yaml",
        "layout_path": source_dir / "layout.json",
        "power_path": source_dir / "power.yaml",
        "package_path": source_dir / "package.yaml",
        "hotspot_path": source_dir / "hotspot.yaml",
        "benchmark_path": source_dir / "benchmark.yaml",
        "y_path": sample_dir / "parsed/temp_layer0.npy",
    }


def _repair_invalid_source(
    *,
    sample_dir: Path,
    sample_uid: str,
    case: dict[str, Any],
    defaults: dict[str, Any],
    sample_index: int,
    seed: int,
    max_layout_attempts: int,
    cleanup_hotspot_workdirs: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    layout, power, benchmark, attempt_info = generate_valid_sample(
        case,
        defaults,
        sample_index,
        seed,
        max_attempts=max_layout_attempts,
    )
    tmp_sample_dir = sample_dir.parent / f"{sample_dir.name}.__repair_tmp__"
    if tmp_sample_dir.exists():
        shutil.rmtree(tmp_sample_dir)
    try:
        tmp_paths = write_sample_sources(tmp_sample_dir, sample_uid, layout, power, benchmark)
        validation = validate_sample_sources(tmp_paths["scenario_path"], case)
        if not validation["passed"]:
            raise RuntimeError(f"temporary repaired source failed validation: {validation['problems']}")
        final_source = sample_dir / "source"
        final_source.mkdir(parents=True, exist_ok=True)
        for filename in ("scenario.yaml", "layout.json", "power.yaml", "package.yaml", "hotspot.yaml", "benchmark.yaml"):
            (tmp_sample_dir / "source" / filename).replace(final_source / filename)
    finally:
        if tmp_sample_dir.exists():
            shutil.rmtree(tmp_sample_dir)
    if cleanup_hotspot_workdirs:
        hotspot_dir = sample_dir / "hotspot"
        if hotspot_dir.exists():
            for child in hotspot_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
    _invalidate_stale_label(sample_dir)
    return layout, power, benchmark, _source_paths(sample_dir), attempt_info


def _invalidate_stale_label(sample_dir: Path) -> None:
    for path in (
        sample_dir / "parsed/temp_layer0.npy",
        sample_dir / "parsed/temp_layer0.tmp.npy",
        sample_dir / "manifest.json",
    ):
        if path.exists():
            path.unlink()


def _format_source_metrics(stats: dict[str, Any]) -> str:
    keys = (
        "whitespace_fraction",
        "occupied_fraction",
        "min_pairwise_edge_distance_mm",
        "mean_power_density_W_per_mm2",
        "max_power_density_W_per_mm2",
        "total_power_W",
    )
    return ", ".join(f"{key}={float(stats[key]):.6g}" for key in keys if key in stats)


def _write_source_validation_report(out_dir: Path, validations: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item for item in validations if not item.get("passed")]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(validations),
        "failed_count": len(failed),
        "passed": not failed,
        "failed_samples": failed,
    }
    (out_dir / "source_validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["sample_uid", "passed", "problems"]
    rows = [
        {
            "sample_uid": item.get("sample_uid", ""),
            "passed": item.get("passed", False),
            "problems": "; ".join(item.get("problems", [])),
        }
        for item in validations
    ]
    _write_csv(out_dir / "source_validation_failures.csv", [row for row in rows if not row["passed"]], fields)
    return report


def _stage(args: argparse.Namespace, samples_per_case: int) -> str:
    if args.dry_run:
        return "dry_run"
    if args.full:
        return "full"
    if samples_per_case <= 5:
        return "smoke"
    return "pilot"


def resolve_hotspot_executable(explicit: Path | None, hotspot_home_arg: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_home = os.environ.get("HOTSPOT_HOME")
    if env_home:
        candidates.append(Path(env_home).expanduser() / "hotspot")
    if hotspot_home_arg is not None:
        candidates.append(hotspot_home_arg.expanduser() / "hotspot")
    try:
        candidates.append(hotspot_home() / "hotspot")
    except Exception:
        pass
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    tried = "\n".join(f"  - {candidate}" for candidate in candidates) or "  - none"
    raise SystemExit("could not resolve HotSpot executable. Tried:\n" + tried)


def _verify_smoke_gate(smoke_root: Path) -> None:
    report_path = smoke_root / "validation_report.json"
    if not report_path.exists():
        raise SystemExit(f"full generation requires labeled smoke validation: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed") or not report.get("require_hotspot_labels"):
        raise SystemExit("full generation requires smoke validation to pass with --require-hotspot-labels")


PATH_COLUMNS = [
    "source_dir",
    "scenario_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "benchmark_path",
    "x_path",
    "y_path",
    "prediction_path",
    "residual_path",
    "graph_path",
    "source_superposition_base_path",
    "source_superposition_residual_path",
]


def _load_active_index(out_dir: Path) -> tuple[Path | None, list[dict[str, str]]]:
    for name in ("all_extension_index.csv", "combined_encoded_index.csv"):
        path = out_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as fp:
                return path, list(csv.DictReader(fp))
    return None, []


def _missing_label_uids_from_index(rows: list[dict[str, str]], out_dir: Path) -> set[str]:
    missing = set()
    for row in rows:
        uid = row.get("sample_uid", "")
        if not uid:
            continue
        y_value = row.get("y_path", "")
        y_path = _resolve_index_path(y_value, out_dir) if y_value else None
        sample_dir = _sample_dir_for_uid(out_dir, uid)
        if not y_path or not y_path.exists() or not _label_valid(sample_dir):
            missing.add(uid)
    return missing


def _rebase_stage_indexes(out_dir: Path) -> int:
    changed = 0
    for name in (
        "all_extension_index.csv",
        "combined_encoded_index.csv",
        "train_index.csv",
        "val_index.csv",
        "test_index.csv",
    ):
        path = out_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        if not rows:
            continue
        row_changed = False
        for row in rows:
            for column in PATH_COLUMNS:
                value = row.get(column, "")
                if not value:
                    continue
                new_value = _portable_index_path(value)
                if new_value != value:
                    row[column] = new_value
                    changed += 1
                    row_changed = True
        if row_changed:
            _write_csv(path, rows, fieldnames or list(rows[0].keys()))
    return changed


def _portable_index_path(value: str) -> str:
    if not value:
        return value
    text = value
    if text.startswith("/"):
        for marker in ("/data/runs/", "/outputs/"):
            if marker in text:
                return marker.strip("/") + "/" + text.split(marker, 1)[1]
    return text


def _resolve_index_path(value: str, out_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (REPO_ROOT / path, out_dir / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def _schedule_decision(
    *,
    uid: str,
    sample_dir: Path,
    active_row: dict[str, str],
    out_dir: Path,
    resume: bool,
    max_retries: int,
    source_validation_passed: bool,
    source_validation_problems: list[str],
) -> dict[str, Any]:
    y_path_string = active_row.get("y_path", "")
    resolved_y_path = _resolve_index_path(y_path_string, out_dir) if y_path_string else sample_dir / "parsed/temp_layer0.npy"
    y_path_exists = resolved_y_path.exists()
    durable_valid = _label_valid_path(resolved_y_path)
    manifest_path = sample_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    prior_status = str(manifest.get("hotspot_status") or manifest.get("status") or "none") if manifest else "none"
    failure_record_presence = bool(manifest and (manifest.get("success") is False or manifest.get("failure_category")))
    retry_count = int(manifest.get("attempts", 0) or 0) if manifest else 0
    retry_exhausted = retry_count > max_retries
    resume_skip_reason = "valid durable label exists and --resume was set" if resume and durable_valid else ""
    if not source_validation_passed:
        should_schedule = False
        reason = "source validation failed: " + "; ".join(source_validation_problems)
    elif retry_exhausted:
        should_schedule = False
        reason = f"retry count {retry_count} exceeds max retries {max_retries}"
    elif durable_valid:
        should_schedule = False
        reason = resume_skip_reason or "valid durable label exists"
    else:
        should_schedule = True
        reason = "scheduled because durable label is missing"
    return {
        "uid": uid,
        "y_path_string": y_path_string,
        "resolved_y_path": str(resolved_y_path),
        "path_y_exists": y_path_exists,
        "sample_directory_exists": sample_dir.exists(),
        "source_directory_exists": (sample_dir / "source").exists(),
        "prior_status": prior_status,
        "failure_record_presence": failure_record_presence,
        "retry_count": retry_count,
        "retry_exhausted": retry_exhausted,
        "resume_skip_reason": resume_skip_reason,
        "final_should_schedule": should_schedule,
        "exact_exclusion_reason": "" if should_schedule else reason,
        "schedule_reason": reason if should_schedule else "",
    }


def _print_schedule_decision(decision: dict[str, Any]) -> None:
    uid = decision["uid"]
    if decision["final_should_schedule"]:
        print(f"UID {uid}: {decision['schedule_reason']}")
    else:
        print(f"UID {uid}: not scheduled because {decision['exact_exclusion_reason']}")
    for key in (
        "y_path_string",
        "resolved_y_path",
        "path_y_exists",
        "sample_directory_exists",
        "source_directory_exists",
        "prior_status",
        "failure_record_presence",
        "retry_count",
        "retry_exhausted",
        "resume_skip_reason",
        "final_should_schedule",
        "exact_exclusion_reason",
    ):
        print(f"  {key}: {decision[key]}")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"success": False, "failure_category": "manifest_parse_failure"}


def _label_valid(sample_dir: Path) -> bool:
    return _label_valid_path(sample_dir / "parsed/temp_layer0.npy")


def _label_valid_path(y_path: Path) -> bool:
    if not y_path.exists():
        return False
    try:
        arr = np.load(y_path)
    except Exception:
        return False
    return arr.shape == (64, 64) and bool(np.isfinite(arr).all())


def _existing_hotspot_status(sample_dir: Path) -> dict[str, str]:
    if _label_valid(sample_dir):
        runtime = ""
        manifest_path = sample_dir / "manifest.json"
        if manifest_path.exists():
            try:
                runtime = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("runtime_s", ""))
            except Exception:
                runtime = ""
        return {"status": "full_package_done", "runtime_s": runtime}
    manifest_path = sample_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("hotspot_status") == "failed" or manifest.get("success") is False:
                return {"status": "failed", "runtime_s": str(manifest.get("runtime_s", ""))}
        except Exception:
            return {"status": "failed", "runtime_s": ""}
    return {"status": "not_run", "runtime_s": ""}


def _run_hotspot_jobs(
    jobs: list[dict[str, Any]],
    *,
    executable: Path,
    config_template: Path,
    keep_hotspot_workdirs: bool,
    cleanup_hotspot_workdirs: bool,
    timeout_s: float | None,
    workers: int,
    max_retries: int,
    out_dir: Path,
    requested_uids: list[str],
    matched_uids: list[str],
    scheduled_uids: list[str],
    skipped_valid_uids: list[str],
    unresolved_reasons: dict[str, str],
) -> dict[str, dict[str, Any]]:
    remaining = list(jobs)
    all_results: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {job["sample_uid"]: 0 for job in jobs}
    completed_total = 0
    runtime_values: list[float] = []
    while remaining:
        batch = remaining
        remaining = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _hotspot_worker,
                    job,
                    executable=str(executable),
                    config_template=str(config_template),
                    keep_hotspot_workdirs=keep_hotspot_workdirs,
                    cleanup_hotspot_workdirs=cleanup_hotspot_workdirs,
                    timeout_s=timeout_s,
                ): job
                for job in batch
            }
            try:
                for future in as_completed(futures):
                    job = futures[future]
                    attempts[job["sample_uid"]] += 1
                    result = future.result()
                    result["attempts"] = attempts[job["sample_uid"]]
                    all_results[job["sample_uid"]] = result
                    completed_total += 1
                    if result.get("runtime_s") not in ("", None):
                        try:
                            runtime_values.append(float(result["runtime_s"]))
                        except Exception:
                            pass
                    if result.get("status") != "full_package_done" and attempts[job["sample_uid"]] <= max_retries:
                        remaining.append(job)
                    mean_runtime = sum(runtime_values) / len(runtime_values) if runtime_values else 0.0
                    pending = len(futures) - completed_total + len(remaining)
                    eta = mean_runtime * pending / max(workers, 1) if mean_runtime else 0.0
                    successes = sum(1 for item in all_results.values() if item.get("status") == "full_package_done")
                    failures = sum(1 for item in all_results.values() if item.get("status") != "full_package_done")
                    print(
                        f"HotSpot progress: completed={completed_total}/{len(jobs)} "
                        f"success={successes} failed={failures} retried={sum(max(0, v - 1) for v in attempts.values())} "
                        f"workers={workers} mean_runtime={mean_runtime:.2f}s ETA={eta:.1f}s",
                        flush=True,
                    )
            except KeyboardInterrupt:
                executor.shutdown(cancel_futures=True)
                raise
    _write_hotspot_reports(
        out_dir,
        list(all_results.values()),
        workers=workers,
        executable=executable,
        requested_uids=requested_uids,
        matched_uids=matched_uids,
        scheduled_uids=scheduled_uids,
        skipped_valid_uids=skipped_valid_uids,
        unresolved_reasons=unresolved_reasons,
    )
    return all_results


def _finish_hotspot_worker(sample_dir: Path, payload: dict[str, Any], start: float) -> dict[str, Any]:
    payload["runtime_s"] = float(time.perf_counter() - start)
    write_manifest(
        sample_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "success": payload["status"] == "full_package_done",
            "hotspot_status": payload["status"],
            "failure_category": payload.get("failure_category", ""),
            "runtime_s": payload["runtime_s"],
            "command": payload.get("command", ""),
            "return_code": payload.get("return_code", ""),
            "stdout_tail": payload.get("stdout_tail", ""),
            "stderr_tail": payload.get("stderr_tail", ""),
            "output_files": payload.get("output_files", {}),
            "parse_status": payload.get("parse_status", ""),
            "shape": payload.get("shape", ""),
            "finite": payload.get("finite", False),
            "temperature_min_K": payload.get("temperature_min_K", ""),
            "temperature_max_K": payload.get("temperature_max_K", ""),
            "cleanup_status": payload.get("cleanup_status", ""),
        },
    )
    return payload


def _write_hotspot_reports(
    out_dir: Path,
    results: list[dict[str, Any]],
    *,
    workers: int,
    executable: Path,
    requested_uids: list[str] | None = None,
    matched_uids: list[str] | None = None,
    scheduled_uids: list[str] | None = None,
    skipped_valid_uids: list[str] | None = None,
    unresolved_reasons: dict[str, str] | None = None,
) -> None:
    results = sorted(results, key=lambda item: item["sample_uid"])
    failures = [item for item in results if item.get("status") != "full_package_done"]
    write_manifest(
        out_dir / "hotspot_generation_report.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workers": workers,
            "hotspot_executable": str(executable),
            "total": len(results),
            "successful": len(results) - len(failures),
            "failed": len(failures),
            "requested_uids": requested_uids or [],
            "matched_uids": matched_uids or [],
            "scheduled_uids": scheduled_uids or [],
            "skipped_valid_uids": skipped_valid_uids or [],
            "unresolved_reasons": unresolved_reasons or {},
            "results": results,
        },
    )
    fields = [
        "sample_uid",
        "case_id",
        "status",
        "failure_category",
        "runtime_s",
        "return_code",
        "stdout_tail",
        "stderr_tail",
        "parse_status",
        "cleanup_status",
        "unresolved_reason",
    ]
    failure_rows = list(failures)
    for uid, reason in sorted((unresolved_reasons or {}).items()):
        if uid not in {row.get("sample_uid") for row in failure_rows}:
            failure_rows.append({"sample_uid": uid, "status": "not_scheduled", "failure_category": "not_scheduled", "unresolved_reason": reason})
    _write_csv(out_dir / "hotspot_failures.csv", failure_rows, fields)
    by_case: dict[str, list[float]] = {}
    for item in results:
        try:
            by_case.setdefault(str(item["case_id"]), []).append(float(item["runtime_s"]))
        except Exception:
            pass
    runtime_rows = []
    for case_id, values in sorted(by_case.items()):
        runtime_rows.append(
            {
                "case_id": case_id,
                "count": len(values),
                "mean_runtime_s": sum(values) / len(values),
                "min_runtime_s": min(values),
                "max_runtime_s": max(values),
            }
        )
    _write_csv(out_dir / "hotspot_runtime_by_case.csv", runtime_rows, ["case_id", "count", "mean_runtime_s", "min_runtime_s", "max_runtime_s"])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _summarize_hotspot_generation(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {}
    for row in rows:
        statuses[row["hotspot_status"]] = statuses.get(row["hotspot_status"], 0) + 1
    failures_path = out_dir / "hotspot_failures.csv"
    return {"status_counts": statuses, "failure_report": str(failures_path.relative_to(REPO_ROOT)) if failures_path.exists() else ""}


def _sample_dir_for_uid(out_dir: Path, sample_uid: str) -> Path:
    prefix, sample_number = sample_uid.rsplit("_sample_", 1)
    case_id = prefix.rsplit("_", 1)[1]
    return out_dir / case_id / f"sample_{int(sample_number):06d}"


def _tail(text: str, limit: int = 4000) -> str:
    return str(text)[-limit:]


def _worker_benchmark_commands(args: argparse.Namespace, out_dir: Path) -> str:
    base = [
        "python3",
        "scripts/build_chiptherm_extension.py",
        "--pilot",
        "--samples-per-case",
        str(_samples_per_case(args)),
        "--out-root",
        str(args.out_root),
        "--seed",
        str(args.seed),
        "--run-hotspot",
        "--resume",
        "--retry-failed",
        "--max-retries",
        str(args.max_retries),
    ]
    lines = ["# Worker scaling commands; run on the same smoke/pilot root for comparable data."]
    for workers in (1, 4, 8, 12, 16):
        lines.append(shlex.join([*base, "--hotspot-workers", str(workers)]))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
