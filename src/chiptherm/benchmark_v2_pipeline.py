from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .benchmark_v2 import BENCHMARK_ID, validate_family_spec
from .benchmark_v2_workloads import load_family, validate_workload, write_workload_tree
from .parsers import parse_block_temps, parse_layer_grid
from .runner import build_hotspot_command, run_hotspot
from .scenario import load_simulation_input
from .validate import validate_simulation_input
from .writers import read_grid_shape, write_flp, write_hotspot_config, write_ptrace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_SCHEMA_VERSION = "benchmark_v2_artifact_manifest/0.1"
DEPENDENCY_SCHEMA_VERSION = "benchmark_v2_dependency_lock/1"
ROOT_MARKER_NAME = ".chiptherm_data_root.json"
PATH_SEMANTICS = "relative_to_declared_data_root"
PILOT_STAGE = "pilot_5x10"
GRID_SHAPE = (64, 64)
PATH_COLUMNS = {
    "x_path",
    "y_path",
    "layout_path",
    "power_path",
    "package_path",
    "hotspot_path",
    "benchmark_path",
    "source_dir",
    "original_temp_path",
    "temp_layer0_path",
    "prediction_path",
    "residual_path",
    "graph_path",
    "source_superposition_base_path",
    "source_superposition_residual_path",
    "source_layout_path",
    "source_power_path",
    "source_package_path",
    "source_hotspot_path",
    "target_rise_path",
    "full_temperature_path",
    "original_x_path",
    "original_y_path",
}
PORTABLE_FORBIDDEN_PREFIXES = ("/Users/", "/nethome/", "/tmp/", "/export/hdd/")
CANONICAL_METADATA_FEATURES = (
    "package_width_mm",
    "package_height_mm",
    "cell_size_x_mm",
    "cell_size_y_mm",
    "total_power_W",
    "chiplet_count",
    "occupied_fraction",
    "whitespace_fraction",
    "mean_power_density_W_per_mm2",
    "max_power_density_W_per_mm2",
    "mean_chiplet_area_mm2",
    "max_chiplet_area_mm2",
    "mean_chiplet_aspect_ratio",
    "spreader_side_m",
    "sink_side_m",
)


@dataclass(frozen=True)
class PilotPaths:
    data_root: Path
    scratch_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.scratch_root / "runs" / self.run_id

    def canonical(self, name: str) -> Path:
        return self.data_root / "canonical" / name

    def derived(self, name: str) -> Path:
        return self.data_root / "derived" / name


@dataclass(frozen=True)
class PilotBuildOptions:
    config_path: Path
    selection_path: Path
    family_dir: Path
    parent_lock_path: Path
    data_root: Path
    scratch_root: Path
    hotspot_home: Path | None
    config_template: Path
    selected_families: tuple[str, ...]
    seed: int
    workers: int
    resume: bool
    dry_run: bool
    keep_hotspot_workdirs: bool
    run_id: str
    source_checkpoint: Path | None = None
    source_lineage: Path | None = None
    residual_checkpoint: Path | None = None
    source_device: str = "cpu"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def data_root_relative(path: str | Path, data_root: str | Path) -> str:
    resolved = Path(path).resolve()
    root = Path(data_root).resolve()
    try:
        value = str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"path is outside declared data root: {resolved}") from exc
    if Path(value).is_absolute() or value.startswith("../"):
        raise ValueError(f"nonportable path: {value}")
    return value


def resolve_data_path(value: str, data_root: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(data_root).resolve() / path


def discover_data_root(index_path: str | Path) -> Path | None:
    current = Path(index_path).resolve().parent
    for candidate in (current, *current.parents):
        marker = candidate / ROOT_MARKER_NAME
        if marker.exists():
            payload = load_json(marker)
            if payload.get("benchmark_id") == BENCHMARK_ID:
                return candidate
    return None


def ensure_root_layout(data_root: Path, scratch_root: Path) -> None:
    for path in (
        data_root / "canonical",
        data_root / "canonical/manifests",
        data_root / "derived",
        data_root / "checkpoints",
        data_root / "evaluations",
        scratch_root / "runs",
        scratch_root / "hotspot_workdirs",
        scratch_root / "retries",
    ):
        path.mkdir(parents=True, exist_ok=True)
    write_json(
        data_root / ROOT_MARKER_NAME,
        {
            "schema_version": "chiptherm_data_root/1",
            "benchmark_id": BENCHMARK_ID,
            "root_id": BENCHMARK_ID,
            "path_semantics": PATH_SEMANTICS,
        },
    )


def verify_phase1_families(
    family_dir: Path,
    family_manifest_path: Path,
    selected_families: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    manifest = load_yaml(family_manifest_path)
    entries = {str(item["family_uid"]): item for item in manifest.get("family_entries", [])}
    if manifest.get("family_count") != 50 or len(entries) != 50:
        raise ValueError("Phase 1 family manifest must contain exactly 50 entries")
    families: list[dict[str, Any]] = []
    for uid in selected_families:
        if uid not in entries:
            raise ValueError(f"selected family missing from Phase 1 manifest: {uid}")
        path = family_dir / f"{uid}.yaml"
        if sha256_file(path) != entries[uid]["family_file_sha256"]:
            raise ValueError(f"Phase 1 family hash mismatch: {uid}")
        family = load_family(path)
        problems = validate_family_spec(family)
        if problems:
            raise ValueError("\n".join(problems))
        if family.get("structural_fingerprint") != entries[uid]["structural_fingerprint"]:
            raise ValueError(f"Phase 1 structural fingerprint mismatch: {uid}")
        families.append(family)
    return families, sha256_file(family_manifest_path)


def load_selection(path: Path, selected_override: Sequence[str] | None = None) -> dict[str, Any]:
    selection = load_yaml(path)
    rows = list(selection.get("selected_families", []))
    configured = [str(row["family_uid"]) for row in rows]
    selected = list(selected_override or configured)
    if len(selected) != 5 or len(set(selected)) != 5:
        raise ValueError("pilot_5x10 requires exactly five unique selected families")
    if selected_override:
        role_by_uid = {str(row["family_uid"]): row for row in rows}
        missing = [uid for uid in selected if uid not in role_by_uid]
        if missing:
            raise ValueError(f"selected overrides are absent from pilot selection config: {missing}")
        selection["selected_families"] = [role_by_uid[uid] for uid in selected]
    return selection


def verify_parent_lock(path: Path, *, family_manifest_path: Path) -> dict[str, Any]:
    lock = load_json(path)
    if lock.get("schema_version") != DEPENDENCY_SCHEMA_VERSION:
        raise ValueError(f"unsupported dependency lock schema: {lock.get('schema_version')}")
    if lock.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("dependency lock benchmark ID mismatch")
    required = lock.get("required_parent_hashes", {})
    checks = {
        "phase1_family_manifest_sha256": sha256_file(family_manifest_path),
        "artifact_manifest_schema_sha256": sha256_file(REPO_ROOT / "configs/benchmark_v2_50family/artifact_manifest_schema.json"),
        "pilot_selection_sha256": sha256_file(REPO_ROOT / "configs/benchmark_v2_50family/pilot_5x10.yaml"),
    }
    for key, actual in checks.items():
        expected = required.get(key)
        if not expected:
            raise ValueError(f"dependency lock has null/missing required field {key}")
        if expected != actual:
            raise ValueError(f"dependency lock mismatch for {key}: expected {expected}, got {actual}")
    return lock


def validate_source_checkpoint_lineage(
    checkpoint: Path,
    lineage_path: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    lineage = load_json(lineage_path)
    checkpoint_sha = sha256_file(checkpoint)
    if lineage.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("source-response checkpoint hash does not match lineage manifest")
    policy = selection["source_response_policy"]
    forbidden = set(policy["oracle_only_families"])
    fit_families: set[str] = set()
    for key in ("training_family_uids", "normalization_family_uids", "selection_family_uids"):
        values = lineage.get(key)
        if not isinstance(values, list):
            raise ValueError(f"source-response lineage is missing {key}")
        fit_families.update(str(value) for value in values)
    overlap = sorted(fit_families & forbidden)
    if overlap:
        raise ValueError(f"source-response checkpoint leaks pilot val/test families: {overlap}")
    if lineage.get("target") != "isolated_source_temperature_rise_K_per_W":
        raise ValueError("source-response lineage target is incompatible")
    return lineage


def copy_selected_families(
    families: Sequence[dict[str, Any]],
    family_dir: Path,
    destination: Path,
    *,
    phase1_manifest_sha256: str,
) -> None:
    stage = destination.parent / f".{destination.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, str]] = []
    for family in families:
        uid = str(family["family_uid"])
        source = family_dir / f"{uid}.yaml"
        shutil.copy2(source, stage / source.name)
        entries.append({"family_uid": uid, "path": source.name, "sha256": sha256_file(source)})
    write_json(
        stage / "selected_family_manifest.json",
        {
            "schema_version": "benchmark_v2_selected_family_manifest/1",
            "benchmark_id": BENCHMARK_ID,
            "phase1_family_manifest_sha256": phase1_manifest_sha256,
            "families": entries,
        },
    )
    promote_directory(stage, destination, resume=True)


def prepare_workloads(
    families: Sequence[dict[str, Any]],
    destination: Path,
    *,
    seed: int,
    run_root: Path,
    resume: bool,
) -> dict[str, Any]:
    if resume and durable_stage_complete(destination) and (destination / "workload_manifest.yaml").exists():
        manifest = load_yaml(destination / "workload_manifest.yaml")
        if manifest.get("family_uids") == sorted(str(item["family_uid"]) for item in families) and manifest.get("base_seed") == seed:
            for family in families:
                for ordinal in range(1, 11):
                    matches = list((destination / str(family["family_uid"])).glob(f"w{ordinal:03d}_*.yaml"))
                    if len(matches) != 1 or validate_workload(load_yaml(matches[0]), family):
                        raise ValueError(f"invalid resumed workload for {family['family_uid']} ordinal {ordinal}")
            return manifest
    stage = run_root / "workloads"
    if stage.exists():
        shutil.rmtree(stage)
    manifest = write_workload_tree(families, stage, base_seed=seed)
    promote_directory(stage, destination, resume=resume)
    return manifest


def workload_rows(workload_root: Path, families: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_uid in sorted(families):
        paths = sorted((workload_root / family_uid).glob("w*.yaml"))
        if len(paths) != 10:
            raise ValueError(f"{family_uid} must have exactly 10 pilot workloads, found {len(paths)}")
        for path in paths:
            workload = load_yaml(path)
            problems = validate_workload(workload, families[family_uid])
            if problems:
                raise ValueError("\n".join(problems))
            rows.append(workload)
    return rows


def write_canonical_sample_source(
    sample_root: Path,
    family: dict[str, Any],
    workload: dict[str, Any],
) -> None:
    source_dir = sample_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    sample_uid = str(workload["sample_uid"])
    layout = json.loads(json.dumps(family["fixed_structure"]["layout"]))
    layout["package"]["name"] = sample_uid
    powers = {str(key): float(value) for key, value in workload["chiplet_power_W"].items()}
    power = {
        "schema_version": 1,
        "units": {"power": "W"},
        "mode": "fixed",
        "active_workload": "nominal",
        "chiplets": powers,
        "workloads": {"nominal": powers},
    }
    scenario = {
        "schema_version": 1,
        "name": sample_uid,
        "description": "ChipTherm Benchmark v2 fixed-family pilot sample.",
        "files": {
            "layout": "layout.json",
            "power": "power.yaml",
            "package": "package.yaml",
            "hotspot": "hotspot.yaml",
        },
    }
    write_yaml(source_dir / "scenario.yaml", scenario)
    (source_dir / "layout.json").write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_yaml(source_dir / "power.yaml", power)
    write_yaml(source_dir / "package.yaml", family["fixed_structure"]["thermal_stack"])
    write_yaml(source_dir / "hotspot.yaml", family["fixed_structure"]["hotspot"])
    write_yaml(source_dir / "family.yaml", family)
    write_yaml(source_dir / "workload.yaml", workload)
    sim = load_simulation_input(source_dir / "scenario.yaml")
    validate_simulation_input(sim)
    if sim.power.active_workload != "nominal" or sim.power.workloads is None:
        raise ValueError(f"{sample_uid}: nominal workload was not preserved")
    if sim.power.chiplet_watts != sim.power.workloads["nominal"]:
        raise ValueError(f"{sample_uid}: top-level and nominal powers differ")


def run_hotspot_sample(
    sample_stage: Path,
    hotspot_home: Path,
    config_template: Path,
    *,
    executable_sha256: str,
    max_attempts: int = 2,
) -> dict[str, Any]:
    source_dir = sample_stage / "source"
    sim = load_simulation_input(source_dir / "scenario.yaml")
    validate_simulation_input(sim)
    hotspot_dir = sample_stage / "hotspot"
    output_dir = sample_stage / "outputs"
    parsed_dir = sample_stage / "parsed"
    for path in (hotspot_dir, output_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)
    flp_path = write_flp(sim.layout, hotspot_dir / "chiplet.flp")
    ptrace_path = write_ptrace(sim.layout, sim.power, hotspot_dir / "power.ptrace")
    config_path = write_hotspot_config(config_template, hotspot_dir / "hotspot.config", sim.package, sim.hotspot)
    rows, cols = read_grid_shape(config_path)
    if (rows, cols) != GRID_SHAPE:
        raise ValueError(f"HotSpot config grid must be {GRID_SHAPE}, got {(rows, cols)}")
    block_path = output_dir / "block.steady"
    grid_path = output_dir / "grid.steady"
    command = build_hotspot_command(
        hotspot_home=hotspot_home,
        config_path=config_path,
        flp_path=flp_path,
        ptrace_path=ptrace_path,
        steady_path=block_path,
        grid_steady_path=grid_path,
    )
    portable_command = [
        "$HOTSPOT_HOME/hotspot",
        "-c", "hotspot/hotspot.config",
        "-f", "hotspot/chiplet.flp",
        "-p", "hotspot/power.ptrace",
        "-model_type", "grid",
        "-steady_file", "outputs/block.steady",
        "-grid_steady_file", "outputs/grid.steady",
    ]
    (sample_stage / "command.txt").write_text(shlex.join(portable_command) + "\n", encoding="utf-8")
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        result = run_hotspot(command, cwd=sample_stage)
        attempts.append(
            {
                "attempt": attempt,
                "return_code": int(result.returncode),
                "runtime_s": time.perf_counter() - attempt_started,
            }
        )
        if result.returncode == 0:
            break
    assert result is not None
    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"HotSpot failed after {len(attempts)} attempts with return code {result.returncode}")
    temperature = parse_layer_grid(grid_path, layer=0, rows=rows, cols=cols).astype(np.float32)
    if temperature.shape != GRID_SHAPE or not np.isfinite(temperature).all():
        raise ValueError(f"invalid HotSpot temperature map shape/values: {temperature.shape}")
    if float(temperature.min()) < 100.0 or float(temperature.max()) > 2000.0:
        raise ValueError("HotSpot temperature values are outside the Kelvin sanity range [100, 2000]")
    np.save(parsed_dir / "temp_layer0.npy", temperature)
    chiplet_names = tuple(item.name for item in sim.layout.chiplets)
    block_temps = parse_block_temps(block_path, names=chiplet_names)
    write_json(parsed_dir / "block_temps.json", block_temps)
    source_hashes = {
        name: sha256_file(source_dir / name)
        for name in ("scenario.yaml", "layout.json", "power.yaml", "package.yaml", "hotspot.yaml", "family.yaml", "workload.yaml")
    }
    manifest = {
        "schema_version": "benchmark_v2_sample_manifest/1",
        "benchmark_id": BENCHMARK_ID,
        "sample_uid": sim.scenario.name,
        "family_uid": load_yaml(source_dir / "workload.yaml")["family_uid"],
        "workload_uid": load_yaml(source_dir / "workload.yaml")["workload_uid"],
        "status": "validated",
        "created_at": utc_now(),
        "path_semantics": "relative_to_sample_root",
        "source_hashes": source_hashes,
        "hotspot": {
            "executable_id": "HOTSPOT_HOME/hotspot",
            "executable_sha256": executable_sha256,
            "command": portable_command,
            "attempts": attempts,
        },
        "runtime_s": time.perf_counter() - started,
        "grid": {"rows": rows, "cols": cols, "orientation": "array[row_y, col_x] with bottom-to-top HotSpot parser order"},
        "temperature": {
            "path": "parsed/temp_layer0.npy",
            "shape": list(temperature.shape),
            "dtype": str(temperature.dtype),
            "units": "K",
            "sha256": sha256_file(parsed_dir / "temp_layer0.npy"),
            "min_K": float(temperature.min()),
            "max_K": float(temperature.max()),
            "mean_K": float(temperature.mean()),
        },
    }
    write_json(sample_stage / "manifest.json", manifest)
    return manifest


def validated_existing_sample(sample_root: Path, family: dict[str, Any], workload: dict[str, Any]) -> bool:
    manifest_path = sample_root / "manifest.json"
    target_path = sample_root / "parsed/temp_layer0.npy"
    if not manifest_path.exists() or not target_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
        target = np.load(target_path, mmap_mode="r")
        source_family = load_yaml(sample_root / "source/family.yaml")
        source_workload = load_yaml(sample_root / "source/workload.yaml")
    except Exception:
        return False
    return bool(
        manifest.get("status") == "validated"
        and manifest.get("sample_uid") == workload.get("sample_uid")
        and source_family.get("structural_fingerprint") == family.get("structural_fingerprint")
        and source_workload.get("content_hash") == workload.get("content_hash")
        and tuple(target.shape) == GRID_SHAPE
        and np.isfinite(np.asarray(target)).all()
    )


def generate_hotspot_samples(
    paths: PilotPaths,
    families: Mapping[str, dict[str, Any]],
    workloads: Sequence[dict[str, Any]],
    *,
    hotspot_home: Path | None,
    config_template: Path,
    workers: int,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    stage_root = paths.run_root / "hotspot_labels"
    stage_root.mkdir(parents=True, exist_ok=True)
    accepted_root = paths.canonical("hotspot_labels")
    scheduled: list[tuple[dict[str, Any], Path]] = []
    skipped: list[str] = []
    for workload in workloads:
        family_uid = str(workload["family_uid"])
        sample_uid = str(workload["sample_uid"])
        accepted = accepted_root / family_uid / sample_uid
        if resume and validated_existing_sample(accepted, families[family_uid], workload):
            skipped.append(sample_uid)
            continue
        stage = stage_root / family_uid / sample_uid
        if stage.exists():
            retry_destination = paths.scratch_root / "retries" / paths.run_id / family_uid / sample_uid / f"retry-{uuid.uuid4().hex[:10]}"
            retry_destination.parent.mkdir(parents=True, exist_ok=True)
            stage.replace(retry_destination)
        write_canonical_sample_source(stage, families[family_uid], workload)
        if dry_run:
            write_json(
                stage / "manifest.json",
                {
                    "schema_version": "benchmark_v2_sample_manifest/1",
                    "benchmark_id": BENCHMARK_ID,
                    "sample_uid": sample_uid,
                    "family_uid": family_uid,
                    "workload_uid": workload["workload_uid"],
                    "status": "source_validated_dry_run",
                    "created_at": utc_now(),
                    "path_semantics": "relative_to_sample_root",
                },
            )
        else:
            scheduled.append((workload, stage))
    failures: list[dict[str, Any]] = []
    completed: list[str] = []
    retry_count = 0
    executable_sha = "dry_run"
    if not dry_run:
        if hotspot_home is None:
            raise ValueError("--hotspot-home is required unless --dry-run is used")
        executable = hotspot_home / "hotspot"
        if not executable.is_file():
            raise FileNotFoundError(executable)
        executable_sha = sha256_file(executable)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_by_item = {
                executor.submit(
                    run_hotspot_sample,
                    stage,
                    hotspot_home,
                    config_template,
                    executable_sha256=executable_sha,
                ): (workload, stage)
                for workload, stage in scheduled
            }
            for future in as_completed(future_by_item):
                workload, stage = future_by_item[future]
                uid = str(workload["sample_uid"])
                try:
                    sample_manifest = future.result()
                    retry_count += max(0, len(sample_manifest.get("hotspot", {}).get("attempts", [])) - 1)
                    destination = accepted_root / str(workload["family_uid"]) / uid
                    promote_directory(stage, destination, resume=False)
                    completed.append(uid)
                except Exception as exc:
                    failure = {
                        "sample_uid": uid,
                        "family_uid": workload["family_uid"],
                        "reason": str(exc),
                        "type": type(exc).__name__,
                        "staging_path": data_root_relative(stage, paths.data_root)
                        if _is_within(stage, paths.data_root)
                        else str(stage.relative_to(paths.scratch_root)),
                    }
                    failures.append(failure)
                    write_json(stage / "failure.json", failure)
    report = {
        "schema_version": "benchmark_v2_hotspot_generation_report/1",
        "requested": len(workloads),
        "scheduled": len(scheduled),
        "completed": len(completed),
        "skipped_valid": len(skipped),
        "failed": len(failures),
        "retry_count": retry_count,
        "completed_uids": sorted(completed),
        "skipped_uids": sorted(skipped),
        "failures": failures,
        "hotspot_executable_sha256": executable_sha,
        "dry_run": dry_run,
        "dry_run_source_count": len(workloads) if dry_run else 0,
    }
    write_json(paths.run_root / "hotspot_generation_report.json", report)
    write_csv(paths.run_root / "hotspot_failures.csv", failures)
    if failures:
        raise RuntimeError(f"HotSpot failed for {len(failures)} pilot samples; staging outputs were retained")
    return report


def raw_index_rows(
    paths: PilotPaths,
    workloads: Sequence[dict[str, Any]],
    families: Mapping[str, dict[str, Any]],
    selection: dict[str, Any],
) -> list[dict[str, str]]:
    sample_assignment = selection["sample_split_assignment"]
    sample_split_by_ordinal = {
        int(ordinal): split
        for split, key in (("train", "train_workload_ordinals"), ("val", "val_workload_ordinals"), ("test", "test_workload_ordinals"))
        for ordinal in sample_assignment[key]
    }
    rows: list[dict[str, str]] = []
    for workload in workloads:
        uid = str(workload["sample_uid"])
        family_uid = str(workload["family_uid"])
        ordinal = int(str(workload["workload_uid"])[1:4])
        sample_root = paths.canonical("hotspot_labels") / family_uid / uid
        target = sample_root / "parsed/temp_layer0.npy"
        if not target.exists():
            raise FileNotFoundError(f"missing validated label for {uid}: {target}")
        rows.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "family_uid": family_uid,
                "workload_uid": str(workload["workload_uid"]),
                "sample_uid": uid,
                "original_sample_uid": uid,
                "case_id": family_uid,
                "dataset_source": BENCHMARK_ID,
                "workload_stratum": str(workload["stratum"]),
                "split": sample_split_by_ordinal[ordinal],
                "sample_split": sample_split_by_ordinal[ordinal],
                "family_split": str(families[family_uid]["primary_split"]),
                "layout_path": data_root_relative(sample_root / "source/layout.json", paths.data_root),
                "power_path": data_root_relative(sample_root / "source/power.yaml", paths.data_root),
                "package_path": data_root_relative(sample_root / "source/package.yaml", paths.data_root),
                "hotspot_path": data_root_relative(sample_root / "source/hotspot.yaml", paths.data_root),
                "benchmark_path": data_root_relative(sample_root / "source/workload.yaml", paths.data_root),
                "source_dir": data_root_relative(sample_root / "source", paths.data_root),
                "temp_layer0_path": data_root_relative(target, paths.data_root),
                "y_path": data_root_relative(target, paths.data_root),
                "original_temp_path": data_root_relative(target, paths.data_root),
                "hotspot_runtime_s": str(load_json(sample_root / "manifest.json").get("runtime_s", "")),
                "num_chiplets": str(len(families[family_uid]["fixed_structure"]["layout"]["chiplets"])),
                "total_power_W": str(workload["total_package_power_W"]),
                "family_structural_fingerprint": str(families[family_uid]["structural_fingerprint"]),
                "workload_content_sha256": str(workload["content_hash"]),
            }
        )
    return rows


def build_derived_pipeline(
    paths: PilotPaths,
    raw_rows: list[dict[str, str]],
    selection: dict[str, Any],
    *,
    source_checkpoint: Path,
    source_lineage: dict[str, Any],
    resume: bool,
    source_device: str,
) -> dict[str, Any]:
    derived_run = paths.run_root / "derived"
    derived_run.mkdir(parents=True, exist_ok=True)
    raw_adapter = derived_run / "raw_index_absolute.csv"
    absolute_rows = absolutize_rows(raw_rows, paths.data_root)
    write_csv(raw_adapter, absolute_rows)

    encoded_dest = paths.derived("encoded_13ch")
    if not (resume and durable_stage_complete(encoded_dest)):
        encoded_stage = derived_run / "encoded_13ch"
        archive_failed_stage(encoded_stage, paths.scratch_root, paths.run_id)
        run_checked([sys.executable, "scripts/encode_dataset.py", "--index", str(raw_adapter), "--out-dir", str(encoded_stage)], paths.run_root)
        finalize_encoded_stage(encoded_stage, raw_adapter)
        prepare_split_files(encoded_stage)
        canonicalize_stage_indices(encoded_stage, encoded_dest, paths.data_root)
        promote_directory(encoded_stage, encoded_dest, resume=False)

    finite_dest = paths.derived("context_17ch")
    if not (resume and durable_stage_complete(finite_dest)):
        finite_input = create_builder_view(encoded_dest, derived_run / "finite_input", paths.data_root)
        finite_stage = derived_run / "context_17ch"
        archive_failed_stage(finite_stage, paths.scratch_root, paths.run_id)
        run_checked(
            [
                sys.executable,
                "scripts/build_finite_source_feature_dataset.py",
                "--source-root", str(finite_input),
                "--out-root", str(finite_stage),
                "--length-scales-mm", "0.5", "1.0", "2.0", "4.0",
                "--quadrature-size", "4",
                "--kernel", "softened_green",
            ],
            paths.run_root,
        )
        canonicalize_stage_indices(finite_stage, finite_dest, paths.data_root)
        promote_directory(finite_stage, finite_dest, resume=False)

    impedance_dest = paths.derived("context_33ch")
    if not (resume and durable_stage_complete(impedance_dest)):
        impedance_input = create_builder_view(finite_dest, derived_run / "impedance_input", paths.data_root)
        impedance_stage = derived_run / "context_33ch"
        archive_failed_stage(impedance_stage, paths.scratch_root, paths.run_id)
        run_checked(
            [
                sys.executable,
                "scripts/build_thermal_impedance_feature_dataset.py",
                "--source-root", str(impedance_input),
                "--out-root", str(impedance_stage),
                "--enclosed-power-radii-mm", "2", "4", "8", "16",
                "--crowding-epsilon-mm", "1.0",
            ],
            paths.run_root,
        )
        canonicalize_stage_indices(impedance_stage, impedance_dest, paths.data_root)
        promote_directory(impedance_stage, impedance_dest, resume=False)

    metadata_dest = paths.derived("metadata")
    if not (resume and durable_stage_complete(metadata_dest)):
        metadata_view = create_builder_view(impedance_dest, derived_run / "metadata_input", paths.data_root)
        run_checked([sys.executable, "scripts/build_metadata_features.py", "--dataset-root", str(metadata_view)], paths.run_root)
        force_canonical_metadata_schema(metadata_view / "metadata_manifest.json")
        metadata_stage = derived_run / "metadata"
        archive_failed_stage(metadata_stage, paths.scratch_root, paths.run_id)
        metadata_stage.mkdir(parents=True, exist_ok=True)
        for name in ("metadata_features.csv", "metadata_manifest.json"):
            shutil.copy2(metadata_view / name, metadata_stage / name)
        canonicalize_portable_documents(metadata_stage, metadata_dest, paths.data_root)
        promote_directory(metadata_stage, metadata_dest, resume=False)

    graph_dest = paths.derived("graphs")
    if not (resume and durable_stage_complete(graph_dest)):
        graph_input = create_builder_view(impedance_dest, derived_run / "graph_input", paths.data_root)
        for name in ("metadata_features.csv", "metadata_manifest.json"):
            shutil.copy2(metadata_dest / name, graph_input / name)
        graph_stage = derived_run / "graphs"
        archive_failed_stage(graph_stage, paths.scratch_root, paths.run_id)
        run_checked(
            [sys.executable, "scripts/build_graph_features.py", "--source-root", str(graph_input), "--out-root", str(graph_stage)],
            paths.run_root,
        )
        canonicalize_stage_indices(graph_stage, graph_dest, paths.data_root)
        promote_directory(graph_stage, graph_dest, resume=False)

    portable_checkpoint = install_checkpoint(source_checkpoint, paths.data_root, "source_response")
    write_json(portable_checkpoint.with_suffix(".lineage.json"), source_lineage)
    source_dest = paths.derived("source_superposition")
    if not (resume and durable_stage_complete(source_dest)):
        graph_view = create_builder_view(graph_dest, derived_run / "source_base_input", paths.data_root)
        source_stage = derived_run / "source_superposition"
        archive_failed_stage(source_stage, paths.scratch_root, paths.run_id)
        run_checked(
            [
                sys.executable,
                "scripts/build_full_source_superposition_base.py",
                "--index", str(graph_view / "combined_encoded_index.csv"),
                "--checkpoint", str(portable_checkpoint),
                "--out-root", str(source_stage),
                "--package-batch-size", "8",
                "--source-batch-size", "64",
                "--device", source_device,
                "--resume",
            ],
            paths.run_root,
        )
        canonicalize_stage_indices(source_stage, source_dest, paths.data_root)
        add_source_lineage_columns(source_stage, source_lineage, portable_checkpoint, paths.data_root)
        promote_directory(source_stage, source_dest, resume=False)

    # Source-response records require the encoded X path as well as the raw
    # source/label paths. The promoted graph index is the first canonical view
    # that contains the complete portable row contract.
    isolation_rows = read_csv(graph_dest / "combined_encoded_index.csv")
    isolation = build_isolation_inputs(isolation_rows, selection, derived_run / "isolation_inputs", paths.data_root)
    return {
        "encoded_13ch": encoded_dest,
        "context_17ch": finite_dest,
        "context_33ch": impedance_dest,
        "metadata": metadata_dest,
        "graphs": graph_dest,
        "source_superposition": source_dest,
        "source_isolation_inputs": isolation,
        "portable_source_checkpoint": portable_checkpoint,
    }


def build_isolation_inputs(
    raw_rows: Sequence[dict[str, str]],
    selection: dict[str, Any],
    output_root: Path,
    data_root: Path,
) -> dict[str, Path]:
    policy = selection["source_response_policy"]
    family_partition = {
        **{uid: "train" for uid in policy["train_eligible_families"]},
        **{uid: "val" for uid in policy["oracle_only_families"][:-1]},
        **{uid: "test" for uid in policy["oracle_only_families"][-1:]},
    }
    selected: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    by_family: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        by_family.setdefault(row["family_uid"], []).append(row)
    for family_uid, partition in family_partition.items():
        candidates = sorted(by_family[family_uid], key=lambda item: item["workload_uid"])
        row = dict(candidates[0])
        row["split"] = partition
        selected[partition].append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for split, rows in selected.items():
        path = output_root / f"{split}_index.csv"
        for row in rows:
            for key in PATH_COLUMNS:
                value = str(row.get(key, "")).strip()
                if not value:
                    continue
                if Path(value).is_absolute() or value.startswith("../"):
                    raise ValueError(
                        f"source-isolation input must remain data-root-relative: "
                        f"sample_uid={row.get('sample_uid')} field={key} value={value!r}"
                    )
                resolved = resolve_data_path(value, data_root)
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"source-isolation input is unresolved: sample_uid={row.get('sample_uid')} "
                        f"field={key} logical path={value!r} resolution root={Path(data_root).resolve()} "
                        f"resolved path={resolved}"
                    )
        write_csv(path, rows)
        result[split] = path
    return result


def run_source_isolation(
    paths: PilotPaths,
    isolation_inputs: Mapping[str, Path],
    selected_families: Sequence[str],
    *,
    hotspot_home: Path,
    config_template: Path,
    resume: bool,
) -> Path:
    destination = paths.canonical("source_isolation")
    if resume and durable_stage_complete(destination):
        return destination
    stage = paths.run_root / "source_isolation"
    command = [
        sys.executable,
        "scripts/build_source_response_dataset.py",
        "--train-index", str(isolation_inputs["train"]),
        "--val-index", str(isolation_inputs["val"]),
        "--test-index", str(isolation_inputs["test"]),
        "--data-root", str(paths.data_root),
        "--out-root", str(stage),
        "--cases", *selected_families,
        "--samples-per-case", "1",
        "--hotspot-home", str(hotspot_home),
        "--config-template", str(config_template),
    ]
    if resume:
        command.append("--resume")
    run_checked(command, paths.run_root)
    canonicalize_stage_indices(stage, destination, paths.data_root)
    promote_directory(stage, destination, resume=False)
    return destination


def create_final_indices(
    paths: PilotPaths,
    source_root: Path,
    selection: dict[str, Any],
    manifest_ids: Mapping[str, str],
) -> Path:
    rows = read_csv(source_root / "combined_encoded_index.csv")
    if len(rows) != 50:
        raise ValueError(f"source-superposition index must contain 50 rows, got {len(rows)}")
    output = paths.derived("indices") / PILOT_STAGE
    stage = paths.run_root / "indices"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    enriched: list[dict[str, str]] = []
    for row in rows:
        result = dict(row)
        result["benchmark_id"] = BENCHMARK_ID
        result["protocol_id"] = "pilot_5x10_all"
        result["metadata_row_id"] = row["sample_uid"]
        result["x_artifact_id"] = "pilot_5x10_context_33ch"
        result["y_artifact_id"] = "pilot_5x10_hotspot_labels"
        result["graph_artifact_id"] = "pilot_5x10_graphs"
        result["metadata_artifact_id"] = "pilot_5x10_metadata"
        result["source_superposition_artifact_id"] = "pilot_5x10_source_superposition"
        for key, value in manifest_ids.items():
            result[key] = value
        enriched.append(result)
    write_csv(stage / "all_index.csv", enriched)
    for protocol, split_key in (("sample_split", "sample_split"), ("family_split", "family_split")):
        for split in ("train", "val", "test"):
            subset: list[dict[str, str]] = []
            for row in enriched:
                if row.get(split_key) != split:
                    continue
                item = dict(row)
                item["split"] = split
                item["protocol_id"] = f"pilot_5x10_{protocol}"
                subset.append(item)
            write_csv(stage / protocol / f"{split}_index.csv", subset, fieldnames=list(enriched[0].keys()))
    for name in ("feature_manifest.json", "graph_manifest.json"):
        source = source_root / name
        if source.exists():
            shutil.copy2(source, stage / name)
    for name in ("metadata_features.csv", "metadata_manifest.json"):
        source = paths.derived("metadata") / name
        shutil.copy2(source, stage / name)
    canonicalize_stage_indices(stage, output, paths.data_root)
    promote_directory(stage, output, resume=False)
    return output


def absolutize_rows(rows: Sequence[dict[str, str]], data_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        for key in PATH_COLUMNS:
            value = item.get(key, "")
            if value:
                item[key] = str(resolve_data_path(value, data_root))
        output.append(item)
    return output


def create_builder_view(source_root: Path, view_root: Path, data_root: Path) -> Path:
    if view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True)
    for csv_path in source_root.glob("*.csv"):
        rows = read_csv(csv_path)
        if any(key in PATH_COLUMNS for key in (rows[0].keys() if rows else [])):
            write_csv(view_root / csv_path.name, absolutize_rows(rows, data_root), fieldnames=list(rows[0].keys()) if rows else [])
        else:
            shutil.copy2(csv_path, view_root / csv_path.name)
    for path in source_root.iterdir():
        if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml", ".yml", ".md"}:
            shutil.copy2(path, view_root / path.name)
    return view_root


def finalize_encoded_stage(root: Path, source_index: Path | None = None) -> None:
    metadata = load_json(root / "encoding_metadata.json")
    encoded = int(metadata.get("num_encoded", 0))
    failed = int(metadata.get("num_failed", 0))
    if encoded != 50 or failed:
        raise RuntimeError(f"13-channel encoding expected 50/0 encoded/failed, got {encoded}/{failed}")
    encoded_rows = read_csv(root / "encoded_index.csv")
    if source_index is not None:
        source_by_uid = {row["sample_uid"]: row for row in read_csv(source_index)}
        missing = [row["sample_uid"] for row in encoded_rows if row["sample_uid"] not in source_by_uid]
        if missing:
            raise ValueError(f"encoded rows cannot be matched to source adapter rows: {missing[:10]}")
        encoded_rows = [{**source_by_uid[row["sample_uid"]], **row} for row in encoded_rows]
    write_csv(root / "combined_encoded_index.csv", encoded_rows)
    with (root / "combined_encoded_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in encoded_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    channel_names = metadata.get("channel_names", [])
    if len(channel_names) != 13:
        raise ValueError(f"13-channel encoder manifest has {len(channel_names)} channels")
    write_json(
        root / "context_manifest.json",
        {
            "schema_version": 1,
            "channel_names": channel_names,
            "context_channels": channel_names[8:],
            "output_channels": 13,
        },
    )


def prepare_split_files(root: Path) -> None:
    rows = read_csv(root / "combined_encoded_index.csv")
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row.get("split") == split]
        write_csv(root / f"{split}_index.csv", subset, fieldnames=list(rows[0].keys()))


def force_canonical_metadata_schema(manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    stats = manifest.get("feature_stats", {})
    missing = [name for name in CANONICAL_METADATA_FEATURES if name not in stats]
    if missing:
        raise ValueError(f"metadata table is missing canonical features: {missing}")
    manifest["active_features"] = list(CANONICAL_METADATA_FEATURES)
    manifest["constant_or_inactive_features"] = [
        name for name in stats if name not in CANONICAL_METADATA_FEATURES
    ]
    manifest["selection_rule"] = (
        "Benchmark-v2 compatibility schema: preserve the validated 15-feature ordering used by the current model; "
        "constant values remain explicit and normalization clamps zero standard deviations."
    )
    write_json(manifest_path, manifest)


def install_checkpoint(checkpoint: Path, data_root: Path, role: str) -> Path:
    digest = sha256_file(checkpoint)
    destination = data_root / "checkpoints" / role / f"{checkpoint.stem}_{digest[:12]}{checkpoint.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError(f"installed checkpoint hash mismatch: {destination}")
    else:
        temporary = destination.with_name(destination.name + f".stage-{uuid.uuid4().hex}")
        shutil.copy2(checkpoint, temporary)
        temporary.replace(destination)
    return destination


def add_source_lineage_columns(
    stage_root: Path,
    lineage: dict[str, Any],
    checkpoint: Path,
    data_root: Path,
) -> None:
    lineage_hash = sha256_json(lineage)
    for csv_path in stage_root.glob("*.csv"):
        rows = read_csv(csv_path)
        if not rows or "sample_uid" not in rows[0]:
            continue
        for row in rows:
            row["source_checkpoint"] = data_root_relative(checkpoint, data_root)
            row["source_checkpoint_lineage_sha256"] = lineage_hash
            row["source_checkpoint_training_family_set_sha256"] = sha256_json(sorted(lineage["training_family_uids"]))
        write_csv(csv_path, rows)
    combined_path = stage_root / "combined_encoded_index.csv"
    if combined_path.exists():
        with (stage_root / "combined_encoded_index.jsonl").open("w", encoding="utf-8") as handle:
            for row in read_csv(combined_path):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    for sidecar in stage_root.glob("maps/*/*/*.json"):
        payload = load_json(sidecar)
        payload["source_checkpoint"] = data_root_relative(checkpoint, data_root)
        payload["source_checkpoint_lineage_sha256"] = lineage_hash
        payload["source_checkpoint_training_family_uids"] = sorted(str(value) for value in lineage["training_family_uids"])
        payload["source_checkpoint_training_family_set_sha256"] = sha256_json(sorted(lineage["training_family_uids"]))
        write_json(sidecar, payload)
    write_json(stage_root / "source_checkpoint_lineage.json", lineage)


def canonicalize_stage_indices(stage_root: Path, destination_root: Path, data_root: Path) -> None:
    canonicalize_portable_documents(stage_root, destination_root, data_root)


def canonicalize_portable_documents(stage_root: Path, destination_root: Path, data_root: Path) -> None:
    for csv_path in stage_root.rglob("*.csv"):
        rows = read_csv(csv_path)
        if not rows:
            continue
        changed = False
        for row in rows:
            for key in PATH_COLUMNS:
                value = row.get(key, "")
                if not value:
                    continue
                resolved = _resolve_stage_value(value, csv_path.parent, stage_root, destination_root, data_root)
                row[key] = data_root_relative(resolved, data_root)
                changed = True
        if changed:
            write_csv(csv_path, rows, fieldnames=list(rows[0].keys()))
    for path in list(stage_root.rglob("*.json")) + list(stage_root.rglob("*.yaml")) + list(stage_root.rglob("*.yml")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sanitized = _sanitize_payload(payload, stage_root, destination_root, data_root)
        if path.suffix == ".json":
            write_json(path, sanitized)
        else:
            write_yaml(path, sanitized)
    for path in stage_root.rglob("*.jsonl"):
        sanitized_lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            sanitized_lines.append(json.dumps(_sanitize_payload(payload, stage_root, destination_root, data_root), sort_keys=True))
        path.write_text("\n".join(sanitized_lines) + ("\n" if sanitized_lines else ""), encoding="utf-8")
    replacements = {
        str(stage_root.resolve()): data_root_relative(destination_root, data_root),
        str(data_root.resolve()): "<CHIPTHERM_V2_DATA_ROOT>",
        str(REPO_ROOT.resolve()): "<CHIPTHERM_REPO_ROOT>",
    }
    for path in list(stage_root.rglob("*.md")) + list(stage_root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        path.write_text(text, encoding="utf-8")


def _resolve_stage_value(value: str, csv_parent: Path, stage_root: Path, destination_root: Path, data_root: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [csv_parent / path, stage_root / path, data_root / path, REPO_ROOT / path]
    existing = next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())
    if _is_within(existing, stage_root):
        return destination_root / existing.relative_to(stage_root.resolve())
    return existing


def _sanitize_payload(payload: Any, stage_root: Path, destination_root: Path, data_root: Path) -> Any:
    if isinstance(payload, dict):
        return {key: _sanitize_payload(value, stage_root, destination_root, data_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_payload(value, stage_root, destination_root, data_root) for value in payload]
    if isinstance(payload, str) and Path(payload).is_absolute():
        path = Path(payload).resolve()
        if _is_within(path, stage_root):
            path = destination_root / path.relative_to(stage_root.resolve())
        elif _is_within(path, data_root / "staging"):
            return "NONRESOLVING_STAGING_PATH_REDACTED"
        if _is_within(path, data_root):
            return data_root_relative(path, data_root)
        if _is_within(path, REPO_ROOT):
            return str(path.relative_to(REPO_ROOT))
        return "NONPORTABLE_PRODUCER_PATH_REDACTED"
    return payload


def promote_directory(stage: Path, destination: Path, *, resume: bool) -> None:
    if destination.exists():
        if resume:
            if not durable_stage_complete(destination):
                raise ValueError(f"resume destination failed durable completion validation: {destination}")
            if stage.exists():
                shutil.rmtree(stage)
            return
        raise FileExistsError(f"refusing to overwrite promoted stage: {destination}")
    completion = make_tree_manifest(stage, exclude_names={".stage_complete.json"})
    write_json(
        stage / ".stage_complete.json",
        {
            "schema_version": "benchmark_v2_stage_completion/1",
            "file_count": completion["file_count"],
            "files": [{"path": item["path"], "sha256": item["sha256"]} for item in completion["files"]],
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(destination)


def durable_stage_complete(destination: Path) -> bool:
    marker = destination / ".stage_complete.json"
    if not destination.is_dir() or not marker.exists():
        return False
    try:
        payload = load_json(marker)
        files = payload.get("files", [])
        return bool(files) and all(
            (destination / item["path"]).is_file()
            and sha256_file(destination / item["path"]) == item["sha256"]
            for item in files
        )
    except Exception:
        return False


def run_checked(command: Sequence[str], run_root: Path) -> None:
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = Path(command[1]).stem if len(command) > 1 else Path(command[0]).stem
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    (logs / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
        )


def archive_failed_stage(stage: Path, scratch_root: Path, run_id: str) -> None:
    if not stage.exists():
        return
    destination = scratch_root / "retries" / run_id / "derived" / stage.name / f"retry-{uuid.uuid4().hex[:10]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(destination)


def make_tree_manifest(root: Path, *, exclude_names: Iterable[str] = ()) -> dict[str, Any]:
    excluded = set(exclude_names)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(path.relative_to(root)), "size_bytes": size, "sha256": sha256_file(path)})
    return {
        "schema_version": "benchmark_v2_tree_manifest/1",
        "root_id": BENCHMARK_ID,
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def write_artifact_manifest(
    artifact_root: Path,
    manifest_path: Path,
    *,
    artifact_id: str,
    artifact_class: str,
    stage: str,
    data_root: Path,
    parents: Sequence[dict[str, str]],
    checks: Sequence[dict[str, Any]],
    tensor_schema: dict[str, Any] | None = None,
    row_count: int | None = None,
    source_count: int | None = None,
    command: Sequence[str] = (),
    seed: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    report_path = artifact_root / "validation_report.json"
    write_json(report_path, {"passed": all(bool(item["passed"]) for item in checks), "checks": list(checks)})
    tree_path = artifact_root / "tree_manifest.json"
    tree = make_tree_manifest(artifact_root, exclude_names={"tree_manifest.json"})
    write_json(tree_path, tree)
    git = git_identity()
    package_report = python_package_report()
    content: dict[str, Any] = {
        "file_count": tree["file_count"],
        "total_bytes": tree["total_bytes"],
        "hash_algorithm": "sha256",
        "tree_manifest_path": data_root_relative(tree_path, data_root),
        "tree_manifest_sha256": sha256_file(tree_path),
    }
    if row_count is not None:
        content["row_count"] = int(row_count)
        content["sample_count"] = int(row_count)
    if source_count is not None:
        content["source_count"] = int(source_count)
    if tensor_schema is not None:
        content["tensor_schema"] = tensor_schema
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "benchmark_id": BENCHMARK_ID,
        "artifact_class": artifact_class,
        "stage": stage,
        "status": "validated",
        "created_at": utc_now(),
        "producer": {
            "command": list(command) or ["benchmark_v2_pipeline"],
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "dirty_diff_sha256": git["dirty_diff_sha256"],
            "code_paths": ["src/chiptherm/benchmark_v2_pipeline.py"],
            "python_version": platform.python_version(),
            "environment_lock_sha256": package_report["sha256"],
            "worker_count": workers,
            "seed": seed,
        },
        "storage": {
            "root_id": BENCHMARK_ID,
            "relative_path": data_root_relative(artifact_root, data_root),
            "path_semantics": PATH_SEMANTICS,
            "persistent": True,
            "retention_class": "publication",
        },
        "parents": list(parents),
        "content": content,
        "validation": {
            "passed": all(bool(item["passed"]) for item in checks),
            "report_path": data_root_relative(report_path, data_root),
            "report_sha256": sha256_file(report_path),
            "checks": list(checks),
        },
        "reproducibility": {
            "regenerable": artifact_class != "canonical_source",
            "reproduction_command": list(command) or None,
            "estimated_cost": "pilot_stage_only",
            "missing_prerequisites": [],
            "must_not_delete_while_referenced": True,
        },
    }
    validate_artifact_manifest(manifest)
    write_json(manifest_path, manifest)
    return manifest


def validate_artifact_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version", "artifact_id", "benchmark_id", "artifact_class", "stage", "status", "created_at",
        "producer", "storage", "parents", "content", "validation", "reproducibility",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"artifact manifest missing fields: {missing}")
    if manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION or manifest["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("artifact manifest identity mismatch")
    storage = manifest["storage"]
    if storage.get("path_semantics") != PATH_SEMANTICS or Path(storage.get("relative_path", "")).is_absolute():
        raise ValueError("artifact manifest has nonportable storage semantics")
    for parent in manifest["parents"]:
        if not parent.get("artifact_id") or len(str(parent.get("manifest_sha256", ""))) != 64:
            raise ValueError("artifact parent reference is incomplete")
    if not manifest["validation"].get("passed"):
        raise ValueError("cannot validate an artifact whose checks failed")


def runtime_dependency_lock(
    options: PilotBuildOptions,
    *,
    phase1_hash: str,
    workload_manifest_path: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    hotspot_identity: dict[str, Any]
    if options.dry_run:
        hotspot_identity = {"executable_id": "dry_run_not_executed", "version": "dry_run", "sha256": "0" * 64}
    else:
        if options.hotspot_home is None:
            raise ValueError("HotSpot home is required")
        executable = options.hotspot_home / "hotspot"
        hotspot_identity = {
            "executable_id": "HOTSPOT_HOME/hotspot",
            "version": hotspot_version(executable),
            "sha256": sha256_file(executable),
        }
    package_report = python_package_report()
    split_hash = sha256_json(
        {
            "pilot_selection": selection,
            "primary_family_split": sha256_file(REPO_ROOT / "configs/benchmark_v2_50family/splits/primary_family_split.yaml"),
        }
    )
    schemas = current_schema_hashes()
    git = git_identity()
    lock = {
        "schema_version": "benchmark_v2_runtime_dependency_lock/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": PILOT_STAGE,
        "created_at": utc_now(),
        "phase1_family_manifest_sha256": phase1_hash,
        "selected_family_uids": list(options.selected_families),
        "workload_manifest_sha256": sha256_file(workload_manifest_path),
        "split_manifest_sha256": split_hash,
        "code_commit": git["commit"],
        "dirty_worktree": git["dirty"],
        "dirty_diff_sha256": git["dirty_diff_sha256"],
        "python_version": platform.python_version(),
        "environment_package_report_sha256": package_report["sha256"],
        "hotspot": hotspot_identity,
        "hotspot_base_config_sha256": sha256_file(options.config_template),
        "raster_schema_sha256": schemas["raster"],
        "metadata_schema_sha256": schemas["metadata"],
        "graph_node_schema_sha256": schemas["graph_node"],
        "graph_edge_schema_sha256": schemas["graph_edge"],
        "source_response_lineage_policy": selection["source_response_policy"],
        "declared_data_root_id": BENCHMARK_ID,
        "declared_scratch_root_id": f"{BENCHMARK_ID}:staging",
        "path_semantics": PATH_SEMANTICS,
    }
    if any(value is None for value in _walk_values(lock)):
        raise ValueError("runtime dependency lock contains null required fields")
    return lock


def current_schema_hashes() -> dict[str, str]:
    from .ml.encoder import CHANNEL_NAMES
    from .ml.graph_models import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES

    return {
        "raster": sha256_json(list(CHANNEL_NAMES)),
        "metadata": sha256_json(list(CANONICAL_METADATA_FEATURES)),
        "graph_node": sha256_json(list(NODE_FEATURE_NAMES)),
        "graph_edge": sha256_json(list(EDGE_FEATURE_NAMES)),
    }


def python_package_report() -> dict[str, Any]:
    packages = sorted(
        (distribution.metadata.get("Name", distribution.name), distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    return {"packages": packages, "sha256": sha256_json(packages)}


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.strip()
    if len(commit) != 40:
        commit = "0" * 40
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=False).stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if status.strip() else None,
    }


def hotspot_version(executable: Path) -> str:
    for flag in ("-v", "--version"):
        result = subprocess.run([str(executable), flag], text=True, capture_output=True, check=False, timeout=10)
        text = (result.stdout or result.stderr).strip().splitlines()
        if text:
            return text[0][:256]
    return "version_not_reported"


def _walk_values(payload: Any) -> Iterable[Any]:
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _walk_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_values(value)
    else:
        yield payload


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_pilot(options: PilotBuildOptions) -> dict[str, Any]:
    started = time.perf_counter()
    if options.workers <= 0:
        raise ValueError("workers must be positive")
    design = load_yaml(options.config_path)
    if design.get("benchmark_id") != BENCHMARK_ID or design.get("stage_gates", {}).get("full_generation_approved") is not False:
        raise ValueError("--config is not the approved Benchmark v2 staged design proposal")
    selection = load_selection(options.selection_path, options.selected_families)
    family_manifest_path = options.family_dir.parent / "family_manifest.yaml"
    verify_parent_lock(options.parent_lock_path, family_manifest_path=family_manifest_path)
    families_list, phase1_hash = verify_phase1_families(options.family_dir, family_manifest_path, options.selected_families)
    families = {str(item["family_uid"]): item for item in families_list}
    paths = PilotPaths(options.data_root.resolve(), options.scratch_root.resolve(), options.run_id)
    ensure_root_layout(paths.data_root, paths.scratch_root)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    copy_selected_families(families_list, options.family_dir, paths.canonical("families"), phase1_manifest_sha256=phase1_hash)
    workload_manifest = prepare_workloads(
        families_list,
        paths.canonical("workloads"),
        seed=options.seed,
        run_root=paths.run_root,
        resume=options.resume,
    )
    workloads = workload_rows(paths.canonical("workloads"), families)
    runtime_lock = runtime_dependency_lock(
        options,
        phase1_hash=phase1_hash,
        workload_manifest_path=paths.canonical("workloads") / "workload_manifest.yaml",
        selection=selection,
    )
    write_json(paths.canonical("manifests") / "runtime_dependency_lock.json", runtime_lock)
    hotspot_report = generate_hotspot_samples(
        paths,
        families,
        workloads,
        hotspot_home=options.hotspot_home,
        config_template=options.config_template,
        workers=options.workers,
        resume=options.resume,
        dry_run=options.dry_run,
    )
    if options.dry_run:
        report = {
            "schema_version": "benchmark_v2_pilot_validation_report/1",
            "benchmark_id": BENCHMARK_ID,
            "stage": PILOT_STAGE,
            "status": "dry_run_validated",
            "run_id": options.run_id,
            "selected_family_uids": list(options.selected_families),
            "workload_count": len(workloads),
            "hotspot": hotspot_report,
            "derived_stages_run": False,
            "runtime_s": time.perf_counter() - started,
            "recommendation": "GO WITH FIXES: run the real HotSpot and derived stages before Phase 3.",
        }
        write_json(paths.canonical("manifests") / "pilot_5x10_build_report.json", report)
        write_json(paths.canonical("manifests") / "pilot_5x10_validation_report.json", report)
        return report
    if options.source_checkpoint is None or options.source_lineage is None:
        raise ValueError("real pilot requires --source-checkpoint and --source-lineage")
    lineage = validate_source_checkpoint_lineage(options.source_checkpoint, options.source_lineage, selection)
    raw_rows = raw_index_rows(paths, workloads, families, selection)
    derived = build_derived_pipeline(
        paths,
        raw_rows,
        selection,
        source_checkpoint=options.source_checkpoint,
        source_lineage=lineage,
        resume=options.resume,
        source_device=options.source_device,
    )
    isolation_root = run_source_isolation(
        paths,
        derived["source_isolation_inputs"],
        options.selected_families,
        hotspot_home=options.hotspot_home,
        config_template=options.config_template,
        resume=options.resume,
    )
    manifest_ids = {
        "runtime_dependency_lock_sha256": sha256_file(paths.canonical("manifests") / "runtime_dependency_lock.json"),
        "workload_manifest_sha256": sha256_file(paths.canonical("workloads") / "workload_manifest.yaml"),
        "phase1_family_manifest_sha256": phase1_hash,
        "source_checkpoint_lineage_sha256": sha256_json(lineage),
    }
    index_root = create_final_indices(paths, derived["source_superposition"], selection, manifest_ids)
    artifact_manifests = write_pilot_artifact_manifests(
        paths,
        options,
        derived=derived,
        isolation_root=isolation_root,
        index_root=index_root,
    )
    report = {
        "schema_version": "benchmark_v2_pilot_validation_report/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": PILOT_STAGE,
        "status": "built_pending_strict_validation",
        "selected_family_uids": list(options.selected_families),
        "selected_family_roles": selection["selected_families"],
        "workload_count": len(workloads),
        "workload_strata": workload_manifest["strata"],
        "hotspot": hotspot_report,
        "derived": {key: data_root_relative(value, paths.data_root) for key, value in derived.items() if isinstance(value, Path)},
        "source_isolation": data_root_relative(isolation_root, paths.data_root),
        "indices": data_root_relative(index_root, paths.data_root),
        "artifact_manifests": artifact_manifests,
        "runtime_s": time.perf_counter() - started,
        "recommendation": "GO WITH FIXES: strict validation and relocation must pass before Phase 3.",
    }
    write_json(paths.canonical("manifests") / "pilot_5x10_build_report.json", report)
    write_json(paths.canonical("manifests") / "pilot_5x10_validation_report.json", report)
    return report


def write_pilot_artifact_manifests(
    paths: PilotPaths,
    options: PilotBuildOptions,
    *,
    derived: Mapping[str, Any],
    isolation_root: Path,
    index_root: Path,
) -> dict[str, str]:
    manifest_root = paths.canonical("manifests") / "artifacts"
    manifest_root.mkdir(parents=True, exist_ok=True)
    command = ["python3", "scripts/build_benchmark_v2.py", "--stage", PILOT_STAGE]
    schemas = current_schema_hashes()
    written: dict[str, str] = {}

    def emit(
        key: str,
        artifact_root: Path,
        artifact_class: str,
        stage: str,
        *,
        parents: Sequence[str] = (),
        tensor_schema: dict[str, Any] | None = None,
        row_count: int | None = 50,
        source_count: int | None = None,
    ) -> None:
        parent_rows = [
            {
                "artifact_id": parent,
                "manifest_sha256": sha256_file(manifest_root / f"{parent}.json"),
                "relationship": "derived_from",
            }
            for parent in parents
        ]
        artifact_id = f"pilot_5x10_{key}"
        path = manifest_root / f"{artifact_id}.json"
        write_artifact_manifest(
            artifact_root,
            path,
            artifact_id=artifact_id,
            artifact_class=artifact_class,
            stage=stage,
            data_root=paths.data_root,
            parents=parent_rows,
            checks=[{"name": "stage_complete", "passed": True, "details": f"validated {key}"}],
            tensor_schema=tensor_schema,
            row_count=row_count,
            source_count=source_count,
            command=command,
            seed=options.seed,
            workers=options.workers,
        )
        written[key] = data_root_relative(path, paths.data_root)

    emit("families", paths.canonical("families"), "canonical_source", "design", row_count=5)
    emit("workloads", paths.canonical("workloads"), "canonical_source", "workloads", parents=("pilot_5x10_families",))
    emit("hotspot_labels", paths.canonical("hotspot_labels"), "canonical_source", "hotspot_labels", parents=("pilot_5x10_workloads",), tensor_schema={"shape": [64, 64], "dtype": "float32", "units": "K", "channel_schema_sha256": schemas["raster"]})
    emit("encoded_13ch", Path(derived["encoded_13ch"]), "model_specific_derived", "encoded_13ch", parents=("pilot_5x10_hotspot_labels",), tensor_schema={"shape": [13, 64, 64], "dtype": "float32", "units": "mixed", "channel_schema_sha256": schemas["raster"]})
    emit("context_17ch", Path(derived["context_17ch"]), "model_specific_derived", "context_17ch", parents=("pilot_5x10_encoded_13ch",), tensor_schema={"shape": [17, 64, 64], "dtype": "float32", "units": "mixed", "channel_schema_sha256": sha256_file(Path(derived["context_17ch"]) / "feature_manifest.json")})
    emit("context_33ch", Path(derived["context_33ch"]), "model_specific_derived", "context_33ch", parents=("pilot_5x10_context_17ch",), tensor_schema={"shape": [33, 64, 64], "dtype": "float32", "units": "mixed", "channel_schema_sha256": sha256_file(Path(derived["context_33ch"]) / "feature_manifest.json")})
    emit("metadata", Path(derived["metadata"]), "model_specific_derived", "metadata", parents=("pilot_5x10_context_33ch",), tensor_schema={"shape": [15], "dtype": "float32", "units": "mixed", "channel_schema_sha256": schemas["metadata"]})
    emit("graphs", Path(derived["graphs"]), "model_specific_derived", "graphs", parents=("pilot_5x10_context_33ch", "pilot_5x10_metadata"), tensor_schema={"shape": ["N", 24], "dtype": "float32", "units": "mixed", "channel_schema_sha256": schemas["graph_node"]})
    emit("source_response_model", Path(derived["portable_source_checkpoint"]).parent, "model_specific_derived", "source_response_model", row_count=None)
    source_count = sum(len(read_csv(isolation_root / f"{split}_index.csv")) for split in ("train", "val", "test"))
    emit("source_isolation", isolation_root, "canonical_source", "source_isolation", parents=("pilot_5x10_hotspot_labels",), row_count=5, source_count=source_count)
    emit("source_superposition", Path(derived["source_superposition"]), "model_specific_derived", "source_superposition", parents=("pilot_5x10_graphs", "pilot_5x10_source_response_model"), tensor_schema={"shape": [64, 64], "dtype": "float32", "units": "K", "channel_schema_sha256": sha256_json(["source_superposition_base_K"])})
    emit("indices", index_root, "generated_required_intermediate", "splits", parents=("pilot_5x10_source_superposition",), tensor_schema=None)
    return written


def validate_pilot_root(
    data_root: str | Path,
    *,
    allow_dry_run: bool = False,
    residual_checkpoint: str | Path | None = None,
    require_relocation: bool = False,
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    marker = load_json(data_root / ROOT_MARKER_NAME)
    if marker.get("benchmark_id") != BENCHMARK_ID or marker.get("path_semantics") != PATH_SEMANTICS:
        raise ValueError("invalid Benchmark v2 data-root marker")
    build_report_path = data_root / "canonical/manifests/pilot_5x10_build_report.json"
    report_path = build_report_path if build_report_path.exists() else data_root / "canonical/manifests/pilot_5x10_validation_report.json"
    build_report = load_json(report_path)
    dry_run = build_report.get("status") == "dry_run_validated"
    if dry_run and not allow_dry_run:
        raise ValueError("pilot is a dry run; real HotSpot and derived stages are required")
    checks: list[dict[str, Any]] = []

    selection = load_yaml(REPO_ROOT / "configs/benchmark_v2_50family/pilot_5x10.yaml")
    selected = [str(item["family_uid"]) for item in selection["selected_families"]]
    workload_root = data_root / "canonical/workloads"
    manifest = load_yaml(workload_root / "workload_manifest.yaml")
    workload_files = sorted(workload_root.glob("f*/w*.yaml"))
    workloads = [load_yaml(path) for path in workload_files]
    checks.append(_check("workload_count", len(workloads) == 50, f"count={len(workloads)}"))
    checks.append(_check("workloads_per_family", all(sum(item["family_uid"] == uid for item in workloads) == 10 for uid in selected)))
    checks.append(_check("all_pilot_strata", sorted(set(item["stratum"] for item in workloads)) == sorted(manifest["strata"])))
    hashes = [str(item["content_hash"]) for item in workloads]
    checks.append(_check("unique_workload_hashes", len(hashes) == len(set(hashes)) == 50))
    family_specs = {uid: load_yaml(data_root / "canonical/families" / f"{uid}.yaml") for uid in selected}
    workload_problems = [problem for workload in workloads for problem in validate_workload(workload, family_specs[str(workload["family_uid"])])]
    checks.append(_check("workload_validation", not workload_problems, "; ".join(workload_problems[:5])))
    portable = audit_portable_documents(data_root)
    checks.append(_check("portable_paths", portable["violation_count"] == 0, f"violations={portable['violation_count']}"))

    if dry_run:
        source_count = len(list((data_root / "staging/runs").glob("*/hotspot_labels/f*/f*/source/scenario.yaml")))
        checks.append(_check("dry_run_sources", source_count >= 50 and build_report.get("hotspot", {}).get("dry_run_source_count") == 50, f"source_files={source_count}"))
        passed = all(item["passed"] for item in checks)
        report = {
            "schema_version": "benchmark_v2_pilot_strict_validation/1",
            "benchmark_id": BENCHMARK_ID,
            "status": "dry_run_passed" if passed else "dry_run_failed",
            "passed": passed,
            "checks": checks,
            "portable_path_audit": portable,
        }
        write_json(data_root / "canonical/manifests/pilot_5x10_strict_validation.json", report)
        if not passed:
            raise ValueError("dry-run validation failed")
        return report

    label_paths = sorted((data_root / "canonical/hotspot_labels").glob("f*/f*/parsed/temp_layer0.npy"))
    checks.append(_check("hotspot_label_count", len(label_paths) == 50, f"count={len(label_paths)}"))
    label_failures = _audit_npy(label_paths, GRID_SHAPE)
    checks.append(_check("hotspot_label_shape_finite", not label_failures, "; ".join(label_failures[:5])))

    shape_contracts = {
        "encoded_13ch": ((13, 64, 64), data_root / "derived/encoded_13ch/combined_encoded_index.csv"),
        "context_17ch": ((17, 64, 64), data_root / "derived/context_17ch/combined_encoded_index.csv"),
        "context_33ch": ((33, 64, 64), data_root / "derived/context_33ch/combined_encoded_index.csv"),
    }
    for stage, (shape, index_path) in shape_contracts.items():
        rows = read_csv(index_path)
        failures = _audit_index_tensor(rows, "x_path", data_root, shape)
        checks.append(_check(f"{stage}_count_shape_finite", len(rows) == 50 and not failures, f"rows={len(rows)} failures={failures[:3]}"))
    metadata_manifest = load_json(data_root / "derived/metadata/metadata_manifest.json")
    metadata_rows = read_csv(data_root / "derived/metadata/metadata_features.csv")
    checks.append(
        _check(
            "metadata_schema",
            len(metadata_rows) == 50 and metadata_manifest.get("active_features") == list(CANONICAL_METADATA_FEATURES),
            f"rows={len(metadata_rows)} active={len(metadata_manifest.get('active_features', []))}",
        )
    )
    graph_rows = read_csv(data_root / "derived/graphs/combined_encoded_index.csv")
    graph_failures = _audit_graphs(graph_rows, data_root)
    checks.append(_check("graph_schema", len(graph_rows) == 50 and not graph_failures, f"rows={len(graph_rows)} failures={graph_failures[:3]}"))
    source_rows = read_csv(data_root / "derived/source_superposition/combined_encoded_index.csv")
    source_failures = _audit_index_tensor(source_rows, "source_superposition_base_path", data_root, GRID_SHAPE)
    checks.append(_check("source_superposition", len(source_rows) == 50 and not source_failures, f"rows={len(source_rows)} failures={source_failures[:3]}"))
    lineage_problems = audit_source_isolation_lineage(data_root, selection)
    checks.append(_check("source_response_lineage", not lineage_problems, "; ".join(lineage_problems[:5])))
    index_root = data_root / "derived/indices/pilot_5x10"
    final_rows = read_csv(index_root / "all_index.csv")
    path_failures = audit_index_paths(final_rows, data_root)
    checks.append(_check("model_ready_index", len(final_rows) == 50 and not path_failures, f"rows={len(final_rows)} unresolved={path_failures[:3]}"))
    split_counts = {
        protocol: {split: len(read_csv(index_root / protocol / f"{split}_index.csv")) for split in ("train", "val", "test")}
        for protocol in ("sample_split", "family_split")
    }
    checks.append(_check("sample_split_counts", split_counts["sample_split"] == {"train": 40, "val": 5, "test": 5}, str(split_counts["sample_split"])))
    checks.append(_check("family_split_counts", sum(split_counts["family_split"].values()) == 50 and all(value > 0 for value in split_counts["family_split"].values()), str(split_counts["family_split"])))
    loader_report = loader_smoke(index_root / "all_index.csv", residual_checkpoint=residual_checkpoint)
    checks.append(_check("loader_smoke", loader_report["passed"], loader_report.get("details", "")))
    relocation_report = data_root / "canonical/manifests/relocation_validation_report.json"
    if require_relocation:
        checks.append(_check("relocation", relocation_report.exists() and load_json(relocation_report).get("passed") is True))

    artifact_manifest_files = sorted((data_root / "canonical/manifests/artifacts").glob("*.json"))
    artifact_errors: list[str] = []
    for path in artifact_manifest_files:
        try:
            validate_artifact_manifest(load_json(path))
        except Exception as exc:
            artifact_errors.append(f"{path.name}: {exc}")
    checks.append(_check("artifact_manifests", len(artifact_manifest_files) >= 8 and not artifact_errors, f"count={len(artifact_manifest_files)} errors={artifact_errors[:3]}"))

    passed = all(bool(item["passed"]) for item in checks)
    artifact_roots = {
        "canonical_families": data_root / "canonical/families",
        "canonical_workloads": data_root / "canonical/workloads",
        "canonical_hotspot_labels": data_root / "canonical/hotspot_labels",
        "canonical_source_isolation": data_root / "canonical/source_isolation",
        "encoded_13ch": data_root / "derived/encoded_13ch",
        "context_17ch": data_root / "derived/context_17ch",
        "context_33ch": data_root / "derived/context_33ch",
        "metadata": data_root / "derived/metadata",
        "graphs": data_root / "derived/graphs",
        "source_superposition": data_root / "derived/source_superposition",
        "indices": data_root / "derived/indices/pilot_5x10",
        "checkpoints": data_root / "checkpoints",
    }
    bytes_by_artifact = {name: _tree_size(path) for name, path in artifact_roots.items()}
    total_bytes = sum(bytes_by_artifact.values())
    scratch_bytes = _tree_size(data_root / "staging")
    hotspot_runtime = float(build_report.get("runtime_s", 0.0))
    report = {
        "schema_version": "benchmark_v2_pilot_strict_validation/1",
        "benchmark_id": BENCHMARK_ID,
        "passed": passed,
        "status": "validated" if passed else "failed",
        "selected_family_uids": selected,
        "workload_count": len(workloads),
        "split_counts": split_counts,
        "checks": checks,
        "loader_smoke": loader_report,
        "portable_path_audit": portable,
        "storage": {
            "pilot_bytes": total_bytes,
            "bytes_per_sample": total_bytes / 50.0,
            "projected_10000_sample_bytes": total_bytes / 50.0 * 10000.0,
            "bytes_by_artifact_class": bytes_by_artifact,
            "scratch_bytes_at_validation": scratch_bytes,
            "peak_scratch_usage_bytes_observed_lower_bound": scratch_bytes,
        },
        "runtime": {
            "pilot_wall_clock_s": hotspot_runtime,
            "projected_10000_sample_s_linear": hotspot_runtime / 50.0 * 10000.0 if hotspot_runtime else None,
        },
        "build_report": build_report,
        "recommendation": "GO" if passed else "NO-GO",
    }
    write_json(data_root / "canonical/manifests/pilot_5x10_strict_validation.json", report)
    write_json(data_root / "canonical/manifests/pilot_5x10_validation_report.json", report)
    if not passed:
        failed = [item for item in checks if not item["passed"]]
        raise ValueError(f"strict pilot validation failed: {failed}")
    return report


def audit_portable_documents(data_root: Path) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    roots = [data_root / "canonical", data_root / "derived"]
    extensions = {".csv", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".md"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in extensions:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for prefix in PORTABLE_FORBIDDEN_PREFIXES:
                if prefix in text:
                    violations.append({"path": str(path.relative_to(data_root)), "prefix": prefix})
            if path.suffix == ".csv":
                for row in read_csv(path):
                    for key in PATH_COLUMNS:
                        value = row.get(key, "")
                        if value and Path(value).is_absolute():
                            violations.append({"path": str(path.relative_to(data_root)), "prefix": f"absolute:{key}"})
    return {"files_scanned": sum(1 for root in roots if root.exists() for path in root.rglob("*") if path.is_file()), "violation_count": len(violations), "violations": violations[:100]}


def audit_index_paths(rows: Sequence[dict[str, str]], data_root: Path) -> list[str]:
    failures: list[str] = []
    for row in rows:
        for key in PATH_COLUMNS:
            value = row.get(key, "")
            if not value:
                continue
            if Path(value).is_absolute():
                failures.append(f"{row.get('sample_uid')} {key} is absolute")
            elif not (data_root / value).exists():
                failures.append(f"{row.get('sample_uid')} {key} missing: {value}")
    return failures


def audit_source_isolation_lineage(data_root: Path, selection: dict[str, Any]) -> list[str]:
    root = data_root / "canonical/source_isolation"
    if not root.exists():
        return ["source isolation root is missing"]
    problems: list[str] = []
    allowed = set(selection["source_response_policy"]["train_eligible_families"])
    for split in ("train", "val", "test"):
        path = root / f"{split}_index.csv"
        if not path.exists():
            problems.append(f"missing source isolation {split} index")
            continue
        rows = read_csv(path)
        for row in rows:
            family = row.get("case_id", "")
            if split == "train" and family not in allowed:
                problems.append(f"forbidden train source family {family}")
            if split != "train" and family in allowed:
                problems.append(f"train family {family} placed in oracle split {split}")
    return problems


def loader_smoke(index_path: Path, *, residual_checkpoint: str | Path | None = None) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from .ml.dataset import ChipThermDataset, chiptherm_collate

        dataset = ChipThermDataset(index_path, target="residual", return_metadata=True, return_graph=True)
        loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, collate_fn=chiptherm_collate)
        batch = next(iter(loader))
        if tuple(batch["x"].shape[1:]) != (33, 64, 64):
            raise ValueError(f"unexpected X batch shape {tuple(batch['x'].shape)}")
        if tuple(batch["physics"].shape[1:]) != GRID_SHAPE:
            raise ValueError(f"unexpected source-base shape {tuple(batch['physics'].shape)}")
        if tuple(batch["metadata_vector"].shape[1:]) != (15,):
            raise ValueError(f"unexpected metadata shape {tuple(batch['metadata_vector'].shape)}")
        if int(batch["graph"]["node_features"].shape[1]) != 24 or int(batch["graph"]["edge_features"].shape[1]) != 15:
            raise ValueError("graph feature dimensions differ from 24/15")
        report: dict[str, Any] = {
            "passed": True,
            "samples": len(dataset),
            "x_shape": list(batch["x"].shape),
            "model_input_channels": 34,
            "metadata_dim": 15,
            "graph_node_dim": 24,
            "graph_edge_dim": 15,
            "forward_smoke": "not_requested",
        }
        if residual_checkpoint is not None:
            report.update(_checkpoint_forward_smoke(batch, Path(residual_checkpoint), torch.device("cpu")))
        return report
    except Exception as exc:
        return {"passed": False, "details": f"{type(exc).__name__}: {exc}"}


def _checkpoint_forward_smoke(batch: dict[str, Any], checkpoint_path: Path, device: Any) -> dict[str, Any]:
    import torch
    from .ml.graph_models import move_graph_to_device, normalize_graph_batch
    from .ml.models import build_model
    from .ml.normalization import NormalizationStats, build_metadata_input, build_model_input

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    if str(config.get("physics_input_mode")) != "source_superposition_v1":
        raise ValueError("residual smoke checkpoint must use source_superposition_v1")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = NormalizationStats(**checkpoint["normalization"])
    model_input = build_model_input(batch["x"].to(device), batch["physics"].to(device), stats, physics_input_mode="source_superposition_v1")
    metadata = build_metadata_input(batch["metadata_vector"].to(device), stats)
    graph = normalize_graph_batch(move_graph_to_device(batch["graph"], device), config.get("graph_normalization"))
    kwargs: dict[str, Any] = {"return_diagnostics": True}
    if str(config.get("mean_head_mode", "direct_k")) == "residual_resistance":
        kwargs["total_power_W"] = batch["total_power_W"].to(device)
    with torch.inference_mode():
        output = model(model_input, metadata, graph, **kwargs)
    final = output["final_temperature"] if isinstance(output, dict) else output
    if final.numel() == 0 or not torch.isfinite(final).all():
        raise ValueError("checkpoint forward returned empty or non-finite output")
    return {"forward_smoke": "passed", "forward_output_shape": list(final.shape), "checkpoint_sha256": sha256_file(checkpoint_path)}


def relocate_pilot(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    residual_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    if destination_root.exists():
        raise FileExistsError(f"relocation destination already exists: {destination_root}")
    shutil.copytree(source_root, destination_root, ignore=shutil.ignore_patterns("staging"))
    source_tree = make_tree_manifest(source_root, exclude_names={"tree_manifest.json"})
    destination_tree = make_tree_manifest(destination_root, exclude_names={"tree_manifest.json"})
    source_hashes = {item["path"]: item["sha256"] for item in source_tree["files"] if not item["path"].startswith("staging/")}
    destination_hashes = {item["path"]: item["sha256"] for item in destination_tree["files"]}
    hash_match = source_hashes == destination_hashes
    strict = validate_pilot_root(destination_root, residual_checkpoint=residual_checkpoint)
    portable = audit_portable_documents(destination_root)
    report = {
        "schema_version": "benchmark_v2_relocation_validation/1",
        "benchmark_id": BENCHMARK_ID,
        "passed": bool(hash_match and strict["passed"] and portable["violation_count"] == 0),
        "hash_match": hash_match,
        "source_file_count": len(source_hashes),
        "destination_file_count": len(destination_hashes),
        "loaded_samples": strict.get("loader_smoke", {}).get("samples", 0),
        "portable_path_violations": portable["violation_count"],
        "loader_smoke": strict.get("loader_smoke"),
    }
    write_json(destination_root / "canonical/manifests/relocation_validation_report.json", report)
    write_json(source_root / "canonical/manifests/relocation_validation_report.json", report)
    if not report["passed"]:
        raise ValueError(f"relocation validation failed: {report}")
    return report


def _audit_npy(paths: Sequence[Path], shape: tuple[int, ...]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        try:
            array = np.load(path, mmap_mode="r")
            if tuple(array.shape) != shape or not np.isfinite(np.asarray(array)).all():
                failures.append(f"{path}: shape={array.shape} finite={np.isfinite(np.asarray(array)).all()}")
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    return failures


def _audit_index_tensor(rows: Sequence[dict[str, str]], column: str, data_root: Path, shape: tuple[int, ...]) -> list[str]:
    paths = [resolve_data_path(row[column], data_root) for row in rows if row.get(column)]
    if len(paths) != len(rows):
        return [f"{column} missing for {len(rows) - len(paths)} rows"]
    return _audit_npy(paths, shape)


def _audit_graphs(rows: Sequence[dict[str, str]], data_root: Path) -> list[str]:
    failures: list[str] = []
    for row in rows:
        try:
            with np.load(resolve_data_path(row["graph_path"], data_root)) as graph:
                if graph["node_features"].ndim != 2 or graph["node_features"].shape[1] != 24:
                    raise ValueError(f"node shape {graph['node_features'].shape}")
                if graph["edge_features"].ndim != 2 or graph["edge_features"].shape[1] != 15:
                    raise ValueError(f"edge shape {graph['edge_features'].shape}")
                if not np.isfinite(graph["node_features"]).all() or not np.isfinite(graph["edge_features"]).all():
                    raise ValueError("non-finite graph features")
        except Exception as exc:
            failures.append(f"{row.get('sample_uid')}: {exc}")
    return failures


def _check(name: str, passed: bool, details: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
