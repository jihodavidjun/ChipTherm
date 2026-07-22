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
from .benchmark_v2_workloads import (
    PHASE2_STAGE,
    PHASE3_STAGE,
    load_family,
    scale_workload_cells,
    validate_workload,
    write_workload_tree,
)
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
PILOT_STAGE = PHASE2_STAGE
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
    "source_checkpoint",
    "target_rise_path",
    "temperature_path",
    "final_temperature",
    "final_temperature_path",
    "full_temperature_path",
    "original_x_path",
    "original_y_path",
}
PORTABLE_FORBIDDEN_PREFIXES = ("/Users/", "/nethome/", "/tmp/", "/export/hdd/")
INFORMATIONAL_PATH_SEMANTICS = "informational_nonresolving"
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
class PilotStageSpec:
    name: str
    family_count: int
    workloads_per_family: int
    sample_count: int
    selection_path: Path
    stage_namespaced: bool


STAGE_SPECS: dict[str, PilotStageSpec] = {
    PHASE2_STAGE: PilotStageSpec(
        PHASE2_STAGE,
        5,
        10,
        50,
        REPO_ROOT / "configs/benchmark_v2_50family/pilot_5x10.yaml",
        False,
    ),
    PHASE3_STAGE: PilotStageSpec(
        PHASE3_STAGE,
        10,
        50,
        500,
        REPO_ROOT / "configs/benchmark_v2_50family/pilot_10x50.yaml",
        True,
    ),
}


def stage_spec(stage: str) -> PilotStageSpec:
    try:
        return STAGE_SPECS[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported Benchmark v2 stage {stage!r}; choices={sorted(STAGE_SPECS)}") from exc


@dataclass(frozen=True)
class PilotPaths:
    data_root: Path
    scratch_root: Path
    run_id: str
    stage: str = PILOT_STAGE

    @property
    def run_root(self) -> Path:
        return self.scratch_root / "runs" / self.run_id

    def canonical(self, name: str) -> Path:
        if stage_spec(self.stage).stage_namespaced and name != "manifests":
            return self.data_root / "canonical" / "stages" / self.stage / name
        return self.data_root / "canonical" / name

    def derived(self, name: str) -> Path:
        if name == "indices":
            return self.data_root / "derived" / "indices"
        if stage_spec(self.stage).stage_namespaced:
            return self.data_root / "derived" / "stages" / self.stage / name
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
    stage: str = PILOT_STAGE


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


def phase2_immutability_snapshot(data_root: Path) -> dict[str, Any]:
    """Hash accepted Phase 2 files without including any Phase 3 namespace."""
    roots = [
        "canonical/families",
        "canonical/workloads",
        "canonical/hotspot_labels",
        "canonical/source_isolation",
        "derived/encoded_13ch",
        "derived/context_17ch",
        "derived/context_33ch",
        "derived/metadata",
        "derived/graphs",
        "derived/source_superposition",
        "derived/indices/pilot_5x10",
    ]
    files: list[dict[str, Any]] = []
    for relative in roots:
        root = data_root / relative
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": data_root_relative(path, data_root),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    manifest_root = data_root / "canonical/manifests"
    for pattern in ("pilot_5x10_*.json", "artifacts/pilot_5x10_*.json"):
        for path in sorted(manifest_root.glob(pattern)):
            files.append(
                {
                    "path": data_root_relative(path, data_root),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return {
        "schema_version": "benchmark_v2_phase2_immutability/1",
        "available": bool(files),
        "file_count": len(files),
        "content_sha256": sha256_json([(row["path"], row["sha256"]) for row in files]),
        "files": files,
    }


def ensure_phase2_snapshot(data_root: Path) -> dict[str, Any]:
    path = data_root / "canonical/manifests/pilot_10x50_phase2_snapshot.json"
    current = phase2_immutability_snapshot(data_root)
    if path.exists():
        expected = load_json(path)
        if expected.get("available") and current.get("content_sha256") != expected.get("content_sha256"):
            raise ValueError("accepted Phase 2 artifacts changed after the Phase 3 immutability snapshot")
        return expected
    write_json(path, current)
    return current


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
    stage = str(selection.get("stage", PILOT_STAGE))
    spec = stage_spec(stage)
    rows = list(selection.get("selected_families", []))
    configured = [str(row["family_uid"]) for row in rows]
    selected = list(selected_override or configured)
    if len(selected) != spec.family_count or len(set(selected)) != spec.family_count:
        raise ValueError(f"{stage} requires exactly {spec.family_count} unique selected families")
    if selected_override:
        role_by_uid = {str(row["family_uid"]): row for row in rows}
        missing = [uid for uid in selected if uid not in role_by_uid]
        if missing:
            raise ValueError(f"selected overrides are absent from pilot selection config: {missing}")
        selection["selected_families"] = [role_by_uid[uid] for uid in selected]
    declared_hash = selection.get("selection_content_sha256")
    if declared_hash:
        hash_payload = dict(selection)
        hash_payload.pop("selection_content_sha256", None)
        actual_hash = sha256_json(hash_payload)
        if declared_hash != actual_hash:
            raise ValueError(f"{stage} selection content hash mismatch: expected {declared_hash}, got {actual_hash}")
    workload_design = selection.get("workload_design", {})
    cell_spec_value = workload_design.get("workload_cell_spec_path")
    if cell_spec_value:
        cell_spec_path = REPO_ROOT / str(cell_spec_value)
        expected_cell_hash = str(workload_design.get("workload_cell_spec_sha256", ""))
        if not cell_spec_path.is_file() or sha256_file(cell_spec_path) != expected_cell_hash:
            raise ValueError(f"{stage} workload-cell specification hash mismatch: {cell_spec_path}")
        frozen_cells = load_yaml(cell_spec_path).get("cells", [])
        generated_cells = scale_workload_cells()
        frozen_identity = [
            (int(row["workload_ordinal"]), str(row["power_regime"]), str(row["topology_regime"]))
            for row in frozen_cells
        ]
        generated_identity = [
            (int(row["workload_ordinal"]), str(row["power_regime"]), str(row["topology_regime"]))
            for row in generated_cells
        ]
        if frozen_identity != generated_identity:
            raise ValueError(f"{stage} workload generator differs from the frozen 50-cell specification")
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
    expected = {
        str(family["family_uid"]): sha256_file(family_dir / f"{family['family_uid']}.yaml")
        for family in families
    }
    if durable_stage_complete(destination) and (destination / "selected_family_manifest.json").exists():
        manifest = load_json(destination / "selected_family_manifest.json")
        existing = {str(row["family_uid"]): str(row["sha256"]) for row in manifest.get("families", [])}
        if existing != expected or manifest.get("phase1_family_manifest_sha256") != phase1_manifest_sha256:
            raise ValueError(f"resumed selected-family stage differs from the requested immutable selection: {destination}")
        return
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
    stage_name: str = PILOT_STAGE,
) -> dict[str, Any]:
    spec = stage_spec(stage_name)
    if resume and durable_stage_complete(destination) and (destination / "workload_manifest.yaml").exists():
        manifest = load_yaml(destination / "workload_manifest.yaml")
        if (
            manifest.get("stage") == stage_name
            and manifest.get("family_uids") == sorted(str(item["family_uid"]) for item in families)
            and manifest.get("base_seed") == seed
            and int(manifest.get("workloads_per_family", -1)) == spec.workloads_per_family
            and int(manifest.get("workload_count", -1)) == spec.sample_count
        ):
            for family in families:
                for ordinal in range(1, spec.workloads_per_family + 1):
                    matches = list((destination / str(family["family_uid"])).glob(f"w{ordinal:03d}_*.yaml"))
                    if len(matches) != 1 or validate_workload(load_yaml(matches[0]), family):
                        raise ValueError(f"invalid resumed workload for {family['family_uid']} ordinal {ordinal}")
            return manifest
        raise ValueError(f"resumed workload stage does not match {stage_name} selection/seed/count: {destination}")
    stage = run_root / "workloads"
    if stage.exists():
        shutil.rmtree(stage)
    manifest = write_workload_tree(families, stage, base_seed=seed, stage=stage_name)
    promote_directory(stage, destination, resume=resume)
    return manifest


def workload_rows(
    workload_root: Path,
    families: Mapping[str, dict[str, Any]],
    *,
    stage_name: str = PILOT_STAGE,
) -> list[dict[str, Any]]:
    expected = stage_spec(stage_name).workloads_per_family
    rows: list[dict[str, Any]] = []
    for family_uid in sorted(families):
        paths = sorted((workload_root / family_uid).glob("w*.yaml"))
        if len(paths) != expected:
            raise ValueError(f"{family_uid} must have exactly {expected} {stage_name} workloads, found {len(paths)}")
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
    reused_phase2: list[str] = []
    for workload in workloads:
        family_uid = str(workload["family_uid"])
        sample_uid = str(workload["sample_uid"])
        accepted = accepted_root / family_uid / sample_uid
        if resume and validated_existing_sample(accepted, families[family_uid], workload):
            skipped.append(sample_uid)
            continue
        phase2_sample = paths.data_root / "canonical" / "hotspot_labels" / family_uid / sample_uid
        if (
            paths.stage == PHASE3_STAGE
            and bool(workload.get("phase2_reference"))
            and validated_existing_sample(phase2_sample, families[family_uid], workload)
        ):
            reused_phase2.append(sample_uid)
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
    peak_staging_bytes = _tree_size(paths.run_root)
    executable_sha = "dry_run"
    if not dry_run:
        if hotspot_home is None:
            raise ValueError("--hotspot-home is required unless --dry-run is used")
        executable = hotspot_home / "hotspot"
        if not executable.is_file():
            raise FileNotFoundError(executable)
        executable_sha = sha256_file(executable)
        progress_started = time.perf_counter()
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
                processed = len(completed) + len(failures)
                peak_staging_bytes = max(peak_staging_bytes, _tree_size(paths.run_root))
                if processed == len(scheduled) or processed % max(1, min(10, len(scheduled))) == 0:
                    elapsed = time.perf_counter() - progress_started
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    eta = (len(scheduled) - processed) / rate if rate > 0 else None
                    promoted_bytes = sum(
                        _tree_size(accepted_root / str(item["family_uid"]) / str(item["sample_uid"]))
                        for item, _ in scheduled
                        if str(item["sample_uid"]) in completed
                    )
                    print(
                        f"HotSpot progress {processed}/{len(scheduled)} "
                        f"family={workload['family_uid']} successes={len(completed)} failures={len(failures)} "
                        f"retries={retry_count} elapsed_s={elapsed:.1f} "
                        f"eta_s={eta:.1f} promoted_bytes={promoted_bytes}"
                        if eta is not None
                        else f"HotSpot progress {processed}/{len(scheduled)} family={workload['family_uid']}"
                    )
    report = {
        "schema_version": "benchmark_v2_hotspot_generation_report/1",
        "requested": len(workloads),
        "scheduled": len(scheduled),
        "completed": len(completed),
        "skipped_valid": len(skipped),
        "reused_phase2": len(reused_phase2),
        "failed": len(failures),
        "retry_count": retry_count,
        "completed_uids": sorted(completed),
        "skipped_uids": sorted(skipped),
        "reused_phase2_uids": sorted(reused_phase2),
        "failures": failures,
        "hotspot_executable_sha256": executable_sha,
        "dry_run": dry_run,
        "dry_run_source_count": len(workloads) - len(reused_phase2) if dry_run else 0,
        "peak_staging_bytes_observed": peak_staging_bytes,
    }
    write_json(paths.run_root / "hotspot_generation_report.json", report)
    write_csv(paths.run_root / "hotspot_failures.csv", failures)
    if failures:
        raise RuntimeError(f"HotSpot failed for {len(failures)} pilot samples; staging outputs were retained")
    if not dry_run:
        collection_rows: list[dict[str, Any]] = []
        reused_set = set(reused_phase2)
        for workload in workloads:
            family_uid = str(workload["family_uid"])
            sample_uid = str(workload["sample_uid"])
            sample_root = accepted_root / family_uid / sample_uid
            if sample_uid in reused_set:
                sample_root = paths.data_root / "canonical/hotspot_labels" / family_uid / sample_uid
            if not validated_existing_sample(sample_root, families[family_uid], workload):
                raise ValueError(f"logical HotSpot collection contains an invalid sample after generation: {sample_uid}")
            collection_rows.append(
                {
                    "sample_uid": sample_uid,
                    "family_uid": family_uid,
                    "workload_uid": workload["workload_uid"],
                    "workload_content_sha256": workload["content_hash"],
                    "sample_root": data_root_relative(sample_root, paths.data_root),
                    "target_path": data_root_relative(sample_root / "parsed/temp_layer0.npy", paths.data_root),
                    "ownership_stage": PHASE2_STAGE if sample_uid in reused_set else paths.stage,
                    "reused_by_content_hash": str(sample_uid in reused_set).lower(),
                }
            )
        write_csv(accepted_root / "sample_index.csv", collection_rows)
        write_json(
            accepted_root / "collection_manifest.json",
            {
                "schema_version": "benchmark_v2_hotspot_collection/1",
                "stage": paths.stage,
                "sample_count": len(collection_rows),
                "reused_phase2_count": len(reused_phase2),
                "path_semantics": PATH_SEMANTICS,
            },
        )
        completion = make_tree_manifest(accepted_root, exclude_names={".stage_complete.json"})
        write_json(
            accepted_root / ".stage_complete.json",
            {
                "schema_version": "benchmark_v2_stage_completion/1",
                "file_count": completion["file_count"],
                "files": [{"path": item["path"], "sha256": item["sha256"]} for item in completion["files"]],
            },
        )
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
        if not (sample_root / "parsed/temp_layer0.npy").exists() and paths.stage == PHASE3_STAGE:
            phase2_sample = paths.data_root / "canonical" / "hotspot_labels" / family_uid / uid
            if bool(workload.get("phase2_reference")) and validated_existing_sample(
                phase2_sample, families[family_uid], workload
            ):
                sample_root = phase2_sample
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
                "workload_cell": str(workload.get("workload_cell", workload["stratum"])),
                "power_regime": str(workload.get("power_regime", "phase2_reference")),
                "topology_regime": str(workload.get("topology_regime", workload["stratum"])),
                "active_chiplet_fraction": str(workload.get("active_chiplet_fraction", "")),
                "dominant_chiplet_share": str(workload.get("dominant_chiplet_share", "")),
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

    source_checkpoint_digest = sha256_file(source_checkpoint)
    expected_installed_checkpoint = paths.data_root / "checkpoints/source_response" / f"{source_checkpoint.stem}_{source_checkpoint_digest[:12]}{source_checkpoint.suffix}"
    source_checkpoint_preexisting = expected_installed_checkpoint.is_file()
    portable_checkpoint = install_checkpoint(source_checkpoint, paths.data_root, "source_response")
    installed_lineage_path = portable_checkpoint.with_suffix(".lineage.json")
    if installed_lineage_path.exists():
        if sha256_json(load_json(installed_lineage_path)) != sha256_json(source_lineage):
            raise ValueError(f"installed source-response lineage differs from requested lineage: {installed_lineage_path}")
    else:
        write_json(installed_lineage_path, source_lineage)
    source_model_dest = paths.derived("source_response_model")
    if not (resume and durable_stage_complete(source_model_dest)):
        source_model_stage = derived_run / "source_response_model"
        archive_failed_stage(source_model_stage, paths.scratch_root, paths.run_id)
        source_model_stage.mkdir(parents=True, exist_ok=True)
        write_json(
            source_model_stage / "checkpoint_reference.json",
            {
                "schema_version": "benchmark_v2_checkpoint_reference/1",
                "checkpoint_path": data_root_relative(portable_checkpoint, paths.data_root),
                "checkpoint_sha256": sha256_file(portable_checkpoint),
                "lineage_sha256": sha256_json(source_lineage),
                "path_semantics": PATH_SEMANTICS,
            },
        )
        promote_directory(source_model_stage, source_model_dest, resume=False)
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
        "source_response_model": source_model_dest,
        "source_checkpoint_preexisting": source_checkpoint_preexisting,
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
    selection: dict[str, Any] | None = None,
) -> Path:
    destination = paths.canonical("source_isolation")
    if resume and durable_stage_complete(destination):
        return destination
    stage = paths.run_root / "source_isolation"
    reused_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    reused_families: set[str] = set()
    if paths.stage == PHASE3_STAGE and selection is not None:
        overlap = {
            str(row["family_uid"])
            for row in selection["selected_families"]
            if bool(row.get("phase2_overlap"))
        }
        phase2_root = paths.data_root / "canonical/source_isolation"
        if durable_stage_complete(phase2_root):
            for split in ("train", "val", "test"):
                for row in read_csv(phase2_root / f"{split}_index.csv"):
                    family_uid = str(row.get("case_id", row.get("family_uid", "")))
                    if family_uid not in overlap:
                        continue
                    target = str(row.get("target_rise_path", ""))
                    if not target or not resolve_data_path(target, paths.data_root).is_file():
                        raise FileNotFoundError(f"cannot reuse Phase 2 source-isolation target for {family_uid}: {target!r}")
                    reused_rows[split].append(dict(row))
                    reused_families.add(family_uid)
    generated_families = [uid for uid in selected_families if uid not in reused_families]
    command = [
        sys.executable,
        "scripts/build_source_response_dataset.py",
        "--train-index", str(isolation_inputs["train"]),
        "--val-index", str(isolation_inputs["val"]),
        "--test-index", str(isolation_inputs["test"]),
        "--data-root", str(paths.data_root),
        "--out-root", str(stage),
        "--cases", *generated_families,
        "--samples-per-case", "1",
        "--hotspot-home", str(hotspot_home),
        "--config-template", str(config_template),
    ]
    if resume:
        command.append("--resume")
    run_checked(command, paths.run_root)
    if reused_families:
        combined: list[dict[str, str]] = []
        for split in ("train", "val", "test"):
            generated = read_csv(stage / f"{split}_index.csv")
            rows = generated + reused_rows[split]
            rows.sort(key=lambda row: (row.get("case_id", ""), row.get("original_sample_uid", ""), int(row.get("source_index") or 0)))
            write_csv(stage / f"{split}_index.csv", rows)
            combined.extend(rows)
        write_csv(stage / "combined_source_index.csv", combined)
        with (stage / "combined_source_index.jsonl").open("w", encoding="utf-8") as handle:
            for row in combined:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest_path = stage / "source_response_manifest.json"
        manifest = load_json(manifest_path)
        manifest["actual_source_rows"] = {
            split: len(read_csv(stage / f"{split}_index.csv")) for split in ("train", "val", "test")
        }
        manifest["actual_total_source_rows"] = len(combined)
        manifest["phase2_reuse"] = {
            "family_uids": sorted(reused_families),
            "source_rows": sum(len(rows) for rows in reused_rows.values()),
            "ownership": "root-relative references to accepted immutable Phase 2 targets; arrays are not duplicated",
        }
        write_json(manifest_path, manifest)
        (stage / "README.md").write_text(
            "# ChipTherm Benchmark v2 Phase 3 source isolation\n\n"
            f"Source rows: {len(combined)}. Reused Phase 2 families: {', '.join(sorted(reused_families))}.\n",
            encoding="utf-8",
        )
    write_json(
        stage / "phase2_reuse_report.json",
        {
            "schema_version": "benchmark_v2_phase2_source_isolation_reuse/1",
            "reused_family_uids": sorted(reused_families),
            "reused_source_rows": sum(len(rows) for rows in reused_rows.values()),
            "generated_family_uids": generated_families,
        },
    )
    canonicalize_stage_indices(stage, destination, paths.data_root)
    promote_directory(stage, destination, resume=False)
    return destination


def create_final_indices(
    paths: PilotPaths,
    source_root: Path,
    selection: dict[str, Any],
    manifest_ids: Mapping[str, str],
    *,
    resume: bool = False,
) -> Path:
    rows = read_csv(source_root / "combined_encoded_index.csv")
    spec = stage_spec(paths.stage)
    if len(rows) != spec.sample_count:
        raise ValueError(f"source-superposition index must contain {spec.sample_count} rows, got {len(rows)}")
    output = paths.derived("indices") / paths.stage
    if resume and durable_stage_complete(output):
        return output
    stage = paths.run_root / "indices"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    enriched: list[dict[str, str]] = []
    for row in rows:
        result = dict(row)
        result["benchmark_id"] = BENCHMARK_ID
        result["protocol_id"] = f"{paths.stage}_all"
        result["metadata_row_id"] = row["sample_uid"]
        result["x_artifact_id"] = f"{paths.stage}_context_33ch"
        result["y_artifact_id"] = f"{paths.stage}_hotspot_labels"
        result["graph_artifact_id"] = f"{paths.stage}_graphs"
        result["metadata_artifact_id"] = f"{paths.stage}_metadata"
        result["source_superposition_artifact_id"] = f"{paths.stage}_source_superposition"
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
                item["protocol_id"] = f"{paths.stage}_{protocol}"
                subset.append(item)
            write_csv(stage / protocol / f"{split}_index.csv", subset, fieldnames=list(enriched[0].keys()))
    write_csv(
        stage / "workload_coverage.csv",
        [
            {
                "family_uid": row["family_uid"],
                "sample_uid": row["sample_uid"],
                "workload_uid": row["workload_uid"],
                "workload_cell": row.get("workload_cell", row.get("workload_stratum", "")),
                "power_regime": row.get("power_regime", ""),
                "topology_regime": row.get("topology_regime", ""),
                "sample_split": row.get("sample_split", ""),
                "family_split": row.get("family_split", ""),
            }
            for row in enriched
        ],
    )
    selected_by_uid = {str(item["family_uid"]): item for item in selection["selected_families"]}
    write_csv(
        stage / "family_coverage.csv",
        [
            {
                "family_uid": family_uid,
                "taxonomy_category": selected_by_uid[family_uid].get("taxonomy_category", selected_by_uid[family_uid].get("role", "")),
                "primary_split": selected_by_uid[family_uid]["primary_split"],
                "source_isolation_eligibility": selected_by_uid[family_uid].get("source_isolation_eligibility", ""),
                "phase2_overlap": str(bool(selected_by_uid[family_uid].get("phase2_overlap", False))).lower(),
                "sample_count": sum(row["family_uid"] == family_uid for row in enriched),
                "workload_cell_count": len({row.get("workload_cell", "") for row in enriched if row["family_uid"] == family_uid}),
            }
            for family_uid in sorted(selected_by_uid)
        ],
    )
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
    expected = len(read_csv(source_index)) if source_index is not None else encoded
    if encoded != expected or failed:
        raise RuntimeError(f"13-channel encoding expected {expected}/0 encoded/failed, got {encoded}/{failed}")
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
    manifest_path = stage_root / "manifest.json"
    if manifest_path.exists():
        payload = load_json(manifest_path)
        payload["source_checkpoint"] = data_root_relative(checkpoint, data_root)
        payload["source_checkpoint_sha256"] = sha256_file(checkpoint)
        payload["source_checkpoint_lineage_sha256"] = lineage_hash
        write_json(manifest_path, payload)
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
    tree = make_tree_manifest(artifact_root, exclude_names={"tree_manifest.json", ".stage_complete.json"})
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
    if (artifact_root / ".stage_complete.json").exists():
        completion = make_tree_manifest(artifact_root, exclude_names={".stage_complete.json"})
        write_json(
            artifact_root / ".stage_complete.json",
            {
                "schema_version": "benchmark_v2_stage_completion/1",
                "file_count": completion["file_count"],
                "files": [{"path": item["path"], "sha256": item["sha256"]} for item in completion["files"]],
            },
        )
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
        "stage": options.stage,
        "created_at": utc_now(),
        "phase1_family_manifest_sha256": phase1_hash,
        "selected_family_uids": list(options.selected_families),
        "workers": int(options.workers),
        "seed": int(options.seed),
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
    spec = stage_spec(options.stage)
    if options.workers <= 0:
        raise ValueError("workers must be positive")
    design = load_yaml(options.config_path)
    if design.get("benchmark_id") != BENCHMARK_ID or design.get("stage_gates", {}).get("full_generation_approved") is not False:
        raise ValueError("--config is not the approved Benchmark v2 staged design proposal")
    selection = load_selection(options.selection_path, options.selected_families)
    if selection.get("stage") != options.stage:
        raise ValueError(f"selection stage {selection.get('stage')!r} does not match requested stage {options.stage!r}")
    family_manifest_path = options.family_dir.parent / "family_manifest.yaml"
    verify_parent_lock(options.parent_lock_path, family_manifest_path=family_manifest_path)
    families_list, phase1_hash = verify_phase1_families(options.family_dir, family_manifest_path, options.selected_families)
    families = {str(item["family_uid"]): item for item in families_list}
    paths = PilotPaths(options.data_root.resolve(), options.scratch_root.resolve(), options.run_id, options.stage)
    ensure_root_layout(paths.data_root, paths.scratch_root)
    phase2_snapshot = ensure_phase2_snapshot(paths.data_root) if options.stage == PHASE3_STAGE else None
    paths.run_root.mkdir(parents=True, exist_ok=True)
    copy_selected_families(families_list, options.family_dir, paths.canonical("families"), phase1_manifest_sha256=phase1_hash)
    workload_manifest = prepare_workloads(
        families_list,
        paths.canonical("workloads"),
        seed=options.seed,
        run_root=paths.run_root,
        resume=options.resume,
        stage_name=options.stage,
    )
    workloads = workload_rows(paths.canonical("workloads"), families, stage_name=options.stage)
    runtime_lock = runtime_dependency_lock(
        options,
        phase1_hash=phase1_hash,
        workload_manifest_path=paths.canonical("workloads") / "workload_manifest.yaml",
        selection=selection,
    )
    runtime_lock_path = paths.canonical("manifests") / (
        "runtime_dependency_lock.json" if options.stage == PHASE2_STAGE else f"{options.stage}_runtime_dependency_lock.json"
    )
    write_json(runtime_lock_path, runtime_lock)
    hotspot_started = time.perf_counter()
    preparation_stage_runtime = hotspot_started - started
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
    hotspot_stage_runtime = time.perf_counter() - hotspot_started
    if options.dry_run:
        report = {
            "schema_version": "benchmark_v2_pilot_validation_report/1",
            "benchmark_id": BENCHMARK_ID,
            "stage": options.stage,
            "status": "dry_run_validated",
            "run_id": options.run_id,
            "selected_family_uids": list(options.selected_families),
            "workload_count": len(workloads),
            "hotspot": hotspot_report,
            "runtime_by_stage_s": {
                "selection_workload_preparation": preparation_stage_runtime,
                "hotspot_planning": hotspot_stage_runtime,
            },
            "derived_stages_run": False,
            "runtime_s": time.perf_counter() - started,
            "phase2_immutability_snapshot": phase2_snapshot,
            "recommendation": "GO WITH MANUAL REVIEW: run real HotSpot and derived stages before scale authorization.",
        }
        write_json(paths.canonical("manifests") / f"{options.stage}_build_report.json", report)
        write_json(paths.canonical("manifests") / f"{options.stage}_validation_report.json", report)
        return report
    if options.source_checkpoint is None or options.source_lineage is None:
        raise ValueError("real pilot requires --source-checkpoint and --source-lineage")
    lineage = validate_source_checkpoint_lineage(options.source_checkpoint, options.source_lineage, selection)
    raw_rows = raw_index_rows(paths, workloads, families, selection)
    derived_started = time.perf_counter()
    derived = build_derived_pipeline(
        paths,
        raw_rows,
        selection,
        source_checkpoint=options.source_checkpoint,
        source_lineage=lineage,
        resume=options.resume,
        source_device=options.source_device,
    )
    derived_stage_runtime = time.perf_counter() - derived_started
    isolation_started = time.perf_counter()
    isolation_root = run_source_isolation(
        paths,
        derived["source_isolation_inputs"],
        options.selected_families,
        hotspot_home=options.hotspot_home,
        config_template=options.config_template,
        resume=options.resume,
        selection=selection,
    )
    isolation_stage_runtime = time.perf_counter() - isolation_started
    finalization_started = time.perf_counter()
    manifest_ids = {
        "runtime_dependency_lock_sha256": sha256_file(runtime_lock_path),
        "workload_manifest_sha256": sha256_file(paths.canonical("workloads") / "workload_manifest.yaml"),
        "phase1_family_manifest_sha256": phase1_hash,
        "source_checkpoint_lineage_sha256": sha256_json(lineage),
    }
    index_root = create_final_indices(
        paths,
        derived["source_superposition"],
        selection,
        manifest_ids,
        resume=options.resume,
    )
    artifact_manifests = write_pilot_artifact_manifests(
        paths,
        options,
        derived=derived,
        isolation_root=isolation_root,
        index_root=index_root,
    )
    finalization_stage_runtime = time.perf_counter() - finalization_started
    report = {
        "schema_version": "benchmark_v2_pilot_validation_report/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": options.stage,
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
        "runtime_by_stage_s": {
            "selection_workload_preparation": preparation_stage_runtime,
            "hotspot_generation": hotspot_stage_runtime,
            "derived_artifacts": derived_stage_runtime,
            "source_isolation": isolation_stage_runtime,
            "indices_and_manifests": finalization_stage_runtime,
        },
        "phase2_reuse": {
            "hotspot_samples": int(hotspot_report.get("reused_phase2", 0)),
            "source_isolation_artifacts": int(load_json(isolation_root / "phase2_reuse_report.json").get("reused_source_rows", 0)),
            "source_isolation_families": load_json(isolation_root / "phase2_reuse_report.json").get("reused_family_uids", []),
            "source_isolation_note": "Accepted Phase 2 target arrays are referenced root-relatively and retain single ownership.",
            "source_response_checkpoint": bool(derived.get("source_checkpoint_preexisting", False)),
            "source_response_checkpoint_sha256": sha256_file(derived["portable_source_checkpoint"]),
        },
        "phase2_immutability_snapshot": phase2_snapshot,
        "expected_sample_count": spec.sample_count,
        "recommendation": "GO WITH MANUAL REVIEW: strict validation, relocation, and visual review must pass.",
    }
    write_json(paths.canonical("manifests") / f"{options.stage}_build_report.json", report)
    write_json(paths.canonical("manifests") / f"{options.stage}_validation_report.json", report)
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
    command = ["python3", "scripts/build_benchmark_v2.py", "--stage", options.stage]
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
        row_count: int | None = None,
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
        artifact_id = f"{options.stage}_{key}"
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

    prefix = options.stage
    spec = stage_spec(options.stage)
    emit("families", paths.canonical("families"), "canonical_source", "design", row_count=spec.family_count)
    emit("workloads", paths.canonical("workloads"), "canonical_source", "workloads", parents=(f"{prefix}_families",), row_count=spec.sample_count)
    hotspot_parents = (f"{prefix}_workloads", f"{PHASE2_STAGE}_hotspot_labels") if options.stage == PHASE3_STAGE else (f"{prefix}_workloads",)
    emit("hotspot_labels", paths.canonical("hotspot_labels"), "canonical_source", "hotspot_labels", parents=hotspot_parents, row_count=spec.sample_count, tensor_schema={"shape": [64, 64], "dtype": "float32", "units": "K", "channel_schema_sha256": schemas["raster"]})
    emit("encoded_13ch", Path(derived["encoded_13ch"]), "model_specific_derived", "encoded_13ch", parents=(f"{prefix}_hotspot_labels",), row_count=spec.sample_count, tensor_schema={"shape": [13, 64, 64], "dtype": "float32", "units": "mixed", "channel_schema_sha256": schemas["raster"]})
    emit("context_17ch", Path(derived["context_17ch"]), "model_specific_derived", "context_17ch", parents=(f"{prefix}_encoded_13ch",), row_count=spec.sample_count, tensor_schema={"shape": [17, 64, 64], "dtype": "float32", "units": "mixed", "channel_schema_sha256": sha256_file(Path(derived["context_17ch"]) / "feature_manifest.json")})
    emit("context_33ch", Path(derived["context_33ch"]), "model_specific_derived", "context_33ch", parents=(f"{prefix}_context_17ch",), row_count=spec.sample_count, tensor_schema={"shape": [33, 64, 64], "dtype": "float32", "units": "mixed", "channel_schema_sha256": sha256_file(Path(derived["context_33ch"]) / "feature_manifest.json")})
    emit("metadata", Path(derived["metadata"]), "model_specific_derived", "metadata", parents=(f"{prefix}_context_33ch",), row_count=spec.sample_count, tensor_schema={"shape": [15], "dtype": "float32", "units": "mixed", "channel_schema_sha256": schemas["metadata"]})
    emit("graphs", Path(derived["graphs"]), "model_specific_derived", "graphs", parents=(f"{prefix}_context_33ch", f"{prefix}_metadata"), row_count=spec.sample_count, tensor_schema={"shape": ["N", 24], "dtype": "float32", "units": "mixed", "channel_schema_sha256": schemas["graph_node"]})
    source_model_parents = (f"{PHASE2_STAGE}_source_response_model",) if options.stage == PHASE3_STAGE and bool(derived.get("source_checkpoint_preexisting")) else ()
    emit("source_response_model", Path(derived["source_response_model"]), "model_specific_derived", "source_response_model", parents=source_model_parents, row_count=None)
    source_count = sum(len(read_csv(isolation_root / f"{split}_index.csv")) for split in ("train", "val", "test"))
    isolation_parents = (f"{prefix}_hotspot_labels", f"{PHASE2_STAGE}_source_isolation") if options.stage == PHASE3_STAGE else (f"{prefix}_hotspot_labels",)
    emit("source_isolation", isolation_root, "canonical_source", "source_isolation", parents=isolation_parents, row_count=spec.family_count, source_count=source_count)
    emit("source_superposition", Path(derived["source_superposition"]), "model_specific_derived", "source_superposition", parents=(f"{prefix}_graphs", f"{prefix}_source_response_model"), row_count=spec.sample_count, tensor_schema={"shape": [64, 64], "dtype": "float32", "units": "K", "channel_schema_sha256": sha256_json(["source_superposition_base_K"])})
    emit("indices", index_root, "generated_required_intermediate", "splits", parents=(f"{prefix}_source_superposition",), row_count=spec.sample_count, tensor_schema=None)
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


def validate_scale_pilot_root(
    data_root: str | Path,
    *,
    allow_dry_run: bool = False,
    residual_checkpoint: str | Path | None = None,
    require_relocation: bool = False,
) -> dict[str, Any]:
    """Strictly validate the isolated 10x50 scale pilot and project full-build cost."""
    data_root = Path(data_root).expanduser().resolve()
    marker = load_json(data_root / ROOT_MARKER_NAME)
    if marker.get("benchmark_id") != BENCHMARK_ID or marker.get("path_semantics") != PATH_SEMANTICS:
        raise ValueError("invalid Benchmark v2 data-root marker")
    spec = stage_spec(PHASE3_STAGE)
    paths = PilotPaths(data_root, data_root / "staging", "strict-validation", PHASE3_STAGE)
    selection = load_selection(spec.selection_path)
    selected = [str(item["family_uid"]) for item in selection["selected_families"]]
    build_report = load_json(data_root / f"canonical/manifests/{PHASE3_STAGE}_build_report.json")
    dry_run = build_report.get("status") == "dry_run_validated"
    if dry_run and not allow_dry_run:
        raise ValueError("scale pilot is a dry run; real HotSpot and derived stages are required")

    checks: list[dict[str, Any]] = []
    workload_root = paths.canonical("workloads")
    workload_manifest = load_yaml(workload_root / "workload_manifest.yaml")
    workload_files = sorted(workload_root.glob("f*/w*.yaml"))
    workloads = [load_yaml(path) for path in workload_files]
    by_family = {uid: [row for row in workloads if row.get("family_uid") == uid] for uid in selected}
    expected_cells = {str(row["workload_cell"]) for row in scale_workload_cells()}
    actual_cells = {uid: {str(row.get("workload_cell", "")) for row in rows} for uid, rows in by_family.items()}
    families = {uid: load_yaml(paths.canonical("families") / f"{uid}.yaml") for uid in selected}
    workload_problems = [
        problem
        for workload in workloads
        for problem in validate_workload(workload, families[str(workload["family_uid"])])
    ]
    hashes = [str(row.get("content_hash", "")) for row in workloads]
    fingerprints = {
        uid: str(family.get("structural_fingerprint", ""))
        for uid, family in families.items()
    }
    geometry_ok = all(
        str(families[uid].get("structural_fingerprint", "")) == fingerprints[uid]
        and len({json.dumps(families[uid]["fixed_structure"]["layout"], sort_keys=True)}) == 1
        for uid in selected
    )
    portable = audit_portable_documents(data_root)
    checks.extend(
        [
            _check("selected_family_count", len(selected) == spec.family_count and len(set(selected)) == spec.family_count, str(selected)),
            _check("workloads_per_family", all(len(rows) == spec.workloads_per_family for rows in by_family.values()), str({key: len(value) for key, value in by_family.items()})),
            _check("planned_workload_count", len(workloads) == spec.sample_count, f"count={len(workloads)}"),
            _check("workload_cell_coverage", all(cells == expected_cells for cells in actual_cells.values()), f"expected={len(expected_cells)} counts={ {key: len(value) for key, value in actual_cells.items()} }"),
            _check("unique_content_hashes", len(hashes) == len(set(hashes)) == spec.sample_count),
            _check("fixed_geometry_per_family", geometry_ok),
            _check("type_aware_power_validation", not workload_problems, "; ".join(workload_problems[:10])),
            _check("portable_paths", portable["violation_count"] == 0, f"violations={portable['violation_count']}"),
        ]
    )

    if dry_run:
        planned = int(build_report.get("hotspot", {}).get("dry_run_source_count", 0)) + int(build_report.get("hotspot", {}).get("reused_phase2", 0))
        checks.append(_check("dry_run_source_accounting", planned == spec.sample_count, f"planned_or_reused={planned}"))
        passed = all(bool(row["passed"]) for row in checks)
        report = {
            "schema_version": "benchmark_v2_scale_pilot_strict_validation/1",
            "benchmark_id": BENCHMARK_ID,
            "stage": PHASE3_STAGE,
            "status": "dry_run_passed" if passed else "dry_run_failed",
            "passed": passed,
            "checks": checks,
            "portable_path_audit": portable,
            "recommendation": "GO WITH MANUAL REVIEW" if passed else "NO-GO",
        }
        write_json(data_root / f"canonical/manifests/{PHASE3_STAGE}_strict_validation.json", report)
        if not passed:
            raise ValueError(f"scale dry-run validation failed: {[row for row in checks if not row['passed']]}")
        return report

    index_root = paths.derived("indices") / PHASE3_STAGE
    final_rows = read_csv(index_root / "all_index.csv")
    path_failures = audit_index_paths(final_rows, data_root)
    final_uids = [row.get("sample_uid", "") for row in final_rows]
    checks.append(_check("model_ready_sample_count", len(final_rows) == spec.sample_count, f"rows={len(final_rows)}"))
    checks.append(_check("zero_unresolved_paths", not path_failures, "; ".join(path_failures[:10])))
    geometry_failures: list[str] = []
    for row in final_rows:
        family_uid = str(row.get("family_uid", ""))
        try:
            layout = load_json(resolve_data_path(row["layout_path"], data_root))
            expected_layout = families[family_uid]["fixed_structure"]["layout"]
            actual_geometry = {"package_size": layout["package"]["size"], "chiplets": layout["chiplets"]}
            expected_geometry = {"package_size": expected_layout["package"]["size"], "chiplets": expected_layout["chiplets"]}
            if sha256_json(actual_geometry) != sha256_json(expected_geometry):
                geometry_failures.append(str(row.get("sample_uid", "")))
        except Exception as exc:
            geometry_failures.append(f"{row.get('sample_uid', '')}: {exc}")
    checks.append(_check("fixed_geometry_source_files", not geometry_failures, "; ".join(geometry_failures[:10])))

    label_paths = [resolve_data_path(row["y_path"], data_root) for row in final_rows if row.get("y_path")]
    label_failures = _audit_npy(label_paths, GRID_SHAPE)
    checks.append(_check("hotspot_label_count", len(label_paths) == spec.sample_count, f"count={len(label_paths)}"))
    checks.append(_check("hotspot_label_shape_finite", not label_failures, "; ".join(label_failures[:10])))

    stage_rows: dict[str, list[dict[str, str]]] = {}
    for name, shape in (("encoded_13ch", (13, 64, 64)), ("context_17ch", (17, 64, 64)), ("context_33ch", (33, 64, 64))):
        rows = read_csv(paths.derived(name) / "combined_encoded_index.csv")
        stage_rows[name] = rows
        failures = _audit_index_tensor(rows, "x_path", data_root, shape)
        checks.append(_check(f"{name}_count_shape_finite", len(rows) == spec.sample_count and not failures, f"rows={len(rows)} failures={failures[:5]}"))

    schema_hashes = current_schema_hashes()
    runtime_lock = load_json(data_root / f"canonical/manifests/{PHASE3_STAGE}_runtime_dependency_lock.json")
    schema_ok = all(
        runtime_lock.get(key) == schema_hashes[source]
        for key, source in (
            ("raster_schema_sha256", "raster"),
            ("metadata_schema_sha256", "metadata"),
            ("graph_node_schema_sha256", "graph_node"),
            ("graph_edge_schema_sha256", "graph_edge"),
        )
    )
    channel_comparisons: dict[str, bool] = {}
    for name, manifest_name in (
        ("encoded_13ch", "context_manifest.json"),
        ("context_17ch", "feature_manifest.json"),
        ("context_33ch", "feature_manifest.json"),
    ):
        phase2_manifest_path = data_root / "derived" / name / manifest_name
        phase3_manifest_path = paths.derived(name) / manifest_name
        if phase2_manifest_path.exists() and phase3_manifest_path.exists():
            phase2_manifest = load_json(phase2_manifest_path)
            phase3_manifest = load_json(phase3_manifest_path)
            channel_comparisons[name] = phase2_manifest.get("channel_names") == phase3_manifest.get("channel_names")
        else:
            channel_comparisons[name] = False
    graph_manifest_path = paths.derived("graphs") / "graph_manifest.json"
    if graph_manifest_path.exists():
        graph_manifest = load_json(graph_manifest_path)
        channel_comparisons["graph_node"] = sha256_json(graph_manifest.get("node_feature_names", [])) == schema_hashes["graph_node"]
        channel_comparisons["graph_edge"] = sha256_json(graph_manifest.get("edge_feature_names", [])) == schema_hashes["graph_edge"]
    else:
        channel_comparisons["graph_node"] = False
        channel_comparisons["graph_edge"] = False
    schema_ok = schema_ok and all(channel_comparisons.values())
    checks.append(_check("channel_order_and_schema_hashes", schema_ok, f"schema_hashes={schema_hashes} phase2_order_match={channel_comparisons}"))

    metadata_manifest = load_json(paths.derived("metadata") / "metadata_manifest.json")
    metadata_rows = read_csv(paths.derived("metadata") / "metadata_features.csv")
    checks.append(
        _check(
            "metadata_count_and_schema",
            len(metadata_rows) == spec.sample_count and metadata_manifest.get("active_features") == list(CANONICAL_METADATA_FEATURES),
            f"rows={len(metadata_rows)} active={metadata_manifest.get('active_features')}",
        )
    )
    graph_rows = read_csv(paths.derived("graphs") / "combined_encoded_index.csv")
    graph_failures = _audit_graphs(graph_rows, data_root)
    checks.append(_check("graph_count_and_schema", len(graph_rows) == spec.sample_count and not graph_failures, f"rows={len(graph_rows)} failures={graph_failures[:5]}"))
    source_rows = read_csv(paths.derived("source_superposition") / "combined_encoded_index.csv")
    source_failures = _audit_index_tensor(source_rows, "source_superposition_base_path", data_root, GRID_SHAPE)
    checks.append(_check("source_superposition_count", len(source_rows) == spec.sample_count and not source_failures, f"rows={len(source_rows)} failures={source_failures[:5]}"))

    isolation_root = paths.canonical("source_isolation")
    lineage_problems = audit_source_isolation_lineage(data_root, selection, isolation_root=isolation_root)
    checks.append(_check("source_response_lineage", not lineage_problems, "; ".join(lineage_problems[:10])))
    train_eligible = set(selection["source_response_policy"]["train_eligible_families"])
    oracle_only = set(selection["source_response_policy"]["oracle_only_families"])
    isolation_train = read_csv(isolation_root / "train_index.csv")
    isolation_val = read_csv(isolation_root / "val_index.csv")
    isolation_test = read_csv(isolation_root / "test_index.csv")
    isolation_all = isolation_train + isolation_val + isolation_test
    isolation_frequency_ok = True
    isolation_frequency_details: dict[str, Any] = {}
    for family_uid in selected:
        rows = [row for row in isolation_all if row.get("case_id", row.get("family_uid", "")) == family_uid]
        expected_sources = len(families[family_uid]["fixed_structure"]["layout"]["chiplets"])
        source_names = {row.get("source_chiplet_name", row.get("source_name", "")) for row in rows}
        original_samples = {row.get("original_sample_uid", "") for row in rows}
        valid = len(rows) == expected_sources and len(source_names) == expected_sources and len(original_samples) == 1
        isolation_frequency_details[family_uid] = {"rows": len(rows), "expected": expected_sources, "original_samples": len(original_samples), "passed": valid}
        isolation_frequency_ok = isolation_frequency_ok and valid
    checks.append(_check("source_isolation_once_per_chiplet_family", isolation_frequency_ok, str(isolation_frequency_details)))
    checks.append(_check("train_only_source_eligibility", {row.get("case_id", "") for row in isolation_train} <= train_eligible))
    oracle_rows = isolation_val + isolation_test
    checks.append(_check("validation_test_oracle_only", {row.get("case_id", "") for row in oracle_rows} <= oracle_only and not ({row.get("case_id", "") for row in oracle_rows} & train_eligible)))

    split_counts = {
        protocol: {split: len(read_csv(index_root / protocol / f"{split}_index.csv")) for split in ("train", "val", "test")}
        for protocol in ("sample_split", "family_split")
    }
    checks.append(_check("model_ready_indices", split_counts["sample_split"] == {"train": 400, "val": 50, "test": 50} and sum(split_counts["family_split"].values()) == spec.sample_count, str(split_counts)))

    artifact_files = sorted((data_root / "canonical/manifests/artifacts").glob(f"{PHASE3_STAGE}_*.json"))
    artifact_errors: list[str] = []
    for path in artifact_files:
        try:
            artifact = load_json(path)
            validate_artifact_manifest(artifact)
            for parent in artifact.get("parents", []):
                parent_path = data_root / "canonical/manifests/artifacts" / f"{parent['artifact_id']}.json"
                if not parent_path.is_file() or sha256_file(parent_path) != parent.get("manifest_sha256"):
                    raise ValueError(f"parent hash mismatch: {parent.get('artifact_id')}")
            tree_value = str(artifact.get("content", {}).get("tree_manifest_path", ""))
            if tree_value:
                tree_path = resolve_data_path(tree_value, data_root)
                if not tree_path.is_file() or sha256_file(tree_path) != artifact["content"].get("tree_manifest_sha256"):
                    raise ValueError("tree manifest hash mismatch")
        except Exception as exc:
            artifact_errors.append(f"{path.name}: {exc}")
    completion_roots = [
        paths.canonical("families"), paths.canonical("workloads"), paths.canonical("hotspot_labels"), paths.canonical("source_isolation"),
        paths.derived("encoded_13ch"), paths.derived("context_17ch"), paths.derived("context_33ch"),
        paths.derived("metadata"), paths.derived("graphs"), paths.derived("source_response_model"), paths.derived("source_superposition"), index_root,
    ]
    completion_failures = [data_root_relative(root, data_root) for root in completion_roots if not durable_stage_complete(root)]
    for row in final_rows:
        source_dir = resolve_data_path(row["source_dir"], data_root)
        sample_root = source_dir.parent
        if not durable_stage_complete(sample_root):
            completion_failures.append(data_root_relative(sample_root, data_root))
    checks.append(_check("artifact_manifests_and_completion", len(artifact_files) >= 10 and not artifact_errors and not completion_failures, f"manifests={len(artifact_files)} errors={artifact_errors[:3]} incomplete={completion_failures}"))

    loader_report = loader_smoke(index_root / "all_index.csv", residual_checkpoint=residual_checkpoint)
    loader_all_report = loader_full_audit(index_root / "all_index.csv")
    checks.append(_check("loader_all_rows", loader_all_report.get("passed") is True and loader_all_report.get("samples") == spec.sample_count, str(loader_all_report)))
    checks.append(_check("checkpoint_forward_smoke", residual_checkpoint is None or loader_report.get("forward_smoke") == "passed", str(loader_report.get("forward_smoke"))))

    uid_sets = {
        "workloads": {str(row["sample_uid"]) for row in workloads},
        "encoded": {row.get("sample_uid", "") for row in stage_rows["encoded_13ch"]},
        "context17": {row.get("sample_uid", "") for row in stage_rows["context_17ch"]},
        "context33": {row.get("sample_uid", "") for row in stage_rows["context_33ch"]},
        "graphs": {row.get("sample_uid", "") for row in graph_rows},
        "source": {row.get("sample_uid", "") for row in source_rows},
        "final": set(final_uids),
    }
    checks.append(_check("no_silent_sample_loss", all(values == uid_sets["workloads"] for values in uid_sets.values()), str({key: len(value) for key, value in uid_sets.items()})))
    hotspot_report = build_report.get("hotspot", {})
    accounted = sum(int(hotspot_report.get(key, 0)) for key in ("completed", "skipped_valid", "reused_phase2", "failed"))
    checks.append(_check("retry_failure_accounting", accounted == spec.sample_count and int(hotspot_report.get("failed", 0)) == 0, f"accounted={accounted} report={hotspot_report}"))

    snapshot_path = data_root / f"canonical/manifests/{PHASE3_STAGE}_phase2_snapshot.json"
    expected_snapshot = load_json(snapshot_path) if snapshot_path.exists() else {"available": False}
    current_snapshot = phase2_immutability_snapshot(data_root)
    immutable = bool(expected_snapshot.get("available")) and expected_snapshot.get("content_sha256") == current_snapshot.get("content_sha256")
    checks.append(_check("phase2_artifacts_unchanged", immutable, f"expected={expected_snapshot.get('content_sha256')} actual={current_snapshot.get('content_sha256')}"))

    relocation_path = data_root / f"canonical/manifests/{PHASE3_STAGE}_relocation_validation.json"
    relocation = load_json(relocation_path) if relocation_path.exists() else {"passed": False, "status": "not_run"}
    visual_review_path = data_root / f"canonical/manifests/{PHASE3_STAGE}_visual_review.json"
    visual_review = load_json(visual_review_path) if visual_review_path.exists() else {"approved": False, "status": "not_reviewed"}
    if require_relocation:
        checks.append(_check("relocation", relocation.get("passed") is True, str(relocation)))

    temperatures = [np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64) for path in label_paths]
    total_powers = [float(row["total_package_power_W"]) for row in workloads]
    active_fractions = [float(row.get("active_chiplet_fraction", 0.0)) for row in workloads]
    dominant_shares = [float(row.get("dominant_chiplet_share", 0.0)) for row in workloads]
    temperature_means = [float(array.mean()) for array in temperatures]
    temperature_peaks = [float(array.max()) for array in temperatures]
    temperature_ranges = [float(array.max() - array.min()) for array in temperatures]
    runtimes = [float(row["hotspot_runtime_s"]) for row in final_rows if str(row.get("hotspot_runtime_s", "")).strip()]

    artifact_roots = {
        "canonical_families": paths.canonical("families"),
        "canonical_workloads": paths.canonical("workloads"),
        "canonical_hotspot_labels": paths.canonical("hotspot_labels"),
        "canonical_source_isolation": isolation_root,
        "encoded_13ch": paths.derived("encoded_13ch"),
        "context_17ch": paths.derived("context_17ch"),
        "context_33ch": paths.derived("context_33ch"),
        "metadata": paths.derived("metadata"),
        "graphs": paths.derived("graphs"),
        "source_response_model_reference": paths.derived("source_response_model"),
        "source_superposition": paths.derived("source_superposition"),
        "indices": index_root,
    }
    bytes_by_class = {name: _tree_size(root) for name, root in artifact_roots.items()}
    files_by_class = {name: _tree_file_count(root) for name, root in artifact_roots.items()}
    total_bytes = sum(bytes_by_class.values())
    total_files = sum(files_by_class.values())
    wall_clock = float(build_report.get("runtime_s", 0.0))
    workers = int(runtime_lock.get("workers", build_report.get("workers", 4)) or 4)
    peak_staging = max(
        int(hotspot_report.get("peak_staging_bytes_observed", 0)),
        _tree_size(data_root / "staging"),
    )
    projections = project_scale_metrics(
        retained_bytes=total_bytes,
        peak_staging_bytes=peak_staging,
        wall_clock_s=wall_clock,
        observed_samples=spec.sample_count,
    )
    passed = all(bool(row["passed"]) for row in checks)
    recommendation = (
        "GO"
        if passed and relocation.get("passed") is True and visual_review.get("approved") is True
        else ("GO WITH MANUAL REVIEW" if passed else "NO-GO")
    )
    report = {
        "schema_version": "benchmark_v2_scale_pilot_strict_validation/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": PHASE3_STAGE,
        "passed": passed,
        "status": "validated" if passed else "failed",
        "recommendation": recommendation,
        "selected_families": selection["selected_families"],
        "workload_design": selection.get("workload_design", {}),
        "expected_sample_count": spec.sample_count,
        "actual_sample_count": len(final_rows),
        "split_counts": split_counts,
        "checks": checks,
        "hotspot": {
            **hotspot_report,
            "runtime_s": _distribution_summary(runtimes),
            "slowest_families": _slowest_groups(final_rows, "family_uid"),
            "slowest_workload_cells": _slowest_groups(final_rows, "workload_cell"),
        },
        "source_isolation_target_count": len(isolation_train) + len(isolation_val) + len(isolation_test),
        "source_isolation_run_count": (
            len(isolation_train) + len(isolation_val) + len(isolation_test)
            - int(build_report.get("phase2_reuse", {}).get("source_isolation_artifacts", 0))
        ),
        "phase2_reuse": build_report.get("phase2_reuse", {}),
        "runtime": {
            "wall_clock_s": wall_clock,
            "by_stage_s": build_report.get("runtime_by_stage_s", {}),
            "observed_worker_count": workers,
            "projected_10000_sample_s_linear": projections["projected_wall_clock_s"],
        },
        "storage": {
            "retained_bytes": total_bytes,
            "bytes_by_artifact_class": bytes_by_class,
            "files_by_artifact_class": files_by_class,
            "inode_file_count": total_files,
            "bytes_per_sample": total_bytes / spec.sample_count,
            "bytes_per_family": total_bytes / spec.family_count,
            "staging_peak_bytes_observed": peak_staging,
            "projected_10000_retained_bytes": projections["projected_retained_bytes"],
            "projected_10000_peak_staging_bytes": projections["projected_peak_staging_bytes"],
        },
        "observed_distributions": {
            "total_power_W": _distribution_summary(total_powers),
            "active_fraction": _distribution_summary(active_fractions),
            "dominant_source_share": _distribution_summary(dominant_shares),
            "temperature_mean_K": _distribution_summary(temperature_means),
            "temperature_peak_K": _distribution_summary(temperature_peaks),
            "temperature_spatial_range_K": _distribution_summary(temperature_ranges),
        },
        "suspicious_hotspot_outputs": [
            final_rows[index].get("sample_uid", "")
            for index, (peak, spatial_range) in enumerate(zip(temperature_peaks, temperature_ranges, strict=True))
            if peak < 250.0 or peak > 1500.0 or spatial_range <= 0.0
        ],
        "portable_path_audit": portable,
        "relocation": relocation,
        "visual_review": visual_review,
        "loader_forward": loader_report,
        "loader_all_rows": loader_all_report,
        "phase2_immutability": {"expected": expected_snapshot, "actual": current_snapshot, "passed": immutable},
        "build_report": build_report,
    }
    write_json(data_root / f"canonical/manifests/{PHASE3_STAGE}_strict_validation.json", report)
    write_json(data_root / f"canonical/manifests/{PHASE3_STAGE}_validation_report.json", report)
    if not passed:
        raise ValueError(f"strict scale-pilot validation failed: {[row for row in checks if not row['passed']]}")
    return report


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "p95": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def project_scale_metrics(
    *,
    retained_bytes: float,
    peak_staging_bytes: float,
    wall_clock_s: float,
    observed_samples: int = 500,
    target_samples: int = 10000,
) -> dict[str, float | None]:
    if observed_samples <= 0 or target_samples <= 0:
        raise ValueError("projection sample counts must be positive")
    factor = target_samples / observed_samples
    return {
        "projected_retained_bytes": retained_bytes * factor,
        "projected_peak_staging_bytes": peak_staging_bytes * factor,
        "projected_wall_clock_s": wall_clock_s * factor if wall_clock_s else None,
    }


def _tree_file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _slowest_groups(rows: Sequence[dict[str, str]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = str(row.get("hotspot_runtime_s", "")).strip()
        if value:
            grouped.setdefault(str(row.get(key, "unknown")), []).append(float(value))
    ranked = sorted(grouped.items(), key=lambda item: np.mean(item[1]), reverse=True)
    return [{key: name, "count": len(values), "mean_runtime_s": float(np.mean(values)), "p95_runtime_s": float(np.quantile(values, 0.95))} for name, values in ranked[:10]]


def audit_portable_documents(data_root: Path) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    violations: list[dict[str, str]] = []
    informational: list[dict[str, str]] = []
    roots = [data_root / "canonical", data_root / "derived"]
    extensions = {".csv", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".md"}
    files_scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in extensions:
                continue
            files_scanned += 1
            relative = str(path.relative_to(data_root))
            if path.suffix == ".csv":
                for row in read_csv(path):
                    for key, value in row.items():
                        _classify_portable_string(
                            str(value or ""),
                            document=relative,
                            field=key,
                            resolving=key in PATH_COLUMNS,
                            informational=False,
                            violations=violations,
                            informational_occurrences=informational,
                        )
                continue
            if path.suffix in {".json", ".yaml", ".yml", ".jsonl"}:
                try:
                    if path.suffix == ".json":
                        payloads = [json.loads(path.read_text(encoding="utf-8"))]
                    elif path.suffix == ".jsonl":
                        payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    else:
                        payloads = [yaml.safe_load(path.read_text(encoding="utf-8"))]
                    for payload in payloads:
                        _audit_portable_value(
                            payload,
                            document=relative,
                            json_path=(),
                            inherited_informational=False,
                            artifact_manifest=isinstance(payload, dict) and payload.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
                            violations=violations,
                            informational_occurrences=informational,
                        )
                    continue
                except Exception as exc:
                    violations.append({"path": relative, "field": "<parse>", "reason": f"parse_error:{type(exc).__name__}"})
                    continue
            text = path.read_text(encoding="utf-8", errors="replace")
            _classify_portable_string(
                text,
                document=relative,
                field="<text>",
                resolving=False,
                informational=False,
                violations=violations,
                informational_occurrences=informational,
            )
    return {
        "schema_version": "benchmark_v2_portability_audit/2",
        "files_scanned": files_scanned,
        "violation_count": len(violations),
        "violations": violations,
        "informational_nonresolving_count": len(informational),
        "informational_nonresolving": informational,
    }


def _audit_portable_value(
    value: Any,
    *,
    document: str,
    json_path: tuple[str, ...],
    inherited_informational: bool,
    artifact_manifest: bool,
    violations: list[dict[str, str]],
    informational_occurrences: list[dict[str, str]],
) -> None:
    if isinstance(value, dict):
        marked = inherited_informational or value.get("path_semantics") == INFORMATIONAL_PATH_SEMANTICS
        for key, item in value.items():
            path = (*json_path, str(key))
            schema_informational = artifact_manifest and (
                path[:2] == ("producer", "command")
                or path[:2] == ("reproducibility", "reproduction_command")
            )
            _audit_portable_value(
                item,
                document=document,
                json_path=path,
                inherited_informational=marked or schema_informational,
                artifact_manifest=artifact_manifest,
                violations=violations,
                informational_occurrences=informational_occurrences,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _audit_portable_value(
                item,
                document=document,
                json_path=(*json_path, str(index)),
                inherited_informational=inherited_informational,
                artifact_manifest=artifact_manifest,
                violations=violations,
                informational_occurrences=informational_occurrences,
            )
        return
    if not isinstance(value, str):
        return
    key = json_path[-1] if json_path else "<root>"
    resolving = key in PATH_COLUMNS or key.endswith("_path") or key in {"path", "relative_path"}
    _classify_portable_string(
        value,
        document=document,
        field=".".join(json_path) or "<root>",
        resolving=resolving,
        informational=inherited_informational,
        violations=violations,
        informational_occurrences=informational_occurrences,
    )


def _classify_portable_string(
    value: str,
    *,
    document: str,
    field: str,
    resolving: bool,
    informational: bool,
    violations: list[dict[str, str]],
    informational_occurrences: list[dict[str, str]],
) -> None:
    forbidden_ids = [
        name
        for prefix, name in zip(PORTABLE_FORBIDDEN_PREFIXES, ("users", "nethome", "tmp", "export_hdd"))
        if prefix in value
    ]
    absolute_resolving = resolving and bool(value.strip()) and Path(value.strip()).is_absolute()
    if not forbidden_ids and not absolute_resolving:
        return
    record = {
        "path": document,
        "field": field,
        "reason": "absolute_resolving_path" if absolute_resolving else f"forbidden_prefix:{','.join(forbidden_ids)}",
    }
    if informational:
        informational_occurrences.append(record)
    else:
        violations.append(record)


def repair_pilot_portability(
    data_root: str | Path,
    *,
    apply: bool = False,
    stage: str = PILOT_STAGE,
) -> dict[str, Any]:
    """Repair portable metadata/index documents without touching array artifacts."""
    root = Path(data_root).expanduser().resolve()
    marker = load_json(root / ROOT_MARKER_NAME)
    if marker.get("benchmark_id") != BENCHMARK_ID or marker.get("path_semantics") != PATH_SEMANTICS:
        raise ValueError(f"invalid Benchmark v2 data root: {root}")
    before = audit_portable_documents(root)
    changes: dict[Path, str] = {}
    extensions = {".csv", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".md"}
    for scope in (root / "canonical", root / "derived"):
        if not scope.exists():
            continue
        for path in scope.rglob("*"):
            if not path.is_file() or path.suffix not in extensions or path.name.endswith("_portability_repair.json"):
                continue
            repaired = _repair_portable_document(path, root)
            original = path.read_text(encoding="utf-8", errors="replace")
            if repaired != original:
                changes[path] = repaired
    if apply:
        for path, content in changes.items():
            _atomic_write_text(path, content)
        changed_paths = set(changes)
        if changed_paths:
            _refresh_artifact_tree_manifests(root, changed_paths)
            _refresh_artifact_manifests(root)
            _refresh_completion_markers(root, changed_paths)
        after = audit_portable_documents(root)
    else:
        after = before
    report = {
        "schema_version": "benchmark_v2_portability_repair/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": stage,
        "mode": "apply" if apply else "dry_run",
        "changed_file_count": len(changes),
        "changed_files": [str(path.relative_to(root)) for path in sorted(changes)],
        "before": before,
        "after": after,
        "thermal_or_array_artifacts_modified": False,
    }
    if apply:
        write_json(root / "canonical/manifests" / f"{stage}_portability_repair.json", report)
        if after["violation_count"]:
            raise ValueError(f"portability repair left {after['violation_count']} resolving violations")
    return report


def _repair_portable_document(path: Path, data_root: Path) -> str:
    original_text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".csv":
        rows = read_csv(path)
        if not rows:
            return original_text
        original_rows = [dict(row) for row in rows]
        fieldnames = list(rows[0].keys())
        for row in rows:
            for key in fieldnames:
                value = str(row.get(key, "") or "")
                if key in PATH_COLUMNS:
                    row[key] = _repair_resolving_path(value, key=key, row=row, data_root=data_root)
                else:
                    row[key] = _sanitize_nonresolving_text(value, data_root)
        if rows == original_rows:
            return original_text
        import io

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()
    if path.suffix == ".json":
        payload = json.loads(original_text)
        repaired = _repair_portable_value(payload, data_root=data_root, informational=False)
        if repaired == payload:
            return original_text
        return json.dumps(repaired, indent=2, sort_keys=True) + "\n"
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in original_text.splitlines() if line.strip()]
        repaired_rows = [_repair_portable_value(row, data_root=data_root, informational=False) for row in rows]
        if repaired_rows == rows:
            return original_text
        return "".join(json.dumps(row, sort_keys=True) + "\n" for row in repaired_rows)
    if path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(original_text)
        repaired = _repair_portable_value(payload, data_root=data_root, informational=False)
        if repaired == payload:
            return original_text
        return yaml.safe_dump(repaired, sort_keys=False, width=120)
    return _sanitize_nonresolving_text(original_text, data_root)


def _repair_portable_value(value: Any, *, data_root: Path, informational: bool, key: str = "") -> Any:
    if isinstance(value, dict):
        marked = informational or value.get("path_semantics") == INFORMATIONAL_PATH_SEMANTICS
        repaired: dict[str, Any] = {}
        for item_key, item in value.items():
            if item_key == "source_checkpoint" and isinstance(item, str) and not marked:
                repaired[item_key] = _repair_resolving_path(item, key=item_key, row=value, data_root=data_root)
            else:
                repaired[item_key] = _repair_portable_value(
                    item,
                    data_root=data_root,
                    informational=marked or item_key in {"command", "reproduction_command"},
                    key=str(item_key),
                )
        return repaired
    if isinstance(value, list):
        return [_repair_portable_value(item, data_root=data_root, informational=informational, key=key) for item in value]
    if not isinstance(value, str):
        return value
    resolving = key in PATH_COLUMNS or key.endswith("_path") or key in {"path", "relative_path"}
    if resolving and not informational:
        return _repair_resolving_path(value, key=key, row={}, data_root=data_root)
    return _sanitize_nonresolving_text(value, data_root)


def _repair_resolving_path(value: str, *, key: str, row: Mapping[str, str], data_root: Path) -> str:
    value = value.strip()
    if not value:
        return value
    if key == "source_checkpoint":
        path = Path(value).expanduser()
        if not path.is_absolute() and (data_root / path).is_file():
            return value
        expected = str(row.get("source_checkpoint_sha256") or row.get("checkpoint_sha256") or "")
        candidates = sorted((data_root / "checkpoints").rglob("*.pt")) if (data_root / "checkpoints").exists() else []
        matches = [candidate for candidate in candidates if not expected or sha256_file(candidate) == expected]
        if len(matches) == 1:
            return data_root_relative(matches[0], data_root)
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    path = path.resolve()
    if _is_within(path, data_root):
        return data_root_relative(path, data_root)
    raise ValueError(
        f"cannot repair resolving path outside declared data root: field={key} value={value!r} root={data_root}"
    )


def _sanitize_nonresolving_text(value: str, data_root: Path) -> str:
    text = value.replace(str(data_root), "<CHIPTHERM_V2_DATA_ROOT>")
    text = text.replace(str(REPO_ROOT.resolve()), "<CHIPTHERM_REPO_ROOT>")
    for prefix, replacement in zip(
        PORTABLE_FORBIDDEN_PREFIXES,
        ("<NONRESOLVING_USERS>/", "<NONRESOLVING_NETHOME>/", "<NONRESOLVING_TMP>/", "<NONRESOLVING_EXPORT_HDD>/"),
    ):
        text = text.replace(prefix, replacement)
    return text


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.repair-{uuid.uuid4().hex}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _refresh_completion_markers(data_root: Path, changed_paths: set[Path]) -> None:
    markers = sorted(
        (path for scope in (data_root / "canonical", data_root / "derived") if scope.exists() for path in scope.rglob(".stage_complete.json")),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for marker in markers:
        stage_root = marker.parent
        if not any(_is_within(path, stage_root) and path != marker for path in changed_paths):
            continue
        completion = make_tree_manifest(stage_root, exclude_names={".stage_complete.json"})
        write_json(
            marker,
            {
                "schema_version": "benchmark_v2_stage_completion/1",
                "file_count": completion["file_count"],
                "files": [{"path": item["path"], "sha256": item["sha256"]} for item in completion["files"]],
            },
        )
        changed_paths.add(marker)


def _refresh_artifact_tree_manifests(data_root: Path, changed_paths: set[Path]) -> None:
    tree_paths = sorted(
        (path for scope in (data_root / "canonical", data_root / "derived") if scope.exists() for path in scope.rglob("tree_manifest.json")),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for tree_path in tree_paths:
        artifact_root = tree_path.parent
        if not any(_is_within(path, artifact_root) and path != tree_path for path in changed_paths):
            continue
        write_json(tree_path, make_tree_manifest(artifact_root, exclude_names={"tree_manifest.json"}))
        changed_paths.add(tree_path)


def _refresh_artifact_manifests(data_root: Path) -> None:
    manifest_root = data_root / "canonical/manifests/artifacts"
    manifest_paths = sorted(manifest_root.glob("*.json")) if manifest_root.exists() else []
    if not manifest_paths:
        return
    by_id: dict[str, Path] = {}
    for path in manifest_paths:
        payload = load_json(path)
        artifact_id = str(payload.get("artifact_id", ""))
        if artifact_id:
            by_id[artifact_id] = path
        tree_value = str(payload.get("content", {}).get("tree_manifest_path", ""))
        if tree_value:
            tree_path = resolve_data_path(tree_value, data_root)
            if tree_path.exists():
                tree = load_json(tree_path)
                payload["content"]["file_count"] = int(tree["file_count"])
                payload["content"]["total_bytes"] = int(tree["total_bytes"])
                payload["content"]["tree_manifest_sha256"] = sha256_file(tree_path)
        report_value = str(payload.get("validation", {}).get("report_path", ""))
        if report_value:
            report_path = resolve_data_path(report_value, data_root)
            if report_path.exists():
                payload["validation"]["report_sha256"] = sha256_file(report_path)
        write_json(path, payload)
    for _ in range(len(manifest_paths) + 1):
        iteration_changed = False
        for path in manifest_paths:
            payload = load_json(path)
            payload_changed = False
            for parent in payload.get("parents", []):
                parent_path = by_id.get(str(parent.get("artifact_id", "")))
                if parent_path is None:
                    continue
                digest = sha256_file(parent_path)
                if parent.get("manifest_sha256") != digest:
                    parent["manifest_sha256"] = digest
                    payload_changed = True
                    iteration_changed = True
            if payload_changed:
                write_json(path, payload)
        if not iteration_changed:
            break


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


def audit_source_isolation_lineage(
    data_root: Path,
    selection: dict[str, Any],
    *,
    isolation_root: Path | None = None,
) -> list[str]:
    root = isolation_root or (data_root / "canonical/source_isolation")
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
        from .ml.dataset import ChipThermDataset, chiptherm_collate

        dataset = ChipThermDataset(index_path, target="residual", return_metadata=True, return_graph=True)
        if not dataset:
            raise ValueError("loader smoke index is empty")
        indices = sorted({0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1})
        batch = chiptherm_collate([dataset[index] for index in indices])
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
            "representative_sample_uids": [dataset.rows[index].get("sample_uid", "") for index in indices],
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


def loader_full_audit(index_path: Path) -> dict[str, Any]:
    """Load and finite-check every row without retaining package arrays in memory."""
    try:
        import torch
        from .ml.dataset import ChipThermDataset

        dataset = ChipThermDataset(index_path, target="residual", return_metadata=True, return_graph=True)
        failures: list[str] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            uid = str(dataset.rows[index].get("sample_uid", index))
            for key in ("x", "target", "physics", "temperature", "metadata_vector"):
                value = sample.get(key)
                if value is None or not torch.isfinite(value).all():
                    failures.append(f"{uid}: {key} is missing or non-finite")
            graph = sample.get("graph", {})
            for key in ("node_features", "edge_features", "chiplet_rects", "package_size"):
                value = graph.get(key)
                if value is None or not torch.isfinite(value).all():
                    failures.append(f"{uid}: graph.{key} is missing or non-finite")
            if failures:
                break
        return {"passed": not failures, "samples": len(dataset), "failures": failures}
    except Exception as exc:
        return {"passed": False, "samples": 0, "failures": [f"{type(exc).__name__}: {exc}"]}


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
    if isinstance(output, dict):
        if "final_temperature" in output:
            final = output["final_temperature"]
        else:
            missing = [key for key in ("mean_rise", "centered_field") if key not in output]
            if missing:
                raise ValueError(
                    f"checkpoint diagnostic output cannot reconstruct final temperature; "
                    f"missing components={missing}, available keys={sorted(output)}"
                )
            centered = output["centered_field"]
            centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
            if str(config.get("mean_head_mode", "direct_k")) == "residual_resistance":
                final = batch["physics"].to(device) + output["mean_rise"][:, None, None] + centered
            else:
                final = batch["ambient_K"].to(device)[:, None, None] + output["mean_rise"][:, None, None] + centered
    else:
        final = output
    if final.numel() == 0 or not torch.isfinite(final).all():
        raise ValueError("checkpoint forward returned empty or non-finite output")
    return {"forward_smoke": "passed", "forward_output_shape": list(final.shape), "checkpoint_sha256": sha256_file(checkpoint_path)}


def relocate_pilot(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    residual_checkpoint: str | Path | None = None,
    stage: str = PILOT_STAGE,
    link_bulk_arrays: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    if destination_root.exists():
        raise FileExistsError(f"relocation destination already exists: {destination_root}")
    def copy_for_relocation(source: str, destination: str) -> str:
        source_path = Path(source)
        if link_bulk_arrays and source_path.suffix in {".npy", ".npz", ".pt"}:
            try:
                os.link(source, destination)
                return destination
            except OSError:
                pass
        return shutil.copy2(source, destination)

    shutil.copytree(
        source_root,
        destination_root,
        ignore=shutil.ignore_patterns("staging"),
        copy_function=copy_for_relocation,
    )
    source_tree = make_tree_manifest(source_root, exclude_names={"tree_manifest.json"})
    destination_tree = make_tree_manifest(destination_root, exclude_names={"tree_manifest.json"})
    source_hashes = {item["path"]: item["sha256"] for item in source_tree["files"] if not item["path"].startswith("staging/")}
    destination_hashes = {item["path"]: item["sha256"] for item in destination_tree["files"]}
    hash_match = source_hashes == destination_hashes
    strict = (
        validate_scale_pilot_root(destination_root, residual_checkpoint=residual_checkpoint)
        if stage == PHASE3_STAGE
        else validate_pilot_root(destination_root, residual_checkpoint=residual_checkpoint)
    )
    portable = audit_portable_documents(destination_root)
    report = {
        "schema_version": "benchmark_v2_relocation_validation/1",
        "benchmark_id": BENCHMARK_ID,
        "stage": stage,
        "bulk_array_copy_mode": "hardlink_with_copy_fallback" if link_bulk_arrays else "copy",
        "passed": bool(hash_match and strict["passed"] and portable["violation_count"] == 0),
        "hash_match": hash_match,
        "source_file_count": len(source_hashes),
        "destination_file_count": len(destination_hashes),
        "loaded_samples": strict.get("loader_smoke", {}).get("samples", 0),
        "portable_path_violations": portable["violation_count"],
        "loader_smoke": strict.get("loader_smoke"),
    }
    report_name = "relocation_validation_report.json" if stage == PHASE2_STAGE else f"{stage}_relocation_validation.json"
    write_json(destination_root / "canonical/manifests" / report_name, report)
    write_json(source_root / "canonical/manifests" / report_name, report)
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
