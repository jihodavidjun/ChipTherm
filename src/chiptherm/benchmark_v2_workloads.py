from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from .validate import POWER_DENSITY_LIMITS_W_PER_MM2 as CANONICAL_POWER_DENSITY_LIMITS


SCHEMA_VERSION = "benchmark_v2_workload/1"
MANIFEST_SCHEMA_VERSION = "benchmark_v2_workload_manifest/1"
DEFAULT_SEED = 20260721
PHASE2_STAGE = "pilot_5x10"
PHASE3_STAGE = "pilot_10x50"
FULL_STAGE = "full_50x200"
IDLE_POWER_FLOOR_W = 0.01  # Numerical lower bound; canonical per-type density floors dominate.
NEAR_DUPLICATE_LINF_FRACTION = 1.0e-3
IDLE_DENSITY_EPS_W_PER_MM2 = 1.0e-6

POWER_DENSITY_RANGES_W_PER_MM2: dict[str, tuple[float, float]] = {
    "CPU": (0.8, 2.4),
    "GPU": (0.7, 2.2),
    "NPU": (0.6, 2.0),
    "HBM": (0.08, 0.25),
    "DRAM": (0.08, 0.35),
    "IO": (0.08, 0.45),
    "ANALOG": (0.05, 0.35),
    "MEMS": (0.03, 0.25),
}


@dataclass(frozen=True)
class WorkloadStratum:
    key: str
    active_fraction: float
    load_fraction: float
    mode: str


@dataclass(frozen=True)
class ScaleTopology:
    key: str
    reference_stratum: str
    active_fraction: float
    mode: str
    description: str
    active_count: int | None = None
    spatial_classification: str = "mixed"


@dataclass(frozen=True)
class ScalePowerRegime:
    key: str
    load_fraction: float | None
    description: str


PILOT_STRATA: tuple[WorkloadStratum, ...] = (
    WorkloadStratum("low_balanced", 1.00, 0.10, "balanced"),
    WorkloadStratum("low_sparse_or_type_specific", 0.30, 0.16, "type_specific"),
    WorkloadStratum("medium_balanced", 1.00, 0.50, "balanced"),
    WorkloadStratum("medium_type_specific", 0.50, 0.58, "type_specific"),
    WorkloadStratum("medium_skewed", 0.70, 0.56, "skewed"),
    WorkloadStratum("high_balanced_dense", 1.00, 0.84, "balanced"),
    WorkloadStratum("high_single_dominant", 0.55, 0.90, "single_dominant"),
    WorkloadStratum("high_interacting_multi_source", 0.65, 0.88, "interacting"),
    WorkloadStratum("sparse_active_subset_stress", 0.15, 0.94, "sparse_subset"),
    WorkloadStratum("dense_active_subset_stress", 0.90, 0.94, "dense_subset"),
)

# The reference regime is exactly the accepted Phase 2 workload for a topology.
# The four additional regimes use absolute within-type load fractions so their
# meaning is stable across chiplet types and package families.
SCALE_POWER_REGIMES: tuple[ScalePowerRegime, ...] = (
    ScalePowerRegime("phase2_reference", None, "Accepted Phase 2 workload, reused by content hash."),
    ScalePowerRegime("very_low", 0.06, "Near-idle active sources."),
    ScalePowerRegime("moderate", 0.36, "Moderate package loading."),
    ScalePowerRegime("high", 0.72, "High package loading."),
    ScalePowerRegime("stress", 0.96, "Near-upper-bound active-source loading."),
)

SCALE_TOPOLOGIES: tuple[ScaleTopology, ...] = (
    ScaleTopology("balanced", "low_balanced", 1.00, "balanced", "All-source balanced activity."),
    ScaleTopology("memory_dominant", "low_sparse_or_type_specific", 0.30, "memory_dominant", "Memory-type activity where available."),
    ScaleTopology("compute_dominant", "medium_balanced", 0.50, "compute_dominant", "CPU/GPU/NPU activity where available."),
    ScaleTopology("type_specific", "medium_type_specific", 0.50, "type_specific", "Dominant chiplet-type subset."),
    ScaleTopology("heterogeneous_cross_type", "medium_skewed", 0.70, "cross_type", "Round-robin heterogeneous activity."),
    ScaleTopology("dense_balanced", "high_balanced_dense", 1.00, "balanced", "Dense balanced activity."),
    ScaleTopology("single_source_dominant", "high_single_dominant", 0.55, "scale_single_dominant", "One dominant source with weak peers."),
    ScaleTopology("clustered_interaction", "high_interacting_multi_source", 0.65, "scale_clustered", "Closest-source cluster interaction."),
    ScaleTopology("spatially_distributed_sparse", "sparse_active_subset_stress", 0.15, "distributed", "Farthest-point sparse activity."),
    ScaleTopology("dense_cross_type", "dense_active_subset_stress", 0.90, "cross_type", "Dense heterogeneous activity."),
)

# Phase 4 completes a 10 x 20 Cartesian design while preserving the accepted
# Phase 3 cells as ordinals 1..50. The five added levels fill gaps between the
# already accepted broad loading regimes.
FULL_ADDITIONAL_POWER_REGIMES: tuple[ScalePowerRegime, ...] = (
    ScalePowerRegime("low", 0.16, "Low active-source loading."),
    ScalePowerRegime("medium_low", 0.28, "Medium-low active-source loading."),
    ScalePowerRegime("medium", 0.52, "Nominal medium active-source loading."),
    ScalePowerRegime("medium_high", 0.64, "Medium-high active-source loading."),
    ScalePowerRegime("very_high", 0.84, "Very-high but sub-stress active-source loading."),
)

FULL_ADDITIONAL_TOPOLOGIES: tuple[ScaleTopology, ...] = (
    ScaleTopology(
        "io_analog_mems_dominant", "medium_type_specific", 0.40, "peripheral_dominant",
        "IO/analog/MEMS activity where those types are available.", spatial_classification="type_specific",
    ),
    ScaleTopology(
        "two_source_near", "high_interacting_multi_source", 0.20, "two_source_near",
        "Exactly two nearest sources interact.", active_count=2, spatial_classification="clustered",
    ),
    ScaleTopology(
        "two_source_far", "high_interacting_multi_source", 0.20, "two_source_far",
        "Exactly two maximally separated sources interact.", active_count=2, spatial_classification="distributed",
    ),
    ScaleTopology(
        "three_source_cluster", "high_interacting_multi_source", 0.30, "three_source_cluster",
        "Three-source compact cluster interaction.", active_count=3, spatial_classification="clustered",
    ),
    ScaleTopology(
        "edge_corner_dominant", "high_single_dominant", 0.30, "edge_corner",
        "Sources nearest package edges/corners dominate.", spatial_classification="edge_corner",
    ),
    ScaleTopology(
        "center_dominant", "high_single_dominant", 0.30, "center",
        "Sources nearest package center dominate.", spatial_classification="center",
    ),
    ScaleTopology(
        "sparse_asymmetric", "sparse_active_subset_stress", 0.18, "sparse_asymmetric",
        "Sparse intentionally asymmetric activity.", spatial_classification="sparse_asymmetric",
    ),
    ScaleTopology(
        "medium_density_asymmetric", "medium_skewed", 0.50, "medium_asymmetric",
        "Medium-density intentionally asymmetric activity.", spatial_classification="medium_asymmetric",
    ),
    ScaleTopology(
        "dense_asymmetric", "dense_active_subset_stress", 0.82, "dense_asymmetric",
        "Dense intentionally asymmetric activity.", spatial_classification="dense_asymmetric",
    ),
    ScaleTopology(
        "symmetric_pairs", "medium_balanced", 0.35, "symmetric_pairs",
        "Approximately package-symmetric source pairs.", active_count=2, spatial_classification="symmetric",
    ),
)

FULL_POWER_REGIMES: tuple[ScalePowerRegime, ...] = SCALE_POWER_REGIMES + FULL_ADDITIONAL_POWER_REGIMES
FULL_TOPOLOGIES: tuple[ScaleTopology, ...] = SCALE_TOPOLOGIES + FULL_ADDITIONAL_TOPOLOGIES


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def load_family(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def generate_family_workloads(
    family: dict[str, Any],
    *,
    base_seed: int = DEFAULT_SEED,
    strata: Sequence[WorkloadStratum] = PILOT_STRATA,
) -> list[dict[str, Any]]:
    family_uid = str(family["family_uid"])
    chiplets = list(family["fixed_structure"]["layout"]["chiplets"])
    if not chiplets:
        raise ValueError(f"{family_uid} has no chiplets")
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    vectors: list[list[float]] = []
    for workload_index, stratum in enumerate(strata, start=1):
        seed = _workload_seed(base_seed, family_uid, workload_index)
        workload = _generate_workload(family, chiplets, stratum, workload_index, seed)
        problems = validate_workload(workload, family)
        if problems:
            raise ValueError("\n".join(problems))
        content_hash = str(workload["content_hash"])
        if content_hash in seen_hashes:
            raise ValueError(f"{family_uid}: duplicate workload hash {content_hash}")
        vector = [float(workload["chiplet_power_W"][str(item["name"])]) for item in chiplets]
        for previous_index, previous in enumerate(vectors, start=1):
            denominator = max(max(previous), max(vector), IDLE_POWER_FLOOR_W)
            distance = max(abs(left - right) for left, right in zip(previous, vector, strict=True)) / denominator
            if distance <= NEAR_DUPLICATE_LINF_FRACTION:
                raise ValueError(
                    f"{family_uid}: workload {workload_index} is near-identical to workload {previous_index} "
                    f"(normalized L-inf={distance:.3g})"
                )
        seen_hashes.add(content_hash)
        vectors.append(vector)
        records.append(workload)
    return records


def scale_workload_cells() -> list[dict[str, Any]]:
    """Return the frozen, ordered 50-cell Phase 3 workload design."""
    cells: list[dict[str, Any]] = []
    ordinal = 1
    for power in SCALE_POWER_REGIMES:
        for topology in SCALE_TOPOLOGIES:
            cells.append(
                {
                    "workload_ordinal": ordinal,
                    "workload_cell": f"{power.key}__{topology.key}",
                    "power_regime": power.key,
                    "power_load_fraction": power.load_fraction,
                    "topology_regime": topology.key,
                    "reference_stratum": topology.reference_stratum,
                    "active_fraction": topology.active_fraction,
                    "mode": topology.mode,
                    "description": f"{power.description} {topology.description}",
                }
            )
            ordinal += 1
    return cells


def full_workload_cells() -> list[dict[str, Any]]:
    """Return the frozen 200-cell Phase 4 design with Phase 3 as prefix."""
    cells = [dict(cell) for cell in scale_workload_cells()]
    ordinal = len(cells) + 1

    # Complete the accepted five power levels for the ten new topologies.
    for power in SCALE_POWER_REGIMES:
        for topology in FULL_ADDITIONAL_TOPOLOGIES:
            cells.append(_full_cell(ordinal, power, topology))
            ordinal += 1
    # Complete all twenty topologies for the five added power levels.
    for power in FULL_ADDITIONAL_POWER_REGIMES:
        for topology in FULL_TOPOLOGIES:
            cells.append(_full_cell(ordinal, power, topology))
            ordinal += 1
    if len(cells) != 200 or len({str(cell["workload_cell"]) for cell in cells}) != 200:
        raise AssertionError("Phase 4 workload-cell construction must produce 200 unique cells")
    return cells


def _full_cell(ordinal: int, power: ScalePowerRegime, topology: ScaleTopology) -> dict[str, Any]:
    return {
        "workload_ordinal": ordinal,
        "workload_cell": f"{power.key}__{topology.key}",
        "power_regime": power.key,
        "power_load_fraction": power.load_fraction,
        "topology_regime": topology.key,
        "reference_stratum": topology.reference_stratum,
        "active_fraction": topology.active_fraction,
        "active_count": topology.active_count,
        "mode": topology.mode,
        "spatial_activity_classification": topology.spatial_classification,
        "description": f"{power.description} {topology.description}",
    }


def generate_scale_family_workloads(
    family: dict[str, Any],
    *,
    base_seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate Phase 3's 50 deterministic cells without changing geometry."""
    reference_by_stratum = {
        str(record["stratum"]): record
        for record in generate_family_workloads(family, base_seed=base_seed)
    }
    chiplets = list(family["fixed_structure"]["layout"]["chiplets"])
    records: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    hashes: set[str] = set()
    ordered_chiplets = sorted(chiplets, key=lambda item: str(item["name"]))
    for cell in scale_workload_cells():
        ordinal = int(cell["workload_ordinal"])
        if cell["power_regime"] == "phase2_reference":
            record = dict(reference_by_stratum[str(cell["reference_stratum"])])
        else:
            stratum = WorkloadStratum(
                str(cell["reference_stratum"]),
                float(cell["active_fraction"]),
                float(cell["power_load_fraction"]),
                str(cell["mode"]),
            )
            seed = _workload_seed(base_seed, str(family["family_uid"]), ordinal)
            record = _generate_workload(family, chiplets, stratum, ordinal, seed)
        record["workload_cell"] = str(cell["workload_cell"])
        record["power_regime"] = str(cell["power_regime"])
        record["topology_regime"] = str(cell["topology_regime"])
        record["sub_stratum"] = str(cell["workload_cell"])
        record["active_chiplet_fraction"] = float(record["active_chiplet_count"]) / len(chiplets)
        record["phase2_reference"] = bool(cell["power_regime"] == "phase2_reference")
        problems = validate_workload(record, family)
        if problems:
            raise ValueError("\n".join(problems))
        content_hash = str(record["content_hash"])
        if content_hash in hashes:
            raise ValueError(f"{family['family_uid']}: duplicate Phase 3 workload hash {content_hash}")
        vector = [float(record["chiplet_power_W"][str(item["name"])]) for item in ordered_chiplets]
        for previous_index, previous in enumerate(vectors, start=1):
            denominator = max(max(previous), max(vector), IDLE_POWER_FLOOR_W)
            distance = max(abs(left - right) for left, right in zip(previous, vector, strict=True)) / denominator
            if distance <= NEAR_DUPLICATE_LINF_FRACTION:
                raise ValueError(
                    f"{family['family_uid']}: Phase 3 workload {ordinal} is near-identical to "
                    f"workload {previous_index} (normalized L-inf={distance:.3g})"
                )
        hashes.add(content_hash)
        vectors.append(vector)
        records.append(record)
    return records


def generate_full_family_workloads(
    family: dict[str, Any],
    *,
    base_seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate Phase 4's 200 cells, retaining Phase 3 rows byte-semantically."""
    phase3 = generate_scale_family_workloads(family, base_seed=base_seed)
    records = [dict(record) for record in phase3]
    chiplets = list(family["fixed_structure"]["layout"]["chiplets"])
    ordered_chiplets = sorted(chiplets, key=lambda item: str(item["name"]))
    reference_loads = {item.key: item.load_fraction for item in PILOT_STRATA}
    for cell in full_workload_cells()[len(phase3) :]:
        ordinal = int(cell["workload_ordinal"])
        load_fraction = cell["power_load_fraction"]
        if load_fraction is None:
            load_fraction = reference_loads[str(cell["reference_stratum"])]
        stratum = WorkloadStratum(
            str(cell["reference_stratum"]),
            float(cell["active_fraction"]),
            float(load_fraction),
            str(cell["mode"]),
        )
        seed = _workload_seed(base_seed, str(family["family_uid"]), ordinal)
        record = _generate_workload(
            family,
            chiplets,
            stratum,
            ordinal,
            seed,
            active_count_override=(int(cell["active_count"]) if cell.get("active_count") is not None else None),
        )
        record["workload_cell"] = str(cell["workload_cell"])
        record["power_regime"] = str(cell["power_regime"])
        record["topology_regime"] = str(cell["topology_regime"])
        record["sub_stratum"] = str(cell["workload_cell"])
        record["broad_stratum"] = _broad_power_stratum(str(cell["power_regime"]))
        record["spatial_activity_classification"] = str(cell["spatial_activity_classification"])
        record["active_chiplet_fraction"] = float(record["active_chiplet_count"]) / len(chiplets)
        record["phase2_reference"] = False
        record["phase3_reference"] = False
        records.append(record)

    for ordinal, (record, cell) in enumerate(zip(records, full_workload_cells(), strict=True), start=1):
        record.setdefault("broad_stratum", _broad_power_stratum(str(cell["power_regime"])))
        record.setdefault("spatial_activity_classification", str(cell.get("spatial_activity_classification", "mixed")))
        record["phase3_reference"] = ordinal <= 50
        record["selected_interaction_source_ids"] = list(record.get("interacting_hot_source_ids", []))
        problems = validate_workload(record, family)
        if problems:
            raise ValueError("\n".join(problems))
    _validate_unique_workload_vectors(records, ordered_chiplets, family_uid=str(family["family_uid"]), stage="Phase 4")
    return records


def _broad_power_stratum(power_regime: str) -> str:
    return {
        "phase2_reference": "reference",
        "very_low": "very_low",
        "low": "low",
        "medium_low": "medium_low",
        "moderate": "medium",
        "medium": "medium",
        "medium_high": "medium_high",
        "high": "high",
        "very_high": "high",
        "stress": "stress",
    }[power_regime]


def _validate_unique_workload_vectors(
    records: Sequence[dict[str, Any]],
    ordered_chiplets: Sequence[dict[str, Any]],
    *,
    family_uid: str,
    stage: str,
) -> None:
    hashes: dict[str, int] = {}
    vectors: list[list[float]] = []
    for ordinal, record in enumerate(records, start=1):
        content_hash = str(record["content_hash"])
        if content_hash in hashes:
            raise ValueError(
                f"{family_uid}: duplicate {stage} workload hash {content_hash} "
                f"at ordinals {hashes[content_hash]} and {ordinal}"
            )
        vector = [float(record["chiplet_power_W"][str(item["name"])]) for item in ordered_chiplets]
        for previous_index, previous in enumerate(vectors, start=1):
            denominator = max(max(previous), max(vector), IDLE_POWER_FLOOR_W)
            distance = max(abs(left - right) for left, right in zip(previous, vector, strict=True)) / denominator
            if distance <= NEAR_DUPLICATE_LINF_FRACTION:
                raise ValueError(
                    f"{family_uid}: {stage} workload {ordinal} is near-identical to workload "
                    f"{previous_index} (normalized L-inf={distance:.3g})"
                )
        hashes[content_hash] = ordinal
        vectors.append(vector)


def validate_workload(workload: dict[str, Any], family: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    family_uid = str(family.get("family_uid", "<missing>"))
    if workload.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"{family_uid}: invalid workload schema")
    if workload.get("family_uid") != family_uid:
        problems.append(f"{family_uid}: workload family mismatch")
    chiplets = list(family["fixed_structure"]["layout"].get("chiplets", []))
    names = [str(item["name"]) for item in chiplets]
    powers = workload.get("chiplet_power_W", {})
    densities = workload.get("chiplet_power_density_W_per_mm2", {})
    if set(powers) != set(names):
        problems.append(f"{family_uid}: power names differ from fixed layout")
    if set(densities) != set(names):
        problems.append(f"{family_uid}: density names differ from fixed layout")
    active_names = set(str(name) for name in workload.get("active_chiplet_names", []))
    if not active_names or not active_names.issubset(names):
        problems.append(f"{family_uid}: invalid active-chiplet set")
    total = 0.0
    for chiplet in chiplets:
        name = str(chiplet["name"])
        chiplet_type = str(chiplet["type"])
        width = float(chiplet["size"]["width"])
        height = float(chiplet["size"]["height"])
        area = width * height
        power = float(powers.get(name, float("nan")))
        density = float(densities.get(name, float("nan")))
        if not math.isfinite(power) or power <= 0.0:
            problems.append(f"{family_uid}: {name} power must be positive and finite")
            continue
        if not math.isfinite(density) or density <= 0.0:
            problems.append(f"{family_uid}: {name} density must be positive and finite")
            continue
        if abs(power / area - density) > max(1.0e-8, abs(density) * 1.0e-7):
            problems.append(f"{family_uid}: {name} power-density mismatch")
        low, high = POWER_DENSITY_RANGES_W_PER_MM2[chiplet_type]
        if name in active_names and not (low - 1.0e-9 <= density <= high + 1.0e-9):
            problems.append(f"{family_uid}: active {name} density {density:g} outside [{low:g}, {high:g}]")
        if name not in active_names:
            idle_density = float(CANONICAL_POWER_DENSITY_LIMITS[chiplet_type][0]) + IDLE_DENSITY_EPS_W_PER_MM2
            expected_idle_power = area * idle_density
            if abs(power - expected_idle_power) > max(1.0e-8, expected_idle_power * 1.0e-7):
                problems.append(f"{family_uid}: inactive {name} must use the canonical {idle_density:g} W/mm^2 idle floor")
        total += power
    if not math.isfinite(total) or total <= 0.0:
        problems.append(f"{family_uid}: total package power must be positive")
    if abs(total - float(workload.get("total_package_power_W", float("nan")))) > max(1.0e-7, total * 1.0e-9):
        problems.append(f"{family_uid}: total package power mismatch")
    if int(workload.get("active_chiplet_count", -1)) != len(active_names):
        problems.append(f"{family_uid}: active chiplet count mismatch")
    expected_hash = workload_content_hash(workload)
    if workload.get("content_hash") != expected_hash:
        problems.append(f"{family_uid}: workload content hash mismatch")
    return problems


def workload_content_hash(workload: dict[str, Any]) -> str:
    return sha256_json(
        {
            "family_uid": workload["family_uid"],
            "chiplet_power_W": {key: float(value) for key, value in sorted(workload["chiplet_power_W"].items())},
            "active_chiplet_names": sorted(str(name) for name in workload["active_chiplet_names"]),
        }
    )


def write_workload_tree(
    families: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    base_seed: int = DEFAULT_SEED,
    stage: str = PHASE2_STAGE,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    family_hashes: dict[str, str] = {}
    for family in families:
        family_uid = str(family["family_uid"])
        family_hashes[family_uid] = str(family.get("structural_fingerprint") or family.get("review", {}).get("structural_fingerprint", ""))
        if stage == FULL_STAGE:
            records = generate_full_family_workloads(family, base_seed=base_seed)
        elif stage == PHASE3_STAGE:
            records = generate_scale_family_workloads(family, base_seed=base_seed)
        else:
            records = generate_family_workloads(family, base_seed=base_seed)
        family_dir = output_root / family_uid
        family_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            path = family_dir / f"{record['workload_uid']}.yaml"
            path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
            all_records.append(record)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_id": "benchmark_v2_50family",
        "stage": stage,
        "base_seed": int(base_seed),
        "family_uids": sorted({str(record["family_uid"]) for record in all_records}),
        "family_structural_fingerprints": family_hashes,
        "family_count": len({record["family_uid"] for record in all_records}),
        "workloads_per_family": (
            len(full_workload_cells()) if stage == FULL_STAGE
            else len(scale_workload_cells()) if stage == PHASE3_STAGE
            else len(PILOT_STRATA)
        ),
        "workload_count": len(all_records),
        "strata": [item.key for item in PILOT_STRATA],
        "workload_cells": (
            full_workload_cells() if stage == FULL_STAGE
            else scale_workload_cells() if stage == PHASE3_STAGE
            else []
        ),
        "workload_hashes": [record["content_hash"] for record in all_records],
        "manifest_content_sha256": sha256_json(
            [{key: record[key] for key in ("family_uid", "workload_uid", "content_hash")} for record in all_records]
        ),
    }
    (output_root / "workload_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def _generate_workload(
    family: dict[str, Any],
    chiplets: list[dict[str, Any]],
    stratum: WorkloadStratum,
    workload_index: int,
    seed: int,
    active_count_override: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    family_uid = str(family["family_uid"])
    ordered = sorted(chiplets, key=lambda item: str(item["name"]))
    active_count = (
        max(1, min(len(ordered), int(active_count_override)))
        if active_count_override is not None
        else max(1, min(len(ordered), int(round(len(ordered) * stratum.active_fraction))))
    )
    package_size = family["fixed_structure"]["layout"]["package"]["size"]
    active = _select_active_chiplets(
        ordered,
        active_count,
        stratum.mode,
        rng,
        package_size=(float(package_size["width"]), float(package_size["height"])),
    )
    active_names = {str(item["name"]) for item in active}
    active_rank = {str(item["name"]): index for index, item in enumerate(active)}
    interacting = _interacting_sources(active, stratum.mode)
    dominant_name = _dominant_name(active, stratum.mode)
    powers: dict[str, float] = {}
    densities: dict[str, float] = {}
    for rank, chiplet in enumerate(ordered):
        name = str(chiplet["name"])
        area = float(chiplet["size"]["width"]) * float(chiplet["size"]["height"])
        if name not in active_names:
            density = float(CANONICAL_POWER_DENSITY_LIMITS[str(chiplet["type"])][0]) + IDLE_DENSITY_EPS_W_PER_MM2
            power = density * area
        else:
            source_rank = active_rank[name]
            low, high = POWER_DENSITY_RANGES_W_PER_MM2[str(chiplet["type"])]
            fraction = stratum.load_fraction
            if stratum.mode == "skewed":
                fraction = min(0.96, max(0.04, fraction * (0.55 + 0.85 * rng.random())))
            elif stratum.mode == "single_dominant":
                fraction = 0.98 if name == dominant_name else 0.24 + 0.10 * rng.random()
            elif stratum.mode == "interacting":
                fraction = 0.96 if name in interacting else 0.42 + 0.12 * rng.random()
            elif stratum.mode == "scale_single_dominant":
                fraction = min(0.98, fraction + 0.10) if name == dominant_name else max(0.03, 0.30 * fraction + 0.04 * rng.random())
            elif stratum.mode == "scale_clustered":
                fraction = min(0.98, fraction + 0.08) if name in interacting else max(0.03, 0.55 * fraction + 0.05 * rng.random())
            elif stratum.mode == "two_source_near":
                fraction = min(0.98, max(0.04, fraction + (0.10 if source_rank % 2 == 0 else -0.06)))
            elif stratum.mode == "two_source_far":
                fraction = min(0.98, max(0.04, fraction + (0.16 if source_rank % 2 == 0 else -0.10)))
            elif stratum.mode == "three_source_cluster":
                fraction = min(0.98, max(0.04, fraction * (1.13 - 0.11 * (source_rank % 3))))
            elif stratum.mode in {"edge_corner", "center"}:
                fraction = min(0.98, max(0.04, fraction * (1.18 if source_rank == 0 else 0.72 + 0.04 * (source_rank % 3))))
            elif stratum.mode in {"sparse_asymmetric", "medium_asymmetric", "dense_asymmetric"}:
                fraction = min(0.98, max(0.03, fraction * (0.52 + 0.14 * ((source_rank * 3 + 1) % 5))))
            elif stratum.mode == "symmetric_pairs":
                fraction = min(0.98, max(0.03, fraction * (0.91 if source_rank % 2 == 0 else 1.09)))
            elif stratum.mode == "peripheral_dominant":
                fraction = min(0.98, max(0.03, fraction * (1.12 if str(chiplet["type"]) in {"IO", "ANALOG", "MEMS"} else 0.68)))
            elif stratum.mode in {"sparse_subset", "dense_subset"}:
                fraction = min(0.98, max(0.65, fraction - 0.08 * rng.random()))
            else:
                fraction = min(0.98, max(0.02, fraction + rng.uniform(-0.025, 0.025)))
            density = low + fraction * (high - low)
            power = density * area
        powers[name] = round(power, 8)
        densities[name] = powers[name] / area
    total_power = float(sum(powers.values()))
    dominant_share = max(powers.values()) / total_power
    workload_uid = f"w{workload_index:03d}_{stratum.key}"
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "benchmark_v2_50family",
        "family_uid": family_uid,
        "workload_uid": workload_uid,
        "sample_uid": f"{family_uid}_{workload_uid}",
        "stratum": stratum.key,
        "deterministic_seed": seed,
        "units": {"power": "W", "power_density": "W/mm^2"},
        "active_workload": "nominal",
        "idle_power_policy": "per_type_canonical_minimum_power_density",
        "total_package_power_W": total_power,
        "chiplet_power_W": powers,
        "chiplet_power_density_W_per_mm2": densities,
        "active_chiplet_names": sorted(active_names),
        "active_chiplet_count": len(active_names),
        "dominant_chiplet_share": dominant_share,
        "interacting_hot_source_ids": interacting,
        "validation_status": "validated",
    }
    record["content_hash"] = workload_content_hash(record)
    return record


def _select_active_chiplets(
    chiplets: list[dict[str, Any]],
    active_count: int,
    mode: str,
    rng: random.Random,
    *,
    package_size: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    if active_count >= len(chiplets):
        return list(chiplets)
    if mode in {"type_specific", "memory_dominant", "compute_dominant", "peripheral_dominant"}:
        preferred_types = {
            "memory_dominant": {"HBM", "DRAM"},
            "compute_dominant": {"CPU", "GPU", "NPU"},
            "peripheral_dominant": {"IO", "ANALOG", "MEMS"},
        }.get(mode)
        if preferred_types is not None:
            preferred = [item for item in chiplets if str(item["type"]) in preferred_types]
            remainder = [item for item in chiplets if item not in preferred]
            rng.shuffle(preferred)
            rng.shuffle(remainder)
            if preferred:
                return (preferred + remainder)[:active_count]
        by_type: dict[str, list[dict[str, Any]]] = {}
        for item in chiplets:
            by_type.setdefault(str(item["type"]), []).append(item)
        types = sorted(by_type, key=lambda key: (-len(by_type[key]), key))
        selected = list(by_type[types[0]])
        remaining = [item for item in chiplets if item not in selected]
        rng.shuffle(remaining)
        return (selected + remaining)[:active_count]
    if mode in {"interacting", "scale_clustered", "two_source_near", "three_source_cluster"}:
        pair = _closest_pair(chiplets)
        remaining = [item for item in chiplets if item not in pair]
        remaining.sort(key=lambda item: min(_center_distance(item, source) for source in pair))
        return (list(pair) + remaining)[:active_count]
    if mode in {"distributed", "two_source_far"}:
        return _farthest_point_subset(chiplets, active_count)
    if mode == "edge_corner":
        package = package_size or _package_size_from_chiplets(chiplets)
        return sorted(chiplets, key=lambda item: (_edge_distance(item, package), str(item["name"])))[:active_count]
    if mode == "center":
        package = package_size or _package_size_from_chiplets(chiplets)
        return sorted(chiplets, key=lambda item: (_package_center_distance(item, package), str(item["name"])))[:active_count]
    if mode == "symmetric_pairs":
        package = package_size or _package_size_from_chiplets(chiplets)
        pair = _most_symmetric_pair(chiplets, package)
        remaining = [item for item in chiplets if item not in pair]
        return (list(pair) + remaining)[:active_count]
    if mode == "cross_type":
        by_type: dict[str, list[dict[str, Any]]] = {}
        for item in chiplets:
            by_type.setdefault(str(item["type"]), []).append(item)
        for values in by_type.values():
            values.sort(key=lambda item: str(item["name"]))
        selected: list[dict[str, Any]] = []
        while len(selected) < active_count:
            changed = False
            for chiplet_type in sorted(by_type):
                if by_type[chiplet_type] and len(selected) < active_count:
                    selected.append(by_type[chiplet_type].pop(0))
                    changed = True
            if not changed:
                break
        return selected
    selected = list(chiplets)
    rng.shuffle(selected)
    return selected[:active_count]


def _interacting_sources(active: list[dict[str, Any]], mode: str) -> list[str]:
    if mode in {"two_source_near", "two_source_far", "symmetric_pairs"}:
        return [str(item["name"]) for item in active[:2]]
    if mode == "three_source_cluster":
        return [str(item["name"]) for item in active[:3]]
    if mode not in {"interacting", "scale_clustered"} or len(active) < 2:
        return []
    pair = _closest_pair(active)
    return [str(item["name"]) for item in pair]


def _dominant_name(active: list[dict[str, Any]], mode: str) -> str | None:
    if mode not in {"single_dominant", "scale_single_dominant"}:
        return None
    return str(max(active, key=lambda item: float(item["size"]["width"]) * float(item["size"]["height"]))["name"])


def _closest_pair(chiplets: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(chiplets) < 2:
        raise ValueError("at least two chiplets are required")
    pairs = [
        (left, right)
        for index, left in enumerate(chiplets)
        for right in chiplets[index + 1 :]
    ]
    return min(pairs, key=lambda pair: (_center_distance(*pair), str(pair[0]["name"]), str(pair[1]["name"])))


def _center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_x = float(left["position"]["x"]) + 0.5 * float(left["size"]["width"])
    left_y = float(left["position"]["y"]) + 0.5 * float(left["size"]["height"])
    right_x = float(right["position"]["x"]) + 0.5 * float(right["size"]["width"])
    right_y = float(right["position"]["y"]) + 0.5 * float(right["size"]["height"])
    return math.hypot(left_x - right_x, left_y - right_y)


def _farthest_point_subset(chiplets: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(chiplets, key=lambda item: str(item["name"]))
    selected = [ordered[0]]
    remaining = ordered[1:]
    while remaining and len(selected) < count:
        candidate = max(
            remaining,
            key=lambda item: (
                min(_center_distance(item, chosen) for chosen in selected),
                str(item["name"]),
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def _package_size_from_chiplets(chiplets: Sequence[dict[str, Any]]) -> tuple[float, float]:
    # Fixed layouts use a common package coordinate frame. Its extents are
    # recoverable here without mutating or rasterizing geometry.
    width = max(float(item["position"]["x"]) + float(item["size"]["width"]) for item in chiplets)
    height = max(float(item["position"]["y"]) + float(item["size"]["height"]) for item in chiplets)
    return width, height


def _edge_distance(chiplet: dict[str, Any], package: tuple[float, float]) -> float:
    x = float(chiplet["position"]["x"])
    y = float(chiplet["position"]["y"])
    width = float(chiplet["size"]["width"])
    height = float(chiplet["size"]["height"])
    return min(x, y, max(0.0, package[0] - x - width), max(0.0, package[1] - y - height))


def _package_center_distance(chiplet: dict[str, Any], package: tuple[float, float]) -> float:
    x = float(chiplet["position"]["x"]) + 0.5 * float(chiplet["size"]["width"])
    y = float(chiplet["position"]["y"]) + 0.5 * float(chiplet["size"]["height"])
    return math.hypot(x - 0.5 * package[0], y - 0.5 * package[1])


def _most_symmetric_pair(
    chiplets: Sequence[dict[str, Any]], package: tuple[float, float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = [
        (left, right)
        for index, left in enumerate(chiplets)
        for right in chiplets[index + 1 :]
    ]
    center_x, center_y = 0.5 * package[0], 0.5 * package[1]

    def score(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, str, str]:
        centers = []
        for item in pair:
            centers.append((
                float(item["position"]["x"]) + 0.5 * float(item["size"]["width"]),
                float(item["position"]["y"]) + 0.5 * float(item["size"]["height"]),
            ))
        symmetry = abs(centers[0][0] + centers[1][0] - 2.0 * center_x) + abs(
            centers[0][1] + centers[1][1] - 2.0 * center_y
        )
        return symmetry, str(pair[0]["name"]), str(pair[1]["name"])

    return min(pairs, key=score)


def _workload_seed(base_seed: int, family_uid: str, workload_index: int) -> int:
    family_number = int(family_uid.removeprefix("f"))
    return int(base_seed) + family_number * 100_003 + workload_index * 1_009
