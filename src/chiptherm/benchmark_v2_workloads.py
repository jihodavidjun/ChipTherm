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
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    family_hashes: dict[str, str] = {}
    for family in families:
        family_uid = str(family["family_uid"])
        family_hashes[family_uid] = str(family.get("structural_fingerprint") or family.get("review", {}).get("structural_fingerprint", ""))
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
        "stage": "pilot_5x10",
        "base_seed": int(base_seed),
        "family_uids": sorted({str(record["family_uid"]) for record in all_records}),
        "family_structural_fingerprints": family_hashes,
        "family_count": len({record["family_uid"] for record in all_records}),
        "workloads_per_family": len(PILOT_STRATA),
        "workload_count": len(all_records),
        "strata": [item.key for item in PILOT_STRATA],
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
) -> dict[str, Any]:
    rng = random.Random(seed)
    family_uid = str(family["family_uid"])
    ordered = sorted(chiplets, key=lambda item: str(item["name"]))
    active_count = max(1, min(len(ordered), int(round(len(ordered) * stratum.active_fraction))))
    active = _select_active_chiplets(ordered, active_count, stratum.mode, rng)
    active_names = {str(item["name"]) for item in active}
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
            low, high = POWER_DENSITY_RANGES_W_PER_MM2[str(chiplet["type"])]
            fraction = stratum.load_fraction
            if stratum.mode == "skewed":
                fraction = min(0.96, max(0.04, fraction * (0.55 + 0.85 * rng.random())))
            elif stratum.mode == "single_dominant":
                fraction = 0.98 if name == dominant_name else 0.24 + 0.10 * rng.random()
            elif stratum.mode == "interacting":
                fraction = 0.96 if name in interacting else 0.42 + 0.12 * rng.random()
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
    chiplets: list[dict[str, Any]], active_count: int, mode: str, rng: random.Random
) -> list[dict[str, Any]]:
    if active_count >= len(chiplets):
        return list(chiplets)
    if mode == "type_specific":
        by_type: dict[str, list[dict[str, Any]]] = {}
        for item in chiplets:
            by_type.setdefault(str(item["type"]), []).append(item)
        types = sorted(by_type, key=lambda key: (-len(by_type[key]), key))
        selected = list(by_type[types[0]])
        remaining = [item for item in chiplets if item not in selected]
        rng.shuffle(remaining)
        return (selected + remaining)[:active_count]
    if mode == "interacting":
        pair = _closest_pair(chiplets)
        remaining = [item for item in chiplets if item not in pair]
        remaining.sort(key=lambda item: min(_center_distance(item, source) for source in pair))
        return (list(pair) + remaining)[:active_count]
    selected = list(chiplets)
    rng.shuffle(selected)
    return selected[:active_count]


def _interacting_sources(active: list[dict[str, Any]], mode: str) -> list[str]:
    if mode != "interacting" or len(active) < 2:
        return []
    pair = _closest_pair(active)
    return [str(item["name"]) for item in pair]


def _dominant_name(active: list[dict[str, Any]], mode: str) -> str | None:
    if mode != "single_dominant":
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


def _workload_seed(base_seed: int, family_uid: str, workload_index: int) -> int:
    family_number = int(family_uid.removeprefix("f"))
    return int(base_seed) + family_number * 100_003 + workload_index * 1_009
