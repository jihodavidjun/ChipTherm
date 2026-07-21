from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .benchmark_extension import DEFAULT_HOTSPOT, DEFAULT_PACKAGE
from .layout import SUPPORTED_CHIPLET_TYPES, layout_from_dict
from .validate import LayoutValidationError, validate_layout


BENCHMARK_ID = "benchmark_v2_50family"
SCHEMA_VERSION = "benchmark_v2_family/1"
MANIFEST_SCHEMA_VERSION = "benchmark_v2_family_manifest/1"
DEFAULT_PROPOSAL = Path("configs/benchmark_v2_50family/design_proposal.yaml")
DEFAULT_FAMILY_DIR = Path("configs/benchmark_v2_50family/families")
DEFAULT_SPLIT_DIR = Path("configs/benchmark_v2_50family/splits")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_v2_stage1")
DEFAULT_REVIEW_PATH = Path("docs/benchmark_v2_stage1_review.md")
DEFAULT_TABLE_PATH = Path("docs/benchmark_v2_family_table.md")
DEFAULT_BASE_SEED = 20260721
GRID_ROWS = 64
GRID_COLS = 64
MIN_GAP_MM = 0.5
GEOMETRY_TOL = 1e-7
NEAR_DUPLICATE_THRESHOLD = 0.150

TYPE_ORDER = ("CPU", "GPU", "NPU", "HBM", "DRAM", "IO", "ANALOG", "MEMS")
TYPE_COLORS = {
    "CPU": "#3b82f6",
    "GPU": "#ef4444",
    "NPU": "#f97316",
    "HBM": "#8b5cf6",
    "DRAM": "#a855f7",
    "IO": "#14b8a6",
    "ANALOG": "#eab308",
    "MEMS": "#64748b",
}
TYPE_AREA_WEIGHTS = {
    "CPU": 1.00,
    "GPU": 1.22,
    "NPU": 0.96,
    "HBM": 0.66,
    "DRAM": 0.62,
    "IO": 0.50,
    "ANALOG": 0.44,
    "MEMS": 0.34,
}
TYPE_ASPECTS = {
    "CPU": 1.08,
    "GPU": 1.18,
    "NPU": 1.00,
    "HBM": 0.92,
    "DRAM": 1.12,
    "IO": 1.34,
    "ANALOG": 1.00,
    "MEMS": 0.86,
}
TYPE_SIDE_LIMITS_MM = {
    "CPU": (2.0, 18.0),
    "GPU": (3.0, 20.0),
    "NPU": (2.0, 15.0),
    "HBM": (2.0, 11.0),
    "DRAM": (2.0, 11.0),
    "IO": (1.5, 11.0),
    "ANALOG": (1.5, 8.0),
    "MEMS": (1.5, 7.5),
}


@dataclass(frozen=True)
class FamilyBlueprint:
    family_uid: str
    category: str
    split: str
    die_count: int
    width_mm: float
    height_mm: float
    whitespace: float
    composition: dict[str, int]
    placement_style: str
    purpose: str
    secondary_tags: tuple[str, ...] = ()
    matched_pair_id: str | None = None
    matched_pair_justification: str | None = None

    @property
    def rotational_group(self) -> int:
        return (int(self.family_uid[1:]) - 1) % 10 + 1

    def generation_seed(self, base_seed: int = DEFAULT_BASE_SEED) -> int:
        return base_seed + int(self.family_uid[1:]) * 1009


def _b(
    family_uid: str,
    category: str,
    split: str,
    die_count: int,
    width_mm: float,
    height_mm: float,
    whitespace: float,
    composition: dict[str, int],
    placement_style: str,
    purpose: str,
    secondary_tags: Sequence[str] = (),
    matched_pair_id: str | None = None,
    matched_pair_justification: str | None = None,
) -> FamilyBlueprint:
    return FamilyBlueprint(
        family_uid=family_uid,
        category=category,
        split=split,
        die_count=die_count,
        width_mm=width_mm,
        height_mm=height_mm,
        whitespace=whitespace,
        composition=composition,
        placement_style=placement_style,
        purpose=purpose,
        secondary_tags=tuple(secondary_tags),
        matched_pair_id=matched_pair_id,
        matched_pair_justification=matched_pair_justification,
    )


FAMILY_BLUEPRINTS: tuple[FamilyBlueprint, ...] = (
    _b("f001", "hpc", "train", 12, 40, 38, 0.48, {"CPU": 2, "GPU": 2, "HBM": 4, "DRAM": 2, "IO": 2}, "symmetric_compute_memory_cluster", "Small symmetric CPU/GPU package with paired memory banks."),
    _b("f002", "hpc", "train", 20, 48, 42, 0.52, {"CPU": 4, "GPU": 4, "HBM": 6, "DRAM": 4, "IO": 2}, "dual_compute_cluster", "Two compute clusters coupled through central memory."),
    _b("f003", "hpc", "train", 28, 54, 48, 0.56, {"CPU": 4, "GPU": 4, "HBM": 8, "DRAM": 8, "IO": 4}, "compute_center_memory_ring", "Compute center with HBM and DRAM ring."),
    _b("f004", "hpc", "train", 36, 62, 58, 0.58, {"CPU": 6, "GPU": 8, "HBM": 10, "DRAM": 8, "IO": 4}, "four_quadrant_clusters", "Large four-quadrant HPC topology."),
    _b("f005", "hpc", "train", 18, 46, 40, 0.50, {"CPU": 3, "NPU": 4, "HBM": 5, "DRAM": 4, "IO": 2}, "asymmetric_accelerator_cluster", "NPU-heavy asymmetric accelerator package."),
    _b("f006", "hpc", "train", 30, 58, 52, 0.60, {"CPU": 4, "GPU": 5, "NPU": 5, "HBM": 6, "DRAM": 6, "IO": 4}, "heterogeneous_compute_islands", "Separated CPU, GPU, and NPU compute islands."),
    _b("f007", "hpc", "val", 24, 52, 46, 0.55, {"CPU": 4, "GPU": 5, "HBM": 7, "DRAM": 5, "IO": 3}, "staggered_compute_memory", "Validation HPC family with staggered compute and memory."),
    _b("f008", "hpc", "test", 32, 66, 52, 0.66, {"CPU": 4, "GPU": 6, "NPU": 4, "HBM": 8, "DRAM": 6, "IO": 4}, "sparse_multi_cluster", "Compound-OOD sparse multi-cluster HPC package.", ("compound_ood", "sparse_hpc")),

    _b("f009", "memory_heavy", "train", 18, 44, 40, 0.54, {"CPU": 2, "HBM": 8, "DRAM": 6, "IO": 2}, "memory_ring", "Memory ring around a small compute center."),
    _b("f010", "memory_heavy", "train", 30, 58, 52, 0.60, {"CPU": 3, "GPU": 3, "HBM": 10, "DRAM": 10, "IO": 4}, "distributed_memory_banks", "Distributed high-count memory banks."),
    _b("f011", "memory_heavy", "train", 20, 50, 42, 0.57, {"NPU": 3, "HBM": 7, "DRAM": 7, "IO": 3}, "asymmetric_memory_banks", "NPU system with asymmetric HBM/DRAM banks."),
    _b("f012", "memory_heavy", "val", 28, 60, 44, 0.62, {"CPU": 3, "GPU": 3, "HBM": 8, "DRAM": 10, "IO": 4}, "edge_memory_banks", "Validation memory family with banks along package edges."),

    _b("f013", "compute_heavy", "train", 10, 36, 34, 0.42, {"CPU": 4, "GPU": 4, "IO": 2}, "compact_compute_cluster", "Compact CPU/GPU source-dense package."),
    _b("f014", "compute_heavy", "train", 18, 44, 38, 0.46, {"GPU": 6, "NPU": 6, "HBM": 4, "IO": 2}, "accelerator_array", "Regular GPU/NPU accelerator array."),
    _b("f015", "compute_heavy", "train", 24, 54, 46, 0.54, {"CPU": 4, "GPU": 8, "NPU": 6, "DRAM": 4, "IO": 2}, "dual_hot_cluster", "Two interacting compute-heavy clusters."),
    _b("f016", "compute_heavy", "test", 18, 50, 40, 0.52, {"CPU": 4, "GPU": 6, "NPU": 6, "IO": 2}, "edge_separated_hot_sources", "Compound-OOD edge-separated high-power compute sources.", ("compound_ood", "edge_compute")),

    _b("f017", "mixed_heterogeneous", "train", 18, 44, 40, 0.48, {"CPU": 3, "GPU": 3, "HBM": 3, "DRAM": 3, "IO": 6}, "functional_quadrants", "Functional CPU/GPU/memory/IO quadrants."),
    _b("f018", "mixed_heterogeneous", "train", 24, 52, 46, 0.55, {"CPU": 3, "NPU": 4, "HBM": 3, "DRAM": 3, "IO": 3, "ANALOG": 4, "MEMS": 4}, "thermal_zones", "High- and low-power functional thermal zones."),
    _b("f019", "mixed_heterogeneous", "train", 32, 60, 54, 0.60, {"CPU": 4, "GPU": 5, "NPU": 4, "HBM": 6, "DRAM": 6, "IO": 7}, "multi_cluster", "Large six-type multi-cluster package."),
    _b("f020", "mixed_heterogeneous", "train", 22, 50, 44, 0.52, {"CPU": 3, "GPU": 4, "HBM": 3, "DRAM": 3, "IO": 4, "ANALOG": 5}, "asymmetric_functional", "Asymmetric mixed digital/analog package."),
    _b("f021", "mixed_heterogeneous", "train", 38, 68, 60, 0.64, {"CPU": 5, "GPU": 5, "NPU": 5, "HBM": 6, "DRAM": 5, "IO": 5, "ANALOG": 4, "MEMS": 3}, "distributed_functional", "Broad heterogeneous package with distributed functions."),
    _b("f022", "mixed_heterogeneous", "train", 16, 40, 36, 0.45, {"CPU": 3, "HBM": 3, "DRAM": 3, "IO": 4, "ANALOG": 3}, "compact_asymmetric", "Small compact heterogeneous package."),
    _b("f023", "mixed_heterogeneous", "val", 28, 58, 50, 0.59, {"CPU": 4, "GPU": 4, "NPU": 3, "HBM": 4, "DRAM": 4, "IO": 4, "ANALOG": 3, "MEMS": 2}, "separated_functional_clusters", "Validation family with separated heterogeneous clusters."),

    _b("f024", "analog_mems", "train", 12, 40, 36, 0.52, {"CPU": 2, "IO": 3, "ANALOG": 3, "MEMS": 4}, "protected_low_power_zone", "Protected analog/MEMS region beside CPU and IO."),
    _b("f025", "analog_mems", "train", 16, 46, 40, 0.56, {"GPU": 3, "HBM": 3, "DRAM": 2, "ANALOG": 4, "MEMS": 4}, "hot_cold_adjacency", "Hot GPU region adjacent to sensitive modules."),
    _b("f026", "analog_mems", "train", 22, 54, 48, 0.63, {"CPU": 3, "NPU": 3, "HBM": 3, "DRAM": 3, "IO": 4, "ANALOG": 3, "MEMS": 3}, "distributed_sensitive_modules", "Sensitive modules distributed around compute."),
    _b("f027", "analog_mems", "test", 16, 54, 46, 0.62, {"CPU": 2, "GPU": 3, "IO": 3, "ANALOG": 4, "MEMS": 4}, "edge_sensitive_hot_center", "Compound-OOD hot center with edge-sensitive devices.", ("compound_ood", "analog_edge")),

    _b("f028", "sparse_low_die", "train", 6, 48, 42, 0.70, {"CPU": 1, "GPU": 2, "HBM": 2, "IO": 1}, "widely_separated", "Six widely separated compute and memory sources."),
    _b("f029", "sparse_low_die", "train", 8, 62, 48, 0.75, {"CPU": 2, "GPU": 2, "HBM": 2, "DRAM": 1, "IO": 1}, "corner_and_center", "Sparse corner and center topology."),
    _b("f030", "sparse_low_die", "val", 7, 56, 46, 0.72, {"GPU": 2, "NPU": 1, "HBM": 2, "CPU": 1, "IO": 1}, "asymmetric_far_sources", "Validation sparse family with asymmetric far sources."),

    _b("f031", "dense_high_die", "train", 48, 56, 52, 0.36, {"CPU": 6, "GPU": 7, "NPU": 5, "HBM": 8, "DRAM": 8, "IO": 8, "ANALOG": 3, "MEMS": 3}, "dense_regular_channels", "Dense many-die layout with regular channels."),
    _b("f032", "dense_high_die", "train", 64, 68, 62, 0.40, {"CPU": 8, "GPU": 8, "NPU": 8, "HBM": 10, "DRAM": 10, "IO": 10, "ANALOG": 5, "MEMS": 5}, "dense_multi_size", "Maximum-count dense heterogeneous layout."),
    _b("f033", "dense_high_die", "test", 56, 62, 56, 0.36, {"CPU": 7, "GPU": 8, "NPU": 6, "HBM": 9, "DRAM": 8, "IO": 9, "ANALOG": 5, "MEMS": 4}, "dense_edge_channels", "Compound-OOD dense heterogeneous package with edge channels.", ("compound_ood", "dense_edge")),

    _b("f034", "compact_clustered", "train", 16, 38, 34, 0.34, {"CPU": 3, "GPU": 4, "HBM": 5, "DRAM": 2, "IO": 2}, "single_tight_cluster", "Single compact compute-memory cluster."),
    _b("f035", "compact_clustered", "train", 24, 46, 40, 0.38, {"CPU": 4, "GPU": 5, "NPU": 4, "HBM": 5, "DRAM": 4, "IO": 2}, "dual_tight_cluster", "Two compact interacting clusters."),
    _b("f036", "compact_clustered", "train", 32, 52, 46, 0.42, {"CPU": 4, "GPU": 5, "HBM": 7, "DRAM": 6, "IO": 6, "NPU": 2, "ANALOG": 2}, "hierarchical_clusters", "Hierarchical compact cluster topology."),

    _b("f037", "distributed", "train", 14, 54, 48, 0.62, {"CPU": 2, "GPU": 3, "HBM": 5, "DRAM": 2, "IO": 2}, "uniform_distributed", "Uniformly distributed compute-memory sources."),
    _b("f038", "distributed", "train", 22, 64, 56, 0.67, {"CPU": 3, "GPU": 4, "HBM": 5, "DRAM": 4, "IO": 4, "ANALOG": 2}, "perimeter_and_center", "Perimeter sources coupled to a center cluster."),
    _b("f039", "distributed", "train", 28, 70, 64, 0.70, {"CPU": 4, "GPU": 4, "NPU": 4, "HBM": 5, "DRAM": 5, "IO": 6}, "separated_islands", "Large package with separated functional islands."),

    _b("f040", "edge_constrained", "train", 14, 46, 40, 0.52, {"CPU": 3, "GPU": 4, "HBM": 4, "DRAM": 1, "IO": 2}, "one_hot_edge_band", "Hot source band constrained near one edge."),
    _b("f041", "edge_constrained", "val", 20, 58, 48, 0.58, {"CPU": 3, "GPU": 5, "NPU": 3, "HBM": 4, "DRAM": 2, "IO": 3}, "two_edge_hot_sources", "Validation topology with hot sources on two edges."),

    _b("f042", "package_scale_aspect", "train", 14, 34, 32, 0.45, {"CPU": 3, "GPU": 3, "HBM": 4, "DRAM": 2, "IO": 2}, "scale_reference_compact", "Small physical-scale reference package."),
    _b("f043", "package_scale_aspect", "train", 24, 72, 38, 0.58, {"CPU": 4, "GPU": 4, "HBM": 6, "DRAM": 6, "IO": 4}, "scale_reference_elongated", "Large elongated reference covering the test aspect range."),
    _b("f044", "package_scale_aspect", "test", 20, 68, 36, 0.54, {"CPU": 3, "GPU": 4, "NPU": 3, "HBM": 4, "DRAM": 3, "IO": 3}, "long_axis_clusters", "Compound-OOD high-aspect package with long-axis clusters.", ("compound_ood", "high_aspect_spacing")),

    _b("f045", "chiplet_size_aspect", "train", 14, 46, 42, 0.56, {"CPU": 3, "GPU": 3, "HBM": 4, "DRAM": 2, "IO": 2}, "elongated_sources_mixed_orientation", "Mixed source aspect ratios and orientations."),
    _b("f046", "chiplet_size_aspect", "train", 24, 58, 52, 0.55, {"CPU": 4, "GPU": 5, "NPU": 3, "HBM": 4, "DRAM": 4, "IO": 4}, "multi_scale_source_sizes", "Large chiplet-size variance within one package."),

    _b("f047", "whitespace", "train", 18, 50, 44, 0.35, {"CPU": 3, "GPU": 4, "HBM": 4, "DRAM": 3, "IO": 4}, "matched_low_whitespace", "Low-whitespace member of a controlled pair.", ("matched_pair",), "whitespace_pair", "Intentional same composition/package pair isolating whitespace."),
    _b("f048", "whitespace", "train", 18, 50, 44, 0.72, {"CPU": 3, "GPU": 4, "HBM": 4, "DRAM": 3, "IO": 4}, "matched_high_whitespace", "High-whitespace member of a controlled pair.", ("matched_pair",), "whitespace_pair", "Intentional same composition/package pair isolating whitespace."),

    _b("f049", "spacing", "train", 18, 58, 48, 0.55, {"CPU": 3, "GPU": 4, "HBM": 4, "DRAM": 3, "IO": 4}, "matched_near_spacing", "Near-spacing member of a controlled pair.", ("matched_pair",), "spacing_pair", "Intentional matched pair isolating source spacing."),
    _b("f050", "spacing", "train", 18, 58, 48, 0.55, {"CPU": 3, "GPU": 4, "HBM": 4, "DRAM": 3, "IO": 4}, "matched_far_spacing", "Far-spacing member of a controlled pair.", ("matched_pair",), "spacing_pair", "Intentional matched pair isolating source spacing."),
)


def load_design_proposal(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def validate_blueprints(proposal: dict[str, Any]) -> None:
    errors: list[str] = []
    if len(FAMILY_BLUEPRINTS) != 50:
        errors.append(f"expected 50 blueprints, got {len(FAMILY_BLUEPRINTS)}")
    proposal_rows = {str(row["id"]): row for row in proposal.get("families", [])}
    seen: set[str] = set()
    taxonomy: dict[str, int] = {}
    splits: dict[str, int] = {}
    for blueprint in FAMILY_BLUEPRINTS:
        if blueprint.family_uid in seen:
            errors.append(f"duplicate family UID {blueprint.family_uid}")
        seen.add(blueprint.family_uid)
        if blueprint.family_uid not in proposal_rows:
            errors.append(f"{blueprint.family_uid} is absent from design proposal")
        else:
            proposed = proposal_rows[blueprint.family_uid]
            if str(proposed.get("category")) != blueprint.category:
                errors.append(f"{blueprint.family_uid}: category differs from proposal")
            if str(proposed.get("split")) != blueprint.split:
                errors.append(f"{blueprint.family_uid}: split differs from proposal")
        if sum(blueprint.composition.values()) != blueprint.die_count:
            errors.append(f"{blueprint.family_uid}: composition does not sum to die count")
        unsupported = set(blueprint.composition) - SUPPORTED_CHIPLET_TYPES
        if unsupported:
            errors.append(f"{blueprint.family_uid}: unsupported types {sorted(unsupported)}")
        taxonomy[blueprint.category] = taxonomy.get(blueprint.category, 0) + 1
        splits[blueprint.split] = splits.get(blueprint.split, 0) + 1
    expected_taxonomy = {str(key): int(value) for key, value in proposal.get("taxonomy_counts", {}).items() if int(value) > 0}
    if taxonomy != expected_taxonomy:
        errors.append(f"taxonomy counts differ: expected {expected_taxonomy}, got {taxonomy}")
    if splits != {"train": 40, "val": 5, "test": 5}:
        errors.append(f"split counts differ: {splits}")
    groups = [blueprint.rotational_group for blueprint in FAMILY_BLUEPRINTS]
    if {group: groups.count(group) for group in sorted(set(groups))} != {group: 5 for group in range(1, 11)}:
        errors.append("rotational groups are not ten groups of five")
    if errors:
        raise ValueError("\n".join(errors))


def instantiate_family(blueprint: FamilyBlueprint, *, base_seed: int = DEFAULT_BASE_SEED) -> dict[str, Any]:
    seed = blueprint.generation_seed(base_seed)
    layout = _generate_layout(blueprint, seed)
    descriptors = compute_layout_descriptors(layout)
    fingerprint_components = structural_fingerprint_components(layout, descriptors)
    fingerprint = sha256_json(fingerprint_components)
    package = deepcopy(DEFAULT_PACKAGE)
    hotspot = deepcopy(DEFAULT_HOTSPOT)
    spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "status": "stage1_fixed_structure",
        "family_uid": blueprint.family_uid,
        "primary_category": blueprint.category,
        "primary_split": blueprint.split,
        "rotational_fold_group": blueprint.rotational_group,
        "secondary_tags": list(blueprint.secondary_tags),
        "placement_style": blueprint.placement_style,
        "purpose": blueprint.purpose,
        "generation": {
            "base_seed": base_seed,
            "candidate_seed": seed,
            "accepted_candidate_attempt": 0,
            "method": "deterministic_structured_cell_construction",
            "geometry_depends_on_workload": False,
        },
        "fixed_structure": {
            "grid": {"rows": GRID_ROWS, "cols": GRID_COLS, "map_mode": "avg"},
            "layout": layout,
            "composition": {key: int(blueprint.composition.get(key, 0)) for key in TYPE_ORDER if blueprint.composition.get(key, 0)},
            "thermal_stack": package,
            "hotspot": hotspot,
            "material_and_cooling_variant": "benchmark_v2_0_fixed_default",
        },
        "constraints": {
            "minimum_legal_gap_mm": MIN_GAP_MM,
            "target_whitespace_fraction": blueprint.whitespace,
            "allowed_chiplet_types": list(TYPE_ORDER),
            "geometry_unit": "mm",
        },
        "descriptors": descriptors,
        "structural_fingerprint": fingerprint,
        "structural_fingerprint_components": fingerprint_components,
        "future_workload_contract": {
            "workloads_per_family": 200,
            "fixed_fields": [
                "package dimensions",
                "grid convention",
                "chiplet IDs/types/rectangles",
                "thermal stack",
                "cooling configuration",
                "ambient convention",
                "HotSpot settings",
            ],
            "variable_fields": ["chiplet activity", "chiplet power_W", "chiplet power_density_W_per_mm2"],
        },
    }
    if blueprint.matched_pair_id:
        spec["intentional_matched_pair"] = {
            "pair_id": blueprint.matched_pair_id,
            "justification": blueprint.matched_pair_justification,
        }
    problems = validate_family_spec(spec)
    if problems:
        raise ValueError(f"{blueprint.family_uid} failed generation validation:\n" + "\n".join(problems))
    return spec


def instantiate_all_families(proposal: dict[str, Any], *, base_seed: int = DEFAULT_BASE_SEED) -> list[dict[str, Any]]:
    validate_blueprints(proposal)
    return [instantiate_family(blueprint, base_seed=base_seed) for blueprint in FAMILY_BLUEPRINTS]


def _generate_layout(blueprint: FamilyBlueprint, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    rows, cols = _grid_shape(blueprint.die_count, blueprint.width_mm, blueprint.height_mm)
    mode = _placement_mode(blueprint.placement_style)
    domain_scale = 0.76 if blueprint.placement_style == "matched_near_spacing" else 1.0
    if mode == "compact" and blueprint.whitespace >= 0.48:
        domain_scale = min(domain_scale, 0.88)

    types = [chiplet_type for chiplet_type in TYPE_ORDER for _ in range(blueprint.composition.get(chiplet_type, 0))]
    if len(types) != blueprint.die_count:
        raise ValueError(f"{blueprint.family_uid}: type list does not match die count")

    for scale_step in range(13):
        scale = min(1.0, domain_scale + scale_step * 0.02)
        slots = _select_slots(
            rows,
            cols,
            blueprint.die_count,
            blueprint.width_mm,
            blueprint.height_mm,
            scale,
            mode,
        )
        assignments = _assign_types_to_slots(types, slots, blueprint.placement_style, rng)
        try:
            chiplets = _size_and_place_chiplets(assignments, blueprint, seed)
            break
        except ValueError:
            if scale >= 1.0:
                raise
    else:
        raise ValueError(f"{blueprint.family_uid}: no feasible structured placement")

    return {
        "schema_version": 1,
        "units": {"length": "mm"},
        "package": {
            "name": blueprint.family_uid,
            "substrate": "silicon_interposer",
            "size": {"width": float(blueprint.width_mm), "height": float(blueprint.height_mm)},
        },
        "chiplets": chiplets,
    }


def _grid_shape(count: int, width: float, height: float) -> tuple[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, count + 1):
        cols = math.ceil(count / rows)
        cell_aspect = (width / cols) / (height / rows)
        unused = rows * cols - count
        cost = abs(math.log(max(cell_aspect, 1e-12))) + 0.075 * unused
        candidates.append((cost, rows, cols))
    _, rows, cols = min(candidates)
    return rows, cols


def _placement_mode(style: str) -> str:
    if style in {"matched_near_spacing", "single_tight_cluster", "dual_tight_cluster", "hierarchical_clusters", "compact_compute_cluster", "compact_asymmetric"}:
        return "compact"
    if any(token in style for token in ("edge", "ring", "perimeter", "corner")):
        return "edge"
    if any(token in style for token in ("distributed", "separated", "islands", "far", "spread", "sparse")):
        return "distributed"
    return "structured"


def _select_slots(
    rows: int,
    cols: int,
    count: int,
    width: float,
    height: float,
    domain_scale: float,
    mode: str,
) -> list[dict[str, float | int]]:
    domain_width = width * domain_scale
    domain_height = height * domain_scale
    x0 = (width - domain_width) / 2.0
    y0 = (height - domain_height) / 2.0
    cell_width = domain_width / cols
    cell_height = domain_height / rows
    slots: list[dict[str, float | int]] = []
    for row in range(rows):
        for col in range(cols):
            cx = x0 + (col + 0.5) * cell_width
            cy = y0 + (row + 0.5) * cell_height
            center_distance = math.hypot((cx - width / 2.0) / width, (cy - height / 2.0) / height)
            edge_distance = min(cx, width - cx, cy, height - cy)
            slots.append(
                {
                    "row": row,
                    "col": col,
                    "cx": cx,
                    "cy": cy,
                    "cell_width": cell_width,
                    "cell_height": cell_height,
                    "center_distance": center_distance,
                    "edge_distance": edge_distance,
                }
            )
    if len(slots) == count:
        selected = slots
    elif mode == "compact":
        selected = sorted(slots, key=lambda item: (item["center_distance"], item["row"], item["col"]))[:count]
    elif mode in {"edge", "distributed"}:
        selected = sorted(slots, key=lambda item: (item["edge_distance"], -item["center_distance"], item["row"], item["col"]))[:count]
    else:
        indexes = np.linspace(0, len(slots) - 1, count).round().astype(int).tolist()
        selected = [slots[index] for index in dict.fromkeys(indexes)]
        if len(selected) < count:
            selected.extend(slot for slot in slots if slot not in selected and len(selected) < count)
    return sorted(selected, key=lambda item: (item["row"], item["col"]))


def _assign_types_to_slots(
    types: list[str],
    slots: list[dict[str, float | int]],
    style: str,
    rng: random.Random,
) -> list[tuple[str, dict[str, float | int]]]:
    compute = {"CPU", "GPU", "NPU"}
    memory = {"HBM", "DRAM"}
    sensitive = {"ANALOG", "MEMS"}
    unassigned = list(slots)
    assignments: list[tuple[str, dict[str, float | int]]] = []
    ordered_types = sorted(types, key=lambda item: (TYPE_ORDER.index(item), item))

    def take(key: Any, reverse: bool = False) -> dict[str, float | int]:
        slot = sorted(unassigned, key=key, reverse=reverse)[0]
        unassigned.remove(slot)
        return slot

    for index, chiplet_type in enumerate(ordered_types):
        if style in {"compute_center_memory_ring", "memory_ring", "distributed_memory_banks", "edge_memory_banks", "asymmetric_memory_banks"}:
            if chiplet_type in memory or chiplet_type == "IO":
                slot = take(lambda item: (item["edge_distance"], -item["center_distance"]))
            else:
                slot = take(lambda item: (item["center_distance"], item["row"], item["col"]))
        elif style in {"protected_low_power_zone", "distributed_sensitive_modules", "edge_sensitive_hot_center"}:
            if chiplet_type in sensitive:
                slot = take(lambda item: (item["cx"], item["cy"]))
            elif chiplet_type in compute:
                slot = take(lambda item: (-float(item["cx"]), item["center_distance"]))
            else:
                slot = take(lambda item: (item["center_distance"], item["row"], item["col"]))
        elif "edge" in style and chiplet_type in compute:
            slot = take(lambda item: (item["edge_distance"], -item["center_distance"]))
        elif style in {"four_quadrant_clusters", "functional_quadrants", "thermal_zones", "multi_cluster", "hierarchical_clusters"}:
            slot = take(lambda item: (item["row"], item["col"]))
        elif style in {"staggered_compute_memory", "dual_compute_cluster", "dual_hot_cluster", "dual_tight_cluster", "long_axis_clusters"}:
            direction = -1.0 if (TYPE_ORDER.index(chiplet_type) % 2) else 1.0
            slot = take(lambda item: (direction * float(item["cx"]), item["cy"]))
        else:
            # RNG is used only to resolve deterministic symmetry ties.
            jitter = {id(item): rng.random() * 1e-9 for item in unassigned}
            slot = take(lambda item: (item["row"], item["col"], jitter[id(item)]))
        assignments.append((chiplet_type, slot))
    return assignments


def _size_and_place_chiplets(
    assignments: list[tuple[str, dict[str, float | int]]],
    blueprint: FamilyBlueprint,
    seed: int,
) -> list[dict[str, Any]]:
    target_area = blueprint.width_mm * blueprint.height_mm * (1.0 - blueprint.whitespace)
    aspects: list[float] = []
    capacities: list[float] = []
    weights: list[float] = []
    for index, (chiplet_type, slot) in enumerate(assignments):
        aspect = TYPE_ASPECTS[chiplet_type]
        if blueprint.placement_style == "elongated_sources_mixed_orientation":
            aspect *= 1.65
        elif blueprint.placement_style == "multi_scale_source_sizes":
            aspect *= 1.0 + 0.25 * ((index % 3) - 1)
        if (index + seed) % 2:
            aspect = 1.0 / aspect
        aspect = max(0.42, min(2.4, aspect))
        available_width = float(slot["cell_width"]) - MIN_GAP_MM
        available_height = float(slot["cell_height"]) - MIN_GAP_MM
        if available_width <= 0.0 or available_height <= 0.0:
            raise ValueError("cell too small for minimum spacing")
        max_area = min(available_width * available_width / aspect, available_height * available_height * aspect)
        aspects.append(aspect)
        capacities.append(max_area * 0.995)
        weight = TYPE_AREA_WEIGHTS[chiplet_type]
        if blueprint.placement_style == "multi_scale_source_sizes":
            weight *= (0.65, 1.0, 1.45)[index % 3]
        weights.append(weight)
    areas = _allocate_weighted_areas(target_area, weights, capacities)

    counters = {chiplet_type: 0 for chiplet_type in TYPE_ORDER}
    chiplets: list[dict[str, Any]] = []
    for index, ((chiplet_type, slot), area, aspect) in enumerate(zip(assignments, areas, aspects, strict=True)):
        chiplet_width = math.sqrt(area * aspect)
        chiplet_height = math.sqrt(area / aspect)
        x = float(slot["cx"]) - chiplet_width / 2.0
        y = float(slot["cy"]) - chiplet_height / 2.0
        name = f"{chiplet_type}{counters[chiplet_type]:02d}"
        counters[chiplet_type] += 1
        chiplets.append(
            {
                "name": name,
                "type": chiplet_type,
                "position": {"x": round(x, 9), "y": round(y, 9)},
                "size": {"width": round(chiplet_width, 9), "height": round(chiplet_height, 9)},
            }
        )
    actual_area = sum(float(item["size"]["width"]) * float(item["size"]["height"]) for item in chiplets)
    if abs(actual_area - target_area) > 1e-5:
        raise ValueError(f"rounded chiplet area differs from target by {actual_area - target_area:.6g} mm^2")
    return chiplets


def _allocate_weighted_areas(total: float, weights: Sequence[float], capacities: Sequence[float]) -> list[float]:
    if total > sum(capacities) + GEOMETRY_TOL:
        raise ValueError(f"target occupied area {total:.6g} exceeds structured placement capacity {sum(capacities):.6g}")
    areas = [0.0] * len(weights)
    active = set(range(len(weights)))
    remaining = total
    while active and remaining > GEOMETRY_TOL:
        weight_sum = sum(weights[index] for index in active)
        capped: list[int] = []
        for index in active:
            proposed = remaining * weights[index] / weight_sum
            room = capacities[index] - areas[index]
            if proposed >= room - GEOMETRY_TOL:
                areas[index] += max(room, 0.0)
                remaining -= max(room, 0.0)
                capped.append(index)
        if not capped:
            for index in active:
                areas[index] += remaining * weights[index] / weight_sum
            remaining = 0.0
            break
        active.difference_update(capped)
    if remaining > 1e-5:
        raise ValueError(f"could not allocate {remaining:.6g} mm^2 of occupied area")
    return areas


def compute_layout_descriptors(layout: dict[str, Any]) -> dict[str, Any]:
    width = float(layout["package"]["size"]["width"])
    height = float(layout["package"]["size"]["height"])
    package_area = width * height
    diagonal = math.hypot(width, height)
    chiplets = list(layout["chiplets"])
    rects = [_rect_tuple(item) for item in chiplets]
    areas = np.asarray([rect[2] * rect[3] for rect in rects], dtype=np.float64)
    aspects = np.asarray([max(rect[2], rect[3]) / min(rect[2], rect[3]) for rect in rects], dtype=np.float64)
    edge_clearances = np.asarray(
        [min(rect[0], width - rect[0] - rect[2], rect[1], height - rect[1] - rect[3]) for rect in rects],
        dtype=np.float64,
    )
    center_distances: list[float] = []
    edge_gaps: list[float] = []
    for index, first in enumerate(rects):
        for second in rects[index + 1:]:
            cx1, cy1 = first[0] + first[2] / 2.0, first[1] + first[3] / 2.0
            cx2, cy2 = second[0] + second[2] / 2.0, second[1] + second[3] / 2.0
            center_distances.append(math.hypot(cx1 - cx2, cy1 - cy2))
            edge_gaps.append(rectangle_edge_gap(first, second))
    type_counts = {chiplet_type: sum(item["type"] == chiplet_type for item in chiplets) for chiplet_type in TYPE_ORDER}
    occupied_area = float(np.sum(areas))
    return {
        "package_width_mm": width,
        "package_height_mm": height,
        "package_area_mm2": package_area,
        "package_aspect_ratio": max(width, height) / min(width, height),
        "chiplet_count": len(chiplets),
        "type_counts": type_counts,
        "occupied_area_mm2": occupied_area,
        "occupied_fraction": occupied_area / package_area,
        "whitespace_fraction": 1.0 - occupied_area / package_area,
        "minimum_chiplet_gap_mm": min(edge_gaps) if edge_gaps else 0.0,
        "mean_pairwise_chiplet_gap_mm": _mean(edge_gaps),
        "minimum_package_edge_clearance_mm": float(np.min(edge_clearances)),
        "mean_package_edge_clearance_mm": float(np.mean(edge_clearances)),
        "chiplet_area_min_mm2": float(np.min(areas)),
        "chiplet_area_mean_mm2": float(np.mean(areas)),
        "chiplet_area_std_mm2": float(np.std(areas)),
        "chiplet_area_max_mm2": float(np.max(areas)),
        "chiplet_aspect_ratio_min": float(np.min(aspects)),
        "chiplet_aspect_ratio_mean": float(np.mean(aspects)),
        "chiplet_aspect_ratio_std": float(np.std(aspects)),
        "chiplet_aspect_ratio_max": float(np.max(aspects)),
        "minimum_pairwise_center_distance_mm": min(center_distances) if center_distances else 0.0,
        "mean_pairwise_center_distance_mm": _mean(center_distances),
        "pairwise_center_distance_histogram": _normalized_hist(center_distances, 0.0, diagonal, 8),
        "pairwise_edge_gap_histogram": _normalized_hist(edge_gaps, 0.0, diagonal, 8),
        "package_edge_distance_histogram": _normalized_hist(edge_clearances.tolist(), 0.0, min(width, height) / 2.0, 8),
        "chiplet_area_quantiles_normalized": _quantiles((areas / package_area).tolist()),
        "chiplet_aspect_ratio_quantiles": _quantiles(aspects.tolist()),
        "spatial_occupancy_descriptor_8x8": _occupancy_descriptor(layout, 8),
        "type_aware_spatial_descriptor_4x4": _type_spatial_descriptor(layout, 4),
    }


def structural_fingerprint_components(layout: dict[str, Any], descriptors: dict[str, Any]) -> dict[str, Any]:
    chiplets = []
    width = descriptors["package_width_mm"]
    height = descriptors["package_height_mm"]
    for item in sorted(layout["chiplets"], key=lambda value: value["name"]):
        chiplets.append(
            {
                "name": item["name"],
                "type": item["type"],
                "x_normalized": round(float(item["position"]["x"]) / width, 10),
                "y_normalized": round(float(item["position"]["y"]) / height, 10),
                "width_normalized": round(float(item["size"]["width"]) / width, 10),
                "height_normalized": round(float(item["size"]["height"]) / height, 10),
            }
        )
    return {
        "physical_package_mm": [width, height],
        "package_aspect_ratio": descriptors["package_aspect_ratio"],
        "chiplet_count": descriptors["chiplet_count"],
        "type_histogram": descriptors["type_counts"],
        "whitespace_fraction": descriptors["whitespace_fraction"],
        "chiplet_area_quantiles_normalized": descriptors["chiplet_area_quantiles_normalized"],
        "chiplet_aspect_ratio_quantiles": descriptors["chiplet_aspect_ratio_quantiles"],
        "pairwise_center_distance_histogram": descriptors["pairwise_center_distance_histogram"],
        "pairwise_edge_gap_histogram": descriptors["pairwise_edge_gap_histogram"],
        "package_edge_distance_histogram": descriptors["package_edge_distance_histogram"],
        "spatial_occupancy_descriptor_8x8": descriptors["spatial_occupancy_descriptor_8x8"],
        "type_aware_spatial_descriptor_4x4": descriptors["type_aware_spatial_descriptor_4x4"],
        "normalized_chiplets": chiplets,
    }


def validate_family_spec(spec: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    uid = str(spec.get("family_uid", "<missing>"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"{uid}: invalid schema_version")
    if spec.get("benchmark_id") != BENCHMARK_ID:
        problems.append(f"{uid}: invalid benchmark_id")
    try:
        layout = spec["fixed_structure"]["layout"]
        validate_layout(layout_from_dict(layout))
    except (KeyError, TypeError, LayoutValidationError, ValueError) as exc:
        problems.append(f"{uid}: layout validation failed: {exc}")
        return problems
    chiplets = list(layout["chiplets"])
    composition = {key: int(value) for key, value in spec["fixed_structure"].get("composition", {}).items()}
    actual_composition = {chiplet_type: sum(item["type"] == chiplet_type for item in chiplets) for chiplet_type in TYPE_ORDER}
    actual_composition = {key: value for key, value in actual_composition.items() if value}
    if composition != actual_composition:
        problems.append(f"{uid}: composition mismatch: expected {composition}, got {actual_composition}")
    if len(chiplets) != sum(composition.values()):
        problems.append(f"{uid}: die count differs from composition")
    if len({item["name"] for item in chiplets}) != len(chiplets):
        problems.append(f"{uid}: duplicate chiplet IDs")
    unsupported = {item["type"] for item in chiplets} - SUPPORTED_CHIPLET_TYPES
    if unsupported:
        problems.append(f"{uid}: unsupported chiplet types {sorted(unsupported)}")
    for item in chiplets:
        low, high = TYPE_SIDE_LIMITS_MM[item["type"]]
        for dimension in ("width", "height"):
            value = float(item["size"][dimension])
            if value < low - GEOMETRY_TOL or value > high + GEOMETRY_TOL:
                problems.append(
                    f"{uid}: {item['name']} {dimension} {value:.6g} mm outside "
                    f"reviewed {item['type']} side range [{low:g}, {high:g}] mm"
                )
    grid = spec["fixed_structure"].get("grid", {})
    hotspot_grid = spec["fixed_structure"].get("hotspot", {}).get("grid", {})
    if (grid.get("rows"), grid.get("cols")) != (GRID_ROWS, GRID_COLS):
        problems.append(f"{uid}: family grid must be 64 x 64")
    if (hotspot_grid.get("rows"), hotspot_grid.get("cols")) != (GRID_ROWS, GRID_COLS):
        problems.append(f"{uid}: HotSpot grid must be 64 x 64")
    if spec["fixed_structure"].get("thermal_stack") != DEFAULT_PACKAGE:
        problems.append(f"{uid}: material/cooling stack differs from v2.0 default")
    if spec["fixed_structure"].get("hotspot") != DEFAULT_HOTSPOT:
        problems.append(f"{uid}: HotSpot settings differ from v2.0 default")
    if spec.get("generation", {}).get("geometry_depends_on_workload") is not False:
        problems.append(f"{uid}: geometry must be independent of workloads")
    recomputed = compute_layout_descriptors(layout)
    stored = spec.get("descriptors", {})
    for name in (
        "package_width_mm",
        "package_height_mm",
        "occupied_area_mm2",
        "whitespace_fraction",
        "minimum_chiplet_gap_mm",
        "minimum_package_edge_clearance_mm",
    ):
        if name not in stored or abs(float(stored[name]) - float(recomputed[name])) > 1e-7:
            problems.append(f"{uid}: descriptor {name} does not match geometry")
    target_whitespace = float(spec.get("constraints", {}).get("target_whitespace_fraction", float("nan")))
    if not math.isfinite(target_whitespace) or abs(recomputed["whitespace_fraction"] - target_whitespace) > 1e-7:
        problems.append(
            f"{uid}: whitespace {recomputed['whitespace_fraction']:.9f} differs from target {target_whitespace:.9f}"
        )
    if recomputed["minimum_chiplet_gap_mm"] < MIN_GAP_MM - GEOMETRY_TOL:
        problems.append(f"{uid}: minimum gap below {MIN_GAP_MM:g} mm")
    expected_fingerprint = sha256_json(structural_fingerprint_components(layout, recomputed))
    if spec.get("structural_fingerprint") != expected_fingerprint:
        problems.append(f"{uid}: structural fingerprint mismatch")
    if _contains_absolute_path(spec):
        problems.append(f"{uid}: contains an absolute path")
    return problems


def validate_family_collection(families: Sequence[dict[str, Any]], proposal: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    if len(families) != 50:
        problems.append(f"expected 50 families, found {len(families)}")
    uids = [str(spec.get("family_uid", "")) for spec in families]
    if len(set(uids)) != len(uids):
        problems.append("family UIDs are not unique")
    fingerprints = [str(spec.get("structural_fingerprint", "")) for spec in families]
    if len(set(fingerprints)) != len(fingerprints):
        problems.append("structural fingerprints are not unique")
    for spec in families:
        problems.extend(validate_family_spec(spec))
    split_counts = _counts(spec["primary_split"] for spec in families)
    if split_counts != {"train": 40, "val": 5, "test": 5}:
        problems.append(f"split counts are {split_counts}, expected 40/5/5")
    group_counts = _counts(int(spec["rotational_fold_group"]) for spec in families)
    if group_counts != {group: 5 for group in range(1, 11)}:
        problems.append(f"rotational groups are not ten groups of five: {group_counts}")
    expected_taxonomy = {str(key): int(value) for key, value in proposal["taxonomy_counts"].items() if int(value)}
    taxonomy_counts = _counts(spec["primary_category"] for spec in families)
    if taxonomy_counts != expected_taxonomy:
        problems.append(f"taxonomy counts differ: expected {expected_taxonomy}, got {taxonomy_counts}")
    stack_hashes = {sha256_json(spec["fixed_structure"]["thermal_stack"]) for spec in families}
    hotspot_hashes = {sha256_json(spec["fixed_structure"]["hotspot"]) for spec in families}
    if len(stack_hashes) != 1 or len(hotspot_hashes) != 1:
        problems.append("material/cooling or HotSpot settings vary across families")
    return {
        "passed": not problems,
        "problems": problems,
        "warnings": warnings,
        "family_count": len(families),
        "split_counts": split_counts,
        "rotational_group_counts": group_counts,
        "taxonomy_counts": taxonomy_counts,
        "material_stack_hash": next(iter(stack_hashes), ""),
        "hotspot_settings_hash": next(iter(hotspot_hashes), ""),
    }


def geometry_for_workload(spec: dict[str, Any], workload_uid: str, workload_seed: int) -> dict[str, Any]:
    del workload_uid, workload_seed
    return deepcopy(spec["fixed_structure"]["layout"])


def rectangle_edge_gap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    dx = max(first[0] - (second[0] + second[2]), second[0] - (first[0] + first[2]), 0.0)
    dy = max(first[1] - (second[1] + second[3]), second[1] - (first[1] + first[3]), 0.0)
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _rect_tuple(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item["position"]["x"]),
        float(item["position"]["y"]),
        float(item["size"]["width"]),
        float(item["size"]["height"]),
    )


def _occupancy_descriptor(layout: dict[str, Any], bins: int) -> list[float]:
    width = float(layout["package"]["size"]["width"])
    height = float(layout["package"]["size"]["height"])
    cell_width = width / bins
    cell_height = height / bins
    out = np.zeros((bins, bins), dtype=np.float64)
    for item in layout["chiplets"]:
        x, y, chiplet_width, chiplet_height = _rect_tuple(item)
        for row in range(bins):
            y_overlap = max(0.0, min(y + chiplet_height, (row + 1) * cell_height) - max(y, row * cell_height))
            if y_overlap <= 0.0:
                continue
            for col in range(bins):
                x_overlap = max(0.0, min(x + chiplet_width, (col + 1) * cell_width) - max(x, col * cell_width))
                if x_overlap > 0.0:
                    out[row, col] += x_overlap * y_overlap / (cell_width * cell_height)
    return [round(float(value), 10) for value in out.ravel()]


def _type_spatial_descriptor(layout: dict[str, Any], bins: int) -> dict[str, list[float]]:
    width = float(layout["package"]["size"]["width"])
    height = float(layout["package"]["size"]["height"])
    output: dict[str, list[float]] = {}
    for chiplet_type in TYPE_ORDER:
        grid = np.zeros((bins, bins), dtype=np.float64)
        selected = [item for item in layout["chiplets"] if item["type"] == chiplet_type]
        for item in selected:
            x, y, chiplet_width, chiplet_height = _rect_tuple(item)
            col = min(bins - 1, int(((x + chiplet_width / 2.0) / width) * bins))
            row = min(bins - 1, int(((y + chiplet_height / 2.0) / height) * bins))
            grid[row, col] += 1.0
        if selected:
            grid /= len(selected)
        output[chiplet_type] = [round(float(value), 10) for value in grid.ravel()]
    return output


def _normalized_hist(values: Sequence[float], low: float, high: float, bins: int) -> list[float]:
    if not values:
        return [0.0] * bins
    clipped = np.clip(np.asarray(values, dtype=np.float64), low, high)
    counts, _ = np.histogram(clipped, bins=bins, range=(low, high))
    normalized = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    return [round(float(value), 10) for value in normalized]


def _quantiles(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0] * 5
    return [round(float(value), 10) for value in np.quantile(np.asarray(values, dtype=np.float64), [0, 0.25, 0.5, 0.75, 1])]


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _counts(values: Iterable[Any]) -> dict[Any, int]:
    output: dict[Any, int] = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return output


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return value.startswith("/") or ":\\" in value
    return False


SCALAR_COVERAGE_FIELDS = (
    "chiplet_count",
    "package_width_mm",
    "package_height_mm",
    "package_area_mm2",
    "package_aspect_ratio",
    "whitespace_fraction",
    "occupied_area_mm2",
    "minimum_chiplet_gap_mm",
    "mean_pairwise_center_distance_mm",
    "chiplet_area_mean_mm2",
    "chiplet_area_std_mm2",
    "CPU_count",
    "GPU_count",
    "NPU_count",
    "memory_count",
    "IO_count",
    "ANALOG_count",
    "MEMS_count",
)


def family_descriptor_row(spec: dict[str, Any]) -> dict[str, Any]:
    descriptors = spec["descriptors"]
    type_counts = descriptors["type_counts"]
    return {
        "family_uid": spec["family_uid"],
        "primary_category": spec["primary_category"],
        "split": spec["primary_split"],
        "rotational_fold_group": spec["rotational_fold_group"],
        "placement_style": spec["placement_style"],
        "structural_fingerprint": spec["structural_fingerprint"],
        "chiplet_count": descriptors["chiplet_count"],
        "package_width_mm": descriptors["package_width_mm"],
        "package_height_mm": descriptors["package_height_mm"],
        "package_area_mm2": descriptors["package_area_mm2"],
        "package_aspect_ratio": descriptors["package_aspect_ratio"],
        "occupied_area_mm2": descriptors["occupied_area_mm2"],
        "occupied_fraction": descriptors["occupied_fraction"],
        "whitespace_fraction": descriptors["whitespace_fraction"],
        "minimum_chiplet_gap_mm": descriptors["minimum_chiplet_gap_mm"],
        "mean_pairwise_chiplet_gap_mm": descriptors["mean_pairwise_chiplet_gap_mm"],
        "minimum_package_edge_clearance_mm": descriptors["minimum_package_edge_clearance_mm"],
        "mean_package_edge_clearance_mm": descriptors["mean_package_edge_clearance_mm"],
        "minimum_pairwise_center_distance_mm": descriptors["minimum_pairwise_center_distance_mm"],
        "mean_pairwise_center_distance_mm": descriptors["mean_pairwise_center_distance_mm"],
        "chiplet_area_min_mm2": descriptors["chiplet_area_min_mm2"],
        "chiplet_area_mean_mm2": descriptors["chiplet_area_mean_mm2"],
        "chiplet_area_std_mm2": descriptors["chiplet_area_std_mm2"],
        "chiplet_area_max_mm2": descriptors["chiplet_area_max_mm2"],
        "chiplet_aspect_ratio_min": descriptors["chiplet_aspect_ratio_min"],
        "chiplet_aspect_ratio_mean": descriptors["chiplet_aspect_ratio_mean"],
        "chiplet_aspect_ratio_std": descriptors["chiplet_aspect_ratio_std"],
        "chiplet_aspect_ratio_max": descriptors["chiplet_aspect_ratio_max"],
        "CPU_count": type_counts["CPU"],
        "GPU_count": type_counts["GPU"],
        "NPU_count": type_counts["NPU"],
        "HBM_count": type_counts["HBM"],
        "DRAM_count": type_counts["DRAM"],
        "memory_count": type_counts["HBM"] + type_counts["DRAM"],
        "IO_count": type_counts["IO"],
        "ANALOG_count": type_counts["ANALOG"],
        "MEMS_count": type_counts["MEMS"],
        "secondary_tags": ";".join(spec.get("secondary_tags", [])),
    }


def structural_feature_vector(spec: dict[str, Any]) -> np.ndarray:
    descriptors = spec["descriptors"]
    type_counts = descriptors["type_counts"]
    count = max(float(descriptors["chiplet_count"]), 1.0)
    scalar = np.asarray(
        [
            descriptors["package_width_mm"] / 78.0,
            descriptors["package_height_mm"] / 78.0,
            descriptors["package_aspect_ratio"] / 2.2,
            descriptors["chiplet_count"] / 72.0,
            descriptors["whitespace_fraction"],
            descriptors["minimum_chiplet_gap_mm"] / 12.0,
            descriptors["mean_pairwise_center_distance_mm"] / math.hypot(78.0, 78.0),
            descriptors["minimum_package_edge_clearance_mm"] / 20.0,
        ],
        dtype=np.float64,
    )
    type_hist = np.asarray([type_counts[item] / count for item in TYPE_ORDER], dtype=np.float64)
    area = np.asarray(descriptors["chiplet_area_quantiles_normalized"], dtype=np.float64) * 20.0
    aspect = np.asarray(descriptors["chiplet_aspect_ratio_quantiles"], dtype=np.float64) / 3.0
    center_hist = np.asarray(descriptors["pairwise_center_distance_histogram"], dtype=np.float64)
    gap_hist = np.asarray(descriptors["pairwise_edge_gap_histogram"], dtype=np.float64)
    edge_hist = np.asarray(descriptors["package_edge_distance_histogram"], dtype=np.float64)
    occupancy = np.asarray(descriptors["spatial_occupancy_descriptor_8x8"], dtype=np.float64)
    type_spatial = np.concatenate(
        [np.asarray(descriptors["type_aware_spatial_descriptor_4x4"][item], dtype=np.float64) for item in TYPE_ORDER]
    )
    blocks = (
        (scalar, 1.8),
        (type_hist, 1.8),
        (area, 1.0),
        (aspect, 0.8),
        (center_hist, 1.0),
        (gap_hist, 0.8),
        (edge_hist, 0.8),
        (occupancy, 1.2),
        (type_spatial, 1.2),
    )
    return np.concatenate([block * weight / math.sqrt(len(block)) for block, weight in blocks])


def pairwise_distance_rows(families: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vectors = {spec["family_uid"]: structural_feature_vector(spec) for spec in families}
    by_uid = {spec["family_uid"]: spec for spec in families}
    rows: list[dict[str, Any]] = []
    for index, first in enumerate(families):
        for second in families[index + 1:]:
            uid_a = first["family_uid"]
            uid_b = second["family_uid"]
            distance = float(np.linalg.norm(vectors[uid_a] - vectors[uid_b]))
            pair_a = first.get("intentional_matched_pair", {}).get("pair_id")
            pair_b = second.get("intentional_matched_pair", {}).get("pair_id")
            intentional = bool(pair_a and pair_a == pair_b)
            justification = first.get("intentional_matched_pair", {}).get("justification", "") if intentional else ""
            rows.append(
                {
                    "family_a": uid_a,
                    "family_b": uid_b,
                    "distance": distance,
                    "same_category": first["primary_category"] == second["primary_category"],
                    "same_split": first["primary_split"] == second["primary_split"],
                    "cross_split": first["primary_split"] != second["primary_split"],
                    "intentional_matched_pair": intentional,
                    "justification": justification,
                    "suspicious": distance < NEAR_DUPLICATE_THRESHOLD or intentional,
                }
            )
    rows.sort(key=lambda row: float(row["distance"]))
    nearest: list[dict[str, Any]] = []
    for spec in families:
        uid = spec["family_uid"]
        candidates = [row for row in rows if row["family_a"] == uid or row["family_b"] == uid]
        best = min(candidates, key=lambda row: float(row["distance"]))
        neighbor = best["family_b"] if best["family_a"] == uid else best["family_a"]
        nearest.append(
            {
                "family_uid": uid,
                "split": spec["primary_split"],
                "nearest_family_uid": neighbor,
                "nearest_split": by_uid[neighbor]["primary_split"],
                "distance": best["distance"],
                "cross_split": best["cross_split"],
                "intentional_matched_pair": best["intentional_matched_pair"],
                "suspicious": best["suspicious"],
                "justification": best["justification"],
            }
        )
    nearest.sort(key=lambda row: row["family_uid"])
    return rows, nearest


def split_coverage_rows(descriptor_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    train = [row for row in descriptor_rows if row["split"] == "train"]
    val = [row for row in descriptor_rows if row["split"] == "val"]
    test = [row for row in descriptor_rows if row["split"] == "test"]
    for name in SCALAR_COVERAGE_FIELDS:
        train_values = np.asarray([float(row[name]) for row in train], dtype=np.float64)
        val_values = np.asarray([float(row[name]) for row in val], dtype=np.float64)
        test_values = np.asarray([float(row[name]) for row in test], dtype=np.float64)
        train_min = float(np.min(train_values))
        train_max = float(np.max(train_values))
        val_overlap = float(np.mean((val_values >= train_min - GEOMETRY_TOL) & (val_values <= train_max + GEOMETRY_TOL)))
        test_overlap = float(np.mean((test_values >= train_min - GEOMETRY_TOL) & (test_values <= train_max + GEOMETRY_TOL)))
        stump_accuracy, threshold, direction = best_balanced_threshold(train_values, test_values)
        output.append(
            {
                "descriptor": name,
                "train_min": train_min,
                "train_max": train_max,
                "train_mean": float(np.mean(train_values)),
                "val_min": float(np.min(val_values)),
                "val_max": float(np.max(val_values)),
                "val_fraction_inside_train_range": val_overlap,
                "test_min": float(np.min(test_values)),
                "test_max": float(np.max(test_values)),
                "test_fraction_inside_train_range": test_overlap,
                "best_test_vs_train_threshold_balanced_accuracy": stump_accuracy,
                "best_threshold": threshold,
                "test_direction": direction,
                "weak_test_overlap": test_overlap < 1.0,
                "trivially_separates_test": stump_accuracy >= 1.0 - 1e-12,
            }
        )
    return output


def best_balanced_threshold(train_values: np.ndarray, test_values: np.ndarray) -> tuple[float, float, str]:
    combined = np.unique(np.concatenate([train_values, test_values]))
    thresholds = [-float("inf")]
    thresholds.extend(float((first + second) / 2.0) for first, second in zip(combined[:-1], combined[1:], strict=True))
    thresholds.append(float("inf"))
    best = (0.5, 0.0, "test_above")
    for threshold in thresholds:
        for direction in ("test_above", "test_below"):
            if direction == "test_above":
                test_recall = float(np.mean(test_values > threshold))
                train_recall = float(np.mean(train_values <= threshold))
            else:
                test_recall = float(np.mean(test_values <= threshold))
                train_recall = float(np.mean(train_values > threshold))
            score = 0.5 * (test_recall + train_recall)
            if score > best[0] + 1e-12:
                best = (score, threshold, direction)
    return best


def taxonomy_summary_rows(families: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = sorted({spec["primary_category"] for spec in families})
    return [
        {
            "primary_category": category,
            "total": sum(spec["primary_category"] == category for spec in families),
            "train": sum(spec["primary_category"] == category and spec["primary_split"] == "train" for spec in families),
            "val": sum(spec["primary_category"] == category and spec["primary_split"] == "val" for spec in families),
            "test": sum(spec["primary_category"] == category and spec["primary_split"] == "test" for spec in families),
        }
        for category in categories
    ]


def family_table_rows(families: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in families:
        d = spec["descriptors"]
        counts = d["type_counts"]
        rows.append(
            {
                "Family": spec["family_uid"],
                "Primary category": spec["primary_category"],
                "Split": spec["primary_split"],
                "Dies": d["chiplet_count"],
                "CPU": counts["CPU"],
                "GPU": counts["GPU"],
                "NPU": counts["NPU"],
                "Memory": counts["HBM"] + counts["DRAM"],
                "IO": counts["IO"],
                "Analog": counts["ANALOG"],
                "MEMS": counts["MEMS"],
                "Package width (mm)": d["package_width_mm"],
                "Package height (mm)": d["package_height_mm"],
                "Whitespace (%)": 100.0 * d["whitespace_fraction"],
                "Minimum gap (mm)": d["minimum_chiplet_gap_mm"],
                "Placement style": spec["placement_style"],
                "Secondary tags": ";".join(spec.get("secondary_tags", [])),
            }
        )
    return rows


def write_stage1_artifacts(
    families: Sequence[dict[str, Any]],
    *,
    proposal_path: Path,
    family_dir: Path,
    split_dir: Path,
    output_dir: Path,
    review_path: Path,
    table_markdown_path: Path,
    base_seed: int,
) -> dict[str, Any]:
    family_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "layout_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    expected_uids = {spec["family_uid"] for spec in families}
    for old_path in family_dir.glob("f*.yaml"):
        if old_path.stem not in expected_uids:
            raise ValueError(f"unexpected preexisting family file would be orphaned: {old_path}")

    family_entries: list[dict[str, Any]] = []
    for spec in families:
        uid = spec["family_uid"]
        family_path = family_dir / f"{uid}.yaml"
        family_path.write_text(yaml.safe_dump(spec, sort_keys=False, width=120), encoding="utf-8")
        compact_layout_path = preview_dir / f"{uid}.json"
        compact_layout_path.write_text(
            json.dumps(spec["fixed_structure"]["layout"], sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        render_family_preview(spec, preview_dir / f"{uid}.png")
        family_entries.append(
            {
                "family_uid": uid,
                "family_file": f"families/{uid}.yaml",
                "family_file_sha256": file_sha256(family_path),
                "structural_fingerprint": spec["structural_fingerprint"],
                "primary_category": spec["primary_category"],
                "primary_split": spec["primary_split"],
                "rotational_fold_group": spec["rotational_fold_group"],
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "status": "stage1_fixed_structure_review",
        "proposal_path": proposal_path.as_posix(),
        "proposal_sha256": file_sha256(proposal_path),
        "generator_source": "src/chiptherm/benchmark_v2.py",
        "generator_source_sha256": file_sha256(Path(__file__)),
        "base_seed": base_seed,
        "family_count": len(families),
        "split_counts": _counts(spec["primary_split"] for spec in families),
        "rotational_group_counts": _counts(int(spec["rotational_fold_group"]) for spec in families),
        "family_entries": family_entries,
        "material_stack_sha256": sha256_json(families[0]["fixed_structure"]["thermal_stack"]),
        "hotspot_settings_sha256": sha256_json(families[0]["fixed_structure"]["hotspot"]),
        "generation_scope": "family_structures_only_no_workloads_no_hotspot_no_labels_no_model_tensors",
    }
    (family_dir.parent / "family_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    write_split_manifests(families, split_dir)
    descriptor_rows = [family_descriptor_row(spec) for spec in families]
    pairwise_rows, nearest_rows = pairwise_distance_rows(families)
    coverage_rows = split_coverage_rows(descriptor_rows)
    taxonomy_rows = taxonomy_summary_rows(families)
    table_rows = family_table_rows(families)
    write_csv(output_dir / "family_descriptors.csv", descriptor_rows)
    write_csv(output_dir / "pairwise_family_distances.csv", pairwise_rows)
    write_csv(output_dir / "nearest_family_pairs.csv", nearest_rows)
    write_csv(output_dir / "split_coverage.csv", coverage_rows)
    write_csv(output_dir / "taxonomy_summary.csv", taxonomy_rows)
    write_csv(output_dir / "family_table.csv", table_rows)
    write_latex_table(output_dir / "family_table.tex", table_rows)
    write_markdown_table(table_markdown_path, table_rows)
    render_contact_sheets(families, preview_dir)

    proposal = load_design_proposal(proposal_path)
    validation = validate_family_collection(families, proposal)
    suspicious = [row for row in pairwise_rows if bool(row["suspicious"])]
    unexplained_cross_split = [
        row for row in suspicious if bool(row["cross_split"]) and not bool(row["intentional_matched_pair"])
    ]
    trivial = [row for row in coverage_rows if bool(row["trivially_separates_test"])]
    weak = [row for row in coverage_rows if bool(row["weak_test_overlap"])]
    missing_previews = [spec["family_uid"] for spec in families if not (preview_dir / f"{spec['family_uid']}.png").exists()]
    if unexplained_cross_split:
        validation["problems"].append(f"{len(unexplained_cross_split)} unexplained suspicious cross-split pairs")
    if trivial:
        validation["problems"].append(
            "test partition is perfectly separable from train by: " + ", ".join(row["descriptor"] for row in trivial)
        )
    if missing_previews:
        validation["problems"].append(f"missing previews for {missing_previews}")
    validation["passed"] = not validation["problems"]
    intentional_suspicious = [row for row in suspicious if bool(row["intentional_matched_pair"])]
    if validation["passed"] and intentional_suspicious:
        recommendation = "GO WITH MANUAL REVIEW"
    elif validation["passed"]:
        recommendation = "GO"
    else:
        recommendation = "NO-GO"
    validation.update(
        {
            "benchmark_id": BENCHMARK_ID,
            "stage": 1,
            "hotspot_runs": 0,
            "workloads_generated": 0,
            "near_duplicate_threshold_proposal": NEAR_DUPLICATE_THRESHOLD,
            "suspicious_pair_count": len(suspicious),
            "intentional_suspicious_pair_count": len(intentional_suspicious),
            "unexplained_cross_split_suspicious_pair_count": len(unexplained_cross_split),
            "weak_coverage_axes": [row["descriptor"] for row in weak],
            "trivially_separating_axes": [row["descriptor"] for row in trivial],
            "manual_review_families": sorted(
                {
                    str(row["family_a"])
                    for row in suspicious
                }
                | {str(row["family_b"]) for row in suspicious}
            ),
            "recommendation": recommendation,
            "provenance": {
                "proposal_sha256": manifest["proposal_sha256"],
                "generator_source_sha256": manifest["generator_source_sha256"],
                "family_manifest_sha256": file_sha256(family_dir.parent / "family_manifest.yaml"),
            },
        }
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_review_report(review_path, validation, suspicious, coverage_rows, taxonomy_rows, nearest_rows)
    return validation


def load_family_specs(family_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(family_dir.glob("f[0-9][0-9][0-9].yaml"))
    families: list[dict[str, Any]] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        families.append(data)
    return families


def write_split_manifests(families: Sequence[dict[str, Any]], split_dir: Path) -> None:
    primary = {
        "schema_version": "benchmark_v2_preliminary_family_split/1",
        "benchmark_id": BENCHMARK_ID,
        "status": "preliminary_stage1_no_model_results_used",
        "train": [spec["family_uid"] for spec in families if spec["primary_split"] == "train"],
        "val": [spec["family_uid"] for spec in families if spec["primary_split"] == "val"],
        "test": [spec["family_uid"] for spec in families if spec["primary_split"] == "test"],
    }
    rotational = {
        "schema_version": "benchmark_v2_rotational_family_groups/1",
        "benchmark_id": BENCHMARK_ID,
        "status": "preliminary_stage1",
        "groups": {
            f"group_{group:02d}": [
                spec["family_uid"] for spec in families if int(spec["rotational_fold_group"]) == group
            ]
            for group in range(1, 11)
        },
        "fold_policy": "fold k test=group k, val=group ((k mod 10)+1), train=remaining eight groups",
    }
    sample = {
        "schema_version": "benchmark_v2_preliminary_workload_split/1",
        "benchmark_id": BENCHMARK_ID,
        "status": "proposal_only_workloads_not_generated",
        "per_family": {"train_workload_count": 160, "val_workload_count": 20, "test_workload_count": 20},
        "assignment_policy": "deterministic stratification within each future workload category",
    }
    for name, payload in (
        ("primary_family_split.yaml", primary),
        ("rotational_family_groups.yaml", rotational),
        ("sample_split_proposal.yaml", sample),
    ):
        (split_dir / name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    headers = list(rows[0]) if rows else []
    lines = [
        "# Benchmark v2 Family Table",
        "",
        "Stage 1 fixed structural families. No workloads or thermal labels are included.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [_format_table_value(row[name]) for name in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    headers = list(rows[0]) if rows else []
    alignment = "l" * len(headers)
    lines = [
        "% Generated Stage 1 family table. No thermal data were used.",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(_latex_escape(name) for name in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_table_value(row[name])) for name in headers) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def _latex_escape(value: str) -> str:
    output = value
    for source, target in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        output = output.replace(source, target)
    return output


def render_family_preview(spec: dict[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas_width, canvas_height = 1000, 820
    left, top, right, bottom = 70, 130, 245, 80
    layout = spec["fixed_structure"]["layout"]
    width = float(layout["package"]["size"]["width"])
    height = float(layout["package"]["size"]["height"])
    chiplets = layout["chiplets"]
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _pil_font(ImageFont, 21)
    subtitle_font = _pil_font(ImageFont, 16)
    label_font = _pil_font(ImageFont, 12 if len(chiplets) <= 24 else 9 if len(chiplets) <= 48 else 7)
    legend_font = _pil_font(ImageFont, 14)
    plot_width = canvas_width - left - right
    plot_height = canvas_height - top - bottom
    scale = min(plot_width / width, plot_height / height)
    package_pixel_width = width * scale
    package_pixel_height = height * scale
    x0 = left + (plot_width - package_pixel_width) / 2.0
    y0 = top + (plot_height - package_pixel_height) / 2.0
    draw.rectangle((x0, y0, x0 + package_pixel_width, y0 + package_pixel_height), fill="#f8fafc", outline="#111827", width=3)

    def pixel_rect(item: dict[str, Any]) -> tuple[float, float, float, float]:
        x, y, chiplet_width, chiplet_height = _rect_tuple(item)
        px0 = x0 + x * scale
        px1 = x0 + (x + chiplet_width) * scale
        py0 = y0 + (height - y - chiplet_height) * scale
        py1 = y0 + (height - y) * scale
        return px0, py0, px1, py1

    for chiplet in chiplets:
        color = TYPE_COLORS[chiplet["type"]]
        rect = pixel_rect(chiplet)
        draw.rectangle(rect, fill=color, outline="#111827", width=1)
        cx = (rect[0] + rect[2]) / 2.0
        cy = (rect[1] + rect[3]) / 2.0
        fill = "#111827" if chiplet["type"] == "ANALOG" else "white"
        draw.text((cx, cy), chiplet["name"], font=label_font, fill=fill, anchor="mm")
    d = spec["descriptors"]
    draw.text(
        (canvas_width / 2.0, 26),
        f"{spec['family_uid']} | {spec['primary_category']} | {spec['primary_split']}",
        font=title_font,
        fill="#111827",
        anchor="ma",
    )
    draw.text(
        (canvas_width / 2.0, 62),
        f"{width:g} x {height:g} mm | {len(chiplets)} dies | whitespace {100*d['whitespace_fraction']:.1f}%",
        font=subtitle_font,
        fill="#334155",
        anchor="ma",
    )
    present_types = [chiplet_type for chiplet_type in TYPE_ORDER if d["type_counts"][chiplet_type]]
    legend_x = canvas_width - right + 30
    legend_y = top + 10
    draw.text((legend_x, legend_y - 32), "Chiplet types", font=subtitle_font, fill="#111827")
    for index, chiplet_type in enumerate(present_types):
        y = legend_y + index * 34
        draw.rectangle((legend_x, y, legend_x + 24, y + 20), fill=TYPE_COLORS[chiplet_type], outline="#111827")
        draw.text((legend_x + 34, y + 2), chiplet_type, font=legend_font, fill="#111827")
    draw.text((x0, y0 + package_pixel_height + 22), "x (mm)", font=legend_font, fill="#334155")
    draw.text((18, y0 + package_pixel_height / 2.0), "y (mm)", font=legend_font, fill="#334155")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def render_contact_sheets(families: Sequence[dict[str, Any]], preview_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    groups: dict[str, list[dict[str, Any]]] = {}
    for spec in families:
        groups.setdefault(f"category_{spec['primary_category']}", []).append(spec)
        groups.setdefault(f"split_{spec['primary_split']}", []).append(spec)
    for name, selected in sorted(groups.items()):
        columns = min(5, len(selected))
        rows = math.ceil(len(selected) / columns)
        tile_width, tile_height, header = 300, 250, 56
        sheet = Image.new("RGB", (columns * tile_width, header + rows * tile_height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (sheet.width / 2.0, 18),
            name.replace("_", " ").title(),
            font=_pil_font(ImageFont, 22),
            fill="#111827",
            anchor="ma",
        )
        for index, spec in enumerate(selected):
            row, col = divmod(index, columns)
            preview = Image.open(preview_dir / f"{spec['family_uid']}.png").convert("RGB")
            preview.thumbnail((tile_width - 12, tile_height - 28))
            px = col * tile_width + (tile_width - preview.width) // 2
            py = header + row * tile_height + 24
            sheet.paste(preview, (px, py))
            draw.text(
                (col * tile_width + tile_width / 2.0, header + row * tile_height + 4),
                spec["family_uid"],
                font=_pil_font(ImageFont, 14),
                fill="#111827",
                anchor="ma",
            )
        sheet.save(preview_dir / f"contact_sheet_{name}.png", format="PNG", optimize=False)


def _pil_font(image_font_module: Any, size: int) -> Any:
    try:
        return image_font_module.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return image_font_module.load_default()


def write_review_report(
    path: Path,
    validation: dict[str, Any],
    suspicious: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    taxonomy: Sequence[dict[str, Any]],
    nearest: Sequence[dict[str, Any]],
) -> None:
    weak = [row for row in coverage if bool(row["weak_test_overlap"])]
    trivial = [row for row in coverage if bool(row["trivially_separates_test"])]
    top_nearest = sorted(nearest, key=lambda row: float(row["distance"]))[:10]
    lines = [
        "# Benchmark v2 Stage 1 Review",
        "",
        "## Scope",
        "",
        "Stage 1 instantiated fixed structural family definitions only. It ran no HotSpot simulations, generated no workloads or thermal labels, and built no model tensors.",
        "",
        "## Acceptance summary",
        "",
        f"- Recommendation: **{validation['recommendation']}**",
        f"- Exactly 50 fixed families present: **{'yes' if validation['family_count'] == 50 else 'no'}**",
        f"- All geometry validation passed: **{'yes' if not [p for p in validation['problems'] if 'layout' in p or 'gap' in p or 'geometry' in p] else 'no'}**",
        f"- Material/cooling and HotSpot settings fixed: **{'yes' if validation['material_stack_hash'] and validation['hotspot_settings_hash'] else 'no'}**",
        f"- Primary split counts: `{validation['split_counts']}`",
        f"- Rotational groups: `{validation['rotational_group_counts']}`",
        f"- Suspicious pair count at distance < {validation['near_duplicate_threshold_proposal']:.3f}, plus intentional matched pairs: {validation['suspicious_pair_count']}",
        f"- Unexplained cross-split suspicious pairs: {validation['unexplained_cross_split_suspicious_pair_count']}",
        f"- Test-separating scalar descriptors: {', '.join(validation['trivially_separating_axes']) or 'none'}",
        "",
        "## Taxonomy",
        "",
        "| Category | Total | Train | Val | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in taxonomy:
        lines.append(f"| {row['primary_category']} | {row['total']} | {row['train']} | {row['val']} | {row['test']} |")
    lines.extend(["", "## Nearest and suspicious families", ""])
    if suspicious:
        lines.extend(["| Family A | Family B | Distance | Cross split | Intentional | Justification |", "|---|---|---:|---|---|---|"])
        for row in suspicious[:20]:
            lines.append(
                f"| {row['family_a']} | {row['family_b']} | {float(row['distance']):.5f} | "
                f"{row['cross_split']} | {row['intentional_matched_pair']} | {row['justification']} |"
            )
    else:
        lines.append("No pair falls below the proposed suspicious-distance threshold.")
    lines.extend(["", "The ten closest per-family neighbor records include:", ""])
    for row in top_nearest:
        lines.append(
            f"- `{row['family_uid']}` -> `{row['nearest_family_uid']}`: {float(row['distance']):.5f} "
            f"(cross split: {row['cross_split']})"
        )
    lines.extend(["", "## Split coverage", ""])
    if weak:
        lines.append("Descriptors with at least one validation/test value outside the train scalar range:")
        for row in weak:
            lines.append(
                f"- `{row['descriptor']}`: test overlap {float(row['test_fraction_inside_train_range']):.2f}, "
                f"best threshold balanced accuracy {float(row['best_test_vs_train_threshold_balanced_accuracy']):.3f}"
            )
    else:
        lines.append("All tested validation/test scalar values lie inside their corresponding train ranges.")
    if trivial:
        lines.append("")
        lines.append("The following descriptors perfectly separate test from train and block Phase 2: " + ", ".join(row["descriptor"] for row in trivial))
    else:
        lines.append("")
        lines.append("No audited individual scalar perfectly separates all five test families from train.")
    lines.extend(
        [
            "",
            "## Compound-OOD assessment",
            "",
            "The designated test families `f008`, `f016`, `f027`, `f033`, and `f044` combine covered marginal geometry/type regimes. Their individual audited scalar values must remain covered by train; the held-out object is the joint topology. No model result was used to choose these assignments.",
            "",
            "## Manual review",
            "",
            "Manual visual review is required for every family before Phase 2. Additional attention should go to the intentional matched pairs `f047/f048` and `f049/f050`, every pair listed above as suspicious, the high-die families `f031-f033`, and the five compound-OOD test families.",
            "",
            "## Validation problems",
            "",
        ]
    )
    if validation["problems"]:
        lines.extend(f"- {problem}" for problem in validation["problems"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Phase 2 gate",
            "",
            f"**{validation['recommendation']}**. `GO WITH MANUAL REVIEW` means all machine acceptance gates pass, but intentional matched/nearest structures still require human approval of their layout previews. This report alone does not authorize workload or HotSpot generation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
