#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.audit_benchmark_v2_family_design_space import (  # noqa: E402
    deterministic_kmeans,
    extract_family_descriptor,
    fit_train_pca,
    fit_train_standardizer,
    generate_recommendation,
    nearest_family_rows,
    rectangle_boundary_clearance,
    rectangle_gap,
    standardized_matrix,
    summarize_numeric_descriptors,
)


def family_spec(uid: str = "f001", split: str = "train", offset: float = 0.0) -> dict:
    return {
        "schema_version": "benchmark_v2_family/1",
        "family_uid": uid,
        "primary_split": split,
        "primary_category": "synthetic",
        "placement_style": "synthetic_pair",
        "secondary_tags": [],
        "fixed_structure": {
            "grid": {"rows": 64, "cols": 64, "map_mode": "avg"},
            "layout": {
                "schema_version": 1,
                "units": {"length": "mm"},
                "package": {
                    "name": uid,
                    "substrate": "silicon_interposer",
                    "size": {"width": 20.0, "height": 10.0},
                },
                "chiplets": [
                    {
                        "name": "CPU00",
                        "type": "CPU",
                        "position": {"x": 1.0 + offset, "y": 1.0},
                        "size": {"width": 4.0, "height": 3.0},
                    },
                    {
                        "name": "GPU00",
                        "type": "GPU",
                        "position": {"x": 10.0 + offset, "y": 5.0},
                        "size": {"width": 5.0, "height": 3.0},
                    },
                ],
            },
            "thermal_stack": {
                "ambient_K": 318.15,
                "initial_temperature_K": 318.15,
                "chip": {
                    "thickness_m": 0.00015,
                    "thermal_conductivity_W_per_mK": 130.0,
                    "volumetric_heat_capacity_J_per_m3K": 1_630_300,
                },
                "interface": {
                    "thickness_m": 2.0e-5,
                    "thermal_conductivity_W_per_mK": 4.0,
                    "volumetric_heat_capacity_J_per_m3K": 4_000_000,
                },
                "spreader": {
                    "side_m": 0.09,
                    "thickness_m": 0.001,
                    "thermal_conductivity_W_per_mK": 400.0,
                    "volumetric_heat_capacity_J_per_m3K": 3_550_000,
                },
                "sink": {
                    "side_m": 0.1,
                    "thickness_m": 0.0069,
                    "thermal_conductivity_W_per_mK": 400.0,
                    "volumetric_heat_capacity_J_per_m3K": 3_550_000,
                    "convection_resistance_K_per_W": 0.12,
                    "convection_capacitance_J_per_K": 140.4,
                },
            },
            "hotspot": {
                "grid": {"rows": 64, "cols": 64, "map_mode": "avg"},
                "sampling_interval_s": 0.01,
                "base_processor_frequency_Hz": 3_000_000_000,
                "leakage_used": False,
                "detailed_package": False,
                "secondary_path": False,
            },
            "material_and_cooling_variant": "fixed_default",
        },
    }


def simple_records() -> list[dict]:
    return [
        {"family_uid": "f001", "split": "train", "primary_category": "a", "a": 0.0, "b": 0.0, "constant": 1.0},
        {"family_uid": "f002", "split": "train", "primary_category": "a", "a": 1.0, "b": 0.0, "constant": 1.0},
        {"family_uid": "f003", "split": "train", "primary_category": "b", "a": 0.0, "b": 1.0, "constant": 1.0},
        {"family_uid": "f004", "split": "train", "primary_category": "b", "a": 1.0, "b": 1.0, "constant": 1.0},
        {"family_uid": "f005", "split": "val", "primary_category": "a", "a": 0.05, "b": 0.0, "constant": 1.0},
        {"family_uid": "f006", "split": "test", "primary_category": "c", "a": 2.0, "b": 2.0, "constant": 1.0},
    ]


def test_descriptor_extraction_and_occupied_area() -> None:
    record = extract_family_descriptor(
        family_spec(),
        config_path=Path("families/f001.yaml"),
        split="train",
        workload={"workload_count": 200},
    )
    assert record["chiplet_count"] == 2
    assert record["distinct_chiplet_type_count"] == 2
    assert abs(record["total_chiplet_area_mm2"] - 27.0) < 1.0e-12
    assert abs(record["occupied_area_ratio"] - 27.0 / 200.0) < 1.0e-12
    assert record["grid_dx_mm"] == 20.0 / 64.0
    assert record["type_CPU_count"] == 1
    assert record["type_GPU_count"] == 1


def test_rectangle_gap_and_boundary_clearance() -> None:
    assert rectangle_gap((0.0, 0.0, 2.0, 2.0), (5.0, 0.0, 2.0, 2.0)) == 3.0
    assert rectangle_gap((0.0, 0.0, 2.0, 2.0), (3.0, 3.0, 1.0, 1.0)) == np.sqrt(2.0)
    assert rectangle_gap((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 2.0, 2.0)) == 0.0
    assert rectangle_boundary_clearance((2.0, 1.0, 4.0, 3.0), 20.0, 10.0) == 1.0


def test_fixed_and_varying_descriptor_detection() -> None:
    summaries = summarize_numeric_descriptors(simple_records(), ("a", "b", "constant"))
    by_name = {row["descriptor"]: row for row in summaries}
    assert by_name["constant"]["variation_class"] == "fixed"
    assert by_name["a"]["variation_class"] == "varying"
    assert by_name["a"]["test_outside_train_count"] == 1


def test_train_only_standardization_ignores_heldout_values() -> None:
    records = simple_records()
    first = fit_train_standardizer(records, ("a", "b", "constant"))
    changed = [dict(row) for row in records]
    changed[-1]["a"] = 1_000_000.0
    changed[-1]["b"] = -1_000_000.0
    second = fit_train_standardizer(changed, ("a", "b", "constant"))
    assert first.names == ("a", "b")
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.scale, second.scale)


def test_nearest_neighbor_ordering_and_redundancy_detection() -> None:
    records = simple_records()
    standardizer = fit_train_standardizer(records, ("a", "b", "constant"))
    matrix, uids = standardized_matrix(records, standardizer)
    nearest, redundant = nearest_family_rows(
        records,
        matrix,
        uids,
        k=2,
        redundancy_threshold=0.20,
    )
    f005 = [row for row in nearest if row["family_uid"] == "f005"]
    assert f005[0]["neighbor_family_uid"] == "f001"
    assert float(f005[0]["rms_standardized_distance"]) < float(f005[1]["rms_standardized_distance"])
    assert any({row["family_a"], row["family_b"]} == {"f001", "f005"} for row in redundant)


def test_missing_optional_cooling_field_is_not_invented() -> None:
    spec = family_spec()
    del spec["fixed_structure"]["thermal_stack"]["sink"]["convection_capacitance_J_per_K"]
    record = extract_family_descriptor(
        spec,
        config_path=Path("families/f001.yaml"),
        split="train",
    )
    assert "sink_convection_capacitance_J_per_K" not in record
    assert record["sink_convection_resistance_K_per_W"] == 0.12


def test_clustering_and_train_fit_pca_are_deterministic() -> None:
    records = simple_records()
    standardizer = fit_train_standardizer(records, ("a", "b", "constant"))
    matrix, uids = standardized_matrix(records, standardizer)
    first_clusters = deterministic_kmeans(matrix, uids, cluster_count=2)
    second_clusters = deterministic_kmeans(matrix, uids, cluster_count=2)
    assert first_clusters == second_clusters
    first_pca = fit_train_pca(records, standardizer)
    second_pca = fit_train_pca(records, standardizer)
    assert np.array_equal(first_pca["coordinates"], second_pca["coordinates"])


def test_recommendation_generation_preserves_existing_benchmark_for_fixed_physics_gap() -> None:
    recommendation = generate_recommendation(
        family_count=50,
        redundant_pairs=[],
        gap_analysis={
            "fixed_dimensions": [
                "package_material_variation",
                "cooling_boundary_condition_variation",
                "layer_stack_variation",
            ],
            "dimension_coverage": {
                "package_material_variation": "fixed_not_varied",
                "cooling_boundary_condition_variation": "fixed_not_varied",
                "layer_stack_variation": "fixed_not_varied",
            },
            "redundant_family_uids": [],
        },
    )
    assert recommendation["code"] == "C"
    assert recommendation["numerical_family_count_sufficient"] is True
    assert "must not rewrite" in recommendation["current_result_validity"]


if __name__ == "__main__":
    test_descriptor_extraction_and_occupied_area()
    test_rectangle_gap_and_boundary_clearance()
    test_fixed_and_varying_descriptor_detection()
    test_train_only_standardization_ignores_heldout_values()
    test_nearest_neighbor_ordering_and_redundancy_detection()
    test_missing_optional_cooling_field_is_not_invented()
    test_clustering_and_train_fit_pca_are_deterministic()
    test_recommendation_generation_preserves_existing_benchmark_for_fixed_physics_gap()
    print("Benchmark v2 family design-space audit tests passed")
