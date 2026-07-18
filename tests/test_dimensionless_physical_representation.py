from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import build_dimensionless_v1_input, build_dimensionless_v2_input  # noqa: E402
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402
from chiptherm.ml.normalization import NormalizationStats, build_model_input  # noqa: E402


CHANNEL_NAMES = [
    "power_density_W_per_mm2",
    "occupancy_mask",
    "CPU_mask",
    "GPU_or_NPU_mask",
    "memory_mask",
    "IO_or_ANALOG_or_MEMS_mask",
    "normalized_x_coordinate",
    "normalized_y_coordinate",
    "total_power_W",
    "package_width_mm",
    "package_height_mm",
    "cell_size_x_mm",
    "cell_size_y_mm",
    "finite_source_L0p5mm",
    "finite_source_L1mm",
    "finite_source_L2mm",
    "finite_source_L4mm",
    "enclosed_power_R2mm_W",
    "enclosed_power_R4mm_W",
    "enclosed_power_R8mm_W",
    "enclosed_power_R16mm_W",
    "distance_to_left_edge_mm",
    "distance_to_right_edge_mm",
    "distance_to_bottom_edge_mm",
    "distance_to_top_edge_mm",
    "minimum_distance_to_package_edge_mm",
    "chiplet_total_power_W",
    "chiplet_width_mm",
    "chiplet_height_mm",
    "chiplet_area_mm2",
    "chiplet_aspect_ratio",
    "chiplet_power_density_W_per_mm2",
    "thermal_crowding_W_per_mm",
]
IDX = {name: index for index, name in enumerate(CHANNEL_NAMES)}


def make_tensor(*, scale_length: float = 1.0, scale_power: float = 1.0) -> torch.Tensor:
    x = torch.zeros((len(CHANNEL_NAMES), 4, 4), dtype=torch.float32)
    width = 8.0 * scale_length
    height = 2.0 * scale_length
    cell_x = width / 4.0
    cell_y = height / 4.0
    total_power = 100.0 * scale_power
    package_area = width * height
    occupied = torch.zeros((4, 4), dtype=torch.float32)
    occupied[:2, :2] = 1.0
    occupied_area = float(occupied.sum().item()) * cell_x * cell_y
    char_pd = total_power / occupied_area
    l_char = package_area**0.5
    x[IDX["occupancy_mask"]] = occupied
    x[IDX["normalized_x_coordinate"]] = torch.linspace(0.125, 0.875, 4).view(1, 4).repeat(4, 1)
    x[IDX["normalized_y_coordinate"]] = torch.linspace(0.125, 0.875, 4).view(4, 1).repeat(1, 4)
    x[IDX["power_density_W_per_mm2"]] = 2.0 * char_pd * occupied
    x[IDX["total_power_W"]] = total_power
    x[IDX["package_width_mm"]] = width
    x[IDX["package_height_mm"]] = height
    x[IDX["cell_size_x_mm"]] = cell_x
    x[IDX["cell_size_y_mm"]] = cell_y
    for name in ("finite_source_L0p5mm", "finite_source_L1mm", "finite_source_L2mm", "finite_source_L4mm"):
        x[IDX[name]] = 3.0 * total_power / l_char
    for value, name in zip((10.0, 25.0, 50.0, 75.0), ("enclosed_power_R2mm_W", "enclosed_power_R4mm_W", "enclosed_power_R8mm_W", "enclosed_power_R16mm_W")):
        x[IDX[name]] = value * scale_power
    x[IDX["distance_to_left_edge_mm"]] = 2.0 * scale_length
    x[IDX["distance_to_right_edge_mm"]] = 6.0 * scale_length
    x[IDX["distance_to_bottom_edge_mm"]] = 0.5 * scale_length
    x[IDX["distance_to_top_edge_mm"]] = 1.5 * scale_length
    x[IDX["minimum_distance_to_package_edge_mm"]] = 0.5 * scale_length
    x[IDX["chiplet_total_power_W"]] = 20.0 * scale_power * occupied
    x[IDX["chiplet_width_mm"]] = 2.0 * scale_length * occupied
    x[IDX["chiplet_height_mm"]] = 1.0 * scale_length * occupied
    x[IDX["chiplet_area_mm2"]] = 2.0 * scale_length * scale_length * occupied
    x[IDX["chiplet_aspect_ratio"]] = 2.0 * occupied
    x[IDX["chiplet_power_density_W_per_mm2"]] = 1.5 * char_pd * occupied
    x[IDX["thermal_crowding_W_per_mm"]] = 4.0 * total_power / l_char
    return x


class DimensionlessPhysicalRepresentationTests(unittest.TestCase):
    def test_uniform_scale_preserves_dimensionless_geometry(self) -> None:
        base = build_dimensionless_v1_input(make_tensor(scale_length=1.0), IDX)
        scaled = build_dimensionless_v1_input(make_tensor(scale_length=2.0), IDX)

        for name in (
            "normalized_x_coordinate",
            "normalized_y_coordinate",
            "package_width_mm",
            "package_height_mm",
            "cell_size_x_mm",
            "cell_size_y_mm",
            "chiplet_width_mm",
            "chiplet_height_mm",
            "chiplet_area_mm2",
            "minimum_distance_to_package_edge_mm",
        ):
            self.assertTrue(torch.allclose(base[IDX[name]], scaled[IDX[name]], atol=1.0e-6), name)

    def test_directional_and_characteristic_distance_normalization(self) -> None:
        out = build_dimensionless_v1_input(make_tensor(), IDX)

        self.assertAlmostEqual(float(out[IDX["distance_to_left_edge_mm"]][0, 0]), 2.0 / 8.0, places=6)
        self.assertAlmostEqual(float(out[IDX["distance_to_top_edge_mm"]][0, 0]), 1.5 / 2.0, places=6)
        self.assertAlmostEqual(float(out[IDX["minimum_distance_to_package_edge_mm"]][0, 0]), 0.5 / 4.0, places=6)
        self.assertAlmostEqual(float(out[IDX["finite_source_L2mm"]][0, 0]), 3.0, places=6)
        self.assertAlmostEqual(float(out[IDX["thermal_crowding_W_per_mm"]][0, 0]), 4.0, places=6)

    def test_power_fraction_invariance_and_absolute_power_available_outside_x(self) -> None:
        base = build_dimensionless_v1_input(make_tensor(scale_power=1.0), IDX)
        scaled = build_dimensionless_v1_input(make_tensor(scale_power=5.0), IDX)

        for name in (
            "power_density_W_per_mm2",
            "total_power_W",
            "chiplet_total_power_W",
            "chiplet_power_density_W_per_mm2",
            "enclosed_power_R16mm_W",
        ):
            self.assertTrue(torch.allclose(base[IDX[name]], scaled[IDX[name]], atol=1.0e-6), name)
        self.assertAlmostEqual(float(base[IDX["total_power_W"]][0, 0]), 1.0, places=6)

    def test_source_base_is_preserved_as_absolute_kelvin_input(self) -> None:
        x = build_dimensionless_v1_input(make_tensor(), IDX).unsqueeze(0)
        physics = torch.full((1, 4, 4), 350.0)
        stats = NormalizationStats(
            schema_version=1,
            power_density_mean=0.0,
            power_density_std=1.0,
            physics_mean=0.0,
            physics_std=1.0,
            residual_mean=0.0,
            residual_std=1.0,
            num_samples=1,
            num_grid_cells=16,
            input_channels=len(CHANNEL_NAMES),
        )

        model_input = build_model_input(x, physics, stats, physics_input_mode="source_superposition_v1")

        self.assertEqual(model_input.shape[1], len(CHANNEL_NAMES) + 1)
        self.assertTrue(torch.allclose(model_input[:, -1], physics))

    def test_invalid_denominators_fail_and_outputs_are_finite(self) -> None:
        x = make_tensor()
        out = build_dimensionless_v1_input(x, IDX)
        self.assertTrue(torch.isfinite(out).all())

        bad = make_tensor()
        bad[IDX["total_power_W"]] = 0.0
        with self.assertRaises(ValueError):
            build_dimensionless_v1_input(bad, IDX)


class DimensionlessV2PhysicalRepresentationTests(unittest.TestCase):
    def test_v2_starts_from_dimensional_tensor_not_v1(self) -> None:
        dimensional = make_tensor()
        v1 = build_dimensionless_v1_input(dimensional, IDX)
        v2_from_dimensional = build_dimensionless_v2_input(dimensional, IDX)
        v2_from_v1 = build_dimensionless_v2_input(v1, IDX)

        self.assertFalse(torch.allclose(v2_from_dimensional, v2_from_v1))
        self.assertTrue(torch.allclose(v2_from_dimensional[IDX["power_density_W_per_mm2"]], dimensional[IDX["power_density_W_per_mm2"]]))

    def test_v2_keeps_power_thermal_and_package_scale_channels_dimensional(self) -> None:
        x = make_tensor()
        out = build_dimensionless_v2_input(x, IDX)

        for name in (
            "power_density_W_per_mm2",
            "total_power_W",
            "package_width_mm",
            "package_height_mm",
            "cell_size_x_mm",
            "cell_size_y_mm",
            "finite_source_L0p5mm",
            "finite_source_L1mm",
            "finite_source_L2mm",
            "finite_source_L4mm",
            "enclosed_power_R2mm_W",
            "enclosed_power_R4mm_W",
            "enclosed_power_R8mm_W",
            "enclosed_power_R16mm_W",
            "thermal_crowding_W_per_mm",
            "chiplet_total_power_W",
            "chiplet_power_density_W_per_mm2",
        ):
            self.assertTrue(torch.equal(out[IDX[name]], x[IDX[name]]), name)

    def test_v2_geometry_ratios_are_correct(self) -> None:
        x = make_tensor()
        out = build_dimensionless_v2_input(x, IDX)

        self.assertAlmostEqual(float(out[IDX["chiplet_width_mm"]][0, 0]), 2.0 / 8.0, places=6)
        self.assertAlmostEqual(float(out[IDX["chiplet_height_mm"]][0, 0]), 1.0 / 2.0, places=6)
        self.assertAlmostEqual(float(out[IDX["chiplet_area_mm2"]][0, 0]), 2.0 / 16.0, places=6)
        self.assertAlmostEqual(float(out[IDX["distance_to_left_edge_mm"]][0, 0]), 2.0 / 8.0, places=6)
        self.assertAlmostEqual(float(out[IDX["distance_to_right_edge_mm"]][0, 0]), 6.0 / 8.0, places=6)
        self.assertAlmostEqual(float(out[IDX["distance_to_bottom_edge_mm"]][0, 0]), 0.5 / 2.0, places=6)
        self.assertAlmostEqual(float(out[IDX["distance_to_top_edge_mm"]][0, 0]), 1.5 / 2.0, places=6)
        self.assertAlmostEqual(float(out[IDX["minimum_distance_to_package_edge_mm"]][0, 0]), 0.5 / 4.0, places=6)

    def test_v2_preserves_masks_coordinates_and_aspect_ratio(self) -> None:
        x = make_tensor()
        out = build_dimensionless_v2_input(x, IDX)

        for name in (
            "occupancy_mask",
            "CPU_mask",
            "GPU_or_NPU_mask",
            "memory_mask",
            "IO_or_ANALOG_or_MEMS_mask",
            "normalized_x_coordinate",
            "normalized_y_coordinate",
            "chiplet_aspect_ratio",
        ):
            self.assertTrue(torch.equal(out[IDX[name]], x[IDX[name]]), name)

    def test_v2_same_channel_count_and_source_base_preserved(self) -> None:
        x = build_dimensionless_v2_input(make_tensor(), IDX).unsqueeze(0)
        physics = torch.full((1, 4, 4), 350.0)
        stats = NormalizationStats(
            schema_version=1,
            power_density_mean=0.0,
            power_density_std=1.0,
            physics_mean=0.0,
            physics_std=1.0,
            residual_mean=0.0,
            residual_std=1.0,
            num_samples=1,
            num_grid_cells=16,
            input_channels=len(CHANNEL_NAMES),
        )

        model_input = build_model_input(x, physics, stats, physics_input_mode="source_superposition_v1")

        self.assertEqual(x.shape[1], len(CHANNEL_NAMES))
        self.assertEqual(model_input.shape[1], len(CHANNEL_NAMES) + 1)
        self.assertTrue(torch.allclose(model_input[:, -1], physics))

    def test_v2_invalid_package_scale_fails_and_outputs_are_finite(self) -> None:
        out = build_dimensionless_v2_input(make_tensor(), IDX)
        self.assertTrue(torch.isfinite(out).all())

        bad = make_tensor()
        bad[IDX["package_width_mm"]] = 0.0
        with self.assertRaises(ValueError):
            build_dimensionless_v2_input(bad, IDX)

    def test_v2_does_not_change_model_capacity_or_active_indices(self) -> None:
        config = {
            "architecture": "miniunet_refine_conditioned_decomposed_feature_fusion",
            "input_channels": len(CHANNEL_NAMES) + 1,
            "base_channels": 4,
            "refine_channels": 4,
            "refine_blocks": 1,
            "refinement_channel_indices": [0, 1, 6, 7, 15, 16, 19, 20, 25, 33],
            "refinement_channel_names": [
                "power_density_W_per_mm2",
                "occupancy_mask",
                "normalized_x_coordinate",
                "normalized_y_coordinate",
                "finite_source_L2mm",
                "finite_source_L4mm",
                "enclosed_power_R8mm_W",
                "enclosed_power_R16mm_W",
                "minimum_distance_to_package_edge_mm",
                "source_superposition_base_K",
            ],
            "metadata_dim": 3,
            "metadata_hidden_dim": 8,
            "metadata_embedding_dim": 8,
            "global_branch_channel_indices": [0, 1, 6, 7, 15, 16, 19, 20, 25, 33],
            "global_branch_channel_names": [
                "power_density_W_per_mm2",
                "occupancy_mask",
                "normalized_x_coordinate",
                "normalized_y_coordinate",
                "finite_source_L2mm",
                "finite_source_L4mm",
                "enclosed_power_R8mm_W",
                "enclosed_power_R16mm_W",
                "minimum_distance_to_package_edge_mm",
                "source_superposition_base_K",
            ],
            "global_hidden_channels": 4,
            "global_pool_size": 8,
            "global_blocks": 1,
            "mean_head_mode": "residual_resistance",
        }
        dimensional_model = build_model({**config, "physical_representation": "dimensional"})
        v2_model = build_model({**config, "physical_representation": "dimensionless_v2"})

        self.assertEqual(count_parameters(dimensional_model), count_parameters(v2_model))
        self.assertEqual(dimensional_model.refinement_channel_indices, v2_model.refinement_channel_indices)
        self.assertEqual(dimensional_model.coarse_model.global_encoder.channel_indices, v2_model.coarse_model.global_encoder.channel_indices)


if __name__ == "__main__":
    unittest.main()
