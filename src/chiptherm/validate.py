from __future__ import annotations

import math
import re

from .layout import SUPPORTED_CHIPLET_TYPES, Chiplet, Layout
from .scenario import PowerModel, SimulationInput


NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
EPS = 1e-12
SPACING_TOLERANCE_MM = 1e-5
MIN_SPACING_MM = 0.5
MAX_TOTAL_POWER_DENSITY_W_PER_MM2 = 3.0
POWER_DENSITY_LIMITS_W_PER_MM2 = {
    "CPU": (0.05, 3.0),
    "GPU": (0.05, 3.0),
    "NPU": (0.05, 3.0),
    "HBM": (0.02, 0.35),
    "DRAM": (0.02, 0.5),
    "IO": (0.02, 0.6),
    "ANALOG": (0.01, 0.7),
    "MEMS": (0.005, 0.5),
}


class ValidationError(ValueError):
    pass


class LayoutValidationError(ValidationError):
    pass


def validate_simulation_input(sim: SimulationInput) -> None:
    errors: list[str] = []
    min_spacing_mm = _min_spacing_mm(sim)
    _validate_layout(sim.layout, errors, min_spacing_mm=min_spacing_mm)
    _validate_power(sim.layout, sim.power, errors)
    _validate_package_and_hotspot(sim, errors)

    if errors:
        raise ValidationError("\n".join(errors))


def validate_layout(layout: Layout) -> None:
    errors: list[str] = []
    _validate_layout(layout, errors, min_spacing_mm=MIN_SPACING_MM)
    if errors:
        raise LayoutValidationError("\n".join(errors))


def _validate_layout(layout: Layout, errors: list[str], *, min_spacing_mm: float) -> None:
    if layout.schema_version != 1:
        errors.append(f"layout.schema_version must be 1, got {layout.schema_version}")

    if layout.units.length not in {"m", "mm", "um"}:
        errors.append("layout.units.length must be one of: m, mm, um")

    if not NAME_RE.match(layout.package.name):
        errors.append("package.name must start with a letter and contain only letters, numbers, and underscores")
    _require_positive("package.size.width", layout.package.size.width, errors)
    _require_positive("package.size.height", layout.package.size.height, errors)

    if not layout.chiplets:
        errors.append("chiplets must contain at least one chiplet")

    seen_names: set[str] = set()
    for chiplet in layout.chiplets:
        _validate_chiplet(layout, chiplet, seen_names, errors)

    for index, first in enumerate(layout.chiplets):
        for second in layout.chiplets[index + 1:]:
            if _rectangles_overlap(first, second):
                errors.append(f"{first.name} overlaps {second.name}")
            spacing = _edge_spacing(first, second)
            spacing_tolerance = (SPACING_TOLERANCE_MM * 1e-3) / layout.length_scale_to_m
            if spacing + spacing_tolerance < _min_spacing_in_layout_units(layout, min_spacing_mm):
                errors.append(f"{first.name} and {second.name} spacing is {spacing:.6g} {layout.units.length}; minimum is {min_spacing_mm:g} mm")


def _validate_chiplet(layout: Layout, chiplet: Chiplet, seen_names: set[str], errors: list[str]) -> None:
    if not NAME_RE.match(chiplet.name):
        errors.append(f"chiplet name {chiplet.name!r} must start with a letter and contain only letters, numbers, and underscores")
    elif chiplet.name in seen_names:
        errors.append(f"chiplet name {chiplet.name!r} is duplicated")
    seen_names.add(chiplet.name)

    if chiplet.type not in SUPPORTED_CHIPLET_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_CHIPLET_TYPES))
        errors.append(f"{chiplet.name}: type must be one of {allowed}")

    _require_finite_nonnegative(f"{chiplet.name}.position.x", chiplet.position.x, errors)
    _require_finite_nonnegative(f"{chiplet.name}.position.y", chiplet.position.y, errors)
    _require_positive(f"{chiplet.name}.size.width", chiplet.size.width, errors)
    _require_positive(f"{chiplet.name}.size.height", chiplet.size.height, errors)

    if chiplet.right_x > layout.package.size.width + EPS:
        errors.append(f"{chiplet.name}: right edge exceeds package width")
    if chiplet.top_y > layout.package.size.height + EPS:
        errors.append(f"{chiplet.name}: top edge exceeds package height")


def _validate_power(layout: Layout, power: PowerModel, errors: list[str]) -> None:
    if power.schema_version != 1:
        errors.append(f"power.schema_version must be 1, got {power.schema_version}")
    if power.units_power != "W":
        errors.append("power.units.power must be 'W'")
    if power.mode != "fixed":
        errors.append("only fixed power mode is supported for the first milestone")

    layout_names = {chiplet.name for chiplet in layout.chiplets}
    power_names = set(power.chiplet_watts)
    for name in sorted(layout_names - power_names):
        errors.append(f"power is missing a value for chiplet {name}")
    for name in sorted(power_names - layout_names):
        errors.append(f"power contains unknown chiplet {name}")

    if power.workloads is not None:
        if power.active_workload is None:
            errors.append("power.active_workload is required when power.workloads is provided")
        elif power.active_workload not in power.workloads:
            errors.append(f"power.active_workload {power.active_workload!r} is not present in workloads")

        for workload, values in power.workloads.items():
            workload_names = set(values)
            for name in sorted(layout_names - workload_names):
                errors.append(f"power.workloads.{workload} is missing chiplet {name}")
            for name in sorted(workload_names - layout_names):
                errors.append(f"power.workloads.{workload} contains unknown chiplet {name}")
            for name, watts in values.items():
                _require_positive(f"power.workloads.{workload}.{name}", watts, errors)

        if power.active_workload in (power.workloads or {}):
            active_values = power.workloads[power.active_workload]
            for name in sorted(layout_names & power_names):
                if abs(power.chiplet_watts[name] - active_values[name]) > 1e-9:
                    errors.append(f"power.chiplets.{name} must match active workload {power.active_workload}")

    area_scale_to_mm2 = _area_scale_to_mm2(layout)
    for chiplet in layout.chiplets:
        watts = power.chiplet_watts.get(chiplet.name)
        if watts is None:
            continue
        _require_positive(f"power.{chiplet.name}", watts, errors)
        area_mm2 = chiplet.area * area_scale_to_mm2
        if area_mm2 <= 0.0:
            continue
        density = watts / area_mm2
        low, high = POWER_DENSITY_LIMITS_W_PER_MM2[chiplet.type]
        if density < low or density > high:
            errors.append(
                f"{chiplet.name}: power density {density:.3g} W/mm^2 is outside "
                f"reasonable {chiplet.type} range [{low}, {high}]"
            )

    total_power = sum(power.chiplet_watts.get(chiplet.name, 0.0) for chiplet in layout.chiplets)
    total_area_mm2 = sum(chiplet.area * area_scale_to_mm2 for chiplet in layout.chiplets)
    if total_area_mm2 > 0.0:
        total_density = total_power / total_area_mm2
        if math.isfinite(total_density) and total_density > MAX_TOTAL_POWER_DENSITY_W_PER_MM2:
            errors.append(
                f"total package power density {total_density:.3g} W/mm^2 exceeds sanity limit "
                f"{MAX_TOTAL_POWER_DENSITY_W_PER_MM2:g} W/mm^2"
            )


def _validate_package_and_hotspot(sim: SimulationInput, errors: list[str]) -> None:
    if sim.scenario.schema_version != 1:
        errors.append(f"scenario.schema_version must be 1, got {sim.scenario.schema_version}")
    if sim.package.schema_version != 1:
        errors.append(f"package.schema_version must be 1, got {sim.package.schema_version}")
    if sim.hotspot.schema_version != 1:
        errors.append(f"hotspot.schema_version must be 1, got {sim.hotspot.schema_version}")

    for option, value in sim.package.options.items():
        _require_positive(f"package option {option}", value, errors)

    if sim.hotspot.model_type != "grid":
        errors.append("hotspot.model_type must be 'grid' for the first milestone")
    if sim.hotspot.grid_rows <= 0 or sim.hotspot.grid_cols <= 0:
        errors.append("hotspot grid rows and cols must be positive")
    if sim.hotspot.grid_map_mode not in {"avg", "min", "max", "center"}:
        errors.append("hotspot.grid.map_mode must be one of: avg, min, max, center")
    _require_positive("hotspot.sampling_interval_s", sim.hotspot.sampling_interval_s, errors)
    _require_positive("hotspot.base_processor_frequency_Hz", sim.hotspot.base_processor_frequency_Hz, errors)

    package_width_m = sim.layout.package.size.width * sim.layout.length_scale_to_m
    package_height_m = sim.layout.package.size.height * sim.layout.length_scale_to_m
    required_side_m = max(package_width_m, package_height_m)
    spreader_side = sim.package.options.get("-s_spreader", float("nan"))
    sink_side = sim.package.options.get("-s_sink", float("nan"))
    if math.isfinite(spreader_side) and spreader_side < required_side_m:
        errors.append("-s_spreader must be at least the larger package dimension")
    if math.isfinite(sink_side) and sink_side < required_side_m:
        errors.append("-s_sink must be at least the larger package dimension")


def _require_positive(field: str, value: float, errors: list[str]) -> None:
    if not math.isfinite(value) or value <= 0.0:
        errors.append(f"{field} must be a positive finite number")


def _require_finite_nonnegative(field: str, value: float, errors: list[str]) -> None:
    if not math.isfinite(value) or value < 0.0:
        errors.append(f"{field} must be a non-negative finite number")


def _rectangles_overlap(first: Chiplet, second: Chiplet) -> bool:
    return _axis_overlap(first.left_x, first.right_x, second.left_x, second.right_x) and _axis_overlap(
        first.bottom_y, first.top_y, second.bottom_y, second.top_y
    )


def _axis_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return a_min < b_max - EPS and b_min < a_max - EPS


def _edge_spacing(first: Chiplet, second: Chiplet) -> float:
    dx = max(first.left_x - second.right_x, second.left_x - first.right_x, 0.0)
    dy = max(first.bottom_y - second.top_y, second.bottom_y - first.top_y, 0.0)
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _min_spacing_in_layout_units(layout: Layout, min_spacing_mm: float) -> float:
    return (min_spacing_mm * 1e-3) / layout.length_scale_to_m


def _area_scale_to_mm2(layout: Layout) -> float:
    scale = layout.length_scale_to_m / 1e-3
    return scale * scale


def _min_spacing_mm(sim: SimulationInput) -> float:
    if sim.benchmark is None:
        return MIN_SPACING_MM
    constraints = sim.benchmark.get("generation_constraints", {})
    value = constraints.get("min_spacing_mm")
    if value is None:
        return MIN_SPACING_MM
    return float(value)
