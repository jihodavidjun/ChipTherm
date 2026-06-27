from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .layout import Layout, load_layout


@dataclass(frozen=True)
class Scenario:
    schema_version: int
    name: str
    root: Path
    layout_path: Path
    power_path: Path
    package_path: Path
    hotspot_path: Path
    benchmark_path: Path | None = None


@dataclass(frozen=True)
class PowerModel:
    schema_version: int
    units_power: str
    mode: str
    chiplet_watts: dict[str, float]
    active_workload: str | None = None
    workloads: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class ThermalPackage:
    schema_version: int
    ambient_K: float
    initial_temperature_K: float
    options: dict[str, float]


@dataclass(frozen=True)
class HotspotSettings:
    schema_version: int
    model_type: str
    grid_rows: int
    grid_cols: int
    grid_map_mode: str
    sampling_interval_s: float
    base_processor_frequency_Hz: float
    leakage_used: bool
    detailed_package: bool
    secondary_path: bool


@dataclass(frozen=True)
class SimulationInput:
    scenario: Scenario
    layout: Layout
    power: PowerModel
    package: ThermalPackage
    hotspot: HotspotSettings
    benchmark: dict[str, Any] | None = None


def load_simulation_input(path: str | Path) -> SimulationInput:
    scenario = load_scenario(path)
    return SimulationInput(
        scenario=scenario,
        layout=load_layout(scenario.layout_path),
        power=load_power(scenario.power_path),
        package=load_package(scenario.package_path),
        hotspot=load_hotspot(scenario.hotspot_path),
        benchmark=load_benchmark(scenario.benchmark_path) if scenario.benchmark_path is not None else None,
    )


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = _load_yaml(path)
    files = data.get("files", {})
    root = path.parent

    return Scenario(
        schema_version=int(data.get("schema_version", 1)),
        name=str(data.get("name", path.parent.name)),
        root=root,
        layout_path=_resolve(root, files.get("layout")),
        power_path=_resolve(root, files.get("power")),
        package_path=_resolve(root, files.get("package")),
        hotspot_path=_resolve(root, files.get("hotspot")),
        benchmark_path=_resolve(root, files.get("benchmark")) if files.get("benchmark") is not None else None,
    )


def load_power(path: str | Path) -> PowerModel:
    data = _load_yaml(path)
    units = data.get("units", {})
    workloads_raw = data.get("workloads")
    workloads = None
    if workloads_raw is not None:
        if not isinstance(workloads_raw, dict):
            raise ValueError(f"{path} workloads must be a mapping")
        workloads = {
            str(workload): {str(name): float(value) for name, value in powers.items()}
            for workload, powers in workloads_raw.items()
        }
    active_workload = data.get("active_workload")
    chiplets = data.get("chiplets")
    if chiplets is None and workloads is not None and active_workload is not None:
        chiplets = workloads[str(active_workload)]
    if chiplets is None:
        chiplets = {}
    return PowerModel(
        schema_version=int(data.get("schema_version", 1)),
        units_power=str(units.get("power", "")),
        mode=str(data.get("mode", "")),
        chiplet_watts={str(name): float(value) for name, value in chiplets.items()},
        active_workload=str(active_workload) if active_workload is not None else None,
        workloads=workloads,
    )


def load_benchmark(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_yaml(path)


def load_package(path: str | Path) -> ThermalPackage:
    data = _load_yaml(path)
    options = {
        "-ambient": _float(data.get("ambient_K")),
        "-init_temp": _float(data.get("initial_temperature_K", data.get("ambient_K"))),
        "-t_chip": _float(data.get("chip", {}).get("thickness_m")),
        "-k_chip": _float(data.get("chip", {}).get("thermal_conductivity_W_per_mK")),
        "-p_chip": _float(data.get("chip", {}).get("volumetric_heat_capacity_J_per_m3K")),
        "-t_interface": _float(data.get("interface", {}).get("thickness_m")),
        "-k_interface": _float(data.get("interface", {}).get("thermal_conductivity_W_per_mK")),
        "-p_interface": _float(data.get("interface", {}).get("volumetric_heat_capacity_J_per_m3K")),
        "-s_spreader": _float(data.get("spreader", {}).get("side_m")),
        "-t_spreader": _float(data.get("spreader", {}).get("thickness_m")),
        "-k_spreader": _float(data.get("spreader", {}).get("thermal_conductivity_W_per_mK")),
        "-p_spreader": _float(data.get("spreader", {}).get("volumetric_heat_capacity_J_per_m3K")),
        "-s_sink": _float(data.get("sink", {}).get("side_m")),
        "-t_sink": _float(data.get("sink", {}).get("thickness_m")),
        "-k_sink": _float(data.get("sink", {}).get("thermal_conductivity_W_per_mK")),
        "-p_sink": _float(data.get("sink", {}).get("volumetric_heat_capacity_J_per_m3K")),
        "-r_convec": _float(data.get("sink", {}).get("convection_resistance_K_per_W")),
        "-c_convec": _float(data.get("sink", {}).get("convection_capacitance_J_per_K")),
    }
    return ThermalPackage(
        schema_version=int(data.get("schema_version", 1)),
        ambient_K=options["-ambient"],
        initial_temperature_K=options["-init_temp"],
        options=options,
    )


def load_hotspot(path: str | Path) -> HotspotSettings:
    data = _load_yaml(path)
    grid = data.get("grid", {})
    return HotspotSettings(
        schema_version=int(data.get("schema_version", 1)),
        model_type=str(data.get("model_type", "grid")),
        grid_rows=int(grid.get("rows", 64)),
        grid_cols=int(grid.get("cols", 64)),
        grid_map_mode=str(grid.get("map_mode", "avg")),
        sampling_interval_s=float(data.get("sampling_interval_s", 0.01)),
        base_processor_frequency_Hz=float(data.get("base_processor_frequency_Hz", 3e9)),
        leakage_used=bool(data.get("leakage_used", False)),
        detailed_package=bool(data.get("detailed_package", False)),
        secondary_path=bool(data.get("secondary_path", False)),
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _resolve(root: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("scenario files must include layout, power, package, and hotspot")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def _float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)
