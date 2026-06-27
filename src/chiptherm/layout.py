from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_CHIPLET_TYPES = {"CPU", "GPU", "HBM", "IO", "NPU", "DRAM", "ANALOG", "MEMS"}


@dataclass(frozen=True)
class Units:
    length: str


@dataclass(frozen=True)
class Size:
    width: float
    height: float


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Chiplet:
    name: str
    type: str
    position: Point
    size: Size
    process_node_nm: int | None = None

    @property
    def left_x(self) -> float:
        return self.position.x

    @property
    def bottom_y(self) -> float:
        return self.position.y

    @property
    def right_x(self) -> float:
        return self.position.x + self.size.width

    @property
    def top_y(self) -> float:
        return self.position.y + self.size.height

    @property
    def area(self) -> float:
        return self.size.width * self.size.height


@dataclass(frozen=True)
class Package:
    name: str
    substrate: str
    size: Size


@dataclass(frozen=True)
class Layout:
    schema_version: int
    units: Units
    package: Package
    chiplets: tuple[Chiplet, ...]

    @property
    def length_scale_to_m(self) -> float:
        return length_scale_to_m(self.units.length)


def length_scale_to_m(unit: str) -> float:
    if unit == "m":
        return 1.0
    if unit == "mm":
        return 1e-3
    if unit == "um":
        return 1e-6
    raise ValueError(f"unsupported length unit: {unit!r}")


def load_layout(path: str | Path) -> Layout:
    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return layout_from_dict(data)


def layout_from_dict(data: dict[str, Any]) -> Layout:
    units = data.get("units", {})
    package = data.get("package", {})
    chiplets = data.get("chiplets", [])

    return Layout(
        schema_version=int(data.get("schema_version", 1)),
        units=Units(length=str(units.get("length", ""))),
        package=Package(
            name=str(package.get("name", "")),
            substrate=str(package.get("substrate", "interposer")),
            size=_size_from_dict(package.get("size", {})),
        ),
        chiplets=tuple(_chiplet_from_dict(item) for item in chiplets),
    )


def _chiplet_from_dict(data: dict[str, Any]) -> Chiplet:
    process_node = data.get("process_node_nm")
    return Chiplet(
        name=str(data.get("name", "")),
        type=str(data.get("type", "")),
        position=_point_from_dict(data.get("position", {})),
        size=_size_from_dict(data.get("size", {})),
        process_node_nm=int(process_node) if process_node is not None else None,
    )


def _size_from_dict(data: dict[str, Any]) -> Size:
    return Size(width=_float(data.get("width")), height=_float(data.get("height")))


def _point_from_dict(data: dict[str, Any]) -> Point:
    return Point(x=_float(data.get("x")), y=_float(data.get("y")))


def _float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)
