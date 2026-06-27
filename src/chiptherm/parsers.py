from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_layer_grid(path: str | Path, layer: int, rows: int, cols: int) -> np.ndarray:
    path = Path(path)
    expected = rows * cols
    values: list[float] = []
    in_layer = False

    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue

            if line.lower().startswith("layer "):
                parts = line.replace(":", " ").split()
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_num = int(parts[1])
                    if in_layer and layer_num != layer:
                        break
                    in_layer = layer_num == layer
                    continue

            if not in_layer:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            values.append(float(parts[-1]))
            if len(values) == expected:
                break

    if len(values) != expected:
        raise ValueError(f"expected {expected} values for Layer {layer}, found {len(values)} in {path}")

    return np.asarray(values, dtype=np.float64).reshape(rows, cols)


def parse_block_temps(path: str | Path, names: list[str] | tuple[str, ...] | None = None) -> dict[str, float]:
    wanted = set(names) if names is not None else None
    temps: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            parts = raw_line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            if wanted is not None and name not in wanted:
                continue
            temps[name] = float(parts[1])

    if wanted is not None:
        missing = wanted - temps.keys()
        if missing:
            names_text = ", ".join(sorted(missing))
            raise ValueError(f"block temperature file is missing: {names_text}")
    return temps
