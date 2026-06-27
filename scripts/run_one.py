#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.parsers import parse_block_temps, parse_layer_grid
from chiptherm.paths import hotspot_home
from chiptherm.runner import build_hotspot_command, run_hotspot
from chiptherm.scenario import load_simulation_input
from chiptherm.validate import validate_simulation_input
from chiptherm.writers import copy_source_files, read_grid_shape, write_flp, write_hotspot_config, write_manifest, write_ptrace


def main() -> int:
    total_start = time.perf_counter()
    runtime: dict[str, float] = {}

    parser = argparse.ArgumentParser(description="Run one ChipTherm HotSpot grid simulation.")
    parser.add_argument("scenario", type=Path, help="Path to scenario.yaml")
    parser.add_argument("--out-dir", required=True, type=Path, help="Run output directory")
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    args = parser.parse_args()

    scenario_path = args.scenario.resolve()
    out_dir = args.out_dir.resolve()
    source_dir = out_dir / "source"
    hotspot_dir = out_dir / "hotspot"
    outputs_dir = out_dir / "outputs"
    parsed_dir = out_dir / "parsed"
    for path in (source_dir, hotspot_dir, outputs_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    sim = load_simulation_input(scenario_path)
    runtime["load_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    validate_simulation_input(sim)
    runtime["validate_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    source_paths = [
        scenario_path,
        sim.scenario.layout_path.resolve(),
        sim.scenario.power_path.resolve(),
        sim.scenario.package_path.resolve(),
        sim.scenario.hotspot_path.resolve(),
    ]
    copy_source_files(source_paths, source_dir)

    flp_path = write_flp(sim.layout, hotspot_dir / "chiplet.flp")
    ptrace_path = write_ptrace(sim.layout, sim.power, hotspot_dir / "power.ptrace")
    config_path = write_hotspot_config(args.config_template, hotspot_dir / "hotspot.config", sim.package, sim.hotspot)

    rows, cols = read_grid_shape(config_path)
    runtime["write_inputs_s"] = time.perf_counter() - stage_start

    block_steady_path = outputs_dir / "block.steady"
    grid_steady_path = outputs_dir / "grid.steady"
    home = (args.hotspot_home or hotspot_home()).resolve()
    command = build_hotspot_command(
        hotspot_home=home,
        config_path=config_path.resolve(),
        flp_path=flp_path.resolve(),
        ptrace_path=ptrace_path.resolve(),
        steady_path=block_steady_path.resolve(),
        grid_steady_path=grid_steady_path.resolve(),
    )
    command_text = shlex.join(command)
    (out_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    stage_start = time.perf_counter()
    result = run_hotspot(command, cwd=hotspot_dir)
    runtime["hotspot_s"] = time.perf_counter() - stage_start
    (outputs_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (outputs_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"HotSpot failed with exit code {result.returncode}. See {outputs_dir / 'stderr.txt'}")

    stage_start = time.perf_counter()
    layer0 = parse_layer_grid(grid_steady_path, layer=0, rows=rows, cols=cols)
    np.save(parsed_dir / "temp_layer0.npy", layer0)

    chiplet_names = tuple(chiplet.name for chiplet in sim.layout.chiplets)
    block_temps = parse_block_temps(block_steady_path, names=chiplet_names)
    (parsed_dir / "block_temps.json").write_text(json.dumps(block_temps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime["parse_s"] = time.perf_counter() - stage_start
    runtime["total_s"] = time.perf_counter() - total_start

    hottest_block, max_block_temp = max(block_temps.items(), key=lambda item: item[1])
    output_summary = {
        "temp_layer0_shape": list(layer0.shape),
        "temp_layer0_min_K": float(layer0.min()),
        "temp_layer0_max_K": float(layer0.max()),
        "temp_layer0_mean_K": float(layer0.mean()),
        "max_block_temperature_K": float(max_block_temp),
        "hottest_block": hottest_block,
        "grid_rows": rows,
        "grid_cols": cols,
    }

    write_manifest(
        out_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_name": sim.scenario.name,
            "runtime": runtime,
            "hotspot": {
                "home": str(home),
                "binary": command[0],
                "git_commit": _git_commit(home),
                "command": command,
                "command_string": command_text,
                "return_code": result.returncode,
            },
            "hotspot_home": str(home),
            "command": command,
            "command_string": command_text,
            "return_code": result.returncode,
            "grid": {"rows": rows, "cols": cols},
            "output_summary": output_summary,
            "sources": {path.name: _sha256(path) for path in source_paths},
            "generated": {
                "flp": str(flp_path.relative_to(out_dir)),
                "ptrace": str(ptrace_path.relative_to(out_dir)),
                "config": str(config_path.relative_to(out_dir)),
                "block_steady": str(block_steady_path.relative_to(out_dir)),
                "grid_steady": str(grid_steady_path.relative_to(out_dir)),
                "temp_layer0": str((parsed_dir / "temp_layer0.npy").relative_to(out_dir)),
                "block_temps": str((parsed_dir / "block_temps.json").relative_to(out_dir)),
            },
        },
    )

    print("ChipTherm run complete")
    print(f"HotSpot runtime: {runtime['hotspot_s']:.3f} s")
    print(f"Total runtime: {runtime['total_s']:.3f} s")
    print(f"Layer0 shape: {rows} x {cols}")
    print(
        "Layer0 min/max/mean: "
        f"{output_summary['temp_layer0_min_K']:.2f} / "
        f"{output_summary['temp_layer0_max_K']:.2f} / "
        f"{output_summary['temp_layer0_mean_K']:.2f} K"
    )
    print(f"Hottest block: {hottest_block}, {max_block_temp:.2f} K")
    print(f"Run dir: {out_dir}")
    return 0


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
