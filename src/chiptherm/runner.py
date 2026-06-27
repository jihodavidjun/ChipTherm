from __future__ import annotations

import subprocess
from pathlib import Path


def build_hotspot_command(
    *,
    hotspot_home: str | Path,
    config_path: str | Path,
    flp_path: str | Path,
    ptrace_path: str | Path,
    steady_path: str | Path,
    grid_steady_path: str | Path,
) -> list[str]:
    executable = Path(hotspot_home) / "hotspot"
    if not executable.exists():
        raise FileNotFoundError(f"HotSpot executable not found at {executable}")
    if not executable.is_file():
        raise FileNotFoundError(f"HotSpot executable path is not a file: {executable}")

    return [
        str(executable),
        "-c",
        str(config_path),
        "-f",
        str(flp_path),
        "-p",
        str(ptrace_path),
        "-model_type",
        "grid",
        "-steady_file",
        str(steady_path),
        "-grid_steady_file",
        str(grid_steady_path),
    ]


def run_hotspot(command: list[str], cwd: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)
