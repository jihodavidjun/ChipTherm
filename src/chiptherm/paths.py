from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def hotspot_home(config_path: str | Path | None = None) -> Path:
    env_value = os.environ.get("HOTSPOT_HOME")
    if env_value:
        return Path(env_value).expanduser()

    config_file = Path(config_path) if config_path is not None else REPO_ROOT / "config.yaml"
    if config_file.exists():
        data: dict[str, Any] = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        root = data.get("hotspot", {}).get("root")
        if root:
            return Path(root).expanduser()

    raise RuntimeError("Set HOTSPOT_HOME or add hotspot.root to config.yaml")
