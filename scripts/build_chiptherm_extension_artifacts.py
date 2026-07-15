#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training-ready encoded/metadata/graph artifacts for ChipTherm extension rows.")
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--index-name", default="all_extension_index.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    extension_root = args.extension_root.resolve()
    out_root = args.out_root.resolve()
    encoded_root = out_root / "encoded_package_plus_power"
    graph_root = out_root / "package_plus_power_graph"
    index = extension_root / args.index_name
    if not index.exists():
        raise SystemExit(f"missing extension index: {index}")

    commands = [
        [
            "python3",
            "scripts/encode_dataset.py",
            "--index",
            str(index),
            "--out-dir",
            str(encoded_root),
        ],
        [
            "python3",
            "scripts/build_metadata_features.py",
            "--dataset-root",
            str(encoded_root),
        ],
        [
            "python3",
            "scripts/build_graph_features.py",
            "--source-root",
            str(encoded_root),
            "--out-root",
            str(graph_root),
            *(["--overwrite"] if args.overwrite else []),
        ],
    ]
    for command in commands:
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
