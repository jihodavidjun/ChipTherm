#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_secondary_protocol import (  # noqa: E402
    DEFAULT_SEED,
    PROTOCOL_NAME,
    build_protocol_indices,
    generate_family_split,
    primary_artifact_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build immutable index views for the Benchmark v2 secondary 35/5/10 family protocol."
    )
    parser.add_argument("--data-root", default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"), type=Path)
    parser.add_argument(
        "--config-manifest",
        default=REPO_ROOT / f"configs/{PROTOCOL_NAME}/split_manifest.json",
        type=Path,
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--out-root", default=None, type=Path)
    parser.add_argument("--source-version-root", default=None, type=Path)
    parser.add_argument("--skip-file-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    root = args.data_root.expanduser().resolve()
    output = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else root / f"derived/protocols/{PROTOCOL_NAME}"
    )
    config_manifest = args.config_manifest.expanduser().resolve()
    configured = json.loads(config_manifest.read_text(encoding="utf-8"))
    generated = generate_family_split(seed=args.seed)
    declared = {
        "train": configured["train_families"],
        "validation": configured["validation_families"],
        "test": configured["test_families"],
    }
    if generated != declared:
        raise SystemExit("checked-in family manifest does not match deterministic RNG generation")
    if output == root / "derived/indices/full_50x200" or "benchmark_v2_50family/splits" in str(output):
        raise SystemExit("secondary protocol output may not target the primary 40/5/5 namespace")
    before = primary_artifact_hashes(REPO_ROOT, root)
    print(f"Protocol: {PROTOCOL_NAME}")
    for name in ("train", "validation", "test"):
        print(f"{name} ({len(generated[name])}): {' '.join(generated[name])}")
    print("Expected package counts: train=5600 internal_val=700 familiar_test=700 heldout_val=1000 heldout_test=2000")
    if args.dry_run:
        print(f"Would write protocol indices under: {output}")
        return 0
    report = build_protocol_indices(
        root,
        output,
        split=generated,
        config_manifest=config_manifest,
        source_version_root=(
            args.source_version_root.expanduser().resolve()
            if args.source_version_root is not None
            else None
        ),
        validate_files=not args.skip_file_validation,
    )
    after = primary_artifact_hashes(REPO_ROOT, root)
    if before != after:
        raise RuntimeError("primary 40/5/5 artifacts changed while building the secondary protocol")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(f"Protocol index manifest: {output / 'protocol_index_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
