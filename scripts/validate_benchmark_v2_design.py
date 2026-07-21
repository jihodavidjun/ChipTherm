#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chiptherm.benchmark_v2 import (  # noqa: E402
    DEFAULT_FAMILY_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROPOSAL,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TABLE_PATH,
    file_sha256,
    load_design_proposal,
    load_family_specs,
    validate_family_collection,
    write_stage1_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate fixed ChipTherm benchmark-v2 family structures without workloads, labels, or HotSpot."
    )
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--family-dir", type=Path, default=DEFAULT_FAMILY_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--table-markdown-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--strict", action="store_true", help="Return nonzero for machine-validation failures.")
    args = parser.parse_args()

    proposal = load_design_proposal(args.proposal)
    families = load_family_specs(args.family_dir)
    collection = validate_family_collection(families, proposal)
    manifest_path = args.family_dir.parent / "family_manifest.yaml"
    manifest_errors: list[str] = []
    base_seed = 0
    if not manifest_path.exists():
        manifest_errors.append(f"missing family manifest: {manifest_path}")
    else:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        base_seed = int(manifest.get("base_seed", 0))
        entries = {str(item.get("family_uid")): item for item in manifest.get("family_entries", [])}
        for spec in families:
            uid = spec["family_uid"]
            family_path = args.family_dir / f"{uid}.yaml"
            entry = entries.get(uid)
            if entry is None:
                manifest_errors.append(f"{uid}: absent from family manifest")
            elif entry.get("family_file_sha256") != file_sha256(family_path):
                manifest_errors.append(f"{uid}: family file hash differs from manifest")
    if collection["problems"] or manifest_errors:
        print("ChipTherm benchmark v2 Stage 1 preflight failed")
        for problem in collection["problems"] + manifest_errors:
            print(f"ERROR: {problem}")
        return 1

    validation = write_stage1_artifacts(
        families,
        proposal_path=args.proposal,
        family_dir=args.family_dir,
        split_dir=args.split_dir,
        output_dir=args.out_dir,
        review_path=args.review_path,
        table_markdown_path=args.table_markdown_path,
        base_seed=base_seed,
    )
    print("ChipTherm benchmark v2 Stage 1 validation")
    print(f"Families: {validation['family_count']}")
    print(f"Valid: {validation['passed']}")
    print(f"Primary split: {validation['split_counts']}")
    print(f"Suspicious pairs: {validation['suspicious_pair_count']}")
    print(f"Unexplained cross-split pairs: {validation['unexplained_cross_split_suspicious_pair_count']}")
    print(f"Weak coverage axes: {validation['weak_coverage_axes']}")
    print(f"Recommendation: {validation['recommendation']}")
    if validation["problems"]:
        for problem in validation["problems"]:
            print(f"ERROR: {problem}")
    return 1 if args.strict and not validation["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
