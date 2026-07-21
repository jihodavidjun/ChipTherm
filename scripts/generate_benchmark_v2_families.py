#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chiptherm.benchmark_v2 import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_FAMILY_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROPOSAL,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TABLE_PATH,
    instantiate_all_families,
    load_design_proposal,
    write_stage1_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Instantiate the 50 fixed ChipTherm benchmark-v2 family structures. This command never runs HotSpot."
    )
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--family-dir", type=Path, default=DEFAULT_FAMILY_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--table-markdown-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()

    proposal = load_design_proposal(args.proposal)
    families = instantiate_all_families(proposal, base_seed=args.seed)
    validation = write_stage1_artifacts(
        families,
        proposal_path=args.proposal,
        family_dir=args.family_dir,
        split_dir=args.split_dir,
        output_dir=args.out_dir,
        review_path=args.review_path,
        table_markdown_path=args.table_markdown_path,
        base_seed=args.seed,
    )
    print("ChipTherm benchmark v2 Stage 1 generation")
    print(f"Families: {validation['family_count']}")
    print(f"Primary split: {validation['split_counts']}")
    print(f"Rotational groups: {validation['rotational_group_counts']}")
    print(f"HotSpot runs: {validation['hotspot_runs']}")
    print(f"Recommendation: {validation['recommendation']}")
    if validation["problems"]:
        for problem in validation["problems"]:
            print(f"ERROR: {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
