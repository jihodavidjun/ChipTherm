#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_family_scaling import (  # noqa: E402
    SOURCE_VERSION,
    build_subset_indices,
    compare_train40_reuse,
    diversity_first_order,
    read_descriptor_artifacts,
    select_primary_descriptor_names,
    write_definition_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Define immutable diversity-first Benchmark v2 family-count subsets."
    )
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument("--source-version", default=SOURCE_VERSION)
    parser.add_argument(
        "--descriptor-table",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1/family_ood_analysis/family_descriptors.csv",
    )
    parser.add_argument(
        "--descriptor-summary",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1/family_ood_analysis/summary.json",
    )
    parser.add_argument(
        "--canonical-run-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        "feature_fusion_train40_source_v1_seed1",
    )
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark_v2_50family/training/package_residual_feature_fusion_v1.yaml",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help="Defaults to DATA_ROOT/derived/indices/family_count_scaling/diversity_first.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/benchmark_v2_50family/family_count_scaling_summary",
    )
    parser.add_argument(
        "--ordering-only",
        action="store_true",
        help="Write the deterministic order without touching benchmark indices.",
    )
    args = parser.parse_args()

    descriptor_table = args.descriptor_table.expanduser().resolve()
    descriptor_summary = args.descriptor_summary.expanduser().resolve()
    rows, summary = read_descriptor_artifacts(descriptor_table, descriptor_summary)
    descriptor_names, excluded = select_primary_descriptor_names(rows, summary)
    ordering = diversity_first_order(rows, descriptor_names)

    if args.ordering_only:
        print("Initial medoid:", ordering["initial_medoid_family"])
        print("Ordering:", " ".join(ordering["ordering"]))
        print("Descriptors:", len(descriptor_names))
        return 0
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    data_root = args.data_root.expanduser().resolve()
    index_root = (
        args.index_root.expanduser().resolve()
        if args.index_root is not None
        else data_root / "derived/indices/family_count_scaling/diversity_first"
    )
    manifests = build_subset_indices(
        data_root=data_root,
        source_version=args.source_version,
        ordering_result=ordering,
        descriptor_table=descriptor_table,
        descriptor_summary=descriptor_summary,
        index_root=index_root,
    )
    equivalence = compare_train40_reuse(
        data_root=data_root,
        source_version=args.source_version,
        s40_manifest=manifests[40],
        canonical_run_root=args.canonical_run_root.expanduser().resolve(),
        canonical_config_path=args.canonical_config.expanduser().resolve(),
    )
    write_definition_outputs(
        output_dir=args.out_dir.expanduser().resolve(),
        ordering_result=ordering,
        excluded_descriptors=excluded,
        manifests=manifests,
        equivalence=equivalence,
        base_training_config=yaml.safe_load(
            args.canonical_config.expanduser().resolve().read_text(encoding="utf-8")
        ),
    )
    print("Initial medoid:", ordering["initial_medoid_family"])
    print("Ordering:", " ".join(ordering["ordering"]))
    for count in (10, 20, 30, 40):
        print(f"S{count}:", " ".join(ordering["ordering"][:count]))
    print("Canonical train40 reusable:", equivalence["canonical_train40_reusable"])
    if not equivalence["canonical_train40_reusable"]:
        failures = [item for item in equivalence["comparisons"] if not item["passed"]]
        raise SystemExit(f"canonical train40 equivalence failed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
