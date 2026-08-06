#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_secondary_protocol import (  # noqa: E402
    validate_normalizer_provenance,
)
from chiptherm.ml.dataset import ChipThermDataset  # noqa: E402
from chiptherm.ml.normalization import (  # noqa: E402
    compute_direct_temperature_target_stats,
    compute_normalization_stats,
    save_normalization_stats,
)
from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponseDataset,
    compute_source_response_normalization,
    save_source_response_normalization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build auditable train-only normalizer previews for a Benchmark v2 protocol.")
    parser.add_argument("--mode", required=True, choices=["source", "residual", "direct"])
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--protocol-index-manifest", required=True, type=Path)
    parser.add_argument("--data-root", default=None, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()
    manifest = json.loads(args.protocol_index_manifest.read_text(encoding="utf-8"))
    with args.train_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    families = sorted({str(row.get("family_uid") or row.get("case_id") or "") for row in rows})
    if args.mode == "source":
        allowed = manifest["source_response_contract"]["fit_family_uids"]
        validate_normalizer_provenance(
            manifest,
            families,
            allowed_families=allowed,
            require_all=True,
        )
    else:
        validate_normalizer_provenance(manifest, families)
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "source":
        dataset = SourceResponseDataset(args.train_index, data_root=args.data_root)
        stats = compute_source_response_normalization(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        stats_path = out / "source_response_normalization.json"
        save_source_response_normalization(stats, stats_path)
    else:
        target = "temperature" if args.mode == "direct" else "residual"
        dataset = ChipThermDataset(args.train_index, target=target, return_metadata=True)
        stats = compute_normalization_stats(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        stats_path = out / f"{args.mode}_normalization.json"
        save_normalization_stats(stats, stats_path)
        if args.mode == "direct":
            direct = compute_direct_temperature_target_stats(
                dataset,
                mode="train_standard",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            (out / "direct_temperature_normalization.json").write_text(
                json.dumps(direct.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    provenance = {
        "schema_version": "benchmark_v2_protocol_normalizer_provenance/1",
        "mode": args.mode,
        "train_index": str(args.train_index),
        "row_count": len(rows),
        "contributing_family_uids": families,
        "forbidden_family_uids": manifest["normalization_contract"]["forbidden_family_uids"],
        "statistics_file": stats_path.name,
        "validation_and_test_targets_used": False,
    }
    (out / f"{args.mode}_normalization_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.mode} train-only normalization to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
