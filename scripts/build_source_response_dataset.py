#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.encoder import active_power_map
from scripts.run_superposition_diagnostic import (
    isolated_power_map,
    load_json,
    load_yaml,
    modified_power_yaml,
    run_power_case,
    safe_name,
    resolve_index_path,
    source_dir_for_row,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a derived ChipTherm source-response dataset.")
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--val-index", required=True, type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        default=None,
        type=Path,
        help="Resolve relative input-index paths against this declared benchmark root.",
    )
    parser.add_argument("--out-root", default=REPO_ROOT / "data/runs/derived/source_response_v1", type=Path)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--samples-per-case", default=None, type=int)
    parser.add_argument("--sample-uids", nargs="*", default=None)
    parser.add_argument("--max-sources-per-sample", default=None, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--hotspot-home", default=None, type=Path)
    parser.add_argument("--config-template", default=REPO_ROOT / "configs/hotspot_base.config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else None
    assert_safe_derived_root(out_root)
    if args.overwrite and out_root.exists() and not args.dry_run:
        shutil.rmtree(out_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    split_inputs = {
        "train": read_rows(args.train_index),
        "val": read_rows(args.val_index),
        "test": read_rows(args.test_index),
    }
    selected_by_split = {
        split: select_split_rows(rows, cases=args.cases, samples_per_case=args.samples_per_case, sample_uids=args.sample_uids, seed=args.seed + offset)
        for offset, (split, rows) in enumerate(split_inputs.items())
    }
    plan = plan_generation(
        selected_by_split,
        max_sources_per_sample=args.max_sources_per_sample,
        data_root=data_root,
    )
    print_plan(plan)
    if args.dry_run:
        return 0

    start = time.perf_counter()
    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    failures: list[dict[str, Any]] = []
    for split, rows in selected_by_split.items():
        for row in rows:
            try:
                records_by_split[split].extend(
                    process_sample(
                        row,
                        split=split,
                        out_root=out_root,
                        max_sources_per_sample=args.max_sources_per_sample,
                        hotspot_home=args.hotspot_home,
                        config_template=args.config_template,
                        resume=args.resume,
                        data_root=data_root,
                    )
                )
            except Exception as exc:
                failures.append({"split": split, "sample_uid": row.get("sample_uid"), "case_id": row.get("case_id"), "error": str(exc), "type": type(exc).__name__})
                print(f"FAILED {split} {row.get('sample_uid')}: {exc}", file=sys.stderr)
                raise

    all_records: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        all_records.extend(records)
        write_csv(out_root / f"{split}_index.csv", records)
    write_csv(out_root / "combined_source_index.csv", all_records)
    write_jsonl(out_root / "combined_source_index.jsonl", all_records)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": repo_relative(out_root),
        "target_definition": "source_temperature_rise_K = isolated_source_temperature_K - ambient_K",
        "split_inheritance": "source rows inherit split from their original package sample",
        "plan": plan,
        "actual_source_rows": {split: len(records) for split, records in records_by_split.items()},
        "actual_total_source_rows": len(all_records),
        "failures": failures,
        "runtime_s": time.perf_counter() - start,
    }
    (out_root / "source_response_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_root, manifest)
    print("Source-response dataset build complete")
    print(f"Output: {out_root}")
    print(f"Source rows: {len(all_records)}")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def select_split_rows(
    rows: list[dict[str, str]],
    *,
    cases: list[str] | None,
    samples_per_case: int | None,
    sample_uids: list[str] | None,
    seed: int,
) -> list[dict[str, str]]:
    if sample_uids:
        wanted = set(sample_uids)
        return [row for row in rows if row["sample_uid"] in wanted or row.get("original_sample_uid") in wanted]
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"]].append(row)
    selected_cases = cases or sorted(rows_by_case)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for case_id in selected_cases:
        candidates = sorted(rows_by_case.get(case_id, []), key=lambda item: item["sample_uid"])
        if not candidates:
            continue
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_case] if samples_per_case is not None else candidates)
    return selected


def plan_generation(
    selected_by_split: dict[str, list[dict[str, str]]],
    *,
    max_sources_per_sample: int | None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    split_counts: dict[str, Any] = {}
    total_runs = 0
    total_bytes = 0
    case_counts: Counter[str] = Counter()
    for split, rows in selected_by_split.items():
        runs = 0
        for row in rows:
            layout = load_json(source_dir_for_row(row, data_root=data_root) / "layout.json")
            count = len(layout.get("chiplets", []))
            if max_sources_per_sample is not None:
                count = min(count, int(max_sources_per_sample))
            runs += count
            case_counts[row["case_id"]] += 1
        split_counts[split] = {"original_samples": len(rows), "source_runs": runs}
        total_runs += runs
        total_bytes += runs * 64 * 64 * 4
    return {
        "splits": split_counts,
        "total_original_samples": sum(len(rows) for rows in selected_by_split.values()),
        "total_isolated_hotspot_runs": total_runs,
        "estimated_target_storage_bytes": total_bytes,
        "estimated_target_storage_MB": total_bytes / 1.0e6,
        "selected_cases": dict(sorted(case_counts.items())),
    }


def print_plan(plan: dict[str, Any]) -> None:
    print("Source-response dataset plan:")
    for split, payload in plan["splits"].items():
        print(f"  {split}: original_samples={payload['original_samples']} source_runs={payload['source_runs']}")
    print(f"Total isolated HotSpot runs: {plan['total_isolated_hotspot_runs']}")
    print(f"Estimated target storage: {plan['estimated_target_storage_MB']:.3f} MB")


def process_sample(
    row: dict[str, str],
    *,
    split: str,
    out_root: Path,
    max_sources_per_sample: int | None,
    hotspot_home: Path | None,
    config_template: Path,
    resume: bool,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    source_dir = source_dir_for_row(row, data_root=data_root)
    layout = load_json(source_dir / "layout.json")
    power_yaml = load_yaml(source_dir / "power.yaml")
    package = load_yaml(source_dir / "package.yaml")
    chiplets = list(layout.get("chiplets", []))
    powers = active_power_map(power_yaml)
    ambient_K = float(package["ambient_K"])
    full_temperature_path = resolve_index_path(
        row["y_path"],
        data_root=data_root,
        field_name="y_path",
        must_exist=True,
    )
    full_temperature = np.load(full_temperature_path).astype(np.float32, copy=False)
    limit = len(chiplets) if max_sources_per_sample is None else min(len(chiplets), int(max_sources_per_sample))
    records: list[dict[str, Any]] = []
    for source_index, chiplet in enumerate(chiplets[:limit]):
        source_name = str(chiplet["name"])
        source_power = float(powers[source_name])
        source_uid = f"{row['sample_uid']}__src{source_index:03d}_{safe_name(source_name)}"
        run_dir = out_root / "hotspot_runs" / split / row["case_id"] / row["sample_uid"] / f"source_{source_index:03d}_{safe_name(source_name)}"
        target_path = out_root / "targets" / split / row["case_id"] / f"{source_uid}_rise.npy"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and target_path.exists():
            rise = np.load(target_path)
            if rise.shape == full_temperature.shape and np.isfinite(rise).all():
                records.append(make_record(row, split, source_uid, source_index, chiplet, source_power, ambient_K, target_path, layout, len(chiplets), limit, data_root=data_root))
                continue
        run = run_power_case(
            source_dir=source_dir,
            output_run_dir=run_dir,
            modified_power=modified_power_yaml(power_yaml, isolated_power_map(powers, source_name)),
            hotspot_home=hotspot_home,
            config_template=config_template,
            resume=resume,
            overwrite=False,
            expected_shape=full_temperature.shape,
        )
        isolated_temperature = np.load(run.temperature_path).astype(np.float32, copy=False)
        rise = isolated_temperature - np.float32(ambient_K)
        np.save(target_path, rise.astype(np.float32, copy=False))
        records.append(make_record(row, split, source_uid, source_index, chiplet, source_power, ambient_K, target_path, layout, len(chiplets), limit, run.runtime_s, data_root=data_root))
    return records


def make_record(
    row: dict[str, str],
    split: str,
    source_uid: str,
    source_index: int,
    chiplet: dict[str, Any],
    source_power: float,
    ambient_K: float,
    target_path: Path,
    layout: dict[str, Any],
    num_chiplets: int,
    num_sources_included: int,
    runtime_s: float | None = None,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    size = chiplet["size"]
    area = float(size["width"]) * float(size["height"])
    source_dir = source_dir_for_row(row, data_root=data_root)
    return {
        "source_response_uid": source_uid,
        "original_sample_uid": row["sample_uid"],
        "canonical_original_sample_uid": row.get("original_sample_uid", ""),
        "case_id": row["case_id"],
        "split": split,
        "dataset_source": row["dataset_source"],
        "original_x_path": row["x_path"],
        "original_y_path": row["y_path"],
        "full_temperature_path": row["y_path"],
        "layout_path": portable_path(source_dir / "layout.json", data_root=data_root),
        "power_path": portable_path(source_dir / "power.yaml", data_root=data_root),
        "package_path": portable_path(source_dir / "package.yaml", data_root=data_root),
        "hotspot_path": portable_path(source_dir / "hotspot.yaml", data_root=data_root),
        "source_index": source_index,
        "source_name": str(chiplet["name"]),
        "source_type": str(chiplet.get("type", "")),
        "source_power_W": source_power,
        "source_area_mm2": area,
        "source_power_density_W_per_mm2": source_power / max(area, 1.0e-12),
        "ambient_K": ambient_K,
        "target_rise_path": portable_path(target_path, data_root=data_root),
        "num_chiplets": num_chiplets,
        "num_sources_included": num_sources_included,
        "source_response_runtime_s": runtime_s,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    text = f"""# ChipTherm Source Response v1

Derived source-isolated dataset. Targets are temperature rise above ambient for
one active source chiplet at a time.

- Total source rows: {manifest['actual_total_source_rows']}
- Target definition: `{manifest['target_definition']}`
- Split inheritance: `{manifest['split_inheritance']}`
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def portable_path(path: Path, *, data_root: Path | None = None) -> str:
    resolved = path.resolve()
    if data_root is not None:
        root = Path(data_root).resolve()
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            pass
    return repo_relative(resolved)


def assert_safe_derived_root(out_root: Path) -> None:
    benchmarks = (REPO_ROOT / "data/runs/benchmarks").resolve()
    try:
        out_root.resolve().relative_to(benchmarks)
    except ValueError:
        return
    raise ValueError(f"refusing to write derived source-response dataset inside canonical benchmark root: {out_root}")


if __name__ == "__main__":
    raise SystemExit(main())
