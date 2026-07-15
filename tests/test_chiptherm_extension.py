#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_extension import (
    approve_pilot,
    estimate_storage,
    generate_sample,
    generate_valid_sample,
    layout_statistics,
    load_extension_config,
    read_index,
    row_for_sample,
    select_cases,
    validate_extension_root,
    validate_sample_sources,
    verify_approval,
    write_audit_reports,
    write_indexes,
    write_sample_sources,
)
from scripts.build_chiptherm_extension_indices import split_per_case
from scripts.build_chiptherm_extension import (
    _load_active_index,
    _missing_label_uids_from_index,
    _rebase_stage_indexes,
    _repair_invalid_source,
    _schedule_decision,
    _write_hotspot_reports,
)
from scripts.rebase_chiptherm_index_paths import rebase_value


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="chiptherm_extension_test_") as tmp:
        root = Path(tmp)
        _test_config_and_generation(root)
        _test_approval_gate(root)
        _test_rebase_and_splits()
        _test_exact_two_sample_retry_bookkeeping(root)
        _test_deterministic_source_repair(root)
    print("chiptherm extension tests passed")
    return 0


def _test_config_and_generation(root: Path) -> None:
    config = load_extension_config()
    assert len(config["cases"]) == 10
    cases = select_cases(config, ["case11", "case17", "case19"])
    rows = []
    stats_rows = []
    validations = []
    for case in cases:
        layout, power, benchmark, _ = generate_valid_sample(case, config["defaults"], sample_index=1, seed=123)
        stats = layout_statistics(layout, power)
        assert stats["chiplet_count"] == case["die_count"]
        low, high = case["whitespace_range"]
        assert float(low) <= stats["whitespace_fraction"] <= float(high)
        assert stats["min_pairwise_edge_distance_mm"] >= 0.5 - 1e-5
        sample_uid = f"test_{case['case_id']}"
        sample_dir = root / "pilot" / case["case_id"] / "sample_000001"
        paths = write_sample_sources(sample_dir, sample_uid, layout, power, benchmark)
        validation = validate_sample_sources(paths["scenario_path"], case)
        assert validation["passed"], validation["problems"]
        original_layout = paths["layout_path"].read_text(encoding="utf-8")
        original_power = paths["power_path"].read_text(encoding="utf-8")
        stats["sample_uid"] = sample_uid
        stats["case_id"] = case["case_id"]
        stats["split"] = case["split_role"]
        rows.append(row_for_sample(sample_uid=sample_uid, case=case, paths=paths, statistics=stats, stage="pilot", hotspot_status="not_run"))
        stats_rows.append(stats)
        validations.append({"sample_uid": sample_uid, "passed": True, "problems": []})
        assert paths["layout_path"].read_text(encoding="utf-8") == original_layout
        assert paths["power_path"].read_text(encoding="utf-8") == original_power
    out_dir = root / "pilot"
    write_indexes(out_dir, rows)
    manifest = write_audit_reports(out_dir, rows, stats_rows, stage="pilot", validation=validations, config_hash="test")
    assert manifest["validation"]["passed"]
    train_rows = read_index(out_dir / "train_index.csv")
    val_rows = read_index(out_dir / "val_index.csv")
    test_rows = read_index(out_dir / "test_index.csv")
    assert {row["case_id"] for row in train_rows} == {"case11"}
    assert {row["case_id"] for row in val_rows} == {"case17"}
    assert {row["case_id"] for row in test_rows} == {"case19"}
    report = validate_extension_root(out_dir)
    assert report["passed"], report["problems"]
    assert report["split_cases"]["train"] == ["case11"]
    assert report["split_cases"]["val"] == ["case17"]
    assert report["split_cases"]["test"] == ["case19"]
    assert estimate_storage(10, include_hotspot_labels=False)["total_GB_for_requested_mode"] < 0.01


def _test_approval_gate(root: Path) -> None:
    pilot_root = root / "pilot"
    approval_path = pilot_root / "pilot_approval.json"
    try:
        verify_approval(pilot_root, approval_path)
    except ValueError:
        pass
    else:
        raise AssertionError("approval gate should fail before approval file exists")
    approval = approve_pilot(pilot_root, approval_path, allow_warnings=True)
    assert approval["approved"]
    verified = verify_approval(pilot_root, approval_path)
    assert verified["approved"]
    manifest_path = pilot_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["touched_for_test"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        verify_approval(pilot_root, approval_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("approval gate should fail after manifest mutation")


def _test_rebase_and_splits() -> None:
    old = "/nethome/jjun49/chiptherm_test/data/runs/benchmarks/benchmark_extension_v1/smoke/case11/sample_000001/source/layout.json"
    rebased = rebase_value(old, old_prefix=None, new_prefix="")
    assert rebased == "data/runs/benchmarks/benchmark_extension_v1/smoke/case11/sample_000001/source/layout.json"
    rows = []
    for case_id in ("case11", "case12"):
        for index in range(1, 11):
            rows.append({"sample_uid": f"{case_id}_sample_{index:06d}", "case_id": case_id, "split": "family"})
    splits = split_per_case(rows, 0.8, 0.1)
    assert len(splits["train"]) == 16
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2
    assert {row["case_id"] for row in splits["train"]} == {"case11", "case12"}


def _test_exact_two_sample_retry_bookkeeping(root: Path) -> None:
    stage = root / "retry_case" / "smoke"
    rows = []
    failed_uids = [
        "benchmark_extension_v1_case12_sample_000002",
        "benchmark_extension_v1_case20_sample_000001",
    ]
    for uid in failed_uids:
        case_id = uid.split("_sample_", 1)[0].rsplit("_", 1)[1]
        sample_num = int(uid.rsplit("_", 1)[1])
        source_dir = stage / case_id / f"sample_{sample_num:06d}" / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        abs_prefix = "/nethome/jjun49/chiptherm_test/data/runs/benchmarks/benchmark_extension_v1/smoke"
        rows.append(
            {
                "sample_uid": uid,
                "case_id": case_id,
                "split": "train" if case_id == "case12" else "test",
                "source_dir": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source",
                "scenario_path": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source/scenario.yaml",
                "layout_path": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source/layout.json",
                "power_path": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source/power.yaml",
                "package_path": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source/package.yaml",
                "hotspot_path": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source/hotspot.yaml",
                "benchmark_path": f"{abs_prefix}/{case_id}/sample_{sample_num:06d}/source/benchmark.yaml",
                "y_path": "",
            }
        )
    for name in ("all_extension_index.csv", "train_index.csv", "test_index.csv"):
        selected = rows if name == "all_extension_index.csv" else [row for row in rows if row["split"] in name]
        _write_test_csv(stage / name, selected or rows)
    changed = _rebase_stage_indexes(stage)
    assert changed > 0
    _, active_rows = _load_active_index(stage)
    assert {row["sample_uid"] for row in active_rows} == set(failed_uids)
    missing = _missing_label_uids_from_index(active_rows, stage)
    assert missing == set(failed_uids)
    for uid in failed_uids:
        case_id = uid.split("_sample_", 1)[0].rsplit("_", 1)[1]
        sample_num = int(uid.rsplit("_", 1)[1])
        sample_dir = stage / case_id / f"sample_{sample_num:06d}"
        row = next(row for row in active_rows if row["sample_uid"] == uid)
        decision = _schedule_decision(
            uid=uid,
            sample_dir=sample_dir,
            active_row=row,
            out_dir=stage,
            resume=True,
            max_retries=2,
            source_validation_passed=True,
            source_validation_problems=[],
        )
        assert decision["final_should_schedule"], decision
        assert decision["retry_count"] == 0
        assert decision["sample_directory_exists"]
        assert decision["source_directory_exists"]
        assert not decision["path_y_exists"]
        assert decision["schedule_reason"] == "scheduled because durable label is missing"
    _write_hotspot_reports(
        stage,
        [],
        workers=4,
        executable=Path("unresolved"),
        requested_uids=failed_uids,
        matched_uids=failed_uids,
        scheduled_uids=[],
        skipped_valid_uids=[],
        unresolved_reasons={uid: "unit-test not scheduled" for uid in failed_uids},
    )
    assert (stage / "hotspot_generation_report.json").exists()
    assert (stage / "hotspot_failures.csv").exists()


def _test_deterministic_source_repair(root: Path) -> None:
    config = load_extension_config()
    cases = {case["case_id"]: case for case in config["cases"]}
    case = cases["case12"]
    sample_index = 2
    sample_uid = "benchmark_extension_v1_case12_sample_000002"
    invalid_layout, invalid_power, invalid_benchmark = generate_sample(case, config["defaults"], sample_index=sample_index, seed=0)
    invalid_stats = layout_statistics(invalid_layout, invalid_power)
    assert invalid_stats["whitespace_fraction"] > float(case["whitespace_range"][1]) + 1e-6

    repaired_layout_a, repaired_power_a, _, repair_info_a = generate_valid_sample(
        case,
        config["defaults"],
        sample_index=sample_index,
        seed=0,
        max_attempts=100,
    )
    repaired_layout_b, repaired_power_b, _, repair_info_b = generate_valid_sample(
        case,
        config["defaults"],
        sample_index=sample_index,
        seed=0,
        max_attempts=100,
    )
    assert repair_info_a["attempt"] > 0
    assert repair_info_a["attempt"] == repair_info_b["attempt"]
    assert repaired_layout_a == repaired_layout_b
    assert repaired_power_a == repaired_power_b
    assert repaired_layout_a["package"]["name"] == sample_uid
    repaired_stats = layout_statistics(repaired_layout_a, repaired_power_a)
    low, high = case["whitespace_range"]
    assert float(low) <= repaired_stats["whitespace_fraction"] <= float(high)

    sample_dir = root / "repair_source" / "smoke" / "case12" / "sample_000002"
    paths = write_sample_sources(sample_dir, sample_uid, invalid_layout, invalid_power, invalid_benchmark)
    stale_label = sample_dir / "parsed/temp_layer0.npy"
    stale_label.parent.mkdir(parents=True, exist_ok=True)
    np.save(stale_label, np.zeros((64, 64), dtype=np.float32))
    assert validate_sample_sources(paths["scenario_path"], case)["passed"] is False
    valid_sibling = sample_dir.parent / "sample_000003" / "parsed/temp_layer0.npy"
    valid_sibling.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_sibling, np.ones((64, 64), dtype=np.float32))
    _repair_invalid_source(
        sample_dir=sample_dir,
        sample_uid=sample_uid,
        case=case,
        defaults=config["defaults"],
        sample_index=sample_index,
        seed=0,
        max_layout_attempts=100,
        cleanup_hotspot_workdirs=False,
    )
    assert validate_sample_sources(sample_dir / "source/scenario.yaml", case)["passed"]
    assert not stale_label.exists()
    assert valid_sibling.exists()


def _write_test_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
