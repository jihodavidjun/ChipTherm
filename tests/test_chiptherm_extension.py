#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_extension import (
    approve_pilot,
    estimate_storage,
    generate_sample,
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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="chiptherm_extension_test_") as tmp:
        root = Path(tmp)
        _test_config_and_generation(root)
        _test_approval_gate(root)
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
        layout, power, benchmark = generate_sample(case, config["defaults"], sample_index=1, seed=123)
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


if __name__ == "__main__":
    raise SystemExit(main())
