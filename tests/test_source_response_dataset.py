#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiptherm.ml.source_response_dataset import (  # noqa: E402
    SourceResponsePackageDataset,
    SourceResponseDataset,
    build_source_input,
    compute_source_response_normalization,
    source_response_package_collate,
)
from chiptherm.ml.encoder import active_power_map  # noqa: E402
from scripts.build_source_response_dataset import select_split_rows  # noqa: E402
from scripts.run_superposition_diagnostic import isolated_power_map, modified_power_yaml  # noqa: E402


def make_fixture(root: Path) -> Path:
    layout = {
        "package": {"size": {"width": 4.0, "height": 4.0}},
        "chiplets": [
            {"name": "CPU0", "type": "CPU", "position": {"x": 0.0, "y": 0.0}, "size": {"width": 2.0, "height": 4.0}},
            {"name": "GPU0", "type": "GPU", "position": {"x": 2.0, "y": 0.0}, "size": {"width": 2.0, "height": 4.0}},
        ],
    }
    layout_path = root / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    x = np.zeros((8, 4, 4), dtype=np.float32)
    x[1, :, :] = 1.0
    x[2, :, :2] = 1.0
    x[3, :, 2:] = 1.0
    coords = (np.arange(4, dtype=np.float32) + 0.5) / 4.0
    x[6] = coords.reshape(1, 4)
    x[7] = coords.reshape(4, 1)
    x_path = root / "x.npy"
    y_path = root / "y.npy"
    target_path = root / "rise.npy"
    np.save(x_path, x)
    np.save(y_path, np.full((4, 4), 320.0, dtype=np.float32))
    np.save(target_path, np.full((4, 4), 2.0, dtype=np.float32))
    index_path = root / "source_index.csv"
    row = {
        "source_response_uid": "sample0__src001_GPU0",
        "original_sample_uid": "sample0",
        "case_id": "caseX",
        "split": "train",
        "dataset_source": "synthetic",
        "original_x_path": str(x_path),
        "original_y_path": str(y_path),
        "full_temperature_path": str(y_path),
        "layout_path": str(layout_path),
        "source_index": "1",
        "source_name": "GPU0",
        "source_type": "GPU",
        "source_power_W": "4.0",
        "source_area_mm2": "8.0",
        "source_power_density_W_per_mm2": "0.5",
        "ambient_K": "318.0",
        "target_rise_path": str(target_path),
        "num_chiplets": "2",
        "num_sources_included": "2",
    }
    with index_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return index_path


def make_group_fixture(root: Path) -> Path:
    layout = {
        "package": {"size": {"width": 4.0, "height": 4.0}},
        "chiplets": [
            {"name": "CPU0", "type": "CPU", "position": {"x": 0.0, "y": 0.0}, "size": {"width": 2.0, "height": 2.0}},
            {"name": "GPU0", "type": "GPU", "position": {"x": 2.0, "y": 0.0}, "size": {"width": 2.0, "height": 2.0}},
            {"name": "MEM0", "type": "memory", "position": {"x": 0.0, "y": 2.0}, "size": {"width": 4.0, "height": 2.0}},
        ],
    }
    layout_path = root / "layout_group.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    x = np.zeros((8, 4, 4), dtype=np.float32)
    x[1] = 1.0
    x[2, :2, :2] = 1.0
    x[3, :2, 2:] = 1.0
    x[4, 2:, :] = 1.0
    coords = (np.arange(4, dtype=np.float32) + 0.5) / 4.0
    x[6] = coords.reshape(1, 4)
    x[7] = coords.reshape(4, 1)
    rows = []
    for package_index, source_count in enumerate((2, 3)):
        uid = f"pkg{package_index}"
        x_path = root / f"{uid}_x.npy"
        y_path = root / f"{uid}_y.npy"
        np.save(x_path, x)
        source_rises = []
        for source_index in range(source_count):
            rise = np.full((4, 4), float(source_index + 1), dtype=np.float32)
            target_path = root / f"{uid}_src{source_index}_rise.npy"
            np.save(target_path, rise)
            source_rises.append(rise)
            chiplet = layout["chiplets"][source_index]
            area = float(chiplet["size"]["width"]) * float(chiplet["size"]["height"])
            rows.append(
                {
                    "source_response_uid": f"{uid}__src{source_index:03d}_{chiplet['name']}",
                    "original_sample_uid": uid,
                    "case_id": "caseX",
                    "split": "train",
                    "dataset_source": "synthetic",
                    "original_x_path": str(x_path),
                    "original_y_path": str(y_path),
                    "full_temperature_path": str(y_path),
                    "layout_path": str(layout_path),
                    "source_index": str(source_index),
                    "source_name": chiplet["name"],
                    "source_type": chiplet["type"],
                    "source_power_W": str(float(source_index + 2)),
                    "source_area_mm2": str(area),
                    "source_power_density_W_per_mm2": str(float(source_index + 2) / area),
                    "ambient_K": "318.0",
                    "target_rise_path": str(target_path),
                    "num_chiplets": str(source_count),
                    "num_sources_included": str(source_count),
                }
            )
        np.save(y_path, np.float32(318.0) + np.sum(source_rises, axis=0))
    index_path = root / "group_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def test_source_input_preserves_geometry_and_only_source_power(tmp_path: Path) -> None:
    index_path = make_fixture(tmp_path)
    dataset = SourceResponseDataset(index_path)
    sample = dataset[0]
    x = sample["x"].numpy()
    assert x.shape == (17, 4, 4)
    assert np.all(x[0] == 1.0)
    assert np.all(x[1, :, :2] == 1.0)
    assert np.all(x[2, :, 2:] == 1.0)
    assert np.all(x[7, :, :2] == 0.0)
    assert np.all(x[7, :, 2:] == 1.0)
    assert np.all(x[8, :, :2] == 0.0)
    assert np.all(x[8, :, 2:] == 0.5)


def test_temperature_rise_and_per_watt_target(tmp_path: Path) -> None:
    index_path = make_fixture(tmp_path)
    dataset = SourceResponseDataset(index_path)
    sample = dataset[0]
    assert np.allclose(sample["target_rise"].numpy(), 2.0)
    assert np.allclose(sample["target_unit"].numpy(), 0.5)


def test_low_power_guard(tmp_path: Path) -> None:
    index_path = make_fixture(tmp_path)
    rows = list(csv.DictReader(index_path.open()))
    rows[0]["source_power_W"] = "0.0"
    with index_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset = SourceResponseDataset(index_path, power_floor_W=2.0)
    assert np.allclose(dataset[0]["target_unit"].numpy(), 1.0)


def test_normalization_stats(tmp_path: Path) -> None:
    index_path = make_fixture(tmp_path)
    stats = compute_source_response_normalization(SourceResponseDataset(index_path), batch_size=1)
    assert stats.num_sources == 1
    assert "source_radius_mm" in stats.channel_names
    assert stats.source_power_min_W == 4.0


def test_package_dataset_keeps_complete_variable_source_groups_together(tmp_path: Path) -> None:
    index_path = make_group_fixture(tmp_path)
    dataset = SourceResponsePackageDataset(index_path)
    assert len(dataset) == 2
    assert [dataset[i]["num_sources"] for i in range(len(dataset))] == [2, 3]
    assert [dataset[i]["original_sample_uid"] for i in range(len(dataset))] == ["pkg0", "pkg1"]


def test_package_collate_packs_sources_and_mapping(tmp_path: Path) -> None:
    index_path = make_group_fixture(tmp_path)
    dataset = SourceResponsePackageDataset(index_path)
    batch = source_response_package_collate([dataset[0], dataset[1]])
    assert batch["x"].shape == (5, 17, 4, 4)
    assert batch["target_rise"].shape == (5, 4, 4)
    assert batch["source_to_package_index"].tolist() == [0, 0, 1, 1, 1]
    assert batch["package_full_temperature"].shape == (2, 4, 4)
    assert batch["package_original_sample_uid"] == ["pkg0", "pkg1"]


def test_package_target_loaded_from_original_y_and_ambient_added_once(tmp_path: Path) -> None:
    index_path = make_group_fixture(tmp_path)
    batch = source_response_package_collate([SourceResponsePackageDataset(index_path)[0]])
    reconstructed = batch["package_ambient_K"][0] + batch["target_rise"].sum(dim=0)
    assert np.allclose(reconstructed.numpy(), batch["package_full_temperature"][0].numpy())


def test_split_selection_independent() -> None:
    rows = [{"case_id": "case01", "sample_uid": f"train_{i}", "split": "train"} for i in range(5)]
    selected_a = select_split_rows(rows, cases=["case01"], samples_per_case=2, sample_uids=None, seed=1)
    selected_b = select_split_rows(rows, cases=["case01"], samples_per_case=2, sample_uids=None, seed=1)
    assert [row["sample_uid"] for row in selected_a] == [row["sample_uid"] for row in selected_b]
    assert all(row["split"] == "train" for row in selected_a)


def test_isolated_power_yaml_preserves_original_and_zeroes_non_sources() -> None:
    original = {
        "mode": "fixed",
        "active_workload": "nominal",
        "chiplets": {"CPU0": 10.0, "GPU0": 4.0, "MEM0": 2.0},
        "workloads": {"nominal": {"CPU0": 10.0, "GPU0": 4.0, "MEM0": 2.0}},
    }
    isolated = isolated_power_map(active_power_map(original), "GPU0")
    modified = modified_power_yaml(original, isolated)
    assert active_power_map(original) == {"CPU0": 10.0, "GPU0": 4.0, "MEM0": 2.0}
    assert modified["active_workload"] == "nominal"
    assert modified["chiplets"]["GPU0"] == 4.0
    assert modified["workloads"]["nominal"]["GPU0"] == 4.0
    for name in ("CPU0", "MEM0"):
        assert modified["chiplets"][name] == 0.0
        assert modified["workloads"]["nominal"][name] == 0.0


def main() -> int:
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
    print("source response dataset tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
