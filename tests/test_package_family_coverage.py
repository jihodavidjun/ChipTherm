from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_package_family_coverage import CoverageConfig, analyze_coverage, compute_family_descriptor


def main() -> None:
    test_descriptor_computation_one_and_two_chiplets()
    test_train_only_scaling_and_no_target_usage_and_stable_ordering()
    test_missing_metadata_is_reported()
    print("package family coverage tests passed")


def test_descriptor_computation_one_and_two_chiplets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        one = make_row(root, "case01_uid0", "case01", [(2, 2, 4, 4, 10.0, "cpu")], package=(20, 10))
        two = make_row(
            root,
            "case02_uid0",
            "case02",
            [(1, 1, 2, 2, 4.0, "gpu"), (7, 1, 2, 2, 6.0, "memory")],
            package=(20, 10),
        )
        config = CoverageConfig(pairwise_hist_bins=4)
        one_desc, warnings = compute_family_descriptor([one], split="train", config=config, metadata_rows={}, metadata_feature_names=[])
        assert warnings == []
        assert one_desc["chiplet_count"] == 1.0
        assert one_desc["pairwise_center_distance_mm_mean"] == 0.0
        assert one_desc["nearest_neighbor_distance_mm_mean"] == 0.0
        assert one_desc["type_cpu_fraction"] == 1.0
        two_desc, _ = compute_family_descriptor([two], split="train", config=config, metadata_rows={}, metadata_feature_names=[])
        assert two_desc["chiplet_count"] == 2.0
        assert np.isclose(two_desc["pairwise_center_distance_mm_mean"], 6.0)
        assert np.isclose(two_desc["nearest_neighbor_distance_mm_mean"], 6.0)
        assert np.isclose(two_desc["total_power_W_mean"], 10.0)
        assert np.isclose(two_desc["hottest_chiplet_power_fraction_mean"], 0.6)


def test_train_only_scaling_and_no_target_usage_and_stable_ordering() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            make_row(root, "case01_uid0", "case01", [(1, 1, 2, 2, 4.0, "cpu")], package=(20, 20)),
            make_row(root, "case02_uid0", "case02", [(1, 1, 3, 3, 9.0, "gpu")], package=(24, 20)),
            make_row(root, "case03_uid0", "case03", [(1, 1, 4, 4, 16.0, "npu")], package=(28, 20)),
            make_row(root, "case17_uid0", "case17", [(1, 1, 2, 2, 4.0, "io")], package=(40, 20)),
            make_row(root, "case19_uid0", "case19", [(1, 1, 2, 2, 4.0, "analog")], package=(50, 20)),
        ]
        for row in rows:
            row["y_path"] = str(root / "does_not_exist_y.npy")
        write_index(root / "train.csv", [rows[2], rows[0], rows[1]])
        write_index(root / "val.csv", [rows[3]])
        write_index(root / "test.csv", [rows[4]])
        write_metadata_sidecars(root, rows[:-1])
        out_a = root / "out_a"
        result_a = analyze_coverage(
            train_index=root / "train.csv",
            val_index=root / "val.csv",
            test_index=root / "test.csv",
            out_dir=out_a,
            config=CoverageConfig(pairwise_hist_bins=3, pca_components=2),
        )
        write_index(root / "train_reordered.csv", [rows[1], rows[2], rows[0]])
        out_b = root / "out_b"
        result_b = analyze_coverage(
            train_index=root / "train_reordered.csv",
            val_index=root / "val.csv",
            test_index=root / "test.csv",
            out_dir=out_b,
            config=CoverageConfig(pairwise_hist_bins=3, pca_components=2),
        )
        desc_a = read_csv(out_a / "family_descriptors.csv")
        desc_b = read_csv(out_b / "family_descriptors.csv")
        assert desc_a == desc_b
        summary = json.loads((out_a / "descriptor_summary.json").read_text())
        assert summary["scaler"]["fit_on"] == "train families only"
        assert set(summary["split_families"]["train"]) == {"case01", "case02", "case03"}
        assert any(item["case_id"] == "case19" and item["split"] == "test" for item in result_a["nearest_training_families"])


def test_missing_metadata_is_reported() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = [make_row(root, "case01_uid0", "case01", [(1, 1, 2, 2, 4.0, "cpu")], package=(20, 20))]
        val = [make_row(root, "case17_uid0", "case17", [(1, 1, 2, 2, 4.0, "gpu")], package=(20, 20))]
        test = [make_row(root, "case19_uid0", "case19", [(1, 1, 2, 2, 4.0, "npu")], package=(20, 20))]
        write_index(root / "train.csv", train)
        write_index(root / "val.csv", val)
        write_index(root / "test.csv", test)
        write_metadata_sidecars(root, train)
        result = analyze_coverage(
            train_index=root / "train.csv",
            val_index=root / "val.csv",
            test_index=root / "test.csv",
            out_dir=root / "out",
            config=CoverageConfig(pairwise_hist_bins=2, pca_components=2),
        )
        assert any("missing metadata sidecar row" in warning for warning in result["warnings"])


def make_row(
    root: Path,
    uid: str,
    case_id: str,
    chips: list[tuple[float, float, float, float, float, str]],
    *,
    package: tuple[float, float],
) -> dict[str, str]:
    graph_dir = root / "graphs" / case_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    base_dir = root / "base" / case_id
    base_dir.mkdir(parents=True, exist_ok=True)
    node_rows = []
    rects = []
    total_power = sum(chip[4] for chip in chips)
    total_area = sum(chip[2] * chip[3] for chip in chips)
    type_order = ("cpu", "gpu", "npu", "memory", "io", "analog", "mems", "other")
    for x, y, width, height, power, chip_type in chips:
        area = width * height
        cx = x + 0.5 * width
        cy = y + 0.5 * height
        type_features = [1.0 if name == chip_type else 0.0 for name in type_order]
        node_rows.append(
            [
                cx,
                cy,
                width,
                height,
                area,
                width / height,
                power,
                power / area,
                x,
                package[0] - (x + width),
                y,
                package[1] - (y + height),
                cx / package[0],
                cy / package[1],
                power / total_power,
                area / total_area,
                *type_features,
            ]
        )
        rects.append([x, y, width, height])
    edges = []
    edge_features = []
    for src, left in enumerate(node_rows):
        for dst, right in enumerate(node_rows):
            if src == dst:
                continue
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            distance = float(np.sqrt(dx * dx + dy * dy))
            edges.append([src, dst])
            edge_features.append([dx, dy, distance, 1.0 / max(distance, 1.0), np.log1p(distance), dy / max(distance, 1e-12), dx / max(distance, 1e-12), left[6], right[6], left[4], right[4], left[7], right[7], min(left[8:12]), min(right[8:12])])
    graph_path = graph_dir / f"{uid}_graph.npz"
    np.savez_compressed(
        graph_path,
        node_features=np.asarray(node_rows, dtype=np.float32),
        edge_index=np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64),
        edge_features=np.asarray(edge_features, dtype=np.float32) if edges else np.empty((0, 15), dtype=np.float32),
        chiplet_rects=np.asarray(rects, dtype=np.float32),
        package_size=np.asarray(package, dtype=np.float32),
    )
    base_path = base_dir / f"{uid}_base.npy"
    np.save(base_path, np.full((64, 64), 320.0 + total_power, dtype=np.float32))
    return {
        "sample_uid": uid,
        "case_id": case_id,
        "dataset_source": "synthetic",
        "split": "",
        "x_path": str(root / "not_used_x.npy"),
        "y_path": str(root / "not_used_y.npy"),
        "graph_path": str(graph_path),
        "source_superposition_base_path": str(base_path),
        "source_base_mode": "source_superposition_v1",
        "num_chiplets": str(len(chips)),
        "total_power_W": str(total_power),
    }


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_sidecars(root: Path, rows: list[dict[str, str]]) -> None:
    names = ["package_width_mm", "package_height_mm", "total_power_W"]
    (root / "metadata_manifest.json").write_text(json.dumps({"active_features": names}) + "\n", encoding="utf-8")
    with (root / "metadata_features.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_uid", *names])
        writer.writeheader()
        for row in rows:
            writer.writerow({"sample_uid": row["sample_uid"], "package_width_mm": "20", "package_height_mm": "20", "total_power_W": row["total_power_W"]})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


if __name__ == "__main__":
    main()
