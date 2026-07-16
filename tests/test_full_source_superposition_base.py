from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset, chiptherm_collate
from chiptherm.ml.models import build_model
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input
from chiptherm.ml.source_response_dataset import SourceResponseNormalizationStats
from scripts.build_full_source_superposition_base import (
    canonical_source_paths,
    infer_package_maps,
    output_row,
    sidecar_path,
    valid_existing_map,
    write_index,
)
from scripts.build_source_superposition_extension_splits import normalize_row_paths


def main() -> None:
    test_dataset_uses_source_base_without_mutating_prediction_path()
    test_write_index_preserves_order_and_columns()
    test_segment_sum_and_ambient_once()
    test_resume_rejects_stale_checkpoint()
    test_canonical_source_paths_use_explicit_extension_paths()
    test_extension_source_rows_keep_compatibility_physics_columns()
    test_source_superposition_extension_row_with_blank_compatibility_paths_collates()
    print("full source-superposition base tests passed")


def test_dataset_uses_source_base_without_mutating_prediction_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        x = np.zeros((33, 64, 64), dtype=np.float32)
        y = np.full((64, 64), 10.0, dtype=np.float32)
        old_physics = np.full((64, 64), 1.0, dtype=np.float32)
        source_base = np.full((64, 64), 7.0, dtype=np.float32)
        for name, array in {
            "x.npy": x,
            "y.npy": y,
            "old_physics.npy": old_physics,
            "source_base.npy": source_base,
        }.items():
            np.save(root / name, array)
        index = root / "index.csv"
        write_csv(
            index,
            [
                {
                    "sample_uid": "sample_a",
                    "original_sample_uid": "case01_sample_a",
                    "case_id": "case01",
                    "dataset_source": "synthetic",
                    "split": "test",
                    "x_path": str(root / "x.npy"),
                    "y_path": str(root / "y.npy"),
                    "prediction_path": str(root / "old_physics.npy"),
                    "residual_path": str(root / "missing_residual.npy"),
                    "source_superposition_base_path": str(root / "source_base.npy"),
                    "source_base_mode": "source_superposition_v1",
                    "hotspot_runtime_s": "",
                    "physics_runtime_s": "",
                    "num_chiplets": "1",
                    "total_power_W": "1.0",
                    "mean_temperature_K": "10.0",
                    "max_temperature_K": "10.0",
                }
            ],
        )
        dataset = ChipThermDataset(index, target="residual", return_graph=False)
        sample = dataset[0]
        assert torch.allclose(sample["physics"], torch.full((64, 64), 7.0))
        assert torch.allclose(sample["residual"], torch.full((64, 64), 3.0))
        assert sample["metadata"]["prediction_path"] == str(root / "old_physics.npy")
        assert sample["metadata"]["effective_prediction_path"] == str(root / "source_base.npy")


def test_write_index_preserves_order_and_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        canonical = [
            canonical_row("uid_a", root),
            canonical_row("uid_b", root),
        ]
        generated = [
            output_row(canonical[0], root / "a.npy", root / "a_res.npy", checkpoint_identity(), "generated"),
            output_row(canonical[1], root / "b.npy", root / "b_res.npy", checkpoint_identity(), "generated"),
        ]
        path = root / "out.csv"
        write_index(path, canonical, generated)
        with path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
            fields = reader.fieldnames or []
        assert [row["sample_uid"] for row in rows] == ["uid_a", "uid_b"]
        assert fields[: len(canonical[0].keys())] == list(canonical[0].keys())
        assert rows[0]["source_superposition_base_path"].endswith("a.npy")


def test_segment_sum_and_ambient_once() -> None:
    class ZeroModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 64, 64), dtype=x.dtype, device=x.device)

    stats = SourceResponseNormalizationStats(
        schema_version=1,
        channel_names=tuple(f"c{i}" for i in range(17)),
        channel_means=tuple(0.0 for _ in range(17)),
        channel_stds=tuple(1.0 for _ in range(17)),
        normalized_channel_indices=tuple(),
        target_unit_mean_K_per_W=2.0,
        target_unit_std_K_per_W=1.0,
        source_power_min_W=1.0,
        source_power_p01_W=1.0,
        source_power_p05_W=1.0,
        source_power_p50_W=1.0,
        source_power_p95_W=1.0,
        source_power_max_W=1.0,
        target_rise_abs_max_K=1.0,
        target_unit_abs_max_K_per_W=1.0,
        power_floor_W=1.0e-6,
        num_sources=2,
    )
    package = {
        "row": {"sample_uid": "p0"},
        "source_inputs": [np.zeros((17, 64, 64), dtype=np.float32), np.zeros((17, 64, 64), dtype=np.float32)],
        "source_powers": np.asarray([3.0, 4.0], dtype=np.float32),
        "ambient_K": 318.15,
        "num_sources": 2,
    }
    base = infer_package_maps([package], ZeroModel(), stats, source_batch_size=4, device=torch.device("cpu"))[0]
    expected = 318.15 + 2.0 * 3.0 + 2.0 * 4.0
    assert np.allclose(base, expected)


def test_resume_rejects_stale_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        map_path = root / "map.npy"
        np.save(map_path, np.ones((64, 64), dtype=np.float32))
        sidecar_path(map_path).write_text(
            json.dumps({"sample_uid": "uid_a", "case_id": "case01", "source_checkpoint_sha256": "old"}) + "\n",
            encoding="utf-8",
        )
        row = {"sample_uid": "uid_a", "case_id": "case01"}
        assert not valid_existing_map(map_path, sidecar_path(map_path), row, {"sha256": "new"})
        assert valid_existing_map(map_path, sidecar_path(map_path), row, {"sha256": "old"})


def test_canonical_source_paths_use_explicit_extension_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source_dir = root / "extension_source"
        source_dir.mkdir()
        for name in ("layout.json", "power.yaml", "package.yaml", "hotspot.yaml"):
            (source_dir / name).write_text("{}\n", encoding="utf-8")
        x = root / "x.npy"
        y = root / "y.npy"
        g = root / "graph.npz"
        np.save(x, np.zeros((13, 64, 64), dtype=np.float32))
        np.save(y, np.zeros((64, 64), dtype=np.float32))
        np.savez(g, node_features=np.zeros((1, 1), dtype=np.float32))
        row = {
            "sample_uid": "benchmark_extension_v1_case11_sample_000001",
            "case_id": "case11",
            "dataset_source": "benchmark_extension_v1_full",
            "source_dir": str(source_dir),
            "x_path": str(x),
            "y_path": str(y),
            "graph_path": str(g),
        }
        paths = canonical_source_paths(row)
        assert paths["source_dir"] == source_dir
        assert paths["layout"] == source_dir / "layout.json"
        assert paths["power"] == source_dir / "power.yaml"


def test_extension_source_rows_keep_compatibility_physics_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        row = {
            "sample_uid": "benchmark_extension_v1_case11_sample_000001",
            "case_id": "case11",
            "dataset_source": "benchmark_extension_v1_full",
            "split": "train",
            "x_path": str(root / "x.npy"),
            "y_path": str(root / "y.npy"),
            "graph_path": str(root / "graph.npz"),
            "num_chiplets": "8",
        }
        result = output_row(
            row,
            root / "base.npy",
            root / "residual.npy",
            {"path": "ckpt.pt", "sha256": "abc", "model_config": {"architecture": "source_response_operator_v1"}},
            "generated",
        )
        assert "prediction_path" in result
        assert "residual_path" in result
        assert result["prediction_path"] == ""
        assert result["residual_path"] == ""
        assert result["source_base_mode"] == "source_superposition_v1"

        normalized = normalize_row_paths(result)
        assert "prediction_path" in normalized
        assert "residual_path" in normalized
        assert normalized["prediction_path"] == ""
        assert normalized["residual_path"] == ""
        assert normalized["source_superposition_base_path"].endswith("base.npy")


def test_source_superposition_extension_row_with_blank_compatibility_paths_collates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        x = np.zeros((33, 64, 64), dtype=np.float32)
        x[0] = 1.0
        y = np.full((64, 64), 320.0, dtype=np.float32)
        base = np.full((64, 64), 319.0, dtype=np.float32)
        residual = y - base
        graph_path = root / "graph.npz"
        np.save(root / "x.npy", x)
        np.save(root / "y.npy", y)
        np.save(root / "source_base.npy", base)
        np.save(root / "source_residual.npy", residual)
        np.savez_compressed(
            graph_path,
            node_features=np.zeros((2, 24), dtype=np.float32),
            edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            edge_features=np.zeros((2, 15), dtype=np.float32),
            chiplet_rects=np.asarray([[0, 0, 1, 1], [2, 2, 1, 1]], dtype=np.float32),
            package_size=np.asarray([64, 64], dtype=np.float32),
        )
        metadata_names = [
            "package_width_mm",
            "package_height_mm",
            "cell_size_x_mm",
            "cell_size_y_mm",
            "total_power_W",
            "chiplet_count",
            "occupied_fraction",
            "whitespace_fraction",
            "mean_power_density_W_per_mm2",
            "max_power_density_W_per_mm2",
            "mean_chiplet_area_mm2",
            "max_chiplet_area_mm2",
            "mean_chiplet_aspect_ratio",
            "spreader_side_m",
            "sink_side_m",
        ]
        (root / "metadata_manifest.json").write_text(json.dumps({"active_features": metadata_names}) + "\n", encoding="utf-8")
        with (root / "metadata_features.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["sample_uid", *metadata_names])
            writer.writeheader()
            for idx in range(4):
                writer.writerow({"sample_uid": f"uid_{idx}", **{name: "1.0" for name in metadata_names}})
        rows = []
        for idx in range(4):
            rows.append(
                {
                    "sample_uid": f"uid_{idx}",
                    "original_sample_uid": f"uid_{idx}",
                    "case_id": "case11",
                    "dataset_source": "synthetic_extension",
                    "split": "test",
                    "x_path": str(root / "x.npy"),
                    "y_path": str(root / "y.npy"),
                    "prediction_path": "",
                    "residual_path": "",
                    "source_superposition_base_path": str(root / "source_base.npy"),
                    "source_superposition_residual_path": str(root / "source_residual.npy"),
                    "source_base_mode": "source_superposition_v1",
                    "graph_path": str(graph_path),
                    "num_chiplets": "2",
                    "total_power_W": "2.0",
                    "hotspot_runtime_s": "",
                    "physics_runtime_s": "",
                }
            )
        index = root / "index.csv"
        write_csv(index, rows)

        dataset = ChipThermDataset(index, target="residual", return_metadata=True, return_graph=True)
        sample = dataset[0]
        assert torch.allclose(sample["physics"], torch.full((64, 64), 319.0))
        assert "physics_v1" not in sample
        assert sample["metadata_vector"].shape[0] == 15
        assert torch.isfinite(sample["metadata_vector"]).all()
        assert np.isfinite(float(sample["metadata"]["hotspot_runtime_s"]))
        assert np.isfinite(float(sample["metadata"]["physics_runtime_s"]))
        assert sample["graph"]["node_features"].shape[-1] == 24
        assert sample["graph"]["edge_features"].shape[-1] == 15
        assert_no_none(sample)

        batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=chiptherm_collate)))
        stats = NormalizationStats(
            schema_version=1,
            power_density_mean=0.0,
            power_density_std=1.0,
            physics_mean=0.0,
            physics_std=1.0,
            residual_mean=0.0,
            residual_std=1.0,
            num_samples=4,
            num_grid_cells=4 * 64 * 64,
            input_channels=33,
            context_channel_indices=tuple(range(8, 33)),
            context_channel_means=tuple(0.0 for _ in range(25)),
            context_channel_stds=tuple(1.0 for _ in range(25)),
            metadata_feature_names=tuple(metadata_names),
            metadata_means=tuple(0.0 for _ in metadata_names),
            metadata_stds=tuple(1.0 for _ in metadata_names),
        )
        model_input = build_model_input(batch["x"], batch["physics"], stats, physics_input_mode="source_superposition_v1")
        assert model_input.shape == (4, 34, 64, 64)
        assert torch.isfinite(model_input).all()
        metadata_input = build_metadata_input(batch["metadata_vector"], stats)
        assert metadata_input is not None and metadata_input.shape == (4, 15)
        assert torch.isfinite(metadata_input).all()
        assert torch.isfinite(batch["graph"]["node_features"]).all()
        assert torch.isfinite(batch["graph"]["edge_features"]).all()

        checkpoint = REPO_ROOT / "outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_gnn_seed1/checkpoints/best.pt"
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(payload["model_config"])
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            with torch.no_grad():
                output = model(model_input, metadata_input, batch["graph"])
            if isinstance(output, dict):
                output_tensor = output.get("final_temperature")
                if output_tensor is None:
                    output_tensor = output.get("prediction")
                if output_tensor is None:
                    output_tensor = next((value for value in output.values() if torch.is_tensor(value)), None)
            else:
                output_tensor = output
            assert output_tensor is not None
            assert torch.isfinite(output_tensor).all()


def assert_no_none(value: object, path: str = "sample") -> None:
    if value is None:
        raise AssertionError(f"{path} is None")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_none(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_none(item, f"{path}[{index}]")


def canonical_row(uid: str, root: Path) -> dict[str, str]:
    return {
        "sample_uid": uid,
        "original_sample_uid": f"case01_{uid}",
        "case_id": "case01",
        "dataset_source": "synthetic",
        "split": "train",
        "x_path": str(root / "x.npy"),
        "y_path": str(root / "y.npy"),
        "prediction_path": str(root / "physics.npy"),
        "residual_path": str(root / "residual.npy"),
        "hotspot_runtime_s": "",
        "physics_runtime_s": "",
        "num_chiplets": "1",
        "total_power_W": "1.0",
        "mean_temperature_K": "0.0",
        "max_temperature_K": "0.0",
        "graph_path": "",
    }


def checkpoint_identity() -> dict[str, object]:
    return {
        "path": "checkpoint.pt",
        "sha256": "abc",
        "model_config": {"architecture": "source_response_operator_v1"},
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
