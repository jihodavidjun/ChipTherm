from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset
from chiptherm.ml.source_response_dataset import SourceResponseNormalizationStats
from scripts.build_full_source_superposition_base import (
    infer_package_maps,
    output_row,
    sidecar_path,
    valid_existing_map,
    write_index,
)


def main() -> None:
    test_dataset_uses_source_base_without_mutating_prediction_path()
    test_write_index_preserves_order_and_columns()
    test_segment_sum_and_ambient_once()
    test_resume_rejects_stale_checkpoint()
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
