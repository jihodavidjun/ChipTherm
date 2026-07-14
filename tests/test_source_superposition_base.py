#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.normalization import NormalizationStats, build_model_input  # noqa: E402
from chiptherm.ml.models import build_model  # noqa: E402
from scripts.build_source_superposition_base_maps import coverage_report, is_complete_source_group  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_indices(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    canonical: dict[str, Path] = {}
    source: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        canonical_rows = [
            {"sample_uid": f"{split}_pkg0", "case_id": "case01", "split": split},
            {"sample_uid": f"{split}_pkg1", "case_id": "case01", "split": split},
        ]
        source_rows = [
            {"original_sample_uid": f"{split}_pkg0", "num_chiplets": "2", "source_index": "0"},
            {"original_sample_uid": f"{split}_pkg0", "num_chiplets": "2", "source_index": "1"},
            {"original_sample_uid": f"{split}_pkg1", "num_chiplets": "3", "source_index": "0"},
        ]
        c_path = root / f"{split}_canonical.csv"
        s_path = root / f"{split}_source.csv"
        write_csv(c_path, canonical_rows)
        write_csv(s_path, source_rows)
        canonical[split] = c_path
        source[split] = s_path
    return canonical, source


def test_coverage_detects_complete_and_incomplete_source_groups(tmp_path: Path) -> None:
    canonical, source = make_indices(tmp_path)
    report = coverage_report(canonical, source)
    for split in ("train", "val", "test"):
        assert report["splits"][split]["covered_sample_uids"] == [f"{split}_pkg0"]
        assert report["splits"][split]["missing_sample_uids"] == [f"{split}_pkg1"]


def test_complete_source_group_validation() -> None:
    assert is_complete_source_group([{"num_chiplets": "2"}, {"num_chiplets": "2"}])
    assert not is_complete_source_group([{"num_chiplets": "2"}])


def test_source_superposition_mode_appends_base_channel_like_v1() -> None:
    stats = NormalizationStats(
        schema_version=1,
        power_density_mean=0.0,
        power_density_std=1.0,
        physics_mean=300.0,
        physics_std=10.0,
        residual_mean=0.0,
        residual_std=1.0,
        num_samples=1,
        num_grid_cells=4,
        input_channels=8,
    )
    x = torch.zeros(1, 8, 2, 2)
    base = torch.ones(1, 2, 2) * 320.0
    model_input = build_model_input(x, base, stats, physics_input_mode="source_superposition_v1")
    assert model_input.shape == (1, 9, 2, 2)
    assert torch.allclose(model_input[:, -1], torch.ones(1, 2, 2) * 2.0)


def test_exact_synthetic_base_zero_residual_reconstructs_target() -> None:
    target = np.ones((4, 4), dtype=np.float32) * 333.0
    source_base = target.copy()
    residual = target - source_base
    assert np.allclose(residual, 0.0)
    assert np.allclose(source_base + residual, target)


def test_checkpoint_roundtrip_with_source_superposition_mode(tmp_path: Path) -> None:
    config = {
        "architecture": "miniunet_refine_decomposed",
        "input_channels": 9,
        "output_channels": 1,
        "base_channels": 4,
        "depth": 2,
        "refine_channels": 4,
        "refine_blocks": 1,
        "refinement_channel_indices": [0, 1, 2],
        "refinement_channel_names": ["power_density_W_per_mm2", "occupancy_mask", "CPU_mask"],
        "physics_input_mode": "source_superposition_v1",
    }
    model = build_model(config)
    x = torch.randn(1, 9, 8, 8)
    y = model(x)
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    loaded = build_model(ckpt["model_config"])
    loaded.load_state_dict(ckpt["model_state_dict"])
    loaded_y = loaded(x)
    assert torch.allclose(y["mean_rise"], loaded_y["mean_rise"])
    assert torch.allclose(y["centered_field"], loaded_y["centered_field"])


def main() -> int:
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
    print("source superposition base tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
