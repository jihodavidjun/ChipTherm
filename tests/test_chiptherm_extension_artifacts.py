#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
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

from scripts.build_chiptherm_extension_artifacts import audit_artifact_paths, build_adapter_index, finalize_encoded_dataset, repair_artifact_indices


def main() -> int:
    source_index = REPO_ROOT / "data/runs/benchmarks/benchmark_extension_v1/full/all_extension_index.csv"
    if not source_index.exists():
        print(f"skipping extension artifact smoke test; missing {source_index}")
        return 0
    with tempfile.TemporaryDirectory(prefix="chiptherm_ext_artifact_test_") as tmp:
        root = Path(tmp)
        four_row_index = root / "extension_4row.csv"
        write_four_row_index(source_index, four_row_index)
        adapter_index = root / "adapter_index.csv"
        rows = build_adapter_index(four_row_index, adapter_index)
        assert len(rows) == 4
        encoded_root = root / "encoded_package_plus_power"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/encode_dataset.py"),
                "--index",
                str(adapter_index),
                "--out-dir",
                str(encoded_root),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        finalize_encoded_dataset(encoded_root)
        encoded_rows = read_rows(encoded_root / "combined_encoded_index.csv")
        assert len(encoded_rows) == 4
        first = encoded_rows[0]
        x = np.load(encoded_root / first["x_path"])
        y = np.load(encoded_root / first["y_path"])
        assert x.shape == (13, 64, 64)
        assert y.shape == (64, 64)
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/build_metadata_features.py"), "--dataset-root", str(encoded_root)],
            cwd=REPO_ROOT,
            check=True,
        )
        manifest = json.loads((encoded_root / "metadata_manifest.json").read_text(encoding="utf-8"))
        assert manifest["num_samples"] == 4
        assert "package_width_mm" in manifest["feature_stats"]
        graph_root = root / "package_plus_power_graph"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/build_graph_features.py"),
                "--source-root",
                str(encoded_root),
                "--out-root",
                str(graph_root),
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        graph_rows = read_rows(graph_root / "combined_encoded_index.csv")
        assert len(graph_rows) == 4
        assert graph_rows[0].get("graph_path")
        repair_artifact_indices(root)
        audit = audit_artifact_paths(root)
        assert audit["unresolved_count"] == 0, audit["unresolved"][:3]
        graph_rows = read_rows(graph_root / "combined_encoded_index.csv")
        first_graph = graph_rows[0]
        assert first_graph["x_path"].startswith("data/runs/") or Path(first_graph["x_path"]).is_absolute()
        assert first_graph["graph_path"].startswith("data/runs/") or Path(first_graph["graph_path"]).is_absolute()
        _test_relocated_path_contract(root, graph_rows)
    print("chiptherm extension artifact smoke test passed")
    return 0


def write_four_row_index(source_index: Path, out_path: Path) -> None:
    with source_index.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    selected = []
    for case_id in ("case11", "case14", "case19", "case20"):
        selected.append(next(row for row in rows if row["case_id"] == case_id))
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _test_relocated_path_contract(source_artifact_root: Path, rows: list[dict[str, str]]) -> None:
    with tempfile.TemporaryDirectory(prefix="chiptherm_ext_reloc_test_") as tmp:
        relocated_repo = Path(tmp) / "repo"
        relocated_artifacts = relocated_repo / "data/runs/benchmarks/benchmark_extension_v1_artifacts"
        relocated_artifacts.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_artifact_root, relocated_artifacts)
        for row in rows:
            for column in ("layout_path", "power_path", "package_path", "hotspot_path", "benchmark_path", "source_dir"):
                value = row.get(column)
                if not value or Path(value).is_absolute():
                    continue
                src = REPO_ROOT / value
                dst = relocated_repo / value
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        audit = audit_artifact_paths(relocated_artifacts, repo_root=relocated_repo)
        assert audit["unresolved_count"] == 0, audit["unresolved"][:3]


if __name__ == "__main__":
    raise SystemExit(main())
