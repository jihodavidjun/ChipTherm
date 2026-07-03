#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.ml.dataset import ChipThermDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the ChipTherm PyTorch dataset interface.")
    parser.add_argument(
        "--index",
        default=REPO_ROOT / "data/runs/benchmarks/dataset_v1/train_index.csv",
        type=Path,
    )
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()

    dataset = ChipThermDataset(args.index, target="residual", return_metadata=True)
    dataset.summary()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print("Batch shapes")
    print(f"X: {tuple(batch['x'].shape)}")
    print(f"Residual target: {tuple(batch['target'].shape)}")
    print(f"Physics: {tuple(batch['physics'].shape)}")
    print(f"Temperature: {tuple(batch['temperature'].shape)}")
    print(f"Residual: {tuple(batch['residual'].shape)}")
    print("Example metadata")
    print(_first_metadata(batch["metadata"]))
    return 0


def _first_metadata(metadata_batch: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in metadata_batch.items():
        if hasattr(value, "shape"):
            item[key] = value[0].item()
        elif isinstance(value, list):
            item[key] = value[0]
        else:
            item[key] = value
    return item


if __name__ == "__main__":
    raise SystemExit(main())
