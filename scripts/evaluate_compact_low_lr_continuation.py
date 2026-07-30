#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.compact_low_lr_continuation import (  # noqa: E402
    CONTINUATION_EPOCHS,
    PRIMARY_PROTOCOL,
    VALIDATION_PROTOCOLS,
    authorize_primary_test,
    checkpoint_id,
    checkpoint_path,
)
from scripts.analyze_compact_low_lr_continuation import analyze  # noqa: E402


def evaluation_command(
    *,
    python: Path,
    data_root: Path,
    source_version: str,
    checkpoint: Path,
    out_dir: Path,
    protocols: tuple[str, ...],
    batch_size: int,
    device: str,
    workers: int,
) -> list[str]:
    return [
        str(python),
        "scripts/evaluate_benchmark_v2_models.py",
        "--data-root",
        str(data_root),
        "--source-version",
        source_version,
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
        "--protocols",
        *protocols,
        "--weights",
        "raw",
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--workers",
        str(workers),
        "--save-predictions",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate and run compact continuation evaluation protocols."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"),
        type=Path,
    )
    parser.add_argument(
        "--source-version",
        default="source_superposition_final_train40_source_v1",
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--canonical-eval-root", type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["selection", "primary-test"],
    )
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    root = args.experiment_root.expanduser().resolve()
    commands: list[list[str]] = []
    if args.stage == "selection":
        for epoch in CONTINUATION_EPOCHS:
            checkpoint = checkpoint_path(root, epoch)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"required continuation checkpoint is missing: {checkpoint}"
                )
            commands.append(
                evaluation_command(
                    python=args.python,
                    data_root=args.data_root.expanduser().resolve(),
                    source_version=args.source_version,
                    checkpoint=checkpoint,
                    out_dir=(
                        root
                        / "evaluation_selection"
                        / checkpoint_id(epoch)
                    ),
                    protocols=VALIDATION_PROTOCOLS,
                    batch_size=args.batch_size,
                    device=args.device,
                    workers=args.workers,
                )
            )
    else:
        if args.canonical_eval_root is None:
            raise SystemExit(
                "--canonical-eval-root is required for primary-test authorization"
            )
        analyze(
            experiment_root=root,
            canonical_eval_root=args.canonical_eval_root.expanduser().resolve(),
            freeze_validation=False,
            include_primary_test=False,
        )
        gate = json.loads(
            (root / "validation_decision_gate.json").read_text(
                encoding="utf-8"
            )
        )
        epoch = authorize_primary_test(
            gate,
            validation_fingerprint_value=gate["validation_fingerprint"],
        )
        commands.append(
            evaluation_command(
                python=args.python,
                data_root=args.data_root.expanduser().resolve(),
                source_version=args.source_version,
                checkpoint=checkpoint_path(root, epoch),
                out_dir=(
                    root
                    / "evaluation_primary_test"
                    / checkpoint_id(epoch)
                ),
                protocols=(PRIMARY_PROTOCOL,),
                batch_size=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
        )
    for command in commands:
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not args.execute:
        print("Evaluation commands validated; no inference was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
