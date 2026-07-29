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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_interpolation_capacity import (  # noqa: E402
    RUN_IDS,
    SOURCE_VERSION,
)


def build_training_command(
    *,
    python: str,
    variant: str,
    data_root: Path,
    output_root: Path,
    config_dir: Path,
    preflight_report: Path,
    device: str,
    workers: int,
    resume: bool,
) -> list[str]:
    config_names = {
        "cosine_ema": "cnn_cosine_ema.yaml",
        "param_matched": "cnn_param_matched.yaml",
    }
    command = [
        python,
        "scripts/train_benchmark_v2_package_residual.py",
        "--data-root",
        str(data_root),
        "--source-version",
        SOURCE_VERSION,
        "--config",
        str(config_dir / config_names[variant]),
        "--preflight-report",
        str(preflight_report),
        "--out-dir",
        str(output_root / RUN_IDS[variant]),
        "--run-id",
        RUN_IDS[variant],
        "--train-family-count",
        "40",
        "--device",
        device,
        "--workers",
        str(workers),
        "--seed",
        "1",
    ]
    if resume:
        command.append("--resume")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run-by-default launcher for one bounded CNN interpolation-capacity variant."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["cosine_ema", "param_matched"],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            f"/export/hdd/{os.environ.get('USER', 'USER')}/chiptherm/"
            "experiment_outputs/benchmark_v2_50family/interpolation_capacity"
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark_v2_50family/interpolation_capacity",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/interpolation_capacity_summary",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    if args.variant == "param_matched":
        gate_path = args.summary_dir.resolve() / "decision_gate.json"
        if not gate_path.is_file():
            raise SystemExit(
                "parameter-matched run is gated until decision_gate.json exists"
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("recommend_param_matched_training") is not True:
            raise SystemExit(
                "decision gate does not recommend parameter-matched training"
            )

    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / RUN_IDS[args.variant]
    completed = run_root / "completed_run_manifest.json"
    if completed.is_file():
        if args.skip_completed:
            print(f"SKIP completed run: {run_root}")
            return 0
        raise SystemExit(
            f"completed run already exists; use --skip-completed to leave it untouched: {run_root}"
        )
    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise SystemExit(
            f"partial output exists; use --resume rather than overwriting it: {run_root}"
        )
    if args.resume and not (run_root / "checkpoints/last.pt").is_file():
        raise SystemExit(f"--resume requires an existing last checkpoint: {run_root}")

    command = build_training_command(
        python=args.python,
        variant=args.variant,
        data_root=args.data_root.expanduser().resolve(),
        output_root=output_root,
        config_dir=args.config_dir.expanduser().resolve(),
        preflight_report=args.preflight_report.expanduser().resolve(),
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )
    print("EXECUTE" if args.execute else "DRY RUN")
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
