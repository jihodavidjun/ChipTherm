#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_family_scaling import RUN_IDS, SOURCE_VERSION  # noqa: E402


def build_training_command(
    *,
    python: str,
    data_root: Path,
    family_count: int,
    index_root: Path,
    output_root: Path,
    preflight_report: Path,
    config: Path,
    device: str,
    workers: int,
    resume: bool,
) -> list[str]:
    if family_count not in RUN_IDS:
        raise ValueError("family_count must be one of 10, 20, or 30")
    command = [
        python,
        "scripts/train_benchmark_v2_package_residual.py",
        "--data-root",
        str(data_root),
        "--source-version",
        SOURCE_VERSION,
        "--config",
        str(config),
        "--preflight-report",
        str(preflight_report),
        "--out-dir",
        str(output_root / RUN_IDS[family_count]),
        "--run-id",
        RUN_IDS[family_count],
        "--train-family-count",
        str(family_count),
        "--prepared-index-root",
        str(index_root / f"train{family_count}"),
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
        description="Dry-run-by-default launcher for one Benchmark v2 family-count run."
    )
    parser.add_argument("--family-count", type=int, choices=sorted(RUN_IDS), required=True)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CHIPTHERM_V2_DATA_ROOT"))
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help="Defaults to DATA_ROOT/derived/indices/family_count_scaling/diversity_first.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            f"/export/hdd/{os.environ.get('USER', 'USER')}/chiptherm/experiment_outputs/"
            "benchmark_v2_50family/family_count_scaling"
        ),
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark_v2_50family/training/package_residual_feature_fusion_v1.yaml",
    )
    parser.add_argument(
        "--definition-dir",
        type=Path,
        default=REPO_ROOT / "outputs/benchmark_v2_50family/family_count_scaling_summary",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit("--data-root or CHIPTHERM_V2_DATA_ROOT is required")
    equivalence_path = args.definition_dir.expanduser().resolve() / "train40_reuse_equivalence.json"
    if not equivalence_path.is_file():
        raise SystemExit(f"train40 equivalence gate is missing: {equivalence_path}")
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if equivalence.get("canonical_train40_reusable") is not True:
        raise SystemExit("canonical train40 reuse equivalence has not passed")
    resolved_config_path = (
        args.definition_dir.expanduser().resolve()
        / "resolved_configs"
        / f"train{args.family_count}.json"
    )
    if not resolved_config_path.is_file():
        raise SystemExit(f"resolved scaling config is missing: {resolved_config_path}")
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    canonical_training = yaml.safe_load(
        args.config.expanduser().resolve().read_text(encoding="utf-8")
    )
    if (
        int(resolved_config.get("family_count", -1)) != args.family_count
        or resolved_config.get("run_id") != RUN_IDS[args.family_count]
        or resolved_config.get("training") != canonical_training
    ):
        raise SystemExit(
            f"resolved scaling config no longer matches the canonical training recipe: "
            f"{resolved_config_path}"
        )

    data_root = args.data_root.expanduser().resolve()
    index_root = (
        args.index_root.expanduser().resolve()
        if args.index_root is not None
        else data_root / "derived/indices/family_count_scaling/diversity_first"
    )
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / RUN_IDS[args.family_count]
    if args.skip_completed and (run_root / "completed_run_manifest.json").is_file():
        print(f"SKIP completed run: {run_root}")
        return 0
    command = build_training_command(
        python=args.python,
        data_root=data_root,
        family_count=args.family_count,
        index_root=index_root,
        output_root=output_root,
        preflight_report=args.preflight_report.expanduser().resolve(),
        config=args.config.expanduser().resolve(),
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )
    print("DRY RUN" if not args.execute else "EXECUTE")
    print(" ".join(command))
    if args.execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
