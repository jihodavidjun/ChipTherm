#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chiptherm.benchmark_v2_interpolation_capacity import (  # noqa: E402
    CANONICAL_RUN_ID,
    PARAMETER_TARGET_RANGE,
    RUN_IDS,
    SAU_FNO_PARAMETER_COUNT,
    SOURCE_VERSION,
    U_FNO_PARAMETER_COUNT,
    deterministic_width_search,
    read_yaml,
    stable_hash,
    validate_two_factor_configs,
)
from chiptherm.ml.models import build_model, count_parameters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the bounded Benchmark v2 CNN interpolation-capacity study."
    )
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark_v2_50family/training/"
        "package_residual_feature_fusion_v1.yaml",
    )
    parser.add_argument(
        "--canonical-run-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/package_residual/"
        f"{CANONICAL_RUN_ID}",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark_v2_50family/interpolation_capacity",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/benchmark_v2_50family/interpolation_capacity_summary",
    )
    args = parser.parse_args()

    canonical_config = read_yaml(args.canonical_config.resolve())
    canonical_run_config = json.loads(
        (args.canonical_run_root.resolve() / "config.json").read_text(encoding="utf-8")
    )
    canonical_model_config = canonical_run_config["model"]
    canonical_parameters = count_parameters(build_model(canonical_model_config))
    cosine_config = read_yaml(args.config_dir.resolve() / "cnn_cosine_ema.yaml")
    wide_constant_config = read_yaml(
        args.config_dir.resolve() / "cnn_param_matched_constant.yaml"
    )
    wide_cosine_config = read_yaml(
        args.config_dir.resolve() / "cnn_param_matched_cosine_ema.yaml"
    )
    invariance = validate_two_factor_configs(
        canonical_config,
        cosine_config,
        wide_constant_config,
        wide_cosine_config,
    )
    width_search = deterministic_width_search(canonical_model_config)
    selected_width = int(width_search["selected_width"])
    configured_widths = {
        int(wide_constant_config["base_channels"]),
        int(wide_constant_config["refine_channels"]),
        int(wide_constant_config["global_hidden_channels"]),
        int(wide_cosine_config["base_channels"]),
        int(wide_cosine_config["refine_channels"]),
        int(wide_cosine_config["global_hidden_channels"]),
    }
    if configured_widths != {selected_width}:
        raise SystemExit(
            f"parameter-matched config widths {configured_widths} differ from "
            f"deterministic selection {selected_width}"
        )
    parameter_model_config = dict(canonical_model_config)
    parameter_model_config.update(
        {
            "base_channels": selected_width,
            "refine_channels": selected_width,
            "global_hidden_channels": selected_width,
        }
    )
    parameter_count = count_parameters(build_model(parameter_model_config))
    if not PARAMETER_TARGET_RANGE[0] <= parameter_count <= PARAMETER_TARGET_RANGE[1]:
        raise SystemExit("parameter-matched model is outside the approved range")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "benchmark_v2_interpolation_parameter_match/1",
        "canonical": {
            "run_id": CANONICAL_RUN_ID,
            "parameter_count": canonical_parameters,
            "model_config_sha256": stable_hash(canonical_model_config),
        },
        "parameter_matched": {
            "run_ids": {
                "constant": RUN_IDS["param_matched_constant"],
                "cosine_ema": RUN_IDS["param_matched_cosine_ema"],
            },
            "base_channels": selected_width,
            "refine_channels": selected_width,
            "global_hidden_channels": selected_width,
            "metadata_hidden_dim": int(
                wide_constant_config["metadata_hidden_dim"]
            ),
            "metadata_embedding_dim": int(
                wide_constant_config["metadata_embedding_dim"]
            ),
            "parameter_count": parameter_count,
            "difference_vs_canonical": parameter_count - canonical_parameters,
            "percentage_increase_vs_canonical": (
                100.0
                * (parameter_count - canonical_parameters)
                / canonical_parameters
            ),
            "difference_vs_ufno": parameter_count - U_FNO_PARAMETER_COUNT,
            "difference_vs_sau_fno": parameter_count - SAU_FNO_PARAMETER_COUNT,
        },
        "width_search": width_search,
        "config_invariance": invariance,
    }
    write_json(out_dir / "parameter_match_report.json", report)
    write_csv(
        out_dir / "run_manifest.csv",
        [
            {
                "entry": "canonical_cnn",
                "run_id": CANONICAL_RUN_ID,
                "run_type": "reused",
                "config": str(args.canonical_config),
                "epochs": canonical_config["epochs"],
                "scheduler": canonical_config["scheduler"],
                "ema": False,
                "parameter_count": canonical_parameters,
                "source_version": SOURCE_VERSION,
            },
            {
                "entry": "small_cosine_ema_epoch100",
                "run_id": RUN_IDS["small_cosine_ema"],
                "run_type": "completed_explicit_checkpoint",
                "config": str(args.config_dir / "cnn_cosine_ema.yaml"),
                "epochs": 100,
                "scheduler": cosine_config["scheduler"],
                "ema": True,
                "parameter_count": canonical_parameters,
                "source_version": SOURCE_VERSION,
            },
            {
                "entry": "small_cosine_ema_epoch150",
                "run_id": RUN_IDS["small_cosine_ema"],
                "run_type": "completed_explicit_checkpoint",
                "config": str(args.config_dir / "cnn_cosine_ema.yaml"),
                "epochs": 150,
                "scheduler": cosine_config["scheduler"],
                "ema": True,
                "parameter_count": canonical_parameters,
                "source_version": SOURCE_VERSION,
            },
            {
                "entry": "wide_constant_epoch100",
                "run_id": RUN_IDS["param_matched_constant"],
                "run_type": "new_manual_run",
                "config": str(
                    args.config_dir / "cnn_param_matched_constant.yaml"
                ),
                "epochs": wide_constant_config["epochs"],
                "scheduler": wide_constant_config["scheduler"],
                "ema": False,
                "parameter_count": parameter_count,
                "source_version": SOURCE_VERSION,
            },
            {
                "entry": "wide_cosine_ema_epoch100",
                "run_id": RUN_IDS["param_matched_cosine_ema"],
                "run_type": "new_manual_run_explicit_checkpoint",
                "config": str(
                    args.config_dir / "cnn_param_matched_cosine_ema.yaml"
                ),
                "epochs": 100,
                "scheduler": wide_cosine_config["scheduler"],
                "ema": True,
                "parameter_count": parameter_count,
                "source_version": SOURCE_VERSION,
            },
            {
                "entry": "wide_cosine_ema_epoch150",
                "run_id": RUN_IDS["param_matched_cosine_ema"],
                "run_type": "new_manual_run_explicit_checkpoint",
                "config": str(
                    args.config_dir / "cnn_param_matched_cosine_ema.yaml"
                ),
                "epochs": 150,
                "scheduler": wide_cosine_config["scheduler"],
                "ema": True,
                "parameter_count": parameter_count,
                "source_version": SOURCE_VERSION,
            },
        ],
    )
    print(f"Canonical parameters: {canonical_parameters}")
    print(
        f"Parameter-matched width/count: {selected_width}/{parameter_count}"
    )
    print(f"Output: {out_dir}")
    return 0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
