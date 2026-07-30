#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch


PERIODIC_PATTERN = re.compile(r"epoch_(\d{4})\.pt$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("model_state_dict")
    ema_state = checkpoint.get("ema_model_state_dict")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_flag": bool(checkpoint.get("best", False)),
        "best_val_mae_K": finite_or_none(checkpoint.get("best_val_mae_K")),
        "global_optimizer_step": int(checkpoint.get("global_optimizer_step", -1)),
        "epochs_without_improvement": int(
            checkpoint.get("epochs_without_improvement", -1)
        ),
        "raw_state_present": isinstance(raw_state, Mapping) and bool(raw_state),
        "ema_state_present": isinstance(ema_state, Mapping) and bool(ema_state),
        "ema_metadata_present": checkpoint.get("ema_state_dict") is not None,
        "optimizer_state_present": checkpoint.get("optimizer_state_dict") is not None,
        "scheduler_state_present": checkpoint.get("scheduler_state_dict") is not None,
        "evaluation_default_weights": checkpoint.get(
            "evaluation_default_weights", "raw"
        ),
        "config_sha256": checkpoint.get("config_sha256"),
        "parameter_count": checkpoint.get("parameter_count"),
    }


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def read_training_log(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def inspect_run(run_root: Path) -> dict[str, Any]:
    checkpoint_root = run_root / "checkpoints"
    paths: list[Path] = []
    for name in ("best.pt", "last.pt"):
        path = checkpoint_root / name
        if path.is_file():
            paths.append(path)
    paths.extend(
        sorted(
            path
            for path in checkpoint_root.glob("epoch_*.pt")
            if PERIODIC_PATTERN.fullmatch(path.name)
        )
    )
    summaries = {path.name: load_checkpoint_summary(path) for path in paths}
    rows = read_training_log(run_root / "train_log.csv")
    logged_epochs = [int(row["epoch"]) for row in rows if row.get("epoch")]
    val_rows = [
        row
        for row in rows
        if str(row.get("val_final_mae_K", "")).strip()
    ]
    latest_logged_epoch = max(logged_epochs, default=None)
    best_logged_epoch = (
        int(min(val_rows, key=lambda row: float(row["val_final_mae_K"]))["epoch"])
        if val_rows
        else None
    )
    periodic_mismatches = []
    periodic_epochs = []
    for name, summary in summaries.items():
        match = PERIODIC_PATTERN.fullmatch(name)
        if not match:
            continue
        filename_epoch = int(match.group(1))
        periodic_epochs.append(filename_epoch)
        if summary["epoch"] != filename_epoch:
            periodic_mismatches.append(
                {
                    "path": summary["path"],
                    "filename_epoch": filename_epoch,
                    "checkpoint_epoch": summary["epoch"],
                }
            )
    latest_evidence = max(
        [epoch for epoch in [latest_logged_epoch, *periodic_epochs] if epoch is not None],
        default=None,
    )
    findings: list[dict[str, Any]] = []
    best = summaries.get("best.pt")
    if best is not None:
        if best_logged_epoch is None:
            findings.append(
                {
                    "artifact": "best.pt",
                    "classification": "unverifiable_without_validation_log",
                }
            )
        elif best["epoch"] == best_logged_epoch:
            findings.append(
                {
                    "artifact": "best.pt",
                    "classification": "consistent_best_internal_validation_epoch",
                    "epoch": best["epoch"],
                }
            )
        else:
            findings.append(
                {
                    "artifact": "best.pt",
                    "classification": "inconsistent_with_training_log",
                    "checkpoint_epoch": best["epoch"],
                    "expected_epoch": best_logged_epoch,
                }
            )
    last = summaries.get("last.pt")
    if last is not None and latest_evidence is not None:
        if last["epoch"] == latest_evidence:
            classification = "consistent_latest_completed_epoch"
        elif last["epoch"] < latest_evidence:
            classification = "stale_or_overwritten_after_training"
        else:
            classification = "newer_than_available_log_and_periodic_evidence"
        findings.append(
            {
                "artifact": "last.pt",
                "classification": classification,
                "checkpoint_epoch": last["epoch"],
                "latest_evidence_epoch": latest_evidence,
            }
        )
    if periodic_mismatches:
        findings.append(
            {
                "artifact": "periodic_checkpoints",
                "classification": "filename_epoch_mismatch",
                "mismatches": periodic_mismatches,
            }
        )
    else:
        findings.append(
            {
                "artifact": "periodic_checkpoints",
                "classification": "consistent",
                "epochs": sorted(periodic_epochs),
            }
        )
    return {
        "schema_version": "benchmark_v2_checkpoint_inspection/1",
        "run_root": str(run_root),
        "trainer_semantics": {
            "best_pt": "saved only when internal-validation MAE strictly improves",
            "last_pt": "saved after every completed epoch",
            "periodic": "saved independently at configured epoch frequency",
            "resume": (
                "restores raw model, EMA, optimizer, scheduler, completed epoch, "
                "global optimizer step, best metric, and early-stopping count"
            ),
        },
        "training_log": {
            "present": (run_root / "train_log.csv").is_file(),
            "rows": len(rows),
            "latest_epoch": latest_logged_epoch,
            "best_validation_epoch": best_logged_epoch,
        },
        "checkpoints": summaries,
        "findings": findings,
        "trainer_patch_required": False,
        "note": (
            "An early best.pt is intentional when it matches the minimum logged "
            "validation MAE. A last.pt older than a valid later periodic checkpoint "
            "cannot be produced by an uninterrupted invocation of the current save loop."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect interpolation-study checkpoint semantics without loading CUDA."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = inspect_run(args.run_root.expanduser().resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for finding in report["findings"]:
        print(
            f"{finding['artifact']}: {finding['classification']}"
        )
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
