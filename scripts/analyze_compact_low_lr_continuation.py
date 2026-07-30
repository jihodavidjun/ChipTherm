#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chiptherm.compact_low_lr_continuation import (  # noqa: E402
    CONTINUATION_EPOCHS,
    EXPECTED_ARCHITECTURE,
    EXPECTED_EPOCHS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PARENT_EPOCH,
    EXPECTED_RECONSTRUCTION,
    PRIMARY_PROTOCOL,
    VALIDATION_PROTOCOLS,
    authorize_primary_test,
    checkpoint_id,
    checkpoint_path,
    continuation_lr,
    evaluation_path,
    now_utc,
    select_checkpoint,
    selection_thresholds,
    sha256_file,
    stable_hash,
    validation_fingerprint,
)
from scripts.analyze_compact_weight_interpolation import (  # noqa: E402
    aggregate_protocol,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_inventory(experiment_root: Path) -> dict[str, Any]:
    manifest_path = experiment_root / "continuation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"continuation manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    parent = manifest["parent_checkpoint"]
    parent_path = Path(parent["path"])
    if not parent_path.is_file():
        raise FileNotFoundError(f"parent checkpoint is missing: {parent_path}")
    if sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("parent checkpoint hash changed after continuation setup")
    parent_checkpoint = torch.load(
        parent_path,
        map_location="cpu",
        weights_only=False,
    )
    parent_state = parent_checkpoint["model_state_dict"]
    records: list[dict[str, Any]] = []
    previous_steps = -1
    for epoch in CONTINUATION_EPOCHS:
        path = checkpoint_path(experiment_root, epoch)
        if not path.is_file():
            raise FileNotFoundError(f"required continuation checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("training_config") or {}
        model = checkpoint.get("model_config") or {}
        lineage = checkpoint.get("training_lineage") or {}
        parent_lineage = lineage.get("parent_checkpoint") or {}
        initialization = lineage.get("initialization") or {}
        optimizer = checkpoint.get("optimizer_state_dict")
        scheduler = checkpoint.get("scheduler_state_dict")
        optimizer_lr = float(optimizer["param_groups"][0]["lr"])
        expected_lr = continuation_lr(epoch)
        global_steps = int(checkpoint.get("global_optimizer_step", -1))
        state = checkpoint.get("model_state_dict") or {}
        finite = bool(state) and all(
            isinstance(value, torch.Tensor) and torch.isfinite(value).all()
            for value in state.values()
        )
        differs_from_parent = any(
            not torch.equal(value, parent_state[name])
            for name, value in state.items()
        )
        checks = {
            "epoch": int(checkpoint.get("epoch", -1)) == epoch,
            "architecture": model.get("architecture") == EXPECTED_ARCHITECTURE,
            "parameter_count": int(
                checkpoint.get(
                    "parameter_count",
                    model.get("total_parameters", -1),
                )
            )
            == EXPECTED_PARAMETER_COUNT,
            "epochs": int(config.get("epochs", -1)) == EXPECTED_EPOCHS,
            "fresh_lineage": initialization.get("new_training_lineage") is True,
            "weights_only_init": initialization.get("mode") == "weights_only",
            "optimizer_not_restored": (
                initialization.get("optimizer_state_restored") is False
            ),
            "scheduler_not_restored": (
                initialization.get("scheduler_state_restored") is False
            ),
            "epoch_not_restored": initialization.get("epoch_restored") is False,
            "ema_not_restored": initialization.get("ema_state_restored") is False,
            "parent_hash": parent_lineage.get("sha256") == parent["sha256"],
            "parent_epoch": int(parent_lineage.get("epoch", -1))
            == EXPECTED_PARENT_EPOCH,
            "parent_weights": parent_lineage.get("weights") == "model_state_dict",
            "resume_false": config.get("resume") is False,
            "init_checkpoint": (
                Path(config.get("init_checkpoint", "")).resolve()
                == parent_path.resolve()
            ),
            "strict_initialization": config.get("require_full_init_checkpoint")
            is True,
            "optimizer_present": isinstance(optimizer, Mapping),
            "scheduler_present": isinstance(scheduler, Mapping),
            "scheduler_epoch": int(scheduler.get("last_epoch", -1)) == epoch,
            "lr": math.isclose(
                optimizer_lr,
                expected_lr,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "ema_disabled": (
                config.get("ema_enabled") is False
                and checkpoint.get("ema_model_state_dict") is None
                and checkpoint.get("ema_state_dict") is None
            ),
            "swa_absent": not any("swa" in str(key).lower() for key in checkpoint),
            "raw_evaluation": checkpoint.get("evaluation_default_weights") == "raw",
            "normalization": checkpoint.get("normalization")
            == parent_checkpoint.get("normalization"),
            "train_index_hash": lineage.get("train_index_sha256")
            == manifest["indices"]["train_index_sha256"],
            "val_index_hash": lineage.get("internal_val_index_sha256")
            == manifest["indices"]["internal_val_index_sha256"],
            "source_version": lineage.get("source_superposition_version")
            == manifest["source_superposition_version"],
            "reconstruction": lineage.get("reconstruction")
            == EXPECTED_RECONSTRUCTION,
            "finite_state": finite,
            "weights_updated": differs_from_parent,
            "optimizer_steps_increase": global_steps > previous_steps,
        }
        if not all(checks.values()):
            raise ValueError(
                f"checkpoint epoch {epoch} violates continuation contract: "
                f"{checks}"
            )
        previous_steps = global_steps
        records.append(
            {
                "epoch": epoch,
                "checkpoint_id": checkpoint_id(epoch),
                "path": str(path),
                "sha256": sha256_file(path),
                "optimizer_lr": optimizer_lr,
                "expected_lr": expected_lr,
                "global_optimizer_step": global_steps,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "checks": checks,
            }
        )
    return {
        "schema_version": "compact_low_lr_checkpoint_inventory/1",
        "passed": True,
        "parent_checkpoint": parent,
        "checkpoint_epochs": list(CONTINUATION_EPOCHS),
        "records": records,
    }


def collect_results(
    *,
    experiment_root: Path,
    canonical_eval_root: Path,
    include_primary_test: bool,
    selected_epoch: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for protocol in VALIDATION_PROTOCOLS:
        path = canonical_eval_root / protocol
        row, family_rows = aggregate_protocol(
            path,
            model="canonical_reference",
            alpha=None,
            protocol=protocol,
            require_predictions=False,
        )
        rows.append(row)
        families.extend(family_rows)
    for epoch in CONTINUATION_EPOCHS:
        model = checkpoint_id(epoch)
        for protocol in VALIDATION_PROTOCOLS:
            path = evaluation_path(experiment_root, epoch, protocol)
            if not (path / "metrics.json").is_file():
                missing.append(
                    {
                        "model": model,
                        "epoch": epoch,
                        "protocol": protocol,
                        "path": str(path),
                    }
                )
                continue
            row, family_rows = aggregate_protocol(
                path,
                model=model,
                alpha=None,
                protocol=protocol,
                require_predictions=True,
            )
            if int(row["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
                raise ValueError(f"{model} parameter count changed")
            rows.append(row)
            families.extend(family_rows)
    if include_primary_test:
        if selected_epoch is None:
            raise ValueError("primary-test collection requires a selected epoch")
        path = evaluation_path(
            experiment_root,
            selected_epoch,
            PRIMARY_PROTOCOL,
        )
        row, family_rows = aggregate_protocol(
            path,
            model=checkpoint_id(selected_epoch),
            alpha=None,
            protocol=PRIMARY_PROTOCOL,
            require_predictions=True,
        )
        rows.append(row)
        families.extend(family_rows)
    return rows, families, missing


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Compact Low-LR Continuation",
        "",
        "The run starts from canonical epoch-94 raw model weights with fresh "
        "AdamW and cosine-scheduler state. Primary-test results are excluded "
        "from checkpoint selection.",
        "",
        "| Checkpoint | Protocol | MAE (K) | RMSE (K) | Hotspot abs. error (K) | Fraction worse |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['protocol']} | "
            f"{float(row['micro_mae_K']):.6f} | "
            f"{float(row['micro_rmse_K']):.6f} | "
            f"{float(row['hotspot_temperature_abs_error_K']):.6f} | "
            f"{float(row['fraction_worse_than_source']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Validation gate: **{summary['validation_gate_status']}**",
            "",
            f"Selection: `{summary['selection'].get('status')}`",
            "",
            "At most one frozen selected checkpoint may be evaluated on the "
            "primary-test families.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    experiment_root: Path,
    canonical_eval_root: Path,
    freeze_validation: bool,
    include_primary_test: bool,
) -> dict[str, Any]:
    experiment_root.mkdir(parents=True, exist_ok=True)
    inventory = checkpoint_inventory(experiment_root)
    inventory_path = experiment_root / "checkpoint_inventory.json"
    write_json(inventory_path, inventory)
    inventory_sha256 = sha256_file(inventory_path)
    gate_path = experiment_root / "validation_decision_gate.json"
    frozen = read_json(gate_path) if gate_path.is_file() else None
    if frozen is not None and frozen.get("status") == "frozen":
        selection = frozen["selection"]
    else:
        selection = {"status": "pending"}
    selected_epoch = (
        authorize_primary_test(frozen)
        if include_primary_test and frozen is not None
        else (
            int(selection["selected_epoch"])
            if selection.get("status") == "selected"
            else None
        )
    )
    if include_primary_test and frozen is None:
        raise ValueError(
            "--include-primary-test requires a frozen validation gate with "
            "one selected checkpoint"
        )
    primary_root = experiment_root / "evaluation_primary_test"
    existing_primary = sorted(primary_root.rglob("metrics.json")) if primary_root.exists() else []
    if not include_primary_test and existing_primary and (
        frozen is None or frozen.get("status") != "frozen"
    ):
        raise ValueError(
            "primary-test metrics exist before the validation gate was frozen"
        )
    if include_primary_test:
        allowed = evaluation_path(
            experiment_root,
            selected_epoch,
            PRIMARY_PROTOCOL,
        ) / "metrics.json"
        extras = [path for path in existing_primary if path != allowed]
        if extras:
            raise ValueError(
                "primary-test metrics exist for an unselected checkpoint: "
                f"{extras}"
            )
    rows, families, missing = collect_results(
        experiment_root=experiment_root,
        canonical_eval_root=canonical_eval_root,
        include_primary_test=include_primary_test,
        selected_epoch=selected_epoch,
    )
    fingerprint = validation_fingerprint(
        rows,
        checkpoint_inventory_sha256=inventory_sha256,
    )
    if frozen is not None and frozen.get("status") == "frozen":
        if frozen.get("validation_fingerprint") != fingerprint:
            raise ValueError("frozen validation fingerprint changed")
        selection = frozen["selection"]
    else:
        selection = select_checkpoint(rows, families)
    validation_complete = not missing
    gate: dict[str, Any] = {
        "schema_version": "compact_low_lr_validation_gate/1",
        "status": "ready_to_freeze" if validation_complete else "pending",
        "validation_fingerprint": fingerprint,
        "checkpoint_inventory_sha256": inventory_sha256,
        "selection": selection,
        "thresholds": selection_thresholds(),
        "primary_test_used_for_selection": False,
        "missing_results": missing,
    }
    if frozen is not None and frozen.get("status") == "frozen":
        gate = frozen
    elif freeze_validation:
        if not validation_complete:
            raise ValueError("cannot freeze with missing validation evaluations")
        gate["status"] = "frozen"
        gate["frozen_at_utc"] = now_utc()
    write_json(gate_path, gate)
    write_json(
        experiment_root / "selected_candidate.json",
        gate.get("selection", selection),
    )
    primary_gate = {
        "schema_version": "compact_low_lr_primary_test_gate/1",
        "status": "closed",
        "validation_gate_status": gate["status"],
        "validation_fingerprint": gate["validation_fingerprint"],
        "selected_epoch": gate.get("selection", {}).get("selected_epoch"),
        "selected_checkpoint": (
            str(
                checkpoint_path(
                    experiment_root,
                    int(gate["selection"]["selected_epoch"]),
                )
            )
            if gate.get("selection", {}).get("status") == "selected"
            else None
        ),
        "maximum_promoted_checkpoints": 1,
        "primary_test_used_for_selection": False,
    }
    if (
        gate["status"] == "frozen"
        and gate.get("selection", {}).get("status") == "selected"
    ):
        primary_gate["status"] = "open_for_selected_checkpoint"
    if include_primary_test:
        selected_epoch = int(gate["selection"]["selected_epoch"])
        primary = next(
            row
            for row in rows
            if row["model"] == checkpoint_id(selected_epoch)
            and row["protocol"] == PRIMARY_PROTOCOL
        )
        primary_gate.update(
            status="complete",
            metrics_path=primary["metrics_path"],
            metrics_sha256=primary["metrics_sha256"],
        )
    write_json(experiment_root / "primary_test_gate.json", primary_gate)
    write_csv(experiment_root / "continuation_metrics.csv", rows)
    write_csv(experiment_root / "continuation_per_family.csv", families)
    summary = {
        "schema_version": "compact_low_lr_continuation_summary/1",
        "created_at_utc": now_utc(),
        "validation_gate_status": gate["status"],
        "selection": gate.get("selection", selection),
        "validation_fingerprint": gate["validation_fingerprint"],
        "checkpoint_inventory_passed": inventory["passed"],
        "missing_results": missing,
        "primary_test_included": include_primary_test,
        "primary_test_used_for_selection": False,
    }
    write_json(experiment_root / "continuation_summary.json", summary)
    write_report(
        experiment_root / "continuation_report.md",
        summary,
        rows,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze and freeze the compact low-LR continuation gate."
    )
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--canonical-eval-root", required=True, type=Path)
    parser.add_argument("--freeze-validation", action="store_true")
    parser.add_argument("--include-primary-test", action="store_true")
    args = parser.parse_args()
    summary = analyze(
        experiment_root=args.experiment_root.expanduser().resolve(),
        canonical_eval_root=args.canonical_eval_root.expanduser().resolve(),
        freeze_validation=args.freeze_validation,
        include_primary_test=args.include_primary_test,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
