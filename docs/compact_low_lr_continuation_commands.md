# Compact Low-LR Continuation Commands

This is a fresh 20-epoch lineage initialized from canonical epoch-94 raw model
weights. It does not use `--resume`, EMA, SWA, warmup, or any primary-test
family during checkpoint selection.

## Server Environment

```bash
cd /nethome/jjun49/chiptherm_test

PY=/tmp/$USER/chiptherm_venv311_clean/bin/python3
DATA=/export/hdd/$USER/chiptherm/benchmark_v2_50family
SRC=source_superposition_final_train40_source_v1
PREFLIGHT=outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json
CANON=outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1
CONT=outputs/benchmark_v2_50family/compact_low_lr_continuation
```

## 1. Validate And Record The Frozen Definition

Dry run:

```bash
$PY scripts/run_compact_low_lr_continuation.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --parent-checkpoint "$CANON/checkpoints/best.pt" \
  --preflight-report "$PREFLIGHT" \
  --out-root "$CONT" \
  --python "$PY" \
  --device cuda \
  --workers 4 \
  --seed 1
```

Write the pre-training manifest, strict initialization report, and exact
configuration diff without training:

```bash
$PY scripts/run_compact_low_lr_continuation.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --parent-checkpoint "$CANON/checkpoints/best.pt" \
  --preflight-report "$PREFLIGHT" \
  --out-root "$CONT" \
  --python "$PY" \
  --device cuda \
  --workers 4 \
  --seed 1 \
  --prepare
```

## 2. Launch The One Continuation Run

```bash
$PY scripts/run_compact_low_lr_continuation.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --parent-checkpoint "$CANON/checkpoints/best.pt" \
  --preflight-report "$PREFLIGHT" \
  --out-root "$CONT" \
  --python "$PY" \
  --device cuda \
  --workers 4 \
  --seed 1 \
  --execute
```

The inner trainer receives `--init-checkpoint` and
`--require-full-init-checkpoint`; it never receives `--resume`. AdamW and
CosineAnnealingLR are created after the model-only load. Required retained
checkpoints are `epoch_0005.pt`, `epoch_0010.pt`, `epoch_0015.pt`, and
`epoch_0020.pt`.

## 3. Evaluate Validation-Side Protocols

This evaluates all four saved checkpoints on only the known-family sample test
and held-out validation families, using raw weights and saving predictions:

```bash
$PY scripts/evaluate_compact_low_lr_continuation.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --experiment-root "$CONT" \
  --stage selection \
  --python "$PY" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --execute
```

No primary-test protocol is reachable in this stage.

## 4. Analyze And Freeze Selection

Generate the inventory, metrics, per-family table, report, and provisional
decision:

```bash
$PY scripts/analyze_compact_low_lr_continuation.py \
  --experiment-root "$CONT" \
  --canonical-eval-root "$CANON/evaluation"
```

Freeze the validation fingerprint and decision:

```bash
$PY scripts/analyze_compact_low_lr_continuation.py \
  --experiment-root "$CONT" \
  --canonical-eval-root "$CANON/evaluation" \
  --freeze-validation

cat "$CONT/validation_decision_gate.json"
cat "$CONT/selected_candidate.json"
cat "$CONT/primary_test_gate.json"
```

If `selected_candidate.json` has `status=no_candidate`, stop. Primary test
remains closed.

## 5. Evaluate The Single Selected Checkpoint On Primary Test

The wrapper revalidates the frozen fingerprint and refuses any unselected
checkpoint:

```bash
$PY scripts/evaluate_compact_low_lr_continuation.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --experiment-root "$CONT" \
  --canonical-eval-root "$CANON/evaluation" \
  --stage primary-test \
  --python "$PY" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --execute
```

Finalize the report after the one authorized primary-test evaluation:

```bash
$PY scripts/analyze_compact_low_lr_continuation.py \
  --experiment-root "$CONT" \
  --canonical-eval-root "$CANON/evaluation" \
  --include-primary-test
```

## Frozen Selection Rule

- known-family MAE `<= 0.135 K`
- held-out-validation MAE `<= 0.940 K`
- maximum validation-family regression from canonical `<= 0.100 K`
- fraction-worse increase from canonical `<= 0.010`
- hotspot absolute-error increase from canonical `<= 0.050 K`
- choose lowest known-family MAE
- candidates within `0.002 K` of the best known-family MAE tie-break on lower
  held-out-validation MAE
- promote at most one checkpoint

