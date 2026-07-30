# Interpolation-Capacity Commands

This study contains exactly six predetermined CNN entries: canonical small
constant, small cosine+EMA at epochs 100/150, wide constant at epoch 100, and
wide cosine+EMA at epochs 100/150. Training, validation, validation freeze, and
primary-test evaluation are separate stages.

## 1. Sync Implementation To GT

From the Mac repository:

```bash
rsync -av --relative \
  src/chiptherm/ml/ema.py \
  src/chiptherm/benchmark_v2_interpolation_capacity.py \
  scripts/train_residual_cnn.py \
  scripts/evaluate_residual_cnn.py \
  scripts/train_benchmark_v2_package_residual.py \
  scripts/evaluate_benchmark_v2_models.py \
  scripts/build_benchmark_v2_interpolation_capacity.py \
  scripts/run_benchmark_v2_interpolation_capacity.py \
  scripts/inspect_benchmark_v2_interpolation_checkpoints.py \
  scripts/analyze_benchmark_v2_interpolation_capacity.py \
  configs/benchmark_v2_50family/interpolation_capacity/cnn_cosine_ema.yaml \
  configs/benchmark_v2_50family/interpolation_capacity/cnn_param_matched_constant.yaml \
  configs/benchmark_v2_50family/interpolation_capacity/cnn_param_matched_cosine_ema.yaml \
  tests/test_benchmark_v2_interpolation_capacity.py \
  docs/interpolation_capacity_experiment.md \
  docs/interpolation_capacity_commands.md \
  "jjun49@chao-srv1.ece.gatech.edu:/nethome/jjun49/chiptherm_test/"
```

## 2. GT Environment

```bash
cd /nethome/jjun49/chiptherm_test
PY=/tmp/$USER/chiptherm_venv311_clean/bin/python3
DATA=/export/hdd/$USER/chiptherm/benchmark_v2_50family
EXP=/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/interpolation_capacity
SRC=source_superposition_final_train40_source_v1
SUMMARY=outputs/benchmark_v2_50family/interpolation_capacity_summary
SMALL=$EXP/feature_fusion_train40_cosine_ema_seed1
WCONST=$EXP/feature_fusion_train40_param_matched_constant_seed1
WCOS=$EXP/feature_fusion_train40_param_matched_cosine_ema_seed1
```

## 3. Inspect Completed Small Checkpoints

This is read-only and loads checkpoints on CPU:

```bash
$PY scripts/inspect_benchmark_v2_interpolation_checkpoints.py \
  --run-root "$SMALL" \
  --out "$SUMMARY/checkpoint_inspection_report.json"
```

Use `epoch_0100.pt` and `epoch_0150.pt` for this run regardless of stale
`best.pt` or `last.pt`.

## 4. Freeze Wide Configs And Parameter Count

```bash
$PY scripts/build_benchmark_v2_interpolation_capacity.py \
  --canonical-run-root outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1 \
  --out-dir "$SUMMARY"
```

Expected wide parameter count: `3,919,642`.

## 5. Dry-Run Wide Constant

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched_constant \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --python "$PY"
```

## 6. Train Wide Constant

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched_constant \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --python "$PY" \
  --execute
```

## 7. Resume Wide Constant

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched_constant \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --python "$PY" \
  --resume \
  --execute
```

## 8. Dry-Run Wide Cosine+EMA

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched_cosine_ema \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --python "$PY"
```

## 9. Train Wide Cosine+EMA

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched_cosine_ema \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --python "$PY" \
  --execute
```

## 10. Resume Wide Cosine+EMA

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched_cosine_ema \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --python "$PY" \
  --resume \
  --execute
```

## 11. Validate Wide Constant Epoch 100

```bash
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$WCONST/checkpoints/epoch_0100.pt" \
  --out-dir "$WCONST/evaluation_epoch0100_raw" \
  --protocols known_family_sample_test primary_validation_families \
  --weights raw \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 12. Validate Wide Cosine Epoch 100 EMA And Raw

```bash
for WEIGHTS in ema raw; do
  $PY scripts/evaluate_benchmark_v2_models.py \
    --data-root "$DATA" \
    --source-version "$SRC" \
    --checkpoint "$WCOS/checkpoints/epoch_0100.pt" \
    --out-dir "$WCOS/evaluation_epoch0100_$WEIGHTS" \
    --protocols known_family_sample_test primary_validation_families \
    --weights "$WEIGHTS" \
    --batch-size 64 \
    --device cuda \
    --workers 4 \
    --save-predictions
done
```

## 13. Validate Wide Cosine Epoch 150 EMA And Raw

```bash
for WEIGHTS in ema raw; do
  $PY scripts/evaluate_benchmark_v2_models.py \
    --data-root "$DATA" \
    --source-version "$SRC" \
    --checkpoint "$WCOS/checkpoints/epoch_0150.pt" \
    --out-dir "$WCOS/evaluation_epoch0150_$WEIGHTS" \
    --protocols known_family_sample_test primary_validation_families \
    --weights "$WEIGHTS" \
    --batch-size 64 \
    --device cuda \
    --workers 4 \
    --save-predictions
done
```

The completed small epoch-100 and epoch-150 validation artifacts must use the
same naming contract:

```bash
for EPOCH in 0100 0150; do
  $PY scripts/evaluate_benchmark_v2_models.py \
    --data-root "$DATA" \
    --source-version "$SRC" \
    --checkpoint "$SMALL/checkpoints/epoch_$EPOCH.pt" \
    --out-dir "$SMALL/evaluation_epoch${EPOCH}_ema" \
    --protocols known_family_sample_test primary_validation_families \
    --weights ema \
    --batch-size 64 \
    --device cuda \
    --workers 4 \
    --save-predictions
done
```

## 14. Build Validation-Only Report

```bash
$PY scripts/analyze_benchmark_v2_interpolation_capacity.py \
  --experiment-root "$EXP" \
  --out-dir "$SUMMARY"
```

The report remains `pending` until all six CNN cells have both validation-side
protocols.

## 15. Inspect And Freeze Validation Interpretation

Inspect the pending/ready report, then freeze:

```bash
cat "$SUMMARY/interpolation_capacity_report.md"

$PY scripts/analyze_benchmark_v2_interpolation_capacity.py \
  --experiment-root "$EXP" \
  --out-dir "$SUMMARY" \
  --freeze-validation

cat "$SUMMARY/validation_decision_gate.json"
```

After freezing, changed validation artifacts cause a hard failure.

## 16. Primary Test: Small Cosine Epoch 100 EMA

```bash
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$SMALL/checkpoints/epoch_0100.pt" \
  --out-dir "$SMALL/evaluation_epoch0100_ema" \
  --protocols primary_test_families \
  --weights ema \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 17. Primary Test: Small Cosine Epoch 150 EMA

```bash
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$SMALL/checkpoints/epoch_0150.pt" \
  --out-dir "$SMALL/evaluation_epoch0150_ema" \
  --protocols primary_test_families \
  --weights ema \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 18. Primary Test: Wide Constant Epoch 100

```bash
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$WCONST/checkpoints/epoch_0100.pt" \
  --out-dir "$WCONST/evaluation_epoch0100_raw" \
  --protocols primary_test_families \
  --weights raw \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 19. Primary Test: Wide Cosine Epoch 100 EMA

```bash
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$WCOS/checkpoints/epoch_0100.pt" \
  --out-dir "$WCOS/evaluation_epoch0100_ema" \
  --protocols primary_test_families \
  --weights ema \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 20. Primary Test: Wide Cosine Epoch 150 EMA

```bash
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$WCOS/checkpoints/epoch_0150.pt" \
  --out-dir "$WCOS/evaluation_epoch0150_ema" \
  --protocols primary_test_families \
  --weights ema \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 21. Optional Wide Cosine Raw Primary Test

Run only if raw weights remain a serious candidate after the frozen validation
analysis:

```bash
EPOCH=0100
$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$WCOS/checkpoints/epoch_$EPOCH.pt" \
  --out-dir "$WCOS/evaluation_epoch${EPOCH}_raw" \
  --protocols primary_test_families \
  --weights raw \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 22. Generate Final Report

```bash
$PY scripts/analyze_benchmark_v2_interpolation_capacity.py \
  --experiment-root "$EXP" \
  --out-dir "$SUMMARY" \
  --include-primary-test
```

This command refuses to run unless the validation decision artifact is already
frozen and its metric fingerprint still matches.

## 23. Sync Results Back To Mac

From the Mac repository:

```bash
rsync -avh --progress \
  "jjun49@chao-srv1.ece.gatech.edu:/export/hdd/jjun49/chiptherm/experiment_outputs/benchmark_v2_50family/interpolation_capacity/" \
  outputs/benchmark_v2_50family/interpolation_capacity_runs/

rsync -avh --progress \
  "jjun49@chao-srv1.ece.gatech.edu:/nethome/jjun49/chiptherm_test/outputs/benchmark_v2_50family/interpolation_capacity_summary/" \
  outputs/benchmark_v2_50family/interpolation_capacity_summary/
```
