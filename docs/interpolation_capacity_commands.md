# Interpolation-Capacity Commands

## 1. Sync And Set Up

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
  scripts/analyze_benchmark_v2_interpolation_capacity.py \
  configs/benchmark_v2_50family/interpolation_capacity/cnn_cosine_ema.yaml \
  configs/benchmark_v2_50family/interpolation_capacity/cnn_param_matched.yaml \
  tests/test_benchmark_v2_interpolation_capacity.py \
  docs/interpolation_capacity_experiment.md \
  docs/interpolation_capacity_commands.md \
  "$USER@chao-srv1.ece.gatech.edu:/nethome/jjun49/chiptherm_test/"
```

On GT:

```bash
cd /nethome/jjun49/chiptherm_test
PY=/tmp/$USER/chiptherm_venv311_clean/bin/python3
DATA=/export/hdd/$USER/chiptherm/benchmark_v2_50family
EXP=/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/interpolation_capacity
SRC=source_superposition_final_train40_source_v1
SUMMARY=outputs/benchmark_v2_50family/interpolation_capacity_summary
```

## 2. Freeze Configs And Parameter Match

```bash
$PY scripts/build_benchmark_v2_interpolation_capacity.py \
  --canonical-run-root outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1 \
  --out-dir "$SUMMARY"
```

## 3. Cosine+EMA Dry Run And Training

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant cosine_ema \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --summary-dir "$SUMMARY" \
  --python "$PY"

$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant cosine_ema \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --summary-dir "$SUMMARY" \
  --python "$PY" \
  --execute
```

Resume by adding `--resume --execute`. A completed run is left untouched with
`--skip-completed`.

## 4. Cosine+EMA Validation And Raw Diagnostic

```bash
COS=$EXP/feature_fusion_train40_cosine_ema_seed1

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$COS/checkpoints/best.pt" \
  --out-dir "$COS/evaluation_selection_ema" \
  --protocols known_family_sample_test primary_validation_families \
  --weights ema \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$COS/checkpoints/best.pt" \
  --out-dir "$COS/evaluation_selection_raw" \
  --protocols known_family_sample_test primary_validation_families \
  --weights raw \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 5. Validation Decision

```bash
$PY scripts/analyze_benchmark_v2_interpolation_capacity.py \
  --experiment-root "$EXP" \
  --out-dir "$SUMMARY" \
  --require-cosine-validation

cat "$SUMMARY/decision_gate.json"
```

## 6. Conditional Parameter-Matched Run

The launcher refuses this stage unless `decision_gate.json` says
`recommend_param_matched_training=true`.

```bash
$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --summary-dir "$SUMMARY" \
  --python "$PY"

$PY scripts/run_benchmark_v2_interpolation_capacity.py \
  --variant param_matched \
  --data-root "$DATA" \
  --output-root "$EXP" \
  --summary-dir "$SUMMARY" \
  --python "$PY" \
  --execute
```

Evaluate EMA and raw:

```bash
WIDE=$EXP/feature_fusion_train40_param_matched_cosine_ema_seed1

for WEIGHTS in ema raw; do
  $PY scripts/evaluate_benchmark_v2_models.py \
    --data-root "$DATA" \
    --source-version "$SRC" \
    --checkpoint "$WIDE/checkpoints/best.pt" \
    --out-dir "$WIDE/evaluation_selection_$WEIGHTS" \
    --protocols known_family_sample_test primary_validation_families \
    --weights "$WEIGHTS" \
    --batch-size 64 \
    --device cuda \
    --workers 4 \
    --save-predictions
done
```

Rebuild the validation-only report:

```bash
$PY scripts/analyze_benchmark_v2_interpolation_capacity.py \
  --experiment-root "$EXP" \
  --out-dir "$SUMMARY" \
  --require-cosine-validation
```

## 7. Explicit Primary-Test Gate

After freezing the selected candidate using validation only:

```bash
SELECTED_RUN="$COS"
SELECTED_WEIGHTS=ema

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$SELECTED_RUN/checkpoints/best.pt" \
  --out-dir "$SELECTED_RUN/evaluation_primary_test_$SELECTED_WEIGHTS" \
  --protocols primary_test_families \
  --weights "$SELECTED_WEIGHTS" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions

$PY scripts/analyze_benchmark_v2_interpolation_capacity.py \
  --experiment-root "$EXP" \
  --out-dir "$SUMMARY" \
  --include-primary-test \
  --require-cosine-validation
```

Set `SELECTED_RUN="$WIDE"` only if validation selected the wider model.

## 8. Sync Results Back

From the Mac repository:

```bash
rsync -avh --progress \
  "$USER@chao-srv1.ece.gatech.edu:/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/interpolation_capacity/" \
  outputs/benchmark_v2_50family/interpolation_capacity_runs/

rsync -avh --progress \
  "$USER@chao-srv1.ece.gatech.edu:/nethome/$USER/chiptherm_test/outputs/benchmark_v2_50family/interpolation_capacity_summary/" \
  outputs/benchmark_v2_50family/interpolation_capacity_summary/
```
