# Family-Count Scaling Commands

Set the server paths once:

```bash
cd /nethome/jjun49/chiptherm_test
PY=/tmp/$USER/chiptherm_venv311_clean/bin/python3
DATA=/export/hdd/$USER/chiptherm/benchmark_v2_50family
EXP=/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/family_count_scaling
SRC=source_superposition_final_train40_source_v1
DEF=outputs/benchmark_v2_50family/family_count_scaling_summary
IDX=$DATA/derived/indices/family_count_scaling/diversity_first
CANON=outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1
```

## 1. Build And Prove Equivalence

This creates index-only subsets and stops if canonical train-40 equivalence
cannot be proven.

```bash
$PY scripts/build_benchmark_v2_family_count_scaling.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --canonical-run-root "$CANON" \
  --index-root "$IDX" \
  --out-dir "$DEF"
```

Dry-run each training command before execution:

```bash
for N in 10 20 30; do
  $PY scripts/run_benchmark_v2_family_count_scaling.py \
    --family-count "$N" \
    --data-root "$DATA" \
    --index-root "$IDX" \
    --output-root "$EXP" \
    --definition-dir "$DEF" \
    --python "$PY"
done
```

## 2. Train10 And Validation Evaluation

```bash
$PY scripts/run_benchmark_v2_family_count_scaling.py \
  --family-count 10 \
  --data-root "$DATA" \
  --index-root "$IDX" \
  --output-root "$EXP" \
  --definition-dir "$DEF" \
  --python "$PY" \
  --execute

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$EXP/family_scaling_diversity_train10_seed1/checkpoints/best.pt" \
  --known-family-index "$IDX/train10/known_family_test_index.csv" \
  --out-dir "$EXP/family_scaling_diversity_train10_seed1/evaluation_selection" \
  --protocols known_family_sample_test primary_validation_families \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 3. Train20 And Validation Evaluation

```bash
$PY scripts/run_benchmark_v2_family_count_scaling.py \
  --family-count 20 \
  --data-root "$DATA" \
  --index-root "$IDX" \
  --output-root "$EXP" \
  --definition-dir "$DEF" \
  --python "$PY" \
  --execute

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$EXP/family_scaling_diversity_train20_seed1/checkpoints/best.pt" \
  --known-family-index "$IDX/train20/known_family_test_index.csv" \
  --out-dir "$EXP/family_scaling_diversity_train20_seed1/evaluation_selection" \
  --protocols known_family_sample_test primary_validation_families \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 4. Train30 And Validation Evaluation

```bash
$PY scripts/run_benchmark_v2_family_count_scaling.py \
  --family-count 30 \
  --data-root "$DATA" \
  --index-root "$IDX" \
  --output-root "$EXP" \
  --definition-dir "$DEF" \
  --python "$PY" \
  --execute

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$EXP/family_scaling_diversity_train30_seed1/checkpoints/best.pt" \
  --known-family-index "$IDX/train30/known_family_test_index.csv" \
  --out-dir "$EXP/family_scaling_diversity_train30_seed1/evaluation_selection" \
  --protocols known_family_sample_test primary_validation_families \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

Resume uses the same launcher with `--execute --resume`. Add
`--skip-completed` to leave completed runs untouched.

## 5. Freeze Validation Interpretation

```bash
$PY scripts/analyze_benchmark_v2_family_count_scaling.py \
  --experiment-root "$EXP" \
  --canonical-train40-root "$CANON" \
  --definition-dir "$DEF" \
  --out-dir "$DEF" \
  --require-validation-complete
```

Review and freeze the validation report before proceeding.

## 6. Explicitly Gated Primary Test

```bash
for N in 10 20 30; do
  RUN=family_scaling_diversity_train${N}_seed1
  $PY scripts/evaluate_benchmark_v2_models.py \
    --data-root "$DATA" \
    --source-version "$SRC" \
    --checkpoint "$EXP/$RUN/checkpoints/best.pt" \
    --out-dir "$EXP/$RUN/evaluation_primary_test" \
    --protocols primary_test_families \
    --batch-size 64 \
    --device cuda \
    --workers 4 \
    --save-predictions
done

$PY scripts/analyze_benchmark_v2_family_count_scaling.py \
  --experiment-root "$EXP" \
  --canonical-train40-root "$CANON" \
  --definition-dir "$DEF" \
  --out-dir "$DEF" \
  --include-primary-test \
  --require-validation-complete
```

## 7. Sync Results To Mac

Run from the Mac repository:

```bash
rsync -avh --progress \
  "$USER@chao-srv1.ece.gatech.edu:/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/family_count_scaling/" \
  outputs/benchmark_v2_50family/family_count_scaling_runs/

rsync -avh --progress \
  "$USER@chao-srv1.ece.gatech.edu:/nethome/$USER/chiptherm_test/outputs/benchmark_v2_50family/family_count_scaling_summary/" \
  outputs/benchmark_v2_50family/family_count_scaling_summary/
```
