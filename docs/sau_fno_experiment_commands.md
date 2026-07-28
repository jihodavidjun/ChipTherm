# ChipTherm SAU-FNO Experiment Commands

These commands are for the GT server. They are stage-gated: train and review
known-family plus held-out-validation results before evaluating primary test
families.

## Environment

```bash
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
export SOURCE_VERSION=source_superposition_final_train40_source_v1
export PREFLIGHT=outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json
export EXPERIMENT_ROOT=/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/sau_fno
export DIRECT_SAU_RUN=$EXPERIMENT_ROOT/direct_temperature_sau_fno_normalized_train40_seed1
export RESIDUAL_SAU_RUN=$EXPERIMENT_ROOT/residual_sau_fno_decomposed_train40_seed1
```

## Compile And Test

```bash
python3 -m py_compile \
  src/chiptherm/ml/sau_fno_models.py \
  src/chiptherm/ml/models.py \
  scripts/train_residual_cnn.py \
  scripts/evaluate_residual_cnn.py \
  scripts/train_benchmark_v2_fno.py \
  scripts/evaluate_benchmark_v2_models.py \
  scripts/analyze_residual_cnn_errors.py \
  scripts/report_benchmark_v2_sau_fno_fairness.py

python3 tests/test_sau_fno_models.py
python3 tests/test_benchmark_v2_sau_fno_training.py
python3 tests/test_fno_models.py
python3 tests/test_benchmark_v2_fno_training.py
python3 tests/test_ufno_models.py
python3 tests/test_benchmark_v2_ufno_training.py
```

## Fairness And Memory Preflight

```bash
python3 scripts/report_benchmark_v2_sau_fno_fairness.py \
  --direct-train-index "$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/$SOURCE_VERSION/sample_split/train_index.csv" \
  --residual-train-index "$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/$SOURCE_VERSION/sample_split/train_index.csv" \
  --direct-fno-config configs/benchmark_v2_50family/training/package_direct_temperature_fno_normalized_seed1.yaml \
  --residual-fno-config configs/benchmark_v2_50family/training/package_residual_fno_decomposed_seed1.yaml \
  --direct-ufno-config configs/benchmark_v2_50family/training/package_direct_temperature_ufno_normalized_seed1.yaml \
  --residual-ufno-config configs/benchmark_v2_50family/training/package_residual_ufno_decomposed_seed1.yaml \
  --direct-sau-fno-config configs/benchmark_v2_50family/training/package_direct_temperature_sau_fno_normalized_seed1.yaml \
  --residual-sau-fno-config configs/benchmark_v2_50family/training/package_residual_sau_fno_decomposed_seed1.yaml \
  --batch-size 64 \
  --out-dir "$EXPERIMENT_ROOT/fairness"
```

The report is static. Before either full run, execute the wrapper's isolated
smoke in a separate directory and inspect peak GPU memory. Do not silently
reduce batch size in the controlled configs.

## Direct Training

```bash
python3 scripts/train_benchmark_v2_fno.py \
  --experiment direct_sau_fno \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --config configs/benchmark_v2_50family/training/package_direct_temperature_sau_fno_normalized_seed1.yaml \
  --preflight-report "$PREFLIGHT" \
  --out-dir "$DIRECT_SAU_RUN" \
  --run-id direct_temperature_sau_fno_normalized_train40_seed1 \
  --device cuda --workers 4 --seed 1
```

## Direct Known/Validation Evaluation

```bash
python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$DIRECT_SAU_RUN/checkpoints/best.pt" \
  --out-dir "$DIRECT_SAU_RUN/evaluation" \
  --batch-size 64 --device cuda --workers 4 \
  --profile-components --save-predictions \
  --protocols known_family_sample_test primary_validation_families
```

## Residual Training

```bash
python3 scripts/train_benchmark_v2_fno.py \
  --experiment residual_sau_fno \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --config configs/benchmark_v2_50family/training/package_residual_sau_fno_decomposed_seed1.yaml \
  --preflight-report "$PREFLIGHT" \
  --out-dir "$RESIDUAL_SAU_RUN" \
  --run-id residual_sau_fno_decomposed_train40_seed1 \
  --device cuda --workers 4 --seed 1
```

## Residual Known/Validation Evaluation

```bash
python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$RESIDUAL_SAU_RUN/checkpoints/best.pt" \
  --out-dir "$RESIDUAL_SAU_RUN/evaluation" \
  --batch-size 64 --device cuda --workers 4 \
  --profile-components --save-predictions --error-analysis \
  --protocols known_family_sample_test primary_validation_families
```

## Primary Test After Validation Review

Run these only after reviewing held-out-validation MAE, RMSE, runtime, and
memory:

```bash
python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$DIRECT_SAU_RUN/checkpoints/best.pt" \
  --out-dir "$DIRECT_SAU_RUN/evaluation" \
  --batch-size 64 --device cuda --workers 4 \
  --profile-components --save-predictions \
  --protocols primary_test_families

python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$RESIDUAL_SAU_RUN/checkpoints/best.pt" \
  --out-dir "$RESIDUAL_SAU_RUN/evaluation" \
  --batch-size 64 --device cuda --workers 4 \
  --profile-components --save-predictions --error-analysis \
  --protocols primary_test_families
```

## Rsync

```bash
rsync -av --relative \
  src/chiptherm/ml/sau_fno_models.py \
  src/chiptherm/ml/models.py \
  scripts/train_residual_cnn.py \
  scripts/evaluate_residual_cnn.py \
  scripts/train_benchmark_v2_fno.py \
  scripts/compare_benchmark_v2_fno_models.py \
  scripts/compare_benchmark_v2_operator_models.py \
  scripts/analyze_residual_cnn_errors.py \
  scripts/report_benchmark_v2_sau_fno_fairness.py \
  configs/benchmark_v2_50family/training/package_direct_temperature_sau_fno_normalized_seed1.yaml \
  configs/benchmark_v2_50family/training/package_residual_sau_fno_decomposed_seed1.yaml \
  docs/sau_fno_architecture_correspondence.md \
  docs/sau_fno_experiment_commands.md \
  tests/test_sau_fno_models.py \
  tests/test_benchmark_v2_sau_fno_training.py \
  tests/test_operator_model_comparison.py \
  "$GT_HOST:$GT_REPO/"
```
