# Benchmark v2 U-FNO Stage-Gated Commands

## Stage 1

Run on the Mac:

```bash
export GT_USER=jjun49
export GT_HOST="$GT_USER@<gt-host>"
export GT_REPO="/nethome/$GT_USER/chiptherm"
rsync -az --info=progress2 \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude 'outputs/' \
  /Users/jihojun/chiptherm/ \
  "$GT_HOST:$GT_REPO/"
```

Run on GT:

```bash
cd /nethome/$USER/chiptherm
source .venv/bin/activate
export CHIPTHERM_REPO="$PWD"
export PYTHONPATH="$CHIPTHERM_REPO/src:$CHIPTHERM_REPO"
export CHIPTHERM_V2_DATA_ROOT="/export/hdd/$USER/chiptherm/benchmark_v2_50family"
export SOURCE_VERSION=source_superposition_final_train40_source_v1
export SOURCE_INDEX_ROOT="$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/$SOURCE_VERSION"
export PREFLIGHT=outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json
export EXPERIMENT_ROOT="/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family"
export DIRECT_UFNO_RUN="$EXPERIMENT_ROOT/ufno/direct_temperature_ufno_normalized_train40_seed1"
export STAGE1_COMPARE="$EXPERIMENT_ROOT/comparisons/operator_stage1_ufno_seed1"

python3 -m py_compile \
  src/chiptherm/ml/ufno_models.py \
  src/chiptherm/ml/models.py \
  scripts/train_residual_cnn.py \
  scripts/evaluate_residual_cnn.py \
  scripts/train_benchmark_v2_fno.py \
  scripts/report_benchmark_v2_ufno_fairness.py \
  scripts/compare_benchmark_v2_fno_models.py \
  scripts/compare_benchmark_v2_operator_models.py

python3 tests/test_ufno_models.py
python3 tests/test_benchmark_v2_ufno_training.py
python3 tests/test_operator_model_comparison.py
python3 tests/test_fno_models.py
python3 tests/test_benchmark_v2_fno_training.py
python3 tests/test_benchmark_v2_fno_sensitivity.py
python3 tests/test_feature_fusion_model.py
python3 tests/test_benchmark_v2_final_training.py

python3 scripts/report_benchmark_v2_ufno_fairness.py \
  --direct-train-index "$SOURCE_INDEX_ROOT/sample_split/train_index.csv" \
  --residual-train-index "$SOURCE_INDEX_ROOT/sample_split/train_index.csv" \
  --direct-fno-config configs/benchmark_v2_50family/training/package_direct_temperature_fno_normalized_seed1.yaml \
  --residual-fno-config configs/benchmark_v2_50family/training/package_residual_fno_decomposed_seed1.yaml \
  --direct-ufno-config configs/benchmark_v2_50family/training/package_direct_temperature_ufno_normalized_seed1.yaml \
  --residual-ufno-config configs/benchmark_v2_50family/training/package_residual_ufno_decomposed_seed1.yaml \
  --direct-cnn-checkpoint outputs/benchmark_v2_50family/package_direct/direct_temperature_feature_fusion_normalized_train40_seed1/checkpoints/best.pt \
  --residual-cnn-checkpoint outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/checkpoints/best.pt \
  --batch-size 64 \
  --out-dir "$EXPERIMENT_ROOT/ufno/fairness"

python3 scripts/train_benchmark_v2_fno.py \
  --experiment direct_ufno \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --config configs/benchmark_v2_50family/training/package_direct_temperature_ufno_normalized_seed1.yaml \
  --preflight-report "$PREFLIGHT" \
  --out-dir "$DIRECT_UFNO_RUN" \
  --run-id direct_temperature_ufno_normalized_train40_seed1 \
  --device cuda \
  --workers 4 \
  --seed 1

python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$DIRECT_UFNO_RUN/checkpoints/best.pt" \
  --out-dir "$DIRECT_UFNO_RUN/evaluation" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --profile-components \
  --save-predictions

python3 scripts/compare_benchmark_v2_operator_models.py \
  --source-only-root outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1 \
  --direct-cnn-root outputs/benchmark_v2_50family/package_direct/direct_temperature_feature_fusion_normalized_train40_seed1/evaluation \
  --residual-cnn-root outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/evaluation \
  --direct-fno-root outputs/benchmark_v2_50family/fno/direct_temperature_fno_normalized_train40_seed1/evaluation \
  --residual-fno-root outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1/evaluation \
  --direct-ufno-root "$DIRECT_UFNO_RUN/evaluation" \
  --out-dir "$STAGE1_COMPARE"
```

Rsync Stage-1 results back to the Mac:

```bash
mkdir -p /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/ufno
mkdir -p /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/comparisons
rsync -az --info=progress2 \
  "$GT_HOST:/export/hdd/$GT_USER/chiptherm/experiment_outputs/benchmark_v2_50family/ufno/direct_temperature_ufno_normalized_train40_seed1/" \
  /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/ufno/direct_temperature_ufno_normalized_train40_seed1/
rsync -az --info=progress2 \
  "$GT_HOST:/export/hdd/$GT_USER/chiptherm/experiment_outputs/benchmark_v2_50family/comparisons/operator_stage1_ufno_seed1/" \
  /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/comparisons/operator_stage1_ufno_seed1/
```

Stop here and inspect Stage 1.

## Stage 2

**Run only after Stage 1 direct U-FNO is approved.**

The residual U-FNO uses the canonical additive reconstruction:

```text
source_superposition_base_K
+ total_power_W * delta_R_eff_pred_K_per_W
+ zero_mean_centered_field_K
```

Run on GT:

```bash
cd /nethome/$USER/chiptherm
source .venv/bin/activate
export CHIPTHERM_REPO="$PWD"
export PYTHONPATH="$CHIPTHERM_REPO/src:$CHIPTHERM_REPO"
export CHIPTHERM_V2_DATA_ROOT="/export/hdd/$USER/chiptherm/benchmark_v2_50family"
export SOURCE_VERSION=source_superposition_final_train40_source_v1
export PREFLIGHT=outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json
export EXPERIMENT_ROOT="/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family"
export DIRECT_UFNO_RUN="$EXPERIMENT_ROOT/ufno/direct_temperature_ufno_normalized_train40_seed1"
export RESIDUAL_UFNO_RUN="$EXPERIMENT_ROOT/ufno/residual_ufno_decomposed_train40_seed1"
export FULL_COMPARE="$EXPERIMENT_ROOT/comparisons/operator_cnn_fno_ufno_seed1"

python3 scripts/train_benchmark_v2_fno.py \
  --experiment residual_ufno \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --config configs/benchmark_v2_50family/training/package_residual_ufno_decomposed_seed1.yaml \
  --preflight-report "$PREFLIGHT" \
  --out-dir "$RESIDUAL_UFNO_RUN" \
  --run-id residual_ufno_decomposed_train40_seed1 \
  --device cuda \
  --workers 4 \
  --seed 1

python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$RESIDUAL_UFNO_RUN/checkpoints/best.pt" \
  --out-dir "$RESIDUAL_UFNO_RUN/evaluation" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --profile-components \
  --save-predictions

python3 scripts/compare_benchmark_v2_operator_models.py \
  --source-only-root outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1 \
  --direct-cnn-root outputs/benchmark_v2_50family/package_direct/direct_temperature_feature_fusion_normalized_train40_seed1/evaluation \
  --residual-cnn-root outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/evaluation \
  --direct-fno-root outputs/benchmark_v2_50family/fno/direct_temperature_fno_normalized_train40_seed1/evaluation \
  --residual-fno-root outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1/evaluation \
  --direct-ufno-root "$DIRECT_UFNO_RUN/evaluation" \
  --residual-ufno-root "$RESIDUAL_UFNO_RUN/evaluation" \
  --out-dir "$FULL_COMPARE"
```

Rsync Stage-2 results back to the Mac:

```bash
rsync -az --info=progress2 \
  "$GT_HOST:/export/hdd/$GT_USER/chiptherm/experiment_outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1/" \
  /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1/
rsync -az --info=progress2 \
  "$GT_HOST:/export/hdd/$GT_USER/chiptherm/experiment_outputs/benchmark_v2_50family/comparisons/operator_cnn_fno_ufno_seed1/" \
  /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/comparisons/operator_cnn_fno_ufno_seed1/
```
