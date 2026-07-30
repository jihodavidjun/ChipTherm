# Integrated ChipTherm Runtime Commands

These commands benchmark the complete uncached source-response plus compact
residual-CNN pipeline. HotSpot is only the recorded CPU reference
(`4.943711 s/package`); it is not launched.

## 1. Sync implementation to GT

Run on the Mac repository:

```bash
export GT_HOST="jjun49@chao-srv1.ece.gatech.edu"
export GT_REPO="/nethome/jjun49/chiptherm_test"
rsync -av --relative \
  src/chiptherm/ml/dataset.py \
  src/chiptherm/ml/integrated_inference.py \
  scripts/evaluate_integrated_chiptherm.py \
  scripts/validate_integrated_chiptherm_equivalence.py \
  scripts/profile_integrated_chiptherm.py \
  tests/test_integrated_inference.py \
  tests/test_integrated_chiptherm_inference.py \
  tests/test_integrated_chiptherm_runtime.py \
  docs/integrated_chiptherm_runtime_commands.md \
  "$GT_HOST:$GT_REPO/"
```

## 2. Environment and paths

Run on GT:

```bash
cd /nethome/$USER/chiptherm_test
export PYTHON=/tmp/$USER/chiptherm_venv311_rebuilt/bin/python3
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
export SOURCE_VERSION=source_superposition_final_train40_source_v1
export SOURCE_CKPT=outputs/benchmark_v2_50family/source_response/final_train40_v1/checkpoints/best.pt
export RESIDUAL_CKPT=outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/checkpoints/best.pt
export SOURCE_INDEX_ROOT="$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/$SOURCE_VERSION"
export TEST_INDEX="$SOURCE_INDEX_ROOT/sample_split/test_index.csv"
export OUT=outputs/benchmark_v2_50family/integrated_runtime

"$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$PYTHON" -m py_compile \
  src/chiptherm/ml/dataset.py \
  src/chiptherm/ml/integrated_inference.py \
  scripts/evaluate_integrated_chiptherm.py \
  scripts/validate_integrated_chiptherm_equivalence.py \
  scripts/profile_integrated_chiptherm.py
```

## 3. Strict equivalence gate

```bash
"$PYTHON" scripts/validate_integrated_chiptherm_equivalence.py \
  --source-checkpoint "$SOURCE_CKPT" \
  --residual-checkpoint "$RESIDUAL_CKPT" \
  --index "$TEST_INDEX" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --compare-cached-index "$TEST_INDEX" \
  --out-dir "$OUT/equivalence" \
  --max-samples 8 \
  --source-batch-size 64 \
  --device cuda \
  --seed 1
```

The reference-versus-optimized gate is `max_abs <= 1e-5 K` and aggregate MAE
difference `<= 1e-6 K`. The cached-map comparison is separately labeled and
uses the historical `0.05 K` CUDA accumulation tolerance.

## 4. Reference profile

```bash
"$PYTHON" scripts/profile_integrated_chiptherm.py \
  --source-checkpoint "$SOURCE_CKPT" \
  --residual-checkpoint "$RESIDUAL_CKPT" \
  --index "$TEST_INDEX" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --out-dir "$OUT/profile_reference" \
  --mode reference \
  --package-batch-size 8 \
  --source-batch-size 64 \
  --warmup-batches 3 \
  --profile-batches 10 \
  --device cuda
```

## 5. Optimized profile

The optimized mode remains FP32 and retains host float64 source accumulation.
Its initial candidate is `torch.inference_mode`; pinned asynchronous transfer
is also measured as a candidate. Neither should be reported as accepted until
the strict equivalence gate and GT timing both pass.

```bash
"$PYTHON" scripts/profile_integrated_chiptherm.py \
  --source-checkpoint "$SOURCE_CKPT" \
  --residual-checkpoint "$RESIDUAL_CKPT" \
  --index "$TEST_INDEX" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --out-dir "$OUT/profile_optimized" \
  --mode optimized \
  --package-batch-size 8 \
  --source-batch-size 64 \
  --warmup-batches 3 \
  --profile-batches 10 \
  --device cuda
```

## 6. Full representative batch sweep

Reference:

```bash
"$PYTHON" scripts/evaluate_integrated_chiptherm.py \
  --source-checkpoint "$SOURCE_CKPT" \
  --residual-checkpoint "$RESIDUAL_CKPT" \
  --index "$TEST_INDEX" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --compare-cached-index "$TEST_INDEX" \
  --out-dir "$OUT/reference_full" \
  --mode reference \
  --package-batch-size 1 8 16 32 64 \
  --source-batch-size 64 \
  --warmup-batches 3 \
  --device cuda \
  --num-workers 0 \
  --profile-components
```

Optimized:

```bash
"$PYTHON" scripts/evaluate_integrated_chiptherm.py \
  --source-checkpoint "$SOURCE_CKPT" \
  --residual-checkpoint "$RESIDUAL_CKPT" \
  --index "$TEST_INDEX" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --compare-cached-index "$TEST_INDEX" \
  --out-dir "$OUT/optimized_full" \
  --mode optimized \
  --package-batch-size 1 8 16 32 64 \
  --source-batch-size 64 \
  --warmup-batches 3 \
  --device cuda \
  --num-workers 0 \
  --profile-components
```

Both commands traverse the complete selected index for every batch size.
`integrated_runtime_by_source_count.csv` records source-count coverage; no
ordered easy prefix is labeled authoritative.

## 7. Inspect and retrieve reports

```bash
jq . "$OUT/equivalence/integrated_equivalence_report.json"
column -s, -t < "$OUT/optimized_full/integrated_runtime_metrics.csv" | less -S
head -80 "$OUT/profile_optimized/profiler_summary.txt"
```

Run on the Mac:

```bash
rsync -av \
  "$GT_HOST:$GT_REPO/outputs/benchmark_v2_50family/integrated_runtime/" \
  outputs/benchmark_v2_50family/integrated_runtime/ \
  --exclude='predictions/'
```
