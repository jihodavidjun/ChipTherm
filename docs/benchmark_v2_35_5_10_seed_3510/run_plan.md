# GT Execution Plan

No command in this document should target the primary 40/5/5 output roots.
Set the roots once:

```bash
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
export EXP_ROOT=/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_family_35_5_10_seed_3510
export PROTOCOL=benchmark_v2_family_35_5_10_seed_3510
export PROTO_ROOT="$CHIPTHERM_V2_DATA_ROOT/derived/protocols/$PROTOCOL"
export SOURCE_RUN="$EXP_ROOT/source_response/source_v1_seed1"
export SOURCE_VERSION=source_superposition_family35_seed3510_source_v1
export SOURCE_VERSION_ROOT="$PROTO_ROOT/source_superposition_artifacts/$SOURCE_VERSION"
export RESIDUAL_INDEX_ROOT="$PROTO_ROOT/source_superposition/$SOURCE_VERSION"
```

## 1. Validate raw Benchmark v2

```bash
python3 scripts/validate_benchmark_v2_full.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --require-relocation
```

## 2. Generate and validate protocol indices

```bash
python3 scripts/build_benchmark_v2_secondary_protocol.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --config-manifest configs/benchmark_v2_family_35_5_10_seed_3510/split_manifest.json \
  --seed 3510 \
  --out-root "$PROTO_ROOT"
```

## 3. Build the source train-only normalizer preview

```bash
python3 scripts/build_benchmark_v2_protocol_normalizers.py \
  --mode source \
  --train-index "$PROTO_ROOT/source_response/train_index.csv" \
  --protocol-index-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --out-dir "$EXP_ROOT/normalizers" \
  --batch-size 64 \
  --num-workers 4
```

## 4. Train source-response model

```bash
test ! -e "$SOURCE_RUN/checkpoints/best.pt"
python3 scripts/train_source_response_model.py \
  --train-index "$PROTO_ROOT/source_response/train_index.csv" \
  --val-index "$PROTO_ROOT/source_response/internal_val_index.csv" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --out-dir "$SOURCE_RUN" \
  --epochs 100 \
  --batch-size 64 \
  --packages-per-batch 1 \
  --lr 1e-3 \
  --base-channels 32 \
  --depth 3 \
  --lambda-source 1.0 \
  --lambda-package 0.0 \
  --package-loss-warmup-epochs 0 \
  --lambda-source-mean 0.0 \
  --scheduler plateau \
  --early-stopping-patience 20 \
  --checkpoint-frequency 10 \
  --lineage-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --device cuda \
  --num-workers 4 \
  --seed 1
```

Evaluate source quality only on the fit and held-in internal-validation
partitions, then freeze the checkpoint identity. Held-out source metrics are
deferred so they cannot influence source-model selection.

```bash
for part in train internal_val; do
  python3 scripts/evaluate_source_response_model.py \
    --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
    --source-index "$PROTO_ROOT/source_response/${part}_index.csv" \
    --data-root "$CHIPTHERM_V2_DATA_ROOT" \
    --out-dir "$SOURCE_RUN/evaluation/$part" \
    --batch-size 64 --device cuda --num-workers 4 --profile-runtime
done

test ! -e "$SOURCE_RUN/source_selection_frozen.txt"
printf '%s\n' \
  "protocol=$PROTOCOL" \
  "checkpoint=$(sha256sum "$SOURCE_RUN/checkpoints/best.pt" | cut -d' ' -f1)" \
  "selection_used=train,internal_val" \
  "heldout_validation_used=false" \
  "heldout_test_used=false" > "$SOURCE_RUN/source_selection_frozen.txt"
chmod a-w "$SOURCE_RUN/source_selection_frozen.txt"
```

## 5. Generate protocol-specific source-superposition maps

This performs source-model inference only. It does not call HotSpot.

```bash
test -f "$SOURCE_RUN/source_selection_frozen.txt"
test ! -e "$SOURCE_VERSION_ROOT/manifest.json"
python3 scripts/build_full_source_superposition_base.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --train-index "$PROTO_ROOT/package/source_generation/train_index.csv" \
  --val-index "$PROTO_ROOT/package/source_generation/val_index.csv" \
  --test-index "$PROTO_ROOT/package/source_generation/test_index.csv" \
  --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
  --out-root "$SOURCE_VERSION_ROOT" \
  --package-batch-size 8 \
  --source-batch-size 64 \
  --device cuda \
  --resume \
  --seed 1
```

Attach the new map paths to protocol views by `sample_uid`:

```bash
python3 scripts/build_benchmark_v2_secondary_protocol.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --config-manifest configs/benchmark_v2_family_35_5_10_seed_3510/split_manifest.json \
  --seed 3510 \
  --out-root "$PROTO_ROOT" \
  --source-version-root "$SOURCE_VERSION_ROOT"
```

## 6. Audit package train-only normalizers

```bash
python3 scripts/build_benchmark_v2_protocol_normalizers.py \
  --mode residual \
  --train-index "$RESIDUAL_INDEX_ROOT/sample_split/train_index.csv" \
  --protocol-index-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --out-dir "$EXP_ROOT/normalizers" --batch-size 64 --num-workers 4

python3 scripts/build_benchmark_v2_protocol_normalizers.py \
  --mode direct \
  --train-index "$PROTO_ROOT/package/sample_split/train_index.csv" \
  --protocol-index-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --out-dir "$EXP_ROOT/normalizers" --batch-size 64 --num-workers 4
```

Each training command below recomputes its normalizer from exactly the same
train index and stores it in the run directory.

## 7. Train ChipTherm

```bash
export CHIP_RUN="$EXP_ROOT/chiptherm/seed1"
test ! -e "$CHIP_RUN/checkpoints/best.pt"
python3 scripts/train_residual_cnn.py \
  --train-index "$RESIDUAL_INDEX_ROOT/sample_split/train_index.csv" \
  --val-index "$RESIDUAL_INDEX_ROOT/sample_split/val_index.csv" \
  --out-dir "$CHIP_RUN" --epochs 100 --batch-size 64 --lr 1e-3 \
  --weight-decay 1e-2 --base-channels 32 \
  --model-architecture miniunet_refine_conditioned_decomposed_feature_fusion \
  --metadata-conditioning --metadata-hidden-dim 64 --metadata-embedding-dim 64 \
  --refine-channels 32 --refine-blocks 4 \
  --physics-input source_superposition_v1 --mean-head-mode residual_resistance \
  --physical-representation dimensional --channel-routing-mode dimensional_baseline \
  --lambda-final 1.0 --lambda-mean 0.1 \
  --global-hidden-channels 32 --global-pool-size 8 \
  --scheduler none --early-stopping-patience 20 --checkpoint-frequency 10 \
  --lineage-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --device cuda --num-workers 4 --seed 1
```

## 8. Train CNN

```bash
export CNN_RUN="$EXP_ROOT/cnn/seed1"
test ! -e "$CNN_RUN/checkpoints/best.pt"
python3 scripts/train_residual_cnn.py \
  --train-index "$PROTO_ROOT/package/sample_split/train_index.csv" \
  --val-index "$PROTO_ROOT/package/sample_split/val_index.csv" \
  --out-dir "$CNN_RUN" --epochs 100 --batch-size 64 --lr 1e-3 \
  --weight-decay 1e-2 --base-channels 32 \
  --model-architecture miniunet_refine_conditioned_direct_temperature_feature_fusion \
  --prediction-mode direct_temperature --direct-target-normalization train_standard \
  --metadata-conditioning --metadata-hidden-dim 64 --metadata-embedding-dim 64 \
  --refine-channels 32 --refine-blocks 4 --physics-input none \
  --physical-representation dimensional --channel-routing-mode dimensional_baseline \
  --global-hidden-channels 32 --global-pool-size 8 \
  --scheduler none --early-stopping-patience 20 --checkpoint-frequency 10 \
  --lineage-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --device cuda --num-workers 4 --seed 1
```

## 9. Train FNO

```bash
export FNO_RUN="$EXP_ROOT/fno/seed1"
test ! -e "$FNO_RUN/checkpoints/best.pt"
python3 scripts/train_residual_cnn.py \
  --train-index "$RESIDUAL_INDEX_ROOT/sample_split/train_index.csv" \
  --val-index "$RESIDUAL_INDEX_ROOT/sample_split/val_index.csv" \
  --out-dir "$FNO_RUN" --epochs 100 --batch-size 64 --lr 1e-3 \
  --weight-decay 1e-2 --base-channels 32 \
  --model-architecture fno2d_residual_decomposed_conditioned \
  --prediction-mode residual_decomposed_fno \
  --metadata-conditioning --metadata-hidden-dim 64 --metadata-embedding-dim 64 \
  --physics-input source_superposition_v1 --mean-head-mode residual_resistance \
  --physical-representation dimensional --channel-routing-mode dimensional_baseline \
  --fno-capacity-profile fno_small --fno-width 32 --fno-layers 4 \
  --fno-modes-x 12 --fno-modes-y 12 --fno-activation gelu \
  --fno-metadata-conditioning film --fno-projection-channels 64 \
  --lambda-final 1.0 --lambda-mean 0.1 \
  --scheduler none --early-stopping-patience 20 --checkpoint-frequency 10 \
  --lineage-manifest "$PROTO_ROOT/protocol_index_manifest.json" \
  --device cuda --num-workers 4 --seed 1
```

## 10. Evaluate familiar and held-out validation before final test

```bash
for item in \
  "familiar_family_sample_test sample_split/test_index.csv" \
  "heldout_validation_families family_split/val_index.csv"; do
  read -r protocol relative_index <<< "$item"
  python3 scripts/evaluate_residual_cnn.py \
    --checkpoint "$CNN_RUN/checkpoints/best.pt" \
    --index "$PROTO_ROOT/package/$relative_index" \
    --out-dir "$CNN_RUN/evaluation/$protocol" \
    --batch-size 64 --device cuda --num-workers 4 --measure-end-to-end --save-predictions
  python3 scripts/evaluate_residual_cnn.py \
    --checkpoint "$FNO_RUN/checkpoints/best.pt" \
    --index "$RESIDUAL_INDEX_ROOT/$relative_index" \
    --out-dir "$FNO_RUN/evaluation/$protocol" \
    --batch-size 64 --device cuda --num-workers 4 --measure-end-to-end --save-predictions
  python3 scripts/evaluate_residual_cnn.py \
    --checkpoint "$CHIP_RUN/checkpoints/best.pt" \
    --index "$RESIDUAL_INDEX_ROOT/$relative_index" \
    --out-dir "$CHIP_RUN/evaluation/$protocol" \
    --batch-size 64 --device cuda --num-workers 4 --measure-end-to-end --save-predictions
done
```

Inspect and freeze model/checkpoint choices, then create a simple immutable
operator record before touching final test:

```bash
test ! -e "$EXP_ROOT/selection_frozen.txt"
printf '%s\n' \
  "protocol=$PROTOCOL" \
  "cnn=$(sha256sum "$CNN_RUN/checkpoints/best.pt" | cut -d' ' -f1)" \
  "fno=$(sha256sum "$FNO_RUN/checkpoints/best.pt" | cut -d' ' -f1)" \
  "chiptherm=$(sha256sum "$CHIP_RUN/checkpoints/best.pt" | cut -d' ' -f1)" \
  "selection_used=familiar_family_sample_test,heldout_validation_families" \
  "final_test_used_for_selection=false" > "$EXP_ROOT/selection_frozen.txt"
chmod a-w "$EXP_ROOT/selection_frozen.txt"
```

Only then run final test:

```bash
test -f "$EXP_ROOT/selection_frozen.txt"
python3 scripts/evaluate_residual_cnn.py \
  --checkpoint "$CNN_RUN/checkpoints/best.pt" \
  --index "$PROTO_ROOT/package/family_split/test_index.csv" \
  --out-dir "$CNN_RUN/evaluation/heldout_final_test_families" \
  --batch-size 64 --device cuda --num-workers 4 --measure-end-to-end --save-predictions
python3 scripts/evaluate_residual_cnn.py \
  --checkpoint "$FNO_RUN/checkpoints/best.pt" \
  --index "$RESIDUAL_INDEX_ROOT/family_split/test_index.csv" \
  --out-dir "$FNO_RUN/evaluation/heldout_final_test_families" \
  --batch-size 64 --device cuda --num-workers 4 --measure-end-to-end --save-predictions
python3 scripts/evaluate_residual_cnn.py \
  --checkpoint "$CHIP_RUN/checkpoints/best.pt" \
  --index "$RESIDUAL_INDEX_ROOT/family_split/test_index.csv" \
  --out-dir "$CHIP_RUN/evaluation/heldout_final_test_families" \
  --batch-size 64 --device cuda --num-workers 4 --measure-end-to-end --save-predictions
```

## 11. Authoritative integrated ChipTherm evaluation

```bash
for item in \
  "familiar_family_sample_test sample_split/test_index.csv" \
  "heldout_validation_families family_split/val_index.csv" \
  "heldout_final_test_families family_split/test_index.csv"; do
  read -r protocol relative_index <<< "$item"
  if [[ "$protocol" == heldout_final_test_families ]]; then
    test -f "$EXP_ROOT/selection_frozen.txt"
  fi
  python3 scripts/evaluate_integrated_chiptherm.py \
    --source-checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
    --residual-checkpoint "$CHIP_RUN/checkpoints/best.pt" \
    --index "$PROTO_ROOT/package/$relative_index" \
    --data-root "$CHIPTHERM_V2_DATA_ROOT" \
    --compare-cached-index "$RESIDUAL_INDEX_ROOT/$relative_index" \
    --out-dir "$CHIP_RUN/integrated/$protocol" \
    --package-batch-size 8 --source-batch-size 64 \
    --device cuda --num-workers 4 --profile-components --save-predictions
done
```

## 12. Aggregate and report

```bash
python3 scripts/aggregate_benchmark_v2_secondary_protocol.py \
  --cnn-eval-root "$CNN_RUN/evaluation" \
  --fno-eval-root "$FNO_RUN/evaluation" \
  --chiptherm-eval-root "$CHIP_RUN/evaluation" \
  --out-dir docs/benchmark_v2_35_5_10_seed_3510
```

## Runtime estimate

Observed local copies of the canonical A6000 logs contain approximately 1.3
minutes of source-model epoch time (85 epochs), 16.4 minutes for the 100-epoch
ChipTherm CNN, 15.4 minutes for the normalized direct CNN, and 9.3 minutes for
the residual FNO. The 35-family runs process 87.5% of the canonical package
training rows, so epoch compute should be slightly lower. Allow 45-75 minutes
for the four sequential training runs including loader startup, normalizer
passes, checkpoints, and validation; allow another 10-25 minutes for all
source-map generation and cached/integrated evaluations. These are planning
ranges, not promised wall times, and should be replaced by measured manifests.
