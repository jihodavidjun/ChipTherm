# Benchmark v2 Final Training Pipeline

This runbook is the authoritative post-dataset sequence for
`benchmark_v2_50family`, stage `full_50x200`. It never regenerates package
HotSpot labels and never mutates accepted pilot or canonical artifacts.

## Scientific Contract

- Primary family split:
  - train: 40 families, 8,000 packages
  - validation: `f007 f012 f023 f030 f041`
  - test: `f008 f016 f027 f033 f044`
- Secondary sample split inside the 40 train families: 6,400 / 800 / 800.
- Source fitting and source-model checkpoint selection use only source targets
  from primary train families. The primary validation and test source targets
  are oracle evaluation only after the source checkpoint is frozen.
- The final package model is
  `miniunet_refine_conditioned_decomposed_feature_fusion` with:

  ```text
  T_pred =
      source_superposition_base
      + total_power_W * delta_R_eff_pred_K_per_W
      + zero_mean_centered_spatial_pred_K
  ```

- The optional GNN freezes that exact CNN and predicts only a zero-mean graph
  correction. It is omitted unless every promotion criterion passes.

## Paths

On the server:

```bash
export CHIPTHERM_REPO=/nethome/$USER/chiptherm
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
export PYTHONPATH="$CHIPTHERM_REPO/src:$CHIPTHERM_REPO"
cd "$CHIPTHERM_REPO"
```

The final source version is:

```bash
export SOURCE_RUN=outputs/benchmark_v2_50family/source_response/final_train40_v1
export SOURCE_VERSION=source_superposition_final_train40_source_v1
export SOURCE_VERSION_ROOT="$CHIPTHERM_V2_DATA_ROOT/derived/stages/full_50x200/$SOURCE_VERSION"
export CNN_RUN=outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1
export GNN_RUN=outputs/benchmark_v2_50family/package_residual/optional_gnn_train40_source_v1_seed1
```

## A. Rsync Implementation

Run on the Mac:

```bash
export GT_HOST=<user>@<gt-host>
export GT_REPO=/nethome/<user>/chiptherm
rsync -az --info=progress2 \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude 'outputs/' \
  /Users/jihojun/chiptherm/ \
  "$GT_HOST:$GT_REPO/"
```

## B. Activate Environment

```bash
ssh "$GT_HOST"
cd /nethome/$USER/chiptherm
source .venv/bin/activate
export CHIPTHERM_REPO="$PWD"
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
export PYTHONPATH="$CHIPTHERM_REPO/src:$CHIPTHERM_REPO"
export SOURCE_RUN=outputs/benchmark_v2_50family/source_response/final_train40_v1
export SOURCE_VERSION=source_superposition_final_train40_source_v1
export SOURCE_VERSION_ROOT="$CHIPTHERM_V2_DATA_ROOT/derived/stages/full_50x200/$SOURCE_VERSION"
export CNN_RUN=outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1
export GNN_RUN=outputs/benchmark_v2_50family/package_residual/optional_gnn_train40_source_v1_seed1
```

## C. Compile and Unit Tests

```bash
python3 -m py_compile \
  src/chiptherm/benchmark_v2_training.py \
  src/chiptherm/ml/source_response_dataset.py \
  scripts/preflight_benchmark_v2_training.py \
  scripts/train_benchmark_v2_source_response.py \
  scripts/evaluate_benchmark_v2_source_response.py \
  scripts/approve_benchmark_v2_source_response.py \
  scripts/regenerate_benchmark_v2_source_superposition.py \
  scripts/validate_benchmark_v2_source_superposition.py \
  scripts/train_benchmark_v2_package_residual.py \
  scripts/evaluate_benchmark_v2_models.py \
  scripts/train_benchmark_v2_optional_gnn.py \
  scripts/compare_benchmark_v2_models.py

python3 tests/test_benchmark_v2_final_training.py
python3 tests/test_source_response_dataset.py
python3 tests/test_source_response_model.py
python3 tests/test_residual_resistance_mean_head.py
python3 tests/test_feature_fusion_model.py
python3 tests/test_graph_rasterizer.py
```

## D. Training Preflight

This creates the immutable source fit/internal-validation/oracle indices and
the deterministic 5/10/20/30/40-family scaling subsets.

```bash
python3 scripts/preflight_benchmark_v2_training.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --out-dir outputs/benchmark_v2_50family/preflight/full_50x200 \
  --seed 20260721
```

Do not continue unless `preflight_report.json` has `passed: true`.

## E. Source-Response Smoke Test

```bash
python3 scripts/train_benchmark_v2_source_response.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --out-dir outputs/benchmark_v2_50family/source_response/smoke_train40_v1 \
  --run-id smoke_train40_v1 \
  --device cuda \
  --workers 4 \
  --seed 1 \
  --smoke-test
```

## F. Final Source-Response Training

```bash
python3 scripts/train_benchmark_v2_source_response.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --config configs/benchmark_v2_50family/training/source_response_final_train40_v1.yaml \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --out-dir "$SOURCE_RUN" \
  --run-id final_train40_v1 \
  --device cuda \
  --workers 4 \
  --seed 1
```

`best.pt` is selected only by reconstructed package full-grid MAE on the
internal validation families drawn from the 40-family training pool.
After an interrupted run has written `checkpoints/last.pt`, rerun the same
command with `--resume`.

## G. Source-Response Evaluation

Final checkpoint:

```bash
python3 scripts/evaluate_benchmark_v2_source_response.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
  --out-dir "$SOURCE_RUN/evaluation" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

Prototype comparator on the same frozen oracle indices:

```bash
python3 scripts/evaluate_benchmark_v2_source_response.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --checkpoint outputs/source_response_operator_v1/prototype_seed1/checkpoints/best.pt \
  --out-dir outputs/benchmark_v2_50family/source_response/prototype_seed1_evaluation \
  --batch-size 64 \
  --device cuda \
  --workers 4
```

## H. Qualitative Audit, Approval, and Freeze

Generate contact sheets:

```bash
python3 scripts/visualize_benchmark_v2_source_response.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --evaluation-root "$SOURCE_RUN/evaluation" \
  --out-dir "$SOURCE_RUN/evaluation/qualitative"
```

After manually inspecting target, prediction, signed residual, absolute
residual, geometry, and active-source placement, sign the audit:

```bash
python3 scripts/visualize_benchmark_v2_source_response.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --evaluation-root "$SOURCE_RUN/evaluation" \
  --out-dir "$SOURCE_RUN/evaluation/qualitative" \
  --reviewed
```

Approve and freeze:

```bash
python3 scripts/approve_benchmark_v2_source_response.py \
  --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
  --training-lineage "$SOURCE_RUN/training_lineage.json" \
  --evaluation-root "$SOURCE_RUN/evaluation" \
  --prototype-metrics outputs/benchmark_v2_50family/source_response/prototype_seed1_evaluation/oracle_primary_test/metrics.json \
  --out-file "$SOURCE_RUN/approval.json"
```

The approval record is immutable. Its checkpoint hash is checked by every
later source-version stage.

## I. Source-Superposition Regeneration Dry Run

```bash
python3 scripts/regenerate_benchmark_v2_source_superposition.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
  --source-lineage "$SOURCE_RUN/approval.json" \
  --approval-file "$SOURCE_RUN/approval.json" \
  --artifact-name "$SOURCE_VERSION" \
  --package-batch-size 8 \
  --source-batch-size 64 \
  --device cuda \
  --run-id final-train40-source-v1 \
  --dry-run
```

## J. Full 10,000-Package Source-Superposition Regeneration

```bash
python3 scripts/regenerate_benchmark_v2_source_superposition.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
  --source-lineage "$SOURCE_RUN/approval.json" \
  --approval-file "$SOURCE_RUN/approval.json" \
  --artifact-name "$SOURCE_VERSION" \
  --package-batch-size 8 \
  --source-batch-size 64 \
  --device cuda \
  --run-id final-train40-source-v1 \
  --resume
```

No HotSpot executable is accepted or invoked by this command. The provisional
source maps are a separate immutable directory.

## K. Strict Source-Version Validation

```bash
python3 scripts/validate_benchmark_v2_source_superposition.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-root "$SOURCE_VERSION_ROOT" \
  --checkpoint "$SOURCE_RUN/checkpoints/best.pt" \
  --approval-file "$SOURCE_RUN/approval.json" \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --out-dir outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1 \
  --spot-check-count 50 \
  --source-batch-size 64 \
  --device cuda \
  --seed 1
```

This checks all 10,000 maps, checkpoint/normalization/lineage hashes,
root-relative paths, source accounting, representative regeneration, and the
preflight immutability snapshot.

## L. Source-Superposition-Only Baseline

Stage K already computes this baseline. To rebuild only the report:

```bash
python3 scripts/evaluate_full_source_superposition_base.py \
  --source-root "$SOURCE_VERSION_ROOT" \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --out-dir outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1
```

## M. Residual-CNN Smoke Test

```bash
python3 scripts/train_benchmark_v2_package_residual.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --out-dir outputs/benchmark_v2_50family/package_residual/smoke_feature_fusion_source_v1 \
  --run-id smoke_feature_fusion_source_v1 \
  --device cuda \
  --workers 4 \
  --seed 1 \
  --smoke-test
```

## N. Final Residual-CNN Training

```bash
python3 scripts/train_benchmark_v2_package_residual.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --config configs/benchmark_v2_50family/training/package_residual_feature_fusion_v1.yaml \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --out-dir "$CNN_RUN" \
  --run-id feature_fusion_train40_source_v1_seed1 \
  --device cuda \
  --workers 4 \
  --seed 1
```

Only the 6,400/800 secondary train/validation rows select this checkpoint.
Primary held-out families do not enter optimization, normalization, scheduling,
or early stopping.
After an interrupted run has written `checkpoints/last.pt`, rerun with
`--resume`.

## O. Residual-CNN Evaluation

The wrapper evaluates known-family sample test, primary validation families,
and primary test families separately:

```bash
python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$CNN_RUN/checkpoints/best.pt" \
  --out-dir "$CNN_RUN/evaluation" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --profile-components \
  --save-predictions \
  --error-analysis
```

## P. Model Comparison

```bash
python3 scripts/compare_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --source-baseline-dir outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1 \
  --cnn-eval-root "$CNN_RUN/evaluation" \
  --out-dir outputs/benchmark_v2_50family/comparisons/final_train40_source_v1
```

When compatible context-only and provisional-source residual checkpoints have
been evaluated through the same three protocol directories, add:

```text
--context-cnn-eval-root <context-evaluation-root>
--provisional-cnn-eval-root <provisional-evaluation-root>
```

The report always includes the ambient, final source-only, and final
source-plus-CNN baselines. It labels known-family and strict family-held-out
results separately.

## Q. Optional GNN Smoke and Training

Smoke:

```bash
python3 scripts/train_benchmark_v2_optional_gnn.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --cnn-checkpoint "$CNN_RUN/checkpoints/best.pt" \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --out-dir outputs/benchmark_v2_50family/package_residual/optional_gnn_smoke \
  --device cuda \
  --workers 4 \
  --seed 1 \
  --smoke-test
```

Training, only after reviewing the CNN result:

```bash
python3 scripts/train_benchmark_v2_optional_gnn.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --cnn-checkpoint "$CNN_RUN/checkpoints/best.pt" \
  --preflight-report outputs/benchmark_v2_50family/preflight/full_50x200/preflight_report.json \
  --config configs/benchmark_v2_50family/training/optional_gnn_v1.yaml \
  --out-dir "$GNN_RUN" \
  --device cuda \
  --workers 4 \
  --seed 1
```

## R. GNN Evaluation and Promotion Gate

```bash
python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint "$GNN_RUN/checkpoints/best.pt" \
  --out-dir "$GNN_RUN/evaluation" \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --profile-components

python3 scripts/compare_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --source-baseline-dir outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1 \
  --cnn-eval-root "$CNN_RUN/evaluation" \
  --gnn-eval-root "$GNN_RUN/evaluation" \
  --gnn-runtime-overhead-fraction <measured-fraction> \
  --gnn-memory-overhead-fraction <measured-fraction> \
  --out-dir outputs/benchmark_v2_50family/comparisons/final_train40_source_v1_with_gnn
```

Promotion requires: at least 0.10 K and 2% primary-test MAE improvement, at
least three of five test families improved, no material RMSE or peak regression,
justified runtime/memory overhead, and a positive paired-bootstrap 95% lower
bound. Otherwise the report states `OMIT GNN FROM PRIMARY MODEL`.

## S. Rsync Reports and Checkpoints Back

Run on the Mac:

```bash
mkdir -p /Users/jihojun/chiptherm/outputs/benchmark_v2_50family
rsync -az --info=progress2 \
  "$GT_HOST:$GT_REPO/outputs/benchmark_v2_50family/" \
  /Users/jihojun/chiptherm/outputs/benchmark_v2_50family/
```

## Scaling Support

Both source and residual wrappers accept:

```text
--train-family-count 5|10|20|30|40
```

Subsets are nested, SHA-256 ordered from the fixed preflight seed, and stored
under the benchmark data root. Primary held-out test performance never chooses
their composition.

## Expected Cost

- Preflight/index generation: seconds to a few minutes; negligible new storage.
- Source training: only the family-level isolated target inventory (roughly a
  few hundred sources), so typically much cheaper than package-CNN training.
- Final source regeneration: approximately one source inference per active
  chiplet across 10,000 packages. Two float32 64×64 maps per package add about
  0.31 GiB before sidecars and indices.
- Residual CNN: the dominant training stage, 100 epochs over 6,400 packages.
- Optional frozen-CNN GNN: similar data passes but fewer trainable parameters;
  it remains scientifically optional.

Actual wall time, throughput, GPU memory, and storage are recorded by each
stage. No accuracy improvement is assumed until server evaluation completes.
