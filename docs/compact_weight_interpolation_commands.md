# Compact Weight-Interpolation Commands

The alpha grid is frozen to `0.00, 0.25, 0.50, 0.75, 1.00`. These commands
never train a model. Primary test is unavailable until validation analysis is
frozen and selects one interior alpha.

## 1. Sync Implementation To GT

Run from the Mac repository:

```bash
rsync -av --relative \
  src/chiptherm/compact_weight_interpolation.py \
  scripts/build_compact_weight_interpolation.py \
  scripts/analyze_compact_weight_interpolation.py \
  tests/test_compact_weight_interpolation.py \
  docs/compact_weight_interpolation_commands.md \
  jjun49@chao-srv1.ece.gatech.edu:/nethome/jjun49/chiptherm_test/
```

## 2. GT Environment

```bash
cd /nethome/jjun49/chiptherm_test
PY=/tmp/$USER/chiptherm_venv311_clean/bin/python3
DATA=/export/hdd/$USER/chiptherm/benchmark_v2_50family
EXP=/export/hdd/$USER/chiptherm/experiment_outputs/benchmark_v2_50family/interpolation_capacity
SRC=source_superposition_final_train40_source_v1
CANON=outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1
COS=$EXP/feature_fusion_train40_cosine_ema_seed1
SOUP=outputs/benchmark_v2_50family/compact_weight_interpolation
ALPHAS="0.00 0.25 0.50 0.75 1.00"
```

## 3. Dry-Run Compatibility Inspection

```bash
$PY scripts/build_compact_weight_interpolation.py \
  --canonical-checkpoint "$CANON/checkpoints/best.pt" \
  --cosine-checkpoint "$COS/checkpoints/epoch_0100.pt" \
  --out-root "$SOUP" \
  --alphas $ALPHAS \
  --dry-run
```

Expected: 102 compatible FP32 states, no non-floating or BatchNorm state.

## 4. Build Five Interpolated Checkpoints

```bash
$PY scripts/build_compact_weight_interpolation.py \
  --canonical-checkpoint "$CANON/checkpoints/best.pt" \
  --cosine-checkpoint "$COS/checkpoints/epoch_0100.pt" \
  --out-root "$SOUP" \
  --alphas $ALPHAS \
  --execute
```

## 5. Verify Checkpoint Hashes And Metadata

```bash
$PY scripts/build_compact_weight_interpolation.py \
  --canonical-checkpoint "$CANON/checkpoints/best.pt" \
  --cosine-checkpoint "$COS/checkpoints/epoch_0100.pt" \
  --out-root "$SOUP" \
  --alphas $ALPHAS \
  --verify-existing

cat "$SOUP/checkpoint_verification_report.json"
```

## 6. Evaluate All Five Alphas On Validation-Side Protocols

```bash
for RUN in \
  compact_soup_alpha000 \
  compact_soup_alpha025 \
  compact_soup_alpha050 \
  compact_soup_alpha075 \
  compact_soup_alpha100
do
  $PY scripts/evaluate_benchmark_v2_models.py \
    --data-root "$DATA" \
    --source-version "$SRC" \
    --checkpoint "$SOUP/$RUN/checkpoints/interpolated.pt" \
    --out-dir "$SOUP/$RUN/evaluation_validation" \
    --protocols known_family_sample_test primary_validation_families \
    --weights raw \
    --batch-size 64 \
    --device cuda \
    --workers 4 \
    --save-predictions
done
```

No primary-test protocol appears in this loop.

## 7. Run Endpoint Reproduction Checks

```bash
$PY scripts/analyze_compact_weight_interpolation.py \
  --experiment-root "$SOUP" \
  --canonical-eval-root "$CANON/evaluation" \
  --cosine-eval-root "$COS/evaluation_epoch0100_ema" \
  --out-dir "$SOUP"

cat "$SOUP/endpoint_reproduction_report.json"
```

Alpha 0 must reproduce canonical raw metrics and alpha 1 must reproduce
epoch-100 EMA metrics within `1e-4 K`; endpoint state tensors are bit-exact.

## 8. Generate Validation-Only Report

```bash
$PY scripts/analyze_compact_weight_interpolation.py \
  --experiment-root "$SOUP" \
  --canonical-eval-root "$CANON/evaluation" \
  --cosine-eval-root "$COS/evaluation_epoch0100_ema" \
  --out-dir "$SOUP"
```

## 9. Inspect Results

```bash
cat "$SOUP/compact_weight_interpolation_report.md"
cat "$SOUP/selected_candidate.json"
```

## 10. Freeze Validation And Candidate Selection

```bash
$PY scripts/analyze_compact_weight_interpolation.py \
  --experiment-root "$SOUP" \
  --canonical-eval-root "$CANON/evaluation" \
  --cosine-eval-root "$COS/evaluation_epoch0100_ema" \
  --out-dir "$SOUP" \
  --freeze-validation

cat "$SOUP/validation_decision_gate.json"
```

If no interior candidate qualifies, stop here.

## 11. Evaluate The Selected Interior Candidate On Primary Test

```bash
SELECTED_RUN=$($PY -c \
  'import json; print(json.load(open("outputs/benchmark_v2_50family/compact_weight_interpolation/selected_candidate.json"))["selected_run_id"])')

$PY scripts/evaluate_benchmark_v2_models.py \
  --data-root "$DATA" \
  --source-version "$SRC" \
  --checkpoint "$SOUP/$SELECTED_RUN/checkpoints/interpolated.pt" \
  --out-dir "$SOUP/$SELECTED_RUN/evaluation_primary_test" \
  --protocols primary_test_families \
  --weights raw \
  --batch-size 64 \
  --device cuda \
  --workers 4 \
  --save-predictions
```

## 12. Generate Final Report

```bash
$PY scripts/analyze_compact_weight_interpolation.py \
  --experiment-root "$SOUP" \
  --canonical-eval-root "$CANON/evaluation" \
  --cosine-eval-root "$COS/evaluation_epoch0100_ema" \
  --out-dir "$SOUP" \
  --include-primary-test
```

This fails if the validation fingerprint changed after freezing.

## 13. Sync Results Back To Mac

```bash
rsync -avh --progress \
  jjun49@chao-srv1.ece.gatech.edu:/nethome/jjun49/chiptherm_test/outputs/benchmark_v2_50family/compact_weight_interpolation/ \
  outputs/benchmark_v2_50family/compact_weight_interpolation/
```

## Frozen Selection Thresholds

- Known-family MAE: at most `0.135 K`.
- Held-out validation MAE: at most `0.940 K`.
- Per-family validation regression: at most `0.10 K`.
- Fraction worse than source: canonical plus at most `0.01` absolute.
- Absolute hotspot-temperature error: canonical plus at most `0.05 K`.
- Candidates within `0.002 K` of best known-family MAE form a tie group;
  lower validation MAE wins.
