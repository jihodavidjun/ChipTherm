# Familiar-Family Source-Baseline Recovery

## Recovered Result

The authoritative familiar-family full-map MAE for the frozen source-superposition baseline is **1.3879 K** (unrounded: `1.387925262451172 K`). The corresponding global full-map RMSE is **2.7843 K**.

This value was **extracted**, not recomputed and not inferred. It is stored in the `physics_baseline` block of:

```text
outputs/benchmark_v2_50family/package_residual/
feature_fusion_train40_source_v1_seed1/evaluation/
known_family_sample_test/metrics.json
```

The evaluator recorded:

```text
samples:       800
cells:         3,276,800
map shape:     64 x 64
MAE:           1.387925262451172 K
RMSE:          2.784344232220096 K
signed error: -0.024629910588264466 K
```

## Protocol And Artifacts

- Source version: `source_superposition_final_train40_source_v1`
- Frozen source-response checkpoint: `outputs/benchmark_v2_50family/source_response/final_train40_v1/checkpoints/best.pt`
- Checkpoint SHA-256: `249bfa021ac738c0644e9349e20317a4353434651fb6132a8a91c9e958512421`
- Familiar-family split: `$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/source_superposition_final_train40_source_v1/sample_split/test_index.csv`
- Authoritative split manifest: `$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/source_superposition_final_train40_source_v1/index_manifest.json`
- Split-manifest SHA-256: `2797a69a82e5d1c7aebd52babbb846275c36745eca6882ebce73b4a54b52530c`
- Per-sample audit: `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/evaluation/known_family_sample_test/metrics_by_sample.csv`

The per-sample file contains 800 unique sample UIDs: 20 held-out workloads from each of the 40 package families represented during package-model training. Its mean of per-sample source-baseline MAEs is `1.3879252916295082 K`, differing from the authoritative global-cell accumulator by only `2.92e-8 K`. Equality is expected here because every map contains 4,096 cells; the reported value remains the evaluator's global-cell result.

## Metric Definition

The reported quantity is

```text
sum over all samples and cells(
    abs(source_superposition_base_K - HotSpot_reference_K)
) / 3,276,800
```

`MetricAccumulator.update` in `scripts/evaluate_residual_cnn.py` accumulates the absolute error sum and divides by the total cell count. This is the same global aggregation used for the direct CNN and full ChipTherm-CNN familiar-family MAEs.

## Reconstruction And Fairness Checks

- `ChipThermDataset._prediction_path_for_row` selects `source_superposition_base_path` when `source_base_mode=source_superposition_v1`.
- The stored base definition is `ambient_K + sum_i source_power_i * source_response_operator(source_i)`, so ambient is already added once and all active chiplet-source contributions are included.
- The base tensor has physical-temperature units of kelvin and is compared directly with the physical HotSpot-generated reference tensor. Normalization is used only when constructing neural-network input and is not applied to this metric.
- In evaluation, `physics_acc.update(physics, temperature)` is called independently of `final_acc.update(pred_temperature, temperature)`. Therefore, no scalar mean correction, centered residual, graph correction, or other residual-model output is included in the source-baseline metric.
- The source-response checkpoint was selected using its source-response training/validation lineage. The package-residual lineage explicitly records `primary_heldout_used_for_selection=false`; no strict held-out test-family result selected the reported checkpoints.
- This is not the strict held-out-family source MAE (`1.6681 K`) and not the full ChipTherm-CNN familiar MAE (`0.1500 K`).

## Two-Regime Ablation

| Configuration | Familiar MAE (K) | Unseen MAE (K) |
|---|---:|---:|
| Direct CNN prediction | 0.3699 | 1.7674 |
| Source-superposition baseline | **1.3879** | 1.6681 |
| Full ChipTherm-CNN | 0.1500 | 1.3306 |

**SRC-ready interpretation:** The source-superposition prior is less accurate than the direct CNN on familiar families but transfers more consistently to unseen package families, while the learned residual correction substantially improves the prior in both regimes.

## Caveat

The source baseline was recovered from the baseline accumulator embedded in the frozen residual-model evaluation rather than from a separate source-only invocation. This does not mix predictions: the evaluator accumulates the source tensor and final model tensor in separate metric objects from the same 800-sample loader, which is preferable for matched-sample comparison.

No CUDA rerun is needed.
