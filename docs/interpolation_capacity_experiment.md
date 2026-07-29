# Benchmark v2 CNN Interpolation-Capacity Study

## Scope

This study has exactly three entries:

1. `canonical_cnn`: reuse
   `feature_fusion_train40_source_v1_seed1`.
2. `cnn_cosine_ema`: canonical 2,188,803-parameter architecture with only a
   bounded cosine schedule, EMA, and longer training budget.
3. `cnn_param_matched`: the same architecture family widened uniformly to
   3,919,642 parameters and trained with the same cosine+EMA recipe.

No data, split, input, metadata, source-superposition, target decomposition,
loss, optimizer family, batch size, reconstruction sign, or checkpoint metric
changes.

## Fixed Reconstruction

```text
T_pred_K =
    source_superposition_base_K
    + total_power_W * delta_R_eff_pred_K_per_W
    + zero_mean_centered_field_K
```

Both correction signs remain `+1`.

## Cosine And EMA

- Initial learning rate: `1e-3`
- Scheduler: `CosineAnnealingLR`
- Maximum epochs: `150`
- Minimum learning rate: `1e-5`
- Warmup: none
- Early-stopping patience: `30`
- EMA decay: `0.999`
- EMA update: once, immediately after every optimizer step
- Validation and best-checkpoint selection: EMA weights
- Checkpoint evaluation default: EMA
- Diagnostic evaluation: explicit `--weights raw`

Checkpoints preserve raw and EMA model states, EMA decay and update count,
optimizer and scheduler states, epoch, global optimizer-step count, best
validation MAE, early-stopping state, stable recipe hash, model configuration,
training lineage, source version through lineage, seed, and parameter count
through model configuration. Legacy checkpoints without EMA continue to use raw
weights under `--weights auto`.

## Parameter Match

The width search increments a shared `base_channels`, `refine_channels`, and
`global_hidden_channels` value from 33 upward and selects the first model in
the approved 3.8–4.2M range. All depths, block counts, heads, metadata widths,
and routing remain fixed.

```text
base_channels          = 43
refine_channels        = 43
global_hidden_channels = 43
metadata_hidden_dim    = 64
metadata_embedding_dim = 64
parameter_count        = 3,919,642
```

This is 1,730,839 parameters above the canonical CNN, 105,992 below U-FNO, and
109,160 below SAU-FNO.

## Decision Gate

The wider run is considered only after cosine+EMA known-family and held-out
validation results exist.

Primary success:

- known-family MAE improves by at least 15%;
- held-out validation worsens by at most `0.05 K`.

Strong success:

- known-family MAE is at most `0.12 K`;
- held-out validation worsens by at most `0.03 K`.

Near-complete closure:

- known-family MAE is at most `0.10 K`;
- held-out validation worsens by at most `0.05 K`.

If strong success is reached, parameter-matched training is not immediately
recommended. Otherwise exactly one wider CNN is permitted. Primary-test
metrics are never read by this decision.

## Runtime Estimate

The canonical 100-epoch log totals approximately 983 seconds. Linear
epoch-count scaling gives roughly 1,474 seconds, or 24.6 minutes, for the
canonical-width 150-epoch run before EMA overhead. Scaling that estimate by the
parameter ratio gives approximately 44 minutes for the width-43 run. Both are
planning estimates; actual early stopping and GPU kernels determine measured
duration.
