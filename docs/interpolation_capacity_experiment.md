# Benchmark v2 CNN Interpolation-Capacity Study

## Scope

This study has exactly six predetermined CNN entries:

1. `canonical_small_constant`: existing 2,188,803-parameter epoch-100 run.
2. `small_cosine_ema_epoch100`: existing explicit `epoch_0100.pt`.
3. `small_cosine_ema_epoch150`: existing explicit `epoch_0150.pt`.
4. `wide_constant_epoch100`: new 3,919,642-parameter constant-LR run.
5. `wide_cosine_ema_epoch100`: explicit epoch-100 checkpoint from one new
   wide cosine+EMA run.
6. `wide_cosine_ema_epoch150`: explicit epoch-150 checkpoint from that same
   run.

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

## Two-Factor Contract

At epoch 100, the four cells isolate width and recipe:

| Width | Constant LR | Cosine+EMA |
|---|---|---|
| 2.19M | canonical | explicit epoch 100 |
| 3.92M | new wide constant | explicit epoch 100 |

Epoch 150 is a predefined bounded-budget sensitivity point for both
cosine+EMA widths. It is not selected using primary-test performance.

The analyzer first produces a validation-only interpretation from
`known_family_sample_test` and `primary_validation_families`. The
`--freeze-validation` action records metric hashes and the interpretation.
`--include-primary-test` is rejected until that artifact is frozen, and it is
also rejected if the validation artifacts subsequently change.

## Runtime Estimate

The canonical 100-epoch log totals approximately 983 seconds. Linear
epoch-count scaling gives roughly 1,474 seconds for the small epoch-150 run.
Parameter-ratio scaling gives a rough 29-minute estimate for wide constant
epoch 100 and 44 minutes for wide cosine+EMA epoch 150. These are planning
estimates only. The final report reads actual epoch runtimes and optimizer
steps when artifacts are available, and never calls equal epochs equal
compute.

## Checkpoint Finding

The current trainer saves `best.pt` only on a strict internal-validation
improvement, writes `last.pt` after every completed epoch, and writes periodic
checkpoints independently. Therefore an epoch-4 `best.pt` can be intentional.
An epoch-4 `last.pt` alongside a valid epoch-150 periodic checkpoint cannot
come from one uninterrupted invocation of the current save loop; it indicates
stale, overwritten, or incompletely synchronized artifact state. The completed
small run is evaluated only through explicit epoch checkpoints, and no trainer
patch is justified by that artifact observation alone.
