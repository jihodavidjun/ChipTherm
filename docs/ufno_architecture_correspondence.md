# ChipTherm U-FNO Architecture Correspondence

## Reference identity

- Model: U-FNO, Wen et al., *Advances in Water Resources* 2022,
  arXiv:2109.03697v3.
- Official local repository: `/Users/jihojun/ufno_reference`.
- Inspected commit: `8315fd7b5bd75282b7efe42ee6b8de86543d13cc`.
- Inspected source: `ufno.py`, `README.md`,
  `train_UFNO_pressure_buildup.ipynb`,
  `train_UFNO_gas_saturation.ipynb`, and the corresponding paper's
  architecture section and diagram.
- License found in the reference repository: CC BY-NC-ND 4.0. Because that
  license does not permit sharing adapted material, ChipTherm independently
  reimplements the published architecture and does not copy the reference
  source.

The ChipTherm implementation is described as a **task-adapted published
U-FNO**. It preserves the defining operator and U-Net topology while adapting
3D transient flow tensors to 2D steady-state thermal fields.

## Architecture mapping

| Element | Published U-FNO | ChipTherm implementation | Classification |
|---|---|---|---|
| Problem | 3D space-time CO2 saturation or pressure buildup | 2D steady-state package temperature | Task-interface adapted |
| Input layout | Channel-last `[B, X, Y, T, C]` before lifting; channel-first internally | Channel-first `[B, C, H, W]`, matching ChipTherm loaders | Dimensionally adapted |
| Lifting | Pointwise fully connected lift to operator width | `1x1 Conv2d` lift, mathematically pointwise over the grid | Dimensionally adapted |
| Operator sequence | Six layers: three Fourier, then three U-Fourier | Six layers with U-Net branches exactly at zero-based indices `3,4,5` | Preserved |
| Fourier path | Truncated 3D real FFT, learned complex weights, inverse FFT | Existing ChipTherm truncated 2D real FFT path | Dimensionally adapted |
| Local path | Pointwise linear map | `1x1 Conv2d` at every block | Preserved |
| U-Net placement | U-Net branch in each of the last three U-Fourier layers | U-Net branch in each of blocks `3,4,5` only | Preserved |
| U-Net depth | Three stride-2 encoder levels and three transpose-convolution decoder levels | Three 2D stride-2 levels and three 2D transpose-convolution levels | Dimensionally adapted |
| U-Net channels | Width remains constant through the branch | Width remains constant: `[32,32,32]` in the primary profile | Preserved |
| Encoder refinement | Extra stride-1 convolution after encoder levels two and three | Same relative placement with `Conv2d` | Dimensionally adapted |
| Skip structure | Concatenate level-two, level-one, and input skips during decoding | Same three concatenative skips | Preserved |
| Downsampling | Kernel 3, stride 2, same padding | `Conv2d`, kernel 3, stride 2, padding 1 | Dimensionally adapted |
| Upsampling | Transposed convolution, kernel 4, stride 2, padding 1 | `ConvTranspose2d` with the same kernel, stride, and padding | Dimensionally adapted |
| U-Net normalization | Batch normalization after encoder convolutions | `BatchNorm2d` at the corresponding locations | Dimensionally adapted |
| U-Net activation | LeakyReLU with slope 0.1 | LeakyReLU with slope 0.1 | Preserved |
| U-Net dropout | Dropout parameter; official trained model uses zero | Dropout configurable and zero in the primary profile | Preserved |
| Branch equation | `spectral(x) + pointwise(x)` for Fourier layers; add `U-Net(x)` in U-Fourier layers | Exact additive branch fusion | Preserved |
| Operator activation | ReLU in the published flow model | GELU, fixed to match the controlled ChipTherm plain-FNO comparison | Task-interface adapted |
| Metadata | No ChipTherm package metadata FiLM | Existing train-normalized 15-feature FiLM after branch sum, identical to plain FNO | Task-interface adapted |
| Projection | Pointwise hidden projection followed by scalar output | Existing `32 -> 64 -> 1` pointwise projection | Task-interface adapted |

## Padding and boundaries

The official wrapper pads the positive ends of its axes before the operator and
crops after projection. The second and third axes use replication while the
first spatial axis is zero-padded. The primary 2D adaptation records this as
`published_mixed`: the positive width edge is replicate-padded by eight cells,
the positive height edge is zero-padded by eight cells, and the output is
cropped back to exactly `64x64`.

This is a faithful dimensional reduction of the reference behavior, not a
claim that mixed asymmetric padding is optimal thermal physics. It is retained
for the primary controlled experiment. No circular convolution padding is used
inside the mini U-Net. The Fourier transform itself remains periodic on the
padded computational field, as in FNO/U-FNO.

## Temporal behavior

The published model represents time as a third operator dimension and predicts
the complete trajectory in one pass. It is not autoregressive. ChipTherm has no
temporal dimension because its target is a steady-state field. Conv3d, FFT3d,
BatchNorm3d, and transposed Conv3d are therefore replaced by their 2D
counterparts. No singleton artificial time axis is retained.

## ChipTherm heads

### Direct normalized temperature

`ufno2d_direct_conditioned` receives the exact 33 canonical spatial channels
and 15 metadata features. It excludes the source-superposition map. Its scalar
map output is train-standardized absolute temperature:

```text
T_norm = (T_K - train_target_mean_K) / train_target_std_K
```

The evaluator inverts this representation before reporting Kelvin metrics.

### Source-superposition residual decomposition

`ufno2d_residual_decomposed_conditioned` receives the same 33 channels plus
the source-superposition base exactly once. It reuses the existing normalized
resistance head and centered-field head:

```text
centered_K = raw_centered_K - spatial_mean(raw_centered_K)

T_pred_K =
    source_superposition_base_K
    + total_power_W * delta_R_eff_pred_K_per_W
    + centered_K
```

Both learned correction signs are explicitly `+1`. The supervised residual is
`HotSpot_K - source_superposition_base_K`; neither the scalar resistance
correction nor the centered spatial correction is negated during
reconstruction.

The raw `total_power_W` batch field is not inferred from normalized metadata.

## Capacity profile

The primary `ufno_published_adapted` profile uses width 32, modes `12x12`,
six operator blocks, three U-Net branches, U-Net depth three, projection width
64, and additive fusion. The six-block count is required by the published
three-Fourier-plus-three-U-Fourier topology, even though the controlled plain
FNO has four layers. Width, modes, conditioning, target definitions, losses,
and projection width remain matched.

No smaller "capacity-matched" U-FNO is selected automatically. Reducing the
number or depth of the three U-Net branches would change the defining
published topology. Parameter count, activation-memory estimate, and measured
runtime are reported as explicit costs of the faithful profile.

## Intentionally omitted features

- The transient time coordinate and autoregressive logic are omitted because
  ChipTherm is steady-state; the reference is itself non-autoregressive.
- Flow-specific input channels, output transforms, and data normalizers are
  omitted. ChipTherm retains its existing controlled 33-channel and
  train-only-normalization contracts.
- Reference repository code is not imported, vendored, or copied because of
  its license. Only the architecture described by the paper and repository
  structure is independently implemented.
- Metadata FiLM is not inserted inside the mini U-Net. It remains at the same
  post-fusion operator-block location used by plain FNO, keeping conditioning
  controlled.
