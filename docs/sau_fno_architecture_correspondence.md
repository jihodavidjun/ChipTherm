# ChipTherm SAU-FNO Architecture Correspondence

## Sources

- Zhen Huang et al., "Self-Attention to Operator Learning-based 3D-IC
  Thermal Simulation," DAC 2025, arXiv:2510.15968v1.
- G. Wen et al., "U-FNO--An Enhanced Fourier Neural Operator-Based
  Deep-Learning Model for Multiphase Flow," arXiv:2109.03697.
- Local U-FNO reference commit
  `8315fd7b5bd75282b7efe42ee6b8de86543d13cc`.

No verified official SAU-FNO source repository was found. This repository
independently implements a metadata-conditioned, steady-state 2D adaptation of
SAU-FNO. It is not described as the official implementation.

## Paper Interpretation

For final U-FNO feature map \(V_t\), the paper defines pointwise projections

\[
A_c=W_hV_t,\quad Q=W_qV_t,\quad K=W_kV_t,
\]

\[
s_{ij}=Q_i^\top K_j,\quad
A_s[i,j]=\operatorname{softmax}_{j}(s_{ij}).
\]

The paper calls \(A_c\) a "channel attention map," but its equation and Figure
2 show a projected value/channel feature, not a separate channel-similarity
matrix. Equation 10 calls the final operation element-wise multiplication even
though \(A_s\in\mathbb{R}^{N\times N}\) and
\(A_c\in\mathbb{R}^{N\times C}\). The dimensionally valid interpretation
depicted by the figure is

\[
V'_t=A_s A_c=\operatorname{softmax}(QK^\top)W_hV_t.
\]

`SAUAttention2d` implements this equation as exact, unscaled, single-head
spatial self-attention. `W_q`, `W_k`, and `W_h` are 1x1 `Conv2d` layers.
Spatial grid cells are tokens; feature channels are embeddings. Softmax is
over the key-token axis for each query token. The controlled profile uses
query, key, and value dimensions equal to U-FNO width 32.

The paper does not specify scaled dot products, an output projection, a
residual connection, normalization, positional embeddings, multiple heads, or
a transformer MLP. None are added. PyTorch
`scaled_dot_product_attention(..., scale=1.0)` is used because it preserves
the unscaled equation while allowing an exact memory-efficient CUDA backend.

## Placement And Padding

The paper reports similar accuracy from attention after every U-FNO layer and
after only the last one or two, then selects attention only at
\(V_t\rightarrow V'_t\) after the final layer. ChipTherm therefore adds
exactly one attention module after block 5 and before the existing projection
head.

The audited U-FNO evaluates its six blocks on a mixed-padded 72x72 domain and
crops back to 64x64 before projection. The paper does not discuss domain
padding. The ChipTherm adaptation applies attention after this existing crop:

```text
33 or 34 channels at 64x64
  -> mixed U-FNO padding to 72x72
  -> six unchanged operator blocks (U-Net branches at 3,4,5)
  -> existing crop to 64x64
  -> one SAUAttention2d
  -> unchanged projection/mean heads
```

This choice keeps attention on physical output cells, avoids treating
artificial padding as thermal tokens, and provides exact attention-disabled
parity with the existing U-FNO backbone.

## Shapes And Memory

For width 32 and a 64x64 field:

- operator tokens before padding: 4,096;
- padded operator tokens: 5,184;
- attention tokens after crop: 4,096;
- Q/K/value: `[B,4096,32]`;
- conceptual attention matrix: `[B,4096,4096]`;
- matrix elements per sample: 16,777,216;
- explicit score storage per sample: 64 MiB fp32 or 32 MiB fp16/bf16;
- explicit score storage at batch 64: 4 GiB fp32 or 2 GiB fp16/bf16.

Those numbers exclude Q/K/value tensors, U-FNO activations, FFT workspaces,
softmax/autograd state, optimizer state, and allocator overhead. Exact SDPA
may avoid materializing the full matrix, but batch 64 is not declared safe
until a one-batch peak-memory test runs on the 48 GB RTX A6000. The primary
configs retain batch 64 for controlled comparison; no silent approximation or
batch-size change is made.

## Direct Contract

`sau_fno2d_direct_conditioned` uses prediction mode
`direct_temperature_sau_fno`.

- Input: 33 canonical non-source channels and 15 metadata features.
- Source-superposition input: excluded.
- Target: train-standardized absolute HotSpot temperature.
- Reconstruction:
  `T_pred_K = T_norm_pred * train_target_std_K + train_target_mean_K`.

## Residual Contract

`sau_fno2d_residual_decomposed_conditioned` uses prediction mode
`residual_decomposed_sau_fno`.

- Input: 33 canonical channels plus one source-superposition base channel.
- Metadata: the same 15 features.
- Raw scalar input: `total_power_W` from the batch.
- Target: `HotSpot_K - source_superposition_base_K`.
- Mean head: train-standardized residual resistance in K/W.
- Centered field: spatial mean subtracted per sample.
- Reconstruction:

```text
T_pred_K =
    source_superposition_base_K
    + total_power_W * delta_R_eff_pred_K_per_W
    + zero_mean_centered_field_pred_K
```

Both correction signs are `+1`, and the source base is included exactly once.

## Controlled Scope

Width 32, modes 12x12, six blocks, U-Net branches `(3,4,5)`, U-Net depth 3,
domain padding 8, mixed padding, GELU, projection width 64, metadata FiLM,
splits, losses, normalization, optimizer, scheduler, epochs, and seed match
the audited U-FNO experiments. Transfer learning and multifidelity training
from the paper are intentionally excluded so the experiment isolates the
attention addition.
