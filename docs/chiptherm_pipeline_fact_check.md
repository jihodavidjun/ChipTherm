# ChipTherm Pipeline Implementation Fact-Check

This report audits the frozen Benchmark v2 source-response and package-residual implementation. It describes executable code and saved run artifacts, not architecture names or earlier summaries.

## 1. Executive conclusion

### Correct source equation

The source-response network does **not** directly emit a physical temperature contribution. Its raw output is a train-standardized unit-response field. For source $s$,

\[
u_s(\mathbf{x}) = f_\theta(I_s)(\mathbf{x}),
\qquad
\widehat Z_s(\mathbf{x}) = \sigma_Z u_s(\mathbf{x}) + \mu_Z,
\]

where $u_s$ is dimensionless and \(\widehat Z_s\) has units K/W. The implementation then computes

\[
\widehat{\Delta T}_s(\mathbf{x}) = P_s\widehat Z_s(\mathbf{x}),
\qquad
T_{\mathrm{base}}(\mathbf{x}) = T_{\mathrm{amb}} + \sum_{s=1}^{N_s}\widehat{\Delta T}_s(\mathbf{x}).
\]

Thus option **C** is exact, and both manuscript forms are equivalent only when \(\widehat{\Delta T}_s=P_s\widehat Z_s\) is stated. The model class calls its output a standardized unit response (`src/chiptherm/ml/source_response_models.py:28-36,70-95`); denormalization and power scaling occur explicitly (`src/chiptherm/ml/source_response_dataset.py:404-411`; `src/chiptherm/ml/integrated_inference.py:488-498`).

### Correct final reconstruction

The frozen residual model uses a residual-resistance mean head:

\[
\widehat T(\mathbf{x}) = T_{\mathrm{base}}(\mathbf{x})
+ P_{\mathrm{tot}}\widehat{\Delta R}_{\mathrm{eff}}
+ \widehat S_0(\mathbf{x}),
\qquad
\langle \widehat S_0\rangle_{\mathbf{x}}=0.
\]

The signs are additive. The scalar head is train-standardized, denormalized to K/W, and multiplied by total package power (`src/chiptherm/ml/models.py:1537-1550`). The spatial field is explicitly centered before reconstruction (`src/chiptherm/ml/models.py:1419-1425`; `scripts/train_residual_cnn.py:2379-2394`). The frozen lineage records the same equation (`outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/training_lineage.json:103-109`).

### Terminology verdict

- **Physics-guided:** accurate with qualification. The pipeline imposes steady-state additive superposition, explicit W-to-K scaling, ambient addition, physical geometry, and a zero-mean residual decomposition. The response kernel itself is learned from isolated HotSpot labels, not derived from a PDE or material law.
- **Thermal impedance:** accurate only as **learned effective source-to-grid unit response (K/W)**. It should not be described as a closed-form, RC-extracted, or guaranteed-positive physical impedance. The frozen source head is linear and unconstrained (`src/chiptherm/ml/source_response_models.py:45-46,68-85`).
- **Physics-derived:** unsupported for the learned source operator. Use "physics-guided learned source-response superposition."

Only additivity across source contributions is hard-coded. Strict homogeneity in power is not guaranteed because source power density is itself a model input before the predicted K/W field is multiplied by source power (`src/chiptherm/ml/source_response_dataset.py:256-280,282-304`). Accordingly, call this a learned additive superposition surrogate, not a provably linear thermal operator.

## 2. Authoritative inference pipeline

1. **Load one package.** The integrated path loads the 33-channel package tensor, scalar ambient, total power, 15 metadata features, layout, power, and package YAML (`src/chiptherm/ml/integrated_inference.py:235-267,598-619`).
2. **Enumerate sources.** Every layout chiplet is one source, in deterministic layout order. The active workload power is matched by chiplet name (`src/chiptherm/ml/integrated_inference.py:602-618,623-632`).
3. **Build a full-grid source input.** For each selected chiplet, the builder retains all-chiplet occupancy/type geometry, identifies the selected source with a mask, and adds source power density, source-relative physical distances, and package-edge distances. Cell centers are used on the package-wide 64x64 grid (`src/chiptherm/ml/source_response_dataset.py:243-304`).
4. **Normalize source inputs.** Source power density and eight physical distance/offset channels are z-scored with train-only statistics; masks and normalized coordinates are unchanged (`src/chiptherm/ml/source_response_dataset.py:38-48,346-381,395-401`).
5. **Predict and denormalize unit response.** The shared U-Net emits $u_s\in\mathbb{R}^{64\times64}$, then the frozen training mean and standard deviation recover \(\widehat Z_s\) in K/W (`src/chiptherm/ml/source_response_models.py:53-85`; `src/chiptherm/ml/source_response_dataset.py:409-411`).
6. **Power-scale each source.** The implementation computes \(P_s\widehat Z_s\) in K (`src/chiptherm/ml/source_response_models.py:111-114`).
7. **Align and sum.** No output crop, translation, placement mask, or external resize is used. Every response is already predicted on the same package-wide grid. Source responses are summed by package, normally with float64 host accumulation; device summation is an optional float32 path (`src/chiptherm/ml/integrated_inference.py:454-468,499-520`). Bilinear interpolation appears inside the source U-Net decoder, not as post-hoc source placement (`src/chiptherm/ml/source_response_models.py:78-82`).
8. **Add ambient once.** The package ambient is added after source summation and the result is saved/returned as float32 absolute temperature in K (`src/chiptherm/ml/integrated_inference.py:515-530`). Source training targets subtract ambient exactly once from isolated HotSpot temperature (`scripts/build_source_response_dataset.py:248-261`).
9. **Build the residual input.** The 33 package channels are normalized and concatenated with the normalized source-superposition base, yielding 34 channels (`src/chiptherm/ml/normalization.py:264-299`).
10. **Condition and correct.** A 15-vector metadata encoder drives FiLM at the bottleneck, decoder, and refinement branch. A local U-Net and a pooled global encoder are fused at 16x16, 32x32, and 64x64 (`src/chiptherm/ml/models.py:226-259,766-846,1302-1425`).
11. **Reconstruct temperature.** The scalar residual-resistance correction and explicitly centered spatial residual are added to the source base (`src/chiptherm/ml/integrated_inference.py:330-341,553-568`).

## 3. Mathematical formulation

Let package $p$ contain chiplets \(s=1,\ldots,N_p\). Let \(I_{p,s}\) be the 17-channel full-grid input for source $s$, \(P_{p,s}\) its active power in W, \(P_{p}=\sum_sP_{p,s}\), and \(T_{\mathrm{amb},p}\) the package ambient in K. Because \(I_{p,s}\) includes source power density, \(\widehat Z_{p,s}\) may depend on \(P_{p,s}\); the implemented sum is additive across sources but is not constrained to be a power-independent linear operator.

### Source-response stage

The isolated training target is

\[
Z_{p,s}^{*}(\mathbf{x})=
\frac{T_{p,s}^{\mathrm{HotSpot}}(\mathbf{x})-T_{\mathrm{amb},p}}
{\max(P_{p,s},P_{\mathrm{floor}})},
\quad [Z^*]=\mathrm{K/W},
\]

with \(P_{\mathrm{floor}}=10^{-6}\) W as a numerical guard (`src/chiptherm/ml/source_response_dataset.py:111-126`; `scripts/build_source_response_dataset.py:258-260`). Train-only target standardization is

\[
\widetilde Z_{p,s}^{*}=(Z_{p,s}^{*}-\mu_Z)/\sigma_Z.
\]

The source loss is SmoothL1 on \(\widetilde Z^*\). In the frozen run, package loss has zero weight, although package-reconstructed validation MAE selects the best checkpoint (`configs/benchmark_v2_50family/training/source_response_final_train40_v1.yaml:9-17`; `scripts/train_source_response_model.py:289-311`). Inference is

\[
\widehat Z_{p,s}=\sigma_Z f_\theta(\widetilde I_{p,s})+\mu_Z,
\quad
T_{\mathrm{base},p}=T_{\mathrm{amb},p}+\sum_sP_{p,s}\widehat Z_{p,s}.
\]

The frozen values are \(\mu_Z=0.1396339199519297\) K/W and \(\sigma_Z=0.044658321184090986\) K/W (`outputs/benchmark_v2_50family/source_response/final_train40_v1/config.json:30-67`).

### Package-residual stage

Define the true package residual

\[
R_p^*(\mathbf{x})=T_p^{\mathrm{HotSpot}}(\mathbf{x})-T_{\mathrm{base},p}(\mathbf{x}),
\]

its scalar mean \(m_p^*=\langle R_p^*\rangle\), its centered component \(S_p^*=R_p^*-m_p^*\), and

\[
\Delta R_{\mathrm{eff},p}^{*}=m_p^*/P_p \quad [\mathrm{K/W}].
\]

These are the implemented targets (`scripts/train_residual_cnn.py:2348-2367`). If the mean head emits raw scalar \(r_p\),

\[
\widehat{\Delta R}_{\mathrm{eff},p}=\sigma_Rr_p+\mu_R,
\qquad
\widehat m_p=P_p\widehat{\Delta R}_{\mathrm{eff},p}.
\]

The frozen train-only values are \(\mu_R=1.081757540383138\times10^{-4}\) K/W and \(\sigma_R=1.044290510960419\times10^{-3}\) K/W (`outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/config.json:8-21`). For coarse map \(C_p\) and detail map \(D_p\),

\[
\widehat S_{0,p}=C_p+D_p-\langle C_p+D_p\rangle,
\]

and

\[
\boxed{\widehat T_p=T_{\mathrm{base},p}+P_p\widehat{\Delta R}_{\mathrm{eff},p}+\widehat S_{0,p}}.
\]

The frozen objective is

\[
\mathcal L=\operatorname{L1}(\widehat T,T^{\mathrm{HotSpot}})
+0.1\operatorname{L1}(\widehat m,m^*).
\]

Centered-field L1 is logged but is not an additional weighted term in this configuration (`scripts/train_residual_cnn.py:116-160`; `configs/benchmark_v2_50family/training/package_residual_feature_fusion_v1.yaml:11-20`).

## 4. Major tensor shapes

| Stage | Tensor | Shape | Physical meaning / units |
|---|---|---:|---|
| Canonical package | `x` | `[B,33,64,64]` | Mixed physical/context package features |
| Source construction | `source_input` | `[N_s,17,64,64]` | One full-grid input per chiplet |
| Source model | raw `u_s` | `[N_s,64,64]` | Standardized, dimensionless K/W target |
| Source denormalization | `Z_hat_s` | `[N_s,64,64]` | Effective unit response, K/W |
| Power scaling | `DeltaT_hat_s` | `[N_s,64,64]` | Per-source temperature rise, K |
| Segment sum + ambient | `T_base` | `[B,64,64]` | Absolute source-superposition temperature, K |
| Residual metadata | raw / normalized metadata | `[B,15]` | Physical descriptors / train z-scores |
| Residual model input | `model_input` | `[B,34,64,64]` | 33 normalized package channels + normalized base |
| Local encoder | features | `[B,32,64,64]`, `[B,64,32,32]`, `[B,128,16,16]` | Local/multiscale features |
| Global encoder | selected input | `[B,5,64,64]` | Power density, occupancy, x/y, source base |
| Global bottleneck | context | `[B,128,8,8]` | Package-scale context |
| Global projections | context | `[B,128,16,16]`, `[B,64,32,32]`, `[B,32,64,64]` | Decoder-scale global features |
| Coarse head | `coarse` | `[B,1,64,64]` | Coarse residual field, K after training |
| Refinement input | selected local + coarse | `[B,9,64,64]` | Channels 0-7 plus coarse map |
| Detail head | `detail` | `[B,1,64,64]` | Full-resolution local correction, K |
| Centered output | `centered_field` | `[B,64,64]` | Zero-mean package residual, K |
| Mean head input | metadata embedding + pooled input | `[B,98]` | 64 embedding + 34 pooled channels |
| Mean head | raw / `delta_R_eff` / `mean_rise` | `[B]` each | standardized scalar / K/W / K |
| Reconstruction | `final_temperature` | `[B,64,64]` | Absolute temperature, K |

Shapes follow the source U-Net (`src/chiptherm/ml/source_response_models.py:53-82`), feature-fusion encoder (`src/chiptherm/ml/models.py:783-816`), global projections (`src/chiptherm/ml/models.py:719-750`), and decomposed wrapper (`src/chiptherm/ml/models.py:1359-1434`).

## 5. Input channels

### 5.1 Source-response input, 17 channels

The exact order is fixed by `SOURCE_RESPONSE_CHANNEL_NAMES` (`src/chiptherm/ml/source_response_dataset.py:18-48`). "None" below means no source-stage standardization.

| i | Channel | Meaning | Units | Source | Normalization |
|---:|---|---|---|---|---|
| 0 | `occupancy_mask` | All-chiplet occupied cells | 0/1 | canonical X[1] | None |
| 1 | `CPU_mask` | CPU cells | 0/1 | canonical X[2] | None |
| 2 | `GPU_or_NPU_mask` | GPU/NPU cells | 0/1 | canonical X[3] | None |
| 3 | `memory_mask` | HBM/DRAM cells | 0/1 | canonical X[4] | None |
| 4 | `IO_or_ANALOG_or_MEMS_mask` | Peripheral/sensitive cells | 0/1 | canonical X[5] | None |
| 5 | `normalized_x_coordinate` | Cell-center x/package width | 1 | canonical X[6] | None |
| 6 | `normalized_y_coordinate` | Cell-center y/package height | 1 | canonical X[7] | None |
| 7 | `source_mask` | Selected source rectangle | 0/1 | layout + cell-center inclusion | None |
| 8 | `source_power_density_W_per_mm2` | Selected-source \(P_s/A_s\) inside source | W/mm2 | layout + active power | Train z-score |
| 9 | `source_dx_mm` | Cell-center x minus source-center x | mm | layout geometry | Train z-score |
| 10 | `source_dy_mm` | Cell-center y minus source-center y | mm | layout geometry | Train z-score |
| 11 | `source_radius_mm` | \(\sqrt{dx^2+dy^2}\) | mm | derived | Train z-score |
| 12 | `distance_to_left_edge_mm` | Cell-center x | mm | package geometry | Train z-score |
| 13 | `distance_to_right_edge_mm` | Width minus x | mm | package geometry | Train z-score |
| 14 | `distance_to_bottom_edge_mm` | Cell-center y | mm | package geometry | Train z-score |
| 15 | `distance_to_top_edge_mm` | Height minus y | mm | package geometry | Train z-score |
| 16 | `minimum_distance_to_package_edge_mm` | Minimum of four edge distances | mm | derived | Train z-score |

All geometry is package-wide; non-source chiplets remain visible in channels 0-4 while only channel 8 carries selected-source active power (`src/chiptherm/ml/source_response_dataset.py:243-304`).

### 5.2 Package-residual input, 34 channels

The 33 pre-base channels are frozen in the feature manifest (`data/runs/derived/source_superposition_base_v1_full/feature_manifest.json:2-47`). Channels 0 and 8-32 use train-only z-scores; masks and normalized coordinates 1-7 are unchanged; channel 33 is z-scored using the train source-base mean/std (`src/chiptherm/ml/normalization.py:264-299`).

| i | Channel | Meaning / formula | Units | Source | Normalization |
|---:|---|---|---|---|---|
| 0 | `power_density_W_per_mm2` | Active chiplet power/area in occupied cells | W/mm2 | layout + power | Train z-score |
| 1 | `occupancy_mask` | Any chiplet at cell center | 0/1 | layout | None |
| 2 | `CPU_mask` | CPU occupancy | 0/1 | layout type | None |
| 3 | `GPU_or_NPU_mask` | GPU/NPU occupancy | 0/1 | layout type | None |
| 4 | `memory_mask` | HBM/DRAM occupancy | 0/1 | layout type | None |
| 5 | `IO_or_ANALOG_or_MEMS_mask` | IO/analog/MEMS occupancy | 0/1 | layout type | None |
| 6 | `normalized_x_coordinate` | \((j+0.5)/64\) | 1 | grid | None |
| 7 | `normalized_y_coordinate` | \((i+0.5)/64\) | 1 | grid | None |
| 8 | `total_power_W` | \(\sum_sP_s\), constant map | W | power YAML | Train z-score |
| 9 | `package_width_mm` | Package width, constant map | mm | layout | Train z-score |
| 10 | `package_height_mm` | Package height, constant map | mm | layout | Train z-score |
| 11 | `cell_size_x_mm` | width/64, constant map | mm | derived | Train z-score |
| 12 | `cell_size_y_mm` | height/64, constant map | mm | derived | Train z-score |
| 13 | `finite_source_L0p5mm` | \(\sum_q w_q/\sqrt{r_q^2+0.5^2}\) | W/mm | quadrature source map | Train z-score |
| 14 | `finite_source_L1mm` | Same, softening 1 mm | W/mm | quadrature source map | Train z-score |
| 15 | `finite_source_L2mm` | Same, softening 2 mm | W/mm | quadrature source map | Train z-score |
| 16 | `finite_source_L4mm` | Same, softening 4 mm | W/mm | quadrature source map | Train z-score |
| 17 | `enclosed_power_R2mm_W` | Quadrature power within 2 mm | W | layout + power | Train z-score |
| 18 | `enclosed_power_R4mm_W` | Quadrature power within 4 mm | W | layout + power | Train z-score |
| 19 | `enclosed_power_R8mm_W` | Quadrature power within 8 mm | W | layout + power | Train z-score |
| 20 | `enclosed_power_R16mm_W` | Quadrature power within 16 mm | W | layout + power | Train z-score |
| 21 | `distance_to_left_edge_mm` | Cell-center x | mm | package geometry | Train z-score |
| 22 | `distance_to_right_edge_mm` | Width minus x | mm | package geometry | Train z-score |
| 23 | `distance_to_bottom_edge_mm` | Cell-center y | mm | package geometry | Train z-score |
| 24 | `distance_to_top_edge_mm` | Height minus y | mm | package geometry | Train z-score |
| 25 | `minimum_distance_to_package_edge_mm` | Minimum edge distance | mm | derived | Train z-score |
| 26 | `chiplet_total_power_W` | Chiplet power inside exact rectangle, zero outside | W | layout + power | Train z-score |
| 27 | `chiplet_width_mm` | Chiplet width inside rectangle | mm | layout | Train z-score |
| 28 | `chiplet_height_mm` | Chiplet height inside rectangle | mm | layout | Train z-score |
| 29 | `chiplet_area_mm2` | Width times height inside rectangle | mm2 | layout | Train z-score |
| 30 | `chiplet_aspect_ratio` | Width/height inside rectangle | 1 | layout | Train z-score |
| 31 | `chiplet_power_density_W_per_mm2` | Chiplet power/area inside rectangle | W/mm2 | layout + power | Train z-score |
| 32 | `thermal_crowding_W_per_mm` | \(\sum_sP_s/\sqrt{\|x-c_s\|^2+1\,\mathrm{mm}^2}\) | W/mm | layout + power | Train z-score |
| 33 | `source_superposition_base_K` | \(T_{amb}+\sum_sP_s\widehat Z_s\) | K | source stage | Train z-score |

The base 13-channel rasterizer uses physical cell-center inclusion (`src/chiptherm/ml/encoder.py:33-120`). The finite-source map uses 4x4 rectangle quadrature and softened inverse-distance kernels (`scripts/build_finite_source_feature_dataset.py:354-393`). Enclosed-power, edge, per-chiplet, and crowding formulas are implemented at `scripts/build_thermal_impedance_feature_dataset.py:262-347,490-568`.

### 5.3 Scalar metadata conditioning

The separate 15-vector is: package width, package height, x/y cell size, total power, chiplet count, occupied fraction, whitespace fraction, mean/max power density, mean/max chiplet area, mean chiplet aspect ratio, spreader side, and sink side. Every feature is standardized with train-only means/stds before a 15->64->64 SiLU MLP (`outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/config.json:270-285`; `src/chiptherm/ml/normalization.py:302-310`; `src/chiptherm/ml/models.py:226-242`). The two side dimensions are globally constant but are nevertheless present in the frozen 15-feature schema.

## 6. Residual architecture and output heads

### What "global/local feature fusion" actually means

- **Local multiscale path:** three two-convolution ReLU blocks with max pooling produce 32-channel 64x64, 64-channel 32x32, and 128-channel 16x16 features. Bilinear decoder upsampling concatenates the 32x32 and 64x64 skips (`src/chiptherm/ml/models.py:22-33,783-815,827-845`).
- **Package-conditioning path:** the standardized 15-vector becomes a 64-vector. FiLM applies learned channelwise scale and bias at the 16x16 bottleneck, 32x32 decoder, 64x64 decoder, and refinement input. FiLM starts as identity (`src/chiptherm/ml/models.py:226-259,799-815`; `src/chiptherm/ml/models.py:348-357`).
- **Global path:** only residual-input channels 0, 1, 6, 7, and 33 are selected. Strided 3x3 convolutions reduce 64->32->16->8; three residual context blocks operate at 8x8; 1x1 projections and bilinear upsampling produce 16x16, 32x32, and 64x64 context (`src/chiptherm/ml/models.py:683-750`; frozen selection in `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/config.json:247-266`).
- **Fusion:** at each 16/32/64 scale, local and global features are concatenated, mixed by 1x1 convolution and SiLU, transformed by a zero-initialized two-convolution residual delta, and added to the local feature (`src/chiptherm/ml/models.py:646-680,809-845`). This is feature-level fusion, not an independently predicted global correction map.
- **Full-resolution detail:** channels 0-7 plus the coarse output enter a 32-channel, four-block residual CNN. Its output projection is zero-initialized (`src/chiptherm/ml/models.py:111-149,1377-1382,1419-1422`).
- **Normalization layers:** there is no BatchNorm or LayerNorm in this frozen CNN. The local convolution blocks use ReLU; global/fusion and metadata MLPs use SiLU; FiLM provides metadata modulation.

### Output heads

| Head/component | Shape | Supervision / role | Centering | Reconstruction role |
|---|---:|---|---|---|
| Source raw head | `[N_s,64,64]` | Standardized isolated K/W unit response | None | Denormalize, then multiply by source W |
| Coarse spatial head | `[B,1,64,64]` | Jointly optimized through final temperature | Only diagnostic coarse copy is centered | Added to detail before final centering |
| Detail head | `[B,1,64,64]` | Jointly optimized through final temperature | Not independently centered | Local correction added to coarse |
| Final centered spatial | `[B,64,64]` | True residual minus its sample mean, K | Explicit mean subtraction in model and reconstruction | Added to base |
| Mean raw head | `[B]` | Standardized \(\Delta R_{eff}\) | N/A | Denormalized to K/W |
| Mean correction | `[B]` | Mean of HotSpot minus source base, K | N/A | \(P_{tot}\widehat{\Delta R}_{eff}\) added to base |
| Final temperature | `[B,64,64]` | HotSpot absolute temperature, K | N/A | Base + scalar + centered field |

The model therefore predicts **two residual correction components**, not absolute temperature and not one normalized residual map.

## 7. Training, sharing, freezing, and caching

- One `SourceResponseOperatorV1` is shared across every chiplet and package. There is no per-source or per-family parameter set (`src/chiptherm/ml/source_response_models.py:28-108`). Source type is inferable through type masks at the selected source location, but neither model receives a chiplet name, source UID, case ID, or family ID.
- The source model is trained first on isolated-source targets. Within the 40 designated training families, 32 families fit weights/normalization and 8 provide internal checkpoint selection; held-out primary validation/test families are excluded (`outputs/benchmark_v2_50family/source_response/final_train40_v1/training_lineage.json:24-111`).
- Residual training does not instantiate or update the source network. It reads precomputed source-superposition maps as the effective `physics` tensor through `source_superposition_base_path` (`src/chiptherm/ml/dataset.py:254-303,585-594`). Consequently, the source stage is frozen and cached during residual training.
- The residual CNN is trained on 6,400 workloads from the 40 training families and selected on 800 internal validation workloads from those same families. Primary held-out families are excluded from optimization and selection (`scripts/train_benchmark_v2_package_residual.py:102-135`; `src/chiptherm/benchmark_v2_training.py:476-483`).
- The authoritative deployment path is uncached: it rebuilds per-source rasters, invokes the frozen source model, sums source responses, then invokes the residual model in one pipeline (`src/chiptherm/ml/integrated_inference.py:235-355,438-530`). Cached residual-only timing is not complete end-to-end inference.

## 8. Frozen parameter count

| Component | Parameters | Accounting note |
|---|---:|---|
| Shared source-response U-Net | 475,585 | Separate frozen checkpoint |
| Residual metadata encoder | 5,184 | Included in residual total |
| Residual coarse feature-fusion model | 2,096,161 | Includes local encoder/decoder, global encoder, fusion blocks, FiLM, coarse head |
| of which global encoder | 1,155,808 | Subset, do not add again |
| of which 16/32/64 fusion blocks | 328,064 / 82,112 / 20,576 | Subsets, total 430,752 |
| Residual full-resolution refinement | 81,057 | Included in residual total |
| Residual scalar mean head | 6,401 | Included in residual total |
| **Residual CNN total** | **2,188,803** | Frozen package checkpoint |
| **Complete two-stage total** | **2,664,388** | 475,585 + 2,188,803 |

The saved source config records 475,585 parameters (`outputs/benchmark_v2_50family/source_response/final_train40_v1/config.json:69-70`). The residual model computes and records its component totals in `config()` (`src/chiptherm/ml/models.py:1552-1583`); the frozen run records the coarse, metadata, refinement, global, and 2,188,803 total counts (`outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/config.json:67,87,114,168,197-198`). The 2.664M figure is an inference-time sum of **separately trained** stages, not a jointly optimized parameter count. The frozen primary model has no GNN branch (`outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/config.json:40-49`).

## 9. Benchmark v2 family and split audit

### Definition and variables

One family fixes package dimensions, the 64x64 grid, chiplet count/names/types/rectangles, thermal stack, cooling, ambient convention, and HotSpot configuration. Geometry does not depend on workload. Within a family, only chiplet activity, chiplet power, and chiplet power density vary (`src/chiptherm/benchmark_v2.py:250-305`; `configs/benchmark_v2_50family/design_proposal.yaml:14-47`).

Across families, the executable blueprints vary package width/height/aspect, chiplet count, functional composition, chiplet sizes/aspects, target whitespace, and structured placement/spacing patterns (`src/chiptherm/benchmark_v2.py:81-200`). The actual fixed family YAMLs span:

- 6 to 64 chiplets;
- package width 34 to 72 mm and height 32 to 64 mm;
- package area 1,088 to 4,480 mm2;
- package aspect ratio 1.053 to 1.895;
- eight types: CPU, GPU, NPU, HBM, DRAM, IO, ANALOG, MEMS.

All 50 family files copy the same `DEFAULT_PACKAGE` and `DEFAULT_HOTSPOT`; generation explicitly rejects stack/config drift (`src/chiptherm/benchmark_v2.py:250-282,695-697`). The fixed stack has ambient 318.15 K, chip/interface/spreader/sink properties, and sink convection resistance 0.12 K/W (`configs/benchmark_v2_50family/families/f001.yaml:135-170`). There is no material or cooling OOD axis in Benchmark v2.0 (`configs/benchmark_v2_50family/design_proposal.yaml:25-34,169-171`).

Each family has 200 deterministic workloads arranged as 10 power regimes x 20 activity/topology regimes (`configs/benchmark_v2_50family/full_50x200.yaml:33-40`). HotSpot grid mode supplies the steady-state 64x64 K reference maps (`src/chiptherm/benchmark_v2_pipeline.py:805-871,890-897`).

### Exact split

- **Training families (40):** f001, f002, f003, f004, f005, f006, f009, f010, f011, f013, f014, f015, f017, f018, f019, f020, f021, f022, f024, f025, f026, f028, f029, f031, f032, f034, f035, f036, f037, f038, f039, f040, f042, f043, f045, f046, f047, f048, f049, f050.
- **Held-out validation families (5):** f007, f012, f023, f030, f041.
- **Held-out test families (5):** f008, f016, f027, f033, f044.

These lists are frozen in `configs/benchmark_v2_50family/splits/primary_family_split.yaml:4-56` and `configs/benchmark_v2_50family/full_50x200.yaml:17-25`.

The familiar-family sample protocol applies to the 40 training families: workload ordinals 1-160 train, 161-180 internal validation, and 181-200 known-family test, giving 6,400/800/800 samples. The family protocol uses all 200 workloads per family, giving 8,000/1,000/1,000 train/held-out-validation/held-out-test samples (`configs/benchmark_v2_50family/full_50x200.yaml:28-32`; `src/chiptherm/benchmark_v2_training.py:476-483,926-935`).

### Tentative-description verdicts

| Tentative item | Verdict | Evidence-based correction |
|---|---|---|
| 50 package families | Verified | Exactly 50 fixed structures. |
| 200 workloads per family | Verified | 10 x 20 workload matrix. |
| 10,000 temperature maps | Verified | Full configuration expects 10,000 HotSpot maps. |
| 6-64 chiplets | Verified | Actual fixed layouts span 6 to 64. |
| Package area 1,088-4,480 mm2 | Verified | Actual package width x height range. |
| Variable package aspect ratio | Verified | 1.053-1.895 across fixed families. |
| Variable chiplet dimensions/types | Verified | Geometry and eight-type compositions differ across families. |
| Varied occupancy, spacing, placement | Verified | Whitespace and placement styles are family axes. |
| Fixed material properties | Verified | One copied stack across all families. |
| Fixed cooling conditions | Verified | One sink/convection configuration. |
| Fixed layer stack | Verified with wording caveat | Fixed configured chip/interface/spreader/sink stack; do not imply extra substrate/interposer layers absent from the schema. |

## 10. Manuscript claim audit

| Proposed manuscript statement | Verdict | Corrected wording |
|---|---|---|
| "The source model predicts each chiplet's temperature contribution." | Revise | "The shared source model predicts a standardized effective unit-response field, which is denormalized to K/W and multiplied by chiplet power to form a K contribution." |
| "ChipTherm learns a thermal-impedance field." | Revise | "ChipTherm learns an effective source-to-grid unit response in K/W from isolated HotSpot labels." |
| "The source model is physics-derived." | Unsupported | "The source model is learned; physics guides its input units, source-power factorization, and additive superposition." |
| "ChipTherm is physics-guided." | Accurate | Add that the guidance is steady-state superposition, explicit power scaling, ambient handling, physical geometry, and residual decomposition, not a PDE guarantee. |
| "The baseline is an analytical thermal model." | Unsupported | "The baseline is a learned source-response superposition model." |
| "The source baseline is the sum of per-chiplet temperature rises." | Accurate | Define each rise as \(P_s\widehat Z_s\), and add ambient once after summation. |
| "Responses are predicted in local source coordinates and placed on the package." | Unsupported | "Each source response is predicted directly on the package-wide 64x64 physical grid; no post-hoc placement is performed." |
| "A global branch predicts an additive global correction map." | Unsupported | "A pooled global encoder injects package-scale features into decoder features at 16x16, 32x32, and 64x64." |
| "The residual CNN predicts a temperature map." | Revise | "It predicts a scalar effective-resistance correction and an explicitly zero-mean spatial residual, then adds both to the source base." |
| "The spatial residual is zero mean." | Accurate | The sum of coarse and detail maps is explicitly centered, and reconstruction centers it again defensively. |
| "The source and residual networks are trained end-to-end." | Unsupported | "They are trained sequentially; source-superposition maps are cached and the source checkpoint is frozen during residual training." |
| "One source network is shared across chiplets." | Accurate | One parameter-shared U-Net handles every source; no source/family-specific weights exist. |
| "The residual model receives source identity." | Unsupported | It receives aggregate rasters, type/geometry/power context, metadata, and the summed source base, but no chiplet name, source UID, case ID, or family ID. |
| "ChipTherm has 2.664M jointly trainable parameters." | Revise | "The integrated two-stage inference path contains 2,664,388 parameters: 475,585 in the separately trained frozen source operator and 2,188,803 in the residual CNN." |
| "Inference uses precomputed source maps." | Revise | "Residual training uses cached source maps; authoritative integrated inference regenerates the source base online." |
| "Benchmark v2 varies package materials and cooling." | Unsupported | Materials, cooling, ambient, HotSpot settings, and stack are fixed in v2.0. |
| "Benchmark v2 contains 50 families and 10,000 HotSpot maps." | Accurate | Each fixed family has 200 power/activity workloads on a 64x64 grid. |
| "The primary family split is 40/5/5." | Accurate | State the exact family IDs and keep the 40-family internal 160/20/20 workload split distinct. |

## 11. SRC-ready Approach and Uniqueness paragraph

ChipTherm predicts steady-state package temperature with a two-stage, physics-guided learned surrogate. First, a parameter-shared source-response U-Net processes each chiplet on the package-wide 64x64 grid using all-chiplet geometry, a selected-source mask, physical source distances, package-edge distances, and source power density. Its output is a train-standardized effective unit-response field; after denormalization to K/W, ChipTherm multiplies it by the chiplet power, sums all source contributions, and adds ambient once. This constructs a learned source-superposition temperature baseline without per-package fitting. A 2.189M-parameter global/local feature-fusion CNN then predicts only the remaining correction. The CNN combines a skip-connected 64-32-16 multiscale path with an 8x8 package-context encoder whose features are injected at three decoder scales. A compact physical-metadata encoder supplies FiLM modulation. The output is decomposed into a scalar effective-resistance correction, converted to kelvin by total package power, and an explicitly zero-mean spatial residual refined at full resolution. The final prediction adds both terms to the source baseline. The shared source operator and residual CNN are trained sequentially; cached source maps are used during residual training, while the authoritative integrated path computes them online. Benchmark v2 evaluates this formulation across 50 fixed package families and 200 workloads per family, with geometry, composition, whitespace, and placement varied across families and a strict 40/5/5 held-out-family split.

## 12. Text diagram for draw.io

```text
PACKAGE INPUTS
layout.json + active power.yaml + package.yaml + 33x64x64 package raster
        |
        +--> enumerate all chiplets in layout order
        |       |
        |       +--> for each source s:
        |             17x64x64 full-package source raster
        |             [all geometry + selected source mask/power/distances]
        |                    |
        |             train-normalize physical source channels
        |                    |
        |             shared SourceResponseOperatorV1 (475,585 params)
        |                    |
        |             standardized field u_s [dimensionless]
        |                    |
        |             Z_hat_s = sigma_Z*u_s + mu_Z [K/W]
        |                    |
        |             DeltaT_s = P_s*Z_hat_s [K]
        |                    |
        +------------- sum aligned 64x64 source fields
                              |
             T_base = T_ambient + sum_s DeltaT_s [K]
                              |
             normalize 33 package channels + T_base
                              |
                     34x64x64 residual input
                              |
        +---------------------+----------------------+
        |                                            |
  LOCAL MULTISCALE PATH                         GLOBAL PATH
  Conv blocks 64->32->16                       select PD/occ/x/y/base
  skips at 64 and 32                           stride to 8x8 context
        |                                            |
        +<-- feature fusion at 16, 32, and 64 ------+
                              |
                   coarse residual field C [K]
                              |
   channels 0-7 + C --> full-resolution detail branch --> D [K]
                              |
                   S0 = center(C + D) [K]

   15 physical metadata --> 64-D encoder --> FiLM + scalar mean head
                                               |
                      delta_R_eff [K/W] --> P_total*delta_R_eff [K]
                                               |
        T_final = T_base + P_total*delta_R_eff + S0 [K]

Integrated parameters: 475,585 source + 2,188,803 residual = 2,664,388
```

## Immediate manuscript corrections

1. Replace "the source model directly predicts a temperature contribution" with "it predicts a standardized K/W unit response that is denormalized and power-scaled."
2. Replace "physics-derived/analytical baseline" with "physics-guided learned source-response superposition baseline."
3. Describe global/local fusion as feature injection at three decoder scales, not a late additive global map.
4. State the additive residual reconstruction and its explicit zero-mean spatial term.
5. Describe 2.664M as the sum of two separately trained inference stages.
6. Do not claim material, cooling, or layer-stack generalization for Benchmark v2.0.
