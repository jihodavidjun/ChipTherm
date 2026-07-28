# ChipTherm Benchmark v2 Zero-Shot Closure

## Protocol

This report closes the zero-shot architecture comparison on
`benchmark_v2_50family`:

- 40 optimization families
- held-out validation families: f007, f012, f023, f030, f041
- held-out primary test families: f008, f016, f027, f033, f044
- 200 workloads per held-out family
- fixed 64x64 HotSpot targets
- source baseline: `source_superposition_final_train40_source_v1`

The four frozen residual models use the same reconstruction:

```text
T_pred_K =
    source_superposition_base_K
    + total_power_W * delta_R_eff_pred_K_per_W
    + zero_mean_centered_field_K
```

Both correction signs are `+1`. Checkpoint lineage records exclude the primary
validation and test families from optimization and checkpoint selection.

## Canonical Results

| Model | Known MAE (K) | Held-out val MAE (K) | Held-out test MAE (K) | Test runtime (ms/sample) | Parameters |
|---|---:|---:|---:|---:|---:|
| Residual CNN | 0.1500 | 0.9130 | **1.3306** | 1.112 | 2,188,803 |
| Residual FNO | 0.1083 | 0.9225 | 1.3711 | **0.805** | 2,394,914 |
| Residual U-FNO | **0.0875** | 0.9200 | 1.3654 | 1.073 | 4,025,634 |
| Residual SAU-FNO | 0.0920 | **0.9138** | 1.3995 | 1.882 | 4,028,802 |

These runtimes are cached model-side measurements. They do not include uncached
source-response construction.

## Family-Wise Findings

The residual CNN wins four of five primary test families:

| Family | CNN | FNO | U-FNO | SAU-FNO | Winner |
|---|---:|---:|---:|---:|---|
| f008 | **0.779** | 0.866 | 0.833 | 0.967 | CNN |
| f016 | **1.016** | 1.035 | 1.026 | 1.018 | CNN |
| f027 | **0.966** | 1.210 | 1.149 | 1.150 | CNN |
| f033 | **0.926** | 1.003 | 0.927 | 0.986 | CNN |
| f044 | 2.967 | **2.741** | 2.891 | 2.877 | FNO |

The aggregate CNN win is broad over four ordinary held-out test families, but
the aggregate value is strongly affected by f044. Its 2.967 K MAE contributes
about 44.6% of the sum of the five CNN family MAEs. The remaining four CNN
family MAEs are tightly grouped from 0.779 to 1.016 K.

Validation-family winners are more mixed: CNN wins f007 and f012, FNO wins
f023, U-FNO wins f030, and SAU-FNO wins f041. This does not translate into an
operator-model aggregate test advantage.

## OOD Method

The family descriptor table contains 166 inference-time descriptors covering:

- package and chiplet geometry
- placement and spacing
- chiplet-type composition
- package-stack and cooling parameters
- workload-aggregated metadata
- source-superposition field statistics

No HotSpot target or residual error is used in a descriptor or distance.
Features are standardized using the 40 training families only. Euclidean
distance is L2 distance in standardized descriptor space. Regularized
Mahalanobis distance uses:

```text
0.9 * covariance + 0.1 * diagonal(covariance)
```

PCA is also fit on training families only.

OOD thresholds are derived from leave-one-out training-family neighborhoods:

- close radius: training nearest-neighbor distance q75
- distant radius: training nearest-neighbor distance q95
- marginal extrapolation: at least one descriptor outside the training min/max
- response anomaly: within the q95 radius, but source-baseline or final-error
  discrepancy from the nearest training family exceeds the corresponding
  training-neighbor q95

Each family receives one primary tier and may retain multiple secondary flags.
Threshold values are saved in `zero_shot_diagnostic_summary.json`.

## Error Decomposition

On primary test families, the CNN's centered-field MAE is 1.137 K and its
mean-correction MAE is 0.622 K. The spatial component is therefore the larger
cross-family failure term, though f044 also exhibits meaningful package-scale
response mismatch. Boundary and hotspot tables are retained separately; this
prevents a good average map metric from hiding local thermal risk.

Nearest-family distance has an exploratory Spearman correlation of about 0.65
with CNN final MAE and 0.62 with CNN centered-field MAE over the ten held-out
families. With only ten observations these are diagnostic associations, not
statistical evidence of causation.

f044 is nearest to f043 in descriptor space but has a much larger thermal
response and residual-model error. It is best interpreted as a
response-anomalous, partly marginal holdout rather than a family with no
geometrically related training neighbor.

## Figures

The analyzer generates the following when matplotlib is available:

- per-family MAE and RMSE
- source improvement by family
- error versus descriptor distance
- centered versus mean error
- boundary and hotspot error
- accuracy/runtime Pareto plot
- train-fit descriptor embedding

Representative common-scale heatmaps additionally require the portable
Benchmark v2 data root because saved predictions alone are insufficient to show
ground truth and source maps honestly.

## Limitations

- OOD/error correlations use only ten held-out families.
- Descriptor distance depends on the documented descriptor representation and
  is not a universal physical similarity metric.
- Cached inference runtimes omit online source-response construction.
- The local workspace contains all four models' metrics, per-sample tables, and
  predictions, but not the raw Benchmark v2 data root.
- The local Python environment lacks matplotlib, so plot generation is deferred
  to the server or another environment with matplotlib installed.

## Conclusion

The zero-shot architecture phase is closed for this benchmark protocol.
U-FNO and SAU-FNO improve known-family interpolation but do not improve
aggregate primary-test accuracy over the smaller residual CNN. Plain FNO is
useful for f044 and is the fastest cached operator, but it is not the aggregate
held-out winner.

The next scientifically useful step is family-count scaling or scoped few-shot
adaptation, with explicit retention of descriptor-close response anomalies.
Future benchmark challenge extensions should emphasize such families rather
than only extending scalar descriptor ranges.

These conclusions are specific to ChipTherm Benchmark v2. They do not establish
universal superiority over published thermal-surrogate methods.
