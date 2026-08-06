# Benchmark v2 f044 Family Audit

## 1. Executive verdict

**Verdict: KEEP WITH STRATIFIED REPORTING.**

No objective benchmark-rule violation, malformed frozen family definition, split leak, workload-generation defect, sample-ID mismatch, or source-response ordering error was found. Family f044 is an intentional compound-OOD test family: a long, narrow package with longitudinally clustered sources and unusually strong directional boundary response. It is difficult for every evaluated model family, not only ChipTherm-CNN.

The dominant failure is also technically coherent. The HotSpot system is almost linear under isolated-source superposition for the audited f044 oracle package (`0.038740 K` oracle reconstruction MAE), while the learned source-response model reconstructs it at `3.665891 K` MAE. Positive and negative source-mean errors occur equally often in that package, but their unequal magnitudes sum to a package-wide cold bias. Across all 200 workloads, the learned source baseline has `3.624509 K` MAE and `-2.522442 K` mean signed error. The residual CNN removes most of that error (`2.966781 K` final MAE) but predicts only `0.232 K` average scalar correction against a `2.522 K` average required correction and leaves substantial low-frequency centered error.

This is evidence of a valid, difficult response regime, not evidence that the frozen test family is defective. Removing or replacing f044 after observing model results would be post-hoc test selection. Report the aggregate five-family result together with f044 and non-f044 strata.

### Scope limitation

The external Benchmark v2 data root was not mounted in this local checkout. Frozen specifications, deterministic workload generation, copied evaluation tables, source-level ordering, UIDs, and analysis outputs were audited. Checks requiring the actual server-side YAML/NPY/NPZ/work directories or file hashes are explicitly marked **NOT VERIFIABLE**. The existing strict-validator implementation covers those checks, but its completed server report was not available here and is not treated as evidence.

## 2. Validation checklist

| Check | Status | Evidence and conclusion |
|---|---|---|
| Frozen family identity and content hash | PASS | `configs/benchmark_v2_50family/family_manifest.yaml:327-329`; the recorded f044 SHA-256 matches `families/f044.yaml`. |
| Split assigned before model results | PASS | `configs/benchmark_v2_50family/splits/primary_family_split.yaml:1-56` records `preliminary_stage1_no_model_results_used`; f044 is a primary-test family. |
| Package dimensions and grid | PASS | `families/f044.yaml:20-34`: `68 x 36 mm`, `64 x 64`; accepted by `validate_family_spec`. |
| Exact chiplet count and composition | PASS | `families/f044.yaml:35-201`: 20 chiplets; CPU/GPU/NPU/HBM/DRAM/IO = 3/4/3/4/3/3. |
| Finite, positive chiplet rectangles and legal types | PASS | `src/chiptherm/benchmark_v2.py:654-725` and `src/chiptherm/validate.py:82-101`; the frozen definition passes these checks. |
| Package-boundary containment | PASS | The frozen definition passes `validate_layout`; containment is enforced at `src/chiptherm/validate.py:82-101`. |
| No chiplet overlap | PASS | The frozen definition passes `validate_layout`; overlap/spacing is enforced at `src/chiptherm/validate.py:46-79`. |
| Minimum spacing | PASS | Frozen minimum gap is `0.906650 mm`, above the declared `0.5 mm` requirement (`families/f044.yaml:239-251`). |
| Occupancy and whitespace | PASS | Occupied fraction `0.460000`; whitespace `0.540000`, exactly the frozen target and within the design proposal. |
| Frozen material stack, cooling, ambient, and HotSpot specifications | PASS | All 50 family YAMLs have one identical thermal-stack hash and one identical HotSpot-settings hash. Equality is enforced at `src/chiptherm/benchmark_v2.py:694-698` and `src/chiptherm/benchmark_v2.py:748-755`. |
| Persisted per-sample `package.yaml` and `hotspot.yaml` match the frozen specifications | NOT VERIFIABLE | The producer writes the frozen structures directly (`src/chiptherm/benchmark_v2_pipeline.py:760-795`) and resume validates their hashes (`src/chiptherm/benchmark_v2_pipeline.py:912-940`), but this checkout contains none of the realized Benchmark v2 source files. |
| Intended family-design envelope | PASS | `configs/benchmark_v2_50family/design_proposal.yaml:123-134` specifies f044 as `compound_ood`, long-axis clusters, 12-28 dies, long package, and 0.46-0.70 whitespace. |
| Deterministic workload count and uniqueness | PASS | Regeneration with the frozen seed produced 200 UIDs and 200 unique content hashes; all passed `validate_workload_record`. |
| Power finiteness, positivity, density identity, and idle policy | PASS | Enforced by `src/chiptherm/benchmark_v2_workloads.py:423-476`; all 200 regenerated records passed. |
| Workload chiplet names match layout | PASS | Enforced by exact name comparison at `src/chiptherm/benchmark_v2_workloads.py:423-476`; all regenerated records passed. |
| Active workload selection | PASS | Every regenerated YAML selects `active_workload=nominal`; generation is defined at `src/chiptherm/benchmark_v2_workloads.py:544-640`. |
| Persisted server workload files equal regenerated content hashes | NOT VERIFIABLE | External Benchmark v2 root is absent. Aggregate inference-time power descriptors match regeneration exactly, which is a strong cross-pipeline consistency check but not a byte-level file audit. |
| Exactly 200 f044 result rows and unique sample UIDs | PASS | Source-base and final-model per-sample tables each contain 200 unique f044 UIDs spanning workload IDs 001-200. |
| Source enumeration and chiplet identity | PASS | The source oracle contains indices 0-19 with names in exact frozen layout order. `source_response_dataset.py:243-304` uses that order. |
| Source input coordinate convention | PASS | `source_response_dataset.py:243-304` uses row=y, column=x cell-center meshes; raster context construction follows the same convention. |
| Raster containment and overlap checks | PASS | `scripts/build_thermal_impedance_feature_dataset.py:473-555` constructs cell centers and exact rectangle masks and rejects overlap. |
| Target and feature map orientation in code paths | PASS | `scripts/encode_dataset.py:170-197` writes raw 64x64 temperature maps; source and context rasterizers consistently use `[row, col]=[y, x]`. No f044-only transpose branch exists. |
| Source/base/target/prediction UID alignment in available outputs | PASS | The matched f043/f044 table has 200 f044 rows, valid decomposition joins, and zero metric cross-check discrepancy. |
| Stored f044 arrays exist, are finite, and have canonical shapes | NOT VERIFIABLE | The server array root is absent. Available copied metric tables are finite for every required numeric field. |
| Repeated target-map hashes or retained failed HotSpot workdirs | NOT VERIFIABLE | Requires the external labels and work directories. The strict pipeline contract checks labels and retry accounting at `benchmark_v2_pipeline.py:3339-3496`. |
| Full 13/17/33-channel tensor schema and graph schema | NOT VERIFIABLE | The strict pipeline implements shape, finiteness, channel-order, metadata, and graph checks at `benchmark_v2_pipeline.py:3344-3401`; the completed report was not present locally. |
| Objective data corruption or benchmark-rule violation | PASS | None found in the verifiable definition, generation, ordering, and result-alignment evidence. |

There are **0 FAIL** findings. The NOT VERIFIABLE entries are evidence-access limits, not suspected defects. In particular, “fixed material stack and cooling” is verified for the benchmark definitions and producer contract, but not independently byte-audited across the external persisted dataset in this checkout.

## 3. Structural-distribution comparison

The full numeric comparison is in `docs/f044_family_feature_comparison.csv`. Percentiles and z-scores use only the 40 training families.

### What is unusual

- **Package aspect ratio:** `1.8889`, training percentile `97.5`, z-score `5.82`. It is rare under the train distribution but remains inside the training range because f043 is `1.8947`. It is not scalar aspect-ratio extrapolation.
- **Normalized source separation:** mean normalized center distance `0.5898`, training percentile `97.5`, z-score `1.90`, still inside the train maximum (`0.5928`). The package combines long-axis separation with clustering rather than violating spacing rules.
- **Directional source-baseline behavior:** boundary-minus-interior source temperature is `1.814 K`, above the train maximum `0.869 K` (z-score `2.18`). This is the clearest package-response range violation in an inference-time descriptor.
- **Low-frequency source structure:** low-frequency source energy fraction is `0.961259`, only `0.000266` below the training minimum. The formal range violation is real but tiny in absolute magnitude; the stronger signal is directional boundary contrast.

### What is ordinary or covered

Package area (`2448 mm^2`), 20-chiplet count, occupied fraction (`0.46`), chiplet areas, canonical chiplet shape aspect ratios, edge clearance, functional-type fractions, total power, power density, active-source count, and dominant-source share all lie within training-family ranges. f044 is therefore not a cartoonishly oversized, overpowered, overcrowded, or malformed package.

### Matched physical control: f043

The existing matched-workload analysis compares all 200 workload IDs of f043 and f044. Their package dimensions, aspect ratios, cell-size anisotropy, and minimum gaps are close, yet their errors differ sharply:

| Family | Source MAE (K) | Final MAE (K) | Mean error (K) | Centered error (K) |
|---|---:|---:|---:|---:|
| f043 | 1.895 | 0.252 | 0.204 | 0.146 |
| f044 | 3.625 | 2.967 | 2.291 | 2.100 |

This comparison argues against package aspect ratio or grid anisotropy alone. The differentiator is the interaction between f044's placement topology and directional/package-boundary thermal response.

### Analysis warning: aspect-ratio naming

`scripts/analyze_benchmark_v2_family_ood.py:413-446` uses orientation-invariant package aspect ratio but labels raw chiplet `width/height` as `chiplet_aspect_ratio`. The frozen f044 descriptors correctly use `max(width,height)/min(width,height)`. This is an analysis-labeling inconsistency, not a dataset defect. Future tables should call the raw quantity `chiplet_width_height_ratio` or use the canonical orientation-invariant definition.

## 4. Workload audit

Deterministic regeneration from the frozen family and benchmark seed produced:

- 200 workloads, 200 unique UIDs, and 200 unique content hashes.
- Total power: minimum `61.845 W`, mean `552.603 W`, maximum `1661.103 W`.
- Explicitly active sources: minimum 2, mean 9.25, maximum 20.
- High-load sources: mean 2.12; maximum 20.
- Dominant-source power share: mean `0.254`; maximum `0.655`.
- Source power: `0.686-181.712 W`; all values finite and positive.
- Every record uses the nominal workload and canonical per-type idle floor.

Against the 40 training families, f044's family-aggregated workload values are covered: total-power mean is at the 75th percentile, mean power density at 82.5th, maximum power density at 75th, active-source count at 55th, and dominant-source share at 27.5th. The 10th-percentile total power is upper-tail (90th family percentile), but no workload-family statistic is outside the training range.

Errors scale broadly with workload severity rather than being caused by a few corrupt rows. Final f044 MAE has median `2.526 K` and 90th percentile `6.245 K`; 83/200 samples exceed `3 K`. The worst 10% contributes only 25.6% of the summed per-sample MAE. Source-baseline and final errors have Spearman correlation `0.993`, and final MAE correlates `0.976` with total power. This is a systematic response-scaling problem.

Actual persisted YAML hashes, target-map hashes, and HotSpot workdirs could not be reopened locally. These checks should be satisfied by archiving and citing the server strict-validation report; they should not be inferred from model metrics.

## 5. HotSpot and preprocessing alignment audit

No f044-specific preprocessing branch was found. The relevant paths use a common convention:

1. The frozen layout is enumerated in list order.
2. Raster rows represent physical y and columns physical x.
3. Cell centers are derived from package dimensions and 64x64 grid spacing.
4. Exact rectangles define occupancy, source masks, and chiplet metrics.
5. Source-response targets and predictions are keyed by source index/name.
6. Package reconstruction requires every source, scales its `K/W` field by source power, sums aligned 64x64 fields, and adds ambient once (`scripts/evaluate_source_response_model.py:302-368`).

The strongest semantic check is the source oracle: summing the ground-truth isolated-source rises reconstructs the f044 package reference to `0.038740 K` MAE. A transpose, wrong source order, duplicated ambient, wrong source power, or gross floorplan mismatch would not normally produce that near-exact reconstruction while the learned reconstruction remains `3.665891 K`. This isolates the failure to learned K/W prediction, subject to the external-file limitations above.

## 6. Error decomposition

### Source-superposition baseline across 200 workloads

| Metric | f044 |
|---|---:|
| Full-field MAE | 3.624509 K |
| RMSE | 4.282 K |
| Mean signed prediction error | -2.522442 K |
| Mean-rise absolute error | 2.522442 K |
| Centered-field MAE | 2.710382 K |
| Hotspot-temperature error | 2.160 K |
| Hotspot-location error | 20.145 cells |
| Low-frequency error-energy fraction | 0.679221 |

Relative to train families, source MAE is `5.89` standard deviations high, signed bias is `-6.34`, centered error is `4.04`, and low-frequency error energy is `4.06`. The source model failure is a mixture of coherent global underprediction and broad spatial mismatch.

### Source-level oracle diagnosis

For the audited package, the learned source reconstruction is `3.665891 K` MAE versus `0.038740 K` for the ground-truth isolated-source sum. Individual learned source-field physical MAEs are modest (`0.0415-1.095 K`). Their signed means split evenly between positive and negative, but unequal magnitudes yield a large net cold bias. CPU and GPU source errors average about `0.80 K`; NPU about `0.62 K`; HBM/DRAM/IO about `0.04-0.06 K`. The package signed error (`-2.793905 K`) nearly equals the summed source mean error (`-2.755170 K`) plus the small oracle remainder.

Conclusion: HotSpot remains nearly superposable; the universal source-response network extrapolates the f044 source-to-grid responses poorly, especially for compute sources in this topology.

### Residual-CNN correction

| Component | MAE |
|---|---:|
| Source baseline | 3.624509 K |
| Required scalar mean correction | 2.522442 K average magnitude |
| Predicted scalar correction | 0.232 K average |
| Scalar mean-correction error | 2.2908 K |
| Centered-spatial correction error | 2.1003 K |
| Final temperature | 2.966781 K |

All 200 true f044 residual means are positive. The CNN correction improves 199/200 samples, so it is not destabilizing the baseline, but it systematically underestimates the necessary package-wide correction and does not fully repair the low-frequency spatial response.

Oracle component analysis reports that f044's baseline error is reducible by approximately `0.867 K` from a perfect mean correction alone, `0.561 K` from a perfect low-frequency correction alone, and `0.319 K` from a perfect high-frequency correction alone. Correcting mean plus low-frequency structure removes `2.243 K`. About 68% of centered squared-error energy is low frequency, with elevated low-frequency boundary error. The most defensible next model diagnosis is therefore **mean plus coarse package-scale response**, not a fine-hotspot-only failure.

## 7. Comparison with other held-out families and models

| Strict test family | ChipTherm-CNN MAE (K) |
|---|---:|
| f008 | 0.779 |
| f016 | 1.016 |
| f027 | 0.966 |
| f033 | 0.926 |
| f044 | 2.967 |

f044 is also difficult for independently formulated models: source-superposition `3.625 K`, direct CNN `2.226 K`, direct FNO `2.533 K`, direct U-FNO `2.691 K`, and direct SAU-FNO `2.370 K`. The direct models reduce error relative to the source-residual pipeline, but none makes f044 ordinary. This cross-model consistency weighs against a checkpoint-specific or residual-sign bug.

The descriptor analysis classifies f044 as **close descriptor neighbor but different thermal response**. Its nearest training family is f043, followed by f015, f002, f040, and f006. Its scalar geometric descriptors have neighbors; its thermal response under long-axis clustered placement does not.

## 8. Recommendation

### Keep f044, with stratified reporting

The benchmark should remain frozen. No objective exclusion rule is triggered. Report:

- the aggregate strict held-out result over all five families;
- each held-out family separately;
- f044 as the preregistered compound-OOD / long-axis-cluster regime;
- the four-family non-f044 aggregate as a descriptive stratum, never as a replacement headline metric;
- source-baseline and final-model performance on f044 to expose where error enters.

Exclusion would become defensible only if server-side reopening demonstrates a reproducible violation such as invalid layout geometry, mismatched UID/source power, malformed/non-finite target, failed simulation retained as valid, or a preprocessing rule applied differently to f044. The exclusion criterion would have to be declared independently of accuracy, applied to all 50 families, followed by a model-blind replacement process and complete reruns of every reported method. No such evidence currently exists.

## 9. Evidence inventory

### Frozen benchmark and validators

- `configs/benchmark_v2_50family/families/f044.yaml:1-251`
- `configs/benchmark_v2_50family/splits/primary_family_split.yaml:1-56`
- `configs/benchmark_v2_50family/design_proposal.yaml:123-134`
- `configs/benchmark_v2_50family/family_manifest.yaml:327-329`
- `src/chiptherm/benchmark_v2.py:191`
- `src/chiptherm/benchmark_v2.py:654-725`
- `src/chiptherm/validate.py:46-101`
- `src/chiptherm/benchmark_v2_workloads.py:26-155`
- `src/chiptherm/benchmark_v2_workloads.py:423-476`
- `src/chiptherm/benchmark_v2_workloads.py:544-640`
- `src/chiptherm/benchmark_v2_pipeline.py:3319-3509`
- `src/chiptherm/benchmark_v2_pipeline.py:760-795`
- `src/chiptherm/benchmark_v2_pipeline.py:912-940`

### Raster, source, and reconstruction semantics

- `scripts/encode_dataset.py:170-197`
- `scripts/build_thermal_impedance_feature_dataset.py:473-555`
- `src/chiptherm/ml/source_response_dataset.py:18-36`
- `src/chiptherm/ml/source_response_dataset.py:243-304`
- `scripts/evaluate_source_response_model.py:302-368`

### Existing result artifacts

- `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/family_ood_analysis/family_descriptors.csv:51`
- `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/family_ood_analysis/heldout_feature_zscores.csv:1496`
- `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/family_ood_analysis/family_ood_report.md:7-35`
- `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/f043_f044_physical_comparison/f043_f044_physical_report.md:3-31`
- `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/residual_decomposition/residual_decomposition_report.md:9-27`
- `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/oracle_residual_components/oracle_residual_component_report.md:18-25`
- `outputs/benchmark_v2_50family/source_superposition/final_train40_source_v1/base_quality_by_case.csv:6`
- `outputs/benchmark_v2_50family/source_response/final_train40_v1/evaluation/oracle_primary_test/metrics_by_case.csv:6`
- `outputs/benchmark_v2_50family/source_response/final_train40_v1/evaluation/oracle_primary_test/package_bias_diagnostics.csv:6`

## 10. Benchmark and manuscript claims requiring revision

1. Describe f044 as an **intentional compound-OOD long-axis-cluster family**, not an anomalous data point discovered after evaluation.
2. Do not claim f044 package aspect ratio is outside the training range. It is high-z but covered by f043; topology and directional response are the more precise distinction.
3. Distinguish physical source superposition from the learned source-response approximation. The f044 oracle (`0.038740 K`) supports superposition; the learned K/W fields are the failing component.
4. Report aggregate and per-family held-out metrics. The aggregate alone hides a meaningful regime-specific generalization failure.
5. State that the remaining f044 error contains both scalar mean undercorrection and low-frequency centered mismatch; it is not primarily a local hotspot-resolution problem.
6. Archive the server-side strict validation report and hashes with the benchmark release. This local audit cannot independently certify missing/duplicate arrays or failed workdirs without the external root.
7. Rename or redefine the orientation-sensitive `chiplet_aspect_ratio` analysis descriptor to avoid conflating chiplet rotation with shape elongation.

## Final determination

- **Objective corruption found:** No.
- **Objective benchmark-rule violation found:** No.
- **Scientifically justified action:** Keep f044 and add stratified reporting.
- **Is changing the frozen split defensible now?** No.
