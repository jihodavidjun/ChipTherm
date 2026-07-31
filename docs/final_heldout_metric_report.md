# Final Frozen Benchmark v2 Held-Out Metric Report

## A. Executive Summary

This report compares the frozen primary CNN, FNO, U-FNO, and SAU-FNO backbones on the same 1,000-sample strict held-out-family protocol. Reference temperatures are HotSpot-generated fields. No inference or model selection was performed to create this report.

The lowest held-out full-map MAE is the physics-guided CNN at 1.3306 K. Physics guidance lowers MAE for every backbone, but it does not uniformly improve RMSE or hotspot metrics: the direct SAU-FNO has the lowest held-out RMSE (2.3983 K), and direct FNO has the lowest recovered absolute peak-value error (1.4970 K). This supports early-stage screening use, not thermal signoff.

## B. Primary Held-Out-Family Results

| Backbone | Mode | Residual Params (M) | Total Params (M) | MAE (K) | RMSE (K) | Signed Peak Error (K) | Absolute Peak Error (K) | Hotspot Location Error (cells) |
|---|---|---|---|---|---|---|---|---|
| Shared source-response | Source-superposition baseline | 0.0000 | 0.4756 | 1.6681 | 3.1485 | 4.4357 | not recoverable from current artifacts | 15.1260 |
| CNN | Direct | n/a (direct) | 2.1818 | 1.7674 | 2.9134 | -1.0427 | 2.0075 | 10.5843 |
| CNN | Physics-guided residual | 2.1888 | 2.6644 | 1.3306 | 2.4861 | 1.3442 | 2.7992 | 12.0850 |
| FNO | Direct | n/a (direct) | 2.3886 | 1.4941 | 2.4471 | 0.8873 | 1.4970 | 9.5505 |
| FNO | Physics-guided residual | 2.3949 | 2.8705 | 1.3711 | 2.5669 | 3.7050 | 4.2500 | 12.1993 |
| U-FNO | Direct | n/a (direct) | 4.0193 | 1.7541 | 2.8210 | -0.1292 | 1.8248 | 9.2051 |
| U-FNO | Physics-guided residual | 4.0256 | 4.5012 | 1.3654 | 2.5497 | 2.8837 | 3.7007 | 12.5164 |
| SAU-FNO | Direct | n/a (direct) | 4.0225 | 1.5453 | 2.3983 | -0.8818 | 1.6106 | 10.8415 |
| SAU-FNO | Physics-guided residual | 4.0288 | 4.5044 | 1.3995 | 2.5607 | 3.2819 | 3.9324 | 12.6819 |

For direct rows, total parameters are the direct backbone. For residual rows, total parameters equal the residual backbone plus the shared 475,585-parameter source-response model; the shared model is counted once per complete system.

Row-level authoritative sources:

- Source baseline: `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/evaluation/primary_test_families/metrics.json`
- CNN / Direct: `outputs/benchmark_v2_50family/package_direct/direct_temperature_feature_fusion_normalized_train40_seed1/evaluation/primary_test_families/metrics.json`
- CNN / Physics-guided residual: `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/evaluation/primary_test_families/metrics.json`
- FNO / Direct: `outputs/benchmark_v2_50family/fno/direct_temperature_fno_normalized_train40_seed1/evaluation/primary_test_families/metrics.json`
- FNO / Physics-guided residual: `outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1/evaluation/primary_test_families/metrics.json`
- U-FNO / Direct: `outputs/benchmark_v2_50family/ufno/direct_temperature_ufno_normalized_train40_seed1/evaluation_primary_test/primary_test_families/metrics.json`
- U-FNO / Physics-guided residual: `outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1/evaluation_primary_test/primary_test_families/metrics.json`
- SAU-FNO / Direct: `outputs/benchmark_v2_50family/sau_fno/direct_temperature_sau_fno_normalized_train40_seed1/evaluation_primary_test/primary_test_families/metrics.json`
- SAU-FNO / Physics-guided residual: `outputs/benchmark_v2_50family/sau_fno/residual_sau_fno_decomposed_train40_seed1/evaluation_primary_test/primary_test_families/metrics.json`

Additional held-out bias and peak-tail diagnostics:

| Backbone | Mode | Mean Signed Error (K) | Abs Peak <1 K | Abs Peak <2 K | Abs Peak <3 K |
|---|---|---|---|---|---|
| Shared source-response | Source-superposition baseline | -0.3045 | not recoverable from current artifacts | not recoverable from current artifacts | not recoverable from current artifacts |
| CNN | Direct | -0.2558 | 0.3320 | 0.6260 | 0.8050 |
| CNN | Physics-guided residual | -0.3463 | 0.2740 | 0.4890 | 0.6460 |
| FNO | Direct | 0.0581 | 0.4600 | 0.7210 | 0.8540 |
| FNO | Physics-guided residual | -0.1904 | 0.2780 | 0.4310 | 0.5600 |
| U-FNO | Direct | 0.5725 | 0.4480 | 0.6680 | 0.8020 |
| U-FNO | Physics-guided residual | -0.2861 | 0.3350 | 0.5010 | 0.6180 |
| SAU-FNO | Direct | 0.0012 | 0.4600 | 0.7510 | 0.8630 |
| SAU-FNO | Physics-guided residual | -0.0977 | 0.2800 | 0.4890 | 0.5980 |

## C. Direct-to-Physics-Guided Improvement

| Backbone | Direct MAE | Physics-guided MAE | MAE Reduction | Direct RMSE | Physics-guided RMSE | RMSE Reduction |
|---|---|---|---|---|---|---|
| CNN | 1.7674 | 1.3306 | 0.4368 | 2.9134 | 2.4861 | 0.4273 |
| FNO | 1.4941 | 1.3711 | 0.1230 | 2.4471 | 2.5669 | -0.1198 |
| U-FNO | 1.7541 | 1.3654 | 0.3887 | 2.8210 | 2.5497 | 0.2712 |
| SAU-FNO | 1.5453 | 1.3995 | 0.1458 | 2.3983 | 2.5607 | -0.1624 |

Positive reduction means the physics-guided model improved the metric; negative RMSE reductions are regressions.

## D. Familiar-Family / Interpolation Results

| Backbone | Mode | MAE (K) | RMSE (K) | Mean Signed Error (K) | Absolute Peak Error (K) | Location Error (cells) |
|---|---|---|---|---|---|---|
| CNN | Direct | 0.3699 | 0.5686 | -0.1438 | 0.5520 | 2.7387 |
| CNN | Physics-guided residual | 0.1500 | 0.2837 | 0.0050 | 0.3098 | 1.3812 |
| FNO | Direct | 0.3748 | 0.5807 | -0.0763 | 0.6501 | 2.5277 |
| FNO | Physics-guided residual | 0.1083 | 0.2071 | -0.0096 | 0.1732 | 1.5382 |
| U-FNO | Direct | 0.3384 | 0.5613 | 0.1698 | 0.5729 | 2.1779 |
| U-FNO | Physics-guided residual | 0.0875 | 0.1624 | 0.0051 | 0.1855 | 1.0316 |
| SAU-FNO | Direct | 0.5282 | 0.7536 | -0.0503 | 1.1172 | 3.2008 |
| SAU-FNO | Physics-guided residual | 0.0920 | 0.1799 | 0.0021 | 0.1700 | 1.3218 |

The familiar-family protocol contains 800 held-out workload samples from known package families.

## E. Per-Held-Out-Family Results

| Backbone | Mode | Family | MAE (K) | RMSE (K) |
|---|---|---|---|---|
| Shared source-response | Source-superposition baseline | f008 | 0.9443 | 1.5066 |
| Shared source-response | Source-superposition baseline | f016 | 1.3805 | 2.5223 |
| Shared source-response | Source-superposition baseline | f027 | 1.3508 | 3.2609 |
| Shared source-response | Source-superposition baseline | f033 | 1.0406 | 1.9174 |
| Shared source-response | Source-superposition baseline | f044 | 3.6245 | 5.1598 |
| CNN | Direct | f008 | 0.9960 | 1.5363 |
| CNN | Direct | f016 | 1.6368 | 2.6495 |
| CNN | Direct | f027 | 1.5943 | 2.5527 |
| CNN | Direct | f033 | 2.3840 | 3.8958 |
| CNN | Direct | f044 | 2.2260 | 3.3714 |
| CNN | Physics-guided residual | f008 | 0.7786 | 1.2733 |
| CNN | Physics-guided residual | f016 | 1.0157 | 1.8610 |
| CNN | Physics-guided residual | f027 | 0.9661 | 2.1006 |
| CNN | Physics-guided residual | f033 | 0.9257 | 1.6290 |
| CNN | Physics-guided residual | f044 | 2.9668 | 4.3306 |
| FNO | Direct | f008 | 1.2045 | 1.6399 |
| FNO | Direct | f016 | 1.2637 | 1.9534 |
| FNO | Direct | f027 | 0.9886 | 1.7173 |
| FNO | Direct | f033 | 1.4808 | 2.4141 |
| FNO | Direct | f044 | 2.5332 | 3.8288 |
| FNO | Physics-guided residual | f008 | 0.8661 | 1.3415 |
| FNO | Physics-guided residual | f016 | 1.0354 | 1.8679 |
| FNO | Physics-guided residual | f027 | 1.2101 | 2.7680 |
| FNO | Physics-guided residual | f033 | 1.0029 | 1.8248 |
| FNO | Physics-guided residual | f044 | 2.7411 | 4.0823 |
| U-FNO | Direct | f008 | 1.1915 | 1.6433 |
| U-FNO | Direct | f016 | 1.4636 | 2.2508 |
| U-FNO | Direct | f027 | 1.0889 | 1.7757 |
| U-FNO | Direct | f033 | 2.3359 | 3.5678 |
| U-FNO | Direct | f044 | 2.6907 | 4.0175 |
| U-FNO | Physics-guided residual | f008 | 0.8330 | 1.2801 |
| U-FNO | Physics-guided residual | f016 | 1.0264 | 1.8778 |
| U-FNO | Physics-guided residual | f027 | 1.1494 | 2.5539 |
| U-FNO | Physics-guided residual | f033 | 0.9274 | 1.6744 |
| U-FNO | Physics-guided residual | f044 | 2.8909 | 4.2444 |
| SAU-FNO | Direct | f008 | 1.0682 | 1.5966 |
| SAU-FNO | Direct | f016 | 1.5732 | 2.3256 |
| SAU-FNO | Direct | f027 | 1.0488 | 1.6951 |
| SAU-FNO | Direct | f033 | 1.6664 | 2.6165 |
| SAU-FNO | Direct | f044 | 2.3698 | 3.3291 |
| SAU-FNO | Physics-guided residual | f008 | 0.9665 | 1.4606 |
| SAU-FNO | Physics-guided residual | f016 | 1.0180 | 1.8341 |
| SAU-FNO | Physics-guided residual | f027 | 1.1498 | 2.6191 |
| SAU-FNO | Physics-guided residual | f033 | 0.9860 | 1.6901 |
| SAU-FNO | Physics-guided residual | f044 | 2.8770 | 4.1918 |

Unweighted variation across the five held-out test families:

| Backbone | Mode | MAE Mean | MAE Min | MAE Max | MAE Std | RMSE Mean | RMSE Min | RMSE Max | RMSE Std |
|---|---|---|---|---|---|---|---|---|---|
| Shared source-response | Source-superposition baseline | 1.6681 | 0.9443 | 3.6245 | 0.9928 | 2.8734 | 1.5066 | 5.1598 | 1.2871 |
| CNN | Direct | 1.7674 | 0.9960 | 2.3840 | 0.4965 | 2.8011 | 1.5363 | 3.8958 | 0.8010 |
| CNN | Physics-guided residual | 1.3306 | 0.7786 | 2.9668 | 0.8219 | 2.2389 | 1.2733 | 4.3306 | 1.0809 |
| FNO | Direct | 1.4941 | 0.9886 | 2.5332 | 0.5426 | 2.3107 | 1.6399 | 3.8288 | 0.8056 |
| FNO | Physics-guided residual | 1.3711 | 0.8661 | 2.7411 | 0.6937 | 2.3769 | 1.3415 | 4.0823 | 0.9693 |
| U-FNO | Direct | 1.7541 | 1.0889 | 2.6907 | 0.6417 | 2.6510 | 1.6433 | 4.0175 | 0.9643 |
| U-FNO | Physics-guided residual | 1.3654 | 0.8330 | 2.8909 | 0.7699 | 2.3261 | 1.2801 | 4.2444 | 1.0442 |
| SAU-FNO | Direct | 1.5453 | 1.0488 | 2.3698 | 0.4836 | 2.3126 | 1.5966 | 3.3291 | 0.6356 |
| SAU-FNO | Physics-guided residual | 1.3995 | 0.9665 | 2.8770 | 0.7415 | 2.3591 | 1.4606 | 4.1918 | 0.9957 |

## F. Metric Definitions

- **Full-map MAE:** mean absolute cell-wise difference from the HotSpot-generated reference field over every sample and all 64x64 cells.
- **Full-map RMSE:** square root of the global mean squared cell error. This report uses `rmse_K`/`global_pixel_rmse_K`, not the separately saved mean of per-sample RMSEs.
- **Mean signed error:** mean of `prediction - reference` over every cell; positive values indicate overprediction.
- **Signed peak-value error:** `max(prediction) - max(reference)`, averaged over samples. The evaluator computes this from each map's independent argmax (`scripts/evaluate_residual_cnn.py:1956-1960`).
- **Absolute peak-value error:** `abs(max(prediction) - max(reference))`, averaged over samples (`scripts/evaluate_residual_cnn.py:1062`). It is not the absolute value of the signed average.
- **True-hotspot-location error:** `prediction[argmax(reference)] - reference[argmax(reference)]`. The current evaluator does not save this metric.
- **Hotspot-location distance:** Euclidean row/column distance in grid cells between `argmax(prediction)` and `argmax(reference)` (`scripts/evaluate_residual_cnn.py:1956-1959`).
- **Physical hotspot distance:** not reported. Saved artifacts retain only scalar cell distance, not directional displacement; family-dependent and potentially anisotropic pitch prevents an unambiguous conversion.

## G. Checkpoint and Protocol Audit

- Source version: `source_superposition_final_train40_source_v1`.
- Authoritative split manifest: `$CHIPTHERM_V2_DATA_ROOT/derived/indices/full_50x200/source_superposition/source_superposition_final_train40_source_v1/index_manifest.json` (SHA-256 `2797a69a82e5d1c7aebd52babbb846275c36745eca6882ebce73b4a54b52530c`).
- Training families (40): f001, f002, f003, f004, f005, f006, f009, f010, f011, f013, f014, f015, f017, f018, f019, f020, f021, f022, f024, f025, f026, f028, f029, f031, f032, f034, f035, f036, f037, f038, f039, f040, f042, f043, f045, f046, f047, f048, f049, f050.
- Held-out validation families (5): f007, f012, f023, f030, f041.
- Strict held-out test families (5): f008, f016, f027, f033, f044.
- All eight model evaluations use identical ordered sample UIDs: 800 familiar-family, 1,000 held-out validation, and 1,000 held-out test samples.
- Every lineage records the same train-index SHA-256 and internal-validation-index SHA-256, and records `primary_heldout_used_for_selection=false`.
- `best.pt` selection is minimum internal-validation final-temperature MAE (`scripts/train_residual_cnn.py:1413-1418`). Held-out validation families gated architecture progression; strict held-out test families did not select checkpoint weights.
- All runs have a nominal 100-epoch budget. Realized histories differ: CNN direct has 99 saved epochs and SAU-FNO direct stopped after 65; this is disclosed because the frozen comparison is not perfectly equal in realized optimization exposure.
- The later compact low-learning-rate continuation and rejected interpolation/soup experiments are excluded.

| Backbone | Mode | Best Epoch | Realized/Budget | Checkpoint SHA-256 | Checkpoint |
|---|---|---|---|---|---|
| Shared source-response | Source-superposition baseline | 65 | 65/100 | 249bfa021ac738c0644e9349e20317a4353434651fb6132a8a91c9e958512421 | `outputs/benchmark_v2_50family/source_response/final_train40_v1/checkpoints/best.pt` |
| CNN | Direct | 79 | 99/100 | 4dc059aa1229332912a585f3760416c045f5aa86d3a0461334a38ddbf27cec14 | `outputs/benchmark_v2_50family/package_direct/direct_temperature_feature_fusion_normalized_train40_seed1/checkpoints/best.pt` |
| CNN | Physics-guided residual | 94 | 100/100 | 4927d8ea274ae5c2e3162ec3a9a244391b6054fde5a07dac5b13aa4818975c1a | `outputs/benchmark_v2_50family/package_residual/feature_fusion_train40_source_v1_seed1/checkpoints/best.pt` |
| FNO | Direct | 82 | 100/100 | e9eb74faae94d73973b0416ee23371e7eb5f8f79701e0d52e30d2b9ee0a7a5b2 | `outputs/benchmark_v2_50family/fno/direct_temperature_fno_normalized_train40_seed1/checkpoints/best.pt` |
| FNO | Physics-guided residual | 100 | 100/100 | c190d9d98779cc7d59dabe33bb67a378ecbdb801afb96af1ff0dfa1b73724f4c | `outputs/benchmark_v2_50family/fno/residual_fno_decomposed_train40_seed1/checkpoints/best.pt` |
| U-FNO | Direct | 85 | 100/100 | bed3e420e63a805dc5c518adfda7859fee19ba3118e9d705eaf251339466a450 | `outputs/benchmark_v2_50family/ufno/direct_temperature_ufno_normalized_train40_seed1/checkpoints/best.pt` |
| U-FNO | Physics-guided residual | 100 | 100/100 | 920f00fcbd90165d0fde2c76f175687f4dbbc2a18364ed56c44ab126e71c899e | `outputs/benchmark_v2_50family/ufno/residual_ufno_decomposed_train40_seed1/checkpoints/best.pt` |
| SAU-FNO | Direct | 45 | 65/100 | f6e9822e2f8367a96fd80b8ef005e458b108a7faf2183742e7d94819ac34685f | `outputs/benchmark_v2_50family/sau_fno/direct_temperature_sau_fno_normalized_train40_seed1/checkpoints/best.pt` |
| SAU-FNO | Physics-guided residual | 97 | 100/100 | a67fb78077c3f4627949beb44e2c8f3bd317338461de98324875ac1161197da0 | `outputs/benchmark_v2_50family/sau_fno/residual_sau_fno_decomposed_train40_seed1/checkpoints/best.pt` |

## H. Missing Data

The following are not recoverable from the locally retained evaluation summaries alone:

- signed and mean absolute error evaluated at the true reference-hotspot location;
- median and 95th-percentile absolute pixel error;
- physical hotspot-location distance;
- source-baseline mean absolute peak error and source-baseline peak-threshold fractions.

Final-temperature prediction arrays are retained, but the external Benchmark v2 target/index tree is not present in this workspace. The first two items can be computed offline on GT from saved predictions and reference arrays without checkpoint inference. Physical distance remains unavailable unless directional argmax displacement and per-axis pitch are recomputed from maps and unambiguous package metadata.

No disagreement was found among duplicate summaries for the selected paths. `rmse_K` equals `global_pixel_rmse_K`; `mean_sample_rmse_K` is a different aggregation and is intentionally not reported as full-map RMSE.

## I. Manual GT Commands for Missing Metrics

No CUDA inference is required because all eight primary-test prediction sets already exist. Re-running inference would not add true-hotspot or pixel-tail fields to the current evaluator. The audit-safe next step is a lightweight CPU post-processing pass on GT using the frozen `family_split/test_index.csv`, its `y_path` arrays, and each run's saved `predictions/*_tpred.npy` arrays. This report intentionally leaves those cells unavailable until that target-backed pass is run.

To reproduce the currently available report artifacts on GT after syncing the saved evaluation trees:

```bash
cd /nethome/$USER/chiptherm
source .venv/bin/activate
python3 scripts/build_final_heldout_metric_report.py
```

If any saved prediction tree is missing on GT, regenerate it with the existing frozen wrapper (substitute only the authoritative checkpoint and output root from the audit table):

```bash
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
export SOURCE_VERSION=source_superposition_final_train40_source_v1
python3 scripts/evaluate_benchmark_v2_models.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --source-version "$SOURCE_VERSION" \
  --checkpoint <AUTHORITATIVE_CHECKPOINT_FROM_TABLE> \
  --out-dir <AUTHORITATIVE_RUN_ROOT>/evaluation_recovery \
  --batch-size 64 --device cuda --workers 4 \
  --protocols primary_test_families --save-predictions
```

## J. SRC-Ready Summary

Across five strictly held-out package families (1,000 workloads), the shared source-superposition baseline achieved 1.668 K MAE. Adding a frozen residual predictor reduced MAE to 1.331 K (CNN), 1.371 K (FNO), 1.365 K (U-FNO), and 1.399 K (SAU-FNO), while direct predictors achieved 1.767 K, 1.494 K, 1.754 K, and 1.545 K, respectively. The physics-guided CNN had the lowest full-map MAE, whereas direct SAU-FNO had the lowest RMSE; these results characterize ChipTherm as an early-stage screening and optimization surrogate against HotSpot-generated reference temperatures, not a signoff replacement.

```latex
\begin{tabular}{llrrr}
\toprule
Backbone & Mode & Params (M) & MAE (K) & RMSE (K) \\
\midrule
CNN & Direct & 2.182 & 1.767 & 2.913 \\
CNN & Residual & 2.664 & 1.331 & 2.486 \\
FNO & Direct & 2.389 & 1.494 & 2.447 \\
FNO & Residual & 2.870 & 1.371 & 2.567 \\
U--FNO & Direct & 4.019 & 1.754 & 2.821 \\
U--FNO & Residual & 4.501 & 1.365 & 2.550 \\
SAU--FNO & Direct & 4.022 & 1.545 & 2.398 \\
SAU--FNO & Residual & 4.504 & 1.399 & 2.561 \\
\bottomrule
\end{tabular}
```

Full-map MAE measures average field fidelity, while peak-value and hotspot-location errors probe different localized failure modes; improving one does not guarantee improving the others.
