# Benchmark v2 Phase 3 Scale Pilot

## Scope

Phase 3 is `pilot_10x50`: ten fixed Phase 1 package families and fifty deterministic workloads per family, for 500 package samples. It is a scale, diversity, storage, and failure-recovery gate before any 50x200 build. It does not change family geometry or train a final residual model.

## Frozen family selection

| Family | Category / role | Primary split | Isolated-source role | Phase 2 overlap |
|---|---|---|---|---|
| f002 | HPC | train | learned-train eligible | yes |
| f007 | HPC | validation | oracle only | no |
| f009 | memory heavy | train | learned-train eligible | no |
| f014 | compute heavy | train | learned-train eligible | no |
| f023 | mixed heterogeneous | validation | oracle only | yes |
| f029 | sparse / long-range | train | learned-train eligible | yes |
| f032 | dense 64-die | train | learned-train eligible | yes |
| f039 | distributed | train | learned-train eligible | no |
| f040 | edge constrained | train | learned-train eligible | no |
| f044 | package-aspect compound OOD | test | oracle only | yes |

The complete rationale and immutable selection-content hash live in `configs/benchmark_v2_50family/pilot_10x50.yaml`.

## Workload matrix

The matrix is five load regimes by ten activity/topology regimes. The `phase2_reference` regime retains each accepted Phase 2 workload's exact source powers and content hash. Four additional within-type load fractions (`very_low=0.06`, `moderate=0.36`, `high=0.72`, `stress=0.96`) are crossed with balanced, memory-dominant, compute-dominant, type-specific, heterogeneous, dense, single-dominant, clustered-interaction, and spatially-distributed regimes. Every family receives each of the 50 cells exactly once.

## Storage and runtime gate

Before execution, treat the scale estimate as a range, not a claim: package-level artifacts should scale roughly 10x from the accepted 50-sample pilot, while isolated-source cost scales with the chiplet counts of five overlapping plus five new families rather than with 500 workloads. HotSpot wall time should be approximately 9x the Phase 2 package-label time when the 50 accepted reference samples are reused. Staging must have at least 2x the projected retained Phase 3 footprint available for retries and atomic promotion.

After strict validation, authoritative measured values are written to:

`canonical/manifests/pilot_10x50_validation_report.json`

That report includes measured retained bytes, inode count, peak observed staging, runtime by stage, HotSpot mean/median/p95, slowest families/cells, power and temperature distributions, 10,000-sample projections, relocation, loader/forward status, Phase 2 reuse, and Phase 2 immutability.

## Authorization rule

- **GO**: all strict checks pass, relocation passes, the visual contact sheet is reviewed, no suspicious thermal output remains, and measured full-build storage/runtime fit the server budget.
- **GO WITH MANUAL REVIEW**: strict data checks pass but relocation or visual review has not yet been recorded.
- **NO-GO**: any missing sample, lineage leak, unresolved path, invalid tensor/schema, inconsistent retry accounting, or Phase 2 mutation.

The implementation recommendation before the real run is **GO WITH MANUAL REVIEW**. Full-build authorization is intentionally deferred to the measured Phase 3 report.
