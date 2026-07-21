# Benchmark v2 Stage 1 Review

## Scope

Stage 1 instantiated fixed structural family definitions only. It ran no HotSpot simulations, generated no workloads or thermal labels, and built no model tensors.

## Acceptance summary

- Recommendation: **GO WITH MANUAL REVIEW**
- Exactly 50 fixed families present: **yes**
- All geometry validation passed: **yes**
- Material/cooling and HotSpot settings fixed: **yes**
- Primary split counts: `{'train': 40, 'val': 5, 'test': 5}`
- Rotational groups: `{1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}`
- Suspicious pair count at distance < 0.150, plus intentional matched pairs: 2
- Unexplained cross-split suspicious pairs: 0
- Test-separating scalar descriptors: none

## Taxonomy

| Category | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| analog_mems | 4 | 3 | 0 | 1 |
| chiplet_size_aspect | 2 | 2 | 0 | 0 |
| compact_clustered | 3 | 3 | 0 | 0 |
| compute_heavy | 4 | 3 | 0 | 1 |
| dense_high_die | 3 | 2 | 0 | 1 |
| distributed | 3 | 3 | 0 | 0 |
| edge_constrained | 2 | 1 | 1 | 0 |
| hpc | 8 | 6 | 1 | 1 |
| memory_heavy | 4 | 3 | 1 | 0 |
| mixed_heterogeneous | 7 | 6 | 1 | 0 |
| package_scale_aspect | 3 | 2 | 0 | 1 |
| spacing | 2 | 2 | 0 | 0 |
| sparse_low_die | 3 | 2 | 1 | 0 |
| whitespace | 2 | 2 | 0 | 0 |

## Nearest and suspicious families

| Family A | Family B | Distance | Cross split | Intentional | Justification |
|---|---|---:|---|---|---|
| f049 | f050 | 0.68316 | False | True | Intentional matched pair isolating source spacing. |
| f047 | f048 | 0.70013 | False | True | Intentional same composition/package pair isolating whitespace. |

The ten closest per-family neighbor records include:

- `f006` -> `f008`: 0.22399 (cross split: True)
- `f008` -> `f006`: 0.22399 (cross split: True)
- `f039` -> `f008`: 0.22425 (cross split: True)
- `f004` -> `f019`: 0.23999 (cross split: False)
- `f019` -> `f004`: 0.23999 (cross split: False)
- `f023` -> `f006`: 0.25861 (cross split: True)
- `f031` -> `f033`: 0.26013 (cross split: True)
- `f033` -> `f031`: 0.26013 (cross split: True)
- `f002` -> `f041`: 0.26254 (cross split: True)
- `f041` -> `f002`: 0.26254 (cross split: True)

## Split coverage

All tested validation/test scalar values lie inside their corresponding train ranges.

No audited individual scalar perfectly separates all five test families from train.

## Compound-OOD assessment

The designated test families `f008`, `f016`, `f027`, `f033`, and `f044` combine covered marginal geometry/type regimes. Their individual audited scalar values must remain covered by train; the held-out object is the joint topology. No model result was used to choose these assignments.

## Manual review

Manual visual review is required for every family before Phase 2. Additional attention should go to the intentional matched pairs `f047/f048` and `f049/f050`, every pair listed above as suspicious, the high-die families `f031-f033`, and the five compound-OOD test families.

## Validation problems

- None.

## Phase 2 gate

**GO WITH MANUAL REVIEW**. `GO WITH MANUAL REVIEW` means all machine acceptance gates pass, but intentional matched/nearest structures still require human approval of their layout previews. This report alone does not authorize workload or HotSpot generation.
