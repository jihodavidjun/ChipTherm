# Benchmark v2 Risk Register

| ID | Risk | Likelihood | Impact | Evidence | Mitigation / gate | Owner decision needed |
|---|---|---|---|---|---|---|
| R1 | V2 repeats current case semantics in which geometry changes per workload, defeating fixed-family evaluation | high | critical | both legacy and extension generators resample structure within a case | separate immutable family template builder from workload builder; Stage 1 fingerprint audit | approve fixed-family definition |
| R2 | Existing merged 20-family caches are mistaken for complete canonical sources | high | critical | compatibility report marks 28,840 live-integrated legacy path fields unavailable; local extension context tree is incomplete | preserve legacy; use explicit artifact class and parent locks; never infer canonical status from a merged index | name canonical release roots |
| R3 | Absolute or stale paths break relocation and downstream regeneration | high | high | checkpoints include `/nethome` paths; historical indices referenced deleted roots | one declared data root, relative paths, relocation test, no per-column fallback | choose path contract |
| R4 | Source-response model leaks held-out-family target information | medium | critical | learned base depends on isolated HotSpot labels and train-only normalization | split-specific source checkpoint lineage; reject incompatible parent family sets | approve leakage policy |
| R5 | Material/cooling variation becomes an unobserved causal variable | high if enabled | critical | active 15 metadata excludes thermal stack fields; source-response input omits them | freeze material/cooling in v2.0; defer OOD physics split | approve v2.0 scope |
| R6 | Fifty families are near duplicates or test is separable by one scalar | medium | high | current labels span broad but sometimes case-identifying scalar ranges | structural fingerprints, nearest-neighbor review, marginal coverage report, compound OOD design | set similarity threshold |
| R7 | Two hundred workloads do not cover extreme source interaction | medium | high | current generator has no explicit workload strata | fixed 200-row stratification; Stage 3 learning/coverage curves | approve workload strata |
| R8 | Power patterns are unrealistic or violate type limits | medium | high | current per-chiplet independent draws let total power emerge | joint sampling with per-type density constraints and design review | define authoritative type ranges |
| R9 | Exact or near duplicate workload leakage recurs | medium | high | prior deterministic sequences produced 990 duplicate rows | versioned seed derivation, content hashes, near-duplicate audit before split | choose near-duplicate rule |
| R10 | Graph storage/runtime grows unexpectedly for dense families | medium | medium | full directed edges scale O(n^2); current graphs average 35 KiB at current counts | cap/review die counts, measure Stage 2/3 bytes and runtime | set maximum dies |
| R11 | Isolated-source generation is naively repeated per workload | medium | critical | current source-response data is package-row based | one source isolation per fixed family/source; prohibit per-workload fanout | approve canonical isolation semantics |
| R12 | Source superposition is invalid under new nonlinear physics | low in frozen v2.0 | critical | formulation assumes linear steady-state response | freeze physics stack and steady-state conventions; run linearity oracle pilot | confirm HotSpot settings |
| R13 | Train-only normalization is reused across protocols | medium | high | checkpoints encode data-specific statistics | protocol/fold-specific normalization artifact with train manifest hash | enforce in loader/trainer |
| R14 | Hard-coded ten/twenty families or 400 workloads blocks generation | high | high | extension config enforces ten cases and scripts embed v1 names/default counts | parameterize before Stage 2; unit tests at 5/10/50 families | authorize compatibility work |
| R15 | Canonical/derived files fill home quota or scratch is purged | high | critical | proposal is 14-16 GiB all retained; `/tmp` is not durable | persistent project root plus scratch, quotas and retention recorded before generation | provide mounts/quota/owner |
| R16 | Intermediate directories are deleted because names imply obsolescence | medium | critical | current descendants still depend on apparently legacy raw/source files | dependency lock, must-not-delete manifest, parent validation | adopt artifact governance |
| R17 | HotSpot failures silently bias accepted families/workloads | medium | high | prior extension required retry/repair hardening | validate before launch, durable failure report, no silent row exclusion, acceptance-rate report | set failure policy |
| R18 | Grid/axis/unit conventions drift between v1 and v2 | medium | critical | physical channels depend on width/height and row/column orientation | synthetic orientation/unit tests and representative v1/v2 statistics | freeze 64 x 64 convention |
| R19 | Benchmark claim lacks external comparison | medium | high | current results are internally comparative | release baselines and include independent prior/compact surrogate where possible | identify external baseline |
| R20 | Results drive post-hoc split/family changes | medium | critical | many possible OOD partitions | freeze split hashes before full model training; publish all declared folds | approve benchmark governance |
| R21 | Legacy benchmark is overwritten or silently folded into v2 | low if gated | critical | current paths and scripts have many v1/v2 aliases | separate version roots and immutable legacy hash manifest | approve separate-version policy |
| R22 | Checkpoint cannot be reproduced because model-ready caches outlive canonical parents | medium | critical | source models build inputs lazily; current extension context directories are not all present locally | retain canonical family/workload/labels, source checkpoint normalization, code/hash lineage; reproduction smoke before release | choose retention set |

## Highest-priority blockers

1. Fixed family semantics are not implemented by the current generator.
2. The authoritative project/scratch storage roots and retention policy are not
   specified.
3. Material/cooling variation must be removed from v2.0 or represented by a
   formally extended input schema and retrained source model.
4. Source-response checkpoint lineage must be protocol-aware.
5. The family-count-agnostic, relative-path, manifest-locked build contract is
   not yet implemented.

These are full-generation blockers, not reasons to reject the benchmark
concept. Stage 1 should resolve them before any HotSpot budget is committed.
