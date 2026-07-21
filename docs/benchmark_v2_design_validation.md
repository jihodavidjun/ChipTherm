# Benchmark v2 Design Validation

## Executive decision

`50 fixed families x 200 workloads` is scientifically sound and better aligned
with the observed held-out-family failure than adding another 2,000 randomized
layouts to the current benchmark. It is approved for specification and pilot
work, subject to the gates in this document. It is not yet approved for full
HotSpot generation.

The most important correction is semantic: a v2 family must be one fixed
structural/package template. The current generators vary chiplet dimensions,
placements, whitespace, and power within rows bearing the same case label.
Those labels are useful scenario classes, but they are not fixed structural
families. V2 must not silently inherit that behavior.

## Why 50 x 200 is appropriate

- The current sample-split MAE of about 1.67 K versus strict family-disjoint
  MAE of about 3.95 K indicates structural coverage is more limiting than raw
  workload count.
- Two hundred deliberately stratified workloads are enough to cover common,
  skewed, sparse, and interacting activity patterns for a fixed structure.
- Fifty families allow geometry axes to be crossed rather than represented by
  a single case each.
- The taxonomy, not the product `10,000`, creates research value. Fifty near
  duplicates would not improve the held-out-family claim.

This allocation should be revisited after Stage 3 using workload learning
curves at 25, 50, 100, 150, and 200 workloads/family. If validation saturates
before 100, the remaining budget should increase family count rather than add
workloads.

## Proposed exclusive primary taxonomy

Categories below are exclusive for accounting. Families may carry secondary
tags such as `edge_constrained`, `high_aspect_package`, or `compound_ood`.
Counts sum to exactly 50.

| Primary category | Families | Typical dies | Package range (mm) | Whitespace | Composition and placement | Intended axis | Fold allocation |
|---|---:|---:|---|---|---|---|---|
| HPC CPU/GPU/HBM/DRAM | 8 | 8-40 | 32-65 per side | 0.35-0.65 | compute clusters with adjacent memory stacks and realistic symmetry | compute-memory coupling and scale | 6 train, 1 val, 1 test |
| memory-heavy | 4 | 10-36 | 35-65 | 0.40-0.70 | HBM/DRAM dominant, memory rings or banks around compute | distributed memory heating | 3 train, 1 val |
| compute-heavy | 4 | 6-28 | 28-58 | 0.30-0.60 | CPU/GPU/NPU dominant, few memory or IO dies | high source density | 3 train, 1 test |
| mixed heterogeneous | 7 | 10-40 | 35-70 | 0.35-0.70 | CPU, GPU/NPU, memory, IO and analog blocks | type and size heterogeneity | 6 train, 1 val |
| analog/MEMS-inclusive | 4 | 7-24 | 30-60 | 0.40-0.72 | analog/MEMS kept away from or coupled to hot compute by design | low/high-power adjacency | 3 train, 1 test |
| sparse low-die | 3 | 4-9 | 35-68 | 0.60-0.82 | widely separated large sources | long-range coupling and whitespace | 2 train, 1 val |
| dense high-die | 3 | 40-72 | 45-72 | 0.25-0.48 | packed small dies with legal channels | dense interaction | 2 train, 1 test |
| compact clustered | 3 | 12-36 | 28-52 | 0.25-0.50 | two or more tight functional clusters | near-field crowding | 3 train |
| distributed | 3 | 10-32 | 45-75 | 0.50-0.78 | sources distributed across package | far-field response | 3 train |
| edge-constrained | 2 | 8-24 | 32-65 | 0.35-0.68 | selected hot dies near one or more boundaries | edge/path interaction | 1 train, 1 val |
| package scale/aspect | 3 | 10-32 | 25-78, aspect up to about 2.1 | 0.35-0.72 | comparable topology at different physical scales/aspects | physical scaling | 2 train, 1 test |
| chiplet size/aspect | 2 | 8-28 | 35-65 | 0.35-0.68 | elongated versus compact source rectangles | finite-source shape | 2 train |
| whitespace-focused | 2 | 10-30 | 38-70 | 0.30-0.80 | matched composition with low/high whitespace | spreading area | 2 train |
| spacing-focused | 2 | 10-30 | 38-70 | 0.38-0.70 | matched package/area with near/far spacing | source coupling | 2 train |
| package/material/cooling variation | 0 in v2.0 | n/a | n/a | n/a | held fixed | deferred until represented by model inputs | deferred |

Five proposed test families should also carry a `compound_ood` secondary tag.
Each combines two geometry attributes whose individual ranges occur in train,
while the joint combination does not. This creates a meaningful compositional
test without making one test family identifiable by an unsupported scalar.

The machine-readable family IDs and allocations appear in
`configs/benchmark_v2_50family/design_proposal.yaml`.

## Workload allocation

Each family receives the same 200 mutually exclusive workload strata.

| Workload stratum | Count | Definition |
|---|---:|---|
| low balanced | 20 | low package power; activity distributed across available functional groups |
| low sparse/type-specific | 15 | low package power; one type or small active subset dominates |
| medium balanced | 35 | central power range with broad activity |
| medium type-specific | 25 | CPU-, GPU/NPU-, memory-, IO-, or analog-dominant within realistic limits |
| medium skewed | 20 | power concentration sampled over several dominant-share bands |
| high balanced dense | 20 | high power with many active chiplets and bounded per-die density |
| high single-dominant | 20 | one source carries a large but realistic fraction of total power |
| high interacting multi-source | 25 | two or more spatially chosen hot sources test near/far coupling |
| sparse active-subset stress | 10 | small active set with a low positive idle floor elsewhere |
| dense active-subset stress | 10 | most dies active at unequal levels |
| **Total** | **200** | |

Power-density bounds are enforced per chiplet type before accepting a
workload. Total package power and spatial activity pattern are sampled jointly,
not independently. Use deterministic stratified or Latin-hypercube draws
inside each stratum, then reject duplicate content hashes and near duplicates.
A low positive idle floor should be used if current source validation requires
positive powers; zero-power semantics require an explicit validator and model
compatibility decision.

The current generators do not satisfy this design. Original and extension
generation draw per-chiplet power densities and allow total power to emerge,
store `idle` and `peak` workloads but simulate `nominal`, and do not explicitly
balance sparse, dominant, or interacting-source regimes. Deterministic seeds
exist, but prior overlapping seed/index sequences caused exact duplicates.

## Family specification requirements

Every family proposal must include:

- fixed package width, height, grid convention, chiplet count, rectangles,
  types, and material/cooling configuration;
- a unique structural fingerprint based on normalized rectangle geometry,
  types, and package properties;
- minimum pairwise separation, overlap, boundary, whitespace, and case-specific
  constraints;
- allowed power-density ranges by chiplet type and workload stratum;
- explicit primary and secondary taxonomy labels;
- generation seed and accepted candidate attempt;
- nearest-family distance and a review note for potentially similar designs.

Near-duplicate rejection should compare geometry after translation/scale-aware
normalization and type-preserving assignment. A single scalar such as package
width, die count, or whitespace must not uniquely separate all test families
from training.

## Model compatibility

### Fully compatible

- Variable package width/height and physical cell sizes at fixed 64 x 64 grid.
- Current 33 raster channels describing power, masks, coordinates,
  finite-source responses, enclosed power, edge distance, instance geometry,
  and thermal crowding.
- Variable chiplet count through graph batching.
- Residual-resistance mean target when total power remains positive.
- Source-superposition reconstruction for a linear steady-state configuration.

### Compatible after retraining

- Wider package, whitespace, die-count, chiplet-size, spacing, and power ranges.
- New type compositions that map into the current grouped type masks and graph
  type encoding.
- Any v2 family-disjoint protocol, because raster, metadata, graph, source
  response, and target normalization must be fit on that protocol's train set.

### Compatible after input/schema extension

- Variable chip, interface, spreader, or sink thickness/conductivity.
- Variable convection resistance or secondary thermal path.
- New chiplet types not representable by the current grouped masks/graph
  vocabulary.
- Different grid resolution or layer-output convention.

The metadata extractor records several thermal parameters, but the active
15-dimensional conditioning vector excludes them because they are constant in
the current corpus. The source-response operator also has no material/cooling
input. Therefore v2.0 must hold material stack, ambient convention, cooling,
and HotSpot settings fixed. Varying them without schema extension would create
an unobserved causal variable.

### Incompatible without architectural or formulation work

- Nonlinear temperature-dependent material behavior that invalidates source
  superposition.
- Transient workloads under a steady-state target/model.
- Multi-layer targets without a defined channel/output extension.

## Pipeline changes required before generation

### Configuration-only

- New versioned family/workload proposal and immutable split definitions.
- External data-root and storage locations.
- Workload-stratum counts, limits, and seeds.

### Small compatibility changes

- Remove exact-ten-family validation in `load_extension_config`.
- Remove `case01-case20`, 400-workload, and extension-v1 naming assumptions.
- Parameterize dataset version, family IDs, workload counts, and output roots.
- Emit one canonical row schema and relative path contract.
- Fail on 100% stage failure and validate parent manifests before continuing.

### Substantial implementation changes

- Separate fixed family template generation from workload generation.
- Add structured workload strata and family-near-duplicate detection.
- Add content-addressed artifact manifests and dependency-lock enforcement.
- Make source-isolation generation family-level rather than workload-level.
- Build split-specific source-response and downstream normalization pipelines.
- Support external storage with scratch staging and relocation tests.

### Research decisions

- Final 50 structures and compound-OOD definitions.
- Whether zero-power inactive dies are scientifically required.
- Whether v2.1 will add material/cooling inputs.
- Number of rotational family-disjoint folds to publish.

## Publication value

A credible claim is: a fixed-template, workload-stratified benchmark that
quantifies within-family interpolation and family-disjoint generalization for
source-resolved chiplet thermal surrogates. Credibility requires released
family definitions, workload assignments, split hashes, HotSpot configuration,
generation code, validation reports, normalization provenance, and baseline
commands.

Required baselines include HotSpot runtime, source-superposition alone,
physics-v1/physics-v2 where reproducible, direct CNN, the validated residual
CNN, graph ablations, and family-disjoint results. At least one externally
recognized or independently implemented compact/surrogate baseline is needed
before a state-of-the-art claim.

Reviewers will regard the benchmark as arbitrary if family counts are selected
without coverage metrics, test families differ by one obvious scalar, power
ranges are unrealistic or unmatched, layouts are unconstrained random
rectangles, or only the best split/seed is reported.

Retain the 20-family corpus unchanged as `benchmark_v1_20family` for backward
compatibility and longitudinal reporting. Do not include it unchanged in the
primary v2 count: its case labels use changing geometry per sample, which is a
different family definition. A separate union/transfer view may evaluate
training on v1 plus v2, but must be labeled as a cross-benchmark experiment.

## Go/no-go

**GO for Stage 1 specification and family review. NO-GO for Stage 2 or full
HotSpot generation until:**

1. all 50 fixed family templates pass structural and nearest-neighbor review;
2. material/cooling variation is explicitly frozen for v2.0;
3. family-count-agnostic configuration and external-storage support are
   designed and tested;
4. canonical path, manifest, checksum, and dependency-lock contracts are
   accepted;
5. source-response leakage policy is implemented per split protocol; and
6. persistent project and scratch roots, quota, and retention owner are known.
