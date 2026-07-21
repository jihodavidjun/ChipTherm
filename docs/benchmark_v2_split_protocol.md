# Benchmark v2 Split Protocol

## Immutable identities

Every row has three stable identities:

- `family_uid`: one fixed structural/package template;
- `workload_uid`: one accepted power/activity assignment within that family;
- `sample_uid`: `family_uid + workload_uid + benchmark version`.

The canonical manifest also stores an X/Y content hash and a structural family
fingerprint. A row may occur in exactly one split within a protocol. Structural
near-duplicate families may not straddle family-disjoint partitions.

All split manifests are generated once from accepted family/workload manifests,
versioned, checksummed, and treated as immutable. Dataset builders consume
split manifests; they do not invent their own random splits.

## Protocol 1: all-family sample split

For every one of 50 families:

- train: 160 workloads
- validation: 20 workloads
- test: 20 workloads

Totals are 8,000 train, 1,000 validation, and 1,000 test samples. Assignment is
stratified by the ten workload categories so every partition covers the
intended power/activity modes. Selection is deterministic from the protocol
seed after content deduplication.

This protocol measures workload interpolation on known structures. It must not
be presented as package-family generalization.

## Protocol 2: primary family-disjoint split

Use 40 train, 5 validation, and 5 test families. All 200 workloads from a
family inherit the family partition, yielding 8,000/1,000/1,000 samples.

Proposed family IDs are defined in `design_proposal.yaml`:

- validation: `f007`, `f012`, `f023`, `f030`, `f041`
- test: `f008`, `f016`, `f027`, `f033`, `f044`
- train: the remaining 40 families

Validation and test cover multiple primary categories. Test families are
compound configurations whose component ranges appear in training, but whose
joint geometry/topology does not. No single scalar should perfectly classify
the partition.

This is the primary package-family generalization result.

## Protocol 3: rotational family-disjoint folds

Partition the 50 families into ten predeclared groups of five using a balanced
taxonomy assignment. For fold `k`, group `k` is test, group `(k + 1) mod 10`
is validation, and the other eight groups are train. This gives 40/5/5 families
and 8,000/1,000/1,000 samples in every fold.

Publish the primary fixed split plus at least five rotational folds if compute
allows; all ten folds are preferred for a benchmark paper. Report mean,
standard deviation, median, and worst-fold metrics. Fold group assignment must
be frozen before model results are examined.

## Protocol 4: geometry OOD

Hold out families at extremes or novel combinations of:

- chiplet count and size distribution;
- whitespace and pairwise separation;
- compact versus distributed placement;
- chiplet aspect ratio and edge placement.

Power-density and total-power marginal ranges remain covered by train so the
test isolates geometry rather than confounding geometry and power. Train/test
family fingerprint distances and marginal range overlap are release artifacts.

## Protocol 5: package-scale OOD

Hold out selected small/large or high-aspect packages while retaining similar
chiplet compositions and workload strata in train. Cell size is derived from
physical package dimensions and the fixed 64 x 64 grid. Report extrapolation
distance for width, height, area, aspect ratio, and cell sizes.

## Protocol 6: package-physics OOD

**Deferred and disabled for v2.0.** Current active model inputs and the
source-response operator do not represent varying material stack or cooling
parameters. Such a split would test missing inputs rather than generalization.
A v2.1 protocol may be added only after schema and source-model support for all
varied physical quantities is implemented and validated.

## Protocol 7: compound OOD

Use five designated test families that combine covered marginal attributes in
novel ways, for example sparse plus high-aspect package or edge-constrained plus
heterogeneous high-power sources. Constituent attribute values and power ranges
must occur in train; only their combination is held out. The combinations and
coverage evidence are fixed before training.

## Normalization isolation

For every protocol and every fold, fit from its train rows only:

- 33-channel raster normalization statistics;
- source-base normalization;
- 15-dimensional metadata statistics;
- graph node and edge statistics;
- residual-resistance target statistics;
- any source-response target normalization.

Validation selects checkpoints and hyperparameters. Test is evaluated once per
locked experiment. Reusing sample-split normalization for a family-disjoint
checkpoint is prohibited even when tensor schemas match.

Checkpoints must store the protocol/fold ID, exact train manifest hash, channel
schema hash, normalization values, and source-response checkpoint artifact ID.

## Source-response leakage policy

Source-response training is part of the learned pipeline, not a label-free
physics preprocessing step. Therefore:

1. A source-response checkpoint used by a family-disjoint model may train only
   on isolated-source targets from train families.
2. Validation-family isolated targets may be used only to select a
   source-response checkpoint if the protocol explicitly treats the source
   model as part of validation. They may never update parameters.
3. Test-family isolated targets may be generated for an oracle diagnostic but
   may not train, calibrate, normalize, select, or stop any model.
4. Cached source-superposition maps inherit the source checkpoint artifact ID;
   builders must reject maps generated from a checkpoint with an incompatible
   split lineage.
5. Integrated evaluation on held-out families must generate the base from the
   permitted train-family source checkpoint.

Because each v2 family has fixed geometry and steady-state conduction is being
modeled as linear, generate at most one canonical isolated response per source
per family for supervision or oracle analysis. Repeating identical structural
isolations for 200 workloads creates duplicate labels and leakage risk.

## Leakage and quality checks

Every protocol builder must fail on:

- duplicate `sample_uid`, X/Y content hash, or workload hash across partitions;
- the same `family_uid` in more than one family-disjoint partition;
- a structural near-duplicate pair crossing family-disjoint partitions below
  the reviewed distance threshold;
- normalization parent rows outside train;
- source checkpoint lineage containing forbidden family IDs;
- mismatched workload-category counts;
- missing or non-finite tensors;
- path or artifact hash mismatch;
- test-only power ranges unless the protocol explicitly declares power OOD.

## Reporting contract

Report sample-weighted and unweighted per-family metrics, mean and centered
field errors, hotspot/chiplet metrics, runtime, family descriptors, and
confidence intervals over seeds/folds. The public result table must label the
protocol and source-response lineage, not merely `test`.
