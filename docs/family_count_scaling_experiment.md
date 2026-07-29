# Benchmark v2 Family-Count Scaling

## Question

This controlled experiment measures zero-shot package-family accuracy as the
number of optimization families increases from 10 to 20 to 30 to 40. The only
experimental variable is the selected training-family prefix. Architecture,
source version, losses, optimizer, learning rate, scheduler, batch size, epoch
limit, early stopping, seed, normalization policy, checkpoint criterion, and
held-out protocols remain canonical.

The canonical source version is
`source_superposition_final_train40_source_v1`. Every selected family
contributes 160 optimization samples, 20 internal-validation samples, and 20
known-family test samples. Families `f007 f012 f023 f030 f041` are fixed
held-out validation families; `f008 f016 f027 f033 f044` are fixed primary test
families. Neither group participates in ordering, normalization, optimization,
or checkpoint selection.

## Diversity Ordering

The ordering is fit only on the canonical 40 training families. Descriptor
normalization uses population mean and standard deviation over those 40
families. The first family is the observed family minimizing total
standardized Euclidean distance to the pool. Each next family maximizes its
minimum distance to the selected set, with lexicographic UID tie-breaking.

The 50 active descriptors cover package dimensions, chiplet count and geometry,
placement centroid/spread, physical pairwise and boundary distances, edge and
corner occupancy, and chiplet-type counts/fractions. The exact ordered names
and fitted statistics are persisted in `experiment_definition.json`.

Material, cooling, grid, and HotSpot-context descriptors are present in the
source artifact but constant over the 40-family pool, so they are recorded as
excluded and cannot affect a standardized distance. Workload-aggregated
metadata and source-superposition map statistics are excluded from the primary
family-structural ordering. Target, residual-error, and model-performance
fields are never eligible.

The resulting order is:

```text
f002 f028 f045 f032 f049 f013 f024 f025 f039 f014
f047 f035 f043 f018 f029 f046 f011 f036 f037 f015
f022 f021 f004 f048 f031 f017 f001 f042 f005 f038
f040 f006 f010 f020 f009 f050 f034 f003 f026 f019
```

Thus:

- S10: `f002 f028 f045 f032 f049 f013 f024 f025 f039 f014`
- S20 adds: `f047 f035 f043 f018 f029 f046 f011 f036 f037 f015`
- S30 adds: `f022 f021 f004 f048 f031 f017 f001 f042 f005 f038`
- S40 adds: `f040 f006 f010 f020 f009 f050 f034 f003 f026 f019`

## Train-40 Gate

The builder copies the canonical S40 sample-split CSVs byte-for-byte and checks
family membership, sample identity, CSV hashes, per-family counts, source
version, resolved training config, input channels, metadata conditioning, seed,
mixed precision, held-out exclusion, correction signs, checkpoint criterion,
and reconstruction equation. Reuse is allowed only when every comparison in
`train40_reuse_equivalence.json` passes.

The canonical reconstruction remains:

```text
T_pred_K =
    source_superposition_base_K
    + total_power_W * delta_R_eff_pred_K_per_W
    + zero_mean_centered_field_K
```

No train-40 command is generated. A failed equivalence gate stops the scaling
launcher.

## Analysis Policy

Validation selection uses only known-family sample tests and the five fixed
held-out validation families. Primary-test loading is opt-in through
`--include-primary-test`; it is run only after the experiment definition and
validation interpretation are frozen.

The final analyzer reports micro and macro family MAE, RMSE, centered-field and
mean-correction errors, hotspot and boundary errors, source improvement,
fraction worse than source, worst-family and f044 MAE, runtime, parameter
count, completed epochs, optimizer updates, and training duration.

The canonical train-40 log contains 100 epochs totaling approximately 983
seconds on its recorded server environment. Linear sample-count projections
are approximately 246 seconds for train10, 491 seconds for train20, and 737
seconds for train30. These are planning estimates, not measured run times.
