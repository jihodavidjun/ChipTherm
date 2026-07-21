# Benchmark v2 Storage Plan

## Recommendation

Store benchmark v2 outside the repository and outside `/tmp/$USER` on a
persistent institutional project filesystem. Use institutional scratch only
for HotSpot work directories and regenerable caches. Keep proposal configs,
manifests, indexes, checksums, and documentation in Git.

`/tmp/$USER` is worker-local temporary space. It is not an authoritative data
location, is not guaranteed to survive reboot or cleanup, and on the audited
machine shares the local APFS volume rather than providing durable project
storage.

## Measured current storage

The following measurements were taken from the current workspace. Directory
sizes include filesystem and CSV overhead and are rounded.

| Artifact | Current population | Measured size | Approximate bytes/package | Scaling character |
|---|---:|---:|---:|---|
| case11-case20 raw validated tree | 4,000 | 1.1 GiB | 282 KiB | workload-dependent |
| raw source configs inside that tree | 4,000 | 26.2 MiB | 6.7 KiB | workload-dependent |
| parsed temperature arrays | 4,000 | 125.5 MiB | 32.1 KiB | workload-dependent |
| retained HotSpot outputs/logs | 4,000 | 737.7 MiB | 188.9 KiB | workload-dependent |
| extension encoded 13-channel tensors | 4,000 | 936 MiB | 240 KiB | workload-dependent |
| current original finite-source dataset | 4,010 | 1.1 GiB | 281 KiB | workload-dependent |
| current original 33-channel impedance dataset | 4,010 | 2.0 GiB | 523 KiB | workload-dependent |
| original graph tensor payload | 4,010 | 133.9 MiB | 34.2 KiB | die-count dependent, roughly O(n^2) edges |
| source-response isolated targets | 6,960 sources | 109.6 MiB | 16.1 KiB/source | source/family dependent |
| source-superposition maps and residuals, 8,010 packages | 8,010 | 263.6 MiB | 33.7 KiB | workload-dependent |
| source-response derived tree including indexes | 300 packages | 159 MiB | not package-scalable | selected diagnostic subset |
| merged source-base index views | 8,010 | 175 MiB | mostly duplicated CSV/index text | view/cache overhead |
| checkpoint files under `outputs/` | 130 checkpoints | 1.19 GB | experiment-dependent | fixed per experiment/model |
| JSON/CSV/plot/array evaluation artifacts under `outputs/` | 3,415 files | 120 MB | experiment-dependent | fixed per evaluation plus saved predictions |

The existing workspace keeps multiple historical 13-, 17-, and 33-channel
representations. Those measurements are useful for capacity planning but do
not imply that v2 should retain all representations indefinitely.

## Estimated 50-family storage

The estimate below assumes 10,000 package workloads on a 64 x 64 grid. It uses
measured extension costs where available and the current float32 33-channel
representation. Graph size is given a range because proposed die counts have
not been finalized.

| Artifact class | Estimate | Notes |
|---|---:|---|
| family templates and compact workload/source files | 0.07 GiB | layout, power, package, HotSpot, and provenance files |
| parsed HotSpot labels | 0.31 GiB | one 64 x 64 float64 array/package at current convention |
| retained HotSpot grid outputs and logs | 1.80 GiB | based on current extension output retention |
| encoded 13-channel X/Y staging | 2.3 GiB | regenerable staging representation |
| finite-source 17-channel X staging | 2.8 GiB | regenerable if 33-channel X is retained |
| canonical 33-channel float32 X | 5.1-5.5 GiB | primary model-ready raster input |
| graph tensors | 0.25-0.50 GiB | full directed graph; upper end allows denser families |
| learned source-superposition base plus residual | 0.40-0.50 GiB | two float32 64 x 64 maps/package plus metadata |
| metadata, indexes, manifests, checksums | 0.10-0.30 GiB | avoid redundant full CSV copies where possible |
| source-response family-isolation targets | 0.02-0.10 GiB | one isolated run per source/fixed family, not per workload |
| **All retained steady-state artifacts** | **about 14-16 GiB** | includes redundant 13- and 17-channel staging |
| **Canonical plus necessary model-ready artifacts** | **about 7-9 GiB** | omits redundant encoded stages and bulky raw grids after verification |
| **Recommended persistent allocation** | **20 GiB minimum** | allows manifests, failures, and revisions |
| **Recommended peak working allocation** | **20-25 GiB** | includes HotSpot workdirs and simultaneous staging |

Do not generate isolated-source labels for every workload. At a representative
20-30 sources/package, that design would create 200,000-300,000 additional
HotSpot jobs, roughly 8 GiB of parsed target arrays, and tens of GiB of raw
grid output. Under a fixed structural family, one unit/isolated response per
source and family is the scientifically appropriate linearity check and is
enough for source-response supervision.

## Proposed filesystem layout

```text
<repo>/
  configs/benchmark_v2_50family/
  docs/
  data_manifests/benchmark_v2_50family/       # lightweight only, if added later

<project_storage>/chiptherm/benchmark_v2_50family/
  canonical/
    families/                                 # immutable fixed family definitions
    workloads/                                # immutable workload assignments
    hotspot_labels/                           # parsed authoritative Y tensors
    source_isolation/                         # canonical isolated labels, if generated
    manifests/
  derived/
    encoded_13ch/
    context_17ch/
    context_33ch/
    graphs/
    source_superposition/
    metadata/
    indices/
  checkpoints/
  evaluations/

<scratch>/chiptherm/benchmark_v2_50family/
  hotspot_workdirs/
  retries/
  staging/
  caches/
```

The root URI must be supplied through a configuration or environment variable,
not embedded as an absolute path in a CSV or checkpoint. Index paths should be
repository/data-root relative, accompanied by a manifest field naming the
resolution root.

## Retention classes

| Class | Examples | Policy |
|---|---|---|
| immutable canonical | family layouts, workload tables, package/HotSpot configs, parsed labels, split assignments | persistent project storage, checksummed, never modified in place |
| expensive reproducible | retained HotSpot grids, isolated-source outputs | retain through publication; archive or drop raw grids only after parsed-label checksums and reproduction validation |
| model-ready required | 33-channel X, graphs, source-superposition maps, metadata tables | persistent while checkpoints or reported metrics depend on them |
| regenerable cache | 13-channel staging, 17-channel staging, temporary workdirs | scratch or lifecycle-managed storage after successful downstream validation |
| lightweight provenance | manifests, schemas, hashes, commands, failure reports | Git or durable manifest store |
| outputs | checkpoints and evaluation reports | project storage, immutable release snapshots for published results |

## Atomicity and integrity

Every stage should write into a staging directory, validate expected row count,
shapes, finite values, split inheritance, and hashes, and then atomically mark
the stage complete. Completion is represented by a signed or checksummed
manifest, not by directory existence. A downstream builder must verify parent
artifact IDs and hashes before reading.

Required checksums are SHA-256 for small configs, indexes, manifests, model
checkpoints, and canonical label arrays. For large trees, store a deterministic
file manifest with one hash per file plus a hash of the sorted manifest.

## Runtime capacity estimate

At the measured 4.6-5.2 seconds per legacy/extension HotSpot package, 10,000
jobs require about 13-15 wall-clock hours serial. A conservative planning range
of 5-15 seconds/job is 14-42 serial hours. Four effective workers suggest
roughly 4-11 compute hours before retry, validation, and I/O overhead. Pilot
measurements must replace this estimate before booking the full run.

## Storage decision

**Use persistent institutional project storage as the authoritative root and
institutional scratch as the build workspace.** A full run is a no-go until the
actual mount, quota, backup policy, and retention owner are recorded in the
dependency lock.
