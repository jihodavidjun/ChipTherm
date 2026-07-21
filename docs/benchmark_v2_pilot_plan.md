# Benchmark v2 Pilot Plan

## Principle

Advance only after each stage produces a complete, checksummed manifest and
passes the stated gates. Failed stages are abandoned by deleting their scratch
staging directory only; canonical parents and the current 20-family benchmark
remain untouched.

Commands below are interface proposals for a future builder. They are not
implemented or run by this audit.

## Stage 1: specification only

**Scale:** 50 family templates, zero HotSpot runs.

Expected artifacts:

- reviewed fixed `layout.json`, package, HotSpot, and material/cooling config
  per family;
- workload-stratum specification and deterministic seed plan;
- family descriptor table, structural fingerprints, and nearest-neighbor
  report;
- proposed sample/family/fold split manifests;
- storage/dependency manifests.

Validation:

- exact family count, unique UIDs, exact die count, no overlap, in-package
  geometry, valid whitespace and separation;
- type composition and power-density bounds are physically reviewed;
- no family near-duplicate below threshold without explicit justification;
- train/validation/test ranges have documented coverage;
- material/cooling and 64 x 64 conventions are identical across v2.0;
- all path roots and storage owners are known.

Go criteria: all 50 templates approved; taxonomy counts and splits balance;
nearest-neighbor review complete; no unsupported varying physical parameter.

Stop criteria: family semantics still vary geometry per workload, a held-out
partition is separable by one scalar, or storage/provenance contracts are not
approved.

Storage: under 10 MiB in the repository/manifests.

Runtime: engineering/review time only.

Proposed future command:

```bash
python3 scripts/validate_benchmark_v2_design.py \
  --proposal configs/benchmark_v2_50family/design_proposal.yaml \
  --out-dir <project_root>/canonical/manifests/stage1
```

Rollback: discard only the unaccepted proposal revision; retain review reports.

## Stage 2: five families x ten workloads

**Scale:** 50 package samples, selected to cover sparse, dense, HPC, mixed, and
package-scale regimes.

Expected artifacts:

- canonical source files and 50 HotSpot labels;
- 13-, 17-, and 33-channel test artifacts;
- graph and metadata tensors;
- a small, train-family-only isolated-source set;
- source-superposition maps and integrated inference smoke results;
- complete failure, timing, storage, and relocation reports.

Validation:

- all source constraints and workload strata pass before HotSpot launch;
- labels are finite 64 x 64 maps with expected units/orientation;
- exact channel names/order and 15/24/15 metadata/node/edge dimensions match
  the canonical model schema;
- source sum adds ambient exactly once and passes numerical tolerance;
- artifact tree relocates to a new root without path rewriting;
- deterministic rerun of selected rows reproduces source/config hashes;
- no parent from the legacy data tree is needed.

Go criteria: 50/50 labels, zero unresolved paths, zero schema failures,
numerical equivalence within established 0.05 K hard tolerance for learned
source-base recomputation, and no systematic HotSpot failures.

Stop criteria: unsupported source types, nonportable paths, schema drift,
unexplained numerical mismatch, or measured storage/runtime more than 2x plan.

Storage: approximately 0.1 GiB retained and 0.2 GiB peak.

Runtime: about 4-13 minutes serial at 5-15 seconds/HotSpot sample, plus
isolated-source and processing time.

Proposed future command:

```bash
python3 scripts/build_benchmark_v2.py \
  --config configs/benchmark_v2_50family/design_proposal.yaml \
  --stage pilot_5x10 \
  --data-root <project_root> \
  --scratch-root <scratch_root> \
  --seed 0
```

Rollback: preserve stage manifest and failure report; remove only the staging
run identified by its run UUID.

## Stage 3: ten families x fifty workloads

**Scale:** 500 package samples. Select two families from each broad regime and
include at least one validation and one test-design candidate without using
their labels for training decisions.

Expected artifacts: complete miniature canonical and model-ready benchmark,
frozen splits, source-response lineage, baseline checkpoint smokes, and
learning curves over 10/25/50 workloads.

Validation:

- all Stage 2 checks at full stage scope;
- workload strata appear in exact proportions and have no duplicate hashes;
- total power, dominant share, active count, and interaction-distance coverage
  meet proposal targets;
- generation throughput, retry rates, per-artifact bytes, graph O(n^2) cost,
  and peak scratch usage are measured;
- train-only normalization lineage is proven;
- a small model run confirms loader/training/evaluation compatibility.

Go criteria: no family has more than 1% generation failures after retry, no
duplicate leakage, all ten families pass descriptor coverage, measured
resource projection fits allocated quota/time, and workload learning curves do
not show a design defect.

Stop criteria: workload categories collapse to near duplicates, family
fingerprints are insufficiently separated, or observed source-response
generalization invalidates the intended family split.

Storage: about 0.7-0.9 GiB necessary retained, 1.0-1.3 GiB if all stages are
kept, and about 1.5 GiB peak.

Runtime: 0.7-2.1 serial HotSpot hours for package labels, plus isolated-source
work; approximately 15-45 minutes at four effective workers before retries.

Proposed future command:

```bash
python3 scripts/build_benchmark_v2.py \
  --config configs/benchmark_v2_50family/design_proposal.yaml \
  --stage pilot_10x50 \
  --data-root <project_root> \
  --scratch-root <scratch_root> \
  --seed 0 \
  --resume
```

Rollback: keep immutable accepted family/workload manifests; discard only
derived stage artifacts and resume from validated package labels if permitted
by their hashes.

## Stage 4: fifty families x two hundred workloads

**Scale:** 10,000 package samples.

Expected artifacts: versioned canonical benchmark, all declared protocol
indices, model-ready 33-channel/graph/source-base data, train-only
normalizations per protocol, benchmark card, and baseline/evaluation scripts.

Validation:

- row counts exactly 10,000 and 200 per family;
- all Stage 2 and Stage 3 checks;
- 8,000/1,000/1,000 counts for primary sample and family splits;
- checksummed parent lineage for every artifact;
- full-index finite/path/shape audit and relocation test;
- family and workload coverage reports signed off before model results;
- uncached runtime includes source-base generation and graph preparation;
- legacy v1 benchmark hashes remain unchanged.

Go criteria: 100% accepted labels or explicitly versioned exclusions before
split freeze, zero unresolved dependencies, storage below allocated 20 GiB
persistent and 25 GiB peak, and independent reproduction of a sampled build.

Stop criteria: missing labels are silently excluded, split membership changes
after seeing results, parent artifacts lack hashes, or source-response/test
family leakage is detected.

Storage: 7-9 GiB canonical plus necessary model-ready data; 14-16 GiB if all
intermediates are retained; reserve 20 GiB persistent and 20-25 GiB peak.

Runtime: planning range 14-42 serial HotSpot hours for package labels, likely
4-11 compute hours with four effective workers, plus processing and retries.

Proposed future command:

```bash
python3 scripts/build_benchmark_v2.py \
  --config configs/benchmark_v2_50family/design_proposal.yaml \
  --stage full \
  --data-root <project_root> \
  --scratch-root <scratch_root> \
  --seed 0 \
  --resume \
  --verify-parent-lock configs/benchmark_v2_50family/dependency_lock.json
```

Rollback: never mutate the release root. Build into a run UUID, retain failure
and provenance manifests, and promote atomically only after validation. A
failed full build is recoverable from immutable accepted package labels.

## Provenance required at every stage

- Git commit and dirty-worktree hash/report;
- Python/package and HotSpot executable hashes/versions;
- design, workload, split, and dependency-lock hashes;
- host, filesystem roots, timestamps, worker count, commands, and seeds;
- accepted attempt/candidate seed for each family/workload;
- parent and child artifact IDs, counts, shapes, dtypes, and checksums;
- failure/retry records and validation reports.
