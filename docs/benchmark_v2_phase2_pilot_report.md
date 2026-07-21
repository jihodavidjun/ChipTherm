# Benchmark v2 Phase 2 Pilot Report

Status: **PENDING SERVER EXECUTION**

This report is populated from
`canonical/manifests/pilot_5x10_validation_report.json` after the real 5 x 10
pilot, strict validation, and relocation test complete. A dry run is not a
Phase 3 release gate.

## Selected Families

| Family | Partition | Pilot role | Source-response use |
|---|---|---|---|
| f002 | train | HPC dual-cluster | training-eligible isolation |
| f023 | val | mixed heterogeneous | oracle-only isolation |
| f029 | train | sparse/long-range | training-eligible isolation |
| f032 | train | dense high-die | training-eligible isolation |
| f044 | test | compound high-aspect | oracle-only isolation |

## Workloads

The pilot contains one deterministic workload from each of the ten planned
strata for every family, for exactly 50 package samples. Geometry is fixed by
the approved Phase 1 family hashes.

## Acceptance Checklist

- [ ] 50/50 HotSpot full-package runs valid after retries
- [ ] 64 x 64 finite Kelvin targets
- [ ] 13/17/33-channel tensors have exact validated ordering
- [ ] metadata dimension is 15
- [ ] graph node/edge dimensions are 24/15
- [ ] source-response train/oracle lineage is split-safe
- [ ] 50 source-superposition maps use the declared frozen checkpoint
- [ ] all portable paths are relative to the declared data root
- [ ] all tree hashes verify after relocation
- [ ] loader smoke passes for all 50 samples
- [ ] checkpoint forward smoke passes
- [ ] no silently omitted failures

## Runtime And Storage

Populate from the machine-readable report:

- HotSpot successes/failures/retries: pending
- wall-clock runtime: pending
- bytes by artifact class: pending
- bytes/sample: pending
- peak scratch usage: pending
- projected 10,000-sample storage/runtime: pending

## Recommendation

**NO-GO until the real pilot, strict validator, and relocation validator pass.**
