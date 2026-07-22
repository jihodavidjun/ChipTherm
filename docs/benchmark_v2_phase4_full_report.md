# ChipTherm Benchmark v2 Phase 4

## Scope

`full_50x200` is the final canonical generation stage: 50 immutable package
families and 200 deterministic workloads per family, for 10,000 package
samples. The primary family split remains 40 train, 5 validation, and 5 test
families. The secondary sample split is defined only over the 40 train
families: 160/20/20 workload ordinals produce 6,400/800/800 rows.

This document describes the implementation and launch contract. Measured build
results belong in
`canonical/manifests/full_50x200_validation_report.json`; the benchmark is not
accepted until strict validation, relocation, and manual visual review pass.

## Frozen Workload Design

The explicit 200-cell design is a 10 by 20 physical activity matrix. The first
50 ordinals are the accepted Phase 3 cells and preserve their content hashes.
Ordinals 51-100 apply the five accepted power levels to ten added spatial
topologies. Ordinals 101-200 apply five added power levels to all twenty
topologies.

Power levels are reference, very-low, low, medium-low, moderate, medium,
medium-high, high, very-high, and stress. Topologies cover balanced,
compute/memory/peripheral/type-specific, cross-type, single-source, near/far
two-source, three-source cluster, distributed sparse, edge/corner, center,
symmetric-pair, and sparse/medium/dense asymmetric activity.

The frozen identities and load fractions are in
`configs/benchmark_v2_50family/full_50x200_workload_cells.yaml`.

## Artifact Contract

Canonical checkpoint-independent artifacts are families, workloads, HotSpot
sources and labels, 13-channel encodings, metadata, graphs, manifests, and
indices. Context tensors are deterministic regenerable physical descriptors.

The prototype source-response checkpoint and maps generated from it are marked
`provisional_source_checkpoint_dependent`. They validate the complete pipeline
but are not the final scientific source model. They may be regenerated later
from the retained graph index and a split-safe source checkpoint without
rerunning HotSpot:

```bash
python3 scripts/regenerate_benchmark_v2_source_superposition.py \
  --data-root "$CHIPTHERM_V2_DATA_ROOT" \
  --checkpoint /path/to/final_split_safe_source_checkpoint.pt \
  --source-lineage /path/to/final_split_safe_source_lineage.json \
  --artifact-name source_superposition_split_safe_v2 \
  --package-batch-size 8 \
  --source-batch-size 64 \
  --device cuda \
  --resume
```

Promotion of a replacement source-base artifact requires a new lineage manifest
and index regeneration; the existing provisional artifact is not overwritten.

## Reuse And Leakage

The accepted Phase 3 prefix contributes 500 reusable package HotSpot samples.
The full build therefore plans 9,500 new package simulations. Source isolation
is performed once per chiplet per fixed family. There are 1,137 chiplets across
the 50 families; 242 accepted Phase 3 source rows are reusable, leaving 895 new
isolated-source runs.

Only the 40 primary train families are eligible for learned source-model
training. Validation and test isolated targets are oracle-only and cannot enter
normalization, fitting, model selection, early stopping, or calibration.

## Resource Gate And Retention

The dry run uses accepted Phase 3 measurements when available and reports free
space, retained bytes, projected bytes by artifact class, source-isolation
bytes, peak staging, inode count, and runtime. A real launch fails closed unless
the configured free-space, retained-size, and staging-size gates all pass.

Must retain: canonical sources/labels, 13-channel encodings, metadata, graphs,
final indices, and manifests. Context 17/33-channel tensors and source-base maps
may be regenerated. HotSpot workdirs, builder views, and retry scratch are
staging-only. Staging cleanup is allowed only after strict and visual acceptance.

## Acceptance

The full validator checks all 10,000 identities and model-ready rows, 200-cell
coverage, immutable geometry, finite 64 by 64 labels, 13/17/33 raster schemas,
15 metadata features, 24/15 graph features, source lineage, exact split counts,
portable paths, artifact/completion lineage, retry accounting, loader coverage,
representative checkpoint forward execution, accepted-pilot immutability,
thermal distributions, and relocation.

Final recommendation states:

- `GO` only after strict validation, relocation, and visual approval.
- `GO WITH MANUAL REVIEW` when automated checks pass but visual approval is pending.
- `NO-GO` for any failed hard check or storage gate.

## Local Verification

The implementation dry run generated and validated all 10,000 workload
identities without HotSpot or CUDA work. It wrote 50 representative source
trees and used 49 MiB locally. In the absence of the external accepted Phase 3
root, the conservative fallback projected 28.222 GB decimal (26.284 GiB) of new
data and 370,466 files and correctly returned `NO-GO` because accepted Phase 3
measurements and reuse artifacts were unavailable. The authoritative server
projection will replace those fallback values and should detect 500 reusable
package labels and 242 reusable isolation rows.

## Operational Commands

On the Mac, set the existing GT SSH alias and repository path, then sync only
implementation files:

```bash
export GT_SSH=jjun49@YOUR_GT_HOST
export GT_REPO=/nethome/jjun49/chiptherm_test
rsync -av --relative \
  ./configs/benchmark_v2_50family/full_50x200.yaml \
  ./configs/benchmark_v2_50family/full_50x200_workload_cells.yaml \
  ./src/chiptherm/benchmark_v2_workloads.py \
  ./src/chiptherm/benchmark_v2_pipeline.py \
  ./scripts/build_benchmark_v2.py \
  ./scripts/generate_benchmark_v2_workloads.py \
  ./scripts/validate_benchmark_v2_full.py \
  ./scripts/validate_benchmark_v2_relocation.py \
  ./scripts/visualize_benchmark_v2_samples.py \
  ./scripts/regenerate_benchmark_v2_source_superposition.py \
  ./tests/test_benchmark_v2_phase4.py \
  ./docs/benchmark_v2_phase4_full_report.md \
  "$GT_SSH:$GT_REPO/"
```

On GT:

```bash
cd /nethome/jjun49/chiptherm_test
export CHIPTHERM_V2_DATA_ROOT=/export/hdd/$USER/chiptherm/benchmark_v2_50family
python3 -m py_compile src/chiptherm/benchmark_v2_workloads.py src/chiptherm/benchmark_v2_pipeline.py scripts/build_benchmark_v2.py scripts/generate_benchmark_v2_workloads.py scripts/validate_benchmark_v2_full.py scripts/validate_benchmark_v2_relocation.py scripts/visualize_benchmark_v2_samples.py scripts/regenerate_benchmark_v2_source_superposition.py
PYTHONPATH=src:. python3 tests/test_benchmark_v2_design.py
PYTHONPATH=src:. python3 tests/test_benchmark_v2_phase2.py
PYTHONPATH=src:. python3 tests/test_benchmark_v2_phase3.py
PYTHONPATH=src:. python3 tests/test_benchmark_v2_phase4.py

python3 scripts/build_benchmark_v2.py --stage full_50x200 --data-root "$CHIPTHERM_V2_DATA_ROOT" --scratch-root "$CHIPTHERM_V2_DATA_ROOT/staging" --seed 20260721 --workers 4 --resume --run-id full-50x200-20260721 --verify-parent-lock configs/benchmark_v2_50family/dependency_lock.json --min-free-gb 100 --min-free-fraction 0.20 --max-retained-gb 2000 --max-staging-gb 500 --dry-run
df -h "$CHIPTHERM_V2_DATA_ROOT"
python3 -m json.tool "$CHIPTHERM_V2_DATA_ROOT/canonical/manifests/full_50x200_resource_projection.json"

python3 scripts/build_benchmark_v2.py --stage full_50x200 --data-root "$CHIPTHERM_V2_DATA_ROOT" --scratch-root "$CHIPTHERM_V2_DATA_ROOT/staging" --hotspot-home "$HOTSPOT_HOME" --seed 20260721 --workers 4 --resume --run-id full-50x200-20260721 --verify-parent-lock configs/benchmark_v2_50family/dependency_lock.json --source-checkpoint outputs/source_response_operator_v1/prototype_seed1/checkpoints/best.pt --source-lineage configs/benchmark_v2_50family/source_response_lineage_prototype_seed1.json --source-device cuda --min-free-gb 100 --min-free-fraction 0.20 --max-retained-gb 2000 --max-staging-gb 500

python3 scripts/validate_benchmark_v2_full.py --data-root "$CHIPTHERM_V2_DATA_ROOT" --residual-checkpoint outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_gnn_seed1/checkpoints/best.pt
python3 scripts/visualize_benchmark_v2_samples.py --stage full_50x200 --data-root "$CHIPTHERM_V2_DATA_ROOT" --out-dir "$CHIPTHERM_V2_DATA_ROOT/reports/full_50x200/visual_audit" --rows-per-sheet 10 --device cpu
# Run only after manually reviewing every generated contact sheet:
python3 -c 'import json,os,pathlib,datetime; p=pathlib.Path(os.environ["CHIPTHERM_V2_DATA_ROOT"])/"canonical/manifests/full_50x200_visual_review.json"; p.write_text(json.dumps({"schema_version":"benchmark_v2_visual_review/1","stage":"full_50x200","approved":True,"status":"manually_approved","reviewed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"audit_manifest":"reports/full_50x200/visual_audit/visual_audit_manifest.json"},indent=2,sort_keys=True)+"\n")'
python3 scripts/validate_benchmark_v2_relocation.py --stage full_50x200 --source-root "$CHIPTHERM_V2_DATA_ROOT" --destination-root "/export/hdd/$USER/chiptherm/benchmark_v2_50family_full_relocation" --residual-checkpoint outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_gnn_seed1/checkpoints/best.pt --link-bulk-arrays
python3 scripts/validate_benchmark_v2_full.py --data-root "$CHIPTHERM_V2_DATA_ROOT" --residual-checkpoint outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_gnn_seed1/checkpoints/best.pt --require-relocation
du -sh "$CHIPTHERM_V2_DATA_ROOT" "$CHIPTHERM_V2_DATA_ROOT"/canonical "$CHIPTHERM_V2_DATA_ROOT"/derived "$CHIPTHERM_V2_DATA_ROOT"/staging
python3 -m json.tool "$CHIPTHERM_V2_DATA_ROOT/canonical/manifests/full_50x200_validation_report.json"
```

An interrupted build uses the exact same real-build command and run ID. To
schedule deterministic ranges, add one of `--start-family f001 --end-family
f010`, `f011/f020`, `f021/f030`, `f031/f040`, or `f041/f050`; no sample identity
changes.

Copy the accepted root to external Mac storage with rsync 2.6.9-compatible
flags:

```bash
export GT_SSH=jjun49@YOUR_GT_HOST
export MAC_BENCHMARK_ROOT=/Volumes/YOUR_DATA_VOLUME/chiptherm/benchmark_v2_50family
mkdir -p "$MAC_BENCHMARK_ROOT"
rsync -av --partial --progress --exclude staging/ \
  "$GT_SSH:/export/hdd/jjun49/chiptherm/benchmark_v2_50family/" \
  "$MAC_BENCHMARK_ROOT/"
```

After strict validation, relocation, and manual visual approval, staging-only
data may be removed explicitly; accepted canonical/derived data and retries are
not part of this command:

```bash
python3 -c 'import json,os,pathlib; r=pathlib.Path(os.environ["CHIPTHERM_V2_DATA_ROOT"])/"canonical/manifests/full_50x200_validation_report.json"; p=json.loads(r.read_text()); assert p["passed"] and p["relocation"]["passed"] and p["visual_review"]["approved"], "cleanup gate not satisfied"' && \
rm -rf "$CHIPTHERM_V2_DATA_ROOT/staging/runs/full-50x200-20260721/hotspot_labels" \
       "$CHIPTHERM_V2_DATA_ROOT/staging/runs/full-50x200-20260721/derived" \
       "$CHIPTHERM_V2_DATA_ROOT/staging/runs/full-50x200-20260721/dry_run_representative_sources"
```
