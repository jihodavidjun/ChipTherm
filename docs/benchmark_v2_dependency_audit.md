# Benchmark v2 Dependency and Reproducibility Audit

## Scope and audit date

This audit was performed on 2026-07-21 at Git commit
`1f701814b619b7c2fdf93613a8dbf4c5a35e9dcc`. It traces the current 20-family
training and evaluation paths backward from the merged source-superposition
indices. It does not declare an artifact obsolete from its name alone and it
does not authorize deletion.

The current benchmark contains 8,010 clean package rows, not exactly 8,000:

- case01-case10: 4,010 deduplicated rows (401 per family)
- case11-case20: 4,000 validated rows (400 per family)

The active all-family protocol contains 6,400 train, 800 validation, and 810
test rows. The active family-disjoint protocol contains 5,600 train, 800
validation, and 800 test rows.

## Evidence inspected

The dependency conclusions in this audit come from the following exact files
and directories.

### Family definitions and raw generation

- `examples/benchmarks/case01` through `examples/benchmarks/case10`
- `configs/benchmark_extension_v1/cases.yaml`
- `configs/hotspot_base.config`
- `data/runs/benchmarks/training_set_1k/dataset_manifest.json`
- `data/runs/benchmarks/training_set_4k_extra/dataset_manifest.json`
- `data/runs/benchmarks/benchmark_extension_v1/full/all_extension_index.csv`
- `data/runs/benchmarks/benchmark_extension_v1/full/manifest.json`
- `data/runs/benchmarks/benchmark_extension_v1/full/full_audit_report.json`
- `data/runs/benchmarks/benchmark_extension_v1/full/validation_report.json`
- `data/runs/benchmarks/benchmark_extension_v1/full/hotspot_generation_report.json`
- `scripts/generate_dataset.py`
- `scripts/generate_benchmark_dataset.py`
- `scripts/build_chiptherm_extension.py`
- `src/chiptherm/benchmark_extension.py`
- `src/chiptherm/scenario.py`, `validate.py`, `writers.py`, and `runner.py`

### Encoding, cleaning, context, metadata, and graph artifacts

- `scripts/encode_dataset.py`
- `scripts/evaluate_physics_baseline.py`
- `scripts/build_combined_dataset.py`
- `scripts/build_context_dataset.py`
- `scripts/build_context_ablation_datasets.py`
- `scripts/build_clean_deduplicated_splits.py`
- `scripts/build_finite_source_feature_dataset.py`
- `scripts/build_thermal_impedance_feature_dataset.py`
- `scripts/build_metadata_features.py`
- `scripts/build_graph_features.py`
- `scripts/build_chiptherm_extension_artifacts.py`
- `scripts/build_extension_context_features.py`
- `data/runs/benchmarks/dataset_v2_clean/package_plus_power/split_manifest.json`
- `data/runs/benchmarks/dataset_v2_clean_impedance/package_plus_power/feature_manifest.json`
- `data/runs/benchmarks/dataset_v2_clean_impedance/package_plus_power/metadata_manifest.json`
- `data/runs/benchmarks/dataset_v2_clean_impedance_graph/package_plus_power/graph_manifest.json`
- `data/runs/benchmarks/benchmark_extension_v1_artifacts/path_audit_report.json`

### Source response and source superposition

- `scripts/run_superposition_diagnostic.py`
- `scripts/build_source_response_dataset.py`
- `scripts/train_source_response_model.py`
- `scripts/evaluate_source_response_model.py`
- `scripts/build_full_source_superposition_base.py`
- `scripts/build_source_superposition_extension_splits.py`
- `scripts/merge_source_superposition_20case.py`
- `src/chiptherm/ml/source_response_dataset.py`
- `src/chiptherm/ml/source_response_models.py`
- `src/chiptherm/ml/integrated_inference.py`
- `data/runs/derived/source_response_v1/source_response_manifest.json`
- `data/runs/derived/source_superposition_base_v1_full/manifest.json`
- `data/runs/derived/source_superposition_base_v1_extension/generation_manifest.json`
- `data/runs/derived/source_superposition_base_v1_extension/extension_split_manifest.json`
- `data/runs/derived/source_superposition_base_v1_20case/merge_manifest.json`
- `data/runs/derived/source_superposition_base_v1_20case/compatibility_report.md`

### Training, normalization, and evaluation

- `src/chiptherm/ml/dataset.py`
- `src/chiptherm/ml/encoder.py`
- `src/chiptherm/ml/normalization.py`
- `src/chiptherm/ml/models.py`
- `src/chiptherm/ml/graph_models.py`
- `scripts/train_residual_cnn.py`
- `scripts/evaluate_residual_cnn.py`
- `scripts/evaluate_integrated_chiptherm.py`
- `scripts/audit_spatial_errors.py`
- `outputs/source_response_operator_v1/prototype_seed1/config.json`
- `outputs/source_response_operator_v1/prototype_seed1/source_response_normalization.json`
- `outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_resistance_mean_all_family_seed1/config.json`
- `outputs/source_superposition_feature_fusion/source_superposition_cnn_feature_fusion_resistance_mean_family_seed1/config.json`
- the corresponding `normalization.json`, `delta_R_eff_normalization.json`,
  `checkpoints/best.pt`, and test `metrics.json` files

## Principal findings

### 1. The two halves of the 20-family benchmark have asymmetric provenance

case11-case20 preserve durable per-sample `layout.json`, `power.yaml`,
`package.yaml`, `hotspot.yaml`, command output, parsed temperature, and a
validated raw index. case01-case10 were deduplicated from `training_set_1k`
and `training_set_4k_extra`; the clean index is an index-only canonical view
whose targets and physics-v1 arrays still point into those older trees.

The merged 20-family compatibility report explicitly records 28,840 missing
live-integrated fields for legacy rows. The original half can be used for
cached training while those fields are blank, but it cannot be treated as a
self-contained canonical source for uncached source-superposition inference.

### 2. The local extension context tree is incomplete relative to active CSVs

The merged extension rows reference
`benchmark_extension_v1_artifacts/package_plus_power_context/...`, but that
directory is not present in this workspace. The historical compatibility
report says cached-training paths were valid when it was generated. This is
evidence of a relocation or partial-copy problem, not evidence that the index
contract is currently self-contained. Benchmark v2 must validate artifacts at
the destination after every copy and must not treat an old validation report
as proof of current availability.

### 3. Current "families" do not have fixed geometry

The original generator perturbs layout position and independently samples
chiplet power density per row. The extension generator resamples chiplet size,
target whitespace, placement, and power for every row. Therefore the current
family-disjoint split means "held-out layout-generator regime," not "held-out
fixed package structure." The proposed fixed-structure, workload-only v2 is a
scientifically cleaner benchmark, but it is a semantic major version.

### 4. Source-response provenance is smaller than the full benchmark but critical

`source_response_v1` contains 6,960 isolated-source targets from 300 selected
package samples: 100 original package samples in each inherited train,
validation, and test split. The trained source-response checkpoint uses 2,320
train sources, selects by reconstructed validation package MAE, and stores
train-only input and K/W target statistics. Reproducing the checkpoint
requires its selected package layouts/powers and isolated targets, not only
the final source-superposition maps.

### 5. Checkpoints contain data-contract assumptions

The validated residual checkpoints encode all of the following:

- exactly 33 X channels plus one source-superposition base channel
- the exact ordered channel set in `feature_manifest.json`
- train-only per-channel statistics for channels 0 and 8-32
- a 15-element metadata vector and its train statistics
- a 24-node-feature / 15-edge-feature graph schema when graph-enabled
- source-base mode `source_superposition_v1`
- residual-resistance target mean/std and total-power reconstruction semantics
- dimensional package representation

A checkpoint is not portable without its config, normalization payload,
feature manifest, metadata manifest, graph manifest, and source checkpoint
identity.

### 6. Path portability is only partially solved

Most active CSV tensor paths are repository-relative. Several manifests and
checkpoint configs retain `/Users/...` or `/nethome/...` paths, and per-map
JSON sidecars retain source checkpoint paths. These absolute fields are often
informational, but the schema does not mark them as such. v2 must distinguish
logical artifact identifiers from local physical paths and must checksum all
parent artifacts.

## Artifact classification

| Artifact class | Current paths | Classification | Why |
|---|---|---|---|
| Legacy family templates | `examples/benchmarks/case01-case10` | canonical source artifact | Defines the original structural generator inputs. |
| Legacy raw package runs | `training_set_1k`, `training_set_4k_extra` | canonical observed source/label artifact | Contains exact generated sources and HotSpot labels used by clean rows and source-response selection. |
| Extension design config | `configs/benchmark_extension_v1/cases.yaml` | canonical source artifact | Defines case11-case20 generator regimes. |
| Extension raw package runs | `benchmark_extension_v1/full` | canonical source/label artifact | Validated layouts, powers, package configs, HotSpot outputs, and labels. |
| 13-channel encoded arrays | legacy `encoded` trees and extension `encoded_package_plus_power` | generated required intermediate | Parent of context tensors and retained target arrays. |
| Physics-v1 prediction/residual arrays | `physics_baseline_global003_residuals` | generated compatibility intermediate | Needed by old residual experiments and legacy CSV schema, not by source-base-only inference. |
| `dataset_v1*` combined/context trees | `dataset_v1`, `dataset_v1_context*` | historical generated intermediate | Parent lineage of the clean deduplicated split. Not safe to call obsolete while exact clean-index reconstruction matters. |
| Clean split indices | `dataset_v2_clean/package_plus_power` | canonical protocol artifact | Defines 4,010 unique rows and leakage-free split membership. |
| Finite-source X arrays | `dataset_v2_clean_finite_source` | regenerable model-specific cache | Deterministic from clean X plus exact layout/power/package source. |
| Impedance/context X arrays | `dataset_v2_clean_impedance` and extension context roots | generated required model artifact | Required by current 33-channel checkpoints; deterministic if all parents remain. |
| Metadata table | `metadata_features.csv` and manifest | generated required model artifact | Required by conditioned checkpoints; deterministic from X and source metadata. |
| Graph tensors | `*_graph/graph_features` | generated required model artifact | Required by GNN checkpoints; deterministic from layout and power. |
| Isolated-source targets | `derived/source_response_v1/targets` | generated but required intermediate | Required to retrain or audit the source-response operator. Expensive HotSpot provenance. |
| Source-response checkpoint | `outputs/source_response_operator_v1/prototype_seed1` | model-specific derived artifact, critical | Parent of every learned source-superposition base. |
| Source-superposition maps/residuals | `derived/source_superposition_base_v1_*` | regenerable model-specific cache | Recomputable from raw source rows and the frozen source checkpoint, if those rows are intact. |
| Merged 20-family split indices | `derived/source_superposition_base_v1_20case/*split*` | canonical protocol artifact | Defines current all-family and family-disjoint experiments. |
| Residual CNN/GNN checkpoints | `outputs/source_superposition_*` | model-specific derived artifact, critical | Includes train-only normalization and architecture/data contract. |
| Metrics, plots, spatial audits | `outputs/**/test_eval`, `error_analysis`, audit roots | evaluation-only artifact | Recomputable from checkpoint plus complete test inputs; publication evidence should still be archived. |
| `source_superposition_base_v1_20case.broken` | named broken merge tree | unknown/quarantined dependency | Candidate for later cleanup only after content/checksum comparison; name alone is insufficient deletion authority. |
| Physics candidate trees | `benchmarks/physics_candidates` | evaluation/model-specific historical artifact | Not on the current source-superposition model path; retain if physics-candidate claims must remain reproducible. |

## Safe-deletion matrix

`safe_to_delete` below means safe for the current validated pipeline and full
reproducibility, not merely safe for one cached evaluation.

| artifact_or_directory | role | direct dependencies | downstream dependents | regenerable | regeneration command | estimated regeneration cost | safe_to_delete | deletion consequence | confidence |
|---|---|---|---|---|---|---|---|---|---|
| `training_set_1k` | legacy raw source and labels | examples, generator, HotSpot | clean Y, source-response selected rows, integrated inference | partially; exact HotSpot binary hash not recorded | `generate_benchmark_dataset.py` template only | about 1,000 HotSpot runs | no | breaks exact legacy labels and selected source-response provenance | high |
| `training_set_4k_extra` | legacy raw source and labels | examples, generator, HotSpot | same as above | partially | same, with historical seed 1 | about 4,000 HotSpot runs | no | breaks most original clean rows and source-response provenance | high |
| `benchmark_extension_v1/full` | validated extension source and labels | v1 config, generator, HotSpot | extension encoding, context, graphs, integrated source base | yes in principle, but expensive | `build_chiptherm_extension.py --full ... --run-hotspot` | 4,000 HotSpot runs | no | breaks extension rebuild and uncached inference | high |
| `dataset_v1_context_ablation/package_plus_power` | clean-split parent | encoded legacy arrays, context builder | `dataset_v2_clean` construction | yes if older parents remain | `build_context_ablation_datasets.py` | moderate I/O | no for provenance | prevents exact clean split re-hash/rebuild | high |
| `dataset_v2_clean/package_plus_power` | clean protocol | context parent | finite, impedance, metadata, graphs | index-only and rebuildable if parent remains | `build_clean_deduplicated_splits.py` | full 5,000-row tensor hashing | no | loses authoritative dedup and split membership | high |
| `dataset_v2_clean_finite_source` | 17-channel cache | clean X, layout/power/package | impedance X | yes | `build_finite_source_feature_dataset.py` | about 26.5 ms/package plus I/O | only after downstream rebuild proof | impedance rebuild requires regeneration | high |
| `dataset_v2_clean_impedance` | active 33-channel X | finite X, source metadata | CNN training, metadata, graph views | yes | `build_thermal_impedance_feature_dataset.py` | about 28.1 ms/package plus I/O | no | current training/evaluation cannot load X | high |
| `dataset_v2_clean_impedance_graph` | original graph view | impedance rows, layout/power | GNN training/evaluation | yes | `build_graph_features.py` | about 7.9 ms/package plus I/O | no for current GNN | graph checkpoints cannot evaluate | high |
| extension 13-channel encoded root | extension target and base encoding | raw extension | context generation | yes | `build_chiptherm_extension_artifacts.py` | moderate I/O | no until context is durable elsewhere | extension context cannot be rebuilt | high |
| extension 33-channel context root | active extension model X | 13-channel, finite/impedance builders | 20-family training/evaluation | yes | `build_extension_context_features.py` | moderate I/O | no | merged extension rows fail to load | high |
| extension graph root | active graph tensors | layouts/powers | GNN evaluation | yes | `build_graph_features.py` | low to moderate | no for graph models | GNN input unavailable | high |
| `source_response_v1/targets` | isolated-source labels | selected raw packages, HotSpot | source-response retraining | yes but expensive | `build_source_response_dataset.py` | 6,960 HotSpot runs, observed 9,510 s generation | no | source model cannot be exactly retrained | high |
| source-response `best.pt` plus normalization | learned source operator | isolated targets | all learned source bases and integrated runtime path | retrainable but not identical | `train_source_response_model.py` | 100-epoch GPU training | no | source bases and final model semantics change | high |
| source-superposition maps | cached model base | source checkpoint and raw layout/power | cached residual training/evaluation | yes | `build_full_source_superposition_base.py` | observed 24-30 ms/package on CUDA | no for cached pipeline | cached model indices fail; uncached inference can recover only with raw sources | high |
| source-superposition residual arrays | convenience target cache | Y and source base | some loaders/training | yes | base builder | low once base exists | conditional | loader can recompute `Y-base` only if schema permits | medium |
| merged split indices and manifests | experiment protocol | original/extension indices | every current 20-family experiment | yes only from exact source splits | `merge_source_superposition_20case.py` | low | no | split definitions and comparison contract are lost | high |
| checkpoint `normalization.json` and config | data/model contract | train split | evaluation and reconstruction | embedded in checkpoint but should be separate | training script | requires retraining if lost from both places | no | evaluation may silently use wrong stats | high |
| raw `grid.steady` files | HotSpot raw output | source configs and HotSpot | parsed Y provenance | yes if exact HotSpot is available | runner | full HotSpot cost | not yet | current manifests do not lock the HotSpot executable/container digest tightly enough | medium |
| evaluation plots | presentation cache | metrics or predictions | paper figures | yes | evaluator/analyzer | low to moderate | conditional | figure provenance lost if predictions/checkpoint unavailable | medium |

## Direct breakage map

- Current CNN training breaks if any active split CSV, 33-channel X tensor,
  Y tensor, source-superposition map, metadata table, checkpoint normalization
  parent, or residual target path disappears.
- Current GNN training additionally breaks if graph tensors or the graph
  manifest disappear.
- Cached evaluation has the same requirements as training except it does not
  require train labels after checkpoint creation.
- Uncached source-superposition reconstruction requires the source checkpoint,
  its normalization, exact layout/power/package rows, source-specific input
  construction code, and source enumeration order.
- Context loading requires the 33-channel X arrays and matching manifest. The
  present workspace lacks the extension context directory referenced by the
  merged rows.
- Spatial audit requires checkpoint, test index, X, Y, source base, metadata,
  and graph rectangles when chiplet-level analysis is requested.
- Extension generation requires `cases.yaml`, generator code, base HotSpot
  config, HotSpot executable, pilot approval, and smoke validation artifacts.

## Proposed dependency protection mechanism

Benchmark v2 should implement the JSON proposals in
`configs/benchmark_v2_50family/` before Stage 2 generation.

Each artifact root should contain:

1. `artifact_manifest.json` conforming to `artifact_manifest_schema.json`.
2. `dependency_lock.json` containing immutable parent artifact IDs, schema
   signatures, code commit, container/HotSpot hashes, generation command, and
   content checksums.
3. A row-level index with logical artifact paths, never host-specific absolute
   paths.
4. A relocation validation report generated after transfer.
5. A build validation command that verifies all parents and refuses to run on
   checksum or schema mismatch.

Deletion policy should be reachability-based: no artifact may be removed while
it is reachable from a released split, released checkpoint, or publication
manifest. No destructive cleanup tool is proposed in this phase.

## Unresolved dependencies

- The exact HotSpot executable/container digest used for the original 5,000
  and extension 4,000 full-package runs is not locked in a repository manifest.
- The local extension 33-channel context tree is absent although current CSVs
  reference it; its authoritative storage location needs confirmation.
- Historical manifests contain absolute paths that are not explicitly marked
  informational versus required.
- The original family templates and extension generator implement different
  notions of a family; benchmark v2 must choose and document one definition.
- There is no mounted institutional scratch/project filesystem visible from
  this workstation, so the final external storage URI cannot be selected here.

