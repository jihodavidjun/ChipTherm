# Benchmark v2 Secondary 35/5/10 Robustness Protocol

This is a secondary family-split robustness experiment over the existing
Benchmark v2 50-family dataset. It does not replace or modify the frozen
primary 40/5/5 protocol or any reported primary result.

The split was generated without model metrics using a local
`random.Random(3510)` shuffle of `f001` through `f050`. The first 35 shuffled
families are training families, the next five are held-out validation
families, and the final ten are held-out final-test families. Lists are sorted
only for display after assignment.

Raw package HotSpot labels, isolated-source simulations, 33-channel context
tensors, metadata, and graphs are reused. Split-dependent source-response and
package models, source-superposition maps, and every learned normalizer must be
rebuilt inside the protocol namespace.

The authoritative checked-in split is
`configs/benchmark_v2_family_35_5_10_seed_3510/split_manifest.json`. Runtime
indices and their hashes are written below the external data root at
`derived/protocols/benchmark_v2_family_35_5_10_seed_3510/`.

