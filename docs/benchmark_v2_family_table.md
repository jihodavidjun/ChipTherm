# Benchmark v2 Family Table

Stage 1 fixed structural families. No workloads or thermal labels are included.

| Family | Primary category | Split | Dies | CPU | GPU | NPU | Memory | IO | Analog | MEMS | Package width (mm) | Package height (mm) | Whitespace (%) | Minimum gap (mm) | Placement style | Secondary tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f001 | hpc | train | 12 | 2 | 2 | 0 | 6 | 2 | 0 | 0 | 40.000 | 38.000 | 48.000 | 0.524 | symmetric_compute_memory_cluster |  |
| f002 | hpc | train | 20 | 4 | 4 | 0 | 10 | 2 | 0 | 0 | 48.000 | 42.000 | 52.000 | 2.020 | dual_compute_cluster |  |
| f003 | hpc | train | 28 | 4 | 4 | 0 | 16 | 4 | 0 | 0 | 54.000 | 48.000 | 56.000 | 0.740 | compute_center_memory_ring |  |
| f004 | hpc | train | 36 | 6 | 8 | 0 | 18 | 4 | 0 | 0 | 62.000 | 58.000 | 58.000 | 1.055 | four_quadrant_clusters |  |
| f005 | hpc | train | 18 | 3 | 0 | 4 | 9 | 2 | 0 | 0 | 46.000 | 40.000 | 50.000 | 0.903 | asymmetric_accelerator_cluster |  |
| f006 | hpc | train | 30 | 4 | 5 | 5 | 12 | 4 | 0 | 0 | 58.000 | 52.000 | 60.000 | 1.902 | heterogeneous_compute_islands |  |
| f007 | hpc | val | 24 | 4 | 5 | 0 | 12 | 3 | 0 | 0 | 52.000 | 46.000 | 55.000 | 1.049 | staggered_compute_memory |  |
| f008 | hpc | test | 32 | 4 | 6 | 4 | 14 | 4 | 0 | 0 | 66.000 | 52.000 | 66.000 | 2.026 | sparse_multi_cluster | compound_ood;sparse_hpc |
| f009 | memory_heavy | train | 18 | 2 | 0 | 0 | 14 | 2 | 0 | 0 | 44.000 | 40.000 | 54.000 | 1.350 | memory_ring |  |
| f010 | memory_heavy | train | 30 | 3 | 3 | 0 | 20 | 4 | 0 | 0 | 58.000 | 52.000 | 60.000 | 0.671 | distributed_memory_banks |  |
| f011 | memory_heavy | train | 20 | 0 | 0 | 3 | 14 | 3 | 0 | 0 | 50.000 | 42.000 | 57.000 | 1.939 | asymmetric_memory_banks |  |
| f012 | memory_heavy | val | 28 | 3 | 3 | 0 | 18 | 4 | 0 | 0 | 60.000 | 44.000 | 62.000 | 0.854 | edge_memory_banks |  |
| f013 | compute_heavy | train | 10 | 4 | 4 | 0 | 0 | 2 | 0 | 0 | 36.000 | 34.000 | 42.000 | 0.521 | compact_compute_cluster |  |
| f014 | compute_heavy | train | 18 | 0 | 6 | 6 | 4 | 2 | 0 | 0 | 44.000 | 38.000 | 46.000 | 0.863 | accelerator_array |  |
| f015 | compute_heavy | train | 24 | 4 | 8 | 6 | 4 | 2 | 0 | 0 | 54.000 | 46.000 | 54.000 | 1.388 | dual_hot_cluster |  |
| f016 | compute_heavy | test | 18 | 4 | 6 | 6 | 0 | 2 | 0 | 0 | 50.000 | 40.000 | 52.000 | 1.842 | edge_separated_hot_sources | compound_ood;edge_compute |
| f017 | mixed_heterogeneous | train | 18 | 3 | 3 | 0 | 6 | 6 | 0 | 0 | 44.000 | 40.000 | 48.000 | 0.529 | functional_quadrants |  |
| f018 | mixed_heterogeneous | train | 24 | 3 | 0 | 4 | 6 | 3 | 4 | 4 | 52.000 | 46.000 | 55.000 | 0.749 | thermal_zones |  |
| f019 | mixed_heterogeneous | train | 32 | 4 | 5 | 4 | 12 | 7 | 0 | 0 | 60.000 | 54.000 | 60.000 | 1.236 | multi_cluster |  |
| f020 | mixed_heterogeneous | train | 22 | 3 | 4 | 0 | 6 | 4 | 5 | 0 | 50.000 | 44.000 | 52.000 | 0.521 | asymmetric_functional |  |
| f021 | mixed_heterogeneous | train | 38 | 5 | 5 | 5 | 11 | 5 | 4 | 3 | 68.000 | 60.000 | 64.000 | 1.721 | distributed_functional |  |
| f022 | mixed_heterogeneous | train | 16 | 3 | 0 | 0 | 6 | 4 | 3 | 0 | 40.000 | 36.000 | 45.000 | 0.945 | compact_asymmetric |  |
| f023 | mixed_heterogeneous | val | 28 | 4 | 4 | 3 | 8 | 4 | 3 | 2 | 58.000 | 50.000 | 59.000 | 1.303 | separated_functional_clusters |  |
| f024 | analog_mems | train | 12 | 2 | 0 | 0 | 0 | 3 | 3 | 4 | 40.000 | 36.000 | 52.000 | 0.736 | protected_low_power_zone |  |
| f025 | analog_mems | train | 16 | 0 | 3 | 0 | 5 | 0 | 4 | 4 | 46.000 | 40.000 | 56.000 | 1.482 | hot_cold_adjacency |  |
| f026 | analog_mems | train | 22 | 3 | 0 | 3 | 6 | 4 | 3 | 3 | 54.000 | 48.000 | 63.000 | 1.017 | distributed_sensitive_modules |  |
| f027 | analog_mems | test | 16 | 2 | 3 | 0 | 0 | 3 | 4 | 4 | 54.000 | 46.000 | 62.000 | 1.005 | edge_sensitive_hot_center | compound_ood;analog_edge |
| f028 | sparse_low_die | train | 6 | 1 | 2 | 0 | 2 | 1 | 0 | 0 | 48.000 | 42.000 | 70.000 | 4.116 | widely_separated |  |
| f029 | sparse_low_die | train | 8 | 2 | 2 | 0 | 3 | 1 | 0 | 0 | 62.000 | 48.000 | 75.000 | 4.758 | corner_and_center |  |
| f030 | sparse_low_die | val | 7 | 1 | 2 | 1 | 2 | 1 | 0 | 0 | 56.000 | 46.000 | 72.000 | 3.598 | asymmetric_far_sources |  |
| f031 | dense_high_die | train | 48 | 6 | 7 | 5 | 16 | 8 | 3 | 3 | 56.000 | 52.000 | 36.000 | 0.517 | dense_regular_channels |  |
| f032 | dense_high_die | train | 64 | 8 | 8 | 8 | 20 | 10 | 5 | 5 | 68.000 | 62.000 | 40.000 | 0.518 | dense_multi_size |  |
| f033 | dense_high_die | test | 56 | 7 | 8 | 6 | 17 | 9 | 5 | 4 | 62.000 | 56.000 | 36.000 | 0.518 | dense_edge_channels | compound_ood;dense_edge |
| f034 | compact_clustered | train | 16 | 3 | 4 | 0 | 7 | 2 | 0 | 0 | 38.000 | 34.000 | 34.000 | 0.520 | single_tight_cluster |  |
| f035 | compact_clustered | train | 24 | 4 | 5 | 4 | 9 | 2 | 0 | 0 | 46.000 | 40.000 | 38.000 | 0.519 | dual_tight_cluster |  |
| f036 | compact_clustered | train | 32 | 4 | 5 | 2 | 13 | 6 | 2 | 0 | 52.000 | 46.000 | 42.000 | 0.518 | hierarchical_clusters |  |
| f037 | distributed | train | 14 | 2 | 3 | 0 | 7 | 2 | 0 | 0 | 54.000 | 48.000 | 62.000 | 2.720 | uniform_distributed |  |
| f038 | distributed | train | 22 | 3 | 4 | 0 | 9 | 4 | 2 | 0 | 64.000 | 56.000 | 67.000 | 2.059 | perimeter_and_center |  |
| f039 | distributed | train | 28 | 4 | 4 | 4 | 10 | 6 | 0 | 0 | 70.000 | 64.000 | 70.000 | 3.027 | separated_islands |  |
| f040 | edge_constrained | train | 14 | 3 | 4 | 0 | 5 | 2 | 0 | 0 | 46.000 | 40.000 | 52.000 | 1.781 | one_hot_edge_band |  |
| f041 | edge_constrained | val | 20 | 3 | 5 | 3 | 6 | 3 | 0 | 0 | 58.000 | 48.000 | 58.000 | 2.153 | two_edge_hot_sources |  |
| f042 | package_scale_aspect | train | 14 | 3 | 3 | 0 | 6 | 2 | 0 | 0 | 34.000 | 32.000 | 45.000 | 0.604 | scale_reference_compact |  |
| f043 | package_scale_aspect | train | 24 | 4 | 4 | 0 | 12 | 4 | 0 | 0 | 72.000 | 38.000 | 58.000 | 0.893 | scale_reference_elongated |  |
| f044 | package_scale_aspect | test | 20 | 3 | 4 | 3 | 7 | 3 | 0 | 0 | 68.000 | 36.000 | 54.000 | 0.907 | long_axis_clusters | compound_ood;high_aspect_spacing |
| f045 | chiplet_size_aspect | train | 14 | 3 | 3 | 0 | 6 | 2 | 0 | 0 | 46.000 | 42.000 | 56.000 | 0.525 | elongated_sources_mixed_orientation |  |
| f046 | chiplet_size_aspect | train | 24 | 4 | 5 | 3 | 8 | 4 | 0 | 0 | 58.000 | 52.000 | 55.000 | 1.136 | multi_scale_source_sizes |  |
| f047 | whitespace | train | 18 | 3 | 4 | 0 | 7 | 4 | 0 | 0 | 50.000 | 44.000 | 35.000 | 0.524 | matched_low_whitespace | matched_pair |
| f048 | whitespace | train | 18 | 3 | 4 | 0 | 7 | 4 | 0 | 0 | 50.000 | 44.000 | 72.000 | 2.746 | matched_high_whitespace | matched_pair |
| f049 | spacing | train | 18 | 3 | 4 | 0 | 7 | 4 | 0 | 0 | 58.000 | 48.000 | 55.000 | 0.523 | matched_near_spacing | matched_pair |
| f050 | spacing | train | 18 | 3 | 4 | 0 | 7 | 4 | 0 | 0 | 58.000 | 48.000 | 55.000 | 1.308 | matched_far_spacing | matched_pair |
