# ChipTherm Graph Rasterizer Optimization

## Bottleneck

The legacy graph rasterizer projected node raster channels to 64x64 maps with
Python loops over graphs and over chiplet nodes. Each node computed a full
distance-to-rectangle halo field, then immediately accumulated into the output
map. Profiling showed this projection dominated the CNN-GNN forward path.

## Legacy Semantics

For each graph, grid-cell centers are placed in physical millimeters:

```text
x = (col + 0.5) / W * package_width_mm
y = (row + 0.5) / H * package_height_mm
```

For each chiplet rectangle `(x0, y0, width, height)`:

```text
x1 = x0 + width
y1 = y0 + height
dx = clamp(max(x0 - x, x - x1), min=0)
dy = clamp(max(y0 - y, y - y1), min=0)
distance = sqrt(dx^2 + dy^2 + 1e-8)
weight = exp(-distance / halo_decay_mm)
```

The `+ 1e-8` is preserved, so inside-rectangle weights are slightly below 1.0.

## Vectorized Formulation

The optimized path packs all nodes across a batch:

```text
node_values: [N, C]
node_batch:  [N]
rectangles:  [N, 4]
weights:     [N, H*W]
output:      [B, C, H, W]
```

It computes all node-to-grid weights with tensor broadcasting, then accumulates
node contributions into graph maps using `index_add_`. To avoid materializing a
large `[N, C, H, W]` tensor, accumulation loops only over small raster-channel
chunks.

## Geometry Cache

`GeometryRasterCache` stores static node-to-grid weights for fixed placement and
package geometry. This is useful when geometry is fixed but node powers or graph
embeddings change across inference calls. The cache validates graph count,
node-to-graph assignment, grid shape, and halo decay before reuse.

## Correctness

Validation command:

```bash
python3 tests/test_graph_rasterizer.py --device cpu --tolerance 1e-5
```

Observed:

```text
synthetic output max abs diff: 0
synthetic grad max abs diff: 0
checkpoint batch max abs diffs:
  final_temperature: 0
  graph_correction: 0
  centered_field: 0
  mean_rise: 0
```

Full test-set evaluation of the frozen graph checkpoint reproduced:

```text
CNN-only MAE/RMSE: 3.186 / 4.954 K
Fused MAE/RMSE:    2.929 / 4.634 K
Graph improvement: 0.257 K
```

## CPU Runtime Snapshot

Short benchmark command:

```bash
python3 scripts/benchmark_graph_rasterizer.py \
  --index data/runs/benchmarks/dataset_v2_clean_graph/package_plus_power/test_index.csv \
  --out-json /private/tmp/chiptherm_graph_rasterizer_benchmark.json \
  --batch-sizes 1 8 32 64 \
  --iterations 5 \
  --warmup 2 \
  --device cpu
```

Observed CPU means:

| Batch | Legacy ms/batch | Vectorized ms/batch | Cached ms/batch |
| ---: | ---: | ---: | ---: |
| 1 | 0.840 | 0.586 | 0.421 |
| 8 | 7.663 | 3.923 | 3.479 |
| 32 | 29.289 | 16.708 | 9.988 |
| 64 | 58.831 | 45.435 | 20.872 |

## Remaining Bottlenecks

The vectorized PyTorch path removes graph/node loops but still constructs a
dense `[N, H*W]` weight matrix each forward pass unless a geometry cache is
used. For repeated fixed-geometry inference, cached raster weights are the best
next optimization. If uncached CUDA profiling still shows high memory traffic, a
future fused CUDA/Triton kernel could combine distance, halo, and scatter-add in
one pass.
