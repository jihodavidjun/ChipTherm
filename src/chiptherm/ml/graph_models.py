from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


NODE_FEATURE_NAMES = (
    "center_x_mm",
    "center_y_mm",
    "width_mm",
    "height_mm",
    "area_mm2",
    "aspect_ratio",
    "total_power_W",
    "power_density_W_per_mm2",
    "distance_to_left_edge_mm",
    "distance_to_right_edge_mm",
    "distance_to_bottom_edge_mm",
    "distance_to_top_edge_mm",
    "normalized_center_x",
    "normalized_center_y",
    "fraction_total_power",
    "fraction_occupied_area",
    "type_cpu",
    "type_gpu",
    "type_npu",
    "type_memory",
    "type_io",
    "type_analog",
    "type_mems",
    "type_other",
)

EDGE_FEATURE_NAMES = (
    "dx_mm",
    "dy_mm",
    "distance_mm",
    "inverse_softened_distance_per_mm",
    "log1p_distance_mm",
    "relative_angle_sin",
    "relative_angle_cos",
    "source_power_W",
    "target_power_W",
    "source_area_mm2",
    "target_area_mm2",
    "source_power_density_W_per_mm2",
    "target_power_density_W_per_mm2",
    "source_min_edge_distance_mm",
    "target_min_edge_distance_mm",
)

EPSILON = 1.0e-8


@dataclass(frozen=True)
class GraphNormalizationStats:
    schema_version: int
    node_feature_names: tuple[str, ...]
    node_means: tuple[float, ...]
    node_stds: tuple[float, ...]
    edge_feature_names: tuple[str, ...]
    edge_means: tuple[float, ...]
    edge_stds: tuple[float, ...]
    num_graphs: int
    num_nodes: int
    num_edges: int
    notes: str = "Computed from train split only. Graph topology and chiplet rectangles are not normalized."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("node_feature_names", "node_means", "node_stds", "edge_feature_names", "edge_means", "edge_stds"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNormalizationStats":
        payload = dict(data)
        for key in ("node_feature_names", "node_means", "node_stds", "edge_feature_names", "edge_means", "edge_stds"):
            payload[key] = tuple(payload.get(key, ()))
        return cls(**payload)


@dataclass(frozen=True)
class GeometryRasterCache:
    """Static node-to-grid weights for repeated inference with unchanged geometry."""

    raster_weights: torch.Tensor
    node_batch: torch.Tensor
    num_graphs: int
    height: int
    width: int
    halo_decay_mm: float
    cache_key: str

    @property
    def device(self) -> torch.device:
        return self.raster_weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.raster_weights.dtype


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.total = torch.zeros(self.dim, dtype=torch.float64)
        self.total_sq = torch.zeros(self.dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        data = values.detach().double().reshape(-1, self.dim).cpu()
        self.count += int(data.shape[0])
        self.total += data.sum(dim=0)
        self.total_sq += (data * data).sum(dim=0)

    @property
    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros(self.dim, dtype=torch.float64)
        return self.total / float(self.count)

    @property
    def std(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones(self.dim, dtype=torch.float64)
        variance = torch.clamp(self.total_sq / float(self.count) - self.mean * self.mean, min=EPSILON)
        return torch.sqrt(variance)


def compute_graph_normalization_stats(dataset: Any) -> GraphNormalizationStats:
    sample = dataset[0].get("graph")
    if sample is None:
        raise ValueError("graph normalization requested, but dataset samples do not contain graph data")
    node_dim = int(sample["node_features"].shape[1])
    edge_dim = int(sample["edge_features"].shape[1])
    node_acc = RunningMoments(node_dim)
    edge_acc = RunningMoments(edge_dim)
    num_nodes = 0
    num_edges = 0
    for index in range(len(dataset)):
        graph = dataset[index].get("graph")
        if graph is None:
            raise ValueError(f"sample {index} is missing graph data")
        node_features = graph["node_features"].float()
        edge_features = graph["edge_features"].float()
        node_acc.update(node_features)
        edge_acc.update(edge_features)
        num_nodes += int(node_features.shape[0])
        num_edges += int(edge_features.shape[0])
    return GraphNormalizationStats(
        schema_version=1,
        node_feature_names=tuple(getattr(dataset, "graph_node_feature_names", ()) or NODE_FEATURE_NAMES[:node_dim]),
        node_means=tuple(float(value) for value in node_acc.mean.tolist()),
        node_stds=tuple(float(value) for value in node_acc.std.tolist()),
        edge_feature_names=tuple(getattr(dataset, "graph_edge_feature_names", ()) or EDGE_FEATURE_NAMES[:edge_dim]),
        edge_means=tuple(float(value) for value in edge_acc.mean.tolist()),
        edge_stds=tuple(float(value) for value in edge_acc.std.tolist()),
        num_graphs=len(dataset),
        num_nodes=num_nodes,
        num_edges=num_edges,
    )


def normalize_graph_batch(graph: dict[str, torch.Tensor], stats: GraphNormalizationStats | dict[str, Any] | None) -> dict[str, torch.Tensor]:
    if stats is None:
        return graph
    if isinstance(stats, dict):
        stats = GraphNormalizationStats.from_dict(stats)
    result = dict(graph)
    node = result["node_features"].float()
    edge = result["edge_features"].float()
    node_mean = torch.tensor(stats.node_means, dtype=node.dtype, device=node.device).view(1, -1)
    node_std = torch.tensor([max(float(value), EPSILON) for value in stats.node_stds], dtype=node.dtype, device=node.device).view(1, -1)
    edge_mean = torch.tensor(stats.edge_means, dtype=edge.dtype, device=edge.device).view(1, -1)
    edge_std = torch.tensor([max(float(value), EPSILON) for value in stats.edge_stds], dtype=edge.dtype, device=edge.device).view(1, -1)
    result["node_features"] = (node - node_mean) / node_std
    result["edge_features"] = (edge - edge_mean) / edge_std
    return result


def move_graph_to_device(graph: dict[str, torch.Tensor] | None, device: torch.device) -> dict[str, torch.Tensor] | None:
    if graph is None:
        return None
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in graph.items()
    }


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("MLP layers must be positive")
        modules: list[nn.Module] = []
        current = int(input_dim)
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        modules.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChipletMessagePassingGNN(nn.Module):
    """Shared-weight chiplet interaction network for variable-sized package graphs."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 96,
        edge_hidden_dim: int = 64,
        layers: int = 4,
        aggregation: str = "sum",
        raster_channels: int = 16,
        use_edge_features: bool = True,
    ) -> None:
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'")
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.edge_hidden_dim = int(edge_hidden_dim)
        self.layers = int(layers)
        self.aggregation = aggregation
        self.raster_channels = int(raster_channels)
        self.use_edge_features = bool(use_edge_features)
        self.node_encoder = MLP(node_feature_dim, hidden_dim, hidden_dim, layers=2)
        self.edge_encoder = MLP(edge_feature_dim, edge_hidden_dim, edge_hidden_dim, layers=2)
        message_input_dim = hidden_dim * 2 + (edge_hidden_dim if self.use_edge_features else 0)
        self.message_mlps = nn.ModuleList(
            [MLP(message_input_dim, hidden_dim, hidden_dim, layers=2) for _ in range(self.layers)]
        )
        self.update_mlps = nn.ModuleList(
            [MLP(hidden_dim * 2, hidden_dim, hidden_dim, layers=2) for _ in range(self.layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.layers)])
        self.node_raster_head = nn.Linear(hidden_dim, raster_channels)
        self.global_head = MLP(hidden_dim, hidden_dim, hidden_dim, layers=2)

    def forward(self, graph: dict[str, torch.Tensor], *, return_diagnostics: bool = False) -> dict[str, torch.Tensor]:
        node_features = graph["node_features"].float()
        edge_features = graph["edge_features"].float()
        edge_index = graph["edge_index"].long()
        node_batch = graph["node_batch"].long()
        num_graphs = int(graph["num_graphs"].item()) if torch.is_tensor(graph["num_graphs"]) else int(graph["num_graphs"])
        h = self.node_encoder(node_features)
        edge_h = self.edge_encoder(edge_features)
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
        else:
            src = torch.empty(0, dtype=torch.long, device=h.device)
            dst = torch.empty(0, dtype=torch.long, device=h.device)
        for message_mlp, update_mlp, norm in zip(self.message_mlps, self.update_mlps, self.norms):
            aggregate = torch.zeros_like(h)
            if src.numel() > 0:
                parts = [h[src], h[dst]]
                if self.use_edge_features:
                    parts.append(edge_h)
                messages = message_mlp(torch.cat(parts, dim=1))
                aggregate.index_add_(0, dst, messages)
                if self.aggregation == "mean":
                    degree = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
                    degree.index_add_(0, dst, torch.ones(dst.shape[0], dtype=h.dtype, device=h.device))
                    aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
            update = update_mlp(torch.cat([h, aggregate], dim=1))
            h = norm(h + update)
        graph_embedding = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
        graph_embedding.index_add_(0, node_batch, h)
        graph_counts = torch.zeros(num_graphs, dtype=h.dtype, device=h.device)
        graph_counts.index_add_(0, node_batch, torch.ones(h.shape[0], dtype=h.dtype, device=h.device))
        graph_embedding = graph_embedding / graph_counts.clamp_min(1.0).unsqueeze(1)
        result = {
            "node_embeddings": h,
            "node_raster_values": self.node_raster_head(h),
            "graph_embedding": self.global_head(graph_embedding),
        }
        if return_diagnostics:
            result["encoded_edges"] = edge_h
        return result

    def forward_profile(self, graph: dict[str, torch.Tensor], synchronize: Any | None = None) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        timings: dict[str, float] = {}

        def tic() -> float:
            if synchronize is not None:
                synchronize()
            return time.perf_counter()

        def toc(name: str, start: float) -> None:
            if synchronize is not None:
                synchronize()
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - start

        node_features = graph["node_features"].float()
        edge_features = graph["edge_features"].float()
        edge_index = graph["edge_index"].long()
        node_batch = graph["node_batch"].long()
        num_graphs = int(graph["num_graphs"].item()) if torch.is_tensor(graph["num_graphs"]) else int(graph["num_graphs"])
        start = tic()
        h = self.node_encoder(node_features)
        toc("node_edge_encoding_s", start)
        start = tic()
        edge_h = self.edge_encoder(edge_features)
        toc("node_edge_encoding_s", start)
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
        else:
            src = torch.empty(0, dtype=torch.long, device=h.device)
            dst = torch.empty(0, dtype=torch.long, device=h.device)
        start = tic()
        for message_mlp, update_mlp, norm in zip(self.message_mlps, self.update_mlps, self.norms):
            aggregate = torch.zeros_like(h)
            if src.numel() > 0:
                parts = [h[src], h[dst]]
                if self.use_edge_features:
                    parts.append(edge_h)
                messages = message_mlp(torch.cat(parts, dim=1))
                aggregate.index_add_(0, dst, messages)
                if self.aggregation == "mean":
                    degree = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
                    degree.index_add_(0, dst, torch.ones(dst.shape[0], dtype=h.dtype, device=h.device))
                    aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
            update = update_mlp(torch.cat([h, aggregate], dim=1))
            h = norm(h + update)
        toc("message_passing_s", start)
        start = tic()
        graph_embedding = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
        graph_embedding.index_add_(0, node_batch, h)
        graph_counts = torch.zeros(num_graphs, dtype=h.dtype, device=h.device)
        graph_counts.index_add_(0, node_batch, torch.ones(h.shape[0], dtype=h.dtype, device=h.device))
        graph_embedding = graph_embedding / graph_counts.clamp_min(1.0).unsqueeze(1)
        graph_embedding = self.global_head(graph_embedding)
        node_raster_values = self.node_raster_head(h)
        toc("graph_pooling_s", start)
        return {
            "node_embeddings": h,
            "node_raster_values": node_raster_values,
            "graph_embedding": graph_embedding,
            "encoded_edges": edge_h,
        }, timings


def rasterize_node_values_legacy(
    node_values: torch.Tensor,
    graph: dict[str, torch.Tensor],
    *,
    height: int = 64,
    width: int = 64,
    halo_decay_mm: float = 4.0,
) -> torch.Tensor:
    """Project node channels to dense maps using chiplet rectangles plus an exponential halo."""
    rects = graph["chiplet_rects"].float()
    package_size = graph["package_size"].float()
    node_batch = graph["node_batch"].long()
    num_graphs = int(graph["num_graphs"].item()) if torch.is_tensor(graph["num_graphs"]) else int(graph["num_graphs"])
    channels = int(node_values.shape[1])
    maps = node_values.new_zeros((num_graphs, channels, height, width))
    rows = torch.arange(height, dtype=node_values.dtype, device=node_values.device) + 0.5
    cols = torch.arange(width, dtype=node_values.dtype, device=node_values.device) + 0.5
    yy_unit, xx_unit = torch.meshgrid(rows / float(height), cols / float(width), indexing="ij")
    decay = max(float(halo_decay_mm), EPSILON)
    for graph_index in range(num_graphs):
        node_indices = torch.nonzero(node_batch == graph_index, as_tuple=False).reshape(-1)
        if node_indices.numel() == 0:
            continue
        pkg_w, pkg_h = package_size[graph_index, 0], package_size[graph_index, 1]
        xx = xx_unit * pkg_w
        yy = yy_unit * pkg_h
        for node_index in node_indices.tolist():
            x0, y0, rect_w, rect_h = rects[node_index]
            x1 = x0 + rect_w
            y1 = y0 + rect_h
            dx = torch.clamp(torch.maximum(x0 - xx, xx - x1), min=0.0)
            dy = torch.clamp(torch.maximum(y0 - yy, yy - y1), min=0.0)
            distance = torch.sqrt(dx * dx + dy * dy + EPSILON)
            weight = torch.exp(-distance / decay)
            maps[graph_index] += node_values[node_index, :, None, None] * weight[None, :, :]
    return maps


def rasterize_node_values_vectorized(
    node_values: torch.Tensor,
    graph: dict[str, torch.Tensor],
    *,
    height: int = 64,
    width: int = 64,
    halo_decay_mm: float = 4.0,
    channel_chunk_size: int = 4,
    cache: GeometryRasterCache | None = None,
) -> torch.Tensor:
    """Vectorized equivalent of :func:`rasterize_node_values_legacy`.

    Shapes:
      node_values: [N, C]
      raster_weights: [N, H*W]
      output: [B, C, H, W]

    The implementation intentionally preserves the legacy distance formula,
    including the small ``+ EPSILON`` inside sqrt, so in-rectangle weights are
    very slightly below one rather than exactly one.
    """
    if node_values.ndim != 2:
        raise ValueError(f"node_values must have shape [N, C], got {tuple(node_values.shape)}")
    node_batch = graph["node_batch"].long()
    package_size = graph["package_size"].float()
    num_graphs = int(package_size.shape[0])
    channels = int(node_values.shape[1])
    pixels = int(height) * int(width)
    if cache is not None:
        validate_raster_cache(cache, graph, height=height, width=width, halo_decay_mm=halo_decay_mm)
        weights = cache.raster_weights.to(device=node_values.device, dtype=node_values.dtype)
        node_batch = cache.node_batch.to(device=node_values.device)
        num_graphs = int(cache.num_graphs)
    else:
        weights = compute_node_raster_weights(graph, height=height, width=width, halo_decay_mm=halo_decay_mm, dtype=node_values.dtype)
    maps_flat = node_values.new_zeros((num_graphs, channels, pixels))
    chunk = max(int(channel_chunk_size), 1)
    for start in range(0, channels, chunk):
        end = min(start + chunk, channels)
        contribution = weights[:, None, :] * node_values[:, start:end, None]
        maps_flat[:, start:end, :].index_add_(0, node_batch, contribution)
    return maps_flat.view(num_graphs, channels, height, width)


def compute_node_raster_weights(
    graph: dict[str, torch.Tensor],
    *,
    height: int = 64,
    width: int = 64,
    halo_decay_mm: float = 4.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    rects = graph["chiplet_rects"].float()
    package_size = graph["package_size"].float()
    node_batch = graph["node_batch"].long()
    device = rects.device
    dtype = dtype or rects.dtype
    rects = rects.to(dtype=dtype)
    package_size = package_size.to(dtype=dtype)
    node_package = package_size.index_select(0, node_batch)
    rows = torch.arange(height, dtype=dtype, device=device) + 0.5
    cols = torch.arange(width, dtype=dtype, device=device) + 0.5
    yy_unit, xx_unit = torch.meshgrid(rows / float(height), cols / float(width), indexing="ij")
    xx_unit_flat = xx_unit.reshape(1, -1)
    yy_unit_flat = yy_unit.reshape(1, -1)
    xx = xx_unit_flat * node_package[:, 0:1]
    yy = yy_unit_flat * node_package[:, 1:2]
    x0 = rects[:, 0:1]
    y0 = rects[:, 1:2]
    x1 = x0 + rects[:, 2:3]
    y1 = y0 + rects[:, 3:4]
    dx = torch.clamp(torch.maximum(x0 - xx, xx - x1), min=0.0)
    dy = torch.clamp(torch.maximum(y0 - yy, yy - y1), min=0.0)
    distance = torch.sqrt(dx * dx + dy * dy + EPSILON)
    return torch.exp(-distance / max(float(halo_decay_mm), EPSILON))


def build_geometry_raster_cache(
    graph: dict[str, torch.Tensor],
    *,
    height: int = 64,
    width: int = 64,
    halo_decay_mm: float = 4.0,
    dtype: torch.dtype = torch.float32,
    cache_key: str | None = None,
) -> GeometryRasterCache:
    weights = compute_node_raster_weights(graph, height=height, width=width, halo_decay_mm=halo_decay_mm, dtype=dtype)
    return GeometryRasterCache(
        raster_weights=weights,
        node_batch=graph["node_batch"].long().clone(),
        num_graphs=int(graph["package_size"].shape[0]),
        height=int(height),
        width=int(width),
        halo_decay_mm=float(halo_decay_mm),
        cache_key=cache_key or geometry_cache_key(graph, height=height, width=width, halo_decay_mm=halo_decay_mm),
    )


def validate_raster_cache(
    cache: GeometryRasterCache,
    graph: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    halo_decay_mm: float,
) -> None:
    if cache.height != int(height) or cache.width != int(width):
        raise ValueError("raster cache grid shape does not match requested rasterization")
    if abs(cache.halo_decay_mm - float(halo_decay_mm)) > 1.0e-12:
        raise ValueError("raster cache halo_decay_mm does not match requested rasterization")
    if cache.num_graphs != int(graph["package_size"].shape[0]):
        raise ValueError("raster cache graph count does not match graph batch")
    if cache.node_batch.shape != graph["node_batch"].shape:
        raise ValueError("raster cache node_batch shape does not match graph batch")
    if not torch.equal(cache.node_batch.cpu(), graph["node_batch"].long().cpu()):
        raise ValueError("raster cache node-to-graph assignment does not match graph batch")


def geometry_cache_key(
    graph: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    halo_decay_mm: float,
) -> str:
    rects = graph["chiplet_rects"].detach().cpu().contiguous()
    package_size = graph["package_size"].detach().cpu().contiguous()
    node_batch = graph["node_batch"].detach().cpu().contiguous()
    values = (
        tuple(rects.reshape(-1).tolist()),
        tuple(package_size.reshape(-1).tolist()),
        tuple(int(value) for value in node_batch.reshape(-1).tolist()),
        int(height),
        int(width),
        float(halo_decay_mm),
    )
    return str(hash(values))


def rasterize_node_values(
    node_values: torch.Tensor,
    graph: dict[str, torch.Tensor],
    *,
    height: int = 64,
    width: int = 64,
    halo_decay_mm: float = 4.0,
    mode: str = "vectorized",
    channel_chunk_size: int = 4,
    cache: GeometryRasterCache | None = None,
) -> torch.Tensor:
    if mode == "legacy":
        return rasterize_node_values_legacy(node_values, graph, height=height, width=width, halo_decay_mm=halo_decay_mm)
    if mode == "vectorized":
        return rasterize_node_values_vectorized(
            node_values,
            graph,
            height=height,
            width=width,
            halo_decay_mm=halo_decay_mm,
            channel_chunk_size=channel_chunk_size,
            cache=cache,
        )
    raise ValueError(f"unsupported rasterizer mode: {mode}")


def zero_initialize_last_linear(module: nn.Module) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linear_layers:
        return
    final = linear_layers[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


def make_silu_mlp(input_dim: int, hidden_dim: int, output_dim: int, *, layers: int, zero_last: bool = False) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be positive")
    modules: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(max(int(layers) - 1, 0)):
        modules.extend([nn.Linear(current, int(hidden_dim)), nn.SiLU()])
        current = int(hidden_dim)
    modules.append(nn.Linear(current, int(output_dim)))
    net = nn.Sequential(*modules)
    if zero_last:
        zero_initialize_last_linear(net)
    return net


class PairwiseThermalImpedanceOperator(nn.Module):
    """Explicit source-target chiplet correction operator.

    The operator consumes the normalized graph batch already used by the generic
    GNN. For every directed edge i -> j it predicts a signed scalar transfer
    coefficient K_ij, multiplies it by the normalized source power feature, and
    aggregates contributions at the target node. A separate zero-initialized
    self term handles local chiplet correction.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        metadata_dim: int = 0,
        hidden_dim: int = 96,
        layers: int = 3,
        source_power_feature_index: int = 6,
    ) -> None:
        super().__init__()
        if source_power_feature_index < 0 or source_power_feature_index >= node_feature_dim:
            raise ValueError("source_power_feature_index is outside the node feature dimension")
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.metadata_dim = int(metadata_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.source_power_feature_index = int(source_power_feature_index)
        pair_input_dim = self.node_feature_dim * 2 + self.edge_feature_dim + self.metadata_dim
        self.pairwise_mlp = make_silu_mlp(pair_input_dim, hidden_dim, 1, layers=layers, zero_last=True)
        self.self_mlp = make_silu_mlp(self.node_feature_dim + self.metadata_dim, hidden_dim, 1, layers=layers, zero_last=True)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        metadata: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        node_features = graph["node_features"].float()
        edge_features = graph["edge_features"].float()
        edge_index = graph["edge_index"].long()
        node_batch = graph["node_batch"].long()
        node_count = int(node_features.shape[0])
        if metadata is None:
            metadata_node = node_features.new_zeros((node_count, 0))
        else:
            metadata = metadata.float()
            metadata_node = metadata.index_select(0, node_batch)

        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            metadata_edge = metadata_node.index_select(0, src)
            pair_input = torch.cat(
                [
                    node_features.index_select(0, src),
                    node_features.index_select(0, dst),
                    edge_features,
                    metadata_edge,
                ],
                dim=1,
            )
            k_values = self.pairwise_mlp(pair_input).squeeze(1)
            source_power = node_features.index_select(0, src)[:, self.source_power_feature_index]
            pairwise_contributions = source_power * k_values
            pairwise_node_sums = node_features.new_zeros(node_count)
            pairwise_node_sums.index_add_(0, dst, pairwise_contributions)
        else:
            k_values = node_features.new_empty((0,))
            pairwise_contributions = node_features.new_empty((0,))
            pairwise_node_sums = node_features.new_zeros(node_count)

        self_input = torch.cat([node_features, metadata_node], dim=1)
        self_corrections = self.self_mlp(self_input).squeeze(1)
        node_corrections = self_corrections + pairwise_node_sums
        result = {
            "node_corrections": node_corrections,
            "self_corrections": self_corrections,
            "pairwise_node_sums": pairwise_node_sums,
            "k_values": k_values,
            "pairwise_contributions": pairwise_contributions,
        }
        if return_diagnostics:
            result["source_power"] = (
                node_features.index_select(0, edge_index[0])[:, self.source_power_feature_index]
                if edge_index.numel() > 0
                else node_features.new_empty((0,))
            )
        return result


PAIRWISE_BASIS_NAMES = (
    "uniform_halo",
    "target_local_u",
    "target_local_v",
    "source_axis_s",
    "source_facing_halfspace",
    "source_opposing_halfspace",
    "target_center_radial_decay",
    "boundary_modulated_halo",
)


class PairwiseBasisThermalOperator(nn.Module):
    """Explicit low-rank source-target spatial correction operator.

    Each directed source-target chiplet edge predicts R coefficients. The
    coefficients weight deterministic, interpretable target-local basis maps
    that depend on source direction, target geometry, target edge distance, and
    package size. No self term or message passing is used in this first version.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        metadata_dim: int = 0,
        basis_rank: int = 8,
        hidden_dim: int = 96,
        layers: int = 3,
        source_power_feature_index: int = 6,
        edge_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if basis_rank < 1 or basis_rank > len(PAIRWISE_BASIS_NAMES):
            raise ValueError(f"basis_rank must be in [1, {len(PAIRWISE_BASIS_NAMES)}]")
        if source_power_feature_index < 0 or source_power_feature_index >= node_feature_dim:
            raise ValueError("source_power_feature_index is outside the node feature dimension")
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.metadata_dim = int(metadata_dim)
        self.basis_rank = int(basis_rank)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.source_power_feature_index = int(source_power_feature_index)
        self.edge_chunk_size = int(edge_chunk_size)
        pair_input_dim = self.node_feature_dim * 2 + self.edge_feature_dim + self.metadata_dim
        self.coefficient_mlp = make_silu_mlp(pair_input_dim, hidden_dim, basis_rank, layers=layers, zero_last=True)

    @property
    def basis_names(self) -> tuple[str, ...]:
        return PAIRWISE_BASIS_NAMES[: self.basis_rank]

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        metadata: torch.Tensor | None = None,
        *,
        height: int = 64,
        width: int = 64,
        halo_decay_mm: float = 4.0,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        node_features = graph["node_features"].float()
        edge_features = graph["edge_features"].float()
        edge_index = graph["edge_index"].long()
        node_batch = graph["node_batch"].long()
        num_graphs = int(graph["num_graphs"].item()) if torch.is_tensor(graph["num_graphs"]) else int(graph["num_graphs"])
        edge_count = int(edge_features.shape[0])
        if metadata is None:
            metadata_node = node_features.new_zeros((node_features.shape[0], 0))
        else:
            metadata_node = metadata.float().index_select(0, node_batch)
        if edge_count > 0:
            src = edge_index[0]
            dst = edge_index[1]
            pair_input = torch.cat(
                [
                    node_features.index_select(0, src),
                    node_features.index_select(0, dst),
                    edge_features,
                    metadata_node.index_select(0, src),
                ],
                dim=1,
            )
            coeff = self.coefficient_mlp(pair_input)
            source_power = node_features.index_select(0, src)[:, self.source_power_feature_index]
            weighted_coeff = coeff * source_power[:, None]
        else:
            coeff = node_features.new_empty((0, self.basis_rank))
            weighted_coeff = node_features.new_empty((0, self.basis_rank))
        field = pairwise_basis_superpose(
            weighted_coeff,
            graph,
            edge_index,
            basis_rank=self.basis_rank,
            height=height,
            width=width,
            halo_decay_mm=halo_decay_mm,
            edge_chunk_size=self.edge_chunk_size,
        )
        result = {
            "field": field.view(num_graphs, height, width),
            "coefficients": coeff,
            "weighted_coefficients": weighted_coeff,
        }
        if return_diagnostics:
            result["basis_names"] = self.basis_names
        return result


def pairwise_basis_superpose(
    weighted_coefficients: torch.Tensor,
    graph: dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    *,
    basis_rank: int,
    height: int = 64,
    width: int = 64,
    halo_decay_mm: float = 4.0,
    edge_chunk_size: int = 512,
) -> torch.Tensor:
    """Sum weighted pairwise basis responses into graph-specific fields.

    This function is vectorized within edge chunks. It intentionally avoids
    Python loops over individual graphs, nodes, or edges, while keeping peak
    memory bounded by chunking the [E, R, H*W] basis tensor.
    """

    package_size = graph["package_size"].float()
    node_batch = graph["node_batch"].long()
    num_graphs = int(package_size.shape[0])
    pixels = int(height) * int(width)
    output = weighted_coefficients.new_zeros((num_graphs, pixels))
    if edge_index.numel() == 0:
        return output
    src_all = edge_index[0].long()
    dst_all = edge_index[1].long()
    rows = torch.arange(height, dtype=weighted_coefficients.dtype, device=weighted_coefficients.device) + 0.5
    cols = torch.arange(width, dtype=weighted_coefficients.dtype, device=weighted_coefficients.device) + 0.5
    yy_unit, xx_unit = torch.meshgrid(rows / float(height), cols / float(width), indexing="ij")
    xx_unit_flat = xx_unit.reshape(1, -1)
    yy_unit_flat = yy_unit.reshape(1, -1)
    chunk_size = max(int(edge_chunk_size), 1)
    for start in range(0, int(weighted_coefficients.shape[0]), chunk_size):
        end = min(start + chunk_size, int(weighted_coefficients.shape[0]))
        src = src_all[start:end]
        dst = dst_all[start:end]
        graph_index = node_batch.index_select(0, dst)
        basis = pairwise_basis_chunk(
            graph,
            src,
            dst,
            graph_index,
            xx_unit_flat,
            yy_unit_flat,
            basis_rank=basis_rank,
            halo_decay_mm=halo_decay_mm,
        )
        contribution = (weighted_coefficients[start:end, :, None] * basis).sum(dim=1)
        output.index_add_(0, graph_index, contribution)
    return output


def pairwise_basis_chunk(
    graph: dict[str, torch.Tensor],
    src: torch.Tensor,
    dst: torch.Tensor,
    graph_index: torch.Tensor,
    xx_unit_flat: torch.Tensor,
    yy_unit_flat: torch.Tensor,
    *,
    basis_rank: int,
    halo_decay_mm: float,
) -> torch.Tensor:
    rects = graph["chiplet_rects"].float().to(dtype=xx_unit_flat.dtype, device=xx_unit_flat.device)
    package_size = graph["package_size"].float().to(dtype=xx_unit_flat.dtype, device=xx_unit_flat.device)
    src_rect = rects.index_select(0, src)
    dst_rect = rects.index_select(0, dst)
    pkg = package_size.index_select(0, graph_index)
    xx = xx_unit_flat * pkg[:, 0:1]
    yy = yy_unit_flat * pkg[:, 1:2]
    x0 = dst_rect[:, 0:1]
    y0 = dst_rect[:, 1:2]
    w = dst_rect[:, 2:3].clamp_min(EPSILON)
    h = dst_rect[:, 3:4].clamp_min(EPSILON)
    x1 = x0 + w
    y1 = y0 + h
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h
    src_cx = src_rect[:, 0:1] + 0.5 * src_rect[:, 2:3]
    src_cy = src_rect[:, 1:2] + 0.5 * src_rect[:, 3:4]
    direction_x = src_cx - cx
    direction_y = src_cy - cy
    direction_norm = torch.sqrt(direction_x * direction_x + direction_y * direction_y + EPSILON)
    direction_x = direction_x / direction_norm
    direction_y = direction_y / direction_norm
    dx_rect = torch.clamp(torch.maximum(x0 - xx, xx - x1), min=0.0)
    dy_rect = torch.clamp(torch.maximum(y0 - yy, yy - y1), min=0.0)
    distance_to_rect = torch.sqrt(dx_rect * dx_rect + dy_rect * dy_rect + EPSILON)
    halo = torch.exp(-distance_to_rect / max(float(halo_decay_mm), EPSILON))
    u = torch.clamp((xx - cx) / (0.5 * w), min=-2.0, max=2.0)
    v = torch.clamp((yy - cy) / (0.5 * h), min=-2.0, max=2.0)
    source_axis = torch.clamp(u * direction_x + v * direction_y, min=-2.0, max=2.0)
    radial = torch.sqrt(u * u + v * v + EPSILON)
    radial_decay = torch.exp(-0.5 * radial * radial)
    min_edge = torch.minimum(
        torch.minimum(x0, pkg[:, 0:1] - x1),
        torch.minimum(y0, pkg[:, 1:2] - y1),
    ).clamp_min(0.0)
    edge_scale = (0.25 * torch.minimum(pkg[:, 0:1], pkg[:, 1:2])).clamp_min(EPSILON)
    boundary_factor = 1.0 - torch.clamp(min_edge / edge_scale, min=0.0, max=1.0)
    terms = [
        halo,
        u * halo,
        v * halo,
        source_axis * halo,
        torch.relu(source_axis) * halo,
        torch.relu(-source_axis) * halo,
        radial_decay * halo,
        boundary_factor * halo,
    ]
    return torch.stack(terms[:basis_rank], dim=1)


def chiplet_cell_weights(
    graph: dict[str, torch.Tensor],
    *,
    height: int = 64,
    width: int = 64,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact rectangle cell-center masks as float weights.

    Cell centers use the same package-coordinate convention as the graph
    rasterizer: row/col centers are at (index + 0.5) / grid_size times the
    package height/width. A cell belongs to a chiplet when its center lies
    inside the closed chiplet rectangle. If a tiny chiplet would otherwise have
    no cells, the nearest cell to the chiplet center is assigned to it.
    """

    rects = graph["chiplet_rects"].float()
    package_size = graph["package_size"].float()
    node_batch = graph["node_batch"].long()
    device = rects.device
    dtype = dtype or rects.dtype
    rects = rects.to(dtype=dtype)
    package_size = package_size.to(dtype=dtype)
    node_package = package_size.index_select(0, node_batch)
    rows = torch.arange(height, dtype=dtype, device=device) + 0.5
    cols = torch.arange(width, dtype=dtype, device=device) + 0.5
    yy_unit, xx_unit = torch.meshgrid(rows / float(height), cols / float(width), indexing="ij")
    xx = xx_unit.reshape(1, -1) * node_package[:, 0:1]
    yy = yy_unit.reshape(1, -1) * node_package[:, 1:2]
    x0 = rects[:, 0:1]
    y0 = rects[:, 1:2]
    x1 = x0 + rects[:, 2:3]
    y1 = y0 + rects[:, 3:4]
    inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    weights = inside.to(dtype=dtype)
    counts = weights.sum(dim=1)
    empty = counts <= 0.0
    if bool(empty.any()):
        centers_x = x0[empty] + 0.5 * rects[empty, 2:3]
        centers_y = y0[empty] + 0.5 * rects[empty, 3:4]
        node_package_empty = node_package[empty]
        col_index = torch.clamp((centers_x / node_package_empty[:, 0:1] * float(width)).floor().long(), min=0, max=width - 1)
        row_index = torch.clamp((centers_y / node_package_empty[:, 1:2] * float(height)).floor().long(), min=0, max=height - 1)
        flat_index = (row_index * width + col_index).reshape(-1)
        empty_indices = torch.nonzero(empty, as_tuple=False).reshape(-1)
        weights[empty_indices] = 0.0
        weights[empty_indices, flat_index] = 1.0
        counts = weights.sum(dim=1)
    return weights.view(-1, height, width), counts.clamp_min(1.0)


def chiplet_mean_temperatures(field: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError(f"field must have shape [B, H, W], got {tuple(field.shape)}")
    weights, counts = chiplet_cell_weights(graph, height=int(field.shape[-2]), width=int(field.shape[-1]), dtype=field.dtype)
    node_batch = graph["node_batch"].long()
    node_fields = field.index_select(0, node_batch)
    return (node_fields * weights).sum(dim=(-2, -1)) / counts.to(field.dtype)


def chiplet_peak_temperatures(field: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError(f"field must have shape [B, H, W], got {tuple(field.shape)}")
    weights, _counts = chiplet_cell_weights(graph, height=int(field.shape[-2]), width=int(field.shape[-1]), dtype=field.dtype)
    node_batch = graph["node_batch"].long()
    node_fields = field.index_select(0, node_batch)
    masked = node_fields.masked_fill(weights <= 0.0, -torch.inf)
    peaks = masked.amax(dim=(-2, -1))
    return torch.where(torch.isfinite(peaks), peaks, node_fields.reshape(node_fields.shape[0], -1).amax(dim=1))


def chiplet_mean_loss(pred: torch.Tensor, target: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
    pred_mean = chiplet_mean_temperatures(pred, graph)
    target_mean = chiplet_mean_temperatures(target, graph)
    return F.smooth_l1_loss(pred_mean, target_mean)


def inter_chiplet_delta_mae(pred_means: torch.Tensor, target_means: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    for graph_index in range(num_graphs):
        indices = torch.nonzero(node_batch == graph_index, as_tuple=False).reshape(-1)
        if int(indices.numel()) < 2:
            continue
        pred = pred_means.index_select(0, indices)
        target = target_means.index_select(0, indices)
        pair_indices = torch.triu_indices(int(indices.numel()), int(indices.numel()), offset=1, device=pred.device)
        pred_delta = pred[pair_indices[0]] - pred[pair_indices[1]]
        target_delta = target[pair_indices[0]] - target[pair_indices[1]]
        values.append((pred_delta - target_delta).abs())
    if not values:
        return pred_means.new_tensor(0.0)
    return torch.cat(values).mean()


def chiplet_metric_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    graph: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    pred_mean = chiplet_mean_temperatures(pred, graph)
    target_mean = chiplet_mean_temperatures(target, graph)
    pred_peak = chiplet_peak_temperatures(pred, graph)
    target_peak = chiplet_peak_temperatures(target, graph)
    node_batch = graph["node_batch"].long()
    return {
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "pred_peak": pred_peak,
        "target_peak": target_peak,
        "mean_abs_error": (pred_mean - target_mean).abs(),
        "peak_abs_error": (pred_peak - target_peak).abs(),
        "delta_mae": inter_chiplet_delta_mae(pred_mean, target_mean, node_batch),
    }
