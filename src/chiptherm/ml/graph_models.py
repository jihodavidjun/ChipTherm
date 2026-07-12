from __future__ import annotations

from dataclasses import asdict, dataclass
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

    def forward(self, graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        return {
            "node_embeddings": h,
            "node_raster_values": self.node_raster_head(h),
            "graph_embedding": self.global_head(graph_embedding),
        }


def rasterize_node_values(
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

