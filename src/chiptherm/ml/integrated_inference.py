from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from chiptherm.ml.graph_models import move_graph_to_device, normalize_graph_batch
from chiptherm.ml.models import build_model, count_parameters
from chiptherm.ml.normalization import NormalizationStats, build_metadata_input, build_model_input
from chiptherm.ml.source_response_dataset import (
    SourceResponseNormalizationStats,
    build_source_input,
    normalize_source_input,
    unnormalize_source_prediction,
)
from chiptherm.ml.source_response_models import build_source_response_model, predict_source_rise


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BASE_MODE = "source_superposition_v1"
GRID_SHAPE = (64, 64)
SUPPORTED_GRAPH_ARCHITECTURES = {
    "miniunet_refine_conditioned_decomposed_graph",
    "miniunet_refine_conditioned_decomposed_global_graph",
    "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
    "miniunet_refine_conditioned_decomposed_pairwise",
    "miniunet_refine_conditioned_decomposed_pairwise_basis",
}
SUPPORTED_CONDITIONED_ARCHITECTURES = {
    "miniunet_refine_conditioned",
    "miniunet_refine_conditioned_decomposed",
    "miniunet_refine_conditioned_decomposed_global",
    "miniunet_refine_conditioned_decomposed_feature_fusion",
    "miniunet_refine_conditioned_decomposed_graph",
    "miniunet_refine_conditioned_decomposed_global_graph",
    "miniunet_refine_conditioned_decomposed_feature_fusion_graph",
    "miniunet_refine_conditioned_decomposed_pairwise",
    "miniunet_refine_conditioned_decomposed_pairwise_basis",
}


class StageTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, float] = {}
        self.methods: dict[str, str] = {}

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def time(self, name: str, *, gpu: bool = False) -> Iterator[None]:
        if gpu and self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                end_event.synchronize()
                elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
                self.values[name] = self.values.get(name, 0.0) + elapsed
                self.methods[name] = "cuda_event"
            return
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.values[name] = self.values.get(name, 0.0) + time.perf_counter() - start
            self.methods[name] = "synchronized_wall_clock"


class IntegratedChipThermModel:
    """Authoritative uncached ChipTherm source-superposition + residual CNN/GNN pipeline."""

    def __init__(
        self,
        *,
        source_checkpoint: str | Path,
        residual_checkpoint: str | Path,
        device: torch.device,
        data_root: str | Path | None = None,
        execution_mode: str = "reference",
        deterministic: bool = False,
        precision: str = "fp32",
        channels_last: bool = False,
        compile_source: bool = False,
        compile_package_cnn: bool = False,
        compile_graph: bool = False,
        device_summation: bool = False,
        non_blocking_transfer: bool = False,
    ) -> None:
        self.device = device
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else None
        self.execution_mode = str(execution_mode)
        if self.execution_mode not in {"reference", "optimized"}:
            raise ValueError("execution_mode must be reference or optimized")
        self.precision = str(precision)
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.precision in {"fp16", "bf16"} and self.device.type != "cuda":
            raise ValueError(f"precision={self.precision} is currently supported only on CUDA")
        self.channels_last = bool(channels_last)
        self.compile_source = bool(compile_source)
        self.compile_package_cnn = bool(compile_package_cnn)
        self.compile_graph = bool(compile_graph)
        self.device_summation = bool(device_summation)
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self.source_checkpoint_path = Path(source_checkpoint).expanduser().resolve()
        self.residual_checkpoint_path = Path(residual_checkpoint).expanduser().resolve()
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.source_checkpoint_sha256 = sha256_file(self.source_checkpoint_path)
        self.residual_checkpoint_sha256 = sha256_file(self.residual_checkpoint_path)

        source_payload = torch.load(self.source_checkpoint_path, map_location=device, weights_only=False)
        self.source_stats = SourceResponseNormalizationStats.from_dict(source_payload["normalization"])
        self.source_model = build_source_response_model(source_payload["model_config"]).to(device)
        self.source_model.load_state_dict(source_payload["model_state_dict"])
        self.source_model.eval()
        if self.channels_last:
            self.source_model = self.source_model.to(memory_format=torch.channels_last)
        if self.compile_source:
            self.source_model = compile_module(self.source_model, "source model")
        self.source_config = dict(source_payload.get("model_config", {}))
        self.source_training_lineage = dict(source_payload.get("training_lineage", {}))
        self.source_normalization_identity = sha256_json(source_payload.get("normalization", {}))

        residual_payload = torch.load(self.residual_checkpoint_path, map_location=device, weights_only=False)
        self.residual_stats = NormalizationStats(**residual_payload["normalization"])
        self.residual_config = dict(residual_payload["model_config"])
        self.residual_training_lineage = dict(residual_payload.get("training_lineage", {}))
        self.residual_normalization_identity = sha256_json(residual_payload.get("normalization", {}))
        self.physics_input_mode = str(self.residual_config.get("physics_input_mode", "v1"))
        if self.physics_input_mode != SOURCE_BASE_MODE:
            raise ValueError(
                f"integrated inference requires residual checkpoint physics_input_mode={SOURCE_BASE_MODE}, "
                f"got {self.physics_input_mode}"
            )
        self.residual_model = build_model(self.residual_config).to(device)
        self.residual_model.load_state_dict(residual_payload["model_state_dict"])
        self.residual_model.eval()
        self.graph_stats = self.residual_config.get("graph_normalization")
        self.residual_architecture = str(self.residual_config.get("architecture", ""))
        self.mean_head_mode = str(self.residual_config.get("mean_head_mode", "direct_k"))
        if self.mean_head_mode not in {"direct_k", "residual_resistance"}:
            raise ValueError(f"unsupported residual mean_head_mode={self.mean_head_mode}")
        self.graph_enabled = self.residual_architecture in SUPPORTED_GRAPH_ARCHITECTURES
        self.conditioned = self.residual_architecture in SUPPORTED_CONDITIONED_ARCHITECTURES
        if self.channels_last:
            self.residual_model = self.residual_model.to(memory_format=torch.channels_last)
        if self.compile_package_cnn:
            if hasattr(self.residual_model, "cnn_model"):
                self.residual_model.cnn_model = compile_module(self.residual_model.cnn_model, "package CNN")
            else:
                self.residual_model = compile_module(self.residual_model, "residual package model")
        if self.compile_graph and hasattr(self.residual_model, "graph_model"):
            self.residual_model.graph_model = compile_module(self.residual_model.graph_model, "graph model")

    def manifest(self) -> dict[str, Any]:
        return {
            "source_checkpoint": str(self.source_checkpoint_path),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_model_config": self.source_config,
            "source_training_lineage": self.source_training_lineage,
            "source_normalization_sha256": self.source_normalization_identity,
            "residual_checkpoint": str(self.residual_checkpoint_path),
            "residual_checkpoint_sha256": self.residual_checkpoint_sha256,
            "residual_model_config": self.residual_config,
            "residual_training_lineage": self.residual_training_lineage,
            "residual_normalization_sha256": self.residual_normalization_identity,
            "source_superposition_version": self.residual_training_lineage.get(
                "source_superposition_version"
            ),
            "source_parameter_count": count_parameters(self.source_model),
            "residual_parameter_count": count_parameters(self.residual_model),
            "physics_input_mode": self.physics_input_mode,
            "base_definition": "ambient_K + sum_i source_power_i * source_response_operator(source_i)",
            "reconstruction": (
                "source_superposition_base_K + total_power_W * delta_R_eff_K_per_W "
                "+ zero_mean_centered_residual_K"
                if self.mean_head_mode == "residual_resistance"
                else "ambient_K + predicted_mean_rise_K + zero_mean_centered_field_K"
            ),
            "mean_head_mode": self.mean_head_mode,
            "data_root": str(self.data_root) if self.data_root is not None else None,
            "execution_mode": self.execution_mode,
            "precision": self.precision,
            "channels_last": self.channels_last,
            "compile_source": self.compile_source,
            "compile_package_cnn": self.compile_package_cnn,
            "compile_graph": self.compile_graph,
            "device_summation": self.device_summation,
            "non_blocking_transfer": self.non_blocking_transfer,
            "software": software_manifest(self.device),
        }

    def device_synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def predict_batch(
        self,
        batch: dict[str, Any],
        rows: list[dict[str, str]],
        *,
        source_batch_size: int,
        profile_components: bool = False,
        graph_correction_scale: float = 1.0,
        return_source_responses: bool = False,
    ) -> dict[str, Any]:
        timer = StageTimer(self.device)
        with timer.time("canonical_batch_host_preparation_s"):
            x_cpu = batch["x"].detach().cpu().float()
        with timer.time("canonical_batch_host_to_device_s", gpu=True):
            x = batch["x"].to(self.device, non_blocking=self.non_blocking_transfer).float()
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            ambient = batch["ambient_K"].to(self.device, non_blocking=self.non_blocking_transfer).float()
            ensure_finite_tensor("spatial_x", x)
            ensure_finite_tensor("ambient_K", ambient)
            total_power = validate_total_power(
                batch.get("total_power_W"),
                batch_size=int(x.shape[0]),
                device=self.device,
                non_blocking=self.non_blocking_transfer,
            )

        source_base, source_counts, source_names, source_responses = self._source_superposition_base(
            rows,
            x_cpu,
            source_batch_size=source_batch_size,
            timer=timer,
            return_source_responses=return_source_responses,
        )
        with timer.time("residual_input_assembly_s", gpu=True):
            source_base = source_base.to(self.device, non_blocking=self.non_blocking_transfer)
            model_input = build_model_input(x, source_base, self.residual_stats, physics_input_mode=SOURCE_BASE_MODE)
            expected_channels = int(self.residual_config.get("input_channels", model_input.shape[1]))
            if int(model_input.shape[1]) != expected_channels:
                raise ValueError(
                    f"residual checkpoint expects {expected_channels} input channels, "
                    f"integrated input provides {model_input.shape[1]}"
                )
            ensure_finite_tensor("residual_model_input", model_input)
            if self.channels_last:
                model_input = model_input.contiguous(memory_format=torch.channels_last)
            metadata_input = build_metadata_input(batch.get("metadata_vector"), self.residual_stats)
            if metadata_input is not None:
                metadata_input = metadata_input.to(self.device, non_blocking=self.non_blocking_transfer)
                ensure_finite_tensor("metadata_input", metadata_input)
            expected_metadata_dim = int(self.residual_config.get("metadata_dim", 0))
            if expected_metadata_dim and (
                metadata_input is None or int(metadata_input.shape[1]) != expected_metadata_dim
            ):
                shape = None if metadata_input is None else tuple(metadata_input.shape)
                raise ValueError(
                    f"residual checkpoint expects metadata_dim={expected_metadata_dim}, got {shape}"
                )
            graph_batch = None
            if self.graph_enabled:
                graph = batch.get("graph")
                if graph is None:
                    raise ValueError("residual graph checkpoint requires graph inputs")
                graph_batch = normalize_graph_batch(move_graph_to_device(graph, self.device), self.graph_stats)
                for key, value in graph_batch.items():
                    if torch.is_tensor(value) and value.is_floating_point():
                        ensure_finite_tensor(f"graph_batch[{key!r}]", value)

        with timer.time("residual_total_forward_s", gpu=True):
            if profile_components and hasattr(self.residual_model, "forward_profile"):
                profile_kwargs: dict[str, Any] = {
                    "synchronize": timer.synchronize,
                    "graph_correction_scale": graph_correction_scale,
                }
                if self.mean_head_mode == "residual_resistance":
                    profile_kwargs["total_power_W"] = total_power
                with self.autocast_context():
                    outputs, residual_timings = self.residual_model.forward_profile(
                        model_input,
                        metadata_input,
                        graph_batch,
                        **profile_kwargs,
                    )
                for name, value in residual_timings.items():
                    timer.values[name] = timer.values.get(name, 0.0) + float(value)
                    timer.methods[name] = "model_forward_profile_synchronized_wall_clock"
            elif self.graph_enabled:
                with self.autocast_context():
                    outputs = self.residual_model(
                        model_input,
                        metadata_input,
                        graph_batch,
                        return_diagnostics=True,
                        graph_correction_scale=graph_correction_scale,
                        ambient=ambient,
                        total_power_W=total_power,
                    )
            elif self.conditioned:
                with self.autocast_context():
                    kwargs: dict[str, Any] = {}
                    if self.mean_head_mode == "residual_resistance":
                        kwargs["total_power_W"] = total_power
                    if getattr(self.residual_model, "feature_fusion_enabled", False):
                        kwargs["return_diagnostics"] = True
                    outputs = self.residual_model(model_input, metadata_input, **kwargs)
            else:
                with self.autocast_context():
                    outputs = self.residual_model(model_input)

        with timer.time("final_reconstruction_s", gpu=True):
            final_temperature = reconstruct_decomposed_temperature(
                outputs,
                ambient,
                source_base=source_base,
                mean_head_mode=self.mean_head_mode,
            )
            cnn_mean = outputs.get("cnn_mean_rise", outputs["mean_rise"])
            cnn_centered = outputs.get("cnn_centered_field", outputs["centered_field"])
            cnn_centered = cnn_centered - cnn_centered.mean(dim=(-2, -1), keepdim=True)
            cnn_base = source_base if self.mean_head_mode == "residual_resistance" else ambient[:, None, None]
            cnn_only_temperature = cnn_base + cnn_mean[:, None, None] + cnn_centered
            graph_correction = outputs.get("graph_correction_field")
            ensure_finite_tensor("final_temperature_K", final_temperature)
        with timer.time("final_device_to_host_transfer_s", gpu=True):
            final_temperature_host = final_temperature.detach().cpu()

        return {
            "ambient_K": ambient,
            "total_power_W": total_power,
            "source_superposition_base_K": source_base,
            "source_responses_K": source_responses,
            "final_temperature_K": final_temperature,
            "final_temperature_host_K": final_temperature_host,
            "cnn_only_temperature_K": cnn_only_temperature,
            "graph_correction_K": graph_correction,
            "outputs": outputs,
            "components": component_payload(outputs, ambient),
            "model_input": model_input,
            "metadata_input": metadata_input,
            "graph_batch": graph_batch,
            "source_counts": source_counts,
            "source_names": source_names,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "residual_checkpoint_sha256": self.residual_checkpoint_sha256,
            "timings": timer.values,
            "timing_methods": timer.methods,
        }

    @torch.no_grad()
    def residual_from_base(
        self,
        batch: dict[str, Any],
        source_base: torch.Tensor,
        *,
        graph_correction_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device, non_blocking=self.non_blocking_transfer).float()
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        ambient = batch["ambient_K"].to(self.device, non_blocking=self.non_blocking_transfer).float()
        total_power = validate_total_power(
            batch.get("total_power_W"),
            batch_size=int(x.shape[0]),
            device=self.device,
            non_blocking=self.non_blocking_transfer,
        )
        source_base = source_base.to(self.device, non_blocking=self.non_blocking_transfer).float()
        model_input = build_model_input(x, source_base, self.residual_stats, physics_input_mode=SOURCE_BASE_MODE)
        if self.channels_last:
            model_input = model_input.contiguous(memory_format=torch.channels_last)
        metadata_input = build_metadata_input(batch.get("metadata_vector"), self.residual_stats)
        if metadata_input is not None:
            metadata_input = metadata_input.to(self.device, non_blocking=self.non_blocking_transfer)
        graph_batch = None
        if self.graph_enabled:
            graph_batch = normalize_graph_batch(move_graph_to_device(batch["graph"], self.device), self.graph_stats)
            with self.autocast_context():
                outputs = self.residual_model(
                    model_input,
                    metadata_input,
                    graph_batch,
                    return_diagnostics=True,
                    graph_correction_scale=graph_correction_scale,
                    ambient=ambient,
                    total_power_W=total_power,
                )
        elif self.conditioned:
            with self.autocast_context():
                kwargs: dict[str, Any] = {}
                if self.mean_head_mode == "residual_resistance":
                    kwargs["total_power_W"] = total_power
                if getattr(self.residual_model, "feature_fusion_enabled", False):
                    kwargs["return_diagnostics"] = True
                outputs = self.residual_model(model_input, metadata_input, **kwargs)
        else:
            with self.autocast_context():
                outputs = self.residual_model(model_input)
        final_temperature = reconstruct_decomposed_temperature(
            outputs,
            ambient,
            source_base=source_base,
            mean_head_mode=self.mean_head_mode,
        )
        cnn_mean = outputs.get("cnn_mean_rise", outputs["mean_rise"])
        cnn_centered = outputs.get("cnn_centered_field", outputs["centered_field"])
        cnn_centered = cnn_centered - cnn_centered.mean(dim=(-2, -1), keepdim=True)
        cnn_base = source_base if self.mean_head_mode == "residual_resistance" else ambient[:, None, None]
        cnn_only_temperature = cnn_base + cnn_mean[:, None, None] + cnn_centered
        return {
            "final_temperature_K": final_temperature,
            "cnn_only_temperature_K": cnn_only_temperature,
            "graph_correction_K": outputs.get("graph_correction_field"),
            "outputs": outputs,
            "components": component_payload(outputs, ambient),
        }

    @torch.no_grad()
    def _source_superposition_base(
        self,
        rows: list[dict[str, str]],
        x_cpu: torch.Tensor,
        *,
        source_batch_size: int,
        timer: StageTimer,
        return_source_responses: bool = False,
    ) -> tuple[torch.Tensor, list[int], list[list[str]], torch.Tensor | None]:
        packages: list[dict[str, Any]] = []
        with timer.time("package_metadata_layout_parsing_s"):
            for row in rows:
                packages.append(load_package_metadata(row, data_root=self.data_root))
        with timer.time("source_raster_construction_s"):
            for package, x_item in zip(packages, x_cpu, strict=True):
                build_package_sources(package, x_item.numpy())
        flat_inputs: list[np.ndarray] = []
        flat_powers: list[float] = []
        package_ids: list[int] = []
        for package_index, package in enumerate(packages):
            for source_input, source_power in zip(package["source_inputs"], package["source_powers"], strict=True):
                flat_inputs.append(source_input)
                flat_powers.append(float(source_power))
                package_ids.append(package_index)
        sums = [np.zeros(GRID_SHAPE, dtype=np.float64) for _ in packages]
        captured_responses: list[np.ndarray] = []
        device_sums: torch.Tensor | None = None
        device_group_ids: torch.Tensor | None = None
        if self.device_summation:
            device_sums = torch.zeros((len(packages), *GRID_SHAPE), dtype=torch.float32, device=self.device)
            device_group_ids = torch.tensor(package_ids, dtype=torch.long, device=self.device)
        with timer.time("source_response_total_s"):
            for start in range(0, len(flat_inputs), source_batch_size):
                stop = min(start + source_batch_size, len(flat_inputs))
                with timer.time("source_host_to_device_transfer_s", gpu=True):
                    x_source = torch.from_numpy(np.stack(flat_inputs[start:stop]).astype(np.float32, copy=False)).to(
                        self.device,
                        non_blocking=self.non_blocking_transfer,
                    )
                    expected_source_channels = int(
                        self.source_config.get("input_channels", x_source.shape[1])
                    )
                    if int(x_source.shape[1]) != expected_source_channels:
                        raise ValueError(
                            f"source checkpoint expects {expected_source_channels} channels, "
                            f"constructed source input provides {x_source.shape[1]}"
                        )
                    if self.channels_last:
                        x_source = x_source.contiguous(memory_format=torch.channels_last)
                    source_power = torch.tensor(flat_powers[start:stop], dtype=torch.float32, device=self.device)
                with timer.time("source_input_normalization_s", gpu=True):
                    source_input = normalize_source_input(x_source, self.source_stats)
                    if self.channels_last:
                        source_input = source_input.contiguous(memory_format=torch.channels_last)
                with timer.time("source_response_model_inference_s", gpu=True):
                    with self.autocast_context():
                        pred_norm = self.source_model(source_input)
                with timer.time("source_kw_denormalization_s", gpu=True):
                    pred_unit = unnormalize_source_prediction(pred_norm, self.source_stats)
                with timer.time("source_power_scaling_s", gpu=True):
                    pred_rise = predict_source_rise(pred_unit, source_power)
                with timer.time("source_segment_sum_s"):
                    if return_source_responses:
                        captured_responses.extend(
                            pred_rise.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                    if self.device_summation:
                        assert device_sums is not None and device_group_ids is not None
                        partial_groups = device_group_ids[start:stop]
                        device_sums.index_add_(0, partial_groups, pred_rise.float())
                    else:
                        rise_cpu = pred_rise.detach().cpu().numpy()
                        accumulate_source_chunk(
                            sums,
                            rise_cpu,
                            package_ids[start:stop],
                        )
        with timer.time("ambient_base_reconstruction_s"):
            if self.device_summation:
                assert device_sums is not None
                sums_array = device_sums.detach().cpu().numpy().astype(np.float64, copy=False)
            else:
                sums_array = sums
            maps = [
                np.asarray(float(package["ambient_K"]) + rise_sum, dtype=np.float32)
                for package, rise_sum in zip(packages, sums_array, strict=True)
            ]
            for row, base in zip(rows, maps, strict=True):
                if base.shape != GRID_SHAPE:
                    raise ValueError(f"{row['sample_uid']} source base has shape {base.shape}, expected {GRID_SHAPE}")
                if not np.isfinite(base).all():
                    raise ValueError(f"{row['sample_uid']} source base contains non-finite values")
            base_tensor = torch.from_numpy(np.stack(maps).astype(np.float32, copy=False))
        return (
            base_tensor,
            [int(package["num_sources"]) for package in packages],
            [list(package["source_names"]) for package in packages],
            (
                torch.from_numpy(np.stack(captured_responses).astype(np.float32, copy=False))
                if return_source_responses and captured_responses
                else None
            ),
        )

    @contextmanager
    def autocast_context(self) -> Iterator[None]:
        if self.precision == "fp32":
            with nullcontext():
                yield
            return
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        with torch.autocast(device_type=self.device.type, dtype=dtype):
            yield


def reconstruct_decomposed_temperature(
    outputs: dict[str, torch.Tensor],
    ambient: torch.Tensor,
    *,
    source_base: torch.Tensor | None = None,
    mean_head_mode: str = "direct_k",
) -> torch.Tensor:
    centered = outputs["centered_field"]
    centered = centered - centered.mean(dim=(-2, -1), keepdim=True)
    if mean_head_mode == "residual_resistance":
        if source_base is None:
            raise ValueError("residual_resistance reconstruction requires source_superposition_base_K")
        return source_base + outputs["mean_rise"][:, None, None] + centered
    if mean_head_mode != "direct_k":
        raise ValueError(f"unsupported mean_head_mode={mean_head_mode}")
    return ambient[:, None, None] + outputs["mean_rise"][:, None, None] + centered


def component_payload(outputs: dict[str, torch.Tensor], ambient: torch.Tensor) -> dict[str, torch.Tensor]:
    del ambient
    components: dict[str, torch.Tensor] = {}
    for key in (
        "mean_rise",
        "cnn_mean_rise",
        "graph_mean_delta",
        "centered_field",
        "cnn_centered_field",
        "graph_correction_field",
        "scaled_graph_correction_field",
    ):
        value = outputs.get(key)
        if torch.is_tensor(value):
            components[key] = value
    if "cnn_centered_field" not in components and "centered_field" in components:
        components["cnn_centered_field"] = components["centered_field"]
    if "cnn_mean_rise" not in components and "mean_rise" in components:
        components["cnn_mean_rise"] = components["mean_rise"]
    return components


def load_package_metadata(
    row: dict[str, str],
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    paths = canonical_source_paths(row, data_root=data_root)
    layout = load_json(paths["layout"])
    power = load_yaml(paths["power"])
    package = load_yaml(paths["package"])
    chiplets = list(layout.get("chiplets", []))
    if not chiplets:
        raise ValueError(f"{paths['layout']} has no chiplets")
    powers = active_power_map(power)
    missing = [str(chiplet.get("name", "")) for chiplet in chiplets if str(chiplet.get("name", "")) not in powers]
    if missing:
        raise ValueError(f"{paths['power']} missing chiplet powers: {', '.join(missing)}")
    return {
        "row": row,
        "layout": layout,
        "chiplets": chiplets,
        "powers": powers,
        "ambient_K": float(package.get("ambient_K", 318.15)),
        "source_names": [str(chiplet["name"]) for chiplet in chiplets],
        "source_powers": [],
        "source_inputs": [],
        "num_sources": len(chiplets),
        "paths": paths,
    }


def build_package_sources(package: dict[str, Any], x_array: np.ndarray) -> None:
    source_inputs: list[np.ndarray] = []
    source_powers: list[float] = []
    for source_index, chiplet in enumerate(package["chiplets"]):
        name = str(chiplet["name"])
        source_power = float(package["powers"][name])
        source_inputs.append(build_source_input(x_array, package["layout"], source_index, source_power))
        source_powers.append(source_power)
    package["source_inputs"] = source_inputs
    package["source_powers"] = np.asarray(source_powers, dtype=np.float32)


def rows_from_batch_metadata(metadata: dict[str, Any], batch_size: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    keys = [
        "sample_uid",
        "original_sample_uid",
        "case_id",
        "dataset_source",
        "split",
        "x_path",
        "y_path",
        "prediction_path",
        "residual_path",
        "graph_path",
        "num_chiplets",
        "total_power_W",
    ]
    for index in range(batch_size):
        row: dict[str, str] = {}
        for key in keys:
            if key in metadata:
                row[key] = stringify_metadata_value(metadata[key], index, batch_size)
        if "sample_uid" not in row:
            raise ValueError("batch metadata does not include sample_uid")
        rows.append(row)
    return rows


def stringify_metadata_value(value: Any, index: int, batch_size: int) -> str:
    if isinstance(value, (list, tuple)):
        item = value[index]
    elif torch.is_tensor(value):
        if value.ndim == 0:
            item = value.item()
        else:
            item = value.detach().cpu().reshape(-1)[index].item()
    else:
        item = value
    if item is None:
        return ""
    return str(item)


def canonical_source_paths(
    row: dict[str, str],
    *,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    source_dir = source_dir_for_row(row, data_root=data_root)
    declared = {
        "layout": row.get("source_layout_path") or row.get("layout_path"),
        "power": row.get("source_power_path") or row.get("power_path"),
        "package": row.get("source_package_path") or row.get("package_path"),
        "hotspot": row.get("source_hotspot_path") or row.get("hotspot_path"),
    }
    resolved = {
        key: resolve_declared_path(value, data_root=data_root)
        if value
        else source_dir / f"{key}.{'json' if key == 'layout' else 'yaml'}"
        for key, value in declared.items()
    }
    missing = [f"{name}={path}" for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{row.get('sample_uid', '<unknown>')} source files are missing under resolution root "
            f"{Path(data_root).expanduser().resolve() if data_root is not None else '<legacy fallback>'}: "
            + ", ".join(missing)
        )
    return {
        "source_dir": source_dir,
        **resolved,
    }


def source_dir_for_row(
    row: dict[str, str],
    *,
    data_root: str | Path | None = None,
) -> Path:
    explicit = row.get("source_dir")
    if explicit:
        return resolve_declared_path(explicit, data_root=data_root)
    for key in ("source_layout_path", "layout_path"):
        if row.get(key):
            return resolve_declared_path(row[key], data_root=data_root).parent
    if data_root is not None:
        raise ValueError(
            f"{row.get('sample_uid', '<unknown>')} has no source_dir/layout path; "
            f"Benchmark v2 rows with explicit data_root={Path(data_root).expanduser().resolve()} "
            "must not use the legacy repository-relative fallback"
        )
    case_id = row["case_id"]
    original = row.get("original_sample_uid") or row["sample_uid"]
    sample_name = original
    prefix = f"{case_id}_"
    if sample_name.startswith(prefix):
        sample_name = sample_name[len(prefix) :]
    return REPO_ROOT / "data/runs/benchmarks" / row["dataset_source"] / case_id / sample_name / "source"


def resolve_declared_path(value: str, *, data_root: str | Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if data_root is not None:
        return Path(data_root).expanduser().resolve() / path
    return resolve_path(value)


def validate_total_power(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError("integrated inference requires raw batch['total_power_W'] tensor")
    if value.ndim not in {1, 2} or (value.ndim == 2 and int(value.shape[1]) != 1):
        raise ValueError(f"total_power_W must have shape [B] or [B,1], got {tuple(value.shape)}")
    if int(value.numel()) != int(batch_size):
        raise ValueError(
            f"total_power_W batch mismatch: expected {batch_size} values, got shape {tuple(value.shape)}"
        )
    result = value.to(device=device, dtype=torch.float32, non_blocking=non_blocking).view(-1)
    if not torch.isfinite(result).all():
        raise ValueError("total_power_W contains non-finite values")
    if torch.any(result <= 0.0):
        raise ValueError("total_power_W must be strictly positive for Benchmark v2 inference")
    return result


def accumulate_source_chunk(
    package_sums: list[np.ndarray],
    source_responses_K: np.ndarray,
    package_ids: list[int],
) -> None:
    if int(source_responses_K.shape[0]) != len(package_ids):
        raise ValueError(
            "source response/package-id count mismatch: "
            f"{source_responses_K.shape[0]} != {len(package_ids)}"
        )
    for response, package_id in zip(source_responses_K, package_ids, strict=True):
        if package_id < 0 or package_id >= len(package_sums):
            raise ValueError(f"source package id {package_id} is outside [0, {len(package_sums)})")
        package_sums[package_id] += response.astype(np.float64, copy=False)


def ensure_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        bad = torch.nonzero(~torch.isfinite(value), as_tuple=False)
        first = bad[0].detach().cpu().tolist() if bad.numel() else []
        raise ValueError(f"{name} contains NaN/Inf; first bad index={first}")


def active_power_map(power: dict[str, Any]) -> dict[str, float]:
    workload = power.get("active_workload", "nominal")
    workloads = power.get("workloads") or {}
    if workload in workloads:
        return {str(name): float(value) for name, value in workloads[workload].items()}
    if "chiplets" in power:
        return {str(name): float(value) for name, value in power["chiplets"].items()}
    raise ValueError("power.yaml has no active workload or chiplets map")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def resolve_path(path_value: str, base: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, REPO_ROOT / path]
    if base is not None:
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_module(module: torch.nn.Module, name: str) -> torch.nn.Module:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError(f"torch.compile requested for {name}, but this PyTorch build does not provide torch.compile")
    return compile_fn(module)


def software_manifest(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload.update(
            {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": torch.cuda.get_device_capability(device),
            }
        )
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    f"--id={device.index or 0}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            payload["nvidia_driver"] = result.stdout.strip().splitlines()[0]
        except (FileNotFoundError, subprocess.SubprocessError, IndexError):
            payload["nvidia_driver"] = None
    return payload
