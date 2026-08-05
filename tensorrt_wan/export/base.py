"""ModelExporter: what each exportable Wan submodule (text encoder, DiT, VAE enc/dec) must supply
for the generic `torch.export -> ONNX -> TensorRT` pipeline in `export.pipeline` to run.

Concrete exporters (`export.exporters.*`) only need to describe example inputs, dynamic shape
ranges, and I/O tensor names — they never call `torch.export`, `torch.onnx`, or the TensorRT
builder API directly. That's what keeps the export pipeline's behavior (and its future bug fixes)
in one place instead of duplicated per module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class DynamicAxis:
    """One dynamic dimension: `name` is the symbolic name torch.export/ONNX will use for it."""

    name: str
    min: int
    opt: int
    max: int


class ModelExporter(ABC):
    """Describes how to export one Wan submodule. Implementations wrap a loaded `nn.Module`;
    they don't load weights themselves — that's the caller's (CLI/API) responsibility, keeping
    checkpoint I/O out of the export pipeline.
    """

    name: str

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @property
    def device(self) -> torch.device:
        """Device `example_inputs()` should build tensors on — inferred from the model's own
        parameters rather than defaulting to CPU. Confirmed necessary against a real export
        attempt: `torch.zeros(...)` with no device defaults to CPU, which torch.export then
        rejects against a GPU-resident model with "Unhandled FakeTensor Device Propagation ...
        found two different devices cpu, cuda:0" — see docs/wan2.2_i2v_14b_notes.md. Every
        `example_inputs()` implementation must build its tensors with `device=self.device`.
        """
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype `example_inputs()` should build floating-point tensors with — fixed at fp16,
        matching this project's default export/inference precision (see `runtime/precision.py`,
        `wan_comfyui_loader.py`'s `TRTWAN_LOADER_DTYPE` default). Deliberately NOT inferred from
        `next(self.model.parameters()).dtype` the way `device` is: Wan models loaded here
        intentionally mix precision (`patch_embedding` stays fp32 for numerical stability, see
        `wan_comfyui_loader.py`), so "the first parameter's dtype" would be unreliable depending
        on parameter iteration order.

        Confirmed necessary against a real failure: without this, fp32-defaulted
        `example_inputs()` silently propagated fp32 activations into fp16-weighted layers
        throughout the graph — `torch.export`/ONNX tolerated the implicit mix without complaint,
        but TensorRT's stricter parser correctly rejected it with "IMatrixMultiplyLayer must have
        same input types... `A` is Float and `B` is Half" — see docs/wan2.2_i2v_14b_notes.md.
        Every `example_inputs()` implementation building a floating-point tensor must pass
        `dtype=self.dtype` (integer tensors, e.g. text encoder token ids, are unaffected).
        """
        return torch.float16

    @abstractmethod
    def example_inputs(self) -> dict[str, torch.Tensor]:
        """Representative inputs for `torch.export`/ONNX tracing, keyed by input tensor name."""
        raise NotImplementedError

    @abstractmethod
    def dynamic_axes(self) -> dict[str, list[DynamicAxis]]:
        """Per-input-tensor dynamic dimensions, for building TensorRT optimization profiles."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def output_names(self) -> list[str]: ...

    @property
    def opset_version(self) -> int:
        # >=23: ONNX added a native RMSNormalization op at opset 23, and torch.onnx's dynamo
        # exporter only maps the fused `aten._fused_rms_norm` op to it at that opset or higher.
        # Wan uses RMSNorm for q/k normalization (qk_norm=True) throughout the DiT, confirmed by
        # a real ONNX export attempt failing at opset 20 with "No ONNX function found for
        # aten._fused_rms_norm" — see docs/wan2.2_i2v_14b_notes.md. Below opset 23, exporting any
        # Wan component that uses RMSNorm fails outright, not just degrades.
        return 23
