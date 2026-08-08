"""ModelExporter: what each exportable Wan submodule (text encoder, DiT, VAE enc/dec) must supply
for the generic `torch.export -> ONNX -> TensorRT` pipeline in `export.pipeline` to run.

Concrete exporters (`export.exporters.*`) only need to describe example inputs, dynamic shape
ranges, and I/O tensor names — they never call `torch.export`, `torch.onnx`, or the TensorRT
builder API directly. That's what keeps the export pipeline's behavior (and its future bug fixes)
in one place instead of duplicated per module.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

_DTYPE_NAMES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


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

    def __init__(self, model: torch.nn.Module, static: bool = False) -> None:
        self.model = model
        # Wires up PLAN.md's "Dim.STATIC per resolution profile" strategy (`ResolutionProfile.dynamic`
        # already declared this at the config layer but nothing ever read it). Confirmed as a real
        # need, not just a performance nicety: a wide dynamic-range `vae_decoder` build (min=32,
        # max=latent_height*2) hit a genuine ~94GiB execution-context allocation failure at real
        # inference on RunPod hardware (2026-08-06) — TensorRT appears to size scratch memory (very
        # plausibly the VAE's bottleneck self-attention matrix, which scales with H*W) for the
        # profile's worst case, not the shape actually used. Concrete subclasses check `self.static`
        # in `dynamic_axes()` and return `{}` when set — matching the exact mechanism that already
        # makes `DiTExporter`'s `context` input (never given a dynamic axis at all) build/run fine:
        # an input with no dynamic axis needs no optimization-profile entry and gets a fully static
        # shape all the way from `torch.export` through the built engine. See
        # docs/wan2.2_i2v_14b_notes.md's 2026-08-06 session.
        self.static = static

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
        """Dtype `example_inputs()` should build floating-point tensors with — reads the same
        `TRTWAN_LOADER_DTYPE` env var `wan_comfyui_loader.py`'s loaders already use to cast the
        model itself (default `fp16`), so exporter and loader can never disagree. Deliberately NOT
        inferred from `next(self.model.parameters()).dtype` the way `device` is: Wan models loaded
        here intentionally mix precision (`patch_embedding` stays fp32 for numerical stability, see
        `wan_comfyui_loader.py`), so "the first parameter's dtype" would be unreliable depending on
        parameter iteration order.

        Confirmed necessary against a real failure: without this, fp32-defaulted
        `example_inputs()` silently propagated fp32 activations into fp16-weighted layers
        throughout the graph — `torch.export`/ONNX tolerated the implicit mix without complaint,
        but TensorRT's stricter parser correctly rejected it with "IMatrixMultiplyLayer must have
        same input types... `A` is Float and `B` is Half" — see docs/wan2.2_i2v_14b_notes.md.
        Every `example_inputs()` implementation building a floating-point tensor must pass
        `dtype=self.dtype` (integer tensors, e.g. text encoder token ids, are unaffected).

        Originally hardcoded to `torch.float16` unconditionally — changed 2026-08-07 while testing
        whether the DiT's TensorRT-only self-attention NaN (see docs/wan2.2_i2v_14b_notes.md) is an
        fp16-dynamic-range artifact; setting `TRTWAN_LOADER_DTYPE=bf16` before export now actually
        produces a consistent bf16 graph instead of silently re-mismatching against a bf16-loaded
        model with fp16 example inputs.
        """
        return _DTYPE_NAMES[os.environ.get("TRTWAN_LOADER_DTYPE", "fp16")]

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

    def shape_digest(self) -> str:
        """Digest of `example_inputs()`'s tensor shapes *and* `dynamic_axes()`'s actual profile
        ranges — distinguishes exporter-kwargs combinations (e.g. `VAEEncoderExporter(frames=1)`
        vs `(frames=81)`, or `static=True` vs `static=False` at the same shape) that a
        `runtime.cache.CacheKey`'s `optimization_profile` *name* alone can't, since that's just a
        string like `"480x832"` with no relation to the exporter's actual traced shape. See
        `CacheKey.input_shape_digest`'s docstring for the real collision this fixes.

        Must include `dynamic_axes()`, not just `example_inputs()`'s shapes: confirmed via a real
        second collision the shapes-only version of this method missed — `static=True` changes
        `dynamic_axes()` (empty dict, fully static profile) without changing `example_inputs()`'s
        shapes at all, so a shapes-only digest hashed identically for a static and a wide-dynamic-
        range build of the same nominal shape, and `build engine` silently served the stale
        dynamic-range engine instead of building the requested static one. See
        docs/wan2.2_i2v_14b_notes.md's 2026-08-06 session.
        """
        shapes = {name: list(tensor.shape) for name, tensor in self.example_inputs().items()}
        axes = {
            name: [[axis.name, axis.min, axis.opt, axis.max] for axis in axis_list]
            for name, axis_list in self.dynamic_axes().items()
        }
        payload = json.dumps({"shapes": shapes, "dynamic_axes": axes}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    @property
    def opset_version(self) -> int:
        # >=23: ONNX added a native RMSNormalization op at opset 23, and torch.onnx's dynamo
        # exporter only maps the fused `aten._fused_rms_norm` op to it at that opset or higher.
        # Wan uses RMSNorm for q/k normalization (qk_norm=True) throughout the DiT, confirmed by
        # a real ONNX export attempt failing at opset 20 with "No ONNX function found for
        # aten._fused_rms_norm" — see docs/wan2.2_i2v_14b_notes.md. Below opset 23, exporting any
        # Wan component that uses RMSNorm fails outright, not just degrades.
        return 23
