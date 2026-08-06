"""Stage 1: PyTorch -> torch.export.ExportedProgram.

Per the project's development rule this is never invoked in this environment — it is exercised
on the RunPod GPU instances during the validation phase.
"""

from __future__ import annotations

import torch

from tensorrt_wan.export.base import ModelExporter
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)


def export_to_torch_export(exporter: ModelExporter) -> torch.export.ExportedProgram:
    """Trace `exporter.model` with `torch.export.export` using its example inputs.

    Dynamic dimensions declared via `exporter.dynamic_axes()` are passed through as
    `torch.export.Dim.AUTO` markers rather than explicit `Dim(name, min=.., max=..)` ranges.
    `Dim(min=, max=)` is an *assertion* — torch.export requires the traced code to behave
    consistently across the entire declared range, which fails on Wan's real DiT: confirmed
    against actual hardware that `pad_to_patch_size`/`rope_encode`'s patch-alignment arithmetic
    (patch_size=(1,2,2)) generates guards that only hold for specific values, not a smooth range,
    producing `ConstraintViolationError: Specializations unexpectedly required` even for spatial
    dims that are genuinely meant to vary (height/width across resolution profiles) — see
    docs/wan2.2_i2v_14b_notes.md. `Dim.AUTO` lets torch.export infer what's actually dynamic from
    the trace instead of us asserting a range upfront that conflicts with the model's real
    constraints — suggested directly in that error's own message. The min/opt/max values in
    `DynamicAxis` are unaffected by this — they're still used as-is by `export.trt_build`'s
    `_build_optimization_profile`, which never inspects torch.export's `Dim` objects at all.
    """
    example_inputs = exporter.example_inputs()
    dynamic_shapes = _build_dynamic_shapes(exporter)

    logger.info("torch.export: %s (inputs=%s)", exporter.name, list(example_inputs))
    return torch.export.export(
        exporter.model,
        args=(),
        kwargs=example_inputs,
        dynamic_shapes=dynamic_shapes or None,
    )


def _build_dynamic_shapes(exporter: ModelExporter) -> dict[str, dict[int, object] | None]:
    """One entry per `example_inputs()` kwarg, not just the ones with dynamic axes.

    Confirmed necessary against a real `torch.export` run on torch 2.10.0 (newer than whatever
    version this was originally written against): passing `dynamic_shapes` as a dict now requires
    its top-level keys to exactly match every arg name in `kwargs`, not just the dynamic-axis
    subset — `torch._dynamo.exc.UserError: ... top-level keys must be the arg names [...] of
    `inputs`, but here they are [...]`. Inputs with no dynamic axes (e.g. `timestep`/`context`,
    fully static once batch specializes — see `export_to_torch_export`'s docstring) get `None`
    rather than being omitted.
    """
    example_inputs = exporter.example_inputs()
    axes_by_input = exporter.dynamic_axes()
    dynamic_shapes: dict[str, dict[int, object] | None] = {}
    for input_name, example in example_inputs.items():
        dims: dict[int, object] = {}
        for axis in axes_by_input.get(input_name, []):
            dim_index = _resolve_axis_index(example, axis.name)
            dims[dim_index] = torch.export.Dim.AUTO
        dynamic_shapes[input_name] = dims or None
    return dynamic_shapes


def _resolve_axis_index(example: torch.Tensor, axis_name: str) -> int:
    """Axis names follow the convention `dim{N}` (e.g. `dim0` for the batch axis)."""
    if not axis_name.startswith("dim") or not axis_name[3:].isdigit():
        raise ValueError(f"Dynamic axis name {axis_name!r} must be of the form 'dimN'")
    index = int(axis_name[3:])
    if index >= example.dim():
        raise ValueError(f"Axis {axis_name!r} out of range for tensor with shape {tuple(example.shape)}")
    return index
