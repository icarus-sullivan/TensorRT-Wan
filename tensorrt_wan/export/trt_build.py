"""Stage 3: ONNX -> serialized TensorRT engine.

Builds one optimization profile per `ResolutionProfile` the caller supplies, so a single engine
covers every configured resolution rather than needing one engine per shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tensorrt_wan.config.schema import PrecisionMode, ResolutionProfile
from tensorrt_wan.export.base import DynamicAxis, ModelExporter
from tensorrt_wan.utils.logging import get_logger

if TYPE_CHECKING:
    import tensorrt as trt

logger = get_logger(__name__)


def build_tensorrt_engine(
    onnx_path: str | Path,
    exporter: ModelExporter,
    resolution_profiles: list[ResolutionProfile],
    precision: PrecisionMode,
    workspace_limit_mb: int | None = None,
) -> bytes:
    """Parse `onnx_path` and build a serialized engine covering `resolution_profiles`.

    Returns the raw serialized engine; writing it to `EngineCache` is the caller's job (see
    `cli.commands.build`) so this function stays agnostic of cache-key/invalidation logic.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX model {onnx_path}:\n{errors}")

    _validate_precision(network, precision)

    config = builder.create_builder_config()
    if workspace_limit_mb is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_limit_mb * (1 << 20))

    for profile_spec in resolution_profiles:
        config.add_optimization_profile(_build_optimization_profile(builder, exporter, profile_spec))

    logger.info(
        "Building TensorRT engine for %s: %d profile(s), precision=%s",
        exporter.name,
        len(resolution_profiles),
        precision,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {exporter.name}")
    return bytes(serialized)


_PRECISION_TO_DTYPE_NAME = {"fp32": "FLOAT", "fp16": "HALF", "bf16": "BF16", "fp8": "FP8"}


def _validate_precision(network: "trt.INetworkDefinition", precision: PrecisionMode) -> None:
    """Confirm the parsed ONNX graph's I/O tensors are actually in `precision`.

    There is no `BuilderFlag.FP16`/`BF16`/`FP8` to set here — confirmed against a real TensorRT
    11.2 build on RunPod hardware that those flags don't exist at all for `STRONGLY_TYPED`
    networks (see docs/wan2.2_i2v_14b_notes.md). A strongly-typed network's precision comes
    entirely from the tensor dtypes already baked into the ONNX graph by the export stage
    (`export.torch_export`/`export.onnx_export`, driven by whatever dtype the loaded model was
    cast to before tracing — see `examples/loaders/wan_comfyui_loader.py`'s `.to(dtype=...)` for
    where that actually happens). There is nothing left for the TensorRT builder to coerce
    post-hoc, so a mismatch here is a real upstream bug (wrong export dtype), not something this
    function can fix — it fails loudly instead of silently building the wrong-precision engine.
    `precision="auto"` is never passed here; callers resolve it to a concrete value first (see
    `runtime.precision.select_precision`), so every entry in the map above is checked exactly.
    """
    import tensorrt as trt

    expected = getattr(trt.DataType, _PRECISION_TO_DTYPE_NAME[precision])
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if tensor.dtype != expected:
            raise RuntimeError(
                f"Requested precision={precision!r} ({expected}) but ONNX input {tensor.name!r} "
                f"is {tensor.dtype}. STRONGLY_TYPED networks take precision entirely from the "
                f"ONNX graph's own tensor dtypes; re-export with the model cast to {precision} "
                f"before torch.export instead of expecting the TensorRT builder to coerce it."
            )


def _build_optimization_profile(
    builder: "trt.Builder",
    exporter: ModelExporter,
    resolution: ResolutionProfile,
) -> "trt.IOptimizationProfile":
    profile = builder.create_optimization_profile()
    example_inputs = exporter.example_inputs()
    for input_name, axes in exporter.dynamic_axes().items():
        example_shape = list(example_inputs[input_name].shape)
        min_shape, opt_shape, max_shape = list(example_shape), list(example_shape), list(example_shape)
        for axis in axes:
            dim_index = _axis_index(axis)
            min_shape[dim_index] = axis.min
            opt_shape[dim_index] = axis.opt
            max_shape[dim_index] = axis.max
        profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
    return profile


def _axis_index(axis: DynamicAxis) -> int:
    if not axis.name.startswith("dim") or not axis.name[3:].isdigit():
        raise ValueError(f"Dynamic axis name {axis.name!r} must be of the form 'dimN'")
    return int(axis.name[3:])
