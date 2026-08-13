"""Stage 3: ONNX -> serialized TensorRT engine.

Builds one optimization profile per `ResolutionProfile` the caller supplies, so a single engine
covers every configured resolution rather than needing one engine per shape.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tensorrt_wan.config.schema import PrecisionMode, ResolutionProfile
from tensorrt_wan.export.base import DynamicAxis, ModelExporter
from tensorrt_wan.lora import onnx_weight_names
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
    timing_cache_path: str | Path | None = None,
) -> "trt.IHostMemory":
    """Parse `onnx_path` and build a serialized engine covering `resolution_profiles`.

    Returns the raw serialized engine (a buffer-protocol object, not a `bytes` copy of it -- see
    the note at the return statement); writing it to `EngineCache` is the caller's job (see
    `cli.commands.build`) so this function stays agnostic of cache-key/invalidation logic.

    `timing_cache_path`, if given, persists TensorRT's per-layer tactic-timing results across
    builds -- the dominant cost of a build is re-benchmarking candidate kernels per layer, and
    that search is keyed on (layer config, GPU), not on the actual weight values. So rebuilding
    the *same* architecture/shapes on the *same* GPU (e.g. iterating on the REFIT flag, or
    building the high-noise and low-noise DiT experts back to back) can reuse prior timings
    almost entirely instead of re-searching from scratch. Never affects correctness -- a stale or
    mismatched cache just means more cache misses (falls back to a real search per layer), not a
    wrong engine.
    """
    import time

    import tensorrt as trt

    phase_start = time.monotonic()

    onnx_path = Path(onnx_path)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX model {onnx_path}:\n{errors}")

    logger.info("Phase timing: ONNX parse took %.1fs", time.monotonic() - phase_start)
    phase_start = time.monotonic()

    _validate_precision(network, precision)

    config = builder.create_builder_config()
    if workspace_limit_mb is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_limit_mb * (1 << 20))

    # TensorRT's build-time cost is almost entirely the CPU-orchestrated tactic search (timing
    # candidate kernel implementations per layer on the GPU, then picking the fastest) -- there's
    # no way to make that GPU-only, but `builder_optimization_level` (0-5) and `max_num_tactics`
    # control how exhaustive that search is. `TRTWAN_BUILD_MODE` selects a preset: `fast`/`balanced`
    # trade final inference speed for much faster iteration (the right tradeoff while debugging,
    # e.g. bisecting the REFIT investigation), `production` (default) stays at level 5 -- this
    # project never wants a silently less-optimized deployment engine. `TRTWAN_BUILDER_OPT_LEVEL`
    # remains available as a one-off manual override of the `production` level specifically.
    build_mode = os.environ.get("TRTWAN_BUILD_MODE", "production")
    if build_mode == "fast":
        config.builder_optimization_level = 1
        config.max_num_tactics = 4
    elif build_mode == "balanced":
        config.builder_optimization_level = 3
        config.max_num_tactics = 16
    elif build_mode == "production":
        config.builder_optimization_level = int(os.environ.get("TRTWAN_BUILDER_OPT_LEVEL", "5"))
    else:
        raise ValueError(f"Unknown TRTWAN_BUILD_MODE={build_mode!r}; want fast|balanced|production")

    # Off by default -- REFIT_INDIVIDUAL (not plain REFIT) marks only the specific weights we name
    # via `network.mark_weights_refittable()`, leaving everything else free to fuse/optimize
    # normally. Confirmed against NVIDIA's docs: plain `REFIT` marks *every* weight refittable and
    # is documented to break more fusions than necessary (e.g. Conv-GELU coefficient fusion) for
    # no benefit here, since LoRA only ever touches attention/FFN weights, never biases or norms.
    # `REFIT_IDENTICAL` is a different, incompatible use case (assumes refit values equal build-time
    # values -- undefined behavior otherwise) and must never be used for real LoRA deltas. See
    # docs/wan2.2_i2v_14b_notes.md's Refit-API entries for the full investigation.
    if os.environ.get("TRTWAN_ENABLE_REFIT", "0") == "1":
        config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
        target_names = onnx_weight_names(onnx_path)
        failed = [name for name in target_names if not network.mark_weights_refittable(name)]
        if failed:
            raise RuntimeError(
                f"mark_weights_refittable failed for {len(failed)}/{len(target_names)} names "
                f"(ONNX initializer name changed vs the graph-walk assumption?): {failed[:10]}..."
            )
        logger.info("Marked %d weights individually refittable for LoRA", len(target_names))

    timing_cache_path = Path(timing_cache_path) if timing_cache_path is not None else None
    if timing_cache_path is not None:
        cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.exists() else b""
        timing_cache = config.create_timing_cache(cache_bytes)
        # ignore_mismatch=True: the header check is against recorded CUDA device properties, and
        # a mismatch (e.g. cache carried over from a different pod) should just mean more cache
        # misses, not a hard failure -- this is a build-speed optimization, never load-bearing.
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    for profile_spec in resolution_profiles:
        config.add_optimization_profile(_build_optimization_profile(builder, exporter, profile_spec))

    logger.info(
        "Building TensorRT engine for %s: %d profile(s), precision=%s (setup phase took %.1fs)",
        exporter.name,
        len(resolution_profiles),
        precision,
        time.monotonic() - phase_start,
    )
    phase_start = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {exporter.name}")
    logger.info("Phase timing: build_serialized_network (tactic search) took %.1fs", time.monotonic() - phase_start)

    if timing_cache_path is not None:
        phase_start = time.monotonic()
        updated_cache = config.get_timing_cache()
        if updated_cache is not None:
            # No bytes(...) copy -- see the note on the return value below, same reasoning.
            timing_cache_path.write_bytes(updated_cache.serialize())
            logger.info(
                "Updated timing cache: %s (took %.1fs)", timing_cache_path, time.monotonic() - phase_start
            )

    # Confirmed on real hardware tonight: `bytes(serialized)` alone was costing ~22 minutes on a
    # 28.6GB engine (measured directly -- the process sat at ~110% CPU, effectively one core, with
    # 0% GPU for that whole span), on top of a build whose actual tactic search took ~21-24
    # minutes. `IHostMemory` implements the buffer protocol (`nbytes`, `__buffer__` -- confirmed
    # via introspection) and `Path.write_bytes()` already wraps its argument in `memoryview()`
    # before writing (confirmed by reading its source), so passing `serialized` straight through
    # avoids that entire redundant 28GB copy -- verified byte-identical against the old
    # `bytes(serialized)` path via a real dummy-engine round-trip + sha256 comparison before this
    # was applied to the real (expensive) build path. Callers (`cli.commands.build`,
    # `export.pipeline`) only ever pass this straight into `EngineCache.put()` -> `write_bytes()`,
    # never slice or hash it themselves, so nothing downstream needs an actual `bytes` instance.
    return serialized


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

    Only floating-point tensors are checked. Confirmed necessary against a real build: the text
    encoder's `input_ids`/`attention_mask` are legitimately `INT64` token/mask tensors — Wan never
    casts those to fp16 (nor should it; they're indices and a 0/1 mask, not activations), so
    enforcing `precision` against them isn't "an upstream export bug" the way a wrong-dtype
    *activation* tensor would be, it's just comparing the wrong kind of value. Non-float dtypes
    (int/bool) are skipped entirely rather than being held to a float precision they were never
    meant to have.

    Checks *outputs* too, not just inputs — added after a real, previously-undetected bug
    (2026-08-07): the text encoder's `text_embeds` output came out `DataType.FLOAT` (fp32) despite
    every *input* being correctly fp16 and this function only ever having checked inputs. A model
    implementation detail (an internal op — plausibly a stability-motivated fp32 LayerNorm, the
    same category of thing `patch_embedding` needed explicit handling for elsewhere) had silently
    produced an fp32 graph output that nothing caught, corrupting every downstream consumer that
    assumed uniform fp16. See docs/wan2.2_i2v_14b_notes.md.
    """
    import tensorrt as trt

    expected = getattr(trt.DataType, _PRECISION_TO_DTYPE_NAME[precision])
    float_dtypes = {getattr(trt.DataType, name) for name in _PRECISION_TO_DTYPE_NAME.values()}
    tensors = [network.get_input(i) for i in range(network.num_inputs)]
    tensors += [network.get_output(i) for i in range(network.num_outputs)]
    for tensor in tensors:
        if tensor.dtype not in float_dtypes:
            continue
        if tensor.dtype != expected:
            raise RuntimeError(
                f"Requested precision={precision!r} ({expected}) but ONNX tensor {tensor.name!r} "
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
