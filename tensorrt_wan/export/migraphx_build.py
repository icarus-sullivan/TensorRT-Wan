"""Stage 3 (MIGraphX target): ONNX -> a compiled onnxruntime InferenceSession backed by AMD's
MIGraphX execution provider, the ROCm-side counterpart to trt_build.py's TensorRT builder.

Unlike TensorRT's `build_serialized_network()`, ONNX Runtime's MIGraphX EP compiles at
`InferenceSession` construction time and (per the official EP docs fetched 2026-09) exposes no
confirmed API for serializing that compiled program to a portable blob the way TensorRT's
`IHostMemory` does -- so this stage caches the *static* ONNX file itself (already deterministic
per resolution profile, see `DiTExporter`'s `static=True`) rather than a fabricated "compiled
engine" artifact. `engine/migraphx_engine.py` recompiles from this cached ONNX file at load time.
Revisit once running against real hardware: MIGraphX/onnxruntime-rocm may expose a faster
AOT-cache provider option (`migraphx_exhaustive_tune` is confirmed real; a dedicated
compiled-model cache path is not confirmed either way) -- see docs/rocm_setup.md.
"""

from __future__ import annotations

from pathlib import Path

from tensorrt_wan.config.schema import PrecisionMode
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)

# migraphx_fp8_enable requires HIP 6.4+ per AMD's EP docs; fp32/auto have no corresponding
# provider-option equivalent and aren't meaningful DiT precisions on this project anyway (the DiT
# always exports bf16 regardless of runtime/precision.py's general architecture table -- see
# export/exporters/dit.py's `dtype` override docstring).
PRECISION_TO_MIGRAPHX_FLAG = {
    "fp16": "migraphx_fp16_enable",
    "bf16": "migraphx_bf16_enable",
    "fp8": "migraphx_fp8_enable",
}


def build_migraphx_program(onnx_path: str | Path, precision: PrecisionMode) -> bytes:
    """Validate `onnx_path` (a *static*-shape export -- MIGraphX's dynamic-shape support is
    inconsistent across ops, see `DiTExporter`'s `static=True`) compiles under the MIGraphX
    execution provider, then return the ONNX file's own bytes for `EngineCache.put()` -- see this
    module's docstring for why that's the cached artifact instead of a compiled-program blob.
    """
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if precision not in PRECISION_TO_MIGRAPHX_FLAG:
        raise ValueError(
            f"precision={precision!r} has no MIGraphX provider-option equivalent; "
            f"want one of {sorted(PRECISION_TO_MIGRAPHX_FLAG)}"
        )

    provider_options = {PRECISION_TO_MIGRAPHX_FLAG[precision]: True, "migraphx_exhaustive_tune": True}
    logger.info("Compiling %s under MIGraphXExecutionProvider (precision=%s)", onnx_path, precision)
    session = ort.InferenceSession(str(onnx_path), providers=[("MIGraphXExecutionProvider", provider_options)])

    # onnxruntime silently falls back to a CPU/default EP if the requested one fails to load the
    # graph rather than raising -- confirmed ORT behavior, not specific to MIGraphX. Check
    # explicitly rather than let a fallback masquerade as a successful MIGraphX build; the actual
    # reason (usually an unsupported op) is in onnxruntime's own log output above this.
    if "MIGraphXExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            f"onnxruntime did not select MIGraphXExecutionProvider for {onnx_path} "
            f"(got {session.get_providers()}) -- the graph likely has an op MIGraphX can't "
            "import; check the onnxruntime log above for the actual reason."
        )
    del session  # only used here to validate the graph compiles; see module docstring

    return onnx_path.read_bytes()
