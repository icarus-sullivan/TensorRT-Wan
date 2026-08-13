"""Self-contained TensorRT-accelerated RIFE frame interpolation for ComfyUI.

Drag-and-drop: copy this one file into any `custom_nodes/*/` package's node list. No dependency
on anything else in this repo -- only `torch`, `tensorrt`, `requests`, and ComfyUI's own
`folder_paths` module.

Modeled directly on github.com/yuvraj108c/ComfyUI-Rife-Tensorrt's fetch/cache/build strategy
(verified against its actual source, not recalled): pretrained RIFE ONNX models are downloaded
from HuggingFace on first use, a TensorRT engine is built from that ONNX the first time a given
(model, precision) combo is requested, and both are cached under ComfyUI's `models/` tree and
reused after. One engine covers a *wide range of resolutions* (256px - 3840px, the same envelope
upstream Rife-TRT uses) via a TensorRT dynamic-shape profile, so arbitrary frame size needs no
rebuild -- unlike the VAE nodes in this package, RIFE has no frame-count axis to worry about (it
always operates on exactly two input frames plus one interpolation timestep).

Deviates from upstream Rife-TRT in one place: engine building uses the raw `tensorrt` Python API
directly (matching this package's own vae_rt.py) rather than upstream's Polygraphy dependency --
same BuilderFlag.FP16 strategy (the downloaded ONNX is fp32; unlike the VAE's self-exported graph,
there's no opportunity to bake precision into the ONNX itself before building), just one fewer
third-party dependency to keep this file portable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

CATEGORY = "TensorRT-RT/RIFE"

RIFE_MODELS = [
    "rife47_ensemble_True_scale_1_sim",
    "rife48_ensemble_True_scale_1_sim",
    "rife49_ensemble_True_scale_1_sim",
]
DEFAULT_RIFE_MODEL = "rife49_ensemble_True_scale_1_sim"
RIFE_ONNX_URL_TEMPLATE = "https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/{model}.onnx"

# Same envelope upstream ComfyUI-Rife-Tensorrt builds with -- one engine serves any resolution in
# this range, no rebuild per input size.
IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 512
IMAGE_DIM_MAX = 3840

DEFAULT_PRECISION = "fp32"  # ponytail: fp16 raises in _build_trt_engine until ONNX->fp16 conversion is added
_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def _require_tensorrt() -> "Any":
    """Import `tensorrt`, or raise a message that tells the user exactly what to do instead of a
    bare `ModuleNotFoundError`/`AttributeError`. No auto-install: `tensorrt` is a compiled
    extension, so installing it into an already-running ComfyUI process wouldn't make it
    importable here anyway -- a restart is unavoidable regardless."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not installed in this ComfyUI's Python environment. Install it with "
            "`pip install tensorrt-cu12` (matching this environment's CUDA/torch build), then "
            "restart ComfyUI -- tensorrt is a compiled extension, so a running process can't pick "
            "up a newly-installed copy. RIFE interpolation has no non-TensorRT fallback and will "
            "fail until this is fixed."
        ) from exc
    return trt


def _models_dir(*parts: str) -> Path:
    import folder_paths

    path = Path(folder_paths.models_dir, *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _free_comfy_vram() -> None:
    """Ask ComfyUI to unload its own dynamically-managed models (e.g. a still-resident DiT) before
    the TensorRT builder's tactic search -- easily the heaviest GPU allocation this file makes.
    Without this, our allocation is invisible to ComfyUI's dynamic VRAM system, which has no reason
    to evict anything on our behalf. Best-effort: never blocks the actual build on this succeeding."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"[TensorRT-RT RIFE] could not free ComfyUI VRAM before a heavy load ({exc}); continuing anyway.")


def _ensure_rife_onnx(model_name: str) -> Path:
    dest = _models_dir("onnx", "rife") / f"{model_name}.onnx"
    if dest.exists():
        return dest

    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    url = RIFE_ONNX_URL_TEMPLATE.format(model=model_name)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _engine_filename(model_name: str, precision: str) -> str:
    trt = _require_tensorrt()

    raw = (
        f"{model_name}|{precision}|{IMAGE_DIM_MIN}-{IMAGE_DIM_OPT}-{IMAGE_DIM_MAX}|trt={trt.__version__}"
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"rife_{model_name}_{precision}_{digest}.engine"


def _build_trt_engine(onnx_path: Path, precision: str, engine_path: Path) -> None:
    trt = _require_tensorrt()

    if precision != "fp32":
        # ponytail: fp32-only for now. Confirmed live against the installed TensorRT (11.2.1.2):
        # BuilderFlag has no FP16/INT8/BF16 member at all anymore -- precision comes purely from
        # the ONNX graph's own tensor dtypes, universally (not just for STRONGLY_TYPED networks,
        # which is what the older vae_rt.py comment assumed from stale docs). The downloaded RIFE
        # ONNX is natively fp32, and this file doesn't yet convert it -- upgrade path: convert the
        # ONNX to fp16 (e.g. onnxconverter_common.float16) before parsing here, then build with a
        # STRONGLY_TYPED network the same way vae_rt.py does.
        raise ValueError(f"precision={precision!r} not yet supported -- only 'fp32' is implemented")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX model {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.builder_optimization_level = int(os.environ.get("TRTWAN_BUILDER_OPT_LEVEL", "5"))

    profile = builder.create_optimization_profile()
    min_shape = (1, 3, IMAGE_DIM_MIN, IMAGE_DIM_MIN)
    opt_shape = (1, 3, IMAGE_DIM_OPT, IMAGE_DIM_OPT)
    max_shape = (1, 3, IMAGE_DIM_MAX, IMAGE_DIM_MAX)
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if input_tensor.name == "timestep":
            profile.set_shape(input_tensor.name, (1,), (1,), (1,))
        else:
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


def _trt_dtype_to_torch(dtype: "Any") -> torch.dtype:
    trt = _require_tensorrt()

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    return mapping[dtype]


class _TensorRTRunner:
    """Deserialize once, bind tensors, execute_async_v3, sync -- same shape as vae_rt.py's
    runner (duplicated rather than imported, so this file stays a standalone drop-in). Keeps the
    same two real-bug-motivated fixes: cast inputs to the engine's own declared dtype, and check
    `set_input_shape`'s return value instead of ignoring it."""

    def __init__(self, engine_path: Path, device: torch.device) -> None:
        self.engine_path = engine_path
        self.device = device
        self._engine = None
        self._context = None
        self._stream: torch.cuda.Stream | None = None

    def load(self) -> None:
        trt = _require_tensorrt()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine at {self.engine_path}")
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.device)

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, engine, stream = self._context, self._engine, self._stream
        if context is None or engine is None or stream is None:
            raise RuntimeError("Engine not loaded; call .load() first")

        outputs: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(stream):
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if name in inputs:
                    target_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
                    tensor = inputs[name].to(self.device, dtype=target_dtype, non_blocking=True).contiguous()
                    if not context.set_input_shape(name, tuple(tensor.shape)):
                        raise RuntimeError(
                            f"set_input_shape failed for input {name!r} with shape "
                            f"{tuple(tensor.shape)} against {self.engine_path} -- likely outside "
                            "the profile this engine was built with."
                        )
                    context.set_tensor_address(name, tensor.data_ptr())
                    inputs[name] = tensor
                else:
                    shape = tuple(context.get_tensor_shape(name))
                    dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
                    out = torch.empty(shape, dtype=dtype, device=self.device)
                    context.set_tensor_address(name, out.data_ptr())
                    outputs[name] = out
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


def _pad_to_multiple(frame: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, int, int]:
    """Pad `frame` (1, 3, H, W) on the right/bottom to the next multiple of `multiple`, returning
    the padded tensor plus the original (h, w) to crop back to after inference.

    Confirmed against the real RIFE architecture source (rife_arch.py): the original PyTorch
    model does exactly this internally (`ph = ((h-1)//64+1)*64`, `F.pad(img0, (0, pw-w, 0,
    ph-h))`, crop `result[..., :h, :w]` after) before running its internal downsample/upsample
    pyramid. That padding-amount computation is itself shape-dependent, so whether it's captured
    as genuinely dynamic in the pre-exported ONNX we download (rather than baked in against
    whatever shape it was traced with) isn't something we control -- an unaligned H/W silently
    misaligns the pyramid at one edge instead of erroring. Padding/cropping here ourselves,
    outside the engine, is correct regardless of what the ONNX graph's internal padding does."""
    h, w = frame.shape[-2:]
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    if ph == h and pw == w:
        return frame, h, w
    return torch.nn.functional.pad(frame, (0, pw - w, 0, ph - h)), h, w


class _RifeRuntime:
    """Owns a lazily-built/cached TensorRT engine for one (model, precision) combo."""

    def __init__(self, model_name: str, precision: str = DEFAULT_PRECISION) -> None:
        self.model_name = model_name
        self.precision = precision
        self.device = torch.device("cuda")
        self._runner: _TensorRTRunner | None = None

    def _ensure_runner(self) -> _TensorRTRunner:
        if self._runner is not None:
            return self._runner

        engine_path = _models_dir("tensorrt", "rife") / _engine_filename(self.model_name, self.precision)
        if not engine_path.exists():
            _free_comfy_vram()
            onnx_path = _ensure_rife_onnx(self.model_name)
            _build_trt_engine(onnx_path, self.precision, engine_path)

        runner = _TensorRTRunner(engine_path, self.device)
        runner.load()
        self._runner = runner
        return runner

    def infer_pair(self, frame_0: torch.Tensor, frame_1: torch.Tensor, timestep: float) -> torch.Tensor:
        """`frame_0`/`frame_1`: (1, 3, H, W) in [0, 1]. Returns the interpolated (1, 3, H, W) frame."""
        runner = self._ensure_runner()
        frame_0p, h, w = _pad_to_multiple(frame_0)
        frame_1p, _, _ = _pad_to_multiple(frame_1)
        timestep_t = torch.tensor([timestep], dtype=torch.float32)
        outputs = runner.infer({"img0": frame_0p, "img1": frame_1p, "timestep": timestep_t})
        return outputs["output"][:, :, :h, :w]


def _interpolate_batch(
    runtime: _RifeRuntime,
    frames: torch.Tensor,
    multiplier: int,
    clear_cache_after_n_frames: int,
) -> torch.Tensor:
    """`frames`: (N, 3, H, W) in [0, 1]. Inserts `multiplier - 1` interpolated frames between each
    adjacent input pair (RIFE's timestep input accepts any value in (0, 1), so each intermediate
    frame is generated directly at `k / multiplier` rather than via recursive bisection)."""
    n = frames.shape[0]
    out = [frames[0:1]]
    for i in range(n - 1):
        f0, f1 = frames[i : i + 1], frames[i + 1 : i + 2]
        for k in range(1, multiplier):
            out.append(runtime.infer_pair(f0, f1, k / multiplier))
        out.append(f1)
        if (i + 1) % clear_cache_after_n_frames == 0:
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


def _resample_fps(
    runtime: _RifeRuntime,
    frames: torch.Tensor,
    source_fps: float,
    target_fps: float,
    clear_cache_after_n_frames: int,
) -> torch.Tensor:
    """`frames`: (N, 3, H, W) in [0, 1], `source_fps` -> `target_fps`.

    Unlike `_interpolate_batch`'s fixed integer multiplier per gap, this maps each *output* frame's
    time position back into source-frame-index space and interpolates at whatever fractional
    timestep that lands on -- the ratio target_fps/source_fps is rarely an integer, so a fixed
    per-gap frame count can't hit arbitrary target rates (16fps -> 25fps needs a different number
    of inserted frames in different gaps, not a constant one)."""
    n = frames.shape[0]
    if n < 2 or source_fps <= 0 or target_fps <= 0:
        return frames

    span = n - 1  # source frame-index span covered by the clip
    out_count = max(2, round(span * target_fps / source_fps) + 1)

    out = []
    for j in range(out_count):
        src_pos = min(j * source_fps / target_fps, span)
        idx = min(int(src_pos), n - 2)
        frac = src_pos - idx
        if frac < 1e-4:
            out.append(frames[idx : idx + 1])
        elif frac > 1.0 - 1e-4:
            out.append(frames[idx + 1 : idx + 2])
        else:
            f0, f1 = frames[idx : idx + 1], frames[idx + 1 : idx + 2]
            out.append(runtime.infer_pair(f0, f1, frac))
        if (j + 1) % clear_cache_after_n_frames == 0:
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------------------------
# ComfyUI nodes
# --------------------------------------------------------------------------------------------


class TensorRTRifeLoader:
    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("TRTWAN_RIFE_RT",)
    RETURN_NAMES = ("rife_rt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (RIFE_MODELS, {"default": DEFAULT_RIFE_MODEL}),
                "precision": (list(_DTYPES), {"default": DEFAULT_PRECISION}),
            }
        }

    def load(self, model: str, precision: str):
        return (_RifeRuntime(model, precision=precision),)


class TensorRTRifeInterpolate:
    CATEGORY = CATEGORY
    FUNCTION = "vfi"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame interpolation"}),
                "rife_rt": ("TRTWAN_RIFE_RT",),
                "multiplier": ("INT", {"default": 2, "min": 2, "tooltip": "Output frames per input gap"}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000}),
            }
        }

    def vfi(self, frames: torch.Tensor, rife_rt: _RifeRuntime, multiplier: int, clear_cache_after_n_frames: int):
        # ComfyUI IMAGE is (N, H, W, C) float in [0, 1]; RIFE wants (N, 3, H, W).
        chw = frames.permute(0, 3, 1, 2).contiguous()
        result = _interpolate_batch(rife_rt, chw, multiplier, clear_cache_after_n_frames)
        out = result.permute(0, 2, 3, 1).contiguous()
        return (out,)


class TensorRTRifeResampleFPS:
    """Like `TensorRTRifeInterpolate`, but converts a fixed source frame rate to an arbitrary
    target frame rate (e.g. 16fps -> 25fps) instead of taking an integer per-gap multiplier."""

    CATEGORY = CATEGORY
    FUNCTION = "resample"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame rate conversion"}),
                "rife_rt": ("TRTWAN_RIFE_RT",),
                "source_fps": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000}),
            }
        }

    def resample(
        self,
        frames: torch.Tensor,
        rife_rt: _RifeRuntime,
        source_fps: float,
        target_fps: float,
        clear_cache_after_n_frames: int,
    ):
        chw = frames.permute(0, 3, 1, 2).contiguous()
        result = _resample_fps(rife_rt, chw, source_fps, target_fps, clear_cache_after_n_frames)
        out = result.permute(0, 2, 3, 1).contiguous()
        return (out,)


NODE_CLASS_MAPPINGS = {
    "TensorRTRifeLoader": TensorRTRifeLoader,
    "TensorRTRifeInterpolate": TensorRTRifeInterpolate,
    "TensorRTRifeResampleFPS": TensorRTRifeResampleFPS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTRifeLoader": "TensorRT RIFE Loader",
    "TensorRTRifeInterpolate": "TensorRT RIFE Interpolate",
    "TensorRTRifeResampleFPS": "TensorRT RIFE Resample FPS",
}
