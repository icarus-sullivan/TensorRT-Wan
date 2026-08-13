"""Self-contained TensorRT-accelerated Wan VAE encode/decode for ComfyUI.

Drag-and-drop: copy this one file into any `custom_nodes/*/` package's node list. No dependency
on anything else in this repo — only `torch`, `tensorrt`, `requests`, and ComfyUI's own `comfy`/
`folder_paths` modules (all already present in a ComfyUI install).

What it does automatically, so the user never has to:
  - Checks `ComfyUI/models/vae/<file>` for the checkpoint; downloads it from HuggingFace if missing.
  - Builds a TensorRT engine the first time a given (component, frame-count) combo is requested,
    then caches it under `ComfyUI/models/tensorrt/vae/` and reuses it on every later call.
  - One engine per component covers a *wide range of resolutions* (256-1088px) via a TensorRT
    dynamic-shape profile -- arbitrary width/height within that range needs no rebuild.

What it can NOT do automatically, and why: the frame-count (T) axis cannot be made genuinely
dynamic. Wan's VAE runs a data-dependent chunked causal-conv loop internally
(`WanVAE.encode`/`.decode`); `torch.export` bakes the loop's trip count in as a constant at trace
time, so an engine built for T=1 only ever accepts T=1. Each distinct frame count you actually
request gets its own engine, built once and cached from then on -- same "build on first use, then
reuse" idea as the resolution envelope, just keyed on a dimension that can't be folded into one
profile.

wan2.1 vs wan2.2 checkpoints are different VAE architectures (16-channel vs 48-channel latent) --
never inferred from a DiT checkpoint name. Default is `wan_2.1_vae.safetensors`, which is correct
for both Wan 2.1 and Wan 2.2's 14B I2V models; `wan2.2_vae.safetensors` (48ch) is only correct for
Wan 2.2's separate 5B TI2V model and is offered as an explicit second dropdown option, not a guess.

Known tradeoff, inherited from this project's own prior investigation: TensorRT-accelerating the
VAE is comparatively low-value (it's cheap next to a DiT denoise loop) and comparatively risky (a
from-scratch reimplementation is more likely to have a subtle correctness bug than to need the
speedup). Every TensorRT call here therefore falls back to eager PyTorch on failure instead of
crashing generation.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Any

import torch

CATEGORY = "TensorRT-RT/VAE"

# --------------------------------------------------------------------------------------------
# Known checkpoints. Verified live against the HuggingFace API (not recalled) before writing this.
# --------------------------------------------------------------------------------------------

DEFAULT_VAE_FILENAME = "wan_2.1_vae.safetensors"

VAE_SOURCES: dict[str, dict[str, Any]] = {
    "wan_2.1_vae.safetensors": {
        "url": (
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/"
            "split_files/vae/wan_2.1_vae.safetensors"
        ),
        "latent_channels": 16,
    },
    "wan2.2_vae.safetensors": {
        "url": (
            "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/"
            "split_files/vae/wan2.2_vae.safetensors"
        ),
        "latent_channels": 48,
    },
}

# Pixel-space (encoder) and latent-space (decoder) H/W envelopes. Narrower than a naive "as wide
# as possible" choice on purpose: a wider range (tried [256, 1280]px) previously OOM'd (~100GB) at
# TensorRT execution-context creation time -- the context sizes scratch memory for the profile's
# worst-case bound, not the actual runtime shape. [256, 1088]px / [32, 136]-latent covers both real
# target shapes (480x832, 720x1088) and is the confirmed-safe bound. Don't widen without retesting
# on a GPU.
ENCODER_HEIGHT = (256, 480, 1088)  # (min, opt, max), pixels
ENCODER_WIDTH = (256, 832, 1088)
DECODER_LATENT_HEIGHT = (32, 60, 136)
DECODER_LATENT_WIDTH = (32, 104, 136)

DEFAULT_PRECISION = "fp16"
_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
_TRT_FLOAT_DTYPE_NAMES = {"fp16": "HALF", "fp32": "FLOAT"}


def _require_tensorrt() -> "Any":
    """Import `tensorrt`, or raise a message that tells the user exactly what to do instead of a
    bare `ModuleNotFoundError`. No auto-install: `tensorrt` is a compiled extension, so installing
    it into an already-running ComfyUI process wouldn't make it importable here anyway -- a
    restart is unavoidable, so there's nothing to gain from doing this silently versus just
    telling the user up front."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not installed in this ComfyUI's Python environment. Install it with "
            "`pip install tensorrt-cu12` (matching this environment's CUDA/torch build), then "
            "restart ComfyUI -- tensorrt is a compiled extension, so a running process can't pick "
            "up a newly-installed copy. Until then, VAE encode/decode falls back to eager PyTorch "
            "automatically; RIFE interpolation has no fallback and will fail until this is fixed."
        ) from exc
    return trt


def _models_dir(*parts: str) -> Path:
    import folder_paths

    path = Path(folder_paths.models_dir, *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _free_comfy_vram() -> None:
    """Ask ComfyUI to unload its own dynamically-managed models (e.g. a still-resident DiT) before
    a heavy GPU allocation of ours (eager model load, TensorRT engine build). Without this, our
    allocations are invisible to ComfyUI's dynamic VRAM system -- it has no reason to evict
    anything on our behalf, so we can collide with a model it's still holding and OOM instead of
    either side backing off, which is exactly what that system exists to prevent. Best-effort:
    never blocks the actual load/build on this succeeding."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"[TensorRT-RT VAE] could not free ComfyUI VRAM before a heavy load ({exc}); continuing anyway.")


def _available_vae_names() -> list[str]:
    import folder_paths

    names = set(VAE_SOURCES) | set(folder_paths.get_filename_list("vae"))
    return sorted(names, key=lambda n: (n != DEFAULT_VAE_FILENAME, n))


def _ensure_vae_checkpoint(filename: str) -> Path:
    """Return a local path to `filename` under ComfyUI's `models/vae/`, downloading it first if
    it isn't there yet. Mirrors ComfyUI-Rife-Tensorrt's `download_file` pattern: plain streamed
    HTTP GET, no `huggingface_hub` dependency, so this file stays dependency-light."""
    import folder_paths

    try:
        return Path(folder_paths.get_full_path_or_raise("vae", filename))
    except FileNotFoundError:
        pass

    if filename not in VAE_SOURCES:
        raise FileNotFoundError(
            f"{filename!r} not found under ComfyUI's models/vae/ and no download source is known "
            f"for it. Known sources: {sorted(VAE_SOURCES)}"
        )

    import requests

    vae_dirs = folder_paths.get_folder_paths("vae")
    dest_dir = Path(vae_dirs[0]) if vae_dirs else _models_dir("vae")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    tmp = dest.with_suffix(dest.suffix + ".part")

    url = VAE_SOURCES[filename]["url"]
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


# --------------------------------------------------------------------------------------------
# Model loading + export prerequisites, ported from this project's own
# examples/loaders/wan_comfyui_loader.py (real, hard-won findings -- not re-derived here).
# --------------------------------------------------------------------------------------------


def _decomposed_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Reference matmul+softmax+matmul attention. TensorRT 11.2's importer for ONNX opset-23's
    native `Attention` op fails (`MyelinCheckException`) for the VAE's bottleneck self-attention;
    decomposing before export sidesteps the native op so the graph is plain MatMul/Softmax nodes
    TensorRT's older fused-MHA pass can still recognize."""
    if enable_gqa:
        n_rep = query.shape[-3] // key.shape[-3]
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=-3)
            value = value.repeat_interleave(n_rep, dim=-3)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    attn_weight = query @ key.transpose(-2, -1) * scale
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        causal_mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device).tril()
        attn_weight = attn_weight.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


@contextlib.contextmanager
def _export_patches():
    """Process-wide monkeypatches, active ONLY for the duration of a `torch.export.export` trace
    -- must be scoped this tightly in a long-running ComfyUI server, not left on after the model
    load that precedes tracing. Confirmed as a real bug: leaving `scaled_dot_product_attention`
    patched process-wide after load would silently degrade every *other* model's eager attention
    (e.g. the DiT sampler's) to the slow decomposed path for the rest of the server's lifetime, and
    leaving the cuDNN-conv workaround disabled after load would reintroduce the real memory bug it
    exists to avoid for every other eager conv call, not just this trace.

    - `scaled_dot_product_attention` -> decomposed matmul+softmax+matmul: TensorRT's importer for
      ONNX opset-23's native `Attention` op fails for this VAE's bottleneck self-attention.
    - `comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND` -> disabled: it calls `torch.cudnn_convolution`
      directly, which has no FakeTensor/meta kernel, so torch.export can't trace it. Non-strict
      torch.export never runs a real cuDNN kernel anyway, so disabling this is safe for a trace.
    """
    import comfy.ops

    prior_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _decomposed_sdpa

    had_workaround_flag = hasattr(comfy.ops, "NVIDIA_MEMORY_CONV_BUG_WORKAROUND")
    if had_workaround_flag:
        prior_workaround = comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND
        comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False

    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = prior_sdpa
        if had_workaround_flag:
            comfy.ops.NVIDIA_MEMORY_CONV_BUG_WORKAROUND = prior_workaround


class _VAEEncodeWrapper(torch.nn.Module):
    """`WanVAE` exposes `.encode(x)`/`.decode(z)`, not a single `forward` -- `torch.export`
    traces `forward`, so each direction needs its own thin wrapper."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels)


class _VAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)


def _add_comfyui_to_path() -> None:
    import sys

    comfyui_root = os.environ.get("COMFYUI_ROOT")
    if comfyui_root and comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)


def _load_wan_vae(checkpoint_path: Path, dtype: torch.dtype, device: str = "cuda") -> torch.nn.Module:
    """Load the Wan VAE's `first_stage_model` (has `.encode`/`.decode`), on `device`/`dtype`,
    ready to either run eagerly (fallback path) or be wrapped + traced for export.

    Deliberately does NOT apply `_export_patches()` -- those monkeypatches must only be active
    during an actual `torch.export.export` trace (see its docstring), not during a normal load,
    and the eager fallback path calls this same loader but wants the real (fused, fast) attention
    kernel and the real cuDNN workaround, not the export-only decomposed/disabled versions."""
    _add_comfyui_to_path()

    import comfy.sd
    import comfy.utils

    sd, metadata = comfy.utils.load_torch_file(str(checkpoint_path), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    first_stage_model = vae.first_stage_model.to(device=device, dtype=dtype)
    first_stage_model.eval()
    return first_stage_model


# --------------------------------------------------------------------------------------------
# ONNX export + TensorRT build, ported from tensorrt_wan/export/{torch_export,onnx_export,
# trt_build}.py. Strongly-typed network: this TensorRT version has no BuilderFlag.FP16/BF16 for
# STRONGLY_TYPED networks, so precision comes entirely from the ONNX graph's own tensor dtypes --
# which is why the model is cast to `dtype` *before* tracing, not coerced by the builder after.
# --------------------------------------------------------------------------------------------


def _export_onnx(
    model: torch.nn.Module,
    example_inputs: dict[str, torch.Tensor],
    input_name: str,
    output_name: str,
    dynamic_dims: dict[int, object],
    onnx_path: Path,
    opset: int = 23,
) -> None:
    # Dim.AUTO, not an explicit Dim(min=, max=) range: torch.export's range assertion fails on
    # patch-alignment-style guards that only hold for specific values, not a smooth range (a real
    # finding against this project's DiT export -- same exporter code path). Dim.AUTO lets
    # torch.export infer what's actually dynamic from the trace instead.
    dynamic_shapes = {input_name: {dim: torch.export.Dim.AUTO for dim in dynamic_dims} or None}
    with _export_patches():
        exported = torch.export.export(model, args=(), kwargs=example_inputs, dynamic_shapes=dynamic_shapes)
    onnx_program = torch.onnx.export(
        exported,
        (),
        kwargs=example_inputs,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamo=True,
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_program.save(str(onnx_path))


def _assert_precision(network: "Any", precision: str) -> None:
    """Confirm the parsed ONNX graph's float I/O tensors actually match `precision` -- catches a
    real, previously-undetected bug class where one internal op (e.g. a stability-motivated fp32
    LayerNorm) silently leaves an unexpected-dtype tensor in an otherwise-uniform graph."""
    trt = _require_tensorrt()

    expected = getattr(trt.DataType, _TRT_FLOAT_DTYPE_NAMES[precision])
    float_dtypes = {getattr(trt.DataType, name) for name in _TRT_FLOAT_DTYPE_NAMES.values()}
    tensors = [network.get_input(i) for i in range(network.num_inputs)]
    tensors += [network.get_output(i) for i in range(network.num_outputs)]
    for tensor in tensors:
        if tensor.dtype not in float_dtypes or tensor.dtype == expected:
            continue
        raise RuntimeError(
            f"Requested precision={precision!r} but ONNX tensor {tensor.name!r} is {tensor.dtype}. "
            "STRONGLY_TYPED networks take precision entirely from the ONNX graph's own dtypes."
        )


def _build_trt_engine(
    onnx_path: Path,
    input_name: str,
    profile_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    precision: str,
    engine_path: Path,
) -> None:
    trt = _require_tensorrt()

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX model {onnx_path}:\n{errors}")

    _assert_precision(network, precision)

    config = builder.create_builder_config()
    # Build-time cost is almost entirely tactic search; level 5 (max) is the default here on
    # purpose -- this is a cached, build-once artifact, never trade final inference speed for a
    # faster build in a real deployment. Override only for fast debug-iteration builds.
    config.builder_optimization_level = int(os.environ.get("TRTWAN_BUILDER_OPT_LEVEL", "5"))

    profile = builder.create_optimization_profile()
    min_shape, opt_shape, max_shape = profile_shapes
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


def _engine_filename(component: str, checkpoint_name: str, precision: str, frames: int) -> str:
    trt = _require_tensorrt()

    stem = checkpoint_name.rsplit(".", 1)[0]
    raw = f"{component}|{checkpoint_name}|{precision}|frames={frames}|trt={trt.__version__}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{component}_{stem}_{precision}_t{frames}_{digest}.engine"


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
    """Deserialize once, bind tensors, execute_async_v3, sync. Ported from
    tensorrt_wan/engine/base.py's `TensorRTEngineWrapper` -- keeps its two real-bug-motivated
    fixes: cast every input to the engine's own declared dtype (set_tensor_address hands TensorRT
    a raw pointer with no dtype conversion -- a mismatched dtype silently reinterprets the same
    bytes rather than erroring), and check `set_input_shape`'s return value (it returns a bool
    rather than raising on a rank/bounds mismatch)."""

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
                    inputs[name] = tensor  # keep contiguous copy alive until execution completes
                else:
                    shape = tuple(context.get_tensor_shape(name))
                    dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
                    out = torch.empty(shape, dtype=dtype, device=self.device)
                    context.set_tensor_address(name, out.data_ptr())
                    outputs[name] = out
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


class _VAERuntime:
    """Owns a lazily-loaded eager Wan VAE module (used both as the export-trace source and as the
    eager-PyTorch fallback) plus a lazily-built/cached dict of TensorRT engines, one per
    (component, frame-count) actually requested. Nothing touches the GPU until the first actual
    `encode()`/`decode()` call -- the loader node just records the checkpoint path/precision, it
    doesn't load anything. Loading eagerly at loader-node time was a real bug: ComfyUI's node
    graph can execute this node while a DiT model is still mid-load/staging, and an unmanaged
    ~250MB-2GB CUDA allocation landing in the middle of that measurably contributed to an OOM."""

    def __init__(self, checkpoint_path: Path, precision: str = DEFAULT_PRECISION) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_path.name
        self.precision = precision
        self.dtype = _DTYPES[precision]
        self.device = torch.device("cuda")
        self.model: torch.nn.Module | None = None
        self._runners: dict[tuple[str, int], _TensorRTRunner] = {}

    def _ensure_model(self) -> torch.nn.Module:
        if self.model is None:
            _free_comfy_vram()
            self.model = _load_wan_vae(self.checkpoint_path, self.dtype, device="cuda")
        return self.model

    def _cache_dir(self) -> Path:
        return _models_dir("tensorrt", "vae")

    def _onnx_dir(self) -> Path:
        return _models_dir("onnx", "vae")

    def _get_or_build(
        self,
        component: str,
        wrapper_cls: type,
        input_name: str,
        output_name: str,
        example: torch.Tensor,
        dynamic_dims: dict[int, object],
        profile_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        frames: int,
    ) -> _TensorRTRunner:
        key = (component, frames)
        if key in self._runners:
            return self._runners[key]

        engine_path = self._cache_dir() / _engine_filename(component, self.checkpoint_name, self.precision, frames)
        if not engine_path.exists():
            # Cache-miss only: torch.export + the TensorRT builder's tactic search is the single
            # heaviest GPU allocation this file makes, easily enough on its own to OOM alongside a
            # still-resident DiT -- free ComfyUI's VRAM before it, not just before the eager load.
            _free_comfy_vram()
            onnx_path = self._onnx_dir() / engine_path.with_suffix(".onnx").name
            wrapped = wrapper_cls(self._ensure_model())
            _export_onnx(wrapped, {input_name: example}, input_name, output_name, dynamic_dims, onnx_path)
            _build_trt_engine(onnx_path, input_name, profile_shapes, self.precision, engine_path)

        runner = _TensorRTRunner(engine_path, self.device)
        runner.load()
        self._runners[key] = runner
        return runner

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """`pixels`: (B, 3, T, H, W) in [-1, 1]. Returns latent (B, C, T', H/8, W/8)."""
        frames = pixels.shape[2]
        h_min, h_opt, h_max = ENCODER_HEIGHT
        w_min, w_opt, w_max = ENCODER_WIDTH
        example = torch.zeros(1, 3, frames, h_opt, w_opt, device=self.device, dtype=self.dtype)
        profile_shapes = (
            (1, 3, frames, h_min, w_min),
            (1, 3, frames, h_opt, w_opt),
            (1, 3, frames, h_max, w_max),
        )
        try:
            runner = self._get_or_build(
                "vae_encoder", _VAEEncodeWrapper, "pixels", "latent", example, {3: None, 4: None},
                profile_shapes, frames,
            )
            return runner.infer({"pixels": pixels})["latent"]
        except Exception as exc:  # noqa: BLE001 -- deliberate TRT -> eager fallback, log + degrade
            print(f"[TensorRT-RT VAE] encode via TensorRT failed ({exc}); falling back to eager PyTorch.")
            with torch.no_grad():
                return self._ensure_model().encode(pixels.to(self.device, dtype=self.dtype))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """`latent`: (B, C, T, H, W). Returns pixels (B, 3, T, H*8, W*8) in [-1, 1]."""
        frames = latent.shape[2]
        h_min, h_opt, h_max = DECODER_LATENT_HEIGHT
        w_min, w_opt, w_max = DECODER_LATENT_WIDTH
        channels = VAE_SOURCES.get(self.checkpoint_name, {}).get("latent_channels", latent.shape[1])
        example = torch.zeros(1, channels, frames, h_opt, w_opt, device=self.device, dtype=self.dtype)
        profile_shapes = (
            (1, channels, frames, h_min, w_min),
            (1, channels, frames, h_opt, w_opt),
            (1, channels, frames, h_max, w_max),
        )
        try:
            runner = self._get_or_build(
                "vae_decoder", _VAEDecodeWrapper, "latent", "pixels", example, {3: None, 4: None},
                profile_shapes, frames,
            )
            return runner.infer({"latent": latent})["pixels"]
        except Exception as exc:  # noqa: BLE001
            print(f"[TensorRT-RT VAE] decode via TensorRT failed ({exc}); falling back to eager PyTorch.")
            with torch.no_grad():
                return self._ensure_model().decode(latent.to(self.device, dtype=self.dtype))


# --------------------------------------------------------------------------------------------
# ComfyUI nodes
# --------------------------------------------------------------------------------------------


class TensorRTWanVAELoader:
    """Resolves (downloading if needed) a Wan VAE checkpoint and wraps it in a `_VAERuntime` that
    builds/caches TensorRT engines on first use. Output feeds `TensorRTWanVAEEncode`/`Decode`."""

    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("TRTWAN_VAE_RT",)
    RETURN_NAMES = ("vae_rt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (_available_vae_names(), {"default": DEFAULT_VAE_FILENAME}),
                "precision": (list(_DTYPES), {"default": DEFAULT_PRECISION}),
            }
        }

    def load(self, vae_name: str, precision: str):
        checkpoint_path = _ensure_vae_checkpoint(vae_name)
        return (_VAERuntime(checkpoint_path, precision=precision),)


class TensorRTWanVAEEncode:
    """IMAGE -> LATENT. An IMAGE batch of N frames is treated as one video's T=N frames (matches
    how TensorRTWanVAEDecode flattens (B,T) back into an IMAGE batch on the way out)."""

    CATEGORY = CATEGORY
    FUNCTION = "encode"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_rt": ("TRTWAN_VAE_RT",), "image": ("IMAGE",)}}

    def encode(self, vae_rt: _VAERuntime, image: torch.Tensor):
        # ComfyUI IMAGE is (N, H, W, C) float in [0, 1]; Wan VAE wants (B, 3, T, H, W) in [-1, 1].
        pixels = image.permute(3, 0, 1, 2).unsqueeze(0).contiguous() * 2.0 - 1.0
        latent = vae_rt.encode(pixels)
        return ({"samples": latent},)


class TensorRTWanVAEDecode:
    """LATENT -> IMAGE. Drop-in for a stock VAEDecode."""

    CATEGORY = CATEGORY
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_rt": ("TRTWAN_VAE_RT",), "samples": ("LATENT",)}}

    def decode(self, vae_rt: _VAERuntime, samples: dict):
        pixels = vae_rt.decode(samples["samples"])  # (B, 3, T, H, W) in [-1, 1]
        frames = (pixels.clamp(-1, 1) + 1.0) / 2.0
        b, c, t, h, w = frames.shape
        frames = frames.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).contiguous()
        return (frames,)


NODE_CLASS_MAPPINGS = {
    "TensorRTWanVAELoader": TensorRTWanVAELoader,
    "TensorRTWanVAEEncode": TensorRTWanVAEEncode,
    "TensorRTWanVAEDecode": TensorRTWanVAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTWanVAELoader": "TensorRT Wan VAE Loader",
    "TensorRTWanVAEEncode": "TensorRT Wan VAE Encode",
    "TensorRTWanVAEDecode": "TensorRT Wan VAE Decode",
}
