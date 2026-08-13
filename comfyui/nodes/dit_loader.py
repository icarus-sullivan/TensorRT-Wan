"""TensorRT-accelerated DiT loader that outputs a real ComfyUI `MODEL` socket, not a project-
specific type.

Confirmed working 2026-08-08 (see docs/known_working/) as a standalone script
(`scripts/real_pipeline_trt_dit_test.py`) before being formalized here: real ComfyUI CLIP, VAE,
conditioning construction, and sampler produced a fully coherent I2V result with *only* the
diffusion model swapped for our TensorRT engine, after a full session investigating a "pure noise"
bug that turned out to live entirely in this project's own (now superseded) sampler/scheduler
reimplementation -- see docs/wan2.2_i2v_14b_notes.md's "BREAKTHROUGH" entry for the full story.

Deliberate scope: only the DiT is TensorRT-accelerated. It's the ~14B-param, 50-100-forward-pass
part of generation and dominates total cost; VAE encode/decode and text encoding are comparatively
cheap and, per this session's findings, far more likely to have subtle correctness bugs in a
from-scratch reimplementation than to need TensorRT's speedup. Use stock ComfyUI CLIPLoader/
VAELoader/CLIPTextEncode/VAEDecode/KSampler(Advanced) nodes around this loader, not this package's
other (now-superseded) TensorRTTextEncoder/TensorRTVAEEncoder/TensorRTVAEDecoder/TensorRTSampler/
TensorRTScheduler nodes -- those reimplement the whole pipeline with this project's own
`TRTWAN_*`-typed sockets, which is exactly the code that caused every bug this session. Not removed
from this package (still registered, still usable for whoever needs a fully-standalone non-ComfyUI
pipeline via `tensorrt_wan.api.WanEngine`), just no longer the recommended path for a real ComfyUI
workflow.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

import comfy.model_base
import comfy.model_management
import comfy.sd
import folder_paths

from tensorrt_wan.engine.base import TensorRTEngineWrapper

# Registers a real ComfyUI model-folder dropdown for engine_name (like unet_name/lora_name
# already get) instead of a free-text path field. `ComfyUI/models/tensorrt_engines/` is a plain
# directory of symlinks (aliases) into wherever the actual multi-GB engine files live
# (trtwan_known_working_engines/, trtwan_engines_refit_test/, etc.) -- nothing gets copied, this
# is just ComfyUI's standard single-directory-per-model-type convention pointed at whatever we've
# chosen to alias in there. Registered at import time (module-level, runs once when ComfyUI loads
# this custom node package), matching how ComfyUI custom nodes conventionally add new model types
# (see folder_paths.folder_names_and_paths). Sidecar .lora_map.json files are aliased in
# alongside their .engine with a matching basename -- TensorRTDiTLoraLoader derives the sidecar
# path from the chosen engine's path, so nothing extra to register for those.
_TENSORRT_ENGINE_DIR = "/workspace/runpod-slim/ComfyUI/models/tensorrt_engines"
if "tensorrt_engines" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt_engines"] = ([], {".engine"})
Path(_TENSORRT_ENGINE_DIR).mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path("tensorrt_engines", _TENSORRT_ENGINE_DIR)

_SAFETENSORS_DTYPES = {
    "F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16,
    "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
    "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _load_shell_fast(unet_path: str):
    """Get a correctly-configured model shell (`model_sampling`/`latent_format`/`concat_keys`)
    without reading the real ~28GB of weight data -- we only need those three attributes off the
    result; `diffusion_model` itself gets replaced with `TensorRTDiTModule` right after this
    returns, so the real weights would just be thrown away anyway.

    Reads only safetensors headers (shape/dtype, no data -- ~0s for ~1100 tensors vs. ~100s for a
    full mmap+copy read) into `torch.empty(..., device="meta")` placeholders, then feeds those into
    ComfyUI's real `load_diffusion_model_state_dict` (same model-config detection path a real load
    uses). That function's `BaseModel.load_model_weights` tries a real `.copy_()` into our meta
    tensors regardless (`assign` comes from `model_patcher.is_dynamic()`, which is hardcoded
    `False` on this ComfyUI version's `CoreModelPatcher` -- an alias to plain `ModelPatcher`, not
    the dynamic subclass, so it's never `True` here) and fails ("Cannot copy out of meta tensor;
    no data!"). Since we don't need that copy to succeed, `load_model_weights` is no-op'd for the
    duration of this one call rather than fighting the meta-tensor copy. Confirmed 2026-08-08 via
    `scripts/prototype_fast_shell_load.py`: identical `model_sampling.shift`/`latent_format`/
    `concat_keys`/`patch_embedding` shape to the real load, ~2300x faster.
    """
    with safe_open(unet_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        fake_sd = {}
        for k in f.keys():
            slice_ = f.get_slice(k)
            dtype = _SAFETENSORS_DTYPES.get(slice_.get_dtype(), torch.float32)
            fake_sd[k] = torch.empty(slice_.get_shape(), dtype=dtype, device="meta")

    orig_load_model_weights = comfy.model_base.BaseModel.load_model_weights
    comfy.model_base.BaseModel.load_model_weights = lambda self, sd, unet_prefix="", assign=False: self
    try:
        model = comfy.sd.load_diffusion_model_state_dict(fake_sd, model_options={}, metadata=metadata)
    finally:
        comfy.model_base.BaseModel.load_model_weights = orig_load_model_weights
    if model is None:
        raise RuntimeError(f"Could not detect model type of: {unet_path}")
    return model


class TensorRTDiTModule(torch.nn.Module):
    """Drop-in replacement for a real `WanModel` as `BaseModel.diffusion_model` -- matches the
    call signature ComfyUI's `apply_model()` (comfy/model_base.py) actually uses:
    `self.diffusion_model(xc, t, context=context, control=control,
    transformer_options=transformer_options, **extra_conds)`. `xc` arrives already
    channel-concatenated (noise+mask+image_latent) and already cast to `self.dtype` by
    `apply_model()` itself (via `get_dtype_inference()` reading this module's own `.dtype`).

    `in_channels`/`max_text_tokens` must match what the TensorRT engine at `engine_path` was
    actually exported with (`DiTExporter`'s `in_channels`/`max_text_tokens` kwargs) -- this class
    has no way to introspect the engine's own bound shapes for these, they're fixed at export time.
    """

    def __init__(
        self,
        engine_path: str,
        in_channels: int = 36,
        max_text_tokens: int = 512,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.max_text_tokens = max_text_tokens
        # checkpoint_path (the *original*, non-TensorRT checkpoint) is only needed by
        # TensorRTDiTLoraLoader (lora_loader.py) -- it has no real base weights to read back a
        # LoRA delta against otherwise (TensorRT's Refitter can't read a never-yet-refit weight,
        # see docs/wan2.2_i2v_14b_notes.md). Stored here rather than requiring the LoRA node to
        # take it as a separate manual input -- TensorRTDiTLoader already knows this path.
        self.checkpoint_path = checkpoint_path
        # WAN22.concat_cond (comfy/model_base.py) introspects
        # `diffusion_model.patch_embedding.weight.shape[1]` to compute how many extra
        # conditioning channels it needs -- only `.weight.shape[1]` is ever read, no real conv
        # needed, this is never called from forward().
        self.patch_embedding = torch.nn.Conv3d(in_channels, 1, kernel_size=1)
        self._wrapper = TensorRTEngineWrapper(engine_path, device=comfy.model_management.get_torch_device())
        self._wrapper.load()

    @property
    def wrapper(self) -> TensorRTEngineWrapper:
        """Exposed for `TensorRTDiTLoraLoader` (comfyui/nodes/lora_loader.py) to call
        `.refit_weights()` on -- nothing in `forward()` needs this, only LoRA application."""
        return self._wrapper

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **extra_conds):
        # Our exported DiT's `context` input has no dynamic axis (max_text_tokens baked in at
        # export time) -- a caller's tokenizer may not pad to exactly that length by default, so
        # pad/truncate defensively rather than let TensorRT reject a mismatched shape.
        if context is not None and context.shape[1] != self.max_text_tokens:
            if context.shape[1] < self.max_text_tokens:
                pad = torch.zeros(
                    context.shape[0], self.max_text_tokens - context.shape[1], context.shape[2],
                    device=context.device, dtype=context.dtype,
                )
                context = torch.cat([context, pad], dim=1)
            else:
                context = context[:, : self.max_text_tokens]

        batch = x.shape[0]
        if batch == 1:
            out = self._wrapper.infer({"x": x, "timestep": timestep, "context": context})
            return out["noise_pred"]

        # ComfyUI's real CFG batches cond+uncond into one batch=N call by default; our engine's
        # batch dim was specialized to 1 at export time (torch.export traced it that way
        # regardless of Dim.AUTO -- see DiTExporter.dynamic_axes()'s docstring). Split into N
        # batch=1 calls and re-concatenate.
        outputs = []
        for i in range(batch):
            t_i = timestep[i : i + 1] if timestep.shape[0] == batch else timestep
            c_i = context[i : i + 1] if context is not None else None
            out = self._wrapper.infer({"x": x[i : i + 1], "timestep": t_i, "context": c_i})
            outputs.append(out["noise_pred"])
        return torch.cat(outputs, dim=0)


class TensorRTDiTLoader:
    """Loads a real Wan diffusion model shell via ComfyUI's own `comfy.sd.load_diffusion_model`
    (for correctly-configured `model_sampling`/`latent_format`/`concat_keys` -- the real weights
    it loads are immediately discarded), then replaces `.diffusion_model` with a TensorRT-backed
    module. Output is a real `MODEL` socket -- wire it directly into a stock `KSampler`/
    `KSamplerAdvanced` (or this repo's own two-phase MoE sampler) exactly like a real
    `UNETLoader`/`Power Lora Loader` chain, no other node in this package required.

    `unet_name` is the *original* (non-TensorRT) checkpoint -- only used to derive the correct
    shell config, never actually run. `engine_name` picks one of this project's own built
    TensorRT `.engine` files, listed from `_TENSORRT_ENGINE_DIRS` above (see docs/runpod_setup.md
    for how to build one) -- a real dropdown, not a path you have to type/paste.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "engine_name": (folder_paths.get_filename_list("tensorrt_engines"),),
                "in_channels": ("INT", {"default": 36, "min": 1, "max": 256}),
                "max_text_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "fast_shell_load": ("BOOLEAN", {"default": True}),
            }
        }

    def load(
        self,
        unet_name: str,
        engine_name: str,
        in_channels: int,
        max_text_tokens: int,
        fast_shell_load: bool = True,
    ):
        engine_path = folder_paths.get_full_path_or_raise("tensorrt_engines", engine_name)
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        if fast_shell_load:
            model = _load_shell_fast(unet_path)
        else:
            model = comfy.sd.load_diffusion_model(unet_path, model_options={})
        model.model.diffusion_model = TensorRTDiTModule(
            engine_path, in_channels=in_channels, max_text_tokens=max_text_tokens, checkpoint_path=unet_path
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"TensorRTDiTLoader": TensorRTDiTLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTDiTLoader": "TensorRT DiT Loader"}
