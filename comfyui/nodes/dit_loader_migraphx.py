"""MIGraphX-accelerated DiT loader for ComfyUI -- the AMD/ROCm counterpart to
`comfyui/nodes/dit_loader.py`'s `TensorRTDiTLoader`, for hardware with no TensorRT at all (see
docs/rocm_setup.md for the full rationale and setup recipe). Reuses `TensorRTDiTLoader`'s fast
meta-shell load (`_load_shell_fast`) unchanged -- that logic is entirely about ComfyUI's own
model-shell construction (`model_sampling`/`latent_format`/`concat_keys`), not TensorRT-specific,
so there's nothing to duplicate here.

Pair with stock ComfyUI CLIPLoader/VAELoader/CLIPTextEncode/VAEDecode/KSampler(Advanced) nodes,
exactly like `TensorRTDiTLoader` -- only the diffusion model differs.
"""

from __future__ import annotations

from pathlib import Path

import torch

import comfy.model_management
import folder_paths

from tensorrt_wan.engine.migraphx_engine import MIGraphXEngineWrapper

from .dit_loader import _load_shell_fast

# Same convention as dit_loader.py's _TENSORRT_ENGINE_DIR: a real ComfyUI model-folder dropdown,
# ".onnx" instead of ".engine" since a MIGraphX build's cached artifact is the static ONNX file
# itself (see export/migraphx_build.py's module docstring for why there's no separate compiled
# "engine" blob the way TensorRT produces).
_MIGRAPHX_ENGINE_DIR = "/workspace/runpod-slim/ComfyUI/models/migraphx_engines"
if "migraphx_engines" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["migraphx_engines"] = ([], {".onnx"})
Path(_MIGRAPHX_ENGINE_DIR).mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path("migraphx_engines", _MIGRAPHX_ENGINE_DIR)


class MIGraphXDiTModule(torch.nn.Module):
    """Drop-in replacement for a real `WanModel` as `BaseModel.diffusion_model` -- identical
    contract to `dit_loader.py`'s `TensorRTDiTModule` (see that class's docstring for the exact
    `apply_model()` call signature this matches and the batch=1 export-shape rationale); only the
    backing engine wrapper differs.
    """

    def __init__(
        self,
        onnx_path: str,
        in_channels: int = 36,
        max_text_tokens: int = 512,
        precision: str = "bf16",
    ) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.max_text_tokens = max_text_tokens
        # See TensorRTDiTModule's identical field: only .weight.shape[1] is ever read
        # (WAN22.concat_cond), never called from forward().
        self.patch_embedding = torch.nn.Conv3d(in_channels, 1, kernel_size=1)
        self._wrapper = MIGraphXEngineWrapper(
            onnx_path, device=comfy.model_management.get_torch_device(), precision=precision
        )
        self._wrapper.load()

    @property
    def wrapper(self) -> MIGraphXEngineWrapper:
        return self._wrapper

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **extra_conds):
        # Same context pad/truncate as TensorRTDiTModule.forward() -- see that method's docstring.
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

        # Same batch=1-per-call splitting as TensorRTDiTModule.forward() -- the MIGraphX build is
        # exported static-shape (DiTExporter's static=True, see export/base.py), specialized to
        # batch=1 same as the TensorRT path.
        outputs = []
        for i in range(batch):
            t_i = timestep[i : i + 1] if timestep.shape[0] == batch else timestep
            c_i = context[i : i + 1] if context is not None else None
            out = self._wrapper.infer({"x": x[i : i + 1], "timestep": t_i, "context": c_i})
            outputs.append(out["noise_pred"])
        return torch.cat(outputs, dim=0)


class MIGraphXDiTLoader:
    """AMD/ROCm counterpart to `TensorRTDiTLoader` -- see that class's docstring for the shared
    shell-loading contract. `engine_name` picks a MIGraphX-targeted `.onnx` build (see
    docs/rocm_setup.md for how to produce one) instead of a `.engine` file.
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
                "engine_name": (folder_paths.get_filename_list("migraphx_engines"),),
                "in_channels": ("INT", {"default": 36, "min": 1, "max": 256}),
                "max_text_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "precision": (["bf16", "fp16", "fp8"], {"default": "bf16"}),
                "fast_shell_load": ("BOOLEAN", {"default": True}),
            }
        }

    def load(
        self,
        unet_name: str,
        engine_name: str,
        in_channels: int,
        max_text_tokens: int,
        precision: str = "bf16",
        fast_shell_load: bool = True,
    ):
        onnx_path = folder_paths.get_full_path_or_raise("migraphx_engines", engine_name)
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        if fast_shell_load:
            model = _load_shell_fast(unet_path)
        else:
            import comfy.sd

            model = comfy.sd.load_diffusion_model(unet_path, model_options={})
        model.model.diffusion_model = MIGraphXDiTModule(
            onnx_path, in_channels=in_channels, max_text_tokens=max_text_tokens, precision=precision
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"MIGraphXDiTLoader": MIGraphXDiTLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MIGraphXDiTLoader": "MIGraphX DiT Loader (AMD/ROCm)"}
