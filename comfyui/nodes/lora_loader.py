"""TensorRT-specific LoRA loader for `TensorRTDiTLoader` models.

ComfyUI's stock LoRA nodes can't work here -- they patch `ModelPatcher`'s parameter dict, but
`TensorRTDiTModule` (dit_loader.py) has no real per-block weights for that patch to land on; the
engine's weights are baked into the `.engine` file at build time. This node instead computes the
LoRA's delta directly against the original checkpoint and applies it to the compiled engine via
TensorRT's Refit API (`tensorrt_wan.engine.lora_refit`), mutating the same `TensorRTDiTModule` in
place -- see docs/wan2.2_i2v_14b_notes.md's Refit-API entries for the full investigation.

Only works on a REFIT-capable engine (built with `TRTWAN_ENABLE_REFIT=1`, see
`tensorrt_wan.export.trt_build`) -- the pinned known-working production engines are not
refit-capable as built, this needs its own build. That same build also writes the weight-name-map
sidecar this node needs (`tensorrt_wan.lora.weight_map_path_for_engine`) -- an engine built before
that sidecar-writing was added won't have one; rebuild it.

Not yet supported: composing multiple LoRAs, and bias/norm/modulation deltas (`.diff_b`/`.diff`/
`.diff_m` LoRA keys) -- only the 400 q/k/v/o/ffn weight matrices are marked refittable on the
engines built so far. See `tensorrt_wan.engine.lora_refit` for exactly what's applied vs skipped.
"""

from __future__ import annotations

import folder_paths

from tensorrt_wan.engine.lora_refit import apply_lora
from tensorrt_wan.lora import weight_map_path_for_engine

from .dit_loader import TensorRTDiTModule


class TensorRTDiTLoraLoader:
    """Chainable like a real LoRA loader: `model -> TensorRTDiTLoraLoader -> model -> ... ->
    KSampler`. `lora_name` is populated from `ComfyUI/models/loras/` exactly like the stock
    `LoraLoader` node. Everything else this node needs (the original checkpoint, the weight-name
    map) is read straight off the `model` -- `TensorRTDiTLoader` already knows both, no need to
    re-enter them here.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "load_lora"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    def load_lora(self, model, lora_name: str, strength: float):
        dit_module = model.model.diffusion_model
        if not isinstance(dit_module, TensorRTDiTModule):
            raise RuntimeError(
                "TensorRTDiTLoraLoader requires a model loaded via TensorRTDiTLoader -- got "
                f"diffusion_model of type {type(dit_module).__name__}. Wire this node's `model` "
                "input directly from a TensorRTDiTLoader node's output."
            )
        if not dit_module.checkpoint_path:
            raise RuntimeError(
                "This model's TensorRTDiTLoader didn't record a checkpoint_path -- reload it "
                "(this should always be set automatically; only stale/older-code loads miss it)."
            )
        weight_map_path = weight_map_path_for_engine(dit_module.wrapper.engine_path)
        if not weight_map_path.exists():
            raise RuntimeError(
                f"No weight-name map at {weight_map_path} -- this engine was built without "
                "TRTWAN_ENABLE_REFIT=1, or before sidecar-writing was added. Rebuild it with "
                "TRTWAN_ENABLE_REFIT=1 (see tensorrt_wan.export.trt_build)."
            )
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        apply_lora(dit_module.wrapper, dit_module.checkpoint_path, weight_map_path, lora_path, strength)
        return (model,)


NODE_CLASS_MAPPINGS = {"TensorRTDiTLoraLoader": TensorRTDiTLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTDiTLoraLoader": "TensorRT DiT LoRA Loader"}
