from . import (
    cache_manager,
    conditioning_manager,
    diagnostics,
    dit_loader,
    engine_builder,
    engine_inspector,
    loader,
    lora_loader,
    precision_selector,
    runtime_manager,
    sampler,
    scheduler,
    text_encoder,
    vae_decoder,
    vae_encoder,
)

_MODULES = (
    runtime_manager,
    precision_selector,
    loader,
    dit_loader,  # recommended path for a real ComfyUI workflow -- see dit_loader.py's docstring
    lora_loader,  # LoRA application for TensorRTDiTLoader models -- see lora_loader.py's docstring
    engine_builder,
    text_encoder,
    vae_encoder,
    vae_decoder,
    conditioning_manager,
    scheduler,
    sampler,
    cache_manager,
    diagnostics,
    engine_inspector,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _module in _MODULES:
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)
