import torch

from tensorrt_wan.conditioning.types import UnifiedConditioning
from tensorrt_wan.engine.dit_engine import DiTEngine
from tensorrt_wan.scheduler.base import Scheduler

from .. import types


class TensorRTSampler:
    """Runs the full denoising loop through the Unified TensorRT DiT Engine.

    Analogous to a stock KSampler: takes a `LATENT` (used for its shape/device — an "Empty Latent
    Video" node upstream, same convention as ComfyUI's own Wan support) and returns a `LATENT`
    with `samples` replaced by the denoised result.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit": (types.DIT_ENGINE,),
                "conditioning": (types.UNIFIED_CONDITIONING,),
                "scheduler": (types.SCHEDULER,),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def run(
        self,
        dit: DiTEngine,
        conditioning: UnifiedConditioning,
        scheduler: Scheduler,
        latent,
        steps: int,
        cfg: float,
        seed: int,
    ):
        shape = latent["samples"].shape
        generator = torch.Generator(device=dit.device).manual_seed(seed)
        initial_latents = torch.randn(shape, generator=generator, device=dit.device)

        denoised = dit.generate(initial_latents, conditioning, scheduler, steps, guidance_scale=cfg)
        return ({"samples": denoised},)


NODE_CLASS_MAPPINGS = {"TensorRTSampler": TensorRTSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTSampler": "TensorRT Sampler"}
