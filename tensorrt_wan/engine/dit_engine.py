"""The Unified TensorRT DiT Engine.

Highest-priority optimization target in the project (see PLAN.md). Every workflow — T2V, I2V,
V2V, ControlNet, IP-Adapter, LoRA, future conditioning methods — ends up calling `denoise_step`
through the same engine instance; workflow-specific behavior lives entirely in what
`ConditioningManager.combine()` fed into `UnifiedConditioning`, never in a workflow-specific
engine subclass.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tensorrt_wan.conditioning.types import UnifiedConditioning
from tensorrt_wan.engine.base import TensorRTEngineWrapper
from tensorrt_wan.scheduler.base import Scheduler
from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)

_NULL_TOKEN_KEY = "text"  # conditioning key classifier-free guidance nulls out for the unconditional pass

# Maps a UnifiedConditioning.embeddings key (ConditioningKind.value) to the tensor name the built
# TensorRT engine actually exposes — confirmed against WanModel.forward(x, timestep, context, ...)
# on real Wan 2.2 hardware (see docs/wan2.2_i2v_14b_notes.md); "text" -> "context" is not a
# cosmetic rename, the engine's I/O tensor is literally named "context".
#
# Only "text" is mapped: Wan channel-concatenates image-latent + mask into `x` itself before
# patch embedding rather than passing them as separate cross-attention tensors (see the same
# notes doc's conditioning-mismatch section) — there's no engine input name for "image"/"control"/
# etc. to map to yet. _build_inputs raises rather than silently dropping those below.
_ENGINE_INPUT_NAME_BY_EMBEDDING_KEY = {"text": "context"}


class DiTEngine:
    """Wraps the single TensorRT engine that performs all diffusion denoising.

    LoRA weight deltas in `UnifiedConditioning.lora_weights` are expected to already be folded
    into the loaded engine at build time (LoRA changes engine weights, not per-step inputs) —
    `denoise_step` accepts them for bookkeeping/diagnostics but a mismatch against what the
    currently-loaded engine was built with should be caught by `RuntimeManager`/`EngineCache`,
    not silently ignored here.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | None = None,
        torch_fallback: torch.nn.Module | None = None,
    ) -> None:
        self.device = device or torch.device("cuda")
        self._wrapper = TensorRTEngineWrapper(engine_path, device=self.device, torch_fallback=torch_fallback)

    def load(self) -> None:
        self._wrapper.load()

    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: UnifiedConditioning,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run one DiT forward pass and return the predicted velocity/noise for this timestep.

        `guidance_scale > 1.0` runs classifier-free guidance as two batched engine calls (one
        with `conditioning` as given, one with the text embedding zeroed) rather than one
        double-batch call, so a single optimization profile covers both the guided and
        unguided case.
        """
        cond_inputs = self._build_inputs(latents, timestep, conditioning)
        cond_out = self._wrapper.infer(cond_inputs)["noise_pred"]

        if guidance_scale == 1.0:
            return cond_out

        uncond_conditioning = _null_conditioning(conditioning)
        uncond_inputs = self._build_inputs(latents, timestep, uncond_conditioning)
        uncond_out = self._wrapper.infer(uncond_inputs)["noise_pred"]

        return uncond_out + guidance_scale * (cond_out - uncond_out)

    def generate(
        self,
        initial_latents: torch.Tensor,
        conditioning: UnifiedConditioning,
        scheduler: Scheduler,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the full denoising loop, returning the final latent tensor.

        The loop itself never leaves the GPU: `scheduler.step` only ever touches its own
        `SchedulerState` buffers and `latents`, so there is no per-step host sync between calling
        this and reading back the last `denoise_step` output.
        """
        state = scheduler.prepare(num_inference_steps, self.device)
        latents = initial_latents
        while not state.done:
            timestep = state.current_timestep
            noise_pred = self.denoise_step(latents, timestep, conditioning, guidance_scale)
            latents = scheduler.step(state, noise_pred, latents)
        return latents

    def _build_inputs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: UnifiedConditioning,
    ) -> dict[str, torch.Tensor]:
        unsupported = set(conditioning.embeddings) - set(_ENGINE_INPUT_NAME_BY_EMBEDDING_KEY)
        if unsupported:
            raise NotImplementedError(
                f"DiTEngine cannot route conditioning kind(s) {sorted(unsupported)} into engine "
                "inputs yet — Wan channel-concatenates image/mask conditioning into `x` before "
                "patch embedding rather than passing it as a separate tensor, and that "
                "concatenation isn't implemented (see docs/wan2.2_i2v_14b_notes.md). Only "
                "text-only (T2V) conditioning is currently supported."
            )

        inputs = {"x": latents, "timestep": timestep}
        for embedding_key, engine_name in _ENGINE_INPUT_NAME_BY_EMBEDDING_KEY.items():
            if embedding_key in conditioning.embeddings:
                inputs[engine_name] = conditioning.embeddings[embedding_key]
        for key, mask in conditioning.masks.items():
            inputs[f"{key}_mask"] = mask
        return inputs


def _null_conditioning(conditioning: UnifiedConditioning) -> UnifiedConditioning:
    """Zero the text embedding for the unconditional CFG pass; other conditioning (image,
    control, IP-Adapter) is left as-is, matching Wan's own CFG convention of nulling only the
    prompt embedding.
    """
    if _NULL_TOKEN_KEY not in conditioning.embeddings:
        return conditioning
    nulled = UnifiedConditioning(
        embeddings=dict(conditioning.embeddings),
        masks=conditioning.masks,
        lora_weights=conditioning.lora_weights,
        metadata=conditioning.metadata,
    )
    nulled.embeddings[_NULL_TOKEN_KEY] = torch.zeros_like(conditioning.embeddings[_NULL_TOKEN_KEY])
    return nulled
