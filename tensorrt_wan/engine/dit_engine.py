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
# Only "text" is mapped here: "first_frame"/"last_frame" are channel-concatenated into `x` itself
# (handled by _concat_image_conditioning below), not passed as separate engine inputs — there's no
# engine input name for those, or for "control"/"ip_adapter", to map to. _build_inputs raises for
# anything left over that isn't one of those two cases.
_ENGINE_INPUT_NAME_BY_EMBEDDING_KEY = {"text": "context"}

# Which side of the temporal axis each image-conditioning kind occupies once concatenated into
# `x` — confirmed channel order is noise(16) ++ mask(4) ++ image_latent(16), verified directly
# against ComfyUI's `WAN21.concat_cond` (comfy/model_base.py), see _concat_image_conditioning's
# docstring; which *temporal* frame index each kind conditions is the obvious reading of the
# kind's name, not independently confirmed against ComfyUI source.
_IMAGE_CONDITIONING_FRAME_INDEX = {"first_frame": 0, "last_frame": -1}


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
        embeddings = dict(conditioning.embeddings)

        image_kinds = {
            kind: embeddings.pop(kind) for kind in _IMAGE_CONDITIONING_FRAME_INDEX if kind in embeddings
        }
        x = _concat_image_conditioning(latents, image_kinds) if image_kinds else latents

        unsupported = set(embeddings) - set(_ENGINE_INPUT_NAME_BY_EMBEDDING_KEY)
        if unsupported:
            raise NotImplementedError(
                f"DiTEngine cannot route conditioning kind(s) {sorted(unsupported)} into engine "
                "inputs yet — only text (cross-attention) and first_frame/last_frame "
                "(channel-concatenated into `x`) are wired up so far."
            )

        # DiTExporter.example_inputs() declared `timestep` as rank-1 (shape (1,)), but
        # SchedulerState.current_timestep (scheduler/state.py) returns a 0-d scalar
        # (`self.timesteps[self.step_index]`, indexing a 1-d tensor with a plain int). Confirmed
        # as a real bug via a real generation run: TensorRT's `set_input_shape` rejected the 0-d
        # tensor with "engineDims.nbDims == dims.nbDims" on every DiT call, and
        # `TensorRTEngineWrapper._infer_trt` didn't check that call's return value — the engine
        # silently ran with a stale/wrong timestep binding instead of raising. `.reshape(1)` is a
        # no-op if `timestep` is already rank-1, so this is safe regardless of scheduler
        # implementation. See docs/wan2.2_i2v_14b_notes.md.
        inputs = {"x": x, "timestep": timestep.reshape(1)}
        for embedding_key, engine_name in _ENGINE_INPUT_NAME_BY_EMBEDDING_KEY.items():
            if embedding_key in embeddings:
                inputs[engine_name] = embeddings[embedding_key]
        for key, mask in conditioning.masks.items():
            if key not in _IMAGE_CONDITIONING_FRAME_INDEX:
                inputs[f"{key}_mask"] = mask
        return inputs


def _concat_image_conditioning(x: torch.Tensor, image_kinds: dict[str, torch.Tensor]) -> torch.Tensor:
    """Channel-concatenates every present image-conditioning kind onto `x`, matching Wan's real
    `noise(16) ++ mask(4) ++ image_latent(16)` channel layout — confirmed by reading
    `WAN21.concat_cond` in ComfyUI's own `comfy/model_base.py` directly (`comfy_extras/
    nodes_wan.py`'s `WanImageToVideo` builds the mask/image separately; `concat_cond` is what
    actually assembles them: `torch.cat((mask, image), dim=1)`, then concatenated after `noise`).
    Exactly one 16-channel image_latent slot and one 4-channel mask slot *total*, not one pair
    per conditioning kind.

    **Real bug, confirmed and fixed:** earlier versions of this function used
    `noise ++ image_latent ++ mask` order (image before mask) — the reverse of the real layout.
    Caught via a decisive eager-vs-TensorRT comparison that showed the *engine* matches the real
    checkpoint almost exactly (cosine_similarity=0.999995) even while end-to-end generation
    produced incoherent output — meaning the bug had to be in what was fed to the model, not the
    model/engine itself. Reading `concat_cond` directly settled it: wrong channel order means the
    model's mask-weight-slice and image-weight-slice were being fed each other's data entirely,
    which explains the complete lack of coherent structure in every generation attempt. See
    docs/wan2.2_i2v_14b_notes.md's "Shift sweep and the decisive eager-vs-TensorRT comparison"
    and the finding that follows it.

    Mask *polarity* was already correct despite looking backwards at a glance: `WanImageToVideo`
    builds a mask with 0=known/1=to-generate, but `concat_cond` inverts it (`mask = 1.0 - mask`)
    before concatenating — net result 1=known/0=to-generate, which is what this function already
    produced. Traced through both stages to confirm rather than assumed.

    Also confirmed via `concat_cond`, still not implemented here: `WanImageToVideo` gray-fills
    (pixel value 0.5) every frame without a real reference image *before* VAE-encoding the whole
    padded video in one call, so the "padding" latent frames are whatever the VAE produces for
    gray input — not zero. This function still zero-pads in latent space directly. Likely a
    smaller-magnitude discrepancy than the channel-order bug was, but not yet fixed or measured;
    worth revisiting once the channel-order fix's actual effect on output quality is confirmed.

    `image_kinds`: e.g. `{"first_frame": (B, C_vae, 1, H, W), "last_frame": (B, C_vae, 1, H, W)}`
    — each value a single encoded frame (`VAEEncoderEngine.encode_image`'s output), not yet
    expanded to `x`'s full temporal length. All kinds present share one combined image_latent/
    mask pair, each placed at its own temporal position — e.g. `first_frame` at index 0 and
    `last_frame` at index -1 simultaneously, both real, only the frames between them zero/gray.
    """
    batch, _, num_frames, height, width = x.shape
    channels = next(iter(image_kinds.values())).shape[1]

    full_image_latent = torch.zeros(batch, channels, num_frames, height, width, device=x.device, dtype=x.dtype)
    mask = torch.zeros(batch, 4, num_frames, height, width, device=x.device, dtype=x.dtype)

    for kind, image_latent in image_kinds.items():
        frame_index = _IMAGE_CONDITIONING_FRAME_INDEX[kind]
        index = frame_index if frame_index >= 0 else num_frames + frame_index
        full_image_latent[:, :, index] = image_latent[:, :, 0]
        mask[:, :, index] = 1.0

    return torch.cat([x, mask, full_image_latent], dim=1)


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
