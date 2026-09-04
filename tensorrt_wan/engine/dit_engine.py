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

from tensorrt_wan.conditioning.types import ConditioningKind, UnifiedConditioning
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
# docstring. Temporal frame index (first_frame=0, last_frame=-1) is now independently confirmed
# too, against `WanFirstLastFrameToVideo.execute()` (comfy_extras/nodes_wan.py):
# `image[:start_image.shape[0]] = start_image` / `image[-end_image.shape[0]:] = end_image`.
_IMAGE_CONDITIONING_FRAME_INDEX = {"first_frame": 0, "last_frame": -1}

# `api/wan_engine.py`'s standalone-API path builds the full-length image_latent/mask pair itself
# (real gray-fill + single vae.encode over the whole padded video, matching
# `WanFirstLastFrameToVideo` exactly) and hands it over already-complete under this one kind —
# skip the zero-pad/single-frame placement logic below entirely for it. The ComfyUI node-graph
# path (comfyui/nodes/vae_encoder.py's `TensorRTVAEEncoder`, one node per frame with no visibility
# into the target video length) can't build that yet and still goes through
# `_IMAGE_CONDITIONING_FRAME_INDEX`'s legacy zero-pad path below — see docs/roadmap.md.
_PREBUILT_IMAGE_VIDEO_KEY = ConditioningKind.IMAGE_VIDEO.value


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
        wrapper: object | None = None,
    ) -> None:
        """`wrapper`, if given, replaces the default `TensorRTEngineWrapper` entirely (and
        `engine_path`/`torch_fallback` are then ignored — the wrapper already has its own
        engine/program path). Duck-typed to any object exposing `.load()`, `.unload()`, and
        `.infer(dict[str, Tensor]) -> dict[str, Tensor]` — no shared base class, since the two
        real implementations' construction-time concerns (TensorRT's LoRA refit/optimization
        profiles vs MIGraphX's fixed-shape static program) are genuinely different, not just an
        implementation detail. This is what lets `engine/migraphx_engine.py`'s
        `MIGraphXEngineWrapper` reuse every bit of this class's conditioning-assembly logic
        below (`_build_inputs`/`_concat_image_conditioning`/`_null_conditioning`) instead of
        duplicating ~200 lines of hard-won, bug-fixed conditioning code into a second engine
        class — see docs/rocm_setup.md.
        """
        self.device = device or torch.device("cuda")
        self._wrapper = wrapper or TensorRTEngineWrapper(engine_path, device=self.device, torch_fallback=torch_fallback)

    def load(self) -> None:
        self._wrapper.load()

    def unload(self) -> None:
        self._wrapper.unload()

    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: UnifiedConditioning,
        guidance_scale: float = 1.0,
        uncond_conditioning: UnifiedConditioning | None = None,
    ) -> torch.Tensor:
        """Run one DiT forward pass and return the predicted velocity/noise for this timestep.

        `guidance_scale > 1.0` runs classifier-free guidance as two batched engine calls (one
        with `conditioning` as given, one unconditional) rather than one double-batch call, so a
        single optimization profile covers both the guided and unguided case.

        `uncond_conditioning`, if given, is used as-is for the unconditional pass -- callers with
        access to a text encoder should build this from a real *empty-string* encoding (see
        `_null_conditioning`'s docstring for why an all-zero embedding is the wrong default) and
        pass it in. Falls back to `_null_conditioning(conditioning)` (zeroing the text embedding)
        only when the caller can't provide a real one -- e.g. no text encoder in scope at this
        call site.
        """
        cond_inputs = self._build_inputs(latents, timestep, conditioning)
        cond_out = self._wrapper.infer(cond_inputs)["noise_pred"]

        if guidance_scale == 1.0:
            return cond_out

        if uncond_conditioning is None:
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
        prebuilt_image_video = embeddings.pop(_PREBUILT_IMAGE_VIDEO_KEY, None)
        if prebuilt_image_video is not None and image_kinds:
            raise NotImplementedError(
                "Both prebuilt image_video conditioning and legacy first_frame/last_frame "
                "conditioning were present in the same UnifiedConditioning — these are two "
                "different callers' conventions and shouldn't mix."
            )
        if prebuilt_image_video is not None:
            mask = conditioning.masks[_PREBUILT_IMAGE_VIDEO_KEY]
            x = torch.cat([latents, mask, prebuilt_image_video], dim=1)
        elif image_kinds:
            x = _concat_image_conditioning(latents, image_kinds)
        else:
            x = latents

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
            if key not in _IMAGE_CONDITIONING_FRAME_INDEX and key != _PREBUILT_IMAGE_VIDEO_KEY:
                inputs[f"{key}_mask"] = mask
        return inputs


def _concat_image_conditioning(x: torch.Tensor, image_kinds: dict[str, torch.Tensor]) -> torch.Tensor:
    """**Legacy/ComfyUI-graph path only** — the standalone `WanEngine` API no longer calls this
    (see `_PREBUILT_IMAGE_VIDEO_KEY` above and `api/wan_engine.py`'s
    `_build_image_to_video_conditioning`, which builds the real gray-fill + single-vae-encode
    conditioning and bypasses this function's placement logic entirely). Still used by
    `comfyui/nodes/vae_encoder.py`'s `TensorRTVAEEncoder` node, which encodes one frame per node
    call with no visibility into the target video's full length — it can't build the real
    algorithm's padded-video encode, so this function's zero-pad approximation remains its only
    option until a joint ComfyUI node (mirroring `WanFirstLastFrameToVideo`'s single-node,
    both-frames-plus-length signature) exists. See docs/roadmap.md.

    Channel-concatenates every present image-conditioning kind onto `x`, matching Wan's real
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

    Also confirmed via `concat_cond`: `WanImageToVideo`/`WanFirstLastFrameToVideo` gray-fill
    (pixel value 0.5) every frame without a real reference image *before* VAE-encoding the whole
    padded video in one call, so the "padding" latent frames are whatever the VAE produces for
    gray input — not zero. This function still zero-pads in latent space directly (real fix now
    lives in `api/wan_engine.py`'s `_build_image_to_video_conditioning` for the standalone path;
    this function keeps the zero-pad approximation for the ComfyUI-graph path only, see this
    function's docstring header).

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
    """Fallback-only: zero the text embedding for the unconditional CFG pass; other conditioning
    (image, control, IP-Adapter) is left as-is.

    Prefer passing a real `uncond_conditioning` (a genuine empty-string text encoding) to
    `denoise_step()` instead of relying on this function when a text encoder is available at the
    call site. Real CFG convention for T5-style text encoders (SD3/Flux/Wan): the unconditional
    pass uses an actual empty-string encoding through the same tokenizer/embedding path, not an
    all-zero tensor -- the model never saw an all-zero embedding during training, so it's
    out-of-distribution input to cross-attention, and CFG's `uncond + scale*(cond-uncond)` formula
    then amplifies whatever that produces. Confirmed via a real eager-mode test this session
    (docs/wan2.2_i2v_14b_notes.md, 2026-08-07/08): swapping to a real empty-string embedding
    produced a visibly smoother/more well-behaved prediction-magnitude trajectory across the
    denoising schedule than zeroing did, though not sufficient alone to fix the broader
    incoherent-output investigation that session was chasing. Kept as a fallback here (not removed)
    for callers with no text encoder in scope (e.g. a raw `DiTEngine` used standalone).
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
