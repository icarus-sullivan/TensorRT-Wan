"""CPU-only, no-GPU tests for the two I2V conditioning paths (see docs/roadmap.md /
docs/wan2.2_i2v_14b_notes.md 2026-08-06 session): the legacy zero-pad `_concat_image_conditioning`
(engine/dit_engine.py, still used by the ComfyUI-graph path) and the per-frame gray-encode
`_build_image_to_video_conditioning` (api/wan_engine.py, standalone-API path — a deliberate
simplification of ComfyUI's real single-whole-video-encode algorithm, forced by a real TensorRT
build limitation on `T>1` vae_encoder engines; see that function's docstring). No TensorRT engine
or real model weights involved — pure tensor-shape/value checks against fake stand-ins, allowed
under PLAN.md's no-GPU-execution rule.
"""

import torch

from tensorrt_wan.api.model_config import WanModelConfig
from tensorrt_wan.api.wan_engine import _build_image_to_video_conditioning
from tensorrt_wan.conditioning.types import ConditioningKind, UnifiedConditioning
from tensorrt_wan.engine.dit_engine import DiTEngine, _concat_image_conditioning


def _dit_engine() -> DiTEngine:
    return object.__new__(DiTEngine)  # _build_inputs doesn't touch self


def _model_config(**overrides) -> WanModelConfig:
    defaults = dict(
        latent_channels=16,
        vae_temporal_scale=4,
        vae_spatial_scale=8,
        text_embed_dim=4096,
        tokenizer_name="fake",
    )
    defaults.update(overrides)
    return WanModelConfig(**defaults)


class _FakeVAEEncoder:
    """`encode_image` returns a latent shaped from the real VAE's spatial downsampling, filled
    with the input's own mean pixel value so tests can tell which encode call produced which
    output frame — enough to exercise `_build_image_to_video_conditioning`'s per-frame call
    pattern without a real model.
    """

    def __init__(self, spatial_scale: int, channels: int = 16) -> None:
        self.spatial_scale = spatial_scale
        self.channels = channels
        self.encode_image_calls = 0

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        self.encode_image_calls += 1
        _, _, height, width = image.shape
        fill = image.mean()
        return torch.full(
            (1, self.channels, 1, height // self.spatial_scale, width // self.spatial_scale), fill
        )


def test_legacy_concat_places_first_and_last_frame_at_opposite_ends():
    x = torch.zeros(1, 16, 5, 2, 2)
    first = torch.full((1, 16, 1, 2, 2), 1.0)
    last = torch.full((1, 16, 1, 2, 2), 2.0)

    out = _concat_image_conditioning(x, {"first_frame": first, "last_frame": last})

    assert out.shape == (1, 16 + 4 + 16, 5, 2, 2)
    mask, image_latent = out[:, 16:20], out[:, 20:]
    assert torch.equal(image_latent[:, :, 0], first[:, :, 0])
    assert torch.equal(image_latent[:, :, -1], last[:, :, 0])
    assert torch.equal(image_latent[:, :, 2], torch.zeros_like(image_latent[:, :, 2]))  # still zero-padded
    assert mask[:, :, 0].eq(1.0).all() and mask[:, :, -1].eq(1.0).all()
    assert mask[:, :, 2].eq(0.0).all()


def test_prebuilt_image_video_path_skips_legacy_placement():
    engine = _dit_engine()
    latents = torch.zeros(1, 16, 3, 2, 2)
    image_latent = torch.full((1, 16, 3, 2, 2), 9.0)
    mask = torch.full((1, 4, 3, 2, 2), 0.5)
    conditioning = UnifiedConditioning(
        embeddings={ConditioningKind.IMAGE_VIDEO.value: image_latent},
        masks={ConditioningKind.IMAGE_VIDEO.value: mask},
    )

    inputs = engine._build_inputs(latents, torch.tensor(500.0), conditioning)

    assert inputs["x"].shape == (1, 36, 3, 2, 2)
    assert torch.equal(inputs["x"][:, 20:], image_latent)
    assert "image_video_mask" not in inputs  # handled via concat, not a generic {key}_mask input


def test_mixing_prebuilt_and_legacy_kinds_raises():
    engine = _dit_engine()
    conditioning = UnifiedConditioning(
        embeddings={
            ConditioningKind.IMAGE_VIDEO.value: torch.zeros(1, 16, 3, 2, 2),
            "first_frame": torch.zeros(1, 16, 1, 2, 2),
        },
        masks={ConditioningKind.IMAGE_VIDEO.value: torch.zeros(1, 4, 3, 2, 2)},
    )
    try:
        engine._build_inputs(torch.zeros(1, 16, 3, 2, 2), torch.tensor(0.0), conditioning)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for mixed conditioning kinds")


def test_gray_fill_conditioning_first_and_last_frame_mark_their_own_index_known():
    config = _model_config()
    vae = _FakeVAEEncoder(spatial_scale=8)

    image_latent, mask = _build_image_to_video_conditioning(
        vae,
        first_frame=torch.ones(1, 3, 16, 16),
        last_frame=torch.full((1, 3, 16, 16), -1.0),
        num_frames=13,  # latent_t = (13-1)//4 + 1 = 4
        height=16,
        width=16,
        model_config=config,
        device=torch.device("cpu"),
    )

    assert image_latent.shape == (1, 16, 4, 2, 2)
    assert mask.shape == (1, 4, 4, 2, 2)
    # Every mask channel is known/unknown together per frame -- no raw-granularity asymmetry,
    # since each latent frame comes from its own independent T=1 encode_image call, not a shared
    # whole-video causal-conv pass.
    assert mask[:, :, 0].eq(1.0).all()
    assert mask[:, :, -1].eq(1.0).all()
    assert mask[:, :, 1:-1].eq(0.0).all()
    assert image_latent[:, :, 0].eq(1.0).all()  # from the first_frame encode
    assert image_latent[:, :, -1].eq(-1.0).all()  # from the last_frame encode
    assert image_latent[:, :, 1].eq(0.0).all()  # from the shared gray encode


def test_gray_fill_conditioning_reuses_one_gray_encode_for_every_padding_frame():
    config = _model_config()
    vae = _FakeVAEEncoder(spatial_scale=8)

    _, _ = _build_image_to_video_conditioning(
        vae,
        first_frame=torch.ones(1, 3, 16, 16),
        last_frame=torch.ones(1, 3, 16, 16),
        num_frames=81,  # latent_t = 21 -- 19 padding frames, but only 1 gray encode call needed
        height=16,
        width=16,
        model_config=config,
        device=torch.device("cpu"),
    )

    # gray (once) + first + last -- not one call per output latent frame.
    assert vae.encode_image_calls == 3


if __name__ == "__main__":
    test_legacy_concat_places_first_and_last_frame_at_opposite_ends()
    test_prebuilt_image_video_path_skips_legacy_placement()
    test_mixing_prebuilt_and_legacy_kinds_raises()
    test_gray_fill_conditioning_first_and_last_frame_mark_their_own_index_known()
    test_gray_fill_conditioning_reuses_one_gray_encode_for_every_padding_frame()
    print("ok")
