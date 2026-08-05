from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor
from tensorrt_wan.engine.vae_engine import VAEEncoderEngine

from .. import types

_KIND_BY_ROLE = {
    "image": ConditioningKind.IMAGE,
    "first_frame": ConditioningKind.FIRST_FRAME,
    "last_frame": ConditioningKind.LAST_FRAME,
}


class TensorRTVAEEncoder:
    """Encodes a ComfyUI IMAGE into a Wan latent for use as image/first-frame/last-frame
    conditioning (I2V, first-frame, and future last-frame workflows all route through here —
    they only differ in `role`, per docs/architecture.md).
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.COND_INPUT,)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_encoder": (types.VAE_ENCODER_ENGINE,),
                "image": ("IMAGE",),
                "role": (list(_KIND_BY_ROLE.keys()), {"default": "image"}),
            }
        }

    def run(self, vae_encoder: VAEEncoderEngine, image, role: str):
        # ComfyUI IMAGE is (B, H, W, C) float in [0, 1]; Wan's VAE expects (B, C, H, W) in [-1, 1].
        pixels = image.permute(0, 3, 1, 2) * 2.0 - 1.0
        latent = vae_encoder.encode_image(pixels)
        return (ConditioningTensor(kind=_KIND_BY_ROLE[role], embedding=latent),)


NODE_CLASS_MAPPINGS = {"TensorRTVAEEncoder": TensorRTVAEEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTVAEEncoder": "TensorRT VAE Encoder"}
