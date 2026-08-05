from tensorrt_wan.engine.vae_engine import VAEDecoderEngine

from .. import types


class TensorRTVAEDecoder:
    """Decodes denoised latents to a ComfyUI IMAGE batch (drop-in for a stock VAEDecode)."""

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_decoder": (types.VAE_DECODER_ENGINE,), "latent": ("LATENT",)}}

    def run(self, vae_decoder: VAEDecoderEngine, latent):
        pixels = vae_decoder.decode(latent["samples"])  # (B, C, T, H, W) in [-1, 1]
        frames = (pixels.clamp(-1, 1) + 1.0) / 2.0
        # ComfyUI IMAGE is (N, H, W, C); flatten batch*time into N, matching how ComfyUI's own
        # video nodes (e.g. VHS) represent a video as an IMAGE batch.
        b, c, t, h, w = frames.shape
        frames = frames.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).contiguous()
        return (frames,)


NODE_CLASS_MAPPINGS = {"TensorRTVAEDecoder": TensorRTVAEDecoder}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTVAEDecoder": "TensorRT VAE Decoder"}
