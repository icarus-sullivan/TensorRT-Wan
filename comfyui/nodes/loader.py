import torch

from tensorrt_wan.api.model_config import WanModelConfig
from tensorrt_wan.api.wan_engine import load_default_tokenizer
from tensorrt_wan.engine.dit_engine import DiTEngine
from tensorrt_wan.engine.text_encoder_engine import TextEncoderEngine
from tensorrt_wan.engine.vae_engine import VAEDecoderEngine, VAEEncoderEngine
from tensorrt_wan.runtime.manager import RuntimeManager

from .. import types


class TensorRTWanLoader:
    """Loads a Wan model's built TensorRT engines (see docs/engine_generation.md) from a model
    directory containing `wan_model.json` + `{text_encoder,dit,vae_encoder,vae_decoder}.engine`.

    Split into four separate engine outputs (rather than one opaque "pipeline" handle) so
    individual stages can be swapped — e.g. connecting a stock ComfyUI VAEDecode instead of
    `vae_decoder`, per the project's "minimal changes to existing workflows" goal.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (
        types.MODEL_CONFIG,
        types.TEXT_ENCODER_ENGINE,
        types.DIT_ENGINE,
        types.VAE_ENCODER_ENGINE,
        types.VAE_DECODER_ENGINE,
    )
    RETURN_NAMES = ("model_config", "text_encoder", "dit", "vae_encoder", "vae_decoder")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": (types.RUNTIME,),
                "model_dir": ("STRING", {"default": ""}),
            }
        }

    def run(self, runtime: RuntimeManager, model_dir: str):
        from pathlib import Path

        model_dir_path = Path(model_dir)
        config = WanModelConfig.load(model_dir_path / "wan_model.json")
        device = torch.device(f"cuda:{runtime.primary_gpu.index}") if runtime.primary_gpu else torch.device("cuda")

        tokenizer = load_default_tokenizer(config.tokenizer_name, config.max_text_tokens)
        text_encoder = TextEncoderEngine(model_dir_path / "text_encoder.engine", tokenizer, device=device)
        dit = DiTEngine(model_dir_path / "dit.engine", device=device)
        vae_encoder = VAEEncoderEngine(model_dir_path / "vae_encoder.engine", device=device)
        vae_decoder = VAEDecoderEngine(model_dir_path / "vae_decoder.engine", device=device)

        for component in (text_encoder, dit, vae_encoder, vae_decoder):
            component.load()
            runtime.register_engine(f"{model_dir}:{component.__class__.__name__}", component)

        return (config, text_encoder, dit, vae_encoder, vae_decoder)


NODE_CLASS_MAPPINGS = {"TensorRTWanLoader": TensorRTWanLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTWanLoader": "TensorRT Wan Loader"}
