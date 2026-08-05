from tensorrt_wan.cli.loader import resolve_loader
from tensorrt_wan.config.schema import DEFAULT_RESOLUTION_PROFILES
from tensorrt_wan.export.exporters import DiTExporter, TextEncoderExporter, VAEDecoderExporter, VAEEncoderExporter
from tensorrt_wan.export.pipeline import run_export_pipeline
from tensorrt_wan.runtime.manager import RuntimeManager

from .. import types

_EXPORTERS = {
    "text_encoder": TextEncoderExporter,
    "dit": DiTExporter,
    "vae_encoder": VAEEncoderExporter,
    "vae_decoder": VAEDecoderExporter,
}


class TensorRTEngineBuilder:
    """Runs the full torch.export -> ONNX -> TensorRT pipeline for one Wan submodule.

    `loader` is a `module.path:function_name` string (see `cli.loader.resolve_loader`) returning
    a loaded `nn.Module` given `checkpoint_path` — this node does not vendor Wan's own model code
    (see `tensorrt_wan/cli/loader.py` docstring for why).
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.RUNTIME, "STRING")
    RETURN_NAMES = ("runtime", "engine_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime": (types.RUNTIME,),
                "component": (sorted(_EXPORTERS.keys()),),
                "loader": ("STRING", {"default": "my_wan_adapter:load_dit"}),
                "checkpoint_path": ("STRING", {"default": ""}),
                "model_hash": ("STRING", {"default": ""}),
                "exporter_kwargs_json": ("STRING", {"default": "{}", "multiline": True}),
                "force_rebuild": ("BOOLEAN", {"default": False}),
            }
        }

    def run(
        self,
        runtime: RuntimeManager,
        component: str,
        loader: str,
        checkpoint_path: str,
        model_hash: str,
        exporter_kwargs_json: str,
        force_rebuild: bool,
    ):
        import json

        load_fn = resolve_loader(loader)
        model = load_fn(checkpoint_path)
        exporter = _EXPORTERS[component](model, **json.loads(exporter_kwargs_json))

        engine_path = run_export_pipeline(
            exporter,
            runtime,
            list(runtime.config.resolution_profiles or DEFAULT_RESOLUTION_PROFILES),
            model_hash=model_hash or checkpoint_path,
            force=force_rebuild,
        )
        return (runtime, str(engine_path))


NODE_CLASS_MAPPINGS = {"TensorRTEngineBuilder": TensorRTEngineBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTEngineBuilder": "TensorRT Engine Builder"}
