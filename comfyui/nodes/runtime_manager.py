from tensorrt_wan.config.schema import CacheConfig, PrecisionConfig, TensorRTWanConfig
from tensorrt_wan.runtime.manager import RuntimeManager

from .. import types


class TensorRTRuntimeManager:
    """GPU/TensorRT capability detection + precision/cache config for the rest of the graph.

    This is normally the first TensorRT-Wan node in a workflow — every other node that touches
    an engine takes its `runtime` output.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.RUNTIME,)
    RETURN_NAMES = ("runtime",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["auto", "fp8", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "cache_dir": ("STRING", {"default": "~/.cache/tensorrt_wan/engines"}),
            }
        }

    def run(self, precision: str, cache_dir: str):
        from pathlib import Path

        config = TensorRTWanConfig(
            precision=PrecisionConfig(mode=precision),
            cache=CacheConfig(directory=Path(cache_dir).expanduser()),
        )
        return (RuntimeManager(config),)


NODE_CLASS_MAPPINGS = {"TensorRTRuntimeManager": TensorRTRuntimeManager}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTRuntimeManager": "TensorRT Runtime Manager"}
