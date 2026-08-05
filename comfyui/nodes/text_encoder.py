from tensorrt_wan.conditioning.types import ConditioningTensor, ConditioningKind
from tensorrt_wan.engine.text_encoder_engine import TextEncoderEngine

from .. import types


class TensorRTTextEncoder:
    """TensorRT-accelerated Wan prompt encoder. Feeds the `text` socket on TensorRT Conditioning
    Manager — analogous to a stock CLIPTextEncode feeding a KSampler's positive/negative input.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.COND_INPUT,)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (types.TEXT_ENCODER_ENGINE,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def run(self, text_encoder: TextEncoderEngine, prompt: str):
        embedding = text_encoder.encode_text(prompt)
        return (ConditioningTensor(kind=ConditioningKind.TEXT, embedding=embedding, metadata={"prompt": prompt}),)


NODE_CLASS_MAPPINGS = {"TensorRTTextEncoder": TensorRTTextEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTTextEncoder": "TensorRT Text Encoder"}
