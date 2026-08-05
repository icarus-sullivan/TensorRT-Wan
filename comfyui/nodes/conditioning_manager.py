from tensorrt_wan.conditioning.manager import ConditioningManager
from tensorrt_wan.conditioning.types import ConditioningTensor

from .. import types


class TensorRTConditioningManager:
    """Merges every connected conditioning source (text, image/first-frame, control, IP-Adapter,
    LoRA) into the single `UnifiedConditioning` the TensorRT Sampler consumes.

    This is the ComfyUI-graph form of `ConditioningManager.combine_encoded()` — each optional
    input is already an encoded `ConditioningTensor` from its own upstream node (TensorRT Text
    Encoder, TensorRT VAE Encoder, a future TensorRT ControlNet/IP-Adapter node), so adding a new
    conditioning method to a workflow means wiring one more optional socket here, not modifying
    this node.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.UNIFIED_CONDITIONING,)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        cond_socket = (types.COND_INPUT,)
        return {
            "required": {},
            "optional": {
                "text": cond_socket,
                "image": cond_socket,
                "control": cond_socket,
                "ip_adapter": cond_socket,
                "lora": cond_socket,
            },
        }

    def run(self, **conditioning_inputs: ConditioningTensor):
        manager = ConditioningManager()
        tensors = [t for t in conditioning_inputs.values() if t is not None]
        return (manager.combine_encoded(tensors),)


NODE_CLASS_MAPPINGS = {"TensorRTConditioningManager": TensorRTConditioningManager}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTConditioningManager": "TensorRT Conditioning Manager"}
