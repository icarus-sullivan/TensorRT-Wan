from tensorrt_wan.scheduler.flow_match import FlowMatchEulerScheduler

from .. import types


class TensorRTScheduler:
    """Configures the flow-matching Euler scheduler TensorRT Sampler uses to step latents.

    Kept as its own node (rather than a widget on the Sampler) so alternate schedulers can be
    swapped in later without changing the Sampler node's signature — see scheduler/base.py.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.SCHEDULER,)
    RETURN_NAMES = ("scheduler",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            }
        }

    def run(self, shift: float):
        return (FlowMatchEulerScheduler(shift=shift),)


NODE_CLASS_MAPPINGS = {"TensorRTScheduler": TensorRTScheduler}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTScheduler": "TensorRT Scheduler"}
