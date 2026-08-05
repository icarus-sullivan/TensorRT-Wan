from tensorrt_wan.runtime.manager import RuntimeManager

from .. import types


class TensorRTPrecisionSelector:
    """Resolves `runtime`'s configured precision preference against the detected GPU and
    reports the decision, without changing the runtime's config — wire this in when a workflow
    wants to *display*/log which precision will be used (see docs/optimization_strategy.md)
    rather than change it; change precision via the Runtime Manager node's `precision` widget.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.RUNTIME, "STRING")
    RETURN_NAMES = ("runtime", "selected_precision")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"runtime": (types.RUNTIME,), "gpu_index": ("INT", {"default": 0, "min": 0})}}

    def run(self, runtime: RuntimeManager, gpu_index: int):
        decision = runtime.select_precision(gpu_index)
        return (runtime, decision.precision)


NODE_CLASS_MAPPINGS = {"TensorRTPrecisionSelector": TensorRTPrecisionSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTPrecisionSelector": "TensorRT Precision Selector"}
