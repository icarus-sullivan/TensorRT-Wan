from tensorrt_wan.runtime.manager import RuntimeManager

from .. import types


class TensorRTDiagnostics:
    """Prints GPU/TensorRT/precision/plugin/cache status to a STRING output — wire into a
    ComfyUI "Show Text" node (or similar) to surface `RuntimeManager.diagnostics()` in the UI.
    """

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.RUNTIME, "STRING")
    RETURN_NAMES = ("runtime", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"runtime": (types.RUNTIME,)}}

    def run(self, runtime: RuntimeManager):
        return (runtime, runtime.diagnostics().as_text())


NODE_CLASS_MAPPINGS = {"TensorRTDiagnostics": TensorRTDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTDiagnostics": "TensorRT Diagnostics"}
