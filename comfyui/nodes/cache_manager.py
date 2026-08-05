from tensorrt_wan.runtime.manager import RuntimeManager

from .. import types


class TensorRTCacheManager:
    """Inspect or clear `runtime`'s engine cache from within a workflow."""

    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = (types.RUNTIME, "STRING")
    RETURN_NAMES = ("runtime", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"runtime": (types.RUNTIME,), "action": (["list", "clear"], {"default": "list"})}}

    def run(self, runtime: RuntimeManager, action: str):
        if action == "clear":
            removed = runtime.cache.clear()
            return (runtime, f"Removed {removed} cached engine(s) from {runtime.cache.directory}")

        entries = runtime.cache.list()
        if not entries:
            return (runtime, "Engine cache is empty.")
        lines = [
            f"{e['model_hash'][:12]} precision={e['precision']} gpu={e['gpu_architecture']} "
            f"profile={e['optimization_profile']}"
            for e in entries
        ]
        return (runtime, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"TensorRTCacheManager": TensorRTCacheManager}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorRTCacheManager": "TensorRT Cache Manager"}
