"""Backend-agnostic resolution picker -- wires this project's known-good `ResolutionProfile`
names (TensorRT optimization-profile shapes and MIGraphX static-build shapes alike, see
`config/schema.py`'s `DEFAULT_RESOLUTION_PROFILES`) into a plain width/height pair for stock
ComfyUI nodes (`EmptyHunyuanLatentVideo`, `KSamplerAdvanced`, etc.) instead of retyping numbers.
No engine/device logic here at all -- pure lookup table.
"""

from __future__ import annotations

from tensorrt_wan.config.schema import DEFAULT_RESOLUTION_PROFILES

_PROFILES_BY_NAME = {profile.name: profile for profile in DEFAULT_RESOLUTION_PROFILES}


class WanResolutionProfile:
    CATEGORY = "TensorRT-Wan"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profile": (list(_PROFILES_BY_NAME),)}}

    def run(self, profile: str):
        chosen = _PROFILES_BY_NAME[profile]
        return (chosen.width, chosen.height)


NODE_CLASS_MAPPINGS = {"WanResolutionProfile": WanResolutionProfile}
NODE_DISPLAY_NAME_MAPPINGS = {"WanResolutionProfile": "Wan Resolution Profile"}
