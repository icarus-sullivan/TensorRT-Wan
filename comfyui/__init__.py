"""TensorRT-Wan ComfyUI custom node package.

ComfyUI imports this directory as a top-level module named after its folder under
`custom_nodes/` — see docs/comfyui_integration.md for why that folder must NOT be named
`tensorrt_wan` (it would collide with the pip-installed `tensorrt_wan` package these nodes
import). Every import in this package is relative for exactly that reason: it must work
regardless of what name ComfyUI assigns this package at load time.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
