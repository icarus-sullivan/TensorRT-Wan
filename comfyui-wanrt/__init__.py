"""TensorRT-RT ComfyUI custom node package: TensorRT-accelerated Wan VAE encode/decode
(`nodes/vae_rt.py`), RIFE frame interpolation (`nodes/rife_rt.py`), and Wan 2.2 model loaders with
optional SageAttention3/MagCache (`nodes/tensorrt_perf.py`).

Each node file is self-contained and drag-and-droppable on its own — copy any one of them
directly into another `custom_nodes/*/` package's node list without this `__init__.py` at all. The
one exception is purely cosmetic: `web/tensorrt_perf.js` (which makes the MagCache preset's
numeric fields auto-populate instead of showing stale defaults) only loads via this package's
`WEB_DIRECTORY`, so it needs this whole `comfyui-wanrt/` folder, not just the one node file --
tensorrt_perf.py's node logic itself is unaffected either way, since it already ignores those
fields outside `MagCache=Custom`.

This package's `__init__.py` just wires all three node files (and the JS) into one ComfyUI
custom-node folder.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
