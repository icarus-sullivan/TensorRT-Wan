"""TensorRT-RT ComfyUI custom node package: TensorRT-accelerated Wan VAE encode/decode
(`nodes/vae_rt.py`) and RIFE frame interpolation (`nodes/rife_rt.py`).

Both node files are self-contained and drag-and-droppable on their own — copy either one
directly into another `custom_nodes/*/` package's node list without this `__init__.py` at all.
This package's `__init__.py` just wires both into one ComfyUI custom-node folder.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
