"""Loads the compiled plugin library and exposes the set of plugin names it registers.

The actual plugin implementations are C++/CUDA in `plugins/csrc/` (see `docs/plugins.md`),
built by `scripts/build_plugins.sh` into `libtensorrt_wan_plugins.so`. Each plugin
self-registers with TensorRT's plugin registry via `REGISTER_TENSORRT_PLUGIN` at library-load
time, so this module's job is just: find the .so, `ctypes.CDLL` it once, and let
`RuntimeManager.load_plugins` confirm the requested names are present.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)

# Must match each plugin's `kName` in plugins/csrc/<op>/plugin.h.
PLUGIN_NAMES: tuple[str, ...] = (
    "RotaryEmbedding",
    "AdaLayerNorm",
    "CustomAttention",
    "PatchEmbed",
    "PatchReconstruct",
    "TimeEmbedding",
    "TemporalResize",
    "FusedActivation",
)

_DEFAULT_LIBRARY_NAME = "libtensorrt_wan_plugins.so"
_loaded_library: ctypes.CDLL | None = None


def default_library_path() -> Path:
    # scripts/build_plugins.sh builds into csrc/build/ by convention.
    return Path(__file__).parent / "csrc" / "build" / _DEFAULT_LIBRARY_NAME


def load_plugin_library(path: str | Path | None = None) -> ctypes.CDLL:
    """Load the plugin shared library exactly once per process.

    Every plugin registers itself as a side effect of the library being loaded (static
    `REGISTER_TENSORRT_PLUGIN` initializers) — there is no per-plugin load step, which is why
    `load_plugin()` below only validates that a name exists rather than loading anything itself.
    """
    global _loaded_library
    if _loaded_library is not None:
        return _loaded_library

    library_path = Path(path) if path else default_library_path()
    if not library_path.exists():
        raise FileNotFoundError(
            f"Plugin library not found at {library_path}. Build it with scripts/build_plugins.sh "
            "(requires a CUDA toolkit + TensorRT SDK; not built by default in this dev phase)."
        )
    logger.info("Loading TensorRT-Wan plugin library: %s", library_path)
    _loaded_library = ctypes.CDLL(str(library_path))
    return _loaded_library


def load_plugin(name: str) -> None:
    """Ensure the plugin library is loaded and `name` is one of the plugins it provides."""
    if name not in PLUGIN_NAMES:
        raise ValueError(f"Unknown plugin {name!r}; known plugins: {', '.join(PLUGIN_NAMES)}")
    load_plugin_library()
    logger.debug("Plugin available: %s", name)
