"""Framework-wide logging.

Adds a TRACE level below DEBUG (TensorRT/CUDA diagnostics are noisy enough that
DEBUG alone isn't granular enough to separate "what precision was chosen" from
"here is every tensor shape at every step").
"""

from __future__ import annotations

import logging
import sys

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class TensorRTWanLogger(logging.Logger):
    def trace(self, msg: str, *args: object, **kwargs: object) -> None:
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


logging.setLoggerClass(TensorRTWanLogger)

_ROOT_NAME = "tensorrt_wan"
_configured = False


def configure(level: int | str = logging.INFO, *, stream: object = None) -> None:
    """Configure the tensorrt_wan root logger. Safe to call more than once."""
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    if not _configured:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(handler)
        root.propagate = False
        _configured = True


def get_logger(name: str) -> TensorRTWanLogger:
    """Return a module logger nested under the tensorrt_wan root, e.g. get_logger(__name__)."""
    if not _configured:
        configure()
    full_name = name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}"
    return logging.getLogger(full_name)  # type: ignore[return-value]
