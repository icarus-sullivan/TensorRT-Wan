"""Automatic PyTorch fallback.

Project rule: if TensorRT cannot execute an operation, fall back to PyTorch, warn, never crash.
`run_with_fallback` is the one place that rule is implemented so every engine wrapper
(text encoder, DiT, VAE encoder/decoder) gets identical behavior instead of reimplementing
its own try/except.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from tensorrt_wan.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class FallbackTriggered(Exception):
    """Raised (and immediately caught by `run_with_fallback`) to record why a fallback fired.

    Kept as a real exception type, rather than a bare log line, so tests and diagnostics can
    assert on *why* a fallback happened instead of just observing that it did.
    """

    def __init__(self, op_name: str, cause: Exception) -> None:
        self.op_name = op_name
        self.cause = cause
        super().__init__(f"{op_name} failed on TensorRT path: {cause}")


def run_with_fallback(
    op_name: str,
    trt_fn: Callable[[], T],
    torch_fn: Callable[[], T],
) -> T:
    """Try `trt_fn`; on any exception, warn and run `torch_fn` instead.

    `trt_fn`/`torch_fn` take no arguments by design — callers close over their inputs, which
    keeps this signature stable regardless of the op being wrapped.
    """
    try:
        return trt_fn()
    except Exception as exc:  # noqa: BLE001 - intentionally broad: any TRT failure must fall back
        fallback_exc = FallbackTriggered(op_name, exc)
        logger.warning(str(fallback_exc))
        return torch_fn()
